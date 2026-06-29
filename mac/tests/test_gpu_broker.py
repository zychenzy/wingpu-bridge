"""Unit tests for wsl/gpu_broker.py.

The broker is deployed to the WSL GPU host, but it is stdlib-only and its GPU /
process calls are isolated behind module functions, so we load it by path and
mock those calls to exercise the full acquire/evict/release/watchdog logic on a
Mac with no GPU.
"""
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

_BROKER_PATH = Path(__file__).resolve().parents[2] / "wsl" / "gpu_broker.py"
_spec = importlib.util.spec_from_file_location("gpu_broker", _BROKER_PATH)
broker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(broker)


class _Args:
    """Minimal argparse.Namespace stand-in for command handlers."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def acquire_args(directory, runtime, **overrides):
    base = dict(
        dir=str(directory), runtime=runtime, pidfile=None, pid=4242, shutdown_url=None,
        vram_mb=None, label=None, wait=0.0, poll=0.0, force=False,
        busy_util=broker.BUSY_UTIL_PCT, vram_floor=broker.VRAM_FLOOR_MB,
        free_timeout=broker.FREE_TIMEOUT_S, stale_grace=broker.STALE_GRACE_S,
    )
    base.update(overrides)
    return _Args(**base)


class GpuBrokerTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        # Default GPU is idle with plenty of free VRAM; patch the probes so no
        # nvidia-smi is ever shelled out, and make sleeps instant.
        self.enterContext(mock.patch.object(broker, "gpu_util_pct", return_value=0))
        self.enterContext(mock.patch.object(broker, "gpu_mem_used_mb", return_value=300))
        self.enterContext(mock.patch.object(broker, "gpu_mem_total_mb", return_value=16384))
        self.enterContext(mock.patch.object(broker.time, "sleep", lambda *_: None))
        self.enterContext(mock.patch.object(broker, "pid_alive", return_value=True))
        self.kill = self.enterContext(mock.patch.object(broker, "kill_pid"))

    # ── helpers ──────────────────────────────────────────────────────────────
    def _holder(self):
        conn = broker.connect(self.dir)
        try:
            return broker.read_holder(conn)
        finally:
            conn.close()

    # ── tests ─────────────────────────────────────────────────────────────────
    def test_acquire_on_free_gpu_grants_and_sets_holder(self):
        rc = broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        self.assertEqual(rc, 0)
        holder = self._holder()
        self.assertEqual(holder["runtime_id"], "llama")
        self.assertEqual(holder["pid"], 111)
        self.assertIsNotNone(holder["lease_token"])

    def test_reacquire_same_runtime_reasserts_same_lease(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        token1 = self._holder()["lease_token"]
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        self.assertEqual(self._holder()["lease_token"], token1)
        self.kill.assert_not_called()

    def test_acquire_evicts_idle_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        # GPU idle (util 0). A second runtime should evict llama and take over.
        rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.assertEqual(rc, 0)
        self.kill.assert_called_once_with(111)
        self.assertEqual(self._holder()["runtime_id"], "locate")

    def test_acquire_denied_when_holder_busy_and_no_wait(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "gpu_util_pct", return_value=90):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222, wait=0.0))
        self.assertEqual(rc, 3)  # busy
        self.kill.assert_not_called()
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_force_evicts_busy_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "gpu_util_pct", return_value=90):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222, force=True))
        self.assertEqual(rc, 0)
        self.kill.assert_called_once_with(111)
        self.assertEqual(self._holder()["runtime_id"], "locate")

    def test_acquire_protects_fresh_dead_holder(self):
        # A holder that just died (within stale_grace) is a momentary blip: do not steal it.
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 3)  # protected -> denied (reason: recovering)
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_acquire_reclaims_stale_dead_holder_when_vram_free(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, stale_grace=0.0))
        self.assertEqual(rc, 0)  # dead + stale + VRAM low (300 < floor) -> reclaimed
        self.assertEqual(self._holder()["runtime_id"], "ocr")

    def test_acquire_denies_stale_reclaim_when_vram_still_occupied(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False), \
             mock.patch.object(broker, "gpu_mem_used_mb", return_value=14000):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333,
                                                 stale_grace=0.0, free_timeout=0.1, wait=0.0))
        self.assertEqual(rc, 3)  # VRAM still occupied -> refuse to grant (avoid OOM)
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_vram_gate_waits_for_drain_before_grant(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111, vram_mb=14000))
        # Simulate VRAM still pinned for two polls, then released.
        used = iter([14000, 14000, 14000, 400, 400, 400])
        with mock.patch.object(broker, "gpu_mem_used_mb", side_effect=lambda: next(used, 400)):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.assertEqual(rc, 0)
        self.assertEqual(self._holder()["runtime_id"], "locate")

    def test_release_by_holder_clears_and_by_other_is_noop(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        broker.cmd_release(_Args(dir=str(self.dir), runtime="ocr", token=None))
        self.assertEqual(self._holder()["runtime_id"], "llama")  # ocr is not holder
        broker.cmd_release(_Args(dir=str(self.dir), runtime="llama", token=None))
        self.assertIsNone(self._holder())

    def test_evict_command_frees_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        rc = broker.cmd_evict(_Args(dir=str(self.dir), runtime=None,
                                    vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0))
        self.assertEqual(rc, 0)
        self.kill.assert_called_once_with(111)
        self.assertIsNone(self._holder())

    def test_pidfile_eviction_signals_and_removes_file(self):
        pidfile = self.dir / "run" / "llama.pid"
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text("9876\n")
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=None, pidfile=str(pidfile)))
        broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.kill.assert_called_once_with(9876)
        self.assertFalse(pidfile.exists())

    def test_watchdog_evicts_idle_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        # idle_timeout 0 → any holder with low util is immediately idle-offloaded.
        action = broker._watchdog_once(self.dir, idle_timeout=0, busy_util=broker.BUSY_UTIL_PCT,
                                       vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0)
        self.assertEqual(action["action"], "idle_offload")
        self.assertEqual(action["runtime"], "llama")
        self.assertIsNone(self._holder())

    def test_watchdog_spares_busy_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "gpu_util_pct", return_value=80):
            action = broker._watchdog_once(self.dir, idle_timeout=0, busy_util=broker.BUSY_UTIL_PCT,
                                           vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0)
        self.assertEqual(action["action"], "none")
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_watchdog_clears_stale_dead_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            action = broker._watchdog_once(self.dir, idle_timeout=99999, busy_util=broker.BUSY_UTIL_PCT,
                                           vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0, stale_grace=0.0)
        self.assertEqual(action["action"], "cleared_stale")
        self.assertIsNone(self._holder())

    def test_watchdog_spares_fresh_dead_holder(self):
        # Within stale_grace, a dead holder is a blip the owner may be restarting: keep it.
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            action = broker._watchdog_once(self.dir, idle_timeout=99999, busy_util=broker.BUSY_UTIL_PCT,
                                           vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0, stale_grace=99999)
        self.assertEqual(action["action"], "none")
        self.assertEqual(self._holder()["runtime_id"], "llama")


if __name__ == "__main__":
    unittest.main()
