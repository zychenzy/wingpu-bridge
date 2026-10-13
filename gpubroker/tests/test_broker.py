"""Unit tests for the stdlib GPU broker.

GPU and process calls are patched, so the suite runs without a GPU or nvidia-smi.
"""
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import gpubroker.broker as broker


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def acquire_args(directory, runtime, **overrides):
    base = dict(
        dir=str(directory), runtime=runtime, pidfile=None, pid=4242, shutdown_url=None,
        vram_mb=None, label=None, wait=0.0, poll=0.0, force=False, no_evict=False,
        busy_util=broker.BUSY_UTIL_PCT, vram_floor=broker.VRAM_FLOOR_MB,
        free_timeout=broker.FREE_TIMEOUT_S, stale_grace=broker.STALE_GRACE_S,
    )
    base.update(overrides)
    return _Args(**base)


class GpuBrokerTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(mock.patch.object(broker, "gpu_util_pct", return_value=0))
        self.enterContext(mock.patch.object(broker, "gpu_mem_used_mb", return_value=300))
        self.enterContext(mock.patch.object(broker, "gpu_mem_total_mb", return_value=16384))
        self.enterContext(mock.patch.object(broker, "gpu_name", return_value="Test GPU"))
        self.enterContext(mock.patch.object(broker, "gpu_compute_apps", return_value=None))
        self.enterContext(mock.patch.object(broker, "gpu_process_sm", return_value=None))
        self.enterContext(mock.patch.object(broker.time, "sleep", lambda *_a, **_k: None))
        self.enterContext(mock.patch.object(broker, "pid_alive", return_value=True))
        self.kill = self.enterContext(mock.patch.object(broker, "kill_pid"))

    def _holder(self):
        conn = broker.connect(self.dir)
        try:
            return broker.read_holder(conn)
        finally:
            conn.close()

    def _events(self):
        conn = broker.connect(self.dir)
        try:
            return broker.list_events(conn, limit=500, since=None)
        finally:
            conn.close()

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
        rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.assertEqual(rc, 0)
        self.kill.assert_called_once_with(111)
        self.assertEqual(self._holder()["runtime_id"], "locate")

    def test_acquire_denied_when_holder_busy_and_no_wait(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "gpu_util_pct", return_value=90):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222, wait=0.0))
        self.assertEqual(rc, 3)
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
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 3)
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_acquire_reclaims_stale_dead_holder_when_vram_free(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, stale_grace=0.0))
        self.assertEqual(rc, 0)
        self.assertEqual(self._holder()["runtime_id"], "ocr")

    def test_acquire_denies_stale_reclaim_when_vram_still_occupied(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False), \
             mock.patch.object(broker, "gpu_mem_used_mb", return_value=14000):
            rc = broker.cmd_acquire(acquire_args(
                self.dir, "ocr", pid=333, stale_grace=0.0, free_timeout=0.1, wait=0.0,
            ))
        self.assertEqual(rc, 3)
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_vram_gate_waits_for_drain_before_grant(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111, vram_mb=14000))
        used = iter([14000, 14000, 14000, 400, 400, 400])
        with mock.patch.object(broker, "gpu_mem_used_mb", side_effect=lambda: next(used, 400)):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.assertEqual(rc, 0)
        self.assertEqual(self._holder()["runtime_id"], "locate")

    def test_release_by_holder_clears_and_by_other_is_noop(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        broker.cmd_release(_Args(dir=str(self.dir), runtime="ocr", token=None))
        self.assertEqual(self._holder()["runtime_id"], "llama")
        broker.cmd_release(_Args(dir=str(self.dir), runtime="llama", token=None))
        self.assertIsNone(self._holder())

    def test_evict_command_frees_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        rc = broker.cmd_evict(_Args(
            dir=str(self.dir), runtime=None,
            vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0,
        ))
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
        action = broker._watchdog_once(
            self.dir, idle_timeout=0, busy_util=broker.BUSY_UTIL_PCT,
            vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0,
        )
        self.assertEqual(action["action"], "idle_offload")
        self.assertEqual(action["runtime"], "llama")
        self.assertIsNone(self._holder())

    def test_watchdog_spares_busy_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "gpu_util_pct", return_value=80):
            action = broker._watchdog_once(
                self.dir, idle_timeout=0, busy_util=broker.BUSY_UTIL_PCT,
                vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0,
            )
        self.assertEqual(action["action"], "none")
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_watchdog_clears_stale_dead_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            action = broker._watchdog_once(
                self.dir, idle_timeout=99999, busy_util=broker.BUSY_UTIL_PCT,
                vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0, stale_grace=0.0,
            )
        self.assertEqual(action["action"], "cleared_stale")
        self.assertIsNone(self._holder())

    def test_watchdog_spares_fresh_dead_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            action = broker._watchdog_once(
                self.dir, idle_timeout=99999, busy_util=broker.BUSY_UTIL_PCT,
                vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0, stale_grace=99999,
            )
        self.assertEqual(action["action"], "none")
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_touch_resets_idle_clock_so_watchdog_spares_an_old_lease(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        conn = broker.connect(self.dir)
        try:
            conn.execute(
                "UPDATE holder SET acquired_at=? WHERE id=1",
                ("2000-01-01T00:00:00+00:00",),
            )
        finally:
            conn.close()
        evicted = broker._watchdog_once(
            self.dir, idle_timeout=100, busy_util=broker.BUSY_UTIL_PCT,
            vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0,
        )
        self.assertEqual(evicted["action"], "idle_offload")

        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        conn = broker.connect(self.dir)
        try:
            conn.execute(
                "UPDATE holder SET acquired_at=? WHERE id=1",
                ("2000-01-01T00:00:00+00:00",),
            )
        finally:
            conn.close()
        broker.cmd_touch(_Args(dir=str(self.dir), runtime="llama"))
        spared = broker._watchdog_once(
            self.dir, idle_timeout=100, busy_util=broker.BUSY_UTIL_PCT,
            vram_floor=broker.VRAM_FLOOR_MB, free_timeout=1.0,
        )
        self.assertEqual(spared["action"], "none")
        self.assertIsNotNone(self._holder()["last_active_at"])

    def test_recent_touch_counts_as_busy_without_gpu_util(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        broker.cmd_touch(_Args(dir=str(self.dir), runtime="llama"))
        with mock.patch.object(broker, "gpu_compute_apps", return_value=[]):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 3)
        self.kill.assert_not_called()

    def test_foreign_gpu_work_does_not_make_the_holder_busy(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        apps = [{"pid": 999999, "name": "game", "vram_mb": 4000}]
        with mock.patch.object(broker, "gpu_util_pct", return_value=95), \
             mock.patch.object(broker, "gpu_compute_apps", return_value=apps):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 0)
        self.assertEqual(self._holder()["runtime_id"], "ocr")

    def test_holder_process_sm_counts_as_busy(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        apps = [{"pid": 111, "name": "llama-server", "vram_mb": 14000}]
        with mock.patch.object(broker, "gpu_util_pct", return_value=0), \
             mock.patch.object(broker, "gpu_compute_apps", return_value=apps), \
             mock.patch.object(broker, "gpu_process_sm", return_value=40):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 3)
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_sole_holder_falls_back_to_whole_gpu_util_when_pmon_is_missing(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        apps = [{"pid": 111, "name": "llama-server", "vram_mb": 14000}]
        with mock.patch.object(broker, "gpu_util_pct", return_value=80), \
             mock.patch.object(broker, "gpu_compute_apps", return_value=apps), \
             mock.patch.object(broker, "gpu_process_sm", return_value=None):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, wait=0.0))
        self.assertEqual(rc, 3)

    def test_kill_and_vram_wait_run_outside_the_lock(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111, vram_mb=1000))
        calls = []

        def kill(pid, grace=12):
            calls.append(("kill", broker.lock_held(), pid))

        def mem():
            seen = sum(1 for item in calls if item[0] == "mem")
            calls.append(("mem", broker.lock_held()))
            return 9000 if seen < 2 else 100

        with mock.patch.object(broker, "kill_pid", side_effect=kill), \
             mock.patch.object(broker, "gpu_mem_used_mb", side_effect=mem):
            rc = broker.cmd_acquire(acquire_args(self.dir, "locate", pid=222))
        self.assertEqual(rc, 0)
        self.assertTrue(any(item[0] == "kill" and item[1] is False for item in calls))
        self.assertTrue(any(item[0] == "mem" and item[1] is False for item in calls))

    def test_shutdown_url_is_posted_before_the_signal(self):
        broker.cmd_acquire(acquire_args(
            self.dir, "llama", pid=111, shutdown_url="http://127.0.0.1:8799/shutdown",
        ))
        order = []
        with mock.patch.object(broker, "http_post", side_effect=lambda *_a, **_k: order.append("post") or True), \
             mock.patch.object(broker, "kill_pid", side_effect=lambda *_a, **_k: order.append("kill")):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333))
        self.assertEqual(rc, 0)
        self.assertEqual(order, ["post", "kill"])

    def test_force_on_fresh_dead_holder_records_acquired(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        with mock.patch.object(broker, "pid_alive", return_value=False):
            rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, force=True))
        self.assertEqual(rc, 0)
        acquired = [row for row in self._events() if row["event"] == "acquired" and row["runtime_id"] == "ocr"]
        self.assertTrue(acquired)

    def test_no_evict_does_not_steal_a_live_holder(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        rc = broker.cmd_acquire(acquire_args(self.dir, "ocr", pid=333, no_evict=True))
        self.assertEqual(rc, 3)
        self.kill.assert_not_called()
        self.assertEqual(self._holder()["runtime_id"], "llama")

    def test_no_evict_reasserts_same_runtime(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        rc = broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111, no_evict=True))
        self.assertEqual(rc, 0)
        self.kill.assert_not_called()

    def test_events_are_capped(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        conn = broker.connect(self.dir)
        try:
            for index in range(broker.MAX_EVENTS + 40):
                broker.record_event(conn, "note", "llama", {"i": index})
            count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        finally:
            conn.close()
        self.assertLessEqual(count, broker.MAX_EVENTS)

    def test_events_command_filters_and_status_reports_clock(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        run = self.dir / "run"
        run.mkdir(parents=True, exist_ok=True)
        (run / "watchdog.pid").write_text("4242\n", encoding="utf-8")
        (run / "watchdog.json").write_text(json.dumps({
            "idle_timeout_s": 2100, "poll_s": 15, "busy_util": 15,
        }), encoding="utf-8")
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = broker.cmd_status(_Args(dir=str(self.dir), json=True, busy_util=15))
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["version"], broker.VERSION)
        self.assertEqual(payload["protocol"], broker.PROTOCOL)
        self.assertEqual(payload["gpu"]["name"], "Test GPU")
        self.assertTrue(payload["watchdog"]["running"])
        self.assertIsNotNone(payload["watchdog"]["next_evict_in_s"])
        self.assertGreater(payload["watchdog"]["next_evict_in_s"], 2000)
        self.assertIn("idle_age_s", payload["holder"])

        buf = io.StringIO()
        with redirect_stdout(buf):
            broker.cmd_events(_Args(dir=str(self.dir), json=True, limit=10, since=None))
        events = json.loads(buf.getvalue())["events"]
        self.assertTrue(any(row["event"] == "acquired" for row in events))

    def test_version_command(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = broker.cmd_version(_Args())
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["version"], broker.VERSION)
        self.assertEqual(payload["protocol"], broker.PROTOCOL)

    def test_status_includes_per_process_vram(self):
        broker.cmd_acquire(acquire_args(self.dir, "llama", pid=111))
        apps = [{"pid": 111, "name": "llama-server", "vram_mb": 15205}]
        buf = io.StringIO()
        with mock.patch.object(broker, "gpu_compute_apps", return_value=apps), redirect_stdout(buf):
            broker.cmd_status(_Args(dir=str(self.dir), json=True, busy_util=15))
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["processes"][0]["vram_mb"], 15205)


if __name__ == "__main__":
    unittest.main()
