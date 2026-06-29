"""Tests for the wingpu <-> GPU broker wiring in wingpu_cli.main and the
standalone broker_client helpers."""
import unittest
from unittest import mock

import wingpu_cli.main as main
import wingpu_cli.broker_client as bc


def _settings(broker_enabled=True):
    return main.Settings(
        connection=main.ConnectionConfig(host="win-gpu", api_key="sk"),
        gateway=main.GatewayConfig(),
        paths=main.PathsConfig(remote_home="/home/czy", remote_src_root="/home/czy/src",
                               remote_models_root="/home/czy/models/Qwen"),
        runtime_defaults=main.RuntimeDefaults(default_runtime="upstream", remote_state_dir="~/.gpu-bridge"),
        runtimes={"upstream": main.RuntimeLane(
            kind="native", source_dir="/s", build_dir="/b",
            server_bin="/b/bin/llama-server", bench_bin="/b/bin/llama-bench",
            supported_cache_types=["q4_0"])},
        state=main.StateConfig(state_dir=main.Path("/tmp/wingpu-broker-tests")),
        broker=main.BrokerConfig(enabled=broker_enabled),
    )


class AcquireLeaseTests(unittest.TestCase):
    def setUp(self):
        # vram estimate touches the catalog/config; isolate it.
        self.enterContext(mock.patch.object(main, "llama_vram_estimate_mb", return_value=None))

    def test_broker_remote_dir_defaults_to_remote_state_dir(self):
        self.assertEqual(main.broker_remote_dir(_settings()), "/home/czy/.gpu-bridge")

    def test_disabled_broker_is_a_noop(self):
        with mock.patch.object(main.broker_client, "acquire") as acquire:
            main.acquire_gpu_lease(_settings(broker_enabled=False), "Qwen", "upstream")
        acquire.assert_not_called()

    def test_grant_with_eviction_does_not_raise(self):
        with mock.patch.object(main.broker_client, "acquire",
                               return_value={"granted": True, "evicted": "locate"}):
            main.acquire_gpu_lease(_settings(), "Qwen", "upstream")  # no exception

    def test_busy_denial_raises(self):
        with mock.patch.object(main.broker_client, "acquire",
                               return_value={"granted": False, "holder": "locate"}):
            with self.assertRaisesRegex(main.WingpuError, "held by 'locate'"):
                main.acquire_gpu_lease(_settings(), "Qwen", "upstream")

    def test_acquire_passes_pidfile_and_force(self):
        captured = {}

        def fake_acquire(_runner, **kwargs):
            captured.update(kwargs)
            return {"granted": True}

        with mock.patch.object(main.broker_client, "acquire", side_effect=fake_acquire):
            main.acquire_gpu_lease(_settings(), "Qwen", "upstream", force=True)
        self.assertEqual(captured["runtime"], "llama")
        self.assertTrue(captured["force"])
        self.assertEqual(captured["pidfile"], "/home/czy/.gpu-bridge/run/upstream.pid")

    def test_unreachable_broker_fails_open(self):
        # No JSON from acquire, and install also yields nothing: warn and proceed.
        with mock.patch.object(main.broker_client, "acquire", return_value=None), \
             mock.patch.object(main, "ensure_broker_installed"):
            main.acquire_gpu_lease(_settings(), "Qwen", "upstream")  # no exception


class LeaseLifecycleTests(unittest.TestCase):
    def test_ensure_runtime_loaded_acquires_before_start(self):
        settings = _settings()
        manager = mock.Mock()
        with mock.patch.object(main, "check_command"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "runtime_lane"), \
             mock.patch.object(main, "supported_cache_types", return_value=["q4_0"]), \
             mock.patch.object(main, "check_ssh_connectivity"), \
             mock.patch.object(main, "ensure_backend_tunnel"), \
             mock.patch.object(main, "backend_ready_for_fast_path", return_value=False), \
             mock.patch.object(main, "runtime_process_info", return_value={"running": False}), \
             mock.patch.object(main, "acquire_gpu_lease", manager.acquire), \
             mock.patch.object(main, "start_remote_runtime", manager.start), \
             mock.patch.object(main, "wait_for_backend_api"):
            main.ensure_runtime_loaded(settings, "Qwen", "upstream", "q4_0", "q4_0", True)
        ordered = [c[0] for c in manager.mock_calls]
        self.assertIn("acquire", ordered)
        self.assertIn("start", ordered)
        self.assertLess(ordered.index("acquire"), ordered.index("start"))

    def test_stop_releases_lease(self):
        settings = _settings()
        with mock.patch.object(main, "stop_gateway"), \
             mock.patch.object(main, "stop_backend_tunnel"), \
             mock.patch.object(main, "stop_remote_runtime"), \
             mock.patch.object(main, "release_gpu_lease") as release:
            main.stop(settings)
        release.assert_called_once_with(settings)

    def test_release_swallows_broker_errors(self):
        with mock.patch.object(main.broker_client, "release", side_effect=RuntimeError("ssh down")):
            main.release_gpu_lease(_settings())  # must not raise


class BrokerClientTests(unittest.TestCase):
    def test_parse_json_takes_last_json_line(self):
        out = "some log noise\n{\"granted\": true, \"runtime\": \"llama\"}\n"
        self.assertEqual(bc._parse_json(out), {"granted": True, "runtime": "llama"})

    def test_parse_json_returns_none_without_json(self):
        self.assertIsNone(bc._parse_json("no json here\n"))

    def test_acquire_builds_invocation_and_parses(self):
        seen = {}

        def runner(script, *, timeout=None):
            seen["script"] = script
            seen["timeout"] = timeout
            return 0, '{"granted": true, "evicted": "ocr"}'

        result = bc.acquire(runner, remote_dir="/home/czy/.gpu-bridge", runtime="llama",
                            pidfile="/home/czy/.gpu-bridge/run/upstream.pid", vram_mb=14000,
                            wait=30, force=True, busy_util=15, vram_floor=2000, free_timeout=60)
        self.assertEqual(result, {"granted": True, "evicted": "ocr"})
        self.assertIn("gpu_broker.py", seen["script"])
        self.assertIn("acquire --runtime llama", seen["script"])
        self.assertIn("--force", seen["script"])
        self.assertIn("--pidfile /home/czy/.gpu-bridge/run/upstream.pid", seen["script"])

    def test_is_installed_detects_presence(self):
        self.assertTrue(bc.is_installed(lambda s, **k: (0, "present\n"), "/d"))
        self.assertFalse(bc.is_installed(lambda s, **k: (0, "missing\n"), "/d"))

    def test_install_uploads_base64_and_reports(self):
        seen = {}

        def runner(script, *, timeout=None):
            seen["script"] = script
            return 0, "broker_installed\n"

        ok = bc.install(runner, "/home/czy/.gpu-bridge", b"print('hi')\n")
        self.assertTrue(ok)
        self.assertIn("base64 -d", seen["script"])
        self.assertIn("gpu_broker.py", seen["script"])


if __name__ == "__main__":
    unittest.main()
