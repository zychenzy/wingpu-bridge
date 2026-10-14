"""Tests for the wingpu <-> GPU broker wiring in wingpu_cli.main and the
shared gpubroker client."""
import io
import unittest
from unittest import mock

import gpubroker.client as bc
import wingpu_cli.main as main


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
        self.assertEqual(captured["shutdown_url"], "http://127.0.0.1:8799/shutdown")
        self.assertEqual(captured["stale_grace"], 45)

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

    def test_install_uploads_on_stdin(self):
        seen = []

        def runner(script, *, timeout=None, stdin=None):
            seen.append((script, stdin))
            if stdin is not None:
                return 0, ""
            return 0, '{"ok": true}\nbroker_installed\n'

        ok = bc.install(runner, "/home/czy/.gpu-bridge", b"print('hi')\n")
        self.assertTrue(ok)
        upload_script, payload = seen[0]
        self.assertEqual(payload, b"print('hi')\n")
        self.assertIn("gpu_broker.py", upload_script)
        self.assertNotIn("base64", upload_script)


class FastPathAndTouchTests(unittest.TestCase):
    def test_fast_path_reasserts_lease_without_restart(self):
        settings = _settings()
        with mock.patch.object(main, "check_command"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "runtime_lane"), \
             mock.patch.object(main, "supported_cache_types", return_value=["q4_0"]), \
             mock.patch.object(main, "check_ssh_connectivity"), \
             mock.patch.object(main, "ensure_backend_tunnel"), \
             mock.patch.object(main, "backend_ready_for_fast_path", return_value=True), \
             mock.patch.object(main, "set_selected_model"), \
             mock.patch.object(main, "set_selected_runtime"), \
             mock.patch.object(main, "set_selected_cache_type"), \
             mock.patch.object(main, "ensure_shutdown_helper") as helper, \
             mock.patch.object(main, "reassert_gpu_lease") as reassert, \
             mock.patch.object(main, "start_remote_runtime") as start, \
             mock.patch.object(main, "acquire_gpu_lease") as acquire:
            main.ensure_runtime_loaded(settings, "Qwen", "upstream", "q4_0", "q4_0", True)
        helper.assert_called_once()
        reassert.assert_called_once()
        start.assert_not_called()
        acquire.assert_not_called()

    def test_reassert_does_not_evict(self):
        captured = {}

        def fake_acquire(_runner, **kwargs):
            captured.update(kwargs)
            return {"granted": True}

        with mock.patch.object(main, "llama_vram_estimate_mb", return_value=None), \
             mock.patch.object(main, "ensure_broker_installed"), \
             mock.patch.object(main.broker_client, "acquire", side_effect=fake_acquire):
            main.reassert_gpu_lease(_settings(), "Qwen", "upstream")
        self.assertTrue(captured["no_evict"])
        self.assertEqual(captured["wait"], 0)
        self.assertEqual(captured["shutdown_url"], "http://127.0.0.1:8799/shutdown")

    def test_gateway_touches_on_request_and_rate_limits(self):
        settings = _settings()
        coord = main.GatewayCoordinator(settings)

        class ImmediateThread:
            def __init__(self, target=None, name=None, daemon=None):
                self.target = target

            def start(self):
                self.target()

        with mock.patch.object(coord, "write_state"), \
             mock.patch.object(main, "touch_gpu_lease", return_value=True) as touch, \
             mock.patch.object(main.threading, "Thread", ImmediateThread):
            coord.begin_request()
            coord.begin_request()
        self.assertEqual(touch.call_count, 1)

    def test_relay_target_comes_from_config(self):
        settings = _settings()
        settings.gateway.backend_tunnel_target_host = "auto"
        self.assertIn("nc 127.0.0.1 8000", main._relay_backend_command(settings)[-1])
        settings.gateway.backend_tunnel_target_host = "10.1.1.1"
        self.assertIn("nc 10.1.1.1 8000", main._relay_backend_command(settings)[-1])

    def test_watchdog_start_forwards_drain_options(self):
        with mock.patch.object(main, "load_settings", return_value=_settings()), \
             mock.patch.object(main, "ensure_broker_installed"), \
             mock.patch.object(main.broker_client, "watchdog_start", return_value=(0, "watchdog_started:1")) as start:
            rc = main.main(["broker", "watchdog", "start"])
        self.assertEqual(rc, 0)
        kwargs = start.call_args.kwargs
        self.assertEqual(kwargs["vram_floor"], 2000)
        self.assertEqual(kwargs["free_timeout"], 60)
        self.assertEqual(kwargs["stale_grace"], 45)

    def test_gateway_logs_read_the_local_file(self):
        settings = _settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        settings.state.gateway_log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
        buf = io.StringIO()
        with mock.patch("sys.stdout", buf):
            main.cmd_logs(settings, "gateway", 2)
        text = buf.getvalue()
        self.assertIn("two", text)
        self.assertIn("three", text)
        self.assertNotIn("one", text)


if __name__ == "__main__":
    unittest.main()
