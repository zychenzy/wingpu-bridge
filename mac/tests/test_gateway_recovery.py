import http.client
import io
import unittest
from unittest import mock

from wingpu_cli.main import (
    ConnectionConfig,
    GatewayConfig,
    PathsConfig,
    RuntimeDefaults,
    RuntimeLane,
    Settings,
    StateConfig,
    ensure_runtime_loaded,
)
import wingpu_cli.main as main


class DummyCoordinator:
    def __init__(self):
        self.settings = make_settings()
        self.begin_calls = 0
        self.end_calls = 0
        self.ensure_calls = 0
        self.recover_calls = []

    def begin_request(self):
        self.begin_calls += 1

    def end_request(self):
        self.end_calls += 1

    def ensure_runtime_loaded(self):
        self.ensure_calls += 1

    def recover_runtime_after_proxy_error(self, exc):
        self.recover_calls.append(type(exc).__name__)


class FakeResponse:
    def __init__(self, body=b'{"ok":true}', status=200, reason="OK", headers=None):
        self.body = body
        self.status = status
        self.reason = reason
        self.headers = headers or [("Content-Type", "application/json")]
        self._sent = False

    def getheaders(self):
        return list(self.headers)

    def read(self, _size=-1):
        if self._sent:
            return b""
        self._sent = True
        return self.body


class FlakyHTTPConnection:
    calls = 0

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def request(self, method, path, body=None, headers=None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}

    def getresponse(self):
        type(self).calls += 1
        if type(self).calls == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return FakeResponse()


class GatewayRecoveryTests(unittest.TestCase):
    def test_main_restart_calls_stop_then_start(self):
        settings = make_settings()
        calls = []
        with mock.patch.object(main, "load_settings", return_value=settings), \
             mock.patch.object(main, "stop", side_effect=lambda _settings: calls.append("stop")), \
             mock.patch.object(main, "start", side_effect=lambda _settings, **kwargs: calls.append(("start", kwargs))):
            result = main.main(["restart"])

        self.assertEqual(result, 0)
        self.assertEqual(calls, ["stop", ("start", {
            "explicit_model": None,
            "explicit_runtime": None,
            "explicit_cache_type_k": None,
            "explicit_cache_type_v": None,
            "flash_attn": None,
        })])

    def test_ensure_runtime_loaded_does_not_fast_path_when_runtime_process_is_missing(self):
        settings = make_settings()
        with mock.patch.object(main, "check_command"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "runtime_lane"), \
             mock.patch.object(main, "supported_cache_types", return_value=["turbo3_0"]), \
             mock.patch.object(main, "check_ssh_connectivity"), \
             mock.patch.object(main, "ensure_backend_tunnel"), \
             mock.patch.object(main, "backend_api_json", return_value={"data": [{"id": "qwen-local"}]}), \
             mock.patch.object(main, "runtime_process_info", return_value={"running": False}), \
             mock.patch.object(main, "stop_remote_runtime") as stop_remote_runtime, \
             mock.patch.object(main, "start_remote_runtime") as start_remote_runtime, \
             mock.patch.object(main, "wait_for_backend_api") as wait_for_backend_api:
            ensure_runtime_loaded(
                settings,
                model_name="Qwen3.6-35B-A3B-UD-IQ3_S",
                runtime_id="turboquant-cuda",
                cache_type_k="turbo3_0",
                cache_type_v="turbo3_0",
                flash_attn=True,
                force_restart=False,
            )

        stop_remote_runtime.assert_not_called()
        start_remote_runtime.assert_called_once()
        wait_for_backend_api.assert_called_once()

    def test_proxy_retries_once_after_remote_disconnect(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": "0", "Authorization": "Bearer sk-local"}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None
        handler._send_json = lambda status, payload: self.fail(f"unexpected json error {status}: {payload}")

        with mock.patch.object(main.http.client, "HTTPConnection", FlakyHTTPConnection):
            handler._proxy()

        self.assertEqual(coordinator.begin_calls, 1)
        self.assertEqual(coordinator.end_calls, 1)
        self.assertEqual(coordinator.ensure_calls, 2)
        self.assertEqual(coordinator.recover_calls, ["RemoteDisconnected"])
        self.assertEqual(sent["status"][0][0], 200)
        self.assertIn(b'{"ok":true}', handler.wfile.getvalue())


def make_settings():
    state = StateConfig(state_dir=io_path())
    return Settings(
        connection=ConnectionConfig(api_key="sk-local"),
        gateway=GatewayConfig(),
        paths=PathsConfig(remote_home="/home/czy", remote_src_root="/home/czy/src", remote_models_root="/home/czy/models/Qwen"),
        runtime_defaults=RuntimeDefaults(default_runtime="turboquant-cuda", served_model_name="qwen-local"),
        runtimes={
            "turboquant-cuda": RuntimeLane(
                kind="native",
                source_dir="/home/czy/src/llama-cpp-turboquant-cuda",
                build_dir="/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89",
                server_bin="/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-server",
                bench_bin="/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-bench",
                supported_cache_types=["turbo3_0"],
            )
        },
        state=state,
    )


def io_path():
    # tests don't touch disk because selected_* helpers are patched in the exercised paths
    return main.Path('/tmp/wingpu-tests')


if __name__ == "__main__":
    unittest.main()
