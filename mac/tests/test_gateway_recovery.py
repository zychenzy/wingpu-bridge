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
    WingpuError,
    ensure_runtime_loaded,
    load_settings,
    remote_runtime_base_dir,
    remote_runtime_log_file,
    remote_runtime_pid_file,
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
    def test_load_settings_fails_fast_when_project_config_is_missing(self):
        config_text = b'''
[paths]
remote_home = "/home/czy"
remote_src_root = "{remote_home}/src"
remote_models_root = "{remote_home}/models/Qwen"

[connection]
host = "gpu-host"
distro = "Ubuntu"
api_key = "sk-local"
local_port = 8000
remote_port = 8000
ssh_connect_timeout = 8
server_alive_interval = 30
server_alive_count_max = 3

[gateway]
listen_host = "127.0.0.1"
backend_host = "127.0.0.1"
backend_local_port = 18000
idle_offload_enabled = true
idle_timeout_seconds = 1800
idle_poll_seconds = 5
request_timeout_seconds = 1800
restart_mode = "on_demand"

[runtime_defaults]
default_runtime = "turboquant-cuda"
served_model_name = "qwen-local"
n_gpu_layers = 99
threads = 8
startup_timeout_seconds = 240
build_jobs = 8
cuda_architectures = "89"
flash_attn = true
remote_state_dir = "~/.gpu-bridge"
default_cache_type_k = "turbo3_0"
default_cache_type_v = "turbo3_0"
cmake_args = []
build_targets = ["llama-server", "llama-bench"]
extra_server_args = []

[runtimes.turboquant-cuda]
kind = "native"
source_dir = "/home/czy/src/llama-cpp-turboquant-cuda"
build_dir = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89"
server_bin = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-bench"
supported_cache_types = ["turbo3_0"]

[state]
state_dir = "/tmp/wingpu-tests"
selected_model_file = "selected_model"
selected_runtime_file = "selected_runtime"
selected_cache_type_k_file = "selected_cache_type_k"
selected_cache_type_v_file = "selected_cache_type_v"
benchmark_dir_name = "benchmarks"
restart_on_model_set = false
gateway_pid_file = "gateway.pid"
gateway_log_file = "gateway.log"
gateway_state_file = "gateway_state.json"
gateway_lock_file = "gateway.lock"
'''
        with mock.patch.object(main, "read_config_bytes", return_value=config_text), \
             mock.patch.object(main, "project_config_path", return_value=None):
            with self.assertRaisesRegex(WingpuError, "project-local wingpu.local.toml"):
                load_settings()

    def test_load_settings_uses_project_config_over_defaults(self):
        config_text = b'''
[paths]
remote_home = "/home/czy"
remote_src_root = "{remote_home}/src"
remote_models_root = "{remote_home}/models/Qwen"

[connection]
host = ""
distro = "Ubuntu"
api_key = "sk-local"
local_port = 8000
remote_port = 8000
ssh_connect_timeout = 8
server_alive_interval = 30
server_alive_count_max = 3

[gateway]
listen_host = "127.0.0.1"
backend_host = "127.0.0.1"
backend_local_port = 18000
idle_offload_enabled = true
idle_timeout_seconds = 1800
idle_poll_seconds = 5
request_timeout_seconds = 1800
restart_mode = "on_demand"

[runtime_defaults]
default_runtime = "turboquant-cuda"
served_model_name = "qwen-local"
n_gpu_layers = 99
threads = 8
startup_timeout_seconds = 240
build_jobs = 8
cuda_architectures = "89"
flash_attn = true
remote_state_dir = "~/.gpu-bridge"
default_cache_type_k = "turbo3_0"
default_cache_type_v = "turbo3_0"
cmake_args = []
build_targets = ["llama-server", "llama-bench"]
extra_server_args = []

[runtimes.turboquant-cuda]
kind = "native"
source_dir = "/home/czy/src/llama-cpp-turboquant-cuda"
build_dir = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89"
server_bin = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama-cpp-turboquant-cuda/build-cuda89/bin/llama-bench"
supported_cache_types = ["turbo3_0"]

[state]
state_dir = "/tmp/wingpu-tests"
selected_model_file = "selected_model"
selected_runtime_file = "selected_runtime"
selected_cache_type_k_file = "selected_cache_type_k"
selected_cache_type_v_file = "selected_cache_type_v"
benchmark_dir_name = "benchmarks"
restart_on_model_set = false
gateway_pid_file = "gateway.pid"
gateway_log_file = "gateway.log"
gateway_state_file = "gateway_state.json"
gateway_lock_file = "gateway.lock"
'''
        with mock.patch.object(main, "read_config_bytes", return_value=config_text), \
             mock.patch.object(main, "project_config_path", return_value=main.Path("/tmp/wingpu.local.toml")), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("pathlib.Path.open", mock.mock_open(read_data=b"[connection]\nhost = \"win-gpu\"\n")):
            settings = load_settings()

        self.assertEqual(settings.connection.host, "win-gpu")

    def test_remote_runtime_paths_expand_home_relative_state_dir(self):
        settings = make_settings()

        self.assertEqual(remote_runtime_base_dir(settings), "/home/czy/.gpu-bridge")
        self.assertEqual(remote_runtime_pid_file(settings, "turboquant-cuda"), "/home/czy/.gpu-bridge/run/turboquant-cuda.pid")
        self.assertEqual(remote_runtime_log_file(settings, "turboquant-cuda"), "/home/czy/.gpu-bridge/logs/turboquant-cuda.log")

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

    def test_models_probe_is_served_locally_without_runtime_start(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "GET"
        handler.path = "/v1/models?foo=bar"
        handler.headers = {}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with mock.patch.object(main, "selected_model", return_value="Qwen3.6-35B-A3B-UD-IQ3_S"), \
             mock.patch.object(main, "selected_runtime", return_value="turboquant-cuda"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "runtime_process_info", side_effect=AssertionError("should not inspect remote runtime")), \
             mock.patch.object(main.http.client, "HTTPConnection", side_effect=AssertionError("proxy should not dial backend")):
            handled = handler._handle_admin()

        self.assertTrue(handled)
        self.assertEqual(coordinator.ensure_calls, 0)
        self.assertEqual(sent["status"][0][0], 200)
        self.assertIn(b'"id": "qwen-local"', handler.wfile.getvalue())

    def test_props_probe_is_served_locally_without_runtime_start(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "GET"
        handler.path = "/v1/props?foo=bar"
        handler.headers = {}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with mock.patch.object(main, "selected_model", return_value="Qwen3.6-35B-A3B-UD-IQ3_S"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "remote_model_path", return_value="/home/czy/models/Qwen/model.gguf"), \
             mock.patch.object(main.http.client, "HTTPConnection", side_effect=AssertionError("proxy should not dial backend")):
            handled = handler._handle_admin()

        self.assertTrue(handled)
        self.assertEqual(coordinator.ensure_calls, 0)
        self.assertEqual(sent["status"][0][0], 200)
        self.assertIn(b'"model_alias": "qwen-local"', handler.wfile.getvalue())
        self.assertIn(b'"n_ctx": 4096', handler.wfile.getvalue())

    def test_version_probe_returns_fast_local_404(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "GET"
        handler.path = "/version?foo=bar"
        handler.headers = {}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None

        with mock.patch.object(main.http.client, "HTTPConnection", side_effect=AssertionError("proxy should not dial backend")):
            handled = handler._handle_admin()

        self.assertTrue(handled)
        self.assertEqual(coordinator.ensure_calls, 0)
        self.assertEqual(sent["status"][0][0], 404)
        self.assertIn(b'"not_found_error"', handler.wfile.getvalue())


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
