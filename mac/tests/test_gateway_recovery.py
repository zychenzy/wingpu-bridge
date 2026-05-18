import http.client
import io
import json
import tempfile
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

    def connect(self):
        pass

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


class InspectingHTTPConnection:
    calls = 0
    bodies = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def connect(self):
        pass

    def request(self, method, path, body=None, headers=None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}
        type(self).bodies.append(body)

    def getresponse(self):
        type(self).calls += 1
        if type(self).calls == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return FakeResponse()


class ConnectFlakyHTTPConnection:
    connect_calls = 0
    bodies = []

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def connect(self):
        type(self).connect_calls += 1
        if type(self).connect_calls == 1:
            raise ConnectionRefusedError("connection refused")

    def request(self, method, path, body=None, headers=None):
        self.method = method
        self.path = path
        self.body = body
        self.headers = headers or {}
        type(self).bodies.append(body)

    def getresponse(self):
        return FakeResponse()


class SingleReadBytesIO(io.BytesIO):
    def __init__(self, initial_bytes: bytes):
        super().__init__(initial_bytes)
        self.read_calls = 0

    def read(self, size=-1):
        self.read_calls += 1
        if self.read_calls > 1:
            raise AssertionError("request body should be buffered and read once")
        return super().read(size)


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
             mock.patch.object(main, "project_config_path", return_value=None), \
             mock.patch.object(main, "global_config_path", return_value=main.Path("/tmp/missing-wingpu.local.toml")):
            with self.assertRaisesRegex(WingpuError, "wingpu.local.toml"):
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

    def test_active_config_prefers_env_config_then_global_config_then_project_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = main.Path(tmpdir)
            env_config = tmp_path / "custom" / "wingpu.local.toml"
            project_config = tmp_path / "project" / "wingpu.local.toml"
            global_config = tmp_path / "home" / ".config" / "wingpu" / "wingpu.local.toml"
            env_config.parent.mkdir(parents=True)
            project_config.parent.mkdir(parents=True)
            global_config.parent.mkdir(parents=True)
            env_config.write_text("[connection]\nhost = \"env-gpu\"\n", encoding="utf-8")
            project_config.write_text("[connection]\nhost = \"project-gpu\"\n", encoding="utf-8")
            global_config.write_text("[connection]\nhost = \"home-gpu\"\n", encoding="utf-8")

            with mock.patch.object(main, "env_config_path", return_value=env_config), \
                 mock.patch.object(main, "project_config_path", return_value=project_config), \
                 mock.patch.object(main, "global_config_path", return_value=global_config):
                self.assertEqual(main.active_local_config_path(), env_config)

            with mock.patch.object(main, "env_config_path", return_value=None), \
                 mock.patch.object(main, "project_config_path", return_value=project_config), \
                 mock.patch.object(main, "global_config_path", return_value=global_config):
                self.assertEqual(main.active_local_config_path(), global_config)

                global_config.unlink()

                self.assertEqual(main.active_local_config_path(), project_config)

    def test_load_settings_uses_global_config_when_project_config_also_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = main.Path(tmpdir)
            project_config = tmp_path / "project" / "wingpu.local.toml"
            global_config = tmp_path / "home" / ".config" / "wingpu" / "wingpu.local.toml"
            project_config.parent.mkdir(parents=True)
            global_config.parent.mkdir(parents=True)
            project_config.write_text("[connection]\nhost = \"project-gpu\"\n", encoding="utf-8")
            global_config.write_text("[connection]\nhost = \"home-gpu\"\n", encoding="utf-8")

            with mock.patch.object(main, "env_config_path", return_value=None), \
                 mock.patch.object(main, "project_config_path", return_value=project_config), \
                 mock.patch.object(
                     main,
                     "global_config_path",
                     side_effect=lambda filename=main.PROJECT_CONFIG_FILENAME: global_config.parent / filename,
                 ):
                settings = load_settings()

        self.assertEqual(settings.connection.host, "home-gpu")
        self.assertEqual(settings.active_config_file, str(global_config))

    def test_load_settings_uses_global_config_when_project_config_is_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            global_config = main.Path(tmpdir) / "wingpu" / "wingpu.local.toml"
            global_config.parent.mkdir(parents=True)
            global_config.write_text("[connection]\nhost = \"home-gpu\"\n", encoding="utf-8")

            with mock.patch.object(main, "env_config_path", return_value=None), \
                 mock.patch.object(main, "project_config_path", return_value=None), \
                 mock.patch.object(
                     main,
                     "global_config_path",
                     side_effect=lambda filename=main.PROJECT_CONFIG_FILENAME: global_config.parent / filename,
                 ):
                settings = load_settings()

        self.assertEqual(settings.connection.host, "home-gpu")
        self.assertEqual(settings.active_config_file, str(global_config))

    def test_remote_runtime_paths_expand_home_relative_state_dir(self):
        settings = make_settings()

        self.assertEqual(remote_runtime_base_dir(settings), "/home/czy/.gpu-bridge")
        self.assertEqual(remote_runtime_pid_file(settings, "turboquant-cuda"), "/home/czy/.gpu-bridge/run/turboquant-cuda.pid")
        self.assertEqual(remote_runtime_log_file(settings, "turboquant-cuda"), "/home/czy/.gpu-bridge/logs/turboquant-cuda.log")

    def test_backend_tunnel_auto_target_uses_wsl_guest_ip(self):
        settings = make_settings()
        with mock.patch.object(main, "run", return_value=mock.Mock(stdout="172.18.59.242 172.17.0.1\n")) as run:
            self.assertEqual(main.backend_tunnel_target_host(settings), "172.18.59.242")

        self.assertIn("hostname", run.call_args.args[0])

    def test_config_global_path_does_not_require_loaded_settings(self):
        target = main.Path("/tmp/wingpu/wingpu.local.toml")
        with mock.patch.object(main, "global_config_path", return_value=target), \
             mock.patch.object(main, "load_settings", side_effect=AssertionError("should not load settings")), \
             mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
            result = main.main(["config", "path", "--global"])

        self.assertEqual(result, 0)
        self.assertEqual(stdout.getvalue().strip(), str(target))

    def test_save_catalog_preserves_config_driven_model_root_and_strips_derived_paths(self):
        catalog = {
            "model_root": "/home/czy/models/Qwen",
            "default_model": "Qwen3.6-35B-A3B-UD-IQ3_S",
            "models": {
                "Qwen3.6-35B-A3B-UD-IQ3_S": {
                    "gguf_relpath": "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
                    "gguf_path": "/home/czy/models/Qwen/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
                    "context_length": 131072,
                    "enabled": True,
                }
            },
        }
        target = io_path() / "qwen_gguf_catalog.json"
        target.parent.mkdir(parents=True, exist_ok=True)

        with mock.patch.object(main, "writable_catalog_path", return_value=target):
            main.save_catalog(catalog, make_settings())

        saved = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(saved["model_root"], "{remote_models_root}")
        self.assertNotIn("gguf_path", saved["models"]["Qwen3.6-35B-A3B-UD-IQ3_S"])
        self.assertEqual(
            saved["models"]["Qwen3.6-35B-A3B-UD-IQ3_S"]["gguf_relpath"],
            "Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf",
        )

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

    def test_proxy_retries_get_once_after_remote_disconnect(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "GET"
        handler.path = "/v1/models"
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

    def test_proxy_does_not_retry_post_after_remote_disconnect(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        payload = b'{"messages":[{"role":"user","content":"hi"}]}'
        handler.headers = {"Content-Length": str(len(payload)), "Authorization": "Bearer sk-local"}
        handler.rfile = SingleReadBytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None
        InspectingHTTPConnection.calls = 0
        InspectingHTTPConnection.bodies = []

        with mock.patch.object(main.http.client, "HTTPConnection", InspectingHTTPConnection):
            handler._proxy()

        self.assertEqual(handler.rfile.read_calls, 1)
        self.assertEqual(InspectingHTTPConnection.bodies, [payload])
        self.assertEqual(coordinator.ensure_calls, 1)
        self.assertEqual(coordinator.recover_calls, [])
        self.assertEqual(sent["status"][0][0], 502)
        self.assertIn(b'"error"', handler.wfile.getvalue())
        self.assertIn(b'request was not retried automatically to avoid duplicate execution', handler.wfile.getvalue())

    def test_proxy_retries_post_when_backend_connect_fails_before_send(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        payload = b'{"messages":[{"role":"user","content":"hi"}]}'
        handler.headers = {"Content-Length": str(len(payload)), "Authorization": "Bearer sk-local"}
        handler.rfile = SingleReadBytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": [], "headers": []}
        handler.send_response = lambda status, reason=None: sent["status"].append((status, reason))
        handler.send_header = lambda key, value: sent["headers"].append((key, value))
        handler.end_headers = lambda: None
        handler._send_json = lambda status, payload: self.fail(f"unexpected json error {status}: {payload}")
        ConnectFlakyHTTPConnection.connect_calls = 0
        ConnectFlakyHTTPConnection.bodies = []

        with mock.patch.object(main.http.client, "HTTPConnection", ConnectFlakyHTTPConnection):
            handler._proxy()

        self.assertEqual(handler.rfile.read_calls, 1)
        self.assertEqual(ConnectFlakyHTTPConnection.connect_calls, 2)
        self.assertEqual(ConnectFlakyHTTPConnection.bodies, [payload])
        self.assertEqual(coordinator.ensure_calls, 2)
        self.assertEqual(coordinator.recover_calls, ["ConnectionRefusedError"])
        self.assertEqual(sent["status"][0][0], 200)

    def test_setup_applies_client_read_timeout(self):
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": DummyCoordinator()})()
        connection = mock.Mock()
        handler.connection = connection
        handler.rfile = io.BytesIO()
        handler.wfile = io.BytesIO()

        with mock.patch.object(main.BaseHTTPRequestHandler, "setup", autospec=True, side_effect=lambda _self: None):
            handler.setup()

        connection.settimeout.assert_called_once_with(handler.coordinator.settings.gateway.client_read_timeout_seconds)

    def test_load_settings_rejects_managed_extra_server_args(self):
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
        local_override = b'''[connection]\nhost = "win-gpu"\n[runtime_defaults]\nextra_server_args = ["--parallel", "2", "-c", "262144"]\n'''
        with mock.patch.object(main, "read_config_bytes", return_value=config_text), \
             mock.patch.object(main, "project_config_path", return_value=main.Path("/tmp/wingpu.local.toml")), \
             mock.patch("pathlib.Path.exists", return_value=True), \
             mock.patch("pathlib.Path.open", mock.mock_open(read_data=local_override)):
            with self.assertRaisesRegex(WingpuError, "managed by wingpu"):
                load_settings()

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
