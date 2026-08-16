import http.client
import io
import json
import os
import tempfile
import time
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
        self.tunnel_refresh_calls = 0

    def begin_request(self):
        self.begin_calls += 1

    def end_request(self):
        self.end_calls += 1

    def ensure_runtime_loaded(self):
        self.ensure_calls += 1

    def recover_runtime_after_proxy_error(self, exc):
        self.recover_calls.append(type(exc).__name__)

    def refresh_backend_tunnel(self):
        self.tunnel_refresh_calls += 1


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

    def read1(self, _size=-1):
        return self.read(_size)


class ChunkedFakeResponse:
    """Mimics an SSE body arriving as separate chunks, as HTTPResponse.read1 would."""

    def __init__(self, chunks, status=200, reason="OK"):
        self.chunks = list(chunks)
        self.status = status
        self.reason = reason
        self.read_calls = 0
        self.read1_calls = 0

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]

    def read(self, _size=-1):
        # read(amt) on a chunked body coalesces: it blocks until amt bytes or EOF.
        self.read_calls += 1
        if not self.chunks:
            return b""
        joined = b"".join(self.chunks)
        self.chunks = []
        return joined

    def read1(self, _size=-1):
        self.read1_calls += 1
        return self.chunks.pop(0) if self.chunks else b""


class StreamingHTTPConnection:
    response = None

    def __init__(self, host, port, timeout=None):
        self.host = host
        self.port = port
        self.timeout = timeout

    def connect(self):
        pass

    def request(self, method, path, body=None, headers=None):
        pass

    def getresponse(self):
        return type(self).response

    def close(self):
        pass


class BrokenPipeWriter(io.BytesIO):
    """A client socket that goes away after the first body write."""

    def __init__(self, fail_after=1):
        super().__init__()
        self.fail_after = fail_after
        self.writes = 0

    def write(self, data):
        self.writes += 1
        if self.writes > self.fail_after:
            raise BrokenPipeError("client went away")
        return super().write(data)


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
default_runtime = "upstream"
served_model_name = "qwen-local"
n_gpu_layers = 99
threads = 8
startup_timeout_seconds = 240
build_jobs = 8
cuda_architectures = "89"
flash_attn = true
remote_state_dir = "~/.gpu-bridge"
default_cache_type_k = "q4_0"
default_cache_type_v = "q4_0"
cmake_args = []
build_targets = ["llama-server", "llama-bench"]
extra_server_args = []

[runtimes.upstream]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]

[runtimes."upstream-mtp"]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]
extra_server_args = ["--spec-type", "draft-mtp"]

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
default_runtime = "upstream"
served_model_name = "qwen-local"
n_gpu_layers = 99
threads = 8
startup_timeout_seconds = 240
build_jobs = 8
cuda_architectures = "89"
flash_attn = true
remote_state_dir = "~/.gpu-bridge"
default_cache_type_k = "q4_0"
default_cache_type_v = "q4_0"
cmake_args = []
build_targets = ["llama-server", "llama-bench"]
extra_server_args = []

[runtimes.upstream]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]

[runtimes."upstream-mtp"]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]
extra_server_args = ["--spec-type", "draft-mtp"]

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
        self.assertEqual(remote_runtime_pid_file(settings, "upstream"), "/home/czy/.gpu-bridge/run/upstream.pid")
        self.assertEqual(remote_runtime_log_file(settings, "upstream"), "/home/czy/.gpu-bridge/logs/upstream.log")
        # each lane gets its own pid/log file, so two lanes never fight over one pidfile
        self.assertEqual(remote_runtime_pid_file(settings, "upstream-mtp"), "/home/czy/.gpu-bridge/run/upstream-mtp.pid")
        self.assertEqual(remote_runtime_log_file(settings, "upstream-mtp"), "/home/czy/.gpu-bridge/logs/upstream-mtp.log")

    def test_backend_relay_uses_ssh_stdio_to_wsl_nc_not_nat_forward(self):
        # The relay must carry bytes over the ssh channel to a WSL-local nc, never an
        # `ssh -L` forward into the WSL NAT IP (that path drops long connections at ~15s).
        settings = make_settings()
        cmd = main._relay_backend_command(settings)
        self.assertEqual(cmd[0], "ssh")
        self.assertEqual(cmd[-2], settings.connection.host)
        self.assertIn(f"nc 127.0.0.1 {settings.connection.remote_port}", cmd[-1])
        self.assertIn("wsl", cmd[-1])
        self.assertNotIn("-L", cmd)  # not a TCP port-forward
        self.assertIn("ControlMaster=auto", cmd)  # multiplexed for cheap per-request ssh

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
            "force_gpu": False,
        })])

    def test_ensure_runtime_loaded_does_not_fast_path_when_runtime_process_is_missing(self):
        settings = make_settings()
        with mock.patch.object(main, "check_command"), \
             mock.patch.object(main, "catalog_entry", return_value={"context_length": 4096}), \
             mock.patch.object(main, "runtime_lane"), \
             mock.patch.object(main, "supported_cache_types", return_value=["q4_0"]), \
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
                runtime_id="upstream-mtp",
                cache_type_k="q4_0",
                cache_type_v="q4_0",
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

    def test_proxy_streams_incrementally_instead_of_buffering_the_body(self):
        # read(64k) on a chunked body blocks until 64KB accumulates, which held every SSE
        # token back until generation finished. The proxy must use read1().
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        handler.send_response = lambda status, reason=None: None
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        chunks = [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]
        response = ChunkedFakeResponse(chunks)
        StreamingHTTPConnection.response = response

        with mock.patch.object(main.http.client, "HTTPConnection", StreamingHTTPConnection):
            handler._proxy()

        self.assertEqual(response.read_calls, 0)
        # one call per chunk plus the terminating empty read
        self.assertEqual(response.read1_calls, len(chunks) + 1)
        self.assertEqual(handler.wfile.getvalue(), b"".join(chunks))

    def test_proxy_does_not_write_an_error_response_after_headers_were_sent(self):
        # Writing _send_json here would push a second HTTP status line into the middle of
        # the body the client is already reading, and raise BrokenPipeError out of the
        # handler as a socketserver traceback.
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Length": "0"}
        handler.rfile = io.BytesIO(b"")
        handler.wfile = BrokenPipeWriter(fail_after=1)
        handler.close_connection = False
        handler.send_response = lambda status, reason=None: None
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        handler._send_json = lambda status, payload: self.fail(
            f"error response written after headers: {status} {payload}"
        )
        handler.log_message = lambda fmt, *args: None
        StreamingHTTPConnection.response = ChunkedFakeResponse(
            [b"data: a\n\n", b"data: b\n\n", b"data: [DONE]\n\n"]
        )

        with mock.patch.object(main.http.client, "HTTPConnection", StreamingHTTPConnection):
            handler._proxy()  # must not raise

        self.assertTrue(handler.close_connection)
        self.assertTrue(handler._response_started)
        self.assertEqual(coordinator.recover_calls, [])
        self.assertEqual(coordinator.tunnel_refresh_calls, 0)
        self.assertEqual(coordinator.end_calls, 1)

    def test_proxy_rebuilds_the_relay_before_reloading_the_model_on_connect_failure(self):
        # A refused connect is usually a dead ssh relay. Rebuilding it costs ~1s;
        # recover_runtime_after_proxy_error force-restarts a healthy 27B for ~40s.
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
        self.assertEqual(coordinator.tunnel_refresh_calls, 1)
        self.assertEqual(coordinator.recover_calls, [])
        self.assertEqual(sent["status"][0][0], 200)

    def test_proxy_escalates_to_runtime_recovery_when_relay_rebuild_does_not_help(self):
        coordinator = DummyCoordinator()
        handler = main.GatewayRequestHandler.__new__(main.GatewayRequestHandler)
        handler.server = type("Server", (), {"coordinator": coordinator})()
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        payload = b'{"messages":[{"role":"user","content":"hi"}]}'
        handler.headers = {"Content-Length": str(len(payload))}
        handler.rfile = SingleReadBytesIO(payload)
        handler.wfile = io.BytesIO()
        handler.close_connection = False
        sent = {"status": []}
        handler.send_response = lambda status, reason=None: sent["status"].append(status)
        handler.send_header = lambda key, value: None
        handler.end_headers = lambda: None
        handler._send_json = lambda status, payload: self.fail(f"unexpected json error {status}: {payload}")
        class TwiceRefusedConnection(ConnectFlakyHTTPConnection):
            """Refuses the first connect and the one after the relay rebuild."""

            def connect(self):
                type(self).connect_calls += 1
                if type(self).connect_calls < 3:
                    raise ConnectionRefusedError("connection refused")

        TwiceRefusedConnection.connect_calls = 0
        TwiceRefusedConnection.bodies = []

        with mock.patch.object(main.http.client, "HTTPConnection", TwiceRefusedConnection):
            handler._proxy()

        self.assertEqual(TwiceRefusedConnection.connect_calls, 3)
        self.assertEqual(coordinator.tunnel_refresh_calls, 1)
        self.assertEqual(coordinator.recover_calls, ["ConnectionRefusedError"])
        self.assertEqual(sent["status"][0], 200)

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
default_runtime = "upstream"
served_model_name = "qwen-local"
n_gpu_layers = 99
threads = 8
startup_timeout_seconds = 240
build_jobs = 8
cuda_architectures = "89"
flash_attn = true
remote_state_dir = "~/.gpu-bridge"
default_cache_type_k = "q4_0"
default_cache_type_v = "q4_0"
cmake_args = []
build_targets = ["llama-server", "llama-bench"]
extra_server_args = []

[runtimes.upstream]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]

[runtimes."upstream-mtp"]
kind = "native"
source_dir = "/home/czy/src/llama.cpp"
build_dir = "/home/czy/src/llama.cpp/build-cuda89"
server_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-server"
bench_bin = "/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench"
supported_cache_types = ["q4_0"]
extra_server_args = ["--spec-type", "draft-mtp"]

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
             mock.patch.object(main, "selected_runtime", return_value="upstream"), \
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

    def test_stale_gateway_self_shuts_down_and_does_not_offload(self):
        # A gateway whose pid no longer owns the pidfile must NOT offload (SIGTERM llama)
        # and must shut itself down, so duplicate gateways cannot fight over the GPU.
        settings = make_settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        settings.state.gateway_pid_path.write_text("999999\n", encoding="utf-8")
        with mock.patch.object(main, "runtime_process_info", return_value={"running": True}):
            coord = main.GatewayCoordinator(settings)
        coord.runtime_loaded = True
        coord.last_request_finished_at = 1.0  # ancient -> would look idle if it were active
        coord.server = mock.Mock()
        with mock.patch.object(main, "stop_remote_runtime") as stop_rt:
            coord.maybe_idle_offload()
        stop_rt.assert_not_called()
        self.assertTrue(coord.stop_event.is_set())

    def test_active_gateway_may_offload_when_idle(self):
        settings = make_settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        settings.state.gateway_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        with mock.patch.object(main, "runtime_process_info", return_value={"running": True}):
            coord = main.GatewayCoordinator(settings)
        coord.runtime_loaded = True
        coord.last_request_finished_at = 1.0  # ancient -> idle
        with mock.patch.object(main, "stop_remote_runtime") as stop_rt, \
             mock.patch.object(main, "stop_backend_tunnel"), \
             mock.patch.object(main, "release_gpu_lease"), \
             mock.patch.object(coord, "write_state"):
            coord.maybe_idle_offload()
        stop_rt.assert_called_once()
        self.assertFalse(coord.stop_event.is_set())

    def test_coordinator_init_does_no_remote_io(self):
        # __init__ runs before the listening socket is bound; an ssh round trip here makes
        # `wingpu gateway start` report "did not become ready" whenever the host is asleep.
        settings = make_settings()
        with mock.patch.object(main, "runtime_process_info", side_effect=AssertionError("remote probe in __init__")):
            coord = main.GatewayCoordinator(settings)
        self.assertFalse(coord.runtime_loaded)
        self.assertEqual(coord.idle_status, "runtime_unknown")

    def test_probe_runtime_state_fills_in_the_initial_belief(self):
        settings = make_settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        coord = main.GatewayCoordinator(settings)
        with mock.patch.object(main, "runtime_process_info", return_value={"running": True}), \
             mock.patch.object(main, "selected_runtime", return_value="upstream"), \
             mock.patch.object(coord, "write_state"):
            coord.probe_runtime_state()
        self.assertTrue(coord.runtime_loaded)
        self.assertEqual(coord.idle_status, "runtime_loaded")

    def test_probe_runtime_state_does_not_overwrite_a_request_that_already_loaded(self):
        settings = make_settings()
        coord = main.GatewayCoordinator(settings)
        coord.runtime_loaded = True
        coord.last_runtime_start_at = time.time()
        with mock.patch.object(main, "runtime_process_info", return_value={"running": False}), \
             mock.patch.object(main, "selected_runtime", return_value="upstream"):
            coord.probe_runtime_state()
        self.assertTrue(coord.runtime_loaded)

    def test_ensure_runtime_loaded_skips_remote_probes_while_recently_verified(self):
        settings = make_settings()
        coord = main.GatewayCoordinator(settings)
        with mock.patch.object(main, "ensure_runtime_loaded") as ensure, \
             mock.patch.object(main, "selected_model", return_value="m"), \
             mock.patch.object(main, "selected_runtime", return_value="upstream"), \
             mock.patch.object(main, "selected_cache_type", return_value="q4_0"), \
             mock.patch.object(coord, "write_state"):
            coord.ensure_runtime_loaded()
            self.assertEqual(ensure.call_count, 1)
            coord.ensure_runtime_loaded()
            coord.ensure_runtime_loaded()
            self.assertEqual(ensure.call_count, 1)  # hot path: no further ssh work

            # Expired TTL falls back to the full check.
            coord.last_runtime_verified_at = time.time() - (main.RUNTIME_VERIFY_TTL_SECONDS + 1)
            coord.ensure_runtime_loaded()
            self.assertEqual(ensure.call_count, 2)

    def test_hot_path_is_disabled_while_the_runtime_is_being_offloaded(self):
        settings = make_settings()
        coord = main.GatewayCoordinator(settings)
        coord.runtime_loaded = True
        coord.last_runtime_verified_at = time.time()
        self.assertTrue(coord._runtime_recently_verified())
        coord.offloading = True
        self.assertFalse(coord._runtime_recently_verified())

    def test_idle_offload_publishes_the_teardown_under_the_state_lock(self):
        settings = make_settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        settings.state.gateway_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
        coord = main.GatewayCoordinator(settings)
        coord.runtime_loaded = True
        coord.last_runtime_verified_at = time.time()
        coord.last_request_finished_at = 1.0  # ancient -> idle
        seen = {}

        def _stop(_settings):
            seen["offloading"] = coord.offloading
            seen["hot_path"] = coord._runtime_recently_verified()

        with mock.patch.object(main, "stop_remote_runtime", side_effect=_stop), \
             mock.patch.object(main, "stop_backend_tunnel"), \
             mock.patch.object(main, "release_gpu_lease"), \
             mock.patch.object(coord, "write_state"):
            coord.maybe_idle_offload()

        self.assertTrue(seen["offloading"])
        self.assertFalse(seen["hot_path"])
        self.assertFalse(coord.offloading)
        self.assertFalse(coord.runtime_loaded)

    def test_control_plane_ssh_multiplexes_over_a_shared_master(self):
        settings = make_settings()
        settings.state.state_dir.mkdir(parents=True, exist_ok=True)
        args = main.ssh_base_args(settings)
        self.assertIn("ControlMaster=auto", args)
        self.assertIn(f"ControlPath={settings.ssh_control_socket}", args)
        self.assertIn(f"ControlPersist={main.SSH_CONTROL_PERSIST_SECONDS}", args)
        # The relay owns a different socket; sharing one would let a control-plane
        # `ssh -O exit` tear down the live backend transport.
        self.assertNotEqual(settings.ssh_control_socket, settings.backend_tunnel_socket)

    def test_stop_backend_tunnel_targets_the_relay_socket_not_the_control_master(self):
        settings = make_settings()
        captured = []
        with mock.patch.object(main, "run", side_effect=lambda argv, **kw: captured.append(argv)):
            main.stop_backend_tunnel(settings)
        argv = captured[0]
        self.assertIn("-S", argv)
        self.assertEqual(argv[argv.index("-S") + 1], str(settings.backend_tunnel_socket))
        self.assertNotIn("ControlMaster=auto", argv)


def make_settings():
    state = StateConfig(state_dir=io_path())
    return Settings(
        connection=ConnectionConfig(api_key="sk-local"),
        gateway=GatewayConfig(),
        paths=PathsConfig(remote_home="/home/czy", remote_src_root="/home/czy/src", remote_models_root="/home/czy/models/Qwen"),
        runtime_defaults=RuntimeDefaults(default_runtime="upstream", served_model_name="qwen-local"),
        # Two lanes so lane-aware code paths (per-lane pid/log files, cache-type
        # validation, extra_server_args) are exercised against a real multi-lane config.
        runtimes={
            "upstream": RuntimeLane(
                kind="native",
                source_dir="/home/czy/src/llama.cpp",
                build_dir="/home/czy/src/llama.cpp/build-cuda89",
                server_bin="/home/czy/src/llama.cpp/build-cuda89/bin/llama-server",
                bench_bin="/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench",
                supported_cache_types=["q4_0"],
            ),
            "upstream-mtp": RuntimeLane(
                kind="native",
                source_dir="/home/czy/src/llama.cpp",
                build_dir="/home/czy/src/llama.cpp/build-cuda89",
                server_bin="/home/czy/src/llama.cpp/build-cuda89/bin/llama-server",
                bench_bin="/home/czy/src/llama.cpp/build-cuda89/bin/llama-bench",
                supported_cache_types=["q4_0"],
                extra_server_args=["--spec-type", "draft-mtp"],
            ),
        },
        state=state,
    )


def io_path():
    # tests don't touch disk because selected_* helpers are patched in the exercised paths
    return main.Path('/tmp/wingpu-tests')


if __name__ == "__main__":
    unittest.main()
