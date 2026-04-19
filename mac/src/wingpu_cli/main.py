from __future__ import annotations

import argparse
import http.client
import json
import os
import posixpath
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import tomllib
import urllib.error
import urllib.request
from importlib import resources as importlib_resources
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BRIDGE_DIR = Path(__file__).resolve().parents[3]
PROJECT_CONFIG_FILENAME = "wingpu.local.toml"

CACHE_TYPE_ALIASES = {
    "turbo2_0": "turbo2",
    "turbo3_0": "turbo3",
    "turbo4_0": "turbo4",
}


class WingpuError(RuntimeError):
    pass


@dataclass(slots=True)
class ConnectionConfig:
    host: str = "gpu-host"
    distro: str = "Ubuntu"
    api_key: str = "local-dev-key"
    local_port: int = 8000
    remote_port: int = 8000
    ssh_connect_timeout: int = 8
    server_alive_interval: int = 30
    server_alive_count_max: int = 3


@dataclass(slots=True)
class RuntimeDefaults:
    default_runtime: str = "upstream"
    served_model_name: str = "qwen-local"
    n_gpu_layers: int = 99
    threads: int = 8
    startup_timeout_seconds: int = 240
    build_jobs: int = 8
    cuda_architectures: str = "89"
    flash_attn: bool = True
    remote_state_dir: str = "~/.gpu-bridge"
    default_cache_type_k: str = "f16"
    default_cache_type_v: str = "f16"
    cmake_args: list[str] = field(default_factory=list)
    build_targets: list[str] = field(default_factory=lambda: ["llama-server", "llama-bench"])
    extra_server_args: list[str] = field(default_factory=list)


@dataclass(slots=True)
class GatewayConfig:
    listen_host: str = "127.0.0.1"
    backend_host: str = "127.0.0.1"
    backend_local_port: int = 18000
    idle_offload_enabled: bool = True
    idle_timeout_seconds: int = 1800
    idle_poll_seconds: int = 5
    request_timeout_seconds: int = 1800
    restart_mode: str = "on_demand"


@dataclass(slots=True)
class PathsConfig:
    remote_home: str = "/home/your-wsl-user"
    remote_src_root: str = "/home/your-wsl-user/src"
    remote_models_root: str = "/home/your-wsl-user/models/Qwen"


@dataclass(slots=True)
class RuntimeLane:
    kind: str
    source_dir: str
    build_dir: str
    server_bin: str
    bench_bin: str
    default_branch: str = "master"
    supported_cache_types: list[str] = field(default_factory=list)
    extra_server_args: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass(slots=True)
class StateConfig:
    state_dir: Path = Path("~/.gpu-chat-bridge").expanduser()
    selected_model_file: str = "selected_model"
    selected_runtime_file: str = "selected_runtime"
    selected_cache_type_k_file: str = "selected_cache_type_k"
    selected_cache_type_v_file: str = "selected_cache_type_v"
    benchmark_dir_name: str = "benchmarks"
    restart_on_model_set: bool = False
    gateway_pid_file: str = "gateway.pid"
    gateway_log_file: str = "gateway.log"
    gateway_state_file: str = "gateway_state.json"
    gateway_lock_file: str = "gateway.lock"

    @property
    def selected_model_path(self) -> Path:
        return self.state_dir / self.selected_model_file

    @property
    def selected_runtime_path(self) -> Path:
        return self.state_dir / self.selected_runtime_file

    @property
    def selected_cache_type_k_path(self) -> Path:
        return self.state_dir / self.selected_cache_type_k_file

    @property
    def selected_cache_type_v_path(self) -> Path:
        return self.state_dir / self.selected_cache_type_v_file

    @property
    def benchmark_dir(self) -> Path:
        return self.state_dir / self.benchmark_dir_name

    @property
    def gateway_pid_path(self) -> Path:
        return self.state_dir / self.gateway_pid_file

    @property
    def gateway_log_path(self) -> Path:
        return self.state_dir / self.gateway_log_file

    @property
    def gateway_state_path(self) -> Path:
        return self.state_dir / self.gateway_state_file

    @property
    def gateway_lock_path(self) -> Path:
        return self.state_dir / self.gateway_lock_file


@dataclass(slots=True)
class Settings:
    connection: ConnectionConfig
    gateway: GatewayConfig
    paths: PathsConfig
    runtime_defaults: RuntimeDefaults
    runtimes: dict[str, RuntimeLane]
    state: StateConfig
    catalog_file: str = "package://wingpu_cli.resources/qwen_gguf_catalog.json"
    defaults_file: str = "package://wingpu_cli.resources/wingpu.defaults.toml"
    project_config_file: str | None = None

    @property
    def backend_tunnel_socket(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.gateway.backend_local_port}.sock"

    @property
    def backend_tunnel_pid_file(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.gateway.backend_local_port}.pid"

    @property
    def backend_tunnel_log_file(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.gateway.backend_local_port}.log"


def merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge_dicts(base[key], value)
        else:
            merged[key] = value
    return merged


class SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def interpolate_value(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format_map(SafeFormatDict(variables))
    if isinstance(value, list):
        return [interpolate_value(item, variables) for item in value]
    if isinstance(value, dict):
        return {key: interpolate_value(item, variables) for key, item in value.items()}
    return value


def resolve_paths_config(raw_paths: dict[str, Any]) -> dict[str, str]:
    resolved = {key: str(value) for key, value in raw_paths.items()}
    for _ in range(8):
        updated = {key: value.format_map(SafeFormatDict(resolved)) for key, value in resolved.items()}
        if updated == resolved:
            break
        resolved = updated
    return resolved


@lru_cache(maxsize=1)
def discover_bridge_dir() -> Path | None:
    project_override = os.getenv("WINGPU_PROJECT_DIR")
    if project_override:
        override = Path(project_override).expanduser().resolve()
        direct = override
        nested = override / "bridge"
        if (direct / "config" / "wingpu.defaults.toml").exists() and (direct / "mac" / "pyproject.toml").exists():
            return direct
        if (nested / "config" / "wingpu.defaults.toml").exists() and (nested / "mac" / "pyproject.toml").exists():
            return nested
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        direct = base
        nested = base / "bridge"
        if (direct / "config" / "wingpu.defaults.toml").exists() and (direct / "mac" / "pyproject.toml").exists():
            return direct
        if (nested / "config" / "wingpu.defaults.toml").exists() and (nested / "mac" / "pyproject.toml").exists():
            return nested
    if (BRIDGE_DIR / "config" / "wingpu.defaults.toml").exists():
        return BRIDGE_DIR
    return None


def repo_config_dir() -> Path | None:
    bridge_dir = discover_bridge_dir()
    if bridge_dir is None:
        return None
    return bridge_dir / "config"


def config_file_path(filename: str) -> Path | None:
    config_dir = repo_config_dir()
    if config_dir is None:
        return None
    candidate = config_dir / filename
    return candidate if candidate.exists() else None


def config_source_label(filename: str) -> str:
    candidate = config_file_path(filename)
    if candidate is not None:
        return str(candidate)
    return f"package://wingpu_cli.resources/{filename}"


def read_config_bytes(filename: str) -> bytes:
    candidate = config_file_path(filename)
    if candidate is not None:
        return candidate.read_bytes()
    return importlib_resources.files("wingpu_cli.resources").joinpath(filename).read_bytes()


def project_config_path() -> Path | None:
    config_dir = repo_config_dir()
    if config_dir is None:
        return None
    return config_dir / PROJECT_CONFIG_FILENAME


def load_settings(host: str | None = None, distro: str | None = None, api_key: str | None = None) -> Settings:
    config = tomllib.loads(read_config_bytes("wingpu.defaults.toml").decode("utf-8"))
    local_project_config = project_config_path()
    if local_project_config is not None and local_project_config.exists():
        with local_project_config.open("rb") as handle:
            config = merge_dicts(config, tomllib.load(handle))

    env_overrides: dict[str, Any] = {"connection": {}, "gateway": {}, "runtime_defaults": {}, "paths": {}}
    if os.getenv("WINGPU_HOST"):
        env_overrides["connection"]["host"] = os.environ["WINGPU_HOST"]
    if os.getenv("WINGPU_DISTRO"):
        env_overrides["connection"]["distro"] = os.environ["WINGPU_DISTRO"]
    if os.getenv("API_KEY"):
        env_overrides["connection"]["api_key"] = os.environ["API_KEY"]
    if os.getenv("LOCAL_PORT"):
        env_overrides["connection"]["local_port"] = int(os.environ["LOCAL_PORT"])
    if os.getenv("REMOTE_PORT"):
        env_overrides["connection"]["remote_port"] = int(os.environ["REMOTE_PORT"])
    if os.getenv("WINGPU_GATEWAY_LISTEN_HOST"):
        env_overrides["gateway"]["listen_host"] = os.environ["WINGPU_GATEWAY_LISTEN_HOST"]
    if os.getenv("WINGPU_GATEWAY_BACKEND_HOST"):
        env_overrides["gateway"]["backend_host"] = os.environ["WINGPU_GATEWAY_BACKEND_HOST"]
    if os.getenv("WINGPU_GATEWAY_BACKEND_LOCAL_PORT"):
        env_overrides["gateway"]["backend_local_port"] = int(os.environ["WINGPU_GATEWAY_BACKEND_LOCAL_PORT"])
    if os.getenv("WINGPU_IDLE_OFFLOAD_ENABLED"):
        env_overrides["gateway"]["idle_offload_enabled"] = os.environ["WINGPU_IDLE_OFFLOAD_ENABLED"].lower() in {"1", "true", "yes", "on"}
    if os.getenv("WINGPU_IDLE_TIMEOUT_SECONDS"):
        env_overrides["gateway"]["idle_timeout_seconds"] = int(os.environ["WINGPU_IDLE_TIMEOUT_SECONDS"])
    if os.getenv("WINGPU_IDLE_POLL_SECONDS"):
        env_overrides["gateway"]["idle_poll_seconds"] = int(os.environ["WINGPU_IDLE_POLL_SECONDS"])
    if os.getenv("WINGPU_GATEWAY_REQUEST_TIMEOUT_SECONDS"):
        env_overrides["gateway"]["request_timeout_seconds"] = int(os.environ["WINGPU_GATEWAY_REQUEST_TIMEOUT_SECONDS"])
    if os.getenv("WINGPU_RESTART_MODE"):
        env_overrides["gateway"]["restart_mode"] = os.environ["WINGPU_RESTART_MODE"]
    if os.getenv("SERVED_MODEL_NAME"):
        env_overrides["runtime_defaults"]["served_model_name"] = os.environ["SERVED_MODEL_NAME"]
    if os.getenv("LLAMA_N_GPU_LAYERS"):
        env_overrides["runtime_defaults"]["n_gpu_layers"] = int(os.environ["LLAMA_N_GPU_LAYERS"])
    if os.getenv("LLAMA_THREADS"):
        env_overrides["runtime_defaults"]["threads"] = int(os.environ["LLAMA_THREADS"])
    if os.getenv("LLAMA_EXTRA_ARGS"):
        env_overrides["runtime_defaults"]["extra_server_args"] = shlex.split(os.environ["LLAMA_EXTRA_ARGS"])
    if os.getenv("WINGPU_RUNTIME"):
        env_overrides["runtime_defaults"]["default_runtime"] = os.environ["WINGPU_RUNTIME"]
    if os.getenv("WINGPU_REMOTE_HOME"):
        env_overrides["paths"]["remote_home"] = os.environ["WINGPU_REMOTE_HOME"]
    if os.getenv("WINGPU_REMOTE_SRC_ROOT"):
        env_overrides["paths"]["remote_src_root"] = os.environ["WINGPU_REMOTE_SRC_ROOT"]
    if os.getenv("WINGPU_REMOTE_MODELS_ROOT"):
        env_overrides["paths"]["remote_models_root"] = os.environ["WINGPU_REMOTE_MODELS_ROOT"]
    config = merge_dicts(config, env_overrides)

    if host:
        config["connection"]["host"] = host
    if distro:
        config["connection"]["distro"] = distro
    if api_key:
        config["connection"]["api_key"] = api_key

    paths_resolved = resolve_paths_config(config["paths"])
    config["runtime_defaults"] = interpolate_value(config["runtime_defaults"], paths_resolved)
    config["runtimes"] = interpolate_value(config["runtimes"], paths_resolved)

    state_dir = Path(config["state"]["state_dir"]).expanduser()
    runtimes = {
        runtime_id: RuntimeLane(**runtime_cfg)
        for runtime_id, runtime_cfg in config["runtimes"].items()
    }
    return Settings(
        connection=ConnectionConfig(**config["connection"]),
        gateway=GatewayConfig(**config["gateway"]),
        paths=PathsConfig(**paths_resolved),
        runtime_defaults=RuntimeDefaults(**config["runtime_defaults"]),
        runtimes=runtimes,
        state=StateConfig(
            state_dir=state_dir,
            selected_model_file=config["state"]["selected_model_file"],
            selected_runtime_file=config["state"]["selected_runtime_file"],
            selected_cache_type_k_file=config["state"]["selected_cache_type_k_file"],
            selected_cache_type_v_file=config["state"]["selected_cache_type_v_file"],
            benchmark_dir_name=config["state"]["benchmark_dir_name"],
            restart_on_model_set=bool(config["state"].get("restart_on_model_set", False)),
            gateway_pid_file=config["state"].get("gateway_pid_file", "gateway.pid"),
            gateway_log_file=config["state"].get("gateway_log_file", "gateway.log"),
            gateway_state_file=config["state"].get("gateway_state_file", "gateway_state.json"),
            gateway_lock_file=config["state"].get("gateway_lock_file", "gateway.lock"),
        ),
        catalog_file=config_source_label("qwen_gguf_catalog.json"),
        defaults_file=config_source_label("wingpu.defaults.toml"),
        project_config_file=str(local_project_config) if local_project_config is not None else None,
    )


def load_catalog(settings: Settings) -> dict[str, Any]:
    catalog = json.loads(read_config_bytes("qwen_gguf_catalog.json").decode("utf-8"))
    variables = asdict(settings.paths)
    catalog = interpolate_value(catalog, variables)
    model_root = catalog.get("model_root")
    if model_root:
        for entry in catalog.get("models", {}).values():
            if "gguf_path" not in entry and "gguf_relpath" in entry:
                entry["gguf_path"] = posixpath.join(model_root, entry["gguf_relpath"])
    return catalog


def writable_catalog_path() -> Path:
    candidate = config_file_path("qwen_gguf_catalog.json")
    if candidate is None:
        raise WingpuError(
            "Cannot modify the model catalog because no project-local qwen_gguf_catalog.json was found."
        )
    return candidate


def save_catalog(catalog: dict[str, Any]) -> None:
    target = writable_catalog_path()
    target.write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")


def catalog_default_model(settings: Settings) -> str:
    return load_catalog(settings)["default_model"]


def catalog_entry(settings: Settings, model_name: str) -> dict[str, Any]:
    catalog = load_catalog(settings)
    entry = catalog["models"].get(model_name)
    if not entry or not entry.get("enabled", False):
        raise WingpuError(f"Unknown or disabled model: {model_name}")
    return entry


def runtime_lane(settings: Settings, runtime_id: str) -> RuntimeLane:
    lane = settings.runtimes.get(runtime_id)
    if lane is None:
        valid = ", ".join(sorted(settings.runtimes))
        raise WingpuError(f"Unknown runtime lane: {runtime_id}. Available: {valid}")
    return lane


def default_runtime_id(settings: Settings) -> str:
    default_runtime = settings.runtime_defaults.default_runtime
    if default_runtime not in settings.runtimes:
        raise WingpuError(f"Configured default runtime not found: {default_runtime}")
    return default_runtime


def selected_model(settings: Settings) -> str:
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    path = settings.state.selected_model_path
    if path.exists():
        name = path.read_text(encoding="utf-8").strip()
        if name:
            try:
                catalog_entry(settings, name)
            except WingpuError:
                pass
            else:
                return name
    return catalog_default_model(settings)


def set_selected_model(settings: Settings, model_name: str) -> None:
    catalog_entry(settings, model_name)
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    settings.state.selected_model_path.write_text(f"{model_name}\n", encoding="utf-8")


def set_catalog_default_model(settings: Settings, model_name: str) -> None:
    catalog_entry(settings, model_name)
    catalog = load_catalog(settings)
    catalog["default_model"] = model_name
    save_catalog(catalog)


def set_catalog_context_length(settings: Settings, model_name: str, context_length: int) -> None:
    if context_length <= 0:
        raise WingpuError("Context length must be a positive integer")
    catalog_entry(settings, model_name)
    catalog = load_catalog(settings)
    catalog["models"][model_name]["context_length"] = int(context_length)
    save_catalog(catalog)


def catalog_context_length(settings: Settings, model_name: str) -> int:
    entry = catalog_entry(settings, model_name)
    return int(entry["context_length"])


def selected_runtime(settings: Settings) -> str:
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    path = settings.state.selected_runtime_path
    if path.exists():
        runtime_id = path.read_text(encoding="utf-8").strip()
        if runtime_id in settings.runtimes:
            return runtime_id
    return default_runtime_id(settings)


def set_selected_runtime(settings: Settings, runtime_id: str) -> None:
    runtime_lane(settings, runtime_id)
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    settings.state.selected_runtime_path.write_text(f"{runtime_id}\n", encoding="utf-8")


def supported_cache_types(settings: Settings, runtime_id: str) -> list[str]:
    return runtime_lane(settings, runtime_id).supported_cache_types


def default_cache_type(settings: Settings, kind: str) -> str:
    return settings.runtime_defaults.default_cache_type_k if kind == "k" else settings.runtime_defaults.default_cache_type_v


def normalize_cache_type(value: str) -> str:
    return CACHE_TYPE_ALIASES.get(value, value)


def cache_type_matches(candidate: str, supported_value: str) -> bool:
    return candidate == supported_value or normalize_cache_type(candidate) == normalize_cache_type(supported_value)


def selected_cache_type(settings: Settings, kind: str, runtime_id: str | None = None) -> str:
    runtime_id = runtime_id or selected_runtime(settings)
    path = settings.state.selected_cache_type_k_path if kind == "k" else settings.state.selected_cache_type_v_path
    default = default_cache_type(settings, kind)
    value = default
    if path.exists():
        candidate = path.read_text(encoding="utf-8").strip()
        if candidate:
            value = candidate
    if not any(cache_type_matches(value, supported) for supported in supported_cache_types(settings, runtime_id)):
        return default
    return value


def set_selected_cache_type(settings: Settings, kind: str, value: str, runtime_id: str | None = None) -> None:
    runtime_id = runtime_id or selected_runtime(settings)
    if not any(cache_type_matches(value, supported) for supported in supported_cache_types(settings, runtime_id)):
        valid = ", ".join(supported_cache_types(settings, runtime_id))
        raise WingpuError(f"Unsupported cache type for {runtime_id}: {value}. Allowed: {valid}")
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    path = settings.state.selected_cache_type_k_path if kind == "k" else settings.state.selected_cache_type_v_path
    path.write_text(f"{value}\n", encoding="utf-8")


def remote_model_path(settings: Settings, model_name: str) -> str:
    return catalog_entry(settings, model_name)["gguf_path"]


def check_command(name: str) -> None:
    if not shutil_which(name):
        raise WingpuError(f"Required command not found: {name}")


def shutil_which(name: str) -> str | None:
    from shutil import which

    return which(name)


def run(
    argv: list[str],
    *,
    input_text: str | None = None,
    check: bool = True,
    capture_output: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        argv,
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if check and result.returncode != 0:
        command = " ".join(shlex.quote(part) for part in argv)
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or f"exit code {result.returncode}"
        raise WingpuError(f"Command failed: {command}\n{detail}")
    return result


def ssh_base_args(settings: Settings) -> list[str]:
    return ["ssh", settings.connection.host]


def check_ssh_connectivity(settings: Settings) -> None:
    print(f"[1/4] Checking SSH connectivity to {settings.connection.host}...")
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={settings.connection.ssh_connect_timeout}",
            settings.connection.host,
            "echo ok",
        ]
    )


def run_wsl_script(settings: Settings, script: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    remote_command = f"wsl -d {shlex.quote(settings.connection.distro)} -- env COLUMNS=120 LINES=40 bash -seuo pipefail"
    return run(ssh_base_args(settings) + [remote_command], input_text=script, check=check)


def local_api_json(settings: Settings, path: str) -> dict[str, Any]:
    url = f"http://{settings.gateway.listen_host}:{settings.connection.local_port}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {settings.connection.api_key}"})
    timeout = max(5, settings.gateway.request_timeout_seconds, settings.runtime_defaults.startup_timeout_seconds + 30)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def gateway_admin_json(settings: Settings, path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> dict[str, Any]:
    url = f"http://{settings.gateway.listen_host}:{settings.connection.local_port}/__wingpu{path}"
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def backend_api_json(settings: Settings, path: str) -> dict[str, Any]:
    url = f"http://{settings.gateway.backend_host}:{settings.gateway.backend_local_port}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {settings.connection.api_key}"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def remote_runtime_base_dir(settings: Settings) -> str:
    return settings.runtime_defaults.remote_state_dir


def remote_runtime_pid_file(settings: Settings, runtime_id: str) -> str:
    return f"{remote_runtime_base_dir(settings)}/run/{runtime_id}.pid"


def remote_runtime_log_file(settings: Settings, runtime_id: str) -> str:
    return f"{remote_runtime_base_dir(settings)}/logs/{runtime_id}.log"


def ensure_backend_tunnel(settings: Settings) -> None:
    print(
        f"[3/4] Ensuring backend SSH tunnel localhost:{settings.gateway.backend_local_port} -> "
        f"{settings.connection.host}:127.0.0.1:{settings.connection.remote_port} ..."
    )
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    sock = str(settings.backend_tunnel_socket)
    check_cmd = ["ssh", "-S", sock, "-O", "check", settings.connection.host]
    if run(check_cmd, check=False).returncode == 0:
        print("Managed backend tunnel already running.")
        return

    settings.backend_tunnel_socket.unlink(missing_ok=True)
    settings.backend_tunnel_pid_file.unlink(missing_ok=True)
    if local_port_in_use(settings.gateway.backend_local_port):
        raise WingpuError(f"Local backend port {settings.gateway.backend_local_port} is already in use by another process.")

    start_cmd = [
        "ssh",
        "-fN",
        "-M",
        "-S",
        sock,
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ServerAliveInterval={settings.connection.server_alive_interval}",
        "-o",
        f"ServerAliveCountMax={settings.connection.server_alive_count_max}",
        "-E",
        str(settings.backend_tunnel_log_file),
        "-L",
        f"{settings.gateway.backend_local_port}:127.0.0.1:{settings.connection.remote_port}",
        settings.connection.host,
    ]
    run(start_cmd)
    time.sleep(1)
    if run(check_cmd, check=False).returncode != 0:
        tunnel_log = settings.backend_tunnel_log_file.read_text(encoding="utf-8", errors="ignore")
        raise WingpuError(f"Failed to establish managed backend SSH tunnel.\n{tunnel_log}")
    pid = port_listener_pid(settings.gateway.backend_local_port)
    if pid:
        settings.backend_tunnel_pid_file.write_text(f"{pid}\n", encoding="utf-8")


def stop_backend_tunnel(settings: Settings) -> None:
    print("[1/2] Stopping backend SSH tunnel...")
    run(["ssh", "-S", str(settings.backend_tunnel_socket), "-O", "exit", settings.connection.host], check=False)
    settings.backend_tunnel_socket.unlink(missing_ok=True)
    settings.backend_tunnel_pid_file.unlink(missing_ok=True)


def stop_remote_runtime(settings: Settings) -> None:
    print("[2/2] Stopping remote llama.cpp runtime...")
    pid_cleanup = "\n".join(
        f'''
PID_FILE="{remote_runtime_pid_file(settings, runtime_id)}"
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    kill "$PID" >/dev/null 2>&1 || true
    sleep 1
    kill -9 "$PID" >/dev/null 2>&1 || true
  fi
  rm -f "$PID_FILE"
fi
'''
        for runtime_id in settings.runtimes
    )
    script = f"""
mkdir -p {shlex.quote(remote_runtime_base_dir(settings))}/run {shlex.quote(remote_runtime_base_dir(settings))}/logs
docker rm -f llama-server-qwen >/dev/null 2>&1 || true
{pid_cleanup}
pkill -f '/llama-server .*--port {settings.connection.remote_port}' >/dev/null 2>&1 || true
echo remote_runtime_stopped
"""
    result = run_wsl_script(settings, script, check=False)
    output = (result.stdout or "").strip()
    if output:
        print(output)


def start_remote_runtime(
    settings: Settings,
    runtime_id: str,
    model_name: str,
    cache_type_k: str,
    cache_type_v: str,
    flash_attn: bool,
) -> None:
    lane = runtime_lane(settings, runtime_id)
    model_entry = catalog_entry(settings, model_name)
    model_path = remote_model_path(settings, model_name)
    extra_args = settings.runtime_defaults.extra_server_args + lane.extra_server_args
    extra_args_str = " ".join(shlex.quote(arg) for arg in extra_args)
    pid_file = remote_runtime_pid_file(settings, runtime_id)
    log_file = remote_runtime_log_file(settings, runtime_id)
    flash_attn_value = 1 if flash_attn else 0
    cache_type_k_cli = normalize_cache_type(cache_type_k)
    cache_type_v_cli = normalize_cache_type(cache_type_v)
    print(f"[2/4] Starting native llama.cpp runtime '{runtime_id}' in WSL...")
    script = f"""
mkdir -p {shlex.quote(remote_runtime_base_dir(settings))}/run {shlex.quote(remote_runtime_base_dir(settings))}/logs
SERVER_BIN={shlex.quote(lane.server_bin)}
MODEL_PATH={shlex.quote(model_path)}
PID_FILE={shlex.quote(pid_file)}
LOG_FILE={shlex.quote(log_file)}
if [[ ! -x "$SERVER_BIN" ]]; then
  echo "Missing runtime binary: $SERVER_BIN" >&2
  exit 1
fi
nohup "$SERVER_BIN" \
  -m "$MODEL_PATH" \
  --alias {shlex.quote(settings.runtime_defaults.served_model_name)} \
  --host 0.0.0.0 \
  --port {settings.connection.remote_port} \
  -c {int(model_entry['context_length'])} \
  -ngl {settings.runtime_defaults.n_gpu_layers} \
  -t {settings.runtime_defaults.threads} \
  --jinja \
  --api-key {shlex.quote(settings.connection.api_key)} \
  -ctk {shlex.quote(cache_type_k_cli)} \
  -ctv {shlex.quote(cache_type_v_cli)} \
  -fa {flash_attn_value} \
  {extra_args_str} > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1
if ! kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  rm -f "$PID_FILE"
  exit 1
fi
"""
    run_wsl_script(settings, script)


def runtime_process_info(settings: Settings, runtime_id: str | None = None) -> dict[str, Any]:
    runtime_id = runtime_id or selected_runtime(settings)
    pid_file = remote_runtime_pid_file(settings, runtime_id)
    log_file = remote_runtime_log_file(settings, runtime_id)
    script = f"""
PID_FILE={shlex.quote(pid_file)}
LOG_FILE={shlex.quote(log_file)}
if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    CMD="$(ps -p "$PID" -o cmd= || true)"
    printf '{{"running":true,"pid":"%s","log_file":"%s","cmd":%s}}\\n' "$PID" "$LOG_FILE" "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$CMD")"
    exit 0
  fi
fi
printf '{{"running":false,"pid":null,"log_file":"%s","cmd":""}}\\n' "$LOG_FILE"
"""
    result = run_wsl_script(settings, script, check=False)
    try:
        return json.loads((result.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {"running": False, "pid": None, "log_file": log_file, "cmd": ""}


def wait_for_backend_api(settings: Settings, runtime_id: str, model_name: str, cache_type_k: str, cache_type_v: str) -> None:
    print("[4/4] Waiting for backend OpenAI-compatible endpoint...")
    deadline = time.time() + settings.runtime_defaults.startup_timeout_seconds
    while time.time() < deadline:
        try:
            backend_api_json(settings, "/v1/models")
        except Exception:
            time.sleep(2)
            continue
        set_selected_model(settings, model_name)
        set_selected_runtime(settings, runtime_id)
        set_selected_cache_type(settings, "k", cache_type_k, runtime_id)
        set_selected_cache_type(settings, "v", cache_type_v, runtime_id)
        print("llama-server ready.")
        print(f"Selected model: {model_name}")
        print(f"Runtime lane:   {runtime_id}")
        print(f"Cache types:    K={cache_type_k} V={cache_type_v}")
        print(f"Served model:   {settings.runtime_defaults.served_model_name}")
        return

    log_result = run_wsl_script(
        settings,
        f"tail -n 120 {shlex.quote(remote_runtime_log_file(settings, runtime_id))} || true",
        check=False,
    )
    detail = (log_result.stdout or "").strip()
    raise WingpuError(f"llama-server did not become ready in time.\n{detail}")


def backend_ready_for_fast_path(settings: Settings, runtime_id: str) -> bool:
    process = runtime_process_info(settings, runtime_id)
    if not process.get("running"):
        return False
    try:
        backend_api_json(settings, "/v1/models")
    except Exception:
        return False
    return True


def ensure_runtime_loaded(
    settings: Settings,
    model_name: str,
    runtime_id: str,
    cache_type_k: str,
    cache_type_v: str,
    flash_attn: bool,
    *,
    force_restart: bool = False,
) -> None:
    check_command("ssh")
    catalog_entry(settings, model_name)
    runtime_lane(settings, runtime_id)
    if not any(cache_type_matches(cache_type_k, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported K cache type for {runtime_id}: {cache_type_k}")
    if not any(cache_type_matches(cache_type_v, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported V cache type for {runtime_id}: {cache_type_v}")

    check_ssh_connectivity(settings)

    if force_restart:
        stop_remote_runtime(settings)
        stop_backend_tunnel(settings)

    ensure_backend_tunnel(settings)
    if not force_restart and backend_ready_for_fast_path(settings, runtime_id):
        set_selected_model(settings, model_name)
        set_selected_runtime(settings, runtime_id)
        set_selected_cache_type(settings, "k", cache_type_k, runtime_id)
        set_selected_cache_type(settings, "v", cache_type_v, runtime_id)
        return

    process = runtime_process_info(settings, runtime_id)
    if process.get("running"):
        stop_remote_runtime(settings)
        stop_backend_tunnel(settings)
        ensure_backend_tunnel(settings)

    start_remote_runtime(settings, runtime_id, model_name, cache_type_k, cache_type_v, flash_attn)
    wait_for_backend_api(settings, runtime_id, model_name, cache_type_k, cache_type_v)


def local_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def port_listener_pid(port: int) -> str | None:
    result = run(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN", "-nP"], check=False)
    return (result.stdout or "").strip().splitlines()[0] if (result.stdout or "").strip() else None


def isoformat_ts(timestamp: float | None) -> str | None:
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def read_gateway_state_file(settings: Settings) -> dict[str, Any]:
    path = settings.state.gateway_state_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def gateway_status(settings: Settings) -> dict[str, Any] | None:
    try:
        return gateway_admin_json(settings, "/status")
    except Exception:
        stale = read_gateway_state_file(settings)
        if stale:
            stale["gateway_up"] = False
        return stale or None


def gateway_is_running(settings: Settings) -> bool:
    status = gateway_status(settings)
    return bool(status and status.get("gateway_up"))


def ensure_gateway_started(settings: Settings) -> None:
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    status = gateway_status(settings)
    if status and status.get("gateway_up"):
        return
    if local_port_in_use(settings.connection.local_port):
        raise WingpuError(
            f"Gateway port {settings.connection.local_port} is already in use by another process."
        )
    log_handle = settings.state.gateway_log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    bridge_dir = discover_bridge_dir()
    if bridge_dir is not None:
        env["WINGPU_PROJECT_DIR"] = str(bridge_dir)
    proc = subprocess.Popen(
        [sys.executable, "-m", "wingpu_cli", "__gateway_serve"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    settings.state.gateway_pid_path.write_text(f"{proc.pid}\n", encoding="utf-8")
    deadline = time.time() + 15
    while time.time() < deadline:
        status = gateway_status(settings)
        if status and status.get("gateway_up"):
            return
        time.sleep(0.5)
    log_text = settings.state.gateway_log_path.read_text(encoding="utf-8", errors="ignore")
    raise WingpuError(f"Gateway did not become ready in time.\n{log_text}")


def stop_gateway(settings: Settings) -> None:
    status = gateway_status(settings)
    if status and status.get("gateway_up"):
        try:
            gateway_admin_json(settings, "/shutdown", method="POST")
        except Exception:
            pass
        deadline = time.time() + 10
        while time.time() < deadline:
            if not local_port_in_use(settings.connection.local_port):
                break
            time.sleep(0.25)
    if settings.state.gateway_pid_path.exists():
        try:
            pid = int(settings.state.gateway_pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            pid = 0
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.time() + 5
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
        settings.state.gateway_pid_path.unlink(missing_ok=True)


def gateway_offload(settings: Settings) -> dict[str, Any]:
    return gateway_admin_json(settings, "/offload", method="POST")


class GatewayCoordinator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.state_lock = threading.RLock()
        self.start_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.active_requests = 0
        self.last_request_started_at: float | None = None
        self.last_request_finished_at: float | None = None
        self.last_runtime_start_at: float | None = None
        self.last_offload_at: float | None = None
        self.runtime_loaded = runtime_process_info(settings, selected_runtime(settings)).get("running", False)
        self.idle_status = "runtime_loaded" if self.runtime_loaded else "runtime_offloaded"
        self.server: ThreadingHTTPServer | None = None

    def write_state(self) -> None:
        with self.state_lock:
            backend_tunnel_pid = None
            if self.settings.backend_tunnel_pid_file.exists():
                backend_tunnel_pid = self.settings.backend_tunnel_pid_file.read_text(encoding="utf-8").strip() or None
            state = {
                "gateway_up": True,
                "gateway_pid": os.getpid(),
                "backend_tunnel_pid": backend_tunnel_pid,
                "active_requests": self.active_requests,
                "runtime_loaded": self.runtime_loaded,
                "idle_status": self.idle_status,
                "idle_offload_enabled": self.settings.gateway.idle_offload_enabled,
                "idle_timeout_seconds": self.settings.gateway.idle_timeout_seconds,
                "last_request_started_at": isoformat_ts(self.last_request_started_at),
                "last_request_finished_at": isoformat_ts(self.last_request_finished_at),
                "last_runtime_start_at": isoformat_ts(self.last_runtime_start_at),
                "last_offload_at": isoformat_ts(self.last_offload_at),
                "selected_model": selected_model(self.settings),
                "selected_runtime": selected_runtime(self.settings),
                "selected_cache_type_k": selected_cache_type(self.settings, "k"),
                "selected_cache_type_v": selected_cache_type(self.settings, "v"),
            }
            tmp_path = self.settings.state.gateway_state_path.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp_path.replace(self.settings.state.gateway_state_path)

    def status_payload(self) -> dict[str, Any]:
        self.write_state()
        payload = read_gateway_state_file(self.settings)
        last_activity = self.last_request_finished_at or self.last_runtime_start_at
        payload["seconds_until_offload"] = None
        if self.settings.gateway.idle_offload_enabled and self.runtime_loaded and self.active_requests == 0 and last_activity is not None:
            payload["seconds_until_offload"] = max(
                0,
                int(self.settings.gateway.idle_timeout_seconds - (time.time() - last_activity)),
            )
        return payload

    def begin_request(self) -> None:
        with self.state_lock:
            self.active_requests += 1
            self.last_request_started_at = time.time()
            self.idle_status = "runtime_loading" if not self.runtime_loaded else "runtime_loaded"
            self.write_state()

    def end_request(self) -> None:
        with self.state_lock:
            self.active_requests = max(0, self.active_requests - 1)
            self.last_request_finished_at = time.time()
            if self.runtime_loaded:
                self.idle_status = "runtime_loaded"
            self.write_state()

    def ensure_runtime_loaded(self) -> None:
        with self.start_lock:
            ensure_runtime_loaded(
                self.settings,
                selected_model(self.settings),
                selected_runtime(self.settings),
                selected_cache_type(self.settings, "k"),
                selected_cache_type(self.settings, "v"),
                self.settings.runtime_defaults.flash_attn,
                force_restart=False,
            )
            with self.state_lock:
                self.runtime_loaded = True
                self.last_runtime_start_at = time.time()
                self.idle_status = "runtime_loaded"
                self.write_state()

    def recover_runtime_after_proxy_error(self, exc: Exception) -> None:
        with self.start_lock:
            with self.state_lock:
                self.runtime_loaded = False
                self.idle_status = f"runtime_recovering:{type(exc).__name__}"
                self.write_state()
            ensure_runtime_loaded(
                self.settings,
                selected_model(self.settings),
                selected_runtime(self.settings),
                selected_cache_type(self.settings, "k"),
                selected_cache_type(self.settings, "v"),
                self.settings.runtime_defaults.flash_attn,
                force_restart=True,
            )
            with self.state_lock:
                self.runtime_loaded = True
                self.last_runtime_start_at = time.time()
                self.idle_status = "runtime_loaded"
                self.write_state()

    def offload_runtime(self, *, reason: str = "manual") -> dict[str, Any]:
        with self.start_lock:
            with self.state_lock:
                if self.active_requests > 0:
                    raise WingpuError("Cannot offload while requests are active.")
            stop_remote_runtime(self.settings)
            stop_backend_tunnel(self.settings)
            with self.state_lock:
                self.runtime_loaded = False
                self.last_offload_at = time.time()
                self.idle_status = f"runtime_offloaded:{reason}"
                self.write_state()
            return self.status_payload()

    def maybe_idle_offload(self) -> None:
        if not self.settings.gateway.idle_offload_enabled:
            return
        with self.state_lock:
            if self.active_requests > 0 or not self.runtime_loaded:
                return
            last_activity = self.last_request_finished_at or self.last_runtime_start_at
            if last_activity is None:
                return
            if time.time() - last_activity < self.settings.gateway.idle_timeout_seconds:
                return
        if self.start_lock.acquire(blocking=False):
            try:
                with self.state_lock:
                    if self.active_requests > 0 or not self.runtime_loaded:
                        return
                stop_remote_runtime(self.settings)
                stop_backend_tunnel(self.settings)
                with self.state_lock:
                    self.runtime_loaded = False
                    self.last_offload_at = time.time()
                    self.idle_status = "runtime_offloaded:idle"
                    self.write_state()
            finally:
                self.start_lock.release()

    def idle_loop(self) -> None:
        while not self.stop_event.wait(self.settings.gateway.idle_poll_seconds):
            try:
                self.maybe_idle_offload()
            except Exception:
                continue


class GatewayRequestHandler(BaseHTTPRequestHandler):
    server_version = "wingpu-gateway/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def coordinator(self) -> GatewayCoordinator:
        return self.server.coordinator  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))
        sys.stdout.flush()

    def _send_json(self, status_code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _handle_admin(self) -> bool:
        if self.path == "/__wingpu/status" and self.command == "GET":
            self._send_json(200, self.coordinator.status_payload())
            return True
        if self.path == "/__wingpu/offload" and self.command == "POST":
            try:
                payload = self.coordinator.offload_runtime(reason="manual")
            except WingpuError as exc:
                self._send_json(409, {"ok": False, "error": str(exc)})
            else:
                self._send_json(200, {"ok": True, "status": payload})
            return True
        if self.path == "/__wingpu/shutdown" and self.command == "POST":
            self._send_json(200, {"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()  # type: ignore[attr-defined]
            self.coordinator.stop_event.set()
            return True
        return False

    @staticmethod
    def _is_retryable_proxy_error(exc: Exception) -> bool:
        if isinstance(
            exc,
            (
                http.client.RemoteDisconnected,
                ConnectionResetError,
                ConnectionAbortedError,
                BrokenPipeError,
                TimeoutError,
                socket.timeout,
            ),
        ):
            return True
        message = str(exc).lower()
        return "remote end closed connection without response" in message or "connection reset" in message

    def _proxy_once(self) -> None:
        self.coordinator.ensure_runtime_loaded()
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else None
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {
                "host",
                "connection",
                "proxy-connection",
                "keep-alive",
                "transfer-encoding",
                "upgrade",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailer",
            }
        }
        headers["Host"] = f"{self.coordinator.settings.gateway.backend_host}:{self.coordinator.settings.gateway.backend_local_port}"
        headers["Connection"] = "close"
        conn = http.client.HTTPConnection(
            self.coordinator.settings.gateway.backend_host,
            self.coordinator.settings.gateway.backend_local_port,
            timeout=self.coordinator.settings.gateway.request_timeout_seconds,
        )
        conn.request(self.command, self.path, body=body, headers=headers)
        resp = conn.getresponse()
        self.send_response(resp.status, resp.reason)
        for key, value in resp.getheaders():
            if key.lower() in {
                "connection",
                "proxy-connection",
                "keep-alive",
                "transfer-encoding",
                "upgrade",
                "proxy-authenticate",
                "proxy-authorization",
                "te",
                "trailer",
            }:
                continue
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        while True:
            chunk = resp.read(64 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        self.close_connection = True

    def _proxy(self) -> None:
        self.coordinator.begin_request()
        try:
            try:
                self._proxy_once()
            except Exception as exc:
                if not self._is_retryable_proxy_error(exc):
                    raise
                self.coordinator.recover_runtime_after_proxy_error(exc)
                self._proxy_once()
        except WingpuError as exc:
            self._send_json(502, {"ok": False, "error": str(exc)})
        except Exception as exc:
            self._send_json(502, {"ok": False, "error": f"Gateway proxy error: {exc}"})
        finally:
            self.coordinator.end_request()

    def do_GET(self) -> None:
        if self._handle_admin():
            return
        self._proxy()

    def do_POST(self) -> None:
        if self._handle_admin():
            return
        self._proxy()

    def do_DELETE(self) -> None:
        if self._handle_admin():
            return
        self._proxy()

    def do_PUT(self) -> None:
        if self._handle_admin():
            return
        self._proxy()

    def do_PATCH(self) -> None:
        if self._handle_admin():
            return
        self._proxy()


def serve_gateway(settings: Settings) -> None:
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    coordinator = GatewayCoordinator(settings)
    settings.state.gateway_pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    coordinator.write_state()
    idle_thread = threading.Thread(target=coordinator.idle_loop, daemon=True)
    idle_thread.start()
    server = ThreadingHTTPServer((settings.gateway.listen_host, settings.connection.local_port), GatewayRequestHandler)
    server.coordinator = coordinator  # type: ignore[attr-defined]
    coordinator.server = server

    def _handle_signal(signum: int, frame: Any) -> None:
        coordinator.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        coordinator.stop_event.set()
        settings.state.gateway_pid_path.unlink(missing_ok=True)


def print_status(settings: Settings) -> None:
    model_name = selected_model(settings)
    runtime_id = selected_runtime(settings)
    cache_type_k = selected_cache_type(settings, "k", runtime_id)
    cache_type_v = selected_cache_type(settings, "v", runtime_id)
    lane = runtime_lane(settings, runtime_id)
    entry = catalog_entry(settings, model_name)
    gateway_info = gateway_status(settings) or read_gateway_state_file(settings)
    print("== Selected model ==")
    print(f"Name:        {model_name}")
    print(f"Model file:  {entry['gguf_path']}")
    print(f"Served as:   {settings.runtime_defaults.served_model_name}")
    print()

    print("== Selected runtime ==")
    print(f"Lane:        {runtime_id}")
    print(f"Source dir:  {lane.source_dir}")
    print(f"Build dir:   {lane.build_dir}")
    print(f"Server bin:  {lane.server_bin}")
    print(f"Bench bin:   {lane.bench_bin}")
    print(f"Cache types: K={cache_type_k} V={cache_type_v}")
    print(f"Flash attn:  {settings.runtime_defaults.flash_attn}")
    print()

    print("== Gateway ==")
    if gateway_info and gateway_info.get("gateway_up"):
        print(f"Gateway:      up on http://{settings.gateway.listen_host}:{settings.connection.local_port}/v1")
        print(f"Idle status:  {gateway_info.get('idle_status')}")
        print(f"Active reqs:  {gateway_info.get('active_requests')}")
        if gateway_info.get("last_request_finished_at"):
            print(f"Last request: {gateway_info.get('last_request_finished_at')}")
        if gateway_info.get("last_runtime_start_at"):
            print(f"Last warm:    {gateway_info.get('last_runtime_start_at')}")
        if gateway_info.get("seconds_until_offload") is not None:
            print(f"Offload in:   {gateway_info.get('seconds_until_offload')}s")
    else:
        print("Gateway:      down")
    print(f"PID file:     {settings.state.gateway_pid_path if settings.state.gateway_pid_path.exists() else 'missing'}")
    print()

    print("== Backend tunnel ==")
    if run(["ssh", "-S", str(settings.backend_tunnel_socket), "-O", "check", settings.connection.host], check=False).returncode == 0:
        print(f"Managed tunnel: active ({settings.backend_tunnel_socket})")
    else:
        print("Managed tunnel: inactive")
    pid_note = "missing"
    if settings.backend_tunnel_pid_file.exists():
        pid = settings.backend_tunnel_pid_file.read_text(encoding="utf-8").strip()
        pid_note = f"{settings.backend_tunnel_pid_file} ({'alive PID ' + pid if pid else 'empty'})"
    print(f"PID file: {pid_note}")
    lsof = run(["lsof", f"-iTCP:{settings.gateway.backend_local_port}", "-sTCP:LISTEN", "-nP"], check=False)
    if (lsof.stdout or "").strip():
        print((lsof.stdout or "").strip())
    print()

    print("== Local API check ==")
    if gateway_info and gateway_info.get("gateway_up"):
        print(f"Reachable: http://{settings.gateway.listen_host}:{settings.connection.local_port}/v1")
    else:
        print("Unavailable: gateway is down")
    print()

    print("== Remote runtime process ==")
    info = runtime_process_info(settings, runtime_id)
    if info.get("running"):
        print(f"[{runtime_id}] pid={info.get('pid')} log={info.get('log_file')}")
        print(info.get("cmd", ""))
    else:
        print("No native runtime process is running.")


def print_models(settings: Settings) -> None:
    ensure_gateway_started(settings)
    data = local_api_json(settings, "/v1/models")
    print(json.dumps(data, indent=2))


def print_model_list(settings: Settings) -> None:
    catalog = load_catalog(settings)
    default = catalog["default_model"]
    for name, entry in catalog["models"].items():
        print(
            f"{name}\tdefault={name == default}\tenabled={entry['enabled']}\tcontext_length={entry['context_length']}\t"
            f"path={entry['gguf_path']}\tnotes={entry['notes']}"
        )


def print_runtime_list(settings: Settings) -> None:
    current = selected_runtime(settings)
    default = default_runtime_id(settings)
    for runtime_id, lane in settings.runtimes.items():
        print(
            f"{runtime_id}\tcurrent={runtime_id == current}\tdefault={runtime_id == default}\t"
            f"kind={lane.kind}\tbuild={lane.build_dir}\tnotes={lane.notes}"
        )


def print_kv_show(settings: Settings) -> None:
    runtime_id = selected_runtime(settings)
    print(f"runtime={runtime_id}")
    print(f"cache_type_k={selected_cache_type(settings, 'k', runtime_id)}")
    print(f"cache_type_v={selected_cache_type(settings, 'v', runtime_id)}")
    print("supported=" + ", ".join(supported_cache_types(settings, runtime_id)))


def print_config_show(settings: Settings) -> None:
    data = {
        "defaults_file": str(settings.defaults_file),
        "project_config_file": settings.project_config_file,
        "catalog_file": str(settings.catalog_file),
        "connection": asdict(settings.connection),
        "gateway": asdict(settings.gateway),
        "paths": asdict(settings.paths),
        "runtime_defaults": asdict(settings.runtime_defaults),
        "runtimes": {runtime_id: asdict(lane) for runtime_id, lane in settings.runtimes.items()},
        "state": {
            "state_dir": str(settings.state.state_dir),
            "selected_model_file": settings.state.selected_model_file,
            "selected_model_path": str(settings.state.selected_model_path),
            "selected_runtime_file": settings.state.selected_runtime_file,
            "selected_runtime_path": str(settings.state.selected_runtime_path),
            "selected_cache_type_k_path": str(settings.state.selected_cache_type_k_path),
            "selected_cache_type_v_path": str(settings.state.selected_cache_type_v_path),
            "benchmark_dir": str(settings.state.benchmark_dir),
            "restart_on_model_set": settings.state.restart_on_model_set,
            "gateway_pid_path": str(settings.state.gateway_pid_path),
            "gateway_log_path": str(settings.state.gateway_log_path),
            "gateway_state_path": str(settings.state.gateway_state_path),
            "backend_tunnel_pid_path": str(settings.backend_tunnel_pid_file),
        },
        "selected": {
            "model": selected_model(settings),
            "runtime": selected_runtime(settings),
            "cache_type_k": selected_cache_type(settings, "k"),
            "cache_type_v": selected_cache_type(settings, "v"),
        },
    }
    print(json.dumps(data, indent=2))


def init_config(force: bool) -> None:
    target = project_config_path()
    if target is None:
        raise WingpuError("Unable to determine project-local config path from the current working directory.")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise WingpuError(f"Config already exists: {target}")
    target.write_text(read_config_bytes("wingpu.defaults.toml").decode("utf-8"), encoding="utf-8")
    print(f"Initialized config: {target}")


def maybe_restart_for_model_change(settings: Settings, model_name: str) -> None:
    if not settings.state.restart_on_model_set:
        return
    stop_gateway(settings)
    stop_backend_tunnel(settings)
    stop_remote_runtime(settings)
    start(settings, model_name)


def build_runtime(settings: Settings, runtime_id: str, clean: bool = False, jobs: int | None = None) -> None:
    lane = runtime_lane(settings, runtime_id)
    jobs = jobs or settings.runtime_defaults.build_jobs
    cmake_args = " ".join(shlex.quote(arg) for arg in settings.runtime_defaults.cmake_args)
    targets = " ".join(shlex.quote(target) for target in settings.runtime_defaults.build_targets)
    clean_cmd = f"rm -rf {shlex.quote(lane.build_dir)}\n" if clean else ""
    print(f"Building runtime lane: {runtime_id}")
    script = f"""
set -euo pipefail
{clean_cmd}mkdir -p {shlex.quote(lane.source_dir)}
cd {shlex.quote(lane.source_dir)}
cmake -S . -B {shlex.quote(lane.build_dir)} -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES={shlex.quote(settings.runtime_defaults.cuda_architectures)} {cmake_args}
cmake --build {shlex.quote(lane.build_dir)} --config Release -j {jobs} --target {targets}
"""
    run_wsl_script(settings, script)
    print(f"Built runtime lane: {runtime_id}")


def benchmark_output_path(settings: Settings, runtime_id: str, model_name: str, cache_type_k: str, cache_type_v: str, output: str | None) -> Path:
    if output:
        return Path(output).expanduser()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = model_name.replace("/", "_")
    settings.state.benchmark_dir.mkdir(parents=True, exist_ok=True)
    return settings.state.benchmark_dir / f"{timestamp}-{runtime_id}-{safe_model}-k{cache_type_k}-v{cache_type_v}.jsonl"


def parse_contexts(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise WingpuError("At least one context value is required")
    return values


def benchmark_run(
    settings: Settings,
    runtime_id: str | None,
    model_name: str | None,
    contexts_raw: str,
    repetitions: int,
    gen_tokens: int,
    batch_size: int,
    ubatch_size: int,
    cache_type_k: str | None,
    cache_type_v: str | None,
    flash_attn: bool | None,
    output: str | None,
) -> None:
    runtime_id = runtime_id or selected_runtime(settings)
    lane = runtime_lane(settings, runtime_id)
    model_name = model_name or selected_model(settings)
    catalog_entry(settings, model_name)
    cache_type_k = cache_type_k or selected_cache_type(settings, "k", runtime_id)
    cache_type_v = cache_type_v or selected_cache_type(settings, "v", runtime_id)
    if not any(cache_type_matches(cache_type_k, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported K cache type for {runtime_id}: {cache_type_k}")
    if not any(cache_type_matches(cache_type_v, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported V cache type for {runtime_id}: {cache_type_v}")
    flash_attn = settings.runtime_defaults.flash_attn if flash_attn is None else flash_attn
    contexts = parse_contexts(contexts_raw)
    out_path = benchmark_output_path(settings, runtime_id, model_name, cache_type_k, cache_type_v, output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path = remote_model_path(settings, model_name)
    flash_attn_value = 1 if flash_attn else 0
    cache_type_k_cli = normalize_cache_type(cache_type_k)
    cache_type_v_cli = normalize_cache_type(cache_type_v)
    remote_script = f"""
set -euo pipefail
BENCH_BIN={shlex.quote(lane.bench_bin)}
MODEL_PATH={shlex.quote(model_path)}
if [[ ! -x "$BENCH_BIN" ]]; then
  echo "Missing benchmark binary: $BENCH_BIN" >&2
  exit 1
fi
run_one() {{
  local phase="$1"
  local context_target="$2"
  local n_prompt="$3"
  local n_gen="$4"
  local n_depth="$5"
  local bench_tmp mem_tmp monitor_pid peak
  bench_tmp="$(mktemp)"
  mem_tmp="$(mktemp)"
  (
    while true; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 >> "$mem_tmp" 2>/dev/null || true
      sleep 0.5
    done
  ) &
  monitor_pid=$!
  "$BENCH_BIN" \
    -o jsonl \
    -r {repetitions} \
    -m "$MODEL_PATH" \
    -p "$n_prompt" \
    -n "$n_gen" \
    -d "$n_depth" \
    -b {batch_size} \
    -ub {ubatch_size} \
    -ctk {shlex.quote(cache_type_k_cli)} \
    -ctv {shlex.quote(cache_type_v_cli)} \
    -t {settings.runtime_defaults.threads} \
    -ngl {settings.runtime_defaults.n_gpu_layers} \
    -fa {flash_attn_value} > "$bench_tmp"
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" 2>/dev/null || true
  peak="$(sort -nr "$mem_tmp" | head -n 1 || true)"
  python3 - "$bench_tmp" "$peak" {shlex.quote(runtime_id)} "$phase" "$context_target" {shlex.quote(cache_type_k)} {shlex.quote(cache_type_v)} <<'PY'
import json
import sys
bench_path, peak_vram, runtime_id, phase, context_target, cache_k, cache_v = sys.argv[1:]
with open(bench_path, 'r', encoding='utf-8') as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        obj['runtime_lane'] = runtime_id
        obj['phase'] = phase
        obj['context_target'] = int(context_target)
        obj['peak_vram_mib'] = int(peak_vram) if peak_vram else None
        obj['selected_cache_type_k'] = cache_k
        obj['selected_cache_type_v'] = cache_v
        print(json.dumps(obj))
PY
  rm -f "$bench_tmp" "$mem_tmp"
}}
"""
    for ctx in contexts:
        remote_script += f"run_one prompt {ctx} {ctx} 0 0\n"
        remote_script += f"run_one decode {ctx} 0 {gen_tokens} {ctx}\n"
    result = run_wsl_script(settings, remote_script)
    content = (result.stdout or "").strip()
    out_path.write_text(content + ("\n" if content else ""), encoding="utf-8")
    rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    print(f"Wrote benchmark results: {out_path}")
    for row in rows:
        phase = row.get("phase")
        ctx = row.get("context_target")
        avg_ts = row.get("avg_ts")
        peak_vram = row.get("peak_vram_mib")
        type_k = row.get("type_k")
        type_v = row.get("type_v")
        print(f"{phase:>6} ctx={ctx:<6} k={type_k:<9} v={type_v:<9} avg_ts={avg_ts:<10} peak_vram_mib={peak_vram}")


def run_admin_wrapper(settings: Settings, wrapper_name: str, argument: str | None = None) -> None:
    cmd = f"sudo -n /usr/local/sbin/{wrapper_name}"
    if argument:
        cmd += f" {shlex.quote(argument)}"
    result = run_wsl_script(settings, cmd, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise WingpuError(f"Admin wrapper failed: {wrapper_name}\n{detail}")
    output = (result.stdout or "").strip()
    if output:
        print(output)


def start(
    settings: Settings,
    explicit_model: str | None = None,
    explicit_runtime: str | None = None,
    explicit_cache_type_k: str | None = None,
    explicit_cache_type_v: str | None = None,
    flash_attn: bool | None = None,
) -> None:
    model_name = explicit_model or selected_model(settings)
    runtime_id = explicit_runtime or selected_runtime(settings)
    cache_type_k = explicit_cache_type_k or selected_cache_type(settings, "k", runtime_id)
    cache_type_v = explicit_cache_type_v or selected_cache_type(settings, "v", runtime_id)
    flash_attn = settings.runtime_defaults.flash_attn if flash_attn is None else flash_attn
    ensure_gateway_started(settings)
    ensure_runtime_loaded(
        settings,
        model_name,
        runtime_id,
        cache_type_k,
        cache_type_v,
        flash_attn,
        force_restart=True,
    )


def stop(settings: Settings) -> None:
    stop_gateway(settings)
    stop_backend_tunnel(settings)
    stop_remote_runtime(settings)


def restart(
    settings: Settings,
    explicit_model: str | None = None,
    explicit_runtime: str | None = None,
    explicit_cache_type_k: str | None = None,
    explicit_cache_type_v: str | None = None,
    flash_attn: bool | None = None,
) -> None:
    stop(settings)
    start(
        settings,
        explicit_model=explicit_model,
        explicit_runtime=explicit_runtime,
        explicit_cache_type_k=explicit_cache_type_k,
        explicit_cache_type_v=explicit_cache_type_v,
        flash_attn=flash_attn,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wingpu", description="Mac helper for the Windows + WSL GPU bridge")
    parser.add_argument("--host", help="Override configured Windows SSH host")
    parser.add_argument("--distro", help="Override configured WSL distro")
    parser.add_argument("--api-key", help="Override configured local API key")

    subparsers = parser.add_subparsers(dest="command", required=True)

    start_parser = subparsers.add_parser("start", help="Start llama.cpp with the selected model")
    start_parser.add_argument("model_name", nargs="?", help="Optional one-shot model override")
    start_parser.add_argument("--runtime", dest="runtime_id", help="Runtime lane override")
    start_parser.add_argument("--cache-type-k", help="Override selected K cache type")
    start_parser.add_argument("--cache-type-v", help="Override selected V cache type")
    flash_group = start_parser.add_mutually_exclusive_group()
    flash_group.add_argument("--flash-attn", dest="flash_attn", action="store_true", help="Force flash attention on")
    flash_group.add_argument("--no-flash-attn", dest="flash_attn", action="store_false", help="Force flash attention off")
    start_parser.set_defaults(flash_attn=None)

    restart_parser = subparsers.add_parser("restart", help="Restart the gateway, backend tunnel, and remote llama.cpp runtime")
    restart_parser.add_argument("model_name", nargs="?", help="Optional one-shot model override")
    restart_parser.add_argument("--runtime", dest="runtime_id", help="Runtime lane override")
    restart_parser.add_argument("--cache-type-k", help="Override selected K cache type")
    restart_parser.add_argument("--cache-type-v", help="Override selected V cache type")
    restart_flash_group = restart_parser.add_mutually_exclusive_group()
    restart_flash_group.add_argument("--flash-attn", dest="flash_attn", action="store_true", help="Force flash attention on")
    restart_flash_group.add_argument("--no-flash-attn", dest="flash_attn", action="store_false", help="Force flash attention off")
    restart_parser.set_defaults(flash_attn=None)

    subparsers.add_parser("stop", help="Stop the gateway, backend tunnel, and remote llama.cpp runtime")
    subparsers.add_parser("status", help="Show selected model, gateway, tunnel, and remote runtime status")
    subparsers.add_parser("models", help="Show the local OpenAI-compatible model list")

    gateway_parser = subparsers.add_parser("gateway", help="Manage the local always-on gateway")
    gateway_subparsers = gateway_parser.add_subparsers(dest="gateway_command", required=True)
    gateway_subparsers.add_parser("start", help="Start the local gateway without warming the remote runtime")
    gateway_subparsers.add_parser("stop", help="Stop the local gateway without touching the remote runtime")
    gateway_subparsers.add_parser("status", help="Show gateway status")

    model_parser = subparsers.add_parser("model", help="Manage the selected GGUF model")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    model_subparsers.add_parser("list", help="Show the configured GGUF catalog")
    model_subparsers.add_parser("current", help="Show the persisted selected model")
    set_parser = model_subparsers.add_parser("set", help="Persist the selected model without starting/stopping")
    set_parser.add_argument("model_name")
    default_parser = model_subparsers.add_parser("default", help="Set the catalog default model")
    default_parser.add_argument("model_name")
    context_parser = model_subparsers.add_parser("context", help="Show or set the catalog context length for a model")
    context_parser.add_argument("model_name")
    context_parser.add_argument("context_length", nargs="?", type=int)

    runtime_parser = subparsers.add_parser("runtime", help="Manage the runtime lane")
    runtime_subparsers = runtime_parser.add_subparsers(dest="runtime_command", required=True)
    runtime_subparsers.add_parser("list", help="Show available runtime lanes")
    runtime_subparsers.add_parser("current", help="Show the persisted selected runtime lane")
    runtime_set_parser = runtime_subparsers.add_parser("set", help="Persist the selected runtime lane")
    runtime_set_parser.add_argument("runtime_id")

    kv_parser = subparsers.add_parser("kv", help="Manage selected KV cache types")
    kv_subparsers = kv_parser.add_subparsers(dest="kv_command", required=True)
    kv_subparsers.add_parser("show", help="Show selected KV cache types")
    kv_set_parser = kv_subparsers.add_parser("set", help="Persist selected KV cache types")
    kv_set_parser.add_argument("--runtime", dest="runtime_id", help="Validate against a specific runtime lane")
    kv_set_parser.add_argument("--k", dest="cache_type_k")
    kv_set_parser.add_argument("--v", dest="cache_type_v")

    build_parser_cmd = subparsers.add_parser("build", help="Build one or more runtime lanes in WSL")
    build_parser_cmd.add_argument("runtime_id", nargs="?", default=None, help="Runtime lane to build, or 'all'")
    build_parser_cmd.add_argument("--clean", action="store_true", help="Remove the build dir before configuring")
    build_parser_cmd.add_argument("--jobs", type=int, help="Parallel build jobs")

    bench_parser = subparsers.add_parser("benchmark", help="Run llama-bench on the selected runtime lane")
    bench_subparsers = bench_parser.add_subparsers(dest="benchmark_command", required=True)
    bench_run = bench_subparsers.add_parser("run", help="Run prompt/decode benchmarks across context sizes")
    bench_run.add_argument("--runtime", dest="runtime_id", help="Runtime lane override")
    bench_run.add_argument("--model", dest="model_name", help="Model override")
    bench_run.add_argument("--contexts", default="8192,16384,32768", help="Comma-separated context targets")
    bench_run.add_argument("--repetitions", type=int, default=3)
    bench_run.add_argument("--gen-tokens", type=int, default=128)
    bench_run.add_argument("--batch-size", type=int, default=2048)
    bench_run.add_argument("--ubatch-size", type=int, default=512)
    bench_run.add_argument("--cache-type-k")
    bench_run.add_argument("--cache-type-v")
    bench_run.add_argument("--output")
    bench_flash_group = bench_run.add_mutually_exclusive_group()
    bench_flash_group.add_argument("--flash-attn", dest="flash_attn", action="store_true")
    bench_flash_group.add_argument("--no-flash-attn", dest="flash_attn", action="store_false")
    bench_run.set_defaults(flash_attn=None)

    admin_parser = subparsers.add_parser("admin", help="Run narrow passwordless maintenance wrappers in WSL")
    admin_subparsers = admin_parser.add_subparsers(dest="admin_command", required=True)
    admin_subparsers.add_parser("build-prereqs", help="Install core build prerequisites")
    admin_subparsers.add_parser("experiment-prereqs", help="Install extra experiment prerequisites")
    admin_subparsers.add_parser("cuda-toolkit", help="Install Ubuntu CUDA toolkit packages")
    admin_install = admin_subparsers.add_parser("install-llamacpp-system", help="Install built llama.cpp binaries to /usr/local/bin")
    admin_install.add_argument("runtime_id", nargs="?", default=None, help="Runtime lane whose build/bin directory should be installed")

    config_parser = subparsers.add_parser("config", help="Inspect or initialize wingpu config")
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)
    config_subparsers.add_parser("path", help="Show the project-local config path")
    config_subparsers.add_parser("show", help="Show the merged active config")
    init_parser = config_subparsers.add_parser("init", help="Write the project-local config template")
    init_parser.add_argument("--force", action="store_true", help="Overwrite existing config")

    internal_gateway = subparsers.add_parser("__gateway_serve", help=argparse.SUPPRESS)
    internal_gateway.set_defaults(internal_gateway=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(host=args.host, distro=args.distro, api_key=args.api_key)

    try:
        if args.command == "__gateway_serve":
            serve_gateway(settings)
        elif args.command == "start":
            start(
                settings,
                explicit_model=args.model_name,
                explicit_runtime=args.runtime_id,
                explicit_cache_type_k=args.cache_type_k,
                explicit_cache_type_v=args.cache_type_v,
                flash_attn=args.flash_attn,
            )
        elif args.command == "restart":
            restart(
                settings,
                explicit_model=args.model_name,
                explicit_runtime=args.runtime_id,
                explicit_cache_type_k=args.cache_type_k,
                explicit_cache_type_v=args.cache_type_v,
                flash_attn=args.flash_attn,
            )
        elif args.command == "stop":
            stop(settings)
        elif args.command == "status":
            print_status(settings)
        elif args.command == "models":
            print_models(settings)
        elif args.command == "gateway":
            if args.gateway_command == "start":
                ensure_gateway_started(settings)
                print(f"Gateway listening on http://{settings.gateway.listen_host}:{settings.connection.local_port}/v1")
            elif args.gateway_command == "stop":
                stop_gateway(settings)
                print("Gateway stopped.")
            elif args.gateway_command == "status":
                data = gateway_status(settings) or read_gateway_state_file(settings) or {"gateway_up": False}
                print(json.dumps(data, indent=2))
        elif args.command == "model":
            if args.model_command == "list":
                print_model_list(settings)
            elif args.model_command == "current":
                print(selected_model(settings))
            elif args.model_command == "set":
                set_selected_model(settings, args.model_name)
                print(f"Selected model set to: {args.model_name}")
                maybe_restart_for_model_change(settings, args.model_name)
            elif args.model_command == "default":
                set_catalog_default_model(settings, args.model_name)
                print(f"Catalog default model set to: {args.model_name}")
            elif args.model_command == "context":
                if args.context_length is None:
                    print(catalog_context_length(settings, args.model_name))
                else:
                    set_catalog_context_length(settings, args.model_name, args.context_length)
                    print(f"Catalog context length set: {args.model_name} -> {args.context_length}")
        elif args.command == "runtime":
            if args.runtime_command == "list":
                print_runtime_list(settings)
            elif args.runtime_command == "current":
                print(selected_runtime(settings))
            elif args.runtime_command == "set":
                set_selected_runtime(settings, args.runtime_id)
                print(f"Selected runtime set to: {args.runtime_id}")
        elif args.command == "kv":
            if args.kv_command == "show":
                print_kv_show(settings)
            elif args.kv_command == "set":
                runtime_id = args.runtime_id or selected_runtime(settings)
                if not args.cache_type_k and not args.cache_type_v:
                    raise WingpuError("Specify at least one of --k or --v")
                if args.cache_type_k:
                    set_selected_cache_type(settings, "k", args.cache_type_k, runtime_id)
                if args.cache_type_v:
                    set_selected_cache_type(settings, "v", args.cache_type_v, runtime_id)
                print(f"Selected KV cache types updated for runtime {runtime_id}")
                print_kv_show(settings)
        elif args.command == "build":
            if args.runtime_id == "all":
                for runtime_id in settings.runtimes:
                    build_runtime(settings, runtime_id, clean=args.clean, jobs=args.jobs)
            else:
                build_runtime(settings, args.runtime_id or selected_runtime(settings), clean=args.clean, jobs=args.jobs)
        elif args.command == "benchmark":
            if args.benchmark_command == "run":
                benchmark_run(
                    settings,
                    runtime_id=args.runtime_id,
                    model_name=args.model_name,
                    contexts_raw=args.contexts,
                    repetitions=args.repetitions,
                    gen_tokens=args.gen_tokens,
                    batch_size=args.batch_size,
                    ubatch_size=args.ubatch_size,
                    cache_type_k=args.cache_type_k,
                    cache_type_v=args.cache_type_v,
                    flash_attn=args.flash_attn,
                    output=args.output,
                )
        elif args.command == "admin":
            if args.admin_command == "build-prereqs":
                run_admin_wrapper(settings, "wingpu-install-build-prereqs")
            elif args.admin_command == "experiment-prereqs":
                run_admin_wrapper(settings, "wingpu-install-experiment-prereqs")
            elif args.admin_command == "cuda-toolkit":
                run_admin_wrapper(settings, "wingpu-install-cuda-toolkit")
            elif args.admin_command == "install-llamacpp-system":
                runtime_id = args.runtime_id or selected_runtime(settings)
                lane = runtime_lane(settings, runtime_id)
                run_admin_wrapper(settings, "wingpu-install-llamacpp-system", f"{lane.build_dir}/bin")
        elif args.command == "config":
            if args.config_command == "path":
                target = project_config_path()
                if target is None:
                    raise WingpuError("Unable to determine project-local config path from the current working directory.")
                print(target)
            elif args.config_command == "show":
                print_config_show(settings)
            elif args.config_command == "init":
                init_config(force=args.force)
        return 0
    except WingpuError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
