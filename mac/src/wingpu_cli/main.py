from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from importlib import resources as importlib_resources
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
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


@dataclass(slots=True)
class Settings:
    connection: ConnectionConfig
    paths: PathsConfig
    runtime_defaults: RuntimeDefaults
    runtimes: dict[str, RuntimeLane]
    state: StateConfig
    catalog_file: str = "package://wingpu_cli.resources/qwen_gguf_catalog.json"
    defaults_file: str = "package://wingpu_cli.resources/wingpu.defaults.toml"
    project_config_file: str | None = None

    @property
    def tunnel_socket(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.connection.local_port}.sock"

    @property
    def tunnel_pid_file(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.connection.local_port}.pid"

    @property
    def tunnel_log_file(self) -> Path:
        return self.state.state_dir / f"tunnel_{self.connection.host}_{self.connection.local_port}.log"


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

    env_overrides: dict[str, Any] = {"connection": {}, "runtime_defaults": {}, "paths": {}}
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
    url = f"http://127.0.0.1:{settings.connection.local_port}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {settings.connection.api_key}"})
    with urllib.request.urlopen(req, timeout=5) as response:
        return json.load(response)


def remote_runtime_base_dir(settings: Settings) -> str:
    return settings.runtime_defaults.remote_state_dir


def remote_runtime_pid_file(settings: Settings, runtime_id: str) -> str:
    return f"{remote_runtime_base_dir(settings)}/run/{runtime_id}.pid"


def remote_runtime_log_file(settings: Settings, runtime_id: str) -> str:
    return f"{remote_runtime_base_dir(settings)}/logs/{runtime_id}.log"


def ensure_tunnel(settings: Settings) -> None:
    print(
        f"[3/4] Ensuring SSH tunnel localhost:{settings.connection.local_port} -> "
        f"{settings.connection.host}:127.0.0.1:{settings.connection.remote_port} ..."
    )
    settings.state.state_dir.mkdir(parents=True, exist_ok=True)
    sock = str(settings.tunnel_socket)
    check_cmd = ["ssh", "-S", sock, "-O", "check", settings.connection.host]
    if run(check_cmd, check=False).returncode == 0:
        print("Managed tunnel already running.")
        return

    settings.tunnel_socket.unlink(missing_ok=True)
    settings.tunnel_pid_file.unlink(missing_ok=True)
    if local_port_in_use(settings.connection.local_port):
        raise WingpuError(f"Local port {settings.connection.local_port} is already in use by another process.")

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
        str(settings.tunnel_log_file),
        "-L",
        f"{settings.connection.local_port}:127.0.0.1:{settings.connection.remote_port}",
        settings.connection.host,
    ]
    run(start_cmd)
    time.sleep(1)
    if run(check_cmd, check=False).returncode != 0:
        tunnel_log = settings.tunnel_log_file.read_text(encoding="utf-8", errors="ignore")
        raise WingpuError(f"Failed to establish managed SSH tunnel.\n{tunnel_log}")
    pid = port_listener_pid(settings.connection.local_port)
    if pid:
        settings.tunnel_pid_file.write_text(f"{pid}\n", encoding="utf-8")


def stop_tunnel(settings: Settings) -> None:
    print("[1/2] Stopping SSH tunnel...")
    run(["ssh", "-S", str(settings.tunnel_socket), "-O", "exit", settings.connection.host], check=False)
    settings.tunnel_socket.unlink(missing_ok=True)
    settings.tunnel_pid_file.unlink(missing_ok=True)


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


def wait_for_api(settings: Settings, runtime_id: str, model_name: str, cache_type_k: str, cache_type_v: str) -> None:
    print("[4/4] Waiting for OpenAI-compatible endpoint...")
    deadline = time.time() + settings.runtime_defaults.startup_timeout_seconds
    while time.time() < deadline:
        try:
            local_api_json(settings, "/v1/models")
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


def local_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def port_listener_pid(port: int) -> str | None:
    result = run(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN", "-nP"], check=False)
    return (result.stdout or "").strip().splitlines()[0] if (result.stdout or "").strip() else None


def print_status(settings: Settings) -> None:
    model_name = selected_model(settings)
    runtime_id = selected_runtime(settings)
    cache_type_k = selected_cache_type(settings, "k", runtime_id)
    cache_type_v = selected_cache_type(settings, "v", runtime_id)
    lane = runtime_lane(settings, runtime_id)
    entry = catalog_entry(settings, model_name)
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

    print("== Local tunnel ==")
    if run(["ssh", "-S", str(settings.tunnel_socket), "-O", "check", settings.connection.host], check=False).returncode == 0:
        print(f"Managed tunnel: active ({settings.tunnel_socket})")
    else:
        print("Managed tunnel: inactive")
    pid_note = "missing"
    if settings.tunnel_pid_file.exists():
        pid = settings.tunnel_pid_file.read_text(encoding="utf-8").strip()
        pid_note = f"{settings.tunnel_pid_file} ({'alive PID ' + pid if pid else 'empty'})"
    print(f"PID file: {pid_note}")
    lsof = run(["lsof", f"-iTCP:{settings.connection.local_port}", "-sTCP:LISTEN", "-nP"], check=False)
    if (lsof.stdout or "").strip():
        print((lsof.stdout or "").strip())
    print()

    print("== Local API check ==")
    try:
        local_api_json(settings, "/v1/models")
        print(f"Reachable: http://127.0.0.1:{settings.connection.local_port}/v1")
    except Exception as exc:
        print(f"Unavailable: {exc}")
    print()

    print("== Remote runtime process ==")
    script = f"""
for RUNTIME_ID in {' '.join(shlex.quote(runtime_id) for runtime_id in settings.runtimes)}; do
  PID_FILE={shlex.quote(remote_runtime_base_dir(settings))}/run/$RUNTIME_ID.pid
  LOG_FILE={shlex.quote(remote_runtime_base_dir(settings))}/logs/$RUNTIME_ID.log
  if [[ -f "$PID_FILE" ]]; then
    PID="$(cat "$PID_FILE")"
    if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
      echo "[$RUNTIME_ID] pid=$PID log=$LOG_FILE"
      ps -p "$PID" -o pid=,etime=,cmd=
    else
      echo "[$RUNTIME_ID] stale pid file: $PID_FILE"
    fi
  fi
done
"""
    result = run_wsl_script(settings, script, check=False)
    output = (result.stdout or "").strip()
    print(output if output else "No native runtime process is running.")


def print_models(settings: Settings) -> None:
    data = local_api_json(settings, "/v1/models")
    print(json.dumps(data, indent=2))


def print_model_list(settings: Settings) -> None:
    catalog = load_catalog(settings)
    default = catalog["default_model"]
    for name, entry in catalog["models"].items():
        print(
            f"{name}\tdefault={name == default}\tenabled={entry['enabled']}\t"
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
    stop_tunnel(settings)
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
    check_command("ssh")
    model_name = explicit_model or selected_model(settings)
    runtime_id = explicit_runtime or selected_runtime(settings)
    cache_type_k = explicit_cache_type_k or selected_cache_type(settings, "k", runtime_id)
    cache_type_v = explicit_cache_type_v or selected_cache_type(settings, "v", runtime_id)
    flash_attn = settings.runtime_defaults.flash_attn if flash_attn is None else flash_attn
    catalog_entry(settings, model_name)
    runtime_lane(settings, runtime_id)
    if not any(cache_type_matches(cache_type_k, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported K cache type for {runtime_id}: {cache_type_k}")
    if not any(cache_type_matches(cache_type_v, supported) for supported in supported_cache_types(settings, runtime_id)):
        raise WingpuError(f"Unsupported V cache type for {runtime_id}: {cache_type_v}")
    check_ssh_connectivity(settings)
    stop_remote_runtime(settings)
    start_remote_runtime(settings, runtime_id, model_name, cache_type_k, cache_type_v, flash_attn)
    ensure_tunnel(settings)
    wait_for_api(settings, runtime_id, model_name, cache_type_k, cache_type_v)


def stop(settings: Settings) -> None:
    stop_tunnel(settings)
    stop_remote_runtime(settings)


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

    subparsers.add_parser("stop", help="Stop the SSH tunnel and remote llama.cpp runtime")
    subparsers.add_parser("status", help="Show selected model, tunnel, and remote runtime status")
    subparsers.add_parser("models", help="Show the local OpenAI-compatible model list")

    model_parser = subparsers.add_parser("model", help="Manage the selected GGUF model")
    model_subparsers = model_parser.add_subparsers(dest="model_command", required=True)
    model_subparsers.add_parser("list", help="Show the configured GGUF catalog")
    model_subparsers.add_parser("current", help="Show the persisted selected model")
    set_parser = model_subparsers.add_parser("set", help="Persist the selected model without starting/stopping")
    set_parser.add_argument("model_name")

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = load_settings(host=args.host, distro=args.distro, api_key=args.api_key)

    try:
        if args.command == "start":
            start(
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
        elif args.command == "model":
            if args.model_command == "list":
                print_model_list(settings)
            elif args.model_command == "current":
                print(selected_model(settings))
            elif args.model_command == "set":
                set_selected_model(settings, args.model_name)
                print(f"Selected model set to: {args.model_name}")
                maybe_restart_for_model_change(settings, args.model_name)
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
