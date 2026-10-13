"""Allowlisted dashboard actions.

Every mutation is a fixed argument vector, or one HTTP call to the local
wingpu gateway. Nothing is passed to a shell.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from . import mock_data

Runner = Callable[[list[str], float], tuple[int, str, str]]

MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,80}$")
LANE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,40}$")
CACHE_RE = re.compile(r"^[a-z0-9_]{2,12}$")
PDF_RE = re.compile(r"^/[A-Za-z0-9._/-]+\.pdf$")
ENGINES = {"whisper", "qwen3", "both"}

LOG_SOURCES = ("llama", "gateway", "watchdog", "ocr", "locate", "asr", "mineru")


class ActionError(Exception):
    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def loopback_host(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def run_argv(argv: list[str], timeout: float) -> tuple[int, str, str]:
    """Run an allowlisted argv. ``shell`` stays false."""
    if not argv or not all(isinstance(part, str) for part in argv):
        raise ActionError("refusing to run a command that is not a list of strings")
    try:
        completed = subprocess.run(
            argv,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ActionError(f"command timed out after {timeout:g}s") from exc
    except FileNotFoundError as exc:
        raise ActionError(f"{argv[0]} is not on PATH") from exc
    return completed.returncode, completed.stdout or "", completed.stderr or ""


def _which(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise ActionError(f"{name} is not on PATH")
    return found


def validate_pdf(path: str, *, must_exist: bool) -> str:
    if not isinstance(path, str) or "\x00" in path:
        raise ActionError("pdf path is not allowed")
    if any(part == ".." for part in path.split("/")):
        raise ActionError("pdf path is not allowed")
    if not PDF_RE.match(path):
        raise ActionError("pdf path must be an absolute .pdf path with no spaces")
    if must_exist and not Path(path).is_file():
        raise ActionError("pdf file was not found")
    return path


def load_catalog(project_dir: Path | None) -> dict[str, Any]:
    """Read the wingpu GGUF catalog and lane list from the repo, if present."""
    roots: list[Path] = []
    if project_dir:
        roots.append(Path(project_dir))
    here = Path(__file__).resolve()
    roots.extend(here.parents)
    catalog_path = None
    defaults_path = None
    for root in roots:
        candidate = root / "config" / "qwen_gguf_catalog.json"
        if candidate.is_file():
            catalog_path = candidate
            defaults = root / "config" / "wingpu.defaults.toml"
            if defaults.is_file():
                defaults_path = defaults
            break
    models: list[dict[str, Any]] = []
    if catalog_path is not None:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        for name, entry in (raw.get("models") or {}).items():
            if not entry.get("enabled", True):
                continue
            models.append({
                "name": name,
                "context_length": entry.get("context_length"),
                "notes": entry.get("notes") or "",
            })
    lanes = ["upstream-mtp", "upstream"]
    cache_types = ["q4_0", "q8_0", "f16"]
    if defaults_path is not None:
        with defaults_path.open("rb") as handle:
            config = tomllib.load(handle)
        found = [key for key in (config.get("runtimes") or {}) if isinstance(key, str)]
        if found:
            lanes = found
        supported = []
        for lane in (config.get("runtimes") or {}).values():
            if isinstance(lane, dict):
                supported.extend(lane.get("supported_cache_types") or [])
        if supported:
            cache_types = list(dict.fromkeys(supported))
    if not models:
        models = copy_models()
    return {"models": models, "lanes": lanes, "cache_types": cache_types}


def copy_models() -> list[dict[str, Any]]:
    return [
        {"name": item["name"], "context_length": item["context_length"], "notes": item["notes"]}
        for item in mock_data.CATALOG["models"]
    ]


def _parse_json_blob(text: str) -> dict[str, Any] | None:
    for line in reversed([line.strip() for line in text.splitlines() if line.strip()]):
        if line.startswith("{"):
            try:
                parsed = json.loads(line)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _http_json(url: str, *, method: str = "GET", timeout: float = 3) -> dict[str, Any] | None:
    req = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


class Controller:
    def __init__(
        self,
        *,
        mock: bool = False,
        project_dir: str | None = None,
        runner: Runner | None = None,
        gateway_url: str = "http://127.0.0.1:8000",
    ) -> None:
        self.mock = mock
        self.project_dir = Path(project_dir) if project_dir else None
        self.runner = runner or run_argv
        self.gateway_url = gateway_url.rstrip("/")
        self.world = mock_data.MockWorld() if mock else None
        self.jobs: dict[str, dict[str, Any]] = {}
        self._mut = threading.Lock()
        self._catalog = load_catalog(self.project_dir)

    def status(self) -> dict[str, Any]:
        if self.world is not None:
            snap = self.world.snapshot()
            snap["catalog"] = self._catalog or snap["catalog"]
            return snap
        return self._live_status()

    def events(self, limit: int = 50) -> dict[str, Any]:
        limit = max(1, min(int(limit), 500))
        if self.world is not None:
            return self.world.events(limit)
        try:
            code, out, err = self.runner(["wingpu", "broker", "events", "--json", "--limit", str(limit)], 20)
        except ActionError as exc:
            return {"ok": False, "error": str(exc), "events": []}
        parsed = _parse_json_blob(out)
        if code != 0 or not parsed:
            return {"ok": False, "error": (err or out or "broker events failed").strip(), "events": []}
        return parsed

    def logs(self, source: str, n: int = 80) -> dict[str, Any]:
        if source not in LOG_SOURCES:
            raise ActionError("unknown log source")
        n = max(1, min(int(n), 1000))
        if self.world is not None:
            return {"ok": True, "source": source, "text": self.world.log_text(source, n)}
        argv = self._log_argv(source, n)
        code, out, err = self.runner(argv, 20)
        text = out or err
        lines = text.splitlines()
        return {"ok": code == 0, "source": source, "text": "\n".join(lines[-n:])}

    def act(self, name: str, body: dict[str, Any] | None) -> dict[str, Any]:
        body = body or {}
        if not isinstance(body, dict):
            raise ActionError("action body must be an object")
        self._validate(name, body)
        if not self._mut.acquire(blocking=False):
            raise ActionError("another action is still running", 409)
        job = {
            "id": uuid.uuid4().hex[:12],
            "action": name,
            "status": "running",
            "output": "",
            "error": "",
        }
        self.jobs[job["id"]] = job
        self._trim_jobs()

        def work() -> None:
            try:
                job["output"] = self._dispatch(name, body)
                job["status"] = "ok"
            except ActionError as exc:
                job["status"] = "error"
                job["error"] = str(exc)
            except Exception as exc:  # noqa: BLE001 — surface it on the job, do not crash the server
                job["status"] = "error"
                job["error"] = str(exc)
            finally:
                self._mut.release()

        if self.mock:
            work()
        else:
            threading.Thread(target=work, name=f"gpu-dashboard-{name}", daemon=True).start()
        return dict(job)

    def job(self, job_id: str) -> dict[str, Any]:
        found = self.jobs.get(job_id)
        if found is None:
            raise ActionError("unknown job", 404)
        return dict(found)

    def _trim_jobs(self) -> None:
        if len(self.jobs) <= 40:
            return
        for key in list(self.jobs)[:-40]:
            self.jobs.pop(key, None)

    def _validate(self, name: str, body: dict[str, Any]) -> None:
        if name not in _HANDLERS:
            raise ActionError("unknown action", 404)
        if name == "evict" and body.get("confirm") is not True:
            raise ActionError("evict needs {\"confirm\": true}")
        if name == "llama-apply":
            self._check_choice("model", body.get("model"), MODEL_RE, [item["name"] for item in self._catalog["models"]])
            self._check_choice("lane", body.get("lane"), LANE_RE, self._catalog["lanes"])
            self._check_choice("cache_k", body.get("cache_k"), CACHE_RE, self._catalog["cache_types"])
            self._check_choice("cache_v", body.get("cache_v"), CACHE_RE, self._catalog["cache_types"])
        if name == "asr-start":
            engine = body.get("engine") or "whisper"
            if engine not in ENGINES:
                raise ActionError("engine must be whisper, qwen3, or both")
        if name == "mineru-run":
            validate_pdf(str(body.get("pdf") or ""), must_exist=not self.mock)
        if name in {"ocr-start", "ocr-stop", "ocr-restart", "locate-start", "locate-stop", "locate-restart",
                    "asr-start", "asr-stop", "asr-restart", "mineru-run"} and body.get("extra"):
            raise ActionError("extra arguments are not accepted")

    def _check_choice(self, label: str, value: Any, pattern: re.Pattern[str], allowed: list[str]) -> None:
        if not isinstance(value, str) or not pattern.match(value) or value not in allowed:
            rendered = ", ".join(allowed)
            raise ActionError(f"{label} must be one of: {rendered}")

    def _dispatch(self, name: str, body: dict[str, Any]) -> str:
        if self.world is not None:
            return _mock_dispatch(self.world, name, body)
        return _HANDLERS[name](self, body)

    def _run(self, argv: list[str], timeout: float) -> str:
        binary = _which(argv[0]) if self.runner is run_argv else argv[0]
        code, out, err = self.runner([binary, *argv[1:]], timeout)
        text = (out or err).strip()
        if code != 0:
            raise ActionError(text or f"{argv[0]} exited {code}")
        return text or f"{argv[0]} ok"

    def _live_status(self) -> dict[str, Any]:
        broker = None
        events: list[Any] = []
        try:
            code, out, _err = self.runner(["wingpu", "broker", "status", "--json"], 20)
            if code == 0:
                broker = _parse_json_blob(out)
        except ActionError:
            broker = None
        try:
            code, out, _err = self.runner(["wingpu", "broker", "events", "--json", "--limit", "12"], 20)
            if code == 0:
                parsed = _parse_json_blob(out)
                events = list((parsed or {}).get("events") or [])
        except ActionError:
            events = []
        gateway = _http_json(f"{self.gateway_url}/__wingpu/status")
        holder = (broker or {}).get("holder") if broker else None
        gpu = (broker or {}).get("gpu") if broker else None
        runtime = holder.get("runtime_id") if isinstance(holder, dict) else None
        return {
            "ok": True,
            "mock": False,
            "sampled_at": None,
            "host": "win-gpu",
            "distro": "Ubuntu",
            "gpu": gpu or {"name": None, "mem_used_mb": None, "mem_total_mb": None, "util_pct": None, "processes": []},
            "broker": {
                "version": (broker or {}).get("version"),
                "protocol": (broker or {}).get("protocol"),
                "reachable": broker is not None,
                "dir": (broker or {}).get("broker_dir"),
            },
            "holder": holder,
            "watchdog": (broker or {}).get("watchdog") or {"running": False, "blocked_reason": "unreachable"},
            "gateway": _gateway_view(gateway),
            "catalog": self._catalog,
            "services": _service_view(runtime, gateway),
            "events": events,
        }

    def _log_argv(self, source: str, n: int) -> list[str]:
        if source in {"llama", "gateway", "watchdog"}:
            return ["wingpu", "logs", "--source", source, "--tail", str(n)]
        if source == "ocr":
            return ["ocrctl", "logs"]
        if source == "locate":
            return ["locatectl", "logs"]
        if source == "asr":
            return ["remote-asr", "logs"]
        return ["minerctl", "logs"]


def _gateway_view(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {
            "up": False,
            "url": "http://127.0.0.1:8000/v1",
            "served_model": "qwen-local",
            "selected_model": None,
            "selected_runtime": None,
            "cache_k": None,
            "cache_v": None,
            "context_length": None,
            "active_requests": None,
            "idle_status": "unknown",
            "seconds_until_offload": None,
            "idle_timeout_seconds": None,
            "idle_offload_enabled": None,
        }
    return {
        "up": bool(payload.get("gateway_up", True)),
        "url": "http://127.0.0.1:8000/v1",
        "served_model": "qwen-local",
        "selected_model": payload.get("selected_model"),
        "selected_runtime": payload.get("selected_runtime"),
        "cache_k": payload.get("selected_cache_type_k"),
        "cache_v": payload.get("selected_cache_type_v"),
        "context_length": payload.get("context_length"),
        "active_requests": payload.get("active_requests"),
        "idle_status": payload.get("idle_status"),
        "seconds_until_offload": payload.get("seconds_until_offload"),
        "idle_timeout_seconds": payload.get("idle_timeout_seconds"),
        "idle_offload_enabled": payload.get("idle_offload_enabled"),
    }


def _service_view(holder: str | None, gateway: dict[str, Any] | None) -> dict[str, Any]:
    loaded = bool(gateway and gateway.get("runtime_loaded"))
    cards = mock_data._services(holder if holder in {"ocr", "locate", "mineru", "live-asr", "live-asr-qwen"} else ("llama" if loaded or holder == "llama" else None))
    if holder in {"live-asr", "live-asr-qwen"}:
        cards["asr"]["engine"] = "qwen3" if holder == "live-asr-qwen" else "whisper"
    return cards


def _mock_dispatch(world: mock_data.MockWorld, name: str, body: dict[str, Any]) -> str:
    if name == "evict":
        return world.evict()
    if name == "broker-install":
        return world.install()
    if name == "watchdog-start":
        return world.watchdog(True)
    if name == "watchdog-stop":
        return world.watchdog(False)
    if name == "llama-offload":
        return world.llama_offload()
    if name == "llama-stop":
        return world.llama_stop()
    if name == "llama-start":
        return world.llama_start()
    if name == "llama-restart":
        world.llama_stop()
        return world.llama_start()
    if name == "llama-apply":
        return world.llama_apply(body["model"], body["lane"], body["cache_k"], body["cache_v"])
    if name == "gateway-restart":
        world.state["gateway"]["up"] = True
        return "gateway restarted (sample)"
    if name == "mineru-run":
        validate_pdf(str(body.get("pdf") or ""), must_exist=False)
        return world.service("mineru", "start")
    engine = body.get("engine") or "whisper"
    mapping = {
        "ocr-start": ("ocr", "start"),
        "ocr-stop": ("ocr", "stop"),
        "ocr-restart": ("ocr", "start"),
        "ocr-status": ("ocr", "status"),
        "locate-start": ("locate", "start"),
        "locate-stop": ("locate", "stop"),
        "locate-restart": ("locate", "start"),
        "locate-status": ("locate", "status"),
        "asr-start": ("live-asr-qwen" if engine == "qwen3" else "live-asr", "start"),
        "asr-stop": ("live-asr-qwen" if engine == "qwen3" else "live-asr", "stop"),
        "asr-restart": ("live-asr-qwen" if engine == "qwen3" else "live-asr", "start"),
        "asr-status": ("live-asr", "status"),
        "mineru-status": ("mineru", "status"),
    }
    runtime, action = mapping[name]
    return world.service(runtime, action, engine=engine if name.startswith("asr") else None)


def _live_evict(ctl: Controller, body: dict[str, Any]) -> str:
    argv = ["wingpu", "broker", "evict"]
    runtime = body.get("runtime")
    if runtime:
        if not isinstance(runtime, str) or not LANE_RE.match(runtime):
            raise ActionError("runtime filter is not allowed")
        argv.extend(["--runtime", runtime])
    return ctl._run(argv, 90)


def _live_install(ctl: Controller, _body: dict[str, Any]) -> str:
    return ctl._run(["wingpu", "broker", "install"], 60)


def _live_watchdog(start: bool):
    def run(ctl: Controller, _body: dict[str, Any]) -> str:
        cmd = "start" if start else "stop"
        return ctl._run(["wingpu", "broker", "watchdog", cmd], 30)
    return run


def _live_llama(kind: str):
    def run(ctl: Controller, body: dict[str, Any]) -> str:
        if kind == "offload":
            payload = _http_json(f"{ctl.gateway_url}/__wingpu/offload", method="POST", timeout=30)
            if payload is None:
                raise ActionError("gateway offload did not respond")
            return json.dumps(payload)
        if kind == "stop":
            return ctl._run(["wingpu", "stop"], 60)
        if kind == "start":
            return ctl._run(["wingpu", "start"], 180)
        if kind == "restart":
            return ctl._run(["wingpu", "restart"], 180)
        if kind == "gateway-restart":
            ctl._run(["wingpu", "gateway", "stop"], 20)
            return ctl._run(["wingpu", "gateway", "start"], 30)
        return ctl._run(["wingpu", "start"], 180)
    return run


def _live_apply(ctl: Controller, body: dict[str, Any]) -> str:
    ctl._run(["wingpu", "model", "set", body["model"]], 20)
    ctl._run(["wingpu", "runtime", "set", body["lane"]], 20)
    ctl._run(["wingpu", "kv", "set", "--k", body["cache_k"], "--v", body["cache_v"]], 20)
    return ctl._run(["wingpu", "restart"], 180)


def _live_service(binary: str, command: str, *, engine: bool = False):
    def run(ctl: Controller, body: dict[str, Any]) -> str:
        argv = [binary, command]
        if engine and command in {"start", "restart"}:
            argv.extend(["--engine", body.get("engine") or "whisper"])
        return ctl._run(argv, 180 if command in {"start", "restart"} else 40)
    return run


def _live_mineru(ctl: Controller, body: dict[str, Any]) -> str:
    pdf = validate_pdf(str(body.get("pdf") or ""), must_exist=True)
    return ctl._run(["minerctl", "run", pdf], 600)


_HANDLERS = {
    "evict": _live_evict,
    "broker-install": _live_install,
    "watchdog-start": _live_watchdog(True),
    "watchdog-stop": _live_watchdog(False),
    "llama-start": _live_llama("start"),
    "llama-stop": _live_llama("stop"),
    "llama-restart": _live_llama("restart"),
    "llama-offload": _live_llama("offload"),
    "llama-apply": _live_apply,
    "gateway-restart": _live_llama("gateway-restart"),
    "ocr-start": _live_service("ocrctl", "start"),
    "ocr-stop": _live_service("ocrctl", "stop"),
    "ocr-restart": _live_service("ocrctl", "restart"),
    "ocr-status": _live_service("ocrctl", "status"),
    "locate-start": _live_service("locatectl", "start"),
    "locate-stop": _live_service("locatectl", "stop"),
    "locate-restart": _live_service("locatectl", "restart"),
    "locate-status": _live_service("locatectl", "status"),
    "asr-start": _live_service("remote-asr", "start", engine=True),
    "asr-stop": _live_service("remote-asr", "stop"),
    "asr-restart": _live_service("remote-asr", "restart", engine=True),
    "asr-status": _live_service("remote-asr", "status"),
    "mineru-run": _live_mineru,
    "mineru-status": _live_service("minerctl", "status"),
}
