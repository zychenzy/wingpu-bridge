"""Sample snapshot used by `--mock`, tests, and screenshots.

Numbers follow a real RTX 4070 Ti SUPER load of Qwen3.8-27B-UD-Q3_K_XL on the
upstream-mtp lane (about 15205 MiB at 65536 context). Nothing here is live.
"""
from __future__ import annotations

import copy
from typing import Any


SAMPLED_AT = "2026-10-13T22:58:00+08:00"

CATALOG = {
    "models": [
        {
            "name": "Qwen3.8-27B-UD-Q3_K_XL",
            "context_length": 65536,
            "notes": "Quality default for the upstream-mtp lane on 16 GB.",
        },
        {
            "name": "Qwen3.8-27B-UD-IQ2_M",
            "context_length": 262144,
            "notes": "Long context. Prefer the upstream lane.",
        },
    ],
    "lanes": ["upstream-mtp", "upstream"],
    "cache_types": ["q4_0", "q4_1", "q8_0", "f16", "bf16"],
}

LLAMA_LOG = "\n".join([
    "[example] srv  load_model: loading model 'Qwen3.8-27B-UD-Q3_K_XL.gguf'",
    "[example] llama_kv_cache: size = 1152.90 MiB (65536 cells, q4_0)",
    "[example] srv  update_slots: all slots are idle",
    "[example] slot launch_slot_: id 0 | task 412 | processing task",
    "[example] slot print_timing: decode 66.3 tok/s",
    "[example] srv  update_slots: all slots are idle",
])

OTHER_LOGS = {
    "gateway": "[example] gateway listening on 127.0.0.1:8000\n[example] relay up, backend 127.0.0.1:18000\n",
    "watchdog": "[example] watchdog poll holder=llama idle_age=480s next=1620s\n",
    "ocr": "[example] ocrctl stopped\n",
    "locate": "[example] locatectl stopped\n",
    "asr": "[example] remote-asr stopped (engine whisper)\n",
    "mineru": "[example] minerctl idle, no job\n",
}


def _services(holder: str | None) -> dict[str, Any]:
    def card(title: str, command: str, state: str, runtime: str, blurb: str, **extra: Any) -> dict[str, Any]:
        return {
            "title": title,
            "command": command,
            "state": state,
            "runtime_id": runtime,
            "holds_gpu": holder == runtime,
            "blurb": blurb,
            **extra,
        }

    llama_state = "loaded" if holder == "llama" else "stopped"
    return {
        "llama": card(
            "llama.cpp",
            "wingpu · 127.0.0.1:8000/v1 · qwen-local",
            llama_state,
            "llama",
            "Gateway on this Mac. The model stays loaded until idle offload or stop.",
        ),
        "ocr": card(
            "Unlimited-OCR",
            "ocrctl · 127.0.0.1:8010",
            "running" if holder == "ocr" else "stopped",
            "ocr",
            "One model at a time. Starting it asks the broker for the card.",
        ),
        "locate": card(
            "LocateAnything",
            "locatectl · 127.0.0.1:8011",
            "running" if holder == "locate" else "stopped",
            "locate",
            "Point and detect server. Same lease as the other bridges.",
        ),
        "asr": card(
            "Live transcription",
            "remote-asr · whisper and qwen3 are separate leases",
            "running" if holder in {"live-asr", "live-asr-qwen"} else "stopped",
            "live-asr" if holder != "live-asr-qwen" else "live-asr-qwen",
            "Whisper keeps the id live-asr. Qwen ASR uses live-asr-qwen.",
            engine="qwen3" if holder == "live-asr-qwen" else "whisper",
        ),
        "mineru": card(
            "MinerU",
            "minerctl · one job, then it exits",
            "running" if holder == "mineru" else "idle",
            "mineru",
            "There is no long-lived server. Run sends a single PDF.",
        ),
    }


def fresh_state() -> dict[str, Any]:
    holder = {
        "runtime_id": "llama",
        "label": "wingpu:upstream-mtp:Qwen3.8-27B-UD-Q3_K_XL",
        "alive": True,
        "busy": False,
        "pid": 412,
        "acquired_at": "2026-10-13T22:31:05+08:00",
        "last_active_at": "2026-10-13T22:50:00+08:00",
        "vram_mb": 12800,
        "idle_age_s": 480.0,
        "state": "active",
        "shutdown_url": "http://127.0.0.1:8799/shutdown",
    }
    return {
        "ok": True,
        "mock": True,
        "sampled_at": SAMPLED_AT,
        "host": "win-gpu",
        "distro": "Ubuntu",
        "gpu": {
            "name": "RTX 4070 Ti SUPER",
            "mem_used_mb": 15205,
            "mem_total_mb": 16376,
            "util_pct": 7,
            "processes": [
                {"pid": 412, "name": "llama-server", "vram_mb": 14880},
            ],
        },
        "broker": {
            "version": "1.0.0",
            "protocol": 2,
            "reachable": True,
            "dir": "/home/czy/.gpu-bridge",
        },
        "holder": holder,
        "watchdog": {
            "running": True,
            "pid": 4412,
            "idle_timeout_s": 2100,
            "poll_s": 15,
            "next_evict_in_s": 1620,
            "blocked_reason": None,
            "idle_age_s": 480.0,
        },
        "gateway": {
            "up": True,
            "url": "http://127.0.0.1:8000/v1",
            "served_model": "qwen-local",
            "selected_model": "Qwen3.8-27B-UD-Q3_K_XL",
            "selected_runtime": "upstream-mtp",
            "cache_k": "q4_0",
            "cache_v": "q4_0",
            "context_length": 65536,
            "active_requests": 0,
            "idle_status": "runtime_loaded",
            "seconds_until_offload": 1320,
            "idle_timeout_seconds": 1800,
            "idle_offload_enabled": True,
        },
        "catalog": copy.deepcopy(CATALOG),
        "services": _services("llama"),
        "events": [
            {"id": 1, "ts": "2026-10-13T20:02:11+08:00", "event": "released", "runtime_id": "mineru", "detail": None},
            {"id": 2, "ts": "2026-10-13T21:12:40+08:00", "event": "idle_offload", "runtime_id": "locate", "detail": None},
            {"id": 3, "ts": "2026-10-13T22:30:58+08:00", "event": "evicted", "runtime_id": "ocr", "detail": "signal"},
            {"id": 4, "ts": "2026-10-13T22:31:05+08:00", "event": "acquired", "runtime_id": "llama", "detail": "upstream-mtp"},
        ],
    }


def _stamp(state: dict[str, Any], event: str, runtime: str, detail: str | None = None) -> None:
    rows = state["events"]
    nxt = max((row["id"] for row in rows), default=0) + 1
    rows.append({
        "id": nxt,
        "ts": SAMPLED_AT,
        "event": event,
        "runtime_id": runtime,
        "detail": detail,
    })
    del rows[:-12]


def _free_gpu(state: dict[str, Any]) -> None:
    state["holder"] = None
    state["gpu"]["mem_used_mb"] = 1480
    state["gpu"]["util_pct"] = 1
    state["gpu"]["processes"] = []
    state["watchdog"]["next_evict_in_s"] = None
    state["watchdog"]["blocked_reason"] = "no_holder"
    state["watchdog"]["idle_age_s"] = None
    state["gateway"]["idle_status"] = "runtime_offloaded"
    state["gateway"]["seconds_until_offload"] = None
    state["services"] = _services(None)


def _hold(state: dict[str, Any], runtime: str, *, label: str, pid: int, proc: str, vram: int) -> None:
    state["holder"] = {
        "runtime_id": runtime,
        "label": label,
        "alive": True,
        "busy": False,
        "pid": pid,
        "acquired_at": SAMPLED_AT,
        "last_active_at": SAMPLED_AT,
        "vram_mb": vram,
        "idle_age_s": 0.0,
        "state": "active",
        "shutdown_url": None,
    }
    state["gpu"]["mem_used_mb"] = min(16376, vram + 400)
    state["gpu"]["util_pct"] = 12
    state["gpu"]["processes"] = [{"pid": pid, "name": proc, "vram_mb": vram}]
    if state["watchdog"]["running"]:
        state["watchdog"]["idle_age_s"] = 0.0
        state["watchdog"]["next_evict_in_s"] = state["watchdog"]["idle_timeout_s"]
        state["watchdog"]["blocked_reason"] = None
    state["services"] = _services(runtime)


class MockWorld:
    def __init__(self) -> None:
        self.state = fresh_state()
        self.logs = {"llama": LLAMA_LOG, **OTHER_LOGS}

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def events(self, limit: int) -> dict[str, Any]:
        rows = list(self.state["events"])[-limit:]
        return {"ok": True, "events": rows}

    def evict(self) -> str:
        holder = self.state["holder"]
        runtime = holder["runtime_id"] if holder else "none"
        _free_gpu(self.state)
        _stamp(self.state, "evicted", runtime, "dashboard")
        return f"evicted {runtime}"

    def install(self) -> str:
        self.state["broker"]["reachable"] = True
        self.state["broker"]["version"] = "1.0.0"
        _stamp(self.state, "install", "broker", "1.0.0")
        return "broker installed (sample)"

    def watchdog(self, start: bool) -> str:
        wd = self.state["watchdog"]
        wd["running"] = start
        wd["pid"] = 4412 if start else None
        if not start:
            wd["next_evict_in_s"] = None
            wd["blocked_reason"] = "stopped"
        elif self.state["holder"] is None:
            wd["next_evict_in_s"] = None
            wd["blocked_reason"] = "no_holder"
        else:
            wd["blocked_reason"] = None
            wd["next_evict_in_s"] = wd["idle_timeout_s"]
        _stamp(self.state, "watchdog_start" if start else "watchdog_stop", "watchdog")
        return "watchdog started" if start else "watchdog stopped"

    def llama_offload(self) -> str:
        if self.state["holder"] and self.state["holder"]["runtime_id"] == "llama":
            _free_gpu(self.state)
        self.state["gateway"]["up"] = True
        self.state["gateway"]["idle_status"] = "runtime_offloaded"
        _stamp(self.state, "released", "llama", "offload")
        return "llama offloaded"

    def llama_stop(self) -> str:
        self.llama_offload()
        self.state["gateway"]["up"] = False
        self.state["gateway"]["idle_status"] = "gateway_down"
        return "llama and gateway stopped"

    def llama_start(self) -> str:
        gw = self.state["gateway"]
        gw["up"] = True
        gw["idle_status"] = "runtime_loaded"
        gw["seconds_until_offload"] = gw["idle_timeout_seconds"]
        model = gw["selected_model"]
        lane = gw["selected_runtime"]
        _hold(
            self.state,
            "llama",
            label=f"wingpu:{lane}:{model}",
            pid=412,
            proc="llama-server",
            vram=14880 if "IQ2" not in model else 11000,
        )
        self.state["holder"]["shutdown_url"] = "http://127.0.0.1:8799/shutdown"
        _stamp(self.state, "acquired", "llama", lane)
        return f"llama started ({model}, {lane})"

    def llama_apply(self, model: str, lane: str, cache_k: str, cache_v: str) -> str:
        entry = next(item for item in self.state["catalog"]["models"] if item["name"] == model)
        gw = self.state["gateway"]
        gw["selected_model"] = model
        gw["selected_runtime"] = lane
        gw["cache_k"] = cache_k
        gw["cache_v"] = cache_v
        gw["context_length"] = entry["context_length"]
        self.llama_start()
        return f"applied {model} on {lane}"

    def service(self, runtime: str, action: str, *, engine: str | None = None) -> str:
        titles = {
            "ocr": ("ocr", "ocr-server", 8200),
            "locate": ("locate", "locate-server", 8600),
            "mineru": ("mineru", "mineru", 9000),
            "live-asr": ("live-asr", "whisper", 6400),
            "live-asr-qwen": ("live-asr-qwen", "qwen-asr", 10400),
        }
        runtime_id, proc, vram = titles[runtime]
        if action == "stop":
            if self.state["holder"] and self.state["holder"]["runtime_id"] == runtime_id:
                _free_gpu(self.state)
            _stamp(self.state, "released", runtime_id)
            return f"{runtime_id} stopped"
        if action == "status":
            return f"{runtime_id} status"
        _hold(self.state, runtime_id, label=runtime_id, pid=900, proc=proc, vram=vram)
        if engine and runtime_id.startswith("live-asr"):
            self.state["services"]["asr"]["engine"] = engine
        _stamp(self.state, "acquired", runtime_id, engine)
        return f"{runtime_id} started"

    def log_text(self, source: str, n: int) -> str:
        text = self.logs.get(source, "")
        lines = text.splitlines()
        return "\n".join(lines[-n:])
