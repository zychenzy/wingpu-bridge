"""Broker settings shared by the bridges.

Each project keeps a ``[broker]`` table in its own TOML. ``load_section``
copies only the keys this package understands, so a project can keep extra
keys without the loader crashing, and a missing key keeps the default.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(slots=True)
class BrokerConfig:
    enabled: bool = True
    runtime_id: str = ""
    remote_dir: str = "~/.gpu-bridge"
    vram_floor_mb: int = 2000
    free_timeout_seconds: float = 60
    acquire_wait_seconds: float = 30
    busy_util_pct: int = 15
    stale_grace_seconds: float = 45
    watchdog_idle_timeout_seconds: float = 900
    watchdog_poll_seconds: float = 15
    # How often a cooperating client should refresh the lease while work is in flight.
    touch_interval_seconds: float = 30
    # Loopback port of the optional graceful-shutdown helper inside WSL. 0 disables it.
    shutdown_port: int = 8799


def load_section(data: Mapping[str, Any] | None) -> BrokerConfig:
    """Build a config from a TOML table, ignoring keys this package does not own."""
    raw = dict(data or {})
    known = {item.name for item in fields(BrokerConfig)}
    return BrokerConfig(**{key: value for key, value in raw.items() if key in known})


def resolve_remote_dir(config: BrokerConfig, *, remote_home: str, fallback: str = "") -> str:
    """Expand ``~`` against the WSL home and fall back when ``remote_dir`` is empty."""
    configured = (config.remote_dir or "").strip() or (fallback or "").strip() or "~/.gpu-bridge"
    if configured == "~":
        return remote_home
    if configured.startswith("~/"):
        return posixpath.join(remote_home, configured[2:])
    return configured


def shutdown_url(config: BrokerConfig, *, host: str = "127.0.0.1") -> str | None:
    """URL the broker POSTs before signalling a holder. None when the helper is disabled."""
    port = int(config.shutdown_port or 0)
    if port <= 0:
        return None
    return f"http://{host}:{port}/shutdown"
