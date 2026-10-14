"""Lease helpers that replace the copy-pasted acquire/release blocks.

Siblings construct a :class:`LeaseSession` with their runner and runtime id.
``allow_install`` stays false: a missing or old broker prints the wingpu
command and the call fails open, matching today's behavior, without uploading
a second copy of the broker.

Touch from the code path that is actually doing work (a request, or a stream
that is open). A timer that touches while the process is merely resident
defeats idle offload.
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Any, Callable, Optional

from . import client
from .config import BrokerConfig, shutdown_url


TouchFn = Callable[[], Optional[dict[str, Any]]]


class ActivityTouch:
    """Rate-limit ``touch`` calls. ``ping`` is safe from a request handler."""

    def __init__(self, touch: TouchFn, interval: float = 30.0) -> None:
        self._touch = touch
        self.interval = max(1.0, float(interval))
        self._last = 0.0
        self._lock = threading.Lock()

    def ping(self) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last < self.interval:
                return False
            self._last = now
        try:
            result = self._touch()
        except Exception:
            with self._lock:
                if self._last == now:
                    self._last = 0.0
            return False
        ok = bool(result and result.get("touched"))
        if not ok:
            with self._lock:
                if self._last == now:
                    self._last = 0.0
        return ok


class TouchWhileActive:
    """Touch on an interval only while ``active`` says work is in flight.

    Live transcription should start this when a stream opens and stop it when
    the stream ends. Leaving it running for the life of the process keeps the
    watchdog from ever seeing the model as idle.
    """

    def __init__(self, touch: TouchFn, *, interval: float = 30.0, active: Callable[[], bool]) -> None:
        self._touch = touch
        self.interval = max(1.0, float(interval))
        self._active = active
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpubroker-touch", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            if not self._active():
                continue
            try:
                self._touch()
            except Exception:
                continue


def _warn(message: str) -> None:
    print(message, file=sys.stderr)


class LeaseSession:
    """Acquire, touch, and release one runtime's lease."""

    def __init__(
        self,
        runner: client.Runner,
        config: BrokerConfig,
        *,
        remote_dir: str,
        allow_install: bool = False,
    ) -> None:
        self.runner = runner
        self.config = config
        self.remote_dir = remote_dir
        self.allow_install = allow_install

    def prepare(self) -> str:
        """Install (wingpu only) or report whether the remote broker matches this package."""
        try:
            status = client.ensure_installed(
                self.runner,
                self.remote_dir,
                upgrade=self.allow_install,
            )
        except client.InstallError as exc:
            _warn(f"[gpubroker] {exc}")
            return "failed"
        if status in {"missing", "mismatch", "failed"} and not self.allow_install:
            _warn(
                f"[gpubroker] remote broker is {status} at {self.remote_dir}. "
                "This command will not upload one. Run `wingpu broker install` on the Mac, "
                "then retry. Continuing without a lease."
            )
        elif status == "failed":
            _warn(f"[gpubroker] could not install the broker at {self.remote_dir}.")
        return status

    def shutdown_url(self) -> str | None:
        return shutdown_url(self.config)

    def acquire(
        self,
        *,
        pidfile: str | None = None,
        pid: int | None = None,
        label: str | None = None,
        vram_mb: int | None = None,
        shutdown: str | None = None,
        force: bool = False,
        no_evict: bool = False,
        wait: float | None = None,
    ) -> dict[str, Any] | None:
        """Acquire the lease. Returns the broker JSON, or None when failing open."""
        if not self.config.enabled:
            return {"ok": True, "granted": True, "skipped": True}
        status = self.prepare()
        if status in {"missing", "mismatch", "failed"} and not self.allow_install:
            return None
        if status == "failed":
            return None
        try:
            return client.acquire(
                self.runner,
                remote_dir=self.remote_dir,
                runtime=self.config.runtime_id,
                pidfile=pidfile,
                pid=pid,
                shutdown_url=shutdown if shutdown is not None else self.shutdown_url(),
                vram_mb=vram_mb,
                label=label,
                wait=self.config.acquire_wait_seconds if wait is None else wait,
                force=force,
                no_evict=no_evict,
                busy_util=self.config.busy_util_pct,
                vram_floor=self.config.vram_floor_mb,
                free_timeout=self.config.free_timeout_seconds,
                stale_grace=self.config.stale_grace_seconds,
            )
        except Exception as exc:
            _warn(f"[gpubroker] acquire failed ({exc}); continuing without a lease.")
            return None

    def release(self) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        try:
            return client.release(
                self.runner,
                remote_dir=self.remote_dir,
                runtime=self.config.runtime_id,
            )
        except Exception:
            return None

    def touch(self) -> dict[str, Any] | None:
        if not self.config.enabled:
            return None
        return client.touch(
            self.runner,
            remote_dir=self.remote_dir,
            runtime=self.config.runtime_id,
        )

    def activity(self) -> ActivityTouch:
        return ActivityTouch(self.touch, interval=self.config.touch_interval_seconds)

    def touch_while_active(self, active: Callable[[], bool]) -> TouchWhileActive:
        return TouchWhileActive(self.touch, interval=self.config.touch_interval_seconds, active=active)


def shutdown_helper_launch(remote_dir: str, *, port: int, pidfile: str) -> str:
    """Bash snippet that starts the helper if it is not already listening.

    The helper file must already be at ``$remote_dir/shutdown_helper.py``
    (``wingpu broker install`` puts it there). The snippet is a no-op when the
    file or the port is missing, so a runtime can still start.
    """
    if port <= 0:
        return "true"
    helper = _q(f"{remote_dir}/shutdown_helper.py")
    pid = _q(pidfile)
    selfpid = _q(f"{remote_dir}/run/shutdown-helper.pid")
    log = _q(f"{remote_dir}/logs/shutdown-helper.log")
    run_dir = _q(f"{remote_dir}/run")
    log_dir = _q(f"{remote_dir}/logs")
    return f"""
HELPER={helper}
PIDFILE={pid}
SELFPID={selfpid}
if [[ -f "$HELPER" ]]; then
  mkdir -p {run_dir} {log_dir}
  if [[ -f "$SELFPID" ]] && kill -0 "$(cat "$SELFPID")" 2>/dev/null; then
    :
  else
    nohup python3 "$HELPER" --port {int(port)} --pidfile "$PIDFILE" --pidfile-self "$SELFPID" \\
      > {log} 2>&1 &
    echo $! > "$SELFPID"
  fi
fi
"""


def _q(value: str) -> str:
    import shlex
    return shlex.quote(value)
