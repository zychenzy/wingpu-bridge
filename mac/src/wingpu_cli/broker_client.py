"""
Mac-side client for the WSL gpu_broker.py lease arbiter.

This module is deliberately standalone: it depends only on a generic `runner`
callable, not on any wingpu internals, so the other bridges (locatectl, ocrctl,
minerctl) can reuse it verbatim in phase 2 by supplying their own runner.

    runner(remote_script: str, *, timeout: float | None = None) -> tuple[int, str]

The runner executes `remote_script` inside the remote WSL shell (for wingpu,
that is `run_wsl_script`) and returns (returncode, stdout_text). Every broker
command prints a single JSON line to stdout, which these helpers parse back.
"""
from __future__ import annotations

import base64
import json
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

Runner = Callable[..., Tuple[int, str]]


def _broker_invocation(remote_dir: str, args: List[str]) -> str:
    quoted = " ".join(shlex.quote(a) for a in args)
    d = shlex.quote(remote_dir)
    return f"python3 {d}/gpu_broker.py --dir {d} {quoted}"


def _parse_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Return the last JSON object printed on stdout, or None if there is none."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    for line in reversed(lines):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                continue
    return None


def _opt(args: List[str], flag: str, value: Optional[Any]) -> None:
    if value is not None:
        args.extend([flag, str(value)])


def is_installed(runner: Runner, remote_dir: str, *, timeout: float = 20) -> bool:
    script = f"test -f {shlex.quote(remote_dir)}/gpu_broker.py && echo present || echo missing"
    _rc, out = runner(script, timeout=timeout)
    return "present" in (out or "")


def install(runner: Runner, remote_dir: str, script_bytes: bytes, *, timeout: float = 60) -> bool:
    """Upload gpu_broker.py to the WSL host (base64 over the runner's stdin channel)."""
    b64 = base64.b64encode(script_bytes).decode()
    d = shlex.quote(remote_dir)
    script = (
        f"mkdir -p {d} {d}/run {d}/logs\n"
        f"echo {b64} | base64 -d > {d}/gpu_broker.py\n"
        f"chmod +x {d}/gpu_broker.py\n"
        f"python3 {d}/gpu_broker.py --dir {d} status >/dev/null 2>&1 || true\n"
        f"echo broker_installed\n"
    )
    rc, out = runner(script, timeout=timeout)
    return rc == 0 and "broker_installed" in (out or "")


def acquire(
    runner: Runner,
    *,
    remote_dir: str,
    runtime: str,
    pidfile: Optional[str] = None,
    pid: Optional[int] = None,
    shutdown_url: Optional[str] = None,
    vram_mb: Optional[int] = None,
    label: Optional[str] = None,
    wait: float = 0.0,
    force: bool = False,
    busy_util: Optional[int] = None,
    vram_floor: Optional[int] = None,
    free_timeout: Optional[float] = None,
    timeout: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    args = ["acquire", "--runtime", runtime]
    _opt(args, "--pidfile", pidfile)
    _opt(args, "--pid", pid)
    _opt(args, "--shutdown-url", shutdown_url)
    _opt(args, "--vram-mb", vram_mb)
    _opt(args, "--label", label)
    _opt(args, "--wait", wait)
    _opt(args, "--busy-util", busy_util)
    _opt(args, "--vram-floor", vram_floor)
    _opt(args, "--free-timeout", free_timeout)
    if force:
        args.append("--force")
    # Allow the broker's own --wait plus VRAM-drain to finish before the SSH call times out.
    call_timeout = timeout if timeout is not None else (wait + (free_timeout or 60) + 30)
    _rc, out = runner(_broker_invocation(remote_dir, args), timeout=call_timeout)
    return _parse_json(out)


def release(runner: Runner, *, remote_dir: str, runtime: str, token: Optional[str] = None,
            timeout: float = 30) -> Optional[Dict[str, Any]]:
    args = ["release", "--runtime", runtime]
    _opt(args, "--token", token)
    _rc, out = runner(_broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def touch(runner: Runner, *, remote_dir: str, runtime: str, timeout: float = 20) -> Optional[Dict[str, Any]]:
    _rc, out = runner(_broker_invocation(remote_dir, ["touch", "--runtime", runtime]), timeout=timeout)
    return _parse_json(out)


def evict(runner: Runner, *, remote_dir: str, runtime: Optional[str] = None,
          timeout: float = 90) -> Optional[Dict[str, Any]]:
    args = ["evict"]
    _opt(args, "--runtime", runtime)
    _rc, out = runner(_broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def status(runner: Runner, *, remote_dir: str, timeout: float = 20) -> Optional[Dict[str, Any]]:
    _rc, out = runner(_broker_invocation(remote_dir, ["status", "--json"]), timeout=timeout)
    return _parse_json(out)


def watchdog_start(runner: Runner, *, remote_dir: str, idle_timeout: float, poll: float,
                   busy_util: int, timeout: float = 30) -> Tuple[int, str]:
    d = shlex.quote(remote_dir)
    pidfile = f"{d}/run/watchdog.pid"
    inner = (
        f"python3 {d}/gpu_broker.py --dir {d} watchdog "
        f"--idle-timeout {int(idle_timeout)} --poll {int(poll)} --busy-util {int(busy_util)}"
    )
    script = (
        f"mkdir -p {d}/run {d}/logs\n"
        f"if [ -f {pidfile} ] && kill -0 \"$(cat {pidfile})\" 2>/dev/null; then\n"
        f"  echo \"watchdog_already_running:$(cat {pidfile})\"\n"
        f"else\n"
        f"  nohup {inner} > {d}/logs/watchdog.log 2>&1 &\n"
        f"  echo \"watchdog_started:$!\"\n"
        f"fi\n"
    )
    return runner(script, timeout=timeout)


def watchdog_stop(runner: Runner, *, remote_dir: str, timeout: float = 30) -> Tuple[int, str]:
    d = shlex.quote(remote_dir)
    pidfile = f"{d}/run/watchdog.pid"
    script = (
        f"if [ -f {pidfile} ]; then\n"
        f"  PID=\"$(cat {pidfile})\"\n"
        f"  if kill -0 \"$PID\" 2>/dev/null; then kill \"$PID\" 2>/dev/null; echo \"watchdog_stopped:$PID\";\n"
        f"  else echo watchdog_not_running; fi\n"
        f"  rm -f {pidfile}\n"
        f"else echo watchdog_not_running; fi\n"
    )
    return runner(script, timeout=timeout)
