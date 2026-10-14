"""
Mac-side client for the WSL gpu broker.

The runner is a callable, not a wingpu import, so every bridge can share it:

    runner(script, *, timeout=None, stdin=None) -> tuple[int, str] | tuple[int, str, str]

``script`` is a short remote bash snippet. When ``stdin`` is set, the runner
must feed those bytes to the remote process and must not paste them into the
SSH command line. ``install`` depends on that: the broker file is tens of
kilobytes and does not fit on Windows' command line. The runner also must not
wrap the snippet in ``bash -c '<script>'`` on the SSH argv. Windows OpenSSH
runs that argv through cmd.exe, which splits ``&&`` and ``>`` inside single
quotes; bash then exits 2 with ``unexpected EOF while looking for matching '``.
``gpubroker.ssh.run_wsl_script`` sends the snippet to ``bash -s`` instead.

A runner may return ``(returncode, stdout)`` or ``(returncode, stdout, stderr)``.
``upgrade`` failure raises ``InstallError`` with the remote exit code and stderr.

Only wingpu should call ``ensure_installed(..., upgrade=True)``. Siblings pass
``upgrade=False`` and, on ``missing`` or ``mismatch``, tell the user to run
``wingpu broker install``.
"""
from __future__ import annotations

import json
import posixpath
import shlex
from typing import Any, Callable, Dict, List, Optional, Tuple

from .broker import PROTOCOL, VERSION

Runner = Callable[..., Tuple[int, str]]


class InstallError(RuntimeError):
    """Uploading or activating the remote broker failed.

    ``returncode`` is the remote write's exit status. ``detail`` is its stderr,
    or stdout when the runner does not report stderr separately.
    """

    def __init__(self, location: str, *, stage: str, returncode: int, detail: str) -> None:
        self.location = location
        self.stage = stage
        self.returncode = int(returncode)
        self.detail = detail or ""
        text = (
            f"Failed to install the GPU broker at {location} "
            f"({stage}, exit {self.returncode})."
        )
        stripped = self.detail.strip()
        if stripped:
            text = f"{text}\n{stripped}"
        super().__init__(text)


def _invoke(runner: Runner, script: str, **kwargs: Any) -> Tuple[int, str, str]:
    """Call ``runner``, accepting either a 2-tuple or a 3-tuple with stderr."""
    result = runner(script, **kwargs)
    rc = int(result[0])
    stdout = "" if len(result) < 2 or result[1] is None else str(result[1])
    stderr = "" if len(result) < 3 or result[2] is None else str(result[2])
    return rc, stdout, stderr


def _streams_detail(stdout: str, stderr: str) -> str:
    parts: List[str] = []
    if stdout.strip():
        parts.append(stdout.strip())
    err = stderr.strip()
    if err and err not in stdout:
        parts.append(err)
    return "\n".join(parts)


def bundled_script_bytes() -> bytes:
    """The stdlib broker file wingpu uploads to WSL."""
    from pathlib import Path
    return Path(__file__).with_name("broker.py").read_bytes()


def bundled_shutdown_helper_bytes() -> bytes:
    from pathlib import Path
    return Path(__file__).with_name("shutdown_helper.py").read_bytes()


def _broker_invocation(remote_dir: str, args: List[str]) -> str:
    quoted = " ".join(shlex.quote(arg) for arg in args)
    directory = shlex.quote(remote_dir)
    return f"python3 {directory}/gpu_broker.py --dir {directory} {quoted}"


def _parse_json(stdout: str) -> Optional[Dict[str, Any]]:
    """Return the last JSON object printed on stdout, or None if there is none."""
    lines = [line.strip() for line in (stdout or "").splitlines() if line.strip()]
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
    _rc, out, _err = _invoke(runner, script, timeout=timeout)
    return "present" in (out or "")


def write_remote_file(runner: Runner, remote_path: str, data: bytes, *, timeout: float = 60) -> bool:
    """Upload ``data`` on the runner's stdin. The remote command stays short.

    Raises ``InstallError`` with the remote exit code and stderr on failure.
    """
    parent = posixpath.dirname(remote_path) or "."
    script = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"cat > {shlex.quote(remote_path)} && "
        f"chmod 755 {shlex.quote(remote_path)}"
    )
    rc, stdout, stderr = _invoke(runner, script, stdin=data, timeout=timeout)
    if rc != 0:
        raise InstallError(
            remote_path,
            stage="write",
            returncode=rc,
            detail=_streams_detail(stdout, stderr),
        )
    return True


def install(runner: Runner, remote_dir: str, script_bytes: bytes, *, timeout: float = 60) -> bool:
    """Upload gpu_broker.py. The file bytes go on stdin, not into the command.

    Raises ``InstallError`` when the upload or the follow-up version check fails.
    """
    try:
        write_remote_file(runner, f"{remote_dir}/gpu_broker.py", script_bytes, timeout=timeout)
    except InstallError as exc:
        raise InstallError(
            remote_dir,
            stage="upload gpu_broker.py",
            returncode=exc.returncode,
            detail=exc.detail,
        ) from exc
    directory = shlex.quote(remote_dir)
    script = (
        f"mkdir -p {directory}/run {directory}/logs && "
        f"python3 {directory}/gpu_broker.py --dir {directory} version && "
        f"echo broker_installed"
    )
    rc, out, err = _invoke(runner, script, timeout=timeout)
    if rc == 0 and "broker_installed" in (out or ""):
        return True
    raise InstallError(
        remote_dir,
        stage="verify gpu_broker.py",
        returncode=rc,
        detail=_streams_detail(out, err),
    )


def install_shutdown_helper(runner: Runner, remote_dir: str, helper_bytes: bytes, *,
                            timeout: float = 60) -> bool:
    """Upload the loopback graceful-shutdown helper next to the broker."""
    try:
        return write_remote_file(runner, f"{remote_dir}/shutdown_helper.py", helper_bytes, timeout=timeout)
    except InstallError as exc:
        raise InstallError(
            remote_dir,
            stage="upload shutdown_helper.py",
            returncode=exc.returncode,
            detail=exc.detail,
        ) from exc


def version(runner: Runner, remote_dir: str, *, timeout: float = 20) -> Optional[Dict[str, Any]]:
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, ["version"]), timeout=timeout)
    return _parse_json(out)


def ensure_installed(runner: Runner, remote_dir: str, script_bytes: Optional[bytes] = None, *,
                     upgrade: bool = False, timeout: float = 60) -> str:
    """Return ``current``, ``installed``, ``upgraded``, ``missing``, or ``mismatch``.

    ``upgrade=True`` is wingpu. Everyone else leaves a missing or different
    broker in place and reports the status. A failed upgrade raises
    ``InstallError`` (remote exit code and stderr) instead of uploading a
    partial file and returning quietly.
    """
    payload = script_bytes if script_bytes is not None else bundled_script_bytes()
    installed = is_installed(runner, remote_dir, timeout=min(timeout, 20))
    remote = version(runner, remote_dir, timeout=min(timeout, 20)) if installed else None
    compatible = bool(
        remote
        and remote.get("version") == VERSION
        and int(remote.get("protocol", -1)) == PROTOCOL
    )
    if compatible:
        return "current"
    if not upgrade:
        return "mismatch" if installed else "missing"
    install(runner, remote_dir, payload, timeout=timeout)
    return "upgraded" if installed else "installed"


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
    no_evict: bool = False,
    busy_util: Optional[int] = None,
    vram_floor: Optional[int] = None,
    free_timeout: Optional[float] = None,
    stale_grace: Optional[float] = None,
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
    _opt(args, "--stale-grace", stale_grace)
    if force:
        args.append("--force")
    if no_evict:
        args.append("--no-evict")
    call_timeout = timeout if timeout is not None else (wait + (free_timeout or 60) + 30)
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, args), timeout=call_timeout)
    return _parse_json(out)


def release(runner: Runner, *, remote_dir: str, runtime: str, token: Optional[str] = None,
            timeout: float = 30) -> Optional[Dict[str, Any]]:
    args = ["release", "--runtime", runtime]
    _opt(args, "--token", token)
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def touch(runner: Runner, *, remote_dir: str, runtime: str, timeout: float = 20) -> Optional[Dict[str, Any]]:
    """Refresh the holder's idle clock. Call this from request handlers, not from a timer while idle."""
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, ["touch", "--runtime", runtime]), timeout=timeout)
    return _parse_json(out)


def evict(runner: Runner, *, remote_dir: str, runtime: Optional[str] = None,
          vram_floor: Optional[int] = None, free_timeout: Optional[float] = None,
          timeout: float = 90) -> Optional[Dict[str, Any]]:
    args = ["evict"]
    _opt(args, "--runtime", runtime)
    _opt(args, "--vram-floor", vram_floor)
    _opt(args, "--free-timeout", free_timeout)
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def status(runner: Runner, *, remote_dir: str, timeout: float = 20,
           busy_util: Optional[int] = None) -> Optional[Dict[str, Any]]:
    args = ["status", "--json"]
    _opt(args, "--busy-util", busy_util)
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def events(runner: Runner, *, remote_dir: str, limit: int = 50, since: Optional[str] = None,
           timeout: float = 20) -> Optional[Dict[str, Any]]:
    args = ["events", "--json", "--limit", str(limit)]
    _opt(args, "--since", since)
    _rc, out, _err = _invoke(runner, _broker_invocation(remote_dir, args), timeout=timeout)
    return _parse_json(out)


def watchdog_start(runner: Runner, *, remote_dir: str, idle_timeout: float, poll: float,
                   busy_util: int, vram_floor: Optional[int] = None,
                   free_timeout: Optional[float] = None, stale_grace: Optional[float] = None,
                   timeout: float = 30) -> Tuple[int, str]:
    directory = shlex.quote(remote_dir)
    pidfile = f"{directory}/run/watchdog.pid"
    flags = (
        f"--idle-timeout {int(idle_timeout)} --poll {int(poll)} --busy-util {int(busy_util)}"
    )
    if vram_floor is not None:
        flags += f" --vram-floor {int(vram_floor)}"
    if free_timeout is not None:
        flags += f" --free-timeout {int(free_timeout)}"
    if stale_grace is not None:
        flags += f" --stale-grace {int(stale_grace)}"
    inner = f"python3 {directory}/gpu_broker.py --dir {directory} watchdog {flags}"
    script = (
        f"mkdir -p {directory}/run {directory}/logs\n"
        f"if [ -f {pidfile} ] && kill -0 \"$(cat {pidfile})\" 2>/dev/null; then\n"
        f"  echo \"watchdog_already_running:$(cat {pidfile})\"\n"
        f"else\n"
        f"  nohup {inner} > {directory}/logs/watchdog.log 2>&1 &\n"
        f"  echo \"watchdog_started:$!\"\n"
        f"fi\n"
    )
    rc, out, _err = _invoke(runner, script, timeout=timeout)
    return rc, out


def watchdog_stop(runner: Runner, *, remote_dir: str, timeout: float = 30) -> Tuple[int, str]:
    directory = shlex.quote(remote_dir)
    pidfile = f"{directory}/run/watchdog.pid"
    script = (
        f"if [ -f {pidfile} ]; then\n"
        f"  PID=\"$(cat {pidfile})\"\n"
        f"  if kill -0 \"$PID\" 2>/dev/null; then kill \"$PID\" 2>/dev/null; echo \"watchdog_stopped:$PID\";\n"
        f"  else echo watchdog_not_running; fi\n"
        f"  rm -f {pidfile}\n"
        f"else echo watchdog_not_running; fi\n"
    )
    rc, out, _err = _invoke(runner, script, timeout=timeout)
    return rc, out
