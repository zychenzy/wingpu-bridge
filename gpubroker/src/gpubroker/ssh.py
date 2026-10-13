"""SSH helpers for the Mac-side bridges.

These replace the copy-pasted ``ssh_run`` / ``_ssh_write_file`` blocks. Two
properties matter:

* ``timeout`` is passed to the process. A hung SSH session ends with return
  code 124 instead of blocking the CLI forever.
* File uploads travel on stdin (see ``write_file``). The remote command stays
  small, so it survives Windows' 32,767-character ``CreateProcess`` limit.

``serve_tcp_relay`` is the transport wingpu uses instead of ``ssh -L``. TCP
into WSL over the Hyper-V NAT is reset after about 15 seconds even while the
stream is active. Carrying bytes on the SSH channel to a WSL-local ``nc``
avoids that hop. OCR and live transcription should use this instead of
``ssh -N -L``.
"""
from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


class SshError(RuntimeError):
    pass


@dataclass(slots=True)
class RunResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(slots=True)
class SshTarget:
    host: str
    distro: str = "Ubuntu"
    connect_timeout: int = 8
    server_alive_interval: int = 30
    server_alive_count_max: int = 3
    control_dir: Path | None = None
    control_persist: int = 300


def control_socket(target: SshTarget) -> Path | None:
    if target.control_dir is None:
        return None
    try:
        target.control_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    safe = target.host.replace("/", "_")
    return target.control_dir / f"ssh_{safe}.sock"


def ssh_option_args(target: SshTarget, *, control: bool = True) -> list[str]:
    args = [
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(target.connect_timeout)}",
        "-o", f"ServerAliveInterval={int(target.server_alive_interval)}",
        "-o", f"ServerAliveCountMax={int(target.server_alive_count_max)}",
    ]
    sock = control_socket(target) if control else None
    if sock is not None:
        args += [
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={sock}",
            "-o", f"ControlPersist={int(target.control_persist)}",
        ]
    return args


def run_argv(argv: Sequence[str], *, input_bytes: bytes | None = None,
             timeout: float | None = None) -> RunResult:
    """Run a local argv. ``timeout`` is enforced; expiry returns code 124."""
    try:
        completed = subprocess.run(
            list(argv),
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or b""
        err = exc.stderr or b""
        if isinstance(out, str):
            out = out.encode()
        if isinstance(err, str):
            err = err.encode()
        note = f"\nCommand timed out after {timeout:g}s".encode()
        return RunResult(124, out.decode("utf-8", "replace"), (err + note).decode("utf-8", "replace"))
    return RunResult(
        completed.returncode,
        (completed.stdout or b"").decode("utf-8", "replace"),
        (completed.stderr or b"").decode("utf-8", "replace"),
    )


def run_wsl_script(target: SshTarget, script: str, *, timeout: float | None = None,
                   stdin: bytes | None = None, check: bool = False) -> RunResult:
    """Run bash inside the WSL distro on ``target.host``.

    With ``stdin`` unset, ``script`` is delivered on SSH's stdin (``bash -s``),
    so it may be large. With ``stdin`` set, ``script`` is the remote ``bash -c``
    argument and must stay short; ``stdin`` is the process input (a file body).
    """
    if stdin is None:
        remote = (
            f"wsl -d {shlex.quote(target.distro)} -- "
            "env COLUMNS=120 LINES=40 bash -seuo pipefail"
        )
        payload = script.encode("utf-8")
    else:
        remote = (
            f"wsl -d {shlex.quote(target.distro)} -- "
            f"env COLUMNS=120 LINES=40 bash -c {shlex.quote(script)}"
        )
        payload = stdin
    argv = ["ssh", *ssh_option_args(target), target.host, remote]
    result = run_argv(argv, input_bytes=payload, timeout=timeout)
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise SshError(f"WSL script failed (exit {result.returncode}).\n{detail}")
    return result


def runner_for(target: SshTarget):
    """Adapt ``run_wsl_script`` to the ``gpubroker.client`` runner signature."""

    def _runner(script: str, *, timeout: float | None = None, stdin: bytes | None = None):
        result = run_wsl_script(target, script, timeout=timeout, stdin=stdin)
        return result.returncode, result.stdout

    return _runner


def write_file(target: SshTarget, remote_path: str, data: bytes, *, timeout: float = 60) -> None:
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
    script = (
        f"mkdir -p {shlex.quote(parent)} && "
        f"cat > {shlex.quote(remote_path)} && "
        f"chmod 644 {shlex.quote(remote_path)}"
    )
    result = run_wsl_script(target, script, stdin=data, timeout=timeout)
    if result.returncode != 0:
        raise SshError(f"Failed to write {remote_path} (exit {result.returncode}).\n{result.stderr}")


def relay_backend_command(target: SshTarget, *, remote_port: int, nc_host: str = "127.0.0.1",
                          control_path: Path | None = None) -> list[str]:
    """SSH argv that connects one accepted socket to ``nc`` inside WSL.

    ``nc_host`` is resolved *inside* WSL. ``127.0.0.1`` is the right value:
    it never crosses the Hyper-V NAT. ``auto`` is not accepted here; callers
    map that sentinel before calling.
    """
    remote = f"wsl -d {shlex.quote(target.distro)} -- nc {shlex.quote(nc_host)} {int(remote_port)}"
    args = ["ssh", *ssh_option_args(target, control=control_path is None)]
    if control_path is not None:
        # The relay's master is separate from the control-plane master.
        args = [
            "ssh",
            "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={int(target.connect_timeout)}",
            "-o", f"ServerAliveInterval={int(target.server_alive_interval)}",
            "-o", f"ServerAliveCountMax={int(target.server_alive_count_max)}",
            "-o", "ControlMaster=auto",
            "-o", f"ControlPath={control_path}",
            "-o", f"ControlPersist={int(target.control_persist)}",
        ]
    args += [target.host, remote]
    return args


def serve_tcp_relay(
    *,
    bind_host: str,
    bind_port: int,
    backend_command: Sequence[str],
    prime_command: Sequence[str] | None = None,
    exit_command: Sequence[str] | None = None,
    pidfile: Path | None = None,
    prime_timeout: float = 20,
    exit_timeout: float = 10,
) -> None:
    """Listen locally and pump each connection over ``backend_command``'s stdio.

    ``backend_command`` is a fixed argv (typically ``relay_backend_command``).
    Nothing in it comes from the connected client.
    """
    if bind_host not in {"127.0.0.1", "localhost", "::1"}:
        raise SshError(f"Relay must bind to loopback, not {bind_host!r}.")

    def _handle(client: socket.socket) -> None:
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            proc = subprocess.Popen(
                list(backend_command),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
        except OSError:
            client.close()
            return
        assert proc.stdin is not None and proc.stdout is not None
        in_fd, out_fd = proc.stdin.fileno(), proc.stdout.fileno()

        def _client_to_backend() -> None:
            try:
                while True:
                    data = client.recv(65536)
                    if not data:
                        break
                    os.write(in_fd, data)
            except OSError:
                pass
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

        pump = threading.Thread(target=_client_to_backend, daemon=True)
        pump.start()
        try:
            while True:
                chunk = os.read(out_fd, 65536)
                if not chunk:
                    break
                client.sendall(chunk)
        except OSError:
            pass
        finally:
            try:
                client.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            client.close()
            if proc.poll() is None:
                proc.terminate()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind_host, bind_port))
    listener.listen(128)
    if pidfile is not None:
        pidfile.parent.mkdir(parents=True, exist_ok=True)
        pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")

    if prime_command:
        run_argv(prime_command, timeout=prime_timeout)

    def _handle_signal(_signum: int, _frame: object) -> None:
        try:
            listener.close()
        except OSError:
            pass

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_signal)
        signal.signal(signal.SIGINT, _handle_signal)
    try:
        while True:
            try:
                client, _addr = listener.accept()
            except OSError:
                break
            threading.Thread(target=_handle, args=(client,), daemon=True).start()
    finally:
        if pidfile is not None:
            pidfile.unlink(missing_ok=True)
        if exit_command:
            run_argv(exit_command, timeout=exit_timeout)
            time.sleep(0)
