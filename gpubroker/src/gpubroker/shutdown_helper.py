#!/usr/bin/env python3
"""Loopback graceful-shutdown helper for a broker holder.

The broker POSTs ``--shutdown-url`` before it signals a pid. This process
binds 127.0.0.1 only, reads the pidfile, and SIGTERMs that pid so the runtime
can exit on its own before the broker's SIGKILL. If this helper is down, the
broker falls through to the signal itself.

wingpu uploads this file next to ``gpu_broker.py`` and starts it with the
llama runtime. Siblings can run the same file against their own pidfile:

    python3 ~/.gpu-bridge/shutdown_helper.py --port 8799 --pidfile .../server.pid
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _read_pid(path: str) -> int | None:
    try:
        return int(Path(path).expanduser().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def stop_pid(pid: int, grace: float) -> str:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return "already_dead"
    except OSError as exc:
        return f"error:{exc}"
    deadline = time.time() + grace
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return "stopped"
        time.sleep(0.2)
    return "term_sent"


def make_handler(pidfile: str, grace: float):
    class Handler(BaseHTTPRequestHandler):
        server_version = "gpubroker-shutdown/1.0"

        def log_message(self, fmt: str, *args: object) -> None:
            return

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/health":
                self._send(404, {"ok": False})
                return
            self._send(200, {"ok": True, "pid": _read_pid(pidfile)})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/shutdown":
                self._send(404, {"ok": False})
                return
            pid = _read_pid(pidfile)
            if not pid:
                self._send(200, {"ok": True, "result": "no_pid"})
                return
            self._send(200, {"ok": True, "result": stop_pid(pid, grace), "pid": pid})

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback graceful shutdown for a GPU runtime")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--pidfile", required=True)
    parser.add_argument("--grace", type=float, default=10.0)
    parser.add_argument("--pidfile-self", default=None, help="Where to write this helper's pid")
    args = parser.parse_args(argv)
    if args.pidfile_self:
        path = Path(args.pidfile_self).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{os.getpid()}\n", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.pidfile, args.grace))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
        if args.pidfile_self:
            Path(args.pidfile_self).expanduser().unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
