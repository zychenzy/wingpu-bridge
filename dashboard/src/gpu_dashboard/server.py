"""Loopback HTTP server for the GPU dashboard.

The process binds to 127.0.0.1 only. Request handlers reject any other peer.
Static files come from this package. Actions are dispatched by name.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .actions import ActionError, Controller, loopback_host

STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAX_BODY = 64 * 1024


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], controller: Controller) -> None:
        self.controller = controller
        super().__init__(server_address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardServer

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:  # noqa: N802
        if not self._peer_ok():
            return
        self._route()

    def do_POST(self) -> None:  # noqa: N802
        if not self._peer_ok():
            return
        self._route()

    def _peer_ok(self) -> bool:
        peer = self.client_address[0]
        if loopback_host(peer):
            return True
        self._json({"ok": False, "error": "loopback only"}, status=403)
        return False

    def _route(self) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        try:
            if path in {"/", "/index.html"}:
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path[len("/static/"):])
                return
            if path == "/api/health":
                self._json({"ok": True, "mock": self.server.controller.mock, "bind": "127.0.0.1"})
                return
            if path == "/api/status":
                self._json(self.server.controller.status())
                return
            if path == "/api/events":
                limit = _query_int(parsed.query, "limit", 50)
                self._json(self.server.controller.events(limit))
                return
            if path.startswith("/api/logs/"):
                source = path[len("/api/logs/"):]
                if not source or "/" in source:
                    raise ActionError("unknown log source", 404)
                n = _query_int(parsed.query, "n", 80)
                self._json(self.server.controller.logs(source, n))
                return
            if path.startswith("/api/jobs/"):
                job_id = path[len("/api/jobs/"):]
                if not job_id or "/" in job_id:
                    raise ActionError("unknown job", 404)
                self._json({"ok": True, "job": self.server.controller.job(job_id)})
                return
            if path.startswith("/api/actions/"):
                if self.command != "POST":
                    raise ActionError("action requires POST", 405)
                name = path[len("/api/actions/"):]
                if not name or "/" in name:
                    raise ActionError("unknown action", 404)
                job = self.server.controller.act(name, self._body())
                self._json({"ok": True, "job": job})
                return
            self._json({"ok": False, "error": "not found"}, status=404)
        except ActionError as exc:
            self._json({"ok": False, "error": str(exc)}, status=exc.status)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ActionError("body is too large")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise ActionError("body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ActionError("body must be a JSON object")
        return payload

    def _static(self, relative: str) -> None:
        root = STATIC_ROOT.resolve()
        candidate = (root / relative).resolve()
        if not _is_inside(candidate, root) or not candidate.is_file():
            self._json({"ok": False, "error": "not found"}, status=404)
            return
        data = candidate.read_bytes()
        mime = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        if candidate.suffix in {".html", ".css", ".js", ".svg", ".woff2"}:
            mime = {
                ".html": "text/html; charset=utf-8",
                ".css": "text/css; charset=utf-8",
                ".js": "text/javascript; charset=utf-8",
                ".svg": "image/svg+xml",
                ".woff2": "font/woff2",
            }[candidate.suffix]
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, payload: dict, *, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _query_int(query: str, key: str, default: int) -> int:
    raw = (parse_qs(query).get(key) or [None])[0]
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ActionError(f"{key} must be an integer") from exc


def serve(host: str, port: int, controller: Controller) -> DashboardServer:
    if host != "127.0.0.1":
        raise ActionError("gpu-dashboard only binds to 127.0.0.1", 2)
    return DashboardServer((host, port), controller)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpu-dashboard", description="Local GPU operations console")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--mock", action="store_true", help="Serve sample data and do not call the CLIs")
    parser.add_argument("--project-dir", default=None, help="Repo root that contains config/qwen_gguf_catalog.json")
    args = parser.parse_args(argv)
    host = "127.0.0.1" if args.host == "localhost" else args.host
    if host != "127.0.0.1":
        print("gpu-dashboard only binds to 127.0.0.1.", file=sys.stderr)
        return 2
    controller = Controller(mock=args.mock, project_dir=args.project_dir)
    httpd = serve(host, args.port, controller)
    bound = httpd.server_address[1]
    print(f"GPU dashboard  http://127.0.0.1:{bound}/", flush=True)
    if args.mock:
        print("Sample data. No SSH, no broker calls.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
