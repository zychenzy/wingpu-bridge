"""Client, SSH, config, and lease-helper tests."""
from __future__ import annotations

import shlex
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from gpubroker import client, config, lease
from gpubroker.broker import PROTOCOL, VERSION
from gpubroker.ssh import (
    RunResult,
    SshError,
    SshTarget,
    bash_s_payload,
    relay_backend_command,
    run_argv,
    serve_tcp_relay,
)


# cmd.exe operators. Single quotes do not quote them; double quotes do.
_CMD_OPERATORS = set("&|<>^()")


def cmd_segments(command: str) -> list[str]:
    """Split ``command`` the way cmd.exe splits on operators.

    Windows OpenSSH runs the remote command through cmd.exe. A ``bash -c``
    argument wrapped in single quotes is not protected: ``&&`` and ``>``
    become command separators, and bash then reports an unmatched ``'``.
    """
    segments: list[str] = []
    buf: list[str] = []
    in_double = False
    for ch in command:
        if ch == '"':
            in_double = not in_double
            buf.append(ch)
            continue
        if not in_double and ch in _CMD_OPERATORS:
            segments.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    segments.append("".join(buf))
    return [segment for segment in segments if segment.strip()]


class ClientTests(unittest.TestCase):
    def test_parse_json_takes_last_json_line(self):
        out = "some log noise\n{\"granted\": true, \"runtime\": \"llama\"}\n"
        self.assertEqual(client._parse_json(out), {"granted": True, "runtime": "llama"})

    def test_parse_json_returns_none_without_json(self):
        self.assertIsNone(client._parse_json("no json here\n"))

    def test_acquire_builds_invocation_and_parses(self):
        seen = {}

        def runner(script, *, timeout=None, stdin=None):
            seen["script"] = script
            seen["timeout"] = timeout
            return 0, '{"granted": true, "evicted": "ocr"}'

        result = client.acquire(
            runner, remote_dir="/home/czy/.gpu-bridge", runtime="llama",
            pidfile="/home/czy/.gpu-bridge/run/upstream.pid", vram_mb=14000,
            wait=30, force=True, busy_util=15, vram_floor=2000, free_timeout=60,
            stale_grace=45, shutdown_url="http://127.0.0.1:8799/shutdown",
        )
        self.assertEqual(result, {"granted": True, "evicted": "ocr"})
        self.assertIn("gpu_broker.py", seen["script"])
        self.assertIn("acquire --runtime llama", seen["script"])
        self.assertIn("--force", seen["script"])
        self.assertIn("--stale-grace 45", seen["script"])
        self.assertIn("--shutdown-url http://127.0.0.1:8799/shutdown", seen["script"])
        self.assertIn("--pidfile /home/czy/.gpu-bridge/run/upstream.pid", seen["script"])

    def test_install_streams_bytes_on_stdin(self):
        seen = []

        def runner(script, *, timeout=None, stdin=None):
            seen.append((script, stdin))
            if stdin is not None:
                return 0, ""
            return 0, '{"version": "%s", "protocol": %s}\nbroker_installed\n' % (VERSION, PROTOCOL)

        ok = client.install(runner, "/home/czy/.gpu-bridge", b"print('broker')\n")
        self.assertTrue(ok)
        upload_script, payload = seen[0]
        self.assertEqual(payload, b"print('broker')\n")
        self.assertNotIn(b"print('broker')", upload_script.encode())
        self.assertNotIn("base64", upload_script)
        self.assertLess(len(upload_script), 500)

    def test_ensure_installed_does_not_upgrade_without_permission(self):
        def runner(script, *, timeout=None, stdin=None):
            if "version" in script:
                return 0, '{"version": "0.0.1", "protocol": 1}\n'
            if "test -f" in script:
                return 0, "present\n"
            raise AssertionError(script)

        status = client.ensure_installed(runner, "/d", upgrade=False)
        self.assertEqual(status, "mismatch")

    def test_ensure_installed_upgrades_when_allowed(self):
        calls = {"install": 0}

        def runner(script, *, timeout=None, stdin=None):
            if stdin is not None:
                calls["install"] += 1
                return 0, ""
            if "test -f" in script:
                return 0, "present\n"
            if "version" in script and calls["install"]:
                return 0, '{"version": "%s", "protocol": %s}\nbroker_installed\n' % (VERSION, PROTOCOL)
            if "version" in script:
                return 0, '{"version": "0.0.1", "protocol": 1}\n'
            return 0, ""

        status = client.ensure_installed(runner, "/d", b"x", upgrade=True)
        self.assertEqual(status, "upgraded")
        self.assertEqual(calls["install"], 1)

    def test_install_failure_includes_remote_exit_and_stderr(self):
        detail = "/bin/bash: -c: line 1: unexpected EOF while looking for matching `'"

        def runner(script, *, timeout=None, stdin=None):
            if stdin is not None:
                return 2, "", detail
            raise AssertionError(script)

        with self.assertRaises(client.InstallError) as caught:
            client.install(runner, "/home/czy/.gpu-bridge", b"print('broker')\n")
        message = str(caught.exception)
        self.assertIn("Failed to install the GPU broker at /home/czy/.gpu-bridge", message)
        self.assertIn("upload gpu_broker.py", message)
        self.assertIn("exit 2", message)
        self.assertIn("unexpected EOF while looking for matching", message)
        self.assertEqual(caught.exception.returncode, 2)
        self.assertEqual(caught.exception.detail, detail)

    def test_install_failure_reads_stderr_folded_into_stdout(self):
        def runner(script, *, timeout=None, stdin=None):
            if stdin is not None:
                return 2, "unexpected EOF while looking for matching `'"
            raise AssertionError(script)

        with self.assertRaises(client.InstallError) as caught:
            client.install(runner, "/home/czy/.gpu-bridge", b"x")
        self.assertIn("exit 2", str(caught.exception))
        self.assertIn("unexpected EOF", caught.exception.detail)

    def test_ensure_installed_does_not_swallow_upload_errors(self):
        def runner(script, *, timeout=None, stdin=None):
            if stdin is not None:
                return 2, "", "write failed"
            if "test -f" in script:
                return 0, "missing\n"
            raise AssertionError(script)

        with self.assertRaises(client.InstallError) as caught:
            client.ensure_installed(runner, "/home/czy/.gpu-bridge", b"x", upgrade=True)
        self.assertIn("exit 2", str(caught.exception))
        self.assertIn("write failed", str(caught.exception))

    def test_watchdog_start_forwards_drain_options(self):
        seen = {}

        def runner(script, *, timeout=None, stdin=None):
            seen["script"] = script
            return 0, "watchdog_started:9\n"

        client.watchdog_start(
            runner, remote_dir="/d", idle_timeout=2100, poll=15, busy_util=15,
            vram_floor=1800, free_timeout=45, stale_grace=30,
        )
        self.assertIn("--vram-floor 1800", seen["script"])
        self.assertIn("--free-timeout 45", seen["script"])
        self.assertIn("--stale-grace 30", seen["script"])
        self.assertIn("--idle-timeout 2100", seen["script"])


class SshTests(unittest.TestCase):
    def test_run_argv_enforces_timeout(self):
        started = time.time()
        result = run_argv(["sleep", "5"], timeout=0.3)
        elapsed = time.time() - started
        self.assertEqual(result.returncode, 124)
        self.assertLess(elapsed, 3)

    def test_run_wsl_script_passes_timeout_and_stdin(self):
        target = SshTarget(host="win-gpu", distro="Ubuntu")
        with mock.patch("gpubroker.ssh.run_argv", return_value=RunResult(0, "ok", "")) as run:
            from gpubroker.ssh import run_wsl_script
            run_wsl_script(target, "echo hi", timeout=12)
            argv = run.call_args.args[0]
            self.assertEqual(run.call_args.kwargs["timeout"], 12)
            self.assertIn("bash -seuo pipefail", argv[-1])
            self.assertIn(b"echo hi", run.call_args.kwargs["input_bytes"])

            run_wsl_script(target, "cat > /tmp/x", stdin=b"payload", timeout=8)
            remote = run.call_args.args[0][-1]
            self.assertEqual(run.call_args.kwargs["timeout"], 8)
            self.assertNotIn("bash -c", remote)
            self.assertEqual(remote, "wsl -d Ubuntu -- env COLUMNS=120 LINES=40 bash -seuo pipefail")
            body = run.call_args.kwargs["input_bytes"]
            self.assertNotEqual(body, b"payload")
            self.assertNotIn(b"payload", remote.encode())
            self.assertIn(b"base64 -d", body)

    def test_upload_remote_command_survives_cmd_quoting(self):
        """The Mac → Windows OpenSSH → cmd.exe → wsl → bash upload must not nest quotes.

        ``bash -c 'mkdir && cat > file'`` is split by cmd.exe on ``&&`` and ``>``
        (single quotes are literal there). bash then exits 2 with
        ``unexpected EOF while looking for matching '`` and the old broker
        file is left in place. The upload command has to stay on ``bash -s``.
        """
        target = SshTarget(host="win-gpu", distro="Ubuntu")
        remote_path = "/home/czy/.gpu-bridge/gpu_broker.py"
        body = b"print('broker')\n# GPUBROKER_B64_EOF\n&& > '\n"
        script = (
            "mkdir -p /home/czy/.gpu-bridge && "
            f"cat > {remote_path} && "
            f"chmod 755 {remote_path}"
        )
        legacy = (
            "wsl -d Ubuntu -- env COLUMNS=120 LINES=40 bash -c "
            + shlex.quote(script)
        )
        legacy_segments = cmd_segments(legacy)
        self.assertGreater(len(legacy_segments), 1)
        self.assertNotEqual(legacy_segments[0].count("'") % 2, 0)

        captured = {}

        def fake_run(argv, *, input_bytes=None, timeout=None):
            captured["argv"] = list(argv)
            captured["stdin"] = input_bytes
            return RunResult(0, "", "")

        with mock.patch("gpubroker.ssh.run_argv", side_effect=fake_run):
            from gpubroker.ssh import runner_for
            ok = client.write_remote_file(runner_for(target), remote_path, body)
        self.assertTrue(ok)
        remote = captured["argv"][-1]
        joined = " ".join(captured["argv"])
        self.assertEqual(remote, "wsl -d Ubuntu -- env COLUMNS=120 LINES=40 bash -seuo pipefail")
        self.assertEqual(cmd_segments(remote), [remote])
        self.assertNotIn("bash -c", remote)
        self.assertNotIn("'", remote)
        self.assertNotIn("&", remote)
        self.assertNotIn(">", remote)
        self.assertLess(len(remote), 200)
        self.assertNotIn(body.decode(), joined)
        self.assertNotIn("print('broker')", joined)
        self.assertEqual(captured["stdin"], bash_s_payload(script, body))
        self.assertIn(remote_path.encode(), captured["stdin"])

        with tempfile.TemporaryDirectory() as temp_dir:
            parent = str(Path(temp_dir) / "sub dir")
            dest = str(Path(parent) / "gpu'broker.py")
            upload = (
                f"mkdir -p {shlex.quote(parent)} && "
                f"cat > {shlex.quote(dest)} && "
                f"chmod 755 {shlex.quote(dest)}"
            )
            payload = bash_s_payload(upload, body)
            self.assertNotIn(body, payload.split(b"\n")[0])
            proc = subprocess.run(
                ["bash", "-seuo", "pipefail"],
                input=payload,
                capture_output=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr.decode())
            written = Path(dest).read_bytes()
            self.assertEqual(written, body)
            self.assertTrue(Path(dest).stat().st_mode & 0o111)

    def test_upload_failure_includes_ssh_stderr(self):
        target = SshTarget(host="win-gpu", distro="Ubuntu")
        detail = "/bin/bash: -c: line 1: unexpected EOF while looking for matching `'\n"
        with mock.patch("gpubroker.ssh.run_argv", return_value=RunResult(2, "", detail)):
            from gpubroker.ssh import runner_for
            with self.assertRaises(client.InstallError) as caught:
                client.install(runner_for(target), "/home/czy/.gpu-bridge", b"print('x')\n")
        self.assertEqual(caught.exception.returncode, 2)
        self.assertIn("unexpected EOF while looking for matching", str(caught.exception))
        self.assertIn("/home/czy/.gpu-bridge", str(caught.exception))
        self.assertIn("upload gpu_broker.py", str(caught.exception))

    def test_relay_command_uses_nc_inside_wsl(self):
        target = SshTarget(host="win-gpu")
        argv = relay_backend_command(target, remote_port=8010, nc_host="127.0.0.1")
        self.assertNotIn("-L", argv)
        self.assertIn("nc 127.0.0.1 8010", argv[-1])

    def test_relay_refuses_non_loopback(self):
        with self.assertRaises(SshError):
            serve_tcp_relay(bind_host="0.0.0.0", bind_port=9, backend_command=["cat"])

    def test_relay_proxies_stdio(self):
        port = _free_port()
        thread = threading.Thread(
            target=serve_tcp_relay,
            kwargs={"bind_host": "127.0.0.1", "bind_port": port, "backend_command": ["cat"]},
            daemon=True,
        )
        thread.start()
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                client_sock = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                break
            except OSError:
                time.sleep(0.05)
        else:
            self.fail("relay did not listen")
        try:
            client_sock.sendall(b"hello-relay")
            client_sock.shutdown(socket.SHUT_WR)
            data = b""
            while b"hello-relay" not in data:
                chunk = client_sock.recv(64)
                if not chunk:
                    break
                data += chunk
        finally:
            client_sock.close()
        self.assertIn(b"hello-relay", data)


class LeaseAndConfigTests(unittest.TestCase):
    def test_load_section_ignores_unknown_keys(self):
        loaded = config.load_section({"runtime_id": "ocr", "not_a_key": 1, "busy_util_pct": 20})
        self.assertEqual(loaded.runtime_id, "ocr")
        self.assertEqual(loaded.busy_util_pct, 20)
        self.assertEqual(loaded.vram_floor_mb, 2000)

    def test_resolve_remote_dir_expands_home(self):
        cfg = config.BrokerConfig(remote_dir="~/bridge")
        self.assertEqual(config.resolve_remote_dir(cfg, remote_home="/home/czy"), "/home/czy/bridge")

    def test_activity_touch_rate_limits(self):
        calls = []
        toucher = lease.ActivityTouch(lambda: calls.append(1) or {"touched": True}, interval=30)
        self.assertTrue(toucher.ping())
        self.assertFalse(toucher.ping())
        self.assertEqual(calls, [1])

    def test_session_refuses_to_install_for_siblings(self):
        def runner(script, *, timeout=None, stdin=None):
            if "test -f" in script:
                return 0, "missing\n"
            raise AssertionError("sibling tried to upload the broker")

        session = lease.LeaseSession(
            runner,
            config.BrokerConfig(runtime_id="locate"),
            remote_dir="/home/czy/.gpu-bridge",
            allow_install=False,
        )
        with mock.patch("gpubroker.lease._warn"):
            result = session.acquire(pidfile="/tmp/server.pid", label="locate")
        self.assertIsNone(result)

    def test_session_reports_install_failure_without_uploading_for_siblings(self):
        def runner(script, *, timeout=None, stdin=None):
            if stdin is not None:
                return 2, "", "unexpected EOF while looking for matching `'"
            if "test -f" in script:
                return 0, "missing\n"
            raise AssertionError("sibling tried to upload the broker")

        sibling = lease.LeaseSession(
            runner,
            config.BrokerConfig(runtime_id="locate"),
            remote_dir="/home/czy/.gpu-bridge",
            allow_install=False,
        )
        with mock.patch("gpubroker.lease._warn") as warn:
            self.assertIsNone(sibling.acquire(pidfile="/tmp/server.pid", label="locate"))
        self.assertIn("will not upload", warn.call_args.args[0])
        self.assertNotIn("unexpected EOF", warn.call_args.args[0])

        installer = lease.LeaseSession(
            runner,
            config.BrokerConfig(runtime_id="llama"),
            remote_dir="/home/czy/.gpu-bridge",
            allow_install=True,
        )
        with mock.patch("gpubroker.lease._warn") as warn:
            self.assertIsNone(installer.acquire(pidfile="/tmp/server.pid", label="wingpu"))
        message = warn.call_args.args[0]
        self.assertIn("exit 2", message)
        self.assertIn("unexpected EOF", message)

    def test_shutdown_launch_snippet_quotes_the_whole_path(self):
        script = lease.shutdown_helper_launch("/home/czy/.gpu-bridge", port=8799, pidfile="/tmp/llama.pid")
        self.assertIn("/home/czy/.gpu-bridge/shutdown_helper.py", script)
        self.assertIn("--port 8799", script)
        self.assertNotIn("base64", script)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


if __name__ == "__main__":
    unittest.main()
