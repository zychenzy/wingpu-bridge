#!/usr/bin/env python3
"""End-to-end tests for bridge fixes.

Tests cover the WSL-side components (bridge_db, bridge_ctl) and
verify all code-review fixes are working correctly.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add wsl/ to path so we can import bridge_db and bridge_ctl
WSL_DIR = Path(__file__).resolve().parents[2] / "wsl"
sys.path.insert(0, str(WSL_DIR))

from bridge_db import BridgeDB


class TestBridgeDBFixes(unittest.TestCase):
    """Tests for bridge_db.py fixes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.db = BridgeDB(self.db_path)

    def _make_job(self, **overrides):
        defaults = {
            "job_id": f"test_{time.monotonic_ns()}",
            "project": "test-project",
            "cmd": "echo hello",
            "log_path": os.path.join(self.tmpdir, "test.log"),
            "containerized": False,
            "max_retries": 0,
        }
        defaults.update(overrides)
        return defaults

    # --- Fix 1: Interrupted jobs with retries are requeued ---

    def test_fix1_stale_job_with_retries_is_requeued(self):
        """Stale running jobs with retries remaining should be requeued, not left as interrupted."""
        job = self._make_job(max_retries=3)
        self.db.submit(job)
        # Claim it to set it to running
        claimed = self.db.claim_next()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["state"], "running")

        # Fake a stale heartbeat by directly updating the DB
        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ?",
                (stale_time, job["job_id"]),
            )

        # Run stale detection
        count = self.db.mark_stale_running_as_interrupted(stale_seconds=120)
        self.assertEqual(count, 1)

        # Job should be requeued, NOT interrupted
        result = self.db.get(job["job_id"])
        self.assertEqual(result["state"], "queued")
        self.assertEqual(result["retry_count"], 1)

    def test_fix1_stale_job_without_retries_stays_interrupted(self):
        """Stale running jobs with no retries remaining should stay interrupted."""
        job = self._make_job(max_retries=0)
        self.db.submit(job)
        claimed = self.db.claim_next()
        self.assertIsNotNone(claimed)

        stale_time = (datetime.now(timezone.utc) - timedelta(seconds=300)).isoformat()
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ?",
                (stale_time, job["job_id"]),
            )

        count = self.db.mark_stale_running_as_interrupted(stale_seconds=120)
        self.assertEqual(count, 1)

        result = self.db.get(job["job_id"])
        self.assertEqual(result["state"], "interrupted")

    # --- Fix 3: request_cancel checks existence first, FK enforcement ---

    def test_fix3_cancel_nonexistent_job_raises(self):
        """Canceling a nonexistent job should raise ValueError immediately."""
        with self.assertRaises(ValueError) as ctx:
            self.db.request_cancel("nonexistent_job_id")
        self.assertIn("Job not found", str(ctx.exception))

    def test_fix3_cancel_existing_job_works(self):
        """Canceling an existing job should succeed."""
        job = self._make_job()
        self.db.submit(job)
        result = self.db.request_cancel(job["job_id"])
        self.assertTrue(result["cancel_requested"])

    def test_fix3_fk_enforcement_enabled(self):
        """Foreign key enforcement should be active on connections."""
        conn = self.db._connect()
        try:
            fk_status = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(fk_status, 1, "Foreign keys should be enabled")
        finally:
            conn.close()

    # --- Fix 6: now_iso() consistency in requeue_after_failure ---

    def test_fix6_requeue_failure_consistent_timestamps(self):
        """When marking failed (no retries left), ended_at and heartbeat_at should be the same."""
        job = self._make_job(max_retries=0)
        self.db.submit(job)
        self.db.claim_next()

        result = self.db.requeue_after_failure(job["job_id"], error="test error", exit_code=1)
        self.assertEqual(result["state"], "failed")
        # ended_at and heartbeat_at should be identical (same now variable)
        self.assertEqual(result["ended_at"], result["heartbeat_at"])

    # --- Fix 14: Retry backoff ---

    def test_fix14_requeued_job_has_retry_after(self):
        """Requeued jobs should have a retry_after timestamp set in the future."""
        job = self._make_job(max_retries=3)
        self.db.submit(job)
        self.db.claim_next()

        result = self.db.requeue_after_failure(job["job_id"], error="test error", exit_code=1)
        self.assertEqual(result["state"], "queued")
        self.assertIsNotNone(result.get("retry_after"))
        # retry_after should be in the future
        retry_after = datetime.fromisoformat(result["retry_after"])
        now = datetime.now(timezone.utc)
        self.assertGreater(retry_after, now)

    def test_fix14_claim_next_respects_retry_after(self):
        """claim_next should not return jobs whose retry_after is in the future."""
        job = self._make_job(max_retries=3)
        self.db.submit(job)
        self.db.claim_next()

        # Requeue with backoff
        self.db.requeue_after_failure(job["job_id"], error="test error", exit_code=1)

        # Should NOT be claimable yet (retry_after is in the future)
        claimed = self.db.claim_next()
        self.assertIsNone(claimed)

    def test_fix14_claim_next_returns_job_after_retry_after(self):
        """claim_next should return jobs whose retry_after has passed."""
        job = self._make_job(max_retries=3)
        self.db.submit(job)
        self.db.claim_next()

        # Requeue, then manually set retry_after to the past
        self.db.requeue_after_failure(job["job_id"], error="test error", exit_code=1)
        past_time = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        with self.db._connect() as conn:
            conn.execute(
                "UPDATE jobs SET retry_after = ? WHERE job_id = ?",
                (past_time, job["job_id"]),
            )

        # Now it should be claimable
        claimed = self.db.claim_next()
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed["job_id"], job["job_id"])

    def test_fix14_exponential_backoff_increases(self):
        """Successive retries should produce increasing retry_after delays."""
        job = self._make_job(max_retries=5)
        self.db.submit(job)

        delays = []
        for i in range(3):
            self.db.claim_next()
            # Set retry_after to past so we can reclaim
            past = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()
            with self.db._connect() as conn:
                conn.execute("UPDATE jobs SET retry_after = ? WHERE job_id = ?", (past, job["job_id"]))
            result = self.db.requeue_after_failure(job["job_id"], error=f"fail {i}", exit_code=1)
            retry_after = datetime.fromisoformat(result["retry_after"])
            delay = (retry_after - datetime.now(timezone.utc)).total_seconds()
            delays.append(delay)
            # Reset retry_after to past for next iteration
            with self.db._connect() as conn:
                conn.execute("UPDATE jobs SET retry_after = ? WHERE job_id = ?", (past, job["job_id"]))

        # Each delay should be larger than the previous (exponential)
        for i in range(1, len(delays)):
            self.assertGreater(delays[i], delays[i - 1],
                f"Delay {i} ({delays[i]:.0f}s) should be > delay {i-1} ({delays[i-1]:.0f}s)")


class TestBridgeCtlFixes(unittest.TestCase):
    """Tests for bridge_ctl.py fixes."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.bridge_dir = Path(self.tmpdir) / "bridge"
        self.bridge_dir.mkdir()
        self.profiles_dir = self.bridge_dir / "profiles"
        self.profiles_dir.mkdir()
        self.state_dir = self.bridge_dir / "state"
        self.state_dir.mkdir()
        self.db_path = str(self.state_dir / "bridge.db")
        self.db = BridgeDB(self.db_path)

    def test_fix7_api_cancel_without_job_id_returns_error(self):
        """The API cancel action with missing job_id should return an error, not crash."""
        # Simulate the API by running bridge_ctl.py in api mode
        req = json.dumps({"action": "cancel", "payload": {}})
        result = subprocess.run(
            [sys.executable, str(WSL_DIR / "bridge_ctl.py"),
             "--bridge-dir", str(self.bridge_dir),
             "--db", self.db_path,
             "api"],
            input=req,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = json.loads(result.stdout.strip())
        self.assertFalse(output["ok"])
        self.assertIn("job_id", output["error"])

    def test_fix7_api_cancel_with_valid_job_id_works(self):
        """The API cancel action with a valid job_id should succeed."""
        job = {
            "job_id": "test_cancel_valid",
            "project": "test",
            "cmd": "echo hi",
            "log_path": str(self.bridge_dir / "logs" / "test.log"),
            "containerized": False,
            "max_retries": 0,
        }
        self.db.submit(job)

        req = json.dumps({"action": "cancel", "payload": {"job_id": "test_cancel_valid"}})
        result = subprocess.run(
            [sys.executable, str(WSL_DIR / "bridge_ctl.py"),
             "--bridge-dir", str(self.bridge_dir),
             "--db", self.db_path,
             "api"],
            input=req,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = json.loads(result.stdout.strip())
        self.assertTrue(output["ok"])
        self.assertTrue(output["job"]["cancel_requested"])


class TestMainPyFixes(unittest.TestCase):
    """Tests for main.py fixes that can be verified without SSH."""

    def test_fix2_lock_ordering_comment_exists(self):
        """The GatewayCoordinator should have a lock ordering comment."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        self.assertIn("Lock ordering: start_lock must always be acquired before state_lock", content)

    def test_fix4_log_handle_closed(self):
        """ensure_gateway_started should close log_handle after Popen."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        # Find the ensure_gateway_started function and verify log_handle.close() is there
        start = content.index("def ensure_gateway_started")
        # Find the next def at the same indentation level
        end = content.index("\ndef ", start + 1)
        func_body = content[start:end]
        self.assertIn("log_handle.close()", func_body)

    def test_fix5_cli_lock_exists(self):
        """cli_lock context manager should be defined."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        self.assertIn("def cli_lock(settings: Settings):", content)
        self.assertIn("fcntl.flock", content)

    def test_fix5_ensure_gateway_uses_cli_lock(self):
        """ensure_gateway_started should use cli_lock."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        start = content.index("def ensure_gateway_started")
        end = content.index("\ndef ", start + 1)
        func_body = content[start:end]
        self.assertIn("cli_lock(settings)", func_body)

    def test_fix8_logging_in_runtime_process_info(self):
        """runtime_process_info should log exceptions instead of silently swallowing."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        start = content.index("def runtime_process_info")
        end = content.index("\ndef ", start + 1)
        func_body = content[start:end]
        self.assertIn("logging.getLogger", func_body)

    def test_fix9_tunnel_poll_loop(self):
        """Backend tunnel setup should use a poll loop instead of fixed sleep."""
        main_py = (Path(__file__).resolve().parents[0] / ".." / "src" / "wingpu_cli" / "main.py").resolve()
        content = main_py.read_text()
        start = content.index("def ensure_backend_tunnel")
        end = content.index("\ndef ", start + 1)
        func_body = content[start:end]
        # Should NOT have the old fixed sleep pattern
        self.assertNotIn("time.sleep(1)\n    if run(check_cmd", func_body)
        # Should have the new poll loop
        self.assertIn("while time.time() < deadline", func_body)


class TestScriptFixes(unittest.TestCase):
    """Tests for shell/PS script fixes."""

    def test_fix10_systemd_false_handled(self):
        """The wsl.conf editor should handle systemd=false."""
        ps1 = Path(__file__).resolve().parents[2] / "windows" / "install-native-wsl-runtime.ps1"
        content = ps1.read_text()
        self.assertIn('normalized.startswith("systemd=")', content)
        self.assertIn('normalized != "systemd=true"', content)

    def test_fix11_install_sh_quoted_paths(self):
        """install.sh should have quoted paths in the systemd unit file."""
        install_sh = Path(__file__).resolve().parents[2] / "wsl" / "install.sh"
        content = install_sh.read_text()
        self.assertIn('WorkingDirectory="$PREFIX"', content)
        self.assertIn('"$PREFIX/worker.py"', content)
        self.assertIn('"$PREFIX/state/bridge.db"', content)

    def test_fix12_update_script_has_cd(self):
        """update_llama.sh should cd to script directory."""
        update_sh = Path(__file__).resolve().parents[2] / "update_llama.sh"
        content = update_sh.read_text()
        self.assertIn('SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', content)
        self.assertIn('cd "$SCRIPT_DIR"', content)

    def test_fix13_boot_sh_pid_tracking(self):
        """boot.sh should write a PID file after nohup."""
        boot_sh = Path(__file__).resolve().parents[2] / "wsl" / "boot.sh"
        content = boot_sh.read_text()
        self.assertIn('echo $! > "$BRIDGE_DIR/state/worker.pid"', content)


if __name__ == "__main__":
    unittest.main()
