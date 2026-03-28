#!/usr/bin/env python3
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  project TEXT NOT NULL,
  image TEXT,
  cmd TEXT NOT NULL,
  gpu TEXT DEFAULT 'all',
  workdir TEXT,
  mount_path TEXT DEFAULT '/workspace',
  env_file TEXT,
  resources_json TEXT,
  containerized INTEGER NOT NULL DEFAULT 1,
  state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  started_at TEXT,
  ended_at TEXT,
  heartbeat_at TEXT,
  exit_code INTEGER,
  log_path TEXT,
  error TEXT,
  retry_count INTEGER NOT NULL DEFAULT 0,
  max_retries INTEGER NOT NULL DEFAULT 0,
  cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_created ON jobs(state, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_heartbeat ON jobs(state, heartbeat_at);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  ts TEXT NOT NULL,
  event TEXT NOT NULL,
  payload TEXT,
  FOREIGN KEY(job_id) REFERENCES jobs(job_id)
);
"""


class BridgeDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @staticmethod
    def now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def tx(self):
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _event(self, conn: sqlite3.Connection, job_id: str, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        conn.execute(
            "INSERT INTO events(job_id, ts, event, payload) VALUES (?, ?, ?, ?)",
            (job_id, self.now_iso(), event, json.dumps(payload or {}, ensure_ascii=True)),
        )

    def submit(self, job: Dict[str, Any]) -> Dict[str, Any]:
        with self.tx() as conn:
            conn.execute(
                """
                INSERT INTO jobs(
                    job_id, project, image, cmd, gpu, workdir, mount_path,
                    env_file, resources_json, containerized, state, created_at,
                    log_path, max_retries
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
                """,
                (
                    job["job_id"],
                    job["project"],
                    job.get("image"),
                    job["cmd"],
                    job.get("gpu", "all"),
                    job.get("workdir"),
                    job.get("mount_path", "/workspace"),
                    job.get("env_file"),
                    json.dumps(job.get("resources", {}), ensure_ascii=True),
                    1 if job.get("containerized", True) else 0,
                    self.now_iso(),
                    job["log_path"],
                    int(job.get("max_retries", 0)),
                ),
            )
            self._event(conn, job["job_id"], "submitted", {"project": job["project"]})
        return self.get(job["job_id"]) or {}

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> Dict[str, Any]:
        limit = max(1, min(limit, 500))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            states = conn.execute(
                "SELECT state, COUNT(*) AS cnt FROM jobs GROUP BY state"
            ).fetchall()
        return {
            "jobs": [self._row_to_dict(r) for r in rows],
            "summary": {r["state"]: r["cnt"] for r in states},
        }

    def claim_next(self) -> Optional[Dict[str, Any]]:
        with self.tx() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_at ASC LIMIT 1"
            ).fetchone()
            if not row:
                return None
            now = self.now_iso()
            conn.execute(
                "UPDATE jobs SET state = 'running', started_at = ?, heartbeat_at = ? WHERE job_id = ?",
                (now, now, row["job_id"]),
            )
            self._event(conn, row["job_id"], "started")
            claimed = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)).fetchone()
        return self._row_to_dict(claimed) if claimed else None

    def heartbeat(self, job_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE jobs SET heartbeat_at = ? WHERE job_id = ? AND state = 'running'",
                (self.now_iso(), job_id),
            )

    def is_cancel_requested(self, job_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def request_cancel(self, job_id: str) -> Dict[str, Any]:
        with self.tx() as conn:
            conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?",
                (job_id,),
            )
            self._event(conn, job_id, "cancel_requested")
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            raise ValueError(f"Job not found: {job_id}")
        return self._row_to_dict(row)

    def finish(self, job_id: str, state: str, exit_code: Optional[int], error: str = "") -> Dict[str, Any]:
        if state not in {"succeeded", "failed", "canceled", "interrupted"}:
            raise ValueError(f"Invalid terminal state: {state}")
        with self.tx() as conn:
            now = self.now_iso()
            conn.execute(
                """
                UPDATE jobs
                SET state = ?, ended_at = ?, exit_code = ?, error = ?, heartbeat_at = ?, cancel_requested = 0
                WHERE job_id = ?
                """,
                (state, now, exit_code, error, now, job_id),
            )
            self._event(conn, job_id, "finished", {"state": state, "exit_code": exit_code, "error": error})
            row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else {}

    def requeue_after_failure(self, job_id: str, error: str, exit_code: Optional[int] = None) -> Dict[str, Any]:
        with self.tx() as conn:
            row = conn.execute("SELECT retry_count, max_retries FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if not row:
                raise ValueError(f"Job not found: {job_id}")
            retry_count = int(row["retry_count"])
            max_retries = int(row["max_retries"])
            if retry_count >= max_retries:
                # No retries left; mark failed.
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = 'failed', ended_at = ?, error = ?, heartbeat_at = ?, exit_code = ?
                    WHERE job_id = ?
                    """,
                    (self.now_iso(), error, self.now_iso(), exit_code, job_id),
                )
                self._event(conn, job_id, "failed", {"retry_count": retry_count, "max_retries": max_retries, "error": error})
            else:
                next_retry = retry_count + 1
                conn.execute(
                    """
                    UPDATE jobs
                    SET state = 'queued', retry_count = ?, started_at = NULL, ended_at = NULL,
                        heartbeat_at = NULL, exit_code = NULL, error = ?, cancel_requested = 0
                    WHERE job_id = ?
                    """,
                    (next_retry, error, job_id),
                )
                self._event(conn, job_id, "requeued", {"retry_count": next_retry, "max_retries": max_retries, "error": error})
            result = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(result) if result else {}

    def mark_stale_running_as_interrupted(self, stale_seconds: int = 120) -> int:
        with self.tx() as conn:
            rows = conn.execute(
                "SELECT job_id, heartbeat_at FROM jobs WHERE state = 'running'"
            ).fetchall()
            now = datetime.now(timezone.utc)
            interrupted = 0
            for row in rows:
                hb_raw = row["heartbeat_at"]
                if not hb_raw:
                    continue
                try:
                    hb = datetime.fromisoformat(hb_raw)
                except ValueError:
                    continue
                age = (now - hb).total_seconds()
                if age > stale_seconds:
                    conn.execute(
                        "UPDATE jobs SET state = 'interrupted', ended_at = ?, error = ? WHERE job_id = ?",
                        (self.now_iso(), f"stale heartbeat ({int(age)}s)", row["job_id"]),
                    )
                    self._event(conn, row["job_id"], "interrupted", {"age_seconds": int(age)})
                    interrupted += 1
        return interrupted

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        try:
            data["resources"] = json.loads(data.get("resources_json") or "{}")
        except json.JSONDecodeError:
            data["resources"] = {}
        data.pop("resources_json", None)
        data["containerized"] = bool(data.get("containerized", 1))
        data["cancel_requested"] = bool(data.get("cancel_requested", 0))
        return data
