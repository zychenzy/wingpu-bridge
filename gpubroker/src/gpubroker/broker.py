#!/usr/bin/env python3
"""
gpu_broker.py — a daemonless GPU lease arbiter for the WSL GPU host.

One NVIDIA GPU is shared by several bridges (wingpu/llama.cpp, locatectl,
ocrctl, minerctl, live transcription). They each load a heavy model that does
not fit alongside the others in VRAM. This program is the single authority on
who currently holds the GPU.

Design notes:
  * Daemonless: state lives in SQLite at $GPU_BROKER_DIR/broker.db, mutual
    exclusion is an flock on broker.lock. Every command is a short-lived process.
    The one long-lived piece is the optional `watchdog`.
  * No third-party dependencies: it runs under whatever python3 WSL ships.
  * wingpu is the only tool that should install or replace this file. Clients
    compare `version` before they talk to it.
  * "Busy" prefers a recent `touch` and, when nvidia-smi can attribute compute
    apps, that holder's own processes. Whole-GPU utilization is only a fallback
    when the holder is the sole compute app (or attribution is impossible).
  * Long waits (process grace, VRAM drain) happen outside broker.lock so
    touch/status/acquire are not stuck behind a 60s drain.

All machine-facing commands print one JSON line to stdout. `status` and
`events` print a human summary unless `--json` is given.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set
from urllib import request as urllib_request

VERSION = "1.0.0"
PROTOCOL = 2

# ── Defaults ────────────────────────────────────────────────────────────────
DEFAULT_DIR = os.environ.get("GPU_BROKER_DIR", "~/.gpu-bridge")
VRAM_FLOOR_MB = 2000        # VRAM is considered "freed" at or below this.
FREE_TIMEOUT_S = 60         # Max wait for VRAM to drain after an eviction.
BUSY_UTIL_PCT = 15          # GPU utilization at or above this counts as busy.
BUSY_GRACE_S = 10           # A touch within this many seconds counts as busy.
STALE_GRACE_S = 45          # A dead holder is reclaimable only this long after acquire/touch.
WATCHDOG_IDLE_S = 900       # Watchdog evicts a holder idle for this long.
WATCHDOG_POLL_S = 15
KILL_GRACE_S = 12           # SIGTERM grace before SIGKILL.
MAX_EVENTS = 500            # events is a ring; older rows are deleted on write.
# A drain left behind by a crashed evictor is abandoned after kill grace + VRAM wait.
STUCK_DRAIN_EXTRA_S = 30

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS holder (
  id            INTEGER PRIMARY KEY CHECK (id = 1),
  runtime_id    TEXT,
  pid           INTEGER,
  pidfile       TEXT,
  shutdown_url  TEXT,
  vram_mb       INTEGER,
  label         TEXT,
  lease_token   TEXT,
  acquired_at   TEXT,
  last_active_at TEXT,
  state         TEXT,
  state_at      TEXT
);
INSERT OR IGNORE INTO holder(id, runtime_id) VALUES (1, NULL);
CREATE TABLE IF NOT EXISTS events (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  event      TEXT NOT NULL,
  runtime_id TEXT,
  detail     TEXT
);
"""

_LOCK_DEPTH = 0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def broker_dir(arg_dir: Optional[str]) -> Path:
    return Path(arg_dir or DEFAULT_DIR).expanduser()


def lock_held() -> bool:
    """True while this process is inside ``file_lock`` (tests use this)."""
    return _LOCK_DEPTH > 0


# ── nvidia-smi probes (patched in tests) ─────────────────────────────────────
def _nvidia_smi_raw(args: List[str]) -> Optional[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", *args],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout or ""


def _nvidia_smi(field: str) -> Optional[List[int]]:
    """Return a per-GPU integer list for ``--query-gpu=<field>``, or None on error."""
    text = _nvidia_smi_raw([f"--query-gpu={field}", "--format=csv,noheader,nounits"])
    if text is None:
        return None
    values: List[int] = []
    for line in text.splitlines():
        token = line.strip().rstrip("%").strip()
        if not token or token.lower() in {"[n/a]", "[not supported]"}:
            continue
        try:
            values.append(int(float(token)))
        except ValueError:
            continue
    return values or None


def gpu_mem_used_mb() -> Optional[int]:
    vals = _nvidia_smi("memory.used")
    return max(vals) if vals else None


def gpu_mem_total_mb() -> Optional[int]:
    vals = _nvidia_smi("memory.total")
    return max(vals) if vals else None


def gpu_util_pct() -> Optional[int]:
    vals = _nvidia_smi("utilization.gpu")
    return max(vals) if vals else None


def gpu_name() -> Optional[str]:
    text = _nvidia_smi_raw(["--query-gpu=name", "--format=csv,noheader"])
    if not text:
        return None
    name = text.splitlines()[0].strip()
    return name or None


def gpu_compute_apps() -> Optional[List[Dict[str, Any]]]:
    """Per-process VRAM from ``nvidia-smi``. None when the query itself fails.

    An empty list means the query succeeded and nothing has a compute context.
    """
    text = _nvidia_smi_raw(
        ["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"]
    )
    if text is None:
        return None
    apps: List[Dict[str, Any]] = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        try:
            apps.append({
                "pid": int(parts[0]),
                "name": parts[1],
                "vram_mb": int(float(parts[2])),
            })
        except ValueError:
            continue
    return apps


def _pmon_sm() -> Optional[Dict[int, int]]:
    """One ``nvidia-smi pmon`` sample of SM% by pid. None when pmon cannot be read."""
    text = _nvidia_smi_raw(["pmon", "-c", "1", "-s", "u"])
    if text is None:
        return None
    sample: Dict[int, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        # gpu pid type sm mem enc dec command
        if len(parts) < 4:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        sm = parts[3]
        if sm in {"-", "N/A"}:
            continue
        try:
            sample[pid] = int(sm)
        except ValueError:
            continue
    return sample


def gpu_process_sm(pids: Set[int], samples: int = 3, interval: float = 0.4) -> Optional[int]:
    """Highest SM% seen for ``pids``, or None when per-process util cannot be read.

    A readable sample that never mentions these pids returns 0.
    """
    if not pids:
        return 0
    peak: Optional[int] = None
    readable = False
    for index in range(max(1, samples)):
        sample = _pmon_sm()
        if sample is None:
            return None
        readable = True
        for pid in pids:
            if pid in sample:
                value = sample[pid]
                peak = value if peak is None else max(peak, value)
        if peak is not None and peak >= 100:
            return peak
        if index + 1 < samples:
            time.sleep(interval)
    if not readable:
        return None
    return 0 if peak is None else peak


def gpu_busy_sampled(busy_util: int, samples: int = 3, interval: float = 0.4) -> bool:
    """True if any whole-GPU sample is at/above ``busy_util``.

    Used only when per-process attribution is unavailable, or the holder is the
    only compute app and ``pmon`` cannot be read. If utilization can't be read
    at all, assume not busy.
    """
    for index in range(max(1, samples)):
        util = gpu_util_pct()
        if util is not None and util >= busy_util:
            return True
        if index + 1 < samples:
            time.sleep(interval)
    return False


# ── process helpers (patched in tests) ───────────────────────────────────────
def pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_pidfile(path: str) -> Optional[int]:
    try:
        return int(Path(path).expanduser().read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def descendant_pids(pid: int) -> Set[int]:
    """``pid`` plus children listed in ``/proc`` (Linux / WSL)."""
    found: Set[int] = set()
    stack = [pid]
    while stack:
        current = stack.pop()
        if current in found or current <= 0:
            continue
        found.add(current)
        children = Path(f"/proc/{current}/task/{current}/children")
        try:
            text = children.read_text(encoding="utf-8")
        except OSError:
            continue
        for token in text.split():
            try:
                stack.append(int(token))
            except ValueError:
                continue
    return found


def holder_pid(holder: Dict[str, Any]) -> Optional[int]:
    pidfile = holder.get("pidfile")
    if pidfile:
        pid = read_pidfile(pidfile)
        if pid:
            return pid
    pid = holder.get("pid")
    if pid:
        try:
            return int(pid)
        except (TypeError, ValueError):
            return None
    return None


def holder_pids(holder: Dict[str, Any]) -> Set[int]:
    pid = holder_pid(holder)
    if not pid:
        return set()
    return descendant_pids(pid)


def kill_pid(pid: int, *, grace: float = KILL_GRACE_S) -> None:
    """SIGTERM, wait up to ``grace`` for exit, then SIGKILL.

    Callers must invoke this outside ``file_lock``. The wait is the multi-second
    section that used to freeze every other broker command.
    """
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.time() + grace
    while time.time() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.3)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def http_post(url: str, timeout: float = 5.0) -> bool:
    try:
        req = urllib_request.Request(url, data=b"", method="POST")
        with urllib_request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


# ── state ────────────────────────────────────────────────────────────────────
@contextmanager
def file_lock(directory: Path) -> Iterator[None]:
    """Serialize broker critical sections across processes via flock."""
    global _LOCK_DEPTH
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "broker.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        _LOCK_DEPTH += 1
        try:
            yield
        finally:
            _LOCK_DEPTH -= 1
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def _ensure_columns(conn: sqlite3.Connection) -> None:
    cols = {row[1] for row in conn.execute("PRAGMA table_info(holder)")}
    if "state" not in cols:
        conn.execute("ALTER TABLE holder ADD COLUMN state TEXT")
    if "state_at" not in cols:
        conn.execute("ALTER TABLE holder ADD COLUMN state_at TEXT")


def connect(directory: Path) -> sqlite3.Connection:
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(directory / "broker.db"), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _ensure_columns(conn)
    return conn


def record_event(conn: sqlite3.Connection, event: str, runtime_id: Optional[str], detail: Any = None) -> None:
    conn.execute(
        "INSERT INTO events(ts, event, runtime_id, detail) VALUES (?, ?, ?, ?)",
        (now_iso(), event, runtime_id, json.dumps(detail) if detail is not None else None),
    )
    cutoff = conn.execute(
        "SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET ?",
        (MAX_EVENTS - 1,),
    ).fetchone()
    if cutoff is not None:
        conn.execute("DELETE FROM events WHERE id < ?", (cutoff[0],))


def read_holder(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM holder WHERE id = 1").fetchone()
    if not row or not row["runtime_id"]:
        return None
    return dict(row)


def set_holder(conn: sqlite3.Connection, fields: Dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE holder SET runtime_id=?, pid=?, pidfile=?, shutdown_url=?, vram_mb=?,
                          label=?, lease_token=?, acquired_at=?, last_active_at=?,
                          state=?, state_at=?
        WHERE id = 1
        """,
        (
            fields.get("runtime_id"), fields.get("pid"), fields.get("pidfile"),
            fields.get("shutdown_url"), fields.get("vram_mb"), fields.get("label"),
            fields.get("lease_token"), fields.get("acquired_at"), fields.get("last_active_at"),
            fields.get("state"), fields.get("state_at"),
        ),
    )


def clear_holder(conn: sqlite3.Connection) -> None:
    set_holder(conn, {})


def touch_holder(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE holder SET last_active_at=? WHERE id = 1", (now_iso(),))


def list_events(conn: sqlite3.Connection, *, limit: int, since: Optional[str]) -> List[Dict[str, Any]]:
    query = "SELECT id, ts, event, runtime_id, detail FROM events"
    params: List[Any] = []
    if since:
        query += " WHERE ts >= ?"
        params.append(since)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, min(int(limit), 500)))
    rows = []
    for row in conn.execute(query, params):
        detail = row["detail"]
        parsed: Any = None
        if detail:
            try:
                parsed = json.loads(detail)
            except ValueError:
                parsed = detail
        rows.append({
            "id": row["id"],
            "ts": row["ts"],
            "event": row["event"],
            "runtime_id": row["runtime_id"],
            "detail": parsed,
        })
    rows.reverse()
    return rows


# ── liveness / busy judgement ─────────────────────────────────────────────────
def holder_alive(holder: Dict[str, Any]) -> bool:
    """Is the holding process still running? Unknown liveness counts as alive."""
    pidfile = holder.get("pidfile")
    if pidfile:
        pid = read_pidfile(pidfile)
        return bool(pid and pid_alive(pid))
    pid = holder.get("pid")
    if pid:
        return pid_alive(int(pid))
    return True


def _recent_touch(holder: Dict[str, Any], busy_grace: float) -> bool:
    last = holder.get("last_active_at")
    if not last:
        return False
    try:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
    except ValueError:
        return False
    return age < busy_grace


def holder_busy(holder: Dict[str, Any], busy_util: int, busy_grace: float) -> bool:
    """Busy if the holder touched recently, or *its* GPU work is above ``busy_util``.

    Whole-GPU utilization counts only when we cannot see compute apps, or the
    holder is the only compute app and per-process SM% cannot be read. A game,
    a desktop compositor, or another bridge's leftover context does not keep
    this holder busy.
    """
    if _recent_touch(holder, busy_grace):
        return True
    apps = gpu_compute_apps()
    if apps is None:
        return gpu_busy_sampled(busy_util)
    pids = holder_pids(holder)
    if not pids:
        if apps:
            return False
        return gpu_busy_sampled(busy_util)
    own = [app for app in apps if app["pid"] in pids]
    others = [app for app in apps if app["pid"] not in pids]
    if not own:
        return False
    sm = gpu_process_sm(pids)
    if sm is not None:
        return sm >= busy_util
    if others:
        return False
    return gpu_busy_sampled(busy_util)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def holder_age_seconds(holder: Dict[str, Any]) -> float:
    """Seconds since the holder was last acquired or touched; inf if unknown."""
    stamps = [dt for dt in (_parse_iso(holder.get("last_active_at")),
                            _parse_iso(holder.get("acquired_at"))) if dt]
    if not stamps:
        return float("inf")
    return (datetime.now(timezone.utc) - max(stamps)).total_seconds()


def idle_age_seconds(holder: Dict[str, Any]) -> Optional[float]:
    """Seconds since the idle clock (last touch, else acquire). None if unknown."""
    last = holder.get("last_active_at") or holder.get("acquired_at")
    stamp = _parse_iso(last)
    if stamp is None:
        return None
    return (datetime.now(timezone.utc) - stamp).total_seconds()


def drain_is_stuck(holder: Dict[str, Any], *, free_timeout: float) -> bool:
    if holder.get("state") != "draining":
        return False
    started = _parse_iso(holder.get("state_at"))
    if started is None:
        return True
    limit = KILL_GRACE_S + float(free_timeout) + STUCK_DRAIN_EXTRA_S
    return (datetime.now(timezone.utc) - started).total_seconds() >= limit


def wait_vram_below(floor_mb: int, timeout: float) -> tuple:
    """Poll until GPU VRAM is at/below floor_mb (or unreadable). Returns (used_mb, ok).

    Do not call this while holding ``broker.lock``.
    """
    used = gpu_mem_used_mb()
    if used is None or used <= floor_mb:
        return used, True
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(1.0)
        used = gpu_mem_used_mb()
        if used is None or used <= floor_mb:
            return used, True
    return used, False


def mark_draining(conn: sqlite3.Connection, holder: Dict[str, Any]) -> None:
    conn.execute(
        "UPDATE holder SET state=?, state_at=? WHERE id = 1 AND lease_token IS ?",
        ("draining", now_iso(), holder.get("lease_token")),
    )


def clear_draining(conn: sqlite3.Connection, token: Optional[str]) -> None:
    conn.execute(
        "UPDATE holder SET state=NULL, state_at=NULL WHERE id = 1 AND lease_token IS ?",
        (token,),
    )


def start_evict(conn: sqlite3.Connection, holder: Dict[str, Any]) -> Dict[str, Any]:
    """Under the lock: remember who we are evicting and mark the row draining.

    The HTTP shutdown, the signal, and the VRAM wait happen in ``finish_evict``,
    outside the lock.
    """
    mark_draining(conn, holder)
    record_event(conn, "evicting", holder.get("runtime_id"), {"token": holder.get("lease_token")})
    return {
        "runtime_id": holder.get("runtime_id"),
        "token": holder.get("lease_token"),
        "pid": holder_pid(holder),
        "pidfile": holder.get("pidfile"),
        "shutdown_url": holder.get("shutdown_url"),
        "vram_mb": holder.get("vram_mb") or 0,
        "used_before": gpu_mem_used_mb(),
    }


def finish_evict(plan: Dict[str, Any], *, vram_floor: int, free_timeout: float,
                 log=lambda _m: None) -> Dict[str, Any]:
    """Outside the lock: graceful POST, SIGTERM/SIGKILL, then wait for VRAM."""
    runtime_id = plan.get("runtime_id")
    used_before = plan.get("used_before")
    log(f"evicting holder '{runtime_id}' (vram_used={used_before}MB)")

    method = "none"
    shutdown_url = plan.get("shutdown_url")
    if shutdown_url and http_post(shutdown_url):
        method = "shutdown_url"
        time.sleep(1.0)

    pid = plan.get("pid")
    if pid and pid_alive(int(pid)):
        kill_pid(int(pid))
        method = "shutdown_url+signal" if method == "shutdown_url" else "signal"
    pidfile = plan.get("pidfile")
    if pidfile:
        try:
            Path(pidfile).expanduser().unlink(missing_ok=True)
        except OSError:
            pass

    target = vram_floor
    holder_vram = plan.get("vram_mb") or 0
    if holder_vram and used_before:
        target = max(vram_floor, int(used_before) - int(holder_vram * 0.8))
    used_after, drained = wait_vram_below(target, free_timeout)
    if not drained:
        log(f"VRAM did not drain below {target}MB within {free_timeout:g}s (used={used_after}MB)")
    return {
        "evicted": runtime_id,
        "method": method,
        "vram_before_mb": used_before,
        "vram_after_mb": used_after,
        "vram_drained": drained,
    }


def complete_evict(conn: sqlite3.Connection, plan: Dict[str, Any], result: Dict[str, Any]) -> bool:
    """Under the lock: drop the holder if it is still the one we started evicting."""
    holder = read_holder(conn)
    if holder is None or holder.get("lease_token") != plan.get("token"):
        return False
    record_event(conn, "evicted", plan.get("runtime_id"), {
        "method": result.get("method"),
        "vram_before": result.get("vram_before_mb"),
        "vram_after": result.get("vram_after_mb"),
        "drained": result.get("vram_drained"),
    })
    clear_holder(conn)
    return True


# ── commands ───────────────────────────────────────────────────────────────--
def emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _new_fields(args: argparse.Namespace, token: str, acquired_at: str,
                prev: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    previous = prev or {}
    return {
        "runtime_id": args.runtime,
        "pid": args.pid,
        "pidfile": args.pidfile,
        "shutdown_url": args.shutdown_url,
        "vram_mb": args.vram_mb,
        "label": args.label,
        "lease_token": previous.get("lease_token") or token,
        "acquired_at": previous.get("acquired_at") or acquired_at,
        "last_active_at": previous.get("last_active_at"),
        "state": "active",
        "state_at": None,
    }


def _grant(conn: sqlite3.Connection, fields: Dict[str, Any], *, holder_was: Optional[str],
           detail: Dict[str, Any], evicted: Any = None, extra: Optional[Dict[str, Any]] = None) -> int:
    set_holder(conn, fields)
    record_event(conn, "acquired", fields["runtime_id"], {"holder_was": holder_was, **detail})
    payload: Dict[str, Any] = {
        "ok": True,
        "granted": True,
        "runtime": fields["runtime_id"],
        "token": fields["lease_token"],
        "evicted": evicted,
    }
    if extra:
        payload.update(extra)
    emit(payload)
    return 0


def _deny(runtime: str, reason: str, holder: Optional[str]) -> int:
    emit({"ok": False, "granted": False, "runtime": runtime, "reason": reason, "holder": holder})
    return 3


def cmd_acquire(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    me = args.runtime
    token = uuid.uuid4().hex
    acquired_at = now_iso()
    deadline = time.time() + (0 if args.force else max(0.0, args.wait))
    waited_for: Optional[str] = None
    wait_reason = "busy"

    while True:
        phase: Optional[tuple] = None
        with file_lock(directory):
            conn = connect(directory)
            try:
                holder = read_holder(conn)
                if holder is None:
                    return _grant(conn, _new_fields(args, token, acquired_at),
                                  holder_was=None, detail={})
                stuck = drain_is_stuck(holder, free_timeout=args.free_timeout)
                if holder.get("state") == "draining" and not stuck:
                    waited_for, wait_reason = holder["runtime_id"], "draining"
                elif holder["runtime_id"] == me:
                    updated = _new_fields(args, token, acquired_at, holder)
                    set_holder(conn, updated)
                    emit({
                        "ok": True, "granted": True, "runtime": me,
                        "token": updated["lease_token"], "evicted": None, "reasserted": True,
                    })
                    return 0
                else:
                    alive = holder_alive(holder)
                    age = holder_age_seconds(holder)
                    if not alive and age >= args.stale_grace:
                        used = gpu_mem_used_mb()
                        vram_free = used is None or used <= args.vram_floor
                        if vram_free:
                            return _grant(
                                conn, _new_fields(args, token, acquired_at),
                                holder_was=holder["runtime_id"],
                                detail={"stale": True},
                                evicted=holder["runtime_id"],
                                extra={"holder_was_stale": True, "vram_after_mb": used, "vram_drained": True},
                            )
                        if args.no_evict:
                            waited_for, wait_reason = holder["runtime_id"], "vram_busy"
                        else:
                            mark_draining(conn, holder)
                            phase = ("vram", dict(holder))
                    elif not alive:
                        if args.force and not args.no_evict:
                            phase = ("evict", start_evict(conn, holder), True)
                        else:
                            waited_for, wait_reason = holder["runtime_id"], "recovering"
                    else:
                        busy = (not args.force) and holder_busy(holder, args.busy_util, BUSY_GRACE_S)
                        if args.no_evict:
                            waited_for, wait_reason = holder["runtime_id"], "busy" if busy else "held"
                        elif not busy:
                            phase = ("evict", start_evict(conn, holder), bool(args.force))
                        else:
                            waited_for, wait_reason = holder["runtime_id"], "busy"
            finally:
                conn.close()

        if phase and phase[0] == "vram":
            snapshot = phase[1]
            _used, drained = wait_vram_below(args.vram_floor, args.free_timeout)
            with file_lock(directory):
                conn = connect(directory)
                try:
                    current = read_holder(conn)
                    same = current is not None and current.get("lease_token") == snapshot.get("lease_token")
                    if same and drained and not holder_alive(current):
                        return _grant(
                            conn, _new_fields(args, token, acquired_at),
                            holder_was=snapshot.get("runtime_id"),
                            detail={"stale": True},
                            evicted=snapshot.get("runtime_id"),
                            extra={"holder_was_stale": True, "vram_after_mb": _used, "vram_drained": True},
                        )
                    if same:
                        clear_draining(conn, snapshot.get("lease_token"))
                finally:
                    conn.close()
            waited_for, wait_reason = snapshot.get("runtime_id"), "vram_busy"
        elif phase and phase[0] == "evict":
            plan = phase[1]
            forced = phase[2]
            result = finish_evict(plan, vram_floor=args.vram_floor, free_timeout=args.free_timeout)
            with file_lock(directory):
                conn = connect(directory)
                try:
                    if complete_evict(conn, plan, result):
                        return _grant(
                            conn, _new_fields(args, token, acquired_at),
                            holder_was=plan.get("runtime_id"),
                            detail={"forced": forced},
                            evicted=result.get("evicted"),
                            extra={"evict": result, "forced": forced},
                        )
                finally:
                    conn.close()
            continue

        if time.time() >= deadline:
            return _deny(me, wait_reason, waited_for)
        time.sleep(min(2.0, args.poll))


def cmd_release(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    with file_lock(directory):
        conn = connect(directory)
        try:
            holder = read_holder(conn)
            if holder is None:
                emit({"ok": True, "released": False, "reason": "no_holder"})
                return 0
            if holder["runtime_id"] != args.runtime:
                emit({"ok": True, "released": False, "reason": "not_holder",
                      "holder": holder["runtime_id"]})
                return 0
            if args.token and holder.get("lease_token") and args.token != holder["lease_token"]:
                emit({"ok": True, "released": False, "reason": "stale_token",
                      "holder": holder["runtime_id"]})
                return 0
            clear_holder(conn)
            record_event(conn, "released", args.runtime, None)
            emit({"ok": True, "released": True, "runtime": args.runtime})
            return 0
        finally:
            conn.close()


def cmd_touch(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    with file_lock(directory):
        conn = connect(directory)
        try:
            holder = read_holder(conn)
            if holder is None or holder["runtime_id"] != args.runtime:
                emit({"ok": True, "touched": False, "holder": holder["runtime_id"] if holder else None})
                return 0
            touch_holder(conn)
            emit({"ok": True, "touched": True, "runtime": args.runtime})
            return 0
        finally:
            conn.close()


def cmd_evict(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    plan: Optional[Dict[str, Any]] = None
    with file_lock(directory):
        conn = connect(directory)
        try:
            holder = read_holder(conn)
            if holder is None:
                emit({"ok": True, "evicted": None, "reason": "no_holder"})
                return 0
            if args.runtime and holder["runtime_id"] != args.runtime:
                emit({"ok": True, "evicted": None, "reason": "not_holder",
                      "holder": holder["runtime_id"]})
                return 0
            if holder.get("state") == "draining" and not drain_is_stuck(holder, free_timeout=args.free_timeout):
                emit({"ok": True, "evicted": None, "reason": "draining", "holder": holder["runtime_id"]})
                return 0
            plan = start_evict(conn, holder)
        finally:
            conn.close()
    assert plan is not None
    result = finish_evict(plan, vram_floor=args.vram_floor, free_timeout=args.free_timeout)
    with file_lock(directory):
        conn = connect(directory)
        try:
            complete_evict(conn, plan, result)
        finally:
            conn.close()
    emit({"ok": True, **result})
    return 0


def holder_view(holder: Optional[Dict[str, Any]], busy_util: int) -> Optional[Dict[str, Any]]:
    if holder is None:
        return None
    age = idle_age_seconds(holder)
    view = {key: holder.get(key) for key in
            ("runtime_id", "pid", "pidfile", "vram_mb", "label", "acquired_at", "last_active_at",
             "shutdown_url", "state")}
    view["alive"] = holder_alive(holder)
    view["busy"] = holder_busy(holder, busy_util, BUSY_GRACE_S)
    view["idle_age_s"] = None if age is None else round(age, 1)
    view["state"] = holder.get("state") or "active"
    return view


def _read_watchdog_meta(directory: Path) -> Dict[str, Any]:
    path = directory / "run" / "watchdog.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def watchdog_state(directory: Path, holder: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    pid = read_pidfile(str(directory / "run" / "watchdog.pid"))
    running = bool(pid and pid_alive(pid))
    meta = _read_watchdog_meta(directory)
    idle_timeout = meta.get("idle_timeout_s")
    age = idle_age_seconds(holder) if holder else None
    next_in: Optional[float] = None
    blocked: Optional[str] = None
    if not running:
        blocked = "stopped"
    elif holder is None:
        blocked = "no_holder"
    elif idle_timeout is None or age is None:
        blocked = "unconfigured"
    else:
        next_in = max(0.0, float(idle_timeout) - age)
    # Busy holders are not evicted when the clock hits zero; say so.
    if running and holder is not None and next_in == 0:
        busy_util = int(meta.get("busy_util") or BUSY_UTIL_PCT)
        if holder_busy(holder, busy_util, BUSY_GRACE_S) and next_in == 0:
            blocked = "busy"
    payload: Dict[str, Any] = {
        "running": running,
        "pid": pid,
        "idle_timeout_s": idle_timeout,
        "poll_s": meta.get("poll_s"),
        "busy_util": meta.get("busy_util"),
        "vram_floor": meta.get("vram_floor"),
        "free_timeout": meta.get("free_timeout"),
        "stale_grace": meta.get("stale_grace"),
        "idle_age_s": None if age is None else round(age, 1),
        "next_evict_in_s": None if next_in is None else int(next_in),
        "blocked_reason": blocked,
    }
    return payload


def cmd_status(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    conn = connect(directory)
    try:
        holder = read_holder(conn)
    finally:
        conn.close()
    gpu = {
        "name": gpu_name(),
        "mem_used_mb": gpu_mem_used_mb(),
        "mem_total_mb": gpu_mem_total_mb(),
        "util_pct": gpu_util_pct(),
    }
    payload = {
        "ok": True,
        "version": VERSION,
        "protocol": PROTOCOL,
        "broker_dir": str(directory),
        "holder": holder_view(holder, args.busy_util),
        "gpu": gpu,
        "processes": gpu_compute_apps(),
        "watchdog": watchdog_state(directory, holder),
    }
    if args.json:
        emit(payload)
        return 0
    h = payload["holder"]
    print(f"broker dir : {directory}")
    print(f"version    : {VERSION} (protocol {PROTOCOL})")
    if h:
        print(f"holder     : {h['runtime_id']}  (pid={h['pid']}, alive={h['alive']}, busy={h['busy']})")
        print(f"  since    : {h['acquired_at']}")
        print(f"  active   : {h['last_active_at']}")
        if h.get("idle_age_s") is not None:
            print(f"  idle age : {h['idle_age_s']}s")
        if h.get("vram_mb"):
            print(f"  vram est : {h['vram_mb']} MB")
    else:
        print("holder     : (none) — GPU is free")
    if gpu["mem_used_mb"] is not None:
        name = f"{gpu['name']}  " if gpu.get("name") else ""
        print(f"gpu vram   : {name}{gpu['mem_used_mb']} / {gpu['mem_total_mb']} MB used, util {gpu['util_pct']}%")
    else:
        print("gpu vram   : (nvidia-smi unavailable)")
    processes = payload["processes"]
    if processes:
        for app in processes:
            print(f"  proc     : pid {app['pid']}  {app['name']}  {app['vram_mb']} MB")
    wd = payload["watchdog"]
    if wd["running"]:
        extra = ""
        if wd.get("next_evict_in_s") is not None:
            extra = f", next idle evict in {wd['next_evict_in_s']}s"
        print(f"watchdog   : running pid {wd['pid']}{extra}")
    else:
        print("watchdog   : stopped")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    conn = connect(directory)
    try:
        rows = list_events(conn, limit=args.limit, since=args.since)
    finally:
        conn.close()
    if args.json:
        emit({"ok": True, "events": rows})
        return 0
    if not rows:
        print("(no events)")
        return 0
    for row in rows:
        runtime = row.get("runtime_id") or ""
        print(f"{row['ts']}  {row['event']:<14}  {runtime}")
    return 0


def cmd_version(_args: argparse.Namespace) -> int:
    emit({"ok": True, "version": VERSION, "protocol": PROTOCOL})
    return 0


def cmd_holder(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    conn = connect(directory)
    try:
        holder = read_holder(conn)
    finally:
        conn.close()
    print(holder["runtime_id"] if holder else "none")
    return 0


def _watchdog_once(directory: Path, *, idle_timeout: float, busy_util: int,
                   vram_floor: int, free_timeout: float, stale_grace: float = STALE_GRACE_S,
                   log=lambda _m: None) -> Dict[str, Any]:
    """One watchdog iteration: clear a dead holder, or evict an idle one.

    The VRAM wait runs outside the lock. Returns an action dict
    ({"action": "none"|"cleared_stale"|"idle_offload", ...}).
    """
    plan: Optional[Dict[str, Any]] = None
    idle_age: Optional[float] = None
    with file_lock(directory):
        conn = connect(directory)
        try:
            holder = read_holder(conn)
            if holder is None:
                return {"action": "none"}
            if holder.get("state") == "draining" and not drain_is_stuck(holder, free_timeout=free_timeout):
                return {"action": "none", "runtime": holder["runtime_id"], "reason": "draining"}
            if not holder_alive(holder) and holder_age_seconds(holder) >= stale_grace:
                record_event(conn, "cleared_stale", holder["runtime_id"], None)
                clear_holder(conn)
                log(f"cleared stale holder '{holder['runtime_id']}'")
                return {"action": "cleared_stale", "runtime": holder["runtime_id"]}
            idle_age = idle_age_seconds(holder)
            if (idle_age is not None and idle_age >= idle_timeout
                    and not holder_busy(holder, busy_util, BUSY_GRACE_S)):
                plan = start_evict(conn, holder)
            else:
                return {"action": "none", "runtime": holder["runtime_id"], "idle_age_s": idle_age}
        finally:
            conn.close()
    assert plan is not None
    result = finish_evict(plan, vram_floor=vram_floor, free_timeout=free_timeout, log=log)
    with file_lock(directory):
        conn = connect(directory)
        try:
            if not complete_evict(conn, plan, result):
                return {"action": "none", "runtime": plan.get("runtime_id"), "reason": "lost_race"}
            record_event(conn, "idle_offload", plan.get("runtime_id"),
                         {"idle_age_s": int(idle_age or 0)})
        finally:
            conn.close()
    log(f"idle-offloaded '{result['evicted']}' after {int(idle_age or 0)}s idle")
    return {"action": "idle_offload", "runtime": result["evicted"], "idle_age_s": int(idle_age or 0)}


def _write_watchdog_meta(directory: Path, args: argparse.Namespace) -> None:
    payload = {
        "idle_timeout_s": args.idle_timeout,
        "poll_s": args.poll,
        "busy_util": args.busy_util,
        "vram_floor": args.vram_floor,
        "free_timeout": args.free_timeout,
        "stale_grace": args.stale_grace,
        "started_at": now_iso(),
        "pid": os.getpid(),
    }
    path = directory / "run" / "watchdog.json"
    path.write_text(json.dumps(payload), encoding="utf-8")


def cmd_watchdog(args: argparse.Namespace) -> int:
    """Long-lived loop: evict whichever runtime has gone idle, freeing VRAM."""
    directory = broker_dir(args.dir)
    run_dir = directory / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pidfile = run_dir / "watchdog.pid"
    pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")
    _write_watchdog_meta(directory, args)

    stop = {"flag": False}

    def _stop(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def log(msg: str) -> None:
        sys.stdout.write(f"[{now_iso()}] watchdog: {msg}\n")
        sys.stdout.flush()

    log(
        f"started (idle_timeout={args.idle_timeout}s, poll={args.poll}s, "
        f"busy_util={args.busy_util}%, vram_floor={args.vram_floor}, "
        f"free_timeout={args.free_timeout}s, stale_grace={args.stale_grace}s)"
    )
    try:
        while not stop["flag"]:
            time.sleep(max(1, args.poll))
            if stop["flag"]:
                break
            try:
                _watchdog_once(
                    directory, idle_timeout=args.idle_timeout, busy_util=args.busy_util,
                    vram_floor=args.vram_floor, free_timeout=args.free_timeout,
                    stale_grace=args.stale_grace, log=log,
                )
            except Exception as exc:  # never let the watchdog die on a transient error
                log(f"error: {exc}")
    finally:
        pidfile.unlink(missing_ok=True)
        log("stopped")
    return 0


# ── argparse ───────────────────────────────────────────────────────────────--
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpu-broker", description="Daemonless GPU lease arbiter")
    parser.add_argument("--dir", default=None, help=f"Broker state dir (default {DEFAULT_DIR})")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_evict_opts(p: argparse.ArgumentParser) -> None:
        p.add_argument("--vram-floor", type=int, default=VRAM_FLOOR_MB,
                       help="VRAM (MB) at/below which the card counts as freed")
        p.add_argument("--free-timeout", type=float, default=FREE_TIMEOUT_S,
                       help="Max seconds to wait for VRAM to drain after eviction")

    p_acq = sub.add_parser("acquire", help="Acquire the GPU for a runtime, evicting the current holder")
    p_acq.add_argument("--runtime", required=True, help="Runtime id (e.g. llama, locate, ocr, mineru)")
    p_acq.add_argument("--pidfile", default=None, help="Holder pidfile the broker may signal to evict")
    p_acq.add_argument("--pid", type=int, default=None, help="Holder pid (if no pidfile)")
    p_acq.add_argument("--shutdown-url", default=None,
                       help="Optional loopback POST the broker calls before SIGTERM")
    p_acq.add_argument("--vram-mb", type=int, default=None, help="Approx VRAM the runtime uses")
    p_acq.add_argument("--label", default=None, help="Free-text label")
    p_acq.add_argument("--wait", type=float, default=0.0, help="Seconds to wait for a busy holder to go idle")
    p_acq.add_argument("--poll", type=float, default=2.0, help="Poll interval while waiting")
    p_acq.add_argument("--force", action="store_true", help="Evict immediately even if the holder is busy")
    p_acq.add_argument("--no-evict", action="store_true",
                       help="Reassert or take a free/abandoned lease, but do not evict another runtime")
    p_acq.add_argument("--busy-util", type=int, default=BUSY_UTIL_PCT, help="GPU util%% considered busy")
    p_acq.add_argument("--stale-grace", type=float, default=STALE_GRACE_S,
                       help="A dead holder is reclaimable only this long after acquire/touch")
    add_evict_opts(p_acq)

    p_rel = sub.add_parser("release", help="Release the GPU if this runtime holds it")
    p_rel.add_argument("--runtime", required=True)
    p_rel.add_argument("--token", default=None, help="Only release if the lease token matches")

    p_touch = sub.add_parser("touch", help="Mark this runtime active (refresh the idle clock)")
    p_touch.add_argument("--runtime", required=True)

    p_ev = sub.add_parser("evict", help="Manually evict the current (or a named) holder")
    p_ev.add_argument("--runtime", default=None, help="Only evict if this runtime holds it")
    add_evict_opts(p_ev)

    p_st = sub.add_parser("status", help="Show holder, GPU, processes, and watchdog status")
    p_st.add_argument("--json", action="store_true")
    p_st.add_argument("--busy-util", type=int, default=BUSY_UTIL_PCT)

    p_events = sub.add_parser("events", help="Show recent broker events")
    p_events.add_argument("--json", action="store_true")
    p_events.add_argument("--limit", type=int, default=50)
    p_events.add_argument("--since", default=None, help="Only events at or after this ISO timestamp")

    sub.add_parser("version", help="Print broker version and protocol")
    sub.add_parser("holder", help="Print the current holder runtime id (or 'none')")

    p_wd = sub.add_parser("watchdog", help="Run the idle-eviction loop (foreground)")
    p_wd.add_argument("--idle-timeout", type=float, default=WATCHDOG_IDLE_S)
    p_wd.add_argument("--poll", type=float, default=WATCHDOG_POLL_S)
    p_wd.add_argument("--busy-util", type=int, default=BUSY_UTIL_PCT)
    p_wd.add_argument("--stale-grace", type=float, default=STALE_GRACE_S)
    add_evict_opts(p_wd)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "acquire": cmd_acquire,
        "release": cmd_release,
        "touch": cmd_touch,
        "evict": cmd_evict,
        "status": cmd_status,
        "events": cmd_events,
        "version": cmd_version,
        "holder": cmd_holder,
        "watchdog": cmd_watchdog,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
