#!/usr/bin/env python3
"""
gpu_broker.py — a daemonless GPU lease arbiter for the WSL GPU host.

One NVIDIA GPU is shared by several bridges (wingpu/llama.cpp, locatectl,
ocrctl, minerctl). They each load a heavy model that does not fit alongside the
others in VRAM, but nothing coordinates them, so two loaded at once means an
out-of-memory failure or an orphan process pinning VRAM.

This program is the single authority on who currently holds the GPU. A bridge
calls `acquire` before it loads its model; if another runtime holds the card the
broker evicts it (graceful HTTP shutdown if offered, else SIGTERM/SIGKILL on its
pidfile) and waits for VRAM to actually drain before granting the lease. A bridge
calls `release` when it stops. An optional `watchdog` loop evicts whichever
runtime has gone idle, freeing VRAM for the whole box.

Design notes:
  * Daemonless: state lives in SQLite at $GPU_BROKER_DIR/broker.db, mutual
    exclusion is an flock on broker.lock. Every command is a short-lived process.
    The one long-lived piece is the optional `watchdog`.
  * No third-party dependencies: it runs under whatever python3 WSL ships.
  * "Busy" is judged from real GPU utilization (nvidia-smi), so it works even for
    bridges that do not cooperate. Cooperating bridges may additionally `touch`.

All machine-facing commands (acquire/release/touch/evict/holder) print one JSON
line to stdout. `status` prints a human summary unless `--json` is given.
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
from typing import Any, Dict, Iterator, List, Optional
from urllib import request as urllib_request

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
  last_active_at TEXT
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


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def broker_dir(arg_dir: Optional[str]) -> Path:
    return Path(arg_dir or DEFAULT_DIR).expanduser()


# ── nvidia-smi probes (patched in tests) ─────────────────────────────────────
def _nvidia_smi(field: str) -> Optional[List[int]]:
    """Return a per-GPU integer list for `--query-gpu=<field>`, or None on error."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    values: List[int] = []
    for line in (out.stdout or "").splitlines():
        token = line.strip().rstrip("%").strip()
        if not token:
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


def gpu_busy_sampled(busy_util: int, samples: int = 3, interval: float = 0.4) -> bool:
    """True if any of a few quick utilization samples is at/above busy_util.

    Sampling several times avoids reading a brief lull between decode steps as idle.
    If utilization can't be read, assume not busy (let the caller fall back to the
    touch heartbeat / liveness checks).
    """
    for i in range(max(1, samples)):
        util = gpu_util_pct()
        if util is not None and util >= busy_util:
            return True
        if i + 1 < samples:
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
    except Exception:
        return None


def kill_pid(pid: int, *, grace: float = KILL_GRACE_S) -> None:
    """SIGTERM, wait up to `grace` for exit, then SIGKILL."""
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
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    handle = (directory / "broker.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle, fcntl.LOCK_UN)
        finally:
            handle.close()


def connect(directory: Path) -> sqlite3.Connection:
    directory.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(directory / "broker.db"), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record_event(conn: sqlite3.Connection, event: str, runtime_id: Optional[str], detail: Any = None) -> None:
    conn.execute(
        "INSERT INTO events(ts, event, runtime_id, detail) VALUES (?, ?, ?, ?)",
        (now_iso(), event, runtime_id, json.dumps(detail) if detail is not None else None),
    )


def read_holder(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT * FROM holder WHERE id = 1").fetchone()
    if not row or not row["runtime_id"]:
        return None
    return dict(row)


def set_holder(conn: sqlite3.Connection, fields: Dict[str, Any]) -> None:
    conn.execute(
        """
        UPDATE holder SET runtime_id=?, pid=?, pidfile=?, shutdown_url=?, vram_mb=?,
                          label=?, lease_token=?, acquired_at=?, last_active_at=?
        WHERE id = 1
        """,
        (
            fields.get("runtime_id"), fields.get("pid"), fields.get("pidfile"),
            fields.get("shutdown_url"), fields.get("vram_mb"), fields.get("label"),
            fields.get("lease_token"), fields.get("acquired_at"), fields.get("last_active_at"),
        ),
    )


def clear_holder(conn: sqlite3.Connection) -> None:
    set_holder(conn, {})


def touch_holder(conn: sqlite3.Connection) -> None:
    conn.execute("UPDATE holder SET last_active_at=? WHERE id = 1", (now_iso(),))


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


def holder_busy(holder: Dict[str, Any], busy_util: int, busy_grace: float) -> bool:
    """Busy if the GPU is computing, or the holder touched its lease very recently."""
    last = holder.get("last_active_at")
    if last:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
            if age < busy_grace:
                return True
        except ValueError:
            pass
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


def wait_vram_below(floor_mb: int, timeout: float) -> tuple:
    """Poll until GPU VRAM is at/below floor_mb (or unreadable). Returns (used_mb, ok)."""
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


def evict(conn: sqlite3.Connection, holder: Dict[str, Any], *, vram_floor: int,
          free_timeout: float, log=lambda _m: None) -> Dict[str, Any]:
    """Stop the holding runtime and wait for its VRAM to drain. Clears the holder."""
    runtime_id = holder.get("runtime_id")
    used_before = gpu_mem_used_mb()
    log(f"evicting holder '{runtime_id}' (vram_used={used_before}MB)")

    method = "none"
    shutdown_url = holder.get("shutdown_url")
    if shutdown_url and http_post(shutdown_url):
        method = "shutdown_url"
        # Give the graceful path a moment to begin tearing down.
        time.sleep(1.0)

    pidfile = holder.get("pidfile")
    pid = read_pidfile(pidfile) if pidfile else holder.get("pid")
    if pid and pid_alive(int(pid)):
        kill_pid(int(pid))
        method = "shutdown_url+signal" if method == "shutdown_url" else "signal"
    if pidfile:
        Path(pidfile).expanduser().unlink(missing_ok=True)

    # Wait for VRAM to actually come back before we let the next model load.
    target = vram_floor
    holder_vram = holder.get("vram_mb") or 0
    if holder_vram and used_before:
        target = max(vram_floor, int(used_before) - int(holder_vram * 0.8))
    drained = True
    deadline = time.time() + free_timeout
    used_after = used_before
    while time.time() < deadline:
        used_after = gpu_mem_used_mb()
        if used_after is None or used_after <= target:
            break
        time.sleep(1.0)
    else:
        drained = False
        log(f"VRAM did not drain below {target}MB within {free_timeout:g}s (used={used_after}MB)")

    record_event(conn, "evicted", runtime_id, {
        "method": method, "vram_before": used_before, "vram_after": used_after, "drained": drained,
    })
    clear_holder(conn)
    return {"evicted": runtime_id, "method": method, "vram_before_mb": used_before,
            "vram_after_mb": used_after, "vram_drained": drained}


# ── commands ───────────────────────────────────────────────────────────────--
def emit(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, sort_keys=True))


def cmd_acquire(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    me = args.runtime
    token = uuid.uuid4().hex
    fields = {
        "runtime_id": me, "pid": args.pid, "pidfile": args.pidfile,
        "shutdown_url": args.shutdown_url, "vram_mb": args.vram_mb, "label": args.label,
        "lease_token": token, "acquired_at": now_iso(), "last_active_at": None,
    }
    # last_active_at is set only by an explicit `touch`; acquiring does not make a
    # runtime look "busy". Until a holder touches, busy-ness is judged from GPU util.
    deadline = time.time() + (0 if args.force else max(0, args.wait))
    waited_for: Optional[str] = None
    wait_reason = "busy"

    while True:
        with file_lock(directory):
            conn = connect(directory)
            try:
                holder = read_holder(conn)
                if holder is None:
                    set_holder(conn, fields)
                    record_event(conn, "acquired", me, {"holder_was": None})
                    emit({"ok": True, "granted": True, "runtime": me, "token": token, "evicted": None})
                    return 0
                if holder["runtime_id"] == me:
                    # Re-assert: refresh registration and activity, keep the same lease.
                    fields["lease_token"] = holder["lease_token"] or token
                    fields["acquired_at"] = holder["acquired_at"] or fields["acquired_at"]
                    fields["last_active_at"] = holder["last_active_at"]  # keep prior touches
                    set_holder(conn, fields)
                    emit({"ok": True, "granted": True, "runtime": me,
                          "token": fields["lease_token"], "evicted": None, "reasserted": True})
                    return 0
                alive = holder_alive(holder)
                if not alive and holder_age_seconds(holder) >= args.stale_grace:
                    # Dead and stale: reclaim, but only once VRAM has actually freed.
                    # A still-occupied card means the "dead" holder is restarting (or
                    # something else is resident); granting now would OOM, so we wait.
                    used, drained = wait_vram_below(args.vram_floor, args.free_timeout)
                    if drained:
                        set_holder(conn, fields)
                        record_event(conn, "acquired", me,
                                     {"holder_was": holder["runtime_id"], "stale": True})
                        emit({"ok": True, "granted": True, "runtime": me, "token": token,
                              "evicted": holder["runtime_id"], "holder_was_stale": True,
                              "vram_after_mb": used, "vram_drained": True})
                        return 0
                    waited_for, wait_reason = holder["runtime_id"], "vram_busy"
                elif not alive:
                    # Dead but still fresh: a momentary blip, the owner may be restarting it.
                    # Protect it; only --force preempts.
                    if args.force:
                        info = evict(conn, holder, vram_floor=args.vram_floor, free_timeout=args.free_timeout)
                        set_holder(conn, fields)
                        emit({"ok": True, "granted": True, "runtime": me, "token": token,
                              "evicted": info["evicted"], "evict": info, "forced": True})
                        return 0
                    waited_for, wait_reason = holder["runtime_id"], "recovering"
                else:
                    busy = (not args.force) and holder_busy(holder, args.busy_util, BUSY_GRACE_S)
                    if not busy:
                        info = evict(conn, holder, vram_floor=args.vram_floor, free_timeout=args.free_timeout)
                        set_holder(conn, fields)
                        record_event(conn, "acquired", me,
                                     {"holder_was": holder["runtime_id"], "forced": bool(args.force)})
                        emit({"ok": True, "granted": True, "runtime": me, "token": token,
                              "evicted": info["evicted"], "evict": info, "forced": bool(args.force)})
                        return 0
                    waited_for, wait_reason = holder["runtime_id"], "busy"
            finally:
                conn.close()
        # Holder is alive and busy and we are not forcing: wait outside the lock.
        if time.time() >= deadline:
            emit({"ok": False, "granted": False, "runtime": me, "reason": wait_reason,
                  "holder": waited_for})
            return 3
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
            info = evict(conn, holder, vram_floor=args.vram_floor, free_timeout=args.free_timeout)
            emit({"ok": True, **info})
            return 0
        finally:
            conn.close()


def holder_view(holder: Optional[Dict[str, Any]], busy_util: int) -> Optional[Dict[str, Any]]:
    if holder is None:
        return None
    view = {k: holder.get(k) for k in
            ("runtime_id", "pid", "pidfile", "vram_mb", "label", "acquired_at", "last_active_at")}
    view["alive"] = holder_alive(holder)
    view["busy"] = holder_busy(holder, busy_util, BUSY_GRACE_S)
    return view


def watchdog_state(directory: Path) -> Dict[str, Any]:
    pid = read_pidfile(str(directory / "run" / "watchdog.pid"))
    return {"running": bool(pid and pid_alive(pid)), "pid": pid}


def cmd_status(args: argparse.Namespace) -> int:
    directory = broker_dir(args.dir)
    conn = connect(directory)
    try:
        holder = read_holder(conn)
    finally:
        conn.close()
    gpu = {"mem_used_mb": gpu_mem_used_mb(), "mem_total_mb": gpu_mem_total_mb(), "util_pct": gpu_util_pct()}
    payload = {
        "broker_dir": str(directory),
        "holder": holder_view(holder, args.busy_util),
        "gpu": gpu,
        "watchdog": watchdog_state(directory),
    }
    if args.json:
        emit(payload)
        return 0
    h = payload["holder"]
    print(f"broker dir : {directory}")
    if h:
        print(f"holder     : {h['runtime_id']}  (pid={h['pid']}, alive={h['alive']}, busy={h['busy']})")
        print(f"  since    : {h['acquired_at']}")
        print(f"  active   : {h['last_active_at']}")
        if h.get("vram_mb"):
            print(f"  vram est : {h['vram_mb']} MB")
    else:
        print("holder     : (none) — GPU is free")
    if gpu["mem_used_mb"] is not None:
        print(f"gpu vram   : {gpu['mem_used_mb']} / {gpu['mem_total_mb']} MB used, util {gpu['util_pct']}%")
    else:
        print("gpu vram   : (nvidia-smi unavailable)")
    wd = payload["watchdog"]
    print(f"watchdog   : {'running pid ' + str(wd['pid']) if wd['running'] else 'stopped'}")
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

    Returns an action dict ({"action": "none"|"cleared_stale"|"idle_offload", ...}).
    """
    with file_lock(directory):
        conn = connect(directory)
        try:
            holder = read_holder(conn)
            if holder is None:
                return {"action": "none"}
            if not holder_alive(holder) and holder_age_seconds(holder) >= stale_grace:
                record_event(conn, "cleared_stale", holder["runtime_id"], None)
                clear_holder(conn)
                log(f"cleared stale holder '{holder['runtime_id']}'")
                return {"action": "cleared_stale", "runtime": holder["runtime_id"]}
            last = holder.get("last_active_at") or holder.get("acquired_at")
            idle_age: Optional[float] = None
            if last:
                try:
                    idle_age = (datetime.now(timezone.utc) - datetime.fromisoformat(last)).total_seconds()
                except ValueError:
                    idle_age = None
            if (idle_age is not None and idle_age >= idle_timeout
                    and not holder_busy(holder, busy_util, BUSY_GRACE_S)):
                info = evict(conn, holder, vram_floor=vram_floor, free_timeout=free_timeout, log=log)
                record_event(conn, "idle_offload", holder["runtime_id"], {"idle_age_s": int(idle_age)})
                log(f"idle-offloaded '{info['evicted']}' after {int(idle_age)}s idle")
                return {"action": "idle_offload", "runtime": info["evicted"], "idle_age_s": int(idle_age)}
            return {"action": "none", "runtime": holder["runtime_id"], "idle_age_s": idle_age}
        finally:
            conn.close()


def cmd_watchdog(args: argparse.Namespace) -> int:
    """Long-lived loop: evict whichever runtime has gone idle, freeing VRAM."""
    directory = broker_dir(args.dir)
    run_dir = directory / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    pidfile = run_dir / "watchdog.pid"
    pidfile.write_text(f"{os.getpid()}\n", encoding="utf-8")

    stop = {"flag": False}

    def _stop(_sig, _frm):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    def log(msg: str) -> None:
        sys.stdout.write(f"[{now_iso()}] watchdog: {msg}\n")
        sys.stdout.flush()

    log(f"started (idle_timeout={args.idle_timeout}s, poll={args.poll}s, busy_util={args.busy_util}%)")
    try:
        while not stop["flag"]:
            time.sleep(max(1, args.poll))
            try:
                _watchdog_once(directory, idle_timeout=args.idle_timeout, busy_util=args.busy_util,
                               vram_floor=args.vram_floor, free_timeout=args.free_timeout,
                               stale_grace=args.stale_grace, log=log)
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
    p_acq.add_argument("--shutdown-url", default=None, help="Optional graceful POST shutdown endpoint")
    p_acq.add_argument("--vram-mb", type=int, default=None, help="Approx VRAM the runtime uses (for logs/gate)")
    p_acq.add_argument("--label", default=None, help="Free-text label")
    p_acq.add_argument("--wait", type=float, default=0.0, help="Seconds to wait for a busy holder to go idle")
    p_acq.add_argument("--poll", type=float, default=2.0, help="Poll interval while waiting")
    p_acq.add_argument("--force", action="store_true", help="Evict immediately even if the holder is busy")
    p_acq.add_argument("--busy-util", type=int, default=BUSY_UTIL_PCT, help="GPU util%% considered busy")
    p_acq.add_argument("--stale-grace", type=float, default=STALE_GRACE_S,
                       help="A dead holder is reclaimable only this long after acquire/touch")
    add_evict_opts(p_acq)

    p_rel = sub.add_parser("release", help="Release the GPU if this runtime holds it")
    p_rel.add_argument("--runtime", required=True)
    p_rel.add_argument("--token", default=None, help="Only release if the lease token matches")

    p_touch = sub.add_parser("touch", help="Mark this runtime active (refresh idle timer)")
    p_touch.add_argument("--runtime", required=True)

    p_ev = sub.add_parser("evict", help="Manually evict the current (or a named) holder")
    p_ev.add_argument("--runtime", default=None, help="Only evict if this runtime holds it")
    add_evict_opts(p_ev)

    p_st = sub.add_parser("status", help="Show holder, GPU, and watchdog status")
    p_st.add_argument("--json", action="store_true")
    p_st.add_argument("--busy-util", type=int, default=BUSY_UTIL_PCT)

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
        "acquire": cmd_acquire, "release": cmd_release, "touch": cmd_touch,
        "evict": cmd_evict, "status": cmd_status, "holder": cmd_holder, "watchdog": cmd_watchdog,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
