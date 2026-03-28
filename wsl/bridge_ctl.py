#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from bridge_db import BridgeDB


def now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def render(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=True, sort_keys=True))


def load_profile(profile_name: Optional[str], profiles_dir: Path) -> Dict[str, Any]:
    if not profile_name:
        return {}
    candidates = [
        profiles_dir / f"{profile_name}.json",
        profiles_dir / profile_name,
        Path(profile_name).expanduser(),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Profile not found: {profile_name}")


def merge_job(profile: Dict[str, Any], payload: Dict[str, Any], bridge_dir: Path) -> Dict[str, Any]:
    merged = dict(profile)
    merged.update({k: v for k, v in payload.items() if v is not None})

    if not merged.get("project"):
        raise ValueError("project is required")
    if not merged.get("cmd"):
        raise ValueError("cmd is required")

    containerized = bool(merged.get("containerized", True))
    if containerized and not merged.get("image"):
        raise ValueError("image is required for containerized jobs")

    job_id = merged.get("job_id") or f"job_{now_compact()}_{uuid.uuid4().hex[:8]}"
    logs_dir = bridge_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    return {
        "job_id": job_id,
        "project": merged["project"],
        "image": merged.get("image"),
        "cmd": merged["cmd"],
        "gpu": str(merged.get("gpu", "all")),
        "workdir": merged.get("workdir"),
        "mount_path": merged.get("mount_path", "/workspace"),
        "env_file": merged.get("env_file"),
        "resources": merged.get("resources", {}),
        "containerized": containerized,
        "max_retries": int(merged.get("max_retries", 0)),
        "log_path": str(logs_dir / f"{job_id}.log"),
    }


def tail_file(path: Path, lines: int = 200, follow: bool = False) -> int:
    if not path.exists():
        print(f"log not found: {path}", file=sys.stderr)
        return 1

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in content[-lines:]:
        print(line)

    if not follow:
        return 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        f.seek(0, os.SEEK_END)
        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                print(line, end="")
        except KeyboardInterrupt:
            return 0


def doctor(bridge_dir: Path, db: BridgeDB) -> Dict[str, Any]:
    checks: Dict[str, Any] = {
        "bridge_dir": str(bridge_dir),
        "db_path": db.db_path,
        "python": {"ok": True, "version": sys.version.split()[0]},
    }

    def run_check(cmd: list[str], timeout: int = 15) -> Dict[str, Any]:
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            ok = cp.returncode == 0
            return {
                "ok": ok,
                "returncode": cp.returncode,
                "stdout": cp.stdout.strip(),
                "stderr": cp.stderr.strip(),
            }
        except Exception as exc:  # pylint: disable=broad-except
            return {"ok": False, "error": str(exc)}

    checks["docker_cli"] = run_check(["docker", "version", "--format", "{{.Client.Version}}"])
    checks["docker_daemon"] = run_check(["docker", "info", "--format", "{{.ServerVersion}}"])
    checks["nvidia_smi"] = run_check(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])

    summary = db.list_jobs(limit=20)
    checks["queue_summary"] = summary.get("summary", {})

    checks["ok"] = all([
        checks["python"].get("ok", False),
        checks["docker_cli"].get("ok", False),
        checks["docker_daemon"].get("ok", False),
        checks["nvidia_smi"].get("ok", False),
    ])
    return checks


def cmd_submit(args: argparse.Namespace, db: BridgeDB, bridge_dir: Path, profiles_dir: Path) -> int:
    profile = load_profile(args.profile, profiles_dir)
    resources = json.loads(args.resources_json) if args.resources_json else None
    payload = {
        "project": args.project,
        "image": args.image,
        "cmd": args.cmd,
        "gpu": args.gpu,
        "workdir": args.workdir,
        "mount_path": args.mount_path,
        "env_file": args.env_file,
        "resources": resources,
        "containerized": None if args.containerized is None else bool(args.containerized),
        "max_retries": args.max_retries,
    }
    job = merge_job(profile, payload, bridge_dir)
    render({"ok": True, "job": db.submit(job)})
    return 0


def cmd_status(args: argparse.Namespace, db: BridgeDB) -> int:
    if args.job_id:
        job = db.get(args.job_id)
        if not job:
            render({"ok": False, "error": "job_not_found", "job_id": args.job_id})
            return 2
        render({"ok": True, "job": job})
        return 0
    render({"ok": True, **db.list_jobs(limit=args.limit)})
    return 0


def cmd_cancel(args: argparse.Namespace, db: BridgeDB) -> int:
    try:
        row = db.request_cancel(args.job_id)
    except ValueError as exc:
        render({"ok": False, "error": str(exc)})
        return 2
    render({"ok": True, "job": row})
    return 0


def cmd_logs(args: argparse.Namespace, db: BridgeDB) -> int:
    job = db.get(args.job_id)
    if not job:
        print(f"job not found: {args.job_id}", file=sys.stderr)
        return 2
    return tail_file(Path(job["log_path"]), lines=args.lines, follow=args.follow)


def cmd_doctor(_args: argparse.Namespace, db: BridgeDB, bridge_dir: Path) -> int:
    render(doctor(bridge_dir, db))
    return 0


def cmd_api(args: argparse.Namespace, db: BridgeDB, bridge_dir: Path, profiles_dir: Path) -> int:
    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        render({"ok": False, "error": f"invalid_json: {exc}"})
        return 2

    action = req.get("action")
    payload = req.get("payload") or {}

    try:
        if action == "submit":
            profile = load_profile(payload.get("profile"), profiles_dir)
            job = merge_job(profile, payload, bridge_dir)
            render({"ok": True, "job": db.submit(job)})
            return 0

        if action == "status":
            job_id = payload.get("job_id")
            if job_id:
                job = db.get(job_id)
                render({"ok": bool(job), "job": job, "error": None if job else "job_not_found"})
                return 0 if job else 2
            limit = int(payload.get("limit", 50))
            render({"ok": True, **db.list_jobs(limit=limit)})
            return 0

        if action == "cancel":
            row = db.request_cancel(payload["job_id"])
            render({"ok": True, "job": row})
            return 0

        if action == "doctor":
            render(doctor(bridge_dir, db))
            return 0

        render({"ok": False, "error": f"unsupported_action: {action}"})
        return 2
    except Exception as exc:  # pylint: disable=broad-except
        render({"ok": False, "error": str(exc)})
        return 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="gpu bridge control")
    parser.add_argument("--bridge-dir", default="~/.gpu-bridge")
    parser.add_argument("--db", default=None)
    parser.add_argument("--profiles-dir", default=None)

    sub = parser.add_subparsers(dest="sub", required=True)

    p_submit = sub.add_parser("submit")
    p_submit.add_argument("--project", required=True)
    p_submit.add_argument("--image")
    p_submit.add_argument("--cmd", required=True)
    p_submit.add_argument("--gpu", default="all")
    p_submit.add_argument("--workdir")
    p_submit.add_argument("--mount-path", default="/workspace")
    p_submit.add_argument("--env-file")
    p_submit.add_argument("--resources-json")
    p_submit.add_argument("--profile")
    p_submit.add_argument("--max-retries", type=int, default=0)
    p_submit.add_argument("--containerized", dest="containerized", action="store_true")
    p_submit.add_argument("--no-container", dest="containerized", action="store_false")
    p_submit.set_defaults(containerized=None)

    p_status = sub.add_parser("status")
    p_status.add_argument("--job-id")
    p_status.add_argument("--limit", type=int, default=50)

    p_cancel = sub.add_parser("cancel")
    p_cancel.add_argument("--job-id", required=True)

    p_logs = sub.add_parser("logs")
    p_logs.add_argument("--job-id", required=True)
    p_logs.add_argument("--lines", type=int, default=200)
    p_logs.add_argument("--follow", action="store_true")

    sub.add_parser("doctor")
    sub.add_parser("api")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bridge_dir = Path(args.bridge_dir).expanduser()
    bridge_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = Path(args.profiles_dir).expanduser() if args.profiles_dir else bridge_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    state_dir = bridge_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    db_path = args.db if args.db else str(state_dir / "bridge.db")
    db = BridgeDB(db_path)

    if args.sub == "submit":
        return cmd_submit(args, db, bridge_dir, profiles_dir)
    if args.sub == "status":
        return cmd_status(args, db)
    if args.sub == "cancel":
        return cmd_cancel(args, db)
    if args.sub == "logs":
        return cmd_logs(args, db)
    if args.sub == "doctor":
        return cmd_doctor(args, db, bridge_dir)
    if args.sub == "api":
        return cmd_api(args, db, bridge_dir, profiles_dir)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
