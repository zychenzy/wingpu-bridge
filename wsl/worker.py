#!/usr/bin/env python3
import argparse
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge_db import BridgeDB


def _expand(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return os.path.expandvars(str(Path(path).expanduser()))


def _docker_cmd(job: Dict[str, Any]) -> List[str]:
    image = job.get("image")
    if not image:
        raise RuntimeError("Containerized job requires image")

    cmd = ["docker", "run", "--rm", "--name", f"gpu-bridge-{job['job_id'][:20]}"]
    gpu = str(job.get("gpu") or "all")
    if gpu.lower() == "all":
        cmd.extend(["--gpus", "all"])
    else:
        cmd.extend(["--gpus", f"device={gpu}"])

    workdir = _expand(job.get("workdir"))
    mount_path = job.get("mount_path") or "/workspace"
    if workdir:
        cmd.extend(["-v", f"{workdir}:{mount_path}"])
        cmd.extend(["-w", mount_path])

    env_file = _expand(job.get("env_file"))
    if env_file and Path(env_file).exists():
        cmd.extend(["--env-file", env_file])

    resources = job.get("resources") or {}
    if resources.get("cpus"):
        cmd.extend(["--cpus", str(resources["cpus"])])
    if resources.get("memory"):
        cmd.extend(["--memory", str(resources["memory"])])
    if resources.get("shm_size"):
        cmd.extend(["--shm-size", str(resources["shm_size"])])

    cmd.extend([image, "bash", "-lc", job["cmd"]])
    return cmd


def _host_cmd(job: Dict[str, Any]) -> List[str]:
    return ["bash", "-lc", job["cmd"]]


def _kill_container_if_exists(job_id: str) -> None:
    name = f"gpu-bridge-{job_id[:20]}"
    subprocess.run(["docker", "rm", "-f", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _run_job(db: BridgeDB, job: Dict[str, Any], heartbeat_interval: int) -> None:
    log_path = Path(_expand(job.get("log_path") or "" ) or "")
    if not log_path.is_absolute():
        log_path = Path.home() / ".gpu-bridge" / "logs" / (job["job_id"] + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    containerized = bool(job.get("containerized", True))
    command = _docker_cmd(job) if containerized else _host_cmd(job)

    cwd = _expand(job.get("workdir")) if not containerized else None

    with log_path.open("a", encoding="utf-8") as lf:
        lf.write(f"[{BridgeDB.now_iso()}] START {job['job_id']}\n")
        lf.write(f"[{BridgeDB.now_iso()}] CMD {' '.join(shlex.quote(x) for x in command)}\n")
        lf.flush()

        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        def reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                lf.write(line)
                lf.flush()

        t = threading.Thread(target=reader, daemon=True)
        t.start()

        canceled = False
        next_heartbeat = time.time() + heartbeat_interval

        while proc.poll() is None:
            if time.time() >= next_heartbeat:
                db.heartbeat(job["job_id"])
                next_heartbeat = time.time() + heartbeat_interval

            if db.is_cancel_requested(job["job_id"]):
                canceled = True
                proc.terminate()
                if containerized:
                    _kill_container_if_exists(job["job_id"])
                break

            time.sleep(1)

        if proc.poll() is None:
            # Still alive after terminate signal.
            time.sleep(3)
            if proc.poll() is None:
                proc.kill()

        t.join(timeout=10)
        rc = proc.returncode if proc.returncode is not None else -9

        lf.write(f"[{BridgeDB.now_iso()}] END {job['job_id']} rc={rc}\n")
        lf.flush()

    if canceled:
        db.finish(job["job_id"], state="canceled", exit_code=rc, error="cancel requested")
        return

    if rc == 0:
        db.finish(job["job_id"], state="succeeded", exit_code=0)
        return

    db.requeue_after_failure(job["job_id"], error=f"exit_code={rc}", exit_code=rc)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU bridge worker")
    parser.add_argument("--db", default="~/.gpu-bridge/state/bridge.db")
    parser.add_argument("--poll-interval", type=int, default=4)
    parser.add_argument("--heartbeat-interval", type=int, default=10)
    parser.add_argument("--stale-seconds", type=int, default=120)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    db = BridgeDB(args.db)

    stop = False

    def _stop(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    while not stop:
        db.mark_stale_running_as_interrupted(stale_seconds=args.stale_seconds)
        job = db.claim_next()
        if job:
            _run_job(db, job, heartbeat_interval=args.heartbeat_interval)
            if args.once:
                break
            continue

        if args.once:
            break
        time.sleep(max(1, args.poll_interval))


if __name__ == "__main__":
    main()
