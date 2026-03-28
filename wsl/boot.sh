#!/usr/bin/env bash
set -euo pipefail

BRIDGE_DIR="${1:-$HOME/.gpu-bridge}"
DB_PATH="$BRIDGE_DIR/state/bridge.db"

mkdir -p "$BRIDGE_DIR/state" "$BRIDGE_DIR/logs" "$BRIDGE_DIR/profiles"

# Initialize DB/schema.
python3 "$BRIDGE_DIR/bridge_ctl.py" --bridge-dir "$BRIDGE_DIR" status --limit 1 >/dev/null

if command -v systemctl >/dev/null 2>&1 && systemctl is-system-running >/dev/null 2>&1 && command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
  sudo systemctl restart gpu-bridge-worker.service || true
  sudo systemctl is-active gpu-bridge-worker.service >/dev/null || sudo systemctl start gpu-bridge-worker.service
else
  # Fallback for non-systemd environments.
  if ! pgrep -f "python3 $BRIDGE_DIR/worker.py" >/dev/null 2>&1; then
    nohup python3 "$BRIDGE_DIR/worker.py" --db "$DB_PATH" >"$BRIDGE_DIR/logs/worker.out" 2>&1 &
  fi
fi

echo "gpu-bridge boot complete: $BRIDGE_DIR"
