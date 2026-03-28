#!/usr/bin/env bash
set -euo pipefail

PREFIX="${1:-$HOME/.gpu-bridge}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$PREFIX" "$PREFIX/state" "$PREFIX/logs" "$PREFIX/profiles"

install -m 0755 "$SCRIPT_DIR/bridge_ctl.py" "$PREFIX/bridge_ctl.py"
install -m 0755 "$SCRIPT_DIR/bridge_db.py" "$PREFIX/bridge_db.py"
install -m 0755 "$SCRIPT_DIR/worker.py" "$PREFIX/worker.py"
install -m 0755 "$SCRIPT_DIR/boot.sh" "$PREFIX/boot.sh"

if [[ -d "$SCRIPT_DIR/../profiles" ]]; then
  cp -f "$SCRIPT_DIR/../profiles"/*.json "$PREFIX/profiles/" 2>/dev/null || true
fi

cat > "$PREFIX/gpu-bridge-worker.service" <<UNIT
[Unit]
Description=GPU Bridge Worker
After=network.target docker.service
Wants=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=$PREFIX
ExecStart=/usr/bin/env python3 $PREFIX/worker.py --db $PREFIX/state/bridge.db
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT

if command -v systemctl >/dev/null 2>&1; then
  echo "[install] systemctl detected; attempting system service install"
  if command -v sudo >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo install -m 0644 "$PREFIX/gpu-bridge-worker.service" /etc/systemd/system/gpu-bridge-worker.service
    sudo systemctl daemon-reload
    sudo systemctl enable gpu-bridge-worker.service
    sudo systemctl restart gpu-bridge-worker.service || true
  else
    echo "[install] warning: passwordless sudo unavailable, skipping system service install"
    echo "[install] warning: boot.sh fallback mode will still start worker on Windows startup task"
  fi
else
  echo "[install] systemctl not found; service installation skipped"
fi

echo "[install] bridge installed at: $PREFIX"
echo "[install] validate with: python3 $PREFIX/bridge_ctl.py doctor"
