#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-$HOME/venvs/qwen-transformers}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_CUDA_INDEX_URL="${TORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

echo "[setup] creating venv at: $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[setup] preflight check for local build headers (needed by bitsandbytes/triton)"
python - <<'PY'
import pathlib
import sys
import sysconfig

include_dir = pathlib.Path(sysconfig.get_config_var("INCLUDEPY") or "")
python_h = include_dir / "Python.h"
if not python_h.exists():
    raise SystemExit(
        "[setup] missing Python headers. Install system deps first:\n"
        "  sudo apt-get update && sudo apt-get install -y python3-dev build-essential"
    )
print(f"[setup] found Python headers at: {python_h}")
PY

echo "[setup] upgrading pip tooling"
python -m pip install --upgrade pip setuptools wheel

echo "[setup] installing PyTorch (CUDA) from: $TORCH_CUDA_INDEX_URL"
python -m pip install --upgrade --index-url "$TORCH_CUDA_INDEX_URL" torch

echo "[setup] installing Transformers stack"
python -m pip install --upgrade \
  transformers \
  accelerate \
  bitsandbytes \
  safetensors \
  sentencepiece \
  huggingface_hub

echo "[setup] validating GPU visibility"
python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu_name={torch.cuda.get_device_name(0)}")
    print(f"cuda_device_count={torch.cuda.device_count()}")
else:
    raise SystemExit("CUDA is not available in torch. Check driver / wheel / WSL setup.")
PY

echo "[setup] done"
echo "[setup] activate with: source $VENV_DIR/bin/activate"
