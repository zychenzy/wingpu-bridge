#!/usr/bin/env python3
"""Back-compat entry point.

The broker that wingpu uploads lives in the gpubroker package
(``gpubroker/src/gpubroker/broker.py``). This shim only exists so a checkout
can still run ``python3 wsl/gpu_broker.py`` during development.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "gpubroker" / "src" / "gpubroker" / "broker.py"


if __name__ == "__main__":
    sys.argv[0] = str(TARGET)
    runpy.run_path(str(TARGET), run_name="__main__")
