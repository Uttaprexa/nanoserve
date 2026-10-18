#!/usr/bin/env bash
# Phase 6 environment: a virtualenv with PyTorch + Hugging Face libraries.
# PyTorch is only used from Python; our kernels are loaded from build/libnanoserve.so via
# ctypes, so PyTorch's bundled CUDA version does not need to match the system nvcc.
set -euo pipefail
sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv >/dev/null
python3 -m venv ~/venv
~/venv/bin/pip install --quiet --upgrade pip
~/venv/bin/pip install --quiet torch transformers accelerate datasets
~/venv/bin/python - << 'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name() if torch.cuda.is_available() else "-")
PY
echo "== Done. Next: make lib && ~/venv/bin/python python/bench_e2e.py =="
