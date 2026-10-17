#!/usr/bin/env bash
# One-time setup on an AWS GPU instance launched from an
# "Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)" image (driver + CUDA preinstalled).
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv

# DLAMI ships CUDA under /usr/local/cuda; make sure nvcc/ncu/nsys are on PATH.
if ! command -v nvcc >/dev/null; then
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}' >> ~/.bashrc
  export PATH=/usr/local/cuda/bin:$PATH
fi
echo "== CUDA =="; nvcc --version | tail -n 2

# Persistence mode avoids driver re-init latency between runs.
sudo nvidia-smi -pm 1 >/dev/null || true

# Allow Nsight Compute to read hardware counters without sudo (takes effect after reboot).
if [ ! -f /etc/modprobe.d/nvidia-profiling.conf ]; then
  echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" | sudo tee /etc/modprobe.d/nvidia-profiling.conf
  echo ">> Reboot once (sudo reboot) so ncu works without sudo. Until then: sudo \$(which ncu) ..."
fi

# Ubuntu 24.04 blocks system-wide pip installs (PEP 668), so use the apt packages.
sudo apt-get update -qq && sudo apt-get install -y -qq python3-pandas python3-matplotlib >/dev/null

# The DLAMI's CUDA toolkit omits the SASS tools; install the ones matching this CUDA version.
CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1-\2/p')
sudo apt-get install -y -qq "cuda-cuobjdump-${CUDA_VER}" "cuda-nvdisasm-${CUDA_VER}" >/dev/null || \
  echo ">> Could not install cuobjdump/nvdisasm; use 'ncu --page source --print-source sass' instead."

echo "== Done. Next: make ARCH=sm_89 && make run =="
