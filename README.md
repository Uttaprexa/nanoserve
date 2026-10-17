# nanoserve

An LLM inference engine where every performance-critical kernel is hand-written in CUDA C++:
GEMM, FlashAttention, paged attention, fused norms/RoPE, quantized GEMM and mixture-of-experts.
PyTorch is used only as a correctness reference. Every kernel ships with a benchmark against the
vendor library, a roofline position and an Nsight Compute explanation of where its time goes.

## Roadmap

| Phase | Deliverable | Baseline to beat | Status |
|---|---|---|---|
| 1 | SGEMM, naive → warptiled; then BF16 tensor-core GEMM (`mma.sync`) | cuBLAS | in progress |
| 2 | FlashAttention-2 forward + paged-attention decode | FlashAttention, PyTorch SDPA | |
| 3 | Fused RMSNorm+residual, RoPE, SwiGLU | unfused PyTorch | |
| 4 | W4A16 / FP8 GEMM with in-register dequant + perplexity check | cuBLASLt, Marlin-style kernels | |
| 5 | MoE: fused top-k router + grouped GEMM with token permutation | PyTorch loop over experts | |
| 6 | Engine: run a real model end to end, tokens/s | llama.cpp, vLLM | |
| 7 | Hopper port (TMA + WGMMA) of the best GEMM/attention kernel | cuBLAS / FA3 on H100 | |

## Quickstart (AWS)

1. Request GPU quota: Service Quotas → EC2 → "Running On-Demand G and VT instances" → 4 or 8 vCPUs.
2. Launch `g6.xlarge` (NVIDIA L4, sm_89) with an **Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)** (x86), 100 GB gp3.
3. On the instance:

```bash
git clone <your-repo> nanoserve && cd nanoserve
./scripts/setup_ec2.sh          # then `sudo reboot` once for profiler permissions
make ARCH=sm_89                 # prints registers/spills per kernel from ptxas
make run                        # correctness + GFLOP/s vs cuBLAS, writes results/*.csv
python3 scripts/plot_sgemm.py results/sgemm_NVIDIA_L4.csv
```

**Stop the instance when you're done.** A stopped instance only bills for its disk.

## Profiling

```bash
# Full Nsight Compute report for one kernel (kernel 4, 4096^3), open the .ncu-rep in the Nsight Compute GUI locally
ncu --set full -k regex:sgemm_blocktile -c 1 -o results/k4_4096 ./build/bench_sgemm --only 4 4096

# Quick memory-vs-compute view
ncu --section SpeedOfLight --section MemoryWorkloadAnalysis -k regex:sgemm -c 4 ./build/bench_sgemm 4096

# Look at the generated machine code
make sass && grep -c FFMA build/bench_sgemm.sass
```

## Phase 1 results (fill in from your runs)

GPU: NVIDIA L4 · FP32 · M = N = K = 4096 · cuBLAS in pedantic FP32 mode (no TF32)

| # | Kernel | GFLOP/s | % of cuBLAS | Key idea | Bottleneck (from ncu) |
|---|---|---|---|---|---|
| 0 | cuBLAS | | 100% | | |
| 1 | naive | | | one thread per output | uncoalesced global loads |
| 2 | coalesced | | | warp reads contiguous B/C | global memory bandwidth |
| 3 | smem_tiling | | | 32×32 tiles in shared memory | shared-memory throughput |
| 4 | blocktile_1d | | | 8 outputs per thread | |
| 5 | blocktile_2d | | | 8×8 outputs per thread | |
| 6 | vectorized | | | float4 loads, transposed As | |
| 7 | warptiling | | | warp-level tile hierarchy | |

## Layout

```
include/common.cuh        error checks, CUDA-event timer
kernels/sgemm/NN_*.cuh    one kernel per file, numbered in optimization order
bench/bench_sgemm.cu      correctness vs cuBLAS + timing + CSV
scripts/                  EC2 setup, plotting
results/                  CSVs, plots, ncu reports
docs/LOG.md               worklog: what changed, what ncu said, what I learned
```
