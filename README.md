# nanoserve

An LLM inference engine where every performance-critical kernel is hand-written in CUDA C++:
GEMM, FlashAttention, paged attention, fused norms/RoPE, quantized GEMM and mixture-of-experts.
PyTorch is used only as a correctness reference. Every kernel ships with a benchmark against the
vendor library, a roofline position and an Nsight Compute explanation of where its time goes.

## Roadmap

| Phase | Deliverable | Baseline to beat | Status |
|---|---|---|---|
| 1 | SGEMM, naive → warptiled; then BF16 tensor-core GEMM (`mma.sync`) | cuBLAS | **done: 91% FP32, 90% BF16** |
| 2 | FlashAttention-2 forward + paged-attention decode | FlashAttention, PyTorch SDPA | next |
| 3 | Fused RMSNorm+residual, RoPE, SwiGLU | unfused PyTorch | |
| 4 | W4A16 / FP8 GEMM with in-register dequant + perplexity check | cuBLASLt, Marlin-style kernels | |
| 5 | MoE: fused top-k router + grouped GEMM with token permutation | PyTorch loop over experts | |
| 6 | Engine: run a real model end to end, tokens/s | llama.cpp, vLLM | |
| 7 | Hopper port (TMA + WGMMA) of the best GEMM/attention kernel | cuBLAS / FA3 on H100 | |

## Phase 1 results: GEMM on an NVIDIA L4

All numbers are % of cuBLAS on the same GPU (median-of-runs to be added). FP32 cuBLAS runs in pedantic
mode (no TF32). The L4 is power-limited (72 W): under sustained load its SM clock dropped from
2040 to 1215 MHz, which is why everyone, cuBLAS included, slows down at 4096.

### FP32 (CUDA cores)

| # | Kernel | 1024 | 2048 | 4096 | Key idea / evidence |
|---|---|---|---|---|---|
| 1 | naive | 1.7% | 1.4% | 1.8% | uncoalesced: 98% memory, 12% compute (ncu) |
| 2 | coalesced | 12.7% | 9.9% | 9.1% | 8.3x from coalescing alone |
| 3 | smem_tiling | 15.5% | 13.1% | 12.7% | only 1.13x: bound by shared-memory loads (2 per FMA) |
| 4 | blocktile_1d | 45.8% | 41.9% | 32.3% | 8 outputs/thread: 3x at the same occupancy |
| 5 | blocktile_2d | 47.6% | 58.6% | 59.6% | 8x8 outputs/thread; wave quantization at 1024 (64 blocks, 58 SMs) |
| 6 | vectorized | 66.0% | 73.7% | 76.5% | SASS: compiler already used LDS.128 in k5; gain is LDG/STG.128 (4x fewer global ops) |
| 7 | warptiling | 76.5% | 70.5% | 85.8% | 16.7% occupancy, fastest kernel |
| 8 | warptile_3blk | 64.9% | 82.4% | 87.5% | register cap for 3 blocks/SM: wins at 2048 (2.21 -> 1.47 waves), small spill |

Best per size: **77-91% of cuBLAS** (run-to-run variance up to ~7%).

![SGEMM results](results/sgemm_NVIDIA_L4.png)

### BF16 tensor cores (FP32 accumulate)

cuBLAS: 54 / 75 / 75 TFLOP/s at 1024 / 2048 / 4096, about 4.6x its FP32 throughput.

| # | Kernel | 1024 | 2048 | 4096 | Key idea / evidence |
|---|---|---|---|---|---|
| 1 | wmma_naive | 23.2% | 15.7% | 10.4% | fragments straight from global memory |
| 2 | wmma_smem | 58.7% | 67.4% | 68.9% | SASS: WMMA smem loads compiled to 48 generic LD.E, no LDSM |
| 3 | wmma_async2 | 60.0% | 70.8% | 64.3% | cp.async double buffering (LDGSTS in SASS) |
| 3 | wmma_async3 | 50.1% | 61.1% | 56.2% | 3 stages = 57 KB smem -> 1 block/SM, slower |
| 4 | mma_w64x32 | 75.3% | 87.4% | 83.9% | ldmatrix + mma.sync: LDSM replaces LD.E |
| 4 | **mma_w64x64** | **76.0%** | **89.9%** | **87.3%** | 64x64 warp tiles, 231 regs, 0 spills: **67.5 TFLOP/s** |

## Quickstart (AWS)

1. Request GPU quota: Service Quotas → EC2 → "Running On-Demand G and VT instances" → 4 or 8 vCPUs.
2. Launch `g6.xlarge` (NVIDIA L4, sm_89) with the **Deep Learning Base AMI with Single CUDA (Ubuntu 24.04)** (x86), 100 GB gp3.
3. On the instance:

```bash
git clone https://github.com/Uttaprexa/nanoserve.git && cd nanoserve
./scripts/setup_ec2.sh          # then `sudo reboot` once for profiler permissions
make                            # builds both benchmarks; prints registers/spills per kernel
./build/bench_sgemm             # FP32: correctness + GFLOP/s vs cuBLAS -> results/*.csv
./build/bench_hgemm             # BF16 tensor cores vs cuBLAS
make sass                       # load widths (LDS/LDSM) and HMMA counts per kernel
```

**Stop the instance when you're done.** A stopped instance only bills for its disk.

## Profiling

```bash
# Full Nsight Compute report for one kernel; open the .ncu-rep in the Nsight Compute GUI locally
ncu --set full -k regex:hgemm_mma -c 1 -o results/mma_4096 ./build/bench_hgemm --only 6 4096

# Waves, occupancy and registers
ncu --section LaunchStats --section Occupancy -k regex:sgemm_warptiling -c 1 ./build/bench_sgemm --only 7 2048
```

## Layout

```
include/common.cuh          error checks, CUDA-event timer
kernels/sgemm/NN_*.cuh      FP32 kernels, numbered in optimization order
kernels/hgemm/NN_*.cuh      BF16 tensor-core kernels; ptx.cuh wraps cp.async, ldmatrix, mma.sync
bench/bench_*.cu            correctness vs cuBLAS + timing + CSV
scripts/                    EC2 setup, build helper, plotting
results/                    CSVs, plots, ncu reports
docs/LOG.md                 worklog: what changed, what ncu said, what I learned
```
