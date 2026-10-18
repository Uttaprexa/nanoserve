# Override ARCH for your GPU: sm_89 = L4/L40S/RTX 40xx (AWS g6/g6e), sm_86 = A10G (g5),
# sm_80 = A100, sm_90a = H100 (needed later for WGMMA/TMA).
.RECIPEPREFIX = >
ARCH      ?= sm_89
NVCC      ?= nvcc
# Find cuobjdump next to the real nvcc, even if it isn't on PATH.
CUDA_BIN  := $(dir $(realpath $(shell command -v $(NVCC))))
CUOBJDUMP ?= $(CUDA_BIN)cuobjdump
NVCCFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo -Xptxas=-v -Iinclude -Ikernels
LDLIBS    := -lcublas

SGEMM_DEPS := include/common.cuh $(wildcard kernels/sgemm/*.cuh)
HGEMM_DEPS := include/common.cuh $(wildcard kernels/hgemm/*.cuh)
ATTN_DEPS  := include/common.cuh $(wildcard kernels/attn/*.cuh) kernels/hgemm/ptx.cuh
FUSED_DEPS := include/common.cuh $(wildcard kernels/fused/*.cuh)
W4_DEPS    := include/common.cuh $(wildcard kernels/quant/*.cuh) kernels/fused/vec.cuh kernels/hgemm/ptx.cuh
MOE_DEPS   := include/common.cuh $(wildcard kernels/moe/*.cuh) kernels/fused/swiglu.cuh kernels/fused/vec.cuh kernels/hgemm/ptx.cuh


.PHONY: all sgemm hgemm attn fused w4a16 moe lib run run-hgemm run-attn run-fused run-w4a16 run-moe sass clean

all: sgemm hgemm attn fused w4a16 moe lib
sgemm: build/bench_sgemm
hgemm: build/bench_hgemm
attn: build/bench_attn
fused: build/bench_fused
w4a16: build/bench_w4a16
moe: build/bench_moe
lib: build/libnanoserve.so

build/bench_sgemm: bench/bench_sgemm.cu $(SGEMM_DEPS)
> @bash scripts/nvcc_build.sh sgemm $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

build/bench_hgemm: bench/bench_hgemm.cu $(HGEMM_DEPS)
> @bash scripts/nvcc_build.sh hgemm $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

build/bench_attn: bench/bench_attn.cu $(ATTN_DEPS)
> @bash scripts/nvcc_build.sh attn $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

build/bench_fused: bench/bench_fused.cu $(FUSED_DEPS)
> @bash scripts/nvcc_build.sh fused $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

build/bench_w4a16: bench/bench_w4a16.cu $(W4_DEPS)
> @bash scripts/nvcc_build.sh w4a16 $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

build/bench_moe: bench/bench_moe.cu $(MOE_DEPS)
> @bash scripts/nvcc_build.sh moe $(NVCC) $(NVCCFLAGS) $< -o $@ $(LDLIBS)

# Shared library for the PyTorch integration (Phase 6), loaded with ctypes.
LIB_DEPS := include/common.cuh $(wildcard kernels/*/*.cuh)
build/libnanoserve.so: csrc/nanoserve_ops.cu $(LIB_DEPS)
> @bash scripts/nvcc_build.sh lib $(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC -shared $< -o $@

run: build/bench_sgemm
> ./build/bench_sgemm

run-hgemm: build/bench_hgemm
> ./build/bench_hgemm

run-attn: build/bench_attn
> ./build/bench_attn

run-fused: build/bench_fused
> ./build/bench_fused

run-w4a16: build/bench_w4a16
> ./build/bench_w4a16

run-moe: build/bench_moe
> ./build/bench_moe

# Dump SASS so you can confirm what the compiler emitted (e.g. FFMA, LDS.128, HMMA).
sass: build/bench_sgemm build/bench_hgemm
> $(CUOBJDUMP) -sass build/bench_sgemm > build/bench_sgemm.sass
> $(CUOBJDUMP) -sass build/bench_hgemm > build/bench_hgemm.sass
> @echo "Shared-memory load widths per kernel:"
> @awk '/Function :/{f=$$3} /LDS/{for(i=1;i<=NF;i++) if($$i ~ /^LDS/){gsub(/[;]/,"",$$i); c[f" "$$i]++}} END{for(k in c) print "  " c[k], k}' build/bench_sgemm.sass build/bench_hgemm.sass | sort -k2
> @echo "Tensor-core instructions per kernel:"
> @awk '/Function :/{f=$$3} /HMMA/{for(i=1;i<=NF;i++) if($$i ~ /^HMMA/){gsub(/[;]/,"",$$i); c[f" "$$i]++}} END{for(k in c) print "  " c[k], k}' build/bench_hgemm.sass | sort -k2

clean:
> rm -rf build
