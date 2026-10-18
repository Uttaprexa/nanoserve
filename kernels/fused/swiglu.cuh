#pragma once
// SwiGLU activation of a Llama MLP: out = silu(gate) * up, where gate and up come from
// one fused projection gu = x @ [W_gate | W_up]: gu [T, 2I] bf16, gate = gu[:, :I], up = gu[:, I:].
// out: [T, I] bf16.
#include "vec.cuh"

__device__ __forceinline__ float silu(float g) { return g / (1.f + __expf(-g)); }

// ---------------- fused: read gate and up once, write out once (6 bytes/element) ----------------
__global__ void swiglu_fused_kernel(const __nv_bfloat16* __restrict__ gu,
                                    __nv_bfloat16* __restrict__ out, int T, int I) {
  const int vpr = I / 8;
  const size_t n8 = (size_t)T * vpr;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / vpr;
    const int c = static_cast<int>(i % vpr);
    const uint4* row = reinterpret_cast<const uint4*>(gu + t * 2 * I);
    float g[8], u[8];
    unpack8(row[c], g);
    unpack8(row[vpr + c], u);
#pragma unroll
    for (int e = 0; e < 8; ++e) g[e] = silu(g[e]) * u[e];
    reinterpret_cast<uint4*>(out)[i] = pack8(g);
  }
}

inline void swiglu_fused(const __nv_bfloat16* gu, __nv_bfloat16* out, int T, int I,
                         cudaStream_t stream = 0) {
  swiglu_fused_kernel<<<grid_for((size_t)T * I / 8), 256, 0, stream>>>(gu, out, T, I);
}

// Same op with gate and up in separate contiguous [T, I] buffers (separate projections, as in
// HF checkpoints, so no fused weight copy is needed).
__global__ void swiglu2_fused_kernel(const __nv_bfloat16* __restrict__ gate,
                                     const __nv_bfloat16* __restrict__ up,
                                     __nv_bfloat16* __restrict__ out, size_t n8) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    float g[8], u[8];
    unpack8(reinterpret_cast<const uint4*>(gate)[i], g);
    unpack8(reinterpret_cast<const uint4*>(up)[i], u);
#pragma unroll
    for (int e = 0; e < 8; ++e) g[e] = silu(g[e]) * u[e];
    reinterpret_cast<uint4*>(out)[i] = pack8(g);
  }
}
inline void swiglu2_fused(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                          int T, int I, cudaStream_t stream = 0) {
  const size_t n8 = (size_t)T * I / 8;
  swiglu2_fused_kernel<<<grid_for(n8), 256, 0, stream>>>(gate, up, out, n8);
}

// ---------------- eager: act_fn(gate) -> tmp (bf16), then tmp * up ----------------
__global__ void eager_silu_kernel(const __nv_bfloat16* __restrict__ gu,
                                  __nv_bfloat16* __restrict__ tmp, int T, int I) {
  const int vpr = I / 8;
  const size_t n8 = (size_t)T * vpr;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / vpr;
    const int c = static_cast<int>(i % vpr);
    float g[8];
    unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I)[c], g);
#pragma unroll
    for (int e = 0; e < 8; ++e) g[e] = silu(g[e]);
    reinterpret_cast<uint4*>(tmp)[i] = pack8(g);
  }
}
__global__ void eager_mul_up_kernel(const __nv_bfloat16* __restrict__ tmp,
                                    const __nv_bfloat16* __restrict__ gu,
                                    __nv_bfloat16* __restrict__ out, int T, int I) {
  const int vpr = I / 8;
  const size_t n8 = (size_t)T * vpr;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / vpr;
    const int c = static_cast<int>(i % vpr);
    float a[8], u[8];
    unpack8(reinterpret_cast<const uint4*>(tmp)[i], a);
    unpack8(reinterpret_cast<const uint4*>(gu + t * 2 * I)[vpr + c], u);
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] *= u[e];
    reinterpret_cast<uint4*>(out)[i] = pack8(a);
  }
}

inline void eager_swiglu(const __nv_bfloat16* gu, __nv_bfloat16* tmp, __nv_bfloat16* out, int T,
                         int I) {
  const int g = grid_for((size_t)T * I / 8);
  eager_silu_kernel<<<g, 256>>>(gu, tmp, T, I);
  eager_mul_up_kernel<<<g, 256>>>(tmp, gu, out, T, I);
}
