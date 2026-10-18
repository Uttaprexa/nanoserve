#pragma once
// Helpers for memory-bound kernels: 8 x bf16 <-> one 16-byte uint4, block sum.
#include <cuda_bf16.h>

__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float2 p = __bfloat1622float2(h[i]);
    f[2 * i] = p.x;
    f[2 * i + 1] = p.y;
  }
}
__device__ __forceinline__ uint4 pack8(const float (&f)[8]) {
  uint4 u;
  __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(&u);
#pragma unroll
  for (int i = 0; i < 4; ++i) h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  return u;
}
__device__ __forceinline__ float round_bf16(float x) {
  return __bfloat162float(__float2bfloat16(x));
}

// Sum over the block (blockDim.x multiple of 32). Safe to call repeatedly.
__device__ float fused_block_sum(float v) {
  __shared__ float warp_sums[32];
  __shared__ float total;
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) warp_sums[warp] = v;
  __syncthreads();
  if (warp == 0) {
    float x = lane < (int)(blockDim.x / 32) ? warp_sums[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    if (lane == 0) total = x;
  }
  __syncthreads();
  return total;
}

// Pure device-to-device copy with 16-byte accesses: the practical bandwidth ceiling.
__global__ void copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

inline int grid_for(size_t n_threads, int block = 256) {
  const size_t g = (n_threads + block - 1) / block;
  return static_cast<int>(g < 65535 * 32 ? g : 65535 * 32);
}
