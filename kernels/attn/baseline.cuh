#pragma once
// Standard (unfused) attention: the baseline FlashAttention is measured against.
//   S = scale * Q K^T     cuBLAS, fp32 output   -> BH x N x N floats in HBM
//   P = softmax(S)        one block per row     -> BH x N x N bf16 in HBM
//   O = P V               cuBLAS, fp32 output
// Layout: Q, K, V are [BH, N, D] row-major bf16.
#include <cfloat>
#include <cuda_bf16.h>

// Block-wide reductions (blockDim.x multiple of 32, <= 1024).
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
template <bool IS_MAX>
__device__ float block_reduce(float v) {
  __shared__ float buf[32];
  __shared__ float result;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  v = IS_MAX ? warp_max(v) : warp_sum(v);
  if (lane == 0) buf[warp] = v;
  __syncthreads();
  if (warp == 0) {
    const int nw = blockDim.x / 32;
    float x = lane < nw ? buf[lane] : (IS_MAX ? -FLT_MAX : 0.f);
    x = IS_MAX ? warp_max(x) : warp_sum(x);
    if (lane == 0) result = x;
  }
  __syncthreads();
  return result;
}

// One block per row of S. Masked entries (causal, key > query) get probability 0.
__global__ void softmax_rows(const float* __restrict__ S, __nv_bfloat16* __restrict__ P, int N,
                             int causal) {
  const size_t row = blockIdx.x;
  const int q = static_cast<int>(row % N);
  const float* s = S + row * N;
  __nv_bfloat16* p = P + row * N;
  const int valid = causal ? q + 1 : N;

  float m = -FLT_MAX;
  for (int j = threadIdx.x; j < valid; j += blockDim.x) m = fmaxf(m, s[j]);
  m = block_reduce<true>(m);

  float sum = 0.f;
  for (int j = threadIdx.x; j < valid; j += blockDim.x) sum += __expf(s[j] - m);
  sum = block_reduce<false>(sum);
  const float inv = 1.f / sum;

  for (int j = threadIdx.x; j < N; j += blockDim.x)
    p[j] = __float2bfloat16(j < valid ? __expf(s[j] - m) * inv : 0.f);
}

// Row-major results through column-major cuBLAS:
//   S^T (N x N, ld N) = K (op T on the d x N col-major view) * Q^T
//   O^T (D x N, ld D) = V^T * P^T
inline void attention_baseline(cublasHandle_t h, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                               const __nv_bfloat16* V, float* S, __nv_bfloat16* P, float* O,
                               int BH, int N, int D, bool causal) {
  const float scale = 1.f / sqrtf(static_cast<float>(D)), one = 1.f, zero = 0.f;
  const long long sQKV = (long long)N * D, sSP = (long long)N * N;
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, N, D, &scale, K,
                                          CUDA_R_16BF, D, sQKV, Q, CUDA_R_16BF, D, sQKV, &zero, S,
                                          CUDA_R_32F, N, sSP, BH, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
  softmax_rows<<<BH * N, 256>>>(S, P, N, causal ? 1 : 0);
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, D, N, N, &one, V,
                                          CUDA_R_16BF, D, sQKV, P, CUDA_R_16BF, N, sSP, &zero, O,
                                          CUDA_R_32F, D, sQKV, BH, CUBLAS_COMPUTE_32F,
                                          CUBLAS_GEMM_DEFAULT));
}
