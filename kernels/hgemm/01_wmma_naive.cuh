#pragma once
// Tensor-core kernel 1: WMMA, straight from global memory.
// C[M,N] (fp32) = alpha * A[M,K] (bf16, row-major) @ B[K,N] (bf16, row-major) + beta * C
//
// Each warp owns one 16x16 tile of C and walks K in steps of 16, loading
// A and B fragments directly from global memory and issuing one tensor-core
// MMA per step. Purpose: learn the fragment API. Expect it to be memory-bound,
// since nothing is reused through shared memory.
// REQUIRES: M % 32 == 0, N % 32 == 0, K % 16 == 0.
#include <cuda_bf16.h>
#include <mma.h>

__global__ void hgemm_wmma_naive(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                 const __nv_bfloat16* B, float beta, float* C) {
  using namespace nvcuda;
  // Block = 2x2 warps (blockDim = 64 x 2), covering a 32x32 tile of C.
  const int tileRow = blockIdx.y * 2 + threadIdx.y;
  const int tileCol = blockIdx.x * 2 + threadIdx.x / 32;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::fill_fragment(acc, 0.f);

  const __nv_bfloat16* Arow = A + (size_t)tileRow * 16 * K;
  const __nv_bfloat16* Bcol = B + tileCol * 16;
  for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a, Arow + k, K);
    wmma::load_matrix_sync(b, Bcol + (size_t)k * N, N);
    wmma::mma_sync(acc, a, b, acc);
  }

  float* Ctile = C + (size_t)tileRow * 16 * N + tileCol * 16;
  for (int i = 0; i < acc.num_elements; ++i) acc.x[i] *= alpha;
  if (beta != 0.f) {
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::load_matrix_sync(c, Ctile, N, wmma::mem_row_major);
    for (int i = 0; i < acc.num_elements; ++i) acc.x[i] += beta * c.x[i];
  }
  wmma::store_matrix_sync(Ctile, acc, N, wmma::mem_row_major);
}

inline void launch_hgemm_wmma_naive(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                    const __nv_bfloat16* B, float beta, float* C) {
  if (M % 32 || N % 32 || K % 16) {
    std::fprintf(stderr, "wmma_naive needs M%%32, N%%32, K%%16 == 0\n");
    std::exit(EXIT_FAILURE);
  }
  dim3 block(64, 2);
  dim3 grid(N / 32, M / 32);
  hgemm_wmma_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
