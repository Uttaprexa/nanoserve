#pragma once
// Tensor-core kernel 3: kernel 2 + cp.async multi-stage pipeline.
//
// Kernel 2 serializes: load tile -> sync -> compute -> sync -> load next.
// Here global->smem copies use cp.async (Ampere+), which runs in the
// background without passing through registers. With STAGES buffers, the
// copy of tile k+STAGES-1 is in flight while the tensor cores work on tile k.
//
// Pipeline invariant (one commit group per iteration, even if empty):
//   iteration kt: wait until <= STAGES-2 groups are pending  -> tile kt landed
//                 __syncthreads()  -> tile kt visible to all warps AND every
//                                     warp finished computing tile kt-1
//                 issue copy of tile kt+STAGES-1 into the buffer of tile kt-1
//                 compute tile kt
//
// Uses dynamic shared memory: STAGES * (BM*(BK+8) + BK*(BN+8)) * 2 bytes
// (2 stages = 37.9 KB, 3 stages = 56.8 KB, which needs the opt-in above 48 KB).
// REQUIRES: M % 128 == 0, N % 128 == 0, K % 32 == 0, 16-byte aligned pointers.
#include <cuda_bf16.h>
#include <mma.h>

#include "ptx.cuh"

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    hgemm_wmma_async(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                     const __nv_bfloat16* B, float beta, float* C) {
  using namespace nvcuda;
  constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int FM = WM / 16, FN = WN / 16;
  constexpr int PAD = 8;
  constexpr int LDA = BK + PAD, LDB = BN + PAD;
  constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;  // elements per stage
  constexpr int A_VEC = BM * BK / 8, B_VEC = BK * BN / 8;
  static_assert(STAGES >= 2, "need at least double buffering");
  static_assert(A_VEC % NUM_THREADS == 0 && B_VEC % NUM_THREADS == 0, "even load split");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Bs = As + STAGES * A_STAGE;

  const int warpIdx = threadIdx.x / 32;
  const int warpRow = warpIdx / WARPS_N;
  const int warpCol = warpIdx % WARPS_N;

  A += (size_t)blockIdx.y * BM * K;
  B += blockIdx.x * BN;

  auto load_tile = [&](int stage, int bk) {
    __nv_bfloat16* as = As + stage * A_STAGE;
    __nv_bfloat16* bs = Bs + stage * B_STAGE;
#pragma unroll
    for (int p = 0; p < A_VEC / NUM_THREADS; ++p) {
      const int v = threadIdx.x + p * NUM_THREADS;
      const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
      cp_async_16(&as[row * LDA + col], &A[(size_t)row * K + bk + col]);
    }
#pragma unroll
    for (int p = 0; p < B_VEC / NUM_THREADS; ++p) {
      const int v = threadIdx.x + p * NUM_THREADS;
      const int row = v / (BN / 8), col = (v % (BN / 8)) * 8;
      cp_async_16(&bs[row * LDB + col], &B[(size_t)(bk + row) * N + col]);
    }
  };

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[FM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[FN];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  const int KT = K / BK;

  // Prologue: start the first STAGES-1 tiles.
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_tile(s, s * BK);
    cp_async_commit();
  }

  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();

    const int next = kt + STAGES - 1;
    if (next < KT) load_tile(next % STAGES, next * BK);
    cp_async_commit();

    const __nv_bfloat16* as = As + (kt % STAGES) * A_STAGE;
    const __nv_bfloat16* bs = Bs + (kt % STAGES) * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &as[(warpRow * WM + i * 16) * LDA + kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &bs[kk * LDB + warpCol * WN + j * 16], LDB);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
  }
  cp_async_wait<0>();  // drain (only empty groups remain, but keep it tidy)

#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) {
      float* Ctile = C + (size_t)(blockIdx.y * BM + warpRow * WM + i * 16) * N +
                     blockIdx.x * BN + warpCol * WN + j * 16;
      for (int e = 0; e < acc[i][j].num_elements; ++e) acc[i][j].x[e] *= alpha;
      if (beta != 0.f) {
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
        wmma::load_matrix_sync(c, Ctile, N, wmma::mem_row_major);
        for (int e = 0; e < c.num_elements; ++e) acc[i][j].x[e] += beta * c.x[e];
      }
      wmma::store_matrix_sync(Ctile, acc[i][j], N, wmma::mem_row_major);
    }
}

template <int STAGES>
inline void launch_hgemm_wmma_async(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                    const __nv_bfloat16* B, float beta, float* C) {
  constexpr int BM = 128, BN = 128, BK = 32, WARPS_M = 2, WARPS_N = 4;
  constexpr int SMEM = STAGES * (BM * (BK + 8) + BK * (BN + 8)) * (int)sizeof(__nv_bfloat16);
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "wmma_async needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  auto kernel = hgemm_wmma_async<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  static bool configured = false;  // opt in to >48 KB dynamic smem once, outside timing
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  dim3 block(WARPS_M * WARPS_N * 32);
  dim3 grid(N / BN, M / BM);
  kernel<<<grid, block, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

inline void launch_hgemm_wmma_async2(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                     const __nv_bfloat16* B, float beta, float* C) {
  launch_hgemm_wmma_async<2>(M, N, K, alpha, A, B, beta, C);
}
inline void launch_hgemm_wmma_async3(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                     const __nv_bfloat16* B, float beta, float* C) {
  launch_hgemm_wmma_async<3>(M, N, K, alpha, A, B, beta, C);
}
