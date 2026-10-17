#pragma once
// Tensor-core kernel 2: WMMA with shared-memory tiling.
// Block tile 128x128, BK = 32, 8 warps arranged 2 (M) x 4 (N).
// Each warp computes a 64x32 tile = 4x2 accumulator fragments of 16x16,
// so every A fragment is reused 2x and every B fragment 4x from registers,
// and every smem tile is reused by many warps.
//
// Global -> smem copies are 128-bit (8 bf16 per thread per load).
// Smem rows are padded by 8 bf16 (16 bytes) so the rows a fragment load
// touches start in different banks. Strides stay multiples of 8 elements and
// fragment pointers stay 32-byte aligned, as WMMA requires.
// REQUIRES: M % 128 == 0, N % 128 == 0, K % 32 == 0.
#include <cuda_bf16.h>
#include <mma.h>

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    hgemm_wmma_smem(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                    const __nv_bfloat16* B, float beta, float* C) {
  using namespace nvcuda;
  constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M;  // 64
  constexpr int WN = BN / WARPS_N;  // 32
  constexpr int FM = WM / 16;       // 4 fragments down
  constexpr int FN = WN / 16;       // 2 fragments across
  constexpr int PAD = 8;
  constexpr int LDA = BK + PAD;     // 40
  constexpr int LDB = BN + PAD;     // 136
  constexpr int A_VEC = BM * BK / 8;  // uint4 loads per A tile
  constexpr int B_VEC = BK * BN / 8;
  static_assert(A_VEC % NUM_THREADS == 0 && B_VEC % NUM_THREADS == 0, "even load split");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0, "16x16x16 fragments");

  __shared__ __align__(128) __nv_bfloat16 As[BM * LDA];
  __shared__ __align__(128) __nv_bfloat16 Bs[BK * LDB];

  const int warpIdx = threadIdx.x / 32;
  const int warpRow = warpIdx / WARPS_N;
  const int warpCol = warpIdx % WARPS_N;

  A += (size_t)blockIdx.y * BM * K;
  B += blockIdx.x * BN;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a[FM];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b[FN];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FM][FN];
#pragma unroll
  for (int i = 0; i < FM; ++i)
#pragma unroll
    for (int j = 0; j < FN; ++j) wmma::fill_fragment(acc[i][j], 0.f);

  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int p = 0; p < A_VEC / NUM_THREADS; ++p) {
      const int v = threadIdx.x + p * NUM_THREADS;
      const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
      *reinterpret_cast<uint4*>(&As[row * LDA + col]) =
          *reinterpret_cast<const uint4*>(&A[(size_t)row * K + col]);
    }
#pragma unroll
    for (int p = 0; p < B_VEC / NUM_THREADS; ++p) {
      const int v = threadIdx.x + p * NUM_THREADS;
      const int row = v / (BN / 8), col = (v % (BN / 8)) * 8;
      *reinterpret_cast<uint4*>(&Bs[row * LDB + col]) =
          *reinterpret_cast<const uint4*>(&B[(size_t)row * N + col]);
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
#pragma unroll
      for (int i = 0; i < FM; ++i)
        wmma::load_matrix_sync(a[i], &As[(warpRow * WM + i * 16) * LDA + kk], LDA);
#pragma unroll
      for (int j = 0; j < FN; ++j)
        wmma::load_matrix_sync(b[j], &Bs[kk * LDB + warpCol * WN + j * 16], LDB);
#pragma unroll
      for (int i = 0; i < FM; ++i)
#pragma unroll
        for (int j = 0; j < FN; ++j) wmma::mma_sync(acc[i][j], a[i], b[j], acc[i][j]);
    }
    __syncthreads();
    A += BK;
    B += (size_t)BK * N;
  }

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

inline void launch_hgemm_wmma_smem(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                   const __nv_bfloat16* B, float beta, float* C) {
  constexpr int BM = 128, BN = 128, BK = 32, WARPS_M = 2, WARPS_N = 4;
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "wmma_smem needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  dim3 block(WARPS_M * WARPS_N * 32);
  dim3 grid(N / BN, M / BM);
  hgemm_wmma_smem<BM, BN, BK, WARPS_M, WARPS_N><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
