#pragma once
// Tensor-core kernel 4: raw mma.sync + ldmatrix + cp.async pipeline.
//
// The SASS of kernels 2-3 showed WMMA fragment loads from shared memory
// compiling to generic 32-bit LD.E (48 per tile), with no LDSM. Here we drop
// WMMA and drive the tensor cores directly:
//   * ldmatrix.x4        loads a 16x16 A fragment (four 8x8 tiles) per instruction
//   * ldmatrix.x4.trans  loads B fragments for two n8 tiles from row-major B
//   * mma.sync m16n8k16  bf16 inputs, fp32 accumulators in registers
// Global->smem copies use the same cp.async pipeline as kernel 3.
//
// Smem rows are padded by 8 bf16 (16 B). For ldmatrix, each 8-row phase reads
// eight 16-byte rows; the padded strides (80 B for A, 272 B for B) put those
// rows in eight different 16-byte bank groups, so loads are conflict-free.
//
// Fragment layouts (PTX ISA, m16n8k16, g = lane/4, t = lane%4):
//   A regs: {r0: (g, 2t..2t+1), r1: (g+8, 2t..), r2: (g, 2t+8..), r3: (g+8, 2t+8..)}
//   B regs: {r0: (k=2t..2t+1, n=g), r1: (k=2t+8.., n=g)}
//   C regs: {c0,c1: (g, 2t..2t+1), c2,c3: (g+8, 2t..2t+1)}
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0, 16-byte aligned pointers.
#include <cuda_bf16.h>

#include "ptx.cuh"

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    hgemm_mma(int M, int N, int K, float alpha, const __nv_bfloat16* A, const __nv_bfloat16* B,
              float beta, float* C) {
  constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;  // warp tile
  constexpr int MI = WM / 16;                          // m16 tiles per warp
  constexpr int NI = WN / 8;                           // n8 tiles per warp
  constexpr int PAD = 8;
  constexpr int LDA = BK + PAD, LDB = BN + PAD;
  constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;
  constexpr int A_VEC = BM * BK / 8, B_VEC = BK * BN / 8;
  static_assert(STAGES >= 2, "need at least double buffering");
  static_assert(WM % 16 == 0 && WN % 16 == 0 && BK % 16 == 0, "tile shapes");
  static_assert(NI % 2 == 0, "ldmatrix.x4.trans loads two n8 tiles at a time");
  static_assert(A_VEC % NUM_THREADS == 0 && B_VEC % NUM_THREADS == 0, "even load split");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Bs = As + STAGES * A_STAGE;

  const int lane = threadIdx.x % 32;
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

  // Per-lane ldmatrix row offsets (constant across the K loop).
  // Matrix index q = lane/8, row-in-matrix r = lane%8.
  //   A (x4):       q -> (row + 8*(q%2), col + 8*(q/2))  => r0..r3 match the A layout
  //   B (x4.trans): q -> (k + 8*(q%2),  n + 8*(q/2))     => {b0,b1} of n-tile 0, then n-tile 1
  const int q = lane / 8, r = lane % 8;
  const int a_row = r + 8 * (q % 2), a_col = 8 * (q / 2);
  const int b_k = r + 8 * (q % 2), b_n = 8 * (q / 2);

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

  const int KT = K / BK;
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
      uint32_t afrag[MI][4];
      uint32_t bfrag[NI][2];
#pragma unroll
      for (int i = 0; i < MI; ++i)
        ldmatrix_x4(afrag[i], &as[(warpRow * WM + i * 16 + a_row) * LDA + kk + a_col]);
#pragma unroll
      for (int j = 0; j < NI; j += 2) {
        uint32_t t4[4];
        ldmatrix_x4_trans(t4, &bs[(kk + b_k) * LDB + warpCol * WN + j * 8 + b_n]);
        bfrag[j][0] = t4[0];
        bfrag[j][1] = t4[1];
        bfrag[j + 1][0] = t4[2];
        bfrag[j + 1][1] = t4[3];
      }
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) mma_bf16_16816(acc[i][j], afrag[i], bfrag[j][0], bfrag[j][1]);
    }
  }
  cp_async_wait<0>();

  // Epilogue: each lane owns (g, 2t..2t+1) and (g+8, 2t..2t+1) of every 16x8 tile.
  const int g = lane / 4, t = lane % 4;
  const bool read_c = (beta != 0.f);
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int row = blockIdx.y * BM + warpRow * WM + i * 16 + g;
      const int col = blockIdx.x * BN + warpCol * WN + j * 8 + 2 * t;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        float2* cp = reinterpret_cast<float2*>(&C[(size_t)(row + 8 * h) * N + col]);
        float2 c = read_c ? *cp : make_float2(0.f, 0.f);
        c.x = alpha * acc[i][j][2 * h + 0] + beta * c.x;
        c.y = alpha * acc[i][j][2 * h + 1] + beta * c.y;
        *cp = c;
      }
    }
}

template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES>
inline void launch_hgemm_mma_cfg(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                 const __nv_bfloat16* B, float beta, float* C) {
  constexpr int SMEM = STAGES * (BM * (BK + 8) + BK * (BN + 8)) * (int)sizeof(__nv_bfloat16);
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "hgemm_mma needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  auto kernel = hgemm_mma<BM, BN, BK, WARPS_M, WARPS_N, STAGES>;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  dim3 block(WARPS_M * WARPS_N * 32);
  dim3 grid(N / BN, M / BM);
  kernel<<<grid, block, SMEM>>>(M, N, K, alpha, A, B, beta, C);
}

// Config A: same tiling as the WMMA kernels (8 warps, 64x32 warp tiles) -> direct comparison.
inline void launch_hgemm_mma_w64x32(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                    const __nv_bfloat16* B, float beta, float* C) {
  launch_hgemm_mma_cfg<128, 128, 32, 2, 4, 2>(M, N, K, alpha, A, B, beta, C);
}
// Config B: 4 warps with 64x64 warp tiles -> more register reuse per ldmatrix, fewer warps.
inline void launch_hgemm_mma_w64x64(int M, int N, int K, float alpha, const __nv_bfloat16* A,
                                    const __nv_bfloat16* B, float beta, float* C) {
  launch_hgemm_mma_cfg<128, 128, 32, 2, 2, 2>(M, N, K, alpha, A, B, beta, C);
}
