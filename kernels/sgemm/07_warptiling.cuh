#pragma once
// Kernel 7: warptiling.
// Adds a warp level between the block tile and the thread tile:
//
//   block tile  BM x BN   (128 x 128, 4 warps)
//   warp tile   WM x WN   (64 x 64, one per warp)
//   warp sub-tile WSUBM x WSUBN, iterated WMITER x WNITER times
//   thread tile TM x TN inside each sub-tile
//
// Why: the 32 threads of a warp now read a compact region of smem, so their
// LDS addresses overlap heavily (broadcasts), and each thread's registers hold
// a larger, better-reused slice of A and B. It also mirrors how tensor-core
// GEMMs are organized, which is where Phase 1 goes next.
//
// Epilogue: when beta == 0 we never read C (saves an M*N global read).
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0, 16-byte aligned pointers.
// MIN_BLOCKS > 1 asks the compiler to cap registers so that many blocks fit per SM
// (used by the warptile_3blk variant to test the wave-quantization hypothesis).
template <int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN, int NUM_THREADS,
          int MIN_BLOCKS = 1>
__global__ void __launch_bounds__(NUM_THREADS, MIN_BLOCKS)
    sgemm_warptiling(int M, int N, int K, float alpha, const float* A, const float* B,
                     float beta, float* C) {
  constexpr int WARPSIZE = 32;
  constexpr int WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  static_assert((BN / WN) * (BM / WM) == NUM_THREADS / WARPSIZE, "one warp per warp tile");
  static_assert(WMITER * WSUBM == WM && WNITER * WSUBN == WN, "warp tile splits evenly");
  static_assert((WSUBN / TN) * (WSUBM / TM) == WARPSIZE, "32 threads cover a sub-tile");
  static_assert((BM * BK) % (4 * NUM_THREADS) == 0, "A loads as float4");
  static_assert((BN * BK) % (4 * NUM_THREADS) == 0, "B loads as float4");
  static_assert(TN % 4 == 0, "float4 epilogue");

  const int warpIdx = threadIdx.x / WARPSIZE;
  const int warpCol = warpIdx % (BN / WN);
  const int warpRow = warpIdx / (BN / WN);
  const int laneIdx = threadIdx.x % WARPSIZE;
  const int threadColInWarp = laneIdx % (WSUBN / TN);
  const int threadRowInWarp = laneIdx / (WSUBN / TN);

  __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  // C points at this warp's 64x64 output tile.
  C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

  // Global -> smem load mapping (each thread moves float4s).
  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  constexpr int ROW_STRIDE_A = (NUM_THREADS * 4) / BK;
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);
  constexpr int ROW_STRIDE_B = NUM_THREADS / (BN / 4);

  float acc[WMITER * TM * WNITER * TN] = {0.f};
  float regM[WMITER * TM];
  float regN[WNITER * TN];

  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += ROW_STRIDE_A) {
      const float4 a =
          reinterpret_cast<const float4*>(&A[(innerRowA + off) * K + innerColA * 4])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA + off] = a.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + off] = a.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + off] = a.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + off] = a.w;
    }
#pragma unroll
    for (int off = 0; off < BK; off += ROW_STRIDE_B) {
      reinterpret_cast<float4*>(&Bs[(innerRowB + off) * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4*>(&B[(innerRowB + off) * N + innerColB * 4])[0];
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int ws = 0; ws < WMITER; ++ws)
#pragma unroll
        for (int i = 0; i < TM; ++i)
          regM[ws * TM + i] =
              As[k * BM + warpRow * WM + ws * WSUBM + threadRowInWarp * TM + i];
#pragma unroll
      for (int ws = 0; ws < WNITER; ++ws)
#pragma unroll
        for (int j = 0; j < TN; ++j)
          regN[ws * TN + j] =
              Bs[k * BN + warpCol * WN + ws * WSUBN + threadColInWarp * TN + j];

#pragma unroll
      for (int wm = 0; wm < WMITER; ++wm)
#pragma unroll
        for (int wn = 0; wn < WNITER; ++wn)
#pragma unroll
          for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j)
              acc[(wm * TM + i) * (WNITER * TN) + wn * TN + j] +=
                  regM[wm * TM + i] * regN[wn * TN + j];
    }
    A += BK;
    B += BK * N;
    __syncthreads();
  }

  const bool read_c = (beta != 0.f);
#pragma unroll
  for (int wm = 0; wm < WMITER; ++wm)
#pragma unroll
    for (int wn = 0; wn < WNITER; ++wn) {
      float* Csub = C + (wm * WSUBM) * N + wn * WSUBN;
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
          float4* cp = reinterpret_cast<float4*>(
              &Csub[(threadRowInWarp * TM + i) * N + threadColInWarp * TN + j]);
          const int r = (wm * TM + i) * (WNITER * TN) + wn * TN + j;
          float4 c = read_c ? *cp : make_float4(0.f, 0.f, 0.f, 0.f);
          c.x = alpha * acc[r + 0] + beta * c.x;
          c.y = alpha * acc[r + 1] + beta * c.y;
          c.z = alpha * acc[r + 2] + beta * c.z;
          c.w = alpha * acc[r + 3] + beta * c.w;
          *cp = c;
        }
    }
}

template <int MIN_BLOCKS>
inline void launch_sgemm_warptiling_impl(int M, int N, int K, float alpha, const float* A,
                                         const float* B, float beta, float* C) {
  constexpr int NUM_THREADS = 128;
  constexpr int BM = 128, BN = 128, BK = 16;
  constexpr int WM = 64, WN = 64, WNITER = 4;
  constexpr int TM = 8, TN = 4;
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "warptiling needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  dim3 block(NUM_THREADS);
  dim3 grid(N / BN, M / BM);
  sgemm_warptiling<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS, MIN_BLOCKS>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

inline void launch_sgemm_warptiling(int M, int N, int K, float alpha, const float* A,
                                    const float* B, float beta, float* C) {
  launch_sgemm_warptiling_impl<1>(M, N, K, alpha, A, B, beta, C);
}

// Same kernel, registers capped at 65536 / (3 * 128) = 170 so 3 blocks fit per SM.
inline void launch_sgemm_warptiling_3blk(int M, int N, int K, float alpha, const float* A,
                                         const float* B, float beta, float* C) {
  launch_sgemm_warptiling_impl<3>(M, N, K, alpha, A, B, beta, C);
}
