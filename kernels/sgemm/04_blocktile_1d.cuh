#pragma once
// Kernel 4: 1D register blocktiling.
// Kernel 3 is limited by shared-memory traffic: every FMA needs two smem loads.
// Here each thread computes TM outputs stacked in one column of C. The B value
// is loaded from smem once and reused TM times from a register, raising
// arithmetic intensity per smem load.
//
// Block computes a BM x BN tile of C with (BM*BN)/TM threads.
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0 (checked by the launcher).
template <int BM, int BN, int BK, int TM>
__global__ void __launch_bounds__((BM * BN) / TM)
    sgemm_blocktile_1d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  constexpr int NUM_THREADS = (BM * BN) / TM;
  static_assert(BM * BK == NUM_THREADS, "each thread loads exactly one A element");
  static_assert(BK * BN == NUM_THREADS, "each thread loads exactly one B element");

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Advance pointers to this block's tile.
  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // Which outputs this thread owns: rows [threadRow*TM, threadRow*TM+TM), column threadCol.
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // Which element this thread loads into smem (coalesced along the fast dimension).
  const int innerColA = threadIdx.x % BK, innerRowA = threadIdx.x / BK;
  const int innerColB = threadIdx.x % BN, innerRowB = threadIdx.x / BN;

  float acc[TM] = {0.f};

  for (int bk = 0; bk < K; bk += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      const float b = Bs[k * BN + threadCol];
#pragma unroll
      for (int i = 0; i < TM; ++i) acc[i] += As[(threadRow * TM + i) * BK + k] * b;
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i) {
    float* c = &C[(threadRow * TM + i) * N + threadCol];
    *c = alpha * acc[i] + beta * (*c);
  }
}

inline void launch_sgemm_blocktile_1d(int M, int N, int K, float alpha, const float* A,
                                      const float* B, float beta, float* C) {
  constexpr int BM = 64, BN = 64, BK = 8, TM = 8;
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "blocktile_1d needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  dim3 block((BM * BN) / TM);
  dim3 grid(N / BN, M / BM);
  sgemm_blocktile_1d<BM, BN, BK, TM><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
