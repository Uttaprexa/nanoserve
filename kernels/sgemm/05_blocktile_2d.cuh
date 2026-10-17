#pragma once
// Kernel 5: 2D register blocktiling.
// Kernel 4 computes a TM x 1 column per thread. Here each thread computes a
// TM x TN tile as an outer product: per k step it loads TM values of A and TN
// values of B from smem into registers, then does TM*TN FMAs. That is
// 64 FMAs per 16 smem loads (4:1), versus 8:9 in kernel 4.
//
// Block computes BM x BN of C with (BM*BN)/(TM*TN) = 256 threads; each thread
// loads several A/B elements per tile (strided loop).
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_blocktile_2d(int M, int N, int K, float alpha, const float* A,
                       const float* B, float beta, float* C) {
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
  constexpr int STRIDE_A = NUM_THREADS / BK;  // rows of A loaded per pass
  constexpr int STRIDE_B = NUM_THREADS / BN;  // rows of B loaded per pass
  static_assert(NUM_THREADS % BK == 0 && BM % STRIDE_A == 0, "A load tiling");
  static_assert(NUM_THREADS % BN == 0 && BK % STRIDE_B == 0, "B load tiling");

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  // This thread's TM x TN output tile.
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  const int innerRowA = threadIdx.x / BK, innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN, innerColB = threadIdx.x % BN;

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
#pragma unroll
    for (int off = 0; off < BM; off += STRIDE_A)
      As[(innerRowA + off) * BK + innerColA] = A[(innerRowA + off) * K + innerColA];
#pragma unroll
    for (int off = 0; off < BK; off += STRIDE_B)
      Bs[(innerRowB + off) * BN + innerColB] = B[(innerRowB + off) * N + innerColB];
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[(threadRow * TM + i) * BK + k];
#pragma unroll
      for (int j = 0; j < TN; ++j) regN[j] = Bs[k * BN + threadCol * TN + j];
#pragma unroll
      for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i * TN + j] += regM[i] * regN[j];
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < TM; ++i)
#pragma unroll
    for (int j = 0; j < TN; ++j) {
      float* c = &C[(threadRow * TM + i) * N + threadCol * TN + j];
      *c = alpha * acc[i * TN + j] + beta * (*c);
    }
}

inline void launch_sgemm_blocktile_2d(int M, int N, int K, float alpha, const float* A,
                                      const float* B, float beta, float* C) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "blocktile_2d needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(N / BN, M / BM);
  sgemm_blocktile_2d<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
