#pragma once
// Kernel 6: vectorized memory access + transposed A tile.
// Two changes on top of kernel 5:
//  1. Global loads/stores use float4 (128-bit): 4x fewer load instructions.
//  2. A is stored TRANSPOSED in smem (As[k][m]). The TM values a thread needs
//     for one k are then contiguous, so the inner loop can use 128-bit LDS.128
//     for both A and B instead of 32-bit loads. Check the SASS to confirm.
// The transposed store costs a 2-way bank conflict on the smem write, which is
// cheap because it happens once per tile, not once per FMA.
//
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0, 16-byte aligned pointers.
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                     const float* B, float beta, float* C) {
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
  static_assert(BM * BK / 4 == NUM_THREADS, "one float4 of A per thread per tile");
  static_assert(BK * BN / 4 == NUM_THREADS, "one float4 of B per thread per tile");
  static_assert(TN % 4 == 0, "float4 epilogue");

  __shared__ float As[BK * BM];  // transposed: As[k * BM + m]
  __shared__ float Bs[BK * BN];

  A += blockIdx.y * BM * K;
  B += blockIdx.x * BN;
  C += blockIdx.y * BM * N + blockIdx.x * BN;

  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  const int innerRowA = threadIdx.x / (BK / 4), innerColA = threadIdx.x % (BK / 4);
  const int innerRowB = threadIdx.x / (BN / 4), innerColB = threadIdx.x % (BN / 4);

  float acc[TM * TN] = {0.f};
  float regM[TM], regN[TN];

  for (int bk = 0; bk < K; bk += BK) {
    const float4 a = reinterpret_cast<const float4*>(&A[innerRowA * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA] = a.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = a.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = a.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = a.w;

    reinterpret_cast<float4*>(&Bs[innerRowB * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4*>(&B[innerRowB * N + innerColB * 4])[0];
    __syncthreads();
    A += BK;
    B += BK * N;

#pragma unroll
    for (int k = 0; k < BK; ++k) {
#pragma unroll
      for (int i = 0; i < TM; ++i) regM[i] = As[k * BM + threadRow * TM + i];
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
    for (int j = 0; j < TN; j += 4) {
      float4* cp = reinterpret_cast<float4*>(&C[(threadRow * TM + i) * N + threadCol * TN + j]);
      float4 c = *cp;
      c.x = alpha * acc[i * TN + j + 0] + beta * c.x;
      c.y = alpha * acc[i * TN + j + 1] + beta * c.y;
      c.z = alpha * acc[i * TN + j + 2] + beta * c.z;
      c.w = alpha * acc[i * TN + j + 3] + beta * c.w;
      *cp = c;
    }
}

inline void launch_sgemm_vectorized(int M, int N, int K, float alpha, const float* A,
                                    const float* B, float beta, float* C) {
  constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
  if (M % BM || N % BN || K % BK) {
    std::fprintf(stderr, "vectorized needs M%%%d, N%%%d, K%%%d == 0\n", BM, BN, BK);
    std::exit(EXIT_FAILURE);
  }
  dim3 block((BM * BN) / (TM * TN));
  dim3 grid(N / BN, M / BM);
  sgemm_vectorized<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
