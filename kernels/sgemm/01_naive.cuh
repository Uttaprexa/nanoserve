#pragma once
// Kernel 1: naive SGEMM. One thread computes one element of C.
// Row-major: C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C.
//
// Deliberately slow: threadIdx.x indexes ROWS, so the 32 threads of a warp
// read 32 different rows of A (stride K) and write 32 different rows of C.
// Global memory accesses are uncoalesced. Kernel 2 fixes exactly this.
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float* A,
                            const float* B, float beta, float* C) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void launch_sgemm_naive(int M, int N, int K, float alpha, const float* A,
                               const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(M, 32), ceil_div(N, 32));
  sgemm_naive<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
