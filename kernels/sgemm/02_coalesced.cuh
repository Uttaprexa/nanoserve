#pragma once
// Kernel 2: global-memory coalescing.
// Same math as kernel 1, but threadIdx.x now indexes COLUMNS. Consecutive
// threads in a warp read consecutive elements of B and write consecutive
// elements of C, so each warp's access becomes a few wide transactions.
// The A[row*K + k] load is the same address for the whole warp (broadcast).
__global__ void sgemm_coalesced(int M, int N, int K, float alpha, const float* A,
                                const float* B, float beta, float* C) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.f;
    for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
  }
}

inline void launch_sgemm_coalesced(int M, int N, int K, float alpha, const float* A,
                                   const float* B, float beta, float* C) {
  dim3 block(32, 32);
  dim3 grid(ceil_div(N, 32), ceil_div(M, 32));
  sgemm_coalesced<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
