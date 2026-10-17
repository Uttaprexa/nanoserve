#pragma once
// Kernel 3: shared-memory tiling.
// Each block loads a TILE x TILE tile of A and of B into shared memory, then
// every thread reuses those values TILE times. This cuts global loads per
// FMA from 2 to 2/TILE. Handles arbitrary M, N, K by zero-padding edge tiles.
//
// Bank-conflict check (warp = one ty row, 32 consecutive tx):
//   As[ty][k] -> same address for the whole warp -> broadcast, no conflict.
//   Bs[k][tx] -> 32 consecutive floats -> 32 distinct banks, no conflict.
template <int TILE>
__global__ void sgemm_smem_tiling(int M, int N, int K, float alpha, const float* A,
                                  const float* B, float beta, float* C) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * TILE + ty;
  const int col = blockIdx.x * TILE + tx;

  float acc = 0.f;
  for (int t = 0; t < K; t += TILE) {
    As[ty][tx] = (row < M && t + tx < K) ? A[row * K + t + tx] : 0.f;
    Bs[ty][tx] = (t + ty < K && col < N) ? B[(t + ty) * N + col] : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

inline void launch_sgemm_smem_tiling(int M, int N, int K, float alpha, const float* A,
                                     const float* B, float beta, float* C) {
  constexpr int TILE = 32;
  dim3 block(TILE, TILE);
  dim3 grid(ceil_div(N, TILE), ceil_div(M, TILE));
  sgemm_smem_tiling<TILE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}
