#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                   __FILE__, __LINE__);                                         \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

#define CUBLAS_CHECK(call)                                                      \
  do {                                                                          \
    cublasStatus_t st_ = (call);                                                \
    if (st_ != CUBLAS_STATUS_SUCCESS) {                                         \
      std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st_, __FILE__,     \
                   __LINE__);                                                   \
      std::exit(EXIT_FAILURE);                                                  \
    }                                                                           \
  } while (0)

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Times `fn` with CUDA events: `warmup` untimed runs, then the mean of `iters` runs (ms).
template <typename F>
float time_kernel_ms(F fn, int warmup = 2, int iters = 10) {
  for (int i = 0; i < warmup; ++i) fn();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) fn();
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  CUDA_CHECK(cudaGetLastError());

  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return ms / iters;
}
