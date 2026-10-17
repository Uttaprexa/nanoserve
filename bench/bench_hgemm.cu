// BF16 tensor-core GEMM benchmark.
// C (fp32) = A (bf16) @ B (bf16), fp32 accumulation, all row-major.
// Baseline: cuBLAS cublasGemmEx on tensor cores with the same types.
// Writes results/hgemm_<gpu>.csv.
//
// Usage: ./build/bench_hgemm [size ...]        (default 1024 2048 4096)
//        ./build/bench_hgemm --only 2 4096
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "common.cuh"
#include "hgemm/01_wmma_naive.cuh"
#include "hgemm/02_wmma_smem.cuh"
#include "hgemm/03_wmma_async.cuh"
#include "hgemm/04_mma_ldmatrix.cuh"

using bf16 = __nv_bfloat16;
using LaunchFn = void (*)(int, int, int, float, const bf16*, const bf16*, float, float*);

struct Kernel {
  int id;
  const char* name;
  LaunchFn launch;
};

static const Kernel kKernels[] = {
    {1, "wmma_naive", launch_hgemm_wmma_naive},
    {2, "wmma_smem", launch_hgemm_wmma_smem},
    {3, "wmma_async2", launch_hgemm_wmma_async2},
    {4, "wmma_async3", launch_hgemm_wmma_async3},
    {5, "mma_w64x32", launch_hgemm_mma_w64x32},
    {6, "mma_w64x64", launch_hgemm_mma_w64x64},
};

// Row-major C = A @ B via column-major cuBLAS: C^T = B^T @ A^T.
static void cublas_hgemm(cublasHandle_t h, int M, int N, int K, float alpha, const bf16* A,
                         const bf16* B, float beta, float* C) {
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, A,
                            CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

static float max_rel_error(const std::vector<float>& out, const std::vector<float>& ref) {
  float worst = 0.f;
  for (size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) return INFINITY;
    worst = std::max(worst, std::fabs(out[i] - ref[i]) / (std::fabs(ref[i]) + 1.f));
  }
  return worst;
}

static std::string sanitize(const char* s) {
  std::string out(s);
  for (char& c : out)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return out;
}

int main(int argc, char** argv) {
  int only = 0;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = std::atoi(argv[++i]);
    else sizes.push_back(std::atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {1024, 2048, 4096};

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs\nBF16 inputs, FP32 accumulate + output\n\n", prop.name,
              prop.major, prop.minor, prop.multiProcessorCount);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  const std::string csv_path = "results/hgemm_" + sanitize(prop.name) + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,kernel_id,kernel,M,N,K,time_ms,gflops,pct_cublas,max_rel_err\n");

  const float alpha = 1.f, beta = 0.f;
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int S : sizes) {
    const int M = S, N = S, K = S;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<bf16> hA(nA), hB(nB);
    for (auto& x : hA) x = __float2bfloat16(dist(rng));
    for (auto& x : hB) x = __float2bfloat16(dist(rng));
    std::vector<float> hRef(nC), hOut(nC);

    bf16 *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(bf16), cudaMemcpyHostToDevice));

    const double flops = 2.0 * M * N * K;
    const int iters = S <= 1024 ? 100 : 20;

    CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));
    cublas_hgemm(handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    const float cublas_ms =
        time_kernel_ms([&] { cublas_hgemm(handle, M, N, K, alpha, dA, dB, beta, dC); }, 3, iters);
    const double cublas_gflops = flops / (cublas_ms * 1e6);

    std::printf("M=N=K=%d\n  %-14s %9.3f ms %9.1f TFLOP/s  100.0%%\n", S, "cuBLAS", cublas_ms,
                cublas_gflops / 1e3);
    std::fprintf(csv, "%s,0,cublas,%d,%d,%d,%.5f,%.2f,100.0,0\n", prop.name, M, N, K, cublas_ms,
                 cublas_gflops);

    for (const Kernel& k : kKernels) {
      if (only && k.id != only) continue;
      CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));
      k.launch(M, N, K, alpha, dA, dB, beta, dC);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaMemcpy(hOut.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
      const float err = max_rel_error(hOut, hRef);
      const bool ok = err < 1e-3f;

      const float ms = time_kernel_ms([&] { k.launch(M, N, K, alpha, dA, dB, beta, dC); }, 2, iters);
      const double gflops = flops / (ms * 1e6);
      const double pct = 100.0 * gflops / cublas_gflops;
      std::printf("  %-14s %9.3f ms %9.1f TFLOP/s %6.1f%%  err=%.1e %s\n", k.name, ms,
                  gflops / 1e3, pct, err, ok ? "" : "  <-- WRONG RESULT");
      std::fprintf(csv, "%s,%d,%s,%d,%d,%d,%.5f,%.2f,%.2f,%.3e\n", prop.name, k.id, k.name, M, N,
                   K, ms, gflops, pct, err);
    }
    std::printf("\n");
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
  }

  std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  std::printf("Wrote %s\n", csv_path.c_str());
  return 0;
}
