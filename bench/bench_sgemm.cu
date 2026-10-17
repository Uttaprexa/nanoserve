// SGEMM benchmark: checks every kernel against cuBLAS, then reports
// time, GFLOP/s and % of cuBLAS. Writes results/sgemm_<gpu>.csv.
//
// Usage: ./build/bench_sgemm [size ...]        (square M=N=K, default 256..4096)
//        ./build/bench_sgemm --only 4 4096     (run one kernel, handy under ncu)
//        ./build/bench_sgemm --beta 1 2048     (beta != 0 forces every kernel to read C)
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common.cuh"
#include "sgemm/01_naive.cuh"
#include "sgemm/02_coalesced.cuh"
#include "sgemm/03_smem_tiling.cuh"
#include "sgemm/04_blocktile_1d.cuh"
#include "sgemm/05_blocktile_2d.cuh"
#include "sgemm/06_vectorized.cuh"
#include "sgemm/07_warptiling.cuh"

using LaunchFn = void (*)(int, int, int, float, const float*, const float*, float, float*);

struct Kernel {
  int id;
  const char* name;
  LaunchFn launch;
};

static const Kernel kKernels[] = {
    {1, "naive", launch_sgemm_naive},
    {2, "coalesced", launch_sgemm_coalesced},
    {3, "smem_tiling", launch_sgemm_smem_tiling},
    {4, "blocktile_1d", launch_sgemm_blocktile_1d},
    {5, "blocktile_2d", launch_sgemm_blocktile_2d},
    {6, "vectorized", launch_sgemm_vectorized},
    {7, "warptiling", launch_sgemm_warptiling},
    {8, "warptile_3blk", launch_sgemm_warptiling_3blk},
};

// Row-major C = A @ B via column-major cuBLAS: compute C^T = B^T @ A^T.
static void cublas_sgemm(cublasHandle_t h, int M, int N, int K, float alpha, const float* A,
                         const float* B, float beta, float* C) {
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
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
  float beta = 0.f;
  std::vector<int> sizes;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--only") && i + 1 < argc) only = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--beta") && i + 1 < argc) beta = std::atof(argv[++i]);
    else sizes.push_back(std::atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {256, 512, 1024, 2048, 4096};

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs | %.1f GB\n\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, prop.totalGlobalMem / 1e9);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  // Keep cuBLAS in true FP32 (no TF32 tensor cores) so the comparison is fair.
  CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

  const std::string csv_path =
      "results/sgemm_" + sanitize(prop.name) + (beta != 0.f ? "_beta1" : "") + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,kernel_id,kernel,M,N,K,time_ms,gflops,pct_cublas,max_rel_err\n");

  const float alpha = 1.f;
  std::printf("alpha = %.1f, beta = %.1f\n\n", alpha, beta);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);

  for (int S : sizes) {
    const int M = S, N = S, K = S;
    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    std::vector<float> hA(nA), hB(nB), hRef(nC), hOut(nC);
    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

    const double flops = 2.0 * M * N * K;
    const int iters = S <= 1024 ? 50 : 10;

    // Reference + baseline timing.
    CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(float)));
    cublas_sgemm(handle, M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    const float cublas_ms =
        time_kernel_ms([&] { cublas_sgemm(handle, M, N, K, alpha, dA, dB, beta, dC); }, 2, iters);
    const double cublas_gflops = flops / (cublas_ms * 1e6);

    std::printf("M=N=K=%d\n  %-14s %9.3f ms %9.1f GFLOP/s  100.0%%\n", S, "cuBLAS", cublas_ms,
                cublas_gflops);
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

      // The naive kernel is painfully slow at 4096; fewer iterations keeps runs short.
      const int it = (k.id == 1 && S >= 2048) ? 2 : iters;
      const float ms = time_kernel_ms([&] { k.launch(M, N, K, alpha, dA, dB, beta, dC); }, 1, it);
      const double gflops = flops / (ms * 1e6);
      const double pct = 100.0 * gflops / cublas_gflops;

      std::printf("  %-14s %9.3f ms %9.1f GFLOP/s %6.1f%%  err=%.1e %s\n", k.name, ms, gflops, pct,
                  err, ok ? "" : "  <-- WRONG RESULT");
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
