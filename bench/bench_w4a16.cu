// Phase 4: W4A16 (4-bit group-quantized weights, bf16 activations) vs cuBLAS bf16.
//   [1] correctness vs a double-precision CPU reference on the dequantized weights,
//       plus the quantization error vs the original bf16 weights
//   [2] decode (M = 1, 8): GEMV, memory-bound -> time, GB/s, % of copy ceiling
//   [3] prefill (M = 512, 2048): tensor-core GEMM, compute-bound -> TFLOP/s
// Layer shapes from Llama-3-8B (N x K): qkv 6144x4096, o 4096x4096, gate_up 28672x4096,
// down 4096x14336. Weights rotate through ~192 MB of copies so timing reads DRAM, not L2.
// Writes results/w4a16_<gpu>.csv.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "common.cuh"
#include "quant/w4a16.cuh"

using bf16 = __nv_bfloat16;
static constexpr int G = 128;

static std::string sanitize(const char* s) {
  std::string out(s);
  for (char& c : out)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return out;
}

__global__ void fill_kernel(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const float u = (h & 0xFFFFFF) / 16777216.f * 2.f - 1.f;
    p[i] = __float2bfloat16(u * 1.732f * scale);
  }
}

struct Layer {  // one weight matrix in bf16 and W4 form, plus rotating copies for timing
  int N, K;
  size_t nW, nWq, nS;
  std::vector<bf16*> Wb;
  std::vector<uint32_t*> Wq;
  std::vector<bf16*> S;
  Layer(int N_, int K_, uint32_t seed, bool rotate) : N(N_), K(K_) {
    nW = (size_t)N * K;
    nWq = nW / 8;
    nS = nW / G;
    const size_t target = rotate ? (192ull << 20) : 1;
    const int copies_b = static_cast<int>(std::max<size_t>(1, target / (nW * 2) + 1));
    const int copies_q = static_cast<int>(std::max<size_t>(1, target / (nWq * 4 + nS * 2) + 1));
    Wb.resize(copies_b);
    Wq.resize(copies_q);
    S.resize(copies_q);
    for (auto& p : Wb) CUDA_CHECK(cudaMalloc(&p, nW * sizeof(bf16)));
    for (int i = 0; i < copies_q; ++i) {
      CUDA_CHECK(cudaMalloc(&Wq[i], nWq * sizeof(uint32_t)));
      CUDA_CHECK(cudaMalloc(&S[i], nS * sizeof(bf16)));
    }
    fill_kernel<<<grid_for(nW), 256>>>(Wb[0], nW, seed, 0.05f);
    quantize_w4_kernel<<<grid_for(nS), 256>>>(Wb[0], Wq[0], S[0], N, K, G);
    for (int i = 1; i < copies_b; ++i)
      CUDA_CHECK(cudaMemcpy(Wb[i], Wb[0], nW * sizeof(bf16), cudaMemcpyDeviceToDevice));
    for (int i = 1; i < copies_q; ++i) {
      CUDA_CHECK(cudaMemcpy(Wq[i], Wq[0], nWq * sizeof(uint32_t), cudaMemcpyDeviceToDevice));
      CUDA_CHECK(cudaMemcpy(S[i], S[0], nS * sizeof(bf16), cudaMemcpyDeviceToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~Layer() {
    for (auto p : Wb) cudaFree(p);
    for (auto p : Wq) cudaFree(p);
    for (auto p : S) cudaFree(p);
  }
};

// Row-major Y[M,N] = X[M,K] @ W[N,K]^T through column-major cuBLAS: Y^T = W (op T) * X^T.
static void cublas_linear(cublasHandle_t h, const bf16* X, const bf16* W, float* Y, int M, int N,
                          int K) {
  const float one = 1.f, zero = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, X,
                            CUDA_R_16BF, K, &zero, Y, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

static std::vector<float> to_host(const float* d, size_t n) {
  std::vector<float> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
  return h;
}
static float max_rel_err(const std::vector<float>& a, const std::vector<float>& ref) {
  float scale = 0.f;
  for (float r : ref) scale = std::max(scale, std::fabs(r));
  float w = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i])) return INFINITY;
    w = std::max(w, std::fabs(a[i] - ref[i]) / (scale + 1e-6f));
  }
  return w;  // max error relative to the largest output magnitude
}
static float rel_fro(const std::vector<float>& a, const std::vector<float>& ref) {
  double num = 0, den = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    num += (double)(a[i] - ref[i]) * (a[i] - ref[i]);
    den += (double)ref[i] * ref[i];
  }
  return static_cast<float>(std::sqrt(num / den));
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs | group size %d\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount, G);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  const float tol = 1e-2f;

  // bandwidth ceiling (same copy kernel as Phase 3)
  double peak_gbs;
  {
    const size_t bytes = 512ull << 20;
    uint4 *a, *b;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 1, bytes));
    const size_t n = bytes / sizeof(uint4);
    const float t = time_kernel_ms([&] { copy_kernel<<<grid_for(n), 256>>>(a, b, n); }, 3, 20);
    peak_gbs = 2.0 * bytes / (t * 1e6);
    cudaFree(a);
    cudaFree(b);
    std::printf("Bandwidth ceiling (copy kernel): %.1f GB/s\n", peak_gbs);
  }

  // ---------------- [1] correctness ----------------
  std::printf("\n[1] Correctness vs CPU reference on dequantized weights (N=256, K=512)\n");
  bool all_ok = true;
  {
    const int N = 256, K = 512, Mmax = 128;
    Layer L(N, K, 11, false);
    bf16* X;
    float* Y;
    CUDA_CHECK(cudaMalloc(&X, (size_t)Mmax * K * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&Y, (size_t)Mmax * N * sizeof(float)));
    fill_kernel<<<grid_for((size_t)Mmax * K), 256>>>(X, (size_t)Mmax * K, 12, 1.f);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<bf16> hX((size_t)Mmax * K), hW(L.nW), hS(L.nS);
    std::vector<uint32_t> hQ(L.nWq);
    CUDA_CHECK(cudaMemcpy(hX.data(), X, hX.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hW.data(), L.Wb[0], hW.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hQ.data(), L.Wq[0], hQ.size() * 4, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hS.data(), L.S[0], hS.size() * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> Wd(L.nW);  // dequantized weights, exactly as the kernels see them
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k) {
        const uint32_t word = hQ[(size_t)n * (K / 8) + k / 8];
        const int q = (word >> (4 * (k % 8))) & 0xF;
        Wd[(size_t)n * K + k] = (q - 8) * __bfloat162float(hS[(size_t)n * (K / G) + k / G]);
      }
    auto cpu_ref = [&](int M) {
      std::vector<float> r((size_t)M * N);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double acc = 0;
          for (int k = 0; k < K; ++k)
            acc += (double)__bfloat162float(hX[(size_t)m * K + k]) * Wd[(size_t)n * K + k];
          r[(size_t)m * N + n] = static_cast<float>(acc);
        }
      return r;
    };
    for (int M : {1, 8}) {
      w4a16_gemv(X, L.Wq[0], L.S[0], Y, M, N, K, G);
      CUDA_CHECK(cudaDeviceSynchronize());
      const float e = max_rel_err(to_host(Y, (size_t)M * N), cpu_ref(M));
      all_ok &= e < tol;
      std::printf("  gemv  M=%-4d err=%.1e   %s\n", M, e, e < tol ? "PASS" : "<-- FAIL");
    }
    {
      const auto ref = cpu_ref(Mmax);
      for (int var = 1; var <= 2; ++var) {
        CUDA_CHECK(cudaMemset(Y, 0, (size_t)Mmax * N * sizeof(float)));
        w4a16_gemm(X, L.Wq[0], L.S[0], Y, Mmax, N, K, G, var);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        const float e = max_rel_err(to_host(Y, (size_t)Mmax * N), ref);
        all_ok &= e < tol;
        std::printf("  gemm  M=%-4d v%d err=%.1e   %s\n", Mmax, var, e, e < tol ? "PASS" : "<-- FAIL");
      }
      // quantization error: W4 output vs output with the original bf16 weights
      cublas_linear(handle, X, L.Wb[0], Y, Mmax, N, K);
      CUDA_CHECK(cudaDeviceSynchronize());
      std::printf("  4-bit quantization error (||Y_w4 - Y_bf16|| / ||Y_bf16||, random weights): %.2f%%\n",
                  100.f * rel_fro(ref, to_host(Y, (size_t)Mmax * N)));
    }
    cudaFree(X);
    cudaFree(Y);
  }
  if (!all_ok) {
    std::printf("\nCorrectness check failed; skipping benchmarks.\n");
    return 1;
  }

  const std::string csv_path = "results/w4a16_" + sanitize(prop.name) + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,phase,layer,M,N,K,impl,time_ms,gbs,tflops,speedup\n");

  struct Shape { const char* name; int N, K; };
  const Shape shapes[] = {{"qkv", 6144, 4096}, {"o", 4096, 4096}, {"gate_up", 28672, 4096},
                          {"down", 4096, 14336}};

  // ---------------- [2] decode ----------------
  std::printf("\n[2] Decode: GEMV, weights streamed from DRAM (ceiling %.1f GB/s)\n", peak_gbs);
  std::printf("  %-8s %3s %12s %10s %10s %8s %10s %9s\n", "layer", "M", "N x K", "bf16 ms",
              "w4 ms", "speedup", "w4 GB/s", "%ceiling");
  for (const Shape& sh : shapes) {
    Layer L(sh.N, sh.K, 21, true);
    for (int M : {1, 8}) {
      bf16* X;
      float* Y;
      CUDA_CHECK(cudaMalloc(&X, (size_t)M * sh.K * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc(&Y, (size_t)M * sh.N * sizeof(float)));
      fill_kernel<<<grid_for((size_t)M * sh.K), 256>>>(X, (size_t)M * sh.K, 22, 1.f);
      int ib = 0, iq = 0;
      const float tb = time_kernel_ms(
          [&] { cublas_linear(handle, X, L.Wb[ib++ % L.Wb.size()], Y, M, sh.N, sh.K); }, 3, 30);
      const float tq = time_kernel_ms(
          [&] {
            const int i = iq++ % static_cast<int>(L.Wq.size());
            w4a16_gemv(X, L.Wq[i], L.S[i], Y, M, sh.N, sh.K, G);
          },
          3, 30);
      const double bytes_q = L.nWq * 4.0 + L.nS * 2.0 + (double)M * sh.K * 2 + (double)M * sh.N * 4;
      const double gbs = bytes_q / (tq * 1e6);
      char nk[32];
      std::snprintf(nk, sizeof nk, "%dx%d", sh.N, sh.K);
      std::printf("  %-8s %3d %12s %10.3f %10.3f %7.2fx %10.1f %8.1f%%\n", sh.name, M, nk, tb, tq,
                  tb / tq, gbs, 100.0 * gbs / peak_gbs);
      std::fprintf(csv, "%s,decode,%s,%d,%d,%d,bf16_cublas,%.5f,,,1\n", prop.name, sh.name, M, sh.N,
                   sh.K, tb);
      std::fprintf(csv, "%s,decode,%s,%d,%d,%d,w4_gemv,%.5f,%.2f,,%.3f\n", prop.name, sh.name, M,
                   sh.N, sh.K, tq, gbs, tb / tq);
      cudaFree(X);
      cudaFree(Y);
    }
  }

  // ---------------- [3] prefill ----------------
  std::printf("\n[3] Prefill: tensor-core GEMM (TFLOP/s)\n");
  std::printf("  %-8s %5s %12s %10s %10s %10s %10s\n", "layer", "M", "N x K", "bf16", "w4 v1",
              "w4 v2", "best/bf16");
  for (const Shape& sh : shapes) {
    Layer L(sh.N, sh.K, 31, true);
    for (int M : {512, 2048}) {
      bf16* X;
      float* Y;
      CUDA_CHECK(cudaMalloc(&X, (size_t)M * sh.K * sizeof(bf16)));
      CUDA_CHECK(cudaMalloc(&Y, (size_t)M * sh.N * sizeof(float)));
      fill_kernel<<<grid_for((size_t)M * sh.K), 256>>>(X, (size_t)M * sh.K, 32, 1.f);
      const double flops = 2.0 * M * sh.N * sh.K;
      int ib = 0;
      const float tb = time_kernel_ms(
          [&] { cublas_linear(handle, X, L.Wb[ib++ % L.Wb.size()], Y, M, sh.N, sh.K); }, 2, 10);
      float tv[3];
      for (int var = 1; var <= 2; ++var) {
        int iq = 0;
        tv[var] = time_kernel_ms(
            [&] {
              const int i = iq++ % static_cast<int>(L.Wq.size());
              w4a16_gemm(X, L.Wq[i], L.S[i], Y, M, sh.N, sh.K, G, var);
            },
            2, 10);
      }
      const double fb = flops / (tb * 1e9), f1 = flops / (tv[1] * 1e9), f2 = flops / (tv[2] * 1e9);
      char nk[32];
      std::snprintf(nk, sizeof nk, "%dx%d", sh.N, sh.K);
      std::printf("  %-8s %5d %12s %10.1f %10.1f %10.1f %9.0f%%\n", sh.name, M, nk, fb, f1, f2,
                  100.0 * std::max(f1, f2) / fb);
      std::fprintf(csv, "%s,prefill,%s,%d,%d,%d,bf16_cublas,%.5f,,%.3f,1\n", prop.name, sh.name, M,
                   sh.N, sh.K, tb, fb);
      for (int var = 1; var <= 2; ++var)
        std::fprintf(csv, "%s,prefill,%s,%d,%d,%d,w4_gemm_v%d,%.5f,,%.3f,%.3f\n", prop.name,
                     sh.name, M, sh.N, sh.K, var, tv[var], flops / (tv[var] * 1e9), tb / tv[var]);
      cudaFree(X);
      cudaFree(Y);
    }
  }
  std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  std::printf("\nWrote %s\n", csv_path.c_str());
  return 0;
}
