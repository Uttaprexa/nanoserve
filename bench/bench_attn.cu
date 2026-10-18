// Attention benchmark: FlashAttention-2 forward vs standard (unfused) attention.
//   Step 1: both implementations vs a double-precision CPU reference (small N).
//   Step 2: speed across sequence lengths; FlashAttention also checked vs the baseline.
// Q, K, V: [BH, N, D] bf16 ~ N(0, 1). Writes results/attn_<gpu>.csv.
//
// Usage: ./build/bench_attn                  (D = 64 and 128, causal and not, N = 512..8192)
//        ./build/bench_attn --d 64 --causal 1 2048 4096
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "common.cuh"
#include "attn/baseline.cuh"
#include "attn/flash_fwd.cuh"

using bf16 = __nv_bfloat16;

static std::string sanitize(const char* s) {
  std::string out(s);
  for (char& c : out)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return out;
}

static float max_rel_err(const std::vector<float>& out, const std::vector<float>& ref) {
  float worst = 0.f;
  for (size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) return INFINITY;
    worst = std::max(worst, std::fabs(out[i] - ref[i]) / (std::fabs(ref[i]) + 1.f));
  }
  return worst;
}

struct Inputs {
  std::vector<bf16> q, k, v;
  bf16 *dQ = nullptr, *dK = nullptr, *dV = nullptr;
  Inputs(int BH, int N, int D, std::mt19937& rng) {
    const size_t n = (size_t)BH * N * D;
    std::normal_distribution<float> nd(0.f, 1.f);
    q.resize(n);
    k.resize(n);
    v.resize(n);
    for (auto* vec : {&q, &k, &v})
      for (auto& x : *vec) x = __float2bfloat16(nd(rng));
    CUDA_CHECK(cudaMalloc(&dQ, n * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dK, n * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&dV, n * sizeof(bf16)));
    CUDA_CHECK(cudaMemcpy(dQ, q.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, k.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, v.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  }
  ~Inputs() {
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dV);
  }
};

struct Outputs {  // device buffers for both implementations
  size_t nO, nS;
  float *S = nullptr, *Obase = nullptr;
  bf16 *P = nullptr, *Oflash = nullptr;
  Outputs(int BH, int N, int D) : nO((size_t)BH * N * D), nS((size_t)BH * N * N) {
    CUDA_CHECK(cudaMalloc(&S, nS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&P, nS * sizeof(bf16)));
    CUDA_CHECK(cudaMalloc(&Obase, nO * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&Oflash, nO * sizeof(bf16)));
  }
  ~Outputs() {
    cudaFree(S);
    cudaFree(P);
    cudaFree(Obase);
    cudaFree(Oflash);
  }
  std::vector<float> base_host() const {
    std::vector<float> h(nO);
    CUDA_CHECK(cudaMemcpy(h.data(), Obase, nO * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
  }
  std::vector<float> flash_host() const {
    std::vector<bf16> tmp(nO);
    CUDA_CHECK(cudaMemcpy(tmp.data(), Oflash, nO * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> h(nO);
    for (size_t i = 0; i < nO; ++i) h[i] = __bfloat162float(tmp[i]);
    return h;
  }
};

static std::vector<float> cpu_attention(const Inputs& in, int BH, int N, int D, bool causal) {
  std::vector<float> out((size_t)BH * N * D);
  const double scale = 1.0 / std::sqrt((double)D);
  std::vector<double> p(N);
  for (int bh = 0; bh < BH; ++bh) {
    const size_t base = (size_t)bh * N * D;
    for (int i = 0; i < N; ++i) {
      const int valid = causal ? i + 1 : N;
      double mx = -1e300;
      for (int j = 0; j < valid; ++j) {
        double d = 0;
        for (int c = 0; c < D; ++c)
          d += (double)__bfloat162float(in.q[base + (size_t)i * D + c]) *
               __bfloat162float(in.k[base + (size_t)j * D + c]);
        p[j] = d * scale;
        mx = std::max(mx, p[j]);
      }
      double sum = 0;
      for (int j = 0; j < valid; ++j) sum += (p[j] = std::exp(p[j] - mx));
      for (int c = 0; c < D; ++c) {
        double acc = 0;
        for (int j = 0; j < valid; ++j)
          acc += p[j] * __bfloat162float(in.v[base + (size_t)j * D + c]);
        out[base + (size_t)i * D + c] = static_cast<float>(acc / sum);
      }
    }
  }
  return out;
}

int main(int argc, char** argv) {
  std::vector<int> dims = {64, 128}, causals = {0, 1}, sizes;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--d") && i + 1 < argc) dims = {std::atoi(argv[++i])};
    else if (!std::strcmp(argv[i], "--causal") && i + 1 < argc) causals = {std::atoi(argv[++i])};
    else sizes.push_back(std::atoi(argv[i]));
  }
  if (sizes.empty()) sizes = {512, 1024, 2048, 4096, 8192};
  const int BH = 16;  // e.g. batch 1 x 16 heads, or batch 2 x 8 heads
  const float tol = 1e-2f;

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);

  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  std::mt19937 rng(1234);

  // ---- Step 1: correctness vs CPU (BH = 2, N = 256) ----
  std::printf("\n[1] Correctness vs double-precision CPU reference (BH=2, N=256)\n");
  bool all_ok = true;
  for (int D : dims)
    for (int causal : causals) {
      const int bh = 2, n = 256;
      Inputs in(bh, n, D, rng);
      Outputs out(bh, n, D);
      attention_baseline(handle, in.dQ, in.dK, in.dV, out.S, out.P, out.Obase, bh, n, D, causal);
      CUDA_CHECK(cudaDeviceSynchronize());
      const auto ref = cpu_attention(in, bh, n, D, causal);
      const float eb = max_rel_err(out.base_host(), ref);
      float ef[3] = {0.f, 0.f, 0.f};
      for (int var = 1; var <= 2; ++var) {
        CUDA_CHECK(cudaMemset(out.Oflash, 0, out.nO * sizeof(bf16)));
        flash_attention_fwd(in.dQ, in.dK, in.dV, out.Oflash, bh, n, D, causal, var);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        ef[var] = max_rel_err(out.flash_host(), ref);
      }
      const bool ok = eb < tol && ef[1] < tol && ef[2] < tol;
      all_ok &= ok;
      std::printf("  D=%-3d causal=%d   baseline err=%.1e   flash_v1 err=%.1e   flash_v2 err=%.1e   %s\n",
                  D, causal, eb, ef[1], ef[2], ok ? "PASS" : "<-- FAIL");
    }
  if (!all_ok) {
    std::printf("\nCorrectness check failed; skipping benchmarks.\n");
    return 1;
  }

  // ---- Step 2: speed ----
  const std::string csv_path = "results/attn_" + sanitize(prop.name) + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,D,causal,BH,N,impl,time_ms,tflops,speedup,max_rel_err,scores_mb\n");

  std::printf("\n[2] Speed, BH=%d (TFLOP/s counts only the useful work; causal = half)\n", BH);
  for (int D : dims)
    for (int causal : causals) {
      std::printf("\n  D=%d causal=%d\n  %6s %11s %9s %11s %9s %11s %9s %9s %9s\n", D, causal, "N",
                  "base ms", "TFLOP/s", "v1 ms", "TFLOP/s", "v2 ms", "TFLOP/s", "best x", "S+P MB");
      for (int N : sizes) {
        Inputs in(BH, N, D, rng);
        Outputs out(BH, N, D);
        const double flops = 4.0 * BH * (double)N * N * D * (causal ? 0.5 : 1.0);
        const double sp_mb = out.nS * (sizeof(float) + sizeof(bf16)) / 1e6;
        const int iters = N <= 2048 ? 20 : 5;

        attention_baseline(handle, in.dQ, in.dK, in.dV, out.S, out.P, out.Obase, BH, N, D, causal);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto base_out = out.base_host();
        const float tb = time_kernel_ms(
            [&] {
              attention_baseline(handle, in.dQ, in.dK, in.dV, out.S, out.P, out.Obase, BH, N, D,
                                 causal);
            },
            1, iters);
        std::fprintf(csv, "%s,%d,%d,%d,%d,baseline,%.5f,%.3f,1.0,0,%.1f\n", prop.name, D, causal,
                     BH, N, tb, flops / (tb * 1e9), sp_mb);

        float tv[3] = {0.f, 0.f, 0.f};
        bool mismatch = false;
        for (int var = 1; var <= 2; ++var) {
          flash_attention_fwd(in.dQ, in.dK, in.dV, out.Oflash, BH, N, D, causal, var);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          const float err = max_rel_err(out.flash_host(), base_out);
          mismatch |= !(err < tol);
          tv[var] = time_kernel_ms(
              [&] { flash_attention_fwd(in.dQ, in.dK, in.dV, out.Oflash, BH, N, D, causal, var); },
              2, iters);
          std::fprintf(csv, "%s,%d,%d,%d,%d,flash_v%d,%.5f,%.3f,%.3f,%.3e,0\n", prop.name, D,
                       causal, BH, N, var, tv[var], flops / (tv[var] * 1e9), tb / tv[var], err);
        }
        const float best = std::min(tv[1], tv[2]);
        std::printf("  %6d %11.3f %9.1f %11.3f %9.1f %11.3f %9.1f %8.2fx %9.0f%s\n", N, tb,
                    flops / (tb * 1e9), tv[1], flops / (tv[1] * 1e9), tv[2],
                    flops / (tv[2] * 1e9), tb / best, sp_mb,
                    mismatch ? "   <-- flash != baseline" : "");
      }
    }
  std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  std::printf("\nWrote %s\n", csv_path.c_str());
  return 0;
}
