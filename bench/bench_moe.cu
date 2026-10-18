// Phase 5: Mixture-of-Experts forward. Fused pipeline vs the per-expert loop of HF Mixtral.
//   [1] correctness vs a double-precision CPU reference (small config)
//   [2] speed for two configurations:
//       mixtral   E=8,  k=2, H=4096, I=14336   (few large experts)
//       qwen-moe  E=60, k=4, H=2048, I=1408    (many small experts, Qwen1.5-MoE-A2.7B)
// Both paths share the router GEMM, top-k and permutation; they differ in the expert MLPs:
//   baseline: copy counts to host (sync), then per active expert: gather, GEMM gate, GEMM up,
//             silu, mul, GEMM down, index_add  (7 launches per expert)
//   fused:    grouped GEMM (gather fused) + fused SwiGLU + grouped GEMM + combine (4 launches)
// Writes results/moe_<gpu>.csv.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "common.cuh"
#include "fused/swiglu.cuh"
#include "moe/moe.cuh"

using bf16 = __nv_bfloat16;

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
template <typename T>
static T* dalloc(size_t n) {
  T* p;
  CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
template <typename T>
static std::vector<T> dl(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}
// Row-major Y[M,N] (ld ldy) = X[M,K] (ld ldx) @ W[N,K]^T
static void linear(cublasHandle_t h, const bf16* X, int ldx, const bf16* W, void* Y,
                   cudaDataType yt, int ldy, int M, int N, int K) {
  const float one = 1.f, zero = 0.f;
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, W, CUDA_R_16BF, K, X,
                            CUDA_R_16BF, ldx, &zero, Y, yt, ldy, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
}

struct MoE {
  int T, E, k, H, I, TK, bm, max_tiles;
  bf16 *X, *Wr, *Wgu, *Wd;
  float *logits, *topw, *sorted_w, *Y2, *out;
  int *ids, *counts, *seg_off, *tile_start, *cursor, *sorted_token, *perm_pos;
  bf16 *GU, *Hact;                         // fused path, permuted rows
  bf16 *Xe, *GUe, *tmp, *He;               // baseline scratch (sized for TK rows)
  float* Ye;
  std::vector<int> h_off;                  // baseline: segment offsets on the host

  MoE(int T_, int E_, int k_, int H_, int I_, uint32_t seed)
      : T(T_), E(E_), k(k_), H(H_), I(I_), TK(T_ * k_) {
    bm = moe_pick_bm(TK, E);
    max_tiles = (TK + bm - 1) / bm + E;
    X = dalloc<bf16>((size_t)T * H);
    Wr = dalloc<bf16>((size_t)E * H);
    Wgu = dalloc<bf16>((size_t)E * 2 * I * H);
    Wd = dalloc<bf16>((size_t)E * H * I);
    logits = dalloc<float>((size_t)T * E);
    topw = dalloc<float>(TK);
    sorted_w = dalloc<float>(TK);
    ids = dalloc<int>(TK);
    counts = dalloc<int>(E);
    seg_off = dalloc<int>(E + 1);
    tile_start = dalloc<int>(E + 1);
    cursor = dalloc<int>(E);
    sorted_token = dalloc<int>(TK);
    perm_pos = dalloc<int>(TK);
    GU = dalloc<bf16>((size_t)TK * 2 * I);
    Hact = dalloc<bf16>((size_t)TK * I);
    Y2 = dalloc<float>((size_t)TK * H);
    out = dalloc<float>((size_t)T * H);
    Xe = dalloc<bf16>((size_t)TK * H);
    GUe = dalloc<bf16>((size_t)TK * 2 * I);
    tmp = dalloc<bf16>((size_t)TK * I);
    He = dalloc<bf16>((size_t)TK * I);
    Ye = dalloc<float>((size_t)TK * H);
    fill_kernel<<<grid_for((size_t)T * H), 256>>>(X, (size_t)T * H, seed, 1.f);
    fill_kernel<<<grid_for((size_t)E * H), 256>>>(Wr, (size_t)E * H, seed + 1, 0.05f);
    fill_kernel<<<grid_for((size_t)E * 2 * I * H), 256>>>(Wgu, (size_t)E * 2 * I * H, seed + 2,
                                                          1.f / std::sqrt((float)H));
    fill_kernel<<<grid_for((size_t)E * H * I), 256>>>(Wd, (size_t)E * H * I, seed + 3,
                                                      1.f / std::sqrt((float)I));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  ~MoE() {
    for (void* p : std::initializer_list<void*>{X, Wr, Wgu, Wd, logits, topw, sorted_w, Y2, out,
                                                ids, counts, seg_off, tile_start, cursor,
                                                sorted_token, perm_pos, GU, Hact, Xe, GUe, tmp, He,
                                                Ye})
      cudaFree(p);
  }

  // Shared by both paths: router, top-k, permutation (all on the GPU).
  void route(cublasHandle_t h) {
    linear(h, X, H, Wr, logits, CUDA_R_32F, E, T, E, H);
    moe_topk_softmax_kernel<<<(T * 32 + 255) / 256, 256>>>(logits, T, E, k, ids, topw);
    CUDA_CHECK(cudaMemsetAsync(counts, 0, E * sizeof(int)));
    moe_count_kernel<<<grid_for(TK), 256>>>(ids, TK, counts);
    moe_scan_kernel<<<1, 32>>>(counts, E, bm, seg_off, tile_start, cursor);
    moe_scatter_kernel<<<grid_for(TK), 256>>>(ids, topw, TK, k, seg_off, cursor, sorted_token,
                                              sorted_w, perm_pos);
  }
  // Fused expert MLPs: 4 launches regardless of E.
  void experts_fused() {
    moe_grouped_gemm<bf16>(bm, X, sorted_token, Wgu, GU, seg_off, tile_start, E, max_tiles, 2 * I, H);
    swiglu_fused(GU, Hact, TK, I);
    moe_grouped_gemm<float>(bm, Hact, nullptr, Wd, Y2, seg_off, tile_start, E, max_tiles, H, I);
    moe_combine_kernel<<<T, 256>>>(Y2, perm_pos, topw, k, H, out);
  }
  // HF-style loop: host needs the counts (device -> host sync), then 7 launches per expert.
  void experts_loop(cublasHandle_t h) {
    h_off.resize(E + 1);
    CUDA_CHECK(cudaMemcpy(h_off.data(), seg_off, (E + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemsetAsync(out, 0, (size_t)T * H * sizeof(float)));
    for (int e = 0; e < E; ++e) {
      const int off = h_off[e], cnt = h_off[e + 1] - h_off[e];
      if (cnt == 0) continue;
      const bf16* wg = Wgu + (size_t)e * 2 * I * H;  // gate rows [0, I), up rows [I, 2I)
      moe_gather_rows_kernel<<<grid_for((size_t)cnt * H / 8), 256>>>(X, sorted_token + off, cnt, H, Xe);
      linear(h, Xe, H, wg, GUe, CUDA_R_16BF, 2 * I, cnt, I, H);                       // w1(x)
      linear(h, Xe, H, wg + (size_t)I * H, GUe + I, CUDA_R_16BF, 2 * I, cnt, I, H);   // w3(x)
      eager_swiglu(GUe, tmp, He, cnt, I);                                              // act * up
      linear(h, He, I, Wd + (size_t)e * H * I, Ye, CUDA_R_32F, H, cnt, H, I);          // w2(.)
      moe_index_add_kernel<<<grid_for((size_t)cnt * H), 256>>>(Ye, sorted_token + off,
                                                               sorted_w + off, cnt, H, out);
    }
  }
};

static float rel_max_err(const std::vector<float>& a, const std::vector<float>& ref) {
  float scale = 0.f, w = 0.f;
  for (float r : ref) scale = std::max(scale, std::fabs(r));
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i])) return INFINITY;
    w = std::max(w, std::fabs(a[i] - ref[i]));
  }
  return w / (scale + 1e-12f);
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));

  // ---------------- [1] correctness ----------------
  std::printf("\n[1] Correctness vs double-precision CPU reference (E=8, k=2, H=256, I=256)\n");
  for (int T : {64, 1024}) {  // 16 rows/expert -> BM=32 tiles; 256 rows/expert -> BM=128 tiles
    const int E = 8, k = 2, H = 256, I = 256;
    MoE m(T, E, k, H, I, 100 + T);
    m.route(handle);
    m.experts_fused();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto out_fused = dl(m.out, (size_t)T * H);
    m.experts_loop(handle);
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto out_loop = dl(m.out, (size_t)T * H);

    const auto hX = dl(m.X, (size_t)T * H), hWgu = dl(m.Wgu, (size_t)E * 2 * I * H),
               hWd = dl(m.Wd, (size_t)E * H * I);
    const auto logits = dl(m.logits, (size_t)T * E);
    const auto ids = dl(m.ids, (size_t)T * k);
    const auto w = dl(m.topw, (size_t)T * k);
    auto f = [](bf16 x) { return __bfloat162float(x); };
    auto bfr = [](double x) { return __bfloat162float(__float2bfloat16(static_cast<float>(x))); };

    // top-k check: CPU selection on the GPU logits (avoids tie flips from rounding)
    bool topk_ok = true;
    for (int t = 0; t < T; ++t) {
      std::vector<int> order(E);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](int a, int b) { return logits[t * E + a] > logits[t * E + b]; });
      double sum = 0;
      for (int j = 0; j < k; ++j) sum += std::exp((double)logits[t * E + order[j]] - logits[t * E + order[0]]);
      for (int j = 0; j < k; ++j) {
        const double ref_w = std::exp((double)logits[t * E + order[j]] - logits[t * E + order[0]]) / sum;
        if (ids[t * k + j] != order[j] || std::fabs(w[t * k + j] - ref_w) > 1e-4) topk_ok = false;
      }
    }
    // full layer reference, using the GPU's routing decisions
    std::vector<float> ref((size_t)T * H, 0.f);
    std::vector<double> gu(2 * I);
    std::vector<float> act(I);
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < k; ++j) {
        const int e = ids[t * k + j];
        for (int n = 0; n < 2 * I; ++n) {
          double s = 0;
          for (int c = 0; c < H; ++c)
            s += (double)f(hX[(size_t)t * H + c]) * f(hWgu[((size_t)e * 2 * I + n) * H + c]);
          gu[n] = bfr(s);  // the GU intermediate is stored in bf16
        }
        for (int c = 0; c < I; ++c) act[c] = bfr(gu[c] / (1.0 + std::exp(-gu[c])) * gu[I + c]);
        for (int n = 0; n < H; ++n) {
          double s = 0;
          for (int c = 0; c < I; ++c) s += (double)act[c] * f(hWd[((size_t)e * H + n) * I + c]);
          ref[(size_t)t * H + n] += static_cast<float>(w[t * k + j] * s);
        }
      }
    const float ef = rel_max_err(out_fused, ref), el = rel_max_err(out_loop, ref);
    const bool ok = topk_ok && ef < 2e-2f && el < 2e-2f;
    std::printf("  T=%-5d BM=%-3d top-k: %s   fused err=%.1e   loop err=%.1e   %s\n", T, m.bm,
                topk_ok ? "match" : "MISMATCH", ef, el, ok ? "PASS" : "<-- FAIL");
    if (!ok) {
      std::printf("\nCorrectness check failed; skipping benchmarks.\n");
      return 1;
    }
  }

  // ---------------- [2] speed ----------------
  const std::string csv_path = "results/moe_" + sanitize(prop.name) + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,config,E,k,H,I,T,active_experts,loop_ms,fused_ms,speedup,fused_tflops,max_err\n");

  struct Cfg { const char* name; int E, k, H, I; };
  const Cfg cfgs[] = {{"mixtral", 8, 2, 4096, 14336}, {"qwen-moe", 60, 4, 2048, 1408}};
  std::printf("\n[2] Speed: expert MLPs + routing (router, top-k, permutation timed in both)\n");
  for (const Cfg& c : cfgs) {
    std::printf("\n  %s: E=%d k=%d H=%d I=%d\n", c.name, c.E, c.k, c.H, c.I);
    std::printf("  %6s %8s %5s %10s %10s %9s %13s %9s\n", "T", "active", "BM", "loop ms",
                "fused ms", "speedup", "fused TFLOP/s", "max err");
    for (int T : {16, 512, 4096}) {
      MoE m(T, c.E, c.k, c.H, c.I, 200 + T);
      m.route(handle);
      m.experts_fused();
      CUDA_CHECK(cudaDeviceSynchronize());
      const auto of = dl(m.out, (size_t)T * c.H);
      m.experts_loop(handle);
      CUDA_CHECK(cudaDeviceSynchronize());
      const float err = rel_max_err(of, dl(m.out, (size_t)T * c.H));
      int active = 0;
      for (int e = 0; e < c.E; ++e) active += m.h_off[e + 1] > m.h_off[e];

      const int iters = T <= 512 ? 10 : 5;
      const float tl = time_kernel_ms([&] { m.route(handle); m.experts_loop(handle); }, 1, iters);
      const float tf = time_kernel_ms([&] { m.route(handle); m.experts_fused(); }, 2, iters);
      const double flops = 6.0 * T * c.k * (double)c.H * c.I;  // gate+up (4HI) + down (2HI) per pair
      std::printf("  %6d %5d/%-2d %5d %10.3f %10.3f %8.2fx %13.1f %9.1e\n", T, active, c.E, m.bm,
                  tl, tf, tl / tf, flops / (tf * 1e9), err);
      std::fprintf(csv, "%s,%s,%d,%d,%d,%d,%d,%d,%.5f,%.5f,%.3f,%.3f,%.3e\n", prop.name, c.name,
                   c.E, c.k, c.H, c.I, T, active, tl, tf, tl / tf, flops / (tf * 1e9), err);
    }
  }
  std::fclose(csv);
  CUBLAS_CHECK(cublasDestroy(handle));
  std::printf("\nWrote %s\n", csv_path.c_str());
  return 0;
}
