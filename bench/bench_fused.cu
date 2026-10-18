// Phase 3: memory-bound fused ops vs the eager PyTorch op sequence.
//   [0] bandwidth ceiling: 16-byte copy kernel and cudaMemcpy D2D
//   [1] correctness of fused and eager vs a double-precision CPU reference
//   [2] speed: fused vs eager, achieved GB/s and % of the copy ceiling
// Shapes follow Llama-3-8B: H = 4096, Hq = 32, Hk = 8, D = 128, I = 14336.
// Writes results/fused_<gpu>.csv.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "common.cuh"
#include "fused/rmsnorm.cuh"
#include "fused/rope.cuh"
#include "fused/swiglu.cuh"

using bf16 = __nv_bfloat16;

static std::string sanitize(const char* s) {
  std::string out(s);
  for (char& c : out)
    if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
  return out;
}
static float bf2f(bf16 x) { return __bfloat162float(x); }
static float bfr(float x) { return __bfloat162float(__float2bfloat16(x)); }

struct DevBuf {  // RAII device buffer of bf16
  bf16* p = nullptr;
  size_t n = 0;
  explicit DevBuf(size_t n_) : n(n_) { CUDA_CHECK(cudaMalloc(&p, n * sizeof(bf16))); }
  ~DevBuf() { cudaFree(p); }
  void upload(const std::vector<bf16>& h) {
    CUDA_CHECK(cudaMemcpy(p, h.data(), n * sizeof(bf16), cudaMemcpyHostToDevice));
  }
  std::vector<float> download() const {
    std::vector<bf16> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = bf2f(h[i]);
    return f;
  }
};

// Fast on-GPU fill for the speed tests (values only need to be finite and varied).
__global__ void fill_kernel(bf16* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    uint32_t h = static_cast<uint32_t>(i) * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13; h *= 3266489917u; h ^= h >> 16;
    const float u = (h & 0xFFFFFF) / 16777216.f * 2.f - 1.f;  // uniform [-1, 1)
    p[i] = __float2bfloat16(u * 1.732f * scale);              // ~unit variance * scale
  }
}
static void fill(bf16* p, size_t n, uint32_t seed, float scale = 1.f) {
  fill_kernel<<<grid_for(n), 256>>>(p, n, seed, scale);
}

static std::vector<bf16> randn(size_t n, std::mt19937& rng, float scale = 1.f) {
  std::normal_distribution<float> nd(0.f, scale);
  std::vector<bf16> v(n);
  for (auto& x : v) x = __float2bfloat16(nd(rng));
  return v;
}
static float max_err(const std::vector<float>& out, const std::vector<float>& ref) {
  float w = 0.f;
  for (size_t i = 0; i < out.size(); ++i) {
    if (!std::isfinite(out[i])) return INFINITY;
    w = std::max(w, std::fabs(out[i] - ref[i]) / (std::fabs(ref[i]) + 1.f));
  }
  return w;
}

int main() {
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s | sm_%d%d | %d SMs\n", prop.name, prop.major, prop.minor,
              prop.multiProcessorCount);
  std::mt19937 rng(7);
  const float tol = 3e-2f, eps = 1e-5f;
  bool all_ok = true;

  // ---------------- [0] bandwidth ceiling ----------------
  double peak_gbs = 0;
  {
    const size_t bytes = 512ull << 20;
    uint4 *a, *b;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 1, bytes));
    const size_t n = bytes / sizeof(uint4);
    const float tk = time_kernel_ms([&] { copy_kernel<<<grid_for(n), 256>>>(a, b, n); }, 3, 20);
    const float tm = time_kernel_ms(
        [&] { CUDA_CHECK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice)); }, 3, 20);
    const double gk = 2.0 * bytes / (tk * 1e6), gm = 2.0 * bytes / (tm * 1e6);
    peak_gbs = std::max(gk, gm);
    std::printf("\n[0] Bandwidth ceiling (512 MB read + 512 MB write)\n");
    std::printf("  copy kernel: %.1f GB/s   cudaMemcpy D2D: %.1f GB/s   -> ceiling %.1f GB/s\n", gk,
                gm, peak_gbs);
    cudaFree(a);
    cudaFree(b);
  }

  const int H = 4096, Hq = 32, Hk = 8, D = 128, I = 14336;
  std::vector<float> inv_freq(D / 2);
  for (int i = 0; i < D / 2; ++i) inv_freq[i] = 1.f / std::pow(10000.f, (2.f * i) / D);
  float* d_inv;
  CUDA_CHECK(cudaMalloc(&d_inv, inv_freq.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_inv, inv_freq.data(), inv_freq.size() * sizeof(float),
                        cudaMemcpyHostToDevice));

  // ---------------- [1] correctness ----------------
  std::printf("\n[1] Correctness vs double-precision CPU reference\n");
  {  // residual add + RMSNorm, T = 8
    const int T = 8;
    const auto hx = randn((size_t)T * H, rng), hr = randn((size_t)T * H, rng),
               hw = randn(H, rng, 0.5f);
    std::vector<float> ref_x((size_t)T * H), ref_y((size_t)T * H);
    for (int t = 0; t < T; ++t) {
      double ss = 0;
      for (int j = 0; j < H; ++j) {
        const size_t i = (size_t)t * H + j;
        ref_x[i] = bfr(bf2f(hx[i]) + bf2f(hr[i]));
        ss += (double)ref_x[i] * ref_x[i];
      }
      const double inv = 1.0 / std::sqrt(ss / H + eps);
      for (int j = 0; j < H; ++j) {
        const size_t i = (size_t)t * H + j;
        ref_y[i] = static_cast<float>(ref_x[i] * inv * bf2f(hw[j]));
      }
    }
    DevBuf x((size_t)T * H), r((size_t)T * H), w(H), y((size_t)T * H);
    float *h, *var;
    CUDA_CHECK(cudaMalloc(&h, (size_t)T * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&var, T * sizeof(float)));
    r.upload(hr);
    w.upload(hw);
    x.upload(hx);
    fused_add_rmsnorm(x.p, r.p, w.p, y.p, T, H, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float fx = max_err(x.download(), ref_x), fy = max_err(y.download(), ref_y);
    x.upload(hx);
    eager_add_rmsnorm(x.p, r.p, w.p, y.p, h, var, T, H, eps);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float ex = max_err(x.download(), ref_x), ey = max_err(y.download(), ref_y);
    const bool ok = std::max({fx, fy, ex, ey}) < tol;
    all_ok &= ok;
    std::printf("  add+RMSNorm  fused x=%.1e y=%.1e   eager x=%.1e y=%.1e   %s\n", fx, fy, ex, ey,
                ok ? "PASS" : "<-- FAIL");
    cudaFree(h);
    cudaFree(var);
  }
  {  // RoPE, T = 1024 (positions up to 1023)
    const int T = 1024;
    const auto hq = randn((size_t)T * Hq * D, rng), hk = randn((size_t)T * Hk * D, rng);
    auto rope_ref = [&](const std::vector<bf16>& in, int Hn) {
      std::vector<float> out(in.size());
      for (int t = 0; t < T; ++t)
        for (int hh = 0; hh < Hn; ++hh)
          for (int i = 0; i < D / 2; ++i) {
            const size_t b = ((size_t)t * Hn + hh) * D;
            const double ang = static_cast<double>(static_cast<float>(t) * inv_freq[i]);
            const double x1 = bf2f(in[b + i]), x2 = bf2f(in[b + i + D / 2]);
            out[b + i] = static_cast<float>(x1 * std::cos(ang) - x2 * std::sin(ang));
            out[b + i + D / 2] = static_cast<float>(x2 * std::cos(ang) + x1 * std::sin(ang));
          }
      return out;
    };
    const auto rq = rope_ref(hq, Hq), rk = rope_ref(hk, Hk);
    DevBuf q((size_t)T * Hq * D), k((size_t)T * Hk * D), qo(q.n), ko(k.n), ct((size_t)T * D),
        st((size_t)T * D), tmp(q.n), a(q.n), b(q.n);
    q.upload(hq);
    k.upload(hk);
    rope_fused(q.p, k.p, d_inv, T, Hq, Hk, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float fq = max_err(q.download(), rq), fk = max_err(k.download(), rk);
    q.upload(hq);
    k.upload(hk);
    rope_tables_kernel<<<grid_for((size_t)T * D), 256>>>(d_inv, ct.p, st.p, T, D);
    eager_rope_one(q.p, qo.p, ct.p, st.p, tmp.p, a.p, b.p, T, Hq, D);
    eager_rope_one(k.p, ko.p, ct.p, st.p, tmp.p, a.p, b.p, T, Hk, D);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float eq = max_err(qo.download(), rq), ek = max_err(ko.download(), rk);
    const bool ok = std::max({fq, fk, eq, ek}) < tol;
    all_ok &= ok;
    std::printf("  RoPE         fused q=%.1e k=%.1e   eager q=%.1e k=%.1e   %s\n", fq, fk, eq, ek,
                ok ? "PASS" : "<-- FAIL");
  }
  {  // SwiGLU, T = 8
    const int T = 8;
    const auto hgu = randn((size_t)T * 2 * I, rng, 2.f);
    std::vector<float> ref((size_t)T * I);
    for (int t = 0; t < T; ++t)
      for (int j = 0; j < I; ++j) {
        const double g = bf2f(hgu[(size_t)t * 2 * I + j]), u = bf2f(hgu[(size_t)t * 2 * I + I + j]);
        ref[(size_t)t * I + j] = static_cast<float>(g / (1.0 + std::exp(-g)) * u);
      }
    DevBuf gu((size_t)T * 2 * I), out((size_t)T * I), tmp((size_t)T * I);
    gu.upload(hgu);
    swiglu_fused(gu.p, out.p, T, I);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float f = max_err(out.download(), ref);
    eager_swiglu(gu.p, tmp.p, out.p, T, I);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float e = max_err(out.download(), ref);
    const bool ok = std::max(f, e) < tol;
    all_ok &= ok;
    std::printf("  SwiGLU       fused   =%.1e           eager   =%.1e             %s\n", f, e,
                ok ? "PASS" : "<-- FAIL");
  }
  CUDA_CHECK(cudaGetLastError());
  if (!all_ok) {
    std::printf("\nCorrectness check failed; skipping benchmarks.\n");
    return 1;
  }

  // ---------------- [2] speed ----------------
  const std::string csv_path = "results/fused_" + sanitize(prop.name) + ".csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (!csv) {
    std::fprintf(stderr, "Cannot open %s (run from repo root)\n", csv_path.c_str());
    return 1;
  }
  std::fprintf(csv, "gpu,op,T,eager_ms,fused_ms,speedup,fused_gbs,eager_gbs,pct_ceiling\n");
  std::printf("\n[2] Speed (GB/s = minimum bytes the op must move / time; ceiling %.1f GB/s)\n",
              peak_gbs);
  std::printf("  %-12s %6s %10s %10s %8s %11s %11s %9s\n", "op", "T", "eager ms", "fused ms",
              "speedup", "fused GB/s", "eager GB/s", "%ceiling");

  // footprint = distinct bytes the fused op touches; if it fits in L2, repeated timing
  // iterations are served from cache and GB/s exceeds the DRAM ceiling.
  const double l2_bytes = prop.l2CacheSize;
  std::printf("  (rows marked L2 fit in the %.0f MB L2 cache: they measure cache, not DRAM)\n",
              l2_bytes / 1e6);
  auto report = [&](const char* op, int T, float te, float tf, double bytes, double footprint) {
    const double gf = bytes / (tf * 1e6), ge = bytes / (te * 1e6);
    std::printf("  %-12s %6d %10.3f %10.3f %7.2fx %11.1f %11.1f %8.1f%%%s\n", op, T, te, tf,
                te / tf, gf, ge, 100.0 * gf / peak_gbs, footprint < l2_bytes ? "  L2" : "");
    std::fprintf(csv, "%s,%s,%d,%.5f,%.5f,%.3f,%.2f,%.2f,%.2f\n", prop.name, op, T, te, tf, te / tf,
                 gf, ge, 100.0 * gf / peak_gbs);
  };

  for (int T : {1024, 4096, 16384}) {
    {  // add + RMSNorm: reads x, r; writes x, y  -> 8 bytes per element
      DevBuf x((size_t)T * H), r((size_t)T * H), w(H), y((size_t)T * H);
      float *h, *var;
      CUDA_CHECK(cudaMalloc(&h, (size_t)T * H * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&var, T * sizeof(float)));
      fill(x.p, x.n, 1);
      fill(r.p, r.n, 2, 0.01f);  // small residual so x stays bounded across repeats
      fill(w.p, w.n, 3, 0.5f);
      const float te = time_kernel_ms(
          [&] { eager_add_rmsnorm(x.p, r.p, w.p, y.p, h, var, T, H, eps); }, 2, 20);
      const float tf = time_kernel_ms([&] { fused_add_rmsnorm(x.p, r.p, w.p, y.p, T, H, eps); }, 2, 20);
      report("add+RMSNorm", T, te, tf, 8.0 * T * H, 6.0 * T * H);
      cudaFree(h);
      cudaFree(var);
    }
    {  // RoPE on Q and K: read + write each element once -> 4 bytes per element
      DevBuf q((size_t)T * Hq * D), k((size_t)T * Hk * D), qo(q.n), ko(k.n), ct((size_t)T * D),
          st((size_t)T * D), tmp(q.n), a(q.n), b(q.n);
      fill(q.p, q.n, 4);
      fill(k.p, k.n, 5);
      rope_tables_kernel<<<grid_for((size_t)T * D), 256>>>(d_inv, ct.p, st.p, T, D);
      const float te = time_kernel_ms(
          [&] {
            eager_rope_one(q.p, qo.p, ct.p, st.p, tmp.p, a.p, b.p, T, Hq, D);
            eager_rope_one(k.p, ko.p, ct.p, st.p, tmp.p, a.p, b.p, T, Hk, D);
          },
          2, 20);
      const float tf = time_kernel_ms([&] { rope_fused(q.p, k.p, d_inv, T, Hq, Hk, D); }, 2, 20);
      report("RoPE", T, te, tf, 4.0 * T * (Hq + Hk) * D, 2.0 * T * (Hq + Hk) * D);
    }
    {  // SwiGLU: read gate + up, write out -> 6 bytes per output element
      DevBuf gu((size_t)T * 2 * I), out((size_t)T * I), tmp((size_t)T * I);
      fill(gu.p, gu.n, 6, 2.f);
      const float te = time_kernel_ms([&] { eager_swiglu(gu.p, tmp.p, out.p, T, I); }, 2, 20);
      const float tf = time_kernel_ms([&] { swiglu_fused(gu.p, out.p, T, I); }, 2, 20);
      report("SwiGLU", T, te, tf, 6.0 * T * I, 6.0 * T * I);
    }
  }
  std::fclose(csv);
  cudaFree(d_inv);
  std::printf("\nWrote %s\n", csv_path.c_str());
  return 0;
}
