#pragma once
// Rotary position embedding (HF Llama "rotate_half" convention), applied to Q and K
// right after the QKV projection. Layouts: q [T, Hq, D], k [T, Hk, D] bf16 (GQA: Hk <= Hq).
// For i < D/2, with angle = pos * inv_freq[i]:
//   out[i]       = x[i] * cos - x[i + D/2] * sin
//   out[i + D/2] = x[i + D/2] * cos + x[i] * sin
// inv_freq[i] = theta^(-2i/D), computed once in fp32 (as HF does).
#include "vec.cuh"

// ---------------- fused: Q and K in one in-place pass, sin/cos computed on the fly ----------------
// One thread rotates 8 pairs: a 16-byte chunk at i and its partner at i + D/2.
// pos0: position of token 0 (decode: current sequence length). q_stride / k_stride: elements
// between consecutive tokens (Hq*D / Hk*D when contiguous; larger inside a fused buffer).
__global__ void rope_fused_kernel(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k,
                                  const float* __restrict__ inv_freq, int T, int Hq, int Hk,
                                  int D, int pos0, long long q_stride, long long k_stride) {
  const int chunks = D / 16;
  const size_t total = (size_t)T * (Hq + Hk) * chunks;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int c = static_cast<int>(idx % chunks);
    const size_t rest = idx / chunks;
    const int h = static_cast<int>(rest % (Hq + Hk));
    const int t = static_cast<int>(rest / (Hq + Hk));
    __nv_bfloat16* base =
        h < Hq ? q + (size_t)t * q_stride + (size_t)h * D : k + (size_t)t * k_stride + (size_t)(h - Hq) * D;
    uint4* lo_p = reinterpret_cast<uint4*>(base + c * 8);
    uint4* hi_p = reinterpret_cast<uint4*>(base + D / 2 + c * 8);
    float lo[8], hi[8];
    unpack8(*lo_p, lo);
    unpack8(*hi_p, hi);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      float s, co;
      sincosf(static_cast<float>(pos0 + t) * inv_freq[c * 8 + e], &s, &co);
      const float x1 = lo[e], x2 = hi[e];
      lo[e] = x1 * co - x2 * s;
      hi[e] = x2 * co + x1 * s;
    }
    *lo_p = pack8(lo);
    *hi_p = pack8(hi);
  }
}

inline void rope_fused(__nv_bfloat16* q, __nv_bfloat16* k, const float* inv_freq, int T, int Hq,
                       int Hk, int D, int pos0 = 0, long long q_stride = -1, long long k_stride = -1,
                       cudaStream_t stream = 0) {
  if (q_stride < 0) q_stride = (long long)Hq * D;
  if (k_stride < 0) k_stride = (long long)Hk * D;
  const size_t total = (size_t)T * (Hq + Hk) * (D / 16);
  rope_fused_kernel<<<grid_for(total), 256, 0, stream>>>(q, k, inv_freq, T, Hq, Hk, D, pos0, q_stride,
                                                         k_stride);
}

// ---------------- eager: HF apply_rotary_pos_emb, one kernel per op ----------------
//   q_out = q * cos + rotate_half(q) * sin     (cos/sin: [T, D] bf16 tables, broadcast over heads)
// Per tensor: rotate_half -> tmp, a = x * cos, b = tmp * sin, out = a + b.
__global__ void rope_tables_kernel(const float* inv_freq, __nv_bfloat16* cos_t,
                                   __nv_bfloat16* sin_t, int T, int D) {
  const size_t n = (size_t)T * D;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const int t = static_cast<int>(i / D), d = static_cast<int>(i % D);
    const float ang = static_cast<float>(t) * inv_freq[d % (D / 2)];  // emb = cat(freqs, freqs)
    cos_t[i] = __float2bfloat16(cosf(ang));
    sin_t[i] = __float2bfloat16(sinf(ang));
  }
}
// tmp = cat(-x[D/2:], x[:D/2]) per head
__global__ void eager_rotate_half_kernel(const uint4* __restrict__ x, uint4* __restrict__ tmp,
                                         int D, size_t n8) {
  const int vpr = D / 8;  // vectors per head row
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t rowv = i / vpr;
    const int c = static_cast<int>(i % vpr);
    const bool first = c < vpr / 2;
    float a[8];
    unpack8(x[rowv * vpr + (first ? c + vpr / 2 : c - vpr / 2)], a);
    if (first)
#pragma unroll
      for (int e = 0; e < 8; ++e) a[e] = -a[e];
    tmp[i] = pack8(a);
  }
}
// out = x * table[t] (table broadcast over H heads)
__global__ void eager_mul_table_kernel(const uint4* __restrict__ x, const uint4* __restrict__ tab,
                                       uint4* __restrict__ out, int H, int D, size_t n8) {
  const int vpr = D / 8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t t = i / ((size_t)H * vpr);
    const int c = static_cast<int>(i % vpr);
    float a[8], b[8];
    unpack8(x[i], a);
    unpack8(tab[t * vpr + c], b);
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] *= b[e];
    out[i] = pack8(a);
  }
}
__global__ void eager_add2_kernel(const uint4* __restrict__ a, const uint4* __restrict__ b,
                                  uint4* __restrict__ out, size_t n8) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n8;
       i += (size_t)gridDim.x * blockDim.x) {
    float x[8], y[8];
    unpack8(a[i], x);
    unpack8(b[i], y);
#pragma unroll
    for (int e = 0; e < 8; ++e) x[e] += y[e];
    out[i] = pack8(x);
  }
}

// One tensor ([T, H, D]); tmp/a/b: scratch of the same size.
inline void eager_rope_one(const __nv_bfloat16* x, __nv_bfloat16* out, const __nv_bfloat16* cos_t,
                           const __nv_bfloat16* sin_t, __nv_bfloat16* tmp, __nv_bfloat16* a,
                           __nv_bfloat16* b, int T, int H, int D) {
  const size_t n8 = (size_t)T * H * D / 8;
  auto U = [](const __nv_bfloat16* p) { return reinterpret_cast<const uint4*>(p); };
  auto W = [](__nv_bfloat16* p) { return reinterpret_cast<uint4*>(p); };
  eager_rotate_half_kernel<<<grid_for(n8), 256>>>(U(x), W(tmp), D, n8);
  eager_mul_table_kernel<<<grid_for(n8), 256>>>(U(x), U(cos_t), W(a), H, D, n8);
  eager_mul_table_kernel<<<grid_for(n8), 256>>>(U(tmp), U(sin_t), W(b), H, D, n8);
  eager_add2_kernel<<<grid_for(n8), 256>>>(U(a), U(b), W(out), n8);
}
