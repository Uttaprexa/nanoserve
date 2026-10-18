#pragma once
// Residual add + RMSNorm, as in every Llama-style decoder layer:
//   x = x + r                       (residual stream, kept for the next layer)
//   y = x * rsqrt(mean(x^2) + eps) * w
// x, r, y: [T, H] bf16 row-major; w: [H] bf16.
#include "vec.cuh"

// ---------------- fused: one kernel, one block per row ----------------
// Each thread keeps its slice of the row in registers between the two phases,
// so x and r are read once and x, y written once: 8 bytes per element total.
template <int MAXV>
__global__ void __launch_bounds__(256)
    fused_add_rmsnorm_kernel(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ r,
                             const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y,
                             int H, float eps) {
  const size_t row = blockIdx.x;
  const int vecs = H / 8;
  uint4* xr = reinterpret_cast<uint4*>(x + row * H);
  const uint4* rr = reinterpret_cast<const uint4*>(r + row * H);
  const uint4* wr = reinterpret_cast<const uint4*>(w);
  uint4* yr = reinterpret_cast<uint4*>(y + row * H);

  float v[MAXV][8];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < MAXV; ++i) {
    const int idx = threadIdx.x + i * blockDim.x;
    if (idx < vecs) {
      float a[8], b[8];
      unpack8(xr[idx], a);
      unpack8(rr[idx], b);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        v[i][e] = round_bf16(a[e] + b[e]);  // the residual stream is stored in bf16
        ss += v[i][e] * v[i][e];
      }
      xr[idx] = pack8(v[i]);
    }
  }
  const float inv = rsqrtf(fused_block_sum(ss) / H + eps);
#pragma unroll
  for (int i = 0; i < MAXV; ++i) {
    const int idx = threadIdx.x + i * blockDim.x;
    if (idx < vecs) {
      float g[8], o[8];
      unpack8(wr[idx], g);
#pragma unroll
      for (int e = 0; e < 8; ++e) o[e] = v[i][e] * inv * g[e];
      yr[idx] = pack8(o);
    }
  }
}

inline void fused_add_rmsnorm(__nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
                              __nv_bfloat16* y, int T, int H, float eps) {
  if (H % 8 || H / 8 > 256 * 4) {
    std::fprintf(stderr, "fused_add_rmsnorm needs H %% 8 == 0 and H <= 8192\n");
    std::exit(EXIT_FAILURE);
  }
  fused_add_rmsnorm_kernel<4><<<T, 256>>>(x, r, w, y, H, eps);
}

// ---------------- eager: the op sequence of HF LlamaRMSNorm, one kernel each ----------------
//   x = x + r;  h = x.float();  var = h.pow(2).mean(-1);  h = h * rsqrt(var + eps);
//   y = w * h.to(bf16)
__global__ void eager_add_kernel(uint4* __restrict__ x, const uint4* __restrict__ r, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float a[8], b[8];
    unpack8(x[i], a);
    unpack8(r[i], b);
#pragma unroll
    for (int e = 0; e < 8; ++e) a[e] += b[e];
    x[i] = pack8(a);
  }
}
__global__ void eager_cast_f32_kernel(const uint4* __restrict__ x, float4* __restrict__ h,
                                      size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    float a[8];
    unpack8(x[i], a);
    h[2 * i] = make_float4(a[0], a[1], a[2], a[3]);
    h[2 * i + 1] = make_float4(a[4], a[5], a[6], a[7]);
  }
}
__global__ void eager_mean_sq_kernel(const float* __restrict__ h, float* __restrict__ var, int H) {
  const float* row = h + (size_t)blockIdx.x * H;
  float ss = 0.f;
  for (int j = threadIdx.x * 4; j < H; j += blockDim.x * 4) {
    const float4 q = *reinterpret_cast<const float4*>(row + j);
    ss += q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
  }
  ss = fused_block_sum(ss);
  if (threadIdx.x == 0) var[blockIdx.x] = ss / H;
}
__global__ void eager_mul_rsqrt_kernel(float4* __restrict__ h, const float* __restrict__ var,
                                       int H, float eps, size_t n4) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4;
       i += (size_t)gridDim.x * blockDim.x) {
    const float s = rsqrtf(var[(i * 4) / H] + eps);
    float4 q = h[i];
    q.x *= s; q.y *= s; q.z *= s; q.w *= s;
    h[i] = q;
  }
}
__global__ void eager_cast_mul_w_kernel(const float4* __restrict__ h, const uint4* __restrict__ w,
                                        uint4* __restrict__ y, int H, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const float4 a = h[2 * i], b = h[2 * i + 1];
    float hb[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w}, g[8];
    unpack8(w[i % (H / 8)], g);
#pragma unroll
    for (int e = 0; e < 8; ++e) hb[e] = round_bf16(hb[e]) * g[e];  // h.to(bf16), then * w in bf16
    y[i] = pack8(hb);
  }
}

// h: [T, H] fp32 scratch, var: [T] fp32 scratch.
inline void eager_add_rmsnorm(__nv_bfloat16* x, const __nv_bfloat16* r, const __nv_bfloat16* w,
                              __nv_bfloat16* y, float* h, float* var, int T, int H, float eps) {
  const size_t n8 = (size_t)T * H / 8, n4 = (size_t)T * H / 4;
  eager_add_kernel<<<grid_for(n8), 256>>>(reinterpret_cast<uint4*>(x),
                                          reinterpret_cast<const uint4*>(r), n8);
  eager_cast_f32_kernel<<<grid_for(n8), 256>>>(reinterpret_cast<const uint4*>(x),
                                               reinterpret_cast<float4*>(h), n8);
  eager_mean_sq_kernel<<<T, 256>>>(h, var, H);
  eager_mul_rsqrt_kernel<<<grid_for(n4), 256>>>(reinterpret_cast<float4*>(h), var, H, eps, n4);
  eager_cast_mul_w_kernel<<<grid_for(n8), 256>>>(reinterpret_cast<const float4*>(h),
                                                 reinterpret_cast<const uint4*>(w),
                                                 reinterpret_cast<uint4*>(y), H, n8);
}
