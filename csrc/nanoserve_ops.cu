// C ABI over the nanoserve kernels, loaded from Python with ctypes (python/nanoserve/ops.py).
// Every function takes raw device pointers and a cudaStream_t (PyTorch's current stream) and
// returns 0 on success or a CUDA error code. All tensors are bf16 unless noted.
// Built as build/libnanoserve.so by `make lib`. No PyTorch headers are involved, so the library
// does not need to match the CUDA version PyTorch was built with.
#include <cuda_bf16.h>

#include "common.cuh"
#include "attn/flash_fwd.cuh"
#include "fused/rmsnorm.cuh"
#include "fused/rope.cuh"
#include "fused/swiglu.cuh"
#include "quant/w4a16.cuh"

using bf16 = __nv_bfloat16;
static int last_status() { return static_cast<int>(cudaGetLastError()); }

extern "C" {

// x: [T, H] residual stream (updated in place to x + r when r != NULL); y = rmsnorm(x) * w.
int ns_rmsnorm(void* x, const void* r, const void* w, void* y, int T, int H, float eps,
               void* stream) {
  fused_add_rmsnorm(static_cast<bf16*>(x), static_cast<const bf16*>(r), static_cast<const bf16*>(w),
                    static_cast<bf16*>(y), T, H, eps, static_cast<cudaStream_t>(stream));
  return last_status();
}

// In-place RoPE on q [T, Hq, D] and k [T, Hk, D] (token strides in elements), positions pos0 + t.
// inv_freq: [D/2] fp32.
int ns_rope(void* q, void* k, const void* inv_freq, int T, int Hq, int Hk, int D, int pos0,
            long long q_stride, long long k_stride, void* stream) {
  rope_fused(static_cast<bf16*>(q), static_cast<bf16*>(k), static_cast<const float*>(inv_freq), T,
             Hq, Hk, D, pos0, q_stride, k_stride, static_cast<cudaStream_t>(stream));
  return last_status();
}

// out [T, I] = silu(gate) * up, gate and up contiguous [T, I].
int ns_swiglu(const void* gate, const void* up, void* out, int T, int I, void* stream) {
  if (((long long)T * I) % 8) return -1;
  swiglu2_fused(static_cast<const bf16*>(gate), static_cast<const bf16*>(up), static_cast<bf16*>(out),
                T, I, static_cast<cudaStream_t>(stream));
  return last_status();
}

// Causal/non-causal FlashAttention for batch 1, token-major tensors:
//   q [N, Hq, D] with token stride q_t; k, v [N, Hk, D] with token stride kv_t; o [N, Hq, D], o_t.
// Head stride is D for all. variant 1 needs N % 64 == 0, variant 2 needs N % 128 == 0.
int ns_flash_fwd(const void* q, const void* k, const void* v, void* o, int N, int Hq, int Hk,
                 int D, long long q_t, long long kv_t, long long o_t, int causal, int variant,
                 void* stream) {
  if (N % (variant == 1 ? 64 : 128) || Hq % Hk) return -1;
  FlashArgs a{};
  a.q = static_cast<const bf16*>(q);
  a.k = static_cast<const bf16*>(k);
  a.v = static_cast<const bf16*>(v);
  a.o = static_cast<bf16*>(o);
  a.N = N; a.Hq = Hq; a.Hk = Hk;
  a.q_b = a.k_b = a.v_b = a.o_b = 0;
  a.q_h = a.k_h = a.v_h = a.o_h = D;
  a.q_t = q_t; a.k_t = kv_t; a.v_t = kv_t; a.o_t = o_t;
  flash_attention_fwd_strided(a, 1, D, causal != 0, variant, static_cast<cudaStream_t>(stream));
  return last_status();
}

// W [N, K] -> Wq [N, K/8] uint32 + S [N, K/G] bf16 (group-wise symmetric 4-bit).
int ns_w4_quantize(const void* W, void* Wq, void* S, int N, int K, int G, void* stream) {
  if (K % G || G % 32) return -1;
  w4_quantize(static_cast<const bf16*>(W), static_cast<uint32_t*>(Wq), static_cast<bf16*>(S), N, K,
              G, static_cast<cudaStream_t>(stream));
  return last_status();
}

// Y [M, N] (bf16) = X [M, K] @ dequant(Wq, S)^T.
// M in {1, 2, 4, 8}: GEMV path (decode). M % 128 == 0: tensor-core GEMM (prefill). Else -2.
int ns_w4_linear(const void* X, const void* Wq, const void* S, void* Y, int M, int N, int K, int G,
                 void* stream) {
  const auto st = static_cast<cudaStream_t>(stream);
  const auto x = static_cast<const bf16*>(X);
  const auto wq = static_cast<const uint32_t*>(Wq);
  const auto s = static_cast<const bf16*>(S);
  auto y = static_cast<bf16*>(Y);
  if (M == 1 || M == 2 || M == 4 || M == 8) {
    w4a16_gemv(x, wq, s, y, M, N, K, G, st);
  } else if (M % 128 == 0 && N % 128 == 0 && K % 32 == 0 && G % 32 == 0) {
    w4a16_gemm(x, wq, s, y, M, N, K, G, 1, st);
  } else {
    return -2;
  }
  return last_status();
}

}  // extern "C"
