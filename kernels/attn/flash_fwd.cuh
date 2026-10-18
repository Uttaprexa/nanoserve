#pragma once
// FlashAttention-2 forward pass, bf16 in / bf16 out, fp32 softmax and accumulation.
//   O = softmax(scale * Q K^T [+ causal mask]) V,   Q, K, V, O: [BH, N, D] row-major
//
// One block = BR query rows of one (batch, head). Each of the NWARPS warps owns
// 16 query rows and keeps, entirely in registers:
//   * its Q fragments (loaded once with ldmatrix),
//   * the running row max m and row sum l (online softmax),
//   * its 16 x D output accumulator O.
// The block streams K/V in BC-row tiles through shared memory (cp.async, double
// buffered). Per tile, each warp:
//   1. S = Q K^T             mma.sync, K fragments via ldmatrix (K rows = keys)
//   2. online softmax        m_new = max(m, rowmax(S)); P = exp2(S - m_new);
//                            O *= exp2(m - m_new);  l = l * exp2(m - m_new) + rowsum(P)
//   3. O += P V              P goes from the S accumulator layout straight into
//                            A-fragment registers (no shared memory round trip),
//                            V fragments via ldmatrix.trans
// The N x N score matrix never leaves registers.
//
// Lane roles inside a 16-row strip (g = lane/4, t = lane%4): a lane holds rows
// g and g+8, columns 2t and 2t+1 of every 8-column tile. The 4 lanes of a quad
// share rows, so row max / row sum need only two __shfl_xor steps.
// REQUIRES: N % BR == 0 (64 or 128), D in {64, 128}.
#include <cfloat>
#include <cuda_bf16.h>

#include "../hgemm/ptx.cuh"

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);  // .x = lo (low 16 bits)
  return *reinterpret_cast<uint32_t*>(&v);
}

template <int D, int BR, int BC, int NWARPS, bool CAUSAL>
__global__ void __launch_bounds__(NWARPS * 32)
    flash_fwd_kernel(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                     const __nv_bfloat16* __restrict__ V, __nv_bfloat16* __restrict__ O, int N,
                     float scale_log2) {
  static_assert(BR == NWARPS * 16, "each warp owns 16 query rows");
  static_assert(D % 16 == 0 && BC % 16 == 0, "tile shapes");
  constexpr int NT = NWARPS * 32;
  constexpr int LD = D + 8;     // padded smem row (16 B pad -> conflict-free ldmatrix)
  constexpr int KCH = D / 16;   // k16 chunks over D for S = Q K^T
  constexpr int SN = BC / 8;    // n8 tiles of S per warp
  constexpr int PK = BC / 16;   // k16 chunks over keys for O += P V
  constexpr int ON = D / 8;     // n8 tiles of O per warp
  constexpr float NEG_INF = -INFINITY;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* Qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // BR x LD
  __nv_bfloat16* Ks = Qs + BR * LD;                                 // 2 x BC x LD
  __nv_bfloat16* Vs = Ks + 2 * BC * LD;                             // 2 x BC x LD

  const int qblk = blockIdx.x;
  const size_t head = (size_t)blockIdx.y * N * D;
  Q += head + (size_t)qblk * BR * D;
  O += head + (size_t)qblk * BR * D;
  K += head;
  V += head;

  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  const int g = lane / 4, t = lane % 4;
  const int lq = lane / 8, lr = lane % 8;  // ldmatrix: which 8x8 matrix, which row

  auto load_rows = [&](__nv_bfloat16* dst, const __nv_bfloat16* src, int rows) {
    constexpr int VPR = D / 8;  // 16-byte vectors per row
    for (int v = tid; v < rows * VPR; v += NT) {
      const int row = v / VPR, col = (v % VPR) * 8;
      cp_async_16(&dst[row * LD + col], &src[(size_t)row * D + col]);
    }
  };

  // Causal: tiles entirely above the diagonal contribute nothing; skip them.
  const int n_tiles = CAUSAL ? ((qblk + 1) * BR + BC - 1) / BC : N / BC;

  // Group 0: Q. Group 1: K/V tile 0.
  load_rows(Qs, Q, BR);
  cp_async_commit();
  load_rows(Ks, K, BC);
  load_rows(Vs, V, BC);
  cp_async_commit();

  cp_async_wait<1>();  // Q has landed
  __syncthreads();
  uint32_t qf[KCH][4];
#pragma unroll
  for (int kc = 0; kc < KCH; ++kc)
    ldmatrix_x4(qf[kc], &Qs[(warp * 16 + lr + 8 * (lq % 2)) * LD + kc * 16 + 8 * (lq / 2)]);

  float o[ON][4];
#pragma unroll
  for (int n = 0; n < ON; ++n) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
  float m0 = NEG_INF, m1 = NEG_INF;  // running max, rows g and g+8 (log2 domain)
  float l0 = 0.f, l1 = 0.f;          // running sum (this lane's columns only)

  const int qrow0 = qblk * BR + warp * 16 + g, qrow1 = qrow0 + 8;

  for (int kt = 0; kt < n_tiles; ++kt) {
    cp_async_wait<0>();  // tile kt has landed
    __syncthreads();     // ...for everyone, and tile kt-1's buffer is free
    if (kt + 1 < n_tiles) {
      const int nb = (kt + 1) & 1;
      load_rows(Ks + nb * BC * LD, K + (size_t)(kt + 1) * BC * D, BC);
      load_rows(Vs + nb * BC * LD, V + (size_t)(kt + 1) * BC * D, BC);
    }
    cp_async_commit();

    // Causal: with BR > BC, late tiles can be fully masked for this warp's 16 rows.
    // Skip the math (the loads above and the barrier at the loop top still happen).
    if (CAUSAL && kt * BC > qblk * BR + warp * 16 + 15) continue;

    const __nv_bfloat16* ks = Ks + (kt & 1) * BC * LD;
    const __nv_bfloat16* vs = Vs + (kt & 1) * BC * LD;

    // ---- 1. S = Q K^T (16 x BC per warp) ----
    float s[SN][4];
#pragma unroll
    for (int n = 0; n < SN; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
#pragma unroll
    for (int kc = 0; kc < KCH; ++kc)
#pragma unroll
      for (int n = 0; n < SN; n += 2) {
        // matrices: {keys n..n+7 | d lo}, {keys n..n+7 | d hi}, {keys n+8.. | d lo}, {.. | d hi}
        uint32_t b[4];
        ldmatrix_x4(b, &ks[(n * 8 + lr + 8 * (lq / 2)) * LD + kc * 16 + 8 * (lq % 2)]);
        mma_bf16_16816(s[n], qf[kc], b[0], b[1]);
        mma_bf16_16816(s[n + 1], qf[kc], b[2], b[3]);
      }

    // ---- 2. online softmax ----
    float mx0 = m0, mx1 = m1;
#pragma unroll
    for (int n = 0; n < SN; ++n) {
#pragma unroll
      for (int e = 0; e < 2; ++e) {
        s[n][e] *= scale_log2;
        s[n][2 + e] *= scale_log2;
        if (CAUSAL) {
          const int key = kt * BC + n * 8 + 2 * t + e;
          if (key > qrow0) s[n][e] = NEG_INF;
          if (key > qrow1) s[n][2 + e] = NEG_INF;
        }
      }
      mx0 = fmaxf(mx0, fmaxf(s[n][0], s[n][1]));
      mx1 = fmaxf(mx1, fmaxf(s[n][2], s[n][3]));
    }
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
    // Tile 0 always contains key 0 (valid for every row), so after it m is finite and
    // a row with no valid key in a later tile just keeps its max (c = 1, P = 0).
    const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);  // first tile: exp2(-inf) = 0
    m0 = mx0;
    m1 = mx1;

    float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
    for (int n = 0; n < SN; ++n) {
      s[n][0] = exp2f(s[n][0] - m0);
      s[n][1] = exp2f(s[n][1] - m0);
      s[n][2] = exp2f(s[n][2] - m1);
      s[n][3] = exp2f(s[n][3] - m1);
      rs0 += s[n][0] + s[n][1];
      rs1 += s[n][2] + s[n][3];
    }
    l0 = l0 * c0 + rs0;
    l1 = l1 * c1 + rs1;
#pragma unroll
    for (int n = 0; n < ON; ++n) {
      o[n][0] *= c0;
      o[n][1] *= c0;
      o[n][2] *= c1;
      o[n][3] *= c1;
    }

    // ---- 3. O += P V ----
    // P's accumulator layout for n8 tiles 2kc and 2kc+1 is exactly the A layout of
    // one m16k16 fragment: {(g, lo), (g+8, lo), (g, hi), (g+8, hi)}.
#pragma unroll
    for (int kc = 0; kc < PK; ++kc) {
      uint32_t pa[4];
      pa[0] = pack_bf16x2(s[2 * kc][0], s[2 * kc][1]);
      pa[1] = pack_bf16x2(s[2 * kc][2], s[2 * kc][3]);
      pa[2] = pack_bf16x2(s[2 * kc + 1][0], s[2 * kc + 1][1]);
      pa[3] = pack_bf16x2(s[2 * kc + 1][2], s[2 * kc + 1][3]);
#pragma unroll
      for (int n = 0; n < ON; n += 2) {
        uint32_t b[4];
        ldmatrix_x4_trans(b, &vs[(kc * 16 + lr + 8 * (lq % 2)) * LD + n * 8 + 8 * (lq / 2)]);
        mma_bf16_16816(o[n], pa, b[0], b[1]);
        mma_bf16_16816(o[n + 1], pa, b[2], b[3]);
      }
    }
  }
  cp_async_wait<0>();

  // ---- epilogue: finish the row sums across the quad, normalize, store bf16 ----
  l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
  l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
  l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
#pragma unroll
  for (int n = 0; n < ON; ++n) {
    const int col = n * 8 + 2 * t;
    *reinterpret_cast<__nv_bfloat162*>(&O[(size_t)(warp * 16 + g) * D + col]) =
        __floats2bfloat162_rn(o[n][0] * inv0, o[n][1] * inv0);
    *reinterpret_cast<__nv_bfloat162*>(&O[(size_t)(warp * 16 + g + 8) * D + col]) =
        __floats2bfloat162_rn(o[n][2] * inv1, o[n][3] * inv1);
  }
}

template <int D, int BR, int BC, int NWARPS, bool CAUSAL>
inline void launch_flash_fwd_cfg(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                 const __nv_bfloat16* V, __nv_bfloat16* O, int BH, int N) {
  constexpr int SMEM = (BR + 4 * BC) * (D + 8) * (int)sizeof(__nv_bfloat16);
  if (N % BR || N % BC) {
    std::fprintf(stderr, "flash_fwd needs N %% %d == 0\n", BR);
    std::exit(EXIT_FAILURE);
  }
  auto kernel = flash_fwd_kernel<D, BR, BC, NWARPS, CAUSAL>;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  const float scale_log2 = 1.4426950408889634f / sqrtf(static_cast<float>(D));
  dim3 grid(N / BR, BH);
  kernel<<<grid, NWARPS * 32, SMEM>>>(Q, K, V, O, N, scale_log2);
}

// variant 1: BR=64 (4 warps), BC=64.    smem: D=64 46 KB (2 blocks/SM), D=128 87 KB (1 block/SM)
// variant 2: BR=128 (8 warps), BC=64 for D=64 (55 KB) / BC=32 for D=128 (70 KB):
//            one block/SM either way, but 8 warps and each K/V tile serves 128 queries.
template <int D, bool CAUSAL>
inline void flash_fwd_dispatch(int variant, const __nv_bfloat16* Q, const __nv_bfloat16* K,
                               const __nv_bfloat16* V, __nv_bfloat16* O, int BH, int N) {
  constexpr int BC2 = (D == 128) ? 32 : 64;
  if (variant == 1) launch_flash_fwd_cfg<D, 64, 64, 4, CAUSAL>(Q, K, V, O, BH, N);
  else launch_flash_fwd_cfg<D, 128, BC2, 8, CAUSAL>(Q, K, V, O, BH, N);
}

inline void flash_attention_fwd(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                const __nv_bfloat16* V, __nv_bfloat16* O, int BH, int N, int D,
                                bool causal, int variant = 1) {
  if (D == 64) {
    causal ? flash_fwd_dispatch<64, true>(variant, Q, K, V, O, BH, N)
           : flash_fwd_dispatch<64, false>(variant, Q, K, V, O, BH, N);
  } else if (D == 128) {
    causal ? flash_fwd_dispatch<128, true>(variant, Q, K, V, O, BH, N)
           : flash_fwd_dispatch<128, false>(variant, Q, K, V, O, BH, N);
  } else {
    std::fprintf(stderr, "flash_fwd supports D = 64 or 128\n");
    std::exit(EXIT_FAILURE);
  }
}
