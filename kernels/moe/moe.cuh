#pragma once
// Mixture-of-Experts layer (Mixtral / Qwen-MoE style), forward pass.
//   logits = X Wr^T                      [T, E]       router (cuBLAS, tiny)
//   ids, w = top-k softmax(logits)       [T, k]       fused, one warp per token
//   permute: group the T*k (token, expert) pairs by expert, on the GPU (no host sync)
//   GU     = grouped GEMM(X[rows], W_gu[e])   [T*k, 2I]  rows gathered while loading A
//   Hact   = silu(GU[:, :I]) * GU[:, I:]      [T*k, I]   Phase 3 fused SwiGLU
//   Y      = grouped GEMM(Hact, W_d[e])       [T*k, H]
//   out[t] = sum_j w[t, j] * Y[pos(t, j)]     [T, H]     fused combine
// Weights: W_gu [E, 2I, H] (gate rows then up rows), W_d [E, H, I], Wr [E, H], bf16.
#include <cfloat>
#include <cuda_bf16.h>

#include "../fused/vec.cuh"
#include "../hgemm/ptx.cuh"

// cp.async with zero-fill: copies 16 bytes if valid, else writes 16 zero bytes.
__device__ __forceinline__ void cp_async_16_zfill(void* smem, const void* gmem, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  const int n = valid ? 16 : 0;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(n)
               : "memory");
}

// ============================ 1. fused top-k + softmax ============================
// One warp per token; E <= 64 (two logits per lane). k rounds of warp arg-max, then a
// softmax over the k selected logits. This equals HF Mixtral's softmax over all experts
// followed by top-k and renormalization: exp(l_i) / sum over the top-k of exp(l_j).
__global__ void moe_topk_softmax_kernel(const float* __restrict__ logits, int T, int E, int k,
                                        int* __restrict__ ids, float* __restrict__ w) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  const int lane = threadIdx.x % 32;
  if (warp_global >= T) return;
  const float* row = logits + (size_t)warp_global * E;
  float v0 = lane < E ? row[lane] : -FLT_MAX;
  float v1 = lane + 32 < E ? row[lane + 32] : -FLT_MAX;
  float sel_v[8];
  int sel_i[8];
  for (int j = 0; j < k; ++j) {
    float best = v0 >= v1 ? v0 : v1;
    int bi = v0 >= v1 ? lane : lane + 32;
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (ov > best || (ov == best && oi < bi)) {
        best = ov;
        bi = oi;
      }
    }
    sel_v[j] = best;
    sel_i[j] = bi;
    if (bi == lane) v0 = -FLT_MAX;
    if (bi == lane + 32) v1 = -FLT_MAX;
  }
  if (lane == 0) {
    float sum = 0.f;
    for (int j = 0; j < k; ++j) sum += __expf(sel_v[j] - sel_v[0]);  // sel_v[0] is the max
    for (int j = 0; j < k; ++j) {
      ids[(size_t)warp_global * k + j] = sel_i[j];
      w[(size_t)warp_global * k + j] = __expf(sel_v[j] - sel_v[0]) / sum;
    }
  }
}

// ============================ 2. permutation (device only) ============================
__global__ void moe_count_kernel(const int* __restrict__ ids, int TK, int* __restrict__ counts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < TK; i += gridDim.x * blockDim.x)
    atomicAdd(&counts[ids[i]], 1);
}
// Single thread: E is small. seg_off = exclusive scan of counts; tile_start = exclusive scan
// of ceil(count / BM) (m-tiles per expert, for the grouped GEMM's tile schedule).
__global__ void moe_scan_kernel(const int* __restrict__ counts, int E, int BM,
                                int* __restrict__ seg_off, int* __restrict__ tile_start,
                                int* __restrict__ cursor) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int off = 0, tiles = 0;
  for (int e = 0; e < E; ++e) {
    seg_off[e] = off;
    tile_start[e] = tiles;
    cursor[e] = 0;
    off += counts[e];
    tiles += (counts[e] + BM - 1) / BM;
  }
  seg_off[E] = off;
  tile_start[E] = tiles;
}
// Each (token, slot) pair claims a position inside its expert's segment.
__global__ void moe_scatter_kernel(const int* __restrict__ ids, const float* __restrict__ w, int TK,
                                   int k, const int* __restrict__ seg_off, int* __restrict__ cursor,
                                   int* __restrict__ sorted_token, float* __restrict__ sorted_w,
                                   int* __restrict__ perm_pos) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < TK; i += gridDim.x * blockDim.x) {
    const int e = ids[i];
    const int pos = seg_off[e] + atomicAdd(&cursor[e], 1);
    sorted_token[pos] = i / k;
    sorted_w[pos] = w[i];
    perm_pos[i] = pos;
  }
}

// ============================ 3. grouped GEMM ============================
// C[r, :] = A[src(r), :] @ W[e(r)]^T for every row r of the permuted problem, where rows
// seg_off[e] .. seg_off[e+1]-1 belong to expert e. One launch covers all experts:
//   blockIdx.x = n tile, blockIdx.y = global m tile; the block binary-searches tile_start to
//   find its expert, so experts with many tokens simply get more m tiles.
// src(r) = a_rows[r] if a_rows (gather fused into the A load), else r.
// Rows past an expert's segment are zero-filled on load and skipped on store.
// W[e] is [N, K] row-major: the smem tile Bs[n][k] feeds non-transposed ldmatrix as mma's B.
// REQUIRES: N % BN == 0, K % BK == 0, 16-byte aligned rows.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, typename OutT>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    moe_grouped_gemm_kernel(const __nv_bfloat16* __restrict__ A, const int* __restrict__ a_rows,
                            const __nv_bfloat16* __restrict__ W, OutT* __restrict__ C,
                            const int* __restrict__ seg_off, const int* __restrict__ tile_start,
                            int E, int N, int K) {
  constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int LDA = BK + 8, LDB = BK + 8;
  constexpr int A_STAGE = BM * LDA, B_STAGE = BN * LDB;
  constexpr int A_VEC = BM * BK / 8, B_VEC = BN * BK / 8;
  constexpr int A_PER = A_VEC / NUM_THREADS;
  static_assert(A_VEC % NUM_THREADS == 0 && B_VEC % NUM_THREADS == 0, "even load split");
  static_assert(NI % 2 == 0 && WM % 16 == 0, "tile shapes");

  const int n_tiles = N / BN;
  const long long b = blockIdx.x;
  if (b >= (long long)tile_start[E] * n_tiles) return;
  const int t = static_cast<int>(b / n_tiles);  // lies in [tile_start[e], tile_start[e+1])
  int lo = 0, hi = E - 1;  // largest e with tile_start[e] <= t (skips empty experts)
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (tile_start[mid] <= t) lo = mid;
    else hi = mid - 1;
  }
  const int e = lo;
  const int mt_e = tile_start[e + 1] - tile_start[e];
  const int local = static_cast<int>(b - (long long)tile_start[e] * n_tiles);
  const int n_tile = local / mt_e, m_local = local % mt_e;
  const int row0 = seg_off[e] + m_local * BM;
  const int row_end = seg_off[e + 1];
  const __nv_bfloat16* We = W + ((size_t)e * N + (size_t)n_tile * BN) * K;

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  __nv_bfloat16* Bs = As + STAGES * A_STAGE;

  const int tid = threadIdx.x, lane = tid % 32, warpIdx = tid / 32;
  const int warpRow = warpIdx / WARPS_N, warpCol = warpIdx % WARPS_N;
  const int lq = lane / 8, lr = lane % 8;

  // The A rows this thread copies are fixed across K tiles: resolve gather + validity once.
  const __nv_bfloat16* a_src[A_PER];
  bool a_ok[A_PER];
  int a_dst[A_PER];
#pragma unroll
  for (int p = 0; p < A_PER; ++p) {
    const int v = tid + p * NUM_THREADS;
    const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
    const int r = row0 + row;
    a_ok[p] = r < row_end;
    const int rr = a_ok[p] ? r : row0;  // any valid row; data is zero-filled anyway
    const int src = a_rows ? a_rows[rr] : rr;
    a_src[p] = A + (size_t)src * K + col;
    a_dst[p] = row * LDA + col;
  }

  auto load_tile = [&](int stage, int bk) {
    __nv_bfloat16* as = As + stage * A_STAGE;
    __nv_bfloat16* bs = Bs + stage * B_STAGE;
#pragma unroll
    for (int p = 0; p < A_PER; ++p) cp_async_16_zfill(&as[a_dst[p]], a_src[p] + bk, a_ok[p]);
#pragma unroll
    for (int p = 0; p < B_VEC / NUM_THREADS; ++p) {
      const int v = tid + p * NUM_THREADS;
      const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
      cp_async_16(&bs[row * LDB + col], &We[(size_t)row * K + bk + col]);
    }
  };

  float acc[MI][NI][4];
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;

  const int KT = K / BK;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_tile(s, s * BK);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int next = kt + STAGES - 1;
    if (next < KT) load_tile(next % STAGES, next * BK);
    cp_async_commit();

    const __nv_bfloat16* as = As + (kt % STAGES) * A_STAGE;
    const __nv_bfloat16* bs = Bs + (kt % STAGES) * B_STAGE;
#pragma unroll
    for (int kk = 0; kk < BK; kk += 16) {
      uint32_t afrag[MI][4], bfrag[NI][2];
#pragma unroll
      for (int i = 0; i < MI; ++i)
        ldmatrix_x4(afrag[i],
                    &as[(warpRow * WM + i * 16 + lr + 8 * (lq % 2)) * LDA + kk + 8 * (lq / 2)]);
#pragma unroll
      for (int j = 0; j < NI; j += 2) {
        uint32_t t4[4];
        ldmatrix_x4(t4, &bs[(warpCol * WN + j * 8 + lr + 8 * (lq / 2)) * LDB + kk + 8 * (lq % 2)]);
        bfrag[j][0] = t4[0];
        bfrag[j][1] = t4[1];
        bfrag[j + 1][0] = t4[2];
        bfrag[j + 1][1] = t4[3];
      }
#pragma unroll
      for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j) mma_bf16_16816(acc[i][j], afrag[i], bfrag[j][0], bfrag[j][1]);
    }
  }
  cp_async_wait<0>();

  const int g = lane / 4, tq = lane % 4;
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row0 + warpRow * WM + i * 16 + g + 8 * h;
        if (r >= row_end) continue;
        const int col = n_tile * BN + warpCol * WN + j * 8 + 2 * tq;
        if constexpr (sizeof(OutT) == 4) {
          *reinterpret_cast<float2*>(&C[(size_t)r * N + col]) =
              make_float2(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        } else {
          *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)r * N + col]) =
              __floats2bfloat162_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
        }
      }
}

// Two tile shapes. BM is also the granularity of the tile schedule built by moe_scan_kernel.
//   BM = 128: 2x2 warps, 64x64 warp tiles (Phase 1's fastest config) for many rows per expert
//   BM = 32:  1x4 warps, 32x32 warp tiles, BK = 64, for decode / many small experts.
//             Decode streams weights from DRAM, and W[e] is [N, K]: each K tile reads BK
//             contiguous elements from each of BN rows. With BK = 32 that is 64 B per row; ncu
//             showed DRAM 91% busy yet moving ~1.5x the weight bytes (the other half of each
//             128 B line was often evicted before the next K step used it). BK = 64 reads one
//             full 128 B line per row per tile.
inline int moe_pick_bm(int rows_total, int E) { return rows_total / E <= 64 ? 32 : 128; }

template <int BM, int WARPS_M, int WARPS_N, int BK, typename OutT>
inline void moe_grouped_gemm_cfg(const __nv_bfloat16* A, const int* a_rows, const __nv_bfloat16* W,
                                 OutT* C, const int* seg_off, const int* tile_start, int E,
                                 int max_m_tiles, int N, int K) {
  constexpr int BN = 128, STAGES = 2;
  constexpr int SMEM = STAGES * (BM + BN) * (BK + 8) * (int)sizeof(__nv_bfloat16);
  if (N % BN || K % BK) {
    std::fprintf(stderr, "moe_grouped_gemm needs N %% %d == 0 and K %% %d == 0\n", BN, BK);
    std::exit(EXIT_FAILURE);
  }
  auto kernel = moe_grouped_gemm_kernel<BM, BN, BK, WARPS_M, WARPS_N, STAGES, OutT>;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  const long long blocks = (long long)max_m_tiles * (N / BN);  // upper bound; extras exit
  kernel<<<static_cast<unsigned>(blocks), WARPS_M * WARPS_N * 32, SMEM>>>(A, a_rows, W, C, seg_off,
                                                                         tile_start, E, N, K);
}

template <typename OutT>
inline void moe_grouped_gemm(int bm, const __nv_bfloat16* A, const int* a_rows,
                             const __nv_bfloat16* W, OutT* C, const int* seg_off,
                             const int* tile_start, int E, int max_m_tiles, int N, int K) {
  if (bm == 32) moe_grouped_gemm_cfg<32, 1, 4, 64, OutT>(A, a_rows, W, C, seg_off, tile_start, E, max_m_tiles, N, K);
  else moe_grouped_gemm_cfg<128, 2, 2, 32, OutT>(A, a_rows, W, C, seg_off, tile_start, E, max_m_tiles, N, K);
}

// ============================ 5. combine ============================
// out[t, :] = sum_j w[t, j] * Y[perm_pos[t, j], :]   (one block per token, float4)
__global__ void moe_combine_kernel(const float* __restrict__ Y, const int* __restrict__ perm_pos,
                                   const float* __restrict__ topw, int k, int H,
                                   float* __restrict__ out) {
  const int t = blockIdx.x;
  for (int c = threadIdx.x * 4; c < H; c += blockDim.x * 4) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int j = 0; j < k; ++j) {
      const float wj = topw[(size_t)t * k + j];
      const float4 y = *reinterpret_cast<const float4*>(&Y[(size_t)perm_pos[(size_t)t * k + j] * H + c]);
      acc.x += wj * y.x; acc.y += wj * y.y; acc.z += wj * y.z; acc.w += wj * y.w;
    }
    *reinterpret_cast<float4*>(&out[(size_t)t * H + c]) = acc;
  }
}

// ============================ baseline helpers (per-expert loop) ============================
// Gather rows: dst[i] = src[idx[i]]  (rows of K bf16)
__global__ void moe_gather_rows_kernel(const __nv_bfloat16* __restrict__ src,
                                       const int* __restrict__ idx, int rows, int K,
                                       __nv_bfloat16* __restrict__ dst) {
  const int vpr = K / 8;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rows * vpr;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / vpr;
    const int c = static_cast<int>(i % vpr);
    reinterpret_cast<uint4*>(dst)[i] = reinterpret_cast<const uint4*>(src + (size_t)idx[r] * K)[c];
  }
}
// out[idx[i]] += w[i] * y[i]   (index_add_ with routing weight; no conflicts within one expert)
__global__ void moe_index_add_kernel(const float* __restrict__ y, const int* __restrict__ idx,
                                     const float* __restrict__ w, int rows, int H,
                                     float* __restrict__ out) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)rows * H;
       i += (size_t)gridDim.x * blockDim.x) {
    const size_t r = i / H;
    const int c = static_cast<int>(i % H);
    out[(size_t)idx[r] * H + c] += w[r] * y[i];
  }
}
