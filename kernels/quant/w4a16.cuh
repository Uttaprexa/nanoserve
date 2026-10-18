#pragma once
// W4A16: 4-bit weights, bf16 activations, fp32 accumulation.
//   Y[M, N] (fp32) = X[M, K] (bf16, row-major) @ W^T,   W: [N, K] (nn.Linear layout)
//
// Weight format (group-wise symmetric, like GPTQ/AWQ without zero-point tensors):
//   Wq [N, K/8] uint32: word j of row n packs k = 8j .. 8j+7, value k = 8j+i in bits 4i..4i+3
//   S  [N, K/G] bf16:   one scale per G consecutive weights along K
//   w[n, k] = (q - 8) * S[n, k / G],  q in [0, 15]
// Memory per weight: 0.5 B + 2/G B  (G = 128 -> 0.516 B, vs 2 B for bf16: 3.9x less).
//
// Two kernels:
//   w4a16_gemv  decode (M <= 8): memory-bound; R output rows per warp, 16-byte weight loads
//   w4a16_gemm  prefill (large M): compute-bound; cp.async stages packed weights in smem,
//               threads dequantize them to a bf16 tile, then the Phase-1 ldmatrix + mma.sync
//               path runs unchanged.
#include <cuda_bf16.h>

#include "../fused/vec.cuh"
#include "../hgemm/ptx.cuh"

__device__ __forceinline__ float w4_warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ void store_out(float* p, float v) { *p = v; }
__device__ __forceinline__ void store_out(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

// 8 packed nibbles -> 8 dequantized floats
__device__ __forceinline__ void dequant8(uint32_t q, float s, float* f) {
#pragma unroll
  for (int i = 0; i < 8; ++i) f[i] = static_cast<float>(static_cast<int>((q >> (4 * i)) & 0xF) - 8) * s;
}

// ============================ decode: GEMV ============================
// Block = 8 warps; each warp owns R consecutive output rows. Each lane streams 16 bytes
// (32 weights) per row per step, so a warp reads 512 contiguous bytes of each of its rows.
//
// Why R rows per warp: X (M x K, tiny) is re-read from L1 by every warp. With one row per
// warp, ncu showed L1/TEX at 89% and DRAM at only 38% for M = 8: the kernel was L1-bound,
// not memory-bound. Loading each X chunk once and reusing it for R rows cuts L1 traffic
// for X by R. M = 1 needs no reuse (X traffic is small), so it keeps R = 1.
template <int M, int R, typename OutT>
__global__ void __launch_bounds__(256)
    w4a16_gemv_kernel(const __nv_bfloat16* __restrict__ X, const uint32_t* __restrict__ Wq,
                      const __nv_bfloat16* __restrict__ S, OutT* __restrict__ Y, int N, int K,
                      int G) {
  const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int n0 = (blockIdx.x * 8 + warp) * R;
  if (n0 >= N) return;
  const int nvec = K / 32, KG = K / G;

  float acc[R][M];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) acc[r][m] = 0.f;

#pragma unroll 2
  for (int v = lane; v < nvec; v += 32) {
    const int k0 = v * 32;
    uint4 w[R];
    float s[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const bool ok = n0 + r < N;
      w[r] = ok ? __ldg(reinterpret_cast<const uint4*>(Wq + (size_t)(n0 + r) * (K / 8)) + v)
                : make_uint4(0u, 0u, 0u, 0u);
      s[r] = ok ? __bfloat162float(S[(size_t)(n0 + r) * KG + k0 / G]) : 0.f;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {  // 8 weights per sub-step
      float xf[M][8];
#pragma unroll
      for (int m = 0; m < M; ++m)  // each X chunk loaded once, reused for R rows
        unpack8(__ldg(reinterpret_cast<const uint4*>(X + (size_t)m * K + k0) + j), xf[m]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint32_t q = j == 0 ? w[r].x : j == 1 ? w[r].y : j == 2 ? w[r].z : w[r].w;
        float wf[8];
        dequant8(q, s[r], wf);
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r][m] += xf[m][i] * wf[i];
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int m = 0; m < M; ++m) {
      const float v = w4_warp_sum(acc[r][m]);
      if (lane == 0 && n0 + r < N) store_out(&Y[(size_t)m * N + n0 + r], v);
    }
}

template <typename OutT>
inline void w4a16_gemv(const __nv_bfloat16* X, const uint32_t* Wq, const __nv_bfloat16* S, OutT* Y,
                       int M, int N, int K, int G, cudaStream_t stream = 0) {
  if (K % 32 || G % 32 || K % G) {
    std::fprintf(stderr, "w4a16_gemv needs K %% 32 == 0, G %% 32 == 0, K %% G == 0\n");
    std::exit(EXIT_FAILURE);
  }
  const dim3 block(256);
  auto grid = [&](int R) { return dim3((N + 8 * R - 1) / (8 * R)); };
  switch (M) {
    case 1: w4a16_gemv_kernel<1, 1, OutT><<<grid(1), block, 0, stream>>>(X, Wq, S, Y, N, K, G); break;
    case 2: w4a16_gemv_kernel<2, 4, OutT><<<grid(4), block, 0, stream>>>(X, Wq, S, Y, N, K, G); break;
    case 4: w4a16_gemv_kernel<4, 4, OutT><<<grid(4), block, 0, stream>>>(X, Wq, S, Y, N, K, G); break;
    case 8: w4a16_gemv_kernel<8, 4, OutT><<<grid(4), block, 0, stream>>>(X, Wq, S, Y, N, K, G); break;
    default:
      std::fprintf(stderr, "w4a16_gemv supports M in {1, 2, 4, 8}\n");
      std::exit(EXIT_FAILURE);
  }
}

// ============================ prefill: tensor-core GEMM ============================
// Per K tile (BK = 32, inside one quantization group):
//   cp.async  A tile (bf16) and packed W tile (BN x BK/8 words) into stage kt % STAGES
//   dequant   packed words -> bf16 tile Bd[n][k] (single buffer), one 16-byte store per word
//   compute   A via ldmatrix.x4, B via ldmatrix.x4 (non-trans: rows of Bd are n, k contiguous,
//             which is exactly mma's "col" B operand), mma.sync m16n8k16
// REQUIRES: M % BM == 0, N % BN == 0, K % BK == 0, G % BK == 0.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N, int STAGES, typename OutT>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32)
    w4a16_gemm_kernel(int M, int N, int K, const __nv_bfloat16* __restrict__ A,
                      const uint32_t* __restrict__ Wq, const __nv_bfloat16* __restrict__ S, int G,
                      OutT* __restrict__ C) {
  constexpr int NUM_THREADS = WARPS_M * WARPS_N * 32;
  constexpr int WM = BM / WARPS_M, WN = BN / WARPS_N;
  constexpr int MI = WM / 16, NI = WN / 8;
  constexpr int LDA = BK + 8, LDB = BK + 8;
  constexpr int A_STAGE = BM * LDA;          // bf16 elements
  constexpr int W_WORDS = BK / 8;            // packed words per row per tile
  constexpr int W_STAGE = BN * W_WORDS;      // uint32 per stage
  constexpr int A_VEC = BM * BK / 8;         // 16-byte copies per A tile
  constexpr int W_VEC = BN * BK / 32;        // 16-byte copies per W tile
  static_assert(BK % 32 == 0 && NI % 2 == 0 && WM % 16 == 0, "tile shapes");
  static_assert(A_VEC % NUM_THREADS == 0, "even A load split");

  extern __shared__ __align__(128) unsigned char smem_raw[];
  __nv_bfloat16* As = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  uint32_t* Ws = reinterpret_cast<uint32_t*>(As + STAGES * A_STAGE);
  __nv_bfloat16* Bd = reinterpret_cast<__nv_bfloat16*>(Ws + STAGES * W_STAGE);

  const int tid = threadIdx.x, lane = tid % 32, warpIdx = tid / 32;
  const int warpRow = warpIdx / WARPS_N, warpCol = warpIdx % WARPS_N;
  const int lq = lane / 8, lr = lane % 8;
  const int KW = K / 8, KG = K / G;

  A += (size_t)blockIdx.y * BM * K;
  Wq += (size_t)blockIdx.x * BN * KW;
  S += (size_t)blockIdx.x * BN * KG;

  auto load_tile = [&](int stage, int bk) {
    __nv_bfloat16* as = As + stage * A_STAGE;
    uint32_t* ws = Ws + stage * W_STAGE;
#pragma unroll
    for (int p = 0; p < A_VEC / NUM_THREADS; ++p) {
      const int v = tid + p * NUM_THREADS;
      const int row = v / (BK / 8), col = (v % (BK / 8)) * 8;
      cp_async_16(&as[row * LDA + col], &A[(size_t)row * K + bk + col]);
    }
    for (int v = tid; v < W_VEC; v += NUM_THREADS) {
      const int row = v / (BK / 32), c = v % (BK / 32);
      cp_async_16(&ws[row * W_WORDS + c * 4], &Wq[(size_t)row * KW + bk / 8 + c * 4]);
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
    __syncthreads();  // tile kt landed for all; every warp finished reading Bd (tile kt-1)
    const int next = kt + STAGES - 1;
    if (next < KT) load_tile(next % STAGES, next * BK);
    cp_async_commit();

    // dequantize packed tile -> Bd[n][k]
    const uint32_t* ws = Ws + (kt % STAGES) * W_STAGE;
    const int grp = (kt * BK) / G;
    for (int v = tid; v < BN * W_WORDS; v += NUM_THREADS) {
      const int row = v / W_WORDS, c = v % W_WORDS;
      float f[8];
      dequant8(ws[row * W_WORDS + c], __bfloat162float(S[(size_t)row * KG + grp]), f);
      *reinterpret_cast<uint4*>(&Bd[row * LDB + c * 8]) = pack8(f);
    }
    __syncthreads();

    const __nv_bfloat16* as = As + (kt % STAGES) * A_STAGE;
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
        ldmatrix_x4(t4, &Bd[(warpCol * WN + j * 8 + lr + 8 * (lq / 2)) * LDB + kk + 8 * (lq % 2)]);
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

  const int g = lane / 4, t = lane % 4;
#pragma unroll
  for (int i = 0; i < MI; ++i)
#pragma unroll
    for (int j = 0; j < NI; ++j) {
      const int row = blockIdx.y * BM + warpRow * WM + i * 16 + g;
      const int col = blockIdx.x * BN + warpCol * WN + j * 8 + 2 * t;
      if constexpr (sizeof(OutT) == 4) {
        *reinterpret_cast<float2*>(&C[(size_t)row * N + col]) = make_float2(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<float2*>(&C[(size_t)(row + 8) * N + col]) =
            make_float2(acc[i][j][2], acc[i][j][3]);
      } else {
        *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)row * N + col]) =
            __floats2bfloat162_rn(acc[i][j][0], acc[i][j][1]);
        *reinterpret_cast<__nv_bfloat162*>(&C[(size_t)(row + 8) * N + col]) =
            __floats2bfloat162_rn(acc[i][j][2], acc[i][j][3]);
      }
    }
}

template <int WARPS_M, int WARPS_N, typename OutT>
inline void w4a16_gemm_cfg(const __nv_bfloat16* A, const uint32_t* Wq, const __nv_bfloat16* S,
                           OutT* C, int M, int N, int K, int G, cudaStream_t stream) {
  constexpr int BM = 128, BN = 128, BK = 32, STAGES = 2;
  constexpr int SMEM = STAGES * BM * (BK + 8) * 2 + STAGES * BN * (BK / 8) * 4 + BN * (BK + 8) * 2;
  if (M % BM || N % BN || K % BK || G % BK) {
    std::fprintf(stderr, "w4a16_gemm needs M%%%d, N%%%d, K%%%d, G%%%d == 0\n", BM, BN, BK, BK);
    std::exit(EXIT_FAILURE);
  }
  auto kernel = w4a16_gemm_kernel<BM, BN, BK, WARPS_M, WARPS_N, STAGES, OutT>;
  static bool configured = false;
  if (!configured) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
    configured = true;
  }
  dim3 grid(N / BN, M / BM);
  kernel<<<grid, WARPS_M * WARPS_N * 32, SMEM, stream>>>(M, N, K, A, Wq, S, G, C);
}
// v1: 4 warps, 64x64 warp tiles (Phase 1's fastest config). v2: 8 warps, 64x32 warp tiles.
template <typename OutT>
inline void w4a16_gemm(const __nv_bfloat16* A, const uint32_t* Wq, const __nv_bfloat16* S, OutT* C,
                       int M, int N, int K, int G, int variant = 1, cudaStream_t stream = 0) {
  if (variant == 1) w4a16_gemm_cfg<2, 2, OutT>(A, Wq, S, C, M, N, K, G, stream);
  else w4a16_gemm_cfg<2, 4, OutT>(A, Wq, S, C, M, N, K, G, stream);
}

// ============================ quantization ============================
// One thread per (row, group): scale = max|w| / 7 (rounded to bf16), q = round(w / scale) + 8.
__global__ void quantize_w4_kernel(const __nv_bfloat16* __restrict__ W, uint32_t* __restrict__ Wq,
                                   __nv_bfloat16* __restrict__ S, int N, int K, int g_size) {
  const int KG = K / g_size;
  const size_t total = (size_t)N * KG;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const size_t n = idx / KG;
    const int g = static_cast<int>(idx % KG);
    const __nv_bfloat16* w = W + n * K + (size_t)g * g_size;
    float mx = 0.f;
    for (int i = 0; i < g_size; ++i) mx = fmaxf(mx, fabsf(__bfloat162float(w[i])));
    const __nv_bfloat16 sb = __float2bfloat16(mx > 0.f ? mx / 7.f : 1.f);
    const float sf = __bfloat162float(sb);
    S[idx] = sb;
    for (int j = 0; j < g_size / 8; ++j) {
      uint32_t word = 0;
      for (int i = 0; i < 8; ++i) {
        const int q = min(15, max(0, __float2int_rn(__bfloat162float(w[j * 8 + i]) / sf) + 8));
        word |= static_cast<uint32_t>(q) << (4 * i);
      }
      Wq[n * (K / 8) + (size_t)g * (g_size / 8) + j] = word;
    }
  }
}
inline void w4_quantize(const __nv_bfloat16* W, uint32_t* Wq, __nv_bfloat16* S, int N, int K, int G,
                        cudaStream_t stream = 0) {
  quantize_w4_kernel<<<grid_for((size_t)N * (K / G)), 256, 0, stream>>>(W, Wq, S, N, K, G);
}
