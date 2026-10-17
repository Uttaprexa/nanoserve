#pragma once
// Thin wrappers around the PTX instructions used by the tensor-core kernels.
#include <cstdint>

// ---- cp.async: global -> shared copy that bypasses registers (sm_80+) ----
__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ---- ldmatrix: each lane supplies one 16-byte row address; the warp loads
// four 8x8 b16 matrices straight into the register layout mma.sync expects.
// Lanes 8i..8i+7 address the rows of matrix i; result r[i] holds, for lane t,
// row t/4, columns 2*(t%4) and 2*(t%4)+1 of matrix i.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}
// Transposed variant: lane t gets rows 2*(t%4), 2*(t%4)+1 at column t/4.
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], const void* smem) {
  const unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(smem));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a)
               : "memory");
}

// ---- mma.sync m16n8k16: D(16x8,f32) += A(16x16,bf16,row) * B(16x8,bf16,col) ----
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4],
                                               uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
