"""ctypes bindings to build/libnanoserve.so, plus PyTorch reference versions of each op.

Each kernel wrapper takes and returns torch tensors on the current CUDA stream. The reference
versions follow Hugging Face's Qwen2/Llama code, so swapping one for the other isolates a kernel.
"""
import ctypes
import os

import torch
import torch.nn.functional as F

_LIB_DEFAULT = os.path.join(os.path.dirname(__file__), "..", "..", "build", "libnanoserve.so")


class Kernels:
    def __init__(self, path: str = _LIB_DEFAULT):
        self.lib = ctypes.CDLL(os.path.abspath(path))
        vp, i32, f32, i64 = ctypes.c_void_p, ctypes.c_int, ctypes.c_float, ctypes.c_longlong
        sig = {
            "ns_rmsnorm": [vp, vp, vp, vp, i32, i32, f32, vp],
            "ns_rope": [vp, vp, vp, i32, i32, i32, i32, i32, i64, i64, vp],
            "ns_swiglu": [vp, vp, vp, i32, i32, vp],
            "ns_flash_fwd": [vp, vp, vp, vp, i32, i32, i32, i32, i64, i64, i64, i32, i32, vp],
            "ns_w4_quantize": [vp, vp, vp, i32, i32, i32, vp],
            "ns_w4_linear": [vp, vp, vp, vp, i32, i32, i32, i32, vp],
        }
        for name, args in sig.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int

    @staticmethod
    def _stream():
        return ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    @staticmethod
    def _check(rc: int, name: str):
        if rc != 0:
            raise RuntimeError(f"{name} failed with code {rc}")

    # x: [T, H] residual stream, updated in place when r is given. Returns rmsnorm(x) * w.
    def rmsnorm(self, x, r, w, eps):
        assert x.is_contiguous() and x.dtype == torch.bfloat16
        y = torch.empty_like(x)
        rc = self.lib.ns_rmsnorm(x.data_ptr(), None if r is None else r.data_ptr(), w.data_ptr(),
                                 y.data_ptr(), x.shape[0], x.shape[1], eps, self._stream())
        self._check(rc, "ns_rmsnorm")
        return y

    # In place on q [T, Hq*D] and k [T, Hk*D] (row stride may exceed Hq*D / Hk*D).
    def rope(self, q, k, inv_freq, n_heads, n_kv_heads, head_dim, pos0):
        rc = self.lib.ns_rope(q.data_ptr(), k.data_ptr(), inv_freq.data_ptr(), q.shape[0], n_heads,
                              n_kv_heads, head_dim, pos0, q.stride(0), k.stride(0), self._stream())
        self._check(rc, "ns_rope")

    def swiglu(self, gate, up):
        assert gate.is_contiguous() and up.is_contiguous()
        out = torch.empty_like(gate)
        rc = self.lib.ns_swiglu(gate.data_ptr(), up.data_ptr(), out.data_ptr(), gate.shape[0],
                                gate.shape[1], self._stream())
        self._check(rc, "ns_swiglu")
        return out

    # q [N, Hq*D], k/v [N, Hk*D] token-major (any row stride); returns o [N, Hq*D].
    def flash_attention(self, q, k, v, n_heads, n_kv_heads, head_dim, causal=True):
        N = q.shape[0]
        variant = 2 if (N % 128 == 0 and N > 1024) else 1
        assert k.stride(0) == v.stride(0)
        o = torch.empty((N, n_heads * head_dim), dtype=q.dtype, device=q.device)
        rc = self.lib.ns_flash_fwd(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), N,
                                   n_heads, n_kv_heads, head_dim, q.stride(0), k.stride(0),
                                   o.stride(0), int(causal), variant, self._stream())
        self._check(rc, "ns_flash_fwd")
        return o

    def w4_quantize(self, W, group=128):
        N, K = W.shape
        Wq = torch.empty((N, K // 8), dtype=torch.int32, device=W.device)
        S = torch.empty((N, K // group), dtype=torch.bfloat16, device=W.device)
        rc = self.lib.ns_w4_quantize(W.contiguous().data_ptr(), Wq.data_ptr(), S.data_ptr(), N, K,
                                     group, self._stream())
        self._check(rc, "ns_w4_quantize")
        return Wq, S

    # x [M, K] -> [M, N]. Pads M up to a supported size (1, 2, 4, 8 or a multiple of 128).
    def w4_linear(self, x, Wq, S, group=128):
        M, K = x.shape
        N = Wq.shape[0]
        Mp = next((m for m in (1, 2, 4, 8) if M <= m), None) or ((M + 127) // 128) * 128
        if Mp != M:
            x = torch.cat([x, x.new_zeros(Mp - M, K)])
        x = x.contiguous()
        y = torch.empty((Mp, N), dtype=torch.bfloat16, device=x.device)
        rc = self.lib.ns_w4_linear(x.data_ptr(), Wq.data_ptr(), S.data_ptr(), y.data_ptr(), Mp, N,
                                   K, group, self._stream())
        self._check(rc, "ns_w4_linear")
        return y[:M]


# ----------------------------- PyTorch references -----------------------------
def rmsnorm_ref(x, r, w, eps):
    """HF Qwen2RMSNorm, with the residual add done first (in place) like the fused kernel."""
    if r is not None:
        x.add_(r)
    h = x.float()
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return w * h.to(x.dtype)


def rope_ref(q, k, inv_freq, n_heads, n_kv_heads, head_dim, pos0):
    """HF apply_rotary_pos_emb (rotate_half), in place on views of q [T, Hq*D] and k [T, Hk*D]."""
    T = q.shape[0]
    pos = torch.arange(pos0, pos0 + T, device=q.device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos().to(q.dtype)[:, None, :], emb.sin().to(q.dtype)[:, None, :]

    def apply(x, H):
        xv = x.view(T, H, head_dim) if x.is_contiguous() else x.unflatten(1, (H, head_dim))
        x1, x2 = xv[..., : head_dim // 2], xv[..., head_dim // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        xv.copy_(xv * cos + rot * sin)

    apply(q, n_heads)
    apply(k, n_kv_heads)


def attention_ref(q, k, v, n_heads, n_kv_heads, head_dim, causal=True):
    """PyTorch SDPA on token-major q/k/v; GQA by repeating K/V heads."""
    N = q.shape[0]
    qh = q.reshape(N, n_heads, head_dim).transpose(0, 1)[None]
    kh = k.reshape(N, n_kv_heads, head_dim).transpose(0, 1)[None]
    vh = v.reshape(N, n_kv_heads, head_dim).transpose(0, 1)[None]
    rep = n_heads // n_kv_heads
    if rep > 1:
        kh, vh = kh.repeat_interleave(rep, dim=1), vh.repeat_interleave(rep, dim=1)
    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal)
    return o[0].transpose(0, 1).reshape(N, n_heads * head_dim)


def swiglu_ref(gate, up):
    return F.silu(gate) * up
