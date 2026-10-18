"""A small Qwen2 (Llama-architecture) runner on top of a Hugging Face checkpoint.

The HF model is only used for its weights and config; the forward pass is written here so each
operation can be switched between a nanoserve kernel and the PyTorch reference:

    use_norm   fused residual-add + RMSNorm   (Phase 3)
    use_rope   in-place RoPE                  (Phase 3)
    use_attn   FlashAttention-2, prompt only  (Phase 2; decode attention uses PyTorch SDPA)
    use_swiglu fused SwiGLU                   (Phase 3)
    w4         4-bit linear layers            (Phase 4; embeddings and lm_head stay bf16)

Batch size 1. Per layer:
    h = norm(x)                    (layer 0 only; later layers get it fused below)
    q, k, v = h Wq + bq, h Wk + bk, h Wv + bv;   rope(q, k)
    a = attention(q, k, v);        x = x + a Wo;  h = norm_post(x)     <- fused add + norm
    m = (silu(h Wg) * (h Wu)) Wd;  x = x + m;     h = norm_next(x)     <- fused add + norm
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .ops import Kernels, attention_ref, rmsnorm_ref, rope_ref, swiglu_ref


@dataclass
class Options:
    use_norm: bool = True
    use_rope: bool = True
    use_attn: bool = True
    use_swiglu: bool = True
    w4: bool = False
    group: int = 128

    def label(self):
        if not any([self.use_norm, self.use_rope, self.use_attn, self.use_swiglu, self.w4]):
            return "torch ops"
        parts = [n for n, on in [("norm", self.use_norm), ("rope", self.use_rope),
                                 ("flash", self.use_attn), ("swiglu", self.use_swiglu)] if on]
        return "+".join(parts) + (" | W4" if self.w4 else " | bf16")


class Qwen2Runner:
    def __init__(self, hf_model, kernels: Kernels, opts: Options = Options()):
        cfg = hf_model.config
        self.k, self.opts = kernels, opts
        self.H = cfg.hidden_size
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", None) or self.H // self.n_heads
        self.eps = cfg.rms_norm_eps
        # RoPE settings moved between Transformers versions: older configs have rope_theta and an
        # optional rope_scaling dict; newer ones keep both in rope_parameters and always fill in
        # rope_type = "default". Only plain (unscaled) RoPE is implemented here.
        rs = getattr(cfg, "rope_scaling", None)
        rp = getattr(cfg, "rope_parameters", None)
        rs = rs if isinstance(rs, dict) else {}
        rp = rp if isinstance(rp, dict) else {}
        rope_type = rs.get("rope_type") or rs.get("type") or rp.get("rope_type") or "default"
        if rope_type != "default":
            raise NotImplementedError(f"rope_type {rope_type!r} is not supported by this runner")
        theta = rp.get("rope_theta") or rs.get("rope_theta") or getattr(cfg, "rope_theta", None) or 10000.0
        dev = hf_model.device
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64,
                                                      device=dev).float() / self.head_dim))
        m = hf_model.model
        self.embed = m.embed_tokens.weight
        self.final_norm = m.norm.weight
        self.lm_head = hf_model.lm_head.weight
        self.layers = []
        for layer in m.layers:
            at, mlp = layer.self_attn, layer.mlp
            L = {
                "ln_in": layer.input_layernorm.weight,
                "ln_post": layer.post_attention_layernorm.weight,
                "bq": at.q_proj.bias, "bk": at.k_proj.bias, "bv": at.v_proj.bias,
                "w": {"q": at.q_proj.weight, "k": at.k_proj.weight, "v": at.v_proj.weight,
                      "o": at.o_proj.weight, "g": mlp.gate_proj.weight, "u": mlp.up_proj.weight,
                      "d": mlp.down_proj.weight},
            }
            self.layers.append(L)
        self.w4 = None
        if opts.w4:
            self.quantize()
        self.cache_k = self.cache_v = None

    # ------------------------------------------------------------------ weights
    def quantize(self):
        self.w4 = []
        for L in self.layers:
            self.w4.append({n: self.k.w4_quantize(W, self.opts.group) for n, W in L["w"].items()})
        torch.cuda.synchronize()

    def linear(self, x, li, name):
        if self.opts.w4:
            Wq, S = self.w4[li][name]
            return self.k.w4_linear(x, Wq, S, self.opts.group)
        return F.linear(x, self.layers[li]["w"][name])

    # ------------------------------------------------------------------ ops
    def norm(self, x, r, w):
        if self.opts.use_norm:
            return self.k.rmsnorm(x, r, w, self.eps)
        return rmsnorm_ref(x, r, w, self.eps)

    def rope(self, q, k, pos0):
        f = self.k.rope if self.opts.use_rope else rope_ref
        f(q, k, self.inv_freq, self.n_heads, self.n_kv, self.head_dim, pos0)

    def swiglu(self, g, u):
        return self.k.swiglu(g, u) if self.opts.use_swiglu else swiglu_ref(g, u)

    # ------------------------------------------------------------------ cache
    def alloc_cache(self, max_len):
        shape = (len(self.layers), self.n_kv, max_len, self.head_dim)
        self.cache_k = torch.empty(shape, dtype=torch.bfloat16, device=self.embed.device)
        self.cache_v = torch.empty_like(self.cache_k)

    def _store_kv(self, li, k, v, pos0):
        T = k.shape[0]
        self.cache_k[li, :, pos0:pos0 + T] = k.view(T, self.n_kv, self.head_dim).transpose(0, 1)
        self.cache_v[li, :, pos0:pos0 + T] = v.view(T, self.n_kv, self.head_dim).transpose(0, 1)

    def _attend_cache(self, li, q, n):
        """Decode attention for one new token over n cached positions (PyTorch SDPA)."""
        qh = q.view(1, self.n_heads, 1, self.head_dim)
        kh = self.cache_k[li, :, :n][None]
        vh = self.cache_v[li, :, :n][None]
        rep = self.n_heads // self.n_kv
        if rep > 1:
            kh, vh = kh.repeat_interleave(rep, dim=1), vh.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(qh, kh, vh)
        return o.reshape(1, self.n_heads * self.head_dim)

    # ------------------------------------------------------------------ forward
    @torch.no_grad()
    def forward(self, ids, pos0=0, use_cache=False, last_only=False):
        """ids: [T] token ids. Prompt mode (pos0 == 0) or one decode step (T == 1, pos0 > 0)."""
        T = ids.shape[0]
        decode = pos0 > 0
        x = F.embedding(ids, self.embed).contiguous()        # residual stream [T, H]
        h = self.norm(x, None, self.layers[0]["ln_in"])
        for li, L in enumerate(self.layers):
            q = self.linear(h, li, "q")
            k = self.linear(h, li, "k")
            v = self.linear(h, li, "v")
            if L["bq"] is not None:
                q += L["bq"]; k += L["bk"]; v += L["bv"]
            self.rope(q, k, pos0)
            if use_cache:
                self._store_kv(li, k, v, pos0)
            if decode:
                a = self._attend_cache(li, q, pos0 + T)
            elif self.opts.use_attn and T % 64 == 0:
                a = self.k.flash_attention(q, k, v, self.n_heads, self.n_kv, self.head_dim)
            else:
                a = attention_ref(q, k, v, self.n_heads, self.n_kv, self.head_dim)
            h = self.norm(x, self.linear(a, li, "o"), L["ln_post"])
            act = self.swiglu(self.linear(h, li, "g"), self.linear(h, li, "u"))
            nxt = self.layers[li + 1]["ln_in"] if li + 1 < len(self.layers) else self.final_norm
            h = self.norm(x, self.linear(act, li, "d"), nxt)
        if last_only:
            h = h[-1:]
        return F.linear(h, self.lm_head)

    @torch.no_grad()
    def generate(self, prompt_ids, n_new):
        """Greedy decoding. Returns the generated ids (list)."""
        T = prompt_ids.shape[0]
        self.alloc_cache(T + n_new)
        logits = self.forward(prompt_ids, 0, use_cache=True, last_only=True)
        tok = logits[-1].argmax()
        out = [tok]
        for i in range(1, n_new):
            logits = self.forward(tok.view(1), T + i - 1, use_cache=True)
            tok = logits[-1].argmax()
            out.append(tok)  # stays on the GPU: no host sync per token
        return torch.stack(out).tolist()
