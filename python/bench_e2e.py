"""Phase 6: nanoserve kernels inside a real LLM (Qwen2.5), against Hugging Face.

  [1] correctness: logits vs HF for each kernel on its own, then all together (bf16 and W4)
  [2] perplexity on WikiText-2: HF bf16 vs nanoserve bf16 vs nanoserve W4
  [3] attention: nanoserve FlashAttention vs PyTorch SDPA at the model's shapes
  [4] prompt processing (prefill) speed
  [5] generation (decode) speed
  [6] sample generations

Usage (from the repo root, with build/libnanoserve.so built by `make lib`):
    ~/venv/bin/python python/bench_e2e.py                         # Qwen2.5-1.5B
    ~/venv/bin/python python/bench_e2e.py --model Qwen/Qwen2.5-7B
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nanoserve.ops import Kernels, attention_ref  # noqa: E402
from nanoserve.qwen import Options, Qwen2Runner  # noqa: E402


def cuda_ms(fn, iters=5, warmup=1):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def load_text(tok):
    text, source, err = None, None, None
    try:
        from datasets import load_dataset
    except Exception as e:
        err = e
    else:
        for repo in ("Salesforce/wikitext", "wikitext"):  # the dataset moved to Salesforce/
            try:
                ds = load_dataset(repo, "wikitext-2-raw-v1", split="test")
                text, source = "\n\n".join(ds["text"]), "WikiText-2 (test)"
                break
            except Exception as e:
                err = e
    if text is None:  # offline fallback: still a fair A/B between implementations
        print(f"  (could not load WikiText-2: {err}; using README.md as text)")
        here = os.path.dirname(os.path.abspath(__file__))
        text, source = open(os.path.join(here, "..", "README.md")).read() * 20, "README.md"
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids, source


@torch.no_grad()
def hf_generate(model, ids, n_new):
    out = model(input_ids=ids[None], use_cache=True)
    past, tok = out.past_key_values, out.logits[0, -1].argmax()
    toks = [tok]
    for _ in range(n_new - 1):
        out = model(input_ids=tok.view(1, 1), past_key_values=past, use_cache=True)
        past, tok = out.past_key_values, out.logits[0, -1].argmax()
        toks.append(tok)
    return torch.stack(toks).tolist()


def compare_logits(a, b):
    a, b = a.float(), b.float()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item() * 100
    d = (a - b).abs()
    return top1, d.max().item(), d.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--ppl-chunks", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--groups", default="128,64,32", help="W4 group sizes for the perplexity sweep")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    gpu = torch.cuda.get_device_name()
    print(f"GPU: {gpu} | torch {torch.__version__} | model {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", device_map=dev).eval()
    cfg = model.config
    n_heads, n_kv = cfg.num_attention_heads, cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads
    print(f"  layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={n_heads} "
          f"kv_heads={n_kv} head_dim={head_dim} intermediate={cfg.intermediate_size}")
    kern = Kernels()
    ids, source = load_text(tok)
    ids = ids.to(dev)
    results = {"gpu": gpu, "model": args.model, "torch": torch.__version__}

    # ---------------- [1] correctness ----------------
    print("\n[1] Logits vs Hugging Face on the first", args.chunk, "tokens of", source)
    x = ids[: args.chunk]
    with torch.no_grad():
        ref = model(x[None]).logits[0]
    configs = [
        Options(False, False, False, False),
        Options(True, False, False, False),
        Options(False, True, False, False),
        Options(False, False, True, False),
        Options(False, False, False, True),
        Options(True, True, True, True),
    ]
    rows = []
    print(f"  {'configuration':34s} {'top-1 agree':>12s} {'max |dlogit|':>13s} {'mean |dlogit|':>14s}")
    for o in configs:
        r = Qwen2Runner(model, kern, o)
        t1, mx, mean = compare_logits(r.forward(x), ref)
        rows.append((o.label(), t1, mx, mean))
        print(f"  {o.label():34s} {t1:11.2f}% {mx:13.3f} {mean:14.4f}")
    t0 = time.time()
    runner_w4 = Qwen2Runner(model, kern, Options(w4=True))
    print(f"  (quantized all linear layers to 4 bits in {time.time() - t0:.1f} s)")
    t1, mx, mean = compare_logits(runner_w4.forward(x), ref)
    rows.append((runner_w4.opts.label(), t1, mx, mean))
    print(f"  {runner_w4.opts.label():34s} {t1:11.2f}% {mx:13.3f} {mean:14.4f}")
    results["correctness"] = rows
    runner = Qwen2Runner(model, kern, Options())

    # ---------------- [2] perplexity ----------------
    n = min(args.ppl_chunks, (ids.numel() - 1) // args.chunk)
    print(f"\n[2] Perplexity on {source}: {n} chunks x {args.chunk} tokens")

    def ppl(logits_fn):
        nll, count = 0.0, 0
        for c in range(n):
            chunk = ids[c * args.chunk:(c + 1) * args.chunk]
            logits = logits_fn(chunk)
            nll += F.cross_entropy(logits[:-1].float(), chunk[1:], reduction="sum").item()
            count += chunk.numel() - 1
        return math.exp(nll / count)

    with torch.no_grad():
        p_hf = ppl(lambda c: model(c[None]).logits[0])
    p_ns = ppl(runner.forward)
    p_w4 = ppl(runner_w4.forward)
    print(f"  HF bf16          {p_hf:8.3f}")
    print(f"  nanoserve bf16   {p_ns:8.3f}   ({100 * (p_ns / p_hf - 1):+.2f}% vs HF)")
    print(f"  nanoserve W4 g128 {p_w4:7.3f}   ({100 * (p_w4 / p_hf - 1):+.2f}% vs HF)   0.516 B/weight")
    sweep = {128: p_w4}
    for g in [int(v) for v in args.groups.split(",") if v.strip() and int(v) != 128]:
        r = Qwen2Runner(model, kern, Options(w4=True, group=g))
        sweep[g] = ppl(r.forward)
        print(f"  nanoserve W4 g{g:<3d} {sweep[g]:7.3f}   ({100 * (sweep[g] / p_hf - 1):+.2f}% vs HF)   "
              f"{0.5 + 2 / g:.3f} B/weight")
        del r
        torch.cuda.empty_cache()
    results["perplexity"] = {"source": source, "hf": p_hf, "nanoserve_bf16": p_ns,
                             "nanoserve_w4_by_group": sweep}

    # ---------------- [3] attention vs SDPA ----------------
    print(f"\n[3] Causal attention at model shapes (Hq={n_heads}, Hkv={n_kv}, D={head_dim}), TFLOP/s")
    print(f"  {'N':>6s} {'SDPA ms':>9s} {'TFLOP/s':>8s} {'ours ms':>9s} {'TFLOP/s':>8s} {'ours/SDPA':>10s} {'max diff':>9s}")
    att = []
    for N in (1024, 2048, 4096):
        q = torch.randn(N, n_heads * head_dim, device=dev, dtype=torch.bfloat16)
        k = torch.randn(N, n_kv * head_dim, device=dev, dtype=torch.bfloat16)
        v = torch.randn(N, n_kv * head_dim, device=dev, dtype=torch.bfloat16)
        rep = n_heads // n_kv
        qh = q.view(N, n_heads, head_dim).transpose(0, 1)[None].contiguous()
        kh = k.view(N, n_kv, head_dim).transpose(0, 1)[None].repeat_interleave(rep, 1).contiguous()
        vh = v.view(N, n_kv, head_dim).transpose(0, 1)[None].repeat_interleave(rep, 1).contiguous()
        t_sdpa = cuda_ms(lambda: F.scaled_dot_product_attention(qh, kh, vh, is_causal=True), 20, 3)
        t_ours = cuda_ms(lambda: kern.flash_attention(q, k, v, n_heads, n_kv, head_dim), 20, 3)
        flops = 2.0 * N * N * head_dim * n_heads  # 4 N^2 D H, halved by the causal mask
        diff = (kern.flash_attention(q, k, v, n_heads, n_kv, head_dim).float()
                - attention_ref(q, k, v, n_heads, n_kv, head_dim).float()).abs().max().item()
        att.append((N, t_sdpa, t_ours))
        print(f"  {N:6d} {t_sdpa:9.3f} {flops / t_sdpa / 1e9:8.1f} {t_ours:9.3f} {flops / t_ours / 1e9:8.1f} "
              f"{t_sdpa / t_ours:9.2f}x {diff:9.4f}")
    results["attention"] = att

    # ---------------- [4] prefill ----------------
    print("\n[4] Prompt processing (prefill, logits for every position), tokens/s")
    print(f"  {'T':>6s} {'HF':>10s} {'nanoserve bf16':>15s} {'nanoserve W4':>13s}")
    pre = []
    for T in (512, 2048):
        xT = ids[:T]
        with torch.no_grad():
            t_hf = cuda_ms(lambda: model(xT[None], use_cache=False), 5)
        t_ns = cuda_ms(lambda: runner.forward(xT), 5)
        t_w4 = cuda_ms(lambda: runner_w4.forward(xT), 5)
        pre.append((T, t_hf, t_ns, t_w4))
        print(f"  {T:6d} {T / t_hf * 1e3:10.0f} {T / t_ns * 1e3:15.0f} {T / t_w4 * 1e3:13.0f}")
    results["prefill_ms"] = pre

    # ---------------- [5] decode ----------------
    G, P = args.gen_tokens, 128
    prompt = ids[:P]
    print(f"\n[5] Generation: {P}-token prompt, {G} new tokens (greedy), decode tokens/s")

    def decode_rate(gen):
        t_all = cuda_ms(lambda: gen(G), 2)
        t_one = cuda_ms(lambda: gen(1), 2)
        return (G - 1) / ((t_all - t_one) / 1e3)

    r_hf = decode_rate(lambda n: hf_generate(model, prompt, n))
    r_ns = decode_rate(lambda n: runner.generate(prompt, n))
    r_w4 = decode_rate(lambda n: runner_w4.generate(prompt, n))
    print(f"  HF bf16          {r_hf:7.1f} tok/s")
    print(f"  nanoserve bf16   {r_ns:7.1f} tok/s   ({r_ns / r_hf:.2f}x)")
    print(f"  nanoserve W4     {r_w4:7.1f} tok/s   ({r_w4 / r_hf:.2f}x)")
    results["decode_tok_s"] = {"hf": r_hf, "nanoserve_bf16": r_ns, "nanoserve_w4": r_w4}

    # ---------------- [6] samples ----------------
    text = "The key idea behind FlashAttention is"
    p_ids = tok(text, return_tensors="pt").input_ids[0].to(dev)
    print(f'\n[6] Greedy continuations of "{text}"')
    g_hf = hf_generate(model, p_ids, 40)
    g_ref = Qwen2Runner(model, kern, Options(False, False, False, False)).generate(p_ids, 40)
    g_ns = runner.generate(p_ids, 40)
    g_w4 = runner_w4.generate(p_ids, 40)

    def match(a, b):
        return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))

    for name, g in (("HF bf16", g_hf), ("runner, torch ops", g_ref), ("nanoserve bf16", g_ns),
                    ("nanoserve W4", g_w4)):
        print(f"  {name:18s} {tok.decode(g)!r}")
    m_ref, m_ns, m_ns_ref = match(g_hf, g_ref), match(g_hf, g_ns), match(g_ref, g_ns)
    print(f"  tokens matching HF: runner with torch ops {m_ref}/40, with nanoserve kernels {m_ns}/40")
    print(f"  nanoserve kernels vs the same runner with torch ops: {m_ns_ref}/40")
    results["samples"] = {"prompt": text, "hf": tok.decode(g_hf), "runner_torch_ops": tok.decode(g_ref),
                          "nanoserve_bf16": tok.decode(g_ns), "nanoserve_w4": tok.decode(g_w4),
                          "match_hf_torch_ops": m_ref, "match_hf_nanoserve": m_ns,
                          "match_torch_ops_nanoserve": m_ns_ref}

    os.makedirs("results", exist_ok=True)
    tag = args.model.split("/")[-1]
    path = f"results/e2e_{tag}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
