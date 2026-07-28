"""M8 - Model-level optimization: the KV cache (small, CPU-only, no GPU needed).

The single most important *model-level* inference optimization. In autoregressive
decoding, every new token attends to ALL previous tokens. The naive way recomputes
the keys/values for the whole sequence at every step -> work grows like O(n^2) per
step, O(n^3) total. The KV cache stores past keys/values and only computes them for
the ONE new token -> O(n) per step, O(n^2) total. Same math, same output, far less work.

This file implements ONE causal self-attention block twice - once without a cache,
once with - proves they produce identical tokens, and times both as the generated
length grows so you SEE the quadratic-vs-linear gap open up. Pure PyTorch on CPU.

Run:  python3 m8_kv_cache.py
Deps: torch (CPU build fine)
"""
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import time
import torch
import torch.nn.functional as F
import sysinfo

torch.manual_seed(0)
D, H = 256, 4          # model dim, heads
HD = D // H
VOCAB = 1000
scale = 1.0 / (HD ** 0.5)

# One fixed attention block + a projection to "logits" so we can pick a next token.
Wq = torch.randn(D, D) * 0.02
Wk = torch.randn(D, D) * 0.02
Wv = torch.randn(D, D) * 0.02
Wo = torch.randn(D, D) * 0.02
Wemb = torch.randn(VOCAB, D) * 0.02    # token id -> embedding
Wout = torch.randn(D, VOCAB) * 0.02    # hidden -> logits

def heads(x):                          # (T, D) -> (H, T, HD)
    T = x.shape[0]
    return x.view(T, H, HD).transpose(0, 1)

def merge(x):                          # (H, T, HD) -> (T, D)
    T = x.shape[1]
    return x.transpose(0, 1).contiguous().view(T, D)

def next_token(hidden_last):           # greedy pick from the last position
    return int(torch.argmax(hidden_last @ Wout))

def generate_nocache(prompt_ids, gen_len):
    """Recompute attention over the ENTIRE sequence every step (the naive way)."""
    ids = list(prompt_ids)
    out = []
    for _ in range(gen_len):
        x = Wemb[torch.tensor(ids)]                      # (T, D) - rebuilt each step
        q, k, v = heads(x @ Wq), heads(x @ Wk), heads(x @ Wv)
        T = x.shape[0]
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        att = torch.softmax((q @ k.transpose(-1, -2)) * scale + mask, dim=-1)
        h = merge(att @ v) @ Wo                         # (T, D)
        nid = next_token(h[-1])
        ids.append(nid); out.append(nid)
    return out


def generate_cache(prompt_ids, gen_len):
    """Keep K,V for past tokens; only compute the new token's q,k,v each step."""
    x = Wemb[torch.tensor(list(prompt_ids))]            # prefill once
    K, V = heads(x @ Wk), heads(x @ Wv)                 # (H, P, HD) cached
    q = heads(x @ Wq)
    att = torch.softmax((q @ K.transpose(-1, -2)) * scale
                        + torch.triu(torch.full((x.shape[0],) * 2, float("-inf")), 1), dim=-1)
    h_last = (merge(att @ V) @ Wo)[-1]
    out = []
    for _ in range(gen_len):
        nid = next_token(h_last); out.append(nid)
        xn = Wemb[nid].view(1, D)                       # ONE new token
        qn, kn, vn = heads(xn @ Wq), heads(xn @ Wk), heads(xn @ Wv)
        K = torch.cat([K, kn], dim=1); V = torch.cat([V, vn], dim=1)  # append to cache
        attn = torch.softmax(qn @ K.transpose(-1, -2) * scale, dim=-1)  # new token sees all
        h_last = (merge(attn @ V) @ Wo)[-1]
    return out


def _time(fn):
    t = time.perf_counter(); r = fn(); return r, time.perf_counter() - t


# --------------------------------------------------------------
# STUDENT TODO: the cache trades memory for speed. Compute the KV-cache size for a
# real model; bytes = 2 (K&V) * layers * seq_len * n_kv_heads * head_dim * dtype_bytes.
# Print it for Llama-3-8B (32 layers, 8 KV heads, head_dim 128, fp16) at seq_len 4096.
# That number is why "paged attention" (M7/vLLM) and quantized KV caches exist.
def student_kv_cache_size():
    raise NotImplementedError("Compute and print the KV-cache size in GB for a real model.")

# --------------------------------------------------------------

if __name__ == "__main__":
    sysinfo.header()
    prompt = list(range(128))        # a 128-"token" prompt

    print("Correctness: do both paths generate the SAME tokens?")
    a = generate_nocache(prompt, 32)
    b = generate_cache(prompt, 32)
    print(f"  identical tokens: {a == b}\n")

    print(f"{'gen_len':>8}{'no-cache(ms)':>14}{'cache(ms)':>12}{'speedup':>9}")
    print("-" * 43)
    for g in (32, 64, 128, 256):
        (_, t0) = _time(lambda: generate_nocache(prompt, g))
        (_, t1) = _time(lambda: generate_cache(prompt, g))
        print(f"{g:>8}{t0*1e3:>14.2f}{t1*1e3:>12.2f}{t0/t1:>8.1f}x")

    print("-" * 43)
    print("No-cache time grows quadratically with length (it re-does all past work each step);")
    print("The cache makes it ~linear. The speedup GROWS with sequence length - which is exactly")
    print("why every real LLM server keeps a KV cache. (Its cost is memory: see the STUDENT TODO.)")
