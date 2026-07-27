"""M1 - Operation roofline (framework level, PyTorch CPU).

Measures achieved GFLOP/s and arithmetic intensity (FLOP/byte) for a suite of
core AI ops, then classifies each as compute- or memory-bound. On CPU, PyTorch
uses the oneDNN backend under the hood, so these are "framework" numbers.

Run:  python3 m1_roofline.py
Deps: torch (CPU build is fine)
Note: sets TORCH_DEVICE_BACKEND_AUTOLOAD=0 so it imports cleanly on all setups.
"""

import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import time
import torch
import torch.nn.functional as F
import sysinfo

torch.manual_seed(0)
DT = torch.float32
BYTES = 4  # float32


def _bench(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def op_matmul():
    m = k = n = 1024
    a = torch.randn(m, k, dtype=DT)
    b = torch.randn(k, n, dtype=DT)
    flops = 2 * m * k * n
    bytes_ = (m * k + k * n + m * n) * BYTES
    return "matmul 1024^3", _bench(lambda: torch.matmul(a, b)), flops, bytes_


def op_conv():
    N, Cin, H, W, Cout, Kk = 8, 64, 56, 56, 64, 3
    x = torch.randn(N, Cin, H, W, dtype=DT)
    w = torch.randn(Cout, Cin, Kk, Kk, dtype=DT)
    Ho, Wo = H - Kk + 1, W - Kk + 1
    flops = 2 * N * Cout * Ho * Wo * Cin * Kk * Kk
    bytes_ = (x.numel() + w.numel() + N * Cout * Ho * Wo) * BYTES
    return "conv2d 3x3", _bench(lambda: F.conv2d(x, w)), flops, bytes_


def op_maxpool():
    N, C, H, W = 16, 64, 112, 112
    x = torch.randn(N, C, H, W, dtype=DT)
    Ho = Wo = H // 2
    flops = N * C * Ho * Wo * 4  # comparisons in 2x2 window (memory-bound anyway)
    bytes_ = (x.numel() + N * C * Ho * Wo) * BYTES
    return "maxpool 2x2", _bench(lambda: F.max_pool2d(x, 2)), flops, bytes_

def op_avgpool():
    N, C, H, W = 16, 64, 112, 112
    x = torch.randn(N, C, H, W, dtype=DT)
    Ho = Wo = H // 2
    flops = N * C * Ho * Wo * 4
    bytes_ = (x.numel() + N * C * Ho * Wo) * BYTES
    return "avgpool 2x2", _bench(lambda: F.avg_pool2d(x, 2)), flops, bytes_

def op_attention():
    B, Hh, S, d = 4, 8, 512, 64
    q = torch.randn(B, Hh, S, d, dtype=DT)
    k = torch.randn(B, Hh, S, d, dtype=DT)
    v = torch.randn(B, Hh, S, d, dtype=DT)
    flops = 4 * B * Hh * S * S * d  # QK^T + softmax*V (approx)
    bytes_ = 3 * q.numel() * BYTES + B * Hh * S * S * BYTES
    return "MHA attention", _bench(lambda: F.scaled_dot_product_attention(q, k, v)), flops, bytes_

def op_upsample():
    N, C, H, W = 16, 32, 64, 64
    x = torch.randn(N, C, H, W, dtype=DT)
    Ho, Wo = H * 2, W * 2
    flops = N * C * Ho * Wo  # ~1 copy per output element -> very low arithmetic intensity
    bytes_ = (x.numel() + N * C * Ho * Wo) * BYTES
    return "upsample 2x", _bench(lambda: F.interpolate(x, scale_factor=2, mode="nearest")), flops, bytes_

# STUDENT TODO: add ONE more op and see where it lands on the roofline.
# Fill in the body, then append `op_student` to the `ops` list in `__main__`.
# Return a tuple: (name, best_time_seconds, flops, bytes_moved)
def op_student():
    # ideas: layernorm, gelu, batchnorm, softmax
    raise NotImplementedError("Pick an op, time it with _bench(...), estimate flops & bytes.")

if __name__ == "__main__":
    sysinfo.header()

    ops = [op_matmul, op_conv, op_maxpool, op_avgpool, op_attention, op_upsample]
    # add op_student here once you implement it

    print(f"{'op':16}{'time(ms)':>10}{'GFLOP/s':>10}{'AI(F/byte)':>12}  class")
    print("-" * 62)

    for op in ops:
        name, t, flops, bytes_ = op()
        gflops = flops / t / 1e9
        ai = flops / bytes_
        cls = "compute-bound" if ai > 25 else "memory-bound"  # ridge ~= M0's FLOP/byte (machine-specific)
        print(f"{name:16}{t*1e3:10.3f}{gflops:10.2f}{ai:12.2f}  {cls}")

    print("-" * 62)
    print("Arithmetic intensity (FLOP/byte) decides the ceiling: high AI -> compute-bound,")
    print("low AI -> memory-bound. Compare GFLOP/s here against M0's peak compute/BW.")
