"""
M3 - Precision & the speed/accuracy trade-off (PyTorch CPU).

Part A: matmul timing across dtypes (fp32 / bf16 / fp16) -> GFLOP/s.
Part B: int8 *dynamic quantization* of a Linear layer -> latency AND accuracy
        (output error vs the fp32 reference). The lesson: lower precision can be
        faster, but it is NOT free -- it costs accuracy.

Run:  python3 m3_precision.py
Deps: torch (CPU build fine)
"""
import os
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
import time
import torch
import torch.nn as nn
import sysinfo

torch.manual_seed(0)


def _bench(fn, reps=20, warmup=5):
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def part_a_matmul_dtypes(n=1024):
    print("Part A - matmul %d^3: fp32 vs bf16 (the CPU-relevant low precision)" % n)
    print(f"{'dtype':10}{'time(ms)':>10}{'GFLOP/s':>10}")
    flops = 2 * n ** 3
    # fp16 is skipped: it is emulated element-wise on most CPUs and can be ~1000x slower.
    for name, dt in [("float32", torch.float32), ("bfloat16", torch.bfloat16)]:
        try:
            a = torch.randn(n, n, dtype=dt)
            b = torch.randn(n, n, dtype=dt)
            t = _bench(lambda: torch.matmul(a, b), reps=10)
            print(f"{name:10}{t*1e3:10.3f}{flops/t/1e9:10.2f}")
        except Exception:
            print(f"{name:10}{'unsupported on this CPU build!':>30}")
            print("  (on CPUs without bf16 acceleration, bf16 can be SLOWER than fp32 -- a real lesson.)")


def _quant_int8(w):
    """Symmetric per-tensor int8 quantize -> dequantize (backend-independent)."""
    scale = w.abs().max() / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127)
    return q * scale  # dequantized approximation of w


def part_b_int8_accuracy(in_f=1024, out_f=1024, batch=64):
    print("\nPart B - int8 quantization: the accuracy cost (manual, backend-independent)")
    x = torch.randn(batch, in_f)
    w = torch.randn(out_f, in_f)
    y_fp32 = x @ w.t()
    w_int8 = _quant_int8(w)
    y_int8 = x @ w_int8.t()
    rel = (y_fp32 - y_int8).norm().item() / y_fp32.norm().item()
    mse = (y_fp32 - y_int8).pow(2).mean().item()
    print(f"  weights quantized to int8 (symmetric per-tensor)")
    print(f"  matmul output error : MSE={mse:.3e}   relative_error={rel:.3%}")
    print("  -> int8 shrinks memory ~4x and enables fast int8 kernels, but introduces")
    print("     this error. Speed gains need an actual int8 kernel (e.g. fbgemm/hardware).")


# -------------------------------------------------------------------------
# STUDENT TODO: quantize the ACTIVATIONS too (not just weights), or try a
# per-row (per-channel) int8 scale instead of one scale for the whole tensor,
# and see whether the relative error goes down. Implement and print it here.
def student_quant_experiment():
    raise NotImplementedError("Try per-channel int8 scaling; compare the error to per-tensor.")
# -------------------------------------------------------------------------


if __name__ == "__main__":
    sysinfo.header()
    part_a_matmul_dtypes()
    part_b_int8_accuracy()
