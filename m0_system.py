"""M0 - System properties + roofline anchors.

Documents the machine and MEASURES the two axes of the roofline:
  - peak compute (GFLOP/s) via a large matmul (BLAS)
  - peak memory bandwidth (GB/s) via a STREAM-triad (a = b + s*c)
The ratio (GFLOP/s / GB/s) is the "ridge point": ops with higher arithmetic
intensity are compute-bound, lower are memory-bound.

Run:  python3 m0_system.py
Deps: numpy
"""

import time
import numpy as np
import sysinfo


def peak_gflops(n=2048, reps=5):
    # fp32 matmul (BLAS) - matches the dtype used by the M1/M3 op benchmarks.
    # This is the "achieved practical peak" via numpy's BLAS, not the theoretical
    # hardware max; a given library/op may run a bit faster or slower.
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    a @ b  # warm-up
    best = min((_time(lambda: a @ b) for _ in range(reps)))
    return (2 * n ** 3) / best / 1e9


def peak_bw_gbps(n=30_000_000, reps=20):
    # a = b + c : 2 reads + 1 write, no temporaries -> a clean memory-bound kernel.
    # Arrays (240 MB each) far exceed L3, so this really hits main memory.
    b = np.random.rand(n)
    c = np.random.rand(n)
    a = np.empty(n)
    np.add(b, c, out=a)  # warm-up
    best = min((_time(lambda: np.add(b, c, out=a)) for _ in range(reps)))
    bytes_moved = 3 * n * 8  # read b, read c, write a (float64 = 8 bytes)
    return bytes_moved / best / 1e9


def _time(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


# ------------------------------------------------------------
# STUDENT TODO: measure bandwidth for a SMALL array that fits in L1/L2 cache
# (a few KB) and compare it to the main-memory number above. You should see the
# cache is many times faster — that gap is why "tiling/blocking" (M2) matters.
def student_cache_bw():
    raise NotImplementedError("Time a=b+c on a tiny (cache-resident) array; report GB/s.")
# ----------------------------------------------------------------------

if __name__ == "__main__":
    sysinfo.header()
    g = peak_gflops()
    bw = peak_bw_gbps()

    print("ROOFLINE ANCHORS (measured on this machine)")
    print("-" * 62)
    print(f"Peak compute (matmul, fp32) : {g:10.2f} GFLOP/s")
    print(f"Peak mem BW (a=b+c kernel)  : {bw:10.2f} GB/s")
    print(f"Ridge point                 : {g / bw:10.2f} FLOP/byte")
    print(
        "  -> ops with arithmetic intensity ABOVE this are compute-bound;"
    )
    print(
        "     BELOW this are memory-bound. (Used by M1/M3 roofline plots.)"
    )
