"""M9 - Prefill/Decode (P/D) disaggregation, a discrete-event simulation (no GPU, stdlib only).

An LLM request has two very different phases:
 - PREFILL: process the whole prompt in one shot. Compute-heavy, bursty, short.
 - DECODE : emit output tokens one at a time. Latency-sensitive, steady, long.

If both run on the SAME workers (colocation), a big prefill blocks ongoing decodes, so
users see spiky time-to-first-token (TTFT) and jittery output under load. DISAGGREGATION
puts prefill on one pool of workers and decode on another, so prefills never stall decodes.
This is what DistServe / vLLM's P/D-disaggregation and Splitwise do.

This sim models both topologies with the SAME total workers and compares TTFT and
end-to-end latency. It's ~120 lines of pure Python - no model, no GPU, no dependencies.

Run:  python3 m9_pd_disagg.py
Knobs (env vars): REQUESTS, RATE (arrivals/sec), WORKERS, PREFILL_P (fraction of workers
given to prefill in the disaggregated case), SEED.
"""

import heapq
import os
import random
import statistics

REQUESTS = int(os.environ.get("REQUESTS", "400"))
RATE = float(os.environ.get("RATE", "2.2"))    # requests/sec (raise it to see colocation suffer)
WORKERS = int(os.environ.get("WORKERS", "8"))
PREFILL_FRAC = float(os.environ.get("PREFILL_P", "0.125"))  # e.g. 1 of 8 workers does prefill
SEED = int(os.environ.get("SEED", "0"))

PREFILL_S_PER_TOK = 0.00005    # 50 us/prompt-token (compute-bound, fast per token, big prompts)
DECODE_S_PER_TOK = 0.012       # 12 ms/output-token (memory-bound, the slow steady part)


def make_requests():
    random.seed(SEED)
    t = 0.0
    reqs = []
    for i in range(REQUESTS):
        t += random.expovariate(RATE)
        prompt = random.randint(200, 1500)     # prompt tokens -> prefill work
        out = random.randint(50, 400)          # output tokens -> decode work
        reqs.append({"id": i, "arrival": t,
                     "prefill": prompt * PREFILL_S_PER_TOK,
                     "decode": out * DECODE_S_PER_TOK})
    return reqs


def _serve(reqs, servers, jobkey):
    """FIFO across 'servers' identical workers; returns (start, finish) per request.
    Each request in arrival order goes to the earliest-free worker. `jobkey(r)` is the
    service time on this pool."""
    free = [0.0] * servers                 # a min-heap of worker free-times
    heapq.heapify(free)
    res = {}
    for r in sorted(reqs, key=lambda r: r["ready"]):
        w = heapq.heappop(free)
        start = max(r["ready"], w)
        finish = start + jobkey(r)
        heapq.heappush(free, finish)
        res[r["id"]] = (start, finish)
    return res


def sim_colocated(reqs):
    """One pool: each worker does prefill THEN decode for a request, back to back."""
    for r in reqs:
        r["ready"] = r["arrival"]
    res = _serve(reqs, WORKERS, lambda r: r["prefill"] + r["decode"])
    ttft, e2e = [], []
    for r in reqs:
        start, finish = res[r["id"]]
        ttft.append(start + r["prefill"] - r["arrival"])   # first token after prefill
        e2e.append(finish - r["arrival"])
    return ttft, e2e


def sim_disaggregated(reqs):
    """Two pools: prefill workers, then decode workers (KV handed over between them)."""
    p_workers = max(1, round(WORKERS * PREFILL_FRAC))
    d_workers = max(1, WORKERS - p_workers)
    for r in reqs:
        r["ready"] = r["arrival"]
    pf = _serve(reqs, p_workers, lambda r: r["prefill"])   # phase 1: prefill
    for r in reqs:
        r["ready"] = pf[r["id"]][1]                         # decode can start once prefill done
    dc = _serve(reqs, d_workers, lambda r: r["decode"])    # phase 2: decode
    ttft, e2e = [], []
    for r in reqs:
        ttft.append(pf[r["id"]][1] - r["arrival"])          # first token right after prefill
        e2e.append(dc[r["id"]][1] - r["arrival"])
    return ttft, e2e, p_workers, d_workers


def p(xs, q):
    xs = sorted(xs)
    return xs[max(0, min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1)))))]


def report(name, ttft, e2e):
    print(f"{name:<22}{p(ttft,50)*1e3:>12.0f}{p(ttft,99)*1e3:>12.0f}"
          f"{p(e2e,50)*1e3:>12.0f}{p(e2e,99)*1e3:>12.0f}")


if __name__ == "__main__":
    reqs = make_requests()
    ct, ce = sim_colocated(list(reqs))
    dt, de, pw, dw = sim_disaggregated(list(reqs))

    print(f"{REQUESTS} requests @ {RATE}/s, {WORKERS} workers "
          f"(disagg split: {pw} prefill + {dw} decode)\n")

    print(f"{'topology':<22}{'TTFT p50(ms)':>12}{'TTFT p99(ms)':>12}{'E2E p50(ms)':>12}{'E2E p99(ms)':>12}")
    print("-" * 82)
    report("colocated", ct, ce)
    report("disaggregated", dt, de)
    print("-" * 82)
    imp = (p(ct, 99) / p(dt, 99)) if p(dt, 99) else float("inf")
    print(f"\nDisaggregation cuts tail TTFT (p99) by ~{imp:.1f}x here: prefill runs on its own pool,")
    print("so a new request gets its first token fast even when every decode worker is busy.")
    print("E2E is governed by the DECODE pool size - that is the point: with disaggregation you")
    print("scale prefill (for TTFT) and decode (for throughput) INDEPENDENTLY. Try RATE=2.8 to")
    print("push colocation toward overload; tune PREFILL_P to rebalance the two pools.")
