"""M7 load-test client - see the throughput vs latency trade-off for yourself.

Fires a fixed number of requests at the M7 server at increasing CONCURRENCY and
reports, for each level: p50/p99 latency and total requests/sec. Uses only the
Python standard library (no extra installs).

Run the server first (python m7_server.py), then:
    python m7_loadtest.py                    # defaults to http://127.0.0.1:8000
    URL=http://host:8000 REQUESTS=64 python m7_loadtest.py

What you should see: at concurrency 1 the server can't batch, so requests/sec is
low. As concurrency rises, the scheduler batches co-arriving requests -> total
requests/sec climbs a lot while per-request latency rises modestly... until the
GPU saturates. Re-run the server with MAX_BATCH=1 and compare: throughput stops
scaling. THAT gap is what dynamic batching buys - the core of LLM serving.
"""

import json
import os
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("URL", "http://127.0.0.1:8000").rstrip("/")
REQUESTS = int(os.environ.get("REQUESTS", "48"))
MAX_NEW = int(os.environ.get("MAX_NEW", "128"))
PROMPT = os.environ.get("PROMPT", "Explain the roofline model in two sentences.")
CONCURRENCIES = [int(x) for x in os.environ.get("CONCURRENCIES", "1,2,4,8,16").split(",")]

def one_request():
    body = json.dumps({"prompt": PROMPT, "max_new_tokens": MAX_NEW}).encode()
    req = urllib.request.Request(URL + "/generate", data=body,
                                 headers={"Content-Type": "application/json"})

    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        resp.read()
    return time.perf_counter() - t0

def pctile(xs, p):
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return xs[k]

def run(concurrency):
    latencies = []
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for lat in pool.map(lambda _: one_request(), range(REQUESTS)):
            latencies.append(lat)
    wall = time.perf_counter() - t0
    return {
        "concurrency": concurrency,
        "p50": pctile(latencies, 50),
        "p99": pctile(latencies, 99),
        "rps": REQUESTS / wall,
    }


if __name__ == "__main__":
    print(f"target={URL}  requests/level={REQUESTS}  max_new={MAX_NEW}")
    # warm-up (first request triggers any lazy CUDA/compile work)
    try:
        one_request()
    except Exception as e:
        raise SystemExit(f"Could not reach the server at {URL}. Is m7_server.py running? ({e})")
    print(f"{'concurrency':>11} {'p50 (s)':>9} {'p99 (s)':>9} {'req/s':>8}")
    print("-" * 41)
    for c in CONCURRENCIES:
        r = run(c)
        print(f"{r['concurrency']:>11} {r['p50']:>9.3f} {r['p99']:>9.3f} {r['rps']:>8.2f}")
    try:
        with urllib.request.urlopen(URL + "/stats", timeout=10) as resp:
            print("\nserver stats:", resp.read().decode())
    except Exception:
        pass
