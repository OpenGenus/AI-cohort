# AI Engineering Lab - Setup, Commands, Expected Output & How to Reason

Modules **M0-M4** run on a **laptop (Docker or native)** or a **Linux CPU droplet**.
**M1.5** (oneDNN benchdnn) is x86-only → run it in **x86 Docker or on the DO CPU droplet**.
GPU modules **M5/M6** run on free Colab or a rented hourly GPU.

Numbers below were captured on our test machine (**Intel Xeon Gold 5220R**, a shared VM). **Your numbers
will differ** - that's why `sysinfo` is printed with every result. What matters is the *pattern*, not the
exact figure.

---

## 0) Setup

**You need:** Python 3.9+ (M0/M1/M3), a C compiler + OpenMP (M2), and git+cmake+curl (M4). M1.5 also needs cmake.

**Recommended - Docker (works on Windows, macOS, Linux, x86 & Apple Silicon):**

```bash
# from this folder
docker run -it --rm -v "$PWD":/lab -w /lab python:3.12 bash

# inside the container:
apt-get update && apt-get install -y build-essential    # gcc + OpenMP for M2
pip install -r requirements.txt
```

**Native Windows:** install Python from python.org, then use a **virtual environment**:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

M2 (C+OpenMP) and M4 (llama.cpp) are easiest under **WSL2** (Ubuntu) on Windows; or use the Docker route.

**Native macOS / Linux:**

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy + CPU torch
# torch CPU wheel explicitly, if needed:
# pip install torch --index-url https://download.pytorch.org/whl/cpu
```

> If `import torch` throws a backend-extension error on some builds, the scripts already set
> `TORCH_DEVICE_BACKEND_AUTOLOAD=0` internally - nothing to do.

---

## A worked example - reasoning from *our* test CPU (do this for YOUR CPU)

> We are using this CPU: **Intel Xeon Gold 5220R (Cascade Lake)**. From M0's `sysinfo`, its ISA
> features are **avx2, avx512f, f16c** ... but **no avx512_bf16** and no **amx_bf16**. So **bf16**
> has no native hardware support on this CPU — it is emulated (upcast to fp32), it runs *slower* than
> fp32.** That exactly explains the M3 result (bf16 GFLOP/s < fp32). Native bf16 came later - **Cooper
> Lake** (AVX512-BF16) and **Sapphire Rapids** (AMX bf16/int8).

**What you should do:** run M0, read your `isa_features` and predict *before* running M3 whether bf16
will help on your machine. If you see `avx512_bf16` / `amx_bf16`, expect bf16 to be faster; if not,
expect it to be slower. Then run M3 and check your prediction. Also note M0's peak is the **achieved**
peak (via a library), which is well below the CPU's **theoretical** peak - ask yourself why (turbo,
threads, memory, a shared VM).

---

## M0 - System properties + roofline anchors

```bash
python3 m0_system.py
```

Expected (our CPU):

```text
cpu              : Intel(R) Xeon(R) Gold 5220R CPU @ 2.20GHz
logical_cores    : 12
physical_cores   : 12
L1d/L1i/L2/L3    : 384KiB / 384KiB / 12MiB / 71.5MiB
isa_features     : avx2, avx512f, f16c
Peak compute (matmul, fp32) : 249.02 GFLOP/s
Peak mem BW (a=b+c kernel)  :   9.95 GB/s
Ridge point                 :  25.03 FLOP/byte
```

Reason: ops with arithmetic intensity **above ~25 FLOP/byte are compute-bound**, below are memory-bound.

## M1 - Op roofline (PyTorch CPU; uses oneDNN under the hood)

```bash
python3 m1_roofline.py
```

Expected (our CPU):

```text
op              time(ms)  GFLOP/s  AI(F/byte)  class
matmul 1024^3      6.23    344.84     170.67   compute-bound
conv2d 3x3        15.93    107.97     137.13   compute-bound
maxpool 2x2       15.28      0.84       0.20   memory-bound
avgpool 2x2        8.87      1.45       0.20   memory-bound
MHA attention      6.66    322.66      46.55   compute-bound
upsample 2x        7.47      1.12       0.20   memory-bound
```

Reason: pooling/upsample have tiny AI (~0.2) → memory-bound + stuck far below peak compute no matter
what. **Analyze:** sweep the matmul size (e.g. 256, 512, 2048, 4096); small matmuls fit in cache and
behave differently from large ones that spill to memory.

## M1.5 - Library speed-of-light + instruction set (oneDNN benchdnn, x86)

```bash
bash m1_5_benchdnn.sh    # one-time: clones + builds oneDNN with -j2 (~10-20 min; memory-safe)
```
Expected (our CPU - **actual captured output**), matmul with AVX-512 vs forced AVX2:

```text
default (AVX-512): brg_matmul:avx512_core    min ~890 GFLOP/s
forced AVX2      : brg_matmul:avx2           min ~945 GFLOP/s
conv (fp32)      : jit:avx512_core           min ~899 GFLOP/s
```

Two insights:

1. **The ladder** (same matmul): your hand kernel (M2) **~40–56** → PyTorch (M1) **~250–350** →
   oneDNN benchdnn (M1.5) **~880+ GFLOP/s**. Each step = expert tuning you didn't have — that gap
   is the job.

2. **Wider SIMD is NOT always faster.** Here AVX2 *beat* AVX-512, because Cascade/Skylake-SP CPUs
   **downclock when running AVX-512** (frequency throttling). The lesson: **measure, don't assume** —
   the instruction set with the widest vectors doesn't automatically win. (Force the ISA with
   `ONEDNN_MAX_CPU_ISA=avx2`.)

(On Apple Silicon, or to skip the ~10–20 min build, treat M1/PyTorch as the library reference.)

## M2 - Custom CPU kernel: the optimization ladder (C + OpenMP)

```bash
bash m2_run.sh
```

Expected (our CPU):

```text
variant         time(s)    GFLOP/s      max_diff
naive           0.5623       3.82       (reference)
tiled           0.1555      13.81       0.00e+00
tiled+openmp    0.0381      56.30       0.00e+00

maxpool 2x2 on 4096x4096 image: 0.017 s, 4.83 GB/s (memory-bound)
```

Reason: naive → cache-blocked (tiled) → tiled+threads climbs ~10×; `max_diff = 0` proves each variant is
still **correct**. **Analyze the gap:** M1/M1.5 (library) hit hundreds of GFLOP/s while your kernel hits
tens — that difference is **SIMD (AVX-512) + expert tuning** you haven't added yet. The pooling line is
memory-bound (reported in GB/s), mirroring M1.

## M3 - Precision: speed & accuracy trade-off (PyTorch CPU)

```bash
python3 m3_precision.py
```

Expected (our CPU):

```text
float32        6.68 ms    321.38 GFLOP/s
bfloat16      15.15 ms    141.80 GFLOP/s    <-- slower; no native bf16 on this CPU (see worked example)
int8 weight-quant matmul error: relative_error = 1.151%
```

Reason: bf16 is slower here because the CPU lacks `avx512_bf16`; int8 shrinks memory ~4× but costs ~1%
error — the speed/accuracy trade-off.

## M4 - LLM inference on CPU (llama.cpp)

```bash
...
```

bash m4_llm_cpu.sh        # one-time: builds llama.cpp + downloads ~460 MB tiny model

Expected (Qwen2.5-0.5B Q4):

threads=1 : pp64 59.10 t/s    tg64 14.75 t/s
threads=4 : pp64 210.13 t/s   tg64 38.73 t/s

`pp` = prefill/prompt t/s, `tg` = generate/decode t/s.
**Analyze:** raise threads further — on a shared/contended machine very high thread counts can regress
(oversubscription); on a dedicated laptop they scale until you hit physical cores.

---

## GPU modules M5 & M6 - run on Google Colab (free T4)

The CPU lab (M0-M4) taught the concepts; M5/M6 re-run the same roofline lens on a **GPU**,
where AI actually ships. They're **Jupyter notebooks**
(`m5_gpu_roofline.ipynb`, `m6_gpu_kernels_llm.ipynb`) meant
for **Google Colab**, which gives you a free NVIDIA **T4 GPU** - no install, no GPU purchase.

### First time with Colab? (step by step)

1. Go to **https://colab.research.google.com** and sign in with a Google account.
2. **File > Upload notebook** → pick `m5_gpu_roofline.ipynb` (or drag it in).
   *(Or put the notebooks in GitHub/Drive and open from there.)*
3. **Turn the GPU on:** **Runtime > Change runtime type > Hardware accelerator > T4 GPU > Save.**
   *(This step is the whole point — without it you get a CPU and the GPU cells no-op.)*
4. **Run everything:** **Runtime > Run all** (or Shift+Enter cell by cell, top to bottom).
5. Verify the GPU: the first cell prints the GPU name (e.g. `Tesla T4`).
   You can also add a cell with `!nvidia-smi` to watch memory/utilization.
6. Colab preinstalls `torch` / `triton` / `transformers`.
   For the optional 8-bit cell, run `!pip install -q bitsandbytes` in a cell first.
   Free sessions time out when idle — just re-run.

> Notebooks have a **CPU fallback** so they won't crash if you forget the GPU, but the numbers only
> mean something on the T4. A free T4 is **shared/throttled** - treat results as indicative
> (for publishable numbers, rent a dedicated GPU briefly).

### M5 - GPU roofline (PyTorch CUDA)

Measures **peak compute per precision** (fp32 / TF32 / fp16 / bf16 - the tensor cores),
**peak HBM bandwidth**, **ridge points**, and **where matmul/conv/attention/pooling land**.
Expected story: fp16/bf16 crush fp32 (tensor cores); the GPU beats your CPU (M0/M1)
by ~10-100x on **compute-bound** ops but far less on **memory-bound** ones -
because arithmetic intensity, not raw FLOPS, still decides the ceiling.

### M6 - GPU kernels + LLM serving

**Part A:** write a fused **Triton** kernel (`relu(x+y)` in one memory pass), beat eager PyTorch,
and compare `torch.compile`. Same lesson as M2 (fewer memory passes = faster), now on the GPU.

**Part B:** serve a small LLM (Qwen2.5-0.5B), measure **decode tokens/s** and how **batching**
raises throughput — the throughput-vs-latency trade-off real serving stacks (vLLM, TGI)
are built around. Optional 8-bit quantization shows the memory/accuracy trade-off
(ties back to M3).

## M7 - Build an inference server with dynamic batching (the serving payoff)

M6 showed a bigger *batch* means more tokens/sec. But real requests arrive one at a time.
`m7_server.py` is a ~120-line FastAPI server that **collects requests arriving close together
and runs them as ONE batch** - the core trick behind every production LLM server.

```bash
pip install -r requirements-serve.txt
python m7_server.py          # serves http://0.0.0.0:8000 (GPU strongly recommended)
# in another terminal:
python m7_loadtest.py        # p50/p99 latency + requests/sec at rising concurrency
```

Expected story: at concurrency 1 the server can't batch, so req/s is low. As concurrency rises,
the scheduler batches co-arriving requests -> **total req/s climbs while per-request latency rises
modestly**, until the GPU saturates. Re-run the server with `MAX_BATCH=1` and compare - throughput
stops scaling. That gap is exactly what dynamic batching buys you.

**Knobs:** `MAX_BATCH` (batch cap) and `MAX_WAIT_MS` (how long to wait for a batch to fill) - the two
dials every serving team tunes. This is *request-level* batching; vLLM/TGI add *token-level* continuous
batching + a paged KV-cache on top of the same idea.

**No GPU? Run it in simulation mode** - only needs `fastapi` + `uvicorn`, no model/torch/download:
```bash
pip install fastapi uvicorn
SIM=1 python m7_server.py        # fake forward pass; batching logic is identical
python m7_loadtest.py            # you still see req/s scale with concurrency, then plateau at MAX_BATCH
```

Example SIM run (req/s scales ~1x -> 8x as concurrency rises, then plateaus at MAX_BATCH=8):
```text
concurrency   p50 (s)   req/s
         1     0.524     1.91
         8     0.519    15.38
        16     1.027    15.49   <- past the batch cap: latency doubles, throughput flat
```

---

## M8 - Model-level optimization: the KV cache

The single most important *model-level* inference trick. During decoding, each new token
attends to all previous tokens; the naive approach recomputes every past key/value every step
(work ~O(n^2) per step). The **KV cache** stores them and computes only the new token
(~O(n) per step) - same output, far less work.

`m8_kv_cache.py` implements one causal-attention block **twice** (no-cache vs cache), proves
they emit **identical tokens**, and times both as the sequence grows. Pure CPU, no GPU.

```bash
python3 m8_kv_cache.py
```
Expected: `identical tokens` True`, and the cache's speedup **grows with length** - which is why
every LLM server keeps one. Its cost is memory (the STUDENT TODO computes the KV-cache size for a
real model - the reason paged / quantized KV caches exist).

## M9 - System-level optimization: prefill/decode (P/D) disaggregation

A request has two phases with opposite profiles: **prefill** (compute-heavy, bursty, short) and
**decode** (latency-sensitive, steady, long). Colocating them lets a big prefill stall in-flight
decodes -> spiky time-to-first-token (TTFT). **Disaggregation** runs prefill and decode on separate
worker pools (DistServe / Splitwise / vLLM P/D).

`m9_pd_disagg.py` is a ~120-line discrete-event simulation (stdlib only, no GPU) that compares both
topologies with the same total workers:

```bash
python3 m9_pd_disagg.py          # knobs: RATE, WORKERS, PREFILL_P
```

Example: TTFT p99 **1289 ms (colocated) -> 118 ms (disaggregated)**, ~11x better, because a new
request's first token no longer waits behind others' decodes. End-to-end latency is governed by the
**decode** pool - the lesson being that disaggregation lets you scale prefill (for TTFT) and decode
(for throughput) *independently*. Raise `RATE` to watch colocation's TTFT balloon.

**Capstone:** fill in `capstone_report_template.md` with **your** CPU (M0-M4) and **your** GPU
(M5-M6) numbers, explained with one roofline. That report is the CV artifact.

---

## Things to analyze across the lab

- **Sizes:** does matmul stay compute-bound at all sizes? Where does it fall off (cache spill)?
- **Utilization:** run `htop` alongside — are all cores saturated, or is it memory-stalled?
- **ISA:** which instruction set does oneDNN pick (M1.5)? AVX2 vs AVX-512 changes the compute ceiling.
- **Threads:** find the thread count where more threads stop helping (M2/M4).
- **Ladder:** line up matmul GFLOP/s for M2 (hand) vs M1 (framework) vs M1.5 (library) — explain each gap.

## Read the real code (where these concepts live)

Once you've *measured*, read how the pros implement it. High-value paths:

**oneDNN** (`github.com/oneapi-src/oneDNN`) — the library M1.5 benchmarks:
- `src/cpu/x64/matmul/brgemm_matmul.cpp` — the fast **brgemm matmul** (the `brg_matmul:avx512_core` you saw).
- `src/cpu/x64/brgemm/` — the block-GEMM **microkernel** machinery (tiling/blocking in practice).
- `src/cpu/x64/cpu_isa_traits.hpp` — **ISA detection** (avx2/avx512/amx): how the library *picks* a kernel.
- `src/cpu/x64/jit_generator.hpp` — the **JIT assembler** (Xbyak) that emits kernels at runtime.
- `src/cpu/x64/jit_avx512_core_*conv*` — AVX-512 **convolution** kernels.

**PyTorch** (`github.com/pytorch/pytorch`) — the framework M1/M3 use:
- `aten/src/ATen/native/` — op **entry points** (matmul, conv, pooling, …).
- `aten/src/ATen/cpu/vec/` — the **SIMD vectorization** layer (avx2/avx512 abstractions).
- `aten/src/ATen/native/cpu/*Kernel.cpp` — the actual **vectorized CPU kernels**.
- `aten/src/ATen/native/mkldnn/` — how PyTorch **dispatches to oneDNN** for conv/matmul.
- `aten/src/ATen/native/transformers/` — **attention / scaled_dot_product_attention** (flash / mem-efficient).
- `aten/src/ATen/native/quantized/` — **int8 quantization** ops.
- `aten/src/ATen/native/cuda/` — **CUDA kernels** (for the GPU modules M5/M6).

**llama.cpp** (`github.com/ggml-org/llama.cpp`) — M4's engine:
- `ggml/src/ggml-cpu/` — the **CPU kernels** (quantized matmul / dot products that drive tokens/sec).
- `ggml/src/ggml-quants.c` — the **quantization formats** (Q4_K, etc.) you benchmarked.

Suggested habit: pick one op you measured (say matmul), and trace it:
PyTorch entry → oneDNN dispatch → brgemm kernel.
That path *is* AI performance engineering.

## Student TODOs

Each script has a `STUDENT TODO` stub:
add an op (M1), measure cache bandwidth (M0), write an optimized matmul variant (M2), try per-channel int8 (M3). Implement them and re-measure.
