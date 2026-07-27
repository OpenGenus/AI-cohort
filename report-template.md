# AI Performance Engineering - Capstone Report

> Fill this in with **your own measured numbers** from M0–M6. The point isn't the numbers themselves -
> it's that you can *explain* them with the roofline. This is the artifact you put on your CV / share.

**Author:** _your name_   **Date:** _YYYY-MM-DD_   **Repo/notebook links:** _..._

---

## 1. The machines

| | CPU (M0-M4) | GPU (M5-M6) |
|---|---|---|
| Model | _e.g. Xeon Gold 5220R (Cascade Lake)_ | _e.g. NVIDIA T4 (Colab)_ |
| Cores / SMs | _12 physical_ | _40 SMs_ |
| ISA / Features | _AVX2, AVX-512, bf16?_ | _CUDA cap, tensor cores_ |
| RAM / VRAM | _39 GB_ | _15 GB_ |
| How obtained | native / droplet / Docker | Colab T4 |

_(paste the `sysinfo` / cell-1 output for each)_

## 2. Roofline anchors (the ceilings)

| Anchor | CPU | GPU |
|---|---|---|
| Peak compute fp32 (GFLOP/s) | | |
| Peak compute bf16 (GFLOP/s) | | _tf32/fp16/bf16 →_ |
| Peak memory bandwidth (GB/s) | | |
| Ridge point (FLOP/byte) | | _per precision_ |

**One-line takeaway:** _e.g. "GPU compute is ~30× the CPU, but memory bandwidth only ~20×, so memory-bound ops close the gap."_

## 3. Where real ops land (M1 / M5)

| Op | CPU GFLOP/s | GPU GFLOP/s | Bound by | Why |
|---|---|---|---|---|
| matmul_1024_4096 | | | compute | high arithmetic intensity |
| conv2d 3×3 | | | compute | |
| attention (SDPA) | | | compute | |
| maxpool / upsample | | | *memory* | AI ≈ 0.2 FLOP/byte |

## 4. The optimization ladder (M2) – matmul, by hand

| Variant | GFLOP/s | Speedup vs naive | What changed |
|---|---|---|---|
| naive | | 1× | |
| tiled (cache blocking) | | | reuse data in cache |
| tiled + OpenMP | | | all cores |
| oneDNN benchmark (M1.5) | | | + SIMD + expert tuning (library ceiling) |

**Gap analysis:** _why is your hand kernel still Nx below the library? (SIMD width, blocking, packing, …)_
**ISA note (M1.5):** _AVX-512 vs `ONEDNN_MAX_CPU_ISA=avx2` – which was faster, and did you see throttling?_

## 5. Precision (M3 / M6)

| Precision | CPU GFLOP/s | GPU GFLOP/s | Accuracy cost |
|-----------|-------------|-------------|---------------|
| fp32 | | | reference |
| bf16 / fp16 | | | _relative error_ |
| int8 | -- | | _relative error_ |

**Takeaway:** _when did low precision help, when did it hurt, and why (hardware support)?_

## 6. Custom kernel (M6, GPU)

- Fused op: _relu(x+y) / SiLU / …_  Correct vs PyTorch: _yes/no_.
- eager: _… ms_ – Triton: _… ms (…×)_ – `torch.compile`: _… ms (…×)_.
- **Insight:** _why the fused kernel wins (memory passes), and whether hand-tuning beat `torch.compile`._

## 7. LLM inference (M4 CPU / M6 GPU)

| | CPU (llama.cpp) | GPU (transformers) |
|---|---|---|
| Model | _Qwen2.5_ | _Qwen2.5-0.5B_ |
| tokens/s @ batch 1 | | |
| tokens/s @ best batch | _(thread sweep)_ | _(batch sweep)_ |
| Quantization used | _Q4_K_ | _fp16 / 8-bit_ |

**Throughput vs latency:** _where did batching/threads stop helping, and why?_

## 8. Conclusions

- 3–5 bullets: the biggest surprises, the clearest roofline lesson, and one thing you'd optimize next.
- Optional: a roofline plot (compute ceiling, BW ceiling, your ops as points) for CPU and GPU.

---

**Method note:** GPU numbers on a shared Colab T4 are indicative (throttled/shared). For publishable figures, re-run on a dedicated GPU and report variance across 3+ runs.
