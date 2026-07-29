## M6 - GPU Kernels + LLM Serving (run on Google Colab, free T4)

# The payoff module. Two parts:
# - **Part A - write a GPU kernel.** Fuse two ops into ONE kernel with **Triton**,
#   beat eager PyTorch, and check `torch.compile`.
# - **Part B - serve an LLM.** Measure **tokens/sec**, **batch throughput**,
#   and (optional) **4-bit quantization** cost.

# **Setup first:** Runtime -> Change runtime type -> T4 GPU.
# Then run top-to-bottom.

# GPU-only cells are guarded, so nothing crashes on CPU -
# but Parts A/B only *do* something on a GPU.

import torch, time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch:', torch.__version__, '| device:', DEVICE)

if DEVICE == 'cuda':
    print('GPU:', torch.cuda.get_device_name())
else:
    print('No GPU - switch Colab runtime to T4 GPU and re-run.')

def bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()

    if DEVICE == 'cuda':
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters / 1000.0

    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters

## Part A - a custom fused kernel with Triton
# `relu(x + y)` in eager PyTorch launches TWO kernels and
# reads/writes memory twice. A single fused Triton kernel
# does it in ONE memory pass. For a memory-bound op, fewer
# passes = faster.

HAVE_TRITON = False

if DEVICE == 'cuda':
    try:
        import triton
        import triton.language as tl
        HAVE_TRITON = True
    except Exception as ex:
        print('triton unavailable:', ex)

if HAVE_TRITON:
    @triton.jit
    def add_relu_kernel(
        x_ptr,
        y_ptr,
        o_ptr,
        n,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        mask = off < n

        x = tl.load(x_ptr + off, mask=mask)
        y = tl.load(y_ptr + off, mask=mask)
        z = x + y
        z = tl.where(z > 0, z, 0.0)
        tl.store(o_ptr + off, z, mask=mask)

    def triton_add_relu(x, y):
        o = torch.empty_like(x)
        n = x.numel()
        grid = lambda meta: (triton.cdiv(n, meta['BLOCK']),)
        add_relu_kernel[grid](x, y, o, n, BLOCK=1024)
        return o

    N = 1 << 22
    x = torch.randn(N, device=DEVICE)
    y = torch.randn(N, device=DEVICE)

    ref = torch.relu(x + y)
    out = triton_add_relu(x, y)

    print('correct :', torch.allclose(out, ref, atol=1e-5))

    te = bench(lambda: torch.relu(x + y))
    tt = bench(lambda: triton_add_relu(x, y))

    print('eager   : %.4f ms' % (te * 1e3))
    print('triton  : %.4f ms (%.2fx vs eager)' % (tt * 1e3, te / tt))

else:
    print('Skipping Triton (needs a CUDA GPU).')

# torch.compile fuses the same graph automatically - compare it to hand-written Triton.
if DEVICE == 'cuda':

    def f(x, y):
        return torch.relu(x + y)

    try:
        fc = torch.compile(f)

        N = 1 << 22
        x = torch.randn(N, device=DEVICE)
        y = torch.randn(N, device=DEVICE)

        fc(x, y)   # triggers compilation

        te = bench(lambda: f(x, y))
        tc = bench(lambda: fc(x, y))

        print('eager        : %.4f ms' % (te * 1e3))
        print('torch.compile: %.4f ms (%.2fx)' % (tc * 1e3, te / tc))

    except Exception as ex:
        print('torch.compile failed:', str(ex)[:80])

else:
    print('Skipping torch.compile demo on CPU.')

## Part B - serve an LLM and measure it

# A small instruct model (Qwen2.5-0.5B) so it loads fast on a free T4.
# We measure **decode tokens/sec** and how **batching** raises throughput -
# the core levers of an inference server.

MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'
model = None; tok = None
if DEVICE == 'cuda':
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16).to(DEVICE).eval()
    print('loaded', MODEL)
else:
    print('Skipping model load on CPU (slow). This runs on Colab T4.')

def throughput(prompt, new=128, batch=1):
    msgs = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok([text] * batch, return_tensors='pt').to(DEVICE)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=new, do_sample=False)
    torch.cuda.synchronize(); dt = time.perf_counter() - t
    gen = (out.shape[1] - enc['input_ids'].shape[1]) * batch
    return gen / dt

if DEVICE == 'cuda':
    print('%-8s %16s' % ('batch', 'tokens/s (total)'))
    print('-' * 26)
    for b in [1, 2, 4, 8]:
        try:
            print('%-8d %16.1f' % (b, throughput('Explain the roofline model in two sentences.', 128, b)))
        except RuntimeError as ex:
            print("%-8d OOM/error: %s" % (b, str(ex)[:40]))
    print('\nBatching raises TOTAL throughput (the GPU was under-fed at batch 1) until you hit')
    print('memory/compute limits - the throughput-vs-latency trade-off every serving team faces.')
else:
    print('(GPU only)')

# OPTIONAL: 8-bit quantization (weights ~2x smaller). On Colab, first run in its own cell:
#     !pip install -q bitsandbytes
if DEVICE == 'cuda':
    try:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        qmodel = AutoModelForCausalLM.from_pretrained(
            MODEL, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map='cuda'
        ).eval()
        vram = torch.cuda.max_memory_allocated() / 1e9
        print(f'8-bit model peak VRAM: ~{vram:.2f} GB (compare to fp16 above)')
        print('Quantization shrinks memory + can raise throughput, at some accuracy cost (see M3).')
        del qmodel; torch.cuda.empty_cache()
    except Exception as ex:
        print('Quantization demo skipped:', str(ex)[:80])
        print('Tip: run !pip install -q bitsandbytes in a cell first.')
else:
    print('(GPU only)')
# ## What to notice
# - Your **Triton** kernel beats eager for the memory-bound op because it does one memory pass instead of two - the exact lesson from M1/M2 (memory movement dominates), now on the GPU.
# - `torch.compile` often matches hand-written fusions for free - know when hand-tuning is worth it.
# - **Batching** raises tokens/sec until you saturate the GPU - that is the throughput vs latency trade-off of real LLM serving. Production stacks (vLLM, TGI) add paged KV-cache + continuous batching on top of exactly this.
#
# ## STUDENT TODO
# 1. Extend the Triton kernel to fuse `x*sigmoid(x)` (SiLU) or add a bias. Verify correctness, measure speedup.
# 2. Plot tokens/s vs batch size. Where does throughput stop rising (GPU saturated)?
# 3. Compare fp16 vs 8-bit: VRAM, tokens/s, and eyeball output quality on the same prompt.
# 4. (Stretch) Install **vLLM** (`pip install vllm`), serve the same model, and compare its tokens/s to plain `transformers.generate`.
# 5. Record your GPU tokens/s into the capstone report next to your CPU (M4) numbers.
