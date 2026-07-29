# ## M5 - GPU Roofline (run on Google Colab, free T4)

# The GPU counterpart of **M0 + M1**. Same lens (the roofline), new hardware.

# You will measure, on a real GPU:
# 1. **Peak compute** across precisions - fp32, TF32, fp16, bf16 (tensor cores).
# 2. **Peak memory bandwidth** (HBM).
# 3. **Ridge points** per precision, and **where real ops land** (matmul / conv / attention / pooling).

# **Setup first:** Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU -> Save,
# then run every cell top-to-bottom (`Shift+Enter`).

# > There is a CPU fallback so it won't crash without a GPU, but the point is the GPU -
# switch the runtime. Colab's free T4 is shared/throttled; numbers are indicative.

# 1) What GPU did Colab give us?
import torch, time

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch', torch.__version__, '| device:', DEVICE)

if DEVICE == 'cuda':
    p = torch.cuda.get_device_properties(0)
    print('GPU      :', p.name)
    print('SMs      :', p.multi_processor_count)
    print('VRAM (GB): %.1f' % (p.total_memory / 1e9))
    print('CUDA cap : %d.%d' % (p.major, p.minor))
else:
    print('No GPU detected - running on CPU (numbers illustrative).')
    print('On Colab: Runtime > Change runtime type > T4 GPU, then re-run.')

# 2) A correct GPU timer. On CUDA we MUST use cuda events + synchronize;
#    because kernel launches are async - time.time() would measure nothing.

def bench(fn, iters=30, warmup=10):
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
        return s.elapsed_time(e) / iters / 1000.0   # ms -> seconds

    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters

print("timer ready")

# 3) PEAK COMPUTE across precisions (big matmul).
# TF32 / fp16 / bf16 use the tensor cores -> should be MUCH faster than fp32.

def peak_matmul(dtype, use_tf32=False, n=4096):
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32

    a = torch.randn(n, n, device=DEVICE, dtype=dtype)
    b = torch.randn(n, n, device=DEVICE, dtype=dtype)

    t = bench(lambda: torch.mm(a, b))
    return 2.0 * n**3 / t / 1e9      # GFLOP/s

configs = [('fp32', torch.float32, False)]

if DEVICE == 'cuda':
    configs += [
        ('tf32', torch.float32, True),
        ('fp16', torch.float16, False),
        ('bf16', torch.bfloat16, False),
    ]

peak = {}

print('%-8s %12s' % ('dtype', 'GFLOP/s'))
print('-' * 22)

for name, dt, tf32 in configs:
    try:
        g = peak_matmul(dt, tf32)
        peak[name] = g
        print('%-8s %12.1f' % (name, g))
    except Exception as ex:
        print('%-8s skipped (%s)' % (name, str(ex)[:40]))

torch.backends.cuda.matmul.allow_tf32 = False

# 4) PEAK MEMORY BANDWIDTH (HBM): a = b + c reads 2 arrays, writes 1.

def peak_bw(n=1 << 25):     # 33M elements
    a = torch.empty(n, device=DEVICE)
    b = torch.randn(n, device=DEVICE)
    c = torch.randn(n, device=DEVICE)

    t = bench(lambda: torch.add(b, c, out=a))
    return 3 * n * 4 / t / 1e9      # 3 arrays × 4 bytes -> GB/s

bw = peak_bw()
print('Peak memory bandwidth: %.1f GB/s' % bw)

# 5) ROOFLINE ANCHORS + ridge point per precision.
#    ridge = peak_compute / peak_bandwidth (FLOP/byte). Above it -> compute-bound.

print('Peak memory BW: %.1f GB/s' % bw)
print('%-8s %12s %14s' % ('dtype', 'GFLOP/s', 'ridge(F/byte)'))
print('-' * 36)

for name, g in peak.items():
    print('%-8s %12.1f %14.1f' % (name, g, g / bw))

print('\nHigher-precision fp32 has a low ceiling; tensor-core dtypes push the ceiling up a lot,')
print('which also pushes the ridge point right (you need more arithmetic intensity to saturate).')

# 6) WHERE REAL OPS LAND (fp16 where it makes sense - that's how GPUs run in practice).

import torch.nn.functional as F
dt = torch.float16 if DEVICE == 'cuda' else torch.float32

def op_matmul():
    n = 4096
    a = torch.randn(n, n, device=DEVICE, dtype=dt)
    b = torch.randn(n, n, device=DEVICE, dtype=dt)
    flops = 2.0 * n**3
    return bench(lambda: torch.mm(a, b)), flops

def op_conv():
    x = torch.randn(16, 64, 56, 56, device=DEVICE, dtype=dt)
    w = torch.randn(64, 64, 3, 3, device=DEVICE, dtype=dt)
    flops = 2.0 * 16 * 64 * 64 * 3 * 3 * 54 * 54
    return bench(lambda: F.conv2d(x, w)), flops

def op_attention():
    b, h, s, d = 8, 8, 1024, 64
    q = torch.randn(b, h, s, d, device=DEVICE, dtype=dt)
    k = torch.randn(b, h, s, d, device=DEVICE, dtype=dt)
    v = torch.randn(b, h, s, d, device=DEVICE, dtype=dt)
    flops = 4.0 * b * h * s * s * d
    return bench(lambda: F.scaled_dot_product_attention(q, k, v)), flops

def op_maxpool():
    x = torch.randn(32, 64, 224, 224, device=DEVICE, dtype=dt)
    flops = float(x.numel())   # ~1 comparison per element -> memory-bound
    return bench(lambda: F.max_pool2d(x, 2)), flops

ridge_fp16 = (peak.get('fp16') or peak.get('fp32', 1.0)) / bw
print('ridge point (~%.0f FLOP/byte)\n' % ridge_fp16)
print('%-16s %10s %12s' % ('op', 'time(ms)', 'GFLOP/s'))
print('-' * 40)

for name, fn in [
    ('matmul', op_matmul),
    ('conv2d 3x3', op_conv),
    ('attention', op_attention),
    ('maxpool 2x2', op_maxpool)
]:
    t, flops = fn()
    # rough bytes moved: assume fp16 in/out; this is a teaching estimate, not exact
    gflops = flops / t / 1e9
    print("%-16s %10.3f %12.1f" % (name, t * 1e3, gflops))

print('\nCompare each GFLOP/s to the peaks in cell 5: matmul/conv/attention should be near the')
print('tensor-core ceiling; pooling stays low (memory-bound), exactly like on CPU (M1).')

# ## What to notice
# - fp16/bf16/TF32 crush fp32 - that is the **tensor cores**.
# - The GPU's peak memory bandwidth (hundreds of GB/s) dwarfs a CPU's (~10 GB/s from M0) -
#   yet pooling is *still* memory-bound. Arithmetic intensity, not raw FLOPs, decides
#   the ceiling. That idea is hardware-independent.
# - Compare your numbers to M0/M1 (CPU): the GPU is ~10-100x on compute-bound ops,
#   but far less on memory-bound ones.

# ### STUDENT TODO
# 1. Sweep the matmul size 'n' in {512, 1024, 2048, 4096, 8192}. Where does GFLOP/s stop rising?
#    (Small matmuls can't fill the GPU.)
# 2. Repeat the ops in fp32 vs fp16 and report the speedup. Which op benefits most? Why?
# 3. Run !nvidia-smi in a new cell while a big matmul loops - watch utilization and memory.
