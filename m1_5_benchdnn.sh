#!/usr/bin/env bash

# M1.5 - Library "speed-of-light" with oneDNN benchdnn (x86).
# Shows what a production-grade library achieves for the SAME ops (matmul/conv),
# AND which instruction set it picks (e.g. avx512_core) via DNNL_VERBOSE.
# Run on an x86 machine (Docker on x86 laptop, or the DO CPU droplet).

# Run:  bash m1_5_benchdnn.sh   (one-time: clones + builds oneDNN, ~5-10 min)
set -e
WORK=${WORK:-./onednn_lab}
mkdir -p "$WORK"; cd "$WORK"

if [ ! -d oneDNN ]; then
    git clone --depth 1 https://github.com/oneapi-src/oneDNN
fi

cmake -S oneDNN -B build -DCMAKE_BUILD_TYPE=Release \
    -DDNNL_BUILD_TESTS=ON -DDNNL_BUILD_EXAMPLES=OFF -DDNNL_BUILD_GRAPH=OFF >/dev/null

# NOTE: oneDNN's AVX-512 kernel files are memory-hungry to compile. Use a small -j
# (2) to avoid the OOM-killer on modest machines. This build takes ~10-20 min once.
cmake --build build -j2 --target benchdnn >/dev/null

BD=./build/tests/benchdnn/benchdnn

echo "===== matmul 1024^3 fp32 : AVX-512 vs AVX2 (instruction-set impact) ====="
echo "-- default (AVX-512) --"
"$BD" --matmul --mode=P --dt=f32 1024x1024:1024x1024 2>/dev/null | grep -E "^perf,cpu"

echo "-- forced AVX2 (ONEDNN_MAX_CPU_ISA=avx2) --"
ONEDNN_MAX_CPU_ISA=avx2 "$BD" --matmul --mode=P --dt=f32 1024x1024:1024x1024 2>/dev/null | grep -E "^perf,cpu"

echo "(impl column: brg_matmul:avx512_core vs brg_matmul:avx2; compare the two Gflops values.)"

echo "===== convolution fp32 ====="
"$BD" --conv --mode=P --dt=f32 mb8ic64ih56oc64kh3ph0n1_5 2>/dev/null | grep -E "^perf,cpu"

echo
echo "INSIGHT: this is the library speed-of-light (compare vs M1 PyTorch and M2 your kernel)."
echo "Note the ISA in the impl name. And wider SIMD (AVX-512) is NOT always faster: on many"
echo "Cascade/Skylake-SP CPUs it triggers frequency throttling, so AVX2 can match or beat it."
echo "Measure, don't assume."
