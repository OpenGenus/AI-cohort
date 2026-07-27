#!/usr/bin/env bash

#!/usr/bin/env bash
# M4 - LLM inference on CPU with llama.cpp.
# Builds llama.cpp, downloads a tiny quantized model, and benchmarks tokens/sec
# across thread counts. Requires: git, cmake, a C/C++ compiler, and internet
# (one-time ~350 MB model download + build).
#
# Run:  bash m4_llm_cpu.sh
set -e
WORK=${WORK:-./llm_lab}
MODEL_URL=${MODEL_URL:-https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf}
MODEL="qwen2.5-0.5b-q4.gguf"    # relative to $WORK (we cd into it below)

mkdir -p "$WORK"; cd "$WORK"

# 1) clone + build llama.cpp (only the tools we need)
if [ ! -d llama.cpp ]; then
    git clone --depth 1 https://github.com/ggml-org/llama.cpp
fi

cmake -S llama.cpp -B llama.cpp/build -DGGML_NATIVE=ON -DLLAMA_CURL=OFF >/dev/null
cmake --build llama.cpp/build -j --target llama-bench >/dev/null

# 2) download a tiny quantized model (once)
if [ ! -f "$MODEL" ]; then
    echo "Downloading model..."
    curl -L "$MODEL_URL" -o "$MODEL"
fi

# 3) benchmark tokens/sec at different thread counts
BENCH=./llama.cpp/build/bin/llama-bench
for T in 1 4 8; do
    echo "===== threads = $T ====="
    "$BENCH" -m "$MODEL" -t "$T" -p 64 -n 64
done
echo "pp = prompt (prefill) tokens/sec, tg = generation (decode) tokens/sec."
