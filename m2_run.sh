#!/usr/bin/env bash
# M2 - compile and run the custom matmul kernel ladder,
set -e
CC=${CC:-gcc}
SRC="$(dirname "$0")/m2_matmul.c"
OUT="${OUT:-./m2_matmul}"
echo "Compiling with: $CC -O3 -fopenmp -march=native"
"$CC" -O3 -fopenmp -march=native "$SRC" -o "$OUT" -lm
echo "Running $OUT"
"$OUT"
