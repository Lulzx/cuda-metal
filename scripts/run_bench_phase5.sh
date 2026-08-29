#!/usr/bin/env bash
# run_bench_phase5.sh — Phase 5 performance gate test.
# Usage: run_bench_phase5.sh <cumetal_bench_exe> <generate_bench_metallib_sh> <output_dir>
#
# Compiles bench_kernels.metal to bench_kernels.metallib in <output_dir>, then
# runs cumetal_bench --all-kernels --max-ratio=2.0.
# Exits with code 77 if xcrun is unavailable (CTest skip).

set -euo pipefail

BENCH_EXE="${1:?usage: $0 <cumetal_bench> <generate_bench_metallib.sh> <output_dir>}"
GEN_SCRIPT="${2:?}"
OUTPUT_DIR="${3:?}"

mkdir -p "$OUTPUT_DIR"

# Generate bench_kernels.metallib — exits 77 if xcrun unavailable.
bash "$GEN_SCRIPT" "$OUTPUT_DIR"

METALLIB="$OUTPUT_DIR/bench_kernels.metallib"
if [[ ! -f "$METALLIB" ]]; then
    echo "ERROR: bench_kernels.metallib was not generated" >&2
    exit 1
fi

# The gate uses the fastest uncontended sample. Twenty iterations could all
# land inside one macOS/Metal scheduling burst; 50 produced a clean sample in
# 10/10 repeated full-gate runs while preserving the same 2x ceiling.
exec "$BENCH_EXE" \
    --metallib "$METALLIB" \
    --all-kernels \
    --elements 262144 \
    --warmup 5 \
    --iterations 50 \
    --max-ratio 2.0
