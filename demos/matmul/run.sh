#!/usr/bin/env bash
# CUDA matmul vs MPS: build, validate, prove GPU execution, then measure.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MODE=full
for arg in "$@"; do
    case "$arg" in
        --quick) MODE=quick ;;
        --check) MODE=check ;;
        --help|-h)
            echo 'Usage: bash demos/matmul/run.sh [--quick|--check]'
            echo 'Default: 4096 cubed, three rounds of 20 iterations.'
            echo '--quick: 512 cubed, three rounds of 3 iterations.'
            echo '--check: correctness and GPU provenance only.'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done
export CUMETAL_BUILD_DIR="${CUMETAL_BUILD_DIR:-$ROOT/build-matmul-release}"
OUT="${CUMETAL_MATMUL_DEMO_OUT:-$ROOT/demos/matmul/out}/$MODE"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export CUMETAL_MATMUL_OUT="$OUT/bin"
# Fix the comparison's math policy and keep tracing outside timed samples.
export CUMETAL_MSL_MATH_MODE=safe
export CUMETAL_TRACE_GPU=0
printf 'Building Release CuMetal (binary shim OFF)...\n'
cmake -S "$ROOT" -B "$CUMETAL_BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
    -DCUMETAL_ENABLE_BINARY_SHIM=OFF > "$OUT/configure.log" 2>&1 || { cat "$OUT/configure.log"; exit 1; }
cmake --build "$CUMETAL_BUILD_DIR" --target cumetalc cumetal_runtime -j6 \
    > "$OUT/build.log" 2>&1 || { cat "$OUT/build.log"; exit 1; }
printf 'Checking all kernel variants and rectangular edge cases...\n'
bash "$ROOT/tools/matmul_bench/test.sh" > "$OUT/correctness.log" 2>&1 || { cat "$OUT/correctness.log"; exit 1; }
CUMETAL_TRACE_GPU=1 "$CUMETAL_MATMUL_OUT/matmul_bench" \
    "$CUMETAL_MATMUL_OUT/kernels.metallib" 128 64 96 1 block64_32 \
    > "$OUT/provenance.log" 2>&1
python3 - "$OUT/provenance.log" <<'PY'
import pathlib, sys
lines = pathlib.Path(sys.argv[1]).read_text().splitlines()
records = [line for line in lines if 'CUMETAL_PROVENANCE event=kernel_launch' in line and 'kernel="block64_32"' in line]
if not records or any('device=apple_gpu' not in line or 'launch_success=true' not in line for line in records):
    raise SystemExit('FAIL: missing or failed Apple-GPU provenance')
if not any(line.startswith('check,block64_32,') and line.endswith('bad=0') for line in lines):
    raise SystemExit('FAIL: missing numerical verification')
print('PASS: numerical checks and Apple-GPU provenance')
PY
if [[ "$MODE" == check ]]; then
    printf 'Validation logs: %s\n' "$OUT"
    exit 0
fi
SIZE=4096
ITERATIONS=20
if [[ "$MODE" == quick ]]; then SIZE=512; ITERATIONS=3; fi
printf 'Measuring %s cubed, 3 rounds x %s iterations...\n' "$SIZE" "$ITERATIONS"
python3 "$ROOT/tools/matmul_bench/measure.py" "$CUMETAL_MATMUL_OUT" \
    --output "$OUT/evidence" --size "$SIZE" --iterations "$ITERATIONS" | tee "$OUT/measure.log"
python3 "$ROOT/demos/matmul/report.py" "$OUT/evidence/results.json" "$SIZE" | tee "$OUT/report.md"
printf '\nDemo artifacts: %s\n' "$OUT"
