#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD="${CUMETAL_BUILD_DIR:-${ROOT}/build}"
OUT="${CUMETAL_MATMUL_OUT:-${ROOT}/build/matmul-bench}"
mkdir -p "$OUT"
"$BUILD/cumetalc" "$ROOT/tools/matmul_bench/kernels.cu" --backend=cumetal-ir --emit=metallib --no-link --overwrite --save-temps -o "$OUT/kernels.metallib" > "$OUT/compile.log" 2>&1
xcrun clang++ -std=c++20 -O3 -DACCELERATE_NEW_LAPACK -fobjc-arc -I"$ROOT/runtime/api" "$ROOT/tools/matmul_bench/main.cpp" "$ROOT/runtime/metal_backend/matmul_bench_reference.mm" -L"$BUILD" -lcumetal -lcublas -Wl,-rpath,"$BUILD" -framework Foundation -framework Metal -framework MetalPerformanceShaders -framework Accelerate -o "$OUT/matmul_bench"
"$OUT/matmul_bench" "$OUT/kernels.metallib" "$@"
