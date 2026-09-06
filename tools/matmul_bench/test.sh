#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
OUT="${CUMETAL_MATMUL_OUT:-${ROOT}/build/matmul-bench}"
bash "$DIR/run.sh" 64 32 48 2 all
for shape in '1 1 1' '17 19 23' '65 37 49' '33 36 52' '129 67 131'; do
    read -r m k n <<< "$shape"
    "$OUT/matmul_bench" "$OUT/kernels.metallib" "$m" "$k" "$n" 2 all
done
if "$OUT/matmul_bench" "$OUT/kernels.metallib" 17 19 23 2 vectorized; then
    echo 'FAIL: unsafe article vector alignment accepted' >&2; exit 1
fi
if "$OUT/matmul_bench" "$OUT/kernels.metallib" 0 32 32 2 all; then
    echo 'FAIL: zero dimension accepted' >&2; exit 1
fi
if "$OUT/matmul_bench" "$OUT/kernels.metallib" 32 32 32 2 unknown; then
    echo 'FAIL: unknown variant accepted' >&2; exit 1
fi
echo 'PASS: all valid variants, rectangular tails, small sizes and invalid arguments'
