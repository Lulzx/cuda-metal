#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_CU="$4"
INCLUDE_DIR="$5"
OUTPUT_METALLIB="$6"

if [[ -z "${CUMETAL_CUDA_CLANG:-}" \
      && ! -x /opt/homebrew/opt/llvm/bin/clang++ \
      && ! -x /usr/local/opt/llvm/bin/clang++ ]]; then
  echo "SKIP: CUDA-capable Homebrew LLVM clang++ not available"
  exit 77
fi

if "$CUMETALC" \
    --cuda-device \
    --cuda-arch invalid \
    --mode experimental \
    --input "$INPUT_CU" \
    --output "${OUTPUT_METALLIB}.invalid" \
    --entry cuda_device_probe \
    -I "$INCLUDE_DIR" \
    -D CUMETAL_FRONTEND_TEST_VALUE=2 \
    --overwrite >/dev/null 2>&1; then
  echo "FAIL: invalid CUDA architecture was accepted" >&2
  exit 1
fi

"$CUMETALC" \
  --cuda-device \
  --mode experimental \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --entry cuda_device_probe \
  --ptx-strict \
  -I "$INCLUDE_DIR" \
  -D CUMETAL_FRONTEND_TEST_VALUE=2 \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUTPUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "cuda_device_probe"' >/dev/null

echo "PASS: cumetalc CUDA device frontend emit+validate completed"
