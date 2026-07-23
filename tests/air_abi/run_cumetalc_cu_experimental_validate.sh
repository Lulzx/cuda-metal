#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_CU="$4"
OUTPUT_METALLIB="$5"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "SKIP: xcrun clang++ not available"
  exit 77
fi

"$CUMETALC" \
  --backend=cumetal-ir \
  --entry vector_add \
  --mode experimental \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$VALIDATOR" \
  "$OUTPUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUTPUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "vector_add"' >/dev/null

echo "PASS: cumetalc .cu experimental emit+validate completed"
