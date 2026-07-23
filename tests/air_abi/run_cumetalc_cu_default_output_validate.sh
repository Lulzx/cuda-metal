#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
VALIDATOR="$2"
INSPECTOR="$3"
INPUT_CU="$4"
WORKDIR="$5"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "SKIP: xcrun clang++ not available"
  exit 77
fi

TMP_DIR="$WORKDIR/cumetalc-cu-default-output"
mkdir -p "$TMP_DIR"

CU_COPY="$TMP_DIR/vector_add.cu"
cp "$INPUT_CU" "$CU_COPY"

"$CUMETALC" \
  --backend=cumetal-ir \
  --entry vector_add \
  --mode experimental \
  "$CU_COPY" \
  --overwrite

OUT_METALLIB="${CU_COPY%.cu}.metallib"
if [[ ! -f "$OUT_METALLIB" ]]; then
  echo "FAIL: expected default output metallib at $OUT_METALLIB"
  exit 1
fi

"$VALIDATOR" \
  "$OUT_METALLIB" \
  --require-function-list \
  --require-metadata

INSPECT_JSON="$("$INSPECTOR" "$OUT_METALLIB" --json)"
echo "$INSPECT_JSON" | rg '"function_count": 1' >/dev/null
echo "$INSPECT_JSON" | rg '"name": "vector_add"' >/dev/null

echo "PASS: cumetalc .cu default output validation completed"
