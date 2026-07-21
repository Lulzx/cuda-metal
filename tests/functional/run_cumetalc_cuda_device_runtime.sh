#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
TEST_BINARY="$2"
INPUT_CU="$3"
OUTPUT_METALLIB="$4"

if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: complete Metal toolchain unavailable"
  exit 77
fi

"$CUMETALC" \
  --cuda-device \
  --mode xcrun \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$TEST_BINARY" "$OUTPUT_METALLIB"
