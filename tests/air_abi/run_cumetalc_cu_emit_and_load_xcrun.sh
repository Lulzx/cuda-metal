#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
LOAD_TEST="$2"
INPUT_CU="$3"
OUTPUT_METALLIB="$4"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find clang++ >/dev/null 2>&1; then
  echo "SKIP: xcrun clang++ not available"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

"$CUMETALC" \
  --backend=cumetal-ir \
  --entry vector_add \
  --mode xcrun \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$LOAD_TEST" "$OUTPUT_METALLIB"

echo "PASS: cumetalc .cu xcrun output loads via MTLDevice.newLibraryWithData"
