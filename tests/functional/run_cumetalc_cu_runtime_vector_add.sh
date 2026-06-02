#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
TEST_BINARY="$2"
INPUT_CU="$3"
OUTPUT_METALLIB="$4"

# If pre-generated .cu -> metallib exists (from prior build), just execute the test.
# cumetalc would need metal for --mode xcrun path; prebuilts allow runtime test.
if [[ -s "$OUTPUT_METALLIB" ]]; then
  echo "Using pre-existing cu-compiled metallib: $OUTPUT_METALLIB"
  exec "$TEST_BINARY" "$OUTPUT_METALLIB"
fi

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
  --mode xcrun \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

"$TEST_BINARY" "$OUTPUT_METALLIB"
