#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
INPUT_METAL="$2"
OUTPUT_METALLIB="$3"

# Reuse a previously compiled metallib only when the Metal toolchain is genuinely missing.
# The artifact check used to come first, so once one build produced a metallib, later edits to
# the reference .metal were never compiled again and the test reported on stale output.
if ! (command -v xcrun >/dev/null 2>&1 && xcrun --find metal >/dev/null 2>&1 &&
      xcrun --find metallib >/dev/null 2>&1) && [[ -s "$OUTPUT_METALLIB" ]]; then
  echo "WARNING: Metal toolchain unavailable; using a previously compiled metallib."
  echo "         $INPUT_METAL was not rebuilt: $OUTPUT_METALLIB"
else
  rm -f "$OUTPUT_METALLIB"
  if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
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

  xcrun metal -c "$INPUT_METAL" -o "$OUTPUT_METALLIB.air"
  xcrun metallib "$OUTPUT_METALLIB.air" -o "$OUTPUT_METALLIB"
fi

"$TEST_BINARY" "$OUTPUT_METALLIB"
