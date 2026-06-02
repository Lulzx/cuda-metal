#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
INPUT_METAL="$2"
OUTPUT_METALLIB="$3"

# Support pre-existing metallib in the build dir (common after initial full-toolchain build).
# Allows running the kernel execution part of the test even without xcrun metal/metallib.
if [[ -s "$OUTPUT_METALLIB" ]]; then
  echo "Using pre-existing metallib (skipping compile): $OUTPUT_METALLIB"
else
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
