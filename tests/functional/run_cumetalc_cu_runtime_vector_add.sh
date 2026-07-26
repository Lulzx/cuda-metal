#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
TEST_BINARY="$2"
INPUT_CU="$3"
OUTPUT_METALLIB="$4"

# Regenerate the metallib from source whenever the toolchain allows it. Reusing a pre-existing
# artifact is a last resort for toolchain-less environments, NOT a fast path: this script used to
# check for the artifact first, so once any build had produced one, the test stopped exercising
# cumetalc entirely and would have stayed green through a total compiler regression.
toolchain_available() {
  command -v xcrun >/dev/null 2>&1 &&
    xcrun --find clang++ >/dev/null 2>&1 &&
    xcrun --find metal >/dev/null 2>&1 &&
    xcrun --find metallib >/dev/null 2>&1
}

if ! toolchain_available; then
  if [[ -s "$OUTPUT_METALLIB" ]]; then
    echo "WARNING: Metal toolchain unavailable; running against a previously generated metallib."
    echo "         This does not verify the current cumetalc: $OUTPUT_METALLIB"
    exec "$TEST_BINARY" "$OUTPUT_METALLIB"
  fi
fi

rm -f "$OUTPUT_METALLIB"

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

"$TEST_BINARY" "$OUTPUT_METALLIB"
