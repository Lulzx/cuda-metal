#!/usr/bin/env bash
set -euo pipefail

GENERATE_SCRIPT="$1"
TEST_BINARY="$2"
REFERENCE_METALLIB="$3"

# If a pre-built or previously generated metallib already exists, use it directly.
# This allows functional tests to run in environments without the full Xcode
# metal/metallib toolchain (e.g. only runtime Metal.framework available).
if [[ -s "$REFERENCE_METALLIB" ]]; then
  echo "Using pre-existing metallib (no xcrun needed): $REFERENCE_METALLIB"
  exec "$TEST_BINARY" "$REFERENCE_METALLIB"
fi

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

"$GENERATE_SCRIPT"

if [[ ! -s "$REFERENCE_METALLIB" ]]; then
  echo "SKIP: reference metallib not available at $REFERENCE_METALLIB"
  exit 77
fi

"$TEST_BINARY" "$REFERENCE_METALLIB"
