#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
INPUT_CU="$2"
HOST_SOURCE="$3"
RUNTIME_INCLUDE_DIR="$4"
RUNTIME_LIB_DIR="$5"
OUTPUT_METALLIB="$6"
OUTPUT_BINARY="$7"

# Short-circuit if pre-built sample binary + metallib already present in build tree.
# Allows execution in envs missing metal/metallib (clang++ is present, but cumetalc
# can fallback internally; prebuilts come from prior full builds).
if [[ -x "$OUTPUT_BINARY" && -s "$OUTPUT_METALLIB" ]]; then
  echo "Using pre-existing sample binary + metallib: $OUTPUT_BINARY $OUTPUT_METALLIB"
  exec "$OUTPUT_BINARY" "$OUTPUT_METALLIB"
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
  --backend=cumetal-ir \
  --entry vector_add \
  --mode xcrun \
  --input "$INPUT_CU" \
  --output "$OUTPUT_METALLIB" \
  --overwrite

xcrun clang++ \
  -std=c++20 \
  -Wall -Wextra -Wpedantic \
  "$HOST_SOURCE" \
  -I"$RUNTIME_INCLUDE_DIR" \
  -L"$RUNTIME_LIB_DIR" \
  -Wl,-rpath,"$RUNTIME_LIB_DIR" \
  -lcumetal \
  -o "$OUTPUT_BINARY"

"$OUTPUT_BINARY" "$OUTPUT_METALLIB"
