#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
INPUT_CU="$2"
HOST_SOURCE="$3"
RUNTIME_INCLUDE_DIR="$4"
RUNTIME_LIB_DIR="$5"
OUTPUT_METALLIB="$6"
OUTPUT_BINARY="$7"

# Always rebuild from source. This used to short-circuit on a pre-existing binary + metallib in
# the build tree, which meant the test verified whatever a previous build had left behind: it
# stayed green across source edits, and would have stayed green if the sample were deleted. Same
# stale-artifact green-wash that cumetal_cuda_projects_compile_link was fixed for.
rm -f "$OUTPUT_BINARY" "$OUTPUT_METALLIB"

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
