#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REF_DIR="$ROOT_DIR/tests/air_abi/reference"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
# Generated artifacts go to REF_OUT_DIR (default: next to the sources).
# Tests that rebuild the kernel pass a private directory.
REF_OUT_DIR="${REF_OUT_DIR:-$REF_DIR}"
mkdir -p "$REF_OUT_DIR"

METAL_FILE="$REF_DIR/vector_add.metal"
LLVM_IR_FILE="$REF_DIR/vector_add_air.ll"
AIR_FILE="$REF_OUT_DIR/reference.air"
METALLIB_FILE="$REF_OUT_DIR/reference.metallib"
EXPERIMENTAL_METALLIB_FILE="$REF_OUT_DIR/reference.experimental.metallib"

# Build in a private directory and rename into place: parallel tests share the
# default output, and an in-place write let a concurrent reader load a missing
# or half-written library.
STAGE_DIR="$(mktemp -d "$REF_OUT_DIR/.generate.XXXXXX")"
trap 'rm -rf "$STAGE_DIR"' EXIT
publish() { mv -f "$STAGE_DIR/$(basename "$1")" "$1"; }

if ! command -v xcrun >/dev/null 2>&1; then
  echo "xcrun not found" >&2
  exit 1
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "xcrun metal is unavailable. Generating experimental reference metallib instead." >&2
  if [[ -x "$BUILD_DIR/cumetal-air-emitter" ]]; then
    "$BUILD_DIR/cumetal-air-emitter" \
      --mode experimental \
      --input "$LLVM_IR_FILE" \
      --output "$STAGE_DIR/$(basename "$EXPERIMENTAL_METALLIB_FILE")" \
      --overwrite
    publish "$EXPERIMENTAL_METALLIB_FILE"
    echo "Generated: $EXPERIMENTAL_METALLIB_FILE"
    exit 0
  fi
  echo "Build cumetal-air-emitter first or install full Xcode." >&2
  exit 1
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "xcrun metallib is unavailable. Generating experimental reference metallib instead." >&2
  if [[ -x "$BUILD_DIR/cumetal-air-emitter" ]]; then
    "$BUILD_DIR/cumetal-air-emitter" \
      --mode experimental \
      --input "$LLVM_IR_FILE" \
      --output "$STAGE_DIR/$(basename "$EXPERIMENTAL_METALLIB_FILE")" \
      --overwrite
    publish "$EXPERIMENTAL_METALLIB_FILE"
    echo "Generated: $EXPERIMENTAL_METALLIB_FILE"
    exit 0
  fi
  echo "Build cumetal-air-emitter first or install full Xcode." >&2
  exit 1
fi

xcrun metal -c "$METAL_FILE" -o "$STAGE_DIR/$(basename "$AIR_FILE")"
xcrun metallib "$STAGE_DIR/$(basename "$AIR_FILE")" -o "$STAGE_DIR/$(basename "$METALLIB_FILE")"
publish "$AIR_FILE"
publish "$METALLIB_FILE"

echo "Generated: $METALLIB_FILE"

if [[ -x "$BUILD_DIR/air_inspect" ]]; then
  "$BUILD_DIR/air_inspect" "$METALLIB_FILE"
fi

if [[ -x "$BUILD_DIR/air_validate" ]]; then
  "$BUILD_DIR/air_validate" "$METALLIB_FILE" --xcrun || true
fi
