#!/usr/bin/env bash
set -euo pipefail

GENERATE_SCRIPT="$1"
TEST_BINARY="$2"
REFERENCE_METALLIB="$3"

# Reuse a previously generated metallib only when the Metal toolchain is genuinely missing (e.g.
# only the runtime Metal.framework is present). Checking for the artifact first, as this script
# used to, meant edits to the reference kernel were silently ignored on every developer machine.
if ! (command -v xcrun >/dev/null 2>&1 && xcrun --find metal >/dev/null 2>&1 &&
      xcrun --find metallib >/dev/null 2>&1); then
  if [[ -s "$REFERENCE_METALLIB" ]]; then
    echo "WARNING: Metal toolchain unavailable; using a previously generated metallib."
    echo "         The reference kernel was not rebuilt: $REFERENCE_METALLIB"
    exec "$TEST_BINARY" "$REFERENCE_METALLIB"
  fi
fi

# Rebuild the reference kernel privately. About twenty tests use this script and
# used to delete and regenerate the shared path in place, so under parallel
# ctest one test loaded a library another was rewriting and failed at random.
# The shared path is only the toolchain-missing fallback above.
PRIVATE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/cumetal-reference.XXXXXX")"
trap 'rm -rf "$PRIVATE_DIR"' EXIT
REFERENCE_METALLIB="$PRIVATE_DIR/reference.metallib"

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

REF_OUT_DIR="$PRIVATE_DIR" "$GENERATE_SCRIPT"

if [[ ! -s "$REFERENCE_METALLIB" ]]; then
  echo "SKIP: reference metallib not available at $REFERENCE_METALLIB"
  exit 77
fi

"$TEST_BINARY" "$REFERENCE_METALLIB"
