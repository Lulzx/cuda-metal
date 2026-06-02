#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

# This exercises the fatbin PTX registration + direct Metal lowering for 'vector_add'
# (which generates .metal then requires xcrun metal to produce the .metallib).
# Without the metal compiler toolchain, skip.
if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available (direct lowering for vector_add requires it)"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

"$TEST_BINARY" "$PTX_PATH"
