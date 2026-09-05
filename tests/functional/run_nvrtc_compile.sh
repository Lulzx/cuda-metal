#!/usr/bin/env bash
# The NVRTC shim compiles through cumetalc, which needs Xcode's Metal
# toolchain. Skip rather than fail where that toolchain is absent.
set -euo pipefail

TEST_BINARY="$1"
CUMETALC="$2"

if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not found"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal is unavailable"
  exit 77
fi

if [[ ! -x "$CUMETALC" ]]; then
  echo "SKIP: cumetalc not found at $CUMETALC"
  exit 77
fi

exec "$TEST_BINARY" "$CUMETALC"
