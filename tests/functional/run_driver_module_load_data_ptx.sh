#!/usr/bin/env bash
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

# Tests cuModuleLoadData with PTX text + fatbin variants, plus launch of the loaded kernel.
# The launch verification requires being able to produce a usable metallib for the kernel;
# the direct lowering or packaging for 'vector_add' needs the metal compiler in this flow.
if ! command -v xcrun >/dev/null 2>&1; then
  echo "SKIP: xcrun not installed"
  exit 77
fi

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "SKIP: xcrun metal not available (driver ptx load+launch test requires it for this kernel)"
  exit 77
fi

if ! xcrun --find metallib >/dev/null 2>&1; then
  echo "SKIP: xcrun metallib not available"
  exit 77
fi

"$TEST_BINARY" "$PTX_PATH"
