#!/usr/bin/env bash
# A PTX kernel that needs the offline Metal compiler must be refused at
# cuModuleLoadData with CUDA_ERROR_JIT_COMPILER_NOT_FOUND and a one-time
# warning naming the fix when the toolchain is absent. DEVELOPER_DIR pointing
# nowhere makes xcrun fail exactly as it does after Xcode is uninstalled.
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"
CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "$CACHE_DIR"' EXIT

set +e
stderr="$(DEVELOPER_DIR=/nonexistent CUMETAL_CACHE_DIR="$CACHE_DIR" \
    "$TEST_BINARY" "$PTX_PATH" expect-missing 2>&1 >/dev/null)"
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
    echo "$stderr"
    exit 1
fi
if [ "$(grep -c "CUMETAL WARNING: kernel JIT needs Apple's offline Metal compiler" <<<"$stderr")" -ne 1 ]; then
    echo "FAIL: expected exactly one missing-toolchain warning"
    echo "$stderr"
    exit 1
fi
if find "$CACHE_DIR" -name '*.metallib' | grep -q .; then
    echo "FAIL: an unrunnable metallib was left in the cache"
    exit 1
fi

# Positive control: with a real toolchain the same module must load.
if xcrun --find metal >/dev/null 2>&1; then
    CUMETAL_CACHE_DIR="$CACHE_DIR" "$TEST_BINARY" "$PTX_PATH" expect-ok
else
    echo "NOTE: xcrun metal not available; positive control skipped"
fi
echo "PASS: missing Metal toolchain is reported at module load"
