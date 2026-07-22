#!/usr/bin/env bash
# Tests the binary shim JIT cache:
#   1. First run of the registration fatbin PTX test compiles the kernel (cache miss).
#   2. Second run reuses the cached metallib (cache hit), skipping xcrun.
#
# Uses CUMETAL_CACHE_DIR to sandbox the cache so we don't interfere with the
# real user cache at $HOME/Library/Caches/io.cumetal.
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
    exit 77
fi
if ! xcrun --find metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal not available (jit cache test for vector_add direct lowering requires it)"
    exit 77
fi
if ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: xcrun metallib not available"
    exit 77
fi

# Sandbox the JIT cache so this test is repeatable and clean.
JIT_CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "$JIT_CACHE_DIR"' EXIT

# First run: expect a JIT compile (cache miss).
first_run_stderr=$(
    CUMETAL_CACHE_DIR="$JIT_CACHE_DIR" \
    CUMETAL_DEBUG_REGISTRATION=1 \
    "$TEST_BINARY" "$PTX_PATH" 2>&1 >/dev/null
)

if ! echo "$first_run_stderr" | grep -q "jit cache miss"; then
    echo "FAIL: first run did not report a jit cache miss"
    echo "stderr was:"
    echo "$first_run_stderr"
    exit 1
fi

if ! echo "$first_run_stderr" | grep -q "args=lazy"; then
    echo "FAIL: fatbin registration eagerly built the PTX argument index"
    echo "stderr was:"
    echo "$first_run_stderr"
    exit 1
fi

# The direct PTX-to-MSL path caches source for newLibraryWithSource; the LLVM
# path caches an AOT metallib. Either is a valid persistent JIT artifact.
cache_file=$(find "$JIT_CACHE_DIR/registration-jit" \
    \( -name "*.metallib" -o -name "*.metal" \) 2>/dev/null | head -1)
if [ -z "$cache_file" ]; then
    echo "FAIL: no .metallib or .metal file found in JIT cache dir after first run"
    exit 1
fi

# Second run: expect a cache hit (no xcrun re-compilation).
second_run_stderr=$(
    CUMETAL_CACHE_DIR="$JIT_CACHE_DIR" \
    CUMETAL_DEBUG_REGISTRATION=1 \
    "$TEST_BINARY" "$PTX_PATH" 2>&1 >/dev/null
)

if ! echo "$second_run_stderr" | grep -q "jit cache hit"; then
    echo "FAIL: second run did not report a jit cache hit"
    echo "stderr was:"
    echo "$second_run_stderr"
    exit 1
fi

# Second run must also succeed (output goes to stdout, test binary exits 0).
CUMETAL_CACHE_DIR="$JIT_CACHE_DIR" "$TEST_BINARY" "$PTX_PATH"

echo "PASS: registration JIT cache miss on first run, cache hit on second run"
