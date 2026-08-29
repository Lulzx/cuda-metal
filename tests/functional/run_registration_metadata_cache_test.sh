#!/usr/bin/env bash
# Prove that a warm registration-JIT cache restores non-empty compiler metadata
# without lowering PTX again. Device printf depends on its recovered format
# table, so the second process checks both the fast path and its semantics.
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: Metal compiler tools are unavailable"
    exit 77
fi

CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "$CACHE_DIR"' EXIT

run_once() {
    CUMETAL_CACHE_DIR="$CACHE_DIR" \
    CUMETAL_DEBUG_REGISTRATION=1 \
        "$TEST_BINARY" "$PTX_PATH" 2>&1
}

cold="$(run_once)"
if ! grep -q "jit cache miss:" <<<"$cold" ||
   ! grep -q "PASS: registration-path device printf" <<<"$cold"; then
    echo "FAIL: cold device-printf registration did not compile and execute"
    echo "$cold"
    exit 1
fi

metadata_count="$(find "$CACHE_DIR/registration-jit" -maxdepth 1 \
    -type f -name '*.metadata' -print | wc -l | tr -d ' ')"
if [ "$metadata_count" -ne 1 ]; then
    echo "FAIL: cold device-printf registration created $metadata_count metadata sidecars"
    exit 1
fi

warm="$(run_once)"
if ! grep -q "jit metadata cache hit:" <<<"$warm" ||
   grep -q "using .* lowering path" <<<"$warm" ||
   ! grep -q "PASS: registration-path device printf" <<<"$warm"; then
    echo "FAIL: warm device-printf registration did not restore cached metadata"
    echo "$warm"
    exit 1
fi

echo "PASS: warm registration cache restores device-printf metadata without PTX lowering"
