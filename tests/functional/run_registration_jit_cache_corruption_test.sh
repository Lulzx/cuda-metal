#!/usr/bin/env bash
# A torn registration-JIT cache entry must be detected and recompiled, never
# reused. Before this was checked, one metallib whose header had been zeroed
# (a process killed mid-write) failed every later launch of that kernel with
# cudaErrorInvalidValue until someone deleted the cache by hand. The same
# now holds for direct-MSL entries: the metadata sidecar records the
# artifact's size + FNV-1a digest, and a hit that fails the integrity check
# is dropped and recompiled.
#
#   1. Cold run compiles the kernel into a sandboxed cache (one artifact
#      entry, .metallib or .metal depending on the lowering path).
#   2. The entry's header is zeroed; the next run must report the bad entry,
#      recompile, compute the right answer, and leave an intact entry.
#   3. Same for a truncated entry.
#   4. No staging (.partial-*) files may be left behind.
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"

if ! xcrun --find metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal not available"
    exit 77
fi

CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "$CACHE_DIR"' EXIT

run() {
    CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_DEBUG_REGISTRATION=1 "$TEST_BINARY" "$PTX_PATH" 2>&1
}

entry() {
    find "$CACHE_DIR/registration-jit" -maxdepth 1 -type f \
        \( -name "*.metallib" -o -name "*.metal" \) \
        ! -name "*.partial-*" -print
}

is_intact() {
    xcrun python3 - "$1" <<'PY'
import os, struct, sys
path = sys.argv[1]
data = open(path, "rb").read(0x18)
if path.endswith(".metallib"):
    ok = len(data) == 0x18 and data[:4] == b"MTLB" and \
         struct.unpack_from("<Q", data, 0x10)[0] == os.path.getsize(path)
else:
    # Direct-MSL entry: cumetalc output starts with its provenance comment
    # and ends with the kernel's closing brace.
    tail = open(path, "rb").read()[-64:].strip()
    ok = len(data) == 0x18 and data[:2] == b"//" and \
         os.path.getsize(path) > 0x400 and tail.endswith(b"}")
sys.exit(0 if ok else 1)
PY
}

cold="$(run)" || { echo "FAIL: cold run failed"; echo "$cold"; exit 1; }
if ! grep -q "jit cache miss:" <<<"$cold"; then
    echo "FAIL: cold run did not miss the sandboxed cache"
    echo "$cold"
    exit 1
fi
cached="$(entry)"
if [ "$(wc -l <<<"$cached" | tr -d ' ')" -ne 1 ] || [ -z "$cached" ]; then
    echo "FAIL: expected exactly one cache entry, got:"
    echo "${cached:-<none>}"
    exit 1
fi
is_intact "$cached" || { echo "FAIL: freshly written cache entry is not intact"; exit 1; }

corrupt_and_check() {
    local label="$1"
    local how="$2"
    xcrun python3 - "$cached" "$how" <<'PY'
import sys
path, how = sys.argv[1], sys.argv[2]
data = bytearray(open(path, "rb").read())
if how == "zero-header":
    data[:64] = bytes(64)
elif how == "truncate":
    data = data[: len(data) // 3]
open(path, "wb").write(data)
PY
    is_intact "$cached" && { echo "FAIL: $label corruption did not take"; exit 1; }
    local out
    out="$(run)" || { echo "FAIL: run after $label corruption failed"; echo "$out"; exit 1; }
    if ! grep -qE "is not a complete metallib|failed integrity check" <<<"$out"; then
        echo "FAIL: $label cache entry was not detected"
        echo "$out"
        exit 1
    fi
    if ! grep -q "^PASS" <<<"$out"; then
        echo "FAIL: kernel did not compute correctly after $label recompile"
        echo "$out"
        exit 1
    fi
    is_intact "$cached" || { echo "FAIL: $label entry was not rewritten intact"; exit 1; }
}

corrupt_and_check "zero-header" zero-header
corrupt_and_check "truncated" truncate

leftovers="$(find "$CACHE_DIR/registration-jit" -name "*.partial-*" -print)"
if [ -n "$leftovers" ]; then
    echo "FAIL: staging files left in the cache:"
    echo "$leftovers"
    exit 1
fi

echo "PASS: torn registration JIT cache entries are detected and recompiled"
