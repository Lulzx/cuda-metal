#!/usr/bin/env bash
# Tests the binary shim JIT cache:
#   1. First run with one libcumetal build compiles the kernel (cache miss).
#   2. Second run with that build reuses the cached artifact (cache hit).
#   3. A libcumetal image with a distinct LC_UUID cannot reuse the first build's
#      artifact (cache miss), but does reuse its own artifact on its second run.
#
# Uses CUMETAL_CACHE_DIR to sandbox the cache so we don't interfere with the
# real user cache at $HOME/Library/Caches/io.cumetal.
set -euo pipefail

TEST_BINARY="$1"
PTX_PATH="$2"
LIBCUMETAL_PATH="$3"

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
if ! xcrun --find python3 >/dev/null 2>&1; then
    echo "SKIP: xcrun python3 not available"
    exit 77
fi
if [ ! -f "$LIBCUMETAL_PATH" ]; then
    echo "FAIL: libcumetal not found at $LIBCUMETAL_PATH"
    exit 1
fi

# Sandbox the JIT cache so this test is repeatable and clean.
JIT_CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "$JIT_CACHE_DIR"' EXIT

BUILD_A_DIR="$JIT_CACHE_DIR/build-a"
BUILD_B_DIR="$JIT_CACHE_DIR/build-b"
mkdir -p "$BUILD_A_DIR" "$BUILD_B_DIR"
cp "$LIBCUMETAL_PATH" "$BUILD_A_DIR/libcumetal.dylib"
cp "$LIBCUMETAL_PATH" "$BUILD_B_DIR/libcumetal.dylib"

# Model a second libcumetal build without recompiling the entire runtime: alter
# only the copied image's LC_UUID. The production cache code discovers this
# value from the loaded Mach-O image via dladdr, so this still exercises the
# real build-identity path. XOR makes the alternate UUID deterministic and
# guarantees that it differs from the original.
xcrun python3 - "$BUILD_B_DIR/libcumetal.dylib" <<'PY'
import struct
import sys

path = sys.argv[1]
with open(path, "rb") as source:
    image = bytearray(source.read())

MH_MAGIC_64 = 0xFEEDFACF
LC_UUID = 0x1B
if len(image) < 32 or struct.unpack_from("<I", image, 0)[0] != MH_MAGIC_64:
    raise SystemExit(f"FAIL: {path} is not a thin 64-bit little-endian Mach-O image")

ncmds = struct.unpack_from("<I", image, 16)[0]
offset = 32
uuid_offset = None
for _ in range(ncmds):
    if offset + 8 > len(image):
        raise SystemExit("FAIL: truncated Mach-O load-command table")
    command, command_size = struct.unpack_from("<II", image, offset)
    if command_size < 8 or offset + command_size > len(image):
        raise SystemExit("FAIL: invalid Mach-O load command")
    if command == LC_UUID:
        if command_size < 24:
            raise SystemExit("FAIL: invalid LC_UUID load command")
        uuid_offset = offset + 8
        break
    offset += command_size

if uuid_offset is None:
    raise SystemExit("FAIL: libcumetal has no LC_UUID load command")

image[uuid_offset] ^= 0xFF
with open(path, "wb") as destination:
    destination.write(image)
PY

# Editing a Mach-O invalidates its signature. Sign both fixture copies so dyld
# treats them exactly like normal local development builds.
/usr/bin/xattr -d com.apple.provenance "$BUILD_A_DIR/libcumetal.dylib" 2>/dev/null || true
/usr/bin/xattr -d com.apple.provenance "$BUILD_B_DIR/libcumetal.dylib" 2>/dev/null || true
/usr/bin/codesign --force --sign - "$BUILD_A_DIR/libcumetal.dylib"
/usr/bin/codesign --force --sign - "$BUILD_B_DIR/libcumetal.dylib"

build_a_uuid="$(xcrun dwarfdump --uuid "$BUILD_A_DIR/libcumetal.dylib")"
build_b_uuid="$(xcrun dwarfdump --uuid "$BUILD_B_DIR/libcumetal.dylib")"
if [ "$build_a_uuid" = "$build_b_uuid" ]; then
    echo "FAIL: libcumetal fixture UUIDs are identical"
    echo "build A: $build_a_uuid"
    echo "build B: $build_b_uuid"
    exit 1
fi

run_with_build() {
    local build_dir="$1"
    DYLD_LIBRARY_PATH="$build_dir${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}" \
    CUMETAL_CACHE_DIR="$JIT_CACHE_DIR" \
    CUMETAL_DEBUG_REGISTRATION=1 \
    "$TEST_BINARY" "$PTX_PATH"
}

first_cache_event() {
    grep -m 1 -E "jit cache (miss|hit):" <<<"$1" || true
}

cache_artifacts() {
    find "$JIT_CACHE_DIR/registration-jit" -maxdepth 1 -type f \
        \( -name "*.metallib" -o -name "*.metal" \) -print 2>/dev/null | sort
}

assert_first_event() {
    local stderr_text="$1"
    local expected="$2"
    local label="$3"
    local event
    event="$(first_cache_event "$stderr_text")"
    if [[ "$event" != *"jit cache $expected:"* ]]; then
        echo "FAIL: $label did not begin with a jit cache $expected"
        echo "first cache event: ${event:-<none>}"
        echo "stderr was:"
        echo "$stderr_text"
        exit 1
    fi
}

# Build A: cold miss, then cross-process reuse.
build_a_first_stderr="$(run_with_build "$BUILD_A_DIR" 2>&1 >/dev/null)"
assert_first_event "$build_a_first_stderr" "miss" "build A first run"
if ! grep -q "args=lazy" <<<"$build_a_first_stderr"; then
    echo "FAIL: fatbin registration eagerly built the PTX argument index"
    echo "stderr was:"
    echo "$build_a_first_stderr"
    exit 1
fi

build_a_artifacts="$(cache_artifacts)"
if [ "$(wc -l <<<"$build_a_artifacts" | tr -d ' ')" -ne 1 ]; then
    echo "FAIL: build A first run did not create exactly one cache artifact"
    echo "artifacts were:"
    echo "$build_a_artifacts"
    exit 1
fi

build_a_second_stderr="$(run_with_build "$BUILD_A_DIR" 2>&1 >/dev/null)"
assert_first_event "$build_a_second_stderr" "hit" "build A second run"
if [ "$(cache_artifacts)" != "$build_a_artifacts" ]; then
    echo "FAIL: build A cache contents changed on reuse"
    exit 1
fi

# Build B has identical PTX and policy but a distinct libcumetal LC_UUID. It
# must create a separate entry, then reuse only that entry on its second run.
build_b_first_stderr="$(run_with_build "$BUILD_B_DIR" 2>&1 >/dev/null)"
assert_first_event "$build_b_first_stderr" "miss" "build B first run"

both_build_artifacts="$(cache_artifacts)"
if [ "$(wc -l <<<"$both_build_artifacts" | tr -d ' ')" -ne 2 ]; then
    echo "FAIL: distinct libcumetal UUID did not create a second cache artifact"
    echo "artifacts were:"
    echo "$both_build_artifacts"
    exit 1
fi
if ! grep -Fqx "$build_a_artifacts" <<<"$both_build_artifacts"; then
    echo "FAIL: build A cache artifact disappeared after build B run"
    exit 1
fi

build_b_second_stderr="$(run_with_build "$BUILD_B_DIR" 2>&1 >/dev/null)"
assert_first_event "$build_b_second_stderr" "hit" "build B second run"
if [ "$(cache_artifacts)" != "$both_build_artifacts" ]; then
    echo "FAIL: build B cache contents changed on reuse"
    exit 1
fi

echo "PASS: registration JIT cache reuses artifacts within one libcumetal UUID and isolates distinct UUIDs"
