#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cumetalc> <fixture.cu> <test-executable> <output.metallib>" >&2
    exit 2
fi

if ! xcrun -f metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal is unavailable"
    exit 77
fi

# --emit metallib, not --cuda-device: this is the invocation CuMetal's NVRTC
# shim builds, and it is the one that puts every kernel of the translation unit
# into the metallib. (--cuda-device emits only the first, which is why the
# kernel under test here is deliberately the second one.)
"$1" "$2" -o "$4" --emit metallib --cuda-arch sm_80 --overwrite

# The sidecar is what carries the aggregate's real size to the launch; without
# it the driver falls back to guessing and the assertion below is vacuous. It
# must also describe both kernels, since the one under test is the second.
abi="$4.cumetal-abi"
if ! grep -q '^arg bytes 32$' "$abi"; then
    echo "FAIL: expected a 32-byte by-value aggregate argument in $abi" >&2
    cat "$abi" >&2
    exit 1
fi
if [[ "$(grep -c '^kernel ' "$abi")" -ne 2 ]]; then
    echo "FAIL: expected an ABI block for each of the two kernels in $abi" >&2
    cat "$abi" >&2
    exit 1
fi

"$3" "$4"
echo "PASS: by-value aggregate kernel parameters reach the device intact"
