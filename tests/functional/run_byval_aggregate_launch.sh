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

"$1" --cuda-device --overwrite "$2" -o "$4"

# The sidecar is what carries the aggregate's real size to the launch; without
# it the driver falls back to guessing and the assertion below is vacuous.
abi="$4.cumetal-abi"
if ! grep -q '^arg bytes 32$' "$abi"; then
    echo "FAIL: expected a 32-byte by-value aggregate argument in $abi" >&2
    cat "$abi" >&2
    exit 1
fi

"$3" "$4"
echo "PASS: by-value aggregate kernel parameters reach the device intact"
