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

"$1" --cuda-device --mode xcrun --ptx-strict \
    --entry nested_device_pointer --overwrite "$2" -o "$4"

CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 "$3" "$4"
echo "PASS: nested CUDA device pointers use native Metal GPU addresses"
