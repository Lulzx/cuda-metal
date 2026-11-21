#!/usr/bin/env bash
# A runtime loop over a by-value struct of pointers lowers to a pointer that
# walks parameter space (`mov %rd, param`; `ld.param [%rd]`; `add %rd, 8`).
# The typed PTX path (--cuda-device --ptx-strict) must compile it and the
# launch must write every column (#221, stwo-zig Cairo memory execution).
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <cumetalc> <fixture.cu> <test-executable>" >&2
    exit 2
fi
if ! xcrun -f metal >/dev/null 2>&1 || ! xcrun -f metallib >/dev/null 2>&1; then
    echo "SKIP: xcrun metal is unavailable"
    exit 77
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/byval_pointer_walk.XXXXXX")"
trap 'rm -rf "$work"' EXIT
metallib="$work/byval_pointer_walk.metallib"

"$1" "$2" -o "$metallib" --cuda-device --ptx-strict --cuda-arch sm_80 \
    --entry byval_pointer_walk --overwrite
CUMETAL_CACHE_DIR="$work/cache" "$3" "$metallib"
