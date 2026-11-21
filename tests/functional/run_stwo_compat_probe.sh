#!/usr/bin/env bash
# The constructs stwo-zig patched CuMetal for or shimmed in a compatibility
# header (#221), compiled without that header and checked on the GPU, through
# the typed PTX frontend stwo-zig uses (--cuda-device --ptx-strict) and through
# the direct typed frontend.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cumetalc> <fixture.cu> <test-executable> <ptx|direct>" >&2
    exit 2
fi
if ! xcrun -f metal >/dev/null 2>&1 || ! xcrun -f metallib >/dev/null 2>&1; then
    echo "SKIP: xcrun metal is unavailable"
    exit 77
fi

work="$(mktemp -d "${TMPDIR:-/tmp}/stwo_compat_probe.XXXXXX")"
trap 'rm -rf "$work"' EXIT
metallib="$work/stwo_compat_probe.metallib"

case "$4" in
    ptx) "$1" "$2" -o "$metallib" --cuda-device --ptx-strict --cuda-arch sm_80 \
             --entry stwo_compat_probe --overwrite ;;
    direct) "$1" "$2" -o "$metallib" --backend=cumetal-ir --emit=metallib --no-link \
                --entry stwo_compat_probe --overwrite ;;
    *) echo "FAIL: unknown frontend '$4'" >&2; exit 2 ;;
esac
CUMETAL_CACHE_DIR="$work/cache" "$3" "$metallib"
