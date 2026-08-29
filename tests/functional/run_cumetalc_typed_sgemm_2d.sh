#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cumetalc> <runner> <source.cu> <output.metallib>" >&2
    exit 2
fi
if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: complete Metal toolchain unavailable"
    exit 77
fi

"$1" "$3" --backend=cumetal-ir --emit=metallib --no-link --overwrite \
    --fp64=fast48 -o "$4"
"$2" "$4"
