#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 7 ]]; then
    echo "usage: $0 <cumetalc> <runner> <source.cu> <output.metallib> <direct|ptx> <device|system|fence> [entry]" >&2
    exit 2
fi

cumetalc="$1"
runner="$2"
source_file="$3"
output="$4"
frontend="$5"
mode="$6"
entry="${7:-}"

if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: complete Metal toolchain unavailable"
    exit 77
fi

args=("${source_file}")
case "${frontend}" in
    direct) ;;
    ptx) args+=(--cuda-device) ;;
    *) echo "invalid frontend: ${frontend}" >&2; exit 2 ;;
esac
args+=(--backend=cumetal-ir --emit=metallib --no-link --overwrite
      --fp64=fast48 -o "${output}")
if [[ -n "${entry}" ]]; then
    args+=(--entry "${entry}")
fi

"${cumetalc}" "${args[@]}"
"${runner}" "${output}" "${mode}"
