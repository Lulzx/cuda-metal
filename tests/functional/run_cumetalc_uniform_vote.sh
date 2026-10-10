#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 <cumetalc> <runner> <source.cu> <output.metallib> <direct|ptx>" >&2
    exit 2
fi
if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: complete Metal toolchain unavailable"
    exit 77
fi

args=("$3" --backend=cumetal-ir --emit=metallib --no-link --overwrite
      --entry warp_uniform_vote -o "$4")
case "$5" in
    direct) ;;
    ptx) args+=(--cuda-device) ;;
    *) echo "FAIL: invalid uniform-vote frontend '$5'" >&2; exit 2 ;;
esac
export CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0
export CUMETAL_ENABLE_LLMC_CPU_EMULATION=0
export CUMETAL_DISABLE_LLMC_EMULATION=1
cache="$(mktemp -d "${4}.cache.XXXXXX")"
log="$(mktemp "${4}.run.XXXXXX")"
trap 'rm -rf "$cache"; rm -f "$log"' EXIT
export CUMETAL_CACHE_DIR="$cache"
"$1" "${args[@]}"
status=0
CUMETAL_TRACE_GPU=1 "$2" "$4" >"$log" 2>&1 || status=$?
cat "$log"
if [[ $status -ne 0 ]]; then
    exit "$status"
fi
if ! grep -q '^PASS: uniform and negated warp votes validated' "$log" ||
   [[ $(grep -c '^CUMETAL_PROVENANCE event=kernel_launch .*device=apple_gpu .*launch_success=true ' "$log") -ne 8 ]]; then
    echo "FAIL: expected numerical validation and eight completed Apple GPU launches" >&2
    exit 1
fi
