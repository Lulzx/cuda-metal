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
      --fp64=fast48)
if [[ -n "${entry}" ]]; then
    args+=(--entry "${entry}")
fi

"${cumetalc}" "${args[@]}" -o "${output}"
if [[ "$mode" == system ]]; then
    values_output="${output%.metallib}-values.metallib"
    "${cumetalc}" "${args[@]}" --entry system_atomic_values -o "${values_output}"
fi
log="$(mktemp "${output}.run.XXXXXX")"
trap 'rm -f "$log"' EXIT
status=0
CUMETAL_TRACE_GPU=1 "${runner}" "${output}" "${mode}" >"$log" 2>&1 || status=$?
if [[ $status -eq 0 && "$mode" == system ]]; then
    CUMETAL_TRACE_GPU=1 "${runner}" "${values_output}" system-values >>"$log" 2>&1 || status=$?
fi
cat "$log"
if [[ $status -ne 0 ]]; then
    exit "$status"
fi
expected_launches=1
if [[ "$mode" == system ]]; then
    expected_launches=2
fi
if [[ $(grep -Ec '^CUMETAL_PROVENANCE event=kernel_launch .*device=apple_gpu .*launch_success=true duration_ns=[1-9][0-9]* ' "$log") -ne $expected_launches ]]; then
    echo "FAIL: expected ${expected_launches} completed Apple GPU launches with positive duration" >&2
    exit 1
fi
