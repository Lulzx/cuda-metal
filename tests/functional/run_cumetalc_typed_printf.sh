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

frontend_args=()
if [[ "$5" == ptx ]]; then
    frontend_args+=(--cuda-device)
elif [[ "$5" != direct ]]; then
    echo "FAIL: invalid typed printf frontend '$5'" >&2
    exit 2
fi

# Bash 3.2 treats an empty array as unset under nounset; preserve zero arguments.
"$1" "$3" ${frontend_args[@]+"${frontend_args[@]}"} --backend=cumetal-ir --emit=metallib \
    --no-link --overwrite --fp64=fast48 -o "$4"
# Each kernel taking an int contributes exactly one caller `arg bytes 4`; the
# hidden ring buffer and capacity must not appear as extra caller args. The
# direct frontend emits every kernel (print_coordinates and
# print_mixed_values); the --cuda-device PTX run emits only the first.
expected_int_args=2
if [[ "$5" == ptx ]]; then expected_int_args=1; fi
if [[ $(grep -c '^arg bytes 4$' "$4.cumetal-abi") -ne ${expected_int_args} ]]; then
    echo "FAIL: typed printf hidden ring ABI leaked into caller arguments" >&2
    exit 1
fi
"$2" "$4"
