#!/usr/bin/env bash
# FP64 support arithmetic must be bit-identical whatever float math flags the
# user picked. Before VF64 support was compiled in its own safe region,
# --use_fast_math defined Metal's fast-math macro for the whole translation
# unit, so VF64's sqrt seeds became fast::sqrt and fast48/wide48 sqrt and
# wide48 hypot returned different bits. VF64's own file-scope pragmas also
# leaked into user code, silently dropping the user's -fmad=true contraction.
#
# Native builds (cumetalc) are compared across default, --use_fast_math and
# -fmad=false; the PTX JIT is compared across CUMETAL_MSL_MATH_MODE=safe/fast.
# The emitter log must show the safe support region followed by the caller's
# restored contract.
set -euo pipefail

ROOT_DIR="${1:?usage: run_fp64_math_flag_invariance.sh <root> <build-dir>}"
BUILD_DIR="${2:?}"
CUMETALC="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cumetalc"
SRC="${ROOT_DIR}/tests/functional/fixtures/fp64_math_flag_invariance.cu"
OUT_DIR="${BUILD_DIR}/fp64_math_flag_invariance"

source "${ROOT_DIR}/tests/cuda_projects/_common.sh"
if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi
mkdir -p "${OUT_DIR}"
CACHE_ROOT="$(mktemp -d)"
trap 'rm -rf "${CACHE_ROOT}"' EXIT

digest() {
    local output
    output="$(CUMETAL_CACHE_DIR="$(mktemp -d "${CACHE_ROOT}/c.XXXXXX")" "$@" 2>&1)" || {
        echo "${output}" >&2
        return 1
    }
    grep '^DIGEST' <<<"${output}"
}

# Under fast48 double hypot is the binary32 fallback evaluated as user float
# code, so the user's flags legitimately apply to it; compare the other five.
fields_for() {
    if [[ "$1" == fast48 ]]; then echo "2-6"; else echo "2-7"; fi
}

status=0
for mode in fast48 wide48 ieee64; do
    reference=""
    for flags in "" "--use_fast_math" "-fmad=false"; do
        bin="${OUT_DIR}/native_${mode}${flags//[^a-z]/_}"
        # shellcheck disable=SC2086
        "${CUMETALC}" "${SRC}" --backend=cumetal-ir --fp64="${mode}" ${flags} \
            --overwrite -o "${bin}" >"${bin}.log" 2>&1 || {
            cat "${bin}.log"
            exit 1
        }
        got="$(digest "${bin}" | cut -d' ' -f"$(fields_for "${mode}")")"
        echo "native ${mode} ${flags:-default}: ${got}"
        if [[ -z "${reference}" ]]; then
            reference="${got}"
        elif [[ "${got}" != "${reference}" ]]; then
            echo "FAIL: native ${mode} FP64 results change under ${flags}"
            status=1
        fi
    done
done

cumetal_cuda_projects_compile_link "${ROOT_DIR}" "$(dirname "${SRC}")" "${OUT_DIR}" \
    "$(basename "${SRC}")" jit_probe >/dev/null
for mode in fast48 wide48 ieee64; do
    reference=""
    for math in safe fast; do
        got="$(CUMETAL_PTX_BACKEND=cumetal-ir CUMETAL_FP64_MODE="${mode}" \
            CUMETAL_MSL_MATH_MODE="${math}" digest "${OUT_DIR}/jit_probe" |
            cut -d' ' -f"$(fields_for "${mode}")")"
        echo "jit ${mode} ${math}: ${got}"
        if [[ -z "${reference}" ]]; then
            reference="${got}"
        elif [[ "${got}" != "${reference}" ]]; then
            echo "FAIL: PTX JIT ${mode} FP64 results change under math mode ${math}"
            status=1
        fi
    done
done

# The staged translation unit: safe support region, then the caller's mode.
check_layout() {
    local flags="$1" restored_math="$2" restored_contract="$3" log
    # shellcheck disable=SC2086
    log="$("${CUMETALC}" "${SRC}" --backend=cumetal-ir --fp64=wide48 ${flags} \
        --emit=metallib --overwrite -o "${OUT_DIR}/probe.metallib" 2>&1)" || {
        echo "${log}"
        return 1
    }
    local actual expected
    actual="$(sed -n '/^staged combined Metal source:/,/^\$ /p' <<<"${log}" |
        grep -E '^#(pragma|include)' |
        sed -e 's#^\#include ".*cumetal_fp64_inline_support.metal"$#SUPPORT#' \
            -e 's#^\#include ".*\.metal"$#USER#')"
    expected="$(printf '%s\n' \
        '#pragma METAL fp math_mode(safe)' '#pragma METAL fp contract(off)' \
        'SUPPORT' \
        "#pragma METAL fp math_mode(${restored_math})" \
        "#pragma METAL fp contract(${restored_contract})" 'USER')"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "FAIL: staged source for '${flags:-default}' is not safe support then caller mode:"
        echo "${actual}"
        return 1
    fi
    if ! grep -q -- 'metal -fno-strict-aliasing -fno-fast-math -ffp-contract=off -c .*combined.metal' \
        <<<"${log}"; then
        echo "FAIL: the support translation unit was not compiled IEEE-safe for '${flags:-default}'"
        return 1
    fi
    echo "layout ${flags:-default}: safe support, then math_mode(${restored_math}) contract(${restored_contract})"
}
check_layout "--use_fast_math" fast fast || status=1
check_layout "-fmad=false" safe off || status=1
check_layout "" safe fast || status=1

if [[ ${status} -ne 0 ]]; then
    exit 1
fi
echo "PASS: FP64 support results are independent of user float math flags"
