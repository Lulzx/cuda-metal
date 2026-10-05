#!/usr/bin/env bash
# Double libdevice calls (exp, log, pow, ...) run in binary32 except where the
# typed backend calls VF64's correctly rounded functions under ieee64. The
# runtime must say so once per process when the fallback applies -- on a warm
# JIT cache hit too, where no MSL is generated -- and stay quiet for pure FP64
# arithmetic and for the ieee64 VF64 calls.
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
# shellcheck source=tests/cuda_projects/_common.sh
source "${ROOT_DIR}/tests/cuda_projects/_common.sh"
if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

SRC_DIR="${ROOT_DIR}/tests/functional/fp64_libdevice_warning"
OUT_DIR="${BUILD_DIR}/fp64_libdevice_warning"
mkdir -p "${OUT_DIR}"
cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" "${SRC_DIR}" "${OUT_DIR}" fp64_libdevice_warning.cu fp64_libdevice_warning

WARNING="CUMETAL WARNING: a kernel calls double-precision math functions"
CACHE_ROOT="$(mktemp -d)"
trap 'rm -rf "${CACHE_ROOT}"' EXIT

# expect <backend> <fp64-mode> <kernel> <expected-warning-count> <label>
expect() {
    local output count
    output="$(CUMETAL_PTX_BACKEND="$1" CUMETAL_FP64_MODE="$2" \
        CUMETAL_CACHE_DIR="${CACHE_ROOT}/$1-$2" "${OUT_DIR}/fp64_libdevice_warning" "$3" 2>&1)" || {
        echo "${output}"
        echo "FAIL: $5: program failed"
        exit 1
    }
    if ! grep -q "^RAN $3 " <<<"${output}"; then
        echo "${output}"
        echo "FAIL: $5: kernel did not run"
        exit 1
    fi
    count="$(grep -c "${WARNING}" <<<"${output}" || true)"
    if [ "${count}" -ne "$4" ]; then
        echo "${output}"
        echo "FAIL: $5: expected $4 binary32-fallback warning(s), saw ${count}"
        exit 1
    fi
    if [ "$4" -ne 0 ] && ! grep -q "CUMETAL_FP64_MODE=$2\." <<<"${output}"; then
        echo "${output}"
        echo "FAIL: $5: warning does not name the active mode"
        exit 1
    fi
}

expect legacy ieee64 exp 1 "legacy cold compile"
expect legacy ieee64 exp 1 "legacy warm cache hit"
expect legacy ieee64 arith 0 "legacy pure arithmetic"
expect cumetal-ir wide48 exp 1 "cumetal-ir wide48 cold compile"
expect cumetal-ir wide48 exp 1 "cumetal-ir wide48 warm cache hit"
expect cumetal-ir ieee64 exp 0 "cumetal-ir ieee64 calls VF64 exp"
expect cumetal-ir ieee64 arith 0 "cumetal-ir pure arithmetic"

# The offline compiler says the same thing, and only when it applies.
CUMETALC="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cumetalc"
OFFLINE="cumetalc: warning: double-precision math functions"
for frontend in direct ptx; do
    flags=(--emit=msl --overwrite)
    [ "${frontend}" = ptx ] && flags+=(--cuda-device --backend=cumetal-ir)
    with_exp="$("${CUMETALC}" "${SRC_DIR}/fp64_libdevice_warning.cu" --fp64=wide48 "${flags[@]}" \
        -o "${OUT_DIR}/offline-${frontend}.metal" 2>&1)"
    vf64_exp="$("${CUMETALC}" "${SRC_DIR}/fp64_libdevice_warning.cu" --fp64=ieee64 "${flags[@]}" \
        -o "${OUT_DIR}/offline-${frontend}-ieee64.metal" 2>&1)"
    without="$("${CUMETALC}" "${ROOT_DIR}/tests/cuda_projects/fp64_reciprocal/fp64_reciprocal.cu" \
        --fp64=wide48 "${flags[@]}" -o "${OUT_DIR}/offline-${frontend}-arith.metal" 2>&1)"
    if ! grep -q "${OFFLINE}" <<<"${with_exp}" || grep -q "${OFFLINE}" <<<"${vf64_exp}" ||
        grep -q "${OFFLINE}" <<<"${without}"; then
        echo "${with_exp}"
        echo "${vf64_exp}"
        echo "${without}"
        echo "FAIL: cumetalc ${frontend}: warning missing for wide48 exp, or present for ieee64 exp or pure arithmetic"
        exit 1
    fi
    if ! grep -q 'vf64_exp_rne(' "${OUT_DIR}/offline-${frontend}-ieee64.metal"; then
        echo "FAIL: cumetalc ${frontend}: ieee64 exp does not call vf64_exp_rne"
        exit 1
    fi
done
echo "PASS: binary32 fallback for double libdevice calls is reported; ieee64 VF64 calls are not"
