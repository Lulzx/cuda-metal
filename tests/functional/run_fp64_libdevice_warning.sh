#!/usr/bin/env bash
# Double libdevice calls (exp, log, pow, ...) run in binary32 under every FP64
# mode. That used to be recorded only as a comment in the generated MSL; the
# runtime must now say so once per process -- on a warm JIT cache hit too,
# where no MSL is generated -- and stay quiet for pure FP64 arithmetic.
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

for backend in cumetal-ir legacy; do
    expect "${backend}" ieee64 exp 1 "${backend} cold compile"
    expect "${backend}" ieee64 exp 1 "${backend} warm cache hit"
    expect "${backend}" ieee64 arith 0 "${backend} pure arithmetic"
done
expect cumetal-ir wide48 exp 1 "cumetal-ir wide48"

# The offline compiler says the same thing, and only when it applies.
CUMETALC="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cumetalc"
OFFLINE="cumetalc: warning: double-precision math functions"
for frontend in direct ptx; do
    flags=(--fp64=ieee64 --emit=msl --overwrite)
    [ "${frontend}" = ptx ] && flags+=(--cuda-device --backend=cumetal-ir)
    with_exp="$("${CUMETALC}" "${SRC_DIR}/fp64_libdevice_warning.cu" "${flags[@]}" \
        -o "${OUT_DIR}/offline-${frontend}.metal" 2>&1)"
    without="$("${CUMETALC}" "${ROOT_DIR}/tests/cuda_projects/fp64_reciprocal/fp64_reciprocal.cu" \
        "${flags[@]}" -o "${OUT_DIR}/offline-${frontend}-arith.metal" 2>&1)"
    if ! grep -q "${OFFLINE}" <<<"${with_exp}" || grep -q "${OFFLINE}" <<<"${without}"; then
        echo "${with_exp}"
        echo "${without}"
        echo "FAIL: cumetalc ${frontend}: warning missing for exp or present for pure arithmetic"
        exit 1
    fi
done
echo "PASS: binary32 fallback for double libdevice calls is reported at run time"
