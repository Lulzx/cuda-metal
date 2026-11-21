#!/usr/bin/env bash
# --fmad decides FMA contraction end to end. The default fuses a * b + c like
# nvcc; --fmad=false must leave two roundings. clang fused the product into
# the device IR and PTX before CuMetal saw it, so --fmad=false used to reach
# only Metal's -ffp-contract and the result stayed fused.
set -euo pipefail

ROOT_DIR="${1:?usage: run_fmad_flag.sh <root> <build-dir>}"
BUILD_DIR="${2:?}"
CUMETALC="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cumetalc"
SRC="${ROOT_DIR}/tests/functional/fixtures/fmad_flag.cu"
OUT_DIR="${BUILD_DIR}/fmad_flag"

source "${ROOT_DIR}/tests/cuda_projects/_common.sh"
if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi
mkdir -p "${OUT_DIR}"
CACHE_ROOT="$(mktemp -d)"
trap 'rm -rf "${CACHE_ROOT}"' EXIT

run_case() {
    local name="$1" expect="$2"
    shift 2
    local bin="${OUT_DIR}/${name}"
    "${CUMETALC}" "${SRC}" -DCUMETAL_EXPECT_FUSED="${expect}" "$@" --overwrite -o "${bin}" \
        >"${bin}.log" 2>&1 || {
        cat "${bin}.log"
        exit 1
    }
    CUMETAL_CACHE_DIR="$(mktemp -d "${CACHE_ROOT}/c.XXXXXX")" "${bin}"
}

run_case default 1
run_case fmad_true 1 --fmad=true
run_case fmad_false 0 --fmad=false

# The PTX deliverable carries the same decision.
"${CUMETALC}" "${SRC}" -DCUMETAL_EXPECT_FUSED=0 --fmad=false --cuda-device --emit ptx \
    --overwrite -o "${OUT_DIR}/fmad_false.ptx" >/dev/null
if grep -q 'fma\.rn\.f32' "${OUT_DIR}/fmad_false.ptx"; then
    echo "FAIL: --fmad=false PTX still contains fma.rn.f32"
    exit 1
fi
"${CUMETALC}" "${SRC}" -DCUMETAL_EXPECT_FUSED=1 --cuda-device --emit ptx \
    --overwrite -o "${OUT_DIR}/default.ptx" >/dev/null
grep -q 'fma\.rn\.f32' "${OUT_DIR}/default.ptx" || {
    echo "FAIL: default PTX does not contract a * b + c"
    exit 1
}
echo "PASS: --fmad controls FMA contraction"
