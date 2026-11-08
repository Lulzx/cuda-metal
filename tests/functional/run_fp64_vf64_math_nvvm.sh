#!/usr/bin/env bash
# The NVVM path: cumetalc compiles the .cu natively (clang LLVM IR through the
# NVVM importer, no PTX JIT at launch). The default ieee64 surface keeps its
# 1-ulp gate; wide48 selects the separate reduced-hypot fixture.
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
# shellcheck source=tests/cuda_projects/_common.sh
source "${ROOT_DIR}/tests/cuda_projects/_common.sh"
if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi
CUMETALC="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}/cumetalc"
SRC="${ROOT_DIR}/tests/cuda_projects/fp64_vf64_math/fp64_vf64_math.cu"
MODE="${3:-ieee64}"
EXTRA=(--fp64="${MODE}")
if [[ "${MODE}" == wide48 ]]; then
    EXTRA+=(-DCUMETAL_TEST_WIDE_HYPOT=1)
    NAMES=(wide_hypot)
    EXPECT="PASS: wide48 hypot analytic and MPFR vectors on Apple GPU"
else
    NAMES=(exp exp2 expm1 log log2 log1p cbrt hypot pow atan atan2 asin acos sin cos tan
           sinh cosh tanh asinh acosh atanh)
    EXPECT="PASS: 22 double libdevice functions within 1 ulp"
fi
OUT_DIR="${BUILD_DIR}/fp64_vf64_math_nvvm_${MODE}"
mkdir -p "${OUT_DIR}"

"${CUMETALC}" "${SRC}" "${EXTRA[@]}" --emit=msl --overwrite -o "${OUT_DIR}/direct.metal"
for name in "${NAMES[@]}"; do
    symbol="vf64_${name}_rne"
    if [[ "${MODE}" == wide48 ]]; then symbol="vf64_${name}"; fi
    if ! grep -q "${symbol}(" "${OUT_DIR}/direct.metal"; then
        echo "FAIL: NVVM path does not call ${symbol} under ${MODE}"
        exit 1
    fi
done

"${CUMETALC}" "${SRC}" "${EXTRA[@]}" --backend=cumetal-ir --overwrite -o "${OUT_DIR}/fp64_vf64_math"
CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "${CACHE_DIR}"' EXIT
OUTPUT="$(CUMETAL_TRACE_GPU=1 CUMETAL_DEBUG_REGISTRATION=1 CUMETAL_CACHE_DIR="${CACHE_DIR}" "${OUT_DIR}/fp64_vf64_math" 2>&1)" || {
    echo "${OUTPUT}"
    exit 1
}
grep -E '^(PASS|FAIL)' <<<"${OUTPUT}"
if grep -q 'JIT compiling' <<<"${OUTPUT}"; then
    echo "FAIL: the native build fell back to the PTX JIT"
    exit 1
fi
grep -q "^${EXPECT}" <<<"${OUTPUT}"
grep -q 'device=apple_gpu.*launch_success=true' <<<"${OUTPUT}"
if [[ "${MODE}" == wide48 ]] && grep -q 'FP64 libdevice calls evaluate through binary32' <<<"${OUTPUT}"; then
    echo "FAIL: wide48 hypot incorrectly reports binary32 evaluation"
    exit 1
fi
