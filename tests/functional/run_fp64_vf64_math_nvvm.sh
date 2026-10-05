#!/usr/bin/env bash
# The NVVM path: cumetalc compiles the .cu natively (clang LLVM IR through the
# NVVM importer, no PTX JIT at launch) under --fp64=ieee64, and the same 1-ulp
# gate must hold. The PTX path is functional_cuda_projects_fp64_vf64_math_ieee64.
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
OUT_DIR="${BUILD_DIR}/fp64_vf64_math_nvvm"
mkdir -p "${OUT_DIR}"

"${CUMETALC}" "${SRC}" --fp64=ieee64 --emit=msl --overwrite -o "${OUT_DIR}/direct.metal"
for name in exp exp2 expm1 log log2 log1p cbrt hypot pow atan atan2 asin acos sin cos tan \
            sinh cosh tanh asinh acosh atanh; do
    if ! grep -q "vf64_${name}_rne(" "${OUT_DIR}/direct.metal"; then
        echo "FAIL: NVVM path does not call vf64_${name}_rne under ieee64"
        exit 1
    fi
done

"${CUMETALC}" "${SRC}" --backend=cumetal-ir --fp64=ieee64 --overwrite -o "${OUT_DIR}/fp64_vf64_math"
CACHE_DIR="$(mktemp -d)"
trap 'rm -rf "${CACHE_DIR}"' EXIT
OUTPUT="$(CUMETAL_DEBUG_REGISTRATION=1 CUMETAL_CACHE_DIR="${CACHE_DIR}" "${OUT_DIR}/fp64_vf64_math" 2>&1)" || {
    echo "${OUTPUT}"
    exit 1
}
grep -E '^(PASS|FAIL)' <<<"${OUTPUT}"
if grep -q 'JIT compiling' <<<"${OUTPUT}"; then
    echo "FAIL: the native build fell back to the PTX JIT"
    exit 1
fi
grep -q '^PASS: 22 double libdevice functions within 1 ulp' <<<"${OUTPUT}"
