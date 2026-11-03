#!/usr/bin/env bash
set -euo pipefail

NVCC="${1:?usage: run_cumetalc_nvcc_fp64.sh <nvcc> <source.cu> <workdir>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"

if ! command -v xcrun >/dev/null 2>&1 ||
   ! xcrun --find metal >/dev/null 2>&1 ||
   ! xcrun --find metallib >/dev/null 2>&1; then
    echo "SKIP: Apple Metal toolchain not installed"
    exit 77
fi

mkdir -p "${WORK_DIR}"
rm -f "${WORK_DIR}/single" "${WORK_DIR}/object" "${WORK_DIR}/kernel.o"

run_and_check() {
    local name="$1"
    local output
    rm -rf "${WORK_DIR}/${name}-cache"
    output="$(CUMETAL_TRACE_GPU=1 CUMETAL_DEBUG_REGISTRATION=1 \
        CUMETAL_CACHE_DIR="${WORK_DIR}/${name}-cache" "${WORK_DIR}/${name}" 2>&1)" || {
        echo "${output}"
        exit 1
    }
    echo "${output}"
    grep -Fq 'PASS: nvcc preserved IEEE64 exponent range' <<<"${output}"
    local launches
    launches=$(awk '
        /CUMETAL_PROVENANCE event=kernel_launch/ && /device=apple_gpu/ && /launch_success=true/ {
            for (i = 1; i <= NF; ++i) {
                if ($i ~ /^duration_ns=/) {
                    split($i, duration, "=")
                    if (duration[2] > 0) ++count
                }
            }
        }
        END { print count + 0 }
    ' <<<"${output}")
    if [[ "${launches}" != 1 ]]; then
        echo "FAIL: expected one completed Apple GPU launch with positive duration"
        exit 1
    fi
    if grep -Eq 'JIT compiling|registration-jit' <<<"${output}"; then
        echo "FAIL: native executable performed PTX JIT"
        exit 1
    fi
}

"${NVCC}" "${SOURCE_CU}" --fp64=ieee64 -o "${WORK_DIR}/single"
run_and_check single
"${NVCC}" -c "${SOURCE_CU}" --fp64=ieee64 -o "${WORK_DIR}/kernel.o"
"${NVCC}" "${WORK_DIR}/kernel.o" -o "${WORK_DIR}/object"
run_and_check object

for option in --fp64=invalid --fp64= --fp64; do
    if "${NVCC}" "${SOURCE_CU}" "${option}" -o "${WORK_DIR}/invalid" \
        >"${WORK_DIR}/invalid.log" 2>&1; then
        echo "FAIL: invalid FP64 option accepted: ${option}"
        exit 1
    fi
    grep -Fq -- '--fp64' "${WORK_DIR}/invalid.log"
done

echo "PASS: nvcc FP64 policy in executable and object compilation"
