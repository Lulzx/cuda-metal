#!/usr/bin/env bash
# Run the FP64 level-1 test on both paths, and require the GPU one to have run.
#
# The GPU path is allowed to decline and fall back to the CPU loop, which means
# a kernel that never compiles still produces correct output and a green test.
# So the forced-on run is additionally required to report launches on the Apple
# GPU for all three kernel shapes -- elementwise axpy, elementwise scal, and the
# reduction that backs dot and nrm2.
set -euo pipefail

BIN="${1:?usage: run_cublas_blas1_metal.sh <test-binary>}"
LOG_DIR="${TMPDIR:-/tmp}/cumetal-blas1-$$"
mkdir -p "${LOG_DIR}"
trap 'rm -rf "${LOG_DIR}"' EXIT

echo "=== forced GPU path (CUMETAL_BLAS_METAL=1) ==="
CUMETAL_BLAS_METAL=1 CUMETAL_DEBUG_CUBLAS_BLAS1=1 "${BIN}" > "${LOG_DIR}/gpu.log" 2>&1 || {
    echo "FAIL: test failed with the GPU path forced"; cat "${LOG_DIR}/gpu.log"; exit 1; }
grep -E "^PASS" "${LOG_DIR}/gpu.log"

fails=0
for kernel in cumetal_daxpy_f64 cumetal_dscal_f64 cumetal_dreduce_f64; do
    count="$(grep -c "${kernel} on the Apple GPU" "${LOG_DIR}/gpu.log" || true)"
    if [[ "${count}" -eq 0 ]]; then
        echo "  FAIL: ${kernel} never ran; the CPU fallback answered instead"
        grep "CUMETAL_DEBUG_CUBLAS_BLAS1" "${LOG_DIR}/gpu.log" | head -5 | sed 's/^/    /'
        fails=$((fails + 1))
    else
        echo "  ${kernel}: ${count} launch(es) on the Apple GPU"
    fi
done

echo "=== forced CPU path (CUMETAL_BLAS_METAL=0) ==="
CUMETAL_BLAS_METAL=0 "${BIN}" > "${LOG_DIR}/cpu.log" 2>&1 || {
    echo "FAIL: test failed with the GPU path disabled"; cat "${LOG_DIR}/cpu.log"; exit 1; }
grep -E "^PASS" "${LOG_DIR}/cpu.log"

if [[ ${fails} -ne 0 ]]; then
    echo "FAIL: ${fails} kernel(s) did not run on the GPU"
    exit 1
fi
echo "PASS: both paths agree with the host reference, and the GPU one ran"
