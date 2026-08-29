#!/usr/bin/env bash
# Require the forced cuSPARSE Metal path to run and to report FP64 honestly.
set -euo pipefail

BIN="${1:?usage: run_cusparse_spmv_metal.sh <test-binary> [kernel-mode]}"
KERNEL_MODE="${2:-scalar}"
LOG_DIR="${TMPDIR:-/tmp}/cumetal-spmv-provenance-$$"
mkdir -p "${LOG_DIR}"
trap 'rm -rf "${LOG_DIR}"' EXIT

CUMETAL_SPARSE_METAL=1 \
CUMETAL_SPARSE_METAL_KERNEL="${KERNEL_MODE}" \
CUMETAL_TRACE_GPU=1 \
    "${BIN}" > "${LOG_DIR}/gpu.log" 2>&1 || {
        echo "FAIL: cuSPARSE conformance failed with ${KERNEL_MODE} forced"
        cat "${LOG_DIR}/gpu.log"
        exit 1
    }

grep -E '^PASS' "${LOG_DIR}/gpu.log"

if grep -E 'kernel="cumetal_spmv_.*_f64" .*semantic_quality=exact' \
        "${LOG_DIR}/gpu.log" >/dev/null; then
    echo "FAIL: an FP64 cuSPARSE Metal kernel claimed exact arithmetic"
    exit 1
fi
if ! grep -E 'kernel="cumetal_spmv_.*_f64" .*semantic_quality=reduced_precision_fp64 .*device=apple_gpu .*launch_success=true' \
        "${LOG_DIR}/gpu.log" >/dev/null; then
    echo "FAIL: no honest successful FP64 cuSPARSE GPU provenance was recorded"
    exit 1
fi
if ! grep -E 'kernel="cumetal_spmv_.*_f32" .*semantic_quality=exact .*device=apple_gpu .*launch_success=true' \
        "${LOG_DIR}/gpu.log" >/dev/null; then
    echo "FAIL: no exact successful FP32 cuSPARSE GPU provenance was recorded"
    exit 1
fi

echo "PASS: ${KERNEL_MODE} cuSPARSE kernels ran with type-accurate provenance"
