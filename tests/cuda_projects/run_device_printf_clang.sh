#!/usr/bin/env bash
# Compile Clang's real CUDA device-printf ABI and verify every GPU record.
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
PTX_BACKEND="${3:-legacy}"
if [[ "${PTX_BACKEND}" != legacy && "${PTX_BACKEND}" != cumetal-ir ]]; then
    echo "FAIL: invalid PTX backend '${PTX_BACKEND}'" >&2
    exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

SOURCE_DIR="${ROOT_DIR}/tests/cuda_projects/device_printf_clang"
OUTPUT_DIR="${BUILD_DIR}/device_printf_clang-${PTX_BACKEND}"
mkdir -p "${OUTPUT_DIR}"
cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" "${SOURCE_DIR}" "${OUTPUT_DIR}" \
    device_printf_clang.cu device_printf_clang

OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

run_status=0
CUMETAL_CACHE_DIR="${CACHE_DIR}" CUMETAL_TRACE_GPU=1 \
    CUMETAL_PTX_BACKEND="${PTX_BACKEND}" \
    "${OUTPUT_DIR}/device_printf_clang" >"${OUTPUT_FILE}" 2>&1 || run_status=$?
cat "${OUTPUT_FILE}"
if (( run_status != 0 )); then
    exit "${run_status}"
fi
if ! grep -q '^HOST_DONE$' "${OUTPUT_FILE}"; then
    echo "FAIL: host completion marker missing"
    exit 1
fi
if ! grep -q 'CUMETAL_PROVENANCE .*source=generic_ptx .*device=apple_gpu .*launch_success=true' \
        "${OUTPUT_FILE}"; then
    echo "FAIL: no successful Apple-GPU provenance record"
    exit 1
fi
if [[ "$(grep -c '^PRINTF\[' "${OUTPUT_FILE}")" -ne 32 ]]; then
    echo "FAIL: expected exactly 32 device printf records"
    exit 1
fi
for block in 0 1 2 3; do
    for thread in 0 1 2 3 4 5 6 7; do
        if [[ "$(grep -Fxc "PRINTF[${block},${thread}]=37" "${OUTPUT_FILE}")" -ne 1 ]]; then
            echo "FAIL: missing or duplicate PRINTF[${block},${thread}]=37"
            exit 1
        fi
    done
done

if [[ "$(grep -Ec '^WIDE signed=-1234567890123 unsigned=1234605616436508552 hex=0x1122334455667788 size=8589934599 ptr=0x[0-9a-f]+ float=3\.125 char=Q percent=%$' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: missing or malformed wide device printf record"
    exit 1
fi

if [[ "$(grep -Fxc 'DYNAMIC int=   -42 float=    3.12 left=7    ' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: missing or malformed dynamic-width/precision device printf record"
    exit 1
fi

if [[ "$(grep -Fxc 'STRING value=CuMetal-device-string' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: missing or malformed tracked device-string printf record"
    exit 1
fi
if [[ "$(grep -Fxc 'MODULE global=CuMetal-global-string' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: missing or malformed registered module-string printf record"
    exit 1
fi
if [[ "$(grep -Fxc 'UNTRACKED value=[string]' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: untracked device string was not rejected safely"
    exit 1
fi
if [[ "$(grep -Fxc 'STRING value=[unterminated-string]' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: unterminated tracked string was not bounded explicitly"
    exit 1
fi

echo "PASS: Clang device printf ABI emitted scalar, dynamic-field, and tracked-string GPU records (${PTX_BACKEND})"
