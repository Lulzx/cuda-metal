#!/usr/bin/env bash
# Native-AOT device printf must carry its format table in the embedded module
# descriptor; GPU dispatch alone is insufficient evidence that records drain.
set -euo pipefail

CUMETALC="${1:?usage: run_cumetalc_native_device_printf.sh <cumetalc> <source.cu> <workdir>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_FILE}"' EXIT

run_status=0
bash "${SCRIPT_DIR}/run_cumetalc_link_executable.sh" \
    "${CUMETALC}" "${SOURCE_CU}" "${WORK_DIR}" \
    --expect-output=HOST_DONE >"${OUTPUT_FILE}" 2>&1 || run_status=$?
cat "${OUTPUT_FILE}"
if (( run_status != 0 )); then
    exit "${run_status}"
fi

if [[ "$(grep -c '^PRINTF\[' "${OUTPUT_FILE}")" -ne 32 ]]; then
    echo "FAIL: expected exactly 32 native-AOT coordinate printf records"
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
    echo "FAIL: missing or malformed native-AOT wide printf record"
    exit 1
fi
if [[ "$(grep -Fxc 'DYNAMIC int=   -42 float=    3.12 left=7    ' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: missing native-AOT dynamic-width printf record"
    exit 1
fi
if [[ "$(grep -Fxc 'STRING value=CuMetal-device-string' "${OUTPUT_FILE}")" -ne 1 ]] ||
   [[ "$(grep -Fxc 'STRING value=[unterminated-string]' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: native-AOT tracked-string printf records are incomplete"
    exit 1
fi
if [[ "$(grep -Fxc 'MODULE global=CuMetal-global-string' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: native-AOT registered module-string printf record is incomplete"
    exit 1
fi
if [[ "$(grep -Fxc 'UNTRACKED value=[string]' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: native-AOT untracked device string was not rejected safely"
    exit 1
fi
if [[ "$(grep -Fxc 'RETURN zero' "${OUTPUT_FILE}")" -ne 1 ]] ||
   grep -q '^RETURN args=' "${OUTPUT_FILE}" ||
   [[ "$(grep -Fxc 'RETURN_VALUES zero=0 args=2' "${OUTPUT_FILE}")" -ne 1 ]]; then
    echo "FAIL: native-AOT printf return/overflow behavior is incomplete"
    exit 1
fi

echo "PASS: native-AOT device printf metadata and GPU records match the CUDA source ABI"
