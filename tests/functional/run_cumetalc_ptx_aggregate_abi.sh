#!/usr/bin/env bash
set -euo pipefail

CUMETALC="$1"
PTX="$2"
OUTPUT="$3"

"${CUMETALC}" "${PTX}" --backend=legacy --entry aggregate_param \
    --overwrite -o "${OUTPUT}"

SIDECAR="${OUTPUT}.cumetal-abi"
if [[ ! -f "${SIDECAR}" ]]; then
    echo "FAIL: aggregate PTX ABI sidecar was not emitted"
    exit 1
fi
if [[ "$(grep -c '^arg bytes 12$' "${SIDECAR}")" -ne 1 ]] ||
   [[ "$(grep -c '^arg buffer 8$' "${SIDECAR}")" -ne 1 ]]; then
    echo "FAIL: aggregate PTX ABI sidecar lost the 12-byte by-value parameter"
    cat "${SIDECAR}"
    exit 1
fi

echo "PASS: aggregate PTX ABI sidecar preserves by-value byte size"
