#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
PROJECT_SUBDIR="${3:?}"
SRC_CU="${4:?}"
OUT_BIN="${5:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

SRC_DIR="${ROOT_DIR}/tests/cuda_projects/${PROJECT_SUBDIR}"
OUT_DIR="${BUILD_DIR}/${PROJECT_SUBDIR}"
mkdir -p "${OUT_DIR}"
cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" "${SRC_DIR}" "${OUT_DIR}" "${SRC_CU}" "${OUT_BIN}"

RUN_OUTPUT="$(CUMETAL_TRACE_GPU=1 "${OUT_DIR}/${OUT_BIN}" 2>&1)" || {
    echo "${RUN_OUTPUT}"
    exit 1
}
echo "${RUN_OUTPUT}"
grep -q "PASS: GGML output-head kernels match CPU references on Apple GPU" \
    <<<"${RUN_OUTPUT}"
grep -q 'device=apple_gpu' <<<"${RUN_OUTPUT}"
if grep -q 'source=approximate_stub' <<<"${RUN_OUTPUT}"; then
    echo "FAIL: strict numerical probe used an approximate stub"
    exit 1
fi
