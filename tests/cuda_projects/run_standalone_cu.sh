#!/usr/bin/env bash
# Build and run one cuda_projects standalone .cu harness.
# Usage: run_standalone_cu.sh <cumetal-root> <ctest-binary-dir> <project-subdir> <source.cu> <binary-name>
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

cumetal_cuda_projects_compile_link "${ROOT_DIR}" "${SRC_DIR}" "${OUT_DIR}" "${SRC_CU}" "${OUT_BIN}"

echo "Running ${OUT_BIN}..."
"${OUT_DIR}/${OUT_BIN}"
echo "PASS: cuda_projects/${PROJECT_SUBDIR}/${OUT_BIN}"
