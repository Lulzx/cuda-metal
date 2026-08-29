#!/usr/bin/env bash
# Build and run one cuda_projects standalone .cu harness.
# Usage: run_standalone_cu.sh <cumetal-root> <ctest-binary-dir> <project-subdir> <source.cu> <binary-name> [legacy|cumetal-ir]
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
PROJECT_SUBDIR="${3:?}"
SRC_CU="${4:?}"
OUT_BIN="${5:?}"
PTX_BACKEND="${6:-legacy}"

if [[ "${PTX_BACKEND}" != legacy && "${PTX_BACKEND}" != cumetal-ir ]]; then
    echo "invalid PTX backend: ${PTX_BACKEND}" >&2
    exit 2
fi

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
# The exit status must be captured, not discarded. `$(cmd || true)` swallowed it,
# so a harness that crashed before printing anything (SIGBUS, SIGSEGV, abort) fell
# through every content check below and was reported PASS -- the loudest possible
# failure read as green. Only classify the run after the status is known.
RUN_STATUS=0
if [[ "${PTX_BACKEND}" == cumetal-ir ]]; then
    TYPED_CACHE="$(mktemp -d "${TMPDIR:-/tmp}/cumetal-typed-runtime.XXXXXX")"
    trap 'rm -rf "${TYPED_CACHE}"' EXIT HUP INT TERM
    RUN_OUTPUT="$(CUMETAL_CACHE_DIR="${TYPED_CACHE}" \
        CUMETAL_PTX_BACKEND=cumetal-ir "${OUT_DIR}/${OUT_BIN}" 2>&1)" || RUN_STATUS=$?
else
    RUN_OUTPUT="$("${OUT_DIR}/${OUT_BIN}" 2>&1)" || RUN_STATUS=$?
fi
echo "$RUN_OUTPUT"

if echo "$RUN_OUTPUT" | grep -q "CUMETAL: registered kernel missing metallib"; then
    echo "UNSUPPORTED: registered kernel lowering is unavailable; harness compile succeeded."
    if [[ "${CUMETAL_CUDA_PROJECT_STRICT_CLASSIFICATION:-0}" == "1" ]]; then
        exit 1
    fi
    echo "SKIP: lowering not supported for this kernel (generic PTX emitter or direct path incomplete for tiled/shared/complex kernels; see docs/known-gaps.md)."
    exit 77
fi
if (( RUN_STATUS >= 128 )); then
    echo "FAIL: cuda_projects/${PROJECT_SUBDIR}/${OUT_BIN} died on signal $(( RUN_STATUS - 128 ))."
    exit 1
fi
if (( RUN_STATUS != 0 )); then
    echo "FAIL: cuda_projects/${PROJECT_SUBDIR}/${OUT_BIN} exited ${RUN_STATUS}."
    exit 1
fi
if echo "$RUN_OUTPUT" | grep -q "FAIL:"; then
    # The kernel launched and ran to completion but produced wrong results.
    # That is a numerical failure, not a coverage gap: unsupported lowering
    # reports "registered kernel missing metallib" and is skipped above. Never
    # downgrade a wrong answer to a skip -- doing so reads as coverage the
    # project does not have.
    echo "FAIL: cuda_projects/${PROJECT_SUBDIR}/${OUT_BIN} produced incorrect results."
    exit 1
fi

echo "PASS: cuda_projects/${PROJECT_SUBDIR}/${OUT_BIN}"
