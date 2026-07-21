#!/usr/bin/env bash
# Compile CUDA source with launch syntax, link to CuMetal, and prove that the
# resulting fatbinary is lowered and committed to an Apple Metal GPU.
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

SOURCE_DIR="${ROOT_DIR}/tests/cuda_projects/vector_add"
OUTPUT_DIR="${BUILD_DIR}/vector_add"
mkdir -p "${OUTPUT_DIR}"

cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" \
    "${SOURCE_DIR}" \
    "${OUTPUT_DIR}" \
    vector_add.cu \
    vector_add

OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

set +e
CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 \
    "${OUTPUT_DIR}/vector_add" >"$OUTPUT_FILE" 2>&1
STATUS=$?
set -e
cat "$OUTPUT_FILE"

if [[ $STATUS -ne 0 ]]; then
    exit "$STATUS"
fi
if ! grep -q 'PASS: CUDA source vector_add produced correct GPU output' "$OUTPUT_FILE"; then
    echo "FAIL: CUDA source program did not verify its result"
    exit 1
fi
if ! grep -q 'CUMETAL_PROVENANCE .*kernel="vector_add" .*source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu .*launch_success=true' "$OUTPUT_FILE"; then
    echo "FAIL: CUDA source result has no evidence of a Metal GPU dispatch"
    exit 1
fi
