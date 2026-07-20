#!/usr/bin/env bash
# Build the upstream NVIDIA cuda-samples vectorAdd source without modifications
# and require correct output from CuMetal's generic PTX Apple-GPU path.
set -euo pipefail

ROOT_DIR="${1:?}"
CUDA_SAMPLES_DIR="${CUMETAL_CUDA_SAMPLES_DIR:-/Users/lulzx/work/cuda-samples}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${SCRIPT_DIR}/_common.sh"

if [[ ! -f "${CUDA_SAMPLES_DIR}/Samples/0_Introduction/vectorAdd/vectorAdd.cu" ]]; then
    echo "SKIP: cuda-samples checkout not found at ${CUDA_SAMPLES_DIR}"
    exit 77
fi

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

bash "${ROOT_DIR}/tests/cuda_projects/cuda_samples_vectoradd/build.sh" \
    "${CUDA_SAMPLES_DIR}"

OUTPUT_FILE="$(mktemp)"
CACHE_DIR="$(mktemp -d)"
trap 'rm -f "$OUTPUT_FILE"; rm -rf "$CACHE_DIR"' EXIT

set +e
CUMETAL_CACHE_DIR="$CACHE_DIR" CUMETAL_TRACE_GPU=1 \
    "${ROOT_DIR}/tests/cuda_projects/cuda_samples_vectoradd/build/vectorAdd" \
    >"$OUTPUT_FILE" 2>&1
STATUS=$?
set -e
cat "$OUTPUT_FILE"

if [[ $STATUS -ne 0 ]]; then
    exit "$STATUS"
fi
if ! grep -q 'Test PASSED' "$OUTPUT_FILE"; then
    echo "FAIL: upstream cuda-samples vectorAdd did not verify its result"
    exit 1
fi
if ! grep -q 'CUMETAL_PROVENANCE .*source=generic_ptx device=apple_gpu .*launch_success=true' \
    "$OUTPUT_FILE"; then
    echo "FAIL: upstream CUDA sample has no successful generic-PTX Apple GPU dispatch"
    exit 1
fi
if grep -Eq 'source=(cpu_fallback|stub)' "$OUTPUT_FILE"; then
    echo "FAIL: upstream CUDA sample used a CPU fallback or stub"
    exit 1
fi
