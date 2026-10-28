#!/usr/bin/env bash
# Typed-backend device heap gate: builds the device heap probe through the
# fatbin shim toolchain and runs it with CUMETAL_PTX_BACKEND=cumetal-ir.
# The probe covers device new/delete, C++ virtual dispatch through a
# vtable, local references, free-list reuse, and heap exhaustion. The
# legacy AOT equivalent is functional_device_heap.
set -euo pipefail

ROOT_DIR="${1:?usage: $0 <repo-root> <workdir>}"
WORK_DIR="${2:?usage: $0 <repo-root> <workdir>}"
mkdir -p "${WORK_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${ROOT_DIR}/tests/cuda_projects/_common.sh"

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" \
    "${SCRIPT_DIR}/fixtures" \
    "${WORK_DIR}" \
    device_heap_probe.cu \
    device_heap_probe

output="$(CUMETAL_PTX_BACKEND=cumetal-ir "${WORK_DIR}/device_heap_probe" 2>&1)"
printf '%s\n' "${output}"
grep -Fq 'PASS: device new/delete, virtual dispatch, references, reuse, and exhaustion' <<<"${output}"

echo "PASS: typed device heap and virtual dispatch behavior"
