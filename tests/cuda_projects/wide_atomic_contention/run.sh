#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:?}"
BUILD_DIR="${2:?}"
OUTPUT_DIR="${BUILD_DIR}/wide_atomic_contention"
mkdir -p "${OUTPUT_DIR}"
cache="$(mktemp -d "${OUTPUT_DIR}/cache.XXXXXX")"
log="$(mktemp "${OUTPUT_DIR}/run.XXXXXX")"
trap 'rm -rf "$cache"; rm -f "$log"' EXIT

export CUMETAL_CACHE_DIR="$cache"
export CUMETAL_PTX_BACKEND=cumetal-ir
export CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0
export CUMETAL_ENABLE_LLMC_CPU_EMULATION=0
export CUMETAL_DISABLE_LLMC_EMULATION=1
status=0
bash "${ROOT_DIR}/tests/cuda_projects/run_strict_standalone_cu.sh" \
    "$ROOT_DIR" "$BUILD_DIR" wide_atomic_contention wide_atomic_contention.cu \
    wide_atomic_contention "PASS: contended 64-bit integer atomics preserve every update" \
    >"$log" 2>&1 || status=$?
cat "$log"
if [[ $status -ne 0 ]]; then
    exit "$status"
fi
if [[ $(grep -c '^CUMETAL_PROVENANCE event=kernel_launch ' "$log") -ne 2 ]]; then
    echo "FAIL: expected exactly two Apple GPU kernel launches" >&2
    exit 1
fi
for kernel in wide_add wide_cas; do
    if ! grep -Eq "^CUMETAL_PROVENANCE event=kernel_launch kernel=\"${kernel}\" source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact device=apple_gpu .*launch_success=true duration_ns=[1-9][0-9]* " "$log"; then
        echo "FAIL: ${kernel} has no successful generic typed-PTX Apple GPU execution" >&2
        exit 1
    fi
done
