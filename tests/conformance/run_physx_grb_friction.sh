#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"
STEPS="${CUMETAL_PHYSX_FRICTION_STEPS:-60}"
EARLY_STEPS="${CUMETAL_PHYSX_FRICTION_EARLY_STEPS:-18}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX GRB friction requires Apple Silicon"
    exit 77
fi
if ! xcrun -f metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal is unavailable"
    exit 77
fi
if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "SKIP: PhysX checkout not found at ${PHYSX_REPO}"
    exit 77
fi

"${ROOT_DIR}/scripts/physx-patches/build_physx_cumetal_grb_macos.sh" >/dev/null

SNIPPET="${BUILD_DIR}/artifacts/bin/UNKNOWN/release/SnippetHelloGRB"
KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
RESULT_DIR="${BUILD_DIR}/conformance-friction"
CPU_DUMP="${RESULT_DIR}/cpu-friction.tsv"
GPU_DUMP="${RESULT_DIR}/gpu-friction.tsv"
OFF_DUMP="${RESULT_DIR}/gpu-frictionless.tsv"
CPU_LOG="${RESULT_DIR}/cpu-friction.log"
GPU_LOG="${RESULT_DIR}/gpu-friction.log"
OFF_LOG="${RESULT_DIR}/gpu-frictionless.log"
mkdir -p "${RESULT_DIR}"

"${SNIPPET}" --cpu --friction --steps "${STEPS}" --dump "${CPU_DUMP}" >"${CPU_LOG}" 2>&1

env CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" --gpu --friction --steps "${STEPS}" --dump "${GPU_DUMP}" >"${GPU_LOG}" 2>&1

env CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" --gpu --frictionless --steps "${STEPS}" --dump "${OFF_DUMP}" >"${OFF_LOG}" 2>&1

for log in "${GPU_LOG}" "${OFF_LOG}"; do
    grep -q 'kernel="contactConstraintBlockPrepareParallelLaunch".*launch_success=true' "${log}"
    grep -q 'kernel="solveStaticBlock".*launch_success=true' "${log}"
    grep -q 'kernel="integrateCoreParallelLaunch".*launch_success=true' "${log}"
    if grep -qi 'internal error\|failed to create compute pipeline\|GPU Hang' "${log}"; then
        echo "FAIL: PhysX GRB friction GPU log contains an internal runtime error"
        exit 1
    fi
done
grep -q 'CuMetal GRB contact mode: friction' "${CPU_LOG}"
grep -q 'CuMetal GRB contact mode: friction' "${GPU_LOG}"
grep -q 'CuMetal GRB contact mode: frictionless' "${OFF_LOG}"

python3 "${SCRIPT_DIR}/compare_physx_grb_friction.py" \
    "${CPU_DUMP}" "${GPU_DUMP}" "${OFF_DUMP}" "${STEPS}" "${EARLY_STEPS}"
