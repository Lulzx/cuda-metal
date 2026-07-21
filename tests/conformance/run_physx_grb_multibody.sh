#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"
STEPS="${CUMETAL_PHYSX_CONFORMANCE_STEPS:-30}"
BODIES="${CUMETAL_PHYSX_MULTIBODY_COUNT:-2}"
REL_TOL="${CUMETAL_PHYSX_REL_TOL:-1e-3}"
ABS_TOL="${CUMETAL_PHYSX_ABS_TOL:-1e-5}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX GRB multibody conformance requires Apple Silicon"
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
RESULT_DIR="${BUILD_DIR}/conformance"
CPU_DUMP="${RESULT_DIR}/physx-grb-multibody-cpu.tsv"
GPU_DUMP="${RESULT_DIR}/physx-grb-multibody-gpu.tsv"
CPU_LOG="${RESULT_DIR}/physx-grb-multibody-cpu.log"
GPU_LOG="${RESULT_DIR}/physx-grb-multibody-gpu.log"
mkdir -p "${RESULT_DIR}"

"${SNIPPET}" --cpu --bodies "${BODIES}" --steps "${STEPS}" \
    --dump "${CPU_DUMP}" >"${CPU_LOG}" 2>&1

env \
    CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" --gpu --bodies "${BODIES}" --steps "${STEPS}" \
    --dump "${GPU_DUMP}" >"${GPU_LOG}" 2>&1

grep -Fq "CuMetal GRB bodies: ${BODIES}" "${CPU_LOG}"
grep -Fq "CuMetal GRB bodies: ${BODIES}" "${GPU_LOG}"
grep -q 'kernel="constraintContactBlockPrePrepLaunch".*launch_success=true.*block=(32,1,1)' "${GPU_LOG}"
grep -q 'kernel="contactConstraintBlockPrepareParallelLaunch".*launch_success=true.*block=(32,1,1)' "${GPU_LOG}"
grep -q 'kernel="solveStaticBlock".*source=metallib.*device=apple_gpu.*launch_success=true' "${GPU_LOG}"
if grep -qi 'internal error\|failed to create compute pipeline' "${GPU_LOG}"; then
    echo "FAIL: PhysX GRB multibody GPU log contains an internal runtime error"
    exit 1
fi

if "${SNIPPET}" --cpu --bodies 0 --steps 1 >/dev/null 2>&1; then
    echo "FAIL: --bodies 0 was accepted"
    exit 1
fi
if "${SNIPPET}" --cpu --bodies 17 --steps 1 >/dev/null 2>&1; then
    echo "FAIL: --bodies 17 was accepted"
    exit 1
fi

python3 "${SCRIPT_DIR}/compare_physx_grb_multibody.py" \
    "${CPU_DUMP}" "${GPU_DUMP}" "${STEPS}" "${BODIES}" \
    "${REL_TOL}" "${ABS_TOL}"
echo "PASS: PhysX GRB multibody conformance used isolated Metal SIMD batches"
