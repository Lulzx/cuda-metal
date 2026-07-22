#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"
STEPS="${CUMETAL_PHYSX_CONFORMANCE_STEPS:-30}"
REL_TOL="${CUMETAL_PHYSX_REL_TOL:-1e-3}"
FRICTION_REL_TOL="${CUMETAL_PHYSX_STACKED_FRICTION_REL_TOL:-3e-3}"
BOX_REL_TOL="${CUMETAL_PHYSX_STACKED_BOX_REL_TOL:-2e-2}"
CONVEX_REL_TOL="${CUMETAL_PHYSX_STACKED_CONVEX_REL_TOL:-1e-2}"
ABS_TOL="${CUMETAL_PHYSX_ABS_TOL:-1e-5}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX GRB stacked conformance requires Apple Silicon"
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
RESULT_DIR="${BUILD_DIR}/conformance-stacked"
mkdir -p "${RESULT_DIR}"

run_pair() {
    local mode="$1"
    local rel_tol="$2"
    local geometry="${3:-sphere}"
    local cpu_dump="${RESULT_DIR}/cpu-${geometry}-${mode}.tsv"
    local gpu_dump="${RESULT_DIR}/gpu-${geometry}-${mode}.tsv"
    local cpu_log="${RESULT_DIR}/cpu-${geometry}-${mode}.log"
    local gpu_log="${RESULT_DIR}/gpu-${geometry}-${mode}.log"

    "${SNIPPET}" --cpu "--${geometry}" --stacked --bodies 2 "--${mode}" --steps "${STEPS}" \
        --dump "${cpu_dump}" >"${cpu_log}" 2>&1
    env \
        CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
        CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
        CUMETAL_SYNC_EACH_LAUNCH=1 \
        CUMETAL_TRACE_GPU=1 \
        DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
        "${SNIPPET}" --gpu "--${geometry}" --stacked --bodies 2 "--${mode}" --steps "${STEPS}" \
        --dump "${gpu_dump}" >"${gpu_log}" 2>&1

    grep -Fq 'CuMetal GRB body layout: stacked' "${cpu_log}"
    grep -Fq 'CuMetal GRB body layout: stacked' "${gpu_log}"
    grep -Fq "CuMetal GRB geometry: ${geometry}" "${cpu_log}"
    grep -Fq "CuMetal GRB geometry: ${geometry}" "${gpu_log}"
    if [[ "${geometry}" == "box" ]]; then
        grep -q 'kernel="boxBoxNphase_Kernel".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    elif [[ "${geometry}" == "convex" ]]; then
        grep -q 'kernel="convexConvexNphase_stage1Kernel".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
        grep -q 'kernel="convexConvexNphase_stage2Kernel".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
        grep -q 'kernel="finishContactsKernel".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
        grep -q 'kernel="convexPlaneNphase_Kernel".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    fi
    grep -q 'kernel="ZeroBodies".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    grep -q 'kernel="solveBlockPartition".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    grep -q 'kernel="writeBackBodies".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    grep -q 'kernel="integrateCoreParallelLaunch".*source=metallib.*device=apple_gpu.*launch_success=true' "${gpu_log}"
    if grep -qi 'internal error\|failed to create compute pipeline' "${gpu_log}"; then
        echo "FAIL: PhysX GRB stacked ${mode} GPU log contains an internal runtime error"
        exit 1
    fi

    python3 "${SCRIPT_DIR}/compare_physx_grb_multibody.py" \
        "${cpu_dump}" "${gpu_dump}" "${STEPS}" 2 "${rel_tol}" "${ABS_TOL}"
}

run_pair friction "${FRICTION_REL_TOL}"
run_pair frictionless "${REL_TOL}"
run_pair frictionless "${BOX_REL_TOL}" box
run_pair frictionless "${CONVEX_REL_TOL}" convex

if "${SNIPPET}" --cpu --stacked --bodies 1 --steps 1 >/dev/null 2>&1; then
    echo "FAIL: --stacked accepted fewer than two bodies"
    exit 1
fi

echo "PASS: PhysX stacked sphere, box, and convex dynamic/dynamic contacts match CPU"
