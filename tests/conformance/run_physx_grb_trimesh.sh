#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"
STEPS="${CUMETAL_PHYSX_CONFORMANCE_STEPS:-30}"
REL_TOL="${CUMETAL_PHYSX_REL_TOL:-1e-3}"
ABS_TOL="${CUMETAL_PHYSX_ABS_TOL:-1e-5}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX GRB triangle-mesh conformance requires Apple Silicon"
    exit 77
fi
if ! xcrun metal --version >/dev/null 2>&1; then
    metal_toolchain="$(xcodebuild -showComponent MetalToolchain -json 2>/dev/null | \
        python3 -c 'import json, sys; data = json.load(sys.stdin); print(data.get("toolchainIdentifier", "") if data.get("status") == "installed" else "")' \
        2>/dev/null || true)"
    if [[ -z "${metal_toolchain}" ]]; then
        echo "SKIP: the optional Xcode Metal Toolchain is unavailable"
        exit 77
    fi
    export TOOLCHAINS="${metal_toolchain}"
fi
if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "SKIP: PhysX checkout not found at ${PHYSX_REPO}"
    exit 77
fi

"${ROOT_DIR}/scripts/physx-patches/build_physx_cumetal_grb_macos.sh" >/dev/null

SNIPPET="${BUILD_DIR}/artifacts/bin/UNKNOWN/release/SnippetHelloGRB"
KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
RESULT_DIR="${BUILD_DIR}/conformance"
CPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-cpu.tsv"
GPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-gpu.tsv"
CPU_LOG="${RESULT_DIR}/physx-grb-trimesh-cpu.log"
GPU_LOG="${RESULT_DIR}/physx-grb-trimesh-gpu.log"
mkdir -p "${RESULT_DIR}"

"${SNIPPET}" --cpu --sphere --trimesh --frictionless --steps "${STEPS}" \
    --dump "${CPU_DUMP}" >"${CPU_LOG}" 2>&1

env \
    CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" --gpu --sphere --trimesh --frictionless --steps "${STEPS}" \
    --dump "${GPU_DUMP}" >"${GPU_LOG}" 2>&1

grep -q 'CuMetal GRB geometry: sphere' "${GPU_LOG}"
grep -q 'CuMetal GRB ground: trimesh' "${CPU_LOG}"
grep -q 'CuMetal GRB ground: trimesh' "${GPU_LOG}"
for kernel in \
    midphaseGeneratePairs \
    sphereTrimeshNarrowphase \
    sortTriangleIndices \
    convexTrimeshPostProcess \
    convexTrimeshCorrelate \
    convexTrimeshFinishContacts; do
    grep -q "kernel=\"${kernel}\".*source=metallib.*device=apple_gpu.*launch_success=true" "${GPU_LOG}"
done
if grep -qi 'internal error\|failed to create compute pipeline' "${GPU_LOG}"; then
    echo "FAIL: PhysX GRB triangle-mesh GPU log contains an internal runtime error"
    exit 1
fi

if "${SNIPPET}" --gpu --box --trimesh --frictionless --steps 1 \
    >"${RESULT_DIR}/physx-grb-trimesh-unsupported.log" 2>&1; then
    echo "FAIL: unsupported GPU box/triangle-mesh mode was accepted"
    exit 1
fi
grep -q 'GPU triangle-mesh mode currently supports one separated frictionless sphere only' \
    "${RESULT_DIR}/physx-grb-trimesh-unsupported.log"

python3 "${SCRIPT_DIR}/compare_physx_grb.py" \
    "${CPU_DUMP}" "${GPU_DUMP}" "${STEPS}" "${REL_TOL}" "${ABS_TOL}"
echo "PASS: PhysX GRB sphere/static-triangle-mesh conformance matched CPU"
