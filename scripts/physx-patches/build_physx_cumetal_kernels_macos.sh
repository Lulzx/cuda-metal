#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${CUMETAL_ROOT}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_GPU_BUILD_DIR:-${CUMETAL_ROOT}/build/physx-cumetal-phase2}"
EMIT_MODE="${CUMETAL_PHYSX_EMIT_MODE:-xcrun}"
CUMETALC="${CUMETALC_EXECUTABLE:-${CUMETAL_ROOT}/build/cumetalc}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this build requires macOS on Apple Silicon" >&2
    exit 1
fi
if [[ "${EMIT_MODE}" != "xcrun" && "${EMIT_MODE}" != "experimental" ]]; then
    echo "error: CUMETAL_PHYSX_EMIT_MODE must be xcrun or experimental" >&2
    exit 1
fi

command -v cmake >/dev/null
command -v ninja >/dev/null
command -v xcrun >/dev/null

"${SCRIPT_DIR}/apply_physx_patches.sh" "${PHYSX_REPO}"
cmake --build "${CUMETAL_ROOT}/build" --target cumetalc air_validate air_inspect --parallel

cmake \
    -S "${PHYSX_REPO}/physx/compiler/public" \
    -B "${BUILD_DIR}" \
    -G Ninja \
    -DTARGET_BUILD_PLATFORM=linux \
    -DPHYSX_ROOT_DIR="${PHYSX_REPO}/physx" \
    -DPX_GENERATE_STATIC_LIBRARIES=ON \
    -DPX_BUILDSNIPPETS=OFF \
    -DPX_BUILDPVDRUNTIME=OFF \
    -DPX_GENERATE_GPU_PROJECTS=OFF \
    -DPX_CUMETAL_GPU_SUBSET=ON \
    -DCUMETALC_EXECUTABLE="${CUMETALC}" \
    -DPX_CUMETAL_EMIT_MODE="${EMIT_MODE}" \
    -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build "${BUILD_DIR}" --target PhysXCumetalGpuKernels --parallel

KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
METALLIB="${KERNEL_DIR}/sphereNphase_Kernel.metallib"
MANIFEST="${KERNEL_DIR}/kernels.json"

"${CUMETAL_ROOT}/build/air_validate" \
    "${METALLIB}" \
    --require-function-list \
    --require-metadata

INSPECT_JSON="$("${CUMETAL_ROOT}/build/air_inspect" "${METALLIB}" --json)"
grep -q '"function_count": 1' <<<"${INSPECT_JSON}"
grep -q '"name": "sphereNphase_Kernel"' <<<"${INSPECT_JSON}"
grep -q '"name": "sphereNphase_Kernel"' "${MANIFEST}"

echo "PASS: PhysX CuMetal sphere-plane kernel subset compiled (${EMIT_MODE})"
