#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${CUMETAL_ROOT}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${CUMETAL_ROOT}/build/physx-cumetal-runtime}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this build requires macOS on Apple Silicon" >&2
    exit 1
fi

command -v cmake >/dev/null
command -v ninja >/dev/null
command -v xcrun >/dev/null
xcrun -f metal >/dev/null

"${SCRIPT_DIR}/apply_physx_patches.sh" "${PHYSX_REPO}"
cmake --build "${CUMETAL_ROOT}/build" --target cumetal_runtime cumetalc --parallel

cmake \
    -S "${PHYSX_REPO}/physx/compiler/public" \
    -B "${BUILD_DIR}" \
    -G Ninja \
    -DTARGET_BUILD_PLATFORM=linux \
    -DPHYSX_ROOT_DIR="${PHYSX_REPO}/physx" \
    -DPX_GENERATE_STATIC_LIBRARIES=ON \
    -DPX_BUILDSNIPPETS=ON \
    -DPX_BUILDPVDRUNTIME=OFF \
    -DPX_GENERATE_GPU_PROJECTS=OFF \
    -DPX_CUMETAL_GPU_SUBSET=ON \
    -DPX_CUMETAL_GPU_RUNTIME=ON \
    -DCUMETAL_ROOT_DIR="${CUMETAL_ROOT}" \
    -DCUMETALC_EXECUTABLE="${CUMETAL_ROOT}/build/cumetalc" \
    -DPX_CUMETAL_EMIT_MODE=xcrun \
    -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_OSX_ARCHITECTURES=arm64

cmake --build "${BUILD_DIR}" --target PhysXCumetalGpuKernels SnippetHelloGRB \
    --parallel "${CUMETAL_PHYSX_BUILD_JOBS:-4}"

KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
SNIPPET="${BUILD_DIR}/artifacts/bin/UNKNOWN/release/SnippetHelloGRB"
RUN_LOG="${BUILD_DIR}/SnippetHelloGRB.cumetal.log"

env \
    CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${CUMETAL_ROOT}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" >"${RUN_LOG}" 2>&1

grep -q 'source=metallib device=apple_gpu.*launch_success=true' "${RUN_LOG}"
grep -q 'CuMetal GRB final velocity: 0 -' "${RUN_LOG}"
grep -q 'SnippetHelloGRB done.' "${RUN_LOG}"
tail -3 "${RUN_LOG}"
echo "PASS: PhysX SnippetHelloGRB executed GPU dynamics through CuMetal"
