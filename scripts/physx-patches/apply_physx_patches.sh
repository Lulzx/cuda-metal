#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${1:-${CUMETAL_ROOT}/../PhysX}"
EXPECTED_COMMIT="5ca9f472105a90d70d957c243cb0ef36fe251a9f"

if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "error: PhysX checkout not found at ${PHYSX_REPO}" >&2
    exit 1
fi

actual_commit="$(git -C "${PHYSX_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "error: expected PhysX ${EXPECTED_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi

patch_marker_is_present() {
    case "$(basename "$1")" in
        0001-macos-arm64-cpu.patch)
            grep -q '"macaarch64"' \
                "${PHYSX_REPO}/physx/source/physxextensions/src/serialization/SnSerialUtils.cpp"
            ;;
        0002-cumetal-gpu-subset.patch)
            grep -q 'PX_CUMETAL_GPU_SUBSET' \
                "${PHYSX_REPO}/physx/compiler/public/CMakeLists.txt" &&
                test -f "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt"
            ;;
        0003-cumetal-apple-gpu-platform.patch)
            grep -q '#define PX_CUMETAL 0' \
                "${PHYSX_REPO}/physx/include/foundation/PxPreprocessor.h" &&
                grep -q 'CUMETAL_NO_DEVICE_PRINTF=1' \
                    "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt"
            ;;
        0004-cumetal-grb-kernel-subset.patch)
            grep -q 'SnippetHelloGRB-sphere-plane-pgs' \
                "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt" &&
                grep -q -- '--cuda-inline-threshold 1000000' \
                    "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt"
            ;;
        0005-cumetal-runtime-grb.patch)
            grep -q 'CUMETAL_PHYSX_KERNEL_DIR' \
                "${PHYSX_REPO}/physx/source/cudamanager/src/CudaKernelWrangler.cpp"
            ;;
        0006-physx-grb-conformance.patch)
            grep -q -- '--dump FILE' \
                "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp" &&
                grep -q 'CuMetal GRB mode:' \
                    "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp"
            ;;
        0007-cumetal-grb-contact.patch)
            grep -q 'CUDA UVA gives mapped host and device pointers' \
                "${PHYSX_REPO}/physx/source/gpusolver/src/PxgSolverCore.cpp"
            ;;
        0008-cumetal-native-warp-paths.patch)
            grep -q 'CUMETAL_PHYSX_KERNEL_DIR' \
                "${PHYSX_REPO}/physx/source/cudamanager/src/CudaKernelWrangler.cpp" &&
                ! grep -q 'partial warp masks as a full SIMD group' \
                    "${PHYSX_REPO}/physx/source/gpusolver/src/CUDA/preIntegration.cuh" &&
                ! grep -q 'CuMetal intentionally emulates partial warp masks' \
                    "${PHYSX_REPO}/physx/source/gpusimulationcontroller/src/CUDA/updateBodiesAndShapes.cu"
            ;;
        0009-cumetal-grb-kinetic-friction.patch)
            { grep -q 'Build its single friction anchor directly' \
                "${PHYSX_REPO}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh" ||
                grep -q "prior frame's impulse is not integrated repeatedly" \
                    "${PHYSX_REPO}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp" ||
                grep -q 'prior-frame impulses are not integrated repeatedly' \
                    "${PHYSX_REPO}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp"; } &&
                grep -q -- '--frictionless' \
                    "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp"
            ;;
        0010-cumetal-grb-rolling-friction.patch)
            { grep -q "prior frame's impulse is not integrated repeatedly" \
                "${PHYSX_REPO}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp" ||
                grep -q 'prior-frame impulses are not integrated repeatedly' \
                    "${PHYSX_REPO}/physx/source/gpusolver/src/PxgCudaSolverCore.cpp"; } &&
                grep -q "host stages the selected target's prior patch" \
                    "${PHYSX_REPO}/physx/source/gpusolver/src/CUDA/contactConstraintBlockPrep.cuh"
            ;;
        0011-cumetal-grb-multibody-static-batching.patch)
            grep -q 'Keep each prepared batch in its own SIMD group' \
                "${PHYSX_REPO}/physx/source/gpusolver/src/PxgSolverCore.cpp" &&
                grep -q -- '--bodies must be between 1 and 16' \
                    "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp"
            ;;
        0012-cumetal-grb-dynamic-contact-batching.patch)
            grep -q 'Avoid staging device pointers and island metadata' \
                "${PHYSX_REPO}/physx/source/gpusolver/src/CUDA/solverMultiBlock.cu" &&
                grep -q -- '--stacked requires at least two bodies' \
                    "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp"
            ;;
        0013-cumetal-grb-box-plane.patch)
            grep -q 'convexPlaneNphase_Kernel' \
                "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt" &&
                grep -q -- '--sphere|--box' \
                    "${PHYSX_REPO}/physx/snippets/snippethellogrb/SnippetHelloGRB.cpp"
            ;;
        0014-cumetal-grb-box-box.patch)
            grep -q 'cudaBox.cu|boxBoxNphase_Kernel' \
                "${PHYSX_REPO}/physx/source/compiler/cmakegpu/cumetal/CMakeLists.txt"
            ;;
        *)
            return 1
            ;;
    esac
}

for patch in "${SCRIPT_DIR}"/*.patch; do
    if patch_marker_is_present "${patch}"; then
        echo "already applied: $(basename "${patch}")"
    elif git -C "${PHYSX_REPO}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "already applied: $(basename "${patch}")"
    elif git -C "${PHYSX_REPO}" apply --check "${patch}"; then
        git -C "${PHYSX_REPO}" apply "${patch}"
        echo "applied: $(basename "${patch}")"
    else
        echo "error: cannot apply $(basename "${patch}") cleanly" >&2
        exit 1
    fi
done
