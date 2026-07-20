#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${CUMETAL_ROOT}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_BUILD_DIR:-${CUMETAL_ROOT}/build/physx-cpu-macos-arm64}"
ARTIFACT_DIR="${BUILD_DIR}/artifacts"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "error: this build requires macOS on Apple Silicon" >&2
    exit 1
fi

command -v cmake >/dev/null
command -v ninja >/dev/null
command -v xcrun >/dev/null

"${SCRIPT_DIR}/apply_physx_patches.sh" "${PHYSX_REPO}"

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
    -DCMAKE_BUILD_TYPE=release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DPX_OUTPUT_LIB_DIR="${ARTIFACT_DIR}" \
    -DPX_OUTPUT_BIN_DIR="${ARTIFACT_DIR}"

cmake --build "${BUILD_DIR}" --target SnippetHelloWorld --parallel

snippet="${ARTIFACT_DIR}/bin/UNKNOWN/release/SnippetHelloWorld"
if [[ ! -x "${snippet}" ]]; then
    echo "error: expected snippet was not built at ${snippet}" >&2
    exit 1
fi

if [[ "$(lipo -archs "${snippet}")" != "arm64" ]]; then
    echo "error: snippet is not a single-architecture arm64 executable" >&2
    exit 1
fi

output="$("${snippet}" 2>&1)"
printf '%s\n' "${output}"

if ! grep -q '^SnippetHelloWorld done\.$' <<<"${output}"; then
    echo "error: CPU snippet did not complete its 100 simulation steps" >&2
    exit 1
fi

echo "PASS: PhysX CPU SnippetHelloWorld completed 100 steps on macOS arm64"
