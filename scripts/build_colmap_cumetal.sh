#!/usr/bin/env bash
# build_colmap_cumetal.sh — Build COLMAP's CUDA path (dense PatchMatch stereo,
# SiftGPU) against CuMetal.
#
# Usage:
#   bash scripts/build_colmap_cumetal.sh
#
# Environment overrides:
#   CUMETAL_BUILD_DIR      CuMetal build to use (default: build-release-shim)
#   CUMETAL_COLMAP_DIR     checkout/build parent (default: $CLAUDE_JOB_DIR/tmp or /tmp)
#   CUMETAL_COLMAP_REF     tag to fetch (default: 4.2.1)
#   CUMETAL_CLANG          clang++ (default: Homebrew LLVM)
#   CUMETAL_JOBS           build parallelism (default: 8)
#
# On success the script prints the colmap binary it produced.
#
# Dependencies (Homebrew): boost ceres-solver eigen flann gflags glog metis
# openimageio suite-sparse faiss libomp glew sqlite.
#
# ── why the build looks like this ────────────────────────────────────────────
# COLMAP is not patched. Every flag below is either the CUDA toolkit hookup or
# a workaround for something unrelated to CuMetal:
#
#   GUI/OPENGL/CGAL/ONNX OFF    headless dense reconstruction needs none of
#                               them; Qt and CGAL are large optional installs.
#   DOWNLOAD_ENABLED=OFF        drops libcurl, whose CMake find module picks the
#   SQLite3_* (Homebrew)        macOS SDK's usr/include for curl and SQLite and
#                               puts it before libc++'s headers on Homebrew LLVM.
#   OpenMP_* (Homebrew libomp)  Homebrew's faiss links Homebrew's libomp; LLVM's
#                               own libomp would load a second OpenMP runtime
#                               into the process and abort at startup.
#   -include type_traits        PoissonRecon's MyAtomic.h uses std::is_pod_v
#                               without including <type_traits>; libc++ 23 no
#                               longer provides it transitively.
#   FETCH_FAISS=OFF             use the Homebrew faiss instead of building it.
#
# Two flags matter to CuMetal:
#   CMAKE_CUDA_ARCHITECTURES=80 the shim's device target; COLMAP otherwise asks
#                               nvidia-smi for "native".
#   --cuda-inline-threshold     inline device helpers the way nvcc does. Clang
#                               leaves LocalRefImage::Read out of line, so the
#                               PatchMatch cost computer stays in private memory
#                               with pointers in it, and the typed PTX backend
#                               cannot prove what a GetSlice() loop (trip count
#                               = an unbounded size_t) may overwrite.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOBS="${CUMETAL_JOBS:-8}"
REF="${CUMETAL_COLMAP_REF:-4.2.1}"
DEFAULT_PARENT="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "${DEFAULT_PARENT}" 2>/dev/null || DEFAULT_PARENT=/tmp
SRC_PARENT="${CUMETAL_COLMAP_DIR:-${DEFAULT_PARENT}/colmap}"
SRC_DIR="${SRC_PARENT}/colmap-${REF}"
BUILD_DIR="${SRC_PARENT}/build-cumetal"

CUMETAL_BUILD="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build-release-shim}"
[[ "${CUMETAL_BUILD}" != /* ]] && CUMETAL_BUILD="${ROOT_DIR}/${CUMETAL_BUILD}"
if [[ ! -f "${CUMETAL_BUILD}/libcumetal.dylib" ]]; then
    echo "ERROR: ${CUMETAL_BUILD}/libcumetal.dylib not found. Build CuMetal first:" >&2
    echo "  cmake -B build-release-shim -DCMAKE_BUILD_TYPE=Release -DCUMETAL_ENABLE_BINARY_SHIM=ON" >&2
    echo "  cmake --build build-release-shim" >&2
    exit 2
fi

CLANG_BIN="${CUMETAL_CLANG:-}"
if [[ -z "${CLANG_BIN}" ]]; then
    for candidate in /opt/homebrew/opt/llvm/bin/clang++ /usr/local/opt/llvm/bin/clang++; do
        [[ -x "${candidate}" ]] && { CLANG_BIN="${candidate}"; break; }
    done
fi
[[ -n "${CLANG_BIN}" ]] || { echo "ERROR: Homebrew LLVM clang++ not found (brew install llvm)" >&2; exit 2; }
CLANG_C="${CLANG_BIN%++}"

BREW="$(brew --prefix 2>/dev/null || echo /opt/homebrew)"
missing=()
for formula in boost ceres-solver eigen flann gflags glog metis openimageio suite-sparse faiss libomp glew sqlite; do
    [[ -d "${BREW}/opt/${formula}" ]] || missing+=("${formula}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing Homebrew dependencies: ${missing[*]}" >&2
    echo "  brew install ${missing[*]}" >&2
    exit 2
fi

# A Homebrew-installed CuMetal (<= 0.8.0) links its CUDA headers into
# ${BREW}/include, which every dependency above puts on the include path. The
# two copies then collide (redefinition of CU_STREAM_DEFAULT ...).
if [[ -L "${BREW}/include/cuda_runtime.h" ]] &&
   [[ "$(readlink "${BREW}/include/cuda_runtime.h")" == *Cellar/cumetal/* ]]; then
    echo "ERROR: Homebrew's cumetal has linked CUDA headers into ${BREW}/include," >&2
    echo "       which collide with this tree's headers. Run: brew unlink cumetal" >&2
    exit 2
fi

FAKE_CUDA="${CUMETAL_BUILD}/cumetal-cuda-toolkit"
echo "Generating CuMetal CUDA toolkit shim in ${FAKE_CUDA} ..."
CUMETAL_BUILD_DIR="${CUMETAL_BUILD}" bash "${ROOT_DIR}/scripts/build_llama_cpp_cumetal.sh" --toolkit-only >/dev/null
[[ -x "${FAKE_CUDA}/bin/nvcc" ]] || { echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing" >&2; exit 2; }

# ── fetch source ─────────────────────────────────────────────────────────────
if [[ ! -d "${SRC_DIR}/.git" ]]; then
    mkdir -p "${SRC_PARENT}"
    echo "Fetching COLMAP ${REF} -> ${SRC_DIR} ..."
    git clone -q --depth 1 --branch "${REF}" https://github.com/colmap/colmap.git "${SRC_DIR}"
fi
echo "COLMAP src: ${SRC_DIR} ($(git -C "${SRC_DIR}" rev-parse HEAD))"

# ── configure + build ────────────────────────────────────────────────────────
echo "=== configuring COLMAP ${REF} with CUDA_ENABLED=ON against CuMetal ==="
PATH="${FAKE_CUDA}/bin:${PATH}" cmake -S "${SRC_DIR}" -B "${BUILD_DIR}" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CLANG_C}" \
    -DCMAKE_CXX_COMPILER="${CLANG_BIN}" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0 \
    -DCMAKE_CXX_FLAGS="-include type_traits" \
    -DCUDA_ENABLED=ON \
    -DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-mmacosx-version-min=15.0 --cuda-inline-threshold 100000" \
    -DCUDAToolkit_ROOT="${FAKE_CUDA}" \
    -DCMAKE_CUDA_COMPILER_LIBRARY_ROOT="${FAKE_CUDA}" \
    -DGUI_ENABLED=OFF -DOPENGL_ENABLED=OFF -DCGAL_ENABLED=OFF -DONNX_ENABLED=OFF \
    -DFETCH_FAISS=OFF -DTESTS_ENABLED=OFF -DCCACHE_ENABLED=OFF -DIPO_ENABLED=OFF \
    -DDOWNLOAD_ENABLED=OFF \
    -DSQLite3_INCLUDE_DIR="${BREW}/opt/sqlite/include" \
    -DSQLite3_LIBRARY="${BREW}/opt/sqlite/lib/libsqlite3.dylib" \
    -DOpenMP_omp_LIBRARY="${BREW}/opt/libomp/lib/libomp.dylib" \
    -DOpenMP_C_INCLUDE_DIR="${BREW}/opt/libomp/include" \
    -DOpenMP_CXX_INCLUDE_DIR="${BREW}/opt/libomp/include" \
    >"${SRC_PARENT}/configure.log" 2>&1 || {
        echo "FAIL: configure; see ${SRC_PARENT}/configure.log" >&2
        tail -20 "${SRC_PARENT}/configure.log" >&2; exit 1; }
grep -q "Enabling CUDA support" "${SRC_PARENT}/configure.log" || {
    echo "FAIL: COLMAP configured without CUDA; see ${SRC_PARENT}/configure.log" >&2; exit 1; }

echo "=== building (several minutes) ==="
PATH="${FAKE_CUDA}/bin:${PATH}" cmake --build "${BUILD_DIR}" --target colmap -j"${JOBS}" \
    >"${SRC_PARENT}/build.log" 2>&1 || {
        echo "FAIL: build; see ${SRC_PARENT}/build.log" >&2
        grep -m5 "error:" "${SRC_PARENT}/build.log" >&2; exit 1; }
echo "colmap: ${BUILD_DIR}/src/colmap/exe/colmap"
echo "libcumetal: ${CUMETAL_BUILD}/libcumetal.dylib"
