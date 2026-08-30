#!/usr/bin/env bash
# build_gromacs_cumetal.sh — Build GROMACS's CUDA GPU path against CuMetal,
# plus a CPU-only build of the same source to compare it against.
#
# Usage:
#   bash scripts/build_gromacs_cumetal.sh            # GPU build only
#   bash scripts/build_gromacs_cumetal.sh --compare  # also build the CPU reference
#
# Environment overrides:
#   CUMETAL_GROMACS_DIR      checkout path (default: $CLAUDE_JOB_DIR/tmp or /tmp)
#   CUMETAL_GROMACS_VERSION  release to fetch (default: 2025.4)
#   CUMETAL_GROMACS_URL      tarball URL (default: ftp.gromacs.org)
#   CUMETAL_CLANG            clang++ to use (default: Homebrew LLVM)
#   CUMETAL_JOBS             build parallelism (default: 6)
#
# On success the script prints the two gmx binaries it produced.
#
# ── why the build looks like this ────────────────────────────────────────────
# Both builds use Homebrew LLVM, not Apple clang. Apple ships clang with OpenMP
# disabled and GROMACS refuses to configure without it; using one compiler for
# the host side of both builds also keeps the CPU/GPU comparison honest.
#
# Three flags work around things that have nothing to do with CuMetal:
#
#   CMAKE_OSX_DEPLOYMENT_TARGET  CMake otherwise leaves it empty, and the
#                                bundled Colvars fails on std::filesystem. CMake
#                                does not forward it to nvcc, so CMAKE_CUDA_FLAGS
#                                repeats it; otherwise the .cu objects target a
#                                newer macOS than the host ones and every link
#                                warns.
#   GMX_USE_COLVARS=NONE         Colvars' colvarproxy_io.cpp reads
#                                __cpp_lib_filesystem without having included
#                                <filesystem>, which libc++ rejects. Colvars is
#                                a free-energy sampling module; no benchmark
#                                here uses it.
#   -include cstddef             gromacs/compat/pointers.h names std::ptrdiff_t
#                                without including <cstddef>; libc++ 23 no
#                                longer leaks it in transitively.
#
# GMX_FFT_LIBRARY=fftpack uses GROMACS's bundled FFT rather than requiring an
# FFTW install. It is slower than FFTW for the CPU PME mesh; the benchmarks
# here are small enough that it does not matter, and it keeps the build
# dependency-free.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOBS="${CUMETAL_JOBS:-6}"
COMPARE=0
[[ "${1:-}" == "--compare" ]] && COMPARE=1

VERSION="${CUMETAL_GROMACS_VERSION:-2025.4}"
DEFAULT_PARENT="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "${DEFAULT_PARENT}" 2>/dev/null || DEFAULT_PARENT=/tmp
SRC_PARENT="${CUMETAL_GROMACS_DIR:-${DEFAULT_PARENT}/gromacs}"
SRC_DIR="${SRC_PARENT}/gromacs-${VERSION}"
URL="${CUMETAL_GROMACS_URL:-https://ftp.gromacs.org/gromacs/gromacs-${VERSION}.tar.gz}"

CLANG_BIN="${CUMETAL_CLANG:-}"
if [[ -z "${CLANG_BIN}" ]]; then
    for candidate in /opt/homebrew/opt/llvm/bin/clang++ /usr/local/opt/llvm/bin/clang++; do
        [[ -x "${candidate}" ]] && { CLANG_BIN="${candidate}"; break; }
    done
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "ERROR: Homebrew LLVM clang++ not found. Apple clang has no OpenMP and" >&2
    echo "       GROMACS will not configure without it. Install with: brew install llvm" >&2
    exit 2
fi
CLANG_C="${CLANG_BIN%++}"

FAKE_CUDA="${ROOT_DIR}/build/cumetal-cuda-toolkit"
if [[ ! -x "${FAKE_CUDA}/bin/nvcc" ]]; then
    echo "Generating CuMetal CUDA toolkit shim ..."
    bash "${ROOT_DIR}/scripts/build_llama_cpp_cumetal.sh" --toolkit-only >/dev/null
fi
if [[ ! -x "${FAKE_CUDA}/bin/nvcc" ]]; then
    echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing." >&2
    exit 2
fi

# ── fetch source ─────────────────────────────────────────────────────────────
if [[ ! -d "${SRC_DIR}" ]]; then
    mkdir -p "${SRC_PARENT}"
    echo "Fetching GROMACS ${VERSION} -> ${SRC_DIR} ..."
    curl -sfL --max-time 900 "${URL}" -o "${SRC_PARENT}/gromacs-${VERSION}.tar.gz"
    tar xzf "${SRC_PARENT}/gromacs-${VERSION}.tar.gz" -C "${SRC_PARENT}"
fi
[[ -f "${SRC_DIR}/CMakeLists.txt" ]] || { echo "ERROR: no GROMACS source at ${SRC_DIR}" >&2; exit 2; }

COMMON_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_OSX_DEPLOYMENT_TARGET=15.0
    -DCMAKE_C_COMPILER="${CLANG_C}"
    -DCMAKE_CXX_COMPILER="${CLANG_BIN}"
    -DCMAKE_CXX_FLAGS="-include cstddef"
    -DGMX_MPI=OFF
    -DGMX_OPENMP=ON
    -DGMX_USE_COLVARS=NONE
    -DGMX_FFT_LIBRARY=fftpack
    -DGMX_HWLOC=OFF
    -DBUILD_TESTING=OFF
    -DGMXAPI=OFF
)

# ── GPU build (CUDA -> CuMetal) ──────────────────────────────────────────────
GPU_BUILD="${SRC_PARENT}/build-cumetal"
echo "=== configuring GROMACS ${VERSION} with GMX_GPU=CUDA against CuMetal ==="
PATH="${FAKE_CUDA}/bin:${PATH}" cmake -S "${SRC_DIR}" -B "${GPU_BUILD}" \
    "${COMMON_ARGS[@]}" \
    -DGMX_GPU=CUDA \
    -DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_CUDA_FLAGS="-mmacosx-version-min=15.0" \
    -DCMAKE_CUDA_COMPILER_LIBRARY_ROOT="${FAKE_CUDA}" \
    -DCUDAToolkit_ROOT="${FAKE_CUDA}" \
    -DGMX_CUDA_TARGET_SM=80 \
    -DGMX_NVSHMEM=OFF \
    >"${SRC_PARENT}/configure-gpu.log" 2>&1 || {
        echo "FAIL: GPU configure; see ${SRC_PARENT}/configure-gpu.log" >&2
        tail -20 "${SRC_PARENT}/configure-gpu.log" >&2; exit 1; }

echo "=== building (this takes a few minutes) ==="
cmake --build "${GPU_BUILD}" -j"${JOBS}" >"${SRC_PARENT}/build-gpu.log" 2>&1 || {
    echo "FAIL: GPU build; see ${SRC_PARENT}/build-gpu.log" >&2
    grep -m5 "error:" "${SRC_PARENT}/build-gpu.log" >&2; exit 1; }
echo "GPU  gmx: ${GPU_BUILD}/bin/gmx"

# ── CPU reference build ──────────────────────────────────────────────────────
if [[ ${COMPARE} -eq 1 ]]; then
    CPU_BUILD="${SRC_PARENT}/build-cpu"
    echo "=== configuring the CPU reference from the same source ==="
    cmake -S "${SRC_DIR}" -B "${CPU_BUILD}" "${COMMON_ARGS[@]}" -DGMX_GPU=OFF \
        >"${SRC_PARENT}/configure-cpu.log" 2>&1 || {
            echo "FAIL: CPU configure; see ${SRC_PARENT}/configure-cpu.log" >&2
            tail -20 "${SRC_PARENT}/configure-cpu.log" >&2; exit 1; }
    cmake --build "${CPU_BUILD}" -j"${JOBS}" >"${SRC_PARENT}/build-cpu.log" 2>&1 || {
        echo "FAIL: CPU build; see ${SRC_PARENT}/build-cpu.log" >&2
        grep -m5 "error:" "${SRC_PARENT}/build-cpu.log" >&2; exit 1; }
    echo "CPU  gmx: ${CPU_BUILD}/bin/gmx"
fi
