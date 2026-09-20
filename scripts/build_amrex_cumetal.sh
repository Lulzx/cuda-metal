#!/usr/bin/env bash
# build_amrex_cumetal.sh — Build AMReX's CUDA GPU backend against CuMetal, plus a
# CPU-only build of the same source to compare it against.
#
# Usage:
#   bash scripts/build_amrex_cumetal.sh            # GPU build only
#   bash scripts/build_amrex_cumetal.sh --compare  # also build the CPU reference
#
# Environment overrides:
#   CUMETAL_AMREX_DIR        checkout parent (default: $CLAUDE_JOB_DIR/tmp or /tmp)
#   CUMETAL_AMREX_VERSION    AMReX release tag to fetch (default: 25.09)
#   CUMETAL_AMREX_TUTORIALS_REV  amrex-tutorials commit (default: pinned below)
#   CUMETAL_CLANG            clang++ to use (default: Homebrew LLVM)
#   CUMETAL_BUILD_DIR        CuMetal build tree to link against (default: ./build)
#   CUMETAL_JOBS             build parallelism (default: 6)
#
# On success the script prints the two HeatEquation binaries it produced and the
# directory holding AMReX's plotfile comparison tools.
#
# ── why the build looks like this ────────────────────────────────────────────
# Nothing from AMReX is vendored in this repository. The script fetches the
# release and the tutorials repository at pinned revisions, then builds AMReX
# twice from one source tree: once with AMReX_GPU_BACKEND=CUDA against CuMetal's
# CUDA toolkit shim, once with AMReX_GPU_BACKEND=NONE as the reference. AMReX
# itself is unmodified -- there are no patches in this path.
#
# Four AMReX options are set away from their defaults, all for reasons that are
# about CuMetal's coverage rather than about correctness of the comparison:
#
#   AMReX_CUDA_FASTMATH=OFF  AMReX turns nvcc's --use_fast_math on by default.
#                            CuMetal compiles generated MSL under CUDA's
#                            floating-point contract, and the demo's whole claim
#                            is a numerical comparison, so the CPU and GPU builds
#                            have to agree on the contract.
#   AMReX_CUDA_MAXREGCOUNT=0 -maxrregcount has no meaning here: ptxas is a shim
#                            and Metal does its own register allocation.
#   AMReX_GPU_RDC=OFF        Relocatable device code fuses per-TU device images
#                            at link time. CuMetal has no such image; each
#                            translation unit registers its own kernels with the
#                            runtime when it loads.
#   AMReX_MPI=OFF            Single process, single GPU. AMReX's CPU and GPU
#                            builds then integrate the same decomposition.
#
# Both builds use Homebrew LLVM rather than Apple clang, because the CUDA build
# needs a clang new enough to accept --cuda-gpu-arch=sm_80, and using one host
# compiler for both keeps the comparison honest.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
JOBS="${CUMETAL_JOBS:-6}"
COMPARE=0
[[ "${1:-}" == "--compare" ]] && COMPARE=1
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && { sed -n '2,18p' "$0"; exit 0; }

VERSION="${CUMETAL_AMREX_VERSION:-25.09}"
# amrex-tutorials has no releases; pin the commit the HeatEquation source is
# taken from so a tutorial edit upstream cannot silently change the gate.
TUTORIALS_REV="${CUMETAL_AMREX_TUTORIALS_REV:-1a73f32e10514d917333ea7a3fd54e661e232144}"

DEFAULT_PARENT="${CLAUDE_JOB_DIR:-/tmp}/tmp"
mkdir -p "${DEFAULT_PARENT}" 2>/dev/null || DEFAULT_PARENT=/tmp
SRC_PARENT="${CUMETAL_AMREX_DIR:-${DEFAULT_PARENT}/amrex}"
AMREX_DIR="${SRC_PARENT}/amrex-${VERSION}"
TUTORIALS_DIR="${SRC_PARENT}/amrex-tutorials"
HEAT_SOURCE="${TUTORIALS_DIR}/ExampleCodes/Basic/HeatEquation_EX0_C/Source"

CLANG_BIN="${CUMETAL_CLANG:-}"
if [[ -z "${CLANG_BIN}" ]]; then
    for candidate in /opt/homebrew/opt/llvm/bin/clang++ /usr/local/opt/llvm/bin/clang++; do
        [[ -x "${candidate}" ]] && { CLANG_BIN="${candidate}"; break; }
    done
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "ERROR: Homebrew LLVM clang++ not found. Install with: brew install llvm" >&2
    exit 2
fi
CLANG_C="${CLANG_BIN%++}"

CUMETAL_ACTIVE_BUILD_DIR="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"
[[ "${CUMETAL_ACTIVE_BUILD_DIR}" != /* ]] && \
    CUMETAL_ACTIVE_BUILD_DIR="${ROOT_DIR}/${CUMETAL_ACTIVE_BUILD_DIR}"
if [[ ! -f "${CUMETAL_ACTIVE_BUILD_DIR}/libcumetal.dylib" ]]; then
    echo "ERROR: libcumetal.dylib not found in ${CUMETAL_ACTIVE_BUILD_DIR}." >&2
    echo "       Build CuMetal first, or set CUMETAL_BUILD_DIR." >&2
    exit 2
fi

FAKE_CUDA="${CUMETAL_ACTIVE_BUILD_DIR}/cumetal-cuda-toolkit"
if [[ ! -x "${FAKE_CUDA}/bin/nvcc" ]]; then
    echo "Generating CuMetal CUDA toolkit shim ..."
    CUMETAL_BUILD_DIR="${CUMETAL_ACTIVE_BUILD_DIR}" \
        bash "${ROOT_DIR}/scripts/build_llama_cpp_cumetal.sh" --toolkit-only >/dev/null
fi
[[ -x "${FAKE_CUDA}/bin/nvcc" ]] || { echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing." >&2; exit 2; }
export PATH="${FAKE_CUDA}/bin:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}"

# ── fetch sources ────────────────────────────────────────────────────────────
mkdir -p "${SRC_PARENT}"
if [[ ! -d "${AMREX_DIR}/.git" ]]; then
    echo "Cloning AMReX ${VERSION} -> ${AMREX_DIR} ..."
    git clone --depth 1 --branch "${VERSION}" \
        https://github.com/AMReX-Codes/amrex.git "${AMREX_DIR}"
fi
if [[ ! -d "${TUTORIALS_DIR}/.git" ]]; then
    echo "Cloning amrex-tutorials -> ${TUTORIALS_DIR} ..."
    git clone https://github.com/AMReX-Codes/amrex-tutorials.git "${TUTORIALS_DIR}"
fi
git -C "${TUTORIALS_DIR}" checkout --quiet "${TUTORIALS_REV}" 2>/dev/null || {
    git -C "${TUTORIALS_DIR}" fetch --quiet origin
    git -C "${TUTORIALS_DIR}" checkout --quiet "${TUTORIALS_REV}"
}
[[ -f "${HEAT_SOURCE}/main.cpp" ]] || { echo "ERROR: ${HEAT_SOURCE}/main.cpp missing" >&2; exit 2; }

COMMON_AMREX_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_C_COMPILER="${CLANG_C}"
    -DCMAKE_CXX_COMPILER="${CLANG_BIN}"
    -DAMReX_MPI=OFF
    -DAMReX_OMP=OFF
    -DAMReX_FORTRAN=OFF
    -DAMReX_SPACEDIM=3
    -DAMReX_INSTALL=ON
)

# ── AMReX, CUDA backend, against CuMetal ─────────────────────────────────────
GPU_BUILD="${AMREX_DIR}/build-cumetal"
GPU_INSTALL="${SRC_PARENT}/install-cumetal"
echo "=== configuring AMReX ${VERSION} (AMReX_GPU_BACKEND=CUDA -> CuMetal) ==="
cmake -S "${AMREX_DIR}" -B "${GPU_BUILD}" "${COMMON_AMREX_FLAGS[@]}" \
    -DAMReX_GPU_BACKEND=CUDA \
    -DAMReX_GPU_RDC=OFF \
    -DAMReX_CUDA_FASTMATH=OFF \
    -DAMReX_CUDA_MAXREGCOUNT=0 \
    -DAMReX_CUDA_SHOW_LINENUMBERS=OFF \
    -DAMReX_CUDA_WARN_CAPTURE_THIS=OFF \
    -DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" \
    -DCUDAToolkit_ROOT="${FAKE_CUDA}" \
    -DCMAKE_CUDA_ARCHITECTURES=80 \
    -DCMAKE_INSTALL_PREFIX="${GPU_INSTALL}"
echo "=== building AMReX (CUDA backend) ==="
cmake --build "${GPU_BUILD}" -j"${JOBS}"
cmake --install "${GPU_BUILD}" >/dev/null

# ── the tutorial, built against each AMReX install ───────────────────────────
# CuMetal supplies the CMakeLists rather than using the tutorials repository's
# own build glue, which expects the whole ExampleCodes tree.
HEAT_CMAKE="${ROOT_DIR}/demos/amrex/heat"
build_heat () {
    local install_prefix="$1" build_dir="$2" backend="$3"
    local args=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_CXX_COMPILER="${CLANG_BIN}"
        -DAMReX_ROOT="${install_prefix}"
        -DHEAT_SOURCE_DIR="${HEAT_SOURCE}"
    )
    if [[ "${backend}" == cuda ]]; then
        args+=(-DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES=80)
    fi
    cmake -S "${HEAT_CMAKE}" -B "${build_dir}" "${args[@]}" >/dev/null
    cmake --build "${build_dir}" -j"${JOBS}" >/dev/null
}

HEAT_GPU_BUILD="${SRC_PARENT}/heat-build-cumetal"
echo "=== building HeatEquation_EX0_C (CuMetal) ==="
build_heat "${GPU_INSTALL}" "${HEAT_GPU_BUILD}" cuda
echo "GPU  heat: ${HEAT_GPU_BUILD}/heat"

if [[ ${COMPARE} -eq 1 ]]; then
    CPU_BUILD="${AMREX_DIR}/build-cpu"
    CPU_INSTALL="${SRC_PARENT}/install-cpu"
    echo "=== configuring AMReX ${VERSION} (AMReX_GPU_BACKEND=NONE, reference) ==="
    # The CPU tree also supplies fcompare/fextrema; they are backend-independent
    # readers of the plotfile format, so one copy serves both runs.
    cmake -S "${AMREX_DIR}" -B "${CPU_BUILD}" "${COMMON_AMREX_FLAGS[@]}" \
        -DAMReX_GPU_BACKEND=NONE \
        -DAMReX_PLOTFILE_TOOLS=ON \
        -DCMAKE_INSTALL_PREFIX="${CPU_INSTALL}"
    echo "=== building AMReX (CPU reference) ==="
    cmake --build "${CPU_BUILD}" -j"${JOBS}"
    cmake --install "${CPU_BUILD}" >/dev/null

    HEAT_CPU_BUILD="${SRC_PARENT}/heat-build-cpu"
    echo "=== building HeatEquation_EX0_C (CPU reference) ==="
    build_heat "${CPU_INSTALL}" "${HEAT_CPU_BUILD}" none
    echo "CPU  heat: ${HEAT_CPU_BUILD}/heat"
    echo "Tools:     ${CPU_INSTALL}/bin"
fi

echo "AMReX src: ${AMREX_DIR}"
