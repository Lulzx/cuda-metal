#!/usr/bin/env bash
# Compile pinned LAMMPS with its bundled Kokkos, using CuMetal, Serial or OpenMP.
# Usage: bash scripts/build_lammps_cumetal.sh
#   [--gpu|--gpu-double|--cpu|--cpu-double|--cpu-omp|--cpu-omp-double|--compare|--probe]
# CUMETAL_LAMMPS_DIR: external checkout; CUMETAL_BUILD_DIR: CuMetal build tree.
# CUMETAL_CLANG: CUDA-capable clang++; CUMETAL_JOBS: parallel build jobs.
# CUMETAL_LAMMPS_TESTS=1: also build the LAMMPS unit tests plus the packages
# whose Kokkos styles they cover, into separate *-tests build directories.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG=stable_30Sep2026
REV=8de817dd79bfe4525d5d39246a212d833e6dee07
SRC="${CUMETAL_LAMMPS_DIR:-/tmp/cumetal-lammps-30Sep2026}"
BUILD="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"
CLANG="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
JOBS="${CUMETAL_JOBS:-6}"
MODE="${1:---gpu}"
case "${MODE}" in
    --gpu|--gpu-double|--cpu|--cpu-double|--cpu-omp|--cpu-omp-double|--compare|--probe) ;;
    -h|--help) sed -n '2,6p' "$0"; exit 0 ;;
    *) echo "Unknown mode: ${MODE}" >&2; exit 2 ;;
esac
[[ "${SRC}" = /* && "${BUILD}" = /* ]] || {
    echo "Use absolute checkout and build paths." >&2; exit 2;
}
[[ -x "${CLANG}" ]] || { echo "Missing clang++: ${CLANG}" >&2; exit 2; }
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid CUMETAL_JOBS" >&2; exit 2; }

if [[ ! -e "${SRC}" ]]; then
    git clone --depth 1 --branch "${TAG}" https://github.com/lammps/lammps.git "${SRC}"
fi
[[ "$(git -C "${SRC}" rev-parse HEAD)" = "${REV}" ]] || {
    echo "Refusing unpinned LAMMPS checkout; expected ${REV}." >&2; exit 2;
}
[[ -z "$(git -C "${SRC}" status --porcelain --untracked-files=no)" ]] || {
    echo "Refusing modified LAMMPS source." >&2; exit 2;
}

# Optional image/compressed output dependencies can put the macOS SDK's C
# headers ahead of Homebrew libc++. LIBDL is for dynamic profiling tools, not
# the neighbour/force kernels exercised here. Disable it in both builds.
COMMON=(
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=20 -DCMAKE_CXX_EXTENSIONS=OFF
    -DBUILD_MPI=OFF -DBUILD_OMP=OFF -DBUILD_SHARED_LIBS=OFF
    -DPKG_KOKKOS=ON -DKOKKOS_PREC=single
    -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_LIBDL=OFF
    -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_ZLIB=OFF -DWITH_GZIP=OFF -DWITH_FFMPEG=OFF
    -DDOWNLOAD_POTENTIALS=OFF
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
)
SUFFIX=
TARGETS=()
if [[ "${CUMETAL_LAMMPS_TESTS:-0}" = 1 ]]; then
    SUFFIX=-tests
    COMMON+=(-DENABLE_TESTING=ON -DPKG_MOLECULE=ON -DPKG_MANYBODY=ON -DPKG_KSPACE=ON)
    # Every unittest executable statically links all of LAMMPS (hundreds of MB
    # each with Kokkos/CUDA; a full tree exceeded 22 GB). The force-style yaml
    # tests run on these six.
    TARGETS=(--target lmp test_pair_style test_bond_style test_angle_style
        test_dihedral_style test_improper_style test_fix_timestep)
fi

CPU_ONLY=0
case "${MODE}" in --cpu|--cpu-double|--cpu-omp|--cpu-omp-double) CPU_ONLY=1 ;; esac
if [[ "${CPU_ONLY}" = 1 || "${MODE}" = --compare ]]; then
    CPU_BUILD="${SRC}/build-cumetal-reference"
    CPU_PREC=single
    [[ "${MODE}" = *-double ]] && CPU_PREC=double
    [[ "${CPU_PREC}" = double ]] && CPU_BUILD="${CPU_BUILD}-double"
    CPU_BACKEND=()
    if [[ "${MODE}" = --cpu-omp* ]]; then
        # Kokkos OpenMP needs LAMMPS BUILD_OMP. Homebrew LLVM ships libomp;
        # Apple Clang has the pragmas but no runtime.
        OMP_LIB="$(dirname "${CLANG}")/../lib"
        [[ -f "${OMP_LIB}/libomp.dylib" ]] || { echo "Missing ${OMP_LIB}/libomp.dylib" >&2; exit 2; }
        CPU_BUILD="${SRC}/build-cumetal-openmp${CPU_BUILD#${SRC}/build-cumetal-reference}"
        CPU_BACKEND=(-DKokkos_ENABLE_OPENMP=ON -DBUILD_OMP=ON
            -DCMAKE_EXE_LINKER_FLAGS="-Xlinker -rpath -Xlinker ${OMP_LIB}")
    fi
    CPU_BUILD="${CPU_BUILD}${SUFFIX}"
    cmake -S "${SRC}/cmake" -B "${CPU_BUILD}" "${COMMON[@]}" \
        -DCMAKE_CXX_COMPILER="${CLANG}" -DKokkos_ENABLE_CUDA=OFF -DKOKKOS_PREC="${CPU_PREC}" \
        ${CPU_BACKEND[@]+"${CPU_BACKEND[@]}"}
    cmake --build "${CPU_BUILD}" -j"${JOBS}" ${TARGETS[@]+"${TARGETS[@]}"}
    echo "CPU lmp: ${CPU_BUILD}/lmp"
fi

if [[ "${CPU_ONLY}" = 0 ]]; then
    [[ -f "${BUILD}/libcumetal.dylib" ]] || {
        echo "Build CuMetal first: cmake -B build && cmake --build build" >&2; exit 2;
    }
    # Source registration is required; the libcuda binary alias is not.
    SYMS="$(nm -gU "${BUILD}/libcumetal.dylib" 2>/dev/null)"
    [[ "${SYMS}" = *"__cudaRegisterFatBinary"* ]] || {
        echo "CuMetal must enable CUMETAL_ENABLE_CUDA_REGISTRATION." >&2; exit 2;
    }
    CUMETAL_BUILD_DIR="${BUILD}" CUMETAL_CLANG="${CLANG}" \
        bash "${ROOT_DIR}/scripts/build_llama_cpp_cumetal.sh" --toolkit-only
    TOOLKIT="${BUILD}/cumetal-cuda-toolkit"
    export PATH="${TOOLKIT}/bin:${PATH}" CUDA_ROOT="${TOOLKIT}"
    export NVCC_WRAPPER_DEFAULT_COMPILER="${CLANG}"
    GPU_BUILD="${SRC}/build-cumetal-cuda${SUFFIX}"
    GPU_PREC=single
    if [[ "${MODE}" = --gpu-double ]]; then
        # FP64 arithmetic runs in CuMetal's software modes; pick one at run
        # time with CUMETAL_FP64_MODE (ieee64 is correctly rounded binary64).
        GPU_BUILD="${SRC}/build-cumetal-cuda-double${SUFFIX}"
        GPU_PREC=double
    fi
    GPU_TESTS=()
    # The utils FFT tests link CUDA::cudart, which this configuration never
    # imports; the force-style tests that are compared do not need them.
    [[ -n "${SUFFIX}" ]] && GPU_TESTS=(-DSKIP_FFT_TESTS=ON)
    cmake -S "${SRC}/cmake" -B "${GPU_BUILD}" "${COMMON[@]}" -DKOKKOS_PREC="${GPU_PREC}" \
        ${GPU_TESTS[@]+"${GPU_TESTS[@]}"} \
        -DCMAKE_CXX_COMPILER="${SRC}/lib/kokkos/bin/nvcc_wrapper" \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
        -DCUDAToolkit_ROOT="${TOOLKIT}" \
        -DCMAKE_CXX_FLAGS="-mllvm -inline-threshold=100000" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${BUILD} -Xlinker -rpath -Xlinker ${BUILD}"
    if [[ "${MODE}" = --probe ]]; then
        python3 "${ROOT_DIR}/demos/lammps/probe.py" "${GPU_BUILD}"
    else
        cmake --build "${GPU_BUILD}" -j"${JOBS}" ${TARGETS[@]+"${TARGETS[@]}"}
        echo "GPU lmp: ${GPU_BUILD}/lmp"
    fi
fi
echo "LAMMPS revision: ${REV}"
