#!/usr/bin/env bash
# Compile pinned LAMMPS with its bundled Kokkos, using CuMetal or Serial.
# Usage: bash scripts/build_lammps_cumetal.sh [--gpu|--cpu|--cpu-double|--compare|--probe]
# CUMETAL_LAMMPS_DIR: external checkout; CUMETAL_BUILD_DIR: CuMetal build tree.
# CUMETAL_CLANG: CUDA-capable clang++; CUMETAL_JOBS: parallel build jobs.
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
    --gpu|--cpu|--cpu-double|--compare|--probe) ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
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

if [[ "${MODE}" = --cpu || "${MODE}" = --cpu-double || "${MODE}" = --compare ]]; then
    CPU_BUILD="${SRC}/build-cumetal-reference"
    CPU_PREC=single
    if [[ "${MODE}" = --cpu-double ]]; then
        CPU_BUILD="${SRC}/build-cumetal-reference-double"
        CPU_PREC=double
    fi
    cmake -S "${SRC}/cmake" -B "${CPU_BUILD}" "${COMMON[@]}" \
        -DCMAKE_CXX_COMPILER="${CLANG}" -DKokkos_ENABLE_CUDA=OFF -DKOKKOS_PREC="${CPU_PREC}"
    cmake --build "${CPU_BUILD}" -j"${JOBS}"
    echo "CPU lmp: ${CPU_BUILD}/lmp"
fi

if [[ "${MODE}" != --cpu && "${MODE}" != --cpu-double ]]; then
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
    GPU_BUILD="${SRC}/build-cumetal-cuda"
    cmake -S "${SRC}/cmake" -B "${GPU_BUILD}" "${COMMON[@]}" \
        -DCMAKE_CXX_COMPILER="${SRC}/lib/kokkos/bin/nvcc_wrapper" \
        -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=ON \
        -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=OFF \
        -DCUDAToolkit_ROOT="${TOOLKIT}" \
        -DCMAKE_CXX_FLAGS="-mllvm -inline-threshold=100000" \
        -DCMAKE_EXE_LINKER_FLAGS="-L${BUILD} -Xlinker -rpath -Xlinker ${BUILD}"
    if [[ "${MODE}" = --probe ]]; then
        python3 "${ROOT_DIR}/demos/lammps/probe.py" "${GPU_BUILD}"
    else
        cmake --build "${GPU_BUILD}" -j"${JOBS}"
        echo "GPU lmp: ${GPU_BUILD}/lmp"
    fi
fi
echo "LAMMPS revision: ${REV}"
