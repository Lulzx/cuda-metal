#!/usr/bin/env bash
# Build and run the 3D SPH dam-break demo on CuMetal.
#
#   ./demos/sph/run.sh --selftest      # GPU vs host brute-force SPH reference
#   ./demos/sph/run.sh                 # full 1920x1080 60 fps dam break -> mp4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
mkdir -p "${OUT_DIR}"

BUILD_DIR="${CUMETAL_BUILD_DIR:-}"
if [[ -z "${BUILD_DIR}" ]]; then
  for cand in build-release build build-nosshim build-noshim; do
    if [[ -f "${ROOT_DIR}/${cand}/libcumetal.dylib" ]]; then
      BUILD_DIR="${ROOT_DIR}/${cand}"
      break
    fi
  done
fi
if [[ -z "${BUILD_DIR}" || ! -f "${BUILD_DIR}/libcumetal.dylib" ]]; then
  echo "FAIL: libcumetal not found. Build CuMetal first (cmake -B build && cmake --build build)."
  exit 1
fi

# shellcheck source=scripts/cumetal_cuda_flags.sh
source "${ROOT_DIR}/scripts/cumetal_cuda_flags.sh"
cumetal_cuda_device_flags

CLANG_BIN="${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
if [[ ! -x "${CLANG_BIN}" ]]; then
  CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
  echo "FAIL: clang++ not found"
  exit 1
fi

export PATH="${BUILD_DIR}/cuda_toolchain:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}"
export CUMETAL_BUILD_DIR="${BUILD_DIR}"

FLAGS=(
  -x cuda
  -std=c++17
  -O2
  -DNDEBUG
  -D__CUDACC__=1
  -D__NVCC__=1
  -Wno-pass-failed
  -Wno-unused-result
  "${CUMETAL_CUDA_DEVICE_FLAGS[@]}"
  -nocudainc
  -nocudalib
  -I"${ROOT_DIR}/runtime/api"
  -include cuda_runtime.h
)

echo "CuMetal build: ${BUILD_DIR}"
echo "Compiling SPH kernels + host..."
LOG="${OUT_DIR}/compile.log"
if ! "${CLANG_BIN}" "${FLAGS[@]}" -c "${SCRIPT_DIR}/main.cu" -o "${OUT_DIR}/main.o" >"${LOG}" 2>&1; then
  cat "${LOG}"
  echo "FAIL: compile"
  exit 1
fi

BIN="${OUT_DIR}/sph_dambreak"
xcrun clang++ "${OUT_DIR}/main.o" \
  -L"${BUILD_DIR}" -lcumetal -Wl,-rpath,"${BUILD_DIR}" \
  -o "${BIN}"

echo "Running ${BIN} $*"
exec "${BIN}" "$@"
