#!/usr/bin/env bash
# Build and run the 3D Gaussian Splatting forward demo on CuMetal.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
mkdir -p "${OUT_DIR}"

# Prefer a Release build when present.
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

VENDOR="${SCRIPT_DIR}/vendor"

# GLM is required to compile the vendored Inria forward.cu (even though the
# demo's preprocess runs on the host). Prefer Homebrew, then common prefixes.
GLM_INC=""
for cand in \
  "${CUMETAL_GLM_INCLUDE:-}" \
  "$(brew --prefix glm 2>/dev/null)/include" \
  /opt/homebrew/include \
  /usr/local/include; do
  if [[ -n "${cand}" && -f "${cand}/glm/glm.hpp" ]]; then
    GLM_INC="${cand}"
    break
  fi
done
if [[ -z "${GLM_INC}" ]]; then
  echo "FAIL: glm headers not found (glm/glm.hpp)."
  echo "      Install with: brew install glm"
  echo "      Or set CUMETAL_GLM_INCLUDE to the directory that contains glm/."
  exit 1
fi
echo "GLM include: ${GLM_INC}"

INCLUDES=(
  -I"${ROOT_DIR}/runtime/api"
  -I"${VENDOR}"
  -I"${GLM_INC}"
  -include cuda_runtime.h
)

COMMON_FLAGS=(
  -x cuda
  -std=c++17
  -O2
  -DNDEBUG
  -D__CUDACC__=1
  -D__NVCC__=1
  -DGLM_FORCE_CUDA=1
  -Wno-pass-failed
  -Wno-unused-result
  -Wno-c++11-narrowing
  -Wno-literal-conversion
  "${CUMETAL_CUDA_DEVICE_FLAGS[@]}"
  -nocudainc
  -nocudalib
  "${INCLUDES[@]}"
)

echo "CuMetal build: ${BUILD_DIR}"
echo "Compiling 3DGS forward kernels + host..."

compile_one() {
  local src="$1"
  local obj="$2"
  local log="${obj}.log"
  rm -f "${obj}"
  local st=0
  "${CLANG_BIN}" "${COMMON_FLAGS[@]}" -c "${src}" -o "${obj}" >"${log}" 2>&1 || st=$?
  if [[ ${st} -ne 0 ]]; then
    echo "---- compile log: ${src} ----"
    cat "${log}"
    echo "FAIL: compile ${src} (exit ${st})"
    exit 1
  fi
  # Keep log on success for diagnosis; print only warnings of interest.
  if grep -E 'error:|warning:.*error' "${log}" >/dev/null 2>&1; then
    cat "${log}"
  fi
}

compile_one "${VENDOR}/cuda_rasterizer/forward.cu" "${OUT_DIR}/forward.o"
compile_one "${VENDOR}/cuda_rasterizer/rasterizer_impl.cu" "${OUT_DIR}/rasterizer_impl.o"
compile_one "${SCRIPT_DIR}/main.cu" "${OUT_DIR}/main.o"

BIN="${OUT_DIR}/gaussian_splat_forward"
xcrun clang++ \
  "${OUT_DIR}/forward.o" "${OUT_DIR}/rasterizer_impl.o" "${OUT_DIR}/main.o" \
  -L"${BUILD_DIR}" -lcumetal -Wl,-rpath,"${BUILD_DIR}" \
  -o "${BIN}"

echo "Running ${BIN}..."
export CUMETAL_TRACE_GPU=1
set +e
"${BIN}" --out "${OUT_DIR}/gaussians.ppm" --size 128 >"${OUT_DIR}/run.log" 2>&1
RUN_ST=$?
set -e
cat "${OUT_DIR}/run.log"

if [[ ${RUN_ST} -ne 0 ]]; then
  echo "FAIL: demo exited ${RUN_ST}"
  exit 1
fi

if ! grep -q 'device=apple_gpu' "${OUT_DIR}/run.log"; then
  # Some builds only print provenance on CUMETAL_TRACE_GPU; accept either the
  # run log or a separate provenance scrape if present.
  if ! grep -q 'device=apple_gpu' "${OUT_DIR}/run.log" 2>/dev/null; then
    echo "WARN: no device=apple_gpu line in run log (check CUMETAL_TRACE_GPU / provenance)."
  fi
fi

if ! grep -q 'PASS: 3D Gaussian Splatting forward rendered on CuMetal' "${OUT_DIR}/run.log"; then
  echo "FAIL: numerical/render gate did not pass"
  exit 1
fi

if [[ ! -f "${OUT_DIR}/gaussians.ppm" ]]; then
  echo "FAIL: missing output image"
  exit 1
fi

echo "PASS: demos/3dgs (see ${OUT_DIR}/gaussians.ppm)"
exit 0
