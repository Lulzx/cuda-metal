#!/usr/bin/env bash
# build_highs_cumetal.sh -- build HiGHS twice from one source tree: its CPU
# build and its CUPDLP_GPU=ON build linked against CuMetal.
#
# The two differ only in -DCUPDLP_GPU / -DCUPDLP_FIND_CUDA. Everything else --
# compiler, optimization level, HiGHS version, MPS reader -- is held fixed, so a
# runtime difference between them is the GPU path and nothing else. That is the
# whole point: comparing against `brew install highs` would confound the result
# with whatever flags Homebrew happens to use.
#
# Usage:
#   bash scripts/build_highs_cumetal.sh [--jobs=N] [--ref=v1.15.1]
#
# Environment:
#   HIGHS_SRC     checkout path (default ~/work/cumetal-bench-ext/HiGHS)
#   CUMETAL_JOBS  build parallelism (default 6; a full-parallel build plus a
#                 running ctest has crashed this laptop before)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HIGHS_SRC="${HIGHS_SRC:-${HOME}/work/cumetal-bench-ext/HiGHS}"
JOBS="${CUMETAL_JOBS:-6}"
REF="v1.15.1"
for arg in "$@"; do
  case "$arg" in
    --jobs=*) JOBS="${arg#--jobs=}" ;;
    --ref=*)  REF="${arg#--ref=}" ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
  esac
done

FAKE_CUDA="${ROOT_DIR}/build/cumetal-cuda-toolkit"
[[ -x "${FAKE_CUDA}/bin/nvcc" ]] || {
  echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing." >&2
  echo "       Run scripts/build_llama_cpp_cumetal.sh once to generate the toolkit." >&2
  exit 2; }
[[ -f "${ROOT_DIR}/build/libcumetal.dylib" ]] || {
  echo "ERROR: build/libcumetal.dylib missing; build CuMetal first." >&2; exit 2; }

if [[ ! -d "${HIGHS_SRC}/.git" ]]; then
  mkdir -p "$(dirname "${HIGHS_SRC}")"
  git clone --quiet --depth 1 --branch "${REF}" \
      https://github.com/ERGO-Code/HiGHS.git "${HIGHS_SRC}"
fi
echo "HiGHS source: ${HIGHS_SRC} ($(git -C "${HIGHS_SRC}" describe --tags 2>/dev/null || echo unknown))"

COMMON=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DZLIB=ON)

echo "=== [1/2] CPU build (CUPDLP_GPU=OFF) ==="
cmake -S "${HIGHS_SRC}" -B "${HIGHS_SRC}/build-cpu" "${COMMON[@]}" \
      -DCUPDLP_GPU=OFF > "${HIGHS_SRC}/build-cpu.configure.log" 2>&1 \
  || { tail -30 "${HIGHS_SRC}/build-cpu.configure.log"; exit 1; }
cmake --build "${HIGHS_SRC}/build-cpu" -j"${JOBS}" > "${HIGHS_SRC}/build-cpu.log" 2>&1 \
  || { tail -40 "${HIGHS_SRC}/build-cpu.log"; exit 1; }

echo "=== [2/2] GPU build (CUPDLP_GPU=ON, CUDA_HOME=${FAKE_CUDA}) ==="
CUDA_HOME="${FAKE_CUDA}" cmake -S "${HIGHS_SRC}" -B "${HIGHS_SRC}/build-gpu" "${COMMON[@]}" \
      -DCUPDLP_GPU=ON -DCUPDLP_FIND_CUDA=ON \
      -DCMAKE_CUDA_COMPILER="${FAKE_CUDA}/bin/nvcc" \
      > "${HIGHS_SRC}/build-gpu.configure.log" 2>&1 \
  || { tail -40 "${HIGHS_SRC}/build-gpu.configure.log"; exit 1; }
CUDA_HOME="${FAKE_CUDA}" cmake --build "${HIGHS_SRC}/build-gpu" -j"${JOBS}" \
      > "${HIGHS_SRC}/build-gpu.log" 2>&1 \
  || { tail -40 "${HIGHS_SRC}/build-gpu.log"; exit 1; }

for b in cpu gpu; do
  exe="${HIGHS_SRC}/build-${b}/bin/highs"
  [[ -x "${exe}" ]] || { echo "FAIL: ${exe} not built"; exit 1; }
  echo "  ${b}: ${exe}"
done
echo "OK"
