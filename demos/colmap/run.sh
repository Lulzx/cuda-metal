#!/usr/bin/env bash
# CuMetal COLMAP demo: dense multi-view stereo (PatchMatch) on the Apple GPU.
#
# Usage:
#   bash demos/colmap/run.sh                     # build, render, reconstruct, gate
#   bash demos/colmap/run.sh --build-dir=path    # CuMetal build (default build-release-shim)
#   bash demos/colmap/run.sh --skip-build        # reuse the last COLMAP build
#
# COLMAP's dense stage -- patch_match_stereo -- has no CPU implementation: on a
# Mac it stops with "Dense stereo reconstruction requires CUDA". This script
# builds unmodified COLMAP 4.2.1 with CUDA_ENABLED=ON against CuMetal
# (scripts/build_colmap_cumetal.sh), then runs the standard dense pipeline
#
#   image_undistorter -> patch_match_stereo -> stereo_fusion
#
# on a synthetic scene rendered by make_scene.py, whose depth is known exactly.
#
# There is no CPU build of PatchMatch to compare against, so the gate
# (gate.py) compares against the truth instead: per-view depth accuracy and
# coverage, and the fused cloud's distance to the analytic surfaces. It also
# refuses to call anything a pass unless CuMetal's trace shows every kernel
# launch ran on the Apple GPU and none failed.
#
# Exit 0 only if the build, every COLMAP stage, the provenance check and the
# accuracy gate pass. The reconstruction takes about 19 minutes on an M4 Pro.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR="${ROOT_DIR}/build-release-shim"
SKIP_BUILD=0
for arg in "$@"; do
  case "$arg" in
    --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
    --skip-build) SKIP_BUILD=1 ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done
[[ "${BUILD_DIR}" != /* ]] && BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
if [[ ! -f "${BUILD_DIR}/libcumetal.dylib" ]]; then
  echo "FAIL: ${BUILD_DIR}/libcumetal.dylib not found. Build CuMetal first:"
  echo "  cmake -B build-release-shim -DCMAKE_BUILD_TYPE=Release -DCUMETAL_ENABLE_BINARY_SHIM=ON"
  echo "  cmake --build build-release-shim"
  exit 1
fi
# COLMAP's .cu files are compiled by Clang in CUDA mode, so their kernels reach
# the runtime through __cudaRegisterFatBinary: the binary shim must be built in.
shim_syms="$(nm -gU "${BUILD_DIR}/libcumetal.dylib" 2>/dev/null | grep -c "__cudaRegisterFatBinary" || true)"
if [[ "${shim_syms:-0}" -eq 0 ]]; then
  echo "SKIP: ${BUILD_DIR}/libcumetal.dylib has no __cudaRegisterFatBinary."
  echo "      Reconfigure with -DCUMETAL_ENABLE_BINARY_SHIM=ON."
  exit 77
fi

mkdir -p "${OUT_DIR}"
COLMAP_LOG="${OUT_DIR}/build.log"
if [[ ${SKIP_BUILD} -eq 0 || ! -f "${COLMAP_LOG}" ]]; then
  echo "=== building COLMAP against CuMetal (first build takes several minutes) ==="
  CUMETAL_BUILD_DIR="${BUILD_DIR}" bash "${ROOT_DIR}/scripts/build_colmap_cumetal.sh" \
      > "${COLMAP_LOG}" 2>&1 || {
        echo "FAIL: build failed; see ${COLMAP_LOG}"; tail -20 "${COLMAP_LOG}"; exit 1; }
fi
COLMAP="$(sed -n 's/^colmap: //p' "${COLMAP_LOG}" | tail -1)"
[[ -x "${COLMAP}" ]] || { echo "FAIL: colmap binary missing (${COLMAP:-?})"; exit 1; }
echo "  $(grep '^COLMAP src:' "${COLMAP_LOG}" | tail -1)"
export DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"

echo "=== rendering the synthetic scene ==="
SCENE="${OUT_DIR}/scene"
rm -rf "${SCENE}"
python3 -I "${SCRIPT_DIR}/make_scene.py" --out "${SCENE}" | tail -2

WS="${OUT_DIR}/dense"
rm -rf "${WS}"
stage() {
  local name="$1"; shift
  local t0=$SECONDS
  if ! "$@" > "${OUT_DIR}/${name}.log" 2>&1; then
    echo "FAIL: ${name} exited non-zero; see ${OUT_DIR}/${name}.log"
    tail -15 "${OUT_DIR}/${name}.log"
    exit 1
  fi
  echo "  ${name}: $((SECONDS - t0)) s"
}

echo "=== COLMAP dense pipeline ==="
stage image_undistorter "${COLMAP}" image_undistorter \
    --image_path "${SCENE}/images" --input_path "${SCENE}/sparse" \
    --output_path "${WS}" --output_type COLMAP
stage patch_match_stereo env CUMETAL_TRACE_GPU=1 "${COLMAP}" patch_match_stereo \
    --workspace_path "${WS}" --PatchMatchStereo.geom_consistency true
stage stereo_fusion "${COLMAP}" stereo_fusion \
    --workspace_path "${WS}" --input_type geometric --output_path "${WS}/fused.ply"

echo "=== provenance ==="
PM_LOG="${OUT_DIR}/patch_match_stereo.log"
ok_launches="$(grep -c 'event=kernel_launch.*device=apple_gpu.*launch_success=true' "${PM_LOG}" || true)"
bad_launches="$(grep -c 'event=kernel_launch.*launch_success=false' "${PM_LOG}" || true)"
missing="$(grep -c 'registered kernel missing metallib' "${PM_LOG}" || true)"
kernels="$(grep -o 'event=kernel_launch kernel="[^"]*"' "${PM_LOG}" | sort -u | wc -l | tr -d ' ')"
echo "  ${ok_launches} Apple-GPU launches of ${kernels} distinct kernels, ${bad_launches} failed, ${missing} unlowered"
if [[ "${ok_launches}" -eq 0 || "${bad_launches}" -ne 0 || "${missing}" -ne 0 ]]; then
  echo "FAIL: provenance -- see ${PM_LOG}"
  exit 1
fi

echo "=== accuracy against ground truth ==="
python3 -I "${SCRIPT_DIR}/gate.py" --scene "${SCENE}" --workspace "${WS}" \
    --figure "${OUT_DIR}/depth_comparison.png" --json "${OUT_DIR}/gate.json"
