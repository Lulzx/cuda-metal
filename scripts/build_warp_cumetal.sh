#!/usr/bin/env bash
# build_warp_cumetal.sh — Fetch NVIDIA Warp, patch it for CuMetal, and compile
# its native CUDA sources through CuMetal's nvcc shim.
#
# Usage:
#   bash scripts/build_warp_cumetal.sh                 # clone + patch + compile sweep
#   bash scripts/build_warp_cumetal.sh --clone-only    # clone + patch, nothing else
#   bash scripts/build_warp_cumetal.sh --build         # also run Warp's own build_lib.py
#
# Environment overrides:
#   CUMETAL_WARP_DIR     checkout path (default: ../warp-cumetal, a sibling of this repo)
#   CUMETAL_WARP_REPO    git remote to clone (default: https://github.com/NVIDIA/warp.git)
#   CUMETAL_WARP_TAG     tag to pin (default: v1.12.0)
#   CUMETAL_BUILD_DIR    CuMetal build tree to use (default: ./build)
#   CUMETAL_JOBS         build parallelism for --build (default: hw.ncpu)
#
# ── why this script exists ───────────────────────────────────────────────────
# Warp needs two upstream changes before it can be built against CuMetal, and
# they live in NVIDIA's repository, not this one. Rather than carry a fork, the
# changes are kept here as patches (scripts/warp-patches/) and applied to a
# clone pinned at v1.12.0. Nothing is pushed anywhere; the checkout is yours.
#
#   0001  crt.h's barebones-Clang include guarded on WP_CUMETAL, an __APPLE__
#         dlopen branch for the driver, --cuda-path honoured on Darwin, and a
#         CuMetal branch in build_dll.py that links -lcuda rather than the
#         NVRTC/PTX static libraries.
#   0002  <new> included for volume_builder.cu's placement new, which nvcc
#         provides implicitly and Clang does not.
#
# ── what the sweep reports ───────────────────────────────────────────────────
# libwarp compiles 11 .cu files. Six compile through CuMetal today; the other
# five need CUB device headers that the CuMetal shim does not yet carry (Phase
# 4 in docs/warp-feasibility.md). The sweep prints one line per file and fails
# only if a file in the known-good set regresses. A file in the known-blocked
# set that starts compiling is reported as news, not as a failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="sweep"
case "${1:-}" in
    --clone-only) MODE="clone" ;;
    --build)      MODE="build" ;;
    "")           ;;
    *)            echo "usage: $0 [--clone-only|--build]" >&2; exit 2 ;;
esac

WARP_DIR="${CUMETAL_WARP_DIR:-${ROOT_DIR}/../warp-cumetal}"
WARP_REPO="${CUMETAL_WARP_REPO:-https://github.com/NVIDIA/warp.git}"
WARP_TAG="${CUMETAL_WARP_TAG:-v1.12.0}"
BUILD_DIR="${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}"
[[ "${BUILD_DIR}" != /* ]] && BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"
JOBS="${CUMETAL_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# ── fetch ────────────────────────────────────────────────────────────────────
if [[ -e "${WARP_DIR}" && ! -d "${WARP_DIR}/.git" ]]; then
    echo "ERROR: ${WARP_DIR} exists but is not a git checkout." >&2
    echo "       Set CUMETAL_WARP_DIR to somewhere else." >&2
    exit 2
fi
if [[ ! -d "${WARP_DIR}/.git" ]]; then
    echo "Cloning ${WARP_REPO} at ${WARP_TAG} -> ${WARP_DIR} ..."
    git -c advice.detachedHead=false clone --quiet --depth 1 \
        --branch "${WARP_TAG}" "${WARP_REPO}" "${WARP_DIR}"
fi
echo "Warp checkout: ${WARP_DIR} ($(git -C "${WARP_DIR}" rev-parse --short HEAD))"

# ── patch ────────────────────────────────────────────────────────────────────
bash "${SCRIPT_DIR}/warp-patches/apply_warp_patches.sh" "${WARP_DIR}"
[[ "${MODE}" == "clone" ]] && exit 0

# ── CuMetal CUDA toolkit shim ────────────────────────────────────────────────
if [[ ! -f "${BUILD_DIR}/libcumetal.dylib" ]]; then
    echo "ERROR: no CuMetal build at ${BUILD_DIR} (libcumetal.dylib missing)." >&2
    echo "       Build CuMetal first, or point CUMETAL_BUILD_DIR at a build tree." >&2
    exit 2
fi
FAKE_CUDA="${BUILD_DIR}/cumetal-cuda-toolkit"
if [[ ! -x "${FAKE_CUDA}/bin/nvcc" ]]; then
    echo "Generating CuMetal CUDA toolkit shim ..."
    CUMETAL_BUILD_DIR="${BUILD_DIR}" bash "${SCRIPT_DIR}/build_llama_cpp_cumetal.sh" --toolkit-only >/dev/null
fi
[[ -x "${FAKE_CUDA}/bin/nvcc" ]] || { echo "ERROR: ${FAKE_CUDA}/bin/nvcc missing." >&2; exit 2; }

# ── Warp's own build ─────────────────────────────────────────────────────────
if [[ "${MODE}" == "build" ]]; then
    echo "=== build_lib.py --cuda_path=${FAKE_CUDA} ==="
    echo "Note: this needs all 11 .cu files; five of them are still blocked on"
    echo "      the CUB device headers. Run without --build for the file-level"
    echo "      status this repo actually claims."
    cd "${WARP_DIR}"
    exec python build_lib.py --cuda_path="${FAKE_CUDA}" --jobs="${JOBS}"
fi

# ── per-file compile sweep ───────────────────────────────────────────────────
NATIVE="${WARP_DIR}/warp/native"
OUT_DIR="${BUILD_DIR}/warp-sweep"
mkdir -p "${OUT_DIR}"

# Mirrors the release nvcc line in warp/_src/build_dll.py, with the CuMetal
# define its own patched build would add from version.json.
compile_one() {
    "${FAKE_CUDA}/bin/nvcc" --std=c++17 -O3 \
        --compiler-options -fPIC,-fvisibility=hidden,-fvisibility-inlines-hidden \
        -gencode=arch=compute_80,code=sm_80 -t0 --extended-lambda -diag-suppress=221 \
        -DNDEBUG -DWP_ENABLE_CUDA=1 -DWP_CUMETAL=1 -DWP_ENABLE_MATHDX=0 \
        -I"${NATIVE}" -o "${OUT_DIR}/$1.o" -c "${NATIVE}/$1.cu"
}

# Known-good as of 2026-09-05; see docs/warp-feasibility.md.
EXPECTED_PASS=(hashgrid mesh scan volume volume_builder warp)
EXPECTED_FAIL=(bvh reduce runlength_encode sort sparse)

is_expected_pass() {
    local candidate
    for candidate in "${EXPECTED_PASS[@]}"; do [[ "${candidate}" == "$1" ]] && return 0; done
    return 1
}

regressions=()
surprises=()
passed=0
echo "=== compiling libwarp's CUDA sources through cumetalc ==="
for name in "${EXPECTED_PASS[@]}" "${EXPECTED_FAIL[@]}"; do
    log="${OUT_DIR}/${name}.log"
    if compile_one "${name}" >"${log}" 2>&1; then
        passed=$((passed + 1))
        if is_expected_pass "${name}"; then
            printf '  PASS  %s.cu\n' "${name}"
        else
            printf '  PASS  %s.cu  (was blocked -- update docs/warp-feasibility.md)\n' "${name}"
            surprises+=("${name}")
        fi
    else
        reason="$(grep -m1 'error:' "${log}" | cut -c1-100)"
        if is_expected_pass "${name}"; then
            printf '  FAIL  %s.cu  REGRESSION: %s\n' "${name}" "${reason}"
            regressions+=("${name}")
        else
            printf '  fail  %s.cu  (known, Phase 4): %s\n' "${name}" "${reason}"
        fi
    fi
done

total=$((${#EXPECTED_PASS[@]} + ${#EXPECTED_FAIL[@]}))
echo "${passed}/${total} of libwarp's CUDA sources compile (logs in ${OUT_DIR})"

if ((${#regressions[@]} > 0)); then
    echo "FAIL: previously compiling files broke: ${regressions[*]}" >&2
    exit 1
fi
((${#surprises[@]} > 0)) && echo "NOTE: newly compiling: ${surprises[*]}"
echo "PASS"
