#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX PBD recreations require Apple Silicon"
    exit 77
fi
if ! xcrun -f metal >/dev/null 2>&1; then
    echo "SKIP: xcrun metal is unavailable"
    exit 77
fi
if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "SKIP: PhysX checkout not found at ${PHYSX_REPO}"
    exit 77
fi

"${ROOT_DIR}/scripts/physx-patches/build_physx_cumetal_pbd_macos.sh" >/dev/null

BIN_DIR="${BUILD_DIR}/artifacts/bin/UNKNOWN/release"
KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
RESULT_DIR="${BUILD_DIR}/conformance"
FLAG_LOG="${RESULT_DIR}/physx-pbd-flag.log"
DRAPE_LOG="${RESULT_DIR}/physx-pbd-drape.log"
INFLATABLE_LOG="${RESULT_DIR}/physx-pbd-inflatable.log"
FROG_LOG="${RESULT_DIR}/physx-pbd-frog.log"
mkdir -p "${RESULT_DIR}"

run_scene() {
    local log_file="$1"
    shift
    env \
        CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
        CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
        CUMETAL_SYNC_EACH_LAUNCH=1 \
        CUMETAL_TRACE_GPU=1 \
        DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
        "$@" >"${log_file}" 2>&1
    if grep -qi 'internal error\|failed to create compute pipeline' "${log_file}"; then
        echo "FAIL: GPU log contains an internal runtime error: ${log_file}" >&2
        exit 1
    fi
}

run_scene "${FLAG_LOG}" "${BIN_DIR}/SnippetPBDCloth"
# The drape needs the full fall-contact-settle window, unlike the other
# scenes whose invariants hold from the first frames.
run_scene "${DRAPE_LOG}" env CUMETAL_PHYSX_DRAPE=1 CUMETAL_CAPTURE_FRAMES=240 \
    "${BIN_DIR}/SnippetPBDCloth"
run_scene "${INFLATABLE_LOG}" "${BIN_DIR}/SnippetPBDInflatable"
run_scene "${FROG_LOG}" env CUMETAL_PHYSX_FROG=1 "${BIN_DIR}/SnippetPBDInflatable"

grep -q 'CUMETAL_PHYSX_FLAG PASS' "${FLAG_LOG}"
grep -q 'kernel="ps_solveAerodynamics2Launch".*source=metallib.*device=apple_gpu.*launch_success=true' \
    "${FLAG_LOG}"
grep -q 'CUMETAL_PHYSX_DRAPE PASS' "${DRAPE_LOG}"
grep -q 'kernel="cumetalClothCollide".*source=metallib.*device=apple_gpu.*launch_success=true' \
    "${DRAPE_LOG}"
grep -q 'CUMETAL_PHYSX_INFLATABLE PASS' "${INFLATABLE_LOG}"
grep -q 'kernel="cumetalInflatablePressure".*source=metallib.*device=apple_gpu.*launch_success=true' \
    "${INFLATABLE_LOG}"
grep -q 'CUMETAL_PHYSX_FROG PASS' "${FROG_LOG}"
grep -q 'kernel="cumetalInflatablePressure".*source=metallib.*device=apple_gpu.*launch_success=true' \
    "${FROG_LOG}"

tail -1 "${FLAG_LOG}"
tail -1 "${DRAPE_LOG}"
tail -1 "${INFLATABLE_LOG}"
tail -1 "${FROG_LOG}"
echo "PASS: PhysX flag, drape, inflatable, and frog recreations used CuMetal Apple GPU kernels"
