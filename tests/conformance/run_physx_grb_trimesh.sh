#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${PHYSX_REPO:-${ROOT_DIR}/../PhysX}"
BUILD_DIR="${CUMETAL_PHYSX_RUNTIME_BUILD_DIR:-${ROOT_DIR}/build/physx-cumetal-runtime}"
STEPS="${CUMETAL_PHYSX_CONFORMANCE_STEPS:-30}"
FRICTION_STEPS="${CUMETAL_PHYSX_FRICTION_STEPS:-60}"
REL_TOL="${CUMETAL_PHYSX_REL_TOL:-1e-3}"
ABS_TOL="${CUMETAL_PHYSX_ABS_TOL:-1e-5}"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
    echo "SKIP: PhysX GRB triangle-mesh conformance requires Apple Silicon"
    exit 77
fi
if ! xcrun metal --version >/dev/null 2>&1; then
    metal_toolchain="$(xcodebuild -showComponent MetalToolchain -json 2>/dev/null | \
        python3 -c 'import json, sys; data = json.load(sys.stdin); print(data.get("toolchainIdentifier", "") if data.get("status") == "installed" else "")' \
        2>/dev/null || true)"
    if [[ -z "${metal_toolchain}" ]]; then
        echo "SKIP: the optional Xcode Metal Toolchain is unavailable"
        exit 77
    fi
    export TOOLCHAINS="${metal_toolchain}"
fi
if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "SKIP: PhysX checkout not found at ${PHYSX_REPO}"
    exit 77
fi

"${ROOT_DIR}/scripts/physx-patches/build_physx_cumetal_grb_macos.sh" >/dev/null

SNIPPET="${BUILD_DIR}/artifacts/bin/UNKNOWN/release/SnippetHelloGRB"
KERNEL_DIR="${BUILD_DIR}/sdk_cumetal_gpu_source_bin/kernels"
RESULT_DIR="${BUILD_DIR}/conformance"
CPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-cpu.tsv"
GPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-gpu.tsv"
CPU_LOG="${RESULT_DIR}/physx-grb-trimesh-cpu.log"
GPU_LOG="${RESULT_DIR}/physx-grb-trimesh-gpu.log"
FRICTION_CPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-friction-cpu.tsv"
FRICTION_GPU_DUMP="${RESULT_DIR}/physx-grb-trimesh-friction-gpu.tsv"
FRICTION_OFF_DUMP="${RESULT_DIR}/physx-grb-trimesh-frictionless-long.tsv"
FRICTION_CPU_LOG="${RESULT_DIR}/physx-grb-trimesh-friction-cpu.log"
FRICTION_GPU_LOG="${RESULT_DIR}/physx-grb-trimesh-friction-gpu.log"
FRICTION_OFF_LOG="${RESULT_DIR}/physx-grb-trimesh-frictionless-long.log"
mkdir -p "${RESULT_DIR}"

"${SNIPPET}" --cpu --sphere --trimesh --frictionless --steps "${STEPS}" \
    --dump "${CPU_DUMP}" >"${CPU_LOG}" 2>&1

python3 - "${CPU_DUMP}" <<'PY'
import csv
import sys

with open(sys.argv[1], newline="") as stream:
    rows = list(csv.DictReader(stream, delimiter="\t"))
if not rows or float(rows[0]["px"]) >= 0.0 or float(rows[-1]["px"]) <= 0.0:
    raise SystemExit("FAIL: triangle-mesh conformance trajectory does not cross the internal seam")
PY

env \
    CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
    CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
    CUMETAL_SYNC_EACH_LAUNCH=1 \
    CUMETAL_TRACE_GPU=1 \
    DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
    "${SNIPPET}" --gpu --sphere --trimesh --frictionless --steps "${STEPS}" \
    --dump "${GPU_DUMP}" >"${GPU_LOG}" 2>&1

"${SNIPPET}" --cpu --sphere --trimesh --friction --steps "${FRICTION_STEPS}" \
    --dump "${FRICTION_CPU_DUMP}" >"${FRICTION_CPU_LOG}" 2>&1

for mode in friction frictionless; do
    if [[ "${mode}" == friction ]]; then
        mode_dump="${FRICTION_GPU_DUMP}"
        mode_log="${FRICTION_GPU_LOG}"
    else
        mode_dump="${FRICTION_OFF_DUMP}"
        mode_log="${FRICTION_OFF_LOG}"
    fi
    env \
        CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 \
        CUMETAL_PHYSX_KERNEL_DIR="${KERNEL_DIR}" \
        CUMETAL_SYNC_EACH_LAUNCH=1 \
        CUMETAL_TRACE_GPU=1 \
        DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}" \
        "${SNIPPET}" --gpu --sphere --trimesh --"${mode}" --steps "${FRICTION_STEPS}" \
        --dump "${mode_dump}" >"${mode_log}" 2>&1
done

grep -q 'CuMetal GRB geometry: sphere' "${GPU_LOG}"
grep -q 'CuMetal GRB ground: trimesh' "${CPU_LOG}"
grep -q 'CuMetal GRB ground: trimesh' "${GPU_LOG}"
for kernel in \
    midphaseGeneratePairs \
    sphereTrimeshNarrowphase \
    sortTriangleIndices \
    convexTrimeshPostProcess \
    convexTrimeshCorrelate \
    convexTrimeshFinishContacts; do
    grep -q "kernel=\"${kernel}\".*source=metallib.*device=apple_gpu.*launch_success=true" "${GPU_LOG}"
done
for log in "${GPU_LOG}" "${FRICTION_GPU_LOG}" "${FRICTION_OFF_LOG}"; do
    if grep -qi 'internal error\|failed to create compute pipeline\|GPU Hang' "${log}"; then
        echo "FAIL: PhysX GRB triangle-mesh GPU log contains an internal runtime error"
        exit 1
    fi
done
grep -q 'CuMetal GRB contact mode: friction' "${FRICTION_CPU_LOG}"
grep -q 'CuMetal GRB contact mode: friction' "${FRICTION_GPU_LOG}"
grep -q 'CuMetal GRB contact mode: frictionless' "${FRICTION_OFF_LOG}"

if "${SNIPPET}" --gpu --box --trimesh --frictionless --steps 1 \
    >"${RESULT_DIR}/physx-grb-trimesh-unsupported.log" 2>&1; then
    echo "FAIL: unsupported GPU box/triangle-mesh mode was accepted"
    exit 1
fi
grep -q 'GPU triangle-mesh mode currently supports one separated sphere only' \
    "${RESULT_DIR}/physx-grb-trimesh-unsupported.log"

python3 "${SCRIPT_DIR}/compare_physx_grb.py" \
    "${CPU_DUMP}" "${GPU_DUMP}" "${STEPS}" "${REL_TOL}" "${ABS_TOL}"
python3 "${SCRIPT_DIR}/compare_physx_grb.py" \
    "${FRICTION_CPU_DUMP}" "${FRICTION_GPU_DUMP}" "${FRICTION_STEPS}" 3e-3 1e-5
python3 - "${FRICTION_GPU_DUMP}" "${FRICTION_OFF_DUMP}" <<'PY'
import csv
import sys

def final(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))[-1]

friction = final(sys.argv[1])
disabled = final(sys.argv[2])
vx = float(friction["vx"])
wz = float(friction["wz"])
off_vx = float(disabled["vx"])
off_wz = float(disabled["wz"])
if not 2.0 < vx < 4.0 or wz >= -2.0 or abs(vx + wz) > 0.02:
    raise SystemExit(f"FAIL: triangle-mesh friction did not reach rolling: vx={vx:g} wz={wz:g}")
if abs(off_vx - 5.0) > 0.02 or abs(off_wz) > 0.02:
    raise SystemExit(f"FAIL: triangle-mesh frictionless control changed motion: vx={off_vx:g} wz={off_wz:g}")
if float(disabled["px"]) - float(friction["px"]) < 1.0:
    raise SystemExit("FAIL: triangle-mesh friction did not materially reduce travel")
PY
echo "PASS: PhysX GRB sphere/static-triangle-mesh contact and friction matched CPU"
