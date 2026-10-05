#!/usr/bin/env bash
# Run the pinned stock LJ benchmark with fresh caches and GPU provenance.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${CUMETAL_LAMMPS_DIR:-/tmp/cumetal-lammps-30Sep2026}"
BUILD="${CUMETAL_BUILD_DIR:-${ROOT}/build}"
LMP="${SRC}/build-cumetal-cuda/lmp"
[[ -x "${LMP}" ]] || { echo "Build LAMMPS with scripts/build_lammps_cumetal.sh first" >&2; exit 2; }
[[ "$(git -C "${SRC}" rev-parse HEAD)" = 8de817dd79bfe4525d5d39246a212d833e6dee07 ]] || exit 2
[[ -z "$(git -C "${SRC}" status --porcelain --untracked-files=no)" ]] || exit 2
mkdir -p "${ROOT}/demos/lammps/out"
OUT="$(mktemp -d "${ROOT}/demos/lammps/out/gpu.XXXXXX")"
export DYLD_LIBRARY_PATH="${BUILD}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export CUMETAL_PTX_BACKEND=cumetal-ir
export CUMETAL_USE_METAL_DEVICE_ADDRESSES=1
export CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0
export CUMETAL_TRACE_GPU=1
export CUMETAL_DEBUG_REGISTRATION=1
for mode in full half; do
    newton=off; [[ "${mode}" = half ]] && newton=on
    mkdir -p "${OUT}/cache-${mode}"
    export CUMETAL_CACHE_DIR="${OUT}/cache-${mode}"
    (cd "${OUT}" && "${LMP}" -k on g 1 -sf kk -pk kokkos neigh "${mode}" newton "${newton}" \
        -in "${SRC}/bench/in.lj" -log "${mode}.lammps.log") > "${OUT}/${mode}.log" 2>&1 || {
        echo "GPU ${mode} run failed; log: ${OUT}/${mode}.log" >&2; exit 1;
    }
    grep -q 'device=apple_gpu' "${OUT}/${mode}.log" || {
        echo "Missing GPU provenance: ${OUT}/${mode}.log" >&2; exit 1;
    }
done
echo "GPU execution logs (numerical validation required): ${OUT}"
