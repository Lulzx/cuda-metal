#!/usr/bin/env bash
# Run the pinned stock Lennard-Jones benchmark on Kokkos Serial, in both
# neighbour modes. This is a CPU reference check, not a CuMetal GPU gate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SRC="${CUMETAL_LAMMPS_DIR:-/tmp/cumetal-lammps-30Sep2026}"
PREC="${1:-single}"
case "${PREC}" in
    single) BUILD_MODE=--cpu; CPU_BUILD="${SRC}/build-cumetal-reference" ;;
    double) BUILD_MODE=--cpu-double; CPU_BUILD="${SRC}/build-cumetal-reference-double" ;;
    *) echo "Expected single or double precision" >&2; exit 2 ;;
esac
mkdir -p "${SCRIPT_DIR}/out"
OUT="$(mktemp -d "${SCRIPT_DIR}/out/reference.XXXXXX")"
echo "Building Kokkos Serial reference; log: ${OUT}/build.log"
bash "${ROOT_DIR}/scripts/build_lammps_cumetal.sh" "${BUILD_MODE}" > "${OUT}/build.log" 2>&1 || {
    tail -30 "${OUT}/build.log"; exit 1;
}
for mode in full half; do
    newton=off; [[ "${mode}" = half ]] && newton=on
    echo "Running stock in.lj: neighbours=${mode}, newton=${newton}"
    "${CPU_BUILD}/lmp" -k on -sf kk \
        -pk kokkos neigh "${mode}" newton "${newton}" \
        -in "${SRC}/bench/in.lj" -log "${OUT}/${mode}.lammps.log" \
        > "${OUT}/${mode}.log" 2>&1 || { tail -30 "${OUT}/${mode}.log"; exit 1; }
done
python3 "${SCRIPT_DIR}/validate_reference.py" "${OUT}" "${PREC}" 2>&1 | tee "${OUT}/validation.log"
echo "CPU reference artifacts: ${OUT}"
