#!/usr/bin/env bash
# CuMetal GROMACS demo: a molecular dynamics engine's CUDA GPU path on Apple Silicon.
#
# Usage:
#   bash demos/gromacs/run.sh              # villin + rnase, ~15 min including builds
#   bash demos/gromacs/run.sh --quick      # villin only
#   bash demos/gromacs/run.sh --all        # adds ADH (~134k atoms)
#   bash demos/gromacs/run.sh --build-dir=path/to/build
#
# Nothing from GROMACS lives in this tree. run.sh calls
# scripts/build_gromacs_cumetal.sh, which fetches the release tarball outside the
# repo and builds it twice from the same source: once with GMX_GPU=CUDA against
# CuMetal, once with GMX_GPU=OFF as the reference. Benchmark inputs come from
# https://gromacs-benchmarks-4ed623.gitlab.io/ (CC BY 4.0).
#
# The gate is a step-by-step energy comparison, not a final number: grompp is
# re-run with nstcalcenergy=1 and thermostatting off so both builds integrate the
# same deterministic trajectory and every energy term is printed at every step.
# MD is chaotic, so agreement is only meaningful over a short window -- 20 steps
# here, which is long enough that a wrong pair list or a stale force buffer has
# already moved the potential by percent, and short enough that single-precision
# rounding has not. A run that finishes without agreeing on the energies is a
# failure, not a pass.
#
# Exit 0 only if every energy term matches the CPU build at every step, the
# GROMACS log confirms the work was assigned to the GPU, and CuMetal's own trace
# shows kernels running on the Apple GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
BENCH_DIR="${OUT_DIR}/benchmarks"
BUILD_DIR=""
QUICK=0
ALL=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --all) ALL=1 ;;
    --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
    -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "${BUILD_DIR}" ]]; then
  for cand in build build-release build-nosshim build-noshim; do
    [[ -f "${ROOT_DIR}/${cand}/libcumetal.dylib" ]] && { BUILD_DIR="${ROOT_DIR}/${cand}"; break; }
  done
fi
if [[ -z "${BUILD_DIR}" || ! -f "${BUILD_DIR}/libcumetal.dylib" ]]; then
  echo "FAIL: libcumetal not found. Build CuMetal first (cmake -B build && cmake --build build)."
  exit 1
fi
export DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"
mkdir -p "${OUT_DIR}" "${BENCH_DIR}"

# villin (5k atoms) is the smallest system in the set and the fastest signal.
# rnase (24k) and ADH (134k) cross the sizes where the pair list stops fitting
# in one dispatch's worth of work.
SYSTEMS=("villin:villin/villin" "rnase:rnase/rnase_cubic")
[[ ${QUICK} -eq 1 ]] && SYSTEMS=("villin:villin/villin")
[[ ${ALL} -eq 1 ]] && SYSTEMS+=("ADH:ADH/adh_cubic")

BENCH_BASE="https://gromacs-benchmarks-4ed623.gitlab.io/download"
STEPS="${CUMETAL_GROMACS_STEPS:-20}"

# ── build both GROMACS trees ─────────────────────────────────────────────────
echo "=== building GROMACS (CuMetal CUDA path + CPU reference) ==="
bash "${ROOT_DIR}/scripts/build_gromacs_cumetal.sh" --compare > "${OUT_DIR}/build.log" 2>&1 || {
    echo "FAIL: build failed; see ${OUT_DIR}/build.log"; tail -20 "${OUT_DIR}/build.log"; exit 1; }
GMX_GPU="$(sed -n 's/^GPU  gmx: //p' "${OUT_DIR}/build.log" | tail -1)"
GMX_CPU="$(sed -n 's/^CPU  gmx: //p' "${OUT_DIR}/build.log" | tail -1)"
for b in "${GMX_GPU}" "${GMX_CPU}"; do
    [[ -x "${b}" ]] || { echo "FAIL: gmx not built (${b:-missing})"; exit 1; }
done
"${GMX_GPU}" -version 2>/dev/null | sed -n 's/^\(GROMACS version\|GPU support\|GPU FFT library\): */  &/p' || true

# ── fetch benchmark inputs ───────────────────────────────────────────────────
for entry in "${SYSTEMS[@]}"; do
    archive="${entry%%:*}"
    [[ -d "${BENCH_DIR}/${archive}" ]] && continue
    echo "fetching ${archive} benchmark ..."
    curl -sfL --max-time 600 "${BENCH_BASE}/${archive}.tar.gz" -o "${BENCH_DIR}/${archive}.tar.gz" \
        || { echo "FAIL: could not fetch ${archive}.tar.gz"; exit 1; }
    tar xzf "${BENCH_DIR}/${archive}.tar.gz" -C "${BENCH_DIR}"
done

fails=0
echo
printf "%-10s %-8s | %-38s | %s\n" system atoms "energy agreement vs CPU build" "GPU tasks"
printf -- "-%.0s" {1..100}; echo

for entry in "${SYSTEMS[@]}"; do
    name="${entry##*:}"
    label="$(basename "${name}")"
    dir="${BENCH_DIR}/${name}"
    [[ -d "${dir}" ]] || { echo "FAIL: ${dir} missing"; fails=$((fails+1)); continue; }

    # The benchmark mdp files run open-ended (nsteps = -1) and only report energy
    # every 1000 steps, which is a throughput setup, not a comparable one. Derive
    # a short deterministic variant: fixed step count, energies every step, and
    # no thermostat -- v-rescale draws random numbers, so leaving it on would make
    # the two builds diverge for a reason that has nothing to do with the GPU.
    sed -e "s/^nsteps .*/nsteps = ${STEPS}/" \
        -e 's/^nstcalcenergy .*/nstcalcenergy = 1/' \
        -e 's/^nstlog .*/nstlog = 1/' \
        -e 's/^nstenergy .*/nstenergy = 1/' \
        -e 's/^tcoupl .*/tcoupl = no/' \
        "${dir}/pme.mdp" > "${OUT_DIR}/${label}.mdp"

    ( cd "${dir}" && "${GMX_CPU}" grompp -f "${OUT_DIR}/${label}.mdp" \
        -o "${OUT_DIR}/${label}.tpr" -maxwarn 2 ) > "${OUT_DIR}/${label}.grompp.log" 2>&1 || {
        echo "FAIL: grompp failed for ${label}; see ${OUT_DIR}/${label}.grompp.log"
        fails=$((fails+1)); continue; }
    # Line 2 of a .gro file is the atom count.
    atoms="$(sed -n '2p' "${dir}/conf.gro" | tr -d ' \r')"

    for which in cpu gpu; do
        rundir="${OUT_DIR}/${label}.${which}"
        rm -rf "${rundir}"; mkdir -p "${rundir}"
        if [[ "${which}" == cpu ]]; then
            binary="${GMX_CPU}"; tasks=(-nb cpu -pme cpu)
        else
            # PME stays on the CPU: CuMetal's cuFFT covers rank-1 transforms only,
            # and GROMACS's PME mesh needs a padded 3D R2C/C2R pair. Everything
            # else -- short-range nonbonded, listed forces, and the LINCS/SETTLE
            # constrained update -- runs on the Apple GPU.
            binary="${GMX_GPU}"; tasks=(-nb gpu -pme cpu -bonded gpu -update gpu)
        fi
        ( cd "${rundir}" && "${binary}" mdrun -s "${OUT_DIR}/${label}.tpr" -deffnm "${which}" \
            -ntmpi 1 -ntomp 4 "${tasks[@]}" -notunepme ) > "${rundir}/mdrun.log" 2>&1 || {
            echo "FAIL: ${label} ${which} mdrun failed; see ${rundir}/mdrun.log"
            tail -5 "${rundir}/mdrun.log"; fails=$((fails+1)); }
    done

    cpu_log="${OUT_DIR}/${label}.cpu/cpu.log"
    gpu_log="${OUT_DIR}/${label}.gpu/gpu.log"
    if [[ ! -f "${cpu_log}" || ! -f "${gpu_log}" ]]; then
        fails=$((fails+1)); continue
    fi

    verdict="$(python3 "${SCRIPT_DIR}/gate.py" "${cpu_log}" "${gpu_log}" \
                 > "${OUT_DIR}/${label}.gate.log" 2>&1 && echo ok || echo bad)"
    summary="$(sed -n 's/^ *largest: //p' "${OUT_DIR}/${label}.gate.log" | head -1)"
    largest="$(sed -n 's/.*rel=\([0-9.e+-]*\).*/\1/p' <<< "${summary}")"

    # A number that matches is not by itself evidence the GPU produced it.
    # GROMACS names each offloaded task in its log; require all three. Nonbonded
    # and bonded share one line ("short-ranged and most bonded interactions on
    # the GPU") when both are offloaded, and it degrades to "short-ranged
    # interactions on the GPU" when only nonbonded is.
    assigned=""
    if grep -q "short-ranged.*interactions on the GPU" "${gpu_log}"; then assigned+="nb "; fi
    if grep -q "bonded interactions on the GPU" "${gpu_log}"; then assigned+="bonded "; fi
    if grep -q "update and constrain coordinates on the GPU" "${gpu_log}"; then assigned+="update"; fi

    printf "%-10s %-8s | %-38s | %s\n" "${label}" "${atoms}" \
           "max rel diff ${largest:-?} over ${STEPS} steps" "${assigned:-NONE}"

    if [[ "${verdict}" != ok ]]; then
        echo "  FAIL: energies disagree with the CPU build"
        sed -n '/^FAIL/p' "${OUT_DIR}/${label}.gate.log" | head -5 | sed 's/^/    /'
        fails=$((fails+1))
    fi
    if [[ "${assigned}" != *nb* || "${assigned}" != *bonded* || "${assigned}" != *update* ]]; then
        echo "  FAIL: GROMACS did not offload all three tasks; the match may be a CPU result"
        fails=$((fails+1))
    fi
done

# ── Apple-GPU provenance ─────────────────────────────────────────────────────
# GROMACS saying "on the GPU" only means GROMACS took its CUDA path. This checks
# that CuMetal then reached Metal, rather than answering from a host fallback.
echo
echo "=== Apple-GPU provenance ==="
prov_label="$(basename "${SYSTEMS[0]##*:}")"
prov_dir="${OUT_DIR}/${prov_label}.provenance"
rm -rf "${prov_dir}"; mkdir -p "${prov_dir}"
( cd "${prov_dir}" && CUMETAL_TRACE_GPU=1 "${GMX_GPU}" mdrun -s "${OUT_DIR}/${prov_label}.tpr" \
    -deffnm p -ntmpi 1 -ntomp 4 -nb gpu -pme cpu -bonded gpu -update gpu -notunepme -nsteps 2 ) \
    > "${prov_dir}/provenance.log" 2>&1 || true
launches="$(grep -c 'device=apple_gpu' "${prov_dir}/provenance.log" || true)"
if [[ "${launches}" -gt 0 ]]; then
    echo "  ${launches} kernel launches traced with device=apple_gpu"
    grep -m1 'device_name=' "${prov_dir}/provenance.log" | sed 's/^/  /' || true
else
    echo "  FAIL: no kernel launch traced on the Apple GPU"
    fails=$((fails+1))
fi
if grep -q 'source=approximate_stub' "${prov_dir}/provenance.log"; then
    echo "  FAIL: an approximate stub kernel participated"
    fails=$((fails+1))
fi

echo
if [[ ${fails} -eq 0 ]]; then
    echo "PASS: GROMACS's CUDA path agrees with its own CPU build on the Apple GPU"
    exit 0
fi
echo "FAIL: ${fails} check(s) failed"
exit 1
