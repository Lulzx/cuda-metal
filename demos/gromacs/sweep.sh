#!/usr/bin/env bash
# Run every case in the GROMACS benchmark set through demos/gromacs' gate.
#
#   bash demos/gromacs/sweep.sh                 # every extracted case
#   bash demos/gromacs/sweep.sh --max-atoms=N   # skip anything larger (default 300000)
#   bash demos/gromacs/sweep.sh villin rnase    # only cases whose path matches
#
# run.sh runs two hand-picked systems. This walks out/benchmarks for every
# directory holding conf.gro + topol.top and runs each .mdp in it -- so the
# reaction-field and virtual-site variants, which run.sh never touches, are
# covered too. The gate is identical: grompp once, integrate the same .tpr on
# the CPU build and on the CuMetal build, compare every energy term at every
# step. Cases are sized-gated because the CPU reference is the slow half.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
BENCH_DIR="${OUT_DIR}/benchmarks"
SWEEP_DIR="${OUT_DIR}/sweep"
MAX_ATOMS=300000
FILTERS=()
for arg in "$@"; do
  case "$arg" in
    --max-atoms=*) MAX_ATOMS="${arg#--max-atoms=}" ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) FILTERS+=("$arg") ;;
  esac
done

BUILD_DIR=""
for cand in build build-release build-noshim; do
  [[ -f "${ROOT_DIR}/${cand}/libcumetal.dylib" ]] && { BUILD_DIR="${ROOT_DIR}/${cand}"; break; }
done
[[ -n "${BUILD_DIR}" ]] || { echo "FAIL: libcumetal not found"; exit 1; }
export DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"

GMX_GPU="/tmp/tmp/gromacs/build-cumetal/bin/gmx"
GMX_CPU="/tmp/tmp/gromacs/build-cpu/bin/gmx"
for b in "${GMX_GPU}" "${GMX_CPU}"; do
  [[ -x "${b}" ]] || { echo "FAIL: ${b} missing; run demos/gromacs/run.sh first"; exit 1; }
done

STEPS="${CUMETAL_GROMACS_STEPS:-20}"
mkdir -p "${SWEEP_DIR}"

printf "%-26s %-10s %-9s | %-30s | %s\n" case atoms mdp "GPU-vs-CPU / CPU-vs-itself" "GPU tasks"
printf -- "-%.0s" {1..105}; echo

fails=0; ran=0; skipped=0
while IFS= read -r conf; do
    dir="$(dirname "${conf}")"
    [[ -f "${dir}/topol.top" ]] || continue
    rel="${dir#"${BENCH_DIR}"/}"
    # The grappa set ships coordinates gzipped and its mdp files one level up in
    # a shared mdp-files/. Read the atom count straight out of the gzip rather
    # than expanding first -- the 6.1M-atom member is not worth unpacking only
    # to find it is over the size limit.
    gz=0
    if [[ "${conf}" == *.gz ]]; then
        # Expanding conf.gro.gz leaves a conf.gro beside it, which find then
        # reports as a second case in the same directory. Let the plain file win.
        [[ -f "${dir}/conf.gro" ]] && continue
        gz=1
    fi
    if [[ ${#FILTERS[@]} -gt 0 ]]; then
        match=0
        for f in "${FILTERS[@]}"; do [[ "${rel}" == *"${f}"* ]] && match=1; done
        [[ ${match} -eq 1 ]] || continue
    fi
    if [[ ${gz} -eq 1 ]]; then atoms="$(gzcat "${conf}" 2>/dev/null | sed -n '2p' | tr -d ' \r')"
    else atoms="$(sed -n '2p' "${conf}" | tr -d ' \r')"; fi
    [[ "${atoms}" =~ ^[0-9]+$ ]] || continue
    if (( atoms > MAX_ATOMS )); then
        printf "%-26s %-10s %-9s | %s\n" "${rel:0:26}" "${atoms}" "-" "SKIP (over --max-atoms=${MAX_ATOMS})"
        skipped=$((skipped+1)); continue
    fi

    if [[ ${gz} -eq 1 ]]; then
        [[ -f "${dir}/conf.gro" ]] || gzcat "${conf}" > "${dir}/conf.gro" || continue
    fi
    # grompp drops an mdout.mdp next to the inputs, so a plain *.mdp glob makes
    # the second run of this script behave differently from the first: the
    # leftover would count as "this case has its own mdp" and suppress the
    # fallback to grappa's shared mdp-files/, silently dropping those cases.
    # Filter it out here, and keep grompp from writing it there at all below.
    mdps=()
    for cand in "${dir}"/*.mdp; do
        [[ -e "${cand}" ]] || continue
        [[ "$(basename "${cand}")" == mdout.mdp ]] && continue
        mdps+=("${cand}")
    done
    if [[ ${#mdps[@]} -eq 0 ]]; then
        for cand in "$(dirname "${dir}")"/mdp-files/*.mdp; do
            [[ -e "${cand}" ]] && mdps+=("${cand}")
        done
    fi
    [[ ${#mdps[@]} -gt 0 ]] || { echo "FAIL: no mdp for ${rel}"; fails=$((fails+1)); continue; }
    for mdp in "${mdps[@]}"; do
        base="$(basename "${mdp}" .mdp)"
        # mdout.mdp is grompp's echo of a previous run, not an input.
        [[ "${base}" == mdout ]] && continue
        tag="$(echo "${rel}_${base}" | tr '/' '_')"
        work="${SWEEP_DIR}/${tag}"
        rm -rf "${work}"; mkdir -p "${work}"

        # Same determinism edits as run.sh: fixed steps, energies every step, no
        # thermostat (v-rescale draws random numbers and would diverge the two
        # builds for reasons unrelated to the GPU).
        sed -e "s/^nsteps .*/nsteps = ${STEPS}/" \
            -e 's/^nstcalcenergy .*/nstcalcenergy = 1/' \
            -e 's/^nstlog .*/nstlog = 1/' \
            -e 's/^nstenergy .*/nstenergy = 1/' "${mdp}" > "${work}/run.mdp"
        # v-rescale draws random numbers, so the thermostat has to be made
        # deterministic or the two builds diverge for a reason that has nothing
        # to do with the GPU. Dropping it is the simple way, but a barostat
        # needs an ensemble temperature and grompp refuses the pair -- so where
        # there is pressure coupling, keep the thermostat and pin its seed
        # instead, which makes both builds draw the same numbers.
        if grep -qiE '^ *pcoupl *=' "${work}/run.mdp" &&
           ! grep -qiE '^ *pcoupl *= *no' "${work}/run.mdp"; then
            sed -i '' -e 's/^ld-seed .*/ld-seed = 1/' -e 's/^ld_seed .*/ld_seed = 1/' "${work}/run.mdp"
            grep -qiE '^ *ld.seed' "${work}/run.mdp" || echo "ld-seed = 1" >> "${work}/run.mdp"
        else
            sed -i '' 's/^tcoupl .*/tcoupl = no/' "${work}/run.mdp"
        fi

        if ! ( cd "${dir}" && "${GMX_CPU}" grompp -f "${work}/run.mdp" -o "${work}/t.tpr" \
                 -po "${work}/mdout.mdp" -maxwarn 5 ) > "${work}/grompp.log" 2>&1; then
            # The smallest water boxes are physically narrower than the cut-off
            # these mdp files ask for. That is a property of the input, not a
            # result about the GPU, so it is not counted as a failure.
            if grep -q "cut-off .* is longer than half the shortest box vector" "${work}/grompp.log"; then
                printf "%-26s %-10s %-9s | %s\n" "${rel:0:26}" "${atoms}" "${base}" \
                       "SKIP (box smaller than the cut-off)"
                skipped=$((skipped+1))
            else
                printf "%-26s %-10s %-9s | %s\n" "${rel:0:26}" "${atoms}" "${base}" \
                       "FAIL: grompp -- $(sed -n '/^Fatal error/{n;p;}' "${work}/grompp.log" | head -1)"
                fails=$((fails+1))
            fi
            continue
        fi

        ok=1
        # The reference, plus the same reference at a different thread count.
        # The second one is not a spare: how far the CPU build lands from itself
        # when only the summation order changes is the floor below which no
        # correct GPU implementation can get, and it grows with system size.
        for n in 4 2 1; do
            rd="${work}/cpu${n}"; mkdir -p "${rd}"
            ( cd "${rd}" && "${GMX_CPU}" mdrun -s "${work}/t.tpr" -deffnm cpu \
                -ntmpi 1 -ntomp "${n}" -nb cpu -pme cpu -notunepme ) > "${rd}/mdrun.log" 2>&1 || ok=0
        done

        # Not every task is offloadable for every system, and GROMACS says so
        # before it runs anything: reaction-field has no PME mesh to put on a
        # GPU, and GPU LINCS handles neither virtual sites nor triangle
        # constraints. Those are GROMACS's own limits, not CuMetal's, so ask for
        # everything and drop whatever it refuses rather than pre-deciding --
        # a task that disappears for any *other* reason then still shows up as a
        # hole in the "GPU tasks" column.
        rd="${work}/gpu"; mkdir -p "${rd}"
        tasks=(-nb gpu -pme gpu -bonded gpu -update gpu)
        declined=""
        for attempt in 1 2 3; do
            ( cd "${rd}" && "${GMX_GPU}" mdrun -s "${work}/t.tpr" -deffnm gpu \
                -ntmpi 1 -ntomp 4 "${tasks[@]}" -notunepme ) > "${rd}/mdrun.log" 2>&1 && break
            drop=""
            grep -q "Cannot compute PME interactions on a GPU" "${rd}/mdrun.log" && drop=pme
            grep -q "Update task can not run on the GPU"       "${rd}/mdrun.log" && drop=update
            # Pure-water systems have no bonded interactions at all.
            grep -q "Bonded interactions can not be computed on a GPU" "${rd}/mdrun.log" && drop=bonded
            [[ -n "${drop}" ]] || { ok=0; break; }
            declined+="${drop} "
            # tasks is (-nb gpu -pme gpu ...): flip the value that follows -<drop>.
            for ((i = 0; i < ${#tasks[@]}; i++)); do
                [[ "${tasks[i]}" == "-${drop}" ]] && tasks[i+1]=cpu
            done
        done
        gpu_log="${work}/gpu/gpu.log"; cpu_log="${work}/cpu4/cpu.log"; env_log="${work}/cpu2/cpu.log"; env_log1="${work}/cpu1/cpu.log"
        if [[ ${ok} -eq 0 || ! -f "${gpu_log}" || ! -f "${cpu_log}" ]]; then
            why="mdrun failed"
            # A task GROMACS itself refuses to offload is a coverage gap, not a
            # CuMetal defect -- report it as such rather than as a failure.
            grep -qi "not supported.*GPU\|cannot be used on the GPU\|is not supported with" \
                "${gpu_log}" 2>/dev/null && why="unsupported by GROMACS GPU path"
            printf "%-26s %-10s %-9s | %s\n" "${rel:0:26}" "${atoms}" "${base}" "FAIL: ${why}"
            sed -n '/^Fatal error/,+6p' "${rd}/mdrun.log" 2>/dev/null | head -7 | sed 's/^/      /'
            fails=$((fails+1)); continue
        fi

        verdict="$(python3 "${SCRIPT_DIR}/gate.py" "${cpu_log}" "${gpu_log}" \
                     --envelope "${env_log}" --envelope "${env_log1}" > "${work}/gate.log" 2>&1 && echo ok || echo bad)"
        largest="$(sed -n 's/^ *largest: //p' "${work}/gate.log" | head -1 |
                   sed -n 's/.*rel=\([0-9.e+-]*\).*/\1/p')"
        floor="$(sed -n 's/.*noise floor //p' "${work}/gate.log" | head -1)"
        assigned=""
        grep -q "short-ranged.*interactions on the GPU" "${gpu_log}" && assigned+="nb "
        grep -q "bonded interactions on the GPU" "${gpu_log}" && assigned+="bonded "
        grep -q "PME tasks will do all aspects on the GPU" "${gpu_log}" && assigned+="pme "
        grep -q "update and constrain coordinates on the GPU" "${gpu_log}" && assigned+="update"

        printf "%-26s %-10s %-9s | %-30s | %s\n" "${rel:0:26}" "${atoms}" "${base}" \
               "${largest:-?} vs ${floor:-?} noise floor" "${assigned:-NONE}${declined:+ (GROMACS declines: ${declined% })}"
        ran=$((ran+1))
        if [[ "${verdict}" != ok ]]; then
            sed -n '/^FAIL/p' "${work}/gate.log" | head -3 | sed 's/^/      /'
            fails=$((fails+1))
        fi
        # nb is the only task every mdp can offload; pme is absent by
        # construction for reaction-field, so only nb is required here.
        [[ "${assigned}" == *nb* ]] || { echo "      FAIL: nonbonded stayed on the CPU"; fails=$((fails+1)); }
    done
done < <(find "${BENCH_DIR}" \( -name conf.gro -o -name conf.gro.gz \) | sort)

echo
echo "${ran} case(s) run, ${skipped} skipped by size, ${fails} failure(s)"
[[ ${fails} -eq 0 ]]
