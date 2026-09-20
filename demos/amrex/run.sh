#!/usr/bin/env bash
# CuMetal AMReX demo: a block-structured AMR framework's CUDA GPU path on Apple Silicon.
#
# Usage:
#   bash demos/amrex/run.sh              # 200 steps, ~10 min including builds
#   bash demos/amrex/run.sh --quick      # 20 steps
#   bash demos/amrex/run.sh --long       # 1000 steps, the tutorial's own input
#   bash demos/amrex/run.sh --build-dir=path/to/build
#
# Nothing from AMReX lives in this tree. run.sh calls
# scripts/build_amrex_cumetal.sh, which fetches AMReX 25.09 and the tutorials
# repository outside the repo and builds AMReX twice from the same source: once
# with AMReX_GPU_BACKEND=CUDA against CuMetal, once with AMReX_GPU_BACKEND=NONE
# as the reference. AMReX is not patched.
#
# The gate is a field comparison, not a number of steps survived. Both builds
# integrate the same 3-D heat equation on the same 32^3 periodic domain from the
# same initial condition, and AMReX's own fcompare reports the per-variable
# L-infinity difference between the two plotfiles. The comparison is run twice:
# once at step 0, which isolates the initial-condition kernel, and once at the
# final step, which adds every stencil update and ghost-cell exchange in between.
#
# Reporting both matters here, because the two failures are different. A wrong
# stencil, a stale ghost cell or a dropped launch moves the final field by
# percent while leaving step 0 untouched. Step 0 instead measures CuMetal's FP64
# transcendentals: the initial condition is a Gaussian, and under FP64 emulation
# `exp` on a double evaluates through binary32 (docs/fp64-policy.md), which is
# the floor on agreement for this problem. Neither number is allowed to hide
# behind the other.
#
# Exit 0 only if both comparisons are within tolerance, the GPU field is not
# degenerate, and CuMetal's trace shows the kernels ran on the Apple GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
BUILD_DIR=""
STEPS=200
for arg in "$@"; do
  case "$arg" in
    --quick) STEPS=20 ;;
    --long)  STEPS=1000 ;;
    --steps=*) STEPS="${arg#--steps=}" ;;
    --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if [[ -z "${BUILD_DIR}" ]]; then
  for cand in build-release-shim build build-nosshim build-noshim; do
    [[ -f "${ROOT_DIR}/${cand}/libcumetal.dylib" ]] && { BUILD_DIR="${ROOT_DIR}/${cand}"; break; }
  done
fi
if [[ -z "${BUILD_DIR}" || ! -f "${BUILD_DIR}/libcumetal.dylib" ]]; then
  echo "FAIL: libcumetal not found. Build CuMetal first (cmake -B build && cmake --build build)."
  exit 1
fi
[[ "${BUILD_DIR}" != /* ]] && BUILD_DIR="${ROOT_DIR}/${BUILD_DIR}"

# AMReX's CUDA path requires the binary shim: the tutorial is compiled by Clang
# in CUDA mode, so its kernels reach the runtime through __cudaRegisterFatBinary
# rather than through cumetalc's embedded metallib.
# grep -q closes the pipe on its first match, so nm dies of SIGPIPE and, under
# `set -o pipefail`, a successful match would read as a failed pipeline.
shim_syms="$(nm -gU "${BUILD_DIR}/libcumetal.dylib" 2>/dev/null | grep -c "__cudaRegisterFatBinary" || true)"
if [[ "${shim_syms:-0}" -eq 0 ]]; then
  echo "SKIP: ${BUILD_DIR}/libcumetal.dylib has no __cudaRegisterFatBinary."
  echo "      Reconfigure with -DCUMETAL_ENABLE_BINARY_SHIM=ON."
  exit 77
fi

mkdir -p "${OUT_DIR}"
export DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"

# AMReX keeps a device-resident array of Array4 descriptors -- each one holding
# the raw pointer to a box's data -- and its reductions and fused multi-box
# kernels index that array on the GPU. Following a pointer that was loaded from
# device memory, rather than passed as a kernel argument, needs the allocation it
# points at to be resident for the dispatch, which is what this mode does. It is
# the same requirement PhysX has (docs/physx-feasibility.md). Without it the
# loads return zeros with no error raised, so it is set here rather than left to
# the caller: see demos/amrex/README.md.
export CUMETAL_USE_METAL_DEVICE_ADDRESSES=1

echo "=== building AMReX (CuMetal CUDA path + CPU reference) ==="
CUMETAL_BUILD_DIR="${BUILD_DIR}" \
  bash "${ROOT_DIR}/scripts/build_amrex_cumetal.sh" --compare > "${OUT_DIR}/build.log" 2>&1 || {
    echo "FAIL: build failed; see ${OUT_DIR}/build.log"; tail -25 "${OUT_DIR}/build.log"; exit 1; }
HEAT_GPU="$(sed -n 's/^GPU  heat: //p' "${OUT_DIR}/build.log" | tail -1)"
HEAT_CPU="$(sed -n 's/^CPU  heat: //p' "${OUT_DIR}/build.log" | tail -1)"
TOOLS="$(sed -n 's/^Tools:     //p' "${OUT_DIR}/build.log" | tail -1)"
AMREX_SRC="$(sed -n 's/^AMReX src: //p' "${OUT_DIR}/build.log" | tail -1)"
for b in "${HEAT_GPU}" "${HEAT_CPU}" "${TOOLS}/amrex_fcompare" "${TOOLS}/amrex_fextrema"; do
    [[ -x "${b}" ]] || { echo "FAIL: missing build product ${b:-?}"; exit 1; }
done
echo "  AMReX source: ${AMREX_SRC} ($(git -C "${AMREX_SRC}" describe --tags 2>/dev/null || echo '?'))"

# The tutorial's own inputs file. n_cell=32 over 8 boxes of 16^3 exercises the
# ghost-cell exchange; a single box would not.
cat > "${OUT_DIR}/inputs" <<EOF
n_cell = 32
max_grid_size = 16
nsteps = ${STEPS}
plot_int = ${STEPS}
dt = 1.e-5
EOF

fails=0
for which in cpu gpu; do
    run_dir="${OUT_DIR}/run-${which}"
    rm -rf "${run_dir}"; mkdir -p "${run_dir}"
    bin="${HEAT_CPU}"; [[ "${which}" == gpu ]] && bin="${HEAT_GPU}"
    trace=""; [[ "${which}" == gpu ]] && trace=1
    ( cd "${run_dir}" && CUMETAL_TRACE_GPU="${trace}" "${bin}" "${OUT_DIR}/inputs" ) \
        > "${run_dir}/run.log" 2>&1 || {
        echo "FAIL: ${which} run exited non-zero; see ${run_dir}/run.log"
        tail -15 "${run_dir}/run.log"; fails=$((fails+1)); }
done

plt="$(printf 'plt%05d' "${STEPS}")"
echo
printf "%-22s %-16s %-16s %s\n" "comparison" "abs (Linf)" "relative" "verdict"
printf -- "-%.0s" {1..74}; echo

# CuMetal's FP64 emulation carries a ~48-bit significand for arithmetic and a
# binary32-accurate exp; 1e-6 sits well above the measured agreement and far
# below what any real lowering defect produces, which is percent-level or total.
TOL="${CUMETAL_AMREX_TOL:-1e-6}"
check_plotfile () {
    local label="$1" tag="$2"
    local cpu_plt="${OUT_DIR}/run-cpu/${tag}" gpu_plt="${OUT_DIR}/run-gpu/${tag}"
    if [[ ! -d "${cpu_plt}" || ! -d "${gpu_plt}" ]]; then
        printf "%-22s %-16s %-16s %s\n" "${label}" "-" "-" "FAIL (plotfile missing)"
        fails=$((fails+1)); return
    fi
    # fcompare exits non-zero whenever the two plotfiles differ at all, which
    # here is always: the verdict is the tolerance below, not fcompare's status.
    local out; out="$("${TOOLS}/amrex_fcompare" "${cpu_plt}" "${gpu_plt}" 2>&1 || true)"
    echo "${out}" > "${OUT_DIR}/fcompare-${tag}.txt"
    # fcompare prints "<variable> <absolute error> <relative error>" per level.
    local abs rel
    abs="$(awk '$1=="phi"{print $2; exit}' <<<"${out}")"
    rel="$(awk '$1=="phi"{print $3; exit}' <<<"${out}")"
    if [[ -z "${rel}" ]]; then
        printf "%-22s %-16s %-16s %s\n" "${label}" "-" "-" "FAIL (unparsed fcompare)"
        fails=$((fails+1)); return
    fi
    local verdict="PASS"
    awk -v r="${rel}" -v t="${TOL}" 'BEGIN{exit !(r+0 <= t+0)}' || { verdict="FAIL"; fails=$((fails+1)); }
    printf "%-22s %-16s %-16s %s\n" "${label}" "${abs}" "${rel}" "${verdict}"
}
check_plotfile "step 0 (init)" "plt00000"
check_plotfile "step ${STEPS} (final)" "${plt}"

# A relative error of 0 against a degenerate field would be meaningless, and an
# all-zero GPU field is exactly how the two defects this demo exposed showed up.
# Require the GPU field to carry the tutorial's actual dynamic range.
gpu_ext="$("${TOOLS}/amrex_fextrema" "${OUT_DIR}/run-gpu/${plt}" 2>/dev/null | awk '$1=="phi"{print $2, $3; exit}')"
read -r gmin gmax <<<"${gpu_ext:-0 0}"
if awk -v a="${gmin}" -v b="${gmax}" 'BEGIN{exit !(b+0 - a+0 > 1e-3)}'; then
    printf "%-22s %-16s %-16s %s\n" "gpu field range" "${gmin}" "${gmax}" "PASS"
else
    printf "%-22s %-16s %-16s %s\n" "gpu field range" "${gmin}" "${gmax}" "FAIL (degenerate)"
    fails=$((fails+1))
fi

# Provenance: a correct number without evidence of Apple-GPU execution is not
# evidence of Apple-GPU execution.
trace="${OUT_DIR}/run-gpu/run.log"
launches="$(grep -c 'CUMETAL_PROVENANCE event=kernel_launch' "${trace}" 2>/dev/null || true)"
apple="$(grep -c 'device=apple_gpu' "${trace}" 2>/dev/null || true)"
failed_launch="$(grep -c 'launch_success=false' "${trace}" 2>/dev/null || true)"
if [[ "${launches}" -gt 0 && "${apple}" -eq "${launches}" && "${failed_launch}" -eq 0 ]]; then
    printf "%-22s %-16s %-16s %s\n" "gpu provenance" "${launches} launches" "device=apple_gpu" "PASS"
else
    printf "%-22s %-16s %-16s %s\n" "gpu provenance" "${launches} launches" "${apple} apple_gpu" "FAIL"
    fails=$((fails+1))
fi

echo
if [[ ${fails} -eq 0 ]]; then
    echo "PASS: AMReX ${STEPS}-step heat equation agrees with the CPU build on the Apple GPU"
    echo "Artifacts: ${OUT_DIR}"
    exit 0
fi
echo "FAIL: ${fails} check(s) failed. Artifacts: ${OUT_DIR}"
exit 1
