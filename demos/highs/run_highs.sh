#!/usr/bin/env bash
# Unmodified HiGHS: compare CUPDLP_GPU=OFF with CUPDLP_GPU=ON through CuMetal.
# The GPU run forces cuSPARSE onto Metal so this proves HiGHS's captured-library
# integration, not only its ordinary translated CUDA kernels.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out/highs"
LP_DIR="${SCRIPT_DIR}/out/lps"
BUILD_DIR="${ROOT_DIR}/build"
HIGHS_SRC="${HIGHS_SRC:-${HOME}/work/cumetal-bench-ext/HiGHS}"
SKIP_BUILD=0

for arg in "$@"; do
    case "${arg}" in
        --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
        --highs-src=*) HIGHS_SRC="${arg#--highs-src=}" ;;
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            sed -n '2,15p' "$0"
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${OUT_DIR}" "${LP_DIR}"

if [[ ${SKIP_BUILD} -eq 0 ]]; then
    echo "=== building unmodified HiGHS (CPU + CUPDLP_GPU) ==="
    HIGHS_SRC="${HIGHS_SRC}" bash "${ROOT_DIR}/scripts/build_highs_cumetal.sh" \
        --build-dir="${BUILD_DIR}" > "${OUT_DIR}/build.log" 2>&1 || {
            echo "FAIL: HiGHS build failed; see ${OUT_DIR}/build.log"
            tail -40 "${OUT_DIR}/build.log"
            exit 1
        }
fi

CPU_BIN="${HIGHS_SRC}/build-cpu/bin/highs"
GPU_BIN="${HIGHS_SRC}/build-gpu/bin/highs"
[[ -x "${CPU_BIN}" ]] || { echo "FAIL: CPU HiGHS binary missing: ${CPU_BIN}"; exit 1; }
[[ -x "${GPU_BIN}" ]] || { echo "FAIL: GPU HiGHS binary missing: ${GPU_BIN}"; exit 1; }
[[ -f "${BUILD_DIR}/libcumetal.dylib" ]] || {
    echo "FAIL: CuMetal runtime missing: ${BUILD_DIR}/libcumetal.dylib"
    exit 1
}

MPS="${LP_DIR}/afiro.mps"
if [[ ! -f "${MPS}" ]]; then
    curl -sfL --max-time 60 \
        "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/afiro.mps" \
        -o "${MPS}" || { echo "FAIL: could not fetch afiro.mps"; exit 1; }
fi

echo "=== solving afiro through HiGHS PDLP ==="
"${CPU_BIN}" --solver pdlp --presolve off "${MPS}" \
    > "${OUT_DIR}/afiro.cpu.log" 2>&1 || true
DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}" \
CUMETAL_SPARSE_METAL=1 \
CUMETAL_TRACE_GPU=1 \
    "${GPU_BIN}" --solver pdlp --presolve off "${MPS}" \
    > "${OUT_DIR}/afiro.gpu.log" 2>&1 || true

read -r cs co cpi cdi cg ci < <(
    python3 "${SCRIPT_DIR}/parse_highs.py" "${OUT_DIR}/afiro.cpu.log")
read -r gs go gpi gdi gg gi < <(
    python3 "${SCRIPT_DIR}/parse_highs.py" "${OUT_DIR}/afiro.gpu.log")
verdict="$(python3 "${SCRIPT_DIR}/gate.py" \
    "${cs}" "${co}" "${cpi}" "${cdi}" "${cg}" \
    "${gs}" "${go}" "${gpi}" "${gdi}" "${gg}")"

launches="$(grep -c 'device=apple_gpu' "${OUT_DIR}/afiro.gpu.log" || true)"
spmv="$(grep -c 'kernel="cumetal_spmv' "${OUT_DIR}/afiro.gpu.log" || true)"
bad_stub="$(grep -c 'source=approximate_stub' "${OUT_DIR}/afiro.gpu.log" || true)"
bad_fp64="$(grep -E -c 'kernel="cumetal_spmv_.*_f64" .*semantic_quality=exact' \
    "${OUT_DIR}/afiro.gpu.log" || true)"
good_fp64="$(grep -E -c 'kernel="cumetal_spmv_.*_f64" .*semantic_quality=reduced_precision_fp64' \
    "${OUT_DIR}/afiro.gpu.log" || true)"

printf "  CPU: %-9s objective=%-17s p=%-9s d=%-9s gap=%s\n" \
    "${cs}" "${co}" "${cpi}" "${cdi}" "${cg}"
printf "  GPU: %-9s objective=%-17s p=%-9s d=%-9s gap=%s\n" \
    "${gs}" "${go}" "${gpi}" "${gdi}" "${gg}"
printf "  objective rel diff: %s; gate: %s\n" "${verdict%%|*}" "${verdict#*|}"
printf "  Apple-GPU launches: %s (%s cuSPARSE SpMV)\n" "${launches}" "${spmv}"

fails=0
if [[ "${verdict#*|}" != "ok" ]]; then
    echo "  FAIL: CUPDLP_GPU result disagrees with the CPU build"
    fails=$((fails + 1))
fi
if [[ "${launches}" -eq 0 ]]; then
    echo "  FAIL: no Apple-GPU launch provenance"
    fails=$((fails + 1))
fi
if [[ "${spmv}" -eq 0 ]]; then
    echo "  FAIL: HiGHS's captured cuSPARSE nodes did not run on Metal"
    fails=$((fails + 1))
fi
if [[ "${bad_stub}" -ne 0 ]]; then
    echo "  FAIL: approximate stub provenance participated"
    fails=$((fails + 1))
fi
if [[ "${bad_fp64}" -ne 0 ]]; then
    echo "  FAIL: an FP64 cuSPARSE kernel claimed exact arithmetic"
    fails=$((fails + 1))
fi
if [[ "${good_fp64}" -eq 0 ]]; then
    echo "  FAIL: no cuSPARSE FP64 launch reported reduced-precision provenance"
    fails=$((fails + 1))
fi

if [[ ${fails} -eq 0 ]]; then
    echo "PASS: unmodified HiGHS CUPDLP_GPU matches its CPU build and executes captured SpMV on Apple GPU"
else
    echo "FAIL: ${fails} HiGHS integration gate(s) failed"
fi
echo "logs: ${OUT_DIR}"
exit $((fails == 0 ? 0 : 1))
