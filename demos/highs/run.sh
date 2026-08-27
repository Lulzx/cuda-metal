#!/usr/bin/env bash
# CuMetal HiGHS/cuPDLP-C demo — an LP solver's CUDA GPU path, on Apple Silicon.
#
# Usage:
#   bash demos/highs/run.sh              # 8-problem corpus, CPU vs Metal (~2-4 min)
#   bash demos/highs/run.sh --quick      # afiro only
#   bash demos/highs/run.sh --build-dir=path/to/build
#
# cuPDLP-C is the PDLP solver HiGHS vendors as its GPU path (HiGHS >= 1.7 exports
# ~104 cupdlp_* symbols of its own). It is built here unmodified except for three
# patches forced by building a standalone cuPDLP-C against a current HiGHS -- none
# of them Metal-related; see scripts/build_cupdlp_cumetal.sh.
#
# Exit 0 only if every problem reaches the same model status on both builds, the
# objectives agree, and the Metal run shows Apple-GPU provenance. A correct
# number without that provenance is not a pass.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/out"
LP_DIR="${OUT_DIR}/lps"
QUICK=0
BUILD_DIR=""
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
  esac
done
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"
mkdir -p "${OUT_DIR}" "${LP_DIR}"

# cuPDLP-C's own afiro plus a spread from HiGHS's own test instances: e226 and
# stair are ill-conditioned, israel is the classic badly-scaled one.
PROBLEMS=(afiro adlittle blending e226 israel shell standata stair)
[[ ${QUICK} -eq 1 ]] && PROBLEMS=(afiro)

echo "=== building cuPDLP-C (CuMetal + CPU reference) ==="
bash "${ROOT_DIR}/scripts/build_cupdlp_cumetal.sh" --compare > "${OUT_DIR}/build.log" 2>&1 || {
    echo "FAIL: build failed; see ${OUT_DIR}/build.log"; tail -20 "${OUT_DIR}/build.log"; exit 1; }
CUPDLP_DIR="$(sed -n 's/^Run: .*DYLD_LIBRARY_PATH=[^ ]* \(.*\)\/build\/bin\/plc .*/\1/p' \
              "${OUT_DIR}/build.log" | tail -1)"
[[ -x "${CUPDLP_DIR}/build/bin/plc" ]] || { echo "FAIL: plc not built"; exit 1; }
export DYLD_LIBRARY_PATH="${BUILD_DIR}:${DYLD_LIBRARY_PATH:-}"

for p in "${PROBLEMS[@]}"; do
    [[ -f "${LP_DIR}/${p}.mps" ]] && continue
    if [[ -f "${CUPDLP_DIR}/example/${p}.mps" ]]; then
        cp "${CUPDLP_DIR}/example/${p}.mps" "${LP_DIR}/"
    else
        curl -sfL --max-time 60 \
          "https://raw.githubusercontent.com/ERGO-Code/HiGHS/master/check/instances/${p}.mps" \
          -o "${LP_DIR}/${p}.mps" || { echo "FAIL: could not fetch ${p}.mps"; exit 1; }
    fi
done

echo
printf "%-10s | %-9s %-17s %6s | %-9s %-17s %6s | %-9s\n" \
       problem "cpu" "objective" "iters" "metal" "objective" "iters" "rel diff"
printf -- "-%.0s" {1..104}; echo

fails=0
for p in "${PROBLEMS[@]}"; do
    mps="${LP_DIR}/${p}.mps"
    # plc writes solution-sum.json into the working directory; keep that in out/.
    ( cd "${OUT_DIR}" && "${CUPDLP_DIR}/build-cpu/bin/plc" -fname "${mps}" -nIterLim 20000 \
        > "${OUT_DIR}/${p}.cpu.log" 2>&1 ) || true
    ( cd "${OUT_DIR}" && "${CUPDLP_DIR}/build/bin/plc" -fname "${mps}" -nIterLim 20000 \
        > "${OUT_DIR}/${p}.metal.log" 2>&1 ) || true
    read -r cs co ci < <(python3 "${SCRIPT_DIR}/parse_plc.py" "${OUT_DIR}/${p}.cpu.log")
    read -r gs go gi < <(python3 "${SCRIPT_DIR}/parse_plc.py" "${OUT_DIR}/${p}.metal.log")
    rel="$(python3 -c "
a,b='$co','$go'
try: print('%.1e' % (abs(float(a)-float(b))/max(abs(float(a)),1e-12)))
except Exception: print('n/a')")"
    verdict=ok
    # "Optimal current" vs "Optimal average" is which iterate PDLP accepted, not
    # a disagreement; compare the status class only.
    if [[ "$cs" != "$gs" ]]; then verdict="STATUS ${cs}!=${gs}"; fails=$((fails+1))
    elif [[ "$rel" == "n/a" ]]; then verdict="NO-RESULT"; fails=$((fails+1))
    elif (( $(python3 -c "print(1 if float('$rel') > 1e-3 else 0)") )); then
        verdict="OBJECTIVE"; fails=$((fails+1))
    fi
    printf "%-10s | %-9s %-17s %6s | %-9s %-17s %6s | %-9s %s\n" \
           "$p" "$cs" "$co" "$ci" "$gs" "$go" "$gi" "$rel" "$verdict"
done

echo
echo "=== Apple-GPU provenance ==="
( cd "${OUT_DIR}" && CUMETAL_TRACE_GPU=1 "${CUPDLP_DIR}/build/bin/plc" \
    -fname "${LP_DIR}/afiro.mps" -nIterLim 50 > "${OUT_DIR}/provenance.log" 2>&1 ) || true
launches="$(grep -c 'device=apple_gpu' "${OUT_DIR}/provenance.log" || true)"
if [[ "${launches}" -gt 0 ]]; then
    echo "  ${launches} kernel launches traced with device=apple_gpu"
    grep -m1 'device_name=' "${OUT_DIR}/provenance.log" | sed 's/^/  /' || true
else
    echo "  FAIL: no kernel launch traced on the Apple GPU"
    fails=$((fails+1))
fi
if grep -q 'source=approximate_stub' "${OUT_DIR}/provenance.log"; then
    echo "  FAIL: an approximate stub kernel participated"
    fails=$((fails+1))
fi

echo
if [[ ${fails} -eq 0 ]]; then
    echo "PASS: cuPDLP-C matches its CPU reference on ${#PROBLEMS[@]} LP(s), running on the Apple GPU"
else
    echo "FAIL: ${fails} problem(s) disagreed"
fi
echo "logs: ${OUT_DIR}"
exit $(( fails == 0 ? 0 : 1 ))
