#!/usr/bin/env bash
# CuMetal Apollo demo — one command that proves CUDA ran on Apple Silicon.
#
# Usage:
#   bash demos/apollo/run.sh              # core ladder (~1–2 min)
#   bash demos/apollo/run.sh --full       # + llm.c GPT-2 FP32 (~several min)
#   bash demos/apollo/run.sh --quick      # vectorAdd + raytracer only
#   bash demos/apollo/run.sh --build-dir=path/to/build
#
# Exit 0 only if every required stage passes with Apple-GPU provenance.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUT_DIR="${CUMETAL_APOLLO_OUT:-${SCRIPT_DIR}/out}"
MODE="core" # core | quick | full
BUILD_DIR=""

for arg in "$@"; do
  case "$arg" in
    --quick) MODE="quick" ;;
    --full) MODE="full" ;;
    --build-dir=*) BUILD_DIR="${arg#--build-dir=}" ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "unknown arg: $arg" >&2
      exit 2
      ;;
  esac
done

# ── resolve build tree ───────────────────────────────────────────────────────
if [[ -z "${BUILD_DIR}" ]]; then
  for candidate in \
      "${CUMETAL_BUILD_DIR:-}" \
      "${ROOT_DIR}/build-release" \
      "${ROOT_DIR}/build" \
      "${ROOT_DIR}/build-nosshim" \
      "${ROOT_DIR}/build-noshim"
  do
    [[ -n "${candidate}" && -x "${candidate}/cumetalc" && -f "${candidate}/libcumetal.dylib" ]] \
      && { BUILD_DIR="${candidate}"; break; }
  done
fi

if [[ -z "${BUILD_DIR}" || ! -x "${BUILD_DIR}/cumetalc" ]]; then
  cat >&2 <<EOF
ERROR: no CuMetal build found (need cumetalc + libcumetal.dylib).

  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j"\$(sysctl -n hw.ncpu)"

Then re-run: bash demos/apollo/run.sh --build-dir=build
EOF
  exit 2
fi

CUMETALC="${BUILD_DIR}/cumetalc"
export DYLD_LIBRARY_PATH="${BUILD_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export PATH="${BUILD_DIR}/cuda_toolchain:${ROOT_DIR}/scripts/cuda_toolchain:${PATH}"
export CUMETAL_TRACE_GPU=1

mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/report.txt"
PROV_LOG="${OUT_DIR}/provenance.log"
: >"${REPORT}"
: >"${PROV_LOG}"

pass=0
fail=0
skip=0
declare -a RESULTS=()

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

log() {
  printf '%s\n' "$*" | tee -a "${REPORT}"
}

banner() {
  log ""
  log "════════════════════════════════════════════════════════════"
  log " $*"
  log "════════════════════════════════════════════════════════════"
}

# Compile + run a .cu, require PASS/OK and apple_gpu provenance.
# Args: stage_id title source_rel expect_regex [extra env assignments as KEY=VAL...]
run_cu_stage() {
  local id="$1" title="$2" src_rel="$3" expect_re="$4"
  shift 4 || true
  local bin="${OUT_DIR}/bin_${id}"
  local logf="${OUT_DIR}/${id}.log"
  local src="${ROOT_DIR}/${src_rel}"

  banner "Stage ${id}: ${title}"
  log "source: ${src_rel}"
  log "time:   $(ts)"

  if [[ ! -f "${src}" ]]; then
    log "SKIP: missing source ${src}"
    RESULTS+=("SKIP  ${id}  ${title}")
    skip=$((skip + 1))
    return 0
  fi

  # optional env for this stage (e.g. CUMETAL_WRITE_PPM=...)
  # Avoid empty-array expansion under `set -u` (macOS /bin/bash 3.2).
  set +e
  if [[ "$#" -gt 0 ]]; then
    env "$@" "${CUMETALC}" "${src}" -o "${bin}" >"${logf}.compile" 2>&1
  else
    "${CUMETALC}" "${src}" -o "${bin}" >"${logf}.compile" 2>&1
  fi
  local rc_c=$?
  set -e
  cat "${logf}.compile" >>"${logf}"
  if [[ ${rc_c} -ne 0 ]]; then
    log "FAIL: cumetalc compile (exit ${rc_c}) — see ${logf}"
    tail -20 "${logf}.compile" | tee -a "${REPORT}" || true
    RESULTS+=("FAIL  ${id}  ${title}  (compile)")
    fail=$((fail + 1))
    return 0
  fi

  set +e
  if [[ "$#" -gt 0 ]]; then
    env "$@" "${bin}" >"${logf}.run" 2>&1
  else
    "${bin}" >"${logf}.run" 2>&1
  fi
  local rc_r=$?
  set -e
  cat "${logf}.run" >>"${logf}"
  # keep every provenance line
  grep 'CUMETAL_PROVENANCE' "${logf}.run" >>"${PROV_LOG}" || true

  local ok=1
  if [[ ${rc_r} -ne 0 ]]; then
    ok=0
    log "FAIL: binary exit ${rc_r}"
  fi
  if ! grep -Eq "${expect_re}" "${logf}.run"; then
    ok=0
    log "FAIL: output did not match /${expect_re}/"
  fi
  if ! grep -q 'device=apple_gpu' "${logf}.run"; then
    ok=0
    log "FAIL: missing device=apple_gpu provenance (CPU result is not a pass)"
  fi
  if ! grep -q 'launch_success=true' "${logf}.run"; then
    ok=0
    log "FAIL: missing launch_success=true"
  fi

  if [[ ${ok} -eq 1 ]]; then
    log "PASS"
    # show one provenance line for the report
    grep 'CUMETAL_PROVENANCE' "${logf}.run" | head -1 | tee -a "${REPORT}" || true
    # show the human PASS/OK line
    grep -E "${expect_re}" "${logf}.run" | head -3 | tee -a "${REPORT}" || true
    RESULTS+=("PASS  ${id}  ${title}")
    pass=$((pass + 1))
  else
    log "---- last 25 lines ----"
    tail -25 "${logf}.run" | tee -a "${REPORT}" || true
    RESULTS+=("FAIL  ${id}  ${title}")
    fail=$((fail + 1))
  fi
}

run_llmc_stage() {
  local id="4" title="llm.c GPT-2 FP32 (forward + backward + AdamW)"
  banner "Stage ${id}: ${title}"
  local llmc_dir="${CUMETAL_LLMC_DIR:-${ROOT_DIR}/../llm.c}"
  local bin="${CUMETAL_LLMC_BIN:-${llmc_dir}/test_gpt2fp32cu}"
  local logf="${OUT_DIR}/llmc.log"

  log "llmc_dir: ${llmc_dir}"
  log "binary:   ${bin}"
  log "time:     $(ts)"

  if [[ ! -x "${bin}" ]]; then
    log "SKIP: no llm.c binary at ${bin}"
    log "      build with: bash scripts/build_llmc_test_gpt2fp32cu.sh ${llmc_dir}"
    RESULTS+=("SKIP  ${id}  ${title}")
    skip=$((skip + 1))
    return 0
  fi
  if [[ ! -f "${llmc_dir}/gpt2_124M.bin" ]]; then
    log "SKIP: missing ${llmc_dir}/gpt2_124M.bin (and debug state assets)"
    RESULTS+=("SKIP  ${id}  ${title}")
    skip=$((skip + 1))
    return 0
  fi

  set +e
  (
    cd "${llmc_dir}"
    # Workload specializations are required for the verified llm.c path.
    export CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="${CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS:-1}"
    export CUMETAL_TRACE_GPU=1
    export DYLD_LIBRARY_PATH="${BUILD_DIR}${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
    "${bin}"
  ) >"${logf}" 2>&1
  local rc=$?
  set -e

  grep 'CUMETAL_PROVENANCE' "${logf}" >>"${PROV_LOG}" || true
  local ok=1
  if [[ ${rc} -ne 0 ]]; then ok=0; log "FAIL: exit ${rc}"; fi
  if ! grep -q 'overall okay: 1' "${logf}"; then
    ok=0
    log "FAIL: missing overall okay: 1"
  fi
  if ! grep -q 'device=apple_gpu' "${logf}"; then
    ok=0
    log "FAIL: missing device=apple_gpu provenance"
  fi
  if grep -q 'cpu_fallback\|CUMETAL_ENABLE_LLMC_CPU_EMULATION' "${logf}" \
      && grep -q 'device=cpu\|source=cpu' "${logf}"; then
    ok=0
    log "FAIL: CPU fallback detected"
  fi

  if [[ ${ok} -eq 1 ]]; then
    log "PASS"
    grep -E 'OK \(LOGITS\)|LOSS OK|TENSOR OK|overall okay|Device 0:' "${logf}" | tee -a "${REPORT}" || true
    grep 'CUMETAL_PROVENANCE' "${logf}" | head -1 | tee -a "${REPORT}" || true
    local nprov
    nprov="$(grep -c 'CUMETAL_PROVENANCE' "${logf}" || true)"
    log "provenance dispatches: ${nprov}"
    RESULTS+=("PASS  ${id}  ${title}")
    pass=$((pass + 1))
  else
    tail -40 "${logf}" | tee -a "${REPORT}" || true
    RESULTS+=("FAIL  ${id}  ${title}")
    fail=$((fail + 1))
  fi
}

# ── header ───────────────────────────────────────────────────────────────────
banner "CuMetal Apollo demo"
log "root:      ${ROOT_DIR}"
log "build:     ${BUILD_DIR}"
log "out:       ${OUT_DIR}"
log "mode:      ${MODE}"
log "host:      $(uname -m)  $(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
log "cpu:       $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
log "started:   $(ts)"
log ""
log "Rules:"
log "  • every stage must print device=apple_gpu and launch_success=true"
log "  • a correct number without GPU provenance is NOT a pass"
log "  • this is a covered-path demo, not a claim of full CUDA"

# ── stages ───────────────────────────────────────────────────────────────────
run_cu_stage "1" "vectorAdd (hello CUDA on Metal)" \
  "samples/vectorAdd/vectorAdd.cu" \
  "PASS: samples/vectorAdd"

if [[ "${MODE}" != "quick" ]]; then
  run_cu_stage "2a" "parallel reduction (1M elements)" \
    "tests/cuda_projects/reduction/reduction_standalone.cu" \
    "PASS: parallel reduction"

  run_cu_stage "2b" "matrix transpose (naive + shared mem)" \
    "tests/cuda_projects/transpose/transpose_standalone.cu" \
    "PASS: transpose_"

  run_cu_stage "2c" "softmax (block + warp)" \
    "tests/cuda_projects/softmax/softmax.cu" \
    "ALL PASS: softmax_cuda"

  run_cu_stage "2d" "SGEMM naive (siboehm kernel)" \
    "tests/cuda_projects/sgemm/sgemm_naive.cu" \
    "PASS: sgemm_naive"
fi

PPM_PATH="${OUT_DIR}/rtiow.ppm"
run_cu_stage "3" "Ray Tracing in One Weekend (GPU == CPU reference)" \
  "tests/cuda_projects/raytracer/rtiow.cu" \
  "OK: GPU render matches CPU reference" \
  "CUMETAL_WRITE_PPM=${PPM_PATH}"

if [[ "${MODE}" == "full" ]]; then
  run_llmc_stage
fi

# ── summary ──────────────────────────────────────────────────────────────────
banner "Apollo result"
log "finished: $(ts)"
log ""
for line in "${RESULTS[@]}"; do
  log "  ${line}"
done
log ""
log "PASS=${pass}  FAIL=${fail}  SKIP=${skip}"
log ""
log "Artifacts:"
log "  report:      ${REPORT}"
log "  provenance:  ${PROV_LOG}"
if [[ -f "${PPM_PATH}" ]]; then
  log "  ray image:   ${PPM_PATH}  (open with Preview / feh / convert)"
fi
log ""

if [[ ${fail} -gt 0 ]]; then
  log "APOLLO FAILED — see logs under ${OUT_DIR}/"
  exit 1
fi

if [[ ${pass} -eq 0 ]]; then
  log "APOLLO FAILED — nothing ran"
  exit 1
fi

log "APOLLO PASSED"
log ""
log "One-liner for strangers:"
log "  CUDA source → CuMetal → Metal on $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')"
log "  with numerical checks and device=apple_gpu provenance on every stage."
exit 0
