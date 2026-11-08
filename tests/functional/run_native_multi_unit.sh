#!/usr/bin/env bash
# Separate native compilation (`nvcc -c` per unit, then link) of two units that
# both contain kernels, with same-named anonymous-namespace and static kernels,
# plus a kernel-free main. Each object used to export the same embedded
# metallib symbols, so any program with kernels in two units failed to link;
# internal-linkage kernels could not be registered at all.
set -euo pipefail

CUMETALC="${1:?usage: run_native_multi_unit.sh <cumetalc> <fixtures> <workdir>}"
FIXTURES="${2:?}"
WORK_DIR="${3:?}"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
    exit 77
fi
rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"
export CUMETAL_NVCC_COMPAT=1
for unit in unit_a unit_b main; do
    "${CUMETALC}" -c "${FIXTURES}/${unit}.cu" -o "${WORK_DIR}/${unit}.o"
done
"${CUMETALC}" "${WORK_DIR}/unit_a.o" "${WORK_DIR}/unit_b.o" "${WORK_DIR}/main.o" \
    -o "${WORK_DIR}/multi_unit"
unset CUMETAL_NVCC_COMPAT
OUTPUT="$(CUMETAL_TRACE_GPU=1 CUMETAL_CACHE_DIR="${WORK_DIR}/cache" "${WORK_DIR}/multi_unit" 2>&1)" || {
    echo "${OUTPUT}"
    exit 1
}
echo "${OUTPUT}" | grep -E '^(PASS|FAIL)'
if grep -q 'JIT compiling' <<<"${OUTPUT}"; then
    echo "FAIL: the native build fell back to the PTX JIT"
    exit 1
fi
LAUNCHES="$(grep -c 'device=apple_gpu.*launch_success=true' <<<"${OUTPUT}" || true)"
if [[ "${LAUNCHES}" != 6 ]]; then
    echo "FAIL: expected 6 successful Apple GPU launches, saw ${LAUNCHES}"
    exit 1
fi
grep -q '^PASS: native units with same-named internal kernels' <<<"${OUTPUT}"
