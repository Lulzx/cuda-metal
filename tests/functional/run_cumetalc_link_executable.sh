#!/usr/bin/env bash
# spec.md Phase 2/3 exit criterion: an unmodified CUDA source file compiles with cumetalc into a
# runnable executable and produces correct output on the Apple GPU.
#
# This gate exists because the project shipped for a long time with a cumetalc that could only
# emit a .metallib, which meant users had to hand-split every program into a .cu plus a host .cpp
# and pass a metallib path at runtime. Do not relax it into a skip: if the driver cannot build a
# working binary from samples/vectorAdd/vectorAdd.cu, the documented primary path is broken.
set -euo pipefail

CUMETALC="${1:?usage: run_cumetalc_link_executable.sh <cumetalc> <source.cu> <workdir>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"

if ! command -v xcrun >/dev/null 2>&1; then
    echo "SKIP: xcrun not installed"
    exit 77
fi

# The driver needs a CUDA-capable clang. Absence is a genuine environment gap, not a defect.
CLANG_BIN="${CUMETAL_CUDA_CLANG:-${CUMETAL_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}}"
if [[ ! -x "${CLANG_BIN}" ]]; then
    CLANG_BIN="$(command -v clang++ || true)"
fi
if [[ -z "${CLANG_BIN}" ]]; then
    echo "SKIP: no clang++ available to drive CUDA compilation"
    exit 77
fi

mkdir -p "${WORK_DIR}"
OUT_BIN="${WORK_DIR}/vectorAdd_linked"
rm -f "${OUT_BIN}"

echo "Building ${SOURCE_CU} -> ${OUT_BIN}"
BUILD_LOG="${WORK_DIR}/link_executable.build.log"
BUILD_STATUS=0
"${CUMETALC}" "${SOURCE_CU}" -o "${OUT_BIN}" >"${BUILD_LOG}" 2>&1 || BUILD_STATUS=$?

# Homebrew LLVM emits a benign "'+ptxNN' is not a recognized feature" line on toolchains new
# enough not to need the flag. Filter it from the printed log only, never from the exit status.
grep -v -E "is not a recognized feature for this target" "${BUILD_LOG}" || true

if [[ ${BUILD_STATUS} -ne 0 ]]; then
    echo "FAIL: cumetalc exited ${BUILD_STATUS} building an executable from ${SOURCE_CU}"
    exit 1
fi
if [[ ! -x "${OUT_BIN}" ]]; then
    echo "FAIL: cumetalc reported success but produced no executable at ${OUT_BIN}"
    exit 1
fi

# The whole point of the driver is that the binary is self-contained: no metallib argument, no
# DYLD_LIBRARY_PATH from the caller. Run it exactly as a user would.
echo "Running ${OUT_BIN}"
RUN_STATUS=0
RUN_OUTPUT="$(CUMETAL_TRACE_GPU=1 "${OUT_BIN}" 2>&1)" || RUN_STATUS=$?
echo "${RUN_OUTPUT}"

if [[ ${RUN_STATUS} -ne 0 ]]; then
    echo "FAIL: linked executable exited ${RUN_STATUS}"
    exit 1
fi
if ! grep -q "PASS:" <<<"${RUN_OUTPUT}"; then
    echo "FAIL: linked executable did not report a numerical PASS"
    exit 1
fi

# Correct output alone does not prove GPU execution -- a host fallback would also print PASS.
# Require positive provenance that the kernel dispatched to the Apple GPU.
if ! grep -q "device=apple_gpu" <<<"${RUN_OUTPUT}"; then
    echo "FAIL: no Apple GPU provenance; the kernel did not dispatch to the GPU"
    exit 1
fi
if ! grep -q "launch_success=true" <<<"${RUN_OUTPUT}"; then
    echo "FAIL: provenance does not report a successful launch"
    exit 1
fi

echo "PASS: cumetalc built and ran a linked CUDA executable on the Apple GPU"
