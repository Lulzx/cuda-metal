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
shift 3
COMPILER_ARGS=("$@")
EXPECT_NATIVE=1
for arg in "${COMPILER_ARGS[@]}"; do
    if [[ "${arg}" == "--backend=legacy" ]]; then
        EXPECT_NATIVE=0
    fi
done

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
# A PATH entry containing spaces used to split cumetalc's shell assignment and
# prevent Clang from starting. Keep this ordinary desktop configuration in the
# primary installed-compiler gate.
PATH_WITH_SPACES="${WORK_DIR}/toolbox scripts"
mkdir -p "${PATH_WITH_SPACES}"
PATH="${PATH_WITH_SPACES}:${PATH}" \
  "${CUMETALC}" "${SOURCE_CU}" "${COMPILER_ARGS[@]}" -o "${OUT_BIN}" \
  >"${BUILD_LOG}" 2>&1 || BUILD_STATUS=$?

cat "${BUILD_LOG}"

if [[ ${BUILD_STATUS} -ne 0 ]]; then
    echo "FAIL: cumetalc exited ${BUILD_STATUS} building an executable from ${SOURCE_CU}"
    exit 1
fi
if grep -q -E "ptx[0-9]+.*is not a recognized feature for this target" "${BUILD_LOG}"; then
    echo "FAIL: the CUDA PTX feature leaked into the Apple host compilation"
    exit 1
fi
if [[ ! -x "${OUT_BIN}" ]]; then
    echo "FAIL: cumetalc reported success but produced no executable at ${OUT_BIN}"
    exit 1
fi
if [[ ${EXPECT_NATIVE} -eq 1 ]] &&
   nm -u "${OUT_BIN}" 2>/dev/null | grep -q '__cudaRegister'; then
    echo "FAIL: native source executable still depends on NVIDIA registration symbols"
    exit 1
fi

# The whole point of the driver is that the binary is self-contained: no metallib argument, no
# DYLD_LIBRARY_PATH from the caller. Run it exactly as a user would.
echo "Running ${OUT_BIN}"
RUN_STATUS=0
RUNTIME_CACHE="${WORK_DIR}/native-runtime-cache"
rm -rf "${RUNTIME_CACHE}"
RUN_OUTPUT="$(CUMETAL_CACHE_DIR="${RUNTIME_CACHE}" CUMETAL_TRACE_GPU=1 \
  CUMETAL_DEBUG_REGISTRATION=1 "${OUT_BIN}" 2>&1)" || RUN_STATUS=$?
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
if [[ ${EXPECT_NATIVE} -eq 1 ]] &&
   { grep -q -E 'JIT compiling|registration-jit' <<<"${RUN_OUTPUT}" ||
     [[ -d "${RUNTIME_CACHE}/registration-jit" ]]; }; then
    echo "FAIL: native source executable performed first-launch PTX JIT"
    exit 1
fi
if [[ ${EXPECT_NATIVE} -eq 1 && ! -d "${RUNTIME_CACHE}/native-aot" ]]; then
    echo "FAIL: native module was not materialized through the native-AOT cache"
    exit 1
fi

if [[ ${EXPECT_NATIVE} -eq 1 ]]; then
    echo "PASS: cumetalc built and ran a native-AOT CUDA executable on the Apple GPU without PTX JIT"
else
    echo "PASS: cumetalc built and ran an explicit legacy-compatibility CUDA executable on the Apple GPU"
fi
