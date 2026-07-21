#!/usr/bin/env bash
set -euo pipefail

LLMC_DIR="${CUMETAL_LLMC_DIR:-}"
BUILD_CMD="${CUMETAL_LLMC_BUILD_CMD:-}"
TEST_CMD="${CUMETAL_LLMC_TEST_CMD:-}"
REQUIRE_NO_EMULATION="${CUMETAL_LLMC_REQUIRE_NO_EMULATION:-1}"

if [[ -z "$LLMC_DIR" ]]; then
  echo "SKIP: set CUMETAL_LLMC_DIR to llm.c checkout root"
  exit 77
fi

if [[ ! -d "$LLMC_DIR" ]]; then
  echo "SKIP: CUMETAL_LLMC_DIR does not exist: $LLMC_DIR"
  exit 77
fi

if [[ ! -f "$LLMC_DIR/gpt2_124M.bin" && ! -f "$LLMC_DIR/dev/data/gpt2_124M.bin" ]]; then
  echo "SKIP: llm.c checkpoint gpt2_124M.bin not found under $LLMC_DIR"
  echo "      Run from cumetal repo: bash scripts/fetch_llmc_assets.sh \"$LLMC_DIR\""
  exit 77
fi
if [[ ! -f "$LLMC_DIR/gpt2_124M_debug_state.bin" && ! -f "$LLMC_DIR/dev/data/gpt2_124M_debug_state.bin" ]]; then
  echo "SKIP: llm.c debug state gpt2_124M_debug_state.bin not found under $LLMC_DIR"
  echo "      Run: bash scripts/fetch_llmc_assets.sh (from cumetal repo)"
  exit 77
fi

# test_gpt2fp32cu links against libcumetal and needs the CUDA runtime
# registration symbols (__cudaRegisterFatBinary et al.), which are only present
# when the binary shim is enabled (default off in Release builds). Detect the
# missing-shim case and skip cleanly instead of failing the build/link opaquely.
CUMETAL_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CUMETAL_BUILD_DIR="${CUMETAL_BUILD_DIR:-${CUMETAL_ROOT_DIR}/build}"
CUMETAL_LIB="${CUMETAL_BUILD_DIR}/libcumetal.dylib"
if [[ -f "${CUMETAL_LIB}" ]] && command -v nm >/dev/null 2>&1; then
  # Capture into a variable (no pipe): `nm | grep -q` under `set -o pipefail`
  # can report failure when grep closes the pipe early (SIGPIPE on nm).
  CUMETAL_SYMS="$(nm -gU "${CUMETAL_LIB}" 2>/dev/null || true)"
  if [[ "${CUMETAL_SYMS}" != *cudaRegisterFatBinary* ]]; then
    echo "SKIP: libcumetal built without the binary shim (CUMETAL_ENABLE_BINARY_SHIM=OFF);"
    echo "      rebuild with -DCUMETAL_ENABLE_BINARY_SHIM=ON to run this conformance test"
    exit 77
  fi
fi

if [[ -n "$BUILD_CMD" ]]; then
  (cd "$LLMC_DIR" && eval "$BUILD_CMD")
fi

if [[ -z "$TEST_CMD" ]]; then
  if [[ -x "$LLMC_DIR/test_gpt2fp32cu" ]]; then
    TEST_CMD="./test_gpt2fp32cu"
  else
    echo "SKIP: set CUMETAL_LLMC_TEST_CMD or provide $LLMC_DIR/test_gpt2fp32cu"
    exit 77
  fi
fi

OUTPUT_FILE="$(mktemp)"
trap 'rm -f "$OUTPUT_FILE"' EXIT

if [[ "$REQUIRE_NO_EMULATION" == "1" ]]; then
  export CUMETAL_DISABLE_LLMC_EMULATION=1
  export CUMETAL_ENABLE_LLMC_CPU_EMULATION=0
  export CUMETAL_TRACE_GPU=1
  echo "INFO: llm.c conformance requires PTX lowering path (LLMC emulation disabled)"
else
  export CUMETAL_ENABLE_LLMC_CPU_EMULATION=1
  export CUMETAL_TRACE_LLMC_EMULATION=1
fi

(cd "$LLMC_DIR" && eval "$TEST_CMD") 2>&1 | tee "$OUTPUT_FILE" || true

if ! rg -qi "loss" "$OUTPUT_FILE"; then
  echo "FAIL: llm.c output did not contain a loss line"
  exit 1
fi

if rg -qi "\\b(fail|error|nan|inf)\\b" "$OUTPUT_FILE"; then
  echo "FAIL: llm.c output contains failure markers"
  exit 1
fi

if rg -q "TENSOR NOT OK" "$OUTPUT_FILE"; then
  echo "FAIL: llm.c reported gradient tensor mismatch"
  exit 1
fi

if rg -q "CUMETAL_LLMC_EMULATION" "$OUTPUT_FILE"; then
  if [[ "$REQUIRE_NO_EMULATION" == "1" ]]; then
    echo "FAIL: llm.c used runtime emulation fallback while no-emulation mode is required"
    exit 1
  fi
  echo "WARN: llm.c used runtime emulation fallback (not pure PTX->LLVM lowering)"
fi

if [[ "$REQUIRE_NO_EMULATION" == "1" ]]; then
  if ! rg -q 'CUMETAL_PROVENANCE .*source=(generic_ptx|specialized_msl|metallib) provenance=(generic_ptx_lowering|library_substitution|workload_specialization|precompiled_metallib) semantic_quality=(exact|tolerance_bounded|semantic_emulation|performance_degraded) device=apple_gpu .*launch_success=true' \
      "$OUTPUT_FILE"; then
    echo "FAIL: llm.c recorded no successful Apple-GPU kernel launch"
    exit 1
  fi
  if rg -q 'CUMETAL_PROVENANCE .*source=(cpu_fallback|stub)' "$OUTPUT_FILE"; then
    echo "FAIL: llm.c used a CPU fallback or stub"
    exit 1
  fi
fi

if rg -q "overall okay: 1" "$OUTPUT_FILE"; then
  echo "PASS: llm.c test_gpt2fp32cu reached numerical parity (overall okay: 1)"
  exit 0
fi

if rg -q "overall okay: 0" "$OUTPUT_FILE" || rg -q "MISMATCH" "$OUTPUT_FILE"; then
  echo "FAIL: llm.c numerical parity mismatch detected"
  exit 1
fi

echo "FAIL: llm.c output missing explicit parity status"
exit 1
