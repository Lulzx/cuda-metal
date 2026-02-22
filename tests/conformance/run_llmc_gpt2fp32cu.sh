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
  echo "INFO: llm.c conformance requires PTX lowering path (LLMC emulation disabled)"
else
  export CUMETAL_TRACE_LLMC_EMULATION=1
fi

(cd "$LLMC_DIR" && eval "$TEST_CMD") | tee "$OUTPUT_FILE"

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
