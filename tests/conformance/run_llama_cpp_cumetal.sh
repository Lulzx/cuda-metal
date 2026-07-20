#!/usr/bin/env bash
# run_llama_cpp_cumetal.sh — Conformance test: llama.cpp GGML CUDA backend via CuMetal
#
# Usage:
#   bash tests/conformance/run_llama_cpp_cumetal.sh
#
# Environment overrides:
#   CUMETAL_LLAMA_DIR        path to llama.cpp checkout (default: ../llama.cpp)
#   CUMETAL_LLAMA_BUILD      path to build dir (default: <llama-dir>/build-cumetal)
#   CUMETAL_LLAMA_CLI        explicit path to llama-cli binary
#   CUMETAL_LLAMA_MODEL      explicit path to GGUF model file (skip auto-download)
#   CUMETAL_LLAMA_PROMPT     inference prompt (default: short factual question)
#   CUMETAL_LLAMA_NGL        GPU layers to offload (default: 1 = verified smoke path)
#   CUMETAL_LLAMA_NTOK       tokens to generate (default: 16)
#   CUMETAL_LLAMA_MODELS_DIR directory to cache downloaded models (default: ~/.cache/cumetal/models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── resolve llama-cli binary ──────────────────────────────────────────────────
LLAMA_DIR="${CUMETAL_LLAMA_DIR:-${ROOT_DIR}/../llama.cpp}"
LLAMA_BUILD="${CUMETAL_LLAMA_BUILD:-${LLAMA_DIR}/build-cumetal}"
LLAMA_CLI="${CUMETAL_LLAMA_CLI:-${LLAMA_BUILD}/bin/llama-cli}"

if [[ ! -x "${LLAMA_CLI}" ]]; then
    for candidate in \
        "${LLAMA_BUILD}/bin/llama-cli" \
        "${LLAMA_DIR}/build/bin/llama-cli" \
        "$(command -v llama-cli 2>/dev/null || true)"
    do
        [[ -x "${candidate}" ]] && { LLAMA_CLI="${candidate}"; break; }
    done
fi

if [[ ! -x "${LLAMA_CLI}" ]]; then
    echo "SKIP: llama-cli not found"
    echo "  Build with: bash ${ROOT_DIR}/scripts/build_llama_cpp_cumetal.sh"
    echo "  Or set CUMETAL_LLAMA_CLI to an existing llama-cli binary"
    exit 77
fi

echo "llama-cli: ${LLAMA_CLI}"

# llama-cli is linked against libcumetal and resolves the CUDA runtime
# registration symbols (__cudaRegisterFatBinary et al.) at load time. Those
# only exist when libcumetal is built with the binary shim (the default for
# non-Release builds). Without them dyld aborts with a bare "Symbol not found".
# Detect the missing-shim case up front and skip with a clear message.
CUMETAL_LIB="${ROOT_DIR}/build/libcumetal.dylib"
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

# ── resolve or download model ─────────────────────────────────────────────────
MODEL_PATH="${CUMETAL_LLAMA_MODEL:-}"

if [[ -z "${MODEL_PATH}" ]]; then
    MODELS_DIR="${CUMETAL_LLAMA_MODELS_DIR:-${HOME}/.cache/cumetal/models}"
    mkdir -p "${MODELS_DIR}"

    # SmolLM2-135M-Instruct Q4_K_M: ~105 MB, fast, ideal for conformance
    MODEL_FILE="SmolLM2-135M-Instruct-Q4_K_M.gguf"
    MODEL_URL="https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/${MODEL_FILE}"
    MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "Downloading ${MODEL_FILE} (~105 MB) ..."
        if command -v curl >/dev/null 2>&1; then
            curl -L --progress-bar -o "${MODEL_PATH}.tmp" "${MODEL_URL}"
        elif command -v wget >/dev/null 2>&1; then
            wget -q --show-progress -O "${MODEL_PATH}.tmp" "${MODEL_URL}"
        else
            echo "SKIP: neither curl nor wget available for model download"
            exit 77
        fi
        mv "${MODEL_PATH}.tmp" "${MODEL_PATH}"
        echo "Model saved: ${MODEL_PATH}"
    else
        echo "Model cached: ${MODEL_PATH}"
    fi
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
    echo "SKIP: model file not found: ${MODEL_PATH}"
    exit 77
fi

# ── inference parameters ──────────────────────────────────────────────────────
# Default to a deterministic factual probe (greedy decode) so the test can verify
# the output is actually CORRECT, not merely non-empty. A tiny instruct model
# decodes "The capital of France is" → "Paris" under greedy sampling (verified
# against stock CPU llama.cpp). CUMETAL_LLAMA_EXPECT is the substring that must
# appear in the generation for the run to be considered numerically correct.
PROMPT="${CUMETAL_LLAMA_PROMPT:-The capital of France is}"
EXPECT="${CUMETAL_LLAMA_EXPECT:-Paris}"
NGL="${CUMETAL_LLAMA_NGL:-1}"
NTOK="${CUMETAL_LLAMA_NTOK:-16}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " llama.cpp CUDA backend conformance test via CuMetal"
echo " Model:  $(basename "${MODEL_PATH}")"
echo " Prompt: ${PROMPT}"
echo " Expect: output should contain '${EXPECT}' (greedy)"
echo " NGL:    ${NGL}  (GPU layers offloaded)"
echo " NTok:   ${NTOK}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Set up library path so llama-cli resolves libcumetal instead of real CUDA
export DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"
export CUMETAL_TRACE_GPU=1

OUTPUT_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_FILE}"' EXIT

START_TS="$(date +%s)"

set +e
"${LLAMA_CLI}" \
    --model "${MODEL_PATH}" \
    --n-gpu-layers "${NGL}" \
    --n-predict "${NTOK}" \
    --prompt "${PROMPT}" \
    --temp 0 \
    --seed 1 \
    --no-mmap \
    --log-disable \
    --single-turn \
    < /dev/null \
    2>&1 | tee "${OUTPUT_FILE}"
EXIT_CODE=$?
set -e

END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))

echo ""
echo "─── Inference complete (${ELAPSED}s wall-clock) ───────────────────────"

# Parse tokens/sec from llama.cpp output
TOKS_PER_SEC=""
if grep -qE "eval time|tok/s" "${OUTPUT_FILE}" 2>/dev/null; then
    TOKS_PER_SEC="$(grep -oE '[0-9]+\.[0-9]+ tokens per second' "${OUTPUT_FILE}" | tail -1 || true)"
fi
[[ -n "${TOKS_PER_SEC}" ]] && echo "Performance: ${TOKS_PER_SEC}"

echo ""

# ── PASS / FAIL ───────────────────────────────────────────────────────────────
FAIL_REASON=""

[[ ${EXIT_CODE} -ne 0 ]] && FAIL_REASON="llama-cli exited with code ${EXIT_CODE}"

OUTPUT_LEN="$(wc -c < "${OUTPUT_FILE}")"
if [[ -z "${FAIL_REASON}" && "${OUTPUT_LEN}" -lt 50 ]]; then
    FAIL_REASON="output too short (${OUTPUT_LEN} bytes) — no tokens generated"
fi

if [[ -z "${FAIL_REASON}" ]] && grep -qiE "Segmentation fault|Bus error|Illegal instruction" "${OUTPUT_FILE}" 2>/dev/null; then
    FAIL_REASON="fatal signal in llama-cli output"
fi

if [[ -z "${FAIL_REASON}" ]] && grep -qiE "CUDA error|cudaError|ggml_cuda.*failed" "${OUTPUT_FILE}" 2>/dev/null; then
    FAIL_REASON="CUDA error reported by GGML backend"
fi

if [[ -z "${FAIL_REASON}" && "${NGL}" -gt 0 ]]; then
    if ! grep -qE 'CUMETAL_PROVENANCE .*source=(generic_ptx|specialized_msl|metallib) device=apple_gpu .*launch_success=true' \
        "${OUTPUT_FILE}" 2>/dev/null; then
        FAIL_REASON="no successful Apple-GPU kernel provenance was recorded"
    elif grep -qE 'CUMETAL_PROVENANCE .*source=(cpu_fallback|stub)' \
        "${OUTPUT_FILE}" 2>/dev/null; then
        FAIL_REASON="CPU fallback or stub provenance was recorded during GPU inference"
    fi
fi

# ── Coherence gate: output must be CORRECT, not merely non-empty ───────────────
# A greedy decode of a factual prompt must contain the expected answer. This is
# the check that separates a genuinely working translation from one that runs to
# completion but emits garbage tokens — the failure mode a "some bytes were
# generated" check silently passes. Set CUMETAL_LLAMA_EXPECT="" to opt out (e.g.
# for a custom prompt with no fixed answer); opting out is explicit, never silent.
if [[ -z "${FAIL_REASON}" && -n "${EXPECT}" ]]; then
    if ! grep -qiF "${EXPECT}" "${OUTPUT_FILE}" 2>/dev/null; then
        FAIL_REASON="incoherent output — generation did not contain '${EXPECT}'. \
CuMetal ran llama.cpp to completion but produced text that does not match a \
correct (stock CPU) run: numerically wrong output, not a crash. See \
docs/known-gaps.md for status."
    fi
fi

if [[ -n "${FAIL_REASON}" ]]; then
    echo "FAIL: ${FAIL_REASON}"
    echo ""
    echo "Last 20 lines of output:"
    tail -20 "${OUTPUT_FILE}"
    exit 1
fi

echo "PASS: llama.cpp produced correct output via CuMetal (NGL=${NGL})."
echo "      Greedy decode of \"${PROMPT}\" contained the expected answer '${EXPECT}'."
if [[ "${NGL}" -eq 0 ]]; then
    echo "      NGL=0: GGML compute ran on CPU; this exercises the binary shim, CUDA"
    echo "      device init, and fatbin registration end-to-end under a real app."
else
    echo "      NGL=${NGL}: GGML CUDA compute kernels were translated to Metal and ran"
    echo "      on the Apple GPU for the offloaded layers."
fi
exit 0
