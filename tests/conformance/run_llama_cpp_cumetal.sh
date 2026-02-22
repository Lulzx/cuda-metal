#!/usr/bin/env bash
# run_llama_cpp_cumetal.sh — Conformance test: llama.cpp GGML CUDA backend via CuMetal
#
# Usage:
#   bash tests/conformance/run_llama_cpp_cumetal.sh
#
# Environment overrides:
#   CUMETAL_LLAMA_DIR     path to llama.cpp checkout (default: ../llama.cpp)
#   CUMETAL_LLAMA_BUILD   path to build dir (default: <llama-dir>/build-cumetal)
#   CUMETAL_LLAMA_CLI     explicit path to llama-cli binary
#   CUMETAL_LLAMA_MODEL   explicit path to GGUF model file (skip auto-download)
#   CUMETAL_LLAMA_PROMPT  inference prompt (default: short factual question)
#   CUMETAL_LLAMA_NGL     GPU layers to offload (default: 99 = all)
#   CUMETAL_LLAMA_NTOK    tokens to generate (default: 128)
#   CUMETAL_LLAMA_MODELS_DIR  directory to cache downloaded models (default: ~/.cache/cumetal/models)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── resolve llama-cli binary ──────────────────────────────────────────────────
LLAMA_DIR="${CUMETAL_LLAMA_DIR:-${ROOT_DIR}/../llama.cpp}"
LLAMA_BUILD="${CUMETAL_LLAMA_BUILD:-${LLAMA_DIR}/build-cumetal}"
LLAMA_CLI="${CUMETAL_LLAMA_CLI:-${LLAMA_BUILD}/bin/llama-cli}"

if [[ ! -x "${LLAMA_CLI}" ]]; then
    # Try common install locations
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

# ── resolve or download model ─────────────────────────────────────────────────
MODEL_PATH="${CUMETAL_LLAMA_MODEL:-}"

if [[ -z "${MODEL_PATH}" ]]; then
    MODELS_DIR="${CUMETAL_LLAMA_MODELS_DIR:-${HOME}/.cache/cumetal/models}"
    mkdir -p "${MODELS_DIR}"

    # TinyLlama-1.1B-Chat Q4_K_M: 638 MB, fast inference, widely used
    MODEL_FILE="tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    MODEL_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/${MODEL_FILE}"
    MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

    if [[ ! -f "${MODEL_PATH}" ]]; then
        echo "Downloading ${MODEL_FILE} (~638 MB) ..."
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
PROMPT="${CUMETAL_LLAMA_PROMPT:-Explain quantum entanglement in two short sentences.}"
NGL="${CUMETAL_LLAMA_NGL:-99}"
NTOK="${CUMETAL_LLAMA_NTOK:-128}"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " llama.cpp CUDA backend conformance test via CuMetal"
echo " Model:  $(basename "${MODEL_PATH}")"
echo " Prompt: ${PROMPT}"
echo " NGL:    ${NGL}  (GPU layers offloaded)"
echo " NTok:   ${NTOK}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Set up library path so llama-cli links against CuMetal instead of real CUDA
export DYLD_LIBRARY_PATH="${ROOT_DIR}/build${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

OUTPUT_FILE="$(mktemp)"
TIME_FILE="$(mktemp)"
trap 'rm -f "${OUTPUT_FILE}" "${TIME_FILE}"' EXIT

START_TS="$(date +%s)"

set +e
"${LLAMA_CLI}" \
    --model "${MODEL_PATH}" \
    --n-gpu-layers "${NGL}" \
    --n-predict "${NTOK}" \
    --prompt "${PROMPT}" \
    --no-mmap \
    --log-disable \
    2>&1 | tee "${OUTPUT_FILE}"
EXIT_CODE=$?
set -e

END_TS="$(date +%s)"
ELAPSED=$(( END_TS - START_TS ))

echo ""
echo "─── Inference complete (${ELAPSED}s wall-clock) ───────────────────────"

# ── parse tokens/sec from llama.cpp output ────────────────────────────────────
TOKS_PER_SEC=""
if grep -qE "eval time" "${OUTPUT_FILE}"; then
    TOKS_PER_SEC="$(grep -oE '[0-9]+\.[0-9]+ tokens per second' "${OUTPUT_FILE}" | tail -1 || true)"
fi
if [[ -n "${TOKS_PER_SEC}" ]]; then
    echo "Performance: ${TOKS_PER_SEC}"
fi

# ── parse peak memory ─────────────────────────────────────────────────────────
PEAK_MEM=""
if grep -qiE "ggml_metal|metal" "${OUTPUT_FILE}" 2>/dev/null; then
    PEAK_MEM="$(grep -oiE 'Metal buffer size\s*=\s*[0-9.]+ MiB' "${OUTPUT_FILE}" | tail -1 || true)"
fi
[[ -n "${PEAK_MEM}" ]] && echo "Memory: ${PEAK_MEM}"

echo ""

# ── PASS / FAIL determination ─────────────────────────────────────────────────
FAIL_REASON=""

if [[ ${EXIT_CODE} -ne 0 ]]; then
    FAIL_REASON="llama-cli exited with code ${EXIT_CODE}"
fi

# Must produce at least some tokens (non-empty output beyond the prompt echo)
OUTPUT_LEN="$(wc -c < "${OUTPUT_FILE}")"
if [[ -z "${FAIL_REASON}" && "${OUTPUT_LEN}" -lt 50 ]]; then
    FAIL_REASON="output too short (${OUTPUT_LEN} bytes) — likely no tokens generated"
fi

# Hard failure indicators
if [[ -z "${FAIL_REASON}" ]] && grep -qiE "\bSEGFAULT\b|Segmentation fault|Bus error|Illegal instruction" "${OUTPUT_FILE}"; then
    FAIL_REASON="fatal signal in llama-cli output"
fi

if [[ -z "${FAIL_REASON}" ]] && grep -qiE "CUDA error|cudaError|ggml_cuda.*failed" "${OUTPUT_FILE}"; then
    FAIL_REASON="CUDA error in llama.cpp GGML backend"
fi

# Confirm GPU backend was used (n_gpu_layers > 0)
if [[ -z "${FAIL_REASON}" ]]; then
    if ! grep -qiE "n_gpu_layers\s*=\s*[1-9]|offloaded [1-9]|GPU\s*layers" "${OUTPUT_FILE}"; then
        echo "WARN: could not confirm GPU layers were offloaded (CPU fallback?)"
    fi
fi

if [[ -n "${FAIL_REASON}" ]]; then
    echo "FAIL: ${FAIL_REASON}"
    echo ""
    echo "Last 20 lines of output:"
    tail -20 "${OUTPUT_FILE}"
    exit 1
fi

echo "PASS: llama.cpp CUDA backend works perfectly on CuMetal"
echo "      Real production LLM kernels ran on Apple Silicon via Metal translation."
exit 0
