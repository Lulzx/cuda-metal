#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKER="${SCRIPT_DIR}/check_llama_cpp_output.sh"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

VALID="${TMP_DIR}/valid-invalid-utf8.log"
{
    printf '\377llama.cpp output with a torn UTF-8 byte sequence\n'
    printf 'The capital of France is Par'
    printf 'CUMETAL_PROVENANCE event=kernel_launch source=specialized_msl '
    printf 'provenance=workload_specialization semantic_quality=exact '
    printf 'device=apple_gpu launch_success=true\n'
    printf 'is.\n'
} > "${VALID}"
"${CHECKER}" "${VALID}" 0 1 Paris

expect_failure() {
    local name="$1"
    local expected="$2"
    shift 2
    local result
    if result="$("${CHECKER}" "$@" 2>&1)"; then
        echo "FAIL: ${name} unexpectedly passed" >&2
        exit 1
    fi
    if [[ "${result}" != *"${expected}"* ]]; then
        echo "FAIL: ${name} reported '${result}', expected '${expected}'" >&2
        exit 1
    fi
}

NO_PROVENANCE="${TMP_DIR}/no-provenance.log"
printf '%060d Paris\n' 0 > "${NO_PROVENANCE}"
expect_failure "missing provenance" "no successful Apple-GPU kernel provenance" \
    "${NO_PROVENANCE}" 0 1 Paris

FALLBACK="${TMP_DIR}/fallback.log"
cp "${VALID}" "${FALLBACK}"
printf 'CUMETAL_PROVENANCE source=cpu_fallback\n' >> "${FALLBACK}"
expect_failure "fallback provenance" "CPU fallback or stub provenance" \
    "${FALLBACK}" 0 1 Paris

expect_failure "incoherent generation" "incoherent output" \
    "${VALID}" 0 1 London

expect_failure "llama failure" "llama-cli exited with code 9" \
    "${VALID}" 9 1 Paris

FAKE_CLI="${TMP_DIR}/llama-cli"
cat > "${FAKE_CLI}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " != *" --simple-io "* ]]; then
    echo "fake llama-cli: --simple-io was not supplied" >&2
    exit 9
fi
printf '%060d\n' 0
printf 'CUMETAL_PROVENANCE event=kernel_launch source=specialized_msl '
printf 'provenance=workload_specialization semantic_quality=exact '
printf 'device=apple_gpu launch_success=true\n'
printf 'The capital of France is Paris.\n'
EOF
chmod +x "${FAKE_CLI}"
touch "${TMP_DIR}/model.gguf"

INTEGRATION_OUTPUT="${TMP_DIR}/integration.log"
CUMETAL_LLAMA_CLI="${FAKE_CLI}" \
CUMETAL_LLAMA_MODEL="${TMP_DIR}/model.gguf" \
CUMETAL_LLAMA_NGL=1 \
CUMETAL_LLAMA_NTOK=1 \
    bash "${SCRIPT_DIR}/run_llama_cpp_cumetal.sh" > "${INTEGRATION_OUTPUT}" 2>&1
if ! LC_ALL=C grep -qF \
    "PASS: llama.cpp produced correct output via CuMetal (NGL=1)." \
    "${INTEGRATION_OUTPUT}"; then
    echo "FAIL: full harness did not pass with capture-safe fake llama output" >&2
    cat "${INTEGRATION_OUTPUT}" >&2
    exit 1
fi

echo "PASS: llama.cpp output gate handles byte streams and rejects invalid runs"
