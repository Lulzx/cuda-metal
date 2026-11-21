#!/usr/bin/env bash
# Native-AOT device printf of string literals (constant memory) through `%s`,
# with output on stdout as CUDA delivers it.
set -euo pipefail

CUMETALC="${1:?usage: run_cumetalc_native_printf_literals.sh <cumetalc> <source.cu> <workdir>}"
SOURCE_CU="${2:?}"
WORK_DIR="${3:?}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_cumetalc_link_executable.sh" \
    "${CUMETALC}" "${SOURCE_CU}" "${WORK_DIR}" --expect-output=HOST_DONE >/dev/null

STDOUT_ONLY="$("${WORK_DIR}/vectorAdd_linked" 2>/dev/null)"
printf '%s\n' "${STDOUT_ONLY}"
for expected in 'LITERAL i=1 s=odd lit=literal pct=%' 'LITERAL i=0 s=even lit=literal pct=%'; do
    if [[ "$(grep -Fxc "${expected}" <<<"${STDOUT_ONLY}")" -ne 1 ]]; then
        echo "FAIL: missing stdout record: ${expected}"
        exit 1
    fi
done
# `%p` keeps the pointer value; only `%s` arguments become string handles.
if ! grep -Eq '^ADDRESS p=0x[0-9a-f]+$' <<<"${STDOUT_ONLY}" ||
   grep -Eq '^ADDRESS p=0x7c5d' <<<"${STDOUT_ONLY}"; then
    echo "FAIL: %p of a string literal did not print a plain address"
    exit 1
fi
echo "PASS: native-AOT printf materialized string literals on stdout"
