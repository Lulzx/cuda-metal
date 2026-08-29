#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <cumetalc> <constant-symbol.cu> <output>" >&2
    exit 2
fi

cumetalc="$1"
source_file="$2"
output="$3"
log="${output}.log"
rm -f "${output}" "${log}"

if "${cumetalc}" "${source_file}" --emit=exe --overwrite -o "${output}" \
    >"${log}" 2>&1; then
    echo "FAIL: native AOT accepted runtime-populated CUDA globals" >&2
    exit 1
fi
if ! grep -q "native AOT does not yet describe runtime-populated" "${log}"; then
    cat "${log}" >&2
    echo "FAIL: native AOT global rejection was not explicit" >&2
    exit 1
fi
if [[ -e "${output}" ]]; then
    echo "FAIL: rejected native AOT compilation left an executable" >&2
    exit 1
fi

echo "PASS: unsupported native-AOT CUDA globals fail explicitly"
