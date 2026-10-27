#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <cumetalc> <runner> <source.ptx> <output.metal>" >&2
    exit 2
fi

cumetalc="$1"
runner="$2"
source_file="$3"
output="$4"

"${cumetalc}" "${source_file}" --backend=cumetal-ir --emit=msl \
    --overwrite -o "${output}"
"${runner}" "${output}"
