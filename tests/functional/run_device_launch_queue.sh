#!/usr/bin/env bash
set -euo pipefail

CUMETALC="${1:?usage: $0 <cumetalc> <source.cu> <workdir>}"
SOURCE="${2:?}"
WORK_DIR="${3:?}"
mkdir -p "${WORK_DIR}"
EXECUTABLE="${WORK_DIR}/device_launch_queue_probe"

"${CUMETALC}" "${SOURCE}" --backend=legacy -o "${EXECUTABLE}"

combined=""
for mode in nested invalid overflow; do
    output="$(CUMETAL_TRACE_GPU=1 "${EXECUTABLE}" "${mode}" 2>&1)"
    printf '%s\n' "${output}"
    combined+="${output}"$'\n'
done

grep -Fq 'PASS: nested device launch queue value=111' <<<"${combined}"
grep -Fq 'PASS: invalid child configuration propagated' <<<"${combined}"
grep -Fq 'PASS: device launch queue overflow propagated' <<<"${combined}"

gpu_launches="$(grep -c 'device=apple_gpu.*launch_success=true' <<<"${combined}")"
if (( gpu_launches < 5 )); then
    echo "FAIL: expected parent, child, leaf, invalid, and overflow Apple-GPU launches" >&2
    exit 1
fi

echo "PASS: nested, invalid, and overflow device-launch queue behavior"
