#!/usr/bin/env bash
# Typed-backend device launch queue gate: builds the RDC probe through the
# fatbin shim toolchain and runs it with CUMETAL_PTX_BACKEND=cumetal-ir so
# parent kernels enqueue child launches through the typed hidden queue
# channel (cudaGetParameterBuffer/cudaLaunchDevice -> queue records drained
# by the runtime). Mirrors run_device_launch_queue.sh, which covers the
# legacy AOT path.
set -euo pipefail

ROOT_DIR="${1:?usage: $0 <repo-root> <workdir>}"
WORK_DIR="${2:?usage: $0 <repo-root> <workdir>}"
mkdir -p "${WORK_DIR}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=tests/cuda_projects/_common.sh
source "${ROOT_DIR}/tests/cuda_projects/_common.sh"

if ! cumetal_cuda_projects_check_prereqs "${ROOT_DIR}"; then
    exit 77
fi

# Device launches need relocatable device code with the old offload driver.
export CUMETAL_CUDA_PROJECT_EXTRA_DEVICE_FLAGS="--no-offload-new-driver -fgpu-rdc"
OBJECT="${WORK_DIR}/device_launch_queue_probe.o"
EXECUTABLE="${WORK_DIR}/device_launch_queue_probe"
# The helper also links, which cannot succeed until the registration stub
# is renamed below; tolerate that expected link failure and require the
# object instead.
cumetal_cuda_projects_compile_link \
    "${ROOT_DIR}" \
    "${SCRIPT_DIR}/fixtures" \
    "${WORK_DIR}" \
    device_launch_queue_probe.cu \
    device_launch_queue_probe.bin || true
if [[ ! -f "${OBJECT}" ]]; then
    echo "FAIL: compiling device_launch_queue_probe.cu produced no object" >&2
    exit 1
fi

# The RDC object references its linked-binary registration stub by a
# per-module hashed name; the runtime exports the unhashed symbol.
LINKED_SYMBOL="$(nm -u "${OBJECT}" 2>/dev/null | sed -nE 's/.*(__cudaRegisterLinkedBinary__nv_[[:alnum:]_]+).*/_\1/p' | head -1 || true)"
if [[ -z "${LINKED_SYMBOL}" ]]; then
    echo "FAIL: RDC object has no linked-binary registration symbol" >&2
    exit 1
fi
OBJCOPY="$(dirname "${CLANG_BIN}")/llvm-objcopy"
"${OBJCOPY}" --redefine-sym "${LINKED_SYMBOL}=___cudaRegisterLinkedBinary" "${OBJECT}"
xcrun clang++ "${OBJECT}" \
    -L"${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}" \
    -lcumetal -Wl,-rpath,"${CUMETAL_BUILD_DIR:-${ROOT_DIR}/build}" \
    -o "${EXECUTABLE}"

combined=""
for mode in nested invalid overflow; do
    output="$(CUMETAL_PTX_BACKEND=cumetal-ir "${EXECUTABLE}" "${mode}" 2>&1)"
    printf '%s\n' "${output}"
    combined+="${output}"$'\n'
done

grep -Fq 'PASS: nested device launch queue value=111' <<<"${combined}"
grep -Fq 'PASS: invalid child configuration propagated' <<<"${combined}"
grep -Fq 'PASS: device launch queue overflow propagated' <<<"${combined}"

echo "PASS: typed nested, invalid, and overflow device-launch queue behavior"
