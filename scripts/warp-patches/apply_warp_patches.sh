#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WARP_REPO="${1:-${CUMETAL_ROOT}/../warp-cumetal}"
EXPECTED_COMMIT="e6c3ba2d54bb048115760b5cd7a4bb2573329ae7"

if [[ ! -d "${WARP_REPO}/.git" ]]; then
    echo "error: Warp checkout not found at ${WARP_REPO}" >&2
    echo "       scripts/build_warp_cumetal.sh clones one for you." >&2
    exit 1
fi

# The patches are diffs against v1.12.0 exactly. A checkout that has already
# been patched still reports the same base commit, so this stays true across
# repeated runs; a checkout moved to another revision is rejected rather than
# fuzzed into.
actual_commit="$(git -C "${WARP_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "error: expected Warp ${EXPECTED_COMMIT} (v1.12.0), found ${actual_commit}" >&2
    exit 1
fi

patch_marker_is_present() {
    case "$(basename "$1")" in
        0001-cumetal-macos-build.patch)
            # One marker per file the patch touches, so a checkout carrying an
            # older revision of this patch is re-patched rather than reported as
            # already done.
            grep -q 'defined(__clang__) && !defined(WP_CUMETAL)' \
                "${WARP_REPO}/warp/native/crt.h" &&
                grep -q 'cumetal_toolkit' "${WARP_REPO}/warp/_src/build_dll.py" &&
                grep -q 'libcumetal.dylib' "${WARP_REPO}/warp/native/cuda_util.cpp" &&
                grep -q 'define-macro=WP_CUMETAL=1' "${WARP_REPO}/warp/native/warp.cu" &&
                grep -q 'CuMetal, which lowers CUDA source' "${WARP_REPO}/warp/_src/context.py"
            ;;
        0002-volume-builder-include-new.patch)
            grep -q '#include <new>' "${WARP_REPO}/warp/native/volume_builder.cu"
            ;;
        *)
            return 1
            ;;
    esac
}

for patch in "${SCRIPT_DIR}"/*.patch; do
    if patch_marker_is_present "${patch}"; then
        echo "already applied: $(basename "${patch}")"
    elif git -C "${WARP_REPO}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "already applied: $(basename "${patch}")"
    elif git -C "${WARP_REPO}" apply --check "${patch}"; then
        git -C "${WARP_REPO}" apply "${patch}"
        echo "applied: $(basename "${patch}")"
    else
        echo "error: cannot apply $(basename "${patch}") cleanly" >&2
        exit 1
    fi
done
