#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUMETAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PHYSX_REPO="${1:-${CUMETAL_ROOT}/../PhysX}"
EXPECTED_COMMIT="5ca9f472105a90d70d957c243cb0ef36fe251a9f"

if [[ ! -d "${PHYSX_REPO}/.git" ]]; then
    echo "error: PhysX checkout not found at ${PHYSX_REPO}" >&2
    exit 1
fi

actual_commit="$(git -C "${PHYSX_REPO}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
    echo "error: expected PhysX ${EXPECTED_COMMIT}, found ${actual_commit}" >&2
    exit 1
fi

for patch in "${SCRIPT_DIR}"/*.patch; do
    if git -C "${PHYSX_REPO}" apply --reverse --check "${patch}" >/dev/null 2>&1; then
        echo "already applied: $(basename "${patch}")"
    elif git -C "${PHYSX_REPO}" apply --check "${patch}"; then
        git -C "${PHYSX_REPO}" apply "${patch}"
        echo "applied: $(basename "${patch}")"
    else
        echo "error: cannot apply $(basename "${patch}") cleanly" >&2
        exit 1
    fi
done
