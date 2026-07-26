#!/usr/bin/env bash
# The release version lives in two places: project(cumetal VERSION ...) in CMakeLists.txt and
# the CUMETAL_VERSION_* macros in runtime/api/cumetal_native.h. The header duplicates it so it
# stays usable standalone, which means the two can drift silently -- a released dylib reporting
# the wrong version is the kind of thing nobody notices until it matters.
#
# This test pins them together, and also checks that the built cumetalc and libcumetal agree.
set -euo pipefail

CMAKE_VERSION="${1:?usage: run_version_matches_test.sh <cmake-project-version> <cumetalc> <native-header>}"
CUMETALC="${2:?}"
NATIVE_HEADER="${3:?}"

fail=0

read_macro() {
    # Matches: #define CUMETAL_VERSION_MAJOR 1
    # POSIX BRE only -- BSD sed on macOS has no \+ quantifier, and using one here silently
    # matched nothing rather than erroring.
    sed -n "s/^#define $1[[:space:]][[:space:]]*\([0-9][0-9]*\).*/\1/p" "${NATIVE_HEADER}" | head -1
}

MAJOR="$(read_macro CUMETAL_VERSION_MAJOR)"
MINOR="$(read_macro CUMETAL_VERSION_MINOR)"
PATCH="$(read_macro CUMETAL_VERSION_PATCH)"

if [[ -z "${MAJOR}" || -z "${MINOR}" || -z "${PATCH}" ]]; then
    echo "FAIL: could not parse CUMETAL_VERSION_{MAJOR,MINOR,PATCH} from ${NATIVE_HEADER}"
    exit 1
fi

HEADER_VERSION="${MAJOR}.${MINOR}.${PATCH}"
if [[ "${HEADER_VERSION}" != "${CMAKE_VERSION}" ]]; then
    echo "FAIL: version drift between build system and header"
    echo "      CMakeLists.txt project(VERSION): ${CMAKE_VERSION}"
    echo "      cumetal_native.h macros:         ${HEADER_VERSION}"
    fail=1
fi

# CUMETAL_VERSION_STRING is written out separately; make sure it agrees with the numbers.
STRING_VERSION="$(sed -n \
    's/^#define CUMETAL_VERSION_STRING[[:space:]][[:space:]]*"\([^"]*\)".*/\1/p' \
    "${NATIVE_HEADER}" | head -1)"
if [[ "${STRING_VERSION}" != "${HEADER_VERSION}" ]]; then
    echo "FAIL: CUMETAL_VERSION_STRING is \"${STRING_VERSION}\" but the macros say ${HEADER_VERSION}"
    fail=1
fi

# cumetalc takes its version from CMake at compile time, so this catches a stale binary too.
CUMETALC_VERSION="$("${CUMETALC}" --version | awk '{print $2}')"
if [[ "${CUMETALC_VERSION}" != "${CMAKE_VERSION}" ]]; then
    echo "FAIL: cumetalc --version reports ${CUMETALC_VERSION}, expected ${CMAKE_VERSION}"
    fail=1
fi

if [[ ${fail} -ne 0 ]]; then
    exit 1
fi

echo "PASS: version ${CMAKE_VERSION} is consistent across CMake, cumetal_native.h, and cumetalc"
