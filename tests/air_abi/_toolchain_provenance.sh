#!/usr/bin/env bash
# Shared toolchain-provenance banner for the AIR ABI regression tests. Source, do not execute.
#
# spec.md §10.5 asks for AIR ABI regression across Xcode 15.0 / 15.4 / 16.0 / 16.2+, because the
# ABI is undocumented and Apple can change it in a minor release (spec §13 rates this the highest
# project risk). A single machine can only exercise the toolchain it has installed. What it can
# always do is say *which* one it exercised: a green AIR ABI run is only meaningful alongside the
# toolchain that produced it, and without that line a passing log is unattributable.
#
# Real multi-version coverage comes from running CI across macOS runner images that ship
# different Xcodes; this banner is what makes those runs distinguishable in the logs.

cumetal_print_toolchain_provenance() {
    local label="${1:-air_abi}"

    local macos_version macos_build xcode_version xcode_build
    macos_version="$(sw_vers -productVersion 2>/dev/null || echo unknown)"
    macos_build="$(sw_vers -buildVersion 2>/dev/null || echo unknown)"

    # `xcodebuild -version` prints two lines: "Xcode 16.2" then "Build version 16C5032a".
    local xcodebuild_output
    xcodebuild_output="$(xcodebuild -version 2>/dev/null || true)"
    xcode_version="$(awk '/^Xcode/ {print $2; exit}' <<<"${xcodebuild_output}")"
    xcode_build="$(awk '/^Build version/ {print $3; exit}' <<<"${xcodebuild_output}")"
    : "${xcode_version:=unknown}"
    : "${xcode_build:=unknown}"

    # The metal compiler version is the value that actually determines AIR output, and it can
    # differ from the Xcode version when a standalone Metal toolchain is selected.
    local metal_version
    metal_version="$(xcrun metal --version 2>/dev/null | head -1 || true)"
    : "${metal_version:=unavailable}"

    local chip
    chip="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"

    echo "CUMETAL_AIR_ABI_PROVENANCE" \
        "test=${label}" \
        "macos=${macos_version}" \
        "macos_build=${macos_build}" \
        "xcode=${xcode_version}" \
        "xcode_build=${xcode_build}" \
        "toolchains=${TOOLCHAINS:-<default>}" \
        "chip=\"${chip}\"" \
        "metal=\"${metal_version}\""
}
