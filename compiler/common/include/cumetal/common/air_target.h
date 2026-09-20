#pragma once

#include <string>

namespace cumetal::common {

// The AIR version and target triple CuMetal emits into generated LLVM IR.
//
// These are not free parameters: `air-lld` from the installed Metal Toolchain
// rejects a module whose `!air.version` does not match the version it was built
// for ("air version set to 2.8.0 ... but expecting 2.9"). Hardcoding a version
// therefore breaks the whole compile path on every Xcode upgrade, so we probe
// the installed toolchain instead.
struct AirTarget {
    int major = 2;
    int minor = 8;
    // e.g. "air64_v29-apple-macosx27.0.0"
    std::string triple;
    // "2.9"
    std::string version_string() const;
};

// Probes `xcrun metal` once per process (cached) for the AIR target triple it
// compiles to, and derives the AIR version from its `air64_vMN` component.
// Falls back to the last known-good 2.8 target when no toolchain is reachable.
// `CUMETAL_AIR_TARGET_TRIPLE` overrides the probe outright.
const AirTarget& detected_air_target();

}  // namespace cumetal::common
