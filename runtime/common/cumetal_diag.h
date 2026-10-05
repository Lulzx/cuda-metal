#pragma once
//
// cumetal_diag.h — lightweight runtime diagnostics helpers.
//
// CuMetal translates a large CUDA surface onto Metal, and a handful of paths are
// intentionally limited (for example FP64 Dekker emulation). Silent
// incorrectness is the project's biggest risk, so limitations must be surfaced
// at runtime rather than only in the spec. These helpers provide a uniform,
// thread-safe, print-once warning channel plus a truthy-env check.
//
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_set>

namespace cumetal {

// True when `name` is set to a truthy value (1/true/yes/on, case-insensitive).
inline bool diag_env_truthy(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return false;
    }
    switch (v[0]) {
        case '1':
        case 't':
        case 'T':
        case 'y':
        case 'Y':
        case 'o':
        case 'O':
            return true;
        default:
            return false;
    }
}

// Emit `message` to stderr at most once per unique `key`, prefixed with
// "CUMETAL WARNING:". Thread-safe. Subsequent calls with the same key are
// suppressed so a warning in a hot kernel-launch path does not flood stderr.
inline void warn_once(const std::string& key, const std::string& message) {
    static std::mutex mutex;
    static std::unordered_set<std::string> seen;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!seen.insert(key).second) {
            return;
        }
    }
    std::fprintf(stderr, "CUMETAL WARNING: %s\n", message.c_str());
    std::fflush(stderr);
}

// A kernel's double libdevice calls (exp, log, pow, trig, ...) have no binary64
// implementation and were lowered through binary32. The generated MSL records
// this only as a comment, so it has to be said at run time.
inline void warn_fp64_libdevice_via_f32(std::string_view fp64_mode) {
    warn_once("fp64-libdevice-via-f32",
              std::string("a kernel calls double-precision math functions (exp, log, pow, "
                          "trig, erf, ...) that have no binary64 implementation; they run in "
                          "binary32 (about 7 significant digits, float range) even though "
                          "CUMETAL_FP64_MODE=") +
                  std::string(fp64_mode) +
                  ". Only FP64 arithmetic, sqrt and fma follow that mode. See "
                  "docs/fp64-policy.md.");
}

}  // namespace cumetal
