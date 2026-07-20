#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace cumetal::ptx {

enum class PtxMetalBackend {
    kLegacy,
    kCumetalIr,
};

struct LowerToMetalOptions {
    bool strict = false;
    std::string entry_name;
    PtxMetalBackend backend = PtxMetalBackend::kLegacy;
};

enum class MetalLoweringKind {
    kNone,
    kGenericCumetalIr,
    kGenericPtx,
    kSpecializedMsl,
    kApproximateStub,
};

struct LowerToMetalResult {
    bool ok = false;
    bool matched = false;
    // True when the matched lowering is an approximate / passthru stub whose
    // output is not numerically correct (e.g. GGML rope/dequant/copy stubs that
    // let a run proceed without computing the real result). Callers must treat
    // such kernels as unsafe: the runtime skips them by default so the program
    // fails loudly instead of silently producing wrong answers.
    bool approximate = false;
    MetalLoweringKind lowering_kind = MetalLoweringKind::kNone;
    std::string entry_name;
    std::string metal_source;
    // printf metadata: if the kernel uses device printf, the compiler injects a hidden
    // ring-buffer argument (spec §5.3).  printf_formats[i] is the format string for id i.
    std::vector<std::string> printf_formats;
    std::vector<std::string> warnings;
    std::string error;
};

LowerToMetalResult lower_ptx_to_metal_source(std::string_view ptx,
                                             const LowerToMetalOptions& options = {});

}  // namespace cumetal::ptx
