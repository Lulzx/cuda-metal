#pragma once
//
// nvrtc_options.h — translation from NVRTC compile options to `cumetalc` flags.
//
// NVRTC's option surface is nvcc's, and most of it describes code generation
// choices that only exist on the PTX/SASS side (vectorization hints, diagnostic
// numbers, fatbin layout, precompiled headers). CuMetal recognises those and
// drops them rather than failing the compile, because a caller that passes
// `--extra-device-vectorization` still wants a module back. Options that do
// carry meaning for a Metal build -- include paths, macros, the target
// architecture -- map onto `cumetalc` flags.
//
// This lives apart from the NVRTC entry points so the mapping can be unit
// tested without spawning a compiler.
//
#include <string>
#include <vector>

namespace cumetal {
namespace nvrtc {

struct TranslatedOptions {
    // Flags to pass to `cumetalc`, in order, excluding the input path, `-o` and
    // `--emit`.
    std::vector<std::string> compiler_args;

    // Options understood by NVRTC that have no Metal analogue and were dropped.
    // Reported through the program log so a caller can see what was skipped.
    std::vector<std::string> ignored;

    // Options this shim does not recognise at all. Also dropped, but worth a
    // louder log line: an unrecognised option is more likely to change meaning.
    std::vector<std::string> unrecognized;

    // Set when the caller asked for a virtual architecture (`compute_NN`),
    // meaning it intends to read PTX back out. CuMetal cannot produce PTX from
    // CUDA source, so `nvrtcCompileProgram` fails early with an explanation
    // instead of handing back a metallib the caller will mis-handle.
    bool ptx_requested = false;

    // The `sm_NN` architecture `cumetalc` is asked to target. CuMetal presents a
    // single Ampere-equivalent device (spec 6.8), so this is sm_80 unless the
    // caller overrides it.
    std::string arch = "sm_80";
};

TranslatedOptions translate_options(const std::vector<std::string>& options);

// The architectures `nvrtcGetSupportedArchs` reports. CuMetal exposes exactly
// one device architecture, so callers that pick CUBIN-vs-PTX by membership in
// this list land on CUBIN.
inline constexpr int kSupportedArch = 80;

}  // namespace nvrtc
}  // namespace cumetal
