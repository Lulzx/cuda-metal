#include "nvrtc_options.h"

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using cumetal::nvrtc::translate_options;
using cumetal::nvrtc::TranslatedOptions;

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool contains(const std::vector<std::string>& values, const std::string& value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

// True when `first` is immediately followed by `second`, which is how cumetalc
// reads its separated flag/value pairs.
bool contains_pair(const std::vector<std::string>& values,
                   const std::string& first,
                   const std::string& second) {
    for (std::size_t i = 0; i + 1 < values.size(); ++i) {
        if (values[i] == first && values[i + 1] == second) return true;
    }
    return false;
}

}  // namespace

int main() {
    {
        const TranslatedOptions result = translate_options({});
        if (!expect(result.arch == "sm_80", "default architecture is sm_80")) return 1;
        if (!expect(contains_pair(result.compiler_args, "--cuda-arch", "sm_80"),
                    "default architecture reaches cumetalc")) {
            return 1;
        }
        if (!expect(!result.ptx_requested, "no PTX requested by default")) return 1;
        if (!expect(contains_pair(result.compiler_args, "-D", "__CUDACC_RTC__=1"),
                    "__CUDACC_RTC__ is predefined as NVRTC does")) {
            return 1;
        }
    }

    // Sources branch on __CUDACC_RTC__ to skip includes a runtime compile has
    // no toolkit for; a caller that explicitly cancels it gets that honoured.
    {
        const TranslatedOptions result =
            translate_options({"--undefine-macro=__CUDACC_RTC__"});
        if (!expect(!contains(result.compiler_args, "__CUDACC_RTC__=1"),
                    "__CUDACC_RTC__ can be undefined by the caller")) {
            return 1;
        }
    }

    // The option set Warp's wp_cuda_compile_program builds for a release build.
    {
        const TranslatedOptions result = translate_options({
            "--gpu-architecture=sm_80",
            "--include-path=/opt/warp/native",
            "--std=c++17",
            "--define-macro=NDEBUG",
            "--define-macro=WP_ENABLE_MATHDX=0",
            "--undefine-macro=WP_VERIFY_FP",
            "--fmad=false",
            "--device-as-default-execution-space",
            "--extra-device-vectorization",
            "--restrict",
            "--diag-suppress=177,550",
        });
        if (!expect(result.arch == "sm_80", "sm_80 architecture parsed")) return 1;
        if (!expect(!result.ptx_requested, "real architecture is not a PTX request")) return 1;
        if (!expect(contains_pair(result.compiler_args, "-I", "/opt/warp/native"),
                    "include path translated")) {
            return 1;
        }
        if (!expect(contains_pair(result.compiler_args, "-D", "NDEBUG"), "macro translated")) {
            return 1;
        }
        if (!expect(contains_pair(result.compiler_args, "-D", "WP_ENABLE_MATHDX=0"),
                    "valued macro translated")) {
            return 1;
        }
        if (!expect(result.unrecognized.empty(), "no unrecognized options in Warp's set")) {
            for (const std::string& option : result.unrecognized) {
                std::fprintf(stderr, "  unrecognized: %s\n", option.c_str());
            }
            return 1;
        }
        if (!expect(contains(result.ignored, "--extra-device-vectorization") &&
                        contains(result.ignored, "--diag-suppress=177,550") &&
                        contains(result.ignored, "--fmad=false") &&
                        contains(result.ignored, "--std=c++17"),
                    "code-generation options are recorded as ignored")) {
            return 1;
        }
    }

    // A virtual architecture means the caller wants PTX back, which CuMetal
    // cannot produce from CUDA source.
    {
        const TranslatedOptions result = translate_options({"--gpu-architecture=compute_75"});
        if (!expect(result.ptx_requested, "compute_XX flags a PTX request")) return 1;
        if (!expect(result.arch == "sm_75", "compute_XX still selects a real target")) return 1;
    }

    // nvcc's arch- and family-specific suffixes name the same hardware
    // generation; Clang rejects them.
    {
        const TranslatedOptions result = translate_options({"-arch=sm_90a"});
        if (!expect(result.arch == "sm_90", "arch suffix stripped")) return 1;
    }

    // Joined and separated spellings both appear in the wild.
    {
        const TranslatedOptions result =
            translate_options({"-I/tmp/one", "-I", "/tmp/two", "-DA=1", "-D", "B"});
        if (!expect(contains_pair(result.compiler_args, "-I", "/tmp/one"), "joined -I")) return 1;
        if (!expect(contains_pair(result.compiler_args, "-I", "/tmp/two"), "separated -I")) {
            return 1;
        }
        if (!expect(contains_pair(result.compiler_args, "-D", "A=1"), "joined -D")) return 1;
        if (!expect(contains_pair(result.compiler_args, "-D", "B"), "separated -D")) return 1;
    }

    // -U cancels a macro defined earlier in the same option list, and is
    // otherwise dropped: CuMetal's device line has no undefine flag.
    {
        const TranslatedOptions result = translate_options(
            {"--define-macro=KEEP=1", "--define-macro=DROP=2", "--undefine-macro=DROP"});
        if (!expect(contains_pair(result.compiler_args, "-D", "KEEP=1"), "unrelated macro kept")) {
            return 1;
        }
        if (!expect(!contains(result.compiler_args, "DROP=2"), "undefined macro removed")) {
            return 1;
        }
    }

    {
        const TranslatedOptions result = translate_options({"--wholly-invented-option"});
        if (!expect(contains(result.unrecognized, "--wholly-invented-option"),
                    "unknown option reported")) {
            return 1;
        }
        // --cuda-arch sm_80 and -D __CUDACC_RTC__=1, and nothing else.
        if (!expect(result.compiler_args.size() == 4,
                    "unknown option does not reach cumetalc")) {
            return 1;
        }
    }

    std::printf("PASS: nvrtc option translation unit tests\n");
    return 0;
}
