#include "nvrtc_options.h"

#include <algorithm>
#include <cstddef>

namespace cumetal {
namespace nvrtc {
namespace {

bool starts_with(const std::string& value, const char* prefix) {
    const std::size_t length = std::string(prefix).size();
    return value.size() >= length && value.compare(0, length, prefix) == 0;
}

std::string after(const std::string& value, const char* prefix) {
    return value.substr(std::string(prefix).size());
}

// `-D NAME=VALUE`, `-D NAME` and `-U NAME` all key off the macro name.
std::string macro_name(const std::string& definition) {
    const std::size_t equals = definition.find('=');
    return equals == std::string::npos ? definition : definition.substr(0, equals);
}

// nvcc spells architectures `sm_90a` / `sm_100f`, where the trailing letter
// selects an arch- or family-specific variant of the same hardware generation.
// Metal has no such split, so the suffix is dropped rather than passed to Clang,
// which would reject it.
std::string strip_arch_suffix(const std::string& arch) {
    std::string result = arch;
    while (!result.empty() && (result.back() == 'a' || result.back() == 'f')) {
        result.pop_back();
    }
    return result;
}

// Options that describe PTX/SASS code generation, host toolchain plumbing or
// diagnostics. Accepting and dropping these keeps a stock NVRTC call site
// working; none of them changes the meaning of the source CuMetal compiles.
bool is_ignorable(const std::string& option) {
    static const char* const kExact[] = {
        "--device-as-default-execution-space",
        "-default-device",
        "--extra-device-vectorization",
        "-extra-device-vectorization",
        "--restrict",
        "-restrict",
        "--use_fast_math",
        "-use_fast_math",
        "--generate-line-info",
        "-lineinfo",
        "--device-debug",
        "-G",
        "--dopt=on",
        "-dopt=on",
        "--relocatable-device-code=true",
        "--relocatable-device-code=false",
        "-rdc=true",
        "-rdc=false",
        "-dlto",
        "--dlto",
        "--device-int128",
        "--builtin-move-forward=true",
        "--builtin-move-forward=false",
        "--builtin-initializer-list=true",
        "--builtin-initializer-list=false",
        "-pch",
        "--pch",
        "--pch-create",
        "--pch-use",
        "--split-compile=0",
        "--minimal",
        "-minimal",
    };
    for (const char* candidate : kExact) {
        if (option == candidate) return true;
    }

    static const char* const kPrefixes[] = {
        "--diag-suppress=",   "-diag-suppress=", "--diag-error=",
        "--diag-warn=",       "--fmad=",         "-fmad=",
        "--ftz=",             "-ftz=",           "--prec-div=",
        "-prec-div=",         "--prec-sqrt=",    "-prec-sqrt=",
        "--maxrregcount=",    "-maxrregcount=",  "--Ofast-compile=",
        "-Ofast-compile=",    "--pch-dir=",      "--pch-create=",
        "--pch-use=",         "--fdevice-time-trace=", "--split-compile=",
        "--optimization-info=", "--gen-opt-lto",
    };
    for (const char* candidate : kPrefixes) {
        if (starts_with(option, candidate)) return true;
    }
    return false;
}

}  // namespace

TranslatedOptions translate_options(const std::vector<std::string>& options) {
    TranslatedOptions result;
    std::vector<std::string> include_dirs;
    std::vector<std::string> defines;
    bool arch_seen = false;

    const auto add_define = [&defines](const std::string& definition) {
        defines.push_back(definition);
    };
    const auto remove_define = [&defines](const std::string& name) {
        defines.erase(
            std::remove_if(defines.begin(), defines.end(),
                           [&name](const std::string& existing) {
                               return macro_name(existing) == name;
                           }),
            defines.end());
    };
    const auto set_arch = [&result, &arch_seen](const std::string& value) {
        if (starts_with(value, "compute_")) {
            result.ptx_requested = true;
            result.arch = "sm_" + strip_arch_suffix(after(value, "compute_"));
        } else if (starts_with(value, "sm_")) {
            result.ptx_requested = false;
            result.arch = "sm_" + strip_arch_suffix(after(value, "sm_"));
        } else {
            return false;
        }
        arch_seen = true;
        return true;
    };

    for (std::size_t i = 0; i < options.size(); ++i) {
        const std::string& option = options[i];
        if (option.empty()) continue;

        // An argument that needs a value may arrive joined or separated. NVRTC
        // itself only documents the joined forms, but callers that build their
        // option vector from an nvcc command line produce both.
        const auto value_of = [&](const char* long_prefix, const char* short_prefix,
                                  std::string* out) {
            if (long_prefix != nullptr && starts_with(option, long_prefix)) {
                *out = after(option, long_prefix);
                return true;
            }
            if (short_prefix == nullptr) return false;
            if (option == short_prefix) {
                if (i + 1 >= options.size()) return false;
                *out = options[++i];
                return true;
            }
            if (starts_with(option, short_prefix)) {
                *out = after(option, short_prefix);
                return true;
            }
            return false;
        };

        std::string value;
        if (value_of("--include-path=", "-I", &value)) {
            if (!value.empty()) include_dirs.push_back(value);
            continue;
        }
        if (value_of("--define-macro=", "-D", &value)) {
            if (!value.empty()) add_define(value);
            continue;
        }
        if (value_of("--undefine-macro=", "-U", &value)) {
            // Clang has no `-U` on CuMetal's device line, and an undefine of a
            // macro CuMetal never defined is a no-op anyway. Honour it against
            // the macros this same option list defined, and drop it otherwise.
            if (!value.empty()) remove_define(macro_name(value));
            continue;
        }
        if (value_of("--gpu-architecture=", "-arch=", &value)) {
            if (!set_arch(value)) result.unrecognized.push_back(option);
            continue;
        }
        if (option == "--gpu-architecture" || option == "-arch") {
            if (i + 1 < options.size()) {
                if (!set_arch(options[++i])) result.unrecognized.push_back(option);
            } else {
                result.unrecognized.push_back(option);
            }
            continue;
        }
        if (value_of("--std=", "-std=", &value)) {
            // CuMetal's device compile is fixed at C++20, which subsumes every
            // standard NVRTC accepts. Record the request without acting on it.
            result.ignored.push_back(option);
            continue;
        }
        if (value_of("--pre-include=", "-include", &value)) {
            if (!value.empty()) {
                result.compiler_args.push_back("--cuda-include");
                result.compiler_args.push_back(value);
            }
            continue;
        }
        if (is_ignorable(option)) {
            result.ignored.push_back(option);
            continue;
        }
        result.unrecognized.push_back(option);
    }

    if (!arch_seen) {
        result.arch = "sm_80";
    }

    std::vector<std::string> args;
    args.push_back("--cuda-arch");
    args.push_back(result.arch);
    for (const std::string& dir : include_dirs) {
        args.push_back("-I");
        args.push_back(dir);
    }
    for (const std::string& definition : defines) {
        args.push_back("-D");
        args.push_back(definition);
    }
    args.insert(args.end(), result.compiler_args.begin(), result.compiler_args.end());
    result.compiler_args = std::move(args);
    return result;
}

}  // namespace nvrtc
}  // namespace cumetal
