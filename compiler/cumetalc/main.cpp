#include "cumetal/air_emitter/emitter.h"
#include "cumetal/common/metallib.h"
#include "cumetal/ir/ir.h"
#include "cumetal/ir/nvvm_importer.h"
#include "cumetal/metal/lower_to_msl.h"
#include "cumetal/ptx/lower_to_metal.h"
#include "cumetal/ptx/lower_to_llvm.h"
#include "cumetal/ptx/parser.h"

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef CUMETAL_SOURCE_DIR
#define CUMETAL_SOURCE_DIR ""
#endif

namespace {

enum class BackendKind {
    kLegacy,
    kCumetalIr,
};

enum class EmitStage {
    kMetallib,
    kLlvm,
    kCumetalIr,
    kMetalIr,
    kMsl,
};

struct CommandResult {
    bool started = false;
    int exit_code = -1;
    std::string output;
};

std::uint32_t ptx_scalar_size(std::string_view type) {
    if (type == ".u8" || type == ".s8" || type == ".b8") return 1;
    if (type == ".u16" || type == ".s16" || type == ".b16") return 2;
    if (type == ".u64" || type == ".s64" || type == ".b64" || type == ".f64") return 8;
    return 4;
}

std::uint32_t ptx_param_size(const cumetal::ptx::Parameter& param) {
    const auto open = param.name.rfind('[');
    const auto close = param.name.rfind(']');
    if (open != std::string::npos && close == param.name.size() - 1 && close > open + 1) {
        const std::string count_text = param.name.substr(open + 1, close - open - 1);
        char* end = nullptr;
        const unsigned long count = std::strtoul(count_text.c_str(), &end, 10);
        if (end != count_text.c_str() && *end == '\0' && count <= 4096) {
            return static_cast<std::uint32_t>(count) * ptx_scalar_size(param.type);
        }
    }
    return ptx_scalar_size(param.type);
}

std::string build_ptx_abi_sidecar(std::string_view ptx_source,
                                  const std::string& requested_entry) {
    cumetal::ptx::ParseOptions parse_options;
    parse_options.strict = false;
    const auto parsed = cumetal::ptx::parse_ptx(ptx_source, parse_options);
    if (!parsed.ok) {
        return {};
    }
    for (const auto& entry : parsed.module.entries) {
        if (!requested_entry.empty() && entry.name != requested_entry) {
            continue;
        }
        std::string text = "CUMETAL_ABI_V1\nkernel " + entry.name + "\n";
        text += "shared " +
                std::to_string(cumetal::ptx::compute_static_shared_bytes(ptx_source,
                                                                         entry.name)) + "\n";
        for (const auto& param : entry.params) {
            text += param.is_pointer ? "arg buffer 8\n"
                                     : "arg bytes " + std::to_string(ptx_param_size(param)) + "\n";
        }
        return text;
    }
    return {};
}

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <file.cu> -o <executable>            # compile and link a CUDA program\n"
                 "       "
              << argv0
              << " <file.cu> -o <file.metallib>         # compile device code only\n\n"
                 "Full options: "
              << argv0
              << " [--input] <file.{metal,cu,ptx,ll,air,bc}> [--output|-o <file.metallib>]"
                 " [--mode xcrun|experimental] [--fallback-experimental]"
                 " [--overwrite] [--skip-validate] [--xcrun-validate]"
                 " [--kernel-name name] [--entry name] [--ptx-strict]"
                 " [--cuda-device] [--cuda-arch sm_XX] [--cuda-clang path]"
                 " [--cuda-inline-threshold value]"
                 " [-I path] [-D name[=value]] [--cuda-include path]"
                 " [--backend legacy|cumetal-ir]"
                 " [--emit llvm|cumetal-ir|metal-ir|msl|metallib|exe]"
                 " [--link|--no-link] [--save-temps]"
                 " [--fp64=native|emulate|warn]\n";
}

std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    for (char& c : ext) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return ext;
}

std::filesystem::path make_temp_path(const std::string& extension_with_dot) {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto pid = static_cast<long long>(::getpid());
    return std::filesystem::temp_directory_path() /
           ("cumetalc-ptx-" + std::to_string(pid) + "-" + std::to_string(now) + extension_with_dot);
}

std::string quote_shell(const std::string& value) {
    std::string quoted;
    quoted.reserve(value.size() + 2);
    quoted.push_back('\'');
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

CommandResult run_command_capture(const std::string& command) {
    CommandResult result;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return result;
    }

    result.started = true;
    char buffer[512];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result.output.append(buffer);
    }

    const int status = pclose(pipe);
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    }

    return result;
}

bool command_exists(const std::string& name) {
    const CommandResult result = run_command_capture("command -v " + name + " >/dev/null 2>&1; echo $?");
    if (!result.started || result.exit_code != 0 || result.output.empty()) {
        return false;
    }
    return result.output[0] == '0';
}

bool xcrun_tool_exists(const std::string& tool_name) {
    const CommandResult result =
        run_command_capture("xcrun --find " + quote_shell(tool_name) + " >/dev/null 2>&1; echo $?");
    if (!result.started || result.exit_code != 0 || result.output.empty()) {
        return false;
    }
    return result.output[0] == '0';
}

// Where cumetalc finds the pieces it needs to build a complete executable. The driver has to
// work both from the build tree and from `cmake --install` output, so nothing here may assume
// CUMETAL_SOURCE_DIR exists on disk.
struct ResourceLayout {
    std::filesystem::path include_dir;    // holds cuda_runtime.h
    std::filesystem::path lib_dir;        // holds libcumetal.dylib
    std::filesystem::path toolchain_dir;  // holds the ptxas/fatbinary shims
    bool ok = false;
};

// Absolute path of this cumetalc binary, used to locate sibling resources in an installed tree.
std::filesystem::path executable_path(const char* argv0) {
    std::error_code ec;
    if (argv0 != nullptr && argv0[0] != '\0') {
        std::filesystem::path candidate(argv0);
        if (candidate.has_parent_path()) {
            auto absolute = std::filesystem::weakly_canonical(candidate, ec);
            if (!ec && std::filesystem::exists(absolute)) return absolute;
        }
        const CommandResult found =
            run_command_capture("command -v " + quote_shell(argv0) + " 2>/dev/null");
        if (found.started && found.exit_code == 0 && !found.output.empty()) {
            std::string path = found.output;
            while (!path.empty() &&
                   std::isspace(static_cast<unsigned char>(path.back())) != 0) {
                path.pop_back();
            }
            if (!path.empty()) {
                auto absolute = std::filesystem::weakly_canonical(path, ec);
                if (!ec) return absolute;
            }
        }
    }
    return {};
}

bool layout_from_root(const std::filesystem::path& include_dir,
                      const std::filesystem::path& lib_dir,
                      const std::filesystem::path& toolchain_dir,
                      ResourceLayout* out) {
    if (!std::filesystem::exists(include_dir / "cuda_runtime.h")) return false;
    if (!std::filesystem::exists(lib_dir / "libcumetal.dylib")) return false;
    out->include_dir = include_dir;
    out->lib_dir = lib_dir;
    out->toolchain_dir = toolchain_dir;
    out->ok = true;
    return true;
}

ResourceLayout resolve_resources(const char* argv0) {
    ResourceLayout layout;

    // 1. Explicit override wins, so a relocated or unusual install is always recoverable.
    if (const char* root = std::getenv("CUMETAL_ROOT"); root != nullptr && root[0] != '\0') {
        const std::filesystem::path prefix(root);
        if (layout_from_root(prefix / "include",
                             prefix / "lib",
                             prefix / "libexec" / "cumetal" / "cuda_toolchain",
                             &layout)) {
            return layout;
        }
    }

    const std::filesystem::path self = executable_path(argv0);
    if (!self.empty()) {
        const std::filesystem::path bin_dir = self.parent_path();

        // 2. Installed layout: <prefix>/bin/cumetalc alongside <prefix>/{include,lib,libexec}.
        const std::filesystem::path prefix = bin_dir.parent_path();
        if (layout_from_root(prefix / "include",
                             prefix / "lib",
                             prefix / "libexec" / "cumetal" / "cuda_toolchain",
                             &layout)) {
            return layout;
        }

        // 3. Build tree: cumetalc sits next to libcumetal.dylib and cuda_toolchain/, with the
        //    headers still back in the source directory.
        const std::filesystem::path source_api =
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api";
        if (layout_from_root(source_api, bin_dir, bin_dir / "cuda_toolchain", &layout)) {
            return layout;
        }
    }

    return layout;
}

// Homebrew LLVM defaults to PTX 4.2, which rejects newer -march values outright. Mirrors
// cumetal_cuda_ptx_feature_flags() in scripts/cumetal_cuda_flags.sh; keep the two in step.
std::string ptx_feature_flags_for_arch(const std::string& arch) {
    const auto feature = [](const char* name) {
        return std::string(" -Xclang -target-feature -Xclang +") + name;
    };
    if (arch == "sm_80" || arch == "sm_86" || arch == "sm_89" || arch.rfind("sm_90", 0) == 0) {
        return feature("ptx70");
    }
    if (arch == "sm_75" || arch == "sm_78") return feature("ptx63");
    if (arch == "sm_70" || arch == "sm_72") return feature("ptx60");
    if (arch == "sm_61") return feature("ptx50");
    return {};
}

std::filesystem::path find_cuda_clang(
    const std::filesystem::path& requested = std::filesystem::path()) {
    if (!requested.empty()) {
        return requested;
    }
    if (const char* configured = std::getenv("CUMETAL_CUDA_CLANG");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    if (const char* configured = std::getenv("CUMETAL_CLANG");
        configured != nullptr && *configured != '\0' &&
        std::filesystem::exists(configured)) {
        return configured;
    }
    static constexpr const char* kCandidates[] = {
        "/opt/homebrew/opt/llvm/bin/clang++",
        "/usr/local/opt/llvm/bin/clang++",
    };
    for (const char* candidate : kCandidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    const CommandResult found = run_command_capture("command -v clang++ 2>/dev/null");
    if (found.started && found.exit_code == 0 && !found.output.empty()) {
        std::string path = found.output;
        while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back())) != 0) {
            path.pop_back();
        }
        if (!path.empty()) return path;
    }
    return {};
}

std::filesystem::path find_llvm_opt(const std::filesystem::path& clang) {
    const std::filesystem::path sibling = clang.parent_path() / "opt";
    if (std::filesystem::exists(sibling)) return sibling;
    const CommandResult found = run_command_capture("command -v opt 2>/dev/null");
    if (found.started && found.exit_code == 0 && !found.output.empty()) {
        std::string path = found.output;
        while (!path.empty() &&
               std::isspace(static_cast<unsigned char>(path.back())) != 0) {
            path.pop_back();
        }
        if (!path.empty()) return path;
    }
    return {};
}

std::string ptx_feature_for_arch(std::string_view arch) {
    if (arch == "sm_90" || arch == "sm_89" || arch == "sm_86" || arch == "sm_80") {
        return "+ptx70";
    }
    if (arch == "sm_78" || arch == "sm_75") return "+ptx63";
    if (arch == "sm_72" || arch == "sm_70") return "+ptx60";
    if (arch == "sm_61") return "+ptx50";
    return {};
}

std::string extension_for_stage(EmitStage stage) {
    switch (stage) {
        case EmitStage::kLlvm: return ".ll";
        case EmitStage::kCumetalIr: return ".cmir";
        case EmitStage::kMetalIr: return ".metalir";
        case EmitStage::kMsl: return ".metal";
        case EmitStage::kMetallib: return ".metallib";
    }
    return ".metallib";
}

bool write_text_output(const std::filesystem::path& output, std::string_view text,
                       bool overwrite, std::string* error) {
    if (std::filesystem::exists(output) && !overwrite) {
        if (error != nullptr) {
            *error = "output already exists (pass --overwrite to replace): " +
                     output.string();
        }
        return false;
    }
    const std::vector<std::uint8_t> bytes(text.begin(), text.end());
    return cumetal::common::write_file_bytes(output, bytes, error);
}

bool emit_inspection_stage(const cumetal::metal::PtxToMslResult& compiled,
                           EmitStage stage, const std::filesystem::path& output,
                           bool overwrite, std::string* error) {
    std::string text;
    if (stage == EmitStage::kCumetalIr) {
        text = cumetal::ir::print(compiled.gpu_ir);
    } else if (stage == EmitStage::kMetalIr) {
        text = cumetal::ir::print(compiled.metal_ir);
    } else if (stage == EmitStage::kMsl) {
        text = compiled.source;
    } else {
        if (error != nullptr) *error = "requested stage is not a textual CuMetal output";
        return false;
    }
    return write_text_output(output, text, overwrite, error);
}

struct ExecutableDriverOptions {
    std::filesystem::path input;
    std::filesystem::path output;
    std::filesystem::path cuda_clang;
    std::string cuda_arch = "sm_80";
    std::string cuda_inline_threshold;
    std::vector<std::filesystem::path> include_dirs;
    std::vector<std::string> defines;
    std::vector<std::filesystem::path> forced_includes;
    bool keep_intermediates = false;
};

// Compile a complete CUDA translation unit -- host code and `<<<>>>` launches included -- into a
// runnable executable linked against libcumetal. This is spec.md's Phase 2/3 exit criterion.
//
// Clang does the whole job: it compiles the host side to the standard CUDA registration ABI
// (__cudaRegisterFatBinary / __cudaPushCallConfiguration / cudaLaunchKernel), and drives the
// device side through the in-tree ptxas and fatbinary shims, which pass PTX through into a
// fatbinary envelope rather than assembling SASS. libcumetal's registration path then JITs that
// PTX to a metallib on first launch. So the executable carries its own device code and needs no
// metallib path at runtime -- which is exactly what the old two-file sample flow could not do.
int run_executable_driver(const ExecutableDriverOptions& options, const char* argv0) {
    const std::filesystem::path compiler = find_cuda_clang(options.cuda_clang);
    if (compiler.empty() || !std::filesystem::exists(compiler)) {
        std::cerr << "cumetalc failed: CUDA-capable clang++ not found; install Homebrew LLVM or "
                     "pass --cuda-clang/CUMETAL_CUDA_CLANG\n";
        return 1;
    }
    if (options.cuda_arch.size() < 4 || options.cuda_arch.substr(0, 3) != "sm_") {
        std::cerr << "cumetalc failed: --cuda-arch must use sm_XX form\n";
        return 2;
    }

    const ResourceLayout layout = resolve_resources(argv0);
    if (!layout.ok) {
        std::cerr << "cumetalc failed: could not locate the CuMetal headers and libcumetal.dylib "
                     "needed to link an executable.\n"
                     "  Set CUMETAL_ROOT to the install prefix, or run cumetalc from the build "
                     "directory.\n";
        return 1;
    }
    if (!std::filesystem::exists(layout.toolchain_dir / "fatbinary") ||
        !std::filesystem::exists(layout.toolchain_dir / "ptxas")) {
        std::cerr << "cumetalc failed: the ptxas/fatbinary shims are missing from "
                  << layout.toolchain_dir << ".\n"
                     "  Clang needs them on PATH to emit device code for a linked executable.\n";
        return 1;
    }

    const std::filesystem::path object_file = make_temp_path(".o");

    // Clang execs the shims as subprocesses, so they must be found via PATH. Prepend rather than
    // replace so a user-provided toolchain earlier in PATH still loses to ours deliberately.
    const char* existing_path = std::getenv("PATH");
    const std::string path_prefix =
        "PATH=" + quote_shell(layout.toolchain_dir.string()) + ":" +
        (existing_path != nullptr ? std::string(existing_path) : std::string("/usr/bin:/bin")) +
        " ";

    std::string compile = path_prefix + quote_shell(compiler.string()) +
                          " -x cuda -std=c++17 -O2"
                          " --cuda-gpu-arch=" +
                          quote_shell(options.cuda_arch) +
                          ptx_feature_flags_for_arch(options.cuda_arch) +
                          " -nocudainc -nocudalib -Wno-unknown-cuda-version -Wno-pass-failed"
                          " -D__CUDACC__=1 -D__NVCC__=1"
                          " -I " +
                          quote_shell(layout.include_dir.string()) + " -include " +
                          quote_shell("cuda_runtime.h");
    if (!options.cuda_inline_threshold.empty()) {
        compile += " -fgpu-inline-threshold=" + quote_shell(options.cuda_inline_threshold) +
                   " -mllvm -inline-all-viable-calls";
    }
    for (const auto& dir : options.include_dirs) {
        compile += " -I " + quote_shell(dir.string());
    }
    for (const auto& define : options.defines) {
        compile += " -D " + quote_shell(define);
    }
    for (const auto& forced : options.forced_includes) {
        compile += " -include " + quote_shell(forced.string());
    }
    compile += " -c " + quote_shell(options.input.string()) + " -o " +
               quote_shell(object_file.string()) + " 2>&1";

    const CommandResult compile_result = run_command_capture(compile);
    if (!compile_result.output.empty()) {
        std::cerr << compile_result.output;
        if (compile_result.output.back() != '\n') std::cerr << '\n';
    }
    if (!compile_result.started || compile_result.exit_code != 0 ||
        !std::filesystem::exists(object_file)) {
        std::error_code ec;
        std::filesystem::remove(object_file, ec);
        std::cerr << "cumetalc failed: CUDA compilation of " << options.input << " failed\n";
        return 1;
    }

    // Link with an rpath so the produced binary runs without the caller exporting
    // DYLD_LIBRARY_PATH first.
    const std::string link = quote_shell(compiler.string()) + " " +
                             quote_shell(object_file.string()) + " -L " +
                             quote_shell(layout.lib_dir.string()) + " -lcumetal -Wl,-rpath," +
                             quote_shell(layout.lib_dir.string()) + " -o " +
                             quote_shell(options.output.string()) + " 2>&1";

    const CommandResult link_result = run_command_capture(link);
    if (!link_result.output.empty()) {
        std::cerr << link_result.output;
        if (link_result.output.back() != '\n') std::cerr << '\n';
    }

    if (!options.keep_intermediates) {
        std::error_code ec;
        std::filesystem::remove(object_file, ec);
    } else {
        std::cerr << "cumetalc: kept intermediate object " << object_file << "\n";
    }

    if (!link_result.started || link_result.exit_code != 0 ||
        !std::filesystem::exists(options.output)) {
        std::cerr << "cumetalc failed: linking " << options.output << " failed\n";
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    cumetal::air_emitter::EmitOptions options;
    BackendKind backend = BackendKind::kLegacy;
    bool backend_set_explicitly = false;
    EmitStage emit_stage = EmitStage::kMetallib;
    bool link_executable = false;
    bool link_requested_explicitly = false;
    bool keep_intermediates = false;
    bool mode_set = false;
    bool positional_input_set = false;
    std::string ptx_entry_name;
    bool ptx_strict = false;
    cumetal::ptx::Fp64Mode ptx_fp64_mode = cumetal::ptx::Fp64Mode::kNative;
    bool cuda_device_frontend = false;
    std::string cuda_arch = "sm_80";
    std::filesystem::path cuda_clang;
    std::string cuda_inline_threshold;
    std::vector<std::filesystem::path> cuda_include_dirs;
    std::vector<std::string> cuda_defines;
    std::vector<std::filesystem::path> cuda_forced_includes;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--input") {
            if (i + 1 >= argc) {
                std::cerr << "--input expects a path\n";
                return 2;
            }
            options.input = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 >= argc) {
                std::cerr << arg << " expects a path\n";
                return 2;
            }
            options.output = argv[++i];
        } else if (arg == "--mode") {
            if (i + 1 >= argc) {
                std::cerr << "--mode expects xcrun or experimental\n";
                return 2;
            }
            const std::string mode = argv[++i];
            if (mode == "xcrun") {
                options.mode = cumetal::air_emitter::EmitMode::kXcrun;
                mode_set = true;
            } else if (mode == "experimental") {
                options.mode = cumetal::air_emitter::EmitMode::kExperimentalContainer;
                mode_set = true;
            } else {
                std::cerr << "invalid --mode: " << mode << "\n";
                return 2;
            }
        } else if (arg == "--fallback-experimental") {
            options.fallback_to_experimental = true;
        } else if (arg == "--overwrite") {
            options.overwrite = true;
        } else if (arg == "--skip-validate") {
            options.validate_output = false;
        } else if (arg == "--xcrun-validate") {
            options.run_xcrun_validate = true;
        } else if (arg == "--kernel-name") {
            if (i + 1 >= argc) {
                std::cerr << "--kernel-name expects a value\n";
                return 2;
            }
            options.kernel_name = argv[++i];
        } else if (arg == "--backend" || arg.rfind("--backend=", 0) == 0) {
            std::string value;
            if (arg == "--backend") {
                if (i + 1 >= argc) {
                    std::cerr << "--backend expects legacy or cumetal-ir\n";
                    return 2;
                }
                value = argv[++i];
            } else {
                value = arg.substr(std::string("--backend=").size());
            }
            if (value == "legacy") {
                backend = BackendKind::kLegacy;
                backend_set_explicitly = true;
            } else if (value == "cumetal-ir") {
                backend = BackendKind::kCumetalIr;
                backend_set_explicitly = true;
            } else {
                std::cerr << "invalid --backend: " << value
                          << " (valid: legacy, cumetal-ir)\n";
                return 2;
            }
        } else if (arg == "--emit" || arg.rfind("--emit=", 0) == 0) {
            std::string value;
            if (arg == "--emit") {
                if (i + 1 >= argc) {
                    std::cerr
                        << "--emit expects llvm, cumetal-ir, metal-ir, msl, metallib, or exe\n";
                    return 2;
                }
                value = argv[++i];
            } else {
                value = arg.substr(std::string("--emit=").size());
            }
            if (value == "exe") {
                emit_stage = EmitStage::kMetallib;
                link_executable = true;
                link_requested_explicitly = true;
            }
            else if (value == "llvm") emit_stage = EmitStage::kLlvm;
            else if (value == "cumetal-ir") emit_stage = EmitStage::kCumetalIr;
            else if (value == "metal-ir") emit_stage = EmitStage::kMetalIr;
            else if (value == "msl") emit_stage = EmitStage::kMsl;
            else if (value == "metallib") emit_stage = EmitStage::kMetallib;
            else {
                std::cerr << "invalid --emit stage: " << value << "\n";
                return 2;
            }
        } else if (arg == "--entry") {
            if (i + 1 >= argc) {
                std::cerr << "--entry expects a value\n";
                return 2;
            }
            ptx_entry_name = argv[++i];
        } else if (arg == "--ptx-strict") {
            ptx_strict = true;
        } else if (arg == "--cuda-device") {
            cuda_device_frontend = true;
        } else if (arg == "--cuda-arch") {
            if (i + 1 >= argc) {
                std::cerr << "--cuda-arch expects a value such as sm_80\n";
                return 2;
            }
            cuda_arch = argv[++i];
        } else if (arg == "--cuda-clang") {
            if (i + 1 >= argc) {
                std::cerr << "--cuda-clang expects a path\n";
                return 2;
            }
            cuda_clang = argv[++i];
        } else if (arg == "--cuda-inline-threshold") {
            if (i + 1 >= argc) {
                std::cerr << "--cuda-inline-threshold expects a non-negative integer\n";
                return 2;
            }
            cuda_inline_threshold = argv[++i];
            if (cuda_inline_threshold.empty() ||
                cuda_inline_threshold.find_first_not_of("0123456789") != std::string::npos) {
                std::cerr << "--cuda-inline-threshold expects a non-negative integer\n";
                return 2;
            }
        } else if (arg == "-I") {
            if (i + 1 >= argc) {
                std::cerr << "-I expects a path\n";
                return 2;
            }
            cuda_include_dirs.emplace_back(argv[++i]);
        } else if (arg.size() > 2 && arg.substr(0, 2) == "-I") {
            cuda_include_dirs.emplace_back(arg.substr(2));
        } else if (arg == "-D") {
            if (i + 1 >= argc) {
                std::cerr << "-D expects a definition\n";
                return 2;
            }
            cuda_defines.emplace_back(argv[++i]);
        } else if (arg.size() > 2 && arg.substr(0, 2) == "-D") {
            cuda_defines.emplace_back(arg.substr(2));
        } else if (arg == "--cuda-include") {
            if (i + 1 >= argc) {
                std::cerr << "--cuda-include expects a path\n";
                return 2;
            }
            cuda_forced_includes.emplace_back(argv[++i]);
        } else if (arg.size() > 7 && arg.substr(0, 7) == "--fp64=") {
            const std::string fp64_mode_str = arg.substr(7);
            if (fp64_mode_str == "native") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kNative;
            } else if (fp64_mode_str == "emulate") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
            } else if (fp64_mode_str == "warn") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kWarn;
            } else {
                std::cerr << "invalid --fp64 mode: " << fp64_mode_str
                          << " (valid: native, emulate, warn)\n";
                return 2;
            }
        } else if (arg == "--link") {
            link_executable = true;
            link_requested_explicitly = true;
        } else if (arg == "--no-link") {
            link_executable = false;
            link_requested_explicitly = true;
        } else if (arg == "--save-temps") {
            keep_intermediates = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            return 2;
        } else if (!positional_input_set && options.input.empty()) {
            options.input = arg;
            positional_input_set = true;
        } else {
            std::cerr << "unexpected positional argument: " << arg << "\n";
            return 2;
        }
    }

    if (options.input.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    // Pick the backend that actually works for this input rather than one global default. The two
    // are complementary, not ranked, and the split follows the frontend feeding them. Measured
    // over the 19-file in-tree .cu corpus (tests/ + samples/, see docs/compiler-architecture.md):
    //
    //   direct .cu           legacy 0/19   cumetal-ir 10/19
    //   --cuda-device (PTX)  legacy 17/19  cumetal-ir  6/19
    //
    // Legacy's direct-.cu mode is the qualifier-stripping prototype documented in
    // docs/known-gaps.md and lowers nothing in this corpus, so typed CuMetal IR is strictly
    // better there. Through the PTX frontend the ordering reverses and defaulting to typed IR
    // would regress the path llm.c, llama.cpp, and PhysX all depend on. --backend overrides.
    if (!backend_set_explicitly && lower_ext(options.input) == ".cu" && !cuda_device_frontend) {
        backend = BackendKind::kCumetalIr;
    }

    // `cumetalc foo.cu -o foo` builds an executable. Infer that from the shape of the request --
    // a .cu input, the default (metallib) emit stage, and an -o that does not name a .metallib --
    // so the nvcc-style invocation works without a special flag. --link/--no-link override.
    if (!link_requested_explicitly && lower_ext(options.input) == ".cu" &&
        emit_stage == EmitStage::kMetallib && !options.output.empty() &&
        lower_ext(options.output) != ".metallib") {
        link_executable = true;
    }

    if (link_executable) {
        if (lower_ext(options.input) != ".cu") {
            std::cerr << "cumetalc failed: --link requires a .cu input (got " << options.input
                      << ")\n";
            return 2;
        }
        if (options.output.empty()) {
            options.output = options.input;
            options.output.replace_extension();
        }
        ExecutableDriverOptions driver;
        driver.input = options.input;
        driver.output = options.output;
        driver.cuda_clang = cuda_clang;
        driver.cuda_arch = cuda_arch;
        driver.cuda_inline_threshold = cuda_inline_threshold;
        driver.include_dirs = cuda_include_dirs;
        driver.defines = cuda_defines;
        driver.forced_includes = cuda_forced_includes;
        driver.keep_intermediates = keep_intermediates;
        return run_executable_driver(driver, argv[0]);
    }

    if (options.output.empty()) {
        options.output = options.input;
        options.output.replace_extension(extension_for_stage(emit_stage));
    }

    if (!mode_set) {
        options.mode = cumetal::air_emitter::EmitMode::kXcrun;
    }

    std::vector<std::filesystem::path> temp_files;
    std::filesystem::path temp_stage_file;
    std::string abi_sidecar;
    std::string input_ext = lower_ext(options.input);
    if (input_ext == ".cu" && cuda_device_frontend) {
        const std::filesystem::path compiler = find_cuda_clang(cuda_clang);
        if (compiler.empty() || !std::filesystem::exists(compiler)) {
            std::cerr
                << "cumetalc failed: CUDA-capable clang++ not found; install Homebrew LLVM or "
                   "pass --cuda-clang/CUMETAL_CUDA_CLANG\n";
            return 1;
        }
        if (cuda_arch.size() < 4 || cuda_arch.substr(0, 3) != "sm_") {
            std::cerr << "cumetalc failed: --cuda-arch must use sm_XX form\n";
            return 2;
        }

        const std::filesystem::path input_cu = options.input;
        const std::filesystem::path runtime_api_dir =
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api";
        temp_stage_file = make_temp_path(".ptx");
        std::string command =
            quote_shell(compiler.string()) +
            " -x cuda --cuda-device-only -S -std=c++17 -O1 -fno-jump-tables"
            " --cuda-gpu-arch=" +
            quote_shell(cuda_arch) +
            " -Xclang -target-feature -Xclang +ptx70"
            " -nocudainc -nocudalib -Wno-unknown-cuda-version -Wno-pass-failed"
            " -D__CUDACC__=1 -D__NVCC__=1";
        if (!cuda_inline_threshold.empty()) {
            command += " -fgpu-inline-threshold=" + quote_shell(cuda_inline_threshold) +
                       " -mllvm -inline-all-viable-calls";
        }
        if (std::filesystem::exists(runtime_api_dir) &&
            std::filesystem::is_directory(runtime_api_dir)) {
            command += " -I " + quote_shell(runtime_api_dir.string()) +
                       " -include " + quote_shell("cuda_runtime.h");
        }
        for (const auto& include_dir : cuda_include_dirs) {
            command += " -I " + quote_shell(include_dir.string());
        }
        for (const auto& define : cuda_defines) {
            command += " -D " + quote_shell(define);
        }
        for (const auto& forced_include : cuda_forced_includes) {
            command += " -include " + quote_shell(forced_include.string());
        }
        command += " " + quote_shell(input_cu.string()) + " -o " +
                   quote_shell(temp_stage_file.string()) + " 2>&1";

        const CommandResult frontend_result = run_command_capture(command);
        if (!frontend_result.output.empty()) {
            std::cerr << frontend_result.output;
            if (frontend_result.output.back() != '\n') {
                std::cerr << '\n';
            }
        }
        if (!frontend_result.started || frontend_result.exit_code != 0 ||
            !std::filesystem::exists(temp_stage_file)) {
            std::error_code ec;
            std::filesystem::remove(temp_stage_file, ec);
            std::cerr << "cumetalc failed: CUDA device frontend compilation failed\n";
            return 1;
        }
        temp_files.push_back(temp_stage_file);
        options.input = temp_stage_file;
        input_ext = ".ptx";
        temp_stage_file.clear();
    }

    if (input_ext == ".ptx") {
        std::string io_error;
        const std::vector<std::uint8_t> ptx_bytes = cumetal::common::read_file_bytes(options.input, &io_error);
        if (ptx_bytes.empty()) {
            std::cerr << "cumetalc failed: "
                      << (io_error.empty() ? "failed to read PTX input" : io_error) << "\n";
            return 1;
        }

        const std::string ptx_source(reinterpret_cast<const char*>(ptx_bytes.data()), ptx_bytes.size());
        abi_sidecar = build_ptx_abi_sidecar(ptx_source, ptx_entry_name);

        if (emit_stage == EmitStage::kLlvm) {
            cumetal::ptx::LowerToLlvmOptions lower_options;
            lower_options.strict = ptx_strict;
            lower_options.entry_name = ptx_entry_name;
            lower_options.fp64_mode = ptx_fp64_mode;
            const auto lowered =
                cumetal::ptx::lower_ptx_to_llvm_ir(std::string_view(ptx_source), lower_options);
            if (!lowered.ok ||
                !write_text_output(options.output, lowered.llvm_ir, options.overwrite, &io_error)) {
                std::cerr << "cumetalc failed: "
                          << (!lowered.ok ? lowered.error : io_error) << "\n";
                return 1;
            }
            std::cout << "wrote " << options.output << "\n";
            return 0;
        }

        if (backend == BackendKind::kCumetalIr) {
            cumetal::metal::PtxToMslOptions compile_options;
            compile_options.strict = true;
            compile_options.entry_name = ptx_entry_name;
            compile_options.source_name = options.input.string();
            const auto compiled =
                cumetal::metal::compile_ptx_to_msl(ptx_source, compile_options);
            for (const std::string& warning : compiled.warnings) {
                std::cerr << "ptx warning: " << warning << "\n";
            }
            if (!compiled.ok) {
                std::cerr << "cumetalc failed: " << compiled.error << "\n";
                return 1;
            }
            if (emit_stage != EmitStage::kMetallib) {
                if (!emit_inspection_stage(compiled, emit_stage, options.output,
                                           options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: " << io_error << "\n";
                    return 1;
                }
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            temp_stage_file = make_temp_path(".metal");
            if (!write_text_output(temp_stage_file, compiled.source, true, &io_error)) {
                std::cerr << "cumetalc failed: failed to write temporary MSL: "
                          << io_error << "\n";
                return 1;
            }
            options.input = temp_stage_file;
            options.kernel_name = compiled.gpu_ir.functions.front().name;
            temp_files.push_back(temp_stage_file);
        } else {
            if (emit_stage == EmitStage::kCumetalIr ||
                emit_stage == EmitStage::kMetalIr) {
                std::cerr << "cumetalc failed: --emit="
                          << (emit_stage == EmitStage::kCumetalIr ? "cumetal-ir" : "metal-ir")
                          << " requires --backend=cumetal-ir\n";
                return 1;
            }
            cumetal::ptx::LowerToMetalOptions lower_to_metal_options;
            lower_to_metal_options.strict = ptx_strict;
            lower_to_metal_options.entry_name = ptx_entry_name;
            const auto lowered_metal =
                cumetal::ptx::lower_ptx_to_metal_source(
                    std::string_view(ptx_source), lower_to_metal_options);
            for (const auto& warning : lowered_metal.warnings) {
                std::cerr << "ptx warning: " << warning << "\n";
            }
            if (!lowered_metal.ok) {
                std::cerr << "cumetalc failed: PTX->Metal lowering failed: "
                          << lowered_metal.error << "\n";
                return 1;
            }
            if (emit_stage == EmitStage::kMsl) {
                if (!lowered_metal.matched ||
                    !write_text_output(options.output, lowered_metal.metal_source,
                                       options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: "
                              << (!lowered_metal.matched
                                      ? "legacy backend did not produce MSL"
                                      : io_error)
                              << "\n";
                    return 1;
                }
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            if (lowered_metal.matched && !lowered_metal.metal_source.empty()) {
                temp_stage_file = make_temp_path(".metal");
                if (!write_text_output(temp_stage_file, lowered_metal.metal_source,
                                       true, &io_error)) {
                    std::cerr << "cumetalc failed: failed to write temporary Metal source: "
                              << io_error << "\n";
                    return 1;
                }
                options.input = temp_stage_file;
                options.kernel_name = lowered_metal.entry_name;
                temp_files.push_back(temp_stage_file);
            } else {
                cumetal::ptx::LowerToLlvmOptions lower_options;
                lower_options.strict = ptx_strict;
                lower_options.entry_name = ptx_entry_name;
                lower_options.fp64_mode = ptx_fp64_mode;
                const auto lowered =
                    cumetal::ptx::lower_ptx_to_llvm_ir(
                        std::string_view(ptx_source), lower_options);
                if (!lowered.ok) {
                    std::cerr << "cumetalc failed: PTX lowering failed: "
                              << lowered.error << "\n";
                    return 1;
                }
                temp_stage_file = make_temp_path(".ll");
                if (!write_text_output(temp_stage_file, lowered.llvm_ir,
                                       true, &io_error)) {
                    std::cerr << "cumetalc failed: failed to write temporary LLVM IR: "
                              << io_error << "\n";
                    return 1;
                }
                options.input = temp_stage_file;
                options.kernel_name = lowered.entry_name;
                temp_files.push_back(temp_stage_file);
            }
        }
    } else if (input_ext == ".ll" || input_ext == ".llvm") {
        if (backend == BackendKind::kCumetalIr) {
            std::string io_error;
            const std::vector<std::uint8_t> bytes =
                cumetal::common::read_file_bytes(options.input, &io_error);
            if (bytes.empty()) {
                std::cerr << "cumetalc failed: " << io_error << "\n";
                return 1;
            }
            const std::string llvm_ir(bytes.begin(), bytes.end());
            const auto compiled =
                cumetal::metal::compile_nvvm_to_msl(llvm_ir, options.input.string(),
                                                    ptx_entry_name);
            if (!compiled.ok) {
                std::cerr << "cumetalc failed: " << compiled.error << "\n";
                return 1;
            }
            if (emit_stage == EmitStage::kLlvm) {
                if (!write_text_output(options.output, llvm_ir, options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: " << io_error << "\n";
                    return 1;
                }
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            if (emit_stage != EmitStage::kMetallib) {
                if (!emit_inspection_stage(compiled, emit_stage, options.output,
                                           options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: " << io_error << "\n";
                    return 1;
                }
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            temp_stage_file = make_temp_path(".metal");
            if (!write_text_output(temp_stage_file, compiled.source, true, &io_error)) {
                std::cerr << "cumetalc failed: " << io_error << "\n";
                return 1;
            }
            options.input = temp_stage_file;
            options.kernel_name = compiled.gpu_ir.functions.front().name;
        } else if (emit_stage != EmitStage::kMetallib) {
            std::cerr << "cumetalc failed: LLVM inspection stages require "
                         "--backend=cumetal-ir (or use the input file directly)\n";
            return 1;
        }
    } else if (input_ext == ".cu") {
        if (backend == BackendKind::kCumetalIr || emit_stage == EmitStage::kLlvm) {
            const std::filesystem::path clang = find_cuda_clang(cuda_clang);
            if (clang.empty()) {
                std::cerr << "cumetalc failed: stock Clang with CUDA support was not found; "
                             "set CUMETAL_CLANG\n";
                return 1;
            }
            if (!cumetal::ir::llvm_frontend_available()) {
                std::cerr << "cumetalc failed: CuMetal was built without LLVM IRReader support\n";
                return 1;
            }
            const std::filesystem::path original_input = options.input;
            const std::filesystem::path raw_device_ll =
                make_temp_path(".raw-device.ll");
            const std::filesystem::path device_ll = make_temp_path(".device.ll");
            const std::filesystem::path runtime_api_dir =
                std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api";
            const std::string arch = cuda_arch;
            const std::string ptx_feature = ptx_feature_for_arch(arch);
            std::string command =
                quote_shell(clang.string()) +
                " -x cuda --cuda-device-only -std=c++20 -O0 "
                "-Xclang -disable-O0-optnone -S -emit-llvm "
                "-gline-tables-only -nocudainc -nocudalib "
                "--cuda-gpu-arch=" + quote_shell(arch) + " ";
            if (!ptx_feature.empty()) {
                command += "-Xclang -target-feature -Xclang " +
                           quote_shell(ptx_feature) + " ";
            }
            command += "-D__CUDACC__=1 -D__NVCC__=1 -I " +
                       quote_shell(runtime_api_dir.string()) +
                       " -include cuda_runtime.h ";
            for (const auto& include_dir : cuda_include_dirs) {
                command += "-I " + quote_shell(include_dir.string()) + " ";
            }
            for (const auto& define : cuda_defines) {
                command += "-D " + quote_shell(define) + " ";
            }
            for (const auto& forced_include : cuda_forced_includes) {
                command += "-include " + quote_shell(forced_include.string()) + " ";
            }
            command += quote_shell(original_input.string()) + " -o " +
                       quote_shell(raw_device_ll.string()) + " 2>&1";
            const CommandResult clang_result = run_command_capture(command);
            if (!clang_result.started || clang_result.exit_code != 0) {
                if (!clang_result.output.empty()) std::cerr << clang_result.output;
                std::cerr << "cumetalc failed: Clang CUDA device compilation failed\n";
                return 1;
            }
            const std::filesystem::path llvm_opt = find_llvm_opt(clang);
            if (llvm_opt.empty()) {
                std::cerr << "cumetalc failed: LLVM opt is required for the "
                             "conservative device-IR normalization pipeline\n";
                return 1;
            }
            const std::string opt_command =
                quote_shell(llvm_opt.string()) +
                " -S -passes=sroa,mem2reg,dce,simplifycfg,lower-switch " +
                quote_shell(raw_device_ll.string()) + " -o " +
                quote_shell(device_ll.string()) + " 2>&1";
            const CommandResult opt_result = run_command_capture(opt_command);
            if (!opt_result.started || opt_result.exit_code != 0) {
                if (!opt_result.output.empty()) std::cerr << opt_result.output;
                std::cerr << "cumetalc failed: conservative LLVM device-IR "
                             "normalization failed\n";
                return 1;
            }
            std::string io_error;
            const std::vector<std::uint8_t> llvm_bytes =
                cumetal::common::read_file_bytes(device_ll, &io_error);
            if (llvm_bytes.empty()) {
                std::cerr << "cumetalc failed: " << io_error << "\n";
                return 1;
            }
            const std::string llvm_ir(llvm_bytes.begin(), llvm_bytes.end());
            if (emit_stage == EmitStage::kLlvm) {
                if (!write_text_output(options.output, llvm_ir, options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: " << io_error << "\n";
                    return 1;
                }
                std::error_code ec;
                std::filesystem::remove(raw_device_ll, ec);
                std::filesystem::remove(device_ll, ec);
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            const auto compiled = cumetal::metal::compile_nvvm_to_msl(
                llvm_ir, original_input.string(), ptx_entry_name);
            if (!compiled.ok) {
                std::cerr << "cumetalc failed: " << compiled.error << "\n";
                return 1;
            }
            if (emit_stage != EmitStage::kMetallib) {
                if (!emit_inspection_stage(compiled, emit_stage, options.output,
                                           options.overwrite, &io_error)) {
                    std::cerr << "cumetalc failed: " << io_error << "\n";
                    return 1;
                }
                std::error_code ec;
                std::filesystem::remove(raw_device_ll, ec);
                std::filesystem::remove(device_ll, ec);
                std::cout << "wrote " << options.output << "\n";
                return 0;
            }
            temp_stage_file = make_temp_path(".metal");
            if (!write_text_output(temp_stage_file, compiled.source, true, &io_error)) {
                std::cerr << "cumetalc failed: " << io_error << "\n";
                return 1;
            }
            std::error_code ec;
            std::filesystem::remove(raw_device_ll, ec);
            std::filesystem::remove(device_ll, ec);
            options.input = temp_stage_file;
            options.kernel_name = compiled.gpu_ir.functions.front().name;
        } else {
        if (!command_exists("xcrun")) {
            std::cerr << "cumetalc failed: xcrun is required for .cu frontend compilation\n";
            return 1;
        }
        if (!xcrun_tool_exists("clang++")) {
            std::cerr << "cumetalc failed: xcrun clang++ not available for .cu frontend compilation\n";
            return 1;
        }

        temp_stage_file = make_temp_path(".ll");
        const std::filesystem::path runtime_api_dir =
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "runtime" / "api";
        const std::string command =
            "xcrun clang++ -std=c++20 -S -emit-llvm -x c++ "
            "-D__global__= -D__host__= -D__device__= -D__shared__= -D__constant__= "
            "-D__managed__= " +
            ((std::filesystem::exists(runtime_api_dir) && std::filesystem::is_directory(runtime_api_dir))
                 ? ("-I " + quote_shell(runtime_api_dir.string()) + " ")
                 : "") +
            quote_shell(options.input.string()) + " -o " + quote_shell(temp_stage_file.string()) + " 2>&1";
        const CommandResult result = run_command_capture(command);
        if (!result.started || result.exit_code != 0) {
            if (!result.output.empty()) {
                std::cerr << result.output;
                if (result.output.back() != '\n') {
                    std::cerr << '\n';
                }
            }
            std::cerr << "cumetalc failed: .cu frontend compilation failed\n";
            return 1;
        }

        options.input = temp_stage_file;
        temp_files.push_back(temp_stage_file);
        }
    }

    const auto result = cumetal::air_emitter::emit_metallib(options);
    for (const auto& temp_file : temp_files) {
        std::error_code ec;
        std::filesystem::remove(temp_file, ec);
    }
    for (const auto& log : result.logs) {
        if (!log.empty()) {
            std::cerr << log;
            if (log.back() != '\n') {
                std::cerr << '\n';
            }
        }
    }

    if (!result.ok) {
        std::cerr << "cumetalc failed: " << result.error << "\n";
        return 1;
    }

    if (!abi_sidecar.empty()) {
        const std::filesystem::path abi_path = options.output.string() + ".cumetal-abi";
        const std::vector<std::uint8_t> abi_bytes(abi_sidecar.begin(), abi_sidecar.end());
        std::string abi_error;
        if (!cumetal::common::write_file_bytes(abi_path, abi_bytes, &abi_error)) {
            std::cerr << "cumetalc failed: unable to write kernel ABI sidecar: "
                      << abi_error << "\n";
            return 1;
        }
    }

    std::cout << "wrote " << result.output << "\n";
    return 0;
}
