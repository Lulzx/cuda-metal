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
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#ifndef CUMETAL_SOURCE_DIR
#define CUMETAL_SOURCE_DIR ""
#endif

#ifndef CUMETAL_VERSION_STRING
#define CUMETAL_VERSION_STRING "unknown"
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
        std::cerr << "cumetalc: PTX ABI sidecar not written: parse failed: " << parsed.error
                  << "\n";
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
    std::cerr << "cumetalc: PTX ABI sidecar not written: no entry named '" << requested_entry
              << "' in the parsed PTX\n";
    return {};
}

// The NVVM route (.cu -> clang LLVM IR -> MSL) never produces PTX, so
// build_ptx_abi_sidecar() never ran for it and the metallib shipped with no ABI
// sidecar. The driver then fell back to scanning kernelParams for a NULL
// terminator that CUDA never promises, walked off the end of the array, and
// segfaulted. The imported IR already carries the kernel's real ABI, so derive
// the sidecar from that instead.
std::string build_ir_abi_sidecar(const cumetal::ir::Module& module,
                                 const std::string& requested_entry) {
    for (const auto& function : module.functions) {
        if (!function.is_kernel || !function.kernel_abi.has_value()) {
            continue;
        }
        if (!requested_entry.empty() && function.name != requested_entry) {
            continue;
        }
        const cumetal::ir::KernelAbi& abi = *function.kernel_abi;
        std::string text = "CUMETAL_ABI_V1\nkernel " + function.name + "\n";
        text += "shared " + std::to_string(abi.static_threadgroup_memory) + "\n";
        for (const auto& argument : abi.arguments) {
            if (argument.hidden_role.has_value()) continue;
            if (argument.kind == cumetal::ir::ArgumentKind::kPointer) {
                text += "arg buffer 8\n";
                continue;
            }
            if (argument.size == 0 || argument.size > 4096) {
                // The runtime rejects these outright; emitting a sidecar the
                // runtime refuses is worse than emitting none.
                std::cerr << "cumetalc: PTX ABI sidecar not written: argument '"
                          << argument.name << "' of kernel '" << function.name
                          << "' has unusable size " << argument.size << "\n";
                return {};
            }
            text += "arg bytes " + std::to_string(argument.size) + "\n";
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
                 " [--fp64=fast48|wide48|ieee64|native|emulate|warn]\n";
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

bool clang_supports_inline_all_viable_calls(const std::filesystem::path& clang) {
    const CommandResult help = run_command_capture(
        quote_shell(clang.string()) + " -cc1 -mllvm --help-hidden 2>&1");
    return help.started &&
           help.output.find("-inline-all-viable-calls") != std::string::npos;
}

std::string cuda_inline_flags(const std::filesystem::path& clang,
                              const std::string& threshold) {
    if (threshold.empty()) return {};
    std::string flags = " -fgpu-inline-threshold=" + quote_shell(threshold);
    // LLVM 22 added the stronger all-viable-calls switch. Older supported
    // Clang versions reject it during option parsing, so retain their GPU
    // inlining threshold without passing an unknown backend option.
    if (clang_supports_inline_all_viable_calls(clang)) {
        flags += " -mllvm -inline-all-viable-calls";
    }
    return flags;
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

std::string llvm_lower_switch_pass(const std::filesystem::path& llvm_opt) {
    const CommandResult passes =
        run_command_capture(quote_shell(llvm_opt.string()) + " --print-passes 2>&1");
    if (passes.started && passes.output.find("lower-switch") != std::string::npos) {
        return "lower-switch";
    }
    // LLVM 18 used the unhyphenated new-pass-manager spelling.
    return "lowerswitch";
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
    BackendKind backend = BackendKind::kCumetalIr;
    cumetal::ptx::Fp64Mode fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    bool keep_intermediates = false;
};

int run_legacy_executable_driver(const ExecutableDriverOptions& options,
                                 const std::filesystem::path& compiler,
                                 const ResourceLayout& layout) {
    if (!std::filesystem::exists(layout.toolchain_dir / "fatbinary") ||
        !std::filesystem::exists(layout.toolchain_dir / "ptxas")) {
        std::cerr << "cumetalc failed: legacy executable mode requires the "
                     "ptxas/fatbinary compatibility shims\n";
        return 1;
    }
    const std::filesystem::path object = make_temp_path(".legacy.o");
    const char* existing_path = std::getenv("PATH");
    const std::string path = layout.toolchain_dir.string() + ":" +
        (existing_path != nullptr ? existing_path : "/usr/bin:/bin");
    std::string compile = "PATH=" + quote_shell(path) + " " +
        quote_shell(compiler.string()) + " -x cuda -std=c++17 -O2 --cuda-gpu-arch=" +
        quote_shell(options.cuda_arch) +
        (ptx_feature_for_arch(options.cuda_arch).empty()
             ? std::string{}
             : " --cuda-feature=" + ptx_feature_for_arch(options.cuda_arch)) +
        " -nocudainc -nocudalib -Wno-unknown-cuda-version -Wno-pass-failed"
        " -D__CUDACC__=1 -D__NVCC__=1 -I " +
        quote_shell(layout.include_dir.string()) + " -include cuda_runtime.h";
    compile += cuda_inline_flags(compiler, options.cuda_inline_threshold);
    for (const auto& dir : options.include_dirs) compile += " -I " + quote_shell(dir.string());
    for (const auto& define : options.defines) compile += " -D " + quote_shell(define);
    for (const auto& forced : options.forced_includes) {
        compile += " -include " + quote_shell(forced.string());
    }
    compile += " -c " + quote_shell(options.input.string()) + " -o " +
               quote_shell(object.string()) + " 2>&1";
    const CommandResult compiled = run_command_capture(compile);
    if (!compiled.output.empty()) std::cerr << compiled.output;
    if (!compiled.started || compiled.exit_code != 0 || !std::filesystem::exists(object)) {
        std::error_code ec;
        std::filesystem::remove(object, ec);
        return 1;
    }
    const std::string link = quote_shell(compiler.string()) + " " +
        quote_shell(object.string()) + " -L " + quote_shell(layout.lib_dir.string()) +
        " -lcumetal -Wl,-rpath," + quote_shell(layout.lib_dir.string()) + " -o " +
        quote_shell(options.output.string()) + " 2>&1";
    const CommandResult linked = run_command_capture(link);
    if (!linked.output.empty()) std::cerr << linked.output;
    if (!options.keep_intermediates) {
        std::error_code ec;
        std::filesystem::remove(object, ec);
    }
    return linked.started && linked.exit_code == 0 && std::filesystem::exists(options.output)
               ? 0
               : 1;
}

struct NativeHostKernel {
    std::string stub_symbol;
    std::string metal_name;
    std::vector<bool> pointer_arguments;
    std::vector<std::uint32_t> argument_sizes;
    std::vector<std::uint32_t> symbol_indices;
};

struct NativeSourceSymbol {
    std::string name;
    std::uint32_t size = 0;
    std::uint32_t alignment = 1;
    std::uint32_t constant_offset = 0;
    bool constant = false;
};

std::vector<NativeSourceSymbol> parse_native_source_symbols(
    std::string_view metal_source) {
    const std::regex record(
        R"(// cumetal-native-symbol: (constant|global) ([^ ]+) ([0-9]+) ([0-9]+) ([0-9]+))"
    );
    const std::string source(metal_source);
    std::vector<NativeSourceSymbol> symbols;
    for (std::sregex_iterator it(source.begin(), source.end(), record), end;
         it != end; ++it) {
        symbols.push_back({
            .name = (*it)[2].str(),
            .size = static_cast<std::uint32_t>(std::stoul((*it)[3].str())),
            .alignment = static_cast<std::uint32_t>(std::stoul((*it)[4].str())),
            .constant_offset = static_cast<std::uint32_t>(std::stoul((*it)[5].str())),
            .constant = (*it)[1].str() == "constant",
        });
    }
    return symbols;
}

std::string device_symbol_from_stub(std::string symbol) {
    static constexpr std::string_view marker = "__device_stub__";
    const std::size_t marker_at = symbol.find(marker);
    if (marker_at == std::string::npos) return {};
    if (marker_at == 0) return symbol.substr(marker.size());

    std::size_t digits_at = marker_at;
    while (digits_at > 0 &&
           std::isdigit(static_cast<unsigned char>(symbol[digits_at - 1])) != 0) {
        --digits_at;
    }
    if (digits_at == marker_at) return {};
    const std::uint64_t component_size =
        std::strtoull(symbol.substr(digits_at, marker_at - digits_at).c_str(),
                      nullptr, 10);
    if (component_size < marker.size()) return {};
    symbol.replace(digits_at, marker_at - digits_at,
                   std::to_string(component_size - marker.size()));
    symbol.erase(digits_at + std::to_string(component_size - marker.size()).size(),
                 marker.size());
    return symbol;
}

bool parse_native_host_kernels(std::string_view llvm_ir,
                               std::vector<NativeHostKernel>* kernels,
                               std::string* error) {
    const std::string source(llvm_ir);
    const std::regex header(
        R"(define[^\n@]*@([^ (\n]+)\(([^\n]*)\)[^{\n]*\{)"
    );
    const std::regex setup(
        R"(@cudaSetupArgument\(ptr [^,%]*?(%[-A-Za-z0-9._]+), i64 ([0-9]+), i64 ([0-9]+)\))"
    );
    for (std::sregex_iterator it(source.begin(), source.end(), header), end;
         it != end; ++it) {
        const std::string stub = (*it)[1].str();
        if (stub.find("__device_stub__") == std::string::npos) continue;
        const std::string metal_name = device_symbol_from_stub(stub);
        if (metal_name.empty()) {
            if (error != nullptr) *error = "cannot derive device symbol from host stub '" + stub + "'";
            return false;
        }
        const std::size_t body_begin =
            static_cast<std::size_t>((*it).position() + (*it).length());
        const std::size_t body_end = source.find("\n}", body_begin);
        if (body_end == std::string::npos) {
            if (error != nullptr) *error = "unterminated host stub '" + stub + "'";
            return false;
        }
        NativeHostKernel kernel{
            .stub_symbol = stub,
            .metal_name = metal_name,
        };
        const std::string body = source.substr(body_begin, body_end - body_begin);
        std::size_t setup_call_count = 0;
        for (std::size_t at = body.find("@cudaSetupArgument(");
             at != std::string::npos;
             at = body.find("@cudaSetupArgument(", at + 1)) {
            ++setup_call_count;
        }
        for (std::sregex_iterator setup_it(body.begin(), body.end(), setup), setup_end;
             setup_it != setup_end; ++setup_it) {
            const std::string storage = (*setup_it)[1].str();
            const auto size = std::stoull((*setup_it)[2].str());
            if (size == 0 || size > 64u * 1024u) {
                if (error != nullptr) *error = "invalid launch argument size in host stub '" + stub + "'";
                return false;
            }
            const std::regex pointer_storage(
                "(?:^|\\n)[[:space:]]*" + storage +
                R"([[:space:]]*=[[:space:]]*alloca ptr(?:,|\n))");
            kernel.pointer_arguments.push_back(
                std::regex_search(body, pointer_storage));
            kernel.argument_sizes.push_back(static_cast<std::uint32_t>(size));
        }
        if (kernel.argument_sizes.size() != setup_call_count) {
            if (error != nullptr) {
                *error = "cannot classify every cudaSetupArgument call in host stub '" +
                         stub + "'";
            }
            return false;
        }
        kernels->push_back(std::move(kernel));
    }
    if (kernels->empty()) {
        if (error != nullptr) *error = "CUDA host compilation produced no kernel launch stubs";
        return false;
    }
    return true;
}

std::uint32_t native_alignment(std::uint32_t size) {
    if (size % 8 == 0) return 8;
    if (size % 4 == 0) return 4;
    if (size % 2 == 0) return 2;
    return 1;
}

std::string native_registration_source(
    const std::vector<NativeHostKernel>& kernels,
    const std::vector<NativeSourceSymbol>& symbols,
    const std::vector<std::uint8_t>& metallib,
    std::string_view provenance,
    std::string_view semantic_quality) {
    std::ostringstream out;
    out << "#include <cumetal_native.h>\n#include <cstdlib>\n\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        out << "extern \"C\" void cm_stub_" << i << "() asm(\""
            << '_' << kernels[i].stub_symbol << "\");\n";
    }
    for (std::size_t i = 0; i < symbols.size(); ++i) {
        out << "extern \"C\" unsigned char cm_symbol_" << i << "[] asm(\"_"
            << symbols[i].name << "\");\n";
    }
    out << "\nstatic const unsigned char cm_metallib[] = {";
    for (std::size_t i = 0; i < metallib.size(); ++i) {
        if (i % 16 == 0) out << "\n  ";
        out << static_cast<unsigned>(metallib[i]) << ',';
    }
    out << "\n};\n";

    std::size_t binding_base = 0;
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        if (kernels[i].argument_sizes.empty()) continue;
        out << "static const CuMetalArgumentDescriptor cm_args_" << i << "[] = {\n";
        for (std::size_t a = 0; a < kernels[i].argument_sizes.size(); ++a) {
            const std::uint32_t size = kernels[i].argument_sizes[a];
            out << "  {" << (kernels[i].pointer_arguments[a]
                                  ? "CUMETAL_NATIVE_ARGUMENT_POINTER"
                                  : "CUMETAL_NATIVE_ARGUMENT_SCALAR")
                << ',' << size << ',' << native_alignment(size) << ','
                << (kernels[i].pointer_arguments[a]
                        ? "CUMETAL_NATIVE_ADDRESS_DEVICE"
                        : "CUMETAL_NATIVE_ADDRESS_NONE")
                << ',' << (binding_base + a) << ",1},\n";
        }
        out << "};\n";
        binding_base += kernels[i].argument_sizes.size();
    }

    out << "static const CuMetalBindingDescriptor cm_bindings[] = {\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        for (std::size_t a = 0; a < kernels[i].argument_sizes.size(); ++a) {
            const std::uint32_t size = kernels[i].argument_sizes[a];
            out << "  {" << (kernels[i].pointer_arguments[a]
                                  ? "CUMETAL_NATIVE_BINDING_BUFFER"
                                  : "CUMETAL_NATIVE_BINDING_BYTES")
                << ',' << a << ',' << a << ',' << size << ','
                << native_alignment(size) << "},\n";
        }
    }
    out << "};\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        if (!kernels[i].symbol_indices.empty()) {
            out << "static const uint32_t cm_kernel_symbols_" << i << "[] = {";
            for (const std::uint32_t symbol : kernels[i].symbol_indices) {
                out << symbol << ',';
            }
            out << "};\n";
        }
    }
    out << "static const CuMetalKernelDescriptor cm_kernels[] = {\n";
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        out << "  {\"" << kernels[i].metal_name << "\",\""
            << kernels[i].metal_name << "\",reinterpret_cast<const void*>(&cm_stub_"
            << i << ")," << kernels[i].argument_sizes.size() << ','
            << (kernels[i].argument_sizes.empty() ? "nullptr" : "cm_args_" + std::to_string(i))
            << ",0,32," << kernels[i].symbol_indices.size() << ','
            << (kernels[i].symbol_indices.empty()
                    ? "nullptr"
                    : "cm_kernel_symbols_" + std::to_string(i))
            << "},\n";
    }
    out << "};\n";
    if (!symbols.empty()) {
        out << "static const CuMetalSymbolDescriptor cm_symbols[] = {\n";
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            out << "  {\"" << symbols[i].name << "\",cm_symbol_" << i << ','
                << symbols[i].size << ',' << symbols[i].alignment << ','
                << symbols[i].constant_offset << ','
                << (symbols[i].constant ? "CUMETAL_NATIVE_SYMBOL_CONSTANT"
                                        : "CUMETAL_NATIVE_SYMBOL_GLOBAL")
                << "},\n";
        }
        out << "};\n";
    }
    out << "static CuMetalModuleHandle cm_module;\n"
           "__attribute__((constructor)) static void cm_register_module() {\n"
           "  const CuMetalModuleDescriptor descriptor = {"
        << "CUMETAL_NATIVE_ABI_VERSION,cm_metallib,sizeof(cm_metallib),"
        << kernels.size() << ",cm_kernels," << binding_base << ','
        << (binding_base == 0 ? "nullptr" : "cm_bindings") << ",\""
        << provenance << "\",\"" << semantic_quality << "\"," << symbols.size()
        << ',' << (symbols.empty() ? "nullptr" : "cm_symbols") << "};\n"
           "  cm_module = cumetalRegisterModule(&descriptor);\n"
           "  if (cm_module == nullptr) std::abort();\n"
           "}\n"
           "__attribute__((destructor)) static void cm_unregister_module() {\n"
           "  if (cm_module != nullptr) cumetalUnregisterModule(cm_module);\n"
           "}\n";
    return out.str();
}

// Compile a complete CUDA translation unit into a native-AOT executable. Clang emits host-only
// launch stubs, the typed direct frontend emits the metallib ahead of time, and a generated
// constructor registers those two halves through cumetal_native.h. The result contains neither a
// fatbinary nor an unresolved __cudaRegister* dependency and performs no first-launch PTX JIT.
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
    if (options.backend == BackendKind::kLegacy) {
        return run_legacy_executable_driver(options, compiler, layout);
    }
    const std::filesystem::path object_file = make_temp_path(".o");
    const std::filesystem::path host_llvm = make_temp_path(".host.ll");
    const std::filesystem::path metal_source = make_temp_path(".metal");
    const std::filesystem::path metallib = make_temp_path(".metallib");
    const std::filesystem::path registration_source = make_temp_path(".native.cpp");
    const std::filesystem::path registration_object = make_temp_path(".native.o");
    const std::vector<std::filesystem::path> intermediates = {
        object_file, host_llvm, metal_source, metallib,
        registration_source, registration_object,
    };
    const auto cleanup = [&]() {
        if (options.keep_intermediates) {
            for (const auto& path : intermediates) {
                if (std::filesystem::exists(path)) {
                    std::cerr << "cumetalc: kept intermediate " << path << "\n";
                }
            }
            return;
        }
        for (const auto& path : intermediates) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    };

    std::string host_flags = quote_shell(compiler.string()) +
                          " -x cuda --cuda-host-only -std=c++17"
                          " --cuda-gpu-arch=" +
                          quote_shell(options.cuda_arch) +
                          " -nocudainc -nocudalib -Wno-unknown-cuda-version -Wno-pass-failed"
                          " -D__CUDACC__=1 -D__NVCC__=1"
                          " -I " +
                          quote_shell(layout.include_dir.string()) + " -include " +
                          quote_shell("cuda_runtime.h");
    host_flags += cuda_inline_flags(compiler, options.cuda_inline_threshold);
    for (const auto& dir : options.include_dirs) {
        host_flags += " -I " + quote_shell(dir.string());
    }
    for (const auto& define : options.defines) {
        host_flags += " -D " + quote_shell(define);
    }
    for (const auto& forced : options.forced_includes) {
        host_flags += " -include " + quote_shell(forced.string());
    }
    const std::string compile = host_flags + " -O2 -c " +
        quote_shell(options.input.string()) + " -o " +
        quote_shell(object_file.string()) + " 2>&1";

    const CommandResult compile_result = run_command_capture(compile);
    if (!compile_result.output.empty()) {
        std::cerr << compile_result.output;
        if (compile_result.output.back() != '\n') std::cerr << '\n';
    }
    if (!compile_result.started || compile_result.exit_code != 0 ||
        !std::filesystem::exists(object_file)) {
        cleanup();
        std::cerr << "cumetalc failed: CUDA compilation of " << options.input << " failed\n";
        return 1;
    }

    const std::string host_ir_command = host_flags +
        " -O0 -Xclang -disable-O0-optnone -S -emit-llvm " +
        quote_shell(options.input.string()) + " -o " + quote_shell(host_llvm.string()) +
        " 2>&1";
    const CommandResult host_ir_result = run_command_capture(host_ir_command);
    if (!host_ir_result.started || host_ir_result.exit_code != 0) {
        if (!host_ir_result.output.empty()) std::cerr << host_ir_result.output;
        cleanup();
        std::cerr << "cumetalc failed: could not inspect native host launch stubs\n";
        return 1;
    }
    std::string io_error;
    const auto host_llvm_bytes = cumetal::common::read_file_bytes(host_llvm, &io_error);
    std::vector<NativeHostKernel> kernels;
    if (!io_error.empty() ||
        !parse_native_host_kernels(
            std::string_view(reinterpret_cast<const char*>(host_llvm_bytes.data()),
                             host_llvm_bytes.size()),
            &kernels, &io_error)) {
        cleanup();
        std::cerr << "cumetalc failed: " << io_error << "\n";
        return 1;
    }

    const std::filesystem::path self = executable_path(argv0);
    if (self.empty()) {
        cleanup();
        std::cerr << "cumetalc failed: cannot locate compiler executable for native AOT\n";
        return 1;
    }
    std::string device_compile = quote_shell(self.string()) + " " +
        quote_shell(options.input.string()) +
        " --backend=cumetal-ir --emit=msl --no-link --overwrite --cuda-clang " +
        quote_shell(compiler.string()) + " --cuda-arch " +
        quote_shell(options.cuda_arch) + " --fp64=" +
        std::string(cumetal::ptx::fp64_mode_name(options.fp64_mode));
    if (!options.cuda_inline_threshold.empty()) {
        device_compile += " --cuda-inline-threshold " +
                          quote_shell(options.cuda_inline_threshold);
    }
    for (const auto& dir : options.include_dirs) {
        device_compile += " -I " + quote_shell(dir.string());
    }
    for (const auto& define : options.defines) {
        device_compile += " -D " + quote_shell(define);
    }
    for (const auto& forced : options.forced_includes) {
        device_compile += " -include " + quote_shell(forced.string());
    }
    device_compile += " -o " + quote_shell(metal_source.string()) + " 2>&1";
    const CommandResult device_result = run_command_capture(device_compile);
    if (!device_result.started || device_result.exit_code != 0) {
        if (!device_result.output.empty()) std::cerr << device_result.output;
        cleanup();
        std::cerr << "cumetalc failed: native AOT device compilation failed\n";
        return 1;
    }
    const auto metal_bytes = cumetal::common::read_file_bytes(metal_source, &io_error);
    if (!io_error.empty()) {
        cleanup();
        std::cerr << "cumetalc failed: " << io_error << "\n";
        return 1;
    }
    const std::string metal_text(metal_bytes.begin(), metal_bytes.end());
    const std::vector<NativeSourceSymbol> symbols =
        parse_native_source_symbols(metal_text);
    for (NativeHostKernel& kernel : kernels) {
        const std::string kernel_prefix = "kernel void " + kernel.metal_name + "(";
        const std::size_t signature_begin = metal_text.find(kernel_prefix);
        if (signature_begin ==
            std::string::npos) {
            cleanup();
            std::cerr << "cumetalc failed: host stub '" << kernel.stub_symbol
                      << "' has no typed Metal kernel '" << kernel.metal_name << "'\n";
            return 1;
        }
        const std::size_t signature_end = metal_text.find(") {", signature_begin);
        if (signature_end == std::string::npos) {
            cleanup();
            std::cerr << "cumetalc failed: malformed typed Metal kernel signature for '"
                      << kernel.metal_name << "'\n";
            return 1;
        }
        const std::string_view signature(
            metal_text.data() + signature_begin, signature_end - signature_begin);
        const bool uses_constants =
            signature.find("cm___cumetal_constant_symbols") != std::string_view::npos;
        for (std::size_t i = 0; i < symbols.size(); ++i) {
            if ((symbols[i].constant && uses_constants) ||
                (!symbols[i].constant &&
                 signature.find("cm___cumetal_global_" + symbols[i].name) !=
                     std::string_view::npos)) {
                kernel.symbol_indices.push_back(static_cast<std::uint32_t>(i));
            }
        }
    }
    if (!symbols.empty()) {
        std::string rewritten_host(host_llvm_bytes.begin(), host_llvm_bytes.end());
        for (const NativeSourceSymbol& symbol : symbols) {
            const std::string internal = "@" + symbol.name + " = internal global";
            const std::size_t at = rewritten_host.find(internal);
            if (at == std::string::npos) {
                cleanup();
                std::cerr << "cumetalc failed: native CUDA symbol '" << symbol.name
                          << "' has no host shadow with supported linkage\n";
                return 1;
            }
            rewritten_host.replace(at, internal.size(),
                                   "@" + symbol.name + " = global");
        }
        if (!write_text_output(host_llvm, rewritten_host, true, &io_error)) {
            cleanup();
            std::cerr << "cumetalc failed: " << io_error << "\n";
            return 1;
        }
        const std::string recompile_host = quote_shell(compiler.string()) +
            " -O2 -c " + quote_shell(host_llvm.string()) + " -o " +
            quote_shell(object_file.string()) + " 2>&1";
        const CommandResult recompiled = run_command_capture(recompile_host);
        if (!recompiled.started || recompiled.exit_code != 0) {
            if (!recompiled.output.empty()) std::cerr << recompiled.output;
            cleanup();
            std::cerr << "cumetalc failed: native symbol host-linkage rewrite failed\n";
            return 1;
        }
    }

    cumetal::air_emitter::EmitOptions emit;
    emit.input = metal_source;
    emit.output = metallib;
    emit.mode = cumetal::air_emitter::EmitMode::kXcrun;
    emit.overwrite = true;
    emit.validate_output = true;
    if (metal_text.find("cm_fp64_") != std::string::npos) {
        emit.textual_include_inputs.push_back(
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" / "metal" /
            "support" / "cumetal_fp64_inline_support.metal");
    }
    const auto emitted = cumetal::air_emitter::emit_metallib(emit);
    if (!emitted.ok) {
        cleanup();
        std::cerr << "cumetalc failed: " << emitted.error << "\n";
        return 1;
    }
    const auto metallib_bytes = cumetal::common::read_file_bytes(metallib, &io_error);
    if (!io_error.empty()) {
        cleanup();
        std::cerr << "cumetalc failed: " << io_error << "\n";
        return 1;
    }
    const auto comment_value = [&](std::string_view key, std::string_view fallback) {
        const std::size_t at = metal_text.find(key);
        if (at == std::string::npos) return std::string(fallback);
        const std::size_t begin = at + key.size();
        const std::size_t end = metal_text.find('\n', begin);
        return metal_text.substr(begin, end == std::string::npos ? end : end - begin);
    };
    const std::string generated = native_registration_source(
        kernels, symbols, metallib_bytes,
        comment_value("// cumetal-provenance: ", "generic_nvvm_lowering"),
        comment_value("// cumetal-semantic-quality: ", "unsupported"));
    if (!write_text_output(registration_source, generated, true, &io_error)) {
        cleanup();
        std::cerr << "cumetalc failed: " << io_error << "\n";
        return 1;
    }
    const std::string registration_compile = quote_shell(compiler.string()) +
        " -std=c++17 -O2 -I " + quote_shell(layout.include_dir.string()) + " -c " +
        quote_shell(registration_source.string()) + " -o " +
        quote_shell(registration_object.string()) + " 2>&1";
    const CommandResult registration_result = run_command_capture(registration_compile);
    if (!registration_result.started || registration_result.exit_code != 0) {
        if (!registration_result.output.empty()) std::cerr << registration_result.output;
        cleanup();
        std::cerr << "cumetalc failed: native registration compilation failed\n";
        return 1;
    }

    // Link with an rpath so the produced binary runs without the caller exporting
    // DYLD_LIBRARY_PATH first.
    const std::string link = quote_shell(compiler.string()) + " " +
                             quote_shell(object_file.string()) + " " +
                             quote_shell(registration_object.string()) + " -L " +
                             quote_shell(layout.lib_dir.string()) + " -lcumetal -Wl,-rpath," +
                             quote_shell(layout.lib_dir.string()) + " -o " +
                             quote_shell(options.output.string()) + " 2>&1";

    const CommandResult link_result = run_command_capture(link);
    if (!link_result.output.empty()) {
        std::cerr << link_result.output;
        if (link_result.output.back() != '\n') std::cerr << '\n';
    }

    cleanup();

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
    bool fp64_mode_set_explicitly = false;
    bool needs_vf64_support = false;
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
            } else if (fp64_mode_str == "emulate" || fp64_mode_str == "fast48") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
            } else if (fp64_mode_str == "wide48") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kWide48;
            } else if (fp64_mode_str == "ieee64") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kIEEE64;
            } else if (fp64_mode_str == "warn") {
                ptx_fp64_mode = cumetal::ptx::Fp64Mode::kWarn;
            } else {
                std::cerr << "invalid --fp64 mode: " << fp64_mode_str
                          << " (valid: fast48, wide48, ieee64, native, emulate, warn)\n";
                return 2;
            }
            fp64_mode_set_explicitly = true;
        } else if (arg == "--link") {
            link_executable = true;
            link_requested_explicitly = true;
        } else if (arg == "--no-link") {
            link_executable = false;
            link_requested_explicitly = true;
        } else if (arg == "--save-temps") {
            keep_intermediates = true;
        } else if (arg == "--version" || arg == "-v") {
            std::cout << "cumetalc " << CUMETAL_VERSION_STRING << "\n";
            return 0;
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
    // over the manifest-controlled 23-file source/sample corpus (see
    // tests/cuda_projects/backend_matrix_manifest.txt and docs/compiler-architecture.md):
    //
    //   direct .cu           legacy 0/23   cumetal-ir 22/23
    //   --cuda-device (PTX)  legacy 23/23  cumetal-ir 22/23
    //
    // These are production-metallib compilation counts, not runtime correctness
    // counts. Legacy's direct-.cu mode is the qualifier-stripping prototype documented in
    // docs/known-gaps.md and lowers nothing in this corpus, so typed CuMetal IR is strictly
    // better there. Through the PTX frontend the ordering reverses and defaulting to typed IR
    // would regress the path llm.c, llama.cpp, and PhysX all depend on. --backend overrides.
    if (!backend_set_explicitly && lower_ext(options.input) == ".cu" && !cuda_device_frontend) {
        backend = BackendKind::kCumetalIr;
    }
    // Direct source compilation selects the typed backend by default, so give
    // it the same usable software-FP64 default as runtime/JIT registration.
    // PTX/offline compatibility retains the historical native default, and an
    // explicit --fp64 policy always wins.
    if (!fp64_mode_set_explicitly && backend == BackendKind::kCumetalIr &&
        lower_ext(options.input) == ".cu" && !cuda_device_frontend) {
        ptx_fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
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
        driver.backend = backend;
        driver.fp64_mode = ptx_fp64_mode;
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
            " -ftrivial-auto-var-init=zero"
            " --cuda-gpu-arch=" +
            quote_shell(cuda_arch) +
            " -Xclang -target-feature -Xclang +ptx70"
            " -nocudainc -nocudalib -Wno-unknown-cuda-version -Wno-pass-failed"
            " -D__CUDACC__=1 -D__NVCC__=1";
        command += cuda_inline_flags(compiler, cuda_inline_threshold);
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
            compile_options.fp64_mode =
                std::string(cumetal::ptx::fp64_mode_name(ptx_fp64_mode));
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
            needs_vf64_support = ptx_source.find(".f64") != std::string::npos;
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
                needs_vf64_support =
                    ptx_source.find(".f64") != std::string::npos &&
                cumetal::ptx::fp64_mode_links_vf64_support(ptx_fp64_mode);
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
                    ptx_entry_name, cumetal::ptx::fp64_mode_name(ptx_fp64_mode));
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
            needs_vf64_support = llvm_ir.find("double") != std::string::npos;
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
                " -S -passes=sroa,mem2reg,dce,simplifycfg," +
                llvm_lower_switch_pass(llvm_opt) + " " +
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
                llvm_ir, original_input.string(), ptx_entry_name,
                cumetal::ptx::fp64_mode_name(ptx_fp64_mode));
            if (!compiled.ok) {
                std::cerr << "cumetalc failed: " << compiled.error << "\n";
                return 1;
            }
            abi_sidecar = build_ir_abi_sidecar(compiled.gpu_ir, ptx_entry_name);
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
            needs_vf64_support = llvm_ir.find("double") != std::string::npos;
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

    if (needs_vf64_support) {
        const bool typed_msl = options.input.extension() == ".metal";
        auto& support_inputs = typed_msl ? options.textual_include_inputs
                                         : options.additional_link_inputs;
        support_inputs.push_back(
            std::filesystem::path(CUMETAL_SOURCE_DIR) / "compiler" / "metal" /
            "support" / (typed_msl ? "cumetal_fp64_inline_support.metal"
                                     : "cumetal_fp64_support.metal"));
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
