#include "nvrtc.h"

#include "cuda.h"
#include "cumetal_diag.h"
#include "nvPTXCompiler.h"
#include "nvrtc_options.h"
#include "../cache/module_cache.h"

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// NVRTC shim: runtime compilation of CUDA C++ by driving `cumetalc`.
//
// The substitution rests on one property of CuMetal's driver: `cuModuleLoadDataEx`
// recognises a Metal library by its MTLB magic, so the bytes NVRTC calls a CUBIN
// can be a metallib and every caller that pipes NVRTC output straight into the
// driver keeps working untouched.
//
// Compilation runs out of process because the CUDA source path is Clang-driven
// and only `cumetalc` knows how to assemble that invocation. That costs a fork
// per compile, which runtime compilation callers already amortise behind their
// own module caches.
//
// The PTX surface is honest about what it cannot do: `cumetalc` lowers CUDA
// source to AIR, never to PTX, so asking for a virtual architecture fails at
// compile time with an explanation rather than at load time with a confusing
// module error.

namespace {

constexpr unsigned int kProgramMagic = 0x6e727463u;  // 'nrtc'

struct Program {
    unsigned int magic = kProgramMagic;
    std::string name;
    std::string source;
    std::string log;
    std::vector<char> cubin;
    bool compiled = false;
    // include name -> contents, as handed to nvrtcCreateProgram.
    std::vector<std::pair<std::string, std::string>> headers;
    std::vector<std::string> name_expressions;
};

Program* as_program(nvrtcProgram prog) {
    auto* program = reinterpret_cast<Program*>(prog);
    if (program == nullptr || program->magic != kProgramMagic) return nullptr;
    return program;
}

std::string quote_shell(const std::string& value) {
    std::string quoted = "'";
    for (const char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

struct CommandResult {
    bool started = false;
    int exit_code = 1;
    std::string output;
};

CommandResult run_command_capture(const std::string& command) {
    CommandResult result;
    std::array<char, 512> buffer{};
    FILE* pipe = ::popen(command.c_str(), "r");
    if (pipe == nullptr) return result;
    result.started = true;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        result.output.append(buffer.data());
    }
    const int status = ::pclose(pipe);
    result.exit_code = (status == -1) ? 1 : (WIFEXITED(status) ? WEXITSTATUS(status) : 1);
    return result;
}

bool is_executable(const std::filesystem::path& path) {
    return !path.empty() && ::access(path.c_str(), X_OK) == 0;
}

// `cumetalc` sits next to the loaded dylib in a build tree and one level up in
// an installed prefix. Resolving relative to this image rather than through PATH
// keeps a program that dlopen'd a specific CuMetal build compiling with that
// same build's compiler.
std::filesystem::path image_directory() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void*>(&kProgramMagic), &info) == 0 ||
        info.dli_fname == nullptr) {
        return {};
    }
    std::error_code ec;
    const auto resolved = std::filesystem::weakly_canonical(info.dli_fname, ec);
    if (ec) return {};
    return resolved.parent_path();
}

std::filesystem::path search_path_for_cumetalc() {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr) return {};
    const std::string path_value(path_env);
    std::size_t start = 0;
    while (start <= path_value.size()) {
        const std::size_t end = path_value.find(':', start);
        const std::string entry = path_value.substr(
            start, end == std::string::npos ? std::string::npos : end - start);
        if (!entry.empty()) {
            const std::filesystem::path candidate = std::filesystem::path(entry) / "cumetalc";
            if (is_executable(candidate)) return candidate;
        }
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return {};
}

const std::filesystem::path& cumetalc_path() {
    static const std::filesystem::path path = [] {
        if (const char* override_path = std::getenv("CUMETAL_NVRTC_COMPILER");
            override_path != nullptr && override_path[0] != '\0') {
            return std::filesystem::path(override_path);
        }
        if (const char* root = std::getenv("CUMETAL_ROOT");
            root != nullptr && root[0] != '\0') {
            const std::filesystem::path candidate =
                std::filesystem::path(root) / "bin" / "cumetalc";
            if (is_executable(candidate)) return candidate;
        }
        const std::filesystem::path image_dir = image_directory();
        if (!image_dir.empty()) {
            for (const std::filesystem::path& candidate :
                 {image_dir / "cumetalc", image_dir.parent_path() / "bin" / "cumetalc"}) {
                if (is_executable(candidate)) return candidate;
            }
        }
        return search_path_for_cumetalc();
    }();
    return path;
}

// A program name arrives from the caller and lands on the filesystem, so keep it
// to a leaf name made of characters that need no quoting beyond what
// quote_shell already does.
std::string sanitize_stem(const std::string& name) {
    std::string stem = std::filesystem::path(name).filename().string();
    if (stem == "." || stem == "..") stem.clear();
    for (char& c : stem) {
        const bool safe = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                          (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (!safe) c = '_';
    }
    if (stem.empty()) stem = "nvrtc_program";
    const std::filesystem::path as_path(stem);
    if (as_path.extension() == ".cu") stem = as_path.stem().string();
    return stem;
}

std::filesystem::path make_workspace(const std::string& stem) {
    static std::atomic<unsigned long long> counter{0};
    const auto sequence = counter.fetch_add(1, std::memory_order_relaxed);
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("cumetal-nvrtc-" + std::to_string(static_cast<long long>(::getpid())) + "-" +
         std::to_string(sequence) + "-" + stem);
    std::error_code ec;
    std::filesystem::create_directories(directory, ec);
    if (ec) return {};
    return directory;
}

bool read_file_bytes(const std::filesystem::path& path, std::vector<char>* out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return false;
    stream.seekg(0, std::ios::end);
    const std::streamoff size = stream.tellg();
    if (size < 0) return false;
    stream.seekg(0, std::ios::beg);
    out->resize(static_cast<std::size_t>(size));
    if (size > 0 && !stream.read(out->data(), size)) return false;
    return true;
}

nvrtcResult copy_out(const std::vector<char>& source, char* destination) {
    if (destination == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    if (!source.empty()) std::memcpy(destination, source.data(), source.size());
    return NVRTC_SUCCESS;
}

// nvPTXCompiler's "compiled program" is CuMetal's input PTX: the driver's module
// loader parses PTX itself, so the pass-through keeps the caller's later
// cuModuleLoadDataEx working.
struct PtxCompiler {
    unsigned int magic = kProgramMagic;
    std::string ptx;
    std::string error_log;
    std::string info_log;
    bool compiled = false;
};

PtxCompiler* as_ptx_compiler(nvPTXCompilerHandle handle) {
    auto* compiler = reinterpret_cast<PtxCompiler*>(handle);
    if (compiler == nullptr || compiler->magic != kProgramMagic) return nullptr;
    return compiler;
}

nvPTXCompileResult copy_log(const std::string& log, char* destination) {
    if (destination == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    std::memcpy(destination, log.c_str(), log.size() + 1);
    return NVPTXCOMPILE_SUCCESS;
}

}  // namespace

extern "C" {

const char* nvrtcGetErrorString(nvrtcResult result) {
    switch (result) {
        case NVRTC_SUCCESS:
            return "NVRTC_SUCCESS";
        case NVRTC_ERROR_OUT_OF_MEMORY:
            return "NVRTC_ERROR_OUT_OF_MEMORY";
        case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
            return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
        case NVRTC_ERROR_INVALID_INPUT:
            return "NVRTC_ERROR_INVALID_INPUT";
        case NVRTC_ERROR_INVALID_PROGRAM:
            return "NVRTC_ERROR_INVALID_PROGRAM";
        case NVRTC_ERROR_INVALID_OPTION:
            return "NVRTC_ERROR_INVALID_OPTION";
        case NVRTC_ERROR_COMPILATION:
            return "NVRTC_ERROR_COMPILATION";
        case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
            return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
        case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
            return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
        case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
            return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
        case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
            return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
        case NVRTC_ERROR_INTERNAL_ERROR:
            return "NVRTC_ERROR_INTERNAL_ERROR";
        case NVRTC_ERROR_TIME_FILE_WRITE_FAILED:
            return "NVRTC_ERROR_TIME_FILE_WRITE_FAILED";
    }
    return "NVRTC_ERROR_INTERNAL_ERROR";
}

nvrtcResult nvrtcVersion(int* major, int* minor) {
    if (major == nullptr || minor == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    // Match the CUDA version CuMetal's headers claim, so a caller comparing
    // NVRTC against CUDA_VERSION sees a consistent toolkit.
    *major = CUDA_VERSION / 1000;
    *minor = (CUDA_VERSION % 1000) / 10;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetNumSupportedArchs(int* numArchs) {
    if (numArchs == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    *numArchs = 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetSupportedArchs(int* supportedArchs) {
    if (supportedArchs == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    supportedArchs[0] = cumetal::nvrtc::kSupportedArch;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCreateProgram(nvrtcProgram* prog,
                               const char* src,
                               const char* name,
                               int numHeaders,
                               const char* const* headers,
                               const char* const* includeNames) {
    if (prog == nullptr || src == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    if (numHeaders < 0) return NVRTC_ERROR_INVALID_INPUT;
    if (numHeaders > 0 && (headers == nullptr || includeNames == nullptr)) {
        return NVRTC_ERROR_INVALID_INPUT;
    }

    auto program = std::make_unique<Program>();
    program->name = (name != nullptr && name[0] != '\0') ? name : "default_program";
    program->source = src;

    // NVRTC's header table is in memory; Clang's is on disk. Keep the pairs here
    // and write them into the compile workspace, so a quoted include resolves
    // the way the caller expects.
    for (int i = 0; i < numHeaders; ++i) {
        if (headers[i] == nullptr || includeNames[i] == nullptr) {
            return NVRTC_ERROR_INVALID_INPUT;
        }
        program->headers.emplace_back(includeNames[i], headers[i]);
    }

    *prog = reinterpret_cast<nvrtcProgram>(program.release());
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcDestroyProgram(nvrtcProgram* prog) {
    if (prog == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    Program* program = as_program(*prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    program->magic = 0;
    delete program;
    *prog = nullptr;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcCompileProgram(nvrtcProgram prog, int numOptions, const char* const* options) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (numOptions < 0 || (numOptions > 0 && options == nullptr)) {
        return NVRTC_ERROR_INVALID_INPUT;
    }

    program->compiled = false;
    program->cubin.clear();
    std::string log;

    std::vector<std::string> option_strings;
    option_strings.reserve(static_cast<std::size_t>(numOptions));
    for (int i = 0; i < numOptions; ++i) {
        if (options[i] == nullptr) return NVRTC_ERROR_INVALID_INPUT;
        option_strings.emplace_back(options[i]);
    }

    const cumetal::nvrtc::TranslatedOptions translated =
        cumetal::nvrtc::translate_options(option_strings);
    for (const std::string& option : translated.ignored) {
        log += "cumetal: ignoring NVRTC option with no Metal equivalent: " + option + "\n";
    }
    for (const std::string& option : translated.unrecognized) {
        log += "cumetal: unrecognized NVRTC option, ignored: " + option + "\n";
    }

    if (translated.ptx_requested) {
        log +=
            "cumetal: a virtual architecture (compute_XX) was requested, but CuMetal lowers CUDA "
            "source to a Metal library and cannot emit PTX. Compile for a real architecture "
            "(sm_XX) and read the result back with nvrtcGetCUBIN.\n";
        program->log = std::move(log);
        return NVRTC_ERROR_INVALID_OPTION;
    }

    const std::filesystem::path& compiler = cumetalc_path();
    if (compiler.empty()) {
        log +=
            "cumetal: cumetalc was not found. Set CUMETAL_NVRTC_COMPILER to its path, set "
            "CUMETAL_ROOT to the installation prefix, or put it on PATH.\n";
        program->log = std::move(log);
        return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
    }

    const std::string stem = sanitize_stem(program->name);
    const std::filesystem::path workspace = make_workspace(stem);
    if (workspace.empty()) {
        log += "cumetal: could not create a temporary compile directory\n";
        program->log = std::move(log);
        return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
    }

    const bool keep_workspace = cumetal::diag_env_truthy("CUMETAL_NVRTC_SAVE_TEMPS");
    const auto cleanup = [&workspace, keep_workspace] {
        if (keep_workspace) return;
        std::error_code ec;
        std::filesystem::remove_all(workspace, ec);
    };

    // Write the in-memory headers beside the source so quoted includes resolve.
    for (const auto& [include_name, contents] : program->headers) {
        const std::filesystem::path header_path =
            workspace / std::filesystem::path(include_name).filename();
        std::ofstream header_stream(header_path, std::ios::binary);
        if (!header_stream) {
            log += "cumetal: could not write header " + include_name + "\n";
            program->log = std::move(log);
            cleanup();
            return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
        }
        header_stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    }

    const std::filesystem::path source_path = workspace / (stem + ".cu");
    const std::filesystem::path output_path = workspace / (stem + ".metallib");
    {
        std::ofstream source_stream(source_path, std::ios::binary);
        if (!source_stream) {
            log += "cumetal: could not write " + source_path.string() + "\n";
            program->log = std::move(log);
            cleanup();
            return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
        }
        source_stream.write(program->source.data(),
                            static_cast<std::streamsize>(program->source.size()));
    }

    std::string command = quote_shell(compiler.string());
    command += " " + quote_shell(source_path.string());
    command += " -o " + quote_shell(output_path.string());
    command += " --emit metallib";
    command += " -I " + quote_shell(workspace.string());
    for (const std::string& arg : translated.compiler_args) {
        command += " " + quote_shell(arg);
    }
    command += " 2>&1";

    if (cumetal::diag_env_truthy("CUMETAL_NVRTC_VERBOSE")) {
        std::fprintf(stderr, "CUMETAL: nvrtc compile: %s\n", command.c_str());
    }

    const CommandResult compile_result = run_command_capture(command);
    log += compile_result.output;
    if (!log.empty() && log.back() != '\n') log += '\n';

    if (!compile_result.started) {
        log += "cumetal: could not launch " + compiler.string() + "\n";
        program->log = std::move(log);
        cleanup();
        return NVRTC_ERROR_BUILTIN_OPERATION_FAILURE;
    }
    if (compile_result.exit_code != 0) {
        program->log = std::move(log);
        cleanup();
        return NVRTC_ERROR_COMPILATION;
    }

    std::vector<char> bytes;
    if (!read_file_bytes(output_path, &bytes) || bytes.empty()) {
        log += "cumetal: cumetalc reported success but produced no metallib at " +
               output_path.string() + "\n";
        program->log = std::move(log);
        cleanup();
        return NVRTC_ERROR_COMPILATION;
    }

    program->cubin = std::move(bytes);

    // cumetalc writes the kernel ABI beside the metallib, and the workspace is
    // about to be deleted. The caller will hand the metallib bytes to
    // cuModuleLoadData, which stages them in the content-addressed module cache;
    // publish the sidecar at that same address now so the driver finds it there
    // instead of guessing argument counts at launch. Best-effort: a failure here
    // costs the ABI metadata, not the compile.
    {
        std::vector<char> abi_bytes;
        if (read_file_bytes(std::filesystem::path(output_path.string() + ".cumetal-abi"),
                            &abi_bytes) &&
            !abi_bytes.empty()) {
            std::filesystem::path staged;
            std::string cache_error;
            if (cumetal::cache::stage_metallib_bytes(program->cubin.data(),
                                                     program->cubin.size(), &staged,
                                                     &cache_error) &&
                !cumetal::cache::stage_metallib_abi_sidecar(staged, abi_bytes.data(),
                                                            abi_bytes.size(), &cache_error)) {
                log += "cumetal: " + cache_error + "\n";
            }
        }
    }

    program->compiled = true;
    program->log = std::move(log);
    cleanup();
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (cubinSizeRet == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    if (!program->compiled) return NVRTC_ERROR_INVALID_PROGRAM;
    *cubinSizeRet = program->cubin.size();
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetCUBIN(nvrtcProgram prog, char* cubin) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (!program->compiled) return NVRTC_ERROR_INVALID_PROGRAM;
    return copy_out(program->cubin, cubin);
}

nvrtcResult nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (ptxSizeRet == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    cumetal::warn_once("nvrtc-ptx",
                       "nvrtcGetPTX is not available: CuMetal compiles CUDA source to a Metal "
                       "library. Use nvrtcGetCUBIN, which cuModuleLoadDataEx accepts.");
    return NVRTC_ERROR_INVALID_PROGRAM;
}

nvrtcResult nvrtcGetPTX(nvrtcProgram prog, char* ptx) {
    (void)ptx;
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    cumetal::warn_once("nvrtc-ptx",
                       "nvrtcGetPTX is not available: CuMetal compiles CUDA source to a Metal "
                       "library. Use nvrtcGetCUBIN, which cuModuleLoadDataEx accepts.");
    return NVRTC_ERROR_INVALID_PROGRAM;
}

nvrtcResult nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (LTOIRSizeRet == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    cumetal::warn_once("nvrtc-ltoir",
                       "nvrtcGetLTOIR is not available: CuMetal has no link-time-optimization IR "
                       "and does not support -dlto.");
    return NVRTC_ERROR_INVALID_PROGRAM;
}

nvrtcResult nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR) {
    (void)LTOIR;
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    cumetal::warn_once("nvrtc-ltoir",
                       "nvrtcGetLTOIR is not available: CuMetal has no link-time-optimization IR "
                       "and does not support -dlto.");
    return NVRTC_ERROR_INVALID_PROGRAM;
}

nvrtcResult nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (logSizeRet == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    *logSizeRet = program->log.size() + 1;
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetProgramLog(nvrtcProgram prog, char* log) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (log == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    std::memcpy(log, program->log.c_str(), program->log.size() + 1);
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcAddNameExpression(nvrtcProgram prog, const char* nameExpression) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (nameExpression == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    if (program->compiled) return NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;
    program->name_expressions.emplace_back(nameExpression);
    return NVRTC_SUCCESS;
}

nvrtcResult nvrtcGetLoweredName(nvrtcProgram prog,
                                const char* nameExpression,
                                const char** loweredName) {
    Program* program = as_program(prog);
    if (program == nullptr) return NVRTC_ERROR_INVALID_PROGRAM;
    if (nameExpression == nullptr || loweredName == nullptr) return NVRTC_ERROR_INVALID_INPUT;
    if (!program->compiled) return NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;

    // CuMetal can answer this only for `extern "C"` entry points, whose lowered
    // name is the expression itself. Anything with template or namespace syntax
    // needs the mangling the device compiler chose, which this shim does not
    // recover from the metallib, so say so rather than guess.
    const std::string expression(nameExpression);
    const bool plain_identifier =
        !expression.empty() &&
        expression.find_first_of("<>():&*, \t") == std::string::npos;
    if (!plain_identifier) return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;

    for (const std::string& registered : program->name_expressions) {
        if (registered == expression) {
            *loweredName = registered.c_str();
            return NVRTC_SUCCESS;
        }
    }
    return NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
}

nvPTXCompileResult nvPTXCompilerGetVersion(unsigned int* major, unsigned int* minor) {
    if (major == nullptr || minor == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    *major = NVPTXCOMPILER_MAX_VERSION_MAJOR;
    *minor = NVPTXCOMPILER_MAX_VERSION_MINOR;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerCreate(nvPTXCompilerHandle* compiler,
                                       size_t ptxCodeLen,
                                       const char* ptxCode) {
    if (compiler == nullptr || ptxCode == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    auto handle = std::make_unique<PtxCompiler>();
    // The caller's length may or may not include a terminator; trim so the
    // program handed back matches the PTX text exactly.
    std::string ptx(ptxCode, ptxCodeLen);
    while (!ptx.empty() && ptx.back() == '\0') ptx.pop_back();
    if (ptx.empty()) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    handle->ptx = std::move(ptx);
    *compiler = reinterpret_cast<nvPTXCompilerHandle>(handle.release());
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerDestroy(nvPTXCompilerHandle* compiler) {
    if (compiler == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    PtxCompiler* handle = as_ptx_compiler(*compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    handle->magic = 0;
    delete handle;
    *compiler = nullptr;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerCompile(nvPTXCompilerHandle compiler,
                                        int numCompileOptions,
                                        const char* const* compileOptions) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (numCompileOptions < 0 || (numCompileOptions > 0 && compileOptions == nullptr)) {
        return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    }
    for (int i = 0; i < numCompileOptions; ++i) {
        if (compileOptions[i] == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    }
    handle->info_log =
        "cumetal: PTX is passed through unchanged; CuMetal's module loader compiles it.\n";
    handle->error_log.clear();
    handle->compiled = true;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle compiler,
                                                       size_t* binaryImageSize) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (binaryImageSize == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    if (!handle->compiled) return NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE;
    // Include the terminator: the driver parses this image as PTX text.
    *binaryImageSize = handle->ptx.size() + 1;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerGetCompiledProgram(nvPTXCompilerHandle compiler,
                                                   void* binaryImage) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (binaryImage == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    if (!handle->compiled) return NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE;
    std::memcpy(binaryImage, handle->ptx.c_str(), handle->ptx.size() + 1);
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerGetErrorLogSize(nvPTXCompilerHandle compiler,
                                                size_t* errorLogSize) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (errorLogSize == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    *errorLogSize = handle->error_log.size() + 1;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerGetErrorLog(nvPTXCompilerHandle compiler, char* errorLog) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    return copy_log(handle->error_log, errorLog);
}

nvPTXCompileResult nvPTXCompilerGetInfoLogSize(nvPTXCompilerHandle compiler, size_t* infoLogSize) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    if (infoLogSize == nullptr) return NVPTXCOMPILE_ERROR_INVALID_INPUT;
    *infoLogSize = handle->info_log.size() + 1;
    return NVPTXCOMPILE_SUCCESS;
}

nvPTXCompileResult nvPTXCompilerGetInfoLog(nvPTXCompilerHandle compiler, char* infoLog) {
    PtxCompiler* handle = as_ptx_compiler(compiler);
    if (handle == nullptr) return NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE;
    return copy_log(handle->info_log, infoLog);
}

}  // extern "C"
