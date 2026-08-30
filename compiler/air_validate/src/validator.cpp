#include "cumetal/air_validate/validator.h"

#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <sstream>
#include <sys/wait.h>
#include <unistd.h>

namespace cumetal::air_validate {
namespace {

struct CommandResult {
    bool started = false;
    int exit_code = -1;
    std::string output;
};

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
    } else {
        result.exit_code = -1;
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

bool has_error(const std::vector<Diagnostic>& diagnostics) {
    for (const auto& diagnostic : diagnostics) {
        if (diagnostic.severity == Severity::kError) {
            return true;
        }
    }
    return false;
}

const char* severity_to_string(Severity severity) {
    switch (severity) {
        case Severity::kInfo:
            return "INFO";
        case Severity::kWarning:
            return "WARN";
        case Severity::kError:
            return "ERROR";
    }
    return "UNKNOWN";
}

bool kernel_has_metadata(const cumetal::common::KernelRecord& kernel, const std::string& key) {
    for (const auto& field : kernel.metadata) {
        if (field.key == key) {
            return true;
        }
    }
    return false;
}

std::optional<std::filesystem::path> make_temp_dir() {
    std::error_code error;
    const std::filesystem::path root = std::filesystem::temp_directory_path(error);
    if (error) return std::nullopt;
    std::string pattern = (root / "cumetal-air-validate-XXXXXX").string();
    if (::mkdtemp(pattern.data()) == nullptr) return std::nullopt;
    return std::filesystem::path(pattern);
}

void validate_kernel_bitcode(const std::filesystem::path& input_path,
                             const std::vector<std::uint8_t>& bytes,
                             const cumetal::common::KernelRecord& kernel,
                             const ValidationOptions& options,
                             std::vector<Diagnostic>* diagnostics) {
    if (diagnostics == nullptr) {
        return;
    }

    if (kernel.bitcode_size == 0 || kernel.bitcode_offset + kernel.bitcode_size > bytes.size()) {
        diagnostics->push_back({.severity = Severity::kError,
                                .message = "kernel " + kernel.name + " has invalid bitcode bounds"});
        return;
    }

    if (kernel.bitcode_signature_ok) {
        diagnostics->push_back({.severity = Severity::kInfo,
                                .message = "kernel " + kernel.name +
                                           " bitcode has LLVM signature prefix"});
        if (!options.verify_bitcode_with_llvm_dis) {
            return;
        }
    }

    if (!options.verify_bitcode_with_llvm_dis) {
        diagnostics->push_back({.severity = Severity::kWarning,
                                .message = "kernel " + kernel.name +
                                           " bitcode does not have a recognized LLVM signature"});
        return;
    }

    if (!command_exists("llvm-dis")) {
        diagnostics->push_back({.severity = Severity::kWarning,
                                .message = "llvm-dis not found; skipping deep bitcode check for " +
                                           kernel.name});
        return;
    }

    const auto temp_dir = make_temp_dir();
    if (!temp_dir.has_value()) {
        diagnostics->push_back({.severity = Severity::kError,
                                .message = "failed to create secure temporary directory for " +
                                           kernel.name});
        return;
    }
    const std::filesystem::path bitcode_file = *temp_dir / "kernel.bc";

    const std::vector<std::uint8_t> slice(bytes.begin() + static_cast<std::ptrdiff_t>(kernel.bitcode_offset),
                                          bytes.begin() + static_cast<std::ptrdiff_t>(kernel.bitcode_offset +
                                                                                      kernel.bitcode_size));
    std::string io_error;
    if (!cumetal::common::write_file_bytes(bitcode_file, slice, &io_error)) {
        diagnostics->push_back({.severity = Severity::kError,
                                .message = "failed to write temporary bitcode for " + kernel.name +
                                           ": " + io_error});
        std::filesystem::remove_all(*temp_dir);
        return;
    }

    const std::string command =
        "llvm-dis " + quote_shell(bitcode_file.string()) + " -o /dev/null 2>&1";
    const CommandResult cmd = run_command_capture(command);
    std::filesystem::remove_all(*temp_dir);

    if (!cmd.started || cmd.exit_code != 0) {
        diagnostics->push_back({.severity = Severity::kError,
                                .message = "llvm-dis rejected kernel " + kernel.name + " from " +
                                           input_path.string()});
        return;
    }

    diagnostics->push_back({.severity = Severity::kInfo,
                            .message = "llvm-dis accepted kernel " + kernel.name + " bitcode"});
}

}  // namespace

ValidationResult validate_file(const std::filesystem::path& path, const ValidationOptions& options) {
    ValidationResult result;

    std::string io_error;
    const std::vector<std::uint8_t> bytes = cumetal::common::read_file_bytes(path, &io_error);
    if (bytes.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kError, .message = io_error.empty() ? "input file is empty" : io_error});
        result.ok = false;
        return result;
    }

    result.summary =
        cumetal::common::inspect_metallib_bytes(path.string(), bytes, options.max_strings);

    if (result.summary.file_size < 16) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "container is too small to be a valid metallib (expected >= 16 bytes)"});
    }

    if (options.strict_magic && !cumetal::common::looks_like_metallib(result.summary)) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "magic header does not match expected MTL* container prefix"});
    }

    if (options.require_bitcode && result.summary.bitcode_sections.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kError,
             .message = "no embedded LLVM bitcode signatures found in container"});
    } else if (!result.summary.bitcode_sections.empty()) {
        result.diagnostics.push_back(
            {.severity = Severity::kInfo,
             .message = "found " + std::to_string(result.summary.bitcode_sections.size()) +
                        " candidate embedded bitcode section(s)"});
    }

    if (result.summary.function_list_parsed) {
        result.diagnostics.push_back(
            {.severity = Severity::kInfo, .message = "parsed function list via " +
                                                     result.summary.function_list_parser});

        if (options.require_function_list && result.summary.kernels.empty()) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "function list is empty in parsed container"});
        }

        std::size_t kernel_count = 0;
        for (const auto& kernel : result.summary.kernels) {
            if (options.require_bitcode) {
                validate_kernel_bitcode(path, bytes, kernel, options, &result.diagnostics);
            }

            if (options.require_kernel_metadata) {
                const bool has_air_kernel = kernel_has_metadata(kernel, "air.kernel");
                const bool has_air_version = kernel_has_metadata(kernel, "air.version");
                if (!has_air_kernel) {
                    result.diagnostics.push_back(
                        {.severity = Severity::kInfo,
                         .message = "function " + kernel.name +
                                    " is a linked helper, not an AIR kernel"});
                    continue;
                }
                ++kernel_count;
                if (!has_air_version) {
                    result.diagnostics.push_back(
                        {.severity = Severity::kError,
                         .message = "kernel " + kernel.name +
                                    " is missing required metadata (air.version)"});
                } else {
                    result.diagnostics.push_back(
                        {.severity = Severity::kInfo,
                         .message = "kernel " + kernel.name +
                                    " includes required metadata fields"});
                }
            }
        }
        if (options.require_kernel_metadata && kernel_count == 0) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "function list contains no AIR kernels"});
        }
    } else if ((options.require_function_list || options.require_kernel_metadata) &&
               !options.run_metal_library_archive) {
        result.diagnostics.push_back({
            .severity = Severity::kWarning,
            .message =
                "container function table is not parsed by built-in parser (expected for Apple metallib); "
                "enable MetalLibraryArchive validation for function/metadata checks"});
    }

    if (options.run_metal_library_archive) {
        std::string command = options.metal_library_archive_command;
        if (command.empty()) {
            command = "swift run --package-path tools/metal_library_archive_bridge cumetal-mla-validate " +
                      quote_shell(path.string()) + " 2>&1";
        } else {
            command += " " + quote_shell(path.string()) + " 2>&1";
        }

        const CommandResult command_result = run_command_capture(command);
        result.metal_library_archive_exit_code = command_result.exit_code;
        result.metal_library_archive_output = command_result.output;

        if (!command_result.started) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "failed to invoke MetalLibraryArchive validation command"});
        } else if (command_result.exit_code != 0) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "MetalLibraryArchive validation command returned non-zero exit code " +
                            std::to_string(command_result.exit_code)});
        } else {
            result.diagnostics.push_back(
                {.severity = Severity::kInfo,
                 .message = "MetalLibraryArchive validation command accepted the container"});
        }
    }

    if (options.run_xcrun_validate) {
        const std::string metallib_command =
            "xcrun metallib --app-store-validate " + quote_shell(path.string()) + " 2>&1";
        CommandResult command_result = run_command_capture(metallib_command);
        std::string command_used = metallib_command;

        const bool needs_legacy_fallback =
            !command_result.started ||
            command_result.output.find("unknown argument: '--app-store-validate'") != std::string::npos ||
            command_result.output.find("unable to find utility \"metallib\"") != std::string::npos;

        if (needs_legacy_fallback) {
            const std::string metal_command =
                "xcrun metal -validate " + quote_shell(path.string()) + " 2>&1";
            command_result = run_command_capture(metal_command);
            command_used = metal_command;
        }

        result.xcrun_exit_code = command_result.exit_code;
        result.xcrun_output = command_result.output;

        if (!command_result.started) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "failed to invoke xcrun validation command"});
        } else if (command_result.exit_code != 0) {
            result.diagnostics.push_back(
                {.severity = Severity::kError,
                 .message = "xcrun validation command returned non-zero exit code " +
                            std::to_string(command_result.exit_code)});
        } else {
            result.diagnostics.push_back(
                {.severity = Severity::kInfo,
                 .message = "xcrun validation command accepted the container"});
        }
        result.diagnostics.push_back(
            {.severity = Severity::kInfo, .message = "xcrun validation command: " + command_used});
    }

    result.ok = !has_error(result.diagnostics);
    return result;
}

std::string format_report(const ValidationResult& result) {
    std::ostringstream out;

    out << "Validation result: " << (result.ok ? "PASS" : "FAIL") << "\n";
    out << "File: " << result.summary.path << "\n";
    out << "Size: " << result.summary.file_size << " bytes\n";
    out << "Magic: " << result.summary.magic_ascii << "\n";
    out << "Bitcode sections: " << result.summary.bitcode_sections.size() << "\n";
    if (result.summary.function_list_parsed) {
        out << "Function parser: " << result.summary.function_list_parser << "\n";
        out << "Kernel count: " << result.summary.kernels.size() << "\n";
    }

    for (const auto& diagnostic : result.diagnostics) {
        out << "[" << severity_to_string(diagnostic.severity) << "] " << diagnostic.message << "\n";
    }

    if (!result.metal_library_archive_output.empty()) {
        out << "\n--- MetalLibraryArchive output ---\n";
        out << result.metal_library_archive_output;
        if (result.metal_library_archive_output.back() != '\n') {
            out << "\n";
        }
    }

    if (!result.xcrun_output.empty()) {
        out << "\n--- xcrun output ---\n";
        out << result.xcrun_output;
        if (result.xcrun_output.back() != '\n') {
            out << "\n";
        }
    }

    return out.str();
}

}  // namespace cumetal::air_validate
