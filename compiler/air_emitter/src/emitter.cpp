#include "cumetal/air_emitter/emitter.h"

#include "cumetal/air_validate/validator.h"
#include "cumetal/common/metallib.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace cumetal::air_emitter {
namespace {

constexpr std::size_t kHeaderSize = 40;
constexpr std::size_t kFunctionEntrySize = 32;

struct CommandResult {
    bool started = false;
    int exit_code = -1;
    std::string output;
};

struct KernelInput {
    std::string name;
    std::vector<cumetal::common::KernelMetadataField> metadata;
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

// Returns absolute path to a tool, checking PATH first, then common macOS/Homebrew locations.
// Returns empty string if not found.
std::string find_tool(const std::string& name) {
    // Check env override first (CUMETAL_TOOL_<NAME> e.g. CUMETAL_TOOL_LLVM_AS)
    std::string env_key = "CUMETAL_TOOL_";
    for (char c : name) {
        env_key += static_cast<char>(std::toupper(static_cast<unsigned char>(c == '-' ? '_' : c)));
    }
    if (const char* env_val = std::getenv(env_key.c_str())) {
        if (std::filesystem::exists(env_val)) {
            return env_val;
        }
    }
    // Check if tool is in PATH via `command -v`
    const CommandResult cv = run_command_capture("command -v " + name + " 2>/dev/null");
    if (cv.started && cv.exit_code == 0 && !cv.output.empty()) {
        const std::string path = cv.output.substr(0, cv.output.find('\n'));
        if (!path.empty() && std::filesystem::exists(path)) {
            return path;
        }
    }
    // Try common absolute locations
    static const std::vector<std::string> search_prefixes = {
        "/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/opt/homebrew/opt/llvm/bin",
        "/opt/homebrew/opt/llvm@20/bin",
        "/opt/homebrew/opt/llvm@19/bin",
        "/opt/homebrew/opt/llvm@18/bin",
        "/opt/homebrew/opt/llvm@17/bin",
    };
    for (const auto& prefix : search_prefixes) {
        const std::string full = prefix + "/" + name;
        if (std::filesystem::exists(full)) {
            return full;
        }
    }
    // Try glob-style search for versioned Homebrew Cellar llvm
    const CommandResult glob = run_command_capture(
        "ls /opt/homebrew/Cellar/llvm@20/*/bin/" + name + " /opt/homebrew/Cellar/llvm/*/bin/" +
        name + " 2>/dev/null | head -1");
    if (glob.started && !glob.output.empty()) {
        const std::string path = glob.output.substr(0, glob.output.find('\n'));
        if (!path.empty() && std::filesystem::exists(path)) {
            return path;
        }
    }
    return "";
}

std::filesystem::path make_temp_dir() {
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto pid = static_cast<long long>(::getpid());
    const std::filesystem::path dir = std::filesystem::temp_directory_path() /
                                      ("cumetal-air-emitter-" + std::to_string(pid) + "-" +
                                       std::to_string(now));
    std::filesystem::create_directories(dir);
    return dir;
}

std::string lower_ext(const std::filesystem::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext;
}

void append_u32_le(std::vector<std::uint8_t>& out, std::uint32_t value) {
    out.push_back(static_cast<std::uint8_t>(value & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 8) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 16) & 0xFF));
    out.push_back(static_cast<std::uint8_t>((value >> 24) & 0xFF));
}

void write_u32_le(std::vector<std::uint8_t>& out, std::size_t offset, std::uint32_t value) {
    if (offset + 4 > out.size()) {
        return;
    }
    out[offset + 0] = static_cast<std::uint8_t>(value & 0xFF);
    out[offset + 1] = static_cast<std::uint8_t>((value >> 8) & 0xFF);
    out[offset + 2] = static_cast<std::uint8_t>((value >> 16) & 0xFF);
    out[offset + 3] = static_cast<std::uint8_t>((value >> 24) & 0xFF);
}

std::string to_text(const std::vector<std::uint8_t>& bytes) {
    return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

std::vector<cumetal::common::KernelMetadataField> dedupe_metadata(
    const std::vector<cumetal::common::KernelMetadataField>& input) {
    std::map<std::string, std::string> keyed;
    for (const auto& field : input) {
        if (!field.key.empty()) {
            keyed[field.key] = field.value;
        }
    }

    std::vector<cumetal::common::KernelMetadataField> output;
    output.reserve(keyed.size());
    for (const auto& [key, value] : keyed) {
        output.push_back({.key = key, .value = value});
    }
    return output;
}

std::vector<KernelInput> parse_kernels_from_llvm_ir(const std::vector<std::uint8_t>& bytes,
                                                     const std::string& fallback_name,
                                                     std::vector<std::string>* logs) {
    const std::string text = to_text(bytes);
    std::istringstream stream(text);
    std::string line;

    struct FunctionDecl {
        std::string name;
        int attr_id = -1;
    };

    std::vector<FunctionDecl> decls;
    std::map<int, std::vector<cumetal::common::KernelMetadataField>> attr_metadata;
    bool saw_air_kernel_module_node = false;
    bool saw_air_compile_options = false;
    bool saw_air_language_version = false;

    const std::regex define_re(
        R"(^\s*define\b.*@([A-Za-z_.$][A-Za-z0-9_.$]*)\(.*\)\s*(?:#([0-9]+))?)");
    const std::regex attr_re(R"(^\s*attributes\s+#([0-9]+)\s*=\s*\{(.*)\}\s*$)");
    const std::regex quoted_key_re(R"re("([^"]+)"(?:="([^"]*)")?)re");

    while (std::getline(stream, line)) {
        if (line.find("!air.kernel") != std::string::npos) {
            saw_air_kernel_module_node = true;
        }
        if (line.find("!air.compile_options") != std::string::npos) {
            saw_air_compile_options = true;
        }
        if (line.find("!air.language_version") != std::string::npos) {
            saw_air_language_version = true;
        }

        std::smatch define_match;
        if (std::regex_search(line, define_match, define_re)) {
            FunctionDecl decl;
            decl.name = define_match[1].str();
            if (define_match[2].matched) {
                decl.attr_id = std::stoi(define_match[2].str());
            }
            decls.push_back(std::move(decl));
            continue;
        }

        std::smatch attr_match;
        if (std::regex_search(line, attr_match, attr_re)) {
            const int attr_id = std::stoi(attr_match[1].str());
            const std::string body = attr_match[2].str();
            std::vector<cumetal::common::KernelMetadataField> fields;

            for (auto it = std::sregex_iterator(body.begin(), body.end(), quoted_key_re);
                 it != std::sregex_iterator(); ++it) {
                const std::smatch token = *it;
                const std::string key = token[1].str();
                const std::string value = token[2].matched ? token[2].str() : "true";
                if (key.rfind("air.", 0) == 0) {
                    fields.push_back({.key = key, .value = value});
                }
            }
            attr_metadata[attr_id] = dedupe_metadata(fields);
        }
    }

    std::vector<KernelInput> kernels;
    kernels.reserve(std::max<std::size_t>(1, decls.size()));

    if (decls.empty()) {
        kernels.push_back({.name = fallback_name,
                           .metadata = {{.key = "air.kernel", .value = "true"},
                                        {.key = "air.version", .value = "2.6"}}});
    } else {
        for (const auto& decl : decls) {
            KernelInput kernel;
            kernel.name = decl.name;

            if (decl.attr_id >= 0) {
                const auto found = attr_metadata.find(decl.attr_id);
                if (found != attr_metadata.end()) {
                    kernel.metadata = found->second;
                }
            }

            bool has_air_kernel = false;
            bool has_air_version = false;
            for (const auto& field : kernel.metadata) {
                has_air_kernel = has_air_kernel || field.key == "air.kernel";
                has_air_version = has_air_version || field.key == "air.version";
            }
            if (!has_air_kernel) {
                kernel.metadata.push_back({.key = "air.kernel", .value = "true"});
                if (logs != nullptr) {
                    logs->push_back("LLVM IR missing explicit air.kernel on " + kernel.name +
                                    "; defaulting to true for experimental container");
                }
            }
            if (!has_air_version) {
                kernel.metadata.push_back({.key = "air.version", .value = "2.6"});
                if (logs != nullptr) {
                    logs->push_back("LLVM IR missing explicit air.version on " + kernel.name +
                                    "; defaulting to 2.6 for experimental container");
                }
            }

            if (saw_air_kernel_module_node) {
                kernel.metadata.push_back({.key = "module.air.kernel", .value = "present"});
            }
            if (saw_air_compile_options) {
                kernel.metadata.push_back({.key = "module.air.compile_options", .value = "present"});
            }
            if (saw_air_language_version) {
                kernel.metadata.push_back({.key = "module.air.language_version", .value = "present"});
            }

            kernel.metadata = dedupe_metadata(kernel.metadata);
            kernels.push_back(std::move(kernel));
        }
    }

    return kernels;
}

std::string serialize_metadata(const std::vector<cumetal::common::KernelMetadataField>& fields) {
    std::string out;
    for (const auto& field : fields) {
        if (field.key.empty()) {
            continue;
        }
        out += field.key;
        out.push_back('=');
        out += field.value;
        out.push_back('\n');
    }
    return out;
}

bool convert_llvm_ir_to_bitcode(const std::filesystem::path& input,
                                std::vector<std::uint8_t>* bitcode,
                                std::vector<std::string>* logs) {
    if (bitcode == nullptr) {
        return false;
    }

    if (!command_exists("llvm-as")) {
        if (logs != nullptr) {
            logs->push_back("llvm-as not found; embedding textual LLVM IR in experimental container");
        }
        return false;
    }

    const std::filesystem::path temp_dir = make_temp_dir();
    const std::filesystem::path temp_bc = temp_dir / "module.bc";

    const std::string command =
        "llvm-as " + quote_shell(input.string()) + " -o " + quote_shell(temp_bc.string()) + " 2>&1";
    const CommandResult result = run_command_capture(command);

    if (logs != nullptr) {
        logs->push_back("$ " + command);
        if (!result.output.empty()) {
            logs->push_back(result.output);
        }
    }

    if (!result.started || result.exit_code != 0) {
        std::filesystem::remove_all(temp_dir);
        return false;
    }

    std::string io_error;
    *bitcode = cumetal::common::read_file_bytes(temp_bc, &io_error);
    std::filesystem::remove_all(temp_dir);

    if (bitcode->empty()) {
        if (logs != nullptr) {
            logs->push_back("failed to read assembled bitcode: " + io_error);
        }
        return false;
    }

    return true;
}

EmitResult emit_experimental(const EmitOptions& options) {
    EmitResult result;
    result.mode_used = EmitMode::kExperimentalContainer;

    std::string io_error;
    const std::vector<std::uint8_t> input_payload =
        cumetal::common::read_file_bytes(options.input, &io_error);
    if (input_payload.empty()) {
        result.error = io_error.empty() ? "input payload is empty" : io_error;
        return result;
    }

    const std::string ext = lower_ext(options.input);
    std::vector<KernelInput> kernels;
    std::vector<std::uint8_t> bitcode_payload = input_payload;

    if (ext == ".ll" || ext == ".llvm") {
        kernels = parse_kernels_from_llvm_ir(input_payload, options.kernel_name, &result.logs);
        const bool assembled = convert_llvm_ir_to_bitcode(options.input, &bitcode_payload, &result.logs);
        if (!assembled) {
            for (auto& kernel : kernels) {
                kernel.metadata.push_back({.key = "bitcode.encoding", .value = "text-llvm"});
                kernel.metadata = dedupe_metadata(kernel.metadata);
            }
        }
    } else {
        kernels.push_back({.name = options.kernel_name,
                           .metadata = {{.key = "air.kernel", .value = "true"},
                                        {.key = "air.version", .value = "2.6"},
                                        {.key = "bitcode.encoding", .value = ext.empty() ? "raw" : ext}}});
    }

    if (kernels.empty()) {
        result.error = "no kernel records could be derived from input";
        return result;
    }

    std::vector<std::uint8_t> function_table;
    function_table.reserve(kernels.size() * kFunctionEntrySize);

    std::vector<std::uint8_t> string_table;
    std::vector<std::uint8_t> payload;

    for (const auto& kernel : kernels) {
        const std::uint32_t name_rel = static_cast<std::uint32_t>(string_table.size());
        string_table.insert(string_table.end(), kernel.name.begin(), kernel.name.end());
        string_table.push_back(0);

        const std::uint32_t bitcode_rel = static_cast<std::uint32_t>(payload.size());
        payload.insert(payload.end(), bitcode_payload.begin(), bitcode_payload.end());
        const std::uint32_t bitcode_size = static_cast<std::uint32_t>(bitcode_payload.size());

        const std::string metadata_blob = serialize_metadata(kernel.metadata);
        const std::uint32_t metadata_rel = static_cast<std::uint32_t>(payload.size());
        payload.insert(payload.end(), metadata_blob.begin(), metadata_blob.end());
        const std::uint32_t metadata_size = static_cast<std::uint32_t>(metadata_blob.size());

        append_u32_le(function_table, name_rel);
        append_u32_le(function_table, bitcode_rel);
        append_u32_le(function_table, bitcode_size);
        append_u32_le(function_table, metadata_rel);
        append_u32_le(function_table, metadata_size);
        append_u32_le(function_table, 0);
        append_u32_le(function_table, 0);
        append_u32_le(function_table, 0);
    }

    const std::uint32_t function_table_offset = static_cast<std::uint32_t>(kHeaderSize);
    const std::uint32_t function_count = static_cast<std::uint32_t>(kernels.size());
    const std::uint32_t string_table_offset =
        function_table_offset + static_cast<std::uint32_t>(function_table.size());
    const std::uint32_t string_table_size = static_cast<std::uint32_t>(string_table.size());
    const std::uint32_t payload_offset = string_table_offset + string_table_size;
    const std::uint32_t payload_size = static_cast<std::uint32_t>(payload.size());
    const std::uint32_t total_size = payload_offset + payload_size;
    const std::uint32_t flags = 0x00000001U;

    std::vector<std::uint8_t> container;
    container.reserve(total_size);

    container.push_back('M');
    container.push_back('T');
    container.push_back('L');
    container.push_back('B');
    append_u32_le(container, 2);
    append_u32_le(container, total_size);
    append_u32_le(container, function_table_offset);
    append_u32_le(container, function_count);
    append_u32_le(container, string_table_offset);
    append_u32_le(container, string_table_size);
    append_u32_le(container, payload_offset);
    append_u32_le(container, payload_size);
    append_u32_le(container, flags);

    container.insert(container.end(), function_table.begin(), function_table.end());
    container.insert(container.end(), string_table.begin(), string_table.end());
    container.insert(container.end(), payload.begin(), payload.end());

    write_u32_le(container, 8, static_cast<std::uint32_t>(container.size()));

    if (!cumetal::common::write_file_bytes(options.output, container, &io_error)) {
        result.error = io_error;
        return result;
    }

    result.output = options.output;
    result.logs.push_back("wrote experimental container v2 with " + std::to_string(function_count) +
                          " kernel record(s)");
    result.ok = true;
    return result;
}

EmitResult emit_with_xcrun(const EmitOptions& options) {
    EmitResult result;
    result.mode_used = EmitMode::kXcrun;

    if (!std::filesystem::exists(options.input)) {
        result.error = "input file does not exist: " + options.input.string();
        return result;
    }

    const std::filesystem::path temp_dir = make_temp_dir();
    const std::filesystem::path temp_air = temp_dir / "kernel.air";

    const std::string extension = lower_ext(options.input);
    bool prepared_air = false;

    if (extension == ".air") {
        std::filesystem::copy_file(options.input, temp_air,
                                   std::filesystem::copy_options::overwrite_existing);
        prepared_air = true;
        result.logs.push_back("using provided .air payload");
    } else if (extension == ".metal") {
        const std::string xcrun_for_metal = find_tool("xcrun");
        const std::string xcrun_metal_bin = xcrun_for_metal.empty() ? "xcrun" : xcrun_for_metal;
        const std::string command = xcrun_metal_bin + " metal -c " + quote_shell(options.input.string()) +
                                    " -o " + quote_shell(temp_air.string()) + " 2>&1";
        const CommandResult cmd = run_command_capture(command);
        result.logs.push_back("$ " + command);
        result.logs.push_back(cmd.output);
        if (!cmd.started || cmd.exit_code != 0) {
            result.error = "failed to compile .metal to .air with xcrun metal";
            std::filesystem::remove_all(temp_dir);
            return result;
        }
        prepared_air = true;
    } else if (extension == ".ll" || extension == ".llvm") {
        // Compile LLVM IR directly to a fully-linked metallib (AOT, GPU ISA included).
        // This avoids Metal JIT crashes that occur when packaging only Air bitcode via
        // `metal -c` + `metallib` and then JIT-compiling at pipeline creation time.
        {
            const std::string xcrun_path = find_tool("xcrun");
            const std::string xcrun_bin = xcrun_path.empty() ? "xcrun" : xcrun_path;
            // Full AOT compilation: metal input.ll -o output.metallib (no -c flag)
            const std::string command = xcrun_bin + " metal " + quote_shell(options.input.string()) + " -o " +
                                        quote_shell(options.output.string()) + " 2>&1";
            const CommandResult cmd = run_command_capture(command);
            result.logs.push_back("$ " + command);
            result.logs.push_back(cmd.output);
            if (cmd.started && cmd.exit_code == 0) {
                result.logs.push_back("used xcrun metal AOT for LLVM IR -> metallib");
                std::filesystem::remove_all(temp_dir);
                result.ok = true;
                result.output = options.output;
                return result;
            }
            // AOT compilation failed — fall back to Air bitcode path
            result.logs.push_back("xcrun metal AOT failed, falling back to Air bitcode path");
            const std::string fallback_cmd = xcrun_bin + " metal -c " + quote_shell(options.input.string()) + " -o " +
                                             quote_shell(temp_air.string()) + " 2>&1";
            const CommandResult fallback = run_command_capture(fallback_cmd);
            result.logs.push_back("$ " + fallback_cmd);
            result.logs.push_back(fallback.output);
            if (fallback.started && fallback.exit_code == 0) {
                result.logs.push_back("used xcrun metal -c for LLVM IR -> AIR");
                prepared_air = true;
            } else {
                const std::string llvm_as_path = find_tool("llvm-as");
                if (!llvm_as_path.empty()) {
                    const std::string llvm_as_cmd =
                        llvm_as_path + " " + quote_shell(options.input.string()) + " -o " +
                        quote_shell(temp_air.string()) + " 2>&1";
                    const CommandResult llvm_as_result = run_command_capture(llvm_as_cmd);
                    result.logs.push_back("$ " + llvm_as_cmd);
                    result.logs.push_back(llvm_as_result.output);
                    if (!llvm_as_result.started || llvm_as_result.exit_code != 0) {
                        result.error = "failed to compile LLVM IR with xcrun metal and assemble fallback";
                        std::filesystem::remove_all(temp_dir);
                        return result;
                    }
                    result.logs.push_back("used llvm-as fallback for LLVM IR -> AIR");
                    prepared_air = true;
                } else {
                    result.error = "failed to compile LLVM IR with xcrun metal (and llvm-as unavailable)";
                    std::filesystem::remove_all(temp_dir);
                    return result;
                }
            }
        }
    } else if (extension == ".bc") {
        std::filesystem::copy_file(options.input, temp_air,
                                   std::filesystem::copy_options::overwrite_existing);
        prepared_air = true;
        result.logs.push_back("treating input .bc as AIR payload for metallib packaging");
    } else {
        result.error = "unsupported input extension for xcrun mode: " + extension;
        std::filesystem::remove_all(temp_dir);
        return result;
    }

    if (!prepared_air) {
        result.error = "could not prepare AIR payload for metallib packaging";
        std::filesystem::remove_all(temp_dir);
        return result;
    }

    const std::string xcrun_for_metallib = find_tool("xcrun");
    const std::string xcrun_metallib_bin = xcrun_for_metallib.empty() ? "xcrun" : xcrun_for_metallib;
    const std::string metallib_command = xcrun_metallib_bin + " metallib " + quote_shell(temp_air.string()) +
                                         " -o " + quote_shell(options.output.string()) + " 2>&1";
    const CommandResult pack_cmd = run_command_capture(metallib_command);
    result.logs.push_back("$ " + metallib_command);
    result.logs.push_back(pack_cmd.output);

    std::filesystem::remove_all(temp_dir);

    if (!pack_cmd.started || pack_cmd.exit_code != 0) {
        result.error = "failed to package metallib with xcrun metallib";
        return result;
    }

    result.ok = true;
    result.output = options.output;
    return result;
}

EmitResult validate_if_requested(const EmitOptions& options, EmitResult result) {
    if (!result.ok || !options.validate_output) {
        return result;
    }

    cumetal::air_validate::ValidationOptions validation_options;
    validation_options.require_bitcode = (result.mode_used == EmitMode::kXcrun);
    validation_options.strict_magic = true;
    validation_options.run_xcrun_validate = options.run_xcrun_validate;
    validation_options.require_function_list = true;
    validation_options.require_kernel_metadata = true;

    const auto validation = cumetal::air_validate::validate_file(options.output, validation_options);
    result.logs.push_back(cumetal::air_validate::format_report(validation));
    if (!validation.ok) {
        result.ok = false;
        result.error = "output metallib did not pass validation";
    }
    return result;
}

}  // namespace

EmitResult emit_metallib(const EmitOptions& options) {
    EmitResult result;

    if (options.input.empty() || options.output.empty()) {
        result.error = "both input and output paths are required";
        return result;
    }

    if (std::filesystem::exists(options.output) && !options.overwrite) {
        result.error = "output already exists (pass --overwrite to replace): " +
                       options.output.string();
        return result;
    }

    if (const auto parent = options.output.parent_path(); !parent.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
    }

    if (options.mode == EmitMode::kXcrun) {
        EmitResult xcrun_result = emit_with_xcrun(options);
        if (!xcrun_result.ok && options.fallback_to_experimental) {
            EmitResult fallback_result = emit_experimental(options);
            fallback_result.logs.insert(fallback_result.logs.begin(),
                                        "xcrun mode failed: " + xcrun_result.error);
            fallback_result.logs.insert(
                fallback_result.logs.begin(),
                "xcrun mode failed, falling back to experimental container mode");
            fallback_result.logs.insert(fallback_result.logs.begin(), xcrun_result.logs.begin(),
                                        xcrun_result.logs.end());
            result = std::move(fallback_result);
        } else {
            result = std::move(xcrun_result);
        }
    } else {
        result = emit_experimental(options);
    }

    return validate_if_requested(options, std::move(result));
}

}  // namespace cumetal::air_emitter
