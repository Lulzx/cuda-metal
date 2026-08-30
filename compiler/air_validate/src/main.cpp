#include "cumetal/air_validate/validator.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

void print_usage(const char* argv0) {
    std::cerr << "Usage: " << argv0
              << " <file.metallib> [--xcrun] [--mla] [--mla-cmd <cmd>]"
                 " [--require-function-list] [--require-metadata]"
                 " [--llvm-dis] [--no-bitcode-required] [--relaxed-magic]\n";
}

std::string json_escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 8);
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        switch (c) {
            case '\\':
                out += "\\\\";
                break;
            case '"':
                out += "\\\"";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out.push_back(hex[c >> 4]);
                    out.push_back(hex[c & 0x0f]);
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    return out;
}

const char* severity_name(cumetal::air_validate::Severity severity) {
    switch (severity) {
        case cumetal::air_validate::Severity::kInfo:
            return "info";
        case cumetal::air_validate::Severity::kWarning:
            return "warning";
        case cumetal::air_validate::Severity::kError:
            return "error";
    }
    return "unknown";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 2;
    }

    cumetal::air_validate::ValidationOptions options;
    bool json = false;
    std::string input;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--xcrun") {
            options.run_xcrun_validate = true;
        } else if (arg == "--mla") {
            options.run_metal_library_archive = true;
        } else if (arg == "--mla-cmd") {
            if (i + 1 >= argc) {
                std::cerr << "--mla-cmd expects a command prefix\n";
                return 2;
            }
            options.metal_library_archive_command = argv[++i];
            options.run_metal_library_archive = true;
        } else if (arg == "--require-function-list") {
            options.require_function_list = true;
        } else if (arg == "--require-metadata") {
            options.require_kernel_metadata = true;
        } else if (arg == "--llvm-dis") {
            options.verify_bitcode_with_llvm_dis = true;
        } else if (arg == "--no-bitcode-required") {
            options.require_bitcode = false;
        } else if (arg == "--relaxed-magic") {
            options.strict_magic = false;
        } else if (arg == "--json") {
            json = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << "\n";
            return 2;
        } else if (input.empty()) {
            input = arg;
        } else {
            std::cerr << "Unexpected positional argument: " << arg << "\n";
            return 2;
        }
    }

    if (input.empty()) {
        print_usage(argv[0]);
        return 2;
    }

    const auto result = cumetal::air_validate::validate_file(input, options);

    if (json) {
        std::cout << "{\n";
        std::cout << "  \"ok\": " << (result.ok ? "true" : "false") << ",\n";
        std::cout << "  \"file\": \"" << json_escape(result.summary.path) << "\",\n";
        std::cout << "  \"size\": " << result.summary.file_size << ",\n";
        std::cout << "  \"magic\": \"" << json_escape(result.summary.magic_ascii) << "\",\n";
        std::cout << "  \"bitcode_sections\": " << result.summary.bitcode_sections.size() << ",\n";
        std::cout << "  \"function_list_parsed\": "
                  << (result.summary.function_list_parsed ? "true" : "false") << ",\n";
        std::cout << "  \"function_list_parser\": \""
                  << json_escape(result.summary.function_list_parser) << "\",\n";
        std::cout << "  \"kernel_count\": " << result.summary.kernels.size() << ",\n";
        std::cout << "  \"diagnostics\": [\n";
        for (std::size_t i = 0; i < result.diagnostics.size(); ++i) {
            const auto& diagnostic = result.diagnostics[i];
            std::cout << "    {\"severity\": \"" << severity_name(diagnostic.severity)
                      << "\", \"message\": \"" << json_escape(diagnostic.message) << "\"}";
            if (i + 1 != result.diagnostics.size()) {
                std::cout << ',';
            }
            std::cout << "\n";
        }
        std::cout << "  ]\n";
        std::cout << "}\n";
    } else {
        std::cout << cumetal::air_validate::format_report(result);
    }

    return result.ok ? 0 : 1;
}
