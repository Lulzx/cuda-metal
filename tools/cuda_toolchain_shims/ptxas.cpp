#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    std::filesystem::path input;
    std::filesystem::path output;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--output-file" && i + 1 < argc) {
            output = argv[++i];
        } else if (arg.rfind("--output-file=", 0) == 0) {
            output = arg.substr(14);
        } else if (!arg.empty() && arg.front() != '-') {
            input = arg;
        }
    }

    if (input.empty() || output.empty()) {
        std::cerr << "ptxas shim: expected input file and --output-file\n";
        return 2;
    }

    std::error_code error;
    std::filesystem::copy_file(
        input, output, std::filesystem::copy_options::overwrite_existing, error);
    if (error) {
        std::cerr << "ptxas shim: " << error.message() << '\n';
        return 1;
    }
    return 0;
}
