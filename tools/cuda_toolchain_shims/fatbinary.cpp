#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace {

void append_u16_le(std::ofstream& output, std::uint16_t value) {
    output.put(static_cast<char>(value & 0xffu));
    output.put(static_cast<char>((value >> 8u) & 0xffu));
}

void append_u32_le(std::ofstream& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xffu));
    }
}

void append_u64_le(std::ofstream& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.put(static_cast<char>((value >> shift) & 0xffu));
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string output_path;
    std::string image_path;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--create" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg.rfind("--create=", 0) == 0) {
            output_path = arg.substr(9);
        } else if (arg.rfind("--image=", 0) == 0 ||
                   arg.rfind("--image3=", 0) == 0) {
            const std::size_t file = arg.rfind("file=");
            if (file != std::string::npos) {
                image_path = arg.substr(file + 5);
            }
        }
    }

    if (output_path.empty() || image_path.empty()) {
        std::cerr << "fatbinary shim: expected --create and an image file\n";
        return 2;
    }

    std::ifstream input(image_path, std::ios::binary);
    if (!input) {
        std::cerr << "fatbinary shim: failed to open " << image_path << '\n';
        return 1;
    }
    const std::vector<char> payload((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
    if (payload.empty()) {
        std::cerr << "fatbinary shim: failed to read " << image_path << '\n';
        return 1;
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    if (!output) {
        std::cerr << "fatbinary shim: failed to create " << output_path << '\n';
        return 1;
    }

    append_u32_le(output, 0xBA55ED50u);
    append_u16_le(output, 1u);
    append_u16_le(output, 16u);
    append_u64_le(output, static_cast<std::uint64_t>(payload.size()));
    output.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    return output ? 0 : 1;
}
