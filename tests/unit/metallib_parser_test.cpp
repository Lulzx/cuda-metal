#include "cumetal/common/metallib.h"

#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool has_metadata_key(const cumetal::common::KernelRecord& kernel, const std::string& key) {
    for (const auto& field : kernel.metadata) {
        if (field.key == key) {
            return true;
        }
    }
    return false;
}

bool has_metadata_value(const cumetal::common::KernelRecord& kernel,
                        const std::string& key,
                        const std::string& value) {
    for (const auto& field : kernel.metadata) {
        if (field.key == key && field.value == value) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <reference.metallib> <reference.experimental.metallib>\n", argv[0]);
        return 2;
    }

    const std::string apple_path = argv[1];
    const std::string experimental_path = argv[2];

    std::string error;
    const std::vector<std::uint8_t> apple_bytes =
        cumetal::common::read_file_bytes(apple_path, &error);
    if (!expect(!apple_bytes.empty(), "read Apple reference metallib")) {
        return 1;
    }

    const auto apple_summary = cumetal::common::inspect_metallib_bytes(apple_path, apple_bytes, 32);
    if (!expect(cumetal::common::looks_like_metallib(apple_summary), "Apple metallib magic check")) {
        return 1;
    }
    if (!expect(apple_summary.function_list_parsed, "Apple metallib function list parsed")) {
        return 1;
    }
    if (!expect(apple_summary.function_list_parser == "metallib-function-list",
                "Apple metallib parser kind")) {
        return 1;
    }
    if (!expect(!apple_summary.kernels.empty(), "Apple metallib has at least one kernel")) {
        return 1;
    }
    if (!expect(apple_summary.kernels.front().name == "vector_add",
                "Apple metallib kernel name is vector_add")) {
        return 1;
    }
    if (!expect(apple_summary.kernels.front().bitcode_size > 0, "Apple metallib bitcode size set")) {
        return 1;
    }
    if (!expect(apple_summary.kernels.front().bitcode_offset + apple_summary.kernels.front().bitcode_size <=
                    apple_bytes.size(),
                "Apple metallib bitcode bounds in-range")) {
        return 1;
    }
    if (!expect(apple_summary.kernels.front().bitcode_signature_ok,
                "Apple metallib bitcode signature recognized")) {
        return 1;
    }
    if (!expect(has_metadata_key(apple_summary.kernels.front(), "air.version"),
                "Apple metallib metadata includes air.version")) {
        return 1;
    }
    if (!expect(has_metadata_key(apple_summary.kernels.front(), "air.kernel"),
                "Apple metallib metadata includes air.kernel marker")) {
        return 1;
    }
    if (!expect(has_metadata_key(apple_summary.kernels.front(), "offt.bitcode"),
                "Apple metallib metadata includes OFFT bitcode offset")) {
        return 1;
    }

    // Production xcrun metallibs do not retain CuMetal's custom metadata, but
    // PTX-lowered AIR preserves <kernel>_param_N reflection names. Verify the
    // parser recovers an exact, contiguous argument count from those strings.
    std::vector<std::uint8_t> reflected_apple = apple_bytes;
    for (const char* name : {"vector_add_param_0", "vector_add_param_1",
                             "vector_add_param_2_12_"}) {
        reflected_apple.insert(reflected_apple.end(), name, name + std::strlen(name) + 1);
    }
    const auto reflected_summary =
        cumetal::common::inspect_metallib_bytes("reflected-reference.metallib",
                                                reflected_apple, 0);
    if (!expect(has_metadata_value(reflected_summary.kernels.front(),
                                   "kernel.arg_count", "3"),
                "Apple metallib AIR reflection yields exact argument count")) {
        return 1;
    }

    std::vector<std::uint8_t> truncated_apple;
    if (apple_bytes.size() > 64) {
        truncated_apple.assign(apple_bytes.begin(), apple_bytes.begin() + 64);
    } else {
        truncated_apple = apple_bytes;
    }
    const auto truncated_summary =
        cumetal::common::inspect_metallib_bytes("truncated-reference.metallib", truncated_apple, 8);
    if (!expect(cumetal::common::looks_like_metallib(truncated_summary), "truncated magic check")) {
        return 1;
    }
    if (!expect(!truncated_summary.function_list_parsed, "truncated metallib should not parse function list")) {
        return 1;
    }

    error.clear();
    const std::vector<std::uint8_t> experimental_bytes =
        cumetal::common::read_file_bytes(experimental_path, &error);
    if (!expect(!experimental_bytes.empty(), "read experimental reference metallib")) {
        return 1;
    }
    const auto experimental_summary =
        cumetal::common::inspect_metallib_bytes(experimental_path, experimental_bytes, 32);
    if (!expect(experimental_summary.function_list_parsed, "experimental function list parsed")) {
        return 1;
    }
    if (!expect(experimental_summary.function_list_parser == "cumetal-experimental",
                "experimental parser kind")) {
        return 1;
    }
    if (!expect(experimental_summary.kernels.size() == 1, "experimental kernel count")) {
        return 1;
    }
    if (!expect(experimental_summary.kernels.front().name == "vector_add",
                "experimental kernel name")) {
        return 1;
    }
    if (!expect(has_metadata_key(experimental_summary.kernels.front(), "air.kernel"),
                "experimental metadata includes air.kernel")) {
        return 1;
    }
    if (!expect(has_metadata_key(experimental_summary.kernels.front(), "air.version"),
                "experimental metadata includes air.version")) {
        return 1;
    }
    if (!expect(has_metadata_value(experimental_summary.kernels.front(), "kernel.arg_count", "3"),
                "experimental metadata includes exact kernel argument count")) {
        return 1;
    }

    std::printf("PASS: metallib parser unit tests\n");
    return 0;
}
