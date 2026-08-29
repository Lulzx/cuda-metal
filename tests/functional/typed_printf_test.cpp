#include "metal_backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {

using cumetal::metal_backend::Buffer;
using cumetal::metal_backend::KernelArg;

KernelArg bytes_arg(std::uint32_t value, std::size_t binding) {
    KernelArg argument;
    argument.kind = KernelArg::Kind::kBytes;
    argument.binding_index = binding;
    argument.bytes.resize(sizeof(value));
    std::memcpy(argument.bytes.data(), &value, sizeof(value));
    return argument;
}

KernelArg buffer_arg(const std::shared_ptr<Buffer>& buffer, std::size_t binding) {
    KernelArg argument;
    argument.kind = KernelArg::Kind::kBuffer;
    argument.buffer = buffer;
    argument.binding_index = binding;
    return argument;
}

bool launch(const char* metallib, const std::shared_ptr<Buffer>& ring,
            std::uint32_t capacity, dim3 grid, dim3 block,
            const std::shared_ptr<cumetal::metal_backend::Stream>& stream,
            std::string* error) {
    constexpr std::uint32_t kValue = 37;
    const std::vector<KernelArg> arguments = {
        bytes_arg(kValue, 0), buffer_arg(ring, 1), bytes_arg(capacity, 2)};
    const cumetal::metal_backend::LaunchConfig config{
        .grid = grid,
        .block = block,
        .shared_memory_bytes = 0,
        .provenance = "generic_typed_printf_test",
        .semantic_quality = "exact",
    };
    return cumetal::metal_backend::launch_kernel(
               metallib, "_Z17print_coordinatesi", config, arguments, stream,
               error) == cudaSuccess &&
           cumetal::metal_backend::stream_synchronize(stream, error) == cudaSuccess;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: usage: %s <device-printf.metallib>\n", argv[0]);
        return 77;
    }
    using namespace cumetal::metal_backend;
    std::string error;
    if (initialize(&error) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: Metal unavailable: %s\n", error.c_str());
        return 77;
    }

    constexpr std::uint32_t kCapacity = 1024;
    std::shared_ptr<Buffer> ring;
    std::shared_ptr<Stream> stream;
    if (allocate_buffer(kCapacity * sizeof(std::uint32_t), &ring, &error) !=
            cudaSuccess ||
        create_stream(&stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed printf setup: %s\n", error.c_str());
        return 1;
    }
    std::memset(ring->contents(), 0, ring->length());
    if (!launch(argv[1], ring, kCapacity, dim3(2, 2, 1), dim3(2, 2, 2),
                stream, &error)) {
        std::fprintf(stderr, "FAIL: typed printf launch: %s\n", error.c_str());
        return 1;
    }

    const auto* words = static_cast<const std::uint32_t*>(ring->contents());
    constexpr std::uint32_t kRecords = 32;
    constexpr std::uint32_t kRecordWords = 5;
    if (words[0] != kRecords * kRecordWords) {
        std::fprintf(stderr, "FAIL: typed printf cursor=%u expected=%u\n",
                     words[0], kRecords * kRecordWords);
        return 1;
    }
    std::array<std::array<bool, 8>, 4> seen{};
    for (std::uint32_t position = 0; position < words[0];
         position += kRecordWords) {
        const std::uint32_t format = words[position + 1];
        const std::uint32_t payload = words[position + 2];
        const std::uint32_t block = words[position + 3];
        const std::uint32_t thread = words[position + 4];
        const std::uint32_t value = words[position + 5];
        if (format != 0 || payload != 3 || block >= 4 || thread >= 8 ||
            value != 37 || seen[block][thread]) {
            std::fprintf(stderr,
                         "FAIL: malformed typed printf record at %u: "
                         "format=%u payload=%u block=%u thread=%u value=%u\n",
                         position, format, payload, block, thread, value);
            return 1;
        }
        seen[block][thread] = true;
    }
    for (const auto& block : seen) {
        for (bool thread : block) {
            if (!thread) {
                std::fprintf(stderr, "FAIL: typed printf omitted a GPU lane\n");
                return 1;
            }
        }
    }

    auto* mutable_words = static_cast<std::uint32_t*>(ring->contents());
    std::fill_n(mutable_words, kCapacity, 0xa5a5a5a5u);
    mutable_words[0] = 0;
    if (!launch(argv[1], ring, 5, dim3(1, 1, 1), dim3(1, 1, 1), stream,
                &error)) {
        std::fprintf(stderr, "FAIL: typed printf bounded launch: %s\n", error.c_str());
        return 1;
    }
    if (mutable_words[0] != 5) {
        std::fprintf(stderr, "FAIL: bounded typed printf cursor=%u expected=5\n",
                     mutable_words[0]);
        return 1;
    }
    for (std::size_t i = 1; i <= 5; ++i) {
        if (mutable_words[i] != 0xa5a5a5a5u) {
            std::fprintf(stderr,
                         "FAIL: bounded typed printf wrote rejected record word %zu\n",
                         i);
            return 1;
        }
    }
    if (destroy_stream(stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed printf stream cleanup: %s\n",
                     error.c_str());
        return 1;
    }
    std::puts("PASS: typed device printf records and bounds on Apple GPU");
    return 0;
}
