#include "metal_backend.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kElements = 1u << 15;
constexpr unsigned int kThreads = 256;
}

int main(int argc, char** argv) {
    if (argc != 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: usage: %s <reference.metallib>\n", argv[0]);
        return 77;
    }

    using namespace cumetal::metal_backend;
    std::string error;
    if (initialize(&error) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: Metal unavailable: %s\n", error.c_str());
        return 77;
    }

    std::shared_ptr<Buffer> a, b, intermediate, output;
    const std::size_t bytes = kElements * sizeof(float);
    if (allocate_buffer(bytes, &a, &error) != cudaSuccess ||
        allocate_buffer(bytes, &b, &error) != cudaSuccess ||
        allocate_buffer(bytes, &intermediate, &error) != cudaSuccess ||
        allocate_buffer(bytes, &output, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: buffer allocation: %s\n", error.c_str());
        return 1;
    }
    auto* host_a = static_cast<float*>(a->contents());
    auto* host_b = static_cast<float*>(b->contents());
    for (std::size_t i = 0; i < kElements; ++i) {
        host_a[i] = static_cast<float>((i * 7) % 31) * 0.25f;
        host_b[i] = static_cast<float>((i * 11) % 37) * 0.5f;
    }

    std::shared_ptr<Stream> producer, consumer;
    if (create_stream(&producer, &error) != cudaSuccess ||
        create_stream(&consumer, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stream creation: %s\n", error.c_str());
        return 1;
    }

    LaunchConfig config{
        .grid = dim3(static_cast<unsigned int>((kElements + kThreads - 1) / kThreads), 1, 1),
        .block = dim3(kThreads, 1, 1),
        .shared_memory_bytes = 0,
        .provenance = "precompiled_metallib",
        .semantic_quality = "exact",
    };
    auto buffer_arg = [](const std::shared_ptr<Buffer>& buffer) {
        KernelArg arg;
        arg.kind = KernelArg::Kind::kBuffer;
        arg.buffer = buffer;
        return arg;
    };

    const std::vector<KernelArg> first{
        buffer_arg(a), buffer_arg(b), buffer_arg(intermediate)};
    const std::vector<KernelArg> second{
        buffer_arg(intermediate), buffer_arg(b), buffer_arg(output)};
    if (launch_kernel(argv[1], "vector_add", config, first, producer, &error) != cudaSuccess ||
        launch_kernel(argv[1], "vector_add", config, second, consumer, &error) != cudaSuccess ||
        stream_synchronize(consumer, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: fenced launch: %s\n", error.c_str());
        return 1;
    }

    const auto* got = static_cast<const float*>(output->contents());
    for (std::size_t i = 0; i < kElements; ++i) {
        const float expected = host_a[i] + 2.0f * host_b[i];
        if (std::fabs(got[i] - expected) > 1e-5f) {
            std::fprintf(stderr, "FAIL: cross-queue mismatch at %zu: got=%g expected=%g\n",
                         i, static_cast<double>(got[i]), static_cast<double>(expected));
            return 1;
        }
    }

    if (destroy_stream(producer, &error) != cudaSuccess ||
        destroy_stream(consumer, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stream destruction: %s\n", error.c_str());
        return 1;
    }
    std::puts("PASS: shared-buffer MTLSharedEvent fence ordered two command queues");
    return 0;
}
