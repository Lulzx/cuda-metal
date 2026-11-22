// Compute encoders carry the kernel name for Instruments / GPU captures
// (#227, ported from MaansAndersson/cuda-metal dac98c14): a batched encoder is
// named after its first kernel and is not relabelled as dispatches join it,
// and labelling does not change results.
#include "metal_backend.h"

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

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

    constexpr std::size_t kElements = 1024;
    std::shared_ptr<Buffer> a, b, out;
    const std::size_t bytes = kElements * sizeof(float);
    if (allocate_buffer(bytes, &a, &error) != cudaSuccess ||
        allocate_buffer(bytes, &b, &error) != cudaSuccess ||
        allocate_buffer(bytes, &out, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation: %s\n", error.c_str());
        return 1;
    }
    auto* ha = static_cast<float*>(a->contents());
    auto* hb = static_cast<float*>(b->contents());
    for (std::size_t i = 0; i < kElements; ++i) {
        ha[i] = static_cast<float>(i);
        hb[i] = 2.0f;
    }

    std::shared_ptr<Stream> stream;
    if (create_stream(&stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stream: %s\n", error.c_str());
        return 1;
    }
    if (!stream_open_compute_encoder_label(stream).empty()) {
        std::fprintf(stderr, "FAIL: a fresh stream reports an open encoder\n");
        return 1;
    }

    LaunchConfig config{
        .grid = dim3(kElements / 256, 1, 1),
        .block = dim3(256, 1, 1),
        .shared_memory_bytes = 0,
        .provenance = "precompiled_metallib",
        .semantic_quality = "exact",
    };
    auto arg = [](const std::shared_ptr<Buffer>& buffer) {
        KernelArg value;
        value.kind = KernelArg::Kind::kBuffer;
        value.buffer = buffer;
        return value;
    };
    const std::vector<KernelArg> args{arg(a), arg(b), arg(out)};

    if (launch_kernel(argv[1], "vector_add", config, args, stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: first launch: %s\n", error.c_str());
        return 1;
    }
    const std::string first = stream_open_compute_encoder_label(stream);
    if (first.empty()) {
        // Batching disabled in this environment (e.g. CUMETAL_TRACE_GPU=1):
        // the encoder was already ended, so there is nothing to inspect.
        if (stream_synchronize(stream, &error) != cudaSuccess) return 1;
        std::puts("SKIP: launch batching is disabled; no open encoder to inspect");
        return 77;
    }
    if (first != "vector_add") {
        std::fprintf(stderr, "FAIL: encoder label '%s', want 'vector_add'\n", first.c_str());
        return 1;
    }
    if (launch_kernel(argv[1], "vector_add", config, args, stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: second launch: %s\n", error.c_str());
        return 1;
    }
    if (stream_open_compute_encoder_label(stream) != "vector_add") {
        std::fprintf(stderr, "FAIL: a batched dispatch relabelled the encoder to '%s'\n",
                     stream_open_compute_encoder_label(stream).c_str());
        return 1;
    }
    if (stream_synchronize(stream, &error) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: synchronize: %s\n", error.c_str());
        return 1;
    }
    if (!stream_open_compute_encoder_label(stream).empty()) {
        std::fprintf(stderr, "FAIL: encoder still open after synchronize\n");
        return 1;
    }
    const auto* result = static_cast<const float*>(out->contents());
    for (std::size_t i = 0; i < kElements; ++i) {
        if (result[i] != static_cast<float>(i) + 2.0f) {
            std::fprintf(stderr, "FAIL: out[%zu] = %f\n", i, static_cast<double>(result[i]));
            return 1;
        }
    }
    std::puts("PASS: compute encoders are labelled with their first kernel");
    return 0;
}
