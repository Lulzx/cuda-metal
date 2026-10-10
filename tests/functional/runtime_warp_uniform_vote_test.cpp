#include "cuda_runtime.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr unsigned int kWords = 6;
constexpr std::uint32_t kSentinel = 0xa5a5a5a5u;
struct Shape { unsigned int blocks, threads, count; };
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }
    const cudaError_t initialized = cudaInit(0);
    if (initialized == cudaErrorNoDevice) {
        std::fprintf(stderr, "SKIP: no supported Metal device\n");
        return 77;
    }
    if (initialized != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit returned %d\n", initialized);
        return 1;
    }
    static const cumetalKernelArgInfo_t kArgs[] = {
        {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BYTES, 4},
    };
    const cumetalKernel_t kernel{
        .metallib_path = argv[1], .kernel_name = "warp_uniform_vote",
        .arg_count = 3, .arg_info = kArgs,
    };
    std::size_t checked = 0;
    // The first shape exits lanes; the second has physically partial SIMD groups.
    for (const Shape shape : {Shape{3, 128, 269}, Shape{2, 45, 90}}) {
        const unsigned int capacity = shape.blocks * shape.threads;
        std::vector<std::uint32_t> predicates(capacity), output(capacity * kWords);
        std::uint32_t* device_predicates = nullptr;
        std::uint32_t* device_output = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&device_predicates), capacity * 4u) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&device_output), output.size() * 4u) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: allocation\n");
            return 1;
        }
        for (unsigned int mode = 0; mode < 4; ++mode) {
            for (unsigned int i = 0; i < capacity; ++i) {
                const unsigned int lane = (i % shape.threads) & 31u;
                const bool value = mode == 0 ? true : mode == 1 ? false :
                                   mode == 2 ? (lane & 1u) == 0u : lane < 16u;
                predicates[i] = value ? 7u : 0u;
            }
            std::fill(output.begin(), output.end(), kSentinel);
            if (cudaMemcpy(device_predicates, predicates.data(), capacity * 4u,
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(device_output, output.data(), output.size() * 4u,
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: input copy\n");
                return 1;
            }
            void* predicate_arg = device_predicates;
            void* output_arg = device_output;
            unsigned int count_arg = shape.count;
            void* arguments[] = {&predicate_arg, &output_arg, &count_arg};
            if (cudaLaunchKernel(&kernel, dim3(shape.blocks), dim3(shape.threads),
                                 arguments, 0, nullptr) != cudaSuccess ||
                cudaDeviceSynchronize() != cudaSuccess ||
                cudaMemcpy(output.data(), device_output, output.size() * 4u,
                           cudaMemcpyDeviceToHost) != cudaSuccess) {
                std::fprintf(stderr, "FAIL: uniform vote launch/readback\n");
                return 1;
            }
            for (unsigned int i = 0; i < capacity; ++i) {
                std::array<std::uint32_t, kWords> expected;
                expected.fill(kSentinel);
                if (i < shape.count) {
                    const unsigned int thread = i % shape.threads;
                    const unsigned int lane = thread & 31u;
                    const unsigned int warp_start = i - lane;
                    const unsigned int warp_end = std::min({warp_start + 32u,
                        i - thread + shape.threads, shape.count});
                    const std::array<std::uint32_t, 4> masks{
                        0xffffffffu, 0xa5a55a5au, 1u << 11u,
                        lane < 16u ? 0x0000ffffu : 0xffff0000u,
                    };
                    for (unsigned int group = 0; group < masks.size(); ++group) {
                        if ((masks[group] & (1u << lane)) == 0u) continue;
                        bool saw_true = false, saw_false = false;
                        std::uint32_t false_lanes = 0;
                        for (unsigned int peer = warp_start; peer < warp_end; ++peer) {
                            const std::uint32_t bit = 1u << (peer - warp_start);
                            if ((masks[group] & bit) == 0u) continue;
                            if (predicates[peer] != 0u) saw_true = true;
                            else { saw_false = true; false_lanes |= bit; }
                        }
                        expected[group] = !(saw_true && saw_false);
                        if (group == 3u) {
                            expected[4] = saw_false;
                            expected[5] = false_lanes;
                        }
                    }
                }
                for (unsigned int word = 0; word < kWords; ++word) {
                    if (output[i * kWords + word] != expected[word]) {
                        std::fprintf(stderr,
                            "FAIL: block=%u mode=%u thread=%u word=%u got=0x%08x expected=0x%08x\n",
                            shape.threads, mode, i, word, output[i * kWords + word], expected[word]);
                        return 1;
                    }
                    ++checked;
                }
            }
        }
        if (cudaFree(device_predicates) != cudaSuccess || cudaFree(device_output) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaFree\n");
            return 1;
        }
    }
    std::printf("PASS: uniform and negated warp votes validated (%zu words, 8 launches)\n", checked);
    return 0;
}
