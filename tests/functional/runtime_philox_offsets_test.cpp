#include "cuda_runtime.h"
#include "curand_kernel.h"

#include <array>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
constexpr unsigned int kDraws = 8;
constexpr unsigned int kThreads = 64;
constexpr std::uint32_t kSentinel = 0xa5a5a5a5u;

std::array<std::uint32_t, kDraws> expected_draws(unsigned long long seed,
                                              unsigned long long subsequence,
                                              unsigned long long offset) {
    // The documented subsequence stride is 2^66 scalar outputs. Four outputs
    // per Philox block put the subsequence in the counter's upper 64 bits and
    // offset / 4 in its lower 64 bits. Do not call the initializer under test.
    // https://docs.nvidia.com/cuda/curand/group__DEVICE.html
    curandStatePhilox4_32_10_t state{};
    const unsigned long long block = offset / 4u;
    state.key[0] = static_cast<std::uint32_t>(seed);
    state.key[1] = static_cast<std::uint32_t>(seed >> 32);
    state.ctr[0] = static_cast<std::uint32_t>(block);
    state.ctr[1] = static_cast<std::uint32_t>(block >> 32);
    state.ctr[2] = static_cast<std::uint32_t>(subsequence);
    state.ctr[3] = static_cast<std::uint32_t>(subsequence >> 32);
    for (unsigned int i = 0; i < offset % 4u; ++i) (void)curand(&state);
    std::array<std::uint32_t, kDraws> result{};
    for (auto& value : result) value = curand(&state);
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }
    // Published Philox4x32-10 counter-zero/key-zero known-answer vector:
    // https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors
    // This anchors the reused round core; the oracle above independently tests
    // initialization and offset placement, not full algorithm equivalence.
    constexpr std::array<std::uint32_t, 4> known{
        0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u,
    };
    const auto zero = expected_draws(0, 0, 0);
    for (unsigned int i = 0; i < known.size(); ++i) {
        if (zero[i] != known[i]) {
            std::fprintf(stderr, "FAIL: host Philox known-answer draw %u\n", i);
            return 1;
        }
    }

    std::vector<unsigned long long> inputs;
    std::vector<std::array<std::uint32_t, kDraws>> expected;
    for (const auto seed : {0ull, 789ull, 0xfedcba9876543210ull}) {
        for (const auto subsequence : {0ull, 1ull, 0x1234567800000009ull, ULLONG_MAX}) {
            for (const auto offset : {0ull, 1ull, 2ull, 3ull, 4ull, 5ull, 7ull, 8ull,
                                      15ull, 16ull, 17ull, (1ull << 34) - 1ull,
                                      1ull << 34, (1ull << 63) + 1ull, ULLONG_MAX}) {
                inputs.insert(inputs.end(), {seed, subsequence, offset});
                expected.push_back(expected_draws(seed, subsequence, offset));
            }
        }
    }
    unsigned int count = static_cast<unsigned int>(expected.size());
    const unsigned int blocks = (count + kThreads - 1u) / kThreads;
    const unsigned int capacity = blocks * kThreads;
    // One leading slot, inactive-thread slots, and one trailing slot are guards.
    std::vector<std::uint32_t> output((capacity + 2u) * kDraws, kSentinel);
    const cudaError_t initialized = cudaInit(0);
    if (initialized == cudaErrorNoDevice) {
        std::fprintf(stderr, "SKIP: no supported Metal device\n");
        return 77;
    }
    if (initialized != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit returned %d\n", initialized);
        return 1;
    }
    unsigned long long* device_inputs = nullptr;
    std::uint32_t* device_output = nullptr;
    auto check = [](cudaError_t status, const char* operation) {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "FAIL: %s returned %d\n", operation, status);
        return false;
    };
    auto release = [&] {
        const bool first = check(cudaFree(device_inputs), "cudaFree(inputs)");
        const bool second = check(cudaFree(device_output), "cudaFree(output)");
        return first && second;
    };
    const std::size_t input_bytes = inputs.size() * sizeof(inputs[0]);
    const std::size_t output_bytes = output.size() * sizeof(output[0]);
    if (!check(cudaMalloc(reinterpret_cast<void**>(&device_inputs), input_bytes), "cudaMalloc(inputs)") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&device_output), output_bytes), "cudaMalloc(output)") ||
        !check(cudaMemcpy(device_inputs, inputs.data(), input_bytes, cudaMemcpyHostToDevice), "copy inputs") ||
        !check(cudaMemcpy(device_output, output.data(), output_bytes, cudaMemcpyHostToDevice), "poison output")) {
        release();
        return 1;
    }
    static const cumetalKernelArgInfo_t kArgs[] = {
        {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BYTES, 4},
    };
    const cumetalKernel_t kernel{
        .metallib_path = argv[1], .kernel_name = "philox_offsets",
        .arg_count = 3, .arg_info = kArgs,
    };
    void* arguments[] = {&device_inputs, &device_output, &count};
    if (!check(cudaLaunchKernel(&kernel, dim3(blocks), dim3(kThreads), arguments, 0, nullptr), "Philox launch") ||
        !check(cudaGetLastError(), "cudaGetLastError") ||
        !check(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost), "read output")) {
        release();
        return 1;
    }
    if (!release()) return 1;
    for (unsigned int slot = 0; slot < capacity + 2u; ++slot) {
        for (unsigned int draw = 0; draw < kDraws; ++draw) {
            const bool active = slot > 0u && slot <= count;
            const std::uint32_t wanted = active ? expected[slot - 1u][draw] : kSentinel;
            const std::uint32_t actual = output[slot * kDraws + draw];
            if (actual == wanted) continue;
            if (active) {
                const auto* input = &inputs[(slot - 1u) * 3u];
                std::fprintf(stderr,
                    "FAIL: seed=%llu subsequence=%llu offset=%llu draw=%u got=0x%08x expected=0x%08x\n",
                    input[0], input[1], input[2], draw, actual, wanted);
            } else {
                std::fprintf(stderr, "FAIL: guard slot=%u draw=%u got=0x%08x expected=0x%08x\n",
                             slot, draw, actual, wanted);
            }
            return 1;
        }
    }
    std::printf("PASS: Philox initialization and offsets validated (%u cases, %u draws, %u guard words)\n",
                count, count * kDraws, (capacity + 2u - count) * kDraws);
    return 0;
}
