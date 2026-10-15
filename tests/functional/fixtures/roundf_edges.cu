#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

__global__ void roundf_edges(const float* input, float* output, unsigned count) {
    const unsigned i = threadIdx.x;
    if (i < count) output[i] = roundf(input[i]);
}

int main() {
    struct Case {
        std::uint32_t input;
        std::uint32_t expected;
    };
    // Exact binary32 inputs make ties and their immediate neighbors explicit.
    // The expected bits also distinguish negative zero from positive zero.
    constexpr Case cases[] = {
        {0x00000000u, 0x00000000u}, {0x80000000u, 0x80000000u},
        {0x00000001u, 0x00000000u}, {0x80000001u, 0x80000000u},
        {0x007fffffu, 0x00000000u}, {0x807fffffu, 0x80000000u},
        {0x3e800000u, 0x00000000u}, {0xbe800000u, 0x80000000u},
        {0x3f000000u, 0x3f800000u}, {0xbf000000u, 0xbf800000u},
        {0x3fc00000u, 0x40000000u}, {0xbfc00000u, 0xc0000000u},
        {0x40200000u, 0x40400000u}, {0xc0200000u, 0xc0400000u},
        {0x3effffffu, 0x00000000u}, {0xbeffffffu, 0x80000000u},
        {0x3f000001u, 0x3f800000u}, {0xbf000001u, 0xbf800000u},
        {0x3fbfffffu, 0x3f800000u}, {0xbfbfffffu, 0xbf800000u},
        {0x3fc00001u, 0x40000000u}, {0xbfc00001u, 0xc0000000u},
        {0x4affffffu, 0x4b000000u}, {0xcaffffffu, 0xcb000000u},
        {0x4b000000u, 0x4b000000u}, {0xcb000000u, 0xcb000000u},
        {0x7f7fffffu, 0x7f7fffffu}, {0xff7fffffu, 0xff7fffffu},
        {0x7f800000u, 0x7f800000u}, {0xff800000u, 0xff800000u},
        {0x7fc00000u, 0x7fc00000u}, {0xffc00000u, 0x7fc00000u},
    };
    constexpr unsigned count = sizeof(cases) / sizeof(cases[0]);
    float input[count];
    std::uint32_t result[count];
    for (unsigned i = 0; i < count; ++i) {
        std::memcpy(&input[i], &cases[i].input, sizeof(float));
        // Finite poison makes an unwritten result fail, including NaN cases.
        result[i] = 0xdeadbeefu;
    }
    auto check = [](cudaError_t status, const char* stage) {
        if (status == cudaSuccess) return true;
        std::fprintf(stderr, "FAIL: %s: %s\n", stage, cudaGetErrorString(status));
        return false;
    };
    float* device_input = nullptr;
    float* device_output = nullptr;
    if (!check(cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(input)),
               "input allocation") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(result)),
               "output allocation") ||
        !check(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice),
               "input copy") ||
        !check(cudaMemcpy(device_output, result, sizeof(result), cudaMemcpyHostToDevice),
               "output poison")) return 1;

    roundf_edges<<<1, 32>>>(device_input, device_output, count);
    if (!check(cudaGetLastError(), "launch") ||
        !check(cudaDeviceSynchronize(), "synchronize") ||
        !check(cudaMemcpy(result, device_output, sizeof(result), cudaMemcpyDeviceToHost),
               "output copy")) return 1;
    for (unsigned i = 0; i < count; ++i) {
        const bool expect_nan = (cases[i].expected & 0x7fffffffu) > 0x7f800000u;
        const bool actual_nan = (result[i] & 0x7fffffffu) > 0x7f800000u;
        if (expect_nan ? !actual_nan : result[i] != cases[i].expected) {
            std::fprintf(stderr,
                         "FAIL: roundf case %u input=%08x output=%08x expected=%08x\n",
                         i, cases[i].input, result[i], cases[i].expected);
            return 1;
        }
    }
    if (!check(cudaFree(device_output), "output free") ||
        !check(cudaFree(device_input), "input free")) return 1;
    std::puts("PASS: roundf ties, neighbors, signed zero, infinities and NaN on Apple GPU");
    return 0;
}
