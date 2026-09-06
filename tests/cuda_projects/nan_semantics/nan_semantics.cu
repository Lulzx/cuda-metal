// IEEE-754 comparison and classification semantics, which CUDA guarantees by
// default: `x != x` is true for NaN, every ordered comparison with NaN is
// false, and isnan/isinf/isfinite/signbit see through the bit pattern.
// Apple's Metal compiler defaults to fast-math, under which `NaN == NaN`
// was true and `x != x` false; CuMetal compiles generated MSL with the CUDA
// contract instead.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

__global__ void nan_semantics_probe(unsigned* output, const float* input, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float x = input[index];
    const float one = 1.0f;
    unsigned bits = 0;
    bits |= (x != x) ? 1u : 0u;
    bits |= (x == x) ? 2u : 0u;
    bits |= (x < one) ? 4u : 0u;
    bits |= (x >= one) ? 8u : 0u;
    bits |= !(x < one) ? 16u : 0u;
    bits |= isnan(x) ? 32u : 0u;
    bits |= isinf(x) ? 64u : 0u;
    bits |= isfinite(x) ? 128u : 0u;
    bits |= signbit(x) ? 256u : 0u;
    bits |= (fmaxf(x, one) == one) ? 512u : 0u;   // CUDA fmaxf ignores a NaN operand
    bits |= (fminf(x, one) == one) ? 1024u : 0u;
    output[index] = bits;
}

int main() {
    float host_input[6];
    std::memset(host_input, 0, sizeof(host_input));
    host_input[0] = std::nanf("");
    host_input[1] = -std::nanf("");
    host_input[2] = INFINITY;
    host_input[3] = -INFINITY;
    host_input[4] = 0.5f;
    host_input[5] = -2.0f;
    unsigned expected[6];
    for (int i = 0; i < 6; ++i) {
        const float x = host_input[i];
        unsigned bits = 0;
        bits |= (x != x) ? 1u : 0u;
        bits |= (x == x) ? 2u : 0u;
        bits |= (x < 1.0f) ? 4u : 0u;
        bits |= (x >= 1.0f) ? 8u : 0u;
        bits |= !(x < 1.0f) ? 16u : 0u;
        bits |= std::isnan(x) ? 32u : 0u;
        bits |= std::isinf(x) ? 64u : 0u;
        bits |= std::isfinite(x) ? 128u : 0u;
        bits |= std::signbit(x) ? 256u : 0u;
        bits |= (std::fmax(x, 1.0f) == 1.0f) ? 512u : 0u;
        bits |= (std::fmin(x, 1.0f) == 1.0f) ? 1024u : 0u;
        expected[i] = bits;
    }

    float* device_input = nullptr;
    unsigned* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }
    nan_semantics_probe<<<1, 32>>>(device_output, device_input, 6);
    unsigned host_output[6];
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }
    for (int i = 0; i < 6; ++i) {
        if (host_output[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: input %d (%g): got 0x%x, expected 0x%x\n", i,
                         host_input[i], host_output[i], expected[i]);
            return 1;
        }
    }
    std::printf("PASS: NaN and infinity follow IEEE comparison and classification semantics\n");
    return 0;
}
