// Non-finite float constants in device code. The direct .cu path imports
// LLVM IR, whose APFloat spells these values "-Inf", "+Inf" and "NaN"; those
// used to reach Metal verbatim ("use of undeclared identifier 'Inf'"), so any
// kernel seeding a max reduction with -INFINITY failed to compile.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

__global__ void non_finite_constants(const float* in, int n, float* out) {
    float maximum = -INFINITY;
    float minimum = INFINITY;
    for (int i = 0; i < n; ++i) {
        maximum = fmaxf(maximum, in[i]);
        minimum = fminf(minimum, in[i]);
    }
    out[0] = maximum;
    out[1] = minimum;
    // n == 0 keeps the seeds, so the raw constants reach the output.
    out[2] = n > 0 ? in[0] : -INFINITY;
    out[3] = n > 0 ? in[0] : INFINITY;
    out[4] = n > 0 ? in[0] : NAN;
}

static std::uint32_t bits(float value) {
    std::uint32_t result = 0;
    std::memcpy(&result, &value, sizeof result);
    return result;
}

int main() {
    const float host_in[3] = {-2.5f, 7.0f, 1.0f};
    float* in = nullptr;
    float* out = nullptr;
    if (cudaMalloc(&in, sizeof host_in) != cudaSuccess ||
        cudaMalloc(&out, 5 * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(in, host_in, sizeof host_in, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::puts("FAIL: setup");
        return 1;
    }
    float result[5] = {};
    non_finite_constants<<<1, 1>>>(in, 3, out);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(result, out, sizeof result, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::puts("FAIL: launch with data");
        return 1;
    }
    if (result[0] != 7.0f || result[1] != -2.5f) {
        std::printf("FAIL: max=%g min=%g\n", result[0], result[1]);
        return 1;
    }
    non_finite_constants<<<1, 1>>>(in, 0, out);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(result, out, sizeof result, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::puts("FAIL: launch without data");
        return 1;
    }
    if (bits(result[0]) != 0xff800000u || bits(result[1]) != 0x7f800000u ||
        bits(result[2]) != 0xff800000u || bits(result[3]) != 0x7f800000u ||
        !std::isnan(result[4])) {
        std::printf("FAIL: seeds %08x %08x %08x %08x %08x\n", bits(result[0]), bits(result[1]),
                    bits(result[2]), bits(result[3]), bits(result[4]));
        return 1;
    }
    cudaFree(in);
    cudaFree(out);
    std::puts("PASS: non-finite float constants reach the device exactly");
    return 0;
}
