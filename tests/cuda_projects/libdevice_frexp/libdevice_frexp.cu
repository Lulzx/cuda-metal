#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdio>

__global__ void frexp_kernel(const float* input, unsigned* mantissa_bits, int* exponent, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    int e = 0;
    const double m = frexp(static_cast<double>(input[i]), &e);
    mantissa_bits[i] = __float_as_uint(static_cast<float>(m));
    exponent[i] = e;
}

int main() {
    constexpr int n = 8;
    const float input[n] = {0.0f, 2.0f, 1.0f, -1.0f, 3.0f, 0.125f, 1024.0f, -96.0f};
    const float expected_m[n] = {0.0f, 0.5f, 0.5f, -0.5f, 0.75f, 0.5f, 0.5f, -0.75f};
    const int expected_e[n] = {0, 2, 1, 1, 2, -2, 11, 7};

    float* d_input = nullptr;
    unsigned* d_mantissa_bits = nullptr;
    int* d_exponent = nullptr;
    cudaMalloc(&d_input, sizeof(input));
    cudaMalloc(&d_mantissa_bits, sizeof(expected_m));
    cudaMalloc(&d_exponent, sizeof(expected_e));
    cudaMemcpy(d_input, input, sizeof(input), cudaMemcpyHostToDevice);
    frexp_kernel<<<1, 32>>>(d_input, d_mantissa_bits, d_exponent, n);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    unsigned mantissa_bits[n] = {};
    int exponent[n] = {};
    cudaMemcpy(mantissa_bits, d_mantissa_bits, sizeof(mantissa_bits), cudaMemcpyDeviceToHost);
    cudaMemcpy(exponent, d_exponent, sizeof(exponent), cudaMemcpyDeviceToHost);
    int failures = 0;
    for (int i = 0; i < n; ++i) {
        float mantissa = 0.0f;
        std::memcpy(&mantissa, &mantissa_bits[i], sizeof(mantissa));
        if (mantissa != expected_m[i] || exponent[i] != expected_e[i]) {
            std::printf("FAIL: frexp[%d] got (%g,%d), expected (%g,%d)\n",
                        i, mantissa, exponent[i], expected_m[i], expected_e[i]);
            ++failures;
        }
    }
    cudaFree(d_exponent);
    cudaFree(d_mantissa_bits);
    cudaFree(d_input);
    if (failures != 0) return 1;
    std::printf("PASS: device frexp returned correct mantissas and exponents\n");
    return 0;
}
