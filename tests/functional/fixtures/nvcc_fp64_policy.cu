#include <cuda_runtime.h>

#include <cstdio>

__global__ void multiply(const double* input, double* output) {
    if (threadIdx.x == 0) output[0] = input[0] * 2.0;
}

int main() {
    const double input = 1.0e-200;
    double output = -777.0;
    double* device_input = nullptr;
    double* device_output = nullptr;
    if (cudaMalloc(&device_input, sizeof(input)) != cudaSuccess ||
        cudaMalloc(&device_output, sizeof(output)) != cudaSuccess) return 1;
    if (cudaMemcpy(device_input, &input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_output, &output, sizeof(output), cudaMemcpyHostToDevice) != cudaSuccess) return 2;
    multiply<<<1, 1>>>(device_input, device_output);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    if (cudaMemcpy(&output, device_output, sizeof(output), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    cudaFree(device_input);
    cudaFree(device_output);
    const double expected = input * 2.0;
    if (output != expected) {
        std::fprintf(stderr, "FAIL: actual %.17g expected %.17g\n", output, expected);
        return 5;
    }
    std::puts("PASS: nvcc preserved IEEE64 exponent range");
    return 0;
}
