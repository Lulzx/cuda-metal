#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

struct Descriptor {
    float* values;
};

__global__ void read_nested_pointer(const Descriptor* descriptor, float* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[0] = descriptor[0].values[1] + 5.0f;
    }
}

__global__ void read_by_value_nested_pointer(Descriptor descriptor, float* output) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        output[1] = descriptor.values[0] + 9.0f;
    }
}

int main() {
    float* values = nullptr;
    Descriptor* descriptor = nullptr;
    float* output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&values), 2 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&descriptor), sizeof(Descriptor)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&output), 2 * sizeof(float)) != cudaSuccess) {
        return 1;
    }

    const float input[2] = {3.0f, 7.0f};
    const Descriptor host_descriptor{values};
    if (cudaMemcpy(values, input, sizeof(input), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(descriptor, &host_descriptor, sizeof(host_descriptor),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return 1;
    }

    read_nested_pointer<<<1, 1>>>(descriptor, output);
    read_by_value_nested_pointer<<<1, 1>>>(host_descriptor, output);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    float results[2]{};
    Descriptor round_trip{};
    if (cudaMemcpy(results, output, sizeof(results), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&round_trip, descriptor, sizeof(round_trip),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    if (std::fabs(results[0] - 12.0f) > 1.0e-6f ||
        std::fabs(results[1] - 12.0f) > 1.0e-6f || round_trip.values != values) {
        std::fprintf(stderr,
                     "FAIL: nested pointer results=(%g,%g) round_trip=%p expected_results=(12,12) expected_pointer=%p\n",
                     results[0], results[1], static_cast<void*>(round_trip.values),
                     static_cast<void*>(values));
        return 1;
    }

    cudaFree(output);
    cudaFree(descriptor);
    cudaFree(values);
    std::puts("PASS: runtime relocates and declares resident nested CUDA pointers in memory and by-value arguments");
    return 0;
}
