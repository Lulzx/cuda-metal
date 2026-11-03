#include <cuda_runtime.h>

#include <array>
#include <cstdio>

__device__ __forceinline__ void update_local(float x, float* output) {
    if (output) {
        output[0] = x + 1.0f;
        output[1] = x - 1.0f;
    }
}

__global__ void compare_null(float* output, const float* maybe) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 67) return;
    float local[2];
    update_local(static_cast<float>(index), local);
    update_local(99.0f, nullptr);
    output[index] = local[0] + local[1] + (maybe == nullptr ? 1.0f : 2.0f) +
                    (nullptr != maybe ? 8.0f : 4.0f);
}

int main() {
    std::array<float, 96> actual;
    float* device = nullptr;
    if (cudaMalloc(&device, sizeof(actual)) != cudaSuccess) return 1;
    for (int nonnull = 0; nonnull < 2; ++nonnull) {
        actual.fill(-12345.0f);
        if (cudaMemcpy(device, actual.data(), sizeof(actual),
                       cudaMemcpyHostToDevice) != cudaSuccess) return 2;
        compare_null<<<1, 96>>>(device, nonnull ? device : nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess) return 3;
        if (cudaMemcpy(actual.data(), device, sizeof(actual),
                       cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
        for (int i = 0; i < 96; ++i) {
            const float expected = i < 67 ? 2.0f * i + (nonnull ? 10.0f : 5.0f)
                                          : -12345.0f;
            if (actual[i] != expected) {
                std::fprintf(stderr, "FAIL: mode %d index %d: %.9g expected %.9g\n",
                             nonnull, i, actual[i], expected);
                return 5;
            }
        }
    }
    cudaFree(device);
    std::puts("PASS: null and nonnull pointer comparisons and guards");
    return 0;
}
