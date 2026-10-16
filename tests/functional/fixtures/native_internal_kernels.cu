// Native-AOT registration must resolve internal kernel stubs without promoting
// unrelated internal host functions.
#include <cuda_runtime.h>
#include <stdio.h>

namespace {
__global__ void anonymous_kernel(int* out, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = 3 * index + 11;
    }
}
}  // namespace

static __global__ void static_kernel(int* out, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = 400 - 2 * index;
    }
}

__global__ void external_kernel(int* out, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = (index ^ 0x35) + 700;
    }
}

static __attribute__((noinline)) int expected_value(int kernel, int index) {
    switch (kernel) {
        case 0: return index * 3 + 11;
        case 1: return 400 - index * 2;
        default: return (index ^ 0x35) + 700;
    }
}

static bool check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        printf("FAIL: %s: %s\n", operation, cudaGetErrorString(error));
        return false;
    }
    return true;
}

int main() {
    constexpr int count = 67;
    constexpr int stride = 96;
    constexpr int poison = -123456789;
    int output[3 * stride];
    for (int& value : output) {
        value = poison;
    }

    int* device = nullptr;
    if (!check(cudaMalloc(&device, sizeof(output)), "allocate") ||
        !check(cudaMemcpy(device, output, sizeof(output), cudaMemcpyHostToDevice),
               "initialize")) {
        return 1;
    }

    anonymous_kernel<<<3, 32>>>(device, count);
    if (!check(cudaGetLastError(), "anonymous kernel launch")) return 1;
    static_kernel<<<3, 32>>>(device + stride, count);
    if (!check(cudaGetLastError(), "static kernel launch")) return 1;
    external_kernel<<<3, 32>>>(device + 2 * stride, count);
    if (!check(cudaGetLastError(), "external kernel launch") ||
        !check(cudaDeviceSynchronize(), "synchronize") ||
        !check(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost),
               "copy results") ||
        !check(cudaFree(device), "free")) {
        return 1;
    }

    for (int kernel = 0; kernel < 3; ++kernel) {
        for (int index = 0; index < stride; ++index) {
            const int expected = index < count ? expected_value(kernel, index) : poison;
            const int actual = output[kernel * stride + index];
            if (actual != expected) {
                printf("FAIL: kernel %d output[%d] = %d, expected %d\n",
                       kernel, index, actual, expected);
                return 1;
            }
        }
    }

    printf("PASS: anonymous, static, and external native-AOT kernels produced correct results\n");
    return 0;
}
