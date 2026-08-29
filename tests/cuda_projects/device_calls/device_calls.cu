#include <cuda_runtime.h>

#include <cstdio>

__device__ __noinline__ int weighted_prefix(const int* first,
                                            const int* second,
                                            bool choose_first,
                                            unsigned count) {
    const int* selected = choose_first ? first : second;
    int sum = 0;
    for (unsigned index = 0; index < count; ++index) {
        const int value = selected[index];
        if (value < 0) {
            return sum - static_cast<int>(index);
        }
        sum += value * static_cast<int>(index + 1);
    }
    return sum;
}

__global__ void device_call_probe(const int* first, const int* second,
                                  int* output) {
    if (threadIdx.x == 0) {
        output[0] = weighted_prefix(first, second, true, 4);
        output[1] = weighted_prefix(first, second, false, 4);
        output[2] = weighted_prefix(first + 1, second + 1, true, 3);
    }
}

int main() {
    const int host_first[4] = {1, 2, 3, 4};
    const int host_second[4] = {5, 6, -7, 8};
    constexpr int expected[3] = {30, 15, 20};

    int* device_first = nullptr;
    int* device_second = nullptr;
    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_first), sizeof(host_first)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_second), sizeof(host_second)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) !=
            cudaSuccess ||
        cudaMemcpy(device_first, host_first, sizeof(host_first),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_second, host_second, sizeof(host_second),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }

    device_call_probe<<<1, 32>>>(device_first, device_second, device_output);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    int actual[3] = {};
    const cudaError_t copy_status =
        cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    cudaFree(device_second);
    cudaFree(device_first);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(sync_status),
                     cudaGetErrorString(copy_status));
        return 1;
    }

    for (int index = 0; index < 3; ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr,
                         "FAIL: output[%d]=%d, expected %d\n",
                         index, actual[index], expected[index]);
            return 1;
        }
    }
    std::printf("PASS: pointer-bearing device calls preserve loops, merges, and early exits\n");
    return 0;
}
