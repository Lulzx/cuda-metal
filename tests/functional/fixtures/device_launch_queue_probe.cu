#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

__global__ void queue_leaf(int* value) {
    atomicAdd(value, 100);
}

__global__ void queue_child(int* value) {
    atomicAdd(value, 10);
    queue_leaf<<<1, 1>>>(value);
}

__global__ void queue_parent(int* value) {
    atomicAdd(value, 1);
    queue_child<<<1, 1>>>(value);
}

__global__ void queue_invalid_parent(int* value) {
    queue_leaf<<<0, 1>>>(value);
}

__global__ void queue_overflow_parent(int* value) {
    for (int i = 0; i < 1024; ++i) {
        queue_leaf<<<1, 1>>>(value);
    }
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "nested";
    int* device_value = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(int)) != cudaSuccess ||
        cudaMemset(device_value, 0, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device setup\n");
        return 1;
    }

    if (std::strcmp(mode, "nested") == 0) {
        queue_parent<<<1, 1>>>(device_value);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: nested queue synchronization\n");
            return 1;
        }
        int value = 0;
        if (cudaMemcpy(&value, device_value, sizeof(value), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            value != 111) {
            std::fprintf(stderr, "FAIL: nested queue value=%d expected=111\n", value);
            return 1;
        }
        cudaFree(device_value);
        std::printf("PASS: nested device launch queue value=111\n");
        return 0;
    }

    if (std::strcmp(mode, "invalid") == 0) {
        queue_invalid_parent<<<1, 1>>>(device_value);
        const cudaError_t status = cudaDeviceSynchronize();
        cudaFree(device_value);
        if (status != cudaErrorInvalidConfiguration) {
            std::fprintf(stderr,
                         "FAIL: invalid child configuration status=%d expected=%d\n",
                         static_cast<int>(status),
                         static_cast<int>(cudaErrorInvalidConfiguration));
            return 1;
        }
        std::printf("PASS: invalid child configuration propagated\n");
        return 0;
    }

    if (std::strcmp(mode, "overflow") == 0) {
        queue_overflow_parent<<<1, 1>>>(device_value);
        const cudaError_t status = cudaDeviceSynchronize();
        cudaFree(device_value);
        if (status != cudaErrorLaunchOutOfResources) {
            std::fprintf(stderr, "FAIL: queue overflow status=%d expected=%d\n",
                         static_cast<int>(status),
                         static_cast<int>(cudaErrorLaunchOutOfResources));
            return 1;
        }
        std::printf("PASS: device launch queue overflow propagated\n");
        return 0;
    }

    std::fprintf(stderr, "usage: %s [nested|invalid|overflow]\n", argv[0]);
    cudaFree(device_value);
    return 64;
}
