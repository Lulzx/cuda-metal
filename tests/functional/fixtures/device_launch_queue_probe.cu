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

__global__ void queue_bump(int* value) {
    atomicAdd(value, 1);
}

__global__ void queue_copy_parent(int* dst, const int* src, int* status) {
    if (threadIdx.x == 0) {
        const cudaError_t copy_status =
            cudaMemcpyAsync(dst, src, sizeof(int), cudaMemcpyDeviceToDevice, 0);
        *status = static_cast<int>(copy_status);
        queue_bump<<<1, 1>>>(dst);
    }
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

    if (std::strcmp(mode, "memcpy") == 0) {
        int* device_src = nullptr;
        int* device_status = nullptr;
        const int seed = 41;
        if (cudaMalloc(reinterpret_cast<void**>(&device_src), sizeof(int)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&device_status), sizeof(int)) != cudaSuccess ||
            cudaMemcpy(device_src, &seed, sizeof(seed), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            std::fprintf(stderr, "FAIL: memcpy queue setup\n");
            return 1;
        }
        queue_copy_parent<<<1, 1>>>(device_value, device_src, device_status);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: memcpy queue synchronization\n");
            return 1;
        }
        int value = 0;
        int status = -1;
        if (cudaMemcpy(&value, device_value, sizeof(value), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(&status, device_status, sizeof(status), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            value != 42 || status != 0) {
            std::fprintf(stderr,
                         "FAIL: device memcpy queue value=%d status=%d expected=42,0\n",
                         value, status);
            return 1;
        }
        cudaFree(device_src);
        cudaFree(device_status);
        cudaFree(device_value);
        std::printf("PASS: device memcpy queue record copied then launched value=42\n");
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

    std::fprintf(stderr, "usage: %s [nested|memcpy|invalid|overflow]\n", argv[0]);
    cudaFree(device_value);
    return 64;
}
