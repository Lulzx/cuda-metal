#include <cuda_runtime.h>

#include <cstdio>

__global__ void print_coordinates(int value) {
    const int block = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x);
    const int thread = static_cast<int>(
        threadIdx.z * blockDim.x * blockDim.y +
        threadIdx.y * blockDim.x + threadIdx.x);
    printf("PRINTF[%d,%d]=%d\n", block, thread, value);
}

__global__ void print_wide_values(const int* pointer) {
    const long long signed_value = -1234567890123LL;
    const unsigned long long unsigned_value = 0x1122334455667788ULL;
    const size_t size_value = static_cast<size_t>(0x200000007ULL);
    printf("WIDE signed=%lld unsigned=%llu hex=%#llx size=%zu ptr=%p "
           "float=%.3f char=%c percent=%%\n",
           signed_value, unsigned_value, unsigned_value, size_value, pointer,
           3.125, 'Q');
}

__global__ void print_dynamic_values() {
    printf("DYNAMIC int=%*d float=%*.*f left=%-*u\n",
           6, -42, 8, 2, 3.125, 5, 7u);
}

int main() {
    print_coordinates<<<dim3(2, 2), dim3(2, 2, 2)>>>(37);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }
    int* pointer = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&pointer), sizeof(int)) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        return 1;
    }
    print_wide_values<<<1, 1>>>(pointer);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: wide cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    print_dynamic_values<<<1, 1>>>();
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: dynamic cudaDeviceSynchronize: %s\n",
                    cudaGetErrorString(error));
        cudaFree(pointer);
        return 1;
    }
    cudaFree(pointer);
    std::printf("HOST_DONE\n");
    return 0;
}
