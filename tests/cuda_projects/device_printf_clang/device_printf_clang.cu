#include <cuda_runtime.h>

#include <cstdio>

__global__ void print_coordinates(int value) {
    const int block = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x);
    const int thread = static_cast<int>(
        threadIdx.z * blockDim.x * blockDim.y +
        threadIdx.y * blockDim.x + threadIdx.x);
    printf("PRINTF[%d,%d]=%d\n", block, thread, value);
}

int main() {
    print_coordinates<<<dim3(2, 2), dim3(2, 2, 2)>>>(37);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }
    std::printf("HOST_DONE\n");
    return 0;
}
