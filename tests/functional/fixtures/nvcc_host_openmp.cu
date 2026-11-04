#include <cuda_runtime.h>
#include <cstdio>

#ifndef HOST_OFFSET
#define HOST_OFFSET 0
#endif

#ifdef WITH_KERNEL
__global__ void fill_values(int* values) {
    values[threadIdx.x] = 3 * static_cast<int>(threadIdx.x) + 1;
}
#endif

int main() {
    int count = 0;
#pragma omp parallel num_threads(2) reduction(+:count)
    { count += 1; }
    if (count + HOST_OFFSET != 19) {
        std::fprintf(stderr, "host options lost: threads=%d offset=%d\n", count, HOST_OFFSET);
        return 1;
    }
#ifndef _OPENMP
    std::fprintf(stderr, "OpenMP compilation disabled\n");
    return 1;
#endif
#ifdef WITH_KERNEL
    constexpr int size = 29;
    int* device = nullptr;
    int values[size] = {};
    if (cudaMalloc(&device, sizeof(values)) != cudaSuccess) return 1;
    fill_values<<<1, size>>>(device);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(values, device, sizeof(values), cudaMemcpyDeviceToHost) != cudaSuccess)
        return 1;
    if (cudaFree(device) != cudaSuccess) return 1;
    for (int i = 0; i < size; ++i) {
        if (values[i] != 3 * i + 1) return 1;
    }
#endif
    std::printf("PASS: host OpenMP options\n");
    return 0;
}
