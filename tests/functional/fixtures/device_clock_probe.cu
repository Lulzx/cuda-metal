#include <cuda_runtime.h>

#include <cstdio>

__global__ void device_clock_probe(unsigned int* elapsed) {
    const unsigned int start = clock();
    unsigned int now = start;
    while (now - start < 4096u) {
        now = clock();
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        elapsed[0] = now - start;
    }
}

int main() {
    unsigned int* device_elapsed = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_elapsed),
                   sizeof(unsigned int)) != cudaSuccess) {
        return 1;
    }
    device_clock_probe<<<1, 1>>>(device_elapsed);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;

    unsigned int elapsed = 0;
    if (cudaMemcpy(&elapsed, device_elapsed, sizeof(elapsed),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    cudaFree(device_elapsed);
    if (elapsed < 4096u) {
        std::fprintf(stderr, "FAIL: device clock did not advance: %u\n", elapsed);
        return 1;
    }
    std::printf("PASS: emulated device clock advanced monotonically (%u ticks)\n",
                elapsed);
    return 0;
}
