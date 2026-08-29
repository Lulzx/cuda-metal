#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdio>

namespace cg = cooperative_groups;

__global__ void cooperative_grid_barrier_probe(unsigned int* partial,
                                                unsigned int* result) {
    cg::grid_group grid = cg::this_grid();
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = blockIdx.x + 1u;
    }
    cg::sync(grid);
    if (grid.thread_rank() == 0) {
        unsigned int total = 0;
        for (unsigned int i = 0; i < gridDim.x; ++i) total += partial[i];
        result[0] = total;
    }
    cg::sync(grid);
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = result[0];
    }
}

int main() {
    constexpr unsigned int kBlocks = 4;
    unsigned int* partial = nullptr;
    unsigned int* result = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   kBlocks * sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&result),
                   sizeof(unsigned int)) != cudaSuccess) {
        return 1;
    }
    void* args[] = {&partial, &result};
    if (cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(cooperative_grid_barrier_probe),
            dim3(kBlocks), dim3(32), args, 0, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cooperative launch or synchronization failed\n");
        return 1;
    }
    unsigned int host[kBlocks] = {};
    if (cudaMemcpy(host, partial, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    for (unsigned int value : host) {
        if (value != 10u) {
            std::fprintf(stderr, "FAIL: grid barrier result=%u expected=10\n", value);
            return 1;
        }
    }
    cudaFree(result);
    cudaFree(partial);
    std::puts("PASS: cooperative grid barrier ordered four threadgroups");
    return 0;
}
