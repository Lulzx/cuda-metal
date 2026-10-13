#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdio>

namespace cg = cooperative_groups;

// Many consecutive grid barriers: every phase reuses the same barrier state,
// so a release that becomes visible before the arrival-count reset wedges a
// later phase instead of merely corrupting one result.
__global__ void cooperative_grid_barrier_stress(unsigned int* partial,
                                                 unsigned int* result,
                                                 unsigned int phases) {
    cg::grid_group grid = cg::this_grid();
    unsigned int checksum = 0;
    for (unsigned int phase = 0; phase < phases; ++phase) {
        if (threadIdx.x == 0) {
            partial[blockIdx.x] = phase * gridDim.x + blockIdx.x + 1u;
        }
        cg::sync(grid);
        if (threadIdx.x == 0) {
            unsigned int total = 0;
            for (unsigned int i = 0; i < gridDim.x; ++i) total += partial[i];
            checksum += total;
        }
        cg::sync(grid);
    }
    if (threadIdx.x == 0) {
        result[blockIdx.x] = checksum;
    }
}

int main() {
    constexpr unsigned int kBlocks = 4;
    constexpr unsigned int kPhases = 256;
    unsigned int* partial = nullptr;
    unsigned int* result = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&partial),
                   kBlocks * sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&result),
                   kBlocks * sizeof(unsigned int)) != cudaSuccess) {
        return 1;
    }
    unsigned int phases = kPhases;
    void* args[] = {&partial, &result, &phases};
    if (cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(cooperative_grid_barrier_stress),
            dim3(kBlocks), dim3(32), args, 0, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cooperative launch or synchronization failed\n");
        return 1;
    }
    unsigned int host[kBlocks] = {};
    if (cudaMemcpy(host, result, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
        return 1;
    }
    // Phase p contributes sum_{b<4} (4p + b + 1) = 16p + 10.
    unsigned int expected = 0;
    for (unsigned int p = 0; p < kPhases; ++p) expected += 16u * p + 10u;
    for (unsigned int value : host) {
        if (value != expected) {
            std::fprintf(stderr, "FAIL: grid barrier checksum=%u expected=%u\n",
                         value, expected);
            return 1;
        }
    }
    cudaFree(result);
    cudaFree(partial);
    std::puts("PASS: cooperative grid barrier ordered 256 phases");
    return 0;
}
