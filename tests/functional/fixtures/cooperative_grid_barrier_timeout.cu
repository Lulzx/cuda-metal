// A grid sync that some threadgroup never reaches must end in
// cudaErrorLaunchTimeout from the grid-sync watchdog instead of hanging the
// host in waitUntilCompleted (#175). Block 0 skips the barrier. That is
// undefined behaviour in CUDA; here it stands in for a threadgroup that Metal
// never made co-resident with the waiters.
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <chrono>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void unbalanced_grid_sync(unsigned int* out) {
    cg::grid_group grid = cg::this_grid();
    if (blockIdx.x != 0) {
        cg::sync(grid);
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = blockIdx.x + 1u;
    }
}

int main() {
    constexpr unsigned int kBlocks = 4;
    unsigned int* out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&out), kBlocks * sizeof(unsigned int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }
    void* args[] = {&out};
    const auto start = std::chrono::steady_clock::now();
    const cudaError_t launch = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(unbalanced_grid_sync), dim3(kBlocks), dim3(32), args, 0,
        nullptr);
    const cudaError_t sync = launch == cudaSuccess ? cudaDeviceSynchronize() : launch;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("unbalanced grid sync: launch=%d sync=%d after %.2f s\n", static_cast<int>(launch),
                static_cast<int>(sync), seconds);
    if (launch != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cooperative launch was rejected (%d)\n",
                     static_cast<int>(launch));
        return 1;
    }
    if (sync != cudaErrorLaunchTimeout) {
        std::fprintf(stderr, "FAIL: expected cudaErrorLaunchTimeout (702), got %d\n",
                     static_cast<int>(sync));
        return 1;
    }
    // Like CUDA's launch timeout, the error is sticky.
    if (cudaDeviceSynchronize() != cudaErrorLaunchTimeout) {
        std::fprintf(stderr, "FAIL: launch timeout was not sticky\n");
        return 1;
    }
    std::puts("PASS: unbalanced cooperative grid sync reported cudaErrorLaunchTimeout");
    return 0;
}
