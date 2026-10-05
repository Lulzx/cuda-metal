// Functional test: a cuFFT call on the legacy default stream is ordered after
// work already queued on a blocking stream. Kokkos (LAMMPS PPPM) fills the
// charge grid on a stream it created with cudaStreamCreate, then calls
// cufftExec with the plan left on stream 0. CuMetal's transform runs on the
// CPU; synchronizing only stream 0 let it read the grid mid-copy.

#include "cufft.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    constexpr int n = 8;
    constexpr int batch = 1 << 21;   // 16M complex doubles: a long copy
    constexpr std::size_t count = static_cast<std::size_t>(n) * batch;
    const std::size_t bytes = count * sizeof(cufftDoubleComplex);

    cufftDoubleComplex* pinned = nullptr;
    cufftDoubleComplex* device = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMallocHost(reinterpret_cast<void**>(&pinned), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device), bytes) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: setup\n");
        return 1;
    }
    // Every transform's input is a unit impulse, so every output is all ones.
    for (std::size_t i = 0; i < count; ++i) pinned[i] = {i % n == 0 ? 1.0 : 0.0, 0.0};
    cudaMemset(device, 0, bytes);
    cudaDeviceSynchronize();

    cufftHandle plan = 0;
    if (cufftPlan1d(&plan, n, CUFFT_Z2Z, batch) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftPlan1d\n");
        return 1;
    }
    for (int round = 0; round < 3; ++round) {
        cudaMemsetAsync(device, 0, bytes, stream);
        cudaMemcpyAsync(device, pinned, bytes, cudaMemcpyHostToDevice, stream);
        if (cufftExecZ2Z(plan, device, device, CUFFT_FORWARD) != CUFFT_SUCCESS) {
            std::fprintf(stderr, "FAIL: cufftExecZ2Z\n");
            return 1;
        }
        std::vector<cufftDoubleComplex> out(count);
        cudaMemcpy(out.data(), device, bytes, cudaMemcpyDeviceToHost);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < count; ++i)
            if (std::abs(out[i].x - 1.0) > 1e-12 || std::abs(out[i].y) > 1e-12) ++wrong;
        if (wrong != 0) {
            std::fprintf(stderr, "FAIL: round %d: %zu of %zu outputs read the grid before the copy finished\n",
                         round, wrong, count);
            return 1;
        }
    }
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
    cudaFree(device);
    cudaFreeHost(pinned);
    std::printf("PASS: cuFFT on stream 0 waits for a blocking stream's queued copy\n");
    return 0;
}
