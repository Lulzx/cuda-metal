// Functional test: batched in-place 1-D Z2Z, the plan shape LAMMPS' Kokkos
// PPPM builds (cufftPlanMany with embed = n, stride 1, distance = length).
// The result is checked against a direct long double DFT, then inverted back.
// Run it with CUMETAL_USE_METAL_DEVICE_ADDRESSES=1 too: there a device pointer
// is the buffer's GPU address, and the CPU transform must not dereference it.

#include "cufft.h"
#include "cuda_runtime.h"

#include <cmath>
#include <complex>
#include <cstdio>
#include <numbers>
#include <vector>

int main() {
    constexpr int n = 12;      // a factor of 3, like PPPM's 12-point grids
    constexpr int length = 16; // distance between transforms (padded rows)
    constexpr int batch = 5;
    constexpr int total = length * batch;

    std::vector<cufftDoubleComplex> host(total);
    for (int i = 0; i < total; ++i) {
        host[i].x = std::sin(0.37 * i) + 0.25 * (i % 3);
        host[i].y = std::cos(0.11 * i * i) - 0.5;
    }
    cufftDoubleComplex* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), sizeof(host[0]) * total) != cudaSuccess ||
        cudaMemcpy(device, host.data(), sizeof(host[0]) * total, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocate\n");
        return 1;
    }
    int size = n, embed = n;
    cufftHandle plan = 0;
    if (cufftPlanMany(&plan, 1, &size, &embed, 1, length, &embed, 1, length, CUFFT_Z2Z, batch) !=
        CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftPlanMany\n");
        return 1;
    }
    if (cufftExecZ2Z(plan, device, device, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: forward cufftExecZ2Z\n");
        return 1;
    }
    std::vector<cufftDoubleComplex> forward(total);
    cudaMemcpy(forward.data(), device, sizeof(host[0]) * total, cudaMemcpyDeviceToHost);

    double worst = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int k = 0; k < n; ++k) {
            std::complex<long double> sum = 0;
            for (int j = 0; j < n; ++j) {
                const long double angle = -2.0L * std::numbers::pi_v<long double> * j * k / n;
                const auto& x = host[b * length + j];
                sum += std::complex<long double>(x.x, x.y) *
                       std::complex<long double>(std::cos(angle), std::sin(angle));
            }
            const auto& y = forward[b * length + k];
            worst = std::max(worst, static_cast<double>(std::abs(
                                        sum - std::complex<long double>(y.x, y.y))));
        }
        // Padding between transforms is outside the plan and must survive.
        for (int k = n; k < length; ++k) {
            const auto& y = forward[b * length + k];
            const auto& x = host[b * length + k];
            if (y.x != x.x || y.y != x.y) {
                std::fprintf(stderr, "FAIL: padding overwritten at batch %d index %d\n", b, k);
                return 1;
            }
        }
    }
    if (!(worst < 1e-12)) {
        std::fprintf(stderr, "FAIL: forward error %.3g\n", worst);
        return 1;
    }

    if (cufftExecZ2Z(plan, device, device, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: inverse cufftExecZ2Z\n");
        return 1;
    }
    std::vector<cufftDoubleComplex> back(total);
    cudaMemcpy(back.data(), device, sizeof(host[0]) * total, cudaMemcpyDeviceToHost);
    double round_trip = 0.0;
    for (int b = 0; b < batch; ++b)
        for (int j = 0; j < n; ++j) {
            const auto& x = host[b * length + j];
            const auto& y = back[b * length + j];
            round_trip = std::max({round_trip, std::abs(y.x / n - x.x), std::abs(y.y / n - x.y)});
        }
    if (!(round_trip < 1e-13)) {
        std::fprintf(stderr, "FAIL: round trip error %.3g\n", round_trip);
        return 1;
    }
    cufftDestroy(plan);
    cudaFree(device);
    std::printf("PASS: batched in-place Z2Z (n=%d, distance=%d, batch=%d): forward %.2g, round trip %.2g\n",
                n, length, batch, worst, round_trip);
    return 0;
}
