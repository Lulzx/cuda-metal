// Where the FP64 level-1 Metal kernels start beating the CPU loop.
//
// Reports completed-call latency, not enqueue latency. The elementwise kernels
// launch asynchronously, so timing them without a synchronize measures how fast
// this process can hand work to the GPU -- a number that looks like a large
// speedup and predicts nothing. Every timing below waits for the work to land.
//
// That makes the elementwise rows pessimistic relative to how they are used: a
// caller that issues several axpys before reading anything gets overlap this
// benchmark deliberately gives up. The reductions have no such gap, since they
// have to synchronize to return a scalar at all.
//
// Not a CTest. It picks a default threshold; it does not assert one.

#include "cublas_v2.h"
#include "cuda_runtime.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Enough repetitions that a single dispatch's jitter does not decide the
// answer, but few enough that the largest sizes still finish.
int reps_for(int n) {
    if (n <= 4096) return 2000;
    if (n <= 65536) return 500;
    if (n <= 1048576) return 100;
    return 30;
}

}  // namespace

int main(int argc, char** argv) {
    const bool force_gpu = argc > 1 && std::string(argv[1]) == "--gpu";
    setenv("CUMETAL_BLAS_METAL", force_gpu ? "1" : "0", 1);

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "cudaInit failed\n");
        return 1;
    }
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    const int sizes[] = {1024, 4096, 16384, 65536, 262144, 1048576, 4194304};
    std::printf("%-10s %12s %12s %12s\n", "n", "axpy us", "dot us", "nrm2 us");

    std::mt19937_64 rng(1);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (const int n : sizes) {
        std::vector<double> hx(n), hy(n);
        for (int i = 0; i < n; ++i) { hx[i] = dist(rng); hy[i] = dist(rng); }

        double* dx = nullptr;
        double* dy = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&dx), n * sizeof(double)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&dy), n * sizeof(double)) != cudaSuccess) {
            std::fprintf(stderr, "cudaMalloc failed at n=%d\n", n);
            return 1;
        }
        cudaMemcpy(dx, hx.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(dy, hy.data(), n * sizeof(double), cudaMemcpyHostToDevice);

        const int reps = reps_for(n);
        const double alpha = 1.0000001;

        // Warm up: the first call on a path pays for the MSL compile, which is
        // cached for the process and would otherwise land entirely on the first
        // size measured.
        cublasDaxpy(handle, n, &alpha, dx, 1, dy, 1);
        double scratch = 0.0;
        cublasDdot(handle, n, dx, 1, dy, 1, &scratch);
        cudaDeviceSynchronize();

        auto t0 = Clock::now();
        for (int r = 0; r < reps; ++r) cublasDaxpy(handle, n, &alpha, dx, 1, dy, 1);
        cudaDeviceSynchronize();
        const double axpy_us = seconds_since(t0) / reps * 1e6;

        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) cublasDdot(handle, n, dx, 1, dy, 1, &scratch);
        const double dot_us = seconds_since(t0) / reps * 1e6;

        t0 = Clock::now();
        for (int r = 0; r < reps; ++r) cublasDnrm2(handle, n, dx, 1, &scratch);
        const double nrm2_us = seconds_since(t0) / reps * 1e6;

        std::printf("%-10d %12.2f %12.2f %12.2f\n", n, axpy_us, dot_us, nrm2_us);
        std::fflush(stdout);

        cudaFree(dx);
        cudaFree(dy);
    }

    cublasDestroy(handle);
    return 0;
}
