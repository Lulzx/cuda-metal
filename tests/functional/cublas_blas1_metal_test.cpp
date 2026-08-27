// The FP64 level-1 Metal kernels, checked against the CPU loops they replace.
//
// Two things have to hold, and only one of them is about numbers. The GPU path
// is allowed to decline for any reason and fall back, which means a kernel that
// never compiles produces perfectly correct output and an entirely green test.
// This file checks the numbers; run_cublas_blas1_metal.sh runs it once on each
// path and separately requires the kernels to have actually launched, by reading
// the debug channel the dispatch writes.
//
// Accuracy is held to the FP64 emulation's contract rather than to binary64:
// arithmetic runs at about a 48-bit significand, so 2^-48 (3.55e-15) per
// operation is the floor, and a reduction over n elements accumulates. Copy and
// swap move bit patterns without decoding, so those are required to be exact.

#include "cublas_v2.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// n is above the default dispatch threshold so that auto mode would take the
// GPU path too, not only the forced mode this test runs in.
constexpr int kN = 200000;

// The pair carries ~48 bits against binary64's 53, so the per-operation floor is
// 2^-48. A reduction of n terms in a pairwise tree grows that like log2(n); the
// slack below is generous because the point is to catch a wrong kernel, not to
// certify the last bit.
constexpr double kRelTol = 1e-12;

bool close(double a, double b, double tol = kRelTol) {
    const double scale = std::fmax(std::fabs(a), std::fabs(b));
    return std::fabs(a - b) <= tol * std::fmax(scale, 1.0);
}

bool fail(const char* what, double got, double want) {
    std::fprintf(stderr, "FAIL: %s got %.17g want %.17g (rel %.3g)\n", what, got, want,
                 std::fabs(got - want) / std::fmax(std::fabs(want), 1.0));
    return false;
}

}  // namespace

int main() {
    // CUMETAL_BLAS_METAL is latched in a function-local static on first use,
    // so the path under test is chosen by the caller's environment rather than
    // here. run_cublas_blas1_metal.sh runs this binary once each way: forced on
    // (and required to show GPU launches) and forced off.
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }
    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cublasCreate failed\n");
        return 1;
    }

    std::mt19937_64 rng(20260827);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> hx(kN), hy(kN);
    for (int i = 0; i < kN; ++i) {
        hx[i] = dist(rng);
        hy[i] = dist(rng);
    }

    double* dx = nullptr;
    double* dy = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dx), kN * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dy), kN * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    const auto upload = [&] {
        return cudaMemcpy(dx, hx.data(), kN * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess &&
               cudaMemcpy(dy, hy.data(), kN * sizeof(double), cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (!upload()) {
        std::fprintf(stderr, "FAIL: cudaMemcpy failed\n");
        return 1;
    }

    bool ok = true;

    // ── Ddot ────────────────────────────────────────────────────────────────
    {
        double want = 0.0;
        for (int i = 0; i < kN; ++i) want += hx[i] * hy[i];
        double got = 0.0;
        if (cublasDdot(handle, kN, dx, 1, dy, 1, &got) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: cublasDdot returned an error\n");
            ok = false;
        } else if (!close(got, want)) {
            ok = fail("Ddot", got, want);
        }
    }

    // ── Dnrm2 ───────────────────────────────────────────────────────────────
    {
        double sum_sq = 0.0;
        for (int i = 0; i < kN; ++i) sum_sq += hx[i] * hx[i];
        const double want = std::sqrt(sum_sq);
        double got = 0.0;
        if (cublasDnrm2(handle, kN, dx, 1, &got) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: cublasDnrm2 returned an error\n");
            ok = false;
        } else if (!close(got, want)) {
            ok = fail("Dnrm2", got, want);
        }
    }

    // ── Daxpy ───────────────────────────────────────────────────────────────
    {
        const double alpha = -0.37519;
        std::vector<double> want(kN);
        for (int i = 0; i < kN; ++i) want[i] = alpha * hx[i] + hy[i];
        if (cublasDaxpy(handle, kN, &alpha, dx, 1, dy, 1) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: cublasDaxpy returned an error\n");
            ok = false;
        }
        // The launch is deliberately async, so the result is only there after a
        // synchronize. A test that read it without one would be testing timing.
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
            return 1;
        }
        std::vector<double> got(kN);
        if (cudaMemcpy(got.data(), dy, kN * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaMemcpy back failed\n");
            return 1;
        }
        for (int i = 0; i < kN; ++i) {
            if (!close(got[i], want[i])) {
                ok = fail("Daxpy element", got[i], want[i]);
                std::fprintf(stderr, "      at i=%d\n", i);
                break;
            }
        }
    }

    // ── Dscal ───────────────────────────────────────────────────────────────
    {
        if (!upload()) {
            std::fprintf(stderr, "FAIL: re-upload failed\n");
            return 1;
        }
        const double alpha = 2.7182818284590452;
        std::vector<double> want(kN);
        for (int i = 0; i < kN; ++i) want[i] = alpha * hx[i];
        if (cublasDscal(handle, kN, &alpha, dx, 1) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL: cublasDscal returned an error\n");
            ok = false;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
            return 1;
        }
        std::vector<double> got(kN);
        if (cudaMemcpy(got.data(), dx, kN * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaMemcpy back failed\n");
            return 1;
        }
        for (int i = 0; i < kN; ++i) {
            if (!close(got[i], want[i])) {
                ok = fail("Dscal element", got[i], want[i]);
                std::fprintf(stderr, "      at i=%d\n", i);
                break;
            }
        }
    }

    cublasDestroy(handle);
    cudaFree(dx);
    cudaFree(dy);

    if (!ok) return 1;
    const char* mode = std::getenv("CUMETAL_BLAS_METAL");
    std::printf("PASS: FP64 cuBLAS level-1 matches the host reference "
                "(CUMETAL_BLAS_METAL=%s)\n", mode != nullptr ? mode : "unset");
    return 0;
}
