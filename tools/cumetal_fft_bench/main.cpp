// cuFFT backend benchmark: the Metal kernels against the Accelerate CPU path.
//
// Both implementations live behind the same cufftExec* entry points and are
// selected by CUMETAL_FFT_METAL, which is read once per process. This tool sets
// it before the first cuFFT call and measures one backend; run it twice to
// compare.
//
// Two numbers per grid, because they answer different questions:
//
//   latency    one R2C, then synchronize. What a caller pays if it needs the
//              spectrum on the host immediately.
//   pipeline   R2C and C2R enqueued back to back, synchronized once. This is
//              what PME actually does -- spread, forward FFT, solve, inverse
//              FFT, gather, all on one stream with no host round trip in the
//              middle -- so it is the number that decides whether moving the
//              transform to the GPU helped.
//
// The CPU path is synchronous, so for it the two are nearly the same and the
// synchronize is free. For the Metal path they differ by exactly the dispatch
// pipelining the GPU can overlap, which is the effect being measured.
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Grid {
    const char* label;
    int n[3];
};

// The two PME meshes the GROMACS demo actually runs, plus power-of-two and
// non-power-of-two grids on either side of them so the Stockham and Bluestein
// paths can be told apart. 56 = 8*7 and 40 = 8*5 both carry a factor vDSP
// cannot handle either, so those rows compare two Bluestein implementations,
// not Bluestein against a native transform.
const Grid kGrids[] = {
    {"32x32x32   pow2", {32, 32, 32}},
    {"40x32x32   villin PME", {40, 32, 32}},
    {"56x56x56   rnase PME", {56, 56, 56}},
    {"64x64x64   pow2", {64, 64, 64}},
    {"96x96x96", {96, 96, 96}},
    {"128x128x128 pow2", {128, 128, 128}},
};

double median(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

bool run_grid(const Grid& grid, cudaStream_t stream, double* latency_us,
              double* pipeline_us) {
    const int n0 = grid.n[0], n1 = grid.n[1], n2 = grid.n[2];
    const int nc = n2 / 2 + 1;
    const std::size_t real_elems = static_cast<std::size_t>(n0) * n1 * n2;
    const std::size_t complex_elems = static_cast<std::size_t>(n0) * n1 * nc;

    cufftReal* d_real = nullptr;
    cufftComplex* d_complex = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_real), real_elems * sizeof(cufftReal)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_complex),
                   complex_elems * sizeof(cufftComplex)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for %s\n", grid.label);
        return false;
    }

    std::vector<float> host(real_elems);
    for (std::size_t i = 0; i < host.size(); ++i) {
        host[i] = std::sin(0.37f * static_cast<float>(i));
    }
    cudaMemcpy(d_real, host.data(), real_elems * sizeof(float), cudaMemcpyHostToDevice);

    int dims[3] = {n0, n1, n2};
    cufftHandle forward = 0, inverse = 0;
    if (cufftPlan3d(&forward, n0, n1, n2, CUFFT_R2C) != CUFFT_SUCCESS ||
        cufftPlan3d(&inverse, n0, n1, n2, CUFFT_C2R) != CUFFT_SUCCESS ||
        cufftSetStream(forward, stream) != CUFFT_SUCCESS ||
        cufftSetStream(inverse, stream) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: plan for %s\n", grid.label);
        return false;
    }
    (void)dims;

    // Warm-up. The first call on the Metal path compiles the MSL, builds the
    // Bluestein filters for every non-power-of-two axis and grows the scratch
    // arena; timing that would measure setup, not the transform.
    for (int i = 0; i < 5; ++i) {
        cufftExecR2C(forward, d_real, d_complex);
        cufftExecC2R(inverse, d_complex, d_real);
    }
    cudaStreamSynchronize(stream);

    const int iterations = real_elems > (1u << 20) ? 20 : 60;

    std::vector<double> latency(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cufftExecR2C(forward, d_real, d_complex);
        cudaStreamSynchronize(stream);
        const auto end = std::chrono::steady_clock::now();
        latency[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    std::vector<double> pipeline(iterations);
    for (int i = 0; i < iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        cufftExecR2C(forward, d_real, d_complex);
        cufftExecC2R(inverse, d_complex, d_real);
        cudaStreamSynchronize(stream);
        const auto end = std::chrono::steady_clock::now();
        pipeline[i] = std::chrono::duration<double, std::micro>(end - start).count();
    }

    *latency_us = median(latency);
    *pipeline_us = median(pipeline);

    cufftDestroy(forward);
    cufftDestroy(inverse);
    cudaFree(d_real);
    cudaFree(d_complex);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* mode = "1";
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cpu") == 0) mode = "0";
        else if (std::strcmp(argv[i], "--metal") == 0) mode = "1";
        else {
            std::fprintf(stderr, "usage: %s [--metal|--cpu]\n", argv[0]);
            return 2;
        }
    }
    // Set before the first cuFFT call: the policy is read once per process.
    setenv("CUMETAL_FFT_METAL", mode, 1);

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate\n");
        return 1;
    }

    std::printf("backend=%s\n", std::strcmp(mode, "0") == 0 ? "accelerate" : "metal");
    std::printf("%-24s %12s %12s\n", "grid", "latency_us", "pipeline_us");
    for (const Grid& grid : kGrids) {
        double latency = 0.0, pipeline = 0.0;
        if (!run_grid(grid, stream, &latency, &pipeline)) return 1;
        std::printf("%-24s %12.1f %12.1f\n", grid.label, latency, pipeline);
        std::fflush(stdout);
    }
    cudaStreamDestroy(stream);
    return 0;
}
