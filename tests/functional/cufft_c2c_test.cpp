// Functional test: cuFFT C2C round-trip (forward + inverse = N * original).
// Uses a 1D C2C plan with a known input and verifies the inverse result equals
// N * input[i] within floating-point tolerance.

#include "cufftXt.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // 56 includes a factor of seven and is rejected by vDSP's DFT setup. This
    // exercises CuMetal's bounded direct-DFT correctness path used by simpleCUFFT.
    const int N = 56;

    // Allocate host arrays.
    std::vector<cufftComplex> h_in(N), h_mid(N), h_out(N);
    for (int i = 0; i < N; ++i) {
        h_in[i].x = static_cast<float>(i % 7) + 1.0f;
        h_in[i].y = static_cast<float>(i % 5) * 0.5f;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }

    // Allocate device memory (UMA: device == host memory).
    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&d_in), N * sizeof(cufftComplex)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&d_out), N * sizeof(cufftComplex)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMemcpyAsync(d_in, h_in.data(), N * sizeof(cufftComplex),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: queued cudaMemcpy H->D\n");
        return 1;
    }

    cufftHandle plan;
    if (cufftPlan1d(&plan, N, CUFFT_C2C, 1) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftPlan1d\n");
        return 1;
    }

    cufftHandle xt_plan = 0;
    long long xt_n[1] = {N};
    long long xt_embed[1] = {N};
    size_t work_size = 1;
    // An explicit inembed describing the contiguous layout is accepted; a
    // mismatched input/execution datatype pair still is not.
    if (cufftCreate(&xt_plan) != CUFFT_SUCCESS ||
        cufftXtMakePlanMany(xt_plan, 1, xt_n, xt_embed, 1, N, CUDA_C_32F,
                            nullptr, 1, N, CUDA_C_32F, 1, &work_size,
                            CUDA_C_32F) != CUFFT_SUCCESS ||
        cufftXtMakePlanMany(xt_plan, 1, xt_n, nullptr, 1, N, CUDA_R_32F,
                            nullptr, 1, N, CUDA_C_32F, 1, &work_size,
                            CUDA_R_32F) != CUFFT_NOT_SUPPORTED ||
        cufftXtMakePlanMany(xt_plan, 1, xt_n, nullptr, 1, N, CUDA_C_32F,
                            nullptr, 1, N, CUDA_C_32F, 1, &work_size,
                            CUDA_C_32F) != CUFFT_SUCCESS ||
        cufftSetStream(xt_plan, stream) != CUFFT_SUCCESS ||
        work_size != 0) {
        std::fprintf(stderr, "FAIL: cufftXtMakePlanMany compatibility contract\n");
        return 1;
    }

    // Forward transform: d_in → d_out.
    if (cufftExecC2C(xt_plan, d_in, d_out, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftExecC2C FORWARD\n");
        return 1;
    }

    // Inverse transform: d_out → d_in (in-place inverse).
    if (cufftExecC2C(plan, d_out, d_in, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftExecC2C INVERSE\n");
        return 1;
    }

    if (cufftDestroy(plan) != CUFFT_SUCCESS ||
        cufftDestroy(xt_plan) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: cufftDestroy\n");
        return 1;
    }

    // Rank 2 and rank 3 now execute rather than being rejected; the numerical
    // contract for them lives in functional_cufft_nd, which checks the output
    // against the transform's definition. Here it is enough that a rank-2 plan
    // runs and leaves the caller's rank-1 data untouched.
    cufftHandle plan_2d = 0;
    if (cufftPlan2d(&plan_2d, 2, 2, CUFFT_C2C) != CUFFT_SUCCESS ||
        cufftExecC2C(plan_2d, d_in, d_out, CUFFT_FORWARD) != CUFFT_SUCCESS ||
        cufftDestroy(plan_2d) != CUFFT_SUCCESS) {
        std::fprintf(stderr, "FAIL: rank-2 cuFFT plan did not execute\n");
        return 1;
    }
    // That transform reads d_in and writes d_out, so the rank-1 round-trip result
    // still sitting in d_in is intact for the value check below.

    if (cudaMemcpy(h_out.data(), d_in, N * sizeof(cufftComplex), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D->H\n");
        return 1;
    }

    // After IFFT, result should be N * h_in[i] (cuFFT does not normalize).
    const float tol = static_cast<float>(N) * 1e-4f;
    for (int i = 0; i < N; ++i) {
        const float expected_re = static_cast<float>(N) * h_in[i].x;
        const float expected_im = static_cast<float>(N) * h_in[i].y;
        if (std::fabs(h_out[i].x - expected_re) > tol ||
            std::fabs(h_out[i].y - expected_im) > tol) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %d: got (%.4f,%.4f) expected (%.4f,%.4f)\n",
                         i, static_cast<double>(h_out[i].x), static_cast<double>(h_out[i].y),
                         static_cast<double>(expected_re), static_cast<double>(expected_im));
            return 1;
        }
    }

    cudaFree(d_in);
    cudaFree(d_out);
    cudaStreamDestroy(stream);

    std::printf("PASS: cuFFT C2C round-trip (N=%d)\n", N);
    return 0;
}
