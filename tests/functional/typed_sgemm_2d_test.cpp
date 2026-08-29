#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: usage: %s <sgemm-2d.metallib>\n", argv[0]);
        return 77;
    }
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: Apple GPU unavailable\n");
        return 77;
    }

    constexpr int kM = 64;
    constexpr int kN = 64;
    constexpr int kK = 16;
    constexpr float kAlpha = 1.25f;
    constexpr float kBeta = -0.5f;
    std::vector<float> a(kM * kK), b(kK * kN), c(kM * kN), expected(kM * kN);
    for (int i = 0; i < kM * kK; ++i) a[i] = static_cast<float>((i * 7) % 17 - 8) * 0.125f;
    for (int i = 0; i < kK * kN; ++i) b[i] = static_cast<float>((i * 11) % 19 - 9) * 0.0625f;
    for (int i = 0; i < kM * kN; ++i) c[i] = static_cast<float>((i * 5) % 13 - 6) * 0.25f;
    for (int row = 0; row < kM; ++row) {
        for (int column = 0; column < kN; ++column) {
            float sum = 0.0f;
            for (int k = 0; k < kK; ++k) {
                sum += a[row * kK + k] * b[k * kN + column];
            }
            expected[row * kN + column] =
                kAlpha * sum + kBeta * c[row * kN + column];
        }
    }

    float *device_a = nullptr, *device_b = nullptr, *device_c = nullptr;
    const std::size_t a_bytes = a.size() * sizeof(float);
    const std::size_t b_bytes = b.size() * sizeof(float);
    const std::size_t c_bytes = c.size() * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&device_a), a_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_b), b_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_c), c_bytes) != cudaSuccess ||
        cudaMemcpy(device_a, a.data(), a_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_b, b.data(), b_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_c, c.data(), c_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed SGEMM allocation or upload failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgs[] = {
        {CUMETAL_ARG_BYTES, sizeof(int)}, {CUMETAL_ARG_BYTES, sizeof(int)},
        {CUMETAL_ARG_BYTES, sizeof(int)}, {CUMETAL_ARG_BYTES, sizeof(float)},
        {CUMETAL_ARG_BUFFER, 0},          {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, sizeof(float)}, {CUMETAL_ARG_BUFFER, 0},
    };
    const std::string metallib = argv[1];
    const cumetalKernel_t kernel{
        .metallib_path = metallib.c_str(),
        .kernel_name = "_Z18sgemm2DBlocktilingILi64ELi64ELi8ELi8ELi8EEviiifPKfS1_fPf",
        .arg_count = 8,
        .arg_info = kArgs,
    };
    int m = kM, n = kN, k = kK;
    float alpha = kAlpha, beta = kBeta;
    void* launch_args[] = {
        &m, &n, &k, &alpha, &device_a, &device_b, &beta, &device_c,
    };
    if (cudaLaunchKernel(&kernel, dim3(1, 1, 1), dim3(64, 1, 1),
                         launch_args, 0, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(c.data(), device_c, c_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed SGEMM launch or readback failed\n");
        return 1;
    }
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    for (std::size_t i = 0; i < c.size(); ++i) {
        if (std::fabs(c[i] - expected[i]) > 1e-3f) {
            std::fprintf(stderr,
                         "FAIL: typed SGEMM mismatch at %zu got=%g expected=%g\n",
                         i, static_cast<double>(c[i]), static_cast<double>(expected[i]));
            return 1;
        }
    }
    std::puts("PASS: typed direct 2D block-tiled SGEMM on Apple GPU");
    return 0;
}
