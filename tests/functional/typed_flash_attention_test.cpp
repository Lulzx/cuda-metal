#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: usage: %s <flash-attention.metallib>\n", argv[0]);
        return 77;
    }
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: Apple GPU unavailable\n");
        return 77;
    }

    constexpr int kN = 32;
    constexpr int kD = 16;
    constexpr int kBc = 32;
    constexpr int kBr = 32;
    constexpr int kTc = 1;
    constexpr int kTr = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(kD));
    std::vector<float> q(kN * kD), k(kN * kD), v(kN * kD);
    std::vector<float> output(kN * kD, 0.0f), expected(kN * kD, 0.0f);
    std::vector<float> l(kN, 0.0f), m(kN, -3.402823466e+38f);
    for (int i = 0; i < kN * kD; ++i) {
        q[i] = static_cast<float>((i * 7) % 23 - 11) * 0.0625f;
        k[i] = static_cast<float>((i * 5) % 19 - 9) * 0.078125f;
        v[i] = static_cast<float>((i * 11) % 29 - 14) * 0.03125f;
    }
    std::vector<float> scores(kN * kN);
    for (int row = 0; row < kN; ++row) {
        float maximum = -3.402823466e+38f;
        for (int column = 0; column < kN; ++column) {
            float sum = 0.0f;
            for (int component = 0; component < kD; ++component) {
                sum += q[row * kD + component] * k[column * kD + component];
            }
            scores[row * kN + column] = sum * scale;
            maximum = std::fmax(maximum, scores[row * kN + column]);
        }
        float denominator = 0.0f;
        for (int column = 0; column < kN; ++column) {
            scores[row * kN + column] =
                std::exp(scores[row * kN + column] - maximum);
            denominator += scores[row * kN + column];
        }
        for (int component = 0; component < kD; ++component) {
            float sum = 0.0f;
            for (int column = 0; column < kN; ++column) {
                sum += scores[row * kN + column] * v[column * kD + component];
            }
            expected[row * kD + component] = sum / denominator;
        }
    }

    float *device_q = nullptr, *device_k = nullptr, *device_v = nullptr;
    float *device_l = nullptr, *device_m = nullptr, *device_output = nullptr;
    const std::size_t matrix_bytes = q.size() * sizeof(float);
    const std::size_t row_bytes = l.size() * sizeof(float);
    if (cudaMalloc(reinterpret_cast<void**>(&device_q), matrix_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_k), matrix_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_v), matrix_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), matrix_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_l), row_bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_m), row_bytes) != cudaSuccess ||
        cudaMemcpy(device_q, q.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_k, k.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_v, v.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_output, output.data(), matrix_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_l, l.data(), row_bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_m, m.data(), row_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed FlashAttention allocation or upload failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgs[] = {
        {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, sizeof(int)}, {CUMETAL_ARG_BYTES, sizeof(int)},
        {CUMETAL_ARG_BYTES, sizeof(int)}, {CUMETAL_ARG_BYTES, sizeof(int)},
        {CUMETAL_ARG_BYTES, sizeof(int)}, {CUMETAL_ARG_BYTES, sizeof(int)},
        {CUMETAL_ARG_BYTES, sizeof(float)}, {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0}, {CUMETAL_ARG_BUFFER, 0},
    };
    const std::string metallib = argv[1];
    const cumetalKernel_t kernel{
        .metallib_path = metallib.c_str(),
        .kernel_name = "_Z14forward_kernelPKfS0_S0_iiiiiifPfS1_S1_",
        .arg_count = 13,
        .arg_info = kArgs,
    };
    int n = kN, d = kD, tc = kTc, tr = kTr, bc = kBc, br = kBr;
    float launch_scale = scale;
    void* launch_args[] = {
        &device_q, &device_k, &device_v, &n, &d, &tc, &tr,
        &bc, &br, &launch_scale, &device_l, &device_m, &device_output,
    };
    constexpr unsigned int kSharedBytes = (3 * kBc * kD + kBc * kBr) * sizeof(float);
    if (cudaLaunchKernel(&kernel, dim3(1, 1, 1), dim3(kBc, 1, 1),
                         launch_args, kSharedBytes, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(output.data(), device_output, matrix_bytes,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed FlashAttention launch or readback failed\n");
        return 1;
    }
    cudaFree(device_q);
    cudaFree(device_k);
    cudaFree(device_v);
    cudaFree(device_output);
    cudaFree(device_l);
    cudaFree(device_m);

    for (std::size_t i = 0; i < output.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 3e-5f) {
            std::fprintf(stderr,
                         "FAIL: typed FlashAttention mismatch at %zu got=%g expected=%g\n",
                         i, static_cast<double>(output[i]),
                         static_cast<double>(expected[i]));
            return 1;
        }
    }
    std::puts("PASS: typed direct FlashAttention on Apple GPU");
    return 0;
}
