#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 14;
constexpr std::size_t kThreadsPerBlock = 256;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "FAIL: metallib not found at %s\n", metallib_path.c_str());
        return 1;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);

    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>(i) * 0.25f;
        host_b[i] = static_cast<float>(i % 17) * 1.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);

    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* launch_args[] = {&arg_a, &arg_b, &arg_c};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
                        1,
                        1);

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: samples/nativeLaunch produced correct output for %zu elements\n", kElementCount);
    return 0;
}
