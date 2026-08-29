#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

// Verifies cooperative launch validation and a basic launch.

namespace {

constexpr std::size_t kElementCount = 256;
constexpr std::size_t kThreadsPerBlock = 256;

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<float> host_a(kElementCount, 1.5f);
    std::vector<float> host_b(kElementCount, 2.5f);
    std::vector<float> host_c(kElementCount, 0.0f);

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);

    if (cudaMalloc(&dev_a, bytes) != cudaSuccess ||
        cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice);

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 1},
        {CUMETAL_ARG_BUFFER, 2},
        {CUMETAL_ARG_BYTES, 3},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 4,
        .arg_info = kArgInfo,
    };

    auto n = static_cast<int>(kElementCount);
    void* args[] = {&dev_a, &dev_b, &dev_c, &n};

    const dim3 block(kThreadsPerBlock, 1, 1);
    const dim3 grid((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, 1);

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess ||
        cudaLaunchCooperativeKernel(
            &kernel, dim3(static_cast<unsigned int>(prop.multiProcessorCount + 1), 1, 1),
            block, args, 0, nullptr) != cudaErrorCooperativeLaunchTooLarge) {
        std::fprintf(stderr, "FAIL: oversized cooperative launch was accepted\n");
        return 1;
    }

    cudaError_t err = cudaLaunchCooperativeKernel(&kernel, grid, block, args, 0, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchCooperativeKernel returned %s\n",
                     cudaGetErrorName(err));
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < kElementCount; ++i) {
        if (host_c[i] != 4.0f) {
            std::fprintf(stderr, "FAIL: result mismatch at [%zu]: got %.4f, expected 4.0\n",
                         i, static_cast<double>(host_c[i]));
            return 1;
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    std::printf("PASS: cooperative launch accepts resident grids and rejects oversized grids\n");
    return 0;
}
