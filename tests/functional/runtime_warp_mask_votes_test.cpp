#include "cuda_runtime.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::uint32_t kThreads = 32;
constexpr std::uint32_t kWordsPerThread = 6;

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
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }

    std::uint32_t* device_output = nullptr;
    constexpr std::size_t kBytes = kThreads * kWordsPerThread * sizeof(std::uint32_t);
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), kBytes) != cudaSuccess ||
        cudaMemset(device_output, 0, kBytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: output allocation\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {{CUMETAL_ARG_BUFFER, 0}};
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "warp_mask_votes",
        .arg_count = 1,
        .arg_info = kArgInfo,
    };
    void* output_arg = device_output;
    void* args[] = {&output_arg};
    if (cudaLaunchKernel(&kernel, dim3(1), dim3(kThreads), args, 0, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: warp_mask_votes launch\n");
        return 1;
    }

    std::vector<std::uint32_t> output(kThreads * kWordsPerThread);
    if (cudaMemcpy(output.data(), device_output, kBytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: output copy\n");
        return 1;
    }

    for (std::uint32_t lane = 0; lane < kThreads; ++lane) {
        const bool lower = lane < 16u;
        const std::array<std::uint32_t, kWordsPerThread> expected{
            lower ? 0x00005555u : 0x55550000u,
            1u,
            1u,
            0xffffffffu,
            lower ? 0u : 16u,
            lower ? (100u + ((lane + 1u) & 15u))
                  : (200u + 16u + ((lane - 15u) & 15u)),
        };
        for (std::uint32_t word = 0; word < kWordsPerThread; ++word) {
            const std::uint32_t actual = output[lane * kWordsPerThread + word];
            if (actual != expected[word]) {
                std::fprintf(stderr,
                             "FAIL: lane %u word %u = 0x%08x, expected 0x%08x\n",
                             lane,
                             word,
                             actual,
                             expected[word]);
                return 1;
            }
        }
    }

    // A malformed source-ABI sidecar must fail the launch loudly rather than
    // silently allocating zero bytes and corrupting shared-memory kernels.
    const std::string sidecar_path = metallib_path + ".cumetal-abi";
    std::ifstream sidecar_input(sidecar_path);
    std::ostringstream saved_sidecar;
    saved_sidecar << sidecar_input.rdbuf();
    if (!sidecar_input && saved_sidecar.str().empty()) {
        std::fprintf(stderr, "FAIL: source ABI sidecar missing\n");
        return 1;
    }
    {
        std::ofstream malformed(sidecar_path, std::ios::trunc);
        malformed << "CUMETAL_ABI_V1\nkernel warp_mask_votes\nshared invalid\narg buffer 8\n";
        if (!malformed) {
            std::fprintf(stderr, "FAIL: unable to write malformed sidecar fixture\n");
            return 1;
        }
    }
    const cudaError_t malformed_status =
        cudaLaunchKernel(&kernel, dim3(1), dim3(kThreads), args, 0, nullptr);
    {
        std::ofstream restored(sidecar_path, std::ios::trunc);
        restored << saved_sidecar.str();
        if (!restored) {
            std::fprintf(stderr, "FAIL: unable to restore source ABI sidecar\n");
            return 1;
        }
    }
    if (malformed_status != cudaErrorInvalidValue) {
        std::fprintf(stderr,
                     "FAIL: malformed ABI sidecar returned %d, expected cudaErrorInvalidValue\n",
                     static_cast<int>(malformed_status));
        return 1;
    }

    if (cudaFree(device_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree\n");
        return 1;
    }
    std::printf("PASS: source partial-mask vote, activemask, shuffle, masked barrier, and static shared allocation\n");
    return 0;
}
