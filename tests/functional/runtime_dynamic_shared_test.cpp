#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

// Tests dynamic (runtime-sized) shared memory via the sharedMem parameter of
// cudaLaunchKernel (spec §6.5 step 6).  The Metal backend maps this to
// setThreadgroupMemoryLength:atIndex:0 so the threadgroup(0) pointer in the
// Metal shader gets the correct allocation.

namespace {

constexpr std::uint32_t kBlockSize = 128;
constexpr std::uint32_t kNumBlocks = 4;
constexpr std::uint32_t kN = kBlockSize * kNumBlocks;

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

    // Fill input: block k contains values k+1 each for all blockSize elements.
    std::vector<float> host_in(kN);
    for (std::uint32_t i = 0; i < kN; ++i) {
        host_in[i] = static_cast<float>(i / kBlockSize + 1);
    }

    std::vector<float> host_out(kNumBlocks, 0.0f);

    void* dev_in = nullptr;
    void* dev_out = nullptr;

    if (cudaMalloc(&dev_in, kN * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev_out, kNumBlocks * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_in, host_in.data(), kN * sizeof(float), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy input failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };

    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "dynamic_shared_reduce",
        .arg_count = 2,
        .arg_info = kArgInfo,
    };

    void* arg_in = dev_in;
    void* arg_out = dev_out;
    void* launch_args[] = {&arg_in, &arg_out};

    const dim3 block_dim(kBlockSize, 1, 1);
    const dim3 grid_dim(kNumBlocks, 1, 1);
    // Pass dynamic shared memory: kBlockSize floats per threadgroup.
    const std::size_t shared_mem = kBlockSize * sizeof(float);

    // Shared-memory limits. CUDA refuses a launch whose shared memory exceeds
    // what the block may use with cudaErrorInvalidValue and runs nothing; the
    // error is not sticky. On Metal the bound is maxThreadgroupMemoryLength,
    // reported as sharedMemPerBlockOptin. This kernel declares no static
    // threadgroup memory, so all of it is available as dynamic memory.
    int optin = 0;
    int per_block = 0;
    if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0) !=
            cudaSuccess ||
        cudaDeviceGetAttribute(&per_block, cudaDevAttrMaxSharedMemoryPerBlock, 0) !=
            cudaSuccess ||
        optin <= 0 || optin != per_block) {
        std::fprintf(stderr, "FAIL: shared memory limits optin=%d per_block=%d\n", optin,
                     per_block);
        return 1;
    }
    const std::size_t max_dynamic = static_cast<std::size_t>(optin);

    const float sentinel = -12345.0f;
    const std::vector<float> sentinels(kNumBlocks, sentinel);
    const std::size_t oversized[] = {
        max_dynamic + 4,  // one word over the device limit
        64 * 1024,        // over CUDA's 48 KB default as well
    };
    for (const std::size_t bytes : oversized) {
        if (cudaMemcpy(dev_out, sentinels.data(), kNumBlocks * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: sentinel copy failed\n");
            return 1;
        }
        if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, bytes, nullptr) !=
            cudaErrorInvalidValue) {
            std::fprintf(stderr, "FAIL: launch with %zu shared bytes was not refused\n", bytes);
            return 1;
        }
        if (cudaGetLastError() != cudaErrorInvalidValue || cudaGetLastError() != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: oversized shared launch error was sticky or lost\n");
            return 1;
        }
        std::vector<float> after(kNumBlocks, 0.0f);
        if (cudaMemcpy(after.data(), dev_out, kNumBlocks * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: sentinel readback failed\n");
            return 1;
        }
        for (const float value : after) {
            if (value != sentinel) {
                std::fprintf(stderr, "FAIL: refused launch (%zu bytes) wrote output\n", bytes);
                return 1;
            }
        }
    }

    // Exactly the advertised maximum is accepted.
    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, max_dynamic, nullptr) !=
            cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch at the dynamic shared limit failed\n");
        return 1;
    }

    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, launch_args, shared_mem, nullptr) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out, kNumBlocks * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy output failed\n");
        return 1;
    }

    // Block k should sum kBlockSize values of (k+1): expected = kBlockSize * (k+1).
    for (std::uint32_t k = 0; k < kNumBlocks; ++k) {
        const float expected = static_cast<float>(kBlockSize) * static_cast<float>(k + 1);
        if (std::fabs(host_out[k] - expected) > 0.5f) {
            std::fprintf(stderr,
                         "FAIL: block %u output=%f expected=%f\n",
                         k,
                         static_cast<double>(host_out[k]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    cudaFree(dev_in);
    cudaFree(dev_out);

    std::printf(
        "PASS: dynamic shared memory reduction (%u blocks × %u threads, sharedMem=%zu bytes)\n",
        kNumBlocks,
        kBlockSize,
        shared_mem);
    return 0;
}
