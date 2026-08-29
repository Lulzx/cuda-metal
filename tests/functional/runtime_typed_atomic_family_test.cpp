#include "cuda_runtime.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

namespace {

bool launch(const std::string& metallib_path, const char* kernel_name,
            void* argument, dim3 grid, dim3 block) {
    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = kernel_name,
        .arg_count = 1,
        .arg_info = kArgInfo,
    };
    void* arguments[] = {&argument};
    return cudaLaunchKernel(&kernel, grid, block, arguments, 0, nullptr) == cudaSuccess;
}

int run_device_atomics(const std::string& metallib_path) {
    constexpr int kBlocks = 64;
    constexpr int kThreads = 256;
    constexpr int kTotal = kBlocks * kThreads;
    std::array<int, 10> host{};
    host[3] = kTotal;
    host[8] = -1;

    int* device = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device), sizeof(host)) != cudaSuccess ||
        cudaMemcpy(device, host.data(), sizeof(host), cudaMemcpyHostToDevice) != cudaSuccess ||
        !launch(metallib_path, "_Z11all_atomicsPi", device,
                dim3(kBlocks, 1, 1), dim3(kThreads, 1, 1)) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host.data(), device, sizeof(host), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed device atomic launch or transfer failed\n");
        return 1;
    }
    cudaFree(device);

    const std::array<int, 10> expected = {
        kTotal * 10, -kTotal * 10, kTotal - 1, 0, -1,
        kTotal, -kTotal, 1, -2, 0,
    };
    if (host != expected) {
        for (std::size_t i = 0; i < host.size(); ++i) {
            if (host[i] != expected[i]) {
                std::fprintf(stderr,
                             "FAIL: typed device atomic slot %zu got %d expected %d\n",
                             i, host[i], expected[i]);
            }
        }
        return 1;
    }
    std::printf("PASS: typed 32-bit device atomic family under contention\n");
    return 0;
}

int run_system_atomics(const std::string& metallib_path) {
    int* slots = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void**>(&slots), 10 * sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMallocManaged failed\n");
        return 1;
    }
    for (int i = 0; i < 10; ++i) slots[i] = 0;
    slots[2] = -1;
    slots[3] = 999;
    slots[7] = 0xff;
    if (!launch(metallib_path, "_Z20system_atomic_kernelPi", slots,
                dim3(1, 1, 1), dim3(32, 1, 1))) {
        std::fprintf(stderr, "FAIL: typed system atomic launch failed\n");
        return 1;
    }
    __sync_fetch_and_add(&slots[0], 7);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed system atomic synchronization failed\n");
        return 1;
    }

    const std::array<int, 10> expected = {103, 0, 31, 0, 14, 106, 77, 0x3c, 0x12, 0x55};
    bool ok = slots[1] >= 0 && slots[1] < 32;
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (i == 1) continue;
        if (slots[i] != expected[i]) {
            std::fprintf(stderr,
                         "FAIL: typed system atomic slot %zu got %d expected %d\n",
                         i, slots[i], expected[i]);
            ok = false;
        }
    }
    cudaFree(slots);
    if (!ok) return 1;
    std::printf("PASS: typed system atomics interoperate with host atomic access\n");
    return 0;
}

int run_fence(const std::string& metallib_path) {
    constexpr int kBlocks = 64;
    constexpr int kThreads = 256;
    constexpr int kTotal = kBlocks * kThreads;
    int* values = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&values),
                   (kTotal + 1) * sizeof(int)) != cudaSuccess ||
        cudaMemset(values, 0, (kTotal + 1) * sizeof(int)) != cudaSuccess ||
        !launch(metallib_path, "_Z18threadfence_kernelPi", values,
                dim3(kBlocks, 1, 1), dim3(kThreads, 1, 1)) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed fence launch failed\n");
        return 1;
    }
    std::array<int, kTotal + 1> host{};
    if (cudaMemcpy(host.data(), values, sizeof(host),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: typed fence readback failed\n");
        return 1;
    }
    cudaFree(values);
    if (host[kTotal] != kTotal) {
        std::fprintf(stderr, "FAIL: typed fence counter got %d expected %d\n",
                     host[kTotal], kTotal);
        return 1;
    }
    for (int i = 0; i < kTotal; ++i) {
        if (host[i] != i) {
            std::fprintf(stderr,
                         "FAIL: typed fence payload %d got %d expected %d\n",
                         i, host[i], i);
            return 1;
        }
    }
    std::printf("PASS: typed device and threadgroup fences preserve payload visibility\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <metallib> <device|system|fence>\n", argv[0]);
        return 64;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", argv[1]);
        return 77;
    }
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }
    const std::string mode = argv[2];
    if (mode == "device") return run_device_atomics(argv[1]);
    if (mode == "system") return run_system_atomics(argv[1]);
    if (mode == "fence") return run_fence(argv[1]);
    std::fprintf(stderr, "invalid mode: %s\n", argv[2]);
    return 64;
}
