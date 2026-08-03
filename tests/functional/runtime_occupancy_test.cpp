#include "cuda_runtime.h"

#include <cstdio>

// Tests rejection behavior for occupancy/function attributes and pointer attributes.

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int numBlocks = -1;
    const void* dummy_func = reinterpret_cast<const void*>(0x1);
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, dummy_func, 256, 0) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: occupancy accepted an invalid function\n");
        return 1;
    }

    int minGridSize = -1;
    int blockSize = -1;
    if (cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, dummy_func, 0, 0) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: potential occupancy accepted an invalid function\n");
        return 1;
    }

    cudaFuncAttributes attr{};
    if (cudaFuncGetAttributes(&attr, dummy_func) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: function attributes accepted an invalid function\n");
        return 1;
    }

    // --- cudaFuncSetCacheConfig (no-op, should succeed) ---
    if (cudaFuncSetCacheConfig(dummy_func, cudaFuncCachePreferL1) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFuncSetCacheConfig failed\n");
        return 1;
    }

    // --- cudaFuncSetSharedMemConfig (no-op, should succeed) ---
    if (cudaFuncSetSharedMemConfig(dummy_func, cudaSharedMemBankSizeEightByte) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFuncSetSharedMemConfig failed\n");
        return 1;
    }

    // --- cudaPointerGetAttributes ---
    // Host pointer: not in allocation table, should be cudaMemoryTypeHost
    cudaPointerAttributes pattr{};
    int local_val = 42;
    if (cudaPointerGetAttributes(&pattr, &local_val) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaPointerGetAttributes (host ptr) failed\n");
        return 1;
    }
    if (pattr.type != cudaMemoryTypeHost && pattr.type != cudaMemoryTypeUnregistered) {
        std::fprintf(stderr,
                     "FAIL: host ptr should be host/unregistered type, got %d\n",
                     static_cast<int>(pattr.type));
        return 1;
    }

    // Device pointer: allocated via cudaMalloc, should be managed/device
    void* dev_ptr = nullptr;
    if (cudaMalloc(&dev_ptr, 64) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    cudaPointerAttributes dattr{};
    if (cudaPointerGetAttributes(&dattr, dev_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaPointerGetAttributes (device ptr) failed\n");
        return 1;
    }
    if (dattr.type != cudaMemoryTypeManaged && dattr.type != cudaMemoryTypeDevice) {
        std::fprintf(stderr,
                     "FAIL: device ptr should be managed/device type, got %d\n",
                     static_cast<int>(dattr.type));
        return 1;
    }

    // --- cudaChooseDevice ---
    int chosen = -1;
    cudaDeviceProp prop{};
    prop.major = 8;
    prop.minor = 0;
    if (cudaChooseDevice(&chosen, &prop) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaChooseDevice failed\n");
        return 1;
    }
    if (chosen != 0) {
        std::fprintf(stderr, "FAIL: cudaChooseDevice returned %d (expected 0)\n", chosen);
        return 1;
    }

    cudaFree(dev_ptr);

    std::printf("PASS: occupancy API, func attrs, pointer attrs (spec §8)\n");
    return 0;
}
