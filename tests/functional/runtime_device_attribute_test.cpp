#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int value = 0;
    if (cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 0) != cudaSuccess || value != 32) {
        std::fprintf(stderr, "FAIL: expected warp size 32\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: multiprocessor count should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: shared memory per block should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrUnifiedAddressing, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: unified addressing should be enabled\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: managed memory should be enabled\n");
        return 1;
    }

    // 0, not 1. Sharing an address space is not the promise this attribute
    // makes: CUDA's is coherent concurrent access, host and kernel reading and
    // writing managed memory at the same time and seeing each other's stores.
    // Metal guarantees the host sees a kernel's writes only once its command
    // buffer completes, and has no CPU-GPU atomic at all. NVIDIA's
    // systemWideAtomics sample branches on this and computed wrong answers when
    // it was 1.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, 0) != cudaSuccess ||
        value != 0) {
        std::fprintf(stderr, "FAIL: concurrent managed access should be reported absent\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMajor, 0) != cudaSuccess || value != 8) {
        std::fprintf(stderr, "FAIL: compute capability major should be 8 (Ampere-equivalent, spec §6.8)\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrComputeCapabilityMinor, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: compute capability minor should be 0 (Ampere-equivalent, spec §6.8)\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxRegistersPerBlock, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: max registers per block should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: clock rate should be positive\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrGpuOverlap, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: GPU overlap should be enabled\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(nullptr, cudaDevAttrWarpSize, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null output pointer should be rejected\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, 9999, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unknown attribute should be rejected\n");
        return 1;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrWarpSize, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: runtime device attribute API behaves correctly\n");
    return 0;
}
