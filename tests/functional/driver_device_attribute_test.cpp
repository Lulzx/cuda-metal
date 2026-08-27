#include "cuda.h"

#include <cstdio>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    int value = 0;
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) != CUDA_SUCCESS || value != 32) {
        std::fprintf(stderr, "FAIL: expected warp size 32\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) != CUDA_SUCCESS ||
        value <= 0) {
        std::fprintf(stderr, "FAIL: multiprocessor count should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, 0) != CUDA_SUCCESS ||
        value <= 0) {
        std::fprintf(stderr, "FAIL: shared memory per block should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, 0) != CUDA_SUCCESS ||
        value != 1) {
        std::fprintf(stderr, "FAIL: unified addressing should be enabled\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY, 0) != CUDA_SUCCESS ||
        value != 1) {
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
    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, 0) != CUDA_SUCCESS ||
        value != 0) {
        std::fprintf(stderr, "FAIL: concurrent managed access should be reported absent\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0) != CUDA_SUCCESS ||
        value != 8) {
        std::fprintf(stderr, "FAIL: compute capability major should be 8 (spec §6.8)\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0) != CUDA_SUCCESS ||
        value != 0) {
        std::fprintf(stderr, "FAIL: compute capability minor should be 0 (spec §6.8)\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, 0) != CUDA_SUCCESS ||
        value <= 0) {
        std::fprintf(stderr, "FAIL: max registers per block should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, 0) != CUDA_SUCCESS || value <= 0) {
        std::fprintf(stderr, "FAIL: clock rate should be positive\n");
        return 1;
    }

    if (cuDeviceGetAttribute(nullptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null output pointer should be rejected\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, static_cast<CUdevice_attribute>(9999), 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: unknown attribute should be rejected\n");
        return 1;
    }

    if (cuDeviceGetAttribute(&value, CU_DEVICE_ATTRIBUTE_WARP_SIZE, 1) != CUDA_ERROR_INVALID_DEVICE) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: driver device attribute API behaves correctly\n");
    return 0;
}
