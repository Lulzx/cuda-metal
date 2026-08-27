#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>

// Verify cudaDeviceProp has all fields specified in spec §6.8 and that
// cudaGetDeviceProperties populates them correctly.

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGetDeviceProperties failed\n");
        return 1;
    }

    // spec §6.8 — basic fields
    if (prop.warpSize != 32) {
        std::fprintf(stderr, "FAIL: warpSize expected 32, got %d\n", prop.warpSize);
        return 1;
    }
    if (prop.totalGlobalMem == 0) {
        std::fprintf(stderr, "FAIL: totalGlobalMem should be non-zero\n");
        return 1;
    }
    if (prop.sharedMemPerBlock == 0) {
        std::fprintf(stderr, "FAIL: sharedMemPerBlock should be non-zero\n");
        return 1;
    }
    if (prop.maxThreadsPerBlock <= 0) {
        std::fprintf(stderr, "FAIL: maxThreadsPerBlock should be positive\n");
        return 1;
    }
    if (prop.multiProcessorCount <= 0) {
        std::fprintf(stderr, "FAIL: multiProcessorCount should be positive\n");
        return 1;
    }
    if (prop.major != 8 || prop.minor != 0) {
        std::fprintf(stderr, "FAIL: compute capability expected 8.0, got %d.%d\n", prop.major, prop.minor);
        return 1;
    }
    if (!prop.unifiedAddressing) {
        std::fprintf(stderr, "FAIL: unifiedAddressing should be 1\n");
        return 1;
    }
    if (!prop.managedMemory) {
        std::fprintf(stderr, "FAIL: managedMemory should be 1\n");
        return 1;
    }
    // 0, not 1. Sharing an address space is not the promise this attribute
    // makes: CUDA's is coherent concurrent access, host and kernel reading and
    // writing managed memory at the same time and seeing each other's stores.
    // Metal guarantees the host sees a kernel's writes only once its command
    // buffer completes, and has no CPU-GPU atomic at all. NVIDIA's
    // systemWideAtomics sample branches on this and computed wrong answers when
    // it was 1.
    if (prop.concurrentManagedAccess) {
        std::fprintf(stderr, "FAIL: concurrentManagedAccess should be 0\n");
        return 1;
    }
    if (std::strlen(prop.name) == 0) {
        std::fprintf(stderr, "FAIL: device name should be non-empty\n");
        return 1;
    }

    // spec §6.8 extended fields (Apple Silicon constants)
    if (prop.clockRate <= 0) {
        std::fprintf(stderr, "FAIL: clockRate should be positive, got %d\n", prop.clockRate);
        return 1;
    }
    if (prop.memoryClockRate <= 0) {
        std::fprintf(stderr, "FAIL: memoryClockRate should be positive, got %d\n", prop.memoryClockRate);
        return 1;
    }
    if (prop.memoryBusWidth <= 0) {
        std::fprintf(stderr, "FAIL: memoryBusWidth should be positive, got %d\n", prop.memoryBusWidth);
        return 1;
    }
    if (prop.totalConstMem == 0) {
        std::fprintf(stderr, "FAIL: totalConstMem should be non-zero\n");
        return 1;
    }
    if (prop.maxThreadsPerMultiProcessor <= 0) {
        std::fprintf(stderr, "FAIL: maxThreadsPerMultiProcessor should be positive, got %d\n", prop.maxThreadsPerMultiProcessor);
        return 1;
    }
    if (prop.l2CacheSize <= 0) {
        std::fprintf(stderr, "FAIL: l2CacheSize should be positive, got %d\n", prop.l2CacheSize);
        return 1;
    }
    if (!prop.canMapHostMemory) {
        std::fprintf(stderr, "FAIL: canMapHostMemory should be 1 on Apple Silicon UMA\n");
        return 1;
    }
    if (!prop.integrated) {
        std::fprintf(stderr, "FAIL: integrated should be 1 on Apple Silicon\n");
        return 1;
    }
    if (!prop.concurrentKernels) {
        std::fprintf(stderr, "FAIL: concurrentKernels should be 1\n");
        return 1;
    }
    if (prop.computeMode != cudaComputeModeDefault) {
        std::fprintf(stderr, "FAIL: computeMode should be cudaComputeModeDefault (0), got %d\n", prop.computeMode);
        return 1;
    }
    if (prop.pageableMemoryAccess) {
        std::fprintf(stderr, "FAIL: pageableMemoryAccess must be 0 until malloc pointers are Metal-bound\n");
        return 1;
    }
    if (prop.pageableMemoryAccessUsesHostPageTables) {
        std::fprintf(stderr, "FAIL: pageableMemoryAccessUsesHostPageTables must be 0\n");
        return 1;
    }
    if (prop.persistingL2CacheMaxSize != 0 || prop.accessPolicyMaxWindowSize != 0) {
        std::fprintf(stderr, "FAIL: unsupported persisting-L2 capabilities must be 0\n");
        return 1;
    }

    // Verify cudaComputeMode enum values compile correctly.
    static_assert(cudaComputeModeDefault         == 0, "cudaComputeModeDefault should be 0");
    static_assert(cudaComputeModeExclusive       == 1, "cudaComputeModeExclusive should be 1");
    static_assert(cudaComputeModeProhibited      == 2, "cudaComputeModeProhibited should be 2");
    static_assert(cudaComputeModeExclusiveProcess == 3, "cudaComputeModeExclusiveProcess should be 3");

    std::printf("PASS: device property struct fully populated (spec §6.8); device: %s\n", prop.name);
    return 0;
}
