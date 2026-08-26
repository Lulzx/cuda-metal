#include "cuda_runtime.h"

#include <cstdio>

// Tests the extended cudaDeviceAttr values added in the last session:
// cudaDevAttrMemoryBusWidth, cudaDevAttrL2CacheSize, cudaDevAttrMaxThreadsPerMultiProcessor,
// cudaDevAttrIntegrated, cudaDevAttrCanMapHostMemory, cudaDevAttrComputeMode,
// cudaDevAttrConcurrentKernels, cudaDevAttrMemoryClockRate,
// cudaDevAttrPageableMemoryAccess, cudaDevAttrPageableMemoryAccessUsesHostPageTables,
// cudaDevAttrPciBusId, cudaDevAttrPciDeviceId, cudaDevAttrPciDomainId,
// cudaDevAttrTccDriver, cudaDevAttrKernelExecTimeout, cudaDevAttrAsyncEngineCount,
// cudaDevAttrSharedMemPerBlockOptin.
// Apple Silicon specific values are checked against their known constants (spec §6.8).

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int value = 0;

    // Memory bus width: 128-bit on Apple Silicon (conservative estimate, spec §6.8).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMemoryBusWidth, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrMemoryBusWidth should be positive, got %d\n", value);
        return 1;
    }

    // L2 cache size: positive value.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrL2CacheSize, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrL2CacheSize should be positive, got %d\n", value);
        return 1;
    }

    // Max threads per multiprocessor.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMaxThreadsPerMultiProcessor, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrMaxThreadsPerMultiProcessor should be positive, got %d\n", value);
        return 1;
    }

    // Integrated: 1 on Apple Silicon (UMA).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrIntegrated, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: cudaDevAttrIntegrated should be 1 on Apple Silicon, got %d\n", value);
        return 1;
    }

    // Can map host memory: 1 on Apple Silicon (UMA).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrCanMapHostMemory, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: cudaDevAttrCanMapHostMemory should be 1 on Apple Silicon, got %d\n", value);
        return 1;
    }

    // Compute mode: 0 = cudaComputeModeDefault.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrComputeMode, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrComputeMode should be 0 (default), got %d\n", value);
        return 1;
    }

    // Concurrent kernels: 1.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentKernels, 0) != cudaSuccess || value != 1) {
        std::fprintf(stderr, "FAIL: cudaDevAttrConcurrentKernels should be 1, got %d\n", value);
        return 1;
    }

    // Memory clock rate: positive.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrMemoryClockRate, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrMemoryClockRate should be positive, got %d\n", value);
        return 1;
    }

    // Arbitrary pageable pointers are not tracked/bound as Metal buffers: 0.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrPageableMemoryAccess should be 0, got %d\n", value);
        return 1;
    }

    // Arbitrary pageable pointers are not tracked/bound as Metal buffers: 0.
    if (cudaDeviceGetAttribute(&value, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrPageableMemoryAccessUsesHostPageTables should be 0, got %d\n", value);
        return 1;
    }

    // PCI Bus / Device / Domain ID: 0 on Apple Silicon (no discrete PCI GPU).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrPciBusId, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrPciBusId should be 0, got %d\n", value);
        return 1;
    }
    if (cudaDeviceGetAttribute(&value, cudaDevAttrPciDeviceId, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrPciDeviceId should be 0, got %d\n", value);
        return 1;
    }
    if (cudaDeviceGetAttribute(&value, cudaDevAttrPciDomainId, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrPciDomainId should be 0, got %d\n", value);
        return 1;
    }

    // TCC driver: 0 (not a Tesla Compute Cluster device).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrTccDriver, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrTccDriver should be 0, got %d\n", value);
        return 1;
    }

    // Kernel exec timeout: 0 (no GPU watchdog on Apple Silicon).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrKernelExecTimeout, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrKernelExecTimeout should be 0, got %d\n", value);
        return 1;
    }

    // Async engine count: 0 (no DMA engines; UMA has no copy engines).
    if (cudaDeviceGetAttribute(&value, cudaDevAttrAsyncEngineCount, 0) != cudaSuccess || value != 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrAsyncEngineCount should be 0, got %d\n", value);
        return 1;
    }

    // Shared mem per block (optin): same as sharedMemPerBlock.
    int base_shared = 0;
    if (cudaDeviceGetAttribute(&base_shared, cudaDevAttrMaxSharedMemoryPerBlock, 0) != cudaSuccess || base_shared <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrMaxSharedMemoryPerBlock should be positive\n");
        return 1;
    }
    if (cudaDeviceGetAttribute(&value, cudaDevAttrSharedMemPerBlockOptin, 0) != cudaSuccess || value <= 0) {
        std::fprintf(stderr, "FAIL: cudaDevAttrSharedMemPerBlockOptin should be positive, got %d\n", value);
        return 1;
    }

    std::printf("PASS: extended device attribute API behaves correctly\n");
    return 0;
}
