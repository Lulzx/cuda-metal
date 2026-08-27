#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGetDeviceProperties(device=0) failed\n");
        return 1;
    }

    if (prop.name[0] == '\0') {
        std::fprintf(stderr, "FAIL: device name should not be empty\n");
        return 1;
    }

    if (prop.warpSize != 32) {
        std::fprintf(stderr, "FAIL: warpSize expected 32\n");
        return 1;
    }

    if (prop.totalGlobalMem == 0) {
        std::fprintf(stderr, "FAIL: totalGlobalMem should be non-zero\n");
        return 1;
    }

    if (prop.maxThreadsPerBlock <= 0 || prop.sharedMemPerBlock <= 0) {
        std::fprintf(stderr, "FAIL: block limits should be positive\n");
        return 1;
    }

    if (prop.major != 8 || prop.minor != 0) {
        std::fprintf(stderr, "FAIL: synthetic compute capability expected 8.0\n");
        return 1;
    }

    if (prop.unifiedAddressing != 1) {
        std::fprintf(stderr, "FAIL: unifiedAddressing should be 1 on Apple Silicon\n");
        return 1;
    }

    if (prop.managedMemory != 1) {
        std::fprintf(stderr, "FAIL: managedMemory should be 1 on Apple Silicon\n");
        return 1;
    }

    // 0, not 1. Sharing an address space is not the promise this attribute
    // makes: CUDA's is coherent concurrent access, host and kernel reading and
    // writing managed memory at the same time and seeing each other's stores.
    // Metal guarantees the host sees a kernel's writes only once its command
    // buffer completes, and has no CPU-GPU atomic at all. NVIDIA's
    // systemWideAtomics sample branches on this and computed wrong answers when
    // it was 1.
    if (prop.concurrentManagedAccess != 0) {
        std::fprintf(stderr, "FAIL: concurrentManagedAccess should be 0: Metal has no "
                             "CPU-GPU coherence during a kernel\n");
        return 1;
    }

    if (prop.maxBufferArguments != 31) {
        std::fprintf(stderr, "FAIL: maxBufferArguments should be 31 (Metal limit)\n");
        return 1;
    }

    if (cudaGetDeviceProperties(nullptr, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null properties pointer should fail\n");
        return 1;
    }

    if (cudaGetDeviceProperties(&prop, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: only device 0 should be supported\n");
        return 1;
    }

    std::printf("PASS: runtime device properties API behaves correctly\n");
    return 0;
}
