#include "cuda.h"

#include <cstdio>

// Tests driver API additions: occupancy, func attrs, stream priority, cooperative launch,
// memset 16/32, device capability, peer access.

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }

    CUcontext ctx = nullptr;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    // --- cuDeviceComputeCapability ---
    int major = 0;
    int minor = 0;
    if (cuDeviceComputeCapability(&major, &minor, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceComputeCapability failed\n");
        return 1;
    }
    if (major <= 0) {
        std::fprintf(stderr, "FAIL: cuDeviceComputeCapability major=%d (expected > 0)\n", major);
        return 1;
    }

    // --- cuDeviceCanAccessPeer ---
    int can_access = -1;
    if (cuDeviceCanAccessPeer(&can_access, dev, dev) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceCanAccessPeer failed\n");
        return 1;
    }
    // Peer access to self should be 0 (no second GPU on Apple Silicon)
    if (can_access != 0) {
        std::fprintf(stderr, "FAIL: cuDeviceCanAccessPeer self should be 0, got %d\n", can_access);
        return 1;
    }

    // --- cuStreamCreateWithPriority ---
    CUstream prio_stream = nullptr;
    if (cuStreamCreateWithPriority(&prio_stream, CU_STREAM_DEFAULT, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreateWithPriority failed\n");
        return 1;
    }
    if (prio_stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreateWithPriority returned null\n");
        return 1;
    }
    if (cuStreamSynchronize(prio_stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamSynchronize on priority stream failed\n");
        return 1;
    }
    if (cuStreamDestroy(prio_stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy on priority stream failed\n");
        return 1;
    }

    // Invalid function handles must be rejected instead of receiving invented
    // occupancy and attribute values.
    CUfunction dummy_func = reinterpret_cast<CUfunction>(0x1);
    int numBlocks = -1;
    if (cuOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocks, dummy_func, 256, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: occupancy accepted an invalid function\n");
        return 1;
    }

    int minGridSize = -1;
    int blockSize = -1;
    if (cuOccupancyMaxPotentialBlockSize(
            &minGridSize, &blockSize, dummy_func, 0, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: potential occupancy accepted an invalid function\n");
        return 1;
    }

    int attr_val = -1;
    if (cuFuncGetAttribute(&attr_val, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           dummy_func) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: function attributes accepted an invalid function\n");
        return 1;
    }

    if (cuFuncSetCacheConfig(dummy_func, CU_FUNC_CACHE_PREFER_L1) !=
        CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuFuncSetCacheConfig accepted invalid function\n");
        return 1;
    }

    // --- cuMemsetD16 ---
    CUdeviceptr dev16 = 0;
    if (cuMemAlloc(&dev16, 8 * sizeof(unsigned short)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc for D16 failed\n");
        return 1;
    }
    if (cuMemsetD16(dev16, 0xBEEF, 8) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemsetD16 failed\n");
        return 1;
    }
    // Verify
    unsigned short host16[8] = {};
    if (cuMemcpyDtoH(host16, dev16, 8 * sizeof(unsigned short)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH for D16 verify failed\n");
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (host16[i] != 0xBEEF) {
            std::fprintf(stderr,
                         "FAIL: cuMemsetD16 host16[%d]=%04x (expected 0xBEEF)\n",
                         i,
                         static_cast<unsigned>(host16[i]));
            return 1;
        }
    }

    // --- cuMemsetD32 ---
    CUdeviceptr dev32 = 0;
    if (cuMemAlloc(&dev32, 8 * sizeof(unsigned int)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc for D32 failed\n");
        return 1;
    }
    if (cuMemsetD32(dev32, 0xDEADBEEF, 8) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemsetD32 failed\n");
        return 1;
    }
    // Verify
    unsigned int host32[8] = {};
    if (cuMemcpyDtoH(host32, dev32, 8 * sizeof(unsigned int)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH for D32 verify failed\n");
        return 1;
    }
    for (int i = 0; i < 8; ++i) {
        if (host32[i] != 0xDEADBEEF) {
            std::fprintf(stderr,
                         "FAIL: cuMemsetD32 host32[%d]=%08x (expected 0xDEADBEEF)\n",
                         i,
                         host32[i]);
            return 1;
        }
    }

    cuMemFree(dev16);
    cuMemFree(dev32);
    cuCtxDestroy(ctx);

    std::printf(
        "PASS: driver extended API — occupancy, func attrs, stream priority, memset16/32, "
        "device capability\n");
    return 0;
}
