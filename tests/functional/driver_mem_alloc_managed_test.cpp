#include "cuda.h"

#include <cstdio>
#include <cstdint>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }
    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    constexpr size_t kBytes = 1024;
    CUdeviceptr ptr = 0;
    if (cuMemAllocManaged(&ptr, kBytes, 0) != CUDA_SUCCESS || ptr == 0) {
        std::fprintf(stderr, "FAIL: cuMemAllocManaged failed\n");
        return 1;
    }

    auto* bytes = reinterpret_cast<std::uint8_t*>(static_cast<std::uintptr_t>(ptr));
    bytes[0] = 0x2a;
    bytes[kBytes - 1] = 0x7f;
    if (bytes[0] != 0x2a || bytes[kBytes - 1] != 0x7f) {
        std::fprintf(stderr, "FAIL: managed allocation should be host-accessible\n");
        return 1;
    }

    if (cuMemFree(ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuMemAllocManaged(nullptr, kBytes, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: null output pointer should fail\n");
        return 1;
    }

    if (cuMemAllocManaged(&ptr, 0, 0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: zero-size allocation should fail\n");
        return 1;
    }

    // GLOBAL attachment is what unified memory already gives, so it must be
    // accepted; CUDA's own C++ overload passes it by default.
    if (cuMemAllocManaged(&ptr, kBytes, CU_MEM_ATTACH_GLOBAL) != CUDA_SUCCESS || ptr == 0) {
        std::fprintf(stderr, "FAIL: CU_MEM_ATTACH_GLOBAL should be accepted\n");
        return 1;
    }
    if (cuMemFree(ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree after CU_MEM_ATTACH_GLOBAL failed\n");
        return 1;
    }

    // HOST is already the initial state of Apple shared storage. CuMetal accepts
    // it without migration while still rejecting SINGLE's per-stream state.
    if (cuMemAllocManaged(&ptr, kBytes, CU_MEM_ATTACH_HOST) != CUDA_SUCCESS || ptr == 0) {
        std::fprintf(stderr, "FAIL: CU_MEM_ATTACH_HOST should be accepted on UMA\n");
        return 1;
    }
    if (cuMemFree(ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree after CU_MEM_ATTACH_HOST failed\n");
        return 1;
    }

    if (cuMemAllocManaged(&ptr, kBytes, CU_MEM_ATTACH_SINGLE) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: CU_MEM_ATTACH_SINGLE should be refused\n");
        return 1;
    }

    if (cuMemAllocManaged(&ptr, kBytes, 0x8) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: undefined managed flags should fail\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver managed allocation API behaves correctly\n");
    return 0;
}
