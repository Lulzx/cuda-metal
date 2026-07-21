#include "cuda.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

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

    constexpr std::size_t kBytes = 1024;
    void* host_ptr = nullptr;
    if (cuMemHostAlloc(&host_ptr, kBytes, 0) != CUDA_SUCCESS || host_ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cuMemHostAlloc failed\n");
        return 1;
    }

    CUdeviceptr device_alias = 0;
    if (cuMemHostGetDevicePointer(&device_alias, host_ptr, 0) != CUDA_SUCCESS ||
        device_alias != static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host_ptr))) {
        std::fprintf(stderr, "FAIL: cuMemHostGetDevicePointer should return host alias\n");
        return 1;
    }

    unsigned int host_flags = 0xdeadbeefu;
    if (cuMemHostGetFlags(&host_flags, host_ptr) != CUDA_SUCCESS || host_flags != 0u) {
        std::fprintf(stderr, "FAIL: cuMemHostGetFlags should report default flags\n");
        return 1;
    }

    void* mapped_host_ptr = nullptr;
    const unsigned int mapped_flags = CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED;
    if (cuMemHostAlloc(&mapped_host_ptr, kBytes, mapped_flags) != CUDA_SUCCESS ||
        mapped_host_ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cuMemHostAlloc(mapped) failed\n");
        return 1;
    }
    if (cuMemHostGetFlags(&host_flags, mapped_host_ptr) != CUDA_SUCCESS || host_flags != mapped_flags) {
        std::fprintf(stderr, "FAIL: cuMemHostGetFlags should report mapped allocation flags\n");
        return 1;
    }

    void* host_offset = static_cast<void*>(static_cast<std::uint8_t*>(host_ptr) + 8);
    if (cuMemHostGetDevicePointer(&device_alias, host_offset, 0) != CUDA_SUCCESS ||
        device_alias != static_cast<CUdeviceptr>(reinterpret_cast<std::uintptr_t>(host_offset))) {
        std::fprintf(stderr, "FAIL: host-offset pointer should preserve its offset\n");
        return 1;
    }

    if (cuMemHostGetFlags(&host_flags, host_offset) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: host-offset pointer should be rejected for flags query\n");
        return 1;
    }

    CUdeviceptr device_ptr = 0;
    if (cuMemAlloc(&device_ptr, kBytes) != CUDA_SUCCESS || device_ptr == 0) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    if (cuMemHostGetDevicePointer(&device_alias,
                                  reinterpret_cast<void*>(static_cast<std::uintptr_t>(device_ptr)),
                                  0) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: device allocation should be rejected by host mapping API\n");
        return 1;
    }

    if (cuMemHostGetFlags(&host_flags,
                          reinterpret_cast<void*>(static_cast<std::uintptr_t>(device_ptr))) !=
        CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: device allocation should be rejected by host flags API\n");
        return 1;
    }

    if (cuMemHostGetDevicePointer(nullptr, host_ptr, 0) != CUDA_ERROR_INVALID_VALUE ||
        cuMemHostGetDevicePointer(&device_alias, nullptr, 0) != CUDA_ERROR_INVALID_VALUE ||
        cuMemHostGetDevicePointer(&device_alias, host_ptr, 1) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: invalid cuMemHostGetDevicePointer args should fail\n");
        return 1;
    }

    if (cuMemHostGetFlags(nullptr, host_ptr) != CUDA_ERROR_INVALID_VALUE ||
        cuMemHostGetFlags(&host_flags, nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: invalid cuMemHostGetFlags args should fail\n");
        return 1;
    }

    if (cuMemHostAlloc(&host_ptr, kBytes, 0x80u) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: unsupported cuMemHostAlloc flags should fail\n");
        return 1;
    }

    if (cuMemFree(device_ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuMemFreeHost(host_ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFreeHost failed\n");
        return 1;
    }
    if (cuMemFreeHost(mapped_host_ptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFreeHost(mapped) failed\n");
        return 1;
    }
    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver host pointer mapping APIs behave correctly\n");
    return 0;
}
