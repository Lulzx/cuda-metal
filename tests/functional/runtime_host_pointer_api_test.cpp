#include "cuda_runtime.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    constexpr std::size_t kBytes = 1024;
    void* host_ptr = nullptr;
    if (cudaHostAlloc(&host_ptr, kBytes, cudaHostAllocDefault) != cudaSuccess || host_ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaHostAlloc failed\n");
        return 1;
    }

    void* device_alias = nullptr;
    if (cudaHostGetDevicePointer(&device_alias, host_ptr, 0) != cudaSuccess ||
        device_alias != host_ptr) {
        std::fprintf(stderr, "FAIL: cudaHostGetDevicePointer should return host alias\n");
        return 1;
    }

    unsigned int host_flags = 0xdeadbeefu;
    if (cudaHostGetFlags(&host_flags, host_ptr) != cudaSuccess || host_flags != cudaHostAllocDefault) {
        std::fprintf(stderr, "FAIL: cudaHostGetFlags should report default flags\n");
        return 1;
    }

    void* mapped_host_ptr = nullptr;
    const unsigned int mapped_flags = cudaHostAllocMapped | cudaHostAllocWriteCombined;
    if (cudaHostAlloc(&mapped_host_ptr, kBytes, mapped_flags) != cudaSuccess || mapped_host_ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaHostAlloc(mapped) failed\n");
        return 1;
    }
    if (cudaHostGetFlags(&host_flags, mapped_host_ptr) != cudaSuccess || host_flags != mapped_flags) {
        std::fprintf(stderr, "FAIL: cudaHostGetFlags should report mapped allocation flags\n");
        return 1;
    }

    void* host_offset = static_cast<void*>(static_cast<std::uint8_t*>(host_ptr) + 8);
    if (cudaHostGetDevicePointer(&device_alias, host_offset, 0) != cudaSuccess ||
        device_alias != host_offset) {
        std::fprintf(stderr, "FAIL: host-offset pointer should preserve its offset\n");
        return 1;
    }

    if (cudaHostGetFlags(&host_flags, host_offset) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: host-offset pointer should be rejected for flags query\n");
        return 1;
    }

    void* device_ptr = nullptr;
    if (cudaMalloc(&device_ptr, kBytes) != cudaSuccess || device_ptr == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaHostGetDevicePointer(&device_alias, device_ptr, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: device allocation should be rejected by host mapping API\n");
        return 1;
    }

    if (cudaHostGetFlags(&host_flags, device_ptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: device allocation should be rejected by host flags API\n");
        return 1;
    }

    if (cudaHostGetDevicePointer(nullptr, host_ptr, 0) != cudaErrorInvalidValue ||
        cudaHostGetDevicePointer(&device_alias, nullptr, 0) != cudaErrorInvalidValue ||
        cudaHostGetDevicePointer(&device_alias, host_ptr, 1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: invalid cudaHostGetDevicePointer args should fail\n");
        return 1;
    }

    if (cudaHostGetFlags(nullptr, host_ptr) != cudaErrorInvalidValue ||
        cudaHostGetFlags(&host_flags, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: invalid cudaHostGetFlags args should fail\n");
        return 1;
    }

    if (cudaHostAlloc(&mapped_host_ptr, kBytes, 0x80u) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unsupported cudaHostAlloc flags should fail\n");
        return 1;
    }

    if (cudaFree(device_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    if (cudaFreeHost(host_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeHost failed\n");
        return 1;
    }
    if (cudaFreeHost(mapped_host_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeHost(mapped) failed\n");
        return 1;
    }

    std::printf("PASS: runtime host pointer mapping APIs behave correctly\n");
    return 0;
}
