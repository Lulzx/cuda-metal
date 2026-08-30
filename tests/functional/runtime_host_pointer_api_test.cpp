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
    auto* typed_host_ptr = static_cast<float*>(host_ptr);
    float* typed_device_alias = nullptr;
    if (cudaHostGetDevicePointer(&typed_device_alias, typed_host_ptr, 0) != cudaSuccess ||
        typed_device_alias != typed_host_ptr) {
        std::fprintf(stderr, "FAIL: typed cudaHostGetDevicePointer overload\n");
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

    void* managed_ptr = nullptr;
    void* host_attached_managed_ptr = nullptr;
    cudaMemLocation host_location{cudaMemLocationTypeHost, 0};
    cudaMemLocation invalid_location{cudaMemLocationTypeInvalid, 0};
    if (cudaMallocManaged(&managed_ptr, kBytes) != cudaSuccess ||
        cudaMallocManaged(&host_attached_managed_ptr, kBytes, cudaMemAttachHost) !=
            cudaSuccess ||
        cudaMemPrefetchAsync(managed_ptr, kBytes, host_location, 0) != cudaSuccess ||
        cudaStreamAttachMemAsync(nullptr, managed_ptr, 0, cudaMemAttachGlobal) != cudaSuccess ||
        cudaStreamAttachMemAsync(nullptr, managed_ptr, 0, cudaMemAttachHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: UMA prefetch/attach compatibility APIs\n");
        return 1;
    }
    int range_value = -1;
    if (::cudaMemPrefetchAsync(managed_ptr, kBytes, 0, nullptr) != cudaSuccess ||
        cudaMemAdvise(managed_ptr, kBytes, cudaMemAdviseSetReadMostly, 0) != cudaSuccess ||
        cudaMemRangeGetAttribute(&range_value, sizeof(range_value),
                                 cudaMemRangeAttributeReadMostly,
                                 managed_ptr, kBytes) != cudaSuccess ||
        range_value != 1) {
        std::fprintf(stderr, "FAIL: validated UMA advisory compatibility APIs\n");
        return 1;
    }
    char* managed_bytes = static_cast<char*>(managed_ptr);
    cudaStream_t invalid_stream = reinterpret_cast<cudaStream_t>(0x12345);
    const cudaError_t invalid_results[] = {
        ::cudaMemPrefetchAsync(nullptr, kBytes, 0, nullptr),
        ::cudaMemPrefetchAsync(managed_ptr, 0, 0, nullptr),
        ::cudaMemPrefetchAsync(managed_bytes + kBytes - 1, 2, 0, nullptr),
        ::cudaMemPrefetchAsync(managed_ptr, kBytes, 1, nullptr),
        ::cudaMemPrefetchAsync(managed_ptr, kBytes, 0, invalid_stream),
        cudaMemAdvise(managed_ptr, kBytes, static_cast<cudaMemoryAdvise>(99), 0),
        cudaMemAdvise(managed_bytes + kBytes - 1, 2, cudaMemAdviseSetReadMostly, 0),
        cudaMemRangeGetAttribute(&range_value, sizeof(range_value),
                                 static_cast<cudaMemRangeAttribute>(99),
                                 managed_ptr, kBytes),
        cudaMemRangeGetAttribute(&range_value, 1, cudaMemRangeAttributeReadMostly,
                                 managed_ptr, kBytes),
        cudaStreamAttachMemAsync(nullptr, managed_ptr, kBytes + 1,
                                 cudaMemAttachGlobal),
    };
    for (size_t i = 0; i < sizeof(invalid_results) / sizeof(invalid_results[0]); ++i) {
        if (invalid_results[i] == cudaErrorInvalidValue) continue;
        std::fprintf(stderr, "FAIL: invalid UMA advisory case %zu returned %d\n",
                     i, invalid_results[i]);
        return 1;
    }
    void* invalid_managed_ptr = nullptr;
    if (cudaMallocManaged(&invalid_managed_ptr, kBytes, cudaMemAttachSingle) !=
            cudaErrorInvalidValue ||
        cudaMemPrefetchAsync(managed_ptr, kBytes, invalid_location, 0) !=
            cudaErrorInvalidValue ||
        cudaMemPrefetchAsync(managed_ptr, kBytes, host_location, 1) !=
            cudaErrorInvalidValue ||
        cudaStreamAttachMemAsync(nullptr, managed_ptr, 0, 0) != cudaErrorInvalidValue ||
        cudaStreamAttachMemAsync(nullptr, nullptr, 0, cudaMemAttachGlobal) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: invalid UMA prefetch/attach arguments should fail\n");
        return 1;
    }
    if (cudaFree(managed_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree(managed) failed\n");
        return 1;
    }
    if (cudaFree(host_attached_managed_ptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree(host-attached managed) failed\n");
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
