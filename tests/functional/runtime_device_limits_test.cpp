#include "cuda_runtime.h"

#include <cstdio>

// Tests cudaDeviceSetLimit / cudaDeviceGetLimit and cudaStreamCreateWithPriority (spec §6.3).

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // --- cudaDeviceGetLimit ---
    size_t stack_size = 0;
    if (cudaDeviceGetLimit(&stack_size, cudaLimitStackSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitStackSize) failed\n");
        return 1;
    }
    if (stack_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitStackSize returned 0\n");
        return 1;
    }

    size_t printf_size = 0;
    if (cudaDeviceGetLimit(&printf_size, cudaLimitPrintfFifoSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitPrintfFifoSize) failed\n");
        return 1;
    }
    if (printf_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitPrintfFifoSize returned 0\n");
        return 1;
    }

    size_t heap_size = 0;
    if (cudaDeviceGetLimit(&heap_size, cudaLimitMallocHeapSize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetLimit(cudaLimitMallocHeapSize) failed\n");
        return 1;
    }
    if (heap_size == 0) {
        std::fprintf(stderr, "FAIL: cudaLimitMallocHeapSize returned 0\n");
        return 1;
    }

    // --- cudaDeviceSetLimit ---
    if (cudaDeviceSetLimit(cudaLimitStackSize, 2048) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSetLimit(cudaLimitStackSize) failed\n");
        return 1;
    }
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16 * 1024 * 1024) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSetLimit(cudaLimitMallocHeapSize) failed\n");
        return 1;
    }
    size_t configured_heap_size = 0;
    if (cudaDeviceGetLimit(&configured_heap_size, cudaLimitMallocHeapSize) != cudaSuccess ||
        configured_heap_size != 16 * 1024 * 1024) {
        std::fprintf(stderr, "FAIL: cudaLimitMallocHeapSize did not persist\n");
        return 1;
    }
    if (cudaDeviceSetLimit(cudaLimitMallocHeapSize, 0) != cudaErrorInvalidValue ||
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, 31) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: undersized device heaps should be rejected\n");
        return 1;
    }

    size_t persisting_l2_size = 1;
    if (cudaDeviceGetLimit(&persisting_l2_size, cudaLimitPersistingL2CacheSize) !=
            cudaSuccess ||
        persisting_l2_size != 0) {
        std::fprintf(stderr, "FAIL: default persisting-L2 hint must be 0\n");
        return 1;
    }
    cudaDeviceProp cache_prop{};
    if (cudaGetDeviceProperties(&cache_prop, 0) != cudaSuccess ||
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 1) != cudaSuccess ||
        cudaDeviceGetLimit(&persisting_l2_size, cudaLimitPersistingL2CacheSize) !=
            cudaSuccess ||
        persisting_l2_size != 1 ||
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                           static_cast<size_t>(cache_prop.persistingL2CacheMaxSize) + 1) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: persisting-L2 hint limit did not validate/round-trip\n");
        return 1;
    }
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: clearing the unsupported persisting-L2 limit failed\n");
        return 1;
    }

    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    if (cudaStreamSetAttribute(nullptr, cudaStreamAttributeAccessPolicyWindow,
                               &attr) != cudaSuccess ||
        cudaStreamGetAttribute(nullptr, cudaStreamAttributeAccessPolicyWindow,
                               &attr) != cudaSuccess ||
        attr.accessPolicyWindow.hitProp != cudaAccessPropertyPersisting ||
        cudaCtxResetPersistingL2Cache() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: persisting-L2 hints should round-trip successfully\n");
        return 1;
    }
    if (cudaStreamSetAttribute(nullptr, cudaStreamAttributeAccessPolicyWindow,
                               nullptr) != cudaErrorInvalidValue ||
        cudaStreamGetAttribute(nullptr, cudaStreamAttributeAccessPolicyWindow,
                               nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: null stream-attribute values should be rejected\n");
        return 1;
    }

    // --- cudaStreamCreateWithPriority ---
    cudaStream_t stream = nullptr;
    if (cudaStreamCreateWithPriority(&stream, cudaStreamDefault, 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithPriority failed\n");
        return 1;
    }
    if (stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithPriority returned null stream\n");
        return 1;
    }

    // Verify the stream works
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize on priority stream failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy on priority stream failed\n");
        return 1;
    }

    std::printf("PASS: device limits and stream priority APIs\n");
    return 0;
}
