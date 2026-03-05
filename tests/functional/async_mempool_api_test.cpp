#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>

static bool test_malloc_free_async() {
    float* dev = nullptr;
    cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&dev), 256, nullptr);
    if (err != cudaSuccess || dev == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMallocAsync returned %d\n", err);
        return false;
    }

    // Write and read back (UMA — synchronous alias)
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    std::memcpy(dev, src, sizeof(src));

    float dst[4] = {};
    std::memcpy(dst, dev, sizeof(dst));
    for (int i = 0; i < 4; ++i) {
        if (dst[i] != src[i]) {
            std::fprintf(stderr, "FAIL: async alloc data mismatch at %d\n", i);
            return false;
        }
    }

    err = cudaFreeAsync(dev, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeAsync returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_mempool_create_destroy() {
    cudaMemPool_t pool = nullptr;
    cudaMemPoolProps props = {};
    props.allocType = 1; // cudaMemAllocationTypePinned
    props.location_type = 1; // cudaMemLocationTypeDevice
    props.location_id = 0;

    cudaError_t err = cudaMemPoolCreate(&pool, &props);
    if (err != cudaSuccess || pool == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMemPoolCreate returned %d\n", err);
        return false;
    }

    err = cudaMemPoolDestroy(pool);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemPoolDestroy returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_default_mempool() {
    cudaMemPool_t pool = nullptr;
    cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, 0);
    if (err != cudaSuccess || pool == nullptr) {
        std::fprintf(stderr, "FAIL: cudaDeviceGetDefaultMemPool returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_malloc_from_pool() {
    cudaMemPool_t pool = nullptr;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    float* dev = nullptr;
    cudaError_t err = cudaMallocFromPoolAsync(reinterpret_cast<void**>(&dev), 128, pool, nullptr);
    if (err != cudaSuccess || dev == nullptr) {
        std::fprintf(stderr, "FAIL: cudaMallocFromPoolAsync returned %d\n", err);
        return false;
    }

    err = cudaFreeAsync(dev, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFreeAsync from pool returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_null_args() {
    if (cudaMallocAsync(nullptr, 64, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaMallocAsync(null) should fail\n");
        return false;
    }
    if (cudaMemPoolCreate(nullptr, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaMemPoolCreate(null) should fail\n");
        return false;
    }
    return true;
}

int main() {
    if (!test_malloc_free_async()) return 1;
    if (!test_mempool_create_destroy()) return 1;
    if (!test_default_mempool()) return 1;
    if (!test_malloc_from_pool()) return 1;
    if (!test_null_args()) return 1;

    std::printf("PASS: Async memory pool API tests\n");
    return 0;
}
