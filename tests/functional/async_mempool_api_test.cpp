#include "cuda_runtime.h"

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <thread>

static_assert(sizeof(cudaMemPoolProps) == 88);
static_assert(offsetof(cudaMemPoolProps, location) == 8);
static_assert(offsetof(cudaMemPoolProps, win32SecurityAttributes) == 16);
static_assert(offsetof(cudaMemPoolProps, reserved) == 24);

namespace {
struct Gate {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
};
void gate_callback(void* raw) {
    auto* gate = static_cast<Gate*>(raw);
    gate->entered.store(true, std::memory_order_release);
    while (!gate->release.load(std::memory_order_acquire)) std::this_thread::yield();
}
}  // namespace

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
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;

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

static bool test_free_follows_stream() {
    cudaStream_t stream = nullptr;
    void* dev = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMallocAsync(&dev, 64, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: async free ordering setup failed\n");
        return false;
    }
    Gate gate;
    if (cudaLaunchHostFunc(stream, gate_callback, &gate) != cudaSuccess) return false;
    while (!gate.entered.load(std::memory_order_acquire)) std::this_thread::yield();
    if (cudaFreeAsync(dev, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: queued cudaFreeAsync failed\n");
        return false;
    }
    cudaPointerAttributes attributes{};
    if (cudaPointerGetAttributes(&attributes, dev) != cudaSuccess ||
        attributes.type != cudaMemoryTypeManaged) {
        std::fprintf(stderr, "FAIL: cudaFreeAsync released storage before its stream reached it\n");
        return false;
    }
    gate.release.store(true, std::memory_order_release);
    if (cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaPointerGetAttributes(&attributes, dev) != cudaSuccess ||
        attributes.type == cudaMemoryTypeManaged) {
        std::fprintf(stderr, "FAIL: cudaFreeAsync did not release storage at stream completion\n");
        return false;
    }
    return cudaStreamDestroy(stream) == cudaSuccess;
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
    if (!test_free_follows_stream()) return 1;
    if (!test_default_mempool()) return 1;
    if (!test_malloc_from_pool()) return 1;
    if (!test_null_args()) return 1;

    std::printf("PASS: Async memory pool API tests\n");
    return 0;
}
