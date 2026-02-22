// Tests MTLHeap auto-threshold: allocations at or above the threshold should use
// the heap path automatically without CUMETAL_MTLHEAP_ALLOC=1.
//
// The test allocates one buffer below the threshold (should use direct newBufferWithLength)
// and one at or above the threshold (should use heap suballoc if auto mode is active).
// Since we cannot observe the internal allocation path directly, we verify:
//   1. Both allocations succeed.
//   2. Data written to each buffer round-trips correctly through cudaMemcpy.
//   3. cudaFree doesn't crash for either path.
//
// Threshold mode is exercised by the test environment:
//   - Default env (no CUMETAL_MTLHEAP_ALLOC): auto mode, threshold 4 MiB.
//   - CUMETAL_MTLHEAP_THRESHOLD_BYTES=65536 overrides the threshold to 64 KiB so
//     the "large" allocation (128 KiB) crosses it on any machine.

#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <vector>

static bool round_trip(std::size_t n_floats, const char* label) {
    const std::size_t bytes = n_floats * sizeof(float);
    void* d = nullptr;
    if (cudaMalloc(&d, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc %s (%zu bytes) failed\n", label, bytes);
        return false;
    }

    std::vector<float> h_src(n_floats), h_dst(n_floats, -1.0f);
    for (std::size_t i = 0; i < n_floats; ++i) h_src[i] = static_cast<float>(i);

    if (cudaMemcpy(d, h_src.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: H2D %s failed\n", label);
        cudaFree(d);
        return false;
    }
    if (cudaMemcpy(h_dst.data(), d, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: D2H %s failed\n", label);
        cudaFree(d);
        return false;
    }
    for (std::size_t i = 0; i < n_floats; ++i) {
        if (h_dst[i] != h_src[i]) {
            std::fprintf(stderr, "FAIL: %s mismatch at %zu: got=%.1f expected=%.1f\n",
                         label, i, static_cast<double>(h_dst[i]),
                         static_cast<double>(h_src[i]));
            cudaFree(d);
            return false;
        }
    }
    cudaFree(d);
    return true;
}

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // Small allocation: 4 KiB (below default 4 MiB threshold, but above 64 KiB override threshold)
    // This exercises the non-heap path in default mode, heap path in override mode.
    // 16384 floats = 64 KiB
    if (!round_trip(16384, "small-64KiB")) return 1;

    // Large allocation: 32768 floats = 128 KiB (above CUMETAL_MTLHEAP_THRESHOLD_BYTES=65536).
    if (!round_trip(32768, "large-128KiB")) return 1;

    // Extra-large: 1M floats = 4 MiB (crosses default threshold even without env override).
    if (!round_trip(1u << 20, "xlarge-4MiB")) return 1;

    std::printf("PASS: heap auto-threshold round-trip correct for all allocation sizes\n");
    return 0;
}
