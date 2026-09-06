#include "cuda_runtime.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>

// Test cudaMemcpy2D, cudaMemset2D, and cudaMemcpy2DAsync with a pitched 2D matrix.
// Uses a 4x8 matrix with row pitch > row width (extra padding per row).

static constexpr int kRows  = 4;
static constexpr int kCols  = 8;  // elements per row (floats)
static constexpr int kExtra = 2;  // extra padding elements per row
static constexpr size_t kWidth  = kCols * sizeof(float);
static constexpr size_t kSPitch = (kCols + kExtra) * sizeof(float);
static constexpr size_t kDPitch = (kCols + kExtra) * sizeof(float);

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    // Allocate host src with pitch (src has valid values + garbage padding).
    float host_src[kRows][kCols + kExtra];
    float host_dst[kRows][kCols + kExtra];

    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            host_src[r][c] = static_cast<float>(r * kCols + c + 1);
        }
        // Fill padding with garbage.
        for (int c = kCols; c < kCols + kExtra; ++c) {
            host_src[r][c] = -999.0f;
        }
    }
    std::memset(host_dst, 0, sizeof(host_dst));

    // cudaMemcpy2D: copy only the valid kCols columns (not padding).
    cudaError_t err = cudaMemcpy2D(
        host_dst, kDPitch,
        host_src, kSPitch,
        kWidth, kRows,
        cudaMemcpyHostToHost);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy2D returned %d\n", err);
        return 1;
    }

    // Verify: valid columns copied correctly, padding untouched (still 0).
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            float expected = static_cast<float>(r * kCols + c + 1);
            if (host_dst[r][c] != expected) {
                std::fprintf(stderr,
                    "FAIL: dst[%d][%d] = %f, expected %f\n",
                    r, c, host_dst[r][c], expected);
                return 1;
            }
        }
        // Padding in dst should be 0 (not copied from src garbage).
        for (int c = kCols; c < kCols + kExtra; ++c) {
            if (host_dst[r][c] != 0.0f) {
                std::fprintf(stderr,
                    "FAIL: padding dst[%d][%d] = %f, expected 0\n",
                    r, c, host_dst[r][c]);
                return 1;
            }
        }
    }

    // cudaMemset2D: set valid columns of dst to a pattern.
    std::memset(host_dst, 0, sizeof(host_dst));
    err = cudaMemset2D(host_dst, kDPitch, 0x42, kWidth, kRows);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemset2D returned %d\n", err);
        return 1;
    }
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols * (int)sizeof(float); ++c) {
            if (reinterpret_cast<unsigned char*>(host_dst)[r * (int)kDPitch + c] != 0x42) {
                std::fprintf(stderr, "FAIL: cudaMemset2D byte mismatch at row %d col %d\n", r, c);
                return 1;
            }
        }
        // Padding should still be 0.
        for (int c = kCols; c < kCols + kExtra; ++c) {
            if (host_dst[r][c] != 0.0f) {
                std::fprintf(stderr,
                    "FAIL: cudaMemset2D corrupted padding dst[%d][%d]\n", r, c);
                return 1;
            }
        }
    }

    // cudaMemcpy2DAsync on null stream.
    std::memset(host_dst, 0, sizeof(host_dst));
    err = cudaMemcpy2DAsync(
        host_dst, kDPitch,
        host_src, kSPitch,
        kWidth, kRows,
        cudaMemcpyHostToHost, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy2DAsync returned %d\n", err);
        return 1;
    }
    if ((err = cudaDeviceSynchronize()) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize returned %d\n", err);
        return 1;
    }
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kCols; ++c) {
            float expected = static_cast<float>(r * kCols + c + 1);
            if (host_dst[r][c] != expected) {
                std::fprintf(stderr,
                    "FAIL: async dst[%d][%d] = %f, expected %f\n",
                    r, c, host_dst[r][c], expected);
                return 1;
            }
        }
    }

    // CUDA's pageable-memory contract: an asynchronous copy or memset whose
    // host end is pageable is synchronous with respect to the host -- the
    // data is in place when the call returns -- while still being stream
    // ordered, so it runs only after the work queued ahead of it. A host
    // function parked on the stream records that it ran; a call that returns
    // with the flag set and the data in place has both properties. (A
    // pageable source is captured at call time, as CUDA specifies, so the
    // parked function must not be expected to influence the copied bytes.)
    struct Gate {
        std::atomic<bool> ran{false};
    } gate;
    auto slow_host_op = [](void* opaque) {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        static_cast<Gate*>(opaque)->ran.store(true, std::memory_order_release);
    };
    cudaStream_t ordered_stream = nullptr;
    if (cudaStreamCreate(&ordered_stream) != cudaSuccess ||
        cudaLaunchHostFunc(ordered_stream, slow_host_op, &gate) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: failed to create async-ordering gate\n");
        return 1;
    }
    std::memset(host_dst, 0, sizeof(host_dst));
    err = cudaMemcpy2DAsync(host_dst, kDPitch, host_src, kSPitch,
                            kWidth, kRows, cudaMemcpyHostToHost, ordered_stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: ordered cudaMemcpy2DAsync returned %d\n", err);
        return 1;
    }
    if (!gate.ran.load(std::memory_order_acquire) || host_dst[0][0] != host_src[0][0]) {
        std::fprintf(stderr,
                     "FAIL: cudaMemcpy2DAsync into pageable memory must complete, in stream "
                     "order, before returning (ran=%d dst=%f)\n",
                     gate.ran.load() ? 1 : 0, static_cast<double>(host_dst[0][0]));
        return 1;
    }
    if (cudaStreamSynchronize(ordered_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize after ordered copy\n");
        return 1;
    }

    gate.ran.store(false, std::memory_order_release);
    if (cudaLaunchHostFunc(ordered_stream, slow_host_op, &gate) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: failed to re-arm async-ordering gate\n");
        return 1;
    }
    if (cudaMemset2DAsync(host_dst, kDPitch, 0x5a, kWidth, kRows,
                          ordered_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: ordered cudaMemset2DAsync failed\n");
        return 1;
    }
    if (!gate.ran.load(std::memory_order_acquire) ||
        reinterpret_cast<const unsigned char*>(host_dst)[0] != 0x5a) {
        std::fprintf(stderr,
                     "FAIL: cudaMemset2DAsync into pageable memory must complete, in stream "
                     "order, before returning\n");
        return 1;
    }
    if (cudaStreamSynchronize(ordered_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize after ordered memset\n");
        return 1;
    }
    cudaStreamDestroy(ordered_stream);

    std::printf("PASS: cudaMemcpy2D, cudaMemset2D, cudaMemcpy2DAsync all correct\n");
    return 0;
}
