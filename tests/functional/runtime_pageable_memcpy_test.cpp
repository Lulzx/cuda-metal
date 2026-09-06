// CUDA's pageable-memory contract for asynchronous copies (cudaMemcpyAsync
// reference): a copy whose host end is pageable is synchronous with respect
// to the host. Host-to-device returns once the source has been staged;
// device-to-host and host-to-host return only once the destination holds the
// data. Pinned memory (cudaHostAlloc) is the only host memory that may be
// copied truly asynchronously.
//
// Programs rely on this constantly -- cudaMemcpyAsync(&value, dev, 4, D2H, s)
// into a stack variable, then reading `value` -- and a runtime that defers the
// write corrupts whatever the caller reuses that memory for. Each check below
// first parks a slow host function on the stream, so a copy that merely got
// queued cannot have run by the time the call returns: the assertion is
// deterministic, not a race the runtime happens to win.
#include "cuda_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {

constexpr std::size_t kBytes = 64 * 1024;
constexpr int kDelayMs = 200;

bool expect(bool condition, const char* what) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", what);
    }
    return condition;
}

void sleep_host_fn(void*) {
    std::this_thread::sleep_for(std::chrono::milliseconds(kDelayMs));
}

bool all_bytes_equal(const std::uint8_t* bytes, std::size_t count, std::uint8_t value) {
    for (std::size_t i = 0; i < count; ++i) {
        if (bytes[i] != value) return false;
    }
    return true;
}

}  // namespace

int main() {
    if (cudaFree(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "SKIP: CUDA runtime unavailable\n");
        return 77;
    }

    void* dev = nullptr;
    void* dev2 = nullptr;
    if (!expect(cudaMalloc(&dev, kBytes) == cudaSuccess && cudaMalloc(&dev2, kBytes) == cudaSuccess,
                "cudaMalloc")) {
        return 1;
    }
    if (!expect(cudaMemset(dev, 0x5A, kBytes) == cudaSuccess, "cudaMemset")) return 1;

    cudaStream_t stream = nullptr;
    if (!expect(cudaStreamCreate(&stream) == cudaSuccess, "cudaStreamCreate")) return 1;

    // Device-to-host into pageable memory holds the data when the call returns,
    // even though the stream was busy: the copy is stream-ordered AND
    // host-synchronous.
    {
        std::vector<std::uint8_t> host(kBytes, 0);
        if (!expect(cudaLaunchHostFunc(stream, sleep_host_fn, nullptr) == cudaSuccess,
                    "cudaLaunchHostFunc")) {
            return 1;
        }
        if (!expect(cudaMemcpyAsync(host.data(), dev, kBytes, cudaMemcpyDeviceToHost, stream) ==
                        cudaSuccess,
                    "cudaMemcpyAsync D2H pageable")) {
            return 1;
        }
        if (!expect(all_bytes_equal(host.data(), kBytes, 0x5A),
                    "D2H copy into pageable memory completed before cudaMemcpyAsync returned")) {
            return 1;
        }
    }

    // Host-to-device from pageable memory: the source may be overwritten the
    // moment the call returns and the device still receives the old bytes.
    {
        std::vector<std::uint8_t> host(kBytes, 0xC3);
        if (!expect(cudaLaunchHostFunc(stream, sleep_host_fn, nullptr) == cudaSuccess,
                    "cudaLaunchHostFunc")) {
            return 1;
        }
        if (!expect(cudaMemcpyAsync(dev2, host.data(), kBytes, cudaMemcpyHostToDevice, stream) ==
                        cudaSuccess,
                    "cudaMemcpyAsync H2D pageable")) {
            return 1;
        }
        std::memset(host.data(), 0x00, kBytes);  // clobber the source immediately
        if (!expect(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize")) {
            return 1;
        }
        std::vector<std::uint8_t> check(kBytes, 0);
        if (!expect(cudaMemcpy(check.data(), dev2, kBytes, cudaMemcpyDeviceToHost) == cudaSuccess,
                    "cudaMemcpy D2H")) {
            return 1;
        }
        if (!expect(all_bytes_equal(check.data(), kBytes, 0xC3),
                    "H2D copy staged the pageable source before cudaMemcpyAsync returned")) {
            return 1;
        }
    }

    // The same contract through the 2D entry point.
    {
        constexpr std::size_t kWidth = 256;
        constexpr std::size_t kHeight = 64;
        constexpr std::size_t kPitch = 320;
        std::vector<std::uint8_t> host(kPitch * kHeight, 0);
        if (!expect(cudaLaunchHostFunc(stream, sleep_host_fn, nullptr) == cudaSuccess,
                    "cudaLaunchHostFunc")) {
            return 1;
        }
        if (!expect(cudaMemcpy2DAsync(host.data(), kPitch, dev, kWidth, kWidth, kHeight,
                                      cudaMemcpyDeviceToHost, stream) == cudaSuccess,
                    "cudaMemcpy2DAsync D2H pageable")) {
            return 1;
        }
        bool ok = true;
        for (std::size_t row = 0; row < kHeight && ok; ++row) {
            ok = all_bytes_equal(host.data() + row * kPitch, kWidth, 0x5A);
        }
        if (!expect(ok, "2D D2H copy into pageable memory completed before the call returned")) {
            return 1;
        }
    }

    // Host-to-host between two pageable buffers: also synchronous.
    {
        std::vector<std::uint8_t> a(kBytes, 0x11);
        std::vector<std::uint8_t> b(kBytes, 0);
        if (!expect(cudaLaunchHostFunc(stream, sleep_host_fn, nullptr) == cudaSuccess,
                    "cudaLaunchHostFunc")) {
            return 1;
        }
        if (!expect(cudaMemcpyAsync(b.data(), a.data(), kBytes, cudaMemcpyHostToHost, stream) ==
                        cudaSuccess,
                    "cudaMemcpyAsync H2H pageable")) {
            return 1;
        }
        if (!expect(all_bytes_equal(b.data(), kBytes, 0x11),
                    "H2H copy between pageable buffers completed before the call returned")) {
            return 1;
        }
    }

    // Pinned memory keeps the asynchronous fast path: the copy is only
    // guaranteed after the stream drains, and it must be correct then. This
    // is the case a runtime that simply synchronised everything would also
    // pass, so it is a correctness check, not a timing one.
    {
        void* pinned = nullptr;
        if (!expect(cudaHostAlloc(&pinned, kBytes, cudaHostAllocDefault) == cudaSuccess,
                    "cudaHostAlloc")) {
            return 1;
        }
        std::memset(pinned, 0, kBytes);
        if (!expect(cudaMemcpyAsync(pinned, dev, kBytes, cudaMemcpyDeviceToHost, stream) ==
                        cudaSuccess,
                    "cudaMemcpyAsync D2H pinned")) {
            return 1;
        }
        if (!expect(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize")) {
            return 1;
        }
        if (!expect(all_bytes_equal(static_cast<const std::uint8_t*>(pinned), kBytes, 0x5A),
                    "D2H copy into pinned memory is complete after stream synchronisation")) {
            return 1;
        }
        if (!expect(cudaFreeHost(pinned) == cudaSuccess, "cudaFreeHost")) return 1;
    }

    // The scenario that corrupted the heap: copy into a temporary, free it
    // immediately, then keep allocating. Guard Malloc turns a late write into
    // a fault; without it the corruption surfaces as a malloc abort later.
    for (int round = 0; round < 8; ++round) {
        auto* temp = static_cast<std::uint8_t*>(std::malloc(kBytes));
        if (!expect(temp != nullptr, "malloc")) return 1;
        if (!expect(cudaLaunchHostFunc(stream, sleep_host_fn, nullptr) == cudaSuccess,
                    "cudaLaunchHostFunc")) {
            return 1;
        }
        if (!expect(cudaMemcpyAsync(temp, dev, kBytes, cudaMemcpyDeviceToHost, stream) ==
                        cudaSuccess,
                    "cudaMemcpyAsync D2H into a temporary")) {
            return 1;
        }
        std::free(temp);
        auto* churn = static_cast<std::uint8_t*>(std::malloc(kBytes));
        if (!expect(churn != nullptr, "malloc")) return 1;
        std::memset(churn, 0xEE, kBytes);
        std::free(churn);
    }
    if (!expect(cudaStreamSynchronize(stream) == cudaSuccess, "cudaStreamSynchronize")) return 1;

    if (!expect(cudaStreamDestroy(stream) == cudaSuccess, "cudaStreamDestroy")) return 1;
    if (!expect(cudaFree(dev) == cudaSuccess && cudaFree(dev2) == cudaSuccess, "cudaFree")) return 1;

    std::printf("PASS: asynchronous copies honour CUDA's pageable-memory contract\n");
    return 0;
}
