#include "cuda.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Host reference for fixtures/stwo_compat_probe.cu.
namespace {
constexpr unsigned int kOutputsPerThread = 17;
constexpr unsigned int kCount = 200;
constexpr unsigned int kThreads = 64;
constexpr unsigned long long kStride = 20;
constexpr unsigned long long kHash = 0xfedcba9876543210ull;
constexpr std::uint32_t kRoundConstants[8] = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

std::uint32_t funnel(std::uint32_t lo, std::uint32_t hi, unsigned n, bool left) {
    const std::uint64_t v = (static_cast<std::uint64_t>(hi) << 32) | lo;
    return static_cast<std::uint32_t>(left ? (v << n) >> 32 : v >> n);
}

std::uint32_t reverse(std::uint32_t v) {
    std::uint32_t r = 0;
    for (int i = 0; i < 32; ++i) r |= ((v >> i) & 1u) << (31 - i);
    return r;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) return 2;

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr words = 0, left = 0, right = 0, out = 0, minimum = 0;
    const std::size_t out_words = static_cast<std::size_t>(kCount) * kStride;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&kernel, module, "stwo_compat_probe") != CUDA_SUCCESS ||
        cuMemAlloc(&words, kCount * 4) != CUDA_SUCCESS ||
        cuMemAlloc(&left, kCount * 4) != CUDA_SUCCESS ||
        cuMemAlloc(&right, kCount * 4) != CUDA_SUCCESS ||
        cuMemAlloc(&out, out_words * 4) != CUDA_SUCCESS ||
        cuMemAlloc(&minimum, 8) != CUDA_SUCCESS) {
        std::fprintf(stderr, "stwo compat probe: setup failed\n");
        return 1;
    }

    std::vector<std::uint32_t> host_words(kCount), host_left(kCount), host_right(kCount);
    std::uint64_t state = 0x243f6a8885a308d3ull;
    for (unsigned int i = 0; i < kCount; ++i) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        host_words[i] = static_cast<std::uint32_t>(state >> 32);
        host_left[i] = 0x10000000u + i;
        host_right[i] = 0x20000000u + i;
    }
    const std::uint64_t initial_minimum = ~0ull;
    const std::vector<std::uint32_t> poison(out_words, 0xdeadbeefu);
    if (cuMemcpyHtoD(words, host_words.data(), kCount * 4) != CUDA_SUCCESS ||
        cuMemcpyHtoD(left, host_left.data(), kCount * 4) != CUDA_SUCCESS ||
        cuMemcpyHtoD(right, host_right.data(), kCount * 4) != CUDA_SUCCESS ||
        cuMemcpyHtoD(out, poison.data(), out_words * 4) != CUDA_SUCCESS ||
        cuMemcpyHtoD(minimum, &initial_minimum, 8) != CUDA_SUCCESS) {
        std::fprintf(stderr, "stwo compat probe: input copy failed\n");
        return 1;
    }

    unsigned long long stride = kStride, hash = kHash;
    unsigned int count = kCount;
    void* args[] = {&words, &left, &right, &out, &minimum, &stride, &hash, &count};
    const unsigned int blocks = (kCount + kThreads - 1) / kThreads;
    const CUresult launched =
        cuLaunchKernel(kernel, blocks, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr);
    if (launched != CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "stwo compat probe: launch failed (%d)\n", static_cast<int>(launched));
        return 1;
    }

    std::vector<std::uint32_t> result(out_words);
    std::uint64_t device_minimum = 0;
    if (cuMemcpyDtoH(result.data(), out, out_words * 4) != CUDA_SUCCESS ||
        cuMemcpyDtoH(&device_minimum, minimum, 8) != CUDA_SUCCESS) {
        std::fprintf(stderr, "stwo compat probe: readback failed\n");
        return 1;
    }

    static const char* const kNames[kOutputsPerThread] = {
        "__funnelshift_l", "__funnelshift_r", "__funnelshift_lc", "__funnelshift_rc", "__brev",
        "__match_any_sync(u32)", "__match_any_sync(u64)", "pointer select",
        "shared atomicExch", "u64 hash high", "u64 hash low", "u64 stride",
        "shf.l.wrap", "shf.r.wrap", "shf.l.clamp", "shf.r.clamp", "__const_$ table"};
    unsigned int failures = 0;
    std::uint64_t expected_minimum = ~0ull;
    for (unsigned int tid = 0; tid < kCount; ++tid) {
        const std::uint32_t low = host_words[tid];
        const std::uint32_t high = host_words[(tid + 1) % kCount];
        const unsigned shift = (low ^ high) % 41u;
        const unsigned clamped = shift < 32 ? shift : 32;
        const unsigned lane = (tid % kThreads) & 31u;
        std::uint32_t match32 = 0, match64 = 0;
        for (unsigned other = 0; other < 32; ++other) {
            if (other % 3 == lane % 3) match32 |= 1u << other;
            if (other % 5 == lane % 5 && (other & 1u) == (lane & 1u)) match64 |= 1u << other;
        }
        const std::uint32_t expected[kOutputsPerThread] = {
            funnel(low, high, shift & 31u, true),
            funnel(low, high, shift & 31u, false),
            funnel(low, high, clamped, true),
            funnel(low, high, clamped, false),
            reverse(low),
            match32,
            match64,
            (low & 1u) ? host_left[tid] : host_right[tid],
            0u,
            static_cast<std::uint32_t>(kHash >> 32),
            static_cast<std::uint32_t>(kHash),
            static_cast<std::uint32_t>(kStride),
            funnel(low, high, shift & 31u, true),
            funnel(low, high, shift & 31u, false),
            funnel(low, high, clamped, true),
            funnel(low, high, clamped, false),
            kRoundConstants[(low >> 3) & 7u],
        };
        const std::uint64_t key = (static_cast<std::uint64_t>(low) << 32) | high;
        if (key < expected_minimum) expected_minimum = key;
        for (unsigned int k = 0; k < kOutputsPerThread; ++k) {
            const std::uint32_t got = result[tid * kStride + k];
            if (got != expected[k] && failures++ < 12) {
                std::fprintf(stderr, "stwo compat probe: thread %u %s = 0x%08x, expected 0x%08x (shift %u)\n",
                             tid, kNames[k], got, expected[k], shift);
            }
        }
        for (unsigned int k = kOutputsPerThread; k < kStride; ++k) {
            if (result[tid * kStride + k] != 0xdeadbeefu && failures++ < 12) {
                std::fprintf(stderr, "stwo compat probe: thread %u wrote past its row\n", tid);
            }
        }
    }
    if (device_minimum != expected_minimum) {
        std::fprintf(stderr, "stwo compat probe: 64-bit atomicMin = 0x%016llx, expected 0x%016llx\n",
                     static_cast<unsigned long long>(device_minimum),
                     static_cast<unsigned long long>(expected_minimum));
        ++failures;
    }
    if (failures != 0) {
        std::fprintf(stderr, "stwo compat probe: %u mismatches\n", failures);
        return 1;
    }

    cuMemFree(words);
    cuMemFree(left);
    cuMemFree(right);
    cuMemFree(out);
    cuMemFree(minimum);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    std::puts("PASS: stwo-zig compatibility constructs match CUDA");
    return 0;
}
