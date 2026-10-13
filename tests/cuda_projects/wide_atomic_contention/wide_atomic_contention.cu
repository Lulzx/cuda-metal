// Cross-threadgroup contention must preserve the payload as well as its lock.
// The seed exercises high bits and carries across the low 32-bit word.
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

constexpr unsigned int kAddBlocks = 64;
constexpr unsigned int kCasBlocks = 16;
constexpr unsigned int kThreads = 256;
constexpr unsigned int kAddTotal = kAddBlocks * kThreads;
constexpr unsigned int kCasTotal = kCasBlocks * kThreads;
constexpr unsigned long long kSeed = 0xfedcba98fffff800ull;

extern "C" __global__ void wide_add(unsigned long long* counter,
                                    unsigned long long* old_values) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    old_values[tid] = atomicAdd(counter, 1ull);
}

extern "C" __global__ void wide_cas(unsigned long long* counter,
                                    unsigned long long* old_values) {
    const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long seen = kSeed;
    unsigned long long previous;
    while ((previous = atomicCAS(counter, seen, seen + 1ull)) != seen) {
        seen = previous;
    }
    old_values[tid] = previous;
}

static bool api(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::printf("FAIL: %s: %s (%d)\n", operation, cudaGetErrorString(error), error);
    return false;
}

static bool verify(const char* operation, unsigned long long counter,
                   const unsigned long long* old_values, unsigned int total) {
    std::vector<unsigned int> seen(total, 0);
    unsigned int invalid = 0, duplicates = 0, missing = 0;
    for (unsigned int tid = 0; tid < total; ++tid) {
        const unsigned long long old = old_values[tid];
        if (old < kSeed || old >= kSeed + total) {
            if (invalid++ < 4)
                std::printf("FAIL: %s tid=%u old=0x%llx outside expected range\n",
                            operation, tid, old);
        } else if (seen[static_cast<unsigned int>(old - kSeed)]++ != 0) {
            ++duplicates;
        }
    }
    for (unsigned int count : seen) missing += count == 0;
    const bool ok = counter == kSeed + total && invalid == 0 &&
                    duplicates == 0 && missing == 0;
    if (!ok)
        std::printf("FAIL: %s counter=0x%llx expected=0x%llx "
                    "invalid=%u duplicates=%u missing=%u\n", operation,
                    counter, kSeed + total, invalid, duplicates, missing);
    return ok;
}

int main() {
    unsigned long long* counters = nullptr;
    unsigned long long* old_values = nullptr;
    auto release = [&]() {
        bool ok = true;
        if (old_values) ok = api(cudaFree(old_values), "cudaFree(old values)") && ok;
        if (counters) ok = api(cudaFree(counters), "cudaFree(counters)") && ok;
        return ok;
    };
    const unsigned long long initial[] = {kSeed, kSeed};
    constexpr size_t output_bytes = (kAddTotal + kCasTotal) * sizeof(unsigned long long);
    if (!api(cudaMalloc(&counters, sizeof(initial)), "cudaMalloc(counters)") ||
        !api(cudaMalloc(&old_values, output_bytes), "cudaMalloc(old values)") ||
        !api(cudaMemcpy(counters, initial, sizeof(initial), cudaMemcpyHostToDevice),
             "cudaMemcpy(seed)") ||
        !api(cudaMemset(old_values, 0xff, output_bytes), "cudaMemset(poison)")) {
        release();
        return 1;
    }

    wide_add<<<kAddBlocks, kThreads>>>(counters, old_values);
    if (!api(cudaGetLastError(), "wide_add launch") ||
        !api(cudaDeviceSynchronize(), "wide_add synchronize")) {
        release();
        return 1;
    }

    // Report lost add updates before attempting the more expensive CAS loop.
    unsigned long long result;
    std::vector<unsigned long long> old(kAddTotal);
    if (!api(cudaMemcpy(&result, counters, sizeof(result), cudaMemcpyDeviceToHost),
             "cudaMemcpy(add counter)") ||
        !api(cudaMemcpy(old.data(), old_values, kAddTotal * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost), "cudaMemcpy(add old values)") ||
        !verify("atomicAdd", result, old.data(), kAddTotal)) {
        release();
        return 1;
    }

    wide_cas<<<kCasBlocks, kThreads>>>(counters + 1, old_values + kAddTotal);
    if (!api(cudaGetLastError(), "wide_cas launch") ||
        !api(cudaDeviceSynchronize(), "wide_cas synchronize")) {
        release();
        return 1;
    }

    if (!api(cudaMemcpy(&result, counters + 1, sizeof(result), cudaMemcpyDeviceToHost),
             "cudaMemcpy(CAS counter)") ||
        !api(cudaMemcpy(old.data(), old_values + kAddTotal,
                        kCasTotal * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
             "cudaMemcpy(CAS old values)")) {
        release();
        return 1;
    }
    bool ok = verify("atomicCAS", result, old.data(), kCasTotal);
    ok = release() && ok;
    if (!ok) return 1;
    std::printf("PASS: contended 64-bit integer atomics preserve every update "
                "(add %u blocks, CAS %u blocks, %u threads each)\n",
                kAddBlocks, kCasBlocks, kThreads);
    return 0;
}
