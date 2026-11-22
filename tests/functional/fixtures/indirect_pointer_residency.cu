// #224: a kernel that follows device addresses stored in device memory --
// descriptor structs, pointer tables -- reaches allocations it never bound.
// Without residency the GPU read zeros and silently dropped the writes for
// every allocation from 1 MiB up, in the default pointer mode. Covers small,
// standalone and heap-backed sizes, a stream created before and one created
// after the allocations, a read-back kernel, and a freed-and-reallocated
// buffer.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

struct Desc {
    int* data;
    unsigned n;
};

__global__ void write_through(const Desc* descs, int count, int value) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int d = 0; d < count; ++d)
        if (i < descs[d].n) descs[d].data[i] = value + d;
}

// Reads through the pointer and writes back: a later kernel on the same
// stream must see the first one's writes.
__global__ void increment_through(const Desc* descs, int count) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int d = 0; d < count; ++d)
        if (i < descs[d].n) descs[d].data[i] += 1;
}

constexpr int kCount = 4;
static const size_t kSizes[kCount] = {1u << 12, 1u << 20, 8u << 20, 64u << 20};

static int check(Desc* host_desc, Desc* descs, cudaStream_t stream, int value,
                 const char* what) {
    const unsigned max_n = host_desc[kCount - 1].n;
    write_through<<<(max_n + 255) / 256, 256, 0, stream>>>(descs, kCount, value);
    increment_through<<<(max_n + 255) / 256, 256, 0, stream>>>(descs, kCount);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        std::printf("FAIL: %s: launch\n", what);
        return 1;
    }
    for (int d = 0; d < kCount; ++d) {
        int* host = static_cast<int*>(std::malloc(kSizes[d]));
        cudaMemcpy(host, host_desc[d].data, kSizes[d], cudaMemcpyDeviceToHost);
        unsigned wrong = 0;
        for (unsigned i = 0; i < host_desc[d].n; ++i) wrong += host[i] != value + d + 1;
        std::free(host);
        if (wrong != 0) {
            std::printf("FAIL: %s: buffer %d (%zu bytes): %u of %u elements wrong\n", what, d,
                        kSizes[d], wrong, host_desc[d].n);
            return 1;
        }
    }
    return 0;
}

int main() {
    cudaStream_t early = nullptr;
    if (cudaStreamCreateWithFlags(&early, cudaStreamNonBlocking) != cudaSuccess) {
        std::printf("FAIL: stream\n");
        return 1;
    }
    Desc host_desc[kCount];
    for (int d = 0; d < kCount; ++d) {
        host_desc[d].n = static_cast<unsigned>(kSizes[d] / sizeof(int));
        if (cudaMalloc(&host_desc[d].data, kSizes[d]) != cudaSuccess) {
            std::printf("FAIL: cudaMalloc\n");
            return 1;
        }
    }
    Desc* descs = nullptr;
    cudaMalloc(&descs, sizeof host_desc);
    cudaMemcpy(descs, host_desc, sizeof host_desc, cudaMemcpyHostToDevice);

    if (check(host_desc, descs, nullptr, 100, "legacy stream")) return 1;
    if (check(host_desc, descs, early, 200, "stream created before the allocations")) return 1;
    cudaStream_t late = nullptr;
    cudaStreamCreateWithFlags(&late, cudaStreamNonBlocking);
    if (check(host_desc, descs, late, 300, "stream created after the allocations")) return 1;

    // Free one allocation and allocate another in its place.
    cudaFree(host_desc[1].data);
    cudaMalloc(&host_desc[1].data, kSizes[1]);
    cudaMemcpy(descs, host_desc, sizeof host_desc, cudaMemcpyHostToDevice);
    if (check(host_desc, descs, late, 400, "after free and reallocation")) return 1;

    std::printf("PASS: kernels reach allocations through stored device addresses\n");
    return 0;
}
