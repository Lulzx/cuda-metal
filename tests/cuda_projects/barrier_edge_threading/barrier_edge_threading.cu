// One thread initializes shared state, all threads update it between two
// barriers, and the same thread reads it back. Both guards test one
// predicate, which once let the PTX edge threader clone the barrier region
// into a lane-0 copy and an everyone-else copy. Metal pairs barriers across
// the copies, so lane 0 zeroed the sum after its SIMD-group peers had added.
#include <cuda_runtime.h>
#include <cstdio>

constexpr int kBlocks = 32;
constexpr int kThreads = 256;

__global__ void count_block(unsigned* out) {
    __shared__ unsigned total;
    if (threadIdx.x == 0) total = 0;
    __syncthreads();
    atomicAdd(&total, threadIdx.x + 1);
    __syncthreads();
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}

int main() {
    unsigned* out;
    cudaMalloc(&out, kBlocks * sizeof(unsigned));
    count_block<<<kBlocks, kThreads>>>(out);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: launch: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    unsigned got[kBlocks];
    cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost);
    const unsigned want = kThreads * (kThreads + 1) / 2;
    int failures = 0;
    for (int b = 0; b < kBlocks; ++b) {
        if (got[b] != want && failures++ < 8)
            std::printf("FAIL: block %d total = %u, want %u\n", b, got[b], want);
    }
    if (failures != 0) return 1;
    std::printf("PASS: guarded shared initialization survives barrier edge threading "
                "(%d blocks)\n", kBlocks);
    return 0;
}
