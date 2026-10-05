// Contended double atomicAdd in device and threadgroup memory. Metal has no
// 64-bit atomics, so CuMetal serializes each address through its lock bank and
// adds with the active FP64 mode. Every addend is a small dyadic rational, so
// the exact sum is representable in all FP64 modes and any lost update shows.
// The header's atomicAdd(double*) is a CAS loop; the clang builtin emits the
// native atom.add.f64 that Kokkos and other sm_60+ code generate.
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

constexpr int kBlocks = 64;
constexpr int kThreads = 256;
constexpr int kTotal = kBlocks * kThreads;
constexpr int kSlots = 4;

__device__ double native_add(double* address, double value) {
    return __nvvm_atom_add_gen_d(address, value);
}

__global__ void contend(double* slots, double* counter_old, double* block_sums) {
    __shared__ double block_sum;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.x == 0) block_sum = 0.0;
    __syncthreads();
    const double value = 0.25 * (tid % 7 + 1);
    native_add(&slots[tid % kSlots], value);
    native_add(&block_sum, value);
    // Every returned old value of a unit counter must be distinct.
    counter_old[tid] = native_add(&slots[kSlots], 1.0);
    __syncthreads();
    if (threadIdx.x == 0) block_sums[blockIdx.x] = block_sum;
}

int main() {
    double *slots, *counter_old, *block_sums;
    cudaMalloc(&slots, (kSlots + 1) * sizeof(double));
    cudaMalloc(&counter_old, kTotal * sizeof(double));
    cudaMalloc(&block_sums, kBlocks * sizeof(double));
    cudaMemset(slots, 0, (kSlots + 1) * sizeof(double));
    contend<<<kBlocks, kThreads>>>(slots, counter_old, block_sums);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: launch: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    std::vector<double> got(kSlots + 1), old(kTotal), blocks(kBlocks);
    cudaMemcpy(got.data(), slots, got.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(old.data(), counter_old, kTotal * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(blocks.data(), block_sums, kBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    int failures = 0;
    std::vector<double> want(kSlots, 0.0), want_blocks(kBlocks, 0.0);
    for (int tid = 0; tid < kTotal; ++tid) {
        want[tid % kSlots] += 0.25 * (tid % 7 + 1);
        want_blocks[tid / kThreads] += 0.25 * (tid % 7 + 1);
    }
    for (int s = 0; s < kSlots; ++s) {
        if (got[s] != want[s]) {
            std::printf("FAIL: slot %d = %.17g, want %.17g\n", s, got[s], want[s]);
            ++failures;
        }
    }
    if (got[kSlots] != kTotal) {
        std::printf("FAIL: counter = %.17g, want %d\n", got[kSlots], kTotal);
        ++failures;
    }
    std::vector<bool> seen(kTotal, false);
    for (int tid = 0; tid < kTotal; ++tid) {
        const double v = old[tid];
        const int i = static_cast<int>(v);
        if (v != i || i < 0 || i >= kTotal || seen[i]) {
            if (failures++ < 8) std::printf("FAIL: thread %d fetched old %.17g\n", tid, v);
            continue;
        }
        seen[i] = true;
    }
    for (int b = 0; b < kBlocks; ++b) {
        if (blocks[b] != want_blocks[b]) {
            if (failures++ < 8)
                std::printf("FAIL: block %d sum = %.17g, want %.17g\n", b, blocks[b],
                            want_blocks[b]);
        }
    }
    if (failures != 0) return 1;
    std::printf("PASS: contended native double atomic add exact in device and threadgroup memory "
                "(%d threads)\n", kTotal);
    return 0;
}
