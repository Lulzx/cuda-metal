// Float atomics on the source-first path. Metal exposes atomic_float add,
// sub and exchange natively; CuMetal's overlay used to spell float atomicAdd
// as inline PTX that only the PTX backend could lower. Every accumulation and
// every Warp adjoint kernel routes through this, so the result must be exact
// for integer-valued inputs and both device and threadgroup storage must
// serialize correctly under contention.
#include <cuda_runtime.h>

#include <cstdio>

constexpr int kThreads = 256;
constexpr int kBlocks = 16;
constexpr int kTotal = kThreads * kBlocks;

__global__ void float_atomic_kernel(float* device_sums, float* old_values,
                                    float* threadgroup_sums) {
    __shared__ float block_sum;
    const int tid = static_cast<int>(threadIdx.x);
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid == 0) block_sum = 0.0f;
    __syncthreads();

    // Device: every thread adds its 1-based index; the old value must be a
    // partial sum that the final total confirms.
    old_values[gid] = atomicAdd(&device_sums[0], static_cast<float>(gid + 1));
    // Intrinsic spelling and subtraction.
    __fAtomicAdd(&device_sums[1], 2.0f);
    atomicSub(reinterpret_cast<int*>(&device_sums[2]), 0);  // keep int path visible
    atomicAdd(&device_sums[2], -1.0f);

    // Threadgroup storage.
    atomicAdd(&block_sum, 1.0f);
    __syncthreads();
    if (tid == 0) threadgroup_sums[blockIdx.x] = block_sum;

    // Exchange returns the previous value; slot 3 ends as some thread's id.
    if (tid == 0) atomicExch(&device_sums[3], static_cast<float>(gid));
}

int main() {
    float* sums = nullptr;
    float* old_values = nullptr;
    float* block_sums = nullptr;
    if (cudaMallocManaged(&sums, 4 * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&old_values, kTotal * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&block_sums, kBlocks * sizeof(float)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    for (int i = 0; i < 4; ++i) sums[i] = 0.0f;
    sums[3] = -1.0f;
    for (int i = 0; i < kTotal; ++i) old_values[i] = -1.0f;
    for (int i = 0; i < kBlocks; ++i) block_sums[i] = -1.0f;

    float_atomic_kernel<<<kBlocks, kThreads>>>(sums, old_values, block_sums);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    int failures = 0;
    // 1 + 2 + ... + kTotal = 8,390,656 is exactly representable in binary32
    // (below 2^24), as is every partial sum, so exact comparison is valid.
    const double expected_total = static_cast<double>(kTotal) * (kTotal + 1) / 2.0;
    if (static_cast<double>(sums[0]) != expected_total) {
        std::printf("FAIL: device atomicAdd(float) total %.17g expected %.17g\n",
                    sums[0], expected_total);
        ++failures;
    }
    if (sums[1] != 2.0f * kTotal) {
        std::printf("FAIL: __fAtomicAdd total %.17g expected %d\n", sums[1], 2 * kTotal);
        ++failures;
    }
    if (sums[2] != -static_cast<float>(kTotal)) {
        std::printf("FAIL: negative atomicAdd(float) total %.17g expected %d\n", sums[2],
                    -kTotal);
        ++failures;
    }
    if (sums[3] < 0.0f || sums[3] >= kTotal ||
        static_cast<int>(sums[3]) % kThreads != 0) {
        std::printf("FAIL: atomicExch(float) left %.17g, not a block-leader id\n", sums[3]);
        ++failures;
    }
    for (int i = 0; i < kBlocks; ++i) {
        if (block_sums[i] != static_cast<float>(kThreads)) {
            std::printf("FAIL: threadgroup atomicAdd(float) block %d got %.17g expected %d\n",
                        i, block_sums[i], kThreads);
            ++failures;
            break;
        }
    }
    // Every returned old value must be a distinct partial sum in range, which
    // is only true if each add observed a consistent previous value.
    int out_of_range = 0;
    for (int i = 0; i < kTotal; ++i) {
        if (old_values[i] < 0.0f || static_cast<double>(old_values[i]) >= expected_total) {
            ++out_of_range;
        }
    }
    if (out_of_range != 0) {
        std::printf("FAIL: %d atomicAdd(float) old values out of range\n", out_of_range);
        ++failures;
    }
    cudaFree(sums);
    cudaFree(old_values);
    cudaFree(block_sums);
    if (failures != 0) return 1;
    std::printf("PASS: float atomics lower to native Metal atomic_float in device and threadgroup storage\n");
    return 0;
}
