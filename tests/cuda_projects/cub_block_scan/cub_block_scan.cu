// cub::BlockScan executing on the GPU.
//
// The shim used to be a host-only sequential fallback: the default
// constructor hard-wired linear_tid_ to zero, the methods carried no
// __host__ __device__ annotation, and no barrier separated the shared-memory
// writes from the reads, so a kernel could not use it as a cooperative block
// scan. This project pins the device behaviour: exclusive/inclusive scans
// with aggregates over multiple blocks and warps, a 3D block, a custom
// associative operator, and temp-storage reuse across the documented caller
// barrier.
#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>

#include <cstdio>

constexpr int kThreads = 256;   // eight warps
constexpr int kBlocks = 4;

// Row-major expected values for a 1D block scanning input[i] = i + 1.
__global__ void exclusive_sum_kernel(const int* input, int* output, int* totals) {
    typedef cub::BlockScan<int, kThreads> Scan;
    __shared__ typename Scan::TempStorage storage;

    const int tid = static_cast<int>(threadIdx.x);
    int prefix = 0;
    int aggregate = 0;
    Scan(storage).ExclusiveSum(input[blockIdx.x * kThreads + tid], prefix, aggregate);
    output[blockIdx.x * kThreads + tid] = prefix;
    if (tid == 0)
        totals[blockIdx.x] = aggregate;
}

// Storage reuse: the same TempStorage serves a second scan after the
// caller's documented __syncthreads().
__global__ void reuse_kernel(int* first, int* second) {
    typedef cub::BlockScan<int, kThreads> Scan;
    __shared__ typename Scan::TempStorage storage;

    const int tid = static_cast<int>(threadIdx.x);
    int out = 0;
    Scan(storage).InclusiveSum(tid + 1, out);
    __syncthreads();  // required before the temp storage is reused
    first[blockIdx.x * kThreads + tid] = out;

    int exclusive = 0;
    Scan(storage).ExclusiveSum(tid + 1, exclusive);
    second[blockIdx.x * kThreads + tid] = exclusive;
}

// A custom associative operator that is not addition.
struct MaxOp {
    __host__ __device__ int operator()(const int& a, const int& b) const {
        return a > b ? a : b;
    }
};

__global__ void custom_op_kernel(const int* input, int* maxima) {
    typedef cub::BlockScan<int, kThreads> Scan;
    __shared__ typename Scan::TempStorage storage;

    const int tid = static_cast<int>(threadIdx.x);
    int running_max = 0;
    Scan(storage).InclusiveScan(input[blockIdx.x * kThreads + tid], running_max, MaxOp());
    maxima[blockIdx.x * kThreads + tid] = running_max;
}

// 3D block: 8 x 4 x 2 = 64 threads, row-major linear index.
__global__ void block_3d_kernel(int* output) {
    typedef cub::BlockScan<int, 8, cub::BLOCK_SCAN_RAKING, 4, 2> Scan;
    __shared__ typename Scan::TempStorage storage;

    const int linear = static_cast<int>(threadIdx.x +
                                        blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
    int prefix = -1;
    Scan(storage).ExclusiveSum(linear + 1, prefix);
    output[linear] = prefix;
}

int main() {
    constexpr int kN = kBlocks * kThreads;
    int* input = nullptr;
    int* output = nullptr;
    int* totals = nullptr;
    int* first = nullptr;
    int* second = nullptr;
    int* maxima = nullptr;
    int* block3d = nullptr;
    if (cudaMallocManaged(&input, kN * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&output, kN * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&totals, kBlocks * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&first, kN * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&second, kN * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&maxima, kN * sizeof(int)) != cudaSuccess ||
        cudaMallocManaged(&block3d, 64 * sizeof(int)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    for (int i = 0; i < kN; ++i) {
        input[i] = (i % kThreads) + 1;
        output[i] = -1;
        first[i] = -1;
        second[i] = -1;
        maxima[i] = -1;
    }

    exclusive_sum_kernel<<<kBlocks, kThreads>>>(input, output, totals);
    reuse_kernel<<<kBlocks, kThreads>>>(first, second);
    custom_op_kernel<<<kBlocks, kThreads>>>(input, maxima);
    block_3d_kernel<<<1, dim3(8, 4, 2)>>>(block3d);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    int failures = 0;
    // Exclusive sum of 1..tid is tid(tid+1)/2; the aggregate is N(N+1)/2.
    for (int i = 0; i < kN; ++i) {
        const int tid = i % kThreads;
        const int expected_exclusive = tid * (tid + 1) / 2;
        if (output[i] != expected_exclusive) {
            std::printf("FAIL: exclusive sum at %d: %d expected %d\n", i, output[i],
                        expected_exclusive);
            ++failures;
        }
        if (first[i] != (tid + 1) * (tid + 2) / 2) {
            std::printf("FAIL: inclusive reuse at %d: %d\n", i, first[i]);
            ++failures;
        }
        if (second[i] != expected_exclusive) {
            std::printf("FAIL: exclusive reuse at %d: %d expected %d\n", i, second[i],
                        expected_exclusive);
            ++failures;
        }
        if (maxima[i] != tid + 1) {
            std::printf("FAIL: running max at %d: %d expected %d\n", i, maxima[i], tid + 1);
            ++failures;
        }
    }
    for (int b = 0; b < kBlocks; ++b) {
        if (totals[b] != kThreads * (kThreads + 1) / 2) {
            std::printf("FAIL: block %d aggregate %d expected %d\n", b, totals[b],
                        kThreads * (kThreads + 1) / 2);
            ++failures;
        }
    }
    // 3D block: exclusive sum of linear+1 is linear*(linear+1)/2.
    for (int i = 0; i < 64; ++i) {
        if (block3d[i] != i * (i + 1) / 2) {
            std::printf("FAIL: 3D block at %d: %d expected %d\n", i, block3d[i],
                        i * (i + 1) / 2);
            ++failures;
        }
    }

    if (failures != 0) {
        std::printf("FAIL: %d checks\n", failures);
        return 1;
    }
    std::printf("PASS: cub::BlockScan exclusive/inclusive sums, aggregates, "
                "custom op, storage reuse, 3D block\n");
    return 0;
}
