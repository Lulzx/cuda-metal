// cub::BlockReduce executing on the GPU.
//
// The shim used to be a host-only sequential fallback: its methods were
// __host__, so a kernel could not call them, and its TempStorage was a plain
// array of T, which is illegal in __shared__ as soon as T has a user-provided
// default constructor. Both bit NVIDIA Warp's bvh.cu, which reduces wp::vec3
// bounds over a partial tile. This project pins the device behaviour: a real
// cooperative reduction, correct for a partial tile, over a type whose default
// constructor keeps it out of __shared__ unless the storage is raw bytes.
#include <cub/cub.h>
#include <cuda_runtime.h>

#include <cstdio>

constexpr int kThreads = 256;
constexpr int kBlocks = 4;
constexpr int kValid = 100;  // partial tile: only the first kValid threads count

struct Vec3 {
    float x, y, z;
    // User-provided, so Vec3 is not trivially default constructible.
    __host__ __device__ Vec3() : x(0.0f), y(0.0f), z(0.0f) {}
    __host__ __device__ Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
};

// A functor rather than a plain function. Warp's bvh.cu passes a function, which
// decays to a pointer and reaches the reduction as an indirect call; that needs
// the optimizer to devirtualize it, and cumetalc's native-AOT path rejects it
// outright (see docs/known-gaps/compiler.md). The reduction is the same either
// way, so this fixture uses the spelling that works on every path.
struct Vec3Max {
    __host__ __device__ Vec3 operator()(const Vec3& a, const Vec3& b) const {
        return Vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
    }
};

__global__ void block_reduce_kernel(float* partial_sums, float* full_sums, Vec3* maxima) {
    typedef cub::BlockReduce<float, kThreads> FloatReduce;
    typedef cub::BlockReduce<Vec3, kThreads> Vec3Reduce;

    __shared__ typename FloatReduce::TempStorage float_temp;
    __shared__ typename Vec3Reduce::TempStorage vec3_temp;

    const int tid = static_cast<int>(threadIdx.x);
    const int gid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    // Partial tile: threads at or past kValid contribute a poison value that
    // must never reach the result.
    const float partial_in = tid < kValid ? static_cast<float>(gid + 1) : 1.0e9f;
    const float partial = FloatReduce(float_temp).Sum(partial_in, kValid);
    __syncthreads();  // temp storage is reused below

    const float full = FloatReduce(float_temp).Sum(1.0f);
    __syncthreads();

    const Vec3 poison(-1.0e9f, -1.0e9f, -1.0e9f);
    const Vec3 vec_in = tid < kValid ? Vec3(static_cast<float>(tid), -static_cast<float>(tid),
                                            static_cast<float>(tid) * 2.0f)
                                     : poison;
    const Vec3 largest = Vec3Reduce(vec3_temp).Reduce(vec_in, Vec3Max(), kValid);

    if (tid == 0) {
        partial_sums[blockIdx.x] = partial;
        full_sums[blockIdx.x] = full;
        maxima[blockIdx.x] = largest;
    }
}

int main() {
    float* partial_sums = nullptr;
    float* full_sums = nullptr;
    Vec3* maxima = nullptr;
    if (cudaMallocManaged(&partial_sums, kBlocks * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&full_sums, kBlocks * sizeof(float)) != cudaSuccess ||
        cudaMallocManaged(&maxima, kBlocks * sizeof(Vec3)) != cudaSuccess) {
        std::printf("FAIL: cudaMallocManaged\n");
        return 1;
    }
    for (int i = 0; i < kBlocks; ++i) {
        partial_sums[i] = -1.0f;
        full_sums[i] = -1.0f;
        maxima[i] = Vec3(-1.0f, -1.0f, -1.0f);
    }

    block_reduce_kernel<<<kBlocks, kThreads>>>(partial_sums, full_sums, maxima);
    if (const cudaError_t error = cudaDeviceSynchronize(); error != cudaSuccess) {
        std::printf("FAIL: cudaDeviceSynchronize: %s\n", cudaGetErrorString(error));
        return 1;
    }

    int failures = 0;
    for (int block = 0; block < kBlocks; ++block) {
        // Threads 0..kValid-1 of block b hold b*kThreads + tid + 1.
        const float expected_partial =
            static_cast<float>(kValid) * static_cast<float>(block * kThreads) +
            static_cast<float>(kValid) * static_cast<float>(kValid + 1) / 2.0f;
        if (partial_sums[block] != expected_partial) {
            std::printf("FAIL: block %d partial sum %.1f, expected %.1f\n", block,
                        partial_sums[block], expected_partial);
            ++failures;
        }
        if (full_sums[block] != static_cast<float>(kThreads)) {
            std::printf("FAIL: block %d full-tile sum %.1f, expected %d\n", block,
                        full_sums[block], kThreads);
            ++failures;
        }
        const Vec3 expected_max(static_cast<float>(kValid - 1), 0.0f,
                                static_cast<float>(kValid - 1) * 2.0f);
        if (maxima[block].x != expected_max.x || maxima[block].y != expected_max.y ||
            maxima[block].z != expected_max.z) {
            std::printf("FAIL: block %d max (%.1f, %.1f, %.1f), expected (%.1f, %.1f, %.1f)\n", block,
                        maxima[block].x, maxima[block].y, maxima[block].z, expected_max.x,
                        expected_max.y, expected_max.z);
            ++failures;
        }
    }

    if (failures != 0) {
        std::printf("FAIL: %d checks\n", failures);
        return 1;
    }
    std::printf("PASS: cub::BlockReduce partial tile, full tile and non-trivial element type\n");
    return 0;
}
