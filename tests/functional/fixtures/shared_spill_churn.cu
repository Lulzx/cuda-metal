// Thousands of shared-memory spill launches of a few MB each, in three grid
// sizes, synchronized one by one under CUMETAL_USE_METAL_DEVICE_ADDRESSES=1:
// the pattern of PhysX's PBD radix sort. Sub-allocating these per-launch spill
// buffers from the runtime's long-lived heap arenas ran Metal out of memory
// after about four thousand launches; spill buffers are now standalone.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void touch_past_limit(unsigned words, unsigned* out) {
    extern __shared__ unsigned s[];
    if (threadIdx.x == 0) s[words - 1] = blockIdx.x;
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = s[words - 1] + 7u;
}

int main() {
    unsigned* d_out = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(&d_out, sizeof(unsigned)) != cudaSuccess || cudaStreamCreate(&stream) != cudaSuccess) {
        std::printf("FAIL: setup\n");
        return 1;
    }
    constexpr int kLaunches = 6000;
    for (int i = 0; i < kLaunches; ++i) {
        // 32 KB + 4 B of dynamic shared memory per block: always spilled.
        touch_past_limit<<<76u * (1u + static_cast<unsigned>(i % 3)), 32, 8193 * 4, stream>>>(8193, d_out);
        if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
            std::printf("FAIL: spilled launch %d\n", i);
            return 1;
        }
    }
    unsigned answer = 0;
    if (cudaMemcpy(&answer, d_out, sizeof answer, cudaMemcpyDeviceToHost) != cudaSuccess || answer != 7u) {
        std::printf("FAIL: result %u\n", answer);
        return 1;
    }
    cudaStreamDestroy(stream);
    cudaFree(d_out);
    std::printf("PASS: %d spilled launches without exhausting Metal memory\n", kLaunches);
    return 0;
}
