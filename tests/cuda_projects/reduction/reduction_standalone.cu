// reduction_standalone.cu — parallel sum reduction
// Uses warp __shfl_down_sync + shared memory tree, following the
// NVIDIA cuda-samples reduce5 pattern with cooperative_groups.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

// Warp-level sum (matches cuda-samples warpReduceSum)
__device__ __forceinline__ float warpReduceSum(unsigned int mask, float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// Block reduction: sequential addressing + warp-shuffle final stage
template <unsigned int blockSize>
__global__ void reduce(const float *g_idata, float *g_odata, unsigned int n) {
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ float sdata[];

    unsigned int tid      = threadIdx.x;
    unsigned int i        = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockSize  * 2 * gridDim.x;

    float mySum = 0.0f;
    while (i < n) {
        mySum += g_idata[i];
        if (i + blockSize < n) mySum += g_idata[i + blockSize];
        i += gridSize;
    }
    sdata[tid] = mySum;
    cg::sync(cta);

    if (blockSize >= 512) { if (tid < 256) sdata[tid] += sdata[tid+256]; cg::sync(cta); }
    if (blockSize >= 256) { if (tid < 128) sdata[tid] += sdata[tid+128]; cg::sync(cta); }
    if (blockSize >= 128) { if (tid <  64) sdata[tid] += sdata[tid+ 64]; cg::sync(cta); }

    // Final warp: for blockSize >= 64, the 64-element intermediate result must be
    // reduced to 32 elements before the warp-shuffle stage.
    if (tid < 32) {
        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
        float warpSum = warpReduceSum(0xffffffffu, sdata[tid]);
        if (tid == 0) g_odata[blockIdx.x] = warpSum;
    }
}

int main(void) {
    const unsigned int N         = 1 << 20;  // 1M floats
    const unsigned int blockSize = 256;
    const unsigned int gridSize  = (N + blockSize*2 - 1) / (blockSize*2);

    float *h_in  = (float*)malloc(N        * sizeof(float));
    float *h_out = (float*)malloc(gridSize * sizeof(float));

    // All ones → expected sum = N
    for (unsigned int i = 0; i < N; i++) h_in[i] = 1.0f;
    const float expected = (float)N;

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N        * sizeof(float));
    cudaMalloc(&d_out, gridSize * sizeof(float));
    cudaMemcpy(d_in, h_in, N*sizeof(float), cudaMemcpyHostToDevice);

    reduce<blockSize><<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, gridSize*sizeof(float), cudaMemcpyDeviceToHost);

    // Final CPU reduction over partial block sums
    float result = 0.0f;
    for (unsigned int i = 0; i < gridSize; i++) result += h_out[i];

    float relerr = fabsf(result - expected) / expected;
    if (relerr > 1e-4f) {
        printf("FAIL: reduction got %.0f expected %.0f relerr %.2e\n",
               result, expected, relerr);
        return 1;
    }
    printf("PASS: parallel reduction (N=%u, sum=%.0f)\n", N, result);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}
