// transpose_standalone.cu — matrix transpose with shared memory (no bank conflicts)
// Adapted from NVIDIA cuda-samples/6_Performance/transpose/transpose.cu
// Uses cooperative_groups cg::this_thread_block() and cta.sync().

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

namespace cg = cooperative_groups;

#define TILE_DIM   32
#define BLOCK_ROWS 8

// Simple copy kernel (baseline)
__global__ void copy_kernel(float *odata, const float *idata, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < width)
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            if (y + i < height)
                odata[(y+i)*width + x] = idata[(y+i)*width + x];
}

// Naive transpose (uncoalesced stores)
__global__ void transpose_naive(float *odata, const float *idata, int width, int height) {
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    if (x < width)
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            if (y + i < height)
                odata[x*height + (y+i)] = idata[(y+i)*width + x];
}

// Coalesced transpose with shared memory (no bank conflicts: +1 padding)
__global__ void transpose_shmem(float *odata, const float *idata, int width, int height) {
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (x < width)
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            if (y + i < height)
                tile[threadIdx.y + i][threadIdx.x] = idata[(y+i)*width + x];
    cta.sync();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    if (x < height)
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            if (y + i < width)
                odata[(y+i)*height + x] = tile[threadIdx.x][threadIdx.y + i];
}

static void transpose_cpu(float *out, const float *in, int W, int H) {
    for (int r = 0; r < H; r++)
        for (int c = 0; c < W; c++)
            out[c*H + r] = in[r*W + c];
}

int main(void) {
    const int W = 1024, H = 1024;
    const float TOL = 0.0f;  // exact for simple values

    size_t sz = (size_t)W * H * sizeof(float);
    float *h_in   = (float*)malloc(sz);
    float *h_out  = (float*)malloc(sz);
    float *h_ref  = (float*)malloc(sz);

    for (int i = 0; i < W*H; i++) h_in[i] = (float)i;
    transpose_cpu(h_ref, h_in, W, H);

    float *d_in, *d_out;
    cudaMalloc(&d_in,  sz);
    cudaMalloc(&d_out, sz);
    cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((W+TILE_DIM-1)/TILE_DIM, (H+TILE_DIM-1)/TILE_DIM);

    // Test 1: naive transpose
    cudaMemset(d_out, 0, sz);
    transpose_naive<<<grid, block>>>(d_out, d_in, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    int errs = 0;
    for (int i = 0; i < W*H; i++) if (h_out[i] != h_ref[i]) errs++;
    if (errs) { printf("FAIL: naive transpose: %d errors\n", errs); return 1; }
    printf("PASS: transpose_naive (%dx%d)\n", W, H);

    // Test 2: shared memory transpose (no bank conflicts)
    cudaMemset(d_out, 0, sz);
    transpose_shmem<<<grid, block>>>(d_out, d_in, W, H);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);
    errs = 0;
    for (int i = 0; i < W*H; i++) if (h_out[i] != h_ref[i]) errs++;
    if (errs) { printf("FAIL: transpose_shmem: %d errors\n", errs); return 1; }
    printf("PASS: transpose_shmem (shared mem, no bank conflicts, %dx%d)\n", W, H);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out); free(h_ref);
    return 0;
}
