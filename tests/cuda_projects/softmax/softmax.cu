// softmax.cu — online numerically-stable softmax with warp reductions
//
// Exercises on CuMetal:
//   - __shared__ memory + __syncthreads()
//   - __shfl_xor_sync warp reductions
//   - expf / fmaxf device math
//   - Multiple kernel launches
//   - cudaMalloc / cudaMemcpy / cudaDeviceSynchronize

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ── Warp-level reductions ─────────────────────────────────────────────────────

__device__ inline float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// ── Kernels ───────────────────────────────────────────────────────────────────

// Block softmax: one block per row, uses shared memory tree-reduction.
// Works for any COLS; blockDim.x should be a power-of-2 <= 1024.
__global__ void softmax_block(float* __restrict__ out,
                              const float* __restrict__ in,
                              int cols) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_in = in + (long)row * cols;
    float*       row_out = out + (long)row * cols;

    // Pass 1 — row maximum
    float local_max = -3.402823466e+38f;
    for (int i = tid; i < cols; i += blockDim.x)
        local_max = fmaxf(local_max, row_in[i]);
    sdata[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float row_max = sdata[0];
    __syncthreads();

    // Pass 2 — exp + local sum
    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float e = expf(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float row_sum = sdata[0];
    __syncthreads();

    // Pass 3 — normalize
    for (int i = tid; i < cols; i += blockDim.x)
        row_out[i] /= row_sum;
}

// Warp softmax: one warp per row, cols <= 32.
__global__ void softmax_warp(float* __restrict__ out,
                             const float* __restrict__ in,
                             int cols) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    const float* row_in = in + (long)row * cols;
    float*       row_out = out + (long)row * cols;

    float val = (tid < cols) ? row_in[tid] : -3.402823466e+38f;
    float row_max = warp_reduce_max(val);
    float e = (tid < cols) ? expf(val - row_max) : 0.0f;
    float row_sum = warp_reduce_sum(e);
    if (tid < cols)
        row_out[tid] = e / row_sum;
}

// ── Reference CPU softmax ──────────────────────────────────────────────────────

static void softmax_cpu(float* out, const float* in, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float* ri = in + (long)r * cols;
        float*       ro = out + (long)r * cols;
        float mx = ri[0];
        for (int i = 1; i < cols; i++) mx = fmaxf(mx, ri[i]);
        float s = 0.0f;
        for (int i = 0; i < cols; i++) { ro[i] = expf(ri[i] - mx); s += ro[i]; }
        for (int i = 0; i < cols; i++) ro[i] /= s;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

static int check(const float* got, const float* ref, int n, float tol,
                 const char* tag) {
    int errs = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(got[i] - ref[i]);
        if (d > tol) {
            if (errs < 4)
                printf("  %s[%d]: got %.6f ref %.6f diff %.2e\n",
                       tag, i, got[i], ref[i], d);
            errs++;
        }
    }
    return errs;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(void) {
    const float TOL = 1e-5f;
    int total_errs = 0;

    // ── Test 1: block softmax (128 cols, 32 rows) ────────────────────────────
    {
        const int ROWS = 32, COLS = 128;
        size_t sz = (size_t)ROWS * COLS * sizeof(float);
        float* h_in  = (float*)malloc(sz);
        float* h_out = (float*)malloc(sz);
        float* h_ref = (float*)malloc(sz);

        for (int i = 0; i < ROWS * COLS; i++)
            h_in[i] = (float)(i % COLS) * 0.1f - 6.4f;
        softmax_cpu(h_ref, h_in, ROWS, COLS);

        float *d_in, *d_out;
        cudaMalloc(&d_in,  sz);
        cudaMalloc(&d_out, sz);
        cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

        int block = 128;
        softmax_block<<<ROWS, block, block * sizeof(float)>>>(d_out, d_in, COLS);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);

        int errs = check(h_out, h_ref, ROWS * COLS, TOL, "block");
        if (errs) {
            printf("FAIL: block softmax: %d errors\n", errs);
            total_errs += errs;
        } else {
            printf("PASS: block softmax (rows=%d cols=%d)\n", ROWS, COLS);
        }

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_out); free(h_ref);
    }

    // ── Test 2: warp softmax (32 cols, 16 rows) ──────────────────────────────
    {
        const int ROWS = 16, COLS = 32;
        size_t sz = (size_t)ROWS * COLS * sizeof(float);
        float* h_in  = (float*)malloc(sz);
        float* h_out = (float*)malloc(sz);
        float* h_ref = (float*)malloc(sz);

        for (int i = 0; i < ROWS * COLS; i++)
            h_in[i] = (float)(i % COLS) * 0.5f - 8.0f;
        softmax_cpu(h_ref, h_in, ROWS, COLS);

        float *d_in, *d_out;
        cudaMalloc(&d_in,  sz);
        cudaMalloc(&d_out, sz);
        cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

        softmax_warp<<<ROWS, 32>>>(d_out, d_in, COLS);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);

        int errs = check(h_out, h_ref, ROWS * COLS, TOL, "warp");
        if (errs) {
            printf("FAIL: warp softmax: %d errors\n", errs);
            total_errs += errs;
        } else {
            printf("PASS: warp softmax (rows=%d cols=%d)\n", ROWS, COLS);
        }

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_out); free(h_ref);
    }

    // ── Test 3: block softmax with wide rows (vocab-like: 512 cols, 8 rows) ──
    {
        const int ROWS = 8, COLS = 512;
        size_t sz = (size_t)ROWS * COLS * sizeof(float);
        float* h_in  = (float*)malloc(sz);
        float* h_out = (float*)malloc(sz);
        float* h_ref = (float*)malloc(sz);

        srand(42);
        for (int i = 0; i < ROWS * COLS; i++)
            h_in[i] = ((float)rand() / (float)RAND_MAX) * 4.0f - 2.0f;
        softmax_cpu(h_ref, h_in, ROWS, COLS);

        float *d_in, *d_out;
        cudaMalloc(&d_in,  sz);
        cudaMalloc(&d_out, sz);
        cudaMemcpy(d_in, h_in, sz, cudaMemcpyHostToDevice);

        int block = 256;
        softmax_block<<<ROWS, block, block * sizeof(float)>>>(d_out, d_in, COLS);
        cudaDeviceSynchronize();
        cudaMemcpy(h_out, d_out, sz, cudaMemcpyDeviceToHost);

        int errs = check(h_out, h_ref, ROWS * COLS, TOL, "block-wide");
        if (errs) {
            printf("FAIL: block softmax (wide): %d errors\n", errs);
            total_errs += errs;
        } else {
            printf("PASS: block softmax wide (rows=%d cols=%d)\n", ROWS, COLS);
        }

        cudaFree(d_in); cudaFree(d_out);
        free(h_in); free(h_out); free(h_ref);
    }

    if (total_errs == 0) {
        printf("ALL PASS: softmax_cuda\n");
        return 0;
    } else {
        printf("FAIL: %d total errors\n", total_errs);
        return 1;
    }
}
