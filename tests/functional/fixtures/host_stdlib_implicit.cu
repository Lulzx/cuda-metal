// Regression: many real CUDA samples only #include <stdio.h> and still call
// malloc/free/exit (and fabs/ceil via the CUDA math overlay). nvcc's headers
// make those visible; cumetalc must too via the force-included cuda_runtime.h.
//
// Do not add <stdlib.h> or <math.h> here — that would defeat the test.

#include <stdio.h>

#define N 4096

__global__ void add_vectors(float* a, float* b, float* c) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C = (float*)malloc(bytes);
    if (A == NULL || B == NULL || C == NULL) {
        printf("FAIL: host allocation failed\n");
        exit(1);
    }

    float *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_B, bytes) != cudaSuccess ||
        cudaMalloc(&d_C, bytes) != cudaSuccess) {
        printf("FAIL: cudaMalloc failed\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);

    int thr_per_blk = 256;
    int blk_in_grid = (int)ceil((float)N / (float)thr_per_blk);
    add_vectors<<<blk_in_grid, thr_per_blk>>>(d_A, d_B, d_C);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: kernel launch/sync failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(1);
    }

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    float tolerance = 1.0e-5f;
    for (int i = 0; i < N; i++) {
        if (fabs(C[i] - 3.0f) > tolerance) {
            printf("FAIL: C[%d] = %f instead of 3.0\n", i, C[i]);
            exit(1);
        }
    }

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("PASS: host stdlib symbols resolved without explicit includes\n");
    return 0;
}
