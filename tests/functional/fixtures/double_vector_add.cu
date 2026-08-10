// Regression: source-path double kernels must launch on Apple GPU via FP64
// emulation (native AIR double fails Metal pipeline creation).
// Host memory is IEEE binary64; results for f32-exact values must be correct.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024

__global__ void add_vectors(double* a, double* b, double* c) {
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if (id < N) {
        c[id] = a[id] + b[id];
    }
}

int main() {
    size_t bytes = N * sizeof(double);
    double* A = (double*)malloc(bytes);
    double* B = (double*)malloc(bytes);
    double* C = (double*)malloc(bytes);
    if (A == NULL || B == NULL || C == NULL) {
        printf("FAIL: host allocation failed\n");
        return 1;
    }

    double *d_A, *d_B, *d_C;
    if (cudaMalloc(&d_A, bytes) != cudaSuccess ||
        cudaMalloc(&d_B, bytes) != cudaSuccess ||
        cudaMalloc(&d_C, bytes) != cudaSuccess) {
        printf("FAIL: cudaMalloc failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = -1.0;
    }

    cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice);
    add_vectors<<<4, 256>>>(d_A, d_B, d_C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("FAIL: launch error %d %s\n", (int)err, cudaGetErrorString(err));
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: sync error\n");
        return 1;
    }

    cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        if (fabs(C[i] - 3.0) > 1e-5) {
            printf("FAIL: C[%d] = %f instead of 3.0\n", i, C[i]);
            return 1;
        }
    }

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("PASS: double vector add launched and produced correct results\n");
    return 0;
}
