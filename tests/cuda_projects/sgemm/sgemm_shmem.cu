// sgemm_shmem.cu — SGEMM_CUDA kernel 3: shared memory blocking (verbatim)
// siboehm/SGEMM_CUDA, src/kernels/3_kernel_shared_mem_blocking.cuh

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C) {
    const int cRow = blockIdx.x;
    const int cCol = blockIdx.y;

    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    const int threadCol = threadIdx.x % BLOCKSIZE;
    const int threadRow = threadIdx.x / BLOCKSIZE;

    A += cRow * BLOCKSIZE * K;
    B += cCol * BLOCKSIZE;
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;

    float tmp = 0.0f;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

static void sgemm_cpu(int M, int N, int K, float alpha,
                      const float *A, const float *B, float beta, float *C) {
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float acc = 0;
            for (int k = 0; k < K; k++) acc += A[m*K+k] * B[k*N+n];
            C[m*N+n] = alpha * acc + beta * C[m*N+n];
        }
}

int main(void) {
    const int M = 128, N = 128, K = 128;
    const float alpha = 1.0f, beta = 0.0f;
    const float TOL = 1e-3f;
    const int BS = 32;

    size_t sa = M*K*sizeof(float), sb = K*N*sizeof(float), sc = M*N*sizeof(float);
    float *h_A=(float*)malloc(sa), *h_B=(float*)malloc(sb);
    float *h_C=(float*)calloc(M*N,sizeof(float)), *h_ref=(float*)calloc(M*N,sizeof(float));

    srand(42);
    for (int i=0;i<M*K;i++) h_A[i]=((float)rand()/(float)0x7fffffff)*2.f-1.f;
    for (int i=0;i<K*N;i++) h_B[i]=((float)rand()/(float)0x7fffffff)*2.f-1.f;
    sgemm_cpu(M,N,K,alpha,h_A,h_B,beta,h_ref);

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,sa); cudaMalloc(&d_B,sb); cudaMalloc(&d_C,sc);
    cudaMemcpy(d_A,h_A,sa,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sb,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,sc,cudaMemcpyHostToDevice);

    dim3 grid(CEIL_DIV(M,BS), CEIL_DIV(N,BS));
    sgemm_shared_mem_block<BS><<<grid, BS*BS>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,sc,cudaMemcpyDeviceToHost);

    int errs=0;
    for (int i=0;i<M*N;i++) {
        float d=fabsf(h_C[i]-h_ref[i]);
        if (d>TOL) { if(errs<4) printf("  [%d]: got %.4f ref %.4f\n",i,h_C[i],h_ref[i]); errs++; }
    }
    if (errs) { printf("FAIL: sgemm_shmem: %d errors\n",errs); return 1; }
    printf("PASS: sgemm_shared_mem_block<%d> (M=%d N=%d K=%d)\n",BS,M,N,K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
