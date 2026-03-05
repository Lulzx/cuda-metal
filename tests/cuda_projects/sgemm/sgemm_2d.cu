// sgemm_2d.cu — SGEMM_CUDA kernel 5: 2D blocktiling with register caches (verbatim)
// siboehm/SGEMM_CUDA, src/kernels/5_kernel_2D_blocktiling.cuh

#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
    const int cRow = blockIdx.y;
    const int cCol = blockIdx.x;
    const int numThreadsBlocktile = (BM * BN) / (TM * TN);
    assert(numThreadsBlocktile == blockDim.x);

    const int threadCol = threadIdx.x % (BN / TN);
    const int threadRow = threadIdx.x / (BN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const int innerRowA = threadIdx.x / BK;
    const int innerColA = threadIdx.x % BK;
    const int strideA   = numThreadsBlocktile / BK;
    const int innerRowB = threadIdx.x / BN;
    const int innerColB = threadIdx.x % BN;
    const int strideB   = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (int lo = 0; lo < BM; lo += strideA)
            As[(innerRowA + lo) * BK + innerColA] = A[(innerRowA + lo) * K + innerColA];
        for (int lo = 0; lo < BK; lo += strideB)
            Bs[(innerRowB + lo) * BN + innerColB] = B[(innerRowB + lo) * N + innerColB];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (int i = 0; i < TM; ++i)
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            for (int i = 0; i < TN; ++i)
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            for (int rm = 0; rm < TM; ++rm)
                for (int rn = 0; rn < TN; ++rn)
                    threadResults[rm * TN + rn] += regM[rm] * regN[rn];
        }
        __syncthreads();
    }
    for (int rm = 0; rm < TM; ++rm)
        for (int rn = 0; rn < TN; ++rn)
            C[(threadRow * TM + rm) * N + threadCol * TN + rn] =
                alpha * threadResults[rm * TN + rn] +
                beta  * C[(threadRow * TM + rm) * N + threadCol * TN + rn];
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
    // BM=64 BN=64 BK=8 TM=8 TN=8 → 64 threads per block
    const int BM=64, BN=64, BK=8, TM=8, TN=8;
    const int M=128, N=128, K=64;
    const float alpha=1.0f, beta=0.0f;
    const float TOL=1e-3f;

    size_t sa=M*K*sizeof(float), sb=K*N*sizeof(float), sc=M*N*sizeof(float);
    float *h_A=(float*)malloc(sa), *h_B=(float*)malloc(sb);
    float *h_C=(float*)calloc(M*N,sizeof(float)), *h_ref=(float*)calloc(M*N,sizeof(float));

    srand(99);
    for (int i=0;i<M*K;i++) h_A[i]=((float)rand()/(float)0x7fffffff)*2.f-1.f;
    for (int i=0;i<K*N;i++) h_B[i]=((float)rand()/(float)0x7fffffff)*2.f-1.f;
    sgemm_cpu(M,N,K,alpha,h_A,h_B,beta,h_ref);

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,sa); cudaMalloc(&d_B,sb); cudaMalloc(&d_C,sc);
    cudaMemcpy(d_A,h_A,sa,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,sb,cudaMemcpyHostToDevice);
    cudaMemcpy(d_C,h_C,sc,cudaMemcpyHostToDevice);

    const int numThreads = (BM*BN)/(TM*TN);
    dim3 grid(CEIL_DIV(N,BN), CEIL_DIV(M,BM));
    sgemm2DBlocktiling<BM,BN,BK,TM,TN><<<grid, numThreads>>>(M,N,K,alpha,d_A,d_B,beta,d_C);
    cudaDeviceSynchronize();
    cudaMemcpy(h_C,d_C,sc,cudaMemcpyDeviceToHost);

    int errs=0;
    for (int i=0;i<M*N;i++) {
        float d=fabsf(h_C[i]-h_ref[i]);
        if (d>TOL) { if(errs<4) printf("  [%d]: got %.4f ref %.4f\n",i,h_C[i],h_ref[i]); errs++; }
    }
    if (errs) { printf("FAIL: sgemm_2d: %d errors\n",errs); return 1; }
    printf("PASS: sgemm2DBlocktiling<BM=%d,BN=%d,BK=%d,TM=%d,TN=%d> (M=%d N=%d K=%d)\n",
           BM,BN,BK,TM,TN,M,N,K);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_ref);
    return 0;
}
