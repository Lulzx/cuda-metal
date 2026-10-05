// One kernel per argument so the lazy registration JIT compiles only the one
// under test. `exp` calls double libdevice exp (binary32 under every FP64
// mode); `arith` is pure FP64 arithmetic, which follows the selected mode.
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

__global__ void k_exp(double* out, const double* in) { out[threadIdx.x] = exp(in[threadIdx.x]); }
__global__ void k_arith(double* out, const double* in) {
    out[threadIdx.x] = in[threadIdx.x] * 3.0 + 0.5;
}

int main(int argc, char** argv) {
    const bool use_exp = argc > 1 && strcmp(argv[1], "exp") == 0;
    double h[32];
    for (int i = 0; i < 32; ++i) h[i] = 0.25 * i;
    double *d_in, *d_out;
    cudaMalloc(&d_in, sizeof h);
    cudaMalloc(&d_out, sizeof h);
    cudaMemcpy(d_in, h, sizeof h, cudaMemcpyHostToDevice);
    if (use_exp) k_exp<<<1, 32>>>(d_out, d_in);
    else k_arith<<<1, 32>>>(d_out, d_in);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("FAIL: launch failed\n");
        return 1;
    }
    cudaMemcpy(h, d_out, sizeof h, cudaMemcpyDeviceToHost);
    printf("RAN %s %.6f\n", use_exp ? "exp" : "arith", h[4]);
    return 0;
}
