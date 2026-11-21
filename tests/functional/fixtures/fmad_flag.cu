// a * b + c with c = -round(a * b): a fused multiply-add returns the
// product's rounding error, two roundings return exactly zero. Built with
// -DCUMETAL_EXPECT_FUSED=1 by default and =0 under --fmad=false.
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#ifndef CUMETAL_EXPECT_FUSED
#error "define CUMETAL_EXPECT_FUSED"
#endif

__global__ void mul_add(const float* a, const float* b, const float* c, float* out, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i] + c[i];
}

int main() {
    const unsigned n = 1024;
    std::vector<float> a(n), b(n), c(n), out(n);
    uint64_t x = 0x1234u;
    for (unsigned i = 0; i < n; ++i) {
        x = 6364136223846793005ull * x + 1442695040888963407ull;
        const uint32_t ua = 0x3f800000u | ((uint32_t)(x >> 32) >> 9);
        x = 6364136223846793005ull * x + 1442695040888963407ull;
        const uint32_t ub = 0x3f800000u | ((uint32_t)(x >> 32) >> 9);
        std::memcpy(&a[i], &ua, 4);
        std::memcpy(&b[i], &ub, 4);
        c[i] = -(a[i] * b[i]);
    }
    float *da, *db, *dc, *dout;
    if (cudaMalloc(&da, n * 4) || cudaMalloc(&db, n * 4) || cudaMalloc(&dc, n * 4) ||
        cudaMalloc(&dout, n * 4))
        return 1;
    cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c.data(), n * 4, cudaMemcpyHostToDevice);
    mul_add<<<(n + 127) / 128, 128>>>(da, db, dc, dout, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) return 1;
    cudaMemcpy(out.data(), dout, n * 4, cudaMemcpyDeviceToHost);
    unsigned nonzero = 0;
    for (unsigned i = 0; i < n; ++i) {
        const float fused = (float)((double)a[i] * (double)b[i] + (double)c[i]);
        const float expected = CUMETAL_EXPECT_FUSED ? fused : 0.0f;
        if (out[i] != expected) {
            std::printf("FAIL: element %u got %a want %a (expect fused=%d)\n", i, out[i], expected,
                        CUMETAL_EXPECT_FUSED);
            return 1;
        }
        nonzero += fused != 0.0f;
    }
    if (nonzero < n / 2) {
        std::printf("FAIL: inputs do not distinguish fused from unfused\n");
        return 1;
    }
    std::printf("PASS: fmad=%s\n", CUMETAL_EXPECT_FUSED ? "true" : "false");
    return 0;
}
