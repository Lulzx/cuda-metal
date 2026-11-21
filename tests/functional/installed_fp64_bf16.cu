// Compiled by an installed cumetalc. Needs headers the install once omitted
// (cuda_bf16.h, math_constants.h, vector_types.h) and the software-FP64
// Metal support sources (issue #204).
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_types.h>

#include <cstdio>

__global__ void kernel(const double* in, double* out, unsigned short* bf, float2* pair) {
    const unsigned i = threadIdx.x;
    out[i] = in[i] / 3.0 + CUDART_PI;
    bf[i] = __bfloat16_as_ushort(__float2bfloat16_rz(static_cast<float>(in[i])));
    pair[i] = make_float2(static_cast<float>(i), -static_cast<float>(i));
}

int main() {
    constexpr int n = 32;
    double host_in[n];
    for (int i = 0; i < n; ++i) host_in[i] = 1.0 + i * 0.25;
    double* in = nullptr;
    double* out = nullptr;
    unsigned short* bf = nullptr;
    float2* pair = nullptr;
    if (cudaMalloc(&in, sizeof(host_in)) != cudaSuccess ||
        cudaMalloc(&out, sizeof(host_in)) != cudaSuccess ||
        cudaMalloc(&bf, n * sizeof(unsigned short)) != cudaSuccess ||
        cudaMalloc(&pair, n * sizeof(float2)) != cudaSuccess ||
        cudaMemcpy(in, host_in, sizeof(host_in), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("FAIL: setup\n");
        return 1;
    }
    kernel<<<1, n>>>(in, out, bf, pair);
    double host_out[n];
    unsigned short host_bf[n];
    float2 host_pair[n];
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(host_out, out, sizeof(host_out), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_bf, bf, sizeof(host_bf), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_pair, pair, sizeof(host_pair), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: launch\n");
        return 1;
    }
    for (int i = 0; i < n; ++i) {
        const double expected = host_in[i] / 3.0 + CUDART_PI;
        const unsigned short expected_bf = __bfloat16_as_ushort(
            __float2bfloat16_rz(static_cast<float>(host_in[i])));
        if (host_out[i] != expected || host_bf[i] != expected_bf ||
            host_pair[i].x != static_cast<float>(i) ||
            host_pair[i].y != -static_cast<float>(i)) {
            std::printf("FAIL: element %d out=%a want=%a bf=%04x want=%04x\n", i,
                        host_out[i], expected, host_bf[i], expected_bf);
            return 1;
        }
    }
    std::printf("PASS\n");
    return 0;
}
