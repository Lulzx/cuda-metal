// #223 (tensorcore evaluation): a pure copy kernel out[i] = a[i] must run, and
// __float2half / __half2float must be callable on the host as in CUDA.
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__global__ void copy(const float* a, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i];
}

__global__ void to_half(const float* in, __half* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

int main() {
    constexpr int n = 1000;
    static float h[n], r[n];
    static __half device_half[n];
    for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i) * 0.37f - 100.0f;
    float *a = nullptr, *o = nullptr;
    __half* hv = nullptr;
    if (cudaMalloc(&a, sizeof h) != cudaSuccess || cudaMalloc(&o, sizeof h) != cudaSuccess ||
        cudaMalloc(&hv, sizeof device_half) != cudaSuccess) {
        std::printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(a, h, sizeof h, cudaMemcpyHostToDevice);
    copy<<<(n + 255) / 256, 256>>>(a, o, n);
    if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: copy kernel launch\n");
        return 1;
    }
    cudaMemcpy(r, o, sizeof r, cudaMemcpyDeviceToHost);
    if (std::memcmp(r, h, sizeof h) != 0) {
        std::printf("FAIL: copy kernel output differs\n");
        return 1;
    }

    // Host and device conversions agree bit for bit.
    to_half<<<(n + 255) / 256, 256>>>(a, hv, n);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: to_half kernel\n");
        return 1;
    }
    cudaMemcpy(device_half, hv, sizeof device_half, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        const __half host_half = __float2half(h[i]);
        if (std::memcmp(&host_half, &device_half[i], sizeof(__half)) != 0) {
            std::printf("FAIL: host __float2half differs from device at %d\n", i);
            return 1;
        }
        const float back = __half2float(host_half);
        if (back != __half2float(device_half[i])) {
            std::printf("FAIL: host __half2float differs at %d\n", i);
            return 1;
        }
    }
    if (__half2float(__float2half(1.5f)) != 1.5f) {
        std::printf("FAIL: host half round trip\n");
        return 1;
    }
    std::printf("PASS: copy kernel and host half conversions\n");
    return 0;
}
