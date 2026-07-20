#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" __global__ void vector_add(const float* a,
                                      const float* b,
                                      float* out,
                                      int count) {
    const int index = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (index < count) {
        out[index] = a[index] + b[index];
    }
}

int main() {
    constexpr int kCount = 8192;
    constexpr int kThreads = 256;
    const std::size_t bytes = static_cast<std::size_t>(kCount) * sizeof(float);

    std::vector<float> host_a(kCount);
    std::vector<float> host_b(kCount);
    std::vector<float> host_out(kCount, 0.0f);
    for (int i = 0; i < kCount; ++i) {
        host_a[i] = static_cast<float>(i) * 0.25f;
        host_b[i] = static_cast<float>(i % 31) * 1.5f;
    }

    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_out = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_a), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_b), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_out), bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        return 1;
    }

    if (cudaMemcpy(device_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: host-to-device copy\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate\n");
        return 1;
    }

    vector_add<<<(kCount + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        device_a, device_b, device_out, kCount);
    if (cudaGetLastError() != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: CUDA kernel launch\n");
        return 1;
    }
    cudaStreamDestroy(stream);

    if (cudaMemcpy(host_out.data(), device_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device-to-host copy\n");
        return 1;
    }

    for (int i = 0; i < kCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (std::fabs(host_out[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_out);
    std::printf("PASS: CUDA source vector_add produced correct GPU output\n");
    return 0;
}
