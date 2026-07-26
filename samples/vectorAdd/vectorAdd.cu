// Ordinary CUDA source: host code, device code, and a <<<>>> launch in one file.
// Nothing here is CuMetal-specific -- this is what nvcc would compile.
//
//   cumetalc vectorAdd.cu -o vectorAdd && ./vectorAdd

#include <cstdio>
#include <cstdlib>
#include <cmath>

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1 << 14;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* host_a = static_cast<float*>(std::malloc(bytes));
    float* host_b = static_cast<float*>(std::malloc(bytes));
    float* host_c = static_cast<float*>(std::malloc(bytes));
    if (host_a == nullptr || host_b == nullptr || host_c == nullptr) {
        std::fprintf(stderr, "FAIL: host allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        host_a[i] = static_cast<float>(i) * 0.25f;
        host_b[i] = static_cast<float>(i % 17) * 1.5f;
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess ||
        cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a, bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;
    vector_add<<<blocks, threads_per_block>>>(dev_a, dev_b, dev_c, n);

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c, dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (std::fabs(host_c[i] - expected) > 1e-5f) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    std::free(host_a);
    std::free(host_b);
    std::free(host_c);

    std::printf("PASS: samples/vectorAdd produced correct output for %d elements\n", n);
    return 0;
}
