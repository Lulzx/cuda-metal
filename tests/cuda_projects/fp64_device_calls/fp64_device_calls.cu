// Doubles crossing device-call boundaries: parameters, returns, pointers to
// double, a struct of doubles by value, and the sign-bit builtins. Software
// FP64 keeps binary64 storage and only routes arithmetic through the mode's
// ALU, so a user function with double in its signature must simply take the
// ordinary call path. This used to be refused as an "unsupported software FP64
// call" for every such function.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

struct Pair {
    double lo;
    double hi;
};

__device__ __noinline__ double load_scaled(const double* p, int i, double scale) {
    return p[i] * scale;
}

__device__ __noinline__ void store_sum(double* out, int i, double a, double b) {
    out[i] = a + b;
}

__device__ __noinline__ Pair widen(double x) {
    Pair r;
    r.lo = x - 1.0;
    r.hi = x + 1.0;
    return r;
}

__device__ __noinline__ double span(Pair p) {
    return p.hi - p.lo;
}

__global__ void fp64_device_calls_probe(double* output, const double* input, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double x = load_scaled(input, index, 2.0);
    const Pair p = widen(x);
    double sum;
    store_sum(&sum, 0, span(p), fabs(-x));
    output[index] = copysign(sum, input[index]) + (signbit(input[index]) ? 100.0 : 0.0);
}

int main() {
    constexpr int kCount = 8;
    double host_input[kCount];
    double expected[kCount];
    for (int i = 0; i < kCount; ++i) {
        host_input[i] = (i % 3 == 1) ? -1.5 * i : 0.25 * i + 0.5;
        const double x = host_input[i] * 2.0;
        const double sum = 2.0 + std::fabs(-x);
        expected[i] = std::copysign(sum, host_input[i]) + (std::signbit(host_input[i]) ? 100.0 : 0.0);
    }
    double* device_input = nullptr;
    double* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }
    fp64_device_calls_probe<<<1, 32>>>(device_output, device_input, kCount);
    double host_output[kCount];
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }
    for (int i = 0; i < kCount; ++i) {
        // fast48 keeps ~48 significand bits; these values are exact in far fewer.
        if (std::fabs(host_output[i] - expected[i]) > 1e-9 * std::fabs(expected[i]) + 1e-12) {
            std::fprintf(stderr, "FAIL: output[%d] = %.17g, expected %.17g\n", i, host_output[i],
                         expected[i]);
            return 1;
        }
    }
    std::printf("PASS: doubles cross device-call boundaries as binary64 storage\n");
    return 0;
}
