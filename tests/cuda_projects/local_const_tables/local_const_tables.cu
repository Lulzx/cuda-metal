// Read-only tables Clang synthesises as private LLVM globals in address space
// 0: a function-local `const T tbl[] = {...}` (`__const.<fn>.<var>`), a
// brace-initialised struct array, a string literal indexed by device code, and
// a zero-initialised static table. Only the `static const` in addrspace(4)
// used to reach the MSL; the rest were referenced without a declaration and
// failed in Apple's compiler as undeclared identifiers.
#include <cuda_runtime.h>

#include <cstdio>

struct Triple {
    float a;
    float b;
    float c;
};

__device__ __noinline__ float polynomial(float x) {
    static const float coefficients[4] = {1.0f, -0.5f, 0.25f, -0.125f};
    float sum = 0.0f;
    for (int index = 0; index < 4; ++index) {
        sum = sum * x + coefficients[index];
    }
    return sum;
}

__device__ __noinline__ int lookup(int index) {
    const int table[6] = {3, 1, 4, 1, 5, 9};
    return table[index % 6];
}

__device__ __noinline__ float pick(int index) {
    const Triple triples[2] = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    return triples[index & 1].b;
}

__device__ __noinline__ int letter(int index) {
    const char* text = "hello";
    return text[index % 5];
}

__device__ __noinline__ int zeros(int index) {
    static const int table[8] = {0};
    return table[index & 7];
}

__global__ void local_const_tables_probe(float* output, const float* input, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = polynomial(input[index]) + static_cast<float>(lookup(index)) +
                        pick(index) + static_cast<float>(letter(index)) +
                        static_cast<float>(zeros(index));
    }
}

int main() {
    constexpr int kCount = 12;
    float host_input[kCount];
    float expected[kCount];
    const float coefficients[4] = {1.0f, -0.5f, 0.25f, -0.125f};
    const int table[6] = {3, 1, 4, 1, 5, 9};
    const float picks[2] = {2.0f, 5.0f};
    const char* text = "hello";
    for (int index = 0; index < kCount; ++index) {
        host_input[index] = 0.25f * static_cast<float>(index) - 1.0f;
        float sum = 0.0f;
        for (int k = 0; k < 4; ++k) sum = sum * host_input[index] + coefficients[k];
        expected[index] = sum + static_cast<float>(table[index % 6]) + picks[index & 1] +
                          static_cast<float>(text[index % 5]);
    }

    float* device_input = nullptr;
    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(host_input)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }

    local_const_tables_probe<<<1, 32>>>(device_output, device_input, kCount);

    float host_output[kCount];
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }
    for (int index = 0; index < kCount; ++index) {
        const float difference = host_output[index] - expected[index];
        if (difference > 1e-5f || difference < -1e-5f) {
            std::fprintf(stderr, "FAIL: output[%d] = %f, expected %f\n", index,
                         host_output[index], expected[index]);
            return 1;
        }
    }
    std::printf("PASS: function-local constant tables are embedded and read correctly\n");
    return 0;
}
