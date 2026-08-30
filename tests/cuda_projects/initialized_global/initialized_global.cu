#include <cuda_runtime.h>

#include <cstdio>

__device__ int mutable_state[4] = {7, -3, 11, 5};
static __device__ int private_state[2] = {13, -2};

__global__ void initialized_global_probe(int* output) {
    output[0] = mutable_state[0];
    output[1] = mutable_state[1];
    output[2] = mutable_state[2];
    output[3] = mutable_state[3];
    output[4] = private_state[0];
    output[5] = private_state[1];
    mutable_state[0] += 10;
    mutable_state[3] *= 2;
    private_state[0] += 4;
    private_state[1] -= 3;
}

__global__ void mutate_private_global_from_second_kernel() {
    private_state[0] += 100;
}

bool check_values(const char* label, const int* actual, const int* expected,
                  int count) {
    for (int index = 0; index < count; ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr,
                         "FAIL: %s[%d] got %d expected %d\n",
                         label, index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}

int main() {
    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), 6 * sizeof(int)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    const int expected_first[6] = {7, -3, 11, 5, 13, -2};
    const int expected_second[6] = {17, -3, 11, 10, 117, -5};
    const int expected_symbol[4] = {27, -3, 11, 20};
    int actual[6] = {};

    initialized_global_probe<<<1, 1>>>(device_output);
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(actual, device_output, sizeof(actual),
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess ||
        !check_values("first launch", actual, expected_first, 6)) {
        std::fprintf(stderr, "FAIL: first launch/copy: %s\n",
                     cudaGetErrorString(status));
        cudaFree(device_output);
        return 1;
    }

    mutate_private_global_from_second_kernel<<<1, 1>>>();
    initialized_global_probe<<<1, 1>>>(device_output);
    status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(actual, device_output, sizeof(actual),
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess ||
        !check_values("second launch", actual, expected_second, 6)) {
        std::fprintf(stderr, "FAIL: second launch/copy: %s\n",
                     cudaGetErrorString(status));
        cudaFree(device_output);
        return 1;
    }

    status = cudaMemcpyFromSymbol(actual, mutable_state, sizeof(expected_symbol), 0,
                                  cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    if (status != cudaSuccess ||
        !check_values("symbol copy", actual, expected_symbol, 4)) {
        std::fprintf(stderr, "FAIL: symbol copy: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }

    std::puts("PASS: visible and private initialized writable globals persist");
    return 0;
}
