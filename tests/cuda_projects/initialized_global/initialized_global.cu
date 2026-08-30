#include <cuda_runtime.h>

#include <cstdio>

__device__ int mutable_state[4] = {7, -3, 11, 5};

__global__ void initialized_global_probe(int* output) {
    output[0] = mutable_state[0];
    output[1] = mutable_state[1];
    output[2] = mutable_state[2];
    output[3] = mutable_state[3];
    mutable_state[0] += 10;
    mutable_state[3] *= 2;
}

bool check_values(const char* label, const int* actual, const int* expected) {
    for (int index = 0; index < 4; ++index) {
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
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), 4 * sizeof(int)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    const int expected_first[4] = {7, -3, 11, 5};
    const int expected_second[4] = {17, -3, 11, 10};
    const int expected_symbol[4] = {27, -3, 11, 20};
    int actual[4] = {};

    initialized_global_probe<<<1, 1>>>(device_output);
    cudaError_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(actual, device_output, sizeof(actual),
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess ||
        !check_values("first launch", actual, expected_first)) {
        std::fprintf(stderr, "FAIL: first launch/copy: %s\n",
                     cudaGetErrorString(status));
        cudaFree(device_output);
        return 1;
    }

    initialized_global_probe<<<1, 1>>>(device_output);
    status = cudaDeviceSynchronize();
    if (status == cudaSuccess) {
        status = cudaMemcpy(actual, device_output, sizeof(actual),
                            cudaMemcpyDeviceToHost);
    }
    if (status != cudaSuccess ||
        !check_values("second launch", actual, expected_second)) {
        std::fprintf(stderr, "FAIL: second launch/copy: %s\n",
                     cudaGetErrorString(status));
        cudaFree(device_output);
        return 1;
    }

    status = cudaMemcpyFromSymbol(actual, mutable_state, sizeof(actual), 0,
                                  cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    if (status != cudaSuccess ||
        !check_values("symbol copy", actual, expected_symbol)) {
        std::fprintf(stderr, "FAIL: symbol copy: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }

    std::puts("PASS: initialized writable global persists across launches");
    return 0;
}
