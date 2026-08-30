#include <cuda_runtime.h>

#include <cstdio>

struct Record {
    int base;
    float scale;
    unsigned bias;
};

__device__ __noinline__ Record make_record(int base, float scale,
                                           unsigned bias) {
    return Record{base, scale, bias};
}

__device__ __noinline__ int consume_record(Record record, int value) {
    return record.base + static_cast<int>(record.scale * value) +
           static_cast<int>(record.bias);
}

__global__ void aggregate_device_call_probe(int* output) {
    if (threadIdx.x == 0) {
        const Record first = make_record(3, 2.5f, 4);
        const Record second = make_record(7, 1.5f, 2);
        output[0] = consume_record(first, 6);
        output[1] = consume_record(second, 4);
    }
}

int main() {
    constexpr int expected[2] = {22, 15};
    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    aggregate_device_call_probe<<<1, 32>>>(device_output);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    int actual[2] = {};
    const cudaError_t copy_status =
        cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(sync_status),
                     cudaGetErrorString(copy_status));
        return 1;
    }

    for (int index = 0; index < 2; ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "FAIL: output[%d]=%d, expected %d\n", index,
                         actual[index], expected[index]);
            return 1;
        }
    }
    std::printf(
        "PASS: aggregate device arguments and returns preserve all fields\n");
    return 0;
}
