#include <cuda_runtime.h>

#include <cstdio>

struct InnerRecord {
    int base;
    float scale;
};

struct OuterRecord {
    InnerRecord inner;
    unsigned bias;
};

__device__ __noinline__ OuterRecord make_record(int base, float scale,
                                                unsigned bias) {
    return OuterRecord{InnerRecord{base, scale}, bias};
}

__device__ __noinline__ int consume_record(OuterRecord record, int value) {
    return record.inner.base +
           static_cast<int>(record.inner.scale * value) +
           static_cast<int>(record.bias);
}

__device__ __noinline__ OuterRecord replace_base(OuterRecord record,
                                                  int base) {
    record.inner.base = base;
    return record;
}

__global__ void nested_aggregate_device_call_probe(int* output) {
    if (threadIdx.x == 0) {
        const OuterRecord first = make_record(3, 2.5f, 4);
        const OuterRecord second = replace_base(first, 7);
        output[0] = consume_record(first, 6);
        output[1] = consume_record(second, 4);
        output[2] = consume_record(
            OuterRecord{InnerRecord{5, 3.0f}, 1}, 4);
    }
}

int main() {
    constexpr int expected[3] = {22, 21, 18};
    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), sizeof(expected)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    nested_aggregate_device_call_probe<<<1, 32>>>(device_output);
    const cudaError_t sync_status = cudaDeviceSynchronize();
    int actual[3] = {};
    const cudaError_t copy_status =
        cudaMemcpy(actual, device_output, sizeof(actual), cudaMemcpyDeviceToHost);
    cudaFree(device_output);
    if (sync_status != cudaSuccess || copy_status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(sync_status),
                     cudaGetErrorString(copy_status));
        return 1;
    }

    for (int index = 0; index < 3; ++index) {
        if (actual[index] != expected[index]) {
            std::fprintf(stderr, "FAIL: output[%d]=%d, expected %d\n", index,
                         actual[index], expected[index]);
            return 1;
        }
    }
    std::printf(
        "PASS: nested aggregate device arguments, returns, updates, and literals preserve all fields\n");
    return 0;
}
