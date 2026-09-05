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

// PTX keeps float temporaries in .b32 registers, so a float-typed instruction
// reads and writes integer-typed operands. `selp.f32` used to assign that bit
// pattern straight into a float result, and a scalar float return used to be
// written into its integer container the same way -- both numeric conversions
// where a reinterpretation was meant. The first returned 1082130432.0 in place
// of 4.0f; the second was invisible only because the round trip happened to
// cancel. Neither is reachable without a device call that returns a float
// chosen by a comparison.
__device__ __noinline__ float larger(float a, float b) { return a > b ? a : b; }

struct Bounds {
    float high;
    float sum;
};

__device__ __noinline__ Bounds bound(float a, float b) {
    Bounds result;
    result.high = a > b ? a : b;
    result.sum = a + b;
    return result;
}

__global__ void float_select_return_probe(float* output) {
    if (threadIdx.x == 0) {
        output[0] = larger(1.0f, 4.0f);
        const Bounds bounds = bound(1.0f, 4.0f);
        output[1] = bounds.high;
        output[2] = bounds.sum;
    }
}

__global__ void aggregate_device_call_probe(int* output) {
    if (threadIdx.x == 0) {
        const Record first = make_record(3, 2.5f, 4);
        const Record second = make_record(7, 1.5f, 2);
        output[0] = consume_record(first, 6);
        output[1] = consume_record(second, 4);
        output[2] = consume_record(Record{5, 3.0f, 1}, 4);
    }
}

int main() {
    constexpr int expected[3] = {22, 15, 18};
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

    float* float_output = nullptr;
    if (cudaMallocManaged(reinterpret_cast<void**>(&float_output),
                          3 * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }
    float_output[0] = float_output[1] = float_output[2] = -1.0f;
    float_select_return_probe<<<1, 32>>>(float_output);
    if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: float probe launch: %s\n",
                     cudaGetErrorString(status));
        return 1;
    }
    constexpr float float_expected[3] = {4.0f, 4.0f, 5.0f};
    for (int index = 0; index < 3; ++index) {
        if (float_output[index] != float_expected[index]) {
            std::fprintf(stderr, "FAIL: float output[%d]=%.1f, expected %.1f\n", index,
                         float_output[index], float_expected[index]);
            return 1;
        }
    }
    cudaFree(float_output);

    std::printf(
        "PASS: aggregate device arguments, returns, promoted literals, and float selects "
        "preserve all fields\n");
    return 0;
}
