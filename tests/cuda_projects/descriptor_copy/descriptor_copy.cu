// A host-populated descriptor -- a struct holding a device pointer -- reaches
// the same accessor both as the kernel's own by-value parameter (device
// resident) and as a private copy the kernel makes to pass by value to a
// helper. The pointer field is never written by device code, so nothing in
// the module says which address space it holds; only the host's launch
// arguments do. This is the shape of every Warp array_t, cuDNN-style tensor
// descriptor, and SoA bundle, and it used to be refused with "unresolved
// generic pointer address space" the moment a private copy existed.
#include <cuda_runtime.h>

#include <cstdio>

struct Descriptor {
    int* data;
    int count;
};

__device__ __noinline__ int& element(const Descriptor& descriptor, int index) {
    return descriptor.data[index];
}

__device__ __noinline__ void store(const Descriptor& descriptor, int index, int value) {
    element(descriptor, index) = value;
}

// By value: the caller copies the descriptor into private memory first.
__device__ __noinline__ int weighted_sum(Descriptor descriptor) {
    int sum = 0;
    for (int index = 0; index < descriptor.count; ++index) {
        sum += element(descriptor, index) * (index + 1);
    }
    return sum;
}

__global__ void descriptor_copy_probe(Descriptor input, Descriptor output) {
    if (threadIdx.x == 0) {
        store(output, 0, weighted_sum(input));
        // The private copy is also written through, then read back through
        // the device-resident parameter: both views alias the same buffer.
        Descriptor copy = output;
        store(copy, 1, element(input, 2) + element(copy, 0));
        output.data[2] = element(output, 1) - element(input, 0);
    }
}

int main() {
    const int host_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    // sum_i (i+1)*input[i] = 1+4+9+16+25+36+49+64 = 204
    const int expected[3] = {204, 3 + 204, 3 + 204 - 1};

    int* device_input = nullptr;
    int* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_input), sizeof(host_input)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_output), 3 * sizeof(int)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input, sizeof(host_input), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemset(device_output, 0, 3 * sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: allocation or upload\n");
        return 1;
    }

    const Descriptor input{device_input, 8};
    const Descriptor output{device_output, 3};
    descriptor_copy_probe<<<1, 32>>>(input, output);

    int host_output[3] = {0, 0, 0};
    const cudaError_t launch_status = cudaGetLastError();
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (launch_status != cudaSuccess || sync_status != cudaSuccess ||
        cudaMemcpy(host_output, device_output, sizeof(host_output), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch or download: %s / %s\n",
                     cudaGetErrorString(launch_status), cudaGetErrorString(sync_status));
        return 1;
    }

    for (int index = 0; index < 3; ++index) {
        if (host_output[index] != expected[index]) {
            std::fprintf(stderr, "FAIL: output[%d] = %d, expected %d\n", index,
                         host_output[index], expected[index]);
            return 1;
        }
    }
    std::printf("PASS: host-populated descriptor fields resolve through private copies\n");
    return 0;
}
