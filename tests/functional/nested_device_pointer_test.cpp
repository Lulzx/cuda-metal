#include "cuda.h"

#include <cmath>
#include <cstdio>

struct NestedPointerDescriptor {
    CUdeviceptr values;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr values = 0;
    CUdeviceptr descriptor = 0;
    CUdeviceptr output = 0;
    float input = 7.0f;
    float result = 0.0f;
    float addend = 5.0f;

    if (cuInit(0) != CUDA_SUCCESS ||
        cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&kernel, module, "nested_device_pointer") != CUDA_SUCCESS ||
        cuMemAlloc(&values, sizeof(input)) != CUDA_SUCCESS ||
        cuMemAlloc(&descriptor, sizeof(NestedPointerDescriptor)) != CUDA_SUCCESS ||
        cuMemAlloc(&output, sizeof(result)) != CUDA_SUCCESS) {
        return 1;
    }

    const NestedPointerDescriptor host_descriptor{values};
    if (cuMemcpyHtoD(values, &input, sizeof(input)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(descriptor, &host_descriptor, sizeof(host_descriptor)) != CUDA_SUCCESS) {
        return 1;
    }

    void* args[] = {&descriptor, &output, &addend};
    if (cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS ||
        cuCtxSynchronize() != CUDA_SUCCESS ||
        cuMemcpyDtoH(&result, output, sizeof(result)) != CUDA_SUCCESS) {
        return 1;
    }

    if (std::fabs(result - 12.0f) > 1.0e-6f) {
        std::fprintf(stderr, "nested pointer result mismatch: got %.9g, expected 12\n", result);
        return 1;
    }

    cuMemFree(output);
    cuMemFree(descriptor);
    cuMemFree(values);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
