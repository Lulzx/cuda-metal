#include <cuda_runtime.h>

struct NestedPointerDescriptor {
    float* values;
};

extern "C" __global__ void nested_device_pointer(
    const NestedPointerDescriptor* descriptor,
    float* output,
    float addend) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        output[0] = descriptor[0].values[0] + addend;
    }
}
