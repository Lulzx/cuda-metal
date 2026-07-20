#include <vector_types.h>

#include "cuda_device_frontend_config.h"

extern "C" __global__ void cuda_device_probe(const float4* input, float4* output) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float4 value = input[index];
    value.x += cumetal_frontend_test_bias();
    output[index] = value;
}
