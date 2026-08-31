#include <cuda_runtime.h>

__global__ void cuda_float_math_overloads(const float* input, float* output) {
    const float x = input[threadIdx.x];
    atomicAdd(&output[threadIdx.x], rsqrt(x) + fma(x, 2.0f, 3.0f));
}
