#include "cuda_runtime.h"

extern "C" __global__ void cuda_float_abs(float* output, float value) {
    output[0] = abs(value);
    output[1] = fabs(value);
}
