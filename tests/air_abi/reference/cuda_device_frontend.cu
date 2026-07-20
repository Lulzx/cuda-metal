#include <vector_types.h>
#include <cuda.h>
#include <sm_35_intrinsics.h>

#include "cuda_device_frontend_config.h"

// Match the opaque declarations used by CUDA consumers that avoid including
// cuda.h themselves. Repeating the real CUDA spelling must be harmless.
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;

struct __builtin_align__(16) CumetalAlignedProbe {
    unsigned long long value;
};

static_assert(sizeof(CUtexObject) == 8, "CUDA texture object ABI");
static_assert(alignof(CumetalAlignedProbe) == 16, "CUDA builtin alignment spelling");

extern "C" __global__ void cuda_device_probe(
    const float4* input, float4* output, unsigned long long* shuffled) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    float4 value = input[index];
    value.x += cumetal_frontend_test_bias();
    output[index] = value;
    const unsigned long long cached = __ldcg(shuffled + index);
    unsigned long long shuffle_input = static_cast<unsigned long long>(index) + cached;
    double shuffle_as_double;
    __builtin_memcpy(&shuffle_as_double, &shuffle_input, sizeof(shuffle_as_double));
    shuffle_as_double = __shfl_sync(0xffffffffu, shuffle_as_double, 0);
    __builtin_memcpy(shuffled + index, &shuffle_as_double, sizeof(shuffle_as_double));
}
