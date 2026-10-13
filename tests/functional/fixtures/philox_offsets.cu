#include "cuda_runtime.h"
#include "curand_kernel.h"

extern "C" __global__ void philox_offsets(const unsigned long long* inputs,
                                         unsigned int* output,
                                         unsigned int count) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    curandStatePhilox4_32_10_t state;
    curand_init(inputs[index * 3u], inputs[index * 3u + 1u],
                inputs[index * 3u + 2u], &state);
    // Reserve a leading guard slot; inactive threads must preserve their slots.
    for (unsigned int draw = 0; draw < 8u; ++draw) {
        output[(index + 1u) * 8u + draw] = curand(&state);
    }
}
