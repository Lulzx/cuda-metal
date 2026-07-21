#include "cuda_runtime.h"

extern "C" __global__ void warp_mask_votes(unsigned int* output) {
    __shared__ unsigned int scratch[32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int member_mask = lane < 16u ? 0x0000ffffu : 0xffff0000u;
    const int even = (lane & 1u) == 0u;
    const unsigned int even_mask = member_mask & 0x55555555u;
    const unsigned int group_base = lane & 16u;

    const unsigned int base = threadIdx.x * 6u;
    output[base + 0u] = __ballot_sync(member_mask, even);
    output[base + 1u] = static_cast<unsigned int>(__any_sync(member_mask, even));
    output[base + 2u] = static_cast<unsigned int>(__all_sync(even_mask, even));
    output[base + 3u] = __activemask();
    output[base + 4u] = __shfl_sync(member_mask, lane, static_cast<int>(group_base));

    // Exercise a real masked warp barrier under divergent execution.  Each
    // half-warp publishes to and reads from its own shared-memory region.
    if (lane < 16u) {
        scratch[lane] = lane + 100u;
        __syncwarp(0x0000ffffu);
        output[base + 5u] = scratch[(lane + 1u) & 15u];
    } else {
        scratch[lane] = lane + 200u;
        __syncwarp(0xffff0000u);
        output[base + 5u] = scratch[16u + ((lane - 15u) & 15u)];
    }
}
