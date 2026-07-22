#include "cuda_runtime.h"

extern "C" __global__ void warp_mask_votes(unsigned int* output) {
    __shared__ unsigned int scratch[32];
    __shared__ unsigned int second_scratch[32];
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int member_mask = lane < 16u ? 0x0000ffffu : 0xffff0000u;
    const int even = (lane & 1u) == 0u;
    const unsigned int even_mask = member_mask & 0x55555555u;
    const unsigned int group_base = lane & 16u;

    const unsigned int base = threadIdx.x * 7u;
    output[base + 0u] = __ballot_sync(member_mask, even);
    output[base + 1u] = static_cast<unsigned int>(__any_sync(member_mask, even));
    output[base + 2u] = static_cast<unsigned int>(__all_sync(even_mask, even));
    output[base + 3u] = __activemask();
    output[base + 4u] = __shfl_sync(member_mask, lane, static_cast<int>(group_base));

    // Exercise a real masked warp barrier under divergent execution.  Each
    // half-warp publishes to and reads from its own shared-memory region.
    if (lane < 16u) {
        scratch[lane] = lane + 100u;
        second_scratch[lane] = lane + 1000u;
        __syncwarp(0x0000ffffu);
        output[base + 5u] = scratch[(lane + 1u) & 15u];
        output[base + 6u] = second_scratch[(lane + 1u) & 15u];
    } else {
        scratch[lane] = lane + 200u;
        second_scratch[lane] = lane + 2000u;
        __syncwarp(0xffff0000u);
        output[base + 5u] = scratch[16u + ((lane - 15u) & 15u)];
        output[base + 6u] = second_scratch[16u + ((lane - 15u) & 15u)];
    }

    // Mirror PhysX's warp-cooperative contact-stream allocation and write:
    // participating lanes ballot, lane zero allocates one packed region,
    // the offset is broadcast, and each lane writes one float4.
    const bool writes_contact = lane < 4u;
    const unsigned int contact_mask = __ballot_sync(0xffffffffu, writes_contact);
    unsigned int byte_offset = 0xffffffffu;
    if (lane == 0u) {
        byte_offset = atomicAdd(output + 224u, __popc(contact_mask) * 16u);
    }
    byte_offset = __shfl_sync(0xffffffffu, byte_offset, 0);
    if (writes_contact) {
        const unsigned int preceding = contact_mask & ((1u << lane) - 1u);
        const unsigned int contact_index = __popc(preceding);
        float4* contact_stream = reinterpret_cast<float4*>(output + 225u);
        contact_stream[byte_offset / 16u + contact_index] =
            make_float4(100.0f + lane, 200.0f + lane, 300.0f + lane, -0.25f * lane);
    }
}
