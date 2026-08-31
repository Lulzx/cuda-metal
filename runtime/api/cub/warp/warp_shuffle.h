#pragma once
// CuMetal CUB shim: free-function warp shuffle helpers.

#include <cuda_runtime.h>

namespace cub {

// CUB's ShuffleIndex helper accepts trivially-copyable values, including CUDA
// vector aggregates such as float3 and float4. CUDA shuffle instructions move
// one 32-bit register at a time, so preserve the object representation and
// shuffle every word from the same source lane.
template <int LOGICAL_WARP_THREADS, typename T>
__device__ __forceinline__ T ShuffleIndex(T input, int src_lane,
                                           unsigned int member_mask = 0xffffffffu) {
    static_assert(LOGICAL_WARP_THREADS > 0 && LOGICAL_WARP_THREADS <= 32,
                  "logical warp width must be in [1, 32]");
    static_assert((LOGICAL_WARP_THREADS & (LOGICAL_WARP_THREADS - 1)) == 0,
                  "logical warp width must be a power of two");
    static_assert(__is_trivially_copyable(T),
                  "cub::ShuffleIndex requires a trivially-copyable value");

    constexpr unsigned int kWordCount =
        (static_cast<unsigned int>(sizeof(T)) + sizeof(unsigned int) - 1u) /
        sizeof(unsigned int);
    unsigned int words[kWordCount] = {};
    __builtin_memcpy(words, &input, sizeof(T));
    for (unsigned int word = 0; word < kWordCount; ++word) {
        words[word] =
            __shfl_sync(member_mask, words[word], src_lane, LOGICAL_WARP_THREADS);
    }

    T output = input;
    __builtin_memcpy(&output, words, sizeof(T));
    return output;
}

} // namespace cub
