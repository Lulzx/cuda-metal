#pragma once

#include "../cooperative_groups.h"

#if defined(__cplusplus)

namespace cooperative_groups {

template <int TileSize, typename T, typename BinaryOp>
__device__ __forceinline__ T reduce(const thread_block_tile<TileSize>& tile, T value, BinaryOp op) {
    if constexpr (TileSize <= 32) {
        // CUDA tiles up to a warp are SIMD collectives. A block-wide barrier is
        // both unnecessary and invalid when a tile reduction is called from a
        // branch taken by only one warp of a multi-warp block.
        for (unsigned int offset = TileSize / 2u; offset > 0; offset >>= 1u) {
            const T other = tile.shfl_down(value, offset);
            if (tile.thread_rank() < offset) {
                value = op(value, other);
            }
        }
        return tile.shfl(value, 0u);
    }

    __shared__ T shared[1024];
    const unsigned int linear_tid = threadIdx.z * (blockDim.y * blockDim.x) + threadIdx.y * blockDim.x +
                                    threadIdx.x;
    const unsigned int tile_base = tile.meta_group_rank() * TileSize;
    const unsigned int tile_rank = tile.thread_rank();

    shared[linear_tid] = value;
    __syncthreads();

    for (unsigned int offset = TileSize / 2u; offset > 0; offset >>= 1u) {
        if (tile_rank < offset) {
            shared[tile_base + tile_rank] =
                op(shared[tile_base + tile_rank], shared[tile_base + tile_rank + offset]);
        }
        __syncthreads();
    }

    return shared[tile_base];
}

// Dynamic groups are not necessarily power-of-two sized or contiguous in
// physical lane space. Gather each dense group rank at the leader, then
// broadcast the exact result. This is intentionally correctness-first; a
// tree reduction can replace it later without changing semantics.
template <typename T, typename BinaryOp>
__device__ __forceinline__ T reduce(const coalesced_group& group, T value, BinaryOp op) {
    T result = value;
    for (unsigned int src_rank = 1u; src_rank < group.size(); ++src_rank) {
        const T candidate = group.shfl(value, src_rank);
        if (group.thread_rank() == 0u) {
            result = op(result, candidate);
        }
    }
    return group.shfl(result, 0u);
}

}  // namespace cooperative_groups

#endif
