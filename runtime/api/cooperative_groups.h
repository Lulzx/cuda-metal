#pragma once

#include "cuda_runtime.h"
#include <type_traits>

#if defined(__cplusplus)

namespace cooperative_groups {

#if !defined(__CUDACC__)

// Ordinary host translation units may include cooperative_groups.h to size
// CUDA scratch-storage types even though CUDA builtins such as threadIdx do
// not exist in that language mode.
template <unsigned int BlockSize>
struct block_tile_memory {
    static_assert(BlockSize > 0, "block tile memory requires a non-empty block");
    unsigned int scratch[(BlockSize + 31u) / 32u];
};

#else

struct thread_block {
    __device__ __forceinline__ unsigned int size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return linear_tid();
    }

    // CUDA cooperative_groups API used by 3DGS / tile-based renderers.
    __device__ __forceinline__ dim3 group_index() const {
        return dim3(blockIdx.x, blockIdx.y, blockIdx.z);
    }

    __device__ __forceinline__ dim3 thread_index() const {
        return dim3(threadIdx.x, threadIdx.y, threadIdx.z);
    }

    __device__ __forceinline__ void sync() const {
        __syncthreads();
    }

private:
    __device__ __forceinline__ unsigned int linear_tid() const {
        return (threadIdx.z * (blockDim.y * blockDim.x)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    }
};

template <int TileSize>
struct thread_block_tile {
    static_assert(TileSize > 0, "TileSize must be positive");

    __device__ __forceinline__ unsigned int size() const { return TileSize; }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return linear_tid() % TileSize;
    }

    __device__ __forceinline__ unsigned int meta_group_rank() const {
        return linear_tid() / TileSize;
    }

    __device__ __forceinline__ unsigned int meta_group_size() const {
        return (block_size() + TileSize - 1u) / TileSize;
    }

    __device__ __forceinline__ void sync() const {
        if (TileSize <= 32) {
            __syncwarp(member_mask());
        } else {
            // Static multi-warp tiles currently require every tile in the
            // thread block to reach the same synchronization point.
            __syncthreads();
        }
    }

    // Warp-level shuffle within the tile.
    template <typename T>
    __device__ __forceinline__ T shfl(T val, unsigned int src_rank) const {
        const int width = TileSize < 32 ? TileSize : 32;
        if constexpr (std::is_class<T>::value || sizeof(T) > sizeof(unsigned long long)) {
            return shuffle_aggregate(val, [=](unsigned int word) {
                return __shfl_sync(member_mask(), word, static_cast<int>(src_rank), width);
            });
        } else {
            return __shfl_sync(member_mask(), val, static_cast<int>(src_rank), width);
        }
    }

    template <typename T>
    __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const {
        const int width = TileSize < 32 ? TileSize : 32;
        if constexpr (std::is_class<T>::value || sizeof(T) > sizeof(unsigned long long)) {
            return shuffle_aggregate(val, [=](unsigned int word) {
                return __shfl_down_sync(member_mask(), word, delta, width);
            });
        } else {
            return __shfl_down_sync(member_mask(), val, delta, width);
        }
    }

    template <typename T>
    __device__ __forceinline__ T shfl_up(T val, unsigned int delta) const {
        const int width = TileSize < 32 ? TileSize : 32;
        if constexpr (std::is_class<T>::value || sizeof(T) > sizeof(unsigned long long)) {
            return shuffle_aggregate(val, [=](unsigned int word) {
                return __shfl_up_sync(member_mask(), word, delta, width);
            });
        } else {
            return __shfl_up_sync(member_mask(), val, delta, width);
        }
    }

    template <typename T>
    __device__ __forceinline__ T shfl_xor(T val, unsigned int lane_mask) const {
        const int width = TileSize < 32 ? TileSize : 32;
        if constexpr (std::is_class<T>::value || sizeof(T) > sizeof(unsigned long long)) {
            return shuffle_aggregate(val, [=](unsigned int word) {
                return __shfl_xor_sync(member_mask(), word,
                                       static_cast<int>(lane_mask), width);
            });
        } else {
            return __shfl_xor_sync(member_mask(), val, static_cast<int>(lane_mask), width);
        }
    }

    __device__ __forceinline__ int any(int predicate) const {
        return __any_sync(member_mask(), predicate);
    }

    __device__ __forceinline__ int all(int predicate) const {
        return __all_sync(member_mask(), predicate);
    }

    __device__ __forceinline__ unsigned int ballot(int predicate) const {
        return __ballot_sync(member_mask(), predicate);
    }

    __device__ __forceinline__ unsigned int member_mask() const {
        const unsigned int width = TileSize < 32 ? static_cast<unsigned int>(TileSize) : 32u;
        const unsigned int lane = linear_tid() & 31u;
        const unsigned int base = (lane / width) * width;
        return (0xffffffffu >> (32u - width)) << base;
    }

private:
    template <typename T, typename ShuffleWord>
    __device__ __forceinline__ T shuffle_aggregate(T value, ShuffleWord shuffle_word) const {
        static_assert(std::is_trivially_copyable<T>::value,
                      "cooperative-groups shuffle values must be trivially copyable");
        union Packed {
            T value;
            unsigned int words[(sizeof(T) + sizeof(unsigned int) - 1) /
                               sizeof(unsigned int)];
        } packed{};
        packed.value = value;
        constexpr unsigned int kWords =
            (sizeof(T) + sizeof(unsigned int) - 1) / sizeof(unsigned int);
        static_assert(kWords <= 8, "cooperative-groups shuffle value is too large");
        if constexpr (kWords >= 1) packed.words[0] = shuffle_word(packed.words[0]);
        if constexpr (kWords >= 2) packed.words[1] = shuffle_word(packed.words[1]);
        if constexpr (kWords >= 3) packed.words[2] = shuffle_word(packed.words[2]);
        if constexpr (kWords >= 4) packed.words[3] = shuffle_word(packed.words[3]);
        if constexpr (kWords >= 5) packed.words[4] = shuffle_word(packed.words[4]);
        if constexpr (kWords >= 6) packed.words[5] = shuffle_word(packed.words[5]);
        if constexpr (kWords >= 7) packed.words[6] = shuffle_word(packed.words[6]);
        if constexpr (kWords >= 8) packed.words[7] = shuffle_word(packed.words[7]);
        return packed.value;
    }

    __device__ __forceinline__ unsigned int linear_tid() const {
        return (threadIdx.z * (blockDim.y * blockDim.x)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    }

    __device__ __forceinline__ unsigned int block_size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
};

// A warp-local group described by an explicit physical-lane membership mask.
// Group ranks are dense even when the participating lanes are not contiguous.
struct coalesced_group {
    __device__ __forceinline__ explicit coalesced_group(unsigned int member_mask)
        : member_mask_(member_mask) {}

    __device__ __forceinline__ unsigned int size() const {
        return static_cast<unsigned int>(__popc(member_mask_));
    }

    __device__ __forceinline__ unsigned int thread_rank() const {
        return static_cast<unsigned int>(__popc(member_mask_ & __lanemask_lt()));
    }

    __device__ __forceinline__ void sync() const { __syncwarp(member_mask_); }

    template <typename T>
    __device__ __forceinline__ T shfl(T value, unsigned int src_rank) const {
        if (src_rank >= size()) {
            return value;
        }
        return __shfl_sync(member_mask_, value, static_cast<int>(lane_for_rank(src_rank)));
    }

    template <typename T>
    __device__ __forceinline__ T shfl_down(T value, unsigned int delta) const {
        const unsigned int target_rank = thread_rank() + delta;
        return target_rank < size() ? shfl(value, target_rank) : value;
    }

    __device__ __forceinline__ int any(int predicate) const {
        return __any_sync(member_mask_, predicate);
    }

    __device__ __forceinline__ int all(int predicate) const {
        return __all_sync(member_mask_, predicate);
    }

    __device__ __forceinline__ unsigned int ballot(int predicate) const {
        return __ballot_sync(member_mask_, predicate) & member_mask_;
    }

    __device__ __forceinline__ unsigned int member_mask() const { return member_mask_; }

private:
    __device__ __forceinline__ unsigned int lane_for_rank(unsigned int rank) const {
        unsigned int remaining = member_mask_;
        for (unsigned int i = 0; i < rank; ++i) {
            remaining &= remaining - 1u;
        }
        return static_cast<unsigned int>(__ffs(static_cast<int>(remaining)) - 1);
    }

    unsigned int member_mask_;
};

__device__ __forceinline__ coalesced_group coalesced_threads() {
    return coalesced_group(__activemask());
}

template <typename ParentGroup>
__device__ __forceinline__ coalesced_group binary_partition(const ParentGroup& parent,
                                                             int predicate) {
    const unsigned int parent_mask = parent.member_mask();
    const unsigned int true_mask = __ballot_sync(parent_mask, predicate != 0) & parent_mask;
    return coalesced_group(predicate != 0 ? true_mask : (parent_mask & ~true_mask));
}

template <typename ParentGroup, typename Label>
__device__ __forceinline__ coalesced_group labeled_partition(const ParentGroup& parent,
                                                              Label label) {
    const unsigned int parent_mask = parent.member_mask();
    unsigned int unclaimed = parent_mask;
    while (unclaimed != 0u) {
        const unsigned int leader_lane =
            static_cast<unsigned int>(__ffs(static_cast<int>(unclaimed)) - 1);
        const Label leader_label =
            __shfl_sync(unclaimed, label, static_cast<int>(leader_lane));
        const unsigned int matching =
            __ballot_sync(unclaimed, label == leader_label) & unclaimed;
        if (label == leader_label) {
            return coalesced_group(matching);
        }
        unclaimed &= ~matching;
    }
    // Every calling lane belongs to parent_mask, so this is unreachable. Keep
    // an empty value as a defensive result for malformed divergent usage.
    return coalesced_group(0u);
}

// Type-erased cooperative group used by CUDA code that accepts either a
// thread block or a statically tiled partition. The stored synchronization
// domain is part of the value: converting a tile must not widen its barrier to
// the whole block, because sibling tiles are allowed to make progress
// independently.
struct thread_group {
    __device__ __forceinline__ thread_group(const thread_block& group)
        : size_(group.size()), rank_(group.thread_rank()), member_mask_(0xffffffffu), warp_group_(false) {}

    template <int TileSize>
    __device__ __forceinline__ thread_group(const thread_block_tile<TileSize>& group)
        : size_(group.size()),
          rank_(group.thread_rank()),
          member_mask_(group.member_mask()),
          warp_group_(TileSize <= 32) {}

    __device__ __forceinline__ thread_group(const coalesced_group& group)
        : size_(group.size()),
          rank_(group.thread_rank()),
          member_mask_(group.member_mask()),
          warp_group_(true) {}

    __device__ __forceinline__ unsigned int size() const { return size_; }
    __device__ __forceinline__ unsigned int thread_rank() const { return rank_; }

    __device__ __forceinline__ void sync() const {
        if (warp_group_) {
            __syncwarp(member_mask_);
        } else {
            __syncthreads();
        }
    }

private:
    unsigned int size_;
    unsigned int rank_;
    unsigned int member_mask_;
    bool warp_group_;
};

__device__ __forceinline__ thread_block this_thread_block() { return {}; }

// CUDA 12.8 introduced an explicit shared-memory scratch object for static
// partitions larger than a warp. CuMetal's current thread_block handle carries
// no state, but accepting the storage-bearing overload is still semantically
// exact for ordinary block operations and <=32-lane tiles. Multi-warp tile
// collectives continue to use the block barrier path above.
template <unsigned int BlockSize>
struct block_tile_memory {
    static_assert(BlockSize > 0, "block tile memory requires a non-empty block");
    unsigned int scratch[(BlockSize + 31u) / 32u];
};

template <unsigned int BlockSize>
__device__ __forceinline__ thread_block
this_thread_block(block_tile_memory<BlockSize>&) {
    return {};
}

template <int TileSize>
__device__ __forceinline__ thread_block_tile<TileSize> tiled_partition(const thread_block&) {
    return {};
}

extern "C" __device__ void __cumetal_grid_sync();

// grid_group — grid-wide collective group. CuMetal lowers this marker call to
// a device-wide sense-reversing barrier for cooperatively launched grids whose
// complete block set can be resident together.
struct grid_group {
    __device__ __forceinline__ unsigned int size() const {
        return gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z;
    }

    __device__ __forceinline__ unsigned int thread_rank() const {
        unsigned int blockRank =
            (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
        unsigned int threadInBlock =
            (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
        unsigned int blockSize = blockDim.x * blockDim.y * blockDim.z;
        return blockRank * blockSize + threadInBlock;
    }

    __device__ __forceinline__ void sync() const {
        __cumetal_grid_sync();
    }
};

__device__ __forceinline__ grid_group this_grid() { return {}; }

// Free-function sync — matches cg::sync(group) usage in CUDA code.
template <typename Group>
__device__ __forceinline__ void sync(Group& g) { g.sync(); }

template <typename T>
struct plus {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct greater {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

template <typename T>
struct less {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};

#endif  // defined(__CUDACC__)

}  // namespace cooperative_groups

#endif
