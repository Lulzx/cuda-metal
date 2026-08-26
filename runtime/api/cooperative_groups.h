#pragma once

#include "cuda_runtime.h"

#if defined(__cplusplus)

namespace cooperative_groups {

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
        return __shfl_sync(member_mask(), val, static_cast<int>(src_rank), width);
    }

    template <typename T>
    __device__ __forceinline__ T shfl_down(T val, unsigned int delta) const {
        const int width = TileSize < 32 ? TileSize : 32;
        return __shfl_down_sync(member_mask(), val, delta, width);
    }

    template <typename T>
    __device__ __forceinline__ T shfl_up(T val, unsigned int delta) const {
        const int width = TileSize < 32 ? TileSize : 32;
        return __shfl_up_sync(member_mask(), val, delta, width);
    }

    template <typename T>
    __device__ __forceinline__ T shfl_xor(T val, unsigned int lane_mask) const {
        const int width = TileSize < 32 ? TileSize : 32;
        return __shfl_xor_sync(member_mask(), val, static_cast<int>(lane_mask), width);
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
    __device__ __forceinline__ unsigned int linear_tid() const {
        return (threadIdx.z * (blockDim.y * blockDim.x)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    }

    __device__ __forceinline__ unsigned int block_size() const {
        return blockDim.x * blockDim.y * blockDim.z;
    }
};

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

template <int TileSize>
__device__ __forceinline__ thread_block_tile<TileSize> tiled_partition(const thread_block&) {
    return {};
}

// grid_group — grid-wide collective group.
// Metal has no cross-threadgroup barrier. CuMetal therefore accepts only a
// single-threadgroup cooperative grid and rejects multi-block cooperative
// launch at the Runtime/Driver API boundary.
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

    // A threadgroup barrier is grid-wide for the only admitted grid shape.
    __device__ __forceinline__ void sync() const {
        __syncthreads();
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


}  // namespace cooperative_groups

#endif
