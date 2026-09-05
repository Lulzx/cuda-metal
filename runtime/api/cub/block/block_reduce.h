#pragma once
// CuMetal CUB shim: BlockReduce.
//
// Two implementations behind one interface. Inside a kernel this is a real
// cooperative tree reduction over threadgroup memory; on the host -- the shim
// is also included by host-only translation units -- it keeps the sequential
// fallback, where a single "thread 0" reduces whatever the caller placed in
// temp storage.

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

namespace cub {

enum BlockReduceAlgorithm {
    BLOCK_REDUCE_RAKING,
    BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
    BLOCK_REDUCE_WARP_REDUCTIONS
};

template <typename T, int BLOCK_DIM_X, BlockReduceAlgorithm ALGORITHM = BLOCK_REDUCE_WARP_REDUCTIONS,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockReduce {
public:
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    // Uninitialized storage rather than `T data[BLOCK_THREADS]`. A __shared__
    // variable may not have an initializer, and an array of a type with a
    // user-provided default constructor is one -- NVIDIA Warp's bvh.cu reduces
    // wp::vec3, which has exactly that. CUB's own TempStorage is uninitialized
    // storage for this reason.
    struct TempStorage {
        alignas(T) unsigned char storage[sizeof(T) * BLOCK_THREADS];

        __host__ __device__ T* data() { return reinterpret_cast<T*>(storage); }
        __host__ __device__ const T* data() const { return reinterpret_cast<const T*>(storage); }
    };

    __host__ __device__ explicit BlockReduce(TempStorage& temp)
        : temp_(temp), linear_tid_(RowMajorTid()) {}
    __host__ __device__ BlockReduce(TempStorage& temp, int linear_tid)
        : temp_(temp), linear_tid_(linear_tid) {}

    // Partial-tile reduce: only the first valid_items threads contribute.
    // As in CUB, valid_items must be the same in every thread of the block and
    // only thread 0's return value is defined.
    template <typename ReduceOp>
    __host__ __device__ T Reduce(T input, ReduceOp op, int valid_items) {
        T* data = temp_.data();
#ifdef __CUDA_ARCH__
        if (valid_items <= 0)
            return input;
        if (linear_tid_ < valid_items)
            data[linear_tid_] = input;
        __syncthreads();

        // Round the first stride up to a power of two so the top element is
        // still folded in when valid_items is not one.
        int stride = 1;
        while (stride < valid_items)
            stride <<= 1;
        for (stride >>= 1; stride > 0; stride >>= 1) {
            // The bound also keeps threads past valid_items from reading the
            // slots they never wrote.
            if (linear_tid_ < stride && linear_tid_ + stride < valid_items)
                data[linear_tid_] = op(data[linear_tid_], data[linear_tid_ + stride]);
            __syncthreads();
        }
        return data[0];
#else
        data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            T result = data[0];
            for (int i = 1; i < valid_items; i++)
                result = op(result, data[i]);
            return result;
        }
        return input;
#endif
    }

    // Full-tile reduce: every thread contributes one item.
    template <typename ReduceOp>
    __host__ __device__ T Reduce(T input, ReduceOp op) {
        return Reduce(input, op, BLOCK_THREADS);
    }

    __host__ __device__ T Sum(T input) { return Reduce(input, SumOp()); }
    __host__ __device__ T Sum(T input, int valid_items) { return Reduce(input, SumOp(), valid_items); }

private:
    // A functor rather than a lambda: this has to be callable from device code,
    // and an unannotated lambda in a __host__ __device__ member is not.
    struct SumOp {
        __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }
    };

    static __host__ __device__ int RowMajorTid() {
#ifdef __CUDA_ARCH__
        return threadIdx.x + BLOCK_DIM_X * (threadIdx.y + BLOCK_DIM_Y * threadIdx.z);
#else
        return 0;
#endif
    }

    TempStorage& temp_;
    int linear_tid_;
};

}  // namespace cub
