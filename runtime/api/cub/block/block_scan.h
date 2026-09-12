#pragma once
// CuMetal CUB shim: BlockScan.
//
// Two implementations behind one interface. Inside a kernel this is a real
// cooperative Hillis-Steele scan over threadgroup memory; on the host -- the
// shim is also included by host-only translation units -- it keeps the
// sequential fallback, where a single "thread 0" scans whatever the caller
// placed in temp storage.

#include <cuda_runtime.h>
#include <cstring>

namespace cub {

enum BlockScanAlgorithm {
    BLOCK_SCAN_RAKING,
    BLOCK_SCAN_RAKING_MEMOIZE,
    BLOCK_SCAN_WARP_SCANS
};

template <typename T, int BLOCK_DIM_X, BlockScanAlgorithm ALGORITHM = BLOCK_SCAN_RAKING,
          int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1, int LEGACY_PTX_ARCH = 0>
class BlockScan {
public:
    static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;

    // Uninitialized storage rather than `T data[BLOCK_THREADS]`. A __shared__
    // variable may not have an initializer, and an array of a type with a
    // user-provided default constructor is one. CUB's own TempStorage is
    // uninitialized storage for this reason.
    struct TempStorage {
        alignas(T) unsigned char storage[sizeof(T) * BLOCK_THREADS];

        __host__ __device__ T* data() { return reinterpret_cast<T*>(storage); }
        __host__ __device__ const T* data() const { return reinterpret_cast<const T*>(storage); }
    };

    __host__ __device__ explicit BlockScan(TempStorage& temp)
        : temp_(temp), linear_tid_(RowMajorTid()) {}
    __host__ __device__ BlockScan(TempStorage& temp, int linear_tid)
        : temp_(temp), linear_tid_(linear_tid) {}

    // Exclusive prefix sum.
    __host__ __device__ void ExclusiveSum(T input, T& output) {
        T aggregate;
        ExclusiveSum(input, output, aggregate);
    }

    // Exclusive prefix sum with the whole block's aggregate.
    __host__ __device__ void ExclusiveSum(T input, T& output, T& block_aggregate) {
        ExclusiveScan(input, output, T{}, block_aggregate, SumOp());
    }

    // Exclusive scan with custom op and initial value.
    template <typename ScanOp>
    __host__ __device__ void ExclusiveScan(T input, T& output, T initial_value, ScanOp op) {
        T aggregate;
        ExclusiveScan(input, output, initial_value, aggregate, op);
    }

    // Exclusive scan with custom op, initial value and block aggregate.
    // exclusive[i] = initial ⊕ x0 ⊕ … ⊕ x_{i-1}; the aggregate is the
    // reduction of the block's input items (CUB does not fold initial_value
    // into it).
    template <typename ScanOp>
    __host__ __device__ void ExclusiveScan(T input, T& output, T initial_value,
                                           T& block_aggregate, ScanOp op) {
        InclusiveScanImpl(input, op);
        T* data = temp_.data();
        // Copy the threadgroup slot into a local first: the IR backend
        // specializes op() per address space and cannot mix a private operand
        // with a threadgroup operand in one call.
        T previous{};
        if (linear_tid_ > 0) previous = data[linear_tid_ - 1];
        output = (linear_tid_ == 0) ? initial_value
                                    : op(initial_value, previous);
        block_aggregate = data[BLOCK_THREADS - 1];
    }

    // Inclusive prefix sum.
    __host__ __device__ void InclusiveSum(T input, T& output) {
        T aggregate;
        InclusiveSum(input, output, aggregate);
    }

    // Inclusive prefix sum with the whole block's aggregate.
    __host__ __device__ void InclusiveSum(T input, T& output, T& block_aggregate) {
        InclusiveScan(input, output, block_aggregate, SumOp());
    }

    // Inclusive scan with custom op.
    template <typename ScanOp>
    __host__ __device__ void InclusiveScan(T input, T& output, ScanOp op) {
        T aggregate;
        InclusiveScan(input, output, aggregate, op);
    }

    // Inclusive scan with custom op and block aggregate.
    template <typename ScanOp>
    __host__ __device__ void InclusiveScan(T input, T& output, T& block_aggregate, ScanOp op) {
        InclusiveScanImpl(input, op);
        output = temp_.data()[linear_tid_];
        block_aggregate = temp_.data()[BLOCK_THREADS - 1];
    }

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

    // Leaves the inclusive prefix in temp storage; every thread's slot holds
    // x0 ⊕ … ⊕ x_i when this returns.
    template <typename ScanOp>
    __host__ __device__ void InclusiveScanImpl(T input, ScanOp op) {
        T* data = temp_.data();
#ifdef __CUDA_ARCH__
        data[linear_tid_] = input;
        __syncthreads();
        // Hillis-Steele: associative op, works for any BLOCK_THREADS and any
        // element type. All reads complete before the barriered write.
        for (int stride = 1; stride < BLOCK_THREADS; stride <<= 1) {
            T partial = data[linear_tid_];
            if (linear_tid_ >= stride) {
                // Same address-space rule as ExclusiveScan: both operands go
                // through private locals.
                const T neighbor = data[linear_tid_ - stride];
                partial = op(neighbor, partial);
            }
            __syncthreads();
            data[linear_tid_] = partial;
            __syncthreads();
        }
#else
        data[linear_tid_] = input;
        if (linear_tid_ == 0) {
            for (int i = 1; i < BLOCK_THREADS; i++)
                data[i] = op(data[i - 1], data[i]);
        }
#endif
    }

    TempStorage& temp_;
    int linear_tid_;
};

} // namespace cub
