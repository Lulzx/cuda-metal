#pragma once
// CuMetal CUB shim: utility types shared by the device-wide algorithms.

#include <cuda_runtime.h>

namespace cub {

// A pair of buffers with a selector saying which one currently holds the data.
// CUB's ping-pong radix sort hands the caller back whichever buffer the last
// pass wrote, so callers must read `Current()` rather than assume the buffer
// they passed in. CuMetal's radix sort is host-backed and sorts in place, so
// the selector it returns is always the one it was given; code written against
// real CUB still works, because reading `Current()` is what that code does.
template <typename T>
struct DoubleBuffer {
    T* d_buffers[2];
    int selector;

    __host__ __device__ DoubleBuffer() : selector(0) {
        d_buffers[0] = nullptr;
        d_buffers[1] = nullptr;
    }

    __host__ __device__ DoubleBuffer(T* d_current, T* d_alternate) : selector(0) {
        d_buffers[0] = d_current;
        d_buffers[1] = d_alternate;
    }

    __host__ __device__ T* Current() const { return d_buffers[selector]; }
    __host__ __device__ T* Alternate() const { return d_buffers[selector ^ 1]; }
};

}  // namespace cub
