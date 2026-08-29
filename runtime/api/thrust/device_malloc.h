#pragma once

#include "cuda_runtime.h"
#include "device_ptr.h"

#include <stdexcept>

namespace thrust {

using system_error = std::runtime_error;

inline device_ptr<void> device_malloc(std::size_t bytes) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, bytes) != cudaSuccess) {
        throw system_error("thrust::device_malloc failed");
    }
    return device_ptr<void>(pointer);
}

template <typename T>
device_ptr<T> device_malloc(std::size_t count) {
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, count * sizeof(T)) != cudaSuccess) {
        throw system_error("thrust::device_malloc failed");
    }
    return device_ptr<T>(static_cast<T*>(pointer));
}

} // namespace thrust
