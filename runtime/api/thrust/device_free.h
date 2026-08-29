#pragma once

#include "cuda_runtime.h"
#include "device_ptr.h"

#include <stdexcept>

namespace thrust {

using system_error = std::runtime_error;

template <typename T>
void device_free(device_ptr<T> pointer) {
    if (cudaFree(pointer.get()) != cudaSuccess) {
        throw system_error("thrust::device_free failed");
    }
}

inline void device_free(device_ptr<void> pointer) {
    if (cudaFree(pointer.get()) != cudaSuccess) {
        throw system_error("thrust::device_free failed");
    }
}

} // namespace thrust
