#pragma once

#include "cooperative_groups.h"
#include "cuda/pipeline"

namespace cooperative_groups {

template <typename Group, typename Destination, typename Source, size_t Alignment>
__device__ __forceinline__ void memcpy_async(
    const Group& group, Destination* destination, const Source* source,
    cuda::aligned_size_t<Alignment> size) {
    unsigned char* dst = reinterpret_cast<unsigned char*>(destination);
    const unsigned char* src = reinterpret_cast<const unsigned char*>(source);
    const size_t bytes = static_cast<size_t>(size);
    for (size_t i = group.thread_rank(); i < bytes; i += group.size()) {
        dst[i] = src[i];
    }
}

template <typename Group>
__device__ __forceinline__ void wait(const Group& group) {
    group.sync();
}

}  // namespace cooperative_groups
