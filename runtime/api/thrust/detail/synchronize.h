#pragma once

#include "cuda_runtime.h"

namespace thrust::detail {

inline void synchronize_before_host_algorithm() {
    // CuMetal implements this clean-room Thrust surface with CPU loops over
    // tracked UMA allocations. CUDA's default Thrust policy is synchronous;
    // wait for preceding stream work before the host touches those buffers.
    (void)cudaDeviceSynchronize();
}

} // namespace thrust::detail
