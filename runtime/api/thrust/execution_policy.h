#pragma once

// CuMetal thrust shim: execution policies.
// The legacy policy-free algorithms are CPU-backed on UMA. CUDA stream policy
// overloads must explicitly implement device execution or refuse it.

#include "cuda_runtime.h"

namespace thrust {

struct device_execution_policy {
    cudaStream_t stream = nullptr;
    constexpr device_execution_policy on(cudaStream_t selected) const { return {selected}; }
};
struct host_execution_policy {};
struct sequential_execution_policy {};

// Standard execution policy tags
static constexpr device_execution_policy device;
static constexpr host_execution_policy host;
static constexpr sequential_execution_policy seq;

// cuda::par is the default device policy
namespace cuda_cub {
    static constexpr device_execution_policy par;
}
namespace cuda {
    static constexpr device_execution_policy par;
}

namespace system {
namespace cuda {
    using thrust::device_execution_policy;
}
}

} // namespace thrust
