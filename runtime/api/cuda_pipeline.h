#pragma once
// CuMetal: cuda_pipeline.h — minimal stubs for CUDA async pipeline primitives.
// These map to sequential operations on Apple Silicon UMA.

#include "cuda_runtime.h"

#ifdef __cplusplus

#include <cstring>

namespace cuda {
namespace pipeline_role {
    enum role_t { producer, consumer };
}

// Minimal pipeline type: no-op on Apple Silicon (cp.async is sequential)
template <typename Scope>
class pipeline {
public:
    pipeline() = default;
    void producer_acquire() {}
    void producer_commit() {}
    void consumer_wait() {}
    void consumer_release() {}
};

// Pipeline shared state
template <typename Scope>
struct pipeline_shared_state {
    // No-op on UMA
};

// Scope tags
struct thread_scope_thread {};
struct thread_scope_block {};
struct thread_scope_device {};
struct thread_scope_system {};

// memcpy_async stub: just do a synchronous copy
template <typename T>
inline void memcpy_async(T* dst, const T* src, size_t count,
                         pipeline<thread_scope_block>&) {
    std::memcpy(dst, src, count);
}

template <typename Group, typename T>
inline void memcpy_async(Group, T* dst, const T* src, size_t count) {
    std::memcpy(dst, src, count);
}

} // namespace cuda

// Legacy __pipeline_ functions.
//
// These are device-side primitives in real CUDA, so they must be callable from
// __global__/__device__ code; declaring them host-only made any device use a
// compile error. Metal has no equivalent of the Ampere async-copy pipeline, so
// the copy runs synchronously and commit/wait become no-ops -- correct, merely
// slower, since a synchronous copy trivially satisfies any later wait.
// std::memcpy is not callable from device code, so copy byte-wise.
__host__ __device__ __forceinline__ void __pipeline_memcpy_async(void* dst, const void* src,
                                                                 size_t count) {
    char* d = static_cast<char*>(dst);
    const char* s = static_cast<const char*>(src);
    for (size_t i = 0; i < count; ++i) {
        d[i] = s[i];
    }
}
__host__ __device__ __forceinline__ void __pipeline_commit() {}
__host__ __device__ __forceinline__ void __pipeline_wait_prior(int) {}

#endif // __cplusplus
