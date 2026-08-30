#pragma once
// CuMetal CUB shim: DeviceReduce — host-side device reduction.
// On Apple Silicon UMA, device memory is host-accessible, so we run sequentially.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cstring>

namespace cub {

struct DeviceReduce {
    // Sum
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Sum(void* d_temp_storage, size_t& temp_storage_bytes,
                           InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                           cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T sum = T{};
        for (int i = 0; i < num_items; i++)
            sum = sum + d_in[i];
        *d_out = sum;
        return cudaSuccess;
    }

    // Min
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Min(void* d_temp_storage, size_t& temp_storage_bytes,
                           InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                           cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T val = d_in[0];
        for (int i = 1; i < num_items; i++)
            if (d_in[i] < val) val = d_in[i];
        *d_out = val;
        return cudaSuccess;
    }

    // Max
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t Max(void* d_temp_storage, size_t& temp_storage_bytes,
                           InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                           cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T val = d_in[0];
        for (int i = 1; i < num_items; i++)
            if (d_in[i] > val) val = d_in[i];
        *d_out = val;
        return cudaSuccess;
    }

    // Reduce with custom op
    template <typename InputIteratorT, typename OutputIteratorT, typename ReduceOp, typename T>
    static cudaError_t Reduce(void* d_temp_storage, size_t& temp_storage_bytes,
                              InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                              ReduceOp op, T init, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        T result = init;
        for (int i = 0; i < num_items; i++)
            result = op(result, d_in[i]);
        *d_out = result;
        return cudaSuccess;
    }

    // ArgMin — returns {index, value} of minimum
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ArgMin(void* d_temp_storage, size_t& temp_storage_bytes,
                              InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                              cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(d_in[0])>::type;
        int best_idx = 0;
        T best_val = d_in[0];
        for (int i = 1; i < num_items; i++) {
            if (d_in[i] < best_val) { best_val = d_in[i]; best_idx = i; }
        }
        // Output is a KeyValuePair-like struct: {key, value}
        *d_out = {best_idx, best_val};
        return cudaSuccess;
    }

    // ArgMax
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ArgMax(void* d_temp_storage, size_t& temp_storage_bytes,
                              InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                              cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(d_in[0])>::type;
        int best_idx = 0;
        T best_val = d_in[0];
        for (int i = 1; i < num_items; i++) {
            if (d_in[i] > best_val) { best_val = d_in[i]; best_idx = i; }
        }
        *d_out = {best_idx, best_val};
        return cudaSuccess;
    }
};

// KeyValuePair used by ArgMin/ArgMax
template <typename KeyT, typename ValueT>
struct KeyValuePair {
    KeyT key;
    ValueT value;
};

} // namespace cub
