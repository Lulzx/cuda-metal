#pragma once
// CuMetal CUB shim: DeviceRadixSort — host-side radix sort.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace cub {

struct DeviceRadixSort {
    // Sort keys ascending
    template <typename KeyT>
    static cudaError_t SortKeys(void* d_temp_storage, size_t& temp_storage_bytes,
                                const KeyT* d_keys_in, KeyT* d_keys_out, int num_items,
                                int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        (void)begin_bit; (void)end_bit;
        if (!d_temp_storage) {
            temp_storage_bytes = num_items * sizeof(KeyT);
            return cudaSuccess;
        }
        std::memcpy(d_keys_out, d_keys_in, num_items * sizeof(KeyT));
        std::sort(d_keys_out, d_keys_out + num_items);
        return cudaSuccess;
    }

    // Sort keys descending
    template <typename KeyT>
    static cudaError_t SortKeysDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                          const KeyT* d_keys_in, KeyT* d_keys_out, int num_items,
                                          int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                          cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        (void)begin_bit; (void)end_bit;
        if (!d_temp_storage) {
            temp_storage_bytes = num_items * sizeof(KeyT);
            return cudaSuccess;
        }
        std::memcpy(d_keys_out, d_keys_in, num_items * sizeof(KeyT));
        std::sort(d_keys_out, d_keys_out + num_items, [](const KeyT& a, const KeyT& b) { return a > b; });
        return cudaSuccess;
    }

    // Sort key-value pairs ascending
    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 const KeyT* d_keys_in, KeyT* d_keys_out,
                                 const ValueT* d_values_in, ValueT* d_values_out,
                                 int num_items,
                                 int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                 cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        (void)begin_bit; (void)end_bit;
        if (!d_temp_storage) {
            temp_storage_bytes = num_items * (sizeof(KeyT) + sizeof(int));
            return cudaSuccess;
        }
        std::vector<int> indices(num_items);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return d_keys_in[a] < d_keys_in[b]; });
        for (int i = 0; i < num_items; i++) {
            d_keys_out[i] = d_keys_in[indices[i]];
            d_values_out[i] = d_values_in[indices[i]];
        }
        return cudaSuccess;
    }

    // Sort key-value pairs descending
    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairsDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                           const KeyT* d_keys_in, KeyT* d_keys_out,
                                           const ValueT* d_values_in, ValueT* d_values_out,
                                           int num_items,
                                           int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        (void)begin_bit; (void)end_bit;
        if (!d_temp_storage) {
            temp_storage_bytes = num_items * (sizeof(KeyT) + sizeof(int));
            return cudaSuccess;
        }
        std::vector<int> indices(num_items);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](int a, int b) { return d_keys_in[a] > d_keys_in[b]; });
        for (int i = 0; i < num_items; i++) {
            d_keys_out[i] = d_keys_in[indices[i]];
            d_values_out[i] = d_values_in[indices[i]];
        }
        return cudaSuccess;
    }
};

} // namespace cub
