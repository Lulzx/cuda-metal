#pragma once
// CuMetal CUB shim: DeviceRadixSort — host-side radix sort.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include "../util_type.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace cub {
namespace detail {

// CUB's radix sort is stable, and callers rely on it: NVIDIA Warp's sparse
// triplet path sorts (row, col) keys and then run-length encodes the result,
// so the order of equal keys decides which block indices pair up. std::sort
// is not stable; std::stable_sort is.
template <typename KeyT, typename ValueT, typename CompareT>
void sort_pairs_host(const KeyT* keys_in, KeyT* keys_out, const ValueT* values_in, ValueT* values_out,
                     int num_items, CompareT compare) {
    std::vector<int> order(num_items);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return compare(keys_in[a], keys_in[b]); });

    // Sorting in place (keys_out == keys_in) is the DoubleBuffer case, so
    // gather through a copy rather than permuting the input as we read it.
    std::vector<KeyT> keys(num_items);
    std::vector<ValueT> values(num_items);
    for (int i = 0; i < num_items; i++) {
        keys[i] = keys_in[order[i]];
        values[i] = values_in[order[i]];
    }
    std::copy(keys.begin(), keys.end(), keys_out);
    std::copy(values.begin(), values.end(), values_out);
}

template <typename KeyT, typename CompareT>
void sort_keys_host(const KeyT* keys_in, KeyT* keys_out, int num_items, CompareT compare) {
    if (keys_out != keys_in)
        std::memcpy(keys_out, keys_in, size_t(num_items) * sizeof(KeyT));
    std::stable_sort(keys_out, keys_out + num_items, compare);
}

template <typename KeyT>
bool ascending(const KeyT& a, const KeyT& b) { return a < b; }

template <typename KeyT>
bool descending(const KeyT& a, const KeyT& b) { return b < a; }

}  // namespace detail

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
        cub::detail::sort_keys_host(d_keys_in, d_keys_out, num_items, cub::detail::ascending<KeyT>);
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
        cub::detail::sort_keys_host(d_keys_in, d_keys_out, num_items, cub::detail::descending<KeyT>);
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
            temp_storage_bytes = num_items * (sizeof(KeyT) + sizeof(ValueT));
            return cudaSuccess;
        }
        cub::detail::sort_pairs_host(d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
                                     cub::detail::ascending<KeyT>);
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
            temp_storage_bytes = num_items * (sizeof(KeyT) + sizeof(ValueT));
            return cudaSuccess;
        }
        cub::detail::sort_pairs_host(d_keys_in, d_keys_out, d_values_in, d_values_out, num_items,
                                     cub::detail::descending<KeyT>);
        return cudaSuccess;
    }

    // ── DoubleBuffer forms ───────────────────────────────────────────────────
    // The sort happens in place in each buffer's current side, so the selector
    // is left alone and Current() keeps pointing at the caller's own buffer.

    template <typename KeyT>
    static cudaError_t SortKeys(void* d_temp_storage, size_t& temp_storage_bytes,
                                DoubleBuffer<KeyT>& d_keys, int num_items,
                                int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                cudaStream_t stream = 0) {
        return SortKeys(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_keys.Current(),
                        num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT>
    static cudaError_t SortKeysDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                          DoubleBuffer<KeyT>& d_keys, int num_items,
                                          int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                          cudaStream_t stream = 0) {
        return SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_keys.Current(),
                                  num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 DoubleBuffer<KeyT>& d_keys, DoubleBuffer<ValueT>& d_values,
                                 int num_items,
                                 int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                 cudaStream_t stream = 0) {
        return SortPairs(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_keys.Current(),
                         d_values.Current(), d_values.Current(), num_items, begin_bit, end_bit, stream);
    }

    template <typename KeyT, typename ValueT>
    static cudaError_t SortPairsDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                           DoubleBuffer<KeyT>& d_keys, DoubleBuffer<ValueT>& d_values,
                                           int num_items,
                                           int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0) {
        return SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_keys.Current(),
                                   d_values.Current(), d_values.Current(), num_items, begin_bit, end_bit,
                                   stream);
    }
};

}  // namespace cub
