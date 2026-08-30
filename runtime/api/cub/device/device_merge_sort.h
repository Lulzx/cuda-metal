#pragma once
// CuMetal CUB shim: DeviceMergeSort — device-level merge sort.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include <algorithm>

namespace cub {

struct DeviceMergeSort {
    // Sort keys in-place
    template <typename KeyIteratorT, typename CompareOpT>
    static cudaError_t SortKeys(void* d_temp_storage, size_t& temp_storage_bytes,
                                KeyIteratorT d_keys, int num_items,
                                CompareOpT compare_op, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        std::stable_sort(d_keys, d_keys + num_items, compare_op);
        return cudaSuccess;
    }

    // Sort key-value pairs in-place
    template <typename KeyIteratorT, typename ValueIteratorT, typename CompareOpT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 KeyIteratorT d_keys, ValueIteratorT d_items,
                                 int num_items, CompareOpT compare_op, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        // Build index permutation and sort by keys
        std::vector<int> idx(num_items);
        for (int i = 0; i < num_items; i++) idx[i] = i;

        using KeyT = typename std::iterator_traits<KeyIteratorT>::value_type;
        using ValT = typename std::iterator_traits<ValueIteratorT>::value_type;

        std::vector<KeyT> keys_copy(d_keys, d_keys + num_items);
        std::vector<ValT> vals_copy(d_items, d_items + num_items);

        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            return compare_op(keys_copy[a], keys_copy[b]);
        });
        for (int i = 0; i < num_items; i++) {
            d_keys[i] = keys_copy[idx[i]];
            d_items[i] = vals_copy[idx[i]];
        }
        return cudaSuccess;
    }

    // Sort keys copy
    template <typename KeyInputIteratorT, typename KeyIteratorT, typename CompareOpT>
    static cudaError_t SortKeysCopy(void* d_temp_storage, size_t& temp_storage_bytes,
                                    KeyInputIteratorT d_input_keys, KeyIteratorT d_output_keys,
                                    int num_items, CompareOpT compare_op, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        std::copy(d_input_keys, d_input_keys + num_items, d_output_keys);
        std::stable_sort(d_output_keys, d_output_keys + num_items, compare_op);
        return cudaSuccess;
    }
};

} // namespace cub
