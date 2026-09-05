#pragma once
// CuMetal CUB shim: DeviceSegmentedRadixSort — host-side per-segment sort.
//
// Segments are described by two arrays of length num_segments: d_begin_offsets[i]
// is the first index of segment i and d_end_offsets[i] is one past its last.
// Segments may be empty (end <= begin) and CUB skips those.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include "../util_type.h"
#include "device_radix_sort.h"

namespace cub {

struct DeviceSegmentedRadixSort {
    template <typename KeyT, typename ValueT, typename OffsetIteratorT, typename CompareT>
    static cudaError_t sort_segments(void* d_temp_storage, size_t& temp_storage_bytes,
                                     KeyT* d_keys, ValueT* d_values, int num_items, int num_segments,
                                     OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                     cudaStream_t stream, CompareT compare) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream); sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = size_t(num_items) * (sizeof(KeyT) + sizeof(ValueT));
            return cudaSuccess;
        }
        for (int segment = 0; segment < num_segments; segment++) {
            const int begin = int(d_begin_offsets[segment]);
            const int end = int(d_end_offsets[segment]);
            if (end <= begin)
                continue;
            const int count = end - begin;
            if (d_values)
                cub::detail::sort_pairs_host(d_keys + begin, d_keys + begin, d_values + begin,
                                             d_values + begin, count, compare);
            else
                cub::detail::sort_keys_host(d_keys + begin, d_keys + begin, count, compare);
        }
        return cudaSuccess;
    }

    template <typename KeyT, typename ValueT, typename OffsetIteratorT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 const KeyT* d_keys_in, KeyT* d_keys_out,
                                 const ValueT* d_values_in, ValueT* d_values_out,
                                 int num_items, int num_segments,
                                 OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                 int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                 cudaStream_t stream = 0) {
        (void)begin_bit; (void)end_bit;
        if (d_temp_storage && d_keys_out != d_keys_in) {
            std::memcpy(d_keys_out, d_keys_in, size_t(num_items) * sizeof(KeyT));
            std::memcpy(d_values_out, d_values_in, size_t(num_items) * sizeof(ValueT));
        }
        return sort_segments(d_temp_storage, temp_storage_bytes, d_keys_out, d_values_out, num_items,
                             num_segments, d_begin_offsets, d_end_offsets, stream,
                             cub::detail::ascending<KeyT>);
    }

    template <typename KeyT, typename ValueT, typename OffsetIteratorT>
    static cudaError_t SortPairsDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                           const KeyT* d_keys_in, KeyT* d_keys_out,
                                           const ValueT* d_values_in, ValueT* d_values_out,
                                           int num_items, int num_segments,
                                           OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                           int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0) {
        (void)begin_bit; (void)end_bit;
        if (d_temp_storage && d_keys_out != d_keys_in) {
            std::memcpy(d_keys_out, d_keys_in, size_t(num_items) * sizeof(KeyT));
            std::memcpy(d_values_out, d_values_in, size_t(num_items) * sizeof(ValueT));
        }
        return sort_segments(d_temp_storage, temp_storage_bytes, d_keys_out, d_values_out, num_items,
                             num_segments, d_begin_offsets, d_end_offsets, stream,
                             cub::detail::descending<KeyT>);
    }

    template <typename KeyT, typename OffsetIteratorT>
    static cudaError_t SortKeys(void* d_temp_storage, size_t& temp_storage_bytes,
                                const KeyT* d_keys_in, KeyT* d_keys_out,
                                int num_items, int num_segments,
                                OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                cudaStream_t stream = 0) {
        (void)begin_bit; (void)end_bit;
        if (d_temp_storage && d_keys_out != d_keys_in)
            std::memcpy(d_keys_out, d_keys_in, size_t(num_items) * sizeof(KeyT));
        return sort_segments<KeyT, KeyT>(d_temp_storage, temp_storage_bytes, d_keys_out, nullptr,
                                         num_items, num_segments, d_begin_offsets, d_end_offsets,
                                         stream, cub::detail::ascending<KeyT>);
    }

    // ── DoubleBuffer forms ───────────────────────────────────────────────────

    template <typename KeyT, typename ValueT, typename OffsetIteratorT>
    static cudaError_t SortPairs(void* d_temp_storage, size_t& temp_storage_bytes,
                                 DoubleBuffer<KeyT>& d_keys, DoubleBuffer<ValueT>& d_values,
                                 int num_items, int num_segments,
                                 OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                 int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                 cudaStream_t stream = 0) {
        (void)begin_bit; (void)end_bit;
        return sort_segments(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_values.Current(),
                             num_items, num_segments, d_begin_offsets, d_end_offsets, stream,
                             cub::detail::ascending<KeyT>);
    }

    template <typename KeyT, typename ValueT, typename OffsetIteratorT>
    static cudaError_t SortPairsDescending(void* d_temp_storage, size_t& temp_storage_bytes,
                                           DoubleBuffer<KeyT>& d_keys, DoubleBuffer<ValueT>& d_values,
                                           int num_items, int num_segments,
                                           OffsetIteratorT d_begin_offsets, OffsetIteratorT d_end_offsets,
                                           int begin_bit = 0, int end_bit = sizeof(KeyT) * 8,
                                           cudaStream_t stream = 0) {
        (void)begin_bit; (void)end_bit;
        return sort_segments(d_temp_storage, temp_storage_bytes, d_keys.Current(), d_values.Current(),
                             num_items, num_segments, d_begin_offsets, d_end_offsets, stream,
                             cub::detail::descending<KeyT>);
    }
};

}  // namespace cub
