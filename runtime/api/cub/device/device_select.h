#pragma once
// CuMetal CUB shim: DeviceSelect — device-level stream compaction.

#include <cuda_runtime.h>
#include <algorithm>

namespace cub {

struct DeviceSelect {
    // Select flagged items
    template <typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT>
    static cudaError_t Flagged(void* d_temp_storage, size_t& temp_storage_bytes,
                               InputIteratorT d_in, FlagIterator d_flags,
                               OutputIteratorT d_out, NumSelectedIteratorT d_num_selected_out,
                               int num_items, cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int count = 0;
        for (int i = 0; i < num_items; i++) {
            if (d_flags[i]) {
                d_out[count++] = d_in[i];
            }
        }
        *d_num_selected_out = count;
        return cudaSuccess;
    }

    // Select items matching a predicate
    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT, typename SelectOp>
    static cudaError_t If(void* d_temp_storage, size_t& temp_storage_bytes,
                          InputIteratorT d_in, OutputIteratorT d_out,
                          NumSelectedIteratorT d_num_selected_out,
                          int num_items, SelectOp select_op, cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int count = 0;
        for (int i = 0; i < num_items; i++) {
            if (select_op(d_in[i])) {
                d_out[count++] = d_in[i];
            }
        }
        *d_num_selected_out = count;
        return cudaSuccess;
    }

    // Remove duplicates (unique)
    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT>
    static cudaError_t Unique(void* d_temp_storage, size_t& temp_storage_bytes,
                              InputIteratorT d_in, OutputIteratorT d_out,
                              NumSelectedIteratorT d_num_selected_out,
                              int num_items, cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        if (num_items == 0) {
            *d_num_selected_out = 0;
            return cudaSuccess;
        }
        int count = 1;
        d_out[0] = d_in[0];
        for (int i = 1; i < num_items; i++) {
            if (!(d_in[i] == d_in[i - 1])) {
                d_out[count++] = d_in[i];
            }
        }
        *d_num_selected_out = count;
        return cudaSuccess;
    }
};

} // namespace cub
