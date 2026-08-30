#pragma once
// CuMetal CUB shim: DevicePartition — device-level partitioning.

#include <cuda_runtime.h>
#include "../detail/host_backed.h"

namespace cub {

struct DevicePartition {
    // Partition items by predicate: selected items first, then unselected
    template <typename InputIteratorT, typename OutputIteratorT, typename NumSelectedIteratorT, typename SelectOp>
    static cudaError_t If(void* d_temp_storage, size_t& temp_storage_bytes,
                          InputIteratorT d_in, OutputIteratorT d_out,
                          NumSelectedIteratorT d_num_selected_out,
                          int num_items, SelectOp select_op, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int selected = 0, rejected = 0;
        // Two-pass: count selected, then place
        for (int i = 0; i < num_items; i++) {
            if (select_op(d_in[i])) selected++;
        }
        int si = 0, ri = selected;
        for (int i = 0; i < num_items; i++) {
            if (select_op(d_in[i])) {
                d_out[si++] = d_in[i];
            } else {
                d_out[ri++] = d_in[i];
            }
        }
        *d_num_selected_out = selected;
        return cudaSuccess;
    }

    // Partition items by flags
    template <typename InputIteratorT, typename FlagIterator, typename OutputIteratorT, typename NumSelectedIteratorT>
    static cudaError_t Flagged(void* d_temp_storage, size_t& temp_storage_bytes,
                               InputIteratorT d_in, FlagIterator d_flags,
                               OutputIteratorT d_out, NumSelectedIteratorT d_num_selected_out,
                               int num_items, cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int selected = 0;
        for (int i = 0; i < num_items; i++) {
            if (d_flags[i]) selected++;
        }
        int si = 0, ri = selected;
        for (int i = 0; i < num_items; i++) {
            if (d_flags[i]) {
                d_out[si++] = d_in[i];
            } else {
                d_out[ri++] = d_in[i];
            }
        }
        *d_num_selected_out = selected;
        return cudaSuccess;
    }

    // Three-way partition
    template <typename InputIteratorT, typename FirstOutputIteratorT, typename SecondOutputIteratorT,
              typename UnselectedOutputIteratorT, typename NumSelectedIteratorT,
              typename SelectFirstPartOp, typename SelectSecondPartOp>
    static cudaError_t If(void* d_temp_storage, size_t& temp_storage_bytes,
                          InputIteratorT d_in,
                          FirstOutputIteratorT d_first_part_out,
                          SecondOutputIteratorT d_second_part_out,
                          UnselectedOutputIteratorT d_unselected_out,
                          NumSelectedIteratorT d_num_selected_out,  // points to int[2]
                          int num_items,
                          SelectFirstPartOp select_first_part_op,
                          SelectSecondPartOp select_second_part_op,
                          cudaStream_t stream = 0) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int c1 = 0, c2 = 0, c3 = 0;
        for (int i = 0; i < num_items; i++) {
            if (select_first_part_op(d_in[i])) {
                d_first_part_out[c1++] = d_in[i];
            } else if (select_second_part_op(d_in[i])) {
                d_second_part_out[c2++] = d_in[i];
            } else {
                d_unselected_out[c3++] = d_in[i];
            }
        }
        d_num_selected_out[0] = c1;
        d_num_selected_out[1] = c2;
        return cudaSuccess;
    }
};

} // namespace cub
