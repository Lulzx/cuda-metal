#pragma once
// CuMetal CUB shim: DeviceRunLengthEncode — device-level RLE.

#include <cuda_runtime.h>

namespace cub {

struct DeviceRunLengthEncode {
    // Encode runs of equal consecutive elements
    template <typename InputIteratorT, typename UniqueOutputIteratorT,
              typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    static cudaError_t Encode(void* d_temp_storage, size_t& temp_storage_bytes,
                              InputIteratorT d_in,
                              UniqueOutputIteratorT d_unique_out,
                              LengthsOutputIteratorT d_counts_out,
                              NumRunsOutputIteratorT d_num_runs_out,
                              int num_items, cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        if (num_items == 0) {
            *d_num_runs_out = 0;
            return cudaSuccess;
        }
        int runs = 0;
        int count = 1;
        auto current = d_in[0];
        for (int i = 1; i < num_items; i++) {
            if (d_in[i] == current) {
                count++;
            } else {
                d_unique_out[runs] = current;
                d_counts_out[runs] = count;
                runs++;
                current = d_in[i];
                count = 1;
            }
        }
        d_unique_out[runs] = current;
        d_counts_out[runs] = count;
        runs++;
        *d_num_runs_out = runs;
        return cudaSuccess;
    }

    // Non-trivial runs only (length > 1)
    template <typename InputIteratorT, typename OffsetsOutputIteratorT,
              typename LengthsOutputIteratorT, typename NumRunsOutputIteratorT>
    static cudaError_t NonTrivialRuns(void* d_temp_storage, size_t& temp_storage_bytes,
                                       InputIteratorT d_in,
                                       OffsetsOutputIteratorT d_offsets_out,
                                       LengthsOutputIteratorT d_lengths_out,
                                       NumRunsOutputIteratorT d_num_runs_out,
                                       int num_items, cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        if (num_items == 0) {
            *d_num_runs_out = 0;
            return cudaSuccess;
        }
        int runs = 0;
        int start = 0;
        for (int i = 1; i <= num_items; i++) {
            if (i == num_items || !(d_in[i] == d_in[start])) {
                int len = i - start;
                if (len > 1) {
                    d_offsets_out[runs] = start;
                    d_lengths_out[runs] = len;
                    runs++;
                }
                start = i;
            }
        }
        *d_num_runs_out = runs;
        return cudaSuccess;
    }
};

} // namespace cub
