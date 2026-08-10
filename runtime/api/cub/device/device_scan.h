#pragma once
// CuMetal CUB shim: DeviceScan — host-side device prefix scan.

#include <cuda_runtime.h>
#include <type_traits>

namespace cub {

struct DeviceScan {
    // Exclusive sum
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t ExclusiveSum(void* d_temp_storage, size_t& temp_storage_bytes,
                                    InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                                    cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T running = T{};
        for (int i = 0; i < num_items; i++) {
            T val = d_in[i];
            d_out[i] = running;
            running = running + val;
        }
        return cudaSuccess;
    }

    // Inclusive sum
    template <typename InputIteratorT, typename OutputIteratorT>
    static cudaError_t InclusiveSum(void* d_temp_storage, size_t& temp_storage_bytes,
                                    InputIteratorT d_in, OutputIteratorT d_out, int num_items,
                                    cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T running = T{};
        for (int i = 0; i < num_items; i++) {
            running = running + d_in[i];
            d_out[i] = running;
        }
        return cudaSuccess;
    }

    // Exclusive scan with custom op
    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp, typename InitValueT>
    static cudaError_t ExclusiveScan(void* d_temp_storage, size_t& temp_storage_bytes,
                                     InputIteratorT d_in, OutputIteratorT d_out,
                                     ScanOp op, InitValueT init, int num_items,
                                     cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        InitValueT running = init;
        for (int i = 0; i < num_items; i++) {
            auto val = d_in[i];
            d_out[i] = running;
            running = op(running, val);
        }
        return cudaSuccess;
    }

    // Inclusive scan with custom op
    template <typename InputIteratorT, typename OutputIteratorT, typename ScanOp>
    static cudaError_t InclusiveScan(void* d_temp_storage, size_t& temp_storage_bytes,
                                     InputIteratorT d_in, OutputIteratorT d_out,
                                     ScanOp op, int num_items,
                                     cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        using T = typename std::remove_reference<decltype(*d_out)>::type;
        T running = d_in[0];
        d_out[0] = running;
        for (int i = 1; i < num_items; i++) {
            running = op(running, d_in[i]);
            d_out[i] = running;
        }
        return cudaSuccess;
    }
};

} // namespace cub
