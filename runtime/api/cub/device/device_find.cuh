#pragma once

#include <cuda_runtime.h>
#include "../detail/host_backed.h"
#include <cstddef>

namespace cub {

struct DeviceFind {
    template <typename InputIt, typename OutputIt, typename Predicate>
    static cudaError_t FindIf(void* storage, std::size_t& bytes, InputIt input,
                              OutputIt output, Predicate predicate, int count,
                              cudaStream_t stream = nullptr) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        if (count < 0) return cudaErrorInvalidValue;
        if (storage == nullptr) {
            bytes = 1;
            return cudaSuccess;
        }
        int found = count;
        for (int i = 0; i < count; ++i) {
            if (predicate(input[i])) {
                found = i;
                break;
            }
        }
        output[0] = found;
        return cudaSuccess;
    }

    template <typename RangeIt, typename ValueIt, typename OutputIt, typename Compare>
    static cudaError_t LowerBound(void* storage, std::size_t& bytes, RangeIt range,
                                  int range_count, ValueIt values, int value_count,
                                  OutputIt output, Compare compare,
                                  cudaStream_t stream = nullptr) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        return bounds<false>(storage, bytes, range, range_count, values,
                             value_count, output, compare);
    }

    template <typename RangeIt, typename ValueIt, typename OutputIt, typename Compare>
    static cudaError_t UpperBound(void* storage, std::size_t& bytes, RangeIt range,
                                  int range_count, ValueIt values, int value_count,
                                  OutputIt output, Compare compare,
                                  cudaStream_t stream = nullptr) {
        if (const cudaError_t sync = cub::detail::sync_host_backed(stream);
            sync != cudaSuccess) {
            return sync;
        }
        return bounds<true>(storage, bytes, range, range_count, values,
                            value_count, output, compare);
    }

private:
    template <bool Upper, typename RangeIt, typename ValueIt, typename OutputIt,
              typename Compare>
    static cudaError_t bounds(void* storage, std::size_t& bytes, RangeIt range,
                              int range_count, ValueIt values, int value_count,
                              OutputIt output, Compare compare) {
        if (range_count < 0 || value_count < 0) return cudaErrorInvalidValue;
        if (storage == nullptr) {
            bytes = 1;
            return cudaSuccess;
        }
        for (int value_index = 0; value_index < value_count; ++value_index) {
            int first = 0;
            int last = range_count;
            while (first < last) {
                const int middle = first + (last - first) / 2;
                const bool advance = Upper ? !compare(values[value_index], range[middle])
                                           : compare(range[middle], values[value_index]);
                if (advance) first = middle + 1;
                else last = middle;
            }
            output[value_index] = first;
        }
        return cudaSuccess;
    }
};

}
