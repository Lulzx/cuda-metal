#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>

namespace cub {

struct DeviceSegmentedScan {
    template <typename InputIt, typename OutputIt, typename BeginOffsetIt,
              typename EndOffsetIt>
    static cudaError_t ExclusiveSegmentedSum(
        void* storage, std::size_t& bytes, InputIt input, OutputIt output,
        BeginOffsetIt begins, EndOffsetIt ends, int segments,
        cudaStream_t = nullptr) {
        if (segments < 0) return cudaErrorInvalidValue;
        if (storage == nullptr) {
            bytes = 1;
            return cudaSuccess;
        }
        using Value = std::remove_cv_t<std::remove_reference_t<decltype(input[0])>>;
        for (int segment = 0; segment < segments; ++segment) {
            Value running{};
            for (std::size_t i = begins[segment]; i < ends[segment]; ++i) {
                const Value value = input[i];
                output[i] = running;
                running = running + value;
            }
        }
        return cudaSuccess;
    }

    template <typename InputIt, typename OutputIt, typename BeginOffsetIt,
              typename EndOffsetIt, typename ScanOp>
    static cudaError_t InclusiveSegmentedScan(
        void* storage, std::size_t& bytes, InputIt input, OutputIt output,
        BeginOffsetIt begins, EndOffsetIt ends, int segments, ScanOp op,
        cudaStream_t = nullptr) {
        if (segments < 0) return cudaErrorInvalidValue;
        if (storage == nullptr) {
            bytes = 1;
            return cudaSuccess;
        }
        for (int segment = 0; segment < segments; ++segment) {
            const std::size_t begin = begins[segment];
            const std::size_t end = ends[segment];
            if (begin >= end) continue;
            auto running = input[begin];
            output[begin] = running;
            for (std::size_t i = begin + 1; i < end; ++i) {
                running = op(running, input[i]);
                output[i] = running;
            }
        }
        return cudaSuccess;
    }
};

}
