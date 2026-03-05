#pragma once
// CuMetal CUB shim: DeviceHistogram — device-level histogram computation.

#include <cuda_runtime.h>
#include <cstring>

namespace cub {

struct DeviceHistogram {
    // Even-width bins histogram
    template <typename SampleIteratorT, typename CounterT>
    static cudaError_t HistogramEven(void* d_temp_storage, size_t& temp_storage_bytes,
                                      SampleIteratorT d_samples,
                                      CounterT* d_histogram,
                                      int num_levels,
                                      float lower_level, float upper_level,
                                      int num_samples,
                                      cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int num_bins = num_levels - 1;
        std::memset(d_histogram, 0, num_bins * sizeof(CounterT));
        float range = upper_level - lower_level;
        if (range <= 0) return cudaSuccess;
        for (int i = 0; i < num_samples; i++) {
            float sample = static_cast<float>(d_samples[i]);
            if (sample >= lower_level && sample < upper_level) {
                int bin = static_cast<int>((sample - lower_level) / range * num_bins);
                if (bin >= num_bins) bin = num_bins - 1;
                d_histogram[bin]++;
            }
        }
        return cudaSuccess;
    }

    // Custom-range bins histogram
    template <typename SampleIteratorT, typename CounterT, typename LevelT>
    static cudaError_t HistogramRange(void* d_temp_storage, size_t& temp_storage_bytes,
                                       SampleIteratorT d_samples,
                                       CounterT* d_histogram,
                                       int num_levels,
                                       const LevelT* d_levels,
                                       int num_samples,
                                       cudaStream_t = 0) {
        if (!d_temp_storage) {
            temp_storage_bytes = 1;
            return cudaSuccess;
        }
        int num_bins = num_levels - 1;
        std::memset(d_histogram, 0, num_bins * sizeof(CounterT));
        for (int i = 0; i < num_samples; i++) {
            auto sample = d_samples[i];
            for (int b = 0; b < num_bins; b++) {
                if (sample >= d_levels[b] && sample < d_levels[b + 1]) {
                    d_histogram[b]++;
                    break;
                }
            }
        }
        return cudaSuccess;
    }
};

} // namespace cub
