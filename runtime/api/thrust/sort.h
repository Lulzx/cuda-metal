#pragma once

// Policy-free sort uses the host; CUDA-policy sort uses CUDA source kernels.

#include <algorithm>
#include <vector>
#include <stdexcept>
#include <string>
#include <limits>
#include <type_traits>
#include "execution_policy.h"
#include "detail/synchronize.h"

namespace thrust {

#if defined(__CUDACC__)
namespace detail {
struct device_less {
    template<class T> __host__ __device__ bool operator()(const T& a, const T& b) const { return a < b; }
};
template<class Iterator, class Value, class Compare>
__global__ __attribute__((used)) void merge_sort_pass(Iterator input, Value* output, size_t count, size_t width, Compare compare) {
    size_t pair = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t pairs = (count - 1) / (2 * width) + 1;
    for(; pair < pairs; pair += size_t(gridDim.x) * blockDim.x) {
        size_t begin = pair * (2 * width);
        size_t middle = begin + (width < count - begin ? width : count - begin);
        size_t end = middle + (width < count - middle ? width : count - middle);
        size_t left = begin, right = middle, destination = begin;
        while(left < middle && right < end) {
            if(compare(input[right], input[left])) output[destination++] = input[right++];
            else output[destination++] = input[left++];
        }
        while(left < middle) output[destination++] = input[left++];
        while(right < end) output[destination++] = input[right++];
    }
}
template<class Iterator, class Value>
__global__ __attribute__((used)) void merge_sort_copy(Iterator output, const Value* input, size_t count) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<count; i+=size_t(gridDim.x)*blockDim.x)
        output[i]=input[i];
}
inline void sort_check(cudaError_t status) {
    if(status != cudaSuccess) throw std::runtime_error(std::string("CuMetal CUDA-policy Thrust sort: ") + cudaGetErrorString(status));
}
template<class Iterator, class Compare>
void device_sort(const device_execution_policy& policy, Iterator first, Iterator last, Compare compare) {
    using Value = typename std::iterator_traits<Iterator>::value_type;
    if constexpr (!std::is_trivially_copyable_v<Value>) {
        throw std::runtime_error("cudaErrorNotSupported: CuMetal CUDA-policy Thrust sort requires trivially copyable values");
    } else {
        auto distance = last - first;
        if(distance < 0) throw std::runtime_error("cudaErrorInvalidValue: reversed CUDA-policy Thrust sort range");
        size_t count = static_cast<size_t>(distance);
        if(count <= 1) { sort_check(cudaStreamSynchronize(policy.stream)); return; }
        if(count > std::numeric_limits<size_t>::max() / sizeof(Value) ||
           count > std::numeric_limits<size_t>::max() / 2)
            throw std::runtime_error("cudaErrorInvalidValue: CUDA-policy Thrust sort range is too large");
        Value* scratch = nullptr;
        sort_check(cudaMalloc(&scratch, count * sizeof(Value)));
        try {
            unsigned copy_blocks = static_cast<unsigned>(std::min<size_t>((count - 1) / 128 + 1, 65535));
            for(size_t width=1; width<count;) {
                size_t pairs=(count-1)/(2*width)+1;
                unsigned blocks=static_cast<unsigned>(std::min<size_t>((pairs-1)/128+1,65535));
                merge_sort_pass<<<blocks,128,0,policy.stream>>>(first,scratch,count,width,compare);
                sort_check(cudaGetLastError());
                merge_sort_copy<<<copy_blocks,128,0,policy.stream>>>(first,scratch,count);
                sort_check(cudaGetLastError());
                if(width >= count - width) break;
                width *= 2;
            }
            sort_check(cudaStreamSynchronize(policy.stream));
        } catch (...) {
            cudaStreamSynchronize(policy.stream);
            cudaFree(scratch);
            throw;
        }
        sort_check(cudaFree(scratch));
    }
}
} // namespace detail
#endif

template <typename Iterator>
void sort(const device_execution_policy& policy, Iterator first, Iterator last) {
#if defined(__CUDACC__)
    detail::device_sort(policy, first, last, detail::device_less{});
#else
    (void)policy; (void)first; (void)last;
    throw std::runtime_error("cudaErrorNotSupported: CuMetal CUDA-policy Thrust sort");
#endif
}
template <typename Iterator, typename Compare>
void sort(const device_execution_policy& policy, Iterator first, Iterator last, Compare compare) {
#if defined(__CUDACC__)
    detail::device_sort(policy, first, last, compare);
#else
    (void)policy; (void)first; (void)last; (void)compare;
    throw std::runtime_error("cudaErrorNotSupported: CuMetal CUDA-policy Thrust sort");
#endif
}
template <typename Keys, typename Values>
void sort_by_key(const device_execution_policy&, Keys, Keys, Values) {
    throw std::runtime_error("cudaErrorNotSupported: CuMetal CUDA-policy Thrust sort_by_key");
}
template <typename Keys, typename Values, typename Compare>
void sort_by_key(const device_execution_policy&, Keys, Keys, Values, Compare) {
    throw std::runtime_error("cudaErrorNotSupported: CuMetal CUDA-policy Thrust sort_by_key");
}

template <typename Iterator>
bool is_sorted(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    return std::is_sorted(first, last);
}

template <typename Iterator, typename Compare>
bool is_sorted(Iterator first, Iterator last, Compare compare) {
    detail::synchronize_before_host_algorithm();
    return std::is_sorted(first, last, compare);
}

template <typename Iterator>
void sort(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    using Value = typename std::iterator_traits<Iterator>::value_type;
    std::vector<Value> values;
    for (Iterator current = first; current != last; ++current) {
        values.emplace_back(*current);
    }
    std::sort(values.begin(), values.end());
    for (const auto& value : values) {
        *first = value;
        ++first;
    }
}

template <typename Iterator, typename Compare>
void sort(Iterator first, Iterator last, Compare comp) {
    detail::synchronize_before_host_algorithm();
    using Value = typename std::iterator_traits<Iterator>::value_type;
    std::vector<Value> values;
    for (Iterator current = first; current != last; ++current) {
        values.emplace_back(*current);
    }
    std::sort(values.begin(), values.end(), comp);
    for (const auto& value : values) {
        *first = value;
        ++first;
    }
}

template <typename KeyIterator, typename ValueIterator>
void sort_by_key(KeyIterator keys_first, KeyIterator keys_last,
                 ValueIterator values_first) {
    detail::synchronize_before_host_algorithm();
    // Zip-sort: create index array, sort indices by key, then permute
    auto n = keys_last - keys_first;
    if (n <= 1) return;

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < (size_t)n; ++i) idx[i] = i;

    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return keys_first[a] < keys_first[b];
    });

    // Apply permutation in-place using cycles
    typedef typename std::iterator_traits<KeyIterator>::value_type K;
    typedef typename std::iterator_traits<ValueIterator>::value_type V;
    std::vector<K> sorted_keys(n);
    std::vector<V> sorted_vals(n);
    for (size_t i = 0; i < (size_t)n; ++i) {
        sorted_keys[i] = keys_first[idx[i]];
        sorted_vals[i] = values_first[idx[i]];
    }
    for (size_t i = 0; i < (size_t)n; ++i) {
        keys_first[i] = sorted_keys[i];
        values_first[i] = sorted_vals[i];
    }
}

template <typename KeyIterator, typename ValueIterator, typename Compare>
void sort_by_key(KeyIterator keys_first, KeyIterator keys_last,
                 ValueIterator values_first, Compare comp) {
    detail::synchronize_before_host_algorithm();
    auto n = keys_last - keys_first;
    if (n <= 1) return;

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < (size_t)n; ++i) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](size_t a, size_t b) {
        return comp(keys_first[a], keys_first[b]);
    });

    typedef typename std::iterator_traits<KeyIterator>::value_type K;
    typedef typename std::iterator_traits<ValueIterator>::value_type V;
    std::vector<K> sorted_keys(n);
    std::vector<V> sorted_vals(n);
    for (size_t i = 0; i < (size_t)n; ++i) {
        sorted_keys[i] = keys_first[idx[i]];
        sorted_vals[i] = values_first[idx[i]];
    }
    for (size_t i = 0; i < (size_t)n; ++i) {
        keys_first[i] = sorted_keys[i];
        values_first[i] = sorted_vals[i];
    }
}

template <typename Iterator>
void stable_sort(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    std::stable_sort(first, last);
}

template <typename Iterator, typename Compare>
void stable_sort(Iterator first, Iterator last, Compare comp) {
    detail::synchronize_before_host_algorithm();
    std::stable_sort(first, last, comp);
}

} // namespace thrust
