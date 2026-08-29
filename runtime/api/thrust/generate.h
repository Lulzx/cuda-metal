#pragma once
#include "detail/synchronize.h"
#include <cstddef>
#include <tuple>
#include <utility>

namespace thrust {

template <typename... Iterators>
class zip_iterator;

#if defined(__CUDACC__)
namespace detail {

template <typename Iterator, typename Value>
__device__ void assign_generated(Iterator first, std::size_t index, Value value) {
    first[static_cast<typename Iterator::difference_type>(index)] = value;
}

template <typename... Iterators, typename Value, std::size_t... Is>
__device__ void assign_generated_zip(zip_iterator<Iterators...> first,
                                     std::size_t index,
                                     Value value,
                                     std::index_sequence<Is...>) {
    const auto& iterators = first.get_iterator_tuple();
    ((std::get<Is>(iterators)[static_cast<typename std::tuple_element<Is, std::tuple<Iterators...>>::type::difference_type>(index)] =
          std::get<Is>(value)), ...);
}

template <typename... Iterators, typename Value>
__device__ void assign_generated(zip_iterator<Iterators...> first,
                                 std::size_t index,
                                 Value value) {
    assign_generated_zip(first, index, value, std::index_sequence_for<Iterators...>{});
}

template <typename Iterator, typename Generator>
__global__ void generate_kernel(Iterator first, std::size_t count, Generator generator) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                              threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = index; i < count; i += stride) {
        assign_generated(first, i, generator());
    }
}

}  // namespace detail
#endif

template <typename Iterator, typename Generator>
void generate(Iterator first, Iterator last, Generator generator) {
    detail::synchronize_before_host_algorithm();
#if defined(__CUDACC__)
    const auto distance = last - first;
    if (distance <= 0) return;
    constexpr unsigned int threads = 256;
    const std::size_t count = static_cast<std::size_t>(distance);
    const unsigned int blocks = static_cast<unsigned int>((count + threads - 1) / threads);
    detail::generate_kernel<<<blocks, threads>>>(first, count, generator);
    (void)cudaDeviceSynchronize();
#else
    for (; first != last; ++first) *first = generator();
#endif
}

}
