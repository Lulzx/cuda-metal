#pragma once

#include <algorithm>
#include "detail/synchronize.h"

namespace thrust {

template <typename Iterator>
Iterator unique(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    return std::unique(first, last);
}

template <typename Iterator, typename BinaryPred>
Iterator unique(Iterator first, Iterator last, BinaryPred pred) {
    detail::synchronize_before_host_algorithm();
    return std::unique(first, last, pred);
}

template <typename InputIterator, typename OutputIterator>
OutputIterator unique_copy(InputIterator first, InputIterator last,
                            OutputIterator result) {
    detail::synchronize_before_host_algorithm();
    return std::unique_copy(first, last, result);
}

template <typename KeyIterator, typename ValueIterator,
          typename KeyOutputIterator, typename ValueOutputIterator>
std::pair<KeyOutputIterator, ValueOutputIterator>
unique_by_key_copy(KeyIterator keys_first, KeyIterator keys_last,
                   ValueIterator values_first, KeyOutputIterator keys_output,
                   ValueOutputIterator values_output) {
    detail::synchronize_before_host_algorithm();
    if (keys_first == keys_last) return {keys_output, values_output};
    auto previous_key = *keys_first;
    *keys_output = previous_key;
    *values_output = *values_first;
    ++keys_output;
    ++values_output;
    ++keys_first;
    ++values_first;
    for (; keys_first != keys_last; ++keys_first, ++values_first) {
        if (!(*keys_first == previous_key)) {
            previous_key = *keys_first;
            *keys_output = previous_key;
            *values_output = *values_first;
            ++keys_output;
            ++values_output;
        }
    }
    return {keys_output, values_output};
}

} // namespace thrust
