#pragma once

#include <iterator>
#include "functional.h"
#include "detail/synchronize.h"

namespace thrust {

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    if (first == last) return result;
    auto previous = *first;
    *result = previous;
    ++first;
    ++result;
    for (; first != last; ++first, ++result) {
        auto current = *first;
        *result = op(current, previous);
        previous = current;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator>
OutputIterator adjacent_difference(InputIterator first, InputIterator last,
                                   OutputIterator result) {
    return adjacent_difference(first, last, result,
        [](const auto& current, const auto& previous) {
            return current - previous;
        });
}

} // namespace thrust
