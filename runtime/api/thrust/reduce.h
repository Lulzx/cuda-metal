#pragma once

// CuMetal thrust shim: reduce, transform_reduce backed by CPU loops.

#include <functional>
#include <numeric>
#include "detail/synchronize.h"

namespace thrust {

template <typename Iterator>
typename std::iterator_traits<Iterator>::value_type
reduce(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    typedef typename std::iterator_traits<Iterator>::value_type T;
    T init = T();
    for (auto it = first; it != last; ++it) init = init + *it;
    return init;
}

template <typename Iterator, typename T>
T reduce(Iterator first, Iterator last, T init) {
    detail::synchronize_before_host_algorithm();
    for (auto it = first; it != last; ++it) init = init + *it;
    return init;
}

template <typename Iterator, typename T, typename BinaryOp>
T reduce(Iterator first, Iterator last, T init, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    for (auto it = first; it != last; ++it) init = op(init, *it);
    return init;
}

template <typename Iterator, typename T, typename BinaryOp, typename UnaryOp>
T transform_reduce(Iterator first, Iterator last, T init, BinaryOp binary_op, UnaryOp unary_op) {
    detail::synchronize_before_host_algorithm();
    for (auto it = first; it != last; ++it) init = binary_op(init, unary_op(*it));
    return init;
}

// min/max element
template <typename Iterator>
Iterator min_element(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    if (first == last) return last;
    Iterator best = first;
    for (auto it = first; it != last; ++it)
        if (*it < *best) best = it;
    return best;
}

template <typename Iterator>
Iterator max_element(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    if (first == last) return last;
    Iterator best = first;
    for (auto it = first; it != last; ++it)
        if (*it > *best) best = it;
    return best;
}

// count / count_if
template <typename Iterator, typename T>
typename std::iterator_traits<Iterator>::difference_type
count(Iterator first, Iterator last, const T& value) {
    detail::synchronize_before_host_algorithm();
    typename std::iterator_traits<Iterator>::difference_type n = 0;
    for (auto it = first; it != last; ++it)
        if (*it == value) ++n;
    return n;
}

template <typename Iterator, typename Pred>
typename std::iterator_traits<Iterator>::difference_type
count_if(Iterator first, Iterator last, Pred pred) {
    detail::synchronize_before_host_algorithm();
    typename std::iterator_traits<Iterator>::difference_type n = 0;
    for (auto it = first; it != last; ++it)
        if (pred(*it)) ++n;
    return n;
}

} // namespace thrust
