#pragma once

// CuMetal thrust shim: transform, fill, copy, for_each backed by CPU loops.

#include <algorithm>
#include <cstring>
#include "detail/synchronize.h"

namespace thrust {

template <typename InputIterator, typename OutputIterator, typename UnaryOp>
OutputIterator transform(InputIterator first, InputIterator last,
                          OutputIterator result, UnaryOp op) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, ++result)
        *result = op(*first);
    return result;
}

template <typename InputIt1, typename InputIt2, typename OutputIterator, typename BinaryOp>
OutputIterator transform(InputIt1 first1, InputIt1 last1,
                          InputIt2 first2, OutputIterator result, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    for (; first1 != last1; ++first1, ++first2, ++result)
        *result = op(*first1, *first2);
    return result;
}

template <typename Iterator, typename T>
void fill(Iterator first, Iterator last, const T& value) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first) *first = value;
}

template <typename Iterator, typename Size, typename T>
Iterator fill_n(Iterator first, Size n, const T& value) {
    detail::synchronize_before_host_algorithm();
    for (Size i = 0; i < n; ++i, ++first) *first = value;
    return first;
}

template <typename InputIterator, typename OutputIterator>
OutputIterator copy(InputIterator first, InputIterator last, OutputIterator result) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, ++result) *result = *first;
    return result;
}

template <typename Iterator, typename T>
void replace(Iterator first, Iterator last, const T& old_val, const T& new_val) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first)
        if (*first == old_val) *first = new_val;
}

template <typename Iterator, typename UnaryOp>
void for_each(Iterator first, Iterator last, UnaryOp op) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first) op(*first);
}

template <typename InputIterator, typename OutputIterator, typename Pred>
OutputIterator copy_if(InputIterator first, InputIterator last,
                        OutputIterator result, Pred pred) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first)
        if (pred(*first)) { *result = *first; ++result; }
    return result;
}

// sequence: fill with 0, 1, 2, ...
template <typename Iterator>
void sequence(Iterator first, Iterator last) {
    detail::synchronize_before_host_algorithm();
    typedef typename std::iterator_traits<Iterator>::value_type T;
    T val = T();
    for (; first != last; ++first, ++val) *first = val;
}

template <typename Iterator, typename T>
void sequence(Iterator first, Iterator last, T init) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, ++init) *first = init;
}

template <typename Iterator, typename T>
void sequence(Iterator first, Iterator last, T init, T step) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, init = init + step) *first = init;
}

} // namespace thrust
