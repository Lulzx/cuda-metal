#pragma once

// CuMetal thrust shim: inclusive/exclusive scan backed by CPU loops.

#include <functional>
#include "functional.h"
#include "detail/synchronize.h"

namespace thrust {

template <typename InputIterator, typename OutputIterator>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result) {
    detail::synchronize_before_host_algorithm();
    if (first == last) return result;
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = acc + *first;
        *result = acc;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator, typename BinaryOp>
OutputIterator inclusive_scan(InputIterator first, InputIterator last,
                              OutputIterator result, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    if (first == last) return result;
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    T acc = *first;
    *result = acc;
    ++first; ++result;
    for (; first != last; ++first, ++result) {
        acc = op(acc, *first);
        *result = acc;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result) {
    typedef typename std::iterator_traits<InputIterator>::value_type T;
    return exclusive_scan(first, last, result, T());
}

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result, T init) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, ++result) {
        *result = init;
        init = init + *first;
    }
    return result;
}

template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOp>
OutputIterator exclusive_scan(InputIterator first, InputIterator last,
                               OutputIterator result, T init, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first, ++result) {
        *result = init;
        init = op(init, *first);
    }
    return result;
}

template <typename KeyIterator, typename InputIterator, typename OutputIterator,
          typename KeyEqual, typename BinaryOp>
OutputIterator inclusive_scan_by_key(KeyIterator keys_first, KeyIterator keys_last,
                                     InputIterator values_first, OutputIterator result,
                                     KeyEqual keys_equal, BinaryOp op) {
    detail::synchronize_before_host_algorithm();
    if (keys_first == keys_last) return result;
    using Key = typename std::iterator_traits<KeyIterator>::value_type;
    using Value = typename std::iterator_traits<InputIterator>::value_type;
    Key previous_key = *keys_first;
    Value accumulator = *values_first;
    *result = accumulator;
    ++keys_first;
    ++values_first;
    ++result;
    for (; keys_first != keys_last;
         ++keys_first, ++values_first, ++result) {
        if (keys_equal(previous_key, *keys_first)) {
            accumulator = op(accumulator, *values_first);
        } else {
            accumulator = *values_first;
        }
        previous_key = *keys_first;
        *result = accumulator;
    }
    return result;
}

} // namespace thrust
