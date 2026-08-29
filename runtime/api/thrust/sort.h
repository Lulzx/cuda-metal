#pragma once

// CuMetal thrust shim: sort backed by std::sort (CPU on UMA).

#include <algorithm>
#include <vector>
#include "detail/synchronize.h"

namespace thrust {

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
