#pragma once
#include "detail/synchronize.h"

namespace thrust {

template <typename Iterator, typename Predicate>
Iterator find_if(Iterator first, Iterator last, Predicate predicate) {
    detail::synchronize_before_host_algorithm();
    for (; first != last; ++first) {
        if (predicate(*first)) return first;
    }
    return last;
}

} // namespace thrust
