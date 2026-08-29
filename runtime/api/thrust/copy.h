#pragma once

#include "transform.h"

namespace thrust {

template <typename InputIt, typename StencilIt, typename OutputIt, typename Predicate>
OutputIt copy_if(InputIt first, InputIt last, StencilIt stencil, OutputIt output,
                 Predicate predicate) {
    detail::synchronize_before_host_algorithm();
    while (first != last) {
        if (predicate(*stencil)) {
            *output = *first;
            ++output;
        }
        ++first;
        ++stencil;
    }
    return output;
}

}
