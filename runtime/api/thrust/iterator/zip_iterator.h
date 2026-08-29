#pragma once

#include <iterator>
#include <tuple>

namespace thrust {

#if defined(__CUDACC__)
#define CUMETAL_THRUST_ZIP_HD __host__ __device__
#else
#define CUMETAL_THRUST_ZIP_HD
#endif

template <typename... Iterators>
class zip_iterator {
    std::tuple<Iterators...> iterators_;

    template <size_t... Is>
    CUMETAL_THRUST_ZIP_HD auto deref(std::index_sequence<Is...>) const {
        return std::tuple<decltype(*std::get<Is>(iterators_))...>(
            *std::get<Is>(iterators_)...);
    }
    template <size_t... Is>
    CUMETAL_THRUST_ZIP_HD void increment(std::index_sequence<Is...>) {
        ((++std::get<Is>(iterators_)), ...);
    }
    template <size_t... Is>
    CUMETAL_THRUST_ZIP_HD void advance_n(ptrdiff_t n, std::index_sequence<Is...>) {
        ((std::get<Is>(iterators_) += n), ...);
    }

public:
    typedef std::tuple<typename std::iterator_traits<Iterators>::value_type...> value_type;
    typedef std::tuple<decltype(*std::declval<Iterators>())...> reference;
    typedef void pointer;
    typedef ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    CUMETAL_THRUST_ZIP_HD zip_iterator() = default;
    CUMETAL_THRUST_ZIP_HD explicit zip_iterator(Iterators... its) : iterators_(its...) {}
    CUMETAL_THRUST_ZIP_HD explicit zip_iterator(std::tuple<Iterators...> t) : iterators_(t) {}

    CUMETAL_THRUST_ZIP_HD reference operator*() const {
        return deref(std::index_sequence_for<Iterators...>{});
    }
    CUMETAL_THRUST_ZIP_HD zip_iterator& operator++() {
        increment(std::index_sequence_for<Iterators...>{});
        return *this;
    }
    CUMETAL_THRUST_ZIP_HD zip_iterator operator++(int) { auto t = *this; ++(*this); return t; }
    CUMETAL_THRUST_ZIP_HD zip_iterator& operator+=(ptrdiff_t n) {
        advance_n(n, std::index_sequence_for<Iterators...>{});
        return *this;
    }
    CUMETAL_THRUST_ZIP_HD zip_iterator operator+(ptrdiff_t n) const { auto t = *this; t += n; return t; }

    CUMETAL_THRUST_ZIP_HD ptrdiff_t operator-(const zip_iterator& o) const {
        return std::get<0>(iterators_) - std::get<0>(o.iterators_);
    }
    CUMETAL_THRUST_ZIP_HD reference operator[](ptrdiff_t n) const { return *(*this + n); }

    CUMETAL_THRUST_ZIP_HD bool operator==(const zip_iterator& o) const {
        return std::get<0>(iterators_) == std::get<0>(o.iterators_);
    }
    CUMETAL_THRUST_ZIP_HD bool operator!=(const zip_iterator& o) const { return !(*this == o); }
    CUMETAL_THRUST_ZIP_HD bool operator<(const zip_iterator& o) const {
        return std::get<0>(iterators_) < std::get<0>(o.iterators_);
    }

    CUMETAL_THRUST_ZIP_HD const std::tuple<Iterators...>& get_iterator_tuple() const { return iterators_; }
};

template <typename... Iterators>
zip_iterator<Iterators...> make_zip_iterator(Iterators... its) {
    return zip_iterator<Iterators...>(its...);
}

template <typename... Iterators>
zip_iterator<Iterators...> make_zip_iterator(std::tuple<Iterators...> t) {
    return zip_iterator<Iterators...>(t);
}

// tuple helpers used with zip_iterator
using std::get;
using std::make_tuple;
using std::tuple;

} // namespace thrust

#undef CUMETAL_THRUST_ZIP_HD
