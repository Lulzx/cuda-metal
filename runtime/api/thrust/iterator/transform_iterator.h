#pragma once
// CuMetal thrust shim: transform_iterator — applies a function on dereference.

#include <iterator>

namespace thrust {

template <typename UnaryFunction, typename Iterator>
class transform_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = decltype(std::declval<UnaryFunction>()(std::declval<typename std::iterator_traits<Iterator>::value_type>()));
    using difference_type = typename std::iterator_traits<Iterator>::difference_type;
    using pointer = value_type*;
    using reference = value_type;

    transform_iterator() = default;
    transform_iterator(Iterator it, UnaryFunction fn) : it_(it), fn_(fn) {}

    reference operator*() const { return fn_(*it_); }
    reference operator[](difference_type n) const { return fn_(it_[n]); }

    transform_iterator& operator++() { ++it_; return *this; }
    transform_iterator operator++(int) { auto tmp = *this; ++it_; return tmp; }
    transform_iterator& operator--() { --it_; return *this; }
    transform_iterator operator--(int) { auto tmp = *this; --it_; return tmp; }

    transform_iterator& operator+=(difference_type n) { it_ += n; return *this; }
    transform_iterator& operator-=(difference_type n) { it_ -= n; return *this; }

    transform_iterator operator+(difference_type n) const { return transform_iterator(it_ + n, fn_); }
    transform_iterator operator-(difference_type n) const { return transform_iterator(it_ - n, fn_); }
    difference_type operator-(const transform_iterator& other) const { return it_ - other.it_; }

    bool operator==(const transform_iterator& o) const { return it_ == o.it_; }
    bool operator!=(const transform_iterator& o) const { return it_ != o.it_; }
    bool operator<(const transform_iterator& o) const { return it_ < o.it_; }
    bool operator>(const transform_iterator& o) const { return it_ > o.it_; }
    bool operator<=(const transform_iterator& o) const { return it_ <= o.it_; }
    bool operator>=(const transform_iterator& o) const { return it_ >= o.it_; }

private:
    Iterator it_{};
    UnaryFunction fn_{};
};

template <typename UnaryFunction, typename Iterator>
transform_iterator<UnaryFunction, Iterator> make_transform_iterator(Iterator it, UnaryFunction fn) {
    return transform_iterator<UnaryFunction, Iterator>(it, fn);
}

} // namespace thrust
