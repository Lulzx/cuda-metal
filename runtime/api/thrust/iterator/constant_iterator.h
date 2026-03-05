#pragma once
// CuMetal thrust shim: constant_iterator — always returns the same value.

#include <iterator>
#include <cstddef>

namespace thrust {

template <typename T>
class constant_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = T;
    using difference_type = ptrdiff_t;
    using pointer = const T*;
    using reference = T;

    constant_iterator() = default;
    explicit constant_iterator(T val, difference_type idx = 0) : val_(val), idx_(idx) {}

    reference operator*() const { return val_; }
    reference operator[](difference_type) const { return val_; }

    constant_iterator& operator++() { ++idx_; return *this; }
    constant_iterator operator++(int) { auto tmp = *this; ++idx_; return tmp; }
    constant_iterator& operator--() { --idx_; return *this; }
    constant_iterator operator--(int) { auto tmp = *this; --idx_; return tmp; }

    constant_iterator& operator+=(difference_type n) { idx_ += n; return *this; }
    constant_iterator& operator-=(difference_type n) { idx_ -= n; return *this; }

    constant_iterator operator+(difference_type n) const { return constant_iterator(val_, idx_ + n); }
    constant_iterator operator-(difference_type n) const { return constant_iterator(val_, idx_ - n); }
    difference_type operator-(const constant_iterator& o) const { return idx_ - o.idx_; }

    bool operator==(const constant_iterator& o) const { return idx_ == o.idx_; }
    bool operator!=(const constant_iterator& o) const { return idx_ != o.idx_; }
    bool operator<(const constant_iterator& o) const { return idx_ < o.idx_; }
    bool operator>(const constant_iterator& o) const { return idx_ > o.idx_; }
    bool operator<=(const constant_iterator& o) const { return idx_ <= o.idx_; }
    bool operator>=(const constant_iterator& o) const { return idx_ >= o.idx_; }

private:
    T val_{};
    difference_type idx_ = 0;
};

template <typename T>
constant_iterator<T> make_constant_iterator(T val) {
    return constant_iterator<T>(val);
}

} // namespace thrust
