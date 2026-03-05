#pragma once
// CuMetal thrust shim: discard_iterator — sink that discards all writes.

#include <iterator>
#include <cstddef>

namespace thrust {

class discard_iterator {
    struct discard_ref {
        template <typename T> discard_ref& operator=(const T&) { return *this; }
    };
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = void;
    using difference_type = ptrdiff_t;
    using pointer = void*;
    using reference = discard_ref;

    discard_iterator() = default;
    explicit discard_iterator(difference_type idx) : idx_(idx) {}

    reference operator*() const { return discard_ref{}; }
    reference operator[](difference_type) const { return discard_ref{}; }

    discard_iterator& operator++() { ++idx_; return *this; }
    discard_iterator operator++(int) { auto tmp = *this; ++idx_; return tmp; }
    discard_iterator& operator--() { --idx_; return *this; }
    discard_iterator operator--(int) { auto tmp = *this; --idx_; return tmp; }

    discard_iterator& operator+=(difference_type n) { idx_ += n; return *this; }
    discard_iterator& operator-=(difference_type n) { idx_ -= n; return *this; }

    discard_iterator operator+(difference_type n) const { return discard_iterator(idx_ + n); }
    discard_iterator operator-(difference_type n) const { return discard_iterator(idx_ - n); }
    difference_type operator-(const discard_iterator& o) const { return idx_ - o.idx_; }

    bool operator==(const discard_iterator& o) const { return idx_ == o.idx_; }
    bool operator!=(const discard_iterator& o) const { return idx_ != o.idx_; }
    bool operator<(const discard_iterator& o) const { return idx_ < o.idx_; }

private:
    difference_type idx_ = 0;
};

inline discard_iterator make_discard_iterator() { return discard_iterator(); }

} // namespace thrust
