#pragma once
// CuMetal thrust shim: permutation_iterator — accesses elements via index indirection.

#include <iterator>

namespace thrust {

template <typename ElementIterator, typename IndexIterator>
class permutation_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::iterator_traits<ElementIterator>::value_type;
    using difference_type = typename std::iterator_traits<IndexIterator>::difference_type;
    using pointer = typename std::iterator_traits<ElementIterator>::pointer;
    using reference = typename std::iterator_traits<ElementIterator>::reference;

    permutation_iterator() = default;
    permutation_iterator(ElementIterator elements, IndexIterator indices)
        : elements_(elements), indices_(indices) {}

    reference operator*() const { return elements_[*indices_]; }
    reference operator[](difference_type n) const { return elements_[indices_[n]]; }

    permutation_iterator& operator++() { ++indices_; return *this; }
    permutation_iterator operator++(int) { auto tmp = *this; ++indices_; return tmp; }
    permutation_iterator& operator--() { --indices_; return *this; }
    permutation_iterator operator--(int) { auto tmp = *this; --indices_; return tmp; }

    permutation_iterator& operator+=(difference_type n) { indices_ += n; return *this; }
    permutation_iterator& operator-=(difference_type n) { indices_ -= n; return *this; }

    permutation_iterator operator+(difference_type n) const { return permutation_iterator(elements_, indices_ + n); }
    permutation_iterator operator-(difference_type n) const { return permutation_iterator(elements_, indices_ - n); }
    difference_type operator-(const permutation_iterator& o) const { return indices_ - o.indices_; }

    bool operator==(const permutation_iterator& o) const { return indices_ == o.indices_; }
    bool operator!=(const permutation_iterator& o) const { return indices_ != o.indices_; }
    bool operator<(const permutation_iterator& o) const { return indices_ < o.indices_; }

private:
    ElementIterator elements_{};
    IndexIterator indices_{};
};

template <typename ElementIterator, typename IndexIterator>
permutation_iterator<ElementIterator, IndexIterator>
make_permutation_iterator(ElementIterator elements, IndexIterator indices) {
    return permutation_iterator<ElementIterator, IndexIterator>(elements, indices);
}

} // namespace thrust
