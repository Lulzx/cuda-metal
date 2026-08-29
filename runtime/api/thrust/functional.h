#pragma once

// CuMetal thrust shim: functional operators.

namespace thrust {

template <typename T = void>
struct plus {
    T operator()(const T& a, const T& b) const { return a + b; }
};
template <>
struct plus<void> {
    template <typename T>
    T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T = void>
struct multiplies {
    T operator()(const T& a, const T& b) const { return a * b; }
};
template <>
struct multiplies<void> {
    template <typename T>
    T operator()(const T& a, const T& b) const { return a * b; }
};

template <typename T = void>
struct minus {
    T operator()(const T& a, const T& b) const { return a - b; }
};

template <typename T = void>
struct maximum {
    T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};
template <>
struct maximum<void> {
    template <typename T>
    T operator()(const T& a, const T& b) const { return a > b ? a : b; }
};

template <typename T = void>
struct minimum {
    T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};
template <>
struct minimum<void> {
    template <typename T>
    T operator()(const T& a, const T& b) const { return a < b ? a : b; }
};

template <typename T = void>
struct negate {
    T operator()(const T& a) const { return -a; }
};

template <typename T = void>
struct identity {
    const T& operator()(const T& a) const { return a; }
};
template <>
struct identity<void> {
    template <typename T>
    const T& operator()(const T& a) const { return a; }
};

template <typename T = void>
struct equal_to {
    bool operator()(const T& a, const T& b) const { return a == b; }
};

template <typename T = void>
struct less {
    bool operator()(const T& a, const T& b) const { return a < b; }
};

template <typename T = void>
struct greater {
    bool operator()(const T& a, const T& b) const { return a > b; }
};

template <typename T = void>
struct greater_equal {
    bool operator()(const T& a, const T& b) const { return a >= b; }
};

template <typename T = void>
struct not_equal_to {
    bool operator()(const T& a, const T& b) const { return a != b; }
};

// Placeholder types used by thrust::transform with binary operations
struct placeholders {
    struct _1_t {} static constexpr _1{};
    struct _2_t {} static constexpr _2{};
};

} // namespace thrust
