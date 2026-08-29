#pragma once

// CuMetal thrust shim: device_ptr — thin wrapper around raw pointer.
// On Apple Silicon UMA, device pointers are host-accessible.

#include <cstddef>
#include <iterator>

namespace thrust {

#if defined(__CUDACC__)
#define CUMETAL_THRUST_HD __host__ __device__
#else
#define CUMETAL_THRUST_HD
#endif

template <typename T>
class device_ptr {
    T* ptr_ = nullptr;
public:
    typedef T value_type;
    typedef T& reference;
    typedef T* pointer;
    typedef ptrdiff_t difference_type;
    typedef std::random_access_iterator_tag iterator_category;

    CUMETAL_THRUST_HD device_ptr() = default;
    CUMETAL_THRUST_HD explicit device_ptr(T* p) : ptr_(p) {}

    CUMETAL_THRUST_HD T* get() const { return ptr_; }
    CUMETAL_THRUST_HD operator T*() const { return ptr_; } // implicit conversion for UMA

    CUMETAL_THRUST_HD T& operator*() const { return *ptr_; }
    CUMETAL_THRUST_HD T* operator->() const { return ptr_; }
    CUMETAL_THRUST_HD T& operator[](ptrdiff_t i) const { return ptr_[i]; }

    CUMETAL_THRUST_HD device_ptr& operator++() { ++ptr_; return *this; }
    CUMETAL_THRUST_HD device_ptr operator++(int) { device_ptr t = *this; ++ptr_; return t; }
    CUMETAL_THRUST_HD device_ptr& operator--() { --ptr_; return *this; }
    CUMETAL_THRUST_HD device_ptr operator--(int) { device_ptr t = *this; --ptr_; return t; }
    CUMETAL_THRUST_HD device_ptr& operator+=(ptrdiff_t n) { ptr_ += n; return *this; }
    CUMETAL_THRUST_HD device_ptr& operator-=(ptrdiff_t n) { ptr_ -= n; return *this; }
    CUMETAL_THRUST_HD device_ptr operator+(ptrdiff_t n) const { return device_ptr(ptr_ + n); }
    CUMETAL_THRUST_HD device_ptr operator-(ptrdiff_t n) const { return device_ptr(ptr_ - n); }
    CUMETAL_THRUST_HD ptrdiff_t operator-(const device_ptr& o) const { return ptr_ - o.ptr_; }

    CUMETAL_THRUST_HD bool operator==(const device_ptr& o) const { return ptr_ == o.ptr_; }
    CUMETAL_THRUST_HD bool operator!=(const device_ptr& o) const { return ptr_ != o.ptr_; }
    CUMETAL_THRUST_HD bool operator<(const device_ptr& o) const { return ptr_ < o.ptr_; }
    CUMETAL_THRUST_HD bool operator>(const device_ptr& o) const { return ptr_ > o.ptr_; }
    CUMETAL_THRUST_HD bool operator<=(const device_ptr& o) const { return ptr_ <= o.ptr_; }
    CUMETAL_THRUST_HD bool operator>=(const device_ptr& o) const { return ptr_ >= o.ptr_; }
};

template <>
class device_ptr<void> {
    void* ptr_ = nullptr;
public:
    using value_type = void;
    using pointer = void*;
    using difference_type = ptrdiff_t;

    CUMETAL_THRUST_HD device_ptr() = default;
    CUMETAL_THRUST_HD explicit device_ptr(void* ptr) : ptr_(ptr) {}
    CUMETAL_THRUST_HD void* get() const { return ptr_; }
    CUMETAL_THRUST_HD operator void*() const { return ptr_; }
};

template <typename T>
device_ptr<T> device_pointer_cast(T* p) { return device_ptr<T>(p); }

template <typename T>
T* raw_pointer_cast(const device_ptr<T>& p) { return p.get(); }

template <typename T>
T* raw_pointer_cast(T* p) { return p; }

template <typename T>
const T* raw_pointer_cast(const T* p) { return p; }

inline void* raw_pointer_cast(const device_ptr<void>& p) { return p.get(); }

template <typename T>
void swap(device_ptr<T>& lhs, device_ptr<T>& rhs) {
    device_ptr<T> temporary = lhs;
    lhs = rhs;
    rhs = temporary;
}

} // namespace thrust

#undef CUMETAL_THRUST_HD
