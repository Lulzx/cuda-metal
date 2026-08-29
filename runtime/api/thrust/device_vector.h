#pragma once

// CuMetal thrust shim: device_vector backed by cudaMalloc on UMA.
// On Apple Silicon, device_vector is just a thin wrapper over host memory.

#include "device_ptr.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <vector>
#include "generate.h"
#include "iterator/zip_iterator.h"

namespace thrust {

template <typename T>
class device_vector {
    T* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;

    void free_() {
        if (data_) { cudaFree(data_); data_ = nullptr; }
        size_ = capacity_ = 0;
    }
    void alloc_(size_t n) {
        if (n > capacity_) {
            free_();
            cudaMalloc(&data_, n * sizeof(T));
            capacity_ = n;
        }
        size_ = n;
    }

public:
    typedef T value_type;
    typedef device_ptr<T> iterator;
    typedef device_ptr<const T> const_iterator;

    device_vector() = default;
    explicit device_vector(size_t n) { alloc_(n); cudaMemset(data_, 0, n * sizeof(T)); }
    device_vector(size_t n, const T& val) {
        alloc_(n);
        std::vector<T> tmp(n, val);
        cudaMemcpy(data_, tmp.data(), n * sizeof(T), cudaMemcpyHostToDevice);
    }
    device_vector(const std::vector<T>& v) {
        alloc_(v.size());
        cudaMemcpy(data_, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
    }
    device_vector(std::initializer_list<T> values) {
        alloc_(values.size());
        if (values.size() != 0) {
            cudaMemcpy(data_, values.begin(), values.size() * sizeof(T), cudaMemcpyHostToDevice);
        }
    }
    device_vector(const device_vector& o) {
        alloc_(o.size_);
        cudaMemcpy(data_, o.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
    }
    device_vector& operator=(const device_vector& o) {
        if (this != &o) {
            alloc_(o.size_);
            cudaMemcpy(data_, o.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
        }
        return *this;
    }
    ~device_vector() { free_(); }

    void resize(size_t n) { alloc_(n); }
    void resize(size_t n, const T& val) {
        size_t old = size_;
        alloc_(n);
        if (n > old) {
            std::vector<T> fill(n - old, val);
            cudaMemcpy(data_ + old, fill.data(), fill.size() * sizeof(T), cudaMemcpyHostToDevice);
        }
    }
    void clear() { size_ = 0; }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    T* data() { return data_; }
    const T* data() const { return data_; }

    device_ptr<T> begin() { return device_ptr<T>(data_); }
    device_ptr<T> end() { return device_ptr<T>(data_ + size_); }
    device_ptr<const T> begin() const { return device_ptr<const T>(data_); }
    device_ptr<const T> end() const { return device_ptr<const T>(data_ + size_); }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
};

} // namespace thrust
