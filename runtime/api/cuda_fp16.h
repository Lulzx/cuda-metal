#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __CUDA_FP16_H__
#define __CUDA_FP16_H__ 1
#endif


// CuMetal cuda_fp16.h — half-precision float shim for Apple Silicon (UMA).
//
// On the host side, __half is a 16-bit IEEE 754 float stored as uint16_t.
// On the device side (when compiled by clang --cuda-gpu-arch via cumetalc),
// it is the native _Float16 type which Metal/AIR supports natively.
//
// Spec §8: "Half-precision atomics: Software emulation via CAS loop"

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#include "cumetal_float16_convert.h"

// ── Host-side __half (not compiled with CUDA device target) ─────────────────
#if !(defined(__clang__) && defined(__CUDA__))

struct __half {
    uint16_t __x;

    __half() = default;

    // Round once to binary16, including subnormals and NaNs.
    explicit __half(float f) : __x(cumetal_float16::from_float<5, 10>(f)) {}

    // Conversion to float
    explicit operator float() const {
        uint32_t sign = (__x >> 15) & 1u;
        uint32_t exp = (__x >> 10) & 0x1fu;
        uint32_t mant = __x & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign << 31;
            } else {
                int unbiased = -14;
                while ((mant & 0x400u) == 0) { mant <<= 1; --unbiased; }
                bits = (sign << 31) | (static_cast<uint32_t>(unbiased + 127) << 23) |
                       ((mant & 0x3ffu) << 13);
            }
        } else if (exp == 31) {
            bits = (sign << 31) | 0x7f800000u | (mant << 13);
        } else {
            bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        float f;
        memcpy(&f, &bits, 4);
        return f;
    }
};

struct __attribute__((aligned(4))) __half2 {
    __half x;
    __half y;

    __half2() = default;
    __half2(__half vx, __half vy) : x(vx), y(vy) {}
    __half2(float vx, float vy) : x(__half(vx)), y(__half(vy)) {}
};
typedef __half half;
typedef __half2 half2;

static_assert(sizeof(__half) == 2, "CuMetal __half must be 16-bit");
static_assert(sizeof(__half2) == 4, "CuMetal __half2 must be 32-bit");

inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
inline __half2 __float2half2_rn(float f) {
    const __half h = __float2half_rn(f);
    return {h, h};
}
inline float __half2float(const __half& h) { return static_cast<float>(h); }
inline __half2 make_half2(__half x, __half y) { return {x, y}; }
inline __half2 make_half2(float x, float y) { return {__float2half_rn(x), __float2half_rn(y)}; }
inline __half __low2half(const __half2& h2) { return h2.x; }
inline __half __high2half(const __half2& h2) { return h2.y; }
inline float __low2float(const __half2& h2) { return __half2float(h2.x); }
inline float __high2float(const __half2& h2) { return __half2float(h2.y); }
inline __half2 __half2half2(const __half& h) { return {h, h}; }
inline __half2 __hadd2(const __half2& a, const __half2& b) {
    return {__half(static_cast<float>(a.x) + static_cast<float>(b.x)),
            __half(static_cast<float>(a.y) + static_cast<float>(b.y))};
}
inline __half2 __hsub2(const __half2& a, const __half2& b) {
    return {__half(static_cast<float>(a.x) - static_cast<float>(b.x)),
            __half(static_cast<float>(a.y) - static_cast<float>(b.y))};
}
inline __half2 __hmul2(const __half2& a, const __half2& b) {
    return {__half(static_cast<float>(a.x) * static_cast<float>(b.x)),
            __half(static_cast<float>(a.y) * static_cast<float>(b.y))};
}
inline __half2 __hfma2(const __half2& a, const __half2& b, const __half2& c) {
    return {__half(static_cast<float>(a.x) * static_cast<float>(b.x) +
                   static_cast<float>(c.x)),
            __half(static_cast<float>(a.y) * static_cast<float>(b.y) +
                   static_cast<float>(c.y))};
}
inline __half2 __hmax2(const __half2& a, const __half2& b) {
    return {
        static_cast<float>(a.x) > static_cast<float>(b.x) ? a.x : b.x,
        static_cast<float>(a.y) > static_cast<float>(b.y) ? a.y : b.y
    };
}
#ifdef CUMETAL_CUDA_VECTOR_TYPES_DEFINED
inline __half2 __float22half2_rn(float2 f) {
    return {__float2half_rn(f.x), __float2half_rn(f.y)};
}
inline float2 __half22float2(const __half2& h2) {
    return {__half2float(h2.x), __half2float(h2.y)};
}
#endif

inline bool __hgt(const __half& a, const __half& b) {
    return static_cast<float>(a) > static_cast<float>(b);
}
inline bool __hlt(const __half& a, const __half& b) {
    return static_cast<float>(a) < static_cast<float>(b);
}
inline bool __hge(const __half& a, const __half& b) {
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline bool __hle(const __half& a, const __half& b) {
    return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool __heq(const __half& a, const __half& b) { return a.__x == b.__x; }
inline bool __hne(const __half& a, const __half& b) { return a.__x != b.__x; }

inline __half __hadd(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) + static_cast<float>(b));
}
inline __half __hmul(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) * static_cast<float>(b));
}
inline __half __hsub(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) - static_cast<float>(b));
}
inline __half __hdiv(const __half& a, const __half& b) {
    return __half(static_cast<float>(a) / static_cast<float>(b));
}
inline __half __hfma(const __half& a, const __half& b, const __half& c) {
    return __half(static_cast<float>(a) * static_cast<float>(b) + static_cast<float>(c));
}
inline __half __hneg(const __half& a) {
    return __half(-static_cast<float>(a));
}
inline __half __habs(const __half& a) {
    __half r = a; r.__x &= 0x7fffu; return r;
}
inline __half __hmax(const __half& a, const __half& b) {
    return static_cast<float>(a) > static_cast<float>(b) ? a : b;
}
inline __half __hmin(const __half& a, const __half& b) {
    return static_cast<float>(a) < static_cast<float>(b) ? a : b;
}

inline __half operator+(const __half& a, const __half& b) { return __hadd(a, b); }
inline __half operator-(const __half& a, const __half& b) { return __hsub(a, b); }
inline __half operator*(const __half& a, const __half& b) { return __hmul(a, b); }
inline __half operator/(const __half& a, const __half& b) { return __hdiv(a, b); }
inline bool operator==(const __half& a, const __half& b) { return __heq(a, b); }
inline bool operator!=(const __half& a, const __half& b) { return __hne(a, b); }
inline bool operator>(const __half& a, const __half& b) { return __hgt(a, b); }
inline bool operator<(const __half& a, const __half& b) { return __hlt(a, b); }
inline __half2 operator+(const __half2& a, const __half2& b) { return __hadd2(a, b); }
inline __half2 operator-(const __half2& a, const __half2& b) { return __hsub2(a, b); }
inline __half2 operator*(const __half2& a, const __half2& b) { return __hmul2(a, b); }
inline __half2& operator+=(__half2& a, const __half2& b) { a = __hadd2(a, b); return a; }
inline __half2& operator-=(__half2& a, const __half2& b) { a = __hsub2(a, b); return a; }
inline __half2& operator*=(__half2& a, const __half2& b) { a = __hmul2(a, b); return a; }

#else  // Device code path (clang CUDA)

// When compiling device code, use the native _Float16 / __fp16 type.
// These are defined by the compiler; no additional typedef needed.
typedef _Float16 __half;
typedef struct __attribute__((aligned(4))) __half2 {
    __half x;
    __half y;
#ifdef __cplusplus
    __host__ __device__ __half2() = default;
    __host__ __device__ __half2(__half vx, __half vy) : x(vx), y(vy) {}
#endif
} __half2;
typedef __half half;
typedef __half2 half2;

static __host__ __device__ __forceinline__ __half __float2half(float f) {
    return static_cast<__half>(f);
}
static __host__ __device__ __forceinline__ __half __float2half_rn(float f) {
    return static_cast<__half>(f);
}
static __device__ __forceinline__ __half2 __float2half2_rn(float f) {
    const __half h = __float2half_rn(f);
    return {h, h};
}
static __host__ __device__ __forceinline__ float __half2float(__half h) {
    return static_cast<float>(h);
}
static __device__ __forceinline__ __half2 make_half2(__half x, __half y) { return {x, y}; }
static __device__ __forceinline__ __half2 __floats2half2_rn(float x, float y) {
    return {__float2half_rn(x), __float2half_rn(y)};
}
static __device__ __forceinline__ __half __low2half(__half2 h2) { return h2.x; }
static __device__ __forceinline__ __half __high2half(__half2 h2) { return h2.y; }
static __host__ __device__ __forceinline__ float __low2float(__half2 h2) { return __half2float(h2.x); }
static __host__ __device__ __forceinline__ float __high2float(__half2 h2) { return __half2float(h2.y); }
static __device__ __forceinline__ __half2 __half2half2(__half h) { return {h, h}; }
static __device__ __forceinline__ __half2 __hadd2(__half2 a, __half2 b) {
    return {a.x + b.x, a.y + b.y};
}
static __device__ __forceinline__ __half2 __hsub2(__half2 a, __half2 b) {
    return {a.x - b.x, a.y - b.y};
}
static __device__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) {
    return {a.x * b.x, a.y * b.y};
}
static __device__ __forceinline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) {
    return {
        static_cast<__half>(__builtin_fmaf(static_cast<float>(a.x),
                                           static_cast<float>(b.x),
                                           static_cast<float>(c.x))),
        static_cast<__half>(__builtin_fmaf(static_cast<float>(a.y),
                                           static_cast<float>(b.y),
                                           static_cast<float>(c.y)))
    };
}
static __device__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b) {
    return {a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y};
}
#ifdef CUMETAL_CUDA_VECTOR_TYPES_DEFINED
static __device__ __forceinline__ __half2 __float22half2_rn(float2 f) {
    return {__float2half_rn(f.x), __float2half_rn(f.y)};
}
static __device__ __forceinline__ float2 __half22float2(__half2 h2) {
    return {__half2float(h2.x), __half2float(h2.y)};
}
static __device__ __forceinline__ __half2 __shfl_sync(unsigned int mask, __half2 val, int srcLane, int width = 32) {
    float2 f = __half22float2(val);
    int x_bits, y_bits, out_bits;
    __builtin_memcpy(&x_bits, &f.x, sizeof(x_bits));
    out_bits = __cumetal_shfl_sync_idx_i32(mask, x_bits, srcLane, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.x, &out_bits, sizeof(f.x));
    __builtin_memcpy(&y_bits, &f.y, sizeof(y_bits));
    out_bits = __cumetal_shfl_sync_idx_i32(mask, y_bits, srcLane, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.y, &out_bits, sizeof(f.y));
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_down_sync(unsigned int mask, __half2 val, unsigned int delta, int width = 32) {
    float2 f = __half22float2(val);
    int x_bits, y_bits, out_bits;
    __builtin_memcpy(&x_bits, &f.x, sizeof(x_bits));
    out_bits = __cumetal_shfl_sync_down_i32(mask, x_bits, delta, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.x, &out_bits, sizeof(f.x));
    __builtin_memcpy(&y_bits, &f.y, sizeof(y_bits));
    out_bits = __cumetal_shfl_sync_down_i32(mask, y_bits, delta, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.y, &out_bits, sizeof(f.y));
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_up_sync(unsigned int mask, __half2 val, unsigned int delta, int width = 32) {
    float2 f = __half22float2(val);
    int x_bits, y_bits, out_bits;
    __builtin_memcpy(&x_bits, &f.x, sizeof(x_bits));
    out_bits = __cumetal_shfl_sync_up_i32(mask, x_bits, delta, ((32 - width) << 8) | 0);
    __builtin_memcpy(&f.x, &out_bits, sizeof(f.x));
    __builtin_memcpy(&y_bits, &f.y, sizeof(y_bits));
    out_bits = __cumetal_shfl_sync_up_i32(mask, y_bits, delta, ((32 - width) << 8) | 0);
    __builtin_memcpy(&f.y, &out_bits, sizeof(f.y));
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_xor_sync(unsigned int mask, __half2 val, int laneMask, int width = 32) {
    float2 f = __half22float2(val);
    unsigned int laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    const int srcLane = static_cast<int>(laneid) ^ laneMask;
    int x_bits, y_bits, out_bits;
    __builtin_memcpy(&x_bits, &f.x, sizeof(x_bits));
    out_bits = __cumetal_shfl_sync_idx_i32(mask, x_bits, srcLane, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.x, &out_bits, sizeof(f.x));
    __builtin_memcpy(&y_bits, &f.y, sizeof(y_bits));
    out_bits = __cumetal_shfl_sync_idx_i32(mask, y_bits, srcLane, ((32 - width) << 8) | 0x1f);
    __builtin_memcpy(&f.y, &out_bits, sizeof(f.y));
    return __float22half2_rn(f);
}
#endif

static __device__ __forceinline__ __half __hadd(__half a, __half b) { return a + b; }
static __device__ __forceinline__ __half __hmul(__half a, __half b) { return a * b; }
static __device__ __forceinline__ __half __hsub(__half a, __half b) { return a - b; }
static __device__ __forceinline__ __half __hdiv(__half a, __half b) { return a / b; }
static __device__ __forceinline__ __half __hfma(__half a, __half b, __half c) {
    return static_cast<__half>(__builtin_fmaf(static_cast<float>(a), static_cast<float>(b), static_cast<float>(c)));
}
static __device__ __forceinline__ __half __hneg(__half a) { return -a; }
static __device__ __forceinline__ __half __habs(__half a) { return a < (__half)0.0f ? -a : a; }
static __device__ __forceinline__ __half __hmax(__half a, __half b) { return a > b ? a : b; }
static __device__ __forceinline__ __half __hmin(__half a, __half b) { return a < b ? a : b; }
static __device__ __forceinline__ bool __hgt(__half a, __half b) { return a > b; }
static __device__ __forceinline__ bool __hlt(__half a, __half b) { return a < b; }
static __device__ __forceinline__ bool __hge(__half a, __half b) { return a >= b; }
static __device__ __forceinline__ bool __hle(__half a, __half b) { return a <= b; }
static __device__ __forceinline__ bool __heq(__half a, __half b) { return a == b; }
static __device__ __forceinline__ bool __hne(__half a, __half b) { return a != b; }
static __device__ __forceinline__ __half2 operator+(__half2 a, __half2 b) { return __hadd2(a, b); }
static __device__ __forceinline__ __half2 operator-(__half2 a, __half2 b) { return __hsub2(a, b); }
static __device__ __forceinline__ __half2 operator*(__half2 a, __half2 b) { return __hmul2(a, b); }
static __device__ __forceinline__ __half2& operator+=(__half2& a, __half2 b) { a = __hadd2(a, b); return a; }
static __device__ __forceinline__ __half2& operator-=(__half2& a, __half2 b) { a = __hsub2(a, b); return a; }
static __device__ __forceinline__ __half2& operator*=(__half2& a, __half2 b) { a = __hmul2(a, b); return a; }
// atomicAdd for __half via CAS loop (spec §8: "Software emulation via CAS loop").
// Uses the 32-bit word containing the 16-bit element for the CAS operation.
static __device__ __forceinline__ __half atomicAdd(__half* addr, __half val) {
    // Map the half address to the containing 32-bit word (must be 2-byte aligned).
    unsigned int* base = reinterpret_cast<unsigned int*>(
        reinterpret_cast<uintptr_t>(addr) & ~static_cast<uintptr_t>(2));
    const bool high = (reinterpret_cast<uintptr_t>(addr) & 2) != 0;

    unsigned int assumed;
    unsigned int old = *base;
    do {
        assumed = old;
        unsigned short existing_bits = high ? (assumed >> 16) : (assumed & 0xffffu);
        __half existing;
        __builtin_memcpy(&existing, &existing_bits, 2);
        __half new_val = existing + val;
        unsigned short new_bits;
        __builtin_memcpy(&new_bits, &new_val, 2);
        unsigned int updated = high ? ((assumed & 0x0000ffffu) | (static_cast<unsigned int>(new_bits) << 16))
                                    : ((assumed & 0xffff0000u) | static_cast<unsigned int>(new_bits));
        old = __uAtomicCAS(base, assumed, updated);
    } while (old != assumed);

    unsigned short result_bits = high ? (old >> 16) : (old & 0xffffu);
    __half result;
    __builtin_memcpy(&result, &result_bits, 2);
    return result;
}

#endif  // device vs host


// Conversion entry points needed by templated CUDA libraries. Keep host and
// device behavior identical; integer and double inputs never round via FP32.
CUMETAL_F16_HD inline __half __cumetal_half_from_bits(uint16_t bits) {
    __half out;
#if defined(__clang__) && defined(__CUDA__)
    __builtin_memcpy(&out, &bits, sizeof(bits));
#else
    out.__x = bits;
#endif
    return out;
}
CUMETAL_F16_HD inline __half __double2half(double value) {
    return __cumetal_half_from_bits(cumetal_float16::from_double<5, 10>(value));
}
CUMETAL_F16_HD inline __half __short2half_rn(short value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __half __ushort2half_rn(unsigned short value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(false, magnitude, 0));
}
CUMETAL_F16_HD inline __half __int2half_rn(int value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __half __uint2half_rn(unsigned int value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(false, magnitude, 0));
}
CUMETAL_F16_HD inline __half __ll2half_rn(long long value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __half __ull2half_rn(unsigned long long value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_half_from_bits(cumetal_float16::pack<5, 10>(false, magnitude, 0));
}
CUMETAL_F16_HD inline short __half2short_rz(__half value) {
    return static_cast<short>(cumetal_float16::signed_integer<16, false>(__half2float(value)));
}
CUMETAL_F16_HD inline short __half2short_rn(__half value) {
    return static_cast<short>(cumetal_float16::signed_integer<16, true>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned short __half2ushort_rz(__half value) {
    return static_cast<unsigned short>(cumetal_float16::unsigned_integer<16, false>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned short __half2ushort_rn(__half value) {
    return static_cast<unsigned short>(cumetal_float16::unsigned_integer<16, true>(__half2float(value)));
}
CUMETAL_F16_HD inline int __half2int_rz(__half value) {
    return static_cast<int>(cumetal_float16::signed_integer<32, false>(__half2float(value)));
}
CUMETAL_F16_HD inline int __half2int_rn(__half value) {
    return static_cast<int>(cumetal_float16::signed_integer<32, true>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned int __half2uint_rz(__half value) {
    return static_cast<unsigned int>(cumetal_float16::unsigned_integer<32, false>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned int __half2uint_rn(__half value) {
    return static_cast<unsigned int>(cumetal_float16::unsigned_integer<32, true>(__half2float(value)));
}
CUMETAL_F16_HD inline long long __half2ll_rz(__half value) {
    return static_cast<long long>(cumetal_float16::signed_integer<64, false>(__half2float(value)));
}
CUMETAL_F16_HD inline long long __half2ll_rn(__half value) {
    return static_cast<long long>(cumetal_float16::signed_integer<64, true>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned long long __half2ull_rz(__half value) {
    return static_cast<unsigned long long>(cumetal_float16::unsigned_integer<64, false>(__half2float(value)));
}
CUMETAL_F16_HD inline unsigned long long __half2ull_rn(__half value) {
    return static_cast<unsigned long long>(cumetal_float16::unsigned_integer<64, true>(__half2float(value)));
}
CUMETAL_F16_HD inline __half hexp(__half value) {
    return __float2half_rn(__builtin_expf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hexp2(__half value) {
    return __float2half_rn(__builtin_exp2f(__half2float(value)));
}
CUMETAL_F16_HD inline __half hlog(__half value) {
    return __float2half_rn(__builtin_logf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hlog2(__half value) {
    return __float2half_rn(__builtin_log2f(__half2float(value)));
}
CUMETAL_F16_HD inline __half hlog10(__half value) {
    return __float2half_rn(__builtin_log10f(__half2float(value)));
}
CUMETAL_F16_HD inline __half hsqrt(__half value) {
    return __float2half_rn(__builtin_sqrtf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hsin(__half value) {
    return __float2half_rn(__builtin_sinf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hcos(__half value) {
    return __float2half_rn(__builtin_cosf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hceil(__half value) {
    return __float2half_rn(__builtin_ceilf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hfloor(__half value) {
    return __float2half_rn(__builtin_floorf(__half2float(value)));
}
CUMETAL_F16_HD inline __half htrunc(__half value) {
    return __float2half_rn(__builtin_truncf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hrint(__half value) {
    return __float2half_rn(cumetal_float16::round_even(__half2float(value)));
}
CUMETAL_F16_HD inline __half hrsqrt(__half value) {
    return __float2half_rn(1.0f / __builtin_sqrtf(__half2float(value)));
}
CUMETAL_F16_HD inline __half hrcp(__half value) {
    return __float2half_rn(1.0f / __half2float(value));
}
CUMETAL_F16_HD inline bool __hisnan(__half value) { return __builtin_isnan(__half2float(value)); }
CUMETAL_F16_HD inline bool __hisinf(__half value) { return __builtin_isinf(__half2float(value)); }

#else  // !__cplusplus

// ── C-mode __half ───────────────────────────────────────────────────────────
// Everything above is C++-only, but cublas_v2.h declares cublasHgemm and
// friends in terms of __half and is included from plain C translation units
// (cuPDLP-C compiles its .c sources with the host compiler, not nvcc). NVIDIA's
// own cuda_fp16.h exposes the storage type to C for exactly this reason, so a
// C consumer can name the type in a prototype even though it cannot do half
// arithmetic. Match that: opaque 16-bit storage, no operators.
typedef struct __half {
    uint16_t __x;
} __half;

typedef struct __attribute__((aligned(4))) __half2 {
    __half x;
    __half y;
} __half2;

typedef __half half;
typedef __half2 half2;

#endif  // __cplusplus
