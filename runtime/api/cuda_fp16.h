#pragma once

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

// ── Host-side __half (not compiled with CUDA device target) ─────────────────
#if !(defined(__clang__) && defined(__CUDA__))

struct __half {
    uint16_t __x;

    __half() = default;

    // Conversion from float
    explicit __half(float f) {
        // IEEE 754 float-to-half conversion
        uint32_t bits;
        memcpy(&bits, &f, 4);
        uint32_t sign = (bits >> 16) & 0x8000u;
        int32_t exp = static_cast<int32_t>((bits >> 23) & 0xff) - 127 + 15;
        uint32_t mant = bits & 0x7fffffu;
        if (exp <= 0) {
            __x = static_cast<uint16_t>(sign);
        } else if (exp >= 31) {
            __x = static_cast<uint16_t>(sign | 0x7c00u);
        } else {
            __x = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
        }
    }

    // Conversion to float
    explicit operator float() const {
        uint32_t sign = (__x >> 15) & 1u;
        uint32_t exp = (__x >> 10) & 0x1fu;
        uint32_t mant = __x & 0x3ffu;
        uint32_t bits;
        if (exp == 0) {
            bits = (sign << 31);
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
};
typedef __half half;
typedef __half2 half2;

static_assert(sizeof(__half) == 2, "CuMetal __half must be 16-bit");
static_assert(sizeof(__half2) == 4, "CuMetal __half2 must be 32-bit");

inline __half __float2half(float f) { return __half(f); }
inline __half __float2half_rn(float f) { return __half(f); }
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
// Half ↔ integer conversions.
inline int   __half2int_rn(const __half& h)    { return static_cast<int>(static_cast<float>(h)); }
inline unsigned int __half2uint_rn(const __half& h) { return static_cast<unsigned int>(static_cast<float>(h)); }
inline short __half2short_rn(const __half& h)  { return static_cast<short>(static_cast<float>(h)); }
inline long long __half2ll_rn(const __half& h) { return static_cast<long long>(static_cast<float>(h)); }
inline __half __int2half_rn(int i)   { return __half(static_cast<float>(i)); }
inline __half __uint2half_rn(unsigned int i) { return __half(static_cast<float>(i)); }
inline __half __short2half_rn(short s) { return __half(static_cast<float>(s)); }
inline __half __ll2half_rn(long long l) { return __half(static_cast<float>(l)); }

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

#else  // Device code path (clang CUDA)

// When compiling device code, use the native _Float16 / __fp16 type.
// These are defined by the compiler; no additional typedef needed.
typedef _Float16 __half;
typedef struct __attribute__((aligned(4))) __half2 {
    __half x;
    __half y;
} __half2;
typedef __half half;
typedef __half2 half2;

static __device__ __forceinline__ __half __float2half(float f) {
    return static_cast<__half>(f);
}
static __device__ __forceinline__ __half __float2half_rn(float f) {
    return static_cast<__half>(f);
}
static __device__ __forceinline__ float __half2float(__half h) {
    return static_cast<float>(h);
}
static __device__ __forceinline__ __half2 make_half2(__half x, __half y) { return {x, y}; }
static __device__ __forceinline__ __half2 make_half2(float x, float y) {
    return {__float2half_rn(x), __float2half_rn(y)};
}
static __device__ __forceinline__ __half __low2half(__half2 h2) { return h2.x; }
static __device__ __forceinline__ __half __high2half(__half2 h2) { return h2.y; }
static __device__ __forceinline__ float __low2float(__half2 h2) { return __half2float(h2.x); }
static __device__ __forceinline__ float __high2float(__half2 h2) { return __half2float(h2.y); }
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
    (void)mask;
    (void)width;
    f.x = __nvvm_shfl_idx_f32(f.x, srcLane, 0x1f);
    f.y = __nvvm_shfl_idx_f32(f.y, srcLane, 0x1f);
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_down_sync(unsigned int mask, __half2 val, unsigned int delta, int width = 32) {
    float2 f = __half22float2(val);
    (void)mask;
    (void)width;
    f.x = __nvvm_shfl_down_f32(f.x, delta, 0x1f);
    f.y = __nvvm_shfl_down_f32(f.y, delta, 0x1f);
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_up_sync(unsigned int mask, __half2 val, unsigned int delta, int width = 32) {
    float2 f = __half22float2(val);
    (void)mask;
    (void)width;
    f.x = __nvvm_shfl_up_f32(f.x, delta, 0);
    f.y = __nvvm_shfl_up_f32(f.y, delta, 0);
    return __float22half2_rn(f);
}
static __device__ __forceinline__ __half2 __shfl_xor_sync(unsigned int mask, __half2 val, int laneMask, int width = 32) {
    float2 f = __half22float2(val);
    (void)mask;
    (void)width;
    f.x = __nvvm_shfl_bfly_f32(f.x, laneMask, 0x1f);
    f.y = __nvvm_shfl_bfly_f32(f.y, laneMask, 0x1f);
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
// Half ↔ integer conversions for device code.
static __device__ __forceinline__ int __half2int_rn(__half h) { return static_cast<int>(h); }
static __device__ __forceinline__ unsigned int __half2uint_rn(__half h) { return static_cast<unsigned int>(h); }
static __device__ __forceinline__ __half __int2half_rn(int i) { return static_cast<__half>(i); }
static __device__ __forceinline__ __half __uint2half_rn(unsigned int i) { return static_cast<__half>(i); }

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

#endif  // __cplusplus
