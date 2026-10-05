#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__ 1
#endif


// CuMetal cuda_bf16.h — minimal bfloat16 compatibility shim for CUDA headers.
// This is header-compatibility focused and provides the subset used by ggml-cuda.

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus
#include "cumetal_float16_convert.h"

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#if defined(__clang__) || defined(__GNUC__)
#define __forceinline__ __inline__ __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

static __host__ __device__ __forceinline__ uint16_t __cumetal_float_to_bf16_bits(float f) {
    return cumetal_float16::from_float<8, 7>(f);
}

static __host__ __device__ __forceinline__ float __cumetal_bf16_bits_to_float(uint16_t bits16) {
    uint32_t bits = static_cast<uint32_t>(bits16) << 16;
    float out;
    __builtin_memcpy(&out, &bits, sizeof(out));
    return out;
}

struct __nv_bfloat16 {
    uint16_t __x;

    __host__ __device__ __forceinline__ __nv_bfloat16() = default;
    __host__ __device__ __forceinline__ __nv_bfloat16(float f)
        : __x(__cumetal_float_to_bf16_bits(f)) {}
    __host__ __device__ __forceinline__ operator float() const {
        return __cumetal_bf16_bits_to_float(__x);
    }
};

struct __attribute__((aligned(4))) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;
};

typedef __nv_bfloat16  nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

static_assert(sizeof(nv_bfloat16) == 2, "CuMetal nv_bfloat16 must be 16-bit");
static_assert(sizeof(nv_bfloat162) == 4, "CuMetal nv_bfloat162 must be 32-bit");

static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16(float f) {
    return nv_bfloat16(f);
}
static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16_rn(float f) {
    return nv_bfloat16(f);
}
static __host__ __device__ __forceinline__ float __bfloat162float(nv_bfloat16 h) {
    return static_cast<float>(h);
}

#ifdef CUMETAL_CUDA_VECTOR_TYPES_DEFINED
static __host__ __device__ __forceinline__ nv_bfloat162 __float22bfloat162_rn(float2 f) {
    return {nv_bfloat16(f.x), nv_bfloat16(f.y)};
}
#endif

static __host__ __device__ __forceinline__ nv_bfloat16 __nv_cvt_e8m0_to_bf16raw(uint8_t x) {
    nv_bfloat16 out;
    out.__x = static_cast<uint16_t>(x) << 7;
    return out;
}


CUMETAL_F16_HD inline __nv_bfloat16 __cumetal_bfloat_from_bits(uint16_t bits) {
    __nv_bfloat16 out; out.__x = bits; return out;
}
CUMETAL_F16_HD inline __nv_bfloat16 __double2bfloat16(double value) {
    return __cumetal_bfloat_from_bits(cumetal_float16::from_double<8, 7>(value));
}
CUMETAL_F16_HD inline __nv_bfloat16 __short2bfloat16_rn(short value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __nv_bfloat16 __ushort2bfloat16_rn(unsigned short value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(false, magnitude, 0));
}
CUMETAL_F16_HD inline __nv_bfloat16 __int2bfloat16_rn(int value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __nv_bfloat16 __uint2bfloat16_rn(unsigned int value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(false, magnitude, 0));
}
CUMETAL_F16_HD inline __nv_bfloat16 __ll2bfloat16_rn(long long value) {
    const uint64_t magnitude = value < 0 ? uint64_t(0) - static_cast<uint64_t>(value) : static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(value < 0, magnitude, 0));
}
CUMETAL_F16_HD inline __nv_bfloat16 __ull2bfloat16_rn(unsigned long long value) {
    const uint64_t magnitude = static_cast<uint64_t>(value);
    return __cumetal_bfloat_from_bits(cumetal_float16::pack<8, 7>(false, magnitude, 0));
}
CUMETAL_F16_HD inline short __bfloat162short_rz(__nv_bfloat16 value) {
    return static_cast<short>(cumetal_float16::signed_integer<16>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline unsigned short __bfloat162ushort_rz(__nv_bfloat16 value) {
    return static_cast<unsigned short>(cumetal_float16::unsigned_integer<16>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline int __bfloat162int_rz(__nv_bfloat16 value) {
    return static_cast<int>(cumetal_float16::signed_integer<32>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline unsigned int __bfloat162uint_rz(__nv_bfloat16 value) {
    return static_cast<unsigned int>(cumetal_float16::unsigned_integer<32>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline long long __bfloat162ll_rz(__nv_bfloat16 value) {
    return static_cast<long long>(cumetal_float16::signed_integer<64>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline unsigned long long __bfloat162ull_rz(__nv_bfloat16 value) {
    return static_cast<unsigned long long>(cumetal_float16::unsigned_integer<64>(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hexp(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_expf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hexp2(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_exp2f(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hlog(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_logf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hlog2(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_log2f(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hlog10(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_log10f(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hsqrt(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_sqrtf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hsin(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_sinf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hcos(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_cosf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hceil(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_ceilf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hfloor(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_floorf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 htrunc(__nv_bfloat16 value) {
    return __float2bfloat16_rn(__builtin_truncf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hrint(__nv_bfloat16 value) {
    return __float2bfloat16_rn(cumetal_float16::round_even(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 __habs(__nv_bfloat16 value) {
    value.__x &= 0x7fff; return value;
}
CUMETAL_F16_HD inline __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(__builtin_fmaxf(__bfloat162float(a), __bfloat162float(b)));
}
CUMETAL_F16_HD inline __nv_bfloat16 __hmin(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(__builtin_fminf(__bfloat162float(a), __bfloat162float(b)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hrsqrt(__nv_bfloat16 value) {
    return __float2bfloat16_rn(1.0f / __builtin_sqrtf(__bfloat162float(value)));
}
CUMETAL_F16_HD inline __nv_bfloat16 hrcp(__nv_bfloat16 value) {
    return __float2bfloat16_rn(1.0f / __bfloat162float(value));
}
CUMETAL_F16_HD inline bool __hisnan(__nv_bfloat16 value) { return __builtin_isnan(__bfloat162float(value)); }
CUMETAL_F16_HD inline bool __hisinf(__nv_bfloat16 value) { return __builtin_isinf(__bfloat162float(value)); }

#endif  // __cplusplus
