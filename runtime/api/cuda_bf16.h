#pragma once

// CuMetal cuda_bf16.h — minimal bfloat16 compatibility shim for CUDA headers.
// This is header-compatibility focused and provides the subset used by ggml-cuda.

#include <stdint.h>
#include <string.h>

#ifdef __cplusplus

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
    uint32_t bits;
    __builtin_memcpy(&bits, &f, sizeof(bits));

    // Round-to-nearest-even on truncation to BF16.
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
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

#endif  // __cplusplus
