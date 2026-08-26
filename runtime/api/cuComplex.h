#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CU_COMPLEX_H_
#define CU_COMPLEX_H_ 1
#endif


// CuMetal cuComplex shim: complex number types for CUDA.
// Compatible with NVIDIA's cuComplex.h used by cuFFT, cuSOLVER, and scientific computing.

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single-precision complex
typedef struct cuFloatComplex {
    float x; // real
    float y; // imaginary
} cuFloatComplex;

typedef cuFloatComplex cuComplex;

static inline cuFloatComplex make_cuFloatComplex(float r, float i) {
    cuFloatComplex c; c.x = r; c.y = i; return c;
}
static inline cuFloatComplex make_cuComplex(float r, float i) {
    return make_cuFloatComplex(r, i);
}
static inline float cuCrealf(cuFloatComplex z) { return z.x; }
static inline float cuCimagf(cuFloatComplex z) { return z.y; }
static inline cuFloatComplex cuConjf(cuFloatComplex z) {
    return make_cuFloatComplex(z.x, -z.y);
}
static inline cuFloatComplex cuCaddf(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x + b.x, a.y + b.y);
}
static inline cuFloatComplex cuCsubf(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x - b.x, a.y - b.y);
}
static inline cuFloatComplex cuCmulf(cuFloatComplex a, cuFloatComplex b) {
    return make_cuFloatComplex(a.x * b.x - a.y * b.y,
                               a.x * b.y + a.y * b.x);
}
static inline cuFloatComplex cuCdivf(cuFloatComplex a, cuFloatComplex b) {
    float d = b.x * b.x + b.y * b.y;
    return make_cuFloatComplex((a.x * b.x + a.y * b.y) / d,
                               (a.y * b.x - a.x * b.y) / d);
}
static inline float cuCabsf(cuFloatComplex z) {
    return sqrtf(z.x * z.x + z.y * z.y);
}
static inline cuFloatComplex cuCnegf(cuFloatComplex z) {
    return make_cuFloatComplex(-z.x, -z.y);
}

// Double-precision complex
typedef struct cuDoubleComplex {
    double x; // real
    double y; // imaginary
} cuDoubleComplex;

static inline cuDoubleComplex make_cuDoubleComplex(double r, double i) {
    cuDoubleComplex c; c.x = r; c.y = i; return c;
}
static inline double cuCreal(cuDoubleComplex z) { return z.x; }
static inline double cuCimag(cuDoubleComplex z) { return z.y; }
static inline cuDoubleComplex cuConj(cuDoubleComplex z) {
    return make_cuDoubleComplex(z.x, -z.y);
}
static inline cuDoubleComplex cuCadd(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}
static inline cuDoubleComplex cuCsub(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}
static inline cuDoubleComplex cuCmul(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y,
                                a.x * b.y + a.y * b.x);
}
static inline cuDoubleComplex cuCdiv(cuDoubleComplex a, cuDoubleComplex b) {
    double d = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / d,
                                (a.y * b.x - a.x * b.y) / d);
}
static inline double cuCabs(cuDoubleComplex z) {
    return sqrt(z.x * z.x + z.y * z.y);
}
static inline cuDoubleComplex cuCneg(cuDoubleComplex z) {
    return make_cuDoubleComplex(-z.x, -z.y);
}

// Conversion
static inline cuDoubleComplex cuComplexFloatToDouble(cuFloatComplex z) {
    return make_cuDoubleComplex((double)z.x, (double)z.y);
}
static inline cuFloatComplex cuComplexDoubleToFloat(cuDoubleComplex z) {
    return make_cuFloatComplex((float)z.x, (float)z.y);
}

#ifdef __cplusplus
}
#endif
