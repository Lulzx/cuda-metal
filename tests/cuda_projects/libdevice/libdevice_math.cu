// Libdevice math surface probe.
//
// The PTX->LLVM lowering maps CUDA math calls (__nv_sqrtf, __nv_tanf, ...) onto
// Metal AIR builtins. Those mappings were previously asserted only by checking
// that the emitted IR text contained the expected symbol name -- which proves
// nothing about whether the AIR symbol exists, links, or computes the right
// function. A wrong or missing mapping does not degrade gracefully: the whole
// kernel fails to lower, or links to something that silently computes garbage.
//
// This harness measures the surface instead of asserting it. Every function
// gets its OWN kernel, so one unsupported call cannot mask the rest, and each
// is scored independently:
//
//   SUPPORTED   launched and matched the host libm within tolerance
//   WRONG       launched but the numbers disagree -> mis-mapped builtin
//   UNSUPPORTED failed to lower -> mapping missing from lower_to_llvm.cpp
//
// Every input is confined to (0,1) and each entry folds its own domain shift
// into the expression, so the same text is valid on host and device.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>

#define N 1024
#define BLOCK 128

// Host references for the inverse-error-function family: the system libm
// lacks erfinv/erfcinv/normcdfinv, so these invert erf/erfc by Newton
// iteration -- deliberately a different algorithm than the Acklam expansion
// the device side runs, so a mis-wired device formula cannot pass silently.
static double host_erfinv(double y) {
    if (std::isnan(y)) return y;
    if (y <= -1.0) return -INFINITY;
    if (y >= 1.0) return INFINITY;
    double t = 0.7 * y;
    for (int i = 0; i < 40; ++i) {
        t -= (erf(t) - y) / (1.1283791670955126 * exp(-t * t));
    }
    return t;
}
static double host_erfcinv(double y) {
    if (std::isnan(y)) return y;
    if (y <= 0.0) return INFINITY;
    if (y >= 2.0) return -INFINITY;
    double t = 0.5;
    for (int i = 0; i < 60; ++i) {
        t += (erfc(t) - y) / (1.1283791670955126 * exp(-t * t));
    }
    return t;
}
static double host_normcdfinv(double p) {
    // normcdf(x) = 0.5 * erfc(-x / sqrt(2)); invert through erfcinv:
    // erfc(-x/sqrt(2)) = 2p  =>  x = -sqrt(2) * erfcinv(2p).
    if (std::isnan(p)) return p;
    if (p <= 0.0) return -INFINITY;
    if (p >= 1.0) return INFINITY;
    return -host_erfcinv(2.0 * p) * 1.4142135623730951;
}

// id, device expression in x/y, host expression in x/y, relative tolerance.
//
// Tolerances split into two regimes. Exact ops (copies, sign, rounding, IEEE
// sqrt) must match to the bit. Everything routed through a Metal `fast_*`
// builtin is a lower-precision approximation by construction, so it gets 2e-3
// relative -- tight enough that a mis-mapped function (e.g. sin bound to cos)
// fails by orders of magnitude, loose enough not to flag honest fast-math error.
#define UNARY_LIST(X)                                                          \
    X(sqrtf,      sqrtf(x),         sqrtf(x),                     1e-6f)       \
    X(rsqrtf,     rsqrtf(x),        1.0f / sqrtf(x),              2e-3f)       \
    X(fabsf,      fabsf(x),         fabsf(x),                     0.0f)        \
    X(expf,       expf(x),          expf(x),                      2e-3f)       \
    X(exp2f,      exp2f(x),         exp2f(x),                     2e-3f)       \
    X(exp10f,     exp10f(x),        powf(10.0f, x),               2e-3f)       \
    X(expm1f,     expm1f(x),        expm1f(x),                    2e-3f)       \
    X(logf,       logf(x),          logf(x),                      2e-3f)       \
    X(log2f,      log2f(x),         log2f(x),                     2e-3f)       \
    X(log10f,     log10f(x),        log10f(x),                    2e-3f)       \
    X(log1pf,     log1pf(x),        log1pf(x),                    2e-3f)       \
    X(sinf,       sinf(x),          sinf(x),                      2e-3f)       \
    X(cosf,       cosf(x),          cosf(x),                      2e-3f)       \
    X(tanf,       tanf(x),          tanf(x),                      2e-3f)       \
    X(asinf,      asinf(x),         asinf(x),                     2e-3f)       \
    X(acosf,      acosf(x),         acosf(x),                     2e-3f)       \
    X(atanf,      atanf(x),         atanf(x),                     2e-3f)       \
    X(sinhf,      sinhf(x),         sinhf(x),                     2e-3f)       \
    X(coshf,      coshf(x),         coshf(x),                     2e-3f)       \
    X(tanhf,      tanhf(x),         tanhf(x),                     2e-3f)       \
    X(asinhf,     asinhf(x),        asinhf(x),                    2e-3f)       \
    X(acoshf,     acoshf(1.0f + x), acoshf(1.0f + x),             2e-3f)       \
    X(atanhf,     atanhf(x),        atanhf(x),                    2e-3f)       \
    X(cbrtf,      cbrtf(x),         cbrtf(x),                     2e-3f)       \
    X(erff,       erff(x),          erff(x),                      2e-3f)       \
    X(erfcf,      erfcf(x),         erfcf(x),                     2e-3f)       \
    X(floorf,     floorf(4.0f * x), floorf(4.0f * x),             0.0f)        \
    X(ceilf,      ceilf(4.0f * x),  ceilf(4.0f * x),              0.0f)        \
    X(truncf,     truncf(4.0f * x), truncf(4.0f * x),             0.0f)        \
    X(roundf,     roundf(4.0f * x), roundf(4.0f * x),             0.0f)        \
    X(rintf,      rintf(4.0f * x),  rintf(4.0f * x),              0.0f)       \
    /* Double rsqrt. GROMACS's nbnxm kernels call it, and until it was mapped   \
       an unlisted libdevice name aborted the whole kernel rather than          \
       degrading. Evaluated through the FP64 pair path, hence the fast-math     \
       tolerance. */                                                           \
    X(rsqrt_d,    (float)rsqrt((double)(x + 0.25f)),                            \
                  (float)(1.0 / sqrt((double)(x + 0.25f))),       2e-3f)       \
    /* Float->int conversions. The suffix names the rounding mode, which has to \
       be applied to the float before the cast; dropping it is the same defect  \
       class as cvt.rni silently truncating. The 4x-2 shift puts negatives in   \
       range, where a dropped mode shows up worst. */                          \
    X(f2i_rn,     (float)__float2int_rn(4.0f * x - 2.0f),                       \
                  nearbyintf(4.0f * x - 2.0f),                    0.0f)        \
    X(f2i_rz,     (float)__float2int_rz(4.0f * x - 2.0f),                       \
                  truncf(4.0f * x - 2.0f),                        0.0f)        \
    X(f2i_ru,     (float)__float2int_ru(4.0f * x - 2.0f),                       \
                  ceilf(4.0f * x - 2.0f),                         0.0f)        \
    X(f2i_rd,     (float)__float2int_rd(4.0f * x - 2.0f),                       \
                  floorf(4.0f * x - 2.0f),                        0.0f)        \
    X(ldexpf,     ldexpf(x, 3),     ldexpf(x, 3),               0.0f)          \
    X(scalbnf,    scalbnf(x, 2),    scalbnf(x, 2),              0.0f)          \
    X(saturatef,  __saturatef(2.0f * x),                                    \
                  fminf(fmaxf(2.0f * x, 0.0f), 1.0f),           0.0f)        \
    /* Double-precision libdevice calls (exp/log/sin/ldexp on double) have no  \
       software-ALU primitive; they lower through binary32 evaluation -- the   \
       decode/f32-builtin/encode path -- so the host expression keeps the      \
       double result and the tolerance reflects binary32 precision. */        \
    X(exp_d,      (float)exp((double)x),   (float)exp((double)x), 2e-3f)      \
    X(log_d,      (float)log((double)x),   (float)log((double)x), 2e-3f)      \
    X(sin_d,      (float)sin((double)x),   (float)sin((double)x), 2e-3f)      \
    X(ldexp_d,    (float)ldexp((double)x, 2),                               \
                  (float)ldexp((double)x, 2),                   1e-6f)       \
    /* IEEE arithmetic spelled as intrinsics. The float forms are the same    \
       rounding as the plain operators; the double forms run through the      \
       software-ALU modes, exact for these magnitudes. */                    \
    X(fadd_rn,    __fadd_rn(x, x),  x + x,                        0.0f)        \
    X(fmul_rn,    __fmul_rn(x, x),  x * x,                        0.0f)        \
    X(frcp_rn,    __frcp_rn(x),     1.0f / x,                     0.0f)        \
    X(fsqrt_rn,   __fsqrt_rn(x),    sqrtf(x),                     1e-6f)       \
    X(frsqrt_rn,  __frsqrt_rn(x),   1.0f / sqrtf(x),              2e-3f)       \
    X(exp10_fast, __exp10f(x),      powf(10.0f, x),               2e-3f)       \
    X(dadd_rn,    (float)__dadd_rn((double)x, (double)x),                    \
                  (float)((double)x + (double)x),               1e-6f)       \
    X(ddiv_rn,    (float)__ddiv_rn((double)x, (double)(x + 0.5f)),           \
                  (float)((double)x / ((double)x + 0.5)),       1e-6f)       \
    X(drcp_rn,    (float)__drcp_rn((double)(x + 0.25f)),                     \
                  (float)(1.0 / ((double)x + 0.25)),            1e-6f)       \
    X(dsqrt_rn,   (float)__dsqrt_rn((double)x),                              \
                  (float)sqrt((double)x),                       1e-6f)       \
    /* Directed-rounding interval intrinsics route through the              \
       correctly-rounded vf64 ALU: binary32 operands widen exactly to        \
       binary64, the op runs under the requested mode, and the result        \
       converts back directed. They differ from the rne host reference by    \
       at most ~1 ulp, so the tolerance just covers that. */                \
    X(fsqrt_rd,   __fsqrt_rd(x),    sqrtf(x),                   1e-5f)       \
    X(fsqrt_ru,   __fsqrt_ru(x),    sqrtf(x),                   1e-5f)       \
    X(frcp_rz,    __frcp_rz(x + 0.25f), 1.0f / (x + 0.25f),     1e-5f)       \
    X(dsqrt_rd,   (float)__dsqrt_rd((double)x),                              \
                  (float)sqrt((double)x),                       1e-5f)       \
    X(dsqrt_ru,   (float)__dsqrt_ru((double)x),                              \
                  (float)sqrt((double)x),                       1e-5f)       \
    X(drcp_rz,    (float)__drcp_rz((double)(x + 0.25f)),                     \
                  (float)(1.0 / ((double)x + 0.25)),            1e-5f)       \
    /* sinpi/cospi are Metal builtins; the double forms take the binary32     \
       fallback like the rest of the transcendentals. */                     \
    X(sinpif,     sinpif(x),        sinf(3.141592653589793f * x), 2e-3f)       \
    X(cospif,     cospif(x),        cosf(3.141592653589793f * x), 2e-3f)       \
    X(sinpi_d,    (float)sinpi((double)x),                                 \
                  (float)sin(3.141592653589793 * (double)x),    2e-3f)       \
    /* logb/ilogb expose the unbiased exponent; x in (0,1) keeps it in       \
       [-7,0]. llrint/llround round 4x-2 to a long long. */                  \
    X(logbf,      logbf(x),         floorf(log2f(x)),             0.0f)        \
    X(ilogbf,     (float)ilogbf(x), floorf(log2f(x)),             0.0f)        \
    X(logb_d,     (float)logb((double)x),                                  \
                  (float)floor(log2((double)x)),                1e-6f)       \
    X(ilogb_d,    (float)ilogb((double)x),                                 \
                  (float)ilogb((double)x),                      0.0f)        \
    X(llrintf,    (float)llrintf(4.0f * x - 2.0f),                           \
                  (float)lrintf(4.0f * x - 2.0f),               0.0f)        \
    X(llroundf,   (float)llroundf(4.0f * x - 2.0f),                          \
                  roundf(4.0f * x - 2.0f),                      0.0f)        \
    X(llrint_d,   (float)llrint(4.0 * (double)x - 2.0),                      \
                  (float)lrint(4.0 * (double)x - 2.0),          0.0f)        \
    /* Double->float/int conversions carry a spelled rounding mode; the host \
       references apply the same mode in double then narrow. */              \
    X(d2f_rn,     __double2float_rn((double)x + 0.001),                      \
                  (float)((double)x + 0.001),                   0.0f)        \
    X(d2i_rn,     (float)__double2int_rn(4.0 * (double)x - 2.0),             \
                  (float)lrint(4.0 * (double)x - 2.0),          0.0f)        \
    X(d2i_rd,     (float)__double2int_rd(4.0 * (double)x - 2.0),             \
                  (float)floor(4.0 * (double)x - 2.0),          0.0f)        \
    X(d2i_ru,     (float)__double2int_ru(4.0 * (double)x - 2.0),             \
                  (float)ceil(4.0 * (double)x - 2.0),           0.0f)        \
    X(d2i_rz,     (float)__double2int_rz(4.0 * (double)x - 2.0),             \
                  (float)trunc(4.0 * (double)x - 2.0),          0.0f)        \
    X(d2u_rn,     (float)__double2uint_rn(4.0 * (double)x),                  \
                  (float)lrint(4.0 * (double)x),                0.0f)        \
    X(i2d_rn,     (float)__int2double_rn((int)(100.0f * x)),                 \
                  (float)(double)(int)(100.0f * x),             0.0f)        \
    X(ll2d_rn,    (float)__ll2double_rn((long long)(1000.0f * x)),           \
                  (float)(double)(long long)(1000.0f * x),      0.0f)        \
    /* No Metal builtin: these run through cumetal_libdevice_support.metal    \
       expansions. Host references are independent (system libm or Newton    \
       inversion), not the same expansion, so a wrong device-side formula    \
       still shows up. */                                                   \
    X(erfcxf,     erfcxf(x),                                               \
                  (float)(exp((double)x * x) * erfc((double)x)), 2e-3f)      \
    X(normcdff,   normcdff(x),                                             \
                  (float)(0.5 * erfc(-(double)x * 0.7071067811865476)),      \
                  2e-3f)       \
    X(tgammaf,    tgammaf(x),       (float)tgamma((double)x),   2e-3f)       \
    X(lgammaf,    lgammaf(x),       (float)lgamma((double)x),   2e-3f)       \
    X(erfinvf,    erfinvf(x),       (float)host_erfinv(x),      2e-3f)       \
    X(erfcinvf,   erfcinvf(x),      (float)host_erfcinv(x),     2e-3f)       \
    X(normcdfinvf, normcdfinvf(x),  (float)host_normcdfinv(x),  2e-3f)       \
    X(erfcx_d,    (float)erfcx((double)x),                                 \
                  (float)(exp((double)x * x) * erfc((double)x)), 2e-3f)      \
    X(normcdf_d,  (float)normcdf((double)x),                               \
                  (float)(0.5 * erfc(-(double)x * 0.7071067811865476)),      \
                  2e-3f)       \
    X(tgamma_d,   (float)tgamma((double)x), (float)tgamma((double)x), 2e-3f) \
    X(lgamma_d,   (float)lgamma((double)x), (float)lgamma((double)x), 2e-3f) \
    X(erfinv_d,   (float)erfinv((double)x), (float)host_erfinv(x),  2e-3f)   \
    X(normcdfinv_d, (float)normcdfinv((double)x),                            \
                  (float)host_normcdfinv(x),                    2e-3f)

#define BINARY_LIST(X)                                                         \
    X(fmaxf,      fmaxf(x, y),      fmaxf(x, y),                  0.0f)        \
    X(fminf,      fminf(x, y),      fminf(x, y),                  0.0f)        \
    X(powf,       powf(x, y),       powf(x, y),                   2e-3f)       \
    X(atan2f,     atan2f(x, y),     atan2f(x, y),                 2e-3f)       \
    X(hypotf,     hypotf(x, y),     hypotf(x, y),                 2e-3f)       \
    X(fmodf,      fmodf(x, y),      fmodf(x, y),                  2e-3f)       \
    X(copysignf,  copysignf(x, -y), copysignf(x, -y),             0.0f)        \
    X(fdimf,      fdimf(x, y),      fdimf(x, y),                  0.0f)        \
    X(remainderf, remainderf(x, y), remainderf(x, y),             2e-3f)       \
    X(fmaf,       fmaf(x, y, x),    fmaf(x, y, x),                1e-6f)       \
    X(nextafterf, nextafterf(x, y), nextafterf(x, y),             0.0f)        \
    /* __fdividef is the fast-division intrinsic (emitted as                   \
       __nv_fast_fdividef): CUDA documents <=2 ulp, so it cannot be compared   \
       against a correctly-rounded x / y at zero tolerance. */                \
    X(fdividef,   __fdividef(x, y), x / y,                        1e-6f)       \
    /* Binary double-precision libdevice calls through the binary32 path. */  \
    X(pow_d,      (float)pow((double)x, (double)y),                            \
                  (float)pow((double)x, (double)y),               2e-3f)       \
    X(hypot_d,    (float)hypot((double)x, (double)y),                          \
                  (float)hypot((double)x, (double)y),             2e-3f)       \
    X(atan2_d,    (float)atan2((double)x, (double)y),                          \
                  (float)atan2((double)x, (double)y),             2e-3f)       \
    X(fmod_d,     (float)fmod((double)x, (double)y),                           \
                  (float)fmod((double)x, (double)y),              2e-3f)       \
    /* Multi-argument norms compose from sqrt on the helper path. */         \
    X(norm3df,    norm3df(x, y, x),   sqrtf(2.0f * x * x + y * y), 1e-6f)      \
    X(norm4df,    norm4df(x, y, x, y),                                       \
                  sqrtf(2.0f * x * x + 2.0f * y * y),           1e-6f)       \
    X(rhypotf,    rhypotf(x, y),    1.0f / sqrtf(x * x + y * y),  1e-6f)       \
    X(rnorm3df,   rnorm3df(x, y, x),                                         \
                  1.0f / sqrtf(2.0f * x * x + y * y),           1e-6f)       \
    X(rnorm4df,   rnorm4df(x, y, x, y),                                      \
                  1.0f / sqrtf(2.0f * x * x + 2.0f * y * y),    1e-6f)       \
    X(norm3d_d,   (float)norm3d((double)x, (double)y, (double)x),            \
                  (float)sqrt(2.0 * x * x + (double)y * y),     1e-6f)       \
    X(rhypot_d,   (float)rhypot((double)x, (double)y),                       \
                  (float)(1.0 / sqrt((double)x * x + (double)y * y)), 1e-6f) \
    X(fsub_rn,    __fsub_rn(x, y),    x - y,                      0.0f)        \
    X(fdiv_rn,    __fdiv_rn(x, y),    x / y,                      0.0f)        \
    X(fmaf_rn,    __fmaf_rn(x, y, x), fmaf(x, y, x),            0.0f)        \
    X(fmaf_ieee_rn, __fmaf_ieee_rn(x, y, x), fmaf(x, y, x),     0.0f)        \
    X(dsub_rn,    (float)__dsub_rn((double)x, (double)y),                    \
                  (float)((double)x - (double)y),               1e-6f)       \
    X(dmul_rn,    (float)__dmul_rn((double)x, (double)y),                    \
                  (float)((double)x * (double)y),               1e-6f)       \
    /* Directed-rounding intrinsics; see the unary list for the widening    \
       scheme. */                                                          \
    X(fadd_rd,    __fadd_rd(x, y),  x + y,                      1e-5f)       \
    X(fadd_ru,    __fadd_ru(x, y),  x + y,                      1e-5f)       \
    X(fadd_rz,    __fadd_rz(x, y),  x + y,                      1e-5f)       \
    X(fsub_rd,    __fsub_rd(x, y),  x - y,                      1e-5f)       \
    X(fmul_ru,    __fmul_ru(x, y),  x * y,                      1e-5f)       \
    X(fdiv_rz,    __fdiv_rz(x, y),  x / y,                      1e-5f)       \
    X(fmaf_rd,    __fmaf_rd(x, y, x), fmaf(x, y, x),            1e-5f)       \
    X(dadd_rd,    (float)__dadd_rd((double)x, (double)y),                    \
                  (float)((double)x + (double)y),               1e-5f)       \
    X(dadd_ru,    (float)__dadd_ru((double)x, (double)y),                    \
                  (float)((double)x + (double)y),               1e-5f)       \
    X(dadd_rz,    (float)__dadd_rz((double)x, (double)y),                    \
                  (float)((double)x + (double)y),               1e-5f)       \
    X(dsub_rd,    (float)__dsub_rd((double)x, (double)y),                    \
                  (float)((double)x - (double)y),               1e-5f)       \
    X(dmul_ru,    (float)__dmul_ru((double)x, (double)y),                    \
                  (float)((double)x * (double)y),               1e-5f)       \
    X(ddiv_rz,    (float)__ddiv_rz((double)x, (double)y),                    \
                  (float)((double)x / (double)y),               1e-5f)       \
    X(fma_rd,     (float)__fma_rd((double)x, (double)y, (double)x),          \
                  (float)fma((double)x, (double)y, (double)x),  1e-5f)

// ------------------------------------------------------------- kernels

#define GEN_UNARY_KERNEL(id, dexpr, hexpr, tol)                                \
    __global__ void k_##id(const float* in, float* out, int n) {               \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                         \
        if (i < n) {                                                           \
            float x = in[i];                                                   \
            out[i] = (dexpr);                                                  \
        }                                                                      \
    }
UNARY_LIST(GEN_UNARY_KERNEL)

#define GEN_BINARY_KERNEL(id, dexpr, hexpr, tol)                               \
    __global__ void k_##id(const float* a, const float* b, float* out, int n) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x;                         \
        if (i < n) {                                                           \
            float x = a[i];                                                    \
            float y = b[i];                                                    \
            out[i] = (dexpr);                                                  \
        }                                                                      \
    }
BINARY_LIST(GEN_BINARY_KERNEL)

// PTX has two lowering paths, and `cvt` is implemented separately in each. The
// scalar kernels above are simple enough to be handled by the direct-MSL
// emitter, so they only ever exercise one of them. Shared memory plus a barrier
// forces this one down the LLVM path, where `cvt.rni.f32.f32` (what clang emits
// for rintf) previously degraded to a plain register copy -- rounding silently
// not happening at all.
__global__ void k_rint_shared(const float* in, float* out, int n) {
    __shared__ float tile[BLOCK];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = (i < n) ? in[i] * 4.0f : 0.0f;
    __syncthreads();
    if (i < n) out[i] = rintf(tile[threadIdx.x]);
}
static float ref_rint_shared(float x, float) { return rintf(x * 4.0f); }

// Pointer-out builtins write a second result through an address argument; they
// cannot fold into the X-list expressions, so each gets its own kernel.
__global__ void k_sincosf(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s, c;
        sincosf(in[i], &s, &c);
        out[i] = s + c;
    }
}
static float ref_sincosf(float x, float) { return sinf(x) + cosf(x); }

__global__ void k_modff(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float ip;
        out[i] = modff(in[i], &ip);
    }
}
static float ref_modff(float x, float) { float ip; return modff(x, &ip); }

__global__ void k_frexpf(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int e;
        out[i] = frexpf(in[i], &e);
    }
}
static float ref_frexpf(float x, float) { int e; return frexpf(x, &e); }

// The double spellings run the same pointer-out ABI with binary64 storage;
// under CuMetal's emulated FP64 they evaluate through binary32, so the host
// reference is computed in double and scored at binary32 tolerance.
__global__ void k_sincos_d(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s, c;
        sincos((double)in[i], &s, &c);
        out[i] = (float)(s + c);
    }
}
static float ref_sincos_d(float x, float) {
    // Host libm has no sincos; the pair is sin(x) + cos(x) either way.
    return (float)(sin((double)x) + cos((double)x));
}

__global__ void k_modf_d(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // The 4x scale gives nonzero integral parts so a broken out-param
        // store shows up in the sum instead of passing silently.
        double ip;
        out[i] = (float)modf(4.0 * (double)in[i] + 0.25, &ip) + (float)ip;
    }
}
static float ref_modf_d(float x, float) {
    double ip;
    return (float)modf(4.0 * (double)x + 0.25, &ip) + (float)ip;
}

__global__ void k_frexp_d(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int e;
        out[i] = (float)frexp((double)in[i], &e) + (float)e;
    }
}
static float ref_frexp_d(float x, float) {
    int e;
    return (float)frexp((double)x, &e) + (float)e;
}

// sincospi keeps the pointer-out ABI on both widths; the double form stores
// re-encoded binary64 words through the out-params.
__global__ void k_sincospif(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s, c;
        sincospif(in[i], &s, &c);
        out[i] = s + c;
    }
}
static float ref_sincospif(float x, float) {
    return sinf(3.141592653589793f * x) + cosf(3.141592653589793f * x);
}

__global__ void k_sincospi_d(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double s, c;
        sincospi((double)in[i], &s, &c);
        out[i] = (float)(s + c);
    }
}
static float ref_sincospi_d(float x, float) {
    return (float)(sin(3.141592653589793 * (double)x) +
                   cos(3.141592653589793 * (double)x));
}

// remquo's quotient out-param is a plain int (not binary64 storage); the
// 8x scale on x gives nonzero quotients, and the 100x remainder scale keeps
// both terms visible in one float.
__global__ void k_remquof(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int q;
        float r = remquof(8.0f * a[i], b[i], &q);
        out[i] = r * 100.0f + (float)q;
    }
}
static float ref_remquof(float x, float y) {
    int q;
    float r = remquof(8.0f * x, y, &q);
    return r * 100.0f + (float)q;
}

__global__ void k_remquo_d(const float* a, const float* b, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int q;
        double r = remquo(8.0 * (double)a[i], (double)b[i], &q);
        out[i] = (float)(r * 100.0 + (double)q);
    }
}
static float ref_remquo_d(float x, float y) {
    int q;
    double r = remquo(8.0 * (double)x, (double)y, &q);
    return (float)(r * 100.0 + (double)q);
}

// Raw binary64 word access: hiloint2double packs two u32 halves into the
// storage word; double2hiint/double2loint extract them. Bit-exact in every
// FP64 mode because the word is the representation.
__global__ void k_hiloint2double(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned hi = 0x3FF00000u + (unsigned)(in[i] * 16.0f);
        out[i] = (float)__hiloint2double(hi, 0x80000000u);
    }
}
static float ref_hiloint2double(float x, float) {
    unsigned hi = 0x3FF00000u + (unsigned)(x * 16.0f);
    unsigned long long bits =
        ((unsigned long long)hi << 32) | 0x80000000ull;
    double d;
    memcpy(&d, &bits, sizeof(d));
    return (float)d;
}

__global__ void k_double2hi(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double dx = (double)in[i] + 1.0;
        out[i] = (float)(__double2hiint(dx) - 0x3FF00000);
    }
}
static float ref_double2hi(float x, float) {
    double d = (double)x + 1.0;
    unsigned long long bits;
    memcpy(&bits, &d, sizeof(bits));
    return (float)((int)(bits >> 32) - 0x3FF00000);
}

__global__ void k_double2lo(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double dx = (double)in[i] + 1.0;
        out[i] = (float)((unsigned)__double2loint(dx) >> 16);
    }
}
static float ref_double2lo(float x, float) {
    double d = (double)x + 1.0;
    unsigned long long bits;
    memcpy(&bits, &d, sizeof(bits));
    return (float)((unsigned)(bits & 0xFFFFFFFFull) >> 16);
}

// --------------------------------------------------------- host refs

#define GEN_UNARY_REF(id, dexpr, hexpr, tol)                                   \
    static float ref_##id(float x, float) { return (hexpr); }
UNARY_LIST(GEN_UNARY_REF)

#define GEN_BINARY_REF(id, dexpr, hexpr, tol)                                  \
    static float ref_##id(float x, float y) { return (hexpr); }
BINARY_LIST(GEN_BINARY_REF)

// ---------------------------------------------------------- scoring

static int g_supported = 0;
static int g_wrong = 0;
static int g_unsupported = 0;
static std::vector<std::string> g_wrong_names;
static std::vector<std::string> g_unsupported_names;

static std::vector<float> h_a, h_b, h_out;
static float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;

// Returns true if the launch lowered and ran. `launch` has already been issued
// by the caller; this only classifies the outcome.
static void score(const char* name, float (*ref)(float, float), float tol) {
    cudaError_t launch_err = cudaGetLastError();
    if (launch_err == cudaSuccess) {
        launch_err = cudaDeviceSynchronize();
    }
    if (launch_err != cudaSuccess) {
        printf("  %-12s UNSUPPORTED  (%s)\n", name, cudaGetErrorString(launch_err));
        ++g_unsupported;
        g_unsupported_names.push_back(name);
        // cudaGetLastError does not drain the pending-launch slot; without a
        // sync the failed launch is re-reported at the NEXT kernel's sync and
        // gets mis-attributed to it.
        (void)cudaDeviceSynchronize();
        cudaGetLastError();
        return;
    }

    if (cudaMemcpy(h_out.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("  %-12s UNSUPPORTED  (result copy failed)\n", name);
        ++g_unsupported;
        g_unsupported_names.push_back(name);
        cudaGetLastError();
        return;
    }

    float worst = 0.0f;
    int worst_i = 0;
    for (int i = 0; i < N; ++i) {
        float expect = ref(h_a[i], h_b[i]);
        float got = h_out[i];
        if (std::isnan(expect) && std::isnan(got)) continue;
        float denom = fmaxf(fabsf(expect), 1e-6f);
        float rel = fabsf(got - expect) / denom;
        if (rel > worst) { worst = rel; worst_i = i; }
    }
    if (!(worst <= tol)) {
        printf("  %-12s WRONG        rel err %.4g > %.4g (x=%.5f y=%.5f got=%.7g want=%.7g)\n",
               name, worst, tol, h_a[worst_i], h_b[worst_i], h_out[worst_i],
               ref(h_a[worst_i], h_b[worst_i]));
        ++g_wrong;
        g_wrong_names.push_back(name);
        return;
    }
    printf("  %-12s SUPPORTED    rel err %.3g\n", name, worst);
    ++g_supported;
}

int main(int argc, char** argv) {
    // A single unsupported call used to abort the whole probe; with one kernel
    // per function the run always completes and reports the full table.
    //
    // Strict by default: every function listed here is currently lowered, so a
    // newly-unsupported one is a regression. `--allow-missing` exists for
    // bringing up a new function without turning the suite red first.
    bool require_all = !(argc > 1 && std::string(argv[1]) == "--allow-missing");

    h_a.resize(N);
    h_b.resize(N);
    h_out.resize(N);
    for (int i = 0; i < N; ++i) {
        // Strictly inside (0,1): valid for log/asin/acos/atanh/sqrt alike, and
        // never hits the exact endpoints where fast builtins are allowed to
        // return inf and the relative check would be meaningless.
        h_a[i] = 0.01f + 0.98f * ((float)i / (float)(N - 1));
        h_b[i] = 0.99f - 0.97f * ((float)i / (float)(N - 1));
    }

    if (cudaMalloc((void**)&d_a, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_b, N * sizeof(float)) != cudaSuccess ||
        cudaMalloc((void**)&d_out, N * sizeof(float)) != cudaSuccess) {
        printf("FAIL: cudaMalloc\n");
        return 1;
    }
    cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK);

    printf("libdevice float surface probe (%d values per function)\n", N);
    printf("unary:\n");
#define RUN_UNARY(id, dexpr, hexpr, tol)                                       \
    cudaMemset(d_out, 0, N * sizeof(float));                                   \
    k_##id<<<grid, block>>>(d_a, d_out, N);                                    \
    score(#id, ref_##id, tol);
    UNARY_LIST(RUN_UNARY)

    printf("binary/ternary:\n");
#define RUN_BINARY(id, dexpr, hexpr, tol)                                      \
    cudaMemset(d_out, 0, N * sizeof(float));                                   \
    k_##id<<<grid, block>>>(d_a, d_b, d_out, N);                               \
    score(#id, ref_##id, tol);
    BINARY_LIST(RUN_BINARY)

    printf("shared-memory path (forces the LLVM lowering path):\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    k_rint_shared<<<grid, block>>>(d_a, d_out, N);
    score("rint_shared", ref_rint_shared, 0.0f);

    printf("pointer-out builtins:\n");
    cudaMemset(d_out, 0, N * sizeof(float));
    k_sincosf<<<grid, block>>>(d_a, d_out, N);
    score("sincosf", ref_sincosf, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_modff<<<grid, block>>>(d_a, d_out, N);
    score("modff", ref_modff, 0.0f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_frexpf<<<grid, block>>>(d_a, d_out, N);
    score("frexpf", ref_frexpf, 0.0f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_sincos_d<<<grid, block>>>(d_a, d_out, N);
    score("sincos_d", ref_sincos_d, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_modf_d<<<grid, block>>>(d_a, d_out, N);
    score("modf_d", ref_modf_d, 1e-6f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_frexp_d<<<grid, block>>>(d_a, d_out, N);
    score("frexp_d", ref_frexp_d, 0.0f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_sincospif<<<grid, block>>>(d_a, d_out, N);
    score("sincospif", ref_sincospif, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_sincospi_d<<<grid, block>>>(d_a, d_out, N);
    score("sincospi_d", ref_sincospi_d, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_remquof<<<grid, block>>>(d_a, d_b, d_out, N);
    score("remquof", ref_remquof, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_remquo_d<<<grid, block>>>(d_a, d_b, d_out, N);
    score("remquo_d", ref_remquo_d, 2e-3f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_hiloint2double<<<grid, block>>>(d_a, d_out, N);
    score("hiloint2double", ref_hiloint2double, 0.0f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_double2hi<<<grid, block>>>(d_a, d_out, N);
    score("double2hiint", ref_double2hi, 0.0f);
    cudaMemset(d_out, 0, N * sizeof(float));
    k_double2lo<<<grid, block>>>(d_a, d_out, N);
    score("double2loint", ref_double2lo, 0.0f);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    const int total = g_supported + g_wrong + g_unsupported;
    printf("\nsummary: %d/%d supported, %d wrong, %d unsupported\n",
           g_supported, total, g_wrong, g_unsupported);
    if (!g_wrong_names.empty()) {
        printf("wrong:");
        for (const auto& n : g_wrong_names) printf(" %s", n.c_str());
        printf("\n");
    }
    if (!g_unsupported_names.empty()) {
        printf("unsupported:");
        for (const auto& n : g_unsupported_names) printf(" %s", n.c_str());
        printf("\n");
    }

    // A mis-mapped builtin is always a bug: it computes the wrong function and
    // no caller can detect that. A missing mapping is a coverage gap; the test
    // gates on it only under --require-all so the probe stays runnable while
    // the surface is still being filled in.
    if (g_wrong > 0) {
        printf("FAIL: %d libdevice function(s) lower to the wrong builtin.\n", g_wrong);
        return 1;
    }
    if (require_all && g_unsupported > 0) {
        printf("FAIL: %d libdevice function(s) have no lowering.\n", g_unsupported);
        return 1;
    }
    printf("PASS: libdevice float surface probe (%d supported, %d unsupported)\n",
           g_supported, g_unsupported);
    return 0;
}
