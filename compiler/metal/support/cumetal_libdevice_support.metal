#include <metal_stdlib>
using namespace metal;

// CuMetal libdevice float helpers: functions with no Metal builtin, expressed
// as numerically-tested binary32 expansions. The typed MSL backend includes
// this file textually when generated source references cm_libdevice_*; the
// LLVM/AIR lowering links it as an additional input for the same symbols.
// Both float and double device calls reach these helpers -- double calls
// decode their binary64 storage to binary32 first (see docs/fp64-policy.md).
//
// Entry points use extern "C": plain C++ linkage would mangle the symbols and
// the emitted LLVM IR calls them by unmangled name, while [[visible]] would
// turn every call -- including helper-to-helper ones -- into an indirect
// visible-function reference that fails at pipeline creation with
// "unresolved visible function reference". extern "C" keeps the unmangled
// name AND an ordinary direct call on both paths.
// Helpers shared between entry points stay file-static for the same reason.

// IEEE round-to-nearest-even via stable primitives. The Apple Metal compiler
// accepts rint(float) but has produced identity-like results on the exercised
// GPU path, so helpers spell the tie-breaking explicitly (same expansion as
// the rint/remainder path in lower_to_msl.cpp).
static float cm_rne_impl(float x) {
    float base = floor(x);
    float fraction = x - base;
    float next = base + 1.0f;
    float tied = fmod(fabs(base), 2.0f) == 0.0f ? base : next;
    float rounded = fraction < 0.5f ? base : (fraction > 0.5f ? next : tied);
    return copysign(rounded, x);
}

// Abramowitz-Stegun 7.1.26: erfc(|x|) ~ P(t) * exp(-x^2), t = 1/(1+p|x|),
// |absolute error| <= 1.5e-7. erfc(x) uses the product directly for x >= 0
// and 2 - tail for x < 0 rather than 1 - erf(x), which would cancel to a
// flat zero for large x.
static float cm_erfc_impl(float x) {
    float ax = fabs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float p = 1.061405429f;
    p = -1.453152027f + t * p;
    p =  1.421413741f + t * p;
    p = -0.284496736f + t * p;
    p =  0.254829592f + t * p;
    float tail = p * t * exp(-(ax * ax));
    return x < 0.0f ? 2.0f - tail : tail;
}

// Inverse of the standard normal CDF via Acklam's rational approximation
// (max |epsilon| ~1.15e-9 in double precision; ample headroom in binary32):
// central region |p - 1/2| <= 0.47575 uses r = q^2, tails evaluate the rational
// directly at q = sqrt(-2 ln r).
static float cm_normcdfinv_impl(float p) {
    if (isnan(p)) return p;
    if (p <= 0.0f) return -INFINITY;
    if (p >= 1.0f) return INFINITY;
    if (p >= 0.02425f && p <= 0.97575f) {
        float q = p - 0.5f;
        float r = q * q;
        float num = -39.69683028665376f;
        num = num * r + 220.9460984245205f;
        num = num * r + -275.9285104469687f;
        num = num * r + 138.3577518672690f;
        num = num * r + -30.66479806614716f;
        num = num * r + 2.506628277459239f;
        float den = -54.47609879822406f;
        den = den * r + 161.5858368580409f;
        den = den * r + -155.6989798598866f;
        den = den * r + 66.80131188771972f;
        den = den * r + -13.28068155288572f;
        return num * q / (den * r + 1.0f);
    }
    float q = sqrt(-2.0f * log(p < 0.5f ? p : 1.0f - p));
    float num = -0.007784894002430293f;
    num = num * q + -0.3223964580411365f;
    num = num * q + -2.400758277161838f;
    num = num * q + -2.549732539343734f;
    num = num * q + 4.374664141464968f;
    num = num * q + 2.938163982698783f;
    float den = 0.007784695709041462f;
    den = den * q + 0.3224671290700398f;
    den = den * q + 2.445134137142996f;
    den = den * q + 3.754408661907416f;
    // The rational already carries the lower-tail sign; negate for the upper.
    float x = num / (den * q + 1.0f);
    return p < 0.5f ? x : -x;
}

// Lanczos approximation (g = 7, n = 9) shared by tgamma and lgamma.
static float cm_lanczos_impl(float x) {
    float a = 0.99999999999980993f;
    a +=  676.5203681218851f     / (x + 1.0f);
    a += -1259.1392167224028f    / (x + 2.0f);
    a +=  771.32342877765313f    / (x + 3.0f);
    a += -176.61502916214059f    / (x + 4.0f);
    a +=  12.507343278686905f    / (x + 5.0f);
    a += -0.13857109526572012f   / (x + 6.0f);
    a +=  9.9843695780195716e-6f / (x + 7.0f);
    a +=  1.5056327351493116e-7f / (x + 8.0f);
    return a;
}

// Direct Lanczos branch, valid for x >= 0.5. Kept separate so the reflection
// branches below can call it without recursion -- recursive fastcc calls are
// miscompiled on the AIR pipeline (Metal has no call stack).
static float cm_tgamma_direct(float x) {
    float z = x - 1.0f;
    float a = cm_lanczos_impl(z);
    float t = z + 7.5f;
    return sqrt(2.0f * 3.141592653589793f) * pow(t, z + 0.5f) * exp(-t) * a;
}

static float cm_lgamma_direct(float x) {
    float z = x - 1.0f;
    float a = cm_lanczos_impl(z);
    float t = z + 7.5f;
    return 0.5f * log(2.0f * 3.141592653589793f) + (z + 0.5f) * log(t) - t +
           log(a);
}

static float cm_tgamma_impl(float x) {
    if (x < 0.5f) {
        // Reflection: Gamma(x) Gamma(1-x) = pi / sin(pi x). 1 - x >= 0.5 here,
        // so the direct branch always applies on the far side.
        float s = sinpi(x);
        if (s == 0.0f) return copysign(INFINITY, s);
        return 3.141592653589793f / (s * cm_tgamma_direct(1.0f - x));
    }
    return cm_tgamma_direct(x);
}

static float cm_lgamma_impl(float x) {
    if (x < 0.5f) {
        float s = sinpi(x);
        if (s == 0.0f) return INFINITY;
        return log(3.141592653589793f / fabs(s)) - cm_lgamma_direct(1.0f - x);
    }
    return cm_lgamma_direct(x);
}

extern "C" float cm_libdevice_rne(float x) { return cm_rne_impl(x); }

// erfcx(x) = exp(x^2) * erfc(x). The direct product overflows for x > ~9.4
// (exp(x^2) -> inf while erfc(x) -> 0), so large x uses the asymptotic
// erfcx(x) ~ 1/(x sqrt(pi)) * (1 - 1/(2x^2) + 3/(4x^4) - 15/(8x^6)).
extern "C" float cm_libdevice_erfcx(float x) {
    if (x <= 9.0f) {
        return exp(x * x) * cm_erfc_impl(x);
    }
    float inv = 1.0f / x;
    float inv2 = inv * inv;
    float series = 1.0f + inv2 * (-0.5f + inv2 * (0.75f + inv2 * -1.875f));
    return inv * 0.5641895835477563f * series;
}

extern "C" float cm_libdevice_normcdf(float x) {
    return 0.5f * cm_erfc_impl(-x * 0.7071067811865476f);
}

extern "C" float cm_libdevice_normcdfinv(float p) {
    return cm_normcdfinv_impl(p);
}

extern "C" float cm_libdevice_erfinv(float y) {
    // erfinv(y) = normcdfinv((y + 1) / 2) / sqrt(2)
    return cm_normcdfinv_impl(0.5f * y + 0.5f) * 0.7071067811865476f;
}

extern "C" float cm_libdevice_erfcinv(float y) {
    // erfcinv(y) = erfinv(1 - y) = normcdfinv(1 - y/2) / sqrt(2)
    return cm_normcdfinv_impl(1.0f - 0.5f * y) * 0.7071067811865476f;
}

extern "C" float cm_libdevice_tgamma(float x) { return cm_tgamma_impl(x); }

extern "C" float cm_libdevice_lgamma(float x) { return cm_lgamma_impl(x); }

// Unbiased exponent as a float; ilogb exists in Metal but logb does not.
// FP_ILOGB0/FP_ILOGBNAN edge cases are spelled out to match CUDA's logb.
extern "C" float cm_libdevice_logb(float x) {
    if (isnan(x)) return x;
    if (isinf(x)) return INFINITY;
    if (x == 0.0f) return -INFINITY;
    return (float)ilogb(x);
}

extern "C" long cm_libdevice_llrint(float x) {
    return (long)cm_rne_impl(x);
}

extern "C" long cm_libdevice_llround(float x) {
    return (long)round(x);
}

// Multi-argument norms without a Metal builtin compose from sqrt. These use
// the plain sum of squares (no scaling), matching the hypot expansion the
// backend already emits; overflow beyond ~1e19 is a documented limitation.
extern "C" float cm_libdevice_norm3d(float x, float y, float z) {
    return sqrt(x * x + y * y + z * z);
}

extern "C" float cm_libdevice_norm4d(float w, float x, float y, float z) {
    return sqrt(w * w + x * x + y * y + z * z);
}

extern "C" float cm_libdevice_rhypot(float x, float y) {
    return 1.0f / sqrt(x * x + y * y);
}

extern "C" float cm_libdevice_rnorm3d(float x, float y, float z) {
    return 1.0f / sqrt(x * x + y * y + z * z);
}

extern "C" float cm_libdevice_rnorm4d(float w, float x, float y, float z) {
    return 1.0f / sqrt(w * w + x * x + y * y + z * z);
}

// Thin wrappers over Metal builtins that have no stable AIR spelling on the
// LLVM lowering path; calling them through cm_libdevice_* keeps the JIT and
// typed backends on identical numerics.
extern "C" float cm_libdevice_sinpi(float x) {
    return sinpi(x);
}

extern "C" float cm_libdevice_cospi(float x) {
    return cospi(x);
}

extern "C" int cm_libdevice_ilogb(float x) {
    return ilogb(x);
}
