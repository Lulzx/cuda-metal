#pragma once

#include <string_view>

namespace cumetal::rt {

// The FP64 emulation shared by every library kernel CuMetal compiles at runtime.
//
// It lives in one place on purpose. Two copies of Dekker arithmetic that drift
// apart would not crash: they would return slightly different numbers from
// cusparseSpMV and cublasDdot for the same input, which is the hardest kind of
// bug to see from a solver's output. Sources that want it prepend this blob and
// then define their own kernels.
//
// The `cumetal-math-mode: safe` marker has to stay at the top of the composed
// file. The algorithms below recover an error term from a cancellation like
// `s - a.hi`, and under fast math the backend is free to reassociate exactly
// that expression away, which turns the low limb into zero and silently drops
// the emulation back to binary32.
inline constexpr std::string_view kF64DekkerMsl = R"MSL(
// cumetal-provenance: library_substitution
// cumetal-math-mode: safe
#include <metal_stdlib>
using namespace metal;

// IEEE-754 binary64 is carried as its bit pattern and decoded into a Dekker FP32
// pair for arithmetic, mirroring what the PTX lowering does so that a library
// call and a compiled kernel agree on the same ~48-bit significand.
//
//   x = sign * 2^(e-52) * M            M a 53-bit integer
//     = sign * (M_hi * 2^(e-23) + M_lo * 2^(e-52))
//
// M_hi is 24 bits so float() is exact. M_lo is 29 bits and rounds to 24, which
// is where the five bits below the pair's 48 are lost.
struct cm_f64 { float hi; float lo; };

static inline cm_f64 cm_f64_decode(ulong bits) {
    const uint biased = uint((bits >> 52) & 0x7FF);
    const bool negative = (bits >> 63) != 0;
    const ulong frac = bits & 0xFFFFFFFFFFFFF;
    if (biased == 0) {
        // Zero, or a binary64 subnormal, which is below what the pair can hold.
        return cm_f64{negative ? -0.0f : 0.0f, 0.0f};
    }
    if (biased == 0x7FF) {
        const float inf = as_type<float>(0x7F800000u);
        return cm_f64{frac != 0 ? as_type<float>(0x7FC00000u) : (negative ? -inf : inf), 0.0f};
    }
    const int e = int(biased) - 1023;
    const float sign = negative ? -1.0f : 1.0f;
    const uint m_hi = uint(frac >> 29) | 0x800000u;   // 24 bits with the implicit 1
    const uint m_lo = uint(frac & 0x1FFFFFFF);        // 29 bits
    return cm_f64{sign * ldexp(float(m_hi), e - 23),
                  sign * ldexp(float(m_lo), e - 52)};
}

static inline ulong cm_f64_encode(cm_f64 v) {
    const uint hb = as_type<uint>(v.hi);
    const uint biased = (hb >> 23) & 0xFF;
    const ulong sign_bit = (hb >> 31) != 0 ? (ulong(1) << 63) : ulong(0);
    if (biased == 0) {
        return sign_bit;                              // zero or a float subnormal
    }
    if (biased == 0xFF) {
        const ulong frac = ulong(hb & 0x7FFFFF) << 29;
        return sign_bit | (ulong(0x7FF) << 52) | frac;
    }
    const int e = int(biased) - 127;
    const uint m_hi = (hb & 0x7FFFFF) | 0x800000u;    // exact, 24 bits
    // The low limb's contribution measured in units of 2^(e-52). Signs are
    // relative to hi, so an opposite-signed residual subtracts.
    const float lo_rel = (hb >> 31) != 0 ? -v.lo : v.lo;
    const long m_lo = long(rint(ldexp(lo_rel, 52 - e)));
    long m = (long(m_hi) << 29) + m_lo;
    int exponent = e;
    // A normalized pair moves the significand by at most one bit either way.
    if (m >= (long(1) << 53)) { m >>= 1; exponent += 1; }
    else if (m < (long(1) << 52)) { m <<= 1; exponent -= 1; }
    if (m < (long(1) << 52) || exponent <= -1023 || exponent >= 1024) {
        // Unnormalized pair or outside binary64's range: fall back to the
        // collapsed value rather than packing something malformed.
        const float collapsed = v.hi + v.lo;
        const uint cb = as_type<uint>(collapsed);
        const uint cbiased = (cb >> 23) & 0xFF;
        const ulong csign = (cb >> 31) != 0 ? (ulong(1) << 63) : ulong(0);
        if (cbiased == 0) { return csign; }
        return csign | (ulong(uint(int(cbiased) - 127 + 1023)) << 52) |
               (ulong(cb & 0x7FFFFF) << 29);
    }
    return sign_bit | (ulong(uint(exponent + 1023)) << 52) | (ulong(m) & 0xFFFFFFFFFFFFF);
}

// Dekker two-sum and two-product. Both need exact IEEE rounding per step.
static inline cm_f64 cm_f64_add(cm_f64 a, cm_f64 b) {
    const float s = a.hi + b.hi;
    const float bv = s - a.hi;
    const float err = (a.hi - (s - bv)) + (b.hi - bv);
    const float lo = (err + a.lo) + b.lo;
    const float hi = s + lo;
    return cm_f64{hi, lo - (hi - s)};
}

static inline cm_f64 cm_f64_mul(cm_f64 a, cm_f64 b) {
    const float p = a.hi * b.hi;
    const float e = fma(a.hi, b.hi, -p);
    const float lo = ((e + a.hi * b.lo) + a.lo * b.hi) + a.lo * b.lo;
    const float hi = p + lo;
    return cm_f64{hi, lo - (hi - p)};
}

// |x|. The pair is normalized, so |lo| is at most half an ulp of hi and the
// sign of hi + lo is the sign of hi whenever hi is nonzero.
static inline cm_f64 cm_f64_abs(cm_f64 a) {
    return a.hi < 0.0f ? cm_f64{-a.hi, -a.lo} : a;
}
)MSL";

}  // namespace cumetal::rt
