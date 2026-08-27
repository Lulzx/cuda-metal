#pragma once

#include <string_view>

namespace cumetal::rt {

// Metal kernels backing cusparseSpMV.
//
// Only the gather shape is here: one output element per compressed row, reading
// its own slice of the index and value arrays. That covers CSR non-transpose and
// CSC transpose, which in the CSR-of-transpose model are the same loop. The
// scatter shapes (CSR transpose, CSC non-transpose) would need atomic
// accumulation into y, and Metal has no FP64 atomic, so those stay on the CPU.
//
// The `cumetal-math-mode: safe` marker matters: the FP64 emulation below is
// Dekker arithmetic, which depends on exact IEEE rounding at each step. Under
// fast math the compiler is free to reassociate `s - a.hi` and friends, which
// is precisely the cancellation the algorithm uses to recover the error term.
inline constexpr std::string_view kSparseKernelsMsl = R"MSL(
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

struct SpmvParams {
    uint  axis;          // compressed rows to walk
    int   base;          // index base, 0 or 1
    uint  beta_is_zero;  // y is written, not accumulated
    uint  pad;
    ulong alpha_bits;    // binary64 bits; the f32 kernels read the low half
    ulong beta_bits;
};

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

kernel void cumetal_spmv_gather_f64(
    device const int*    offsets [[buffer(0)]],
    device const int*    indices [[buffer(1)]],
    device const ulong*  values  [[buffer(2)]],
    device const ulong*  x       [[buffer(3)]],
    device ulong*        y       [[buffer(4)]],
    constant SpmvParams& p       [[buffer(5)]],
    uint                 gid     [[thread_position_in_grid]]) {
    if (gid >= p.axis) { return; }

    cm_f64 acc = cm_f64{0.0f, 0.0f};
    const int begin = offsets[gid] - p.base;
    const int end = offsets[gid + 1] - p.base;
    for (int k = begin; k < end; ++k) {
        acc = cm_f64_add(acc, cm_f64_mul(cm_f64_decode(values[k]),
                                         cm_f64_decode(x[indices[k] - p.base])));
    }

    cm_f64 out = cm_f64_mul(cm_f64_decode(p.alpha_bits), acc);
    if (p.beta_is_zero == 0) {
        // beta == 0 means y is written, not read: CUDA does not require the
        // existing contents to be finite, so skip the multiply entirely.
        out = cm_f64_add(out, cm_f64_mul(cm_f64_decode(p.beta_bits),
                                         cm_f64_decode(y[gid])));
    }
    y[gid] = cm_f64_encode(out);
}

kernel void cumetal_spmv_gather_f32(
    device const int*    offsets [[buffer(0)]],
    device const int*    indices [[buffer(1)]],
    device const float*  values  [[buffer(2)]],
    device const float*  x       [[buffer(3)]],
    device float*        y       [[buffer(4)]],
    constant SpmvParams& p       [[buffer(5)]],
    uint                 gid     [[thread_position_in_grid]]) {
    if (gid >= p.axis) { return; }

    float acc = 0.0f;
    const int begin = offsets[gid] - p.base;
    const int end = offsets[gid + 1] - p.base;
    for (int k = begin; k < end; ++k) {
        acc = fma(values[k], x[indices[k] - p.base], acc);
    }

    const float alpha = as_type<float>(uint(p.alpha_bits));
    const float beta = as_type<float>(uint(p.beta_bits));
    y[gid] = p.beta_is_zero != 0 ? alpha * acc : fma(beta, y[gid], alpha * acc);
}

// One simdgroup per compressed row. The scalar kernel above walks a row in a
// single thread, so its cost is the longest row; this one splits that row across
// the simdgroup's lanes and reduces, cutting the serial depth by the simdgroup
// width. The strided read (lane i takes k, k+width, ...) is also the coalesced
// one, where the scalar kernel has each thread walking its own disjoint run.
//
// It is not a strict improvement: a row shorter than the simdgroup width leaves
// lanes idle, so a matrix of uniformly short rows runs this kernel with most of
// its threads doing nothing. Which of the two runs is chosen from the longest
// row, in cusparse.cpp.
//
// Rows are walked with a grid stride rather than one row per simdgroup. The
// host cannot size a grid in simdgroups without knowing the execution width,
// and guessing 32 would silently leave rows uncomputed on a device that reports
// anything else -- a wrong answer, not a slow one. The stride is read from the
// dispatch instead, so any grid at all covers every row.

kernel void cumetal_spmv_gather_simd_f64(
    device const int*    offsets [[buffer(0)]],
    device const int*    indices [[buffer(1)]],
    device const ulong*  values  [[buffer(2)]],
    device const ulong*  x       [[buffer(3)]],
    device ulong*        y       [[buffer(4)]],
    constant SpmvParams& p       [[buffer(5)]],
    uint tg    [[threadgroup_position_in_grid]],
    uint tgs   [[threadgroups_per_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint sgs   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    // The bound is uniform across the simdgroup, so every lane runs the same
    // number of iterations and the shuffles below never wait on a lane that has
    // already left the loop.
    const uint stride = tgs * sgs;
    for (uint row = tg * sgs + sg; row < p.axis; row += stride) {
        cm_f64 acc = cm_f64{0.0f, 0.0f};
        const int begin = offsets[row] - p.base;
        const int end = offsets[row + 1] - p.base;
        for (int k = begin + int(lane); k < end; k += int(width)) {
            acc = cm_f64_add(acc, cm_f64_mul(cm_f64_decode(values[k]),
                                             cm_f64_decode(x[indices[k] - p.base])));
        }

        // No simd_sum for a pair, so reduce by hand. Lane 0 only ever reads
        // lanes that hold a live partial sum, which is why its result is the
        // one used. Summing a row in a different order than the scalar kernel
        // does gives a different rounding, not a worse one.
        for (uint off = width >> 1; off > 0; off >>= 1) {
            const cm_f64 other = cm_f64{simd_shuffle_down(acc.hi, off),
                                        simd_shuffle_down(acc.lo, off)};
            acc = cm_f64_add(acc, other);
        }
        if (lane != 0) { continue; }

        cm_f64 out = cm_f64_mul(cm_f64_decode(p.alpha_bits), acc);
        if (p.beta_is_zero == 0) {
            out = cm_f64_add(out, cm_f64_mul(cm_f64_decode(p.beta_bits),
                                             cm_f64_decode(y[row])));
        }
        y[row] = cm_f64_encode(out);
    }
}

kernel void cumetal_spmv_gather_simd_f32(
    device const int*    offsets [[buffer(0)]],
    device const int*    indices [[buffer(1)]],
    device const float*  values  [[buffer(2)]],
    device const float*  x       [[buffer(3)]],
    device float*        y       [[buffer(4)]],
    constant SpmvParams& p       [[buffer(5)]],
    uint tg    [[threadgroup_position_in_grid]],
    uint tgs   [[threadgroups_per_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint sgs   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {
    const uint stride = tgs * sgs;
    for (uint row = tg * sgs + sg; row < p.axis; row += stride) {
        float acc = 0.0f;
        const int begin = offsets[row] - p.base;
        const int end = offsets[row + 1] - p.base;
        for (int k = begin + int(lane); k < end; k += int(width)) {
            acc = fma(values[k], x[indices[k] - p.base], acc);
        }
        acc = simd_sum(acc);
        if (lane != 0) { continue; }

        const float alpha = as_type<float>(uint(p.alpha_bits));
        const float beta = as_type<float>(uint(p.beta_bits));
        y[row] = p.beta_is_zero != 0 ? alpha * acc : fma(beta, y[row], alpha * acc);
    }
}
)MSL";

}  // namespace cumetal::rt
