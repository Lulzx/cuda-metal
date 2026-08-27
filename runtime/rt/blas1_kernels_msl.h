#pragma once

#include <string_view>

namespace cumetal::rt {

// Metal kernels backing the FP64 cuBLAS level-1 calls.
//
// These exist because cuPDLP-C's GPU path calls cublasDaxpy, cublasDdot,
// cublasDnrm2 and cublasDscal directly on vectors the length of the LP, and
// serving them from a scalar CPU loop over unified memory made them a quarter
// of a measured PDLP solve while the sparse products were already on the GPU.
//
// Prepend kF64DekkerMsl before compiling: cm_f64 and its arithmetic come from
// there so that a dot product and an SpMV round the same way.
//
// The reductions stop at one partial per threadgroup and leave the last step to
// the host. Metal has no FP64 atomic, so a single-pass kernel would need the
// address-hashed lock bank, and a second reduction pass would cost another
// dispatch to fold a few hundred values. Folding those on the CPU is both
// cheaper and more accurate, since the host adds them in real binary64 rather
// than in the emulated pair.
inline constexpr std::string_view kBlas1KernelsMsl = R"MSL(

struct Blas1Params {
    uint  n;
    uint  incx;
    uint  incy;
    uint  op;          // reduction selector; see kReduceOp* in cublas.cpp
    ulong alpha_bits;  // binary64 bits
};

constant uint kOpDot    = 0u;   // sum x[i] * y[i]
constant uint kOpSumSq  = 1u;   // sum x[i] * x[i], for nrm2
constant uint kOpAbsSum = 2u;   // sum |x[i]|, for asum

kernel void cumetal_daxpy_f64(
    device ulong*         y     [[buffer(0)]],
    device const ulong*   x     [[buffer(1)]],
    constant Blas1Params& p     [[buffer(2)]],
    uint                  gid   [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    const uint xi = gid * p.incx;
    const uint yi = gid * p.incy;
    y[yi] = cm_f64_encode(cm_f64_add(cm_f64_mul(cm_f64_decode(p.alpha_bits),
                                                cm_f64_decode(x[xi])),
                                     cm_f64_decode(y[yi])));
}

kernel void cumetal_dscal_f64(
    device ulong*         x     [[buffer(0)]],
    constant Blas1Params& p     [[buffer(1)]],
    uint                  gid   [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    const uint xi = gid * p.incx;
    x[xi] = cm_f64_encode(cm_f64_mul(cm_f64_decode(p.alpha_bits),
                                     cm_f64_decode(x[xi])));
}

// Copy and swap move bit patterns and never decode, so they are exact rather
// than emulated: the 48-bit contract does not apply to them.
kernel void cumetal_dcopy_f64(
    device ulong*         y     [[buffer(0)]],
    device const ulong*   x     [[buffer(1)]],
    constant Blas1Params& p     [[buffer(2)]],
    uint                  gid   [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    y[gid * p.incy] = x[gid * p.incx];
}

kernel void cumetal_dswap_f64(
    device ulong*         x     [[buffer(0)]],
    device ulong*         y     [[buffer(1)]],
    constant Blas1Params& p     [[buffer(2)]],
    uint                  gid   [[thread_position_in_grid]]) {
    if (gid >= p.n) { return; }
    const uint xi = gid * p.incx;
    const uint yi = gid * p.incy;
    const ulong t = x[xi];
    x[xi] = y[yi];
    y[yi] = t;
}

// One partial per threadgroup, written as binary64 bits for the host to fold.
//
// Each thread walks the vector with a grid stride so that any grid size covers
// every element, then the simdgroup reduces by shuffle and the simdgroup leaders
// meet in threadgroup memory. Summing in this order gives a different rounding
// than the CPU loop's strict left-to-right, not a worse one -- pairwise summation
// over n elements has error growing like log n rather than n.
kernel void cumetal_dreduce_f64(
    device ulong*         partials [[buffer(0)]],
    device const ulong*   x        [[buffer(1)]],
    device const ulong*   y        [[buffer(2)]],
    constant Blas1Params& p        [[buffer(3)]],
    threadgroup float2*   scratch  [[threadgroup(0)]],
    uint gid   [[thread_position_in_grid]],
    uint gsz   [[threads_per_grid]],
    uint tg    [[threadgroup_position_in_grid]],
    uint sg    [[simdgroup_index_in_threadgroup]],
    uint sgs   [[simdgroups_per_threadgroup]],
    uint lane  [[thread_index_in_simdgroup]],
    uint width [[threads_per_simdgroup]]) {

    cm_f64 acc = cm_f64{0.0f, 0.0f};
    for (uint i = gid; i < p.n; i += gsz) {
        const cm_f64 xv = cm_f64_decode(x[i * p.incx]);
        cm_f64 term;
        if (p.op == kOpDot) {
            term = cm_f64_mul(xv, cm_f64_decode(y[i * p.incy]));
        } else if (p.op == kOpSumSq) {
            term = cm_f64_mul(xv, xv);
        } else {
            term = cm_f64_abs(xv);
        }
        acc = cm_f64_add(acc, term);
    }

    // No simd_sum for a pair, so reduce by hand. Every lane is live here --
    // the loop bound is a grid stride, not a per-lane bound -- so the shuffles
    // never read a lane that has already returned.
    for (uint off = width >> 1; off > 0; off >>= 1) {
        const cm_f64 other = cm_f64{simd_shuffle_down(acc.hi, off),
                                    simd_shuffle_down(acc.lo, off)};
        acc = cm_f64_add(acc, other);
    }
    if (lane == 0) { scratch[sg] = float2(acc.hi, acc.lo); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg != 0 || lane != 0) { return; }
    cm_f64 total = cm_f64{0.0f, 0.0f};
    for (uint i = 0; i < sgs; ++i) {
        total = cm_f64_add(total, cm_f64{scratch[i].x, scratch[i].y});
    }
    partials[tg] = cm_f64_encode(total);
}
)MSL";

}  // namespace cumetal::rt
