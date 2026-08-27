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
// Prepend kF64DekkerMsl (f64_dekker_msl.h) before compiling: cm_f64 and its
// arithmetic live there so that an SpMV and a cublasDdot round the same way.
inline constexpr std::string_view kSparseKernelsMsl = R"MSL(
struct SpmvParams {
    uint  axis;          // compressed rows to walk
    int   base;          // index base, 0 or 1
    uint  beta_is_zero;  // y is written, not accumulated
    uint  pad;
    ulong alpha_bits;    // binary64 bits; the f32 kernels read the low half
    ulong beta_bits;
};

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
