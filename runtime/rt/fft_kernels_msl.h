#pragma once

#include <string_view>

namespace cumetal::rt {

// Metal kernels backing the single-precision cuFFT transforms.
//
// These exist because GROMACS's PME runs an R2C and a C2R over the whole mesh on
// every step. With the transform on the CPU the rest of PME was on the GPU and
// the FFT was a synchronous excursion back to the host in the middle of it.
//
// The engine is Stockham autosort, radix 2, out of place. Autosort matters here:
// the classic Cooley-Tukey formulation needs a bit-reversal permutation, which
// on a GPU is either a separate scattered pass or a divergent index computation
// inside every butterfly. Stockham folds the permutation into the data movement
// each pass already performs, so every pass is one dispatch with a fully regular
// access pattern and no permutation kernel at all.
//
// Lengths that are not a power of two -- 40, which is villin's slowest PME axis,
// is the ordinary case -- go through Bluestein on the same kernels: a chirp
// multiply, a power-of-two convolution, and a second chirp multiply. Nothing
// about that path returns to the host.
//
// Everything is float2. Metal has no FP64, and the emulated pair would cost more
// than the host transform it replaced, so the double-precision cuFFT entry
// points keep the CPU implementation.
inline constexpr std::string_view kFftKernelsMsl = R"MSL(

#include <metal_stdlib>
using namespace metal;

// A transform axis viewed as `outer` x `length` x `inner`, which is what any
// axis of a row-major N-D grid looks like: `inner` is the product of the faster
// axes, `outer` the product of the slower ones. Element (o, e, i) sits at
// (o * length + e) * inner + i. One kernel therefore serves every axis of every
// rank, including the contiguous case (inner == 1) that Bluestein works in.
struct FftPassParams {
    uint outer;
    uint length;
    uint inner;
    uint half_span;    // `l` in the Stockham recurrence: pairs still to split
    uint block_span;   // `m`: elements already contiguous within a sub-transform
    float sign;        // -1 forward, +1 inverse
    uint pad0;
    uint pad1;
};

// One Stockham pass. Threads map one-to-one onto butterflies:
//   thread -> (line, idx), idx -> (j, k) with j = idx / m, k = idx % m
// so the two reads and two writes are contiguous across k and the whole
// dispatch is free of divergence.
kernel void cumetal_fft_stockham_f32(
    device const float2*  src [[buffer(0)]],
    device float2*        dst [[buffer(1)]],
    constant FftPassParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint half_length = p.length / 2u;
    const uint total = p.outer * p.inner * half_length;
    if (tid >= total) return;

    const uint butterfly = tid % half_length;
    const uint line = tid / half_length;
    const uint inner_index = line % p.inner;
    const uint outer_index = line / p.inner;

    const uint j = butterfly / p.block_span;
    const uint k = butterfly % p.block_span;

    const uint base = outer_index * p.length * p.inner + inner_index;
    const uint lo = k + j * p.block_span;
    const uint hi = lo + p.half_span * p.block_span;

    const float2 a = src[base + lo * p.inner];
    const float2 b = src[base + hi * p.inner];

    // theta = sign * pi * j / half_span. Forming it from the integer ratio
    // rather than accumulating a rotation keeps every twiddle independently
    // rounded, so the error does not grow along the pass.
    const float theta = p.sign * M_PI_F * float(j) / float(p.half_span);
    const float wr = precise::cos(theta);
    const float wi = precise::sin(theta);

    const float2 sum = a + b;
    const float2 diff = a - b;
    const float2 twiddled = float2(diff.x * wr - diff.y * wi, diff.x * wi + diff.y * wr);

    const uint out_lo = k + 2u * j * p.block_span;
    const uint out_hi = out_lo + p.block_span;
    dst[base + out_lo * p.inner] = sum;
    dst[base + out_hi * p.inner] = twiddled;
}

// Every pass of a transform, staged in threadgroup memory.
//
// The pass kernel above writes the whole grid back to device memory between
// passes, so a length-L transform reads and writes it log2(L) times. When one
// line fits in threadgroup memory that is all avoidable: a threadgroup loads its
// line once, runs every pass against threadgroup memory with a barrier between
// them, and stores once. For a 128-point axis that is 7 device round trips
// replaced by 1.
//
// The line is the unit rather than the grid because a threadgroup barrier only
// orders threads within a threadgroup; a transform staged across threadgroups
// would need a device-wide barrier between passes, which is the dispatch
// boundary the multi-pass kernel already is.
//
// Two threadgroup buffers of L complex values ping-pong, so the caller sizes
// threadgroup memory at 2*L*sizeof(float2) and declines when that exceeds the
// device budget.
kernel void cumetal_fft_stockham_line_f32(
    device const float2*  src [[buffer(0)]],
    device float2*        dst [[buffer(1)]],
    constant FftPassParams& p [[buffer(2)]],
    threadgroup float2*   scratch [[threadgroup(0)]],
    uint line   [[threadgroup_position_in_grid]],
    uint lid    [[thread_position_in_threadgroup]],
    uint stride [[threads_per_threadgroup]])
{
    const uint length = p.length;
    if (line >= p.outer * p.inner) return;

    const uint inner_index = line % p.inner;
    const uint outer_index = line / p.inner;
    const uint base = outer_index * length * p.inner + inner_index;

    threadgroup float2* read = scratch;
    threadgroup float2* write = scratch + length;

    for (uint j = lid; j < length; j += stride) {
        read[j] = src[base + j * p.inner];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint butterflies = length / 2u;
    for (uint half_span = length / 2u, block_span = 1u; half_span >= 1u;
         half_span /= 2u, block_span *= 2u) {
        for (uint t = lid; t < butterflies; t += stride) {
            const uint j = t / block_span;
            const uint k = t % block_span;
            const uint lo = k + j * block_span;
            const uint hi = lo + half_span * block_span;
            const float2 a = read[lo];
            const float2 b = read[hi];
            const float theta = p.sign * M_PI_F * float(j) / float(half_span);
            const float wr = precise::cos(theta);
            const float wi = precise::sin(theta);
            const float2 diff = a - b;
            const uint out_lo = k + 2u * j * block_span;
            write[out_lo] = a + b;
            write[out_lo + block_span] =
                float2(diff.x * wr - diff.y * wi, diff.x * wi + diff.y * wr);
        }
        // Orders this pass's writes to `write` before it becomes the next pass's
        // source, and equally this pass's reads of `read` before the next pass
        // overwrites it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2* swap = read;
        read = write;
        write = swap;
        if (half_span == 1u) break;
    }

    for (uint j = lid; j < length; j += stride) {
        dst[base + j * p.inner] = read[j];
    }
}

// ── cuFFT advanced data layout ───────────────────────────────────────────────
//
// Element (x0, x1, x2) of batch b lives at b*dist + stride*((x0*embed1 + x1)*embed2 + x2).
// embed[0] never participates, matching cuFFT. `dims` are the logical extents of
// the domain being addressed, which differ between the real and complex sides of
// a real transform.
struct FftLayoutParams {
    uint dim0;
    uint dim1;
    uint dim2;
    uint embed1;
    uint embed2;
    uint stride;
    uint base;        // batch offset, already multiplied by dist
    uint fast_full;   // full extent of the fastest axis, for Hermitian mirroring
};

static inline uint fft_layout_offset(constant FftLayoutParams& p, uint i0, uint i1, uint i2)
{
    return p.base + p.stride * ((i0 * p.embed1 + i1) * p.embed2 + i2);
}

static inline void fft_unrank(uint linear, constant FftLayoutParams& p,
                              thread uint& i0, thread uint& i1, thread uint& i2)
{
    i2 = linear % p.dim2;
    const uint rest = linear / p.dim2;
    i1 = rest % p.dim1;
    i0 = rest / p.dim1;
}

// Real grid -> complex field with zero imaginary part.
kernel void cumetal_fft_load_real_f32(
    device const float*   src   [[buffer(0)]],
    device float2*        field [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    field[tid] = float2(src[fft_layout_offset(p, i0, i1, i2)], 0.0f);
}

// Complex grid -> complex field.
kernel void cumetal_fft_load_complex_f32(
    device const float2*  src   [[buffer(0)]],
    device float2*        field [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    field[tid] = src[fft_layout_offset(p, i0, i1, i2)];
}

// Complex field -> complex grid.
kernel void cumetal_fft_store_complex_f32(
    device const float2*  field [[buffer(0)]],
    device float2*        dst   [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    dst[fft_layout_offset(p, i0, i1, i2)] = field[tid];
}

// Complex field -> real grid, taking the real part.
kernel void cumetal_fft_store_real_f32(
    device const float2*  field [[buffer(0)]],
    device float*         dst   [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    dst[fft_layout_offset(p, i0, i1, i2)] = field[tid].x;
}

// Keep the non-redundant half of the fastest axis. `dims` describe the
// destination (complex) field; `fast_full` is the source's fastest extent.
kernel void cumetal_fft_truncate_f32(
    device const float2*  full_field [[buffer(0)]],
    device float2*        half_field [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    half_field[tid] = full_field[(i0 * p.dim1 + i1) * p.fast_full + i2];
}

// Rebuild the full fastest axis from the stored half by conjugate mirroring.
// This is only valid once the slower axes are already back in real space: at
// that point each remaining line is the spectrum of a real sequence on its own,
// so the mirror is within the line rather than across the whole grid.
// `dims` describe the destination (full) field, `fast_full` the half's extent.
kernel void cumetal_fft_hermitian_expand_f32(
    device const float2*  half_field [[buffer(0)]],
    device float2*        full_field [[buffer(1)]],
    constant FftLayoutParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.dim0 * p.dim1 * p.dim2;
    if (tid >= total) return;
    uint i0, i1, i2;
    fft_unrank(tid, p, i0, i1, i2);
    const bool mirrored = i2 >= p.fast_full;
    const uint source_k = mirrored ? (p.dim2 - i2) : i2;
    const float2 v = half_field[(i0 * p.dim1 + i1) * p.fast_full + source_k];
    full_field[tid] = mirrored ? float2(v.x, -v.y) : v;
}

// ── Bluestein ────────────────────────────────────────────────────────────────
//
// x[j] is multiplied by chirp[j] and zero-extended to `padded`; the convolution
// with the precomputed filter is a pointwise product between two power-of-two
// transforms; the result is scaled and multiplied by chirp again.
struct FftBluesteinParams {
    uint lines;
    uint length;     // n, the true transform length
    uint padded;     // m, a power of two >= 2n-1
    uint outer;      // field geometry, as in FftPassParams
    uint inner;
    float scale;     // 1/m, folded into the post pass
    uint pad0;
    uint pad1;
};

// Gather one line of the strided field, apply the chirp, zero the tail.
kernel void cumetal_fft_bluestein_pre_f32(
    device const float2*  field  [[buffer(0)]],
    device const float2*  chirp  [[buffer(1)]],
    device float2*        work   [[buffer(2)]],
    constant FftBluesteinParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.lines * p.padded;
    if (tid >= total) return;
    const uint slot = tid % p.padded;
    const uint line = tid / p.padded;
    if (slot >= p.length) {
        work[tid] = float2(0.0f, 0.0f);
        return;
    }
    const uint inner_index = line % p.inner;
    const uint outer_index = line / p.inner;
    const float2 x =
        field[(outer_index * p.length + slot) * p.inner + inner_index];
    const float2 c = chirp[slot];
    work[tid] = float2(x.x * c.x - x.y * c.y, x.x * c.y + x.y * c.x);
}

// Pointwise product with the transformed filter, which is shared by every line.
kernel void cumetal_fft_bluestein_mul_f32(
    device float2*        work   [[buffer(0)]],
    device const float2*  filter [[buffer(1)]],
    constant FftBluesteinParams& p [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.lines * p.padded;
    if (tid >= total) return;
    const float2 a = work[tid];
    const float2 b = filter[tid % p.padded];
    work[tid] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Scale, apply the chirp again, scatter back into the strided field.
kernel void cumetal_fft_bluestein_post_f32(
    device const float2*  work   [[buffer(0)]],
    device const float2*  chirp  [[buffer(1)]],
    device float2*        field  [[buffer(2)]],
    constant FftBluesteinParams& p [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    const uint total = p.lines * p.length;
    if (tid >= total) return;
    const uint slot = tid % p.length;
    const uint line = tid / p.length;
    const float2 v = work[line * p.padded + slot] * p.scale;
    const float2 c = chirp[slot];
    const uint inner_index = line % p.inner;
    const uint outer_index = line / p.inner;
    field[(outer_index * p.length + slot) * p.inner + inner_index] =
        float2(v.x * c.x - v.y * c.y, v.x * c.y + v.y * c.x);
}

)MSL";

}  // namespace cumetal::rt
