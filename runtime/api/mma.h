#pragma once

// Clean-room CUDA WMMA source-compatibility layer. CuMetal distributes each
// documented matrix tile across a 32-lane SIMD group. BF16 uses public Metal
// SIMD-group matrix intrinsics; the other supported shapes use ALU and shuffles.

#include "cuda_runtime.h"
#include "cuda_bf16.h"
#include "cuda_fp16.h"

namespace nvcuda {
namespace wmma {

#if defined(__clang__) && defined(__CUDA__)
// Compiler-recognized collective. The PTX frontend lowers this marker to
// Metal's public 8x8 BF16 simdgroup-matrix load/MAC/store intrinsics. Keeping
// the marker at the fragment-operation boundary preserves CUDA's public
// fragment storage and permits user code to inspect or modify `x[]` between
// WMMA operations.
extern "C" __device__ void __cumetal_wmma_bf16_mma_8x8(
    float* destination,
    const __nv_bfloat16* matrix_a,
    const __nv_bfloat16* matrix_b);
extern "C" __device__ void __cumetal_wmma_f32_mma_8x8(
    float* destination,
    const float* matrix_a,
    const float* matrix_b);
#endif

struct matrix_a {};
struct matrix_b {};
struct accumulator {};
struct row_major {};
struct col_major {};

enum layout_t { mem_row_major, mem_col_major };

namespace precision {
using tf32 = float;
}

template <typename Use, int M, int N, int K>
struct __cumetal_fragment_elements;
template <int M, int N, int K>
struct __cumetal_fragment_elements<matrix_a, M, N, K> {
    static constexpr int value = M * K;
};
template <int M, int N, int K>
struct __cumetal_fragment_elements<matrix_b, M, N, K> {
    static constexpr int value = K * N;
};
template <int M, int N, int K>
struct __cumetal_fragment_elements<accumulator, M, N, K> {
    static constexpr int value = M * N;
};

template <typename Use, typename T>
struct __cumetal_fragment_storage { using type = T; };
template <>
struct __cumetal_fragment_storage<matrix_a, half> { using type = float; };
template <>
struct __cumetal_fragment_storage<matrix_b, half> { using type = float; };
template <>
struct __cumetal_fragment_storage<matrix_a, __nv_bfloat16> { using type = float; };
template <>
struct __cumetal_fragment_storage<matrix_b, __nv_bfloat16> { using type = float; };

template <typename T>
struct __cumetal_wmma_compute { using type = T; };
// The public Metal ISA has no FP64 ALU. CuMetal's software DMMA evaluates the
// product accumulation in FP32 and converts at the accumulator boundary; this
// matches the integer-valued CUDA sample workload but is not full FP64 WMMA.
template <>
struct __cumetal_wmma_compute<double> { using type = float; };

template <typename Use, int M, int N, int K, typename T, typename Layout = void>
struct fragment {
    using use = Use;
    using element_type = T;
    using layout = Layout;
    static constexpr int logical_elements =
        __cumetal_fragment_elements<Use, M, N, K>::value;
    static constexpr int num_elements = (logical_elements + 31) / 32;
    using storage_type = typename __cumetal_fragment_storage<Use, T>::type;
    storage_type x[num_elements];
};

template <typename Layout>
struct __cumetal_is_row_major { static constexpr bool value = false; };
template <>
struct __cumetal_is_row_major<row_major> { static constexpr bool value = true; };

__device__ __forceinline__ int __cumetal_wmma_lane() {
    return static_cast<int>(threadIdx.x) & 31;
}

template <typename T>
__device__ __forceinline__ T __cumetal_wmma_convert(T value) { return value; }

template <typename To, typename From>
__device__ __forceinline__ To __cumetal_wmma_convert_as(From value) {
    return static_cast<To>(value);
}

template <typename Use, int M, int N, int K, typename T, typename Layout,
          typename Value>
__device__ __forceinline__ void fill_fragment(
    fragment<Use, M, N, K, T, Layout>& frag, Value value) {
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        frag.x[i] = static_cast<T>(value);
    }
}

template <int M, int N, int K, typename T, typename Layout>
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_a, M, N, K, T, Layout>& frag,
    const T* pointer, unsigned int leading_dimension) {
    const int lane = __cumetal_wmma_lane();
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        const int logical = lane * frag.num_elements + i;
        if (logical < M * K) {
            const int row = logical / K;
            const int col = logical - row * K;
            const int offset = __cumetal_is_row_major<Layout>::value
                                   ? row * static_cast<int>(leading_dimension) + col
                                   : col * static_cast<int>(leading_dimension) + row;
            frag.x[i] = static_cast<typename fragment<matrix_a, M, N, K, T, Layout>::storage_type>(
                pointer[offset]);
        } else {
            frag.x[i] = typename fragment<matrix_a, M, N, K, T, Layout>::storage_type{};
        }
    }
}

template <int M, int N, int K, typename T, typename Layout>
__device__ __forceinline__ void load_matrix_sync(
    fragment<matrix_b, M, N, K, T, Layout>& frag,
    const T* pointer, unsigned int leading_dimension) {
    const int lane = __cumetal_wmma_lane();
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        const int logical = lane * frag.num_elements + i;
        if (logical < K * N) {
            const int row = logical / N;
            const int col = logical - row * N;
            const int offset = __cumetal_is_row_major<Layout>::value
                                   ? row * static_cast<int>(leading_dimension) + col
                                   : col * static_cast<int>(leading_dimension) + row;
            frag.x[i] = static_cast<typename fragment<matrix_b, M, N, K, T, Layout>::storage_type>(
                pointer[offset]);
        } else {
            frag.x[i] = typename fragment<matrix_b, M, N, K, T, Layout>::storage_type{};
        }
    }
}

template <int M, int N, int K, typename T>
__device__ __forceinline__ void load_matrix_sync(
    fragment<accumulator, M, N, K, T>& frag,
    const T* pointer, unsigned int leading_dimension, layout_t layout) {
    const int lane = __cumetal_wmma_lane();
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        const int logical = lane * frag.num_elements + i;
        if (logical < M * N) {
            const int row = logical / N;
            const int col = logical - row * N;
            frag.x[i] = pointer[layout == mem_row_major
                                    ? row * static_cast<int>(leading_dimension) + col
                                    : col * static_cast<int>(leading_dimension) + row];
        } else {
            frag.x[i] = T{};
        }
    }
}

template <typename To, typename Use, int M, int N, int K, typename T, typename Layout>
__device__ __forceinline__ To __cumetal_wmma_get(
    const fragment<Use, M, N, K, T, Layout>& frag, int logical) {
    const int owner = logical / frag.num_elements;
    const int local_index = logical - owner * frag.num_elements;
    To local = To{};
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        if (i == local_index) local = static_cast<To>(frag.x[i]);
    }
    return __shfl_sync(0xffffffffu, local, owner, 32);
}

template <int M, int N, int K, typename TA, typename LayoutA,
          typename TB, typename LayoutB, typename TC>
__device__ __forceinline__ void mma_sync(
    fragment<accumulator, M, N, K, TC>& destination,
    const fragment<matrix_a, M, N, K, TA, LayoutA>& a,
    const fragment<matrix_b, M, N, K, TB, LayoutB>& b,
    const fragment<accumulator, M, N, K, TC>& c) {
#if defined(__CUDA_ARCH__)
    if constexpr (M == 16 && N == 16 && K == 8 &&
                  __is_same(TA, float) && __is_same(TB, float) &&
                  __is_same(TC, float)) {
        __shared__ float staged_a[32][64];
        __shared__ float staged_b[32][64];
        __shared__ float staged_c[32][64];
        const int lane = __cumetal_wmma_lane();
        const int linear_thread = static_cast<int>(threadIdx.x) +
            static_cast<int>(blockDim.x) *
                (static_cast<int>(threadIdx.y) +
                 static_cast<int>(blockDim.y) * static_cast<int>(threadIdx.z));
        const int warp = linear_thread >> 5;
#pragma unroll
        for (int output_row_tile = 0; output_row_tile < 2; ++output_row_tile) {
#pragma unroll
            for (int output_col_tile = 0; output_col_tile < 2; ++output_col_tile) {
#pragma unroll
                for (int i = 0; i < a.num_elements; ++i) {
                    const int logical = lane * a.num_elements + i;
                    const int row = logical / K;
                    const int col = logical - row * K;
                    if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8) {
                        staged_a[warp][(row - output_row_tile * 8) * 8 + col] =
                            a.x[i];
                    }
                }
#pragma unroll
                for (int i = 0; i < b.num_elements; ++i) {
                    const int logical = lane * b.num_elements + i;
                    const int row = logical / N;
                    const int col = logical - row * N;
                    if (col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                        staged_b[warp][row * 8 + col - output_col_tile * 8] = b.x[i];
                    }
                }
#pragma unroll
                for (int i = 0; i < c.num_elements; ++i) {
                    const int logical = lane * c.num_elements + i;
                    const int row = logical / N;
                    const int col = logical - row * N;
                    if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8 &&
                        col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                        staged_c[warp][(row - output_row_tile * 8) * 8 +
                                       col - output_col_tile * 8] = c.x[i];
                    }
                }
                __syncwarp();
                __cumetal_wmma_f32_mma_8x8(staged_c[warp], staged_a[warp],
                                           staged_b[warp]);
                __syncwarp();
#pragma unroll
                for (int i = 0; i < destination.num_elements; ++i) {
                    const int logical = lane * destination.num_elements + i;
                    const int row = logical / N;
                    const int col = logical - row * N;
                    if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8 &&
                        col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                        destination.x[i] =
                            staged_c[warp][(row - output_row_tile * 8) * 8 +
                                           col - output_col_tile * 8];
                    }
                }
                __syncwarp();
            }
        }
        return;
    }

    if constexpr (M == 8 && N == 8 && K == 4 &&
                  __is_same(TA, double) && __is_same(TB, double) &&
                  __is_same(TC, double)) {
        // CUDA DMMA is emulated with FP32 compute on current Apple GPUs. Stage
        // its 8x4 and 4x8 operands as zero-padded 8x8 tiles so Metal's public
        // FP32 SIMD-group matrix operation can perform the same computation
        // without a watchdog-scale scalar/shuffle loop.
        __shared__ float staged_a[32][64];
        __shared__ float staged_b[32][64];
        __shared__ float staged_c[32][64];

        const int lane = __cumetal_wmma_lane();
        const int linear_thread = static_cast<int>(threadIdx.x) +
            static_cast<int>(blockDim.x) *
                (static_cast<int>(threadIdx.y) +
                 static_cast<int>(blockDim.y) * static_cast<int>(threadIdx.z));
        const int warp = linear_thread >> 5;

        staged_a[warp][lane] = 0.0f;
        staged_a[warp][lane + 32] = 0.0f;
        staged_b[warp][lane] = 0.0f;
        staged_b[warp][lane + 32] = 0.0f;
#pragma unroll
        for (int i = 0; i < a.num_elements; ++i) {
            const int logical = lane * a.num_elements + i;
            if (logical < M * K) {
                const int row = logical / K;
                const int col = logical - row * K;
                staged_a[warp][row * 8 + col] = static_cast<float>(a.x[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < b.num_elements; ++i) {
            const int logical = lane * b.num_elements + i;
            if (logical < K * N) {
                staged_b[warp][logical] = static_cast<float>(b.x[i]);
            }
        }
#pragma unroll
        for (int i = 0; i < c.num_elements; ++i) {
            const int logical = lane * c.num_elements + i;
            if (logical < M * N) {
                staged_c[warp][logical] = static_cast<float>(c.x[i]);
            }
        }
        __syncwarp();
        __cumetal_wmma_f32_mma_8x8(staged_c[warp], staged_a[warp],
                                   staged_b[warp]);
        __syncwarp();
#pragma unroll
        for (int i = 0; i < destination.num_elements; ++i) {
            const int logical = lane * destination.num_elements + i;
            destination.x[i] = logical < M * N
                                   ? static_cast<double>(staged_c[warp][logical])
                                   : 0.0;
        }
        return;
    }

    if constexpr (M == 16 && N == 16 && K == 16 &&
                  __is_same(TA, __nv_bfloat16) &&
                  __is_same(TB, __nv_bfloat16) &&
                  __is_same(TC, float)) {
        // One 8x8 A/B/C staging tile per possible warp in a 1024-thread CUDA
        // block. Reusing it across the eight native operations keeps static
        // threadgroup memory to 16 KiB instead of materializing full 16x16
        // fragments for every warp.
        __shared__ __nv_bfloat16 staged_a[32][64];
        __shared__ __nv_bfloat16 staged_b[32][64];
        __shared__ float staged_c[32][64];

        const int lane = __cumetal_wmma_lane();
        const int linear_thread = static_cast<int>(threadIdx.x) +
            static_cast<int>(blockDim.x) *
                (static_cast<int>(threadIdx.y) +
                 static_cast<int>(blockDim.y) * static_cast<int>(threadIdx.z));
        const int warp = linear_thread >> 5;

#pragma unroll
        for (int output_row_tile = 0; output_row_tile < 2; ++output_row_tile) {
#pragma unroll
            for (int output_col_tile = 0; output_col_tile < 2; ++output_col_tile) {
#pragma unroll
                for (int inner_tile = 0; inner_tile < 2; ++inner_tile) {
#pragma unroll
                    for (int i = 0; i < a.num_elements; ++i) {
                        const int logical = lane * a.num_elements + i;
                        const int row = logical / K;
                        const int col = logical - row * K;
                        if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8 &&
                            col >= inner_tile * 8 && col < inner_tile * 8 + 8) {
                            const int tile_index = (row - output_row_tile * 8) * 8 +
                                                   (col - inner_tile * 8);
                            staged_a[warp][tile_index] = __nv_bfloat16(a.x[i]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < b.num_elements; ++i) {
                        const int logical = lane * b.num_elements + i;
                        const int row = logical / N;
                        const int col = logical - row * N;
                        if (row >= inner_tile * 8 && row < inner_tile * 8 + 8 &&
                            col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                            const int tile_index = (row - inner_tile * 8) * 8 +
                                                   (col - output_col_tile * 8);
                            staged_b[warp][tile_index] = __nv_bfloat16(b.x[i]);
                        }
                    }
                    if (inner_tile == 0) {
#pragma unroll
                        for (int i = 0; i < c.num_elements; ++i) {
                            const int logical = lane * c.num_elements + i;
                            const int row = logical / N;
                            const int col = logical - row * N;
                            if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8 &&
                                col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                                const int tile_index = (row - output_row_tile * 8) * 8 +
                                                       (col - output_col_tile * 8);
                                staged_c[warp][tile_index] = c.x[i];
                            }
                        }
                    }
                    __syncwarp();
                    __cumetal_wmma_bf16_mma_8x8(staged_c[warp], staged_a[warp],
                                                staged_b[warp]);
                    __syncwarp();
                }

#pragma unroll
                for (int i = 0; i < destination.num_elements; ++i) {
                    const int logical = lane * destination.num_elements + i;
                    const int row = logical / N;
                    const int col = logical - row * N;
                    if (row >= output_row_tile * 8 && row < output_row_tile * 8 + 8 &&
                        col >= output_col_tile * 8 && col < output_col_tile * 8 + 8) {
                        const int tile_index = (row - output_row_tile * 8) * 8 +
                                               (col - output_col_tile * 8);
                        destination.x[i] = staged_c[warp][tile_index];
                    }
                }
                __syncwarp();
            }
        }
        return;
    }
#endif
    using Compute = typename __cumetal_wmma_compute<TC>::type;
    constexpr int kAccumulatorElements =
        fragment<accumulator, M, N, K, TC>::num_elements;
    const int lane = __cumetal_wmma_lane();
    Compute sums[kAccumulatorElements];
#pragma unroll
    for (int i = 0; i < kAccumulatorElements; ++i) {
        sums[i] = static_cast<Compute>(c.x[i]);
    }

    // A lane owns consecutive accumulator columns, so all of its output
    // elements share one A row. Hoisting that shuffle out of the column loop
    // nearly halves the software WMMA communication cost.
    const int first_logical = lane * kAccumulatorElements;
    const int row = first_logical / N;
#pragma unroll
    for (int inner = 0; inner < K; ++inner) {
        const Compute av = __cumetal_wmma_get<Compute>(a, row * K + inner);
#pragma unroll
        for (int i = 0; i < kAccumulatorElements; ++i) {
            const int logical = first_logical + i;
            if (logical < M * N) {
                const int col = logical - row * N;
                const Compute bv = __cumetal_wmma_get<Compute>(b, inner * N + col);
                sums[i] = static_cast<Compute>(sums[i] + av * bv);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kAccumulatorElements; ++i) {
        const int logical = first_logical + i;
        if (logical < M * N) {
            destination.x[i] = static_cast<TC>(sums[i]);
        } else {
            destination.x[i] = TC{};
        }
    }
}

template <int M, int N, int K, typename T>
__device__ __forceinline__ void store_matrix_sync(
    T* pointer, const fragment<accumulator, M, N, K, T>& frag,
    unsigned int leading_dimension, layout_t layout) {
    const int lane = __cumetal_wmma_lane();
#pragma unroll
    for (int i = 0; i < frag.num_elements; ++i) {
        const int logical = lane * frag.num_elements + i;
        if (logical < M * N) {
            const int row = logical / N;
            const int col = logical - row * N;
            pointer[layout == mem_row_major
                        ? row * static_cast<int>(leading_dimension) + col
                        : col * static_cast<int>(leading_dimension) + row] = frag.x[i];
        }
    }
}

__device__ __forceinline__ float __float_to_tf32(float value) {
    union { float f; unsigned int u; } bits{value};
    const unsigned int lsb = (bits.u >> 13) & 1u;
    bits.u += 0x0fffu + lsb;
    bits.u &= 0xffffe000u;
    return bits.f;
}

}  // namespace wmma
}  // namespace nvcuda
