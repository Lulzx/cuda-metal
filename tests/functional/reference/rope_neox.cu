#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>

struct rope_corr_dims {
    float low;
    float high;
};

// GPT-NeoX RoPE forward (no freq-factor scaling), matching the GGML
// rope_neox ABI: one thread per element pair, rows flattened as
// i1 + i2 * ne01 across blockIdx.x, pairs across threadIdx.y.
template <typename DstT>
__global__ void rope_neox_impl(const float* src, DstT* dst, int ne00, int ne01,
                               int ne02, int s01, int s02, int s03, int s1,
                               int s2, int s3, int n_dims, const int* pos,
                               float freq_scale, float ext_factor,
                               float attn_factor, rope_corr_dims corr_dims,
                               float theta_scale, const float* freq_factors,
                               const long long* row_indices,
                               int set_rows_stride) {
    (void)ne00; (void)s03; (void)s3; (void)freq_factors; (void)row_indices;
    (void)set_rows_stride;
    const int row = static_cast<int>(blockIdx.x);
    const int i2 = row / ne01;
    const int i1 = row % ne01;
    const int pair = static_cast<int>(threadIdx.y);
    const int i0 = pair * 2;
    const int src_base = i2 * s02 + i1 * s01;
    const int dst_base = i2 * s2 + i1 * s1;
    if (i0 >= n_dims) {
        dst[dst_base + i0] = static_cast<DstT>(src[src_base + i0]);
        dst[dst_base + i0 + 1] = static_cast<DstT>(src[src_base + i0 + 1]);
        return;
    }
    const float theta_extrap =
        static_cast<float>(pos[i2]) * powf(theta_scale, static_cast<float>(i0) * 0.5f);
    const float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    float mscale = attn_factor;
    if (ext_factor != 0.0f) {
        const float y = (static_cast<float>(i0 / 2) - corr_dims.low) /
                        fmaxf(0.001f, corr_dims.high - corr_dims.low);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, y));
        const float mix = ramp * ext_factor;
        theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float cosine = cosf(theta) * mscale;
    const float sine = sinf(theta) * mscale;
    const float x0 = src[src_base + i0 / 2];
    const float x1 = src[src_base + i0 / 2 + n_dims / 2];
    dst[dst_base + i0 / 2] = static_cast<DstT>(x0 * cosine - x1 * sine);
    dst[dst_base + i0 / 2 + n_dims / 2] = static_cast<DstT>(x0 * sine + x1 * cosine);
}

template __global__ void rope_neox_impl<float>(
    const float*, float*, int, int, int, int, int, int, int, int, int, int,
    const int*, float, float, float, rope_corr_dims, float, const float*,
    const long long*, int);
template __global__ void rope_neox_impl<__half>(
    const float*, __half*, int, int, int, int, int, int, int, int, int, int,
    const int*, float, float, float, rope_corr_dims, float, const float*,
    const long long*, int);
