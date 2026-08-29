#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

struct RopeCorrDims {
    float v[2];
};

extern "C" __global__ void rms_norm_f32_probe(
    const float* x, float* dst, int ncols, int64_t stride_row,
    int64_t stride_channel, int64_t stride_sample, float eps,
    const float* mul, int64_t mul_stride_row, int64_t mul_stride_channel,
    int64_t mul_stride_sample, uint3 mul_ncols, uint3 mul_nrows,
    uint3 mul_nchannels, uint3 mul_nsamples, const float* add,
    int64_t add_stride_row, int64_t add_stride_channel,
    int64_t add_stride_sample, uint3 add_ncols, uint3 add_nrows,
    uint3 add_nchannels, uint3 add_nsamples) {
    __shared__ float partials[256];
    const unsigned tid = threadIdx.x;
    const unsigned row = blockIdx.x;
    const unsigned channel = blockIdx.y;
    const unsigned sample = blockIdx.z;
    const float* row_x = x + static_cast<int64_t>(sample) * stride_sample +
                         static_cast<int64_t>(channel) * stride_channel +
                         static_cast<int64_t>(row) * stride_row;
    float* row_dst = dst +
        ((static_cast<int64_t>(sample) * gridDim.y + channel) * gridDim.x + row) * ncols;
    float sum = 0.0f;
    for (int col = static_cast<int>(tid); col < ncols; col += blockDim.x) {
        const float value = row_x[col];
        sum += value * value;
    }
    partials[tid] = sum;
    __syncthreads();
    for (unsigned span = blockDim.x / 2; span > 0; span >>= 1) {
        if (tid < span) partials[tid] += partials[tid + span];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(partials[0] / ncols + eps);
    const float* row_mul = mul;
    if (row_mul != nullptr) {
        const unsigned mr = mul_nrows.z ? row % mul_nrows.z : 0;
        const unsigned mc = mul_nchannels.z ? channel % mul_nchannels.z : 0;
        const unsigned ms = mul_nsamples.z ? sample % mul_nsamples.z : 0;
        row_mul += static_cast<int64_t>(ms) * mul_stride_sample +
                   static_cast<int64_t>(mc) * mul_stride_channel +
                   static_cast<int64_t>(mr) * mul_stride_row;
    }
    const float* row_add = add;
    if (row_add != nullptr) {
        const unsigned ar = add_nrows.z ? row % add_nrows.z : 0;
        const unsigned ac = add_nchannels.z ? channel % add_nchannels.z : 0;
        const unsigned as = add_nsamples.z ? sample % add_nsamples.z : 0;
        row_add += static_cast<int64_t>(as) * add_stride_sample +
                   static_cast<int64_t>(ac) * add_stride_channel +
                   static_cast<int64_t>(ar) * add_stride_row;
    }
    for (int col = static_cast<int>(tid); col < ncols; col += blockDim.x) {
        float value = row_x[col] * scale;
        if (row_mul != nullptr) value *= row_mul[mul_ncols.z ? col % mul_ncols.z : 0];
        if (row_add != nullptr) value += row_add[add_ncols.z ? col % add_ncols.z : 0];
        row_dst[col] = value;
    }
}

extern "C" __global__ void rms_norm_f32_plain_probe(
    const float* x, float* dst, int ncols, int64_t stride_row,
    int64_t stride_channel, int64_t stride_sample, float eps) {
    __shared__ float partials[256];
    const unsigned tid = threadIdx.x;
    const unsigned row = blockIdx.x;
    const unsigned channel = blockIdx.y;
    const unsigned sample = blockIdx.z;
    const float* row_x = x + static_cast<int64_t>(sample) * stride_sample +
                         static_cast<int64_t>(channel) * stride_channel +
                         static_cast<int64_t>(row) * stride_row;
    float* row_dst = dst +
        ((static_cast<int64_t>(sample) * gridDim.y + channel) * gridDim.x + row) * ncols;
    float sum = 0.0f;
    for (int col = static_cast<int>(tid); col < ncols; col += blockDim.x) {
        const float value = row_x[col];
        sum += value * value;
    }
    partials[tid] = sum;
    __syncthreads();
    for (unsigned span = blockDim.x / 2; span > 0; span >>= 1) {
        if (tid < span) partials[tid] += partials[tid + span];
        __syncthreads();
    }
    const float scale = 1.0f / sqrtf(partials[0] / ncols + eps);
    for (int col = static_cast<int>(tid); col < ncols; col += blockDim.x) {
        row_dst[col] = row_x[col] * scale;
    }
}

extern "C" __global__ void k_bin_bcast_op_addff_probe(
    const float* src0, const float* src1, float* dst, int ne0, int ne1,
    int ne2, uint3 ne3, uint3 ne10, uint3 ne11, uint3 ne12, uint3 ne13,
    int s1, int s2, int s3, int s00, int s01, int s02, int s03, int s10,
    int s11, int s12, int s13, const float* extra_src1) {
    const unsigned i0_start = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned i1 = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
    const unsigned i2 = ne3.z ? z / ne3.z : 0;
    const unsigned i3 = z - i2 * ne3.z;
    if (i0_start >= static_cast<unsigned>(ne0) || i1 >= static_cast<unsigned>(ne1) ||
        i2 >= static_cast<unsigned>(ne2) || i3 >= ne3.z) return;
    const unsigned i11 = ne11.z ? i1 % ne11.z : i1;
    const unsigned i12 = ne12.z ? i2 % ne12.z : i2;
    const unsigned i13 = ne13.z ? i3 % ne13.z : i3;
    const int64_t base0 = static_cast<int64_t>(i3) * s03 +
                          static_cast<int64_t>(i2) * s02 +
                          static_cast<int64_t>(i1) * s01;
    const int64_t base1 = static_cast<int64_t>(i13) * s13 +
                          static_cast<int64_t>(i12) * s12 +
                          static_cast<int64_t>(i11) * s11;
    const int64_t base_dst = static_cast<int64_t>(i3) * s3 +
                             static_cast<int64_t>(i2) * s2 +
                             static_cast<int64_t>(i1) * s1;
    const unsigned stride = gridDim.x * blockDim.x;
    for (unsigned i0 = i0_start; i0 < static_cast<unsigned>(ne0); i0 += stride) {
        const unsigned i10 = ne10.z ? i0 % ne10.z : i0;
        const float rhs = extra_src1 ? extra_src1[base1 + i10 * s10]
                                     : src1[base1 + i10 * s10];
        dst[base_dst + i0] = src0[base0 + i0 * s00] + rhs;
    }
}

extern "C" __global__ void dequantize_block_q8_0_f16_probe(
    const void* source, __half* dst, int64_t count) {
    const unsigned char* packed = static_cast<const unsigned char*>(source);
    const int64_t group_base = static_cast<int64_t>(blockIdx.x) * 2048;
    for (int64_t local = threadIdx.x; local < 2048; local += blockDim.x) {
        const int64_t out = group_base + local;
        if (out >= count) return;
        const int64_t block = out / 32;
        const int lane = static_cast<int>(out % 32);
        const unsigned char* record = packed + block * 34;
        const __half scale = *reinterpret_cast<const __half*>(record);
        const signed char quant = *reinterpret_cast<const signed char*>(record + 2 + lane);
        dst[out] = __float2half_rn(__half2float(scale) * static_cast<float>(quant));
    }
}

extern "C" __global__ void dequantize_block_q6_KIDF16_E_probe(
    const void* source, __half* dst) {
    if (threadIdx.x >= 64) return;
    const unsigned char* record = static_cast<const unsigned char*>(source) +
                                  static_cast<int64_t>(blockIdx.x) * 210;
    const unsigned ip = threadIdx.x >> 5;
    const unsigned il = threadIdx.x & 31;
    const unsigned scale_index = 8 * ip + (il >> 4);
    const unsigned ql_index = 64 * ip + il;
    const unsigned char qh = record[128 + 32 * ip + il];
    const unsigned char ql0 = record[ql_index];
    const unsigned char ql32 = record[ql_index + 32];
    const int q[4] = {
        static_cast<int>((ql0 & 15) | (((qh >> 0) & 3) << 4)) - 32,
        static_cast<int>((ql32 & 15) | (((qh >> 2) & 3) << 4)) - 32,
        static_cast<int>((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32,
        static_cast<int>((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32,
    };
    const signed char* scales = reinterpret_cast<const signed char*>(record + 192);
    const float delta = __half2float(*reinterpret_cast<const __half*>(record + 208));
    const int64_t out = static_cast<int64_t>(blockIdx.x) * 256 + 128 * ip + il;
    for (int segment = 0; segment < 4; ++segment) {
        dst[out + 32 * segment] = __float2half_rn(
            delta * static_cast<float>(scales[scale_index + 2 * segment]) * q[segment]);
    }
}

extern "C" __global__ void unary_gated_op_kernel_op_silu_EEfEv_probe(
    const float* x, const float* gate, float* dst, int64_t count,
    int64_t width, int64_t x_stride, int64_t gate_stride) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count || width <= 0) return;
    const int64_t row = i / width;
    const int64_t col = i - row * width;
    const float value = x[row * x_stride + col];
    dst[i] = value / (1.0f + expf(-value)) * gate[row * gate_stride + col];
}

extern "C" __global__ void rope_normILb1ELb0EfDF16_Ev_probe(
    const float* src, __half* dst, int ne00, int ne01, int ne02, int s01,
    int s02, int s03, int s1, int s2, int s3, int n_dims,
    const int32_t* positions, float freq_scale, float ext_factor,
    float attn_factor, RopeCorrDims corr, float theta_scale,
    const float* freq_factors, const int64_t* row_indices,
    int set_rows_stride) {
    (void)freq_factors;
    const int i0 = 2 * (static_cast<int>(blockDim.y) * blockIdx.y + threadIdx.y);
    if (i0 >= ne00) return;
    const int row = static_cast<int>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int rows12 = ne01 * ne02;
    const int i3 = row / rows12;
    const int rem = row - i3 * rows12;
    const int i2 = rem / ne01;
    const int i1 = rem - i2 * ne01;
    int out = i0 + i1 * s1 + i2 * s2 + i3 * s3;
    const int in = i0 + i1 * s01 + i2 * s02 + i3 * s03;
    if (set_rows_stride != 0) out = i1 * s1 + i0 + row_indices[i2] * set_rows_stride;
    const float x0 = src[in];
    const float x1 = src[in + 1];
    if (i0 >= n_dims) {
        dst[out] = __float2half_rn(x0);
        dst[out + 1] = __float2half_rn(x1);
        return;
    }
    const float extrap = positions[i2] * powf(theta_scale, 0.5f * i0);
    const float interp = freq_scale * extrap;
    float theta = interp;
    float scale = attn_factor;
    if (ext_factor != 0.0f) {
        const float denominator = fmaxf(0.001f, corr.v[1] - corr.v[0]);
        const float ramp = 1.0f - fminf(1.0f, fmaxf(0.0f, (0.5f * i0 - corr.v[0]) / denominator));
        const float mix = ramp * ext_factor;
        theta = interp * (1.0f - mix) + extrap * mix;
        scale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    const float cosine = cosf(theta) * scale;
    const float sine = sinf(theta) * scale;
    dst[out] = __float2half_rn(x0 * cosine - x1 * sine);
    dst[out + 1] = __float2half_rn(x0 * sine + x1 * cosine);
}

extern "C" __global__ void convert_unaryIfDF16_E_probe(
    const void* source, __half* dst, int64_t ne00, int64_t ne01,
    int64_t ne0203, uint3 ne02_fd, int64_t s01, int64_t s02,
    int64_t s03) {
    const int64_t i00 = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i00 >= ne00 || ne02_fd.z == 0) return;
    const int64_t i01 = blockIdx.y;
    const float* src = static_cast<const float*>(source);
    for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
        const int64_t i02 = i0203 % ne02_fd.z;
        const int64_t i03 = i0203 / ne02_fd.z;
        const int64_t in = i03 * s03 + i02 * s02 + i01 * s01 + i00;
        const int64_t out = (i0203 * ne01 + i01) * ne00 + i00;
        dst[out] = __float2half_rn(src[in]);
    }
}

extern "C" __global__ void convert_unaryIDF16_fE_probe(
    const void* source, float* dst, int64_t ne00, int64_t ne01,
    int64_t ne0203, uint3 ne02_fd, int64_t s01, int64_t s02,
    int64_t s03) {
    const int64_t i00 = static_cast<int64_t>(blockDim.x) * blockIdx.x + threadIdx.x;
    if (i00 >= ne00 || ne02_fd.z == 0) return;
    const int64_t i01 = blockIdx.y;
    const __half* src = static_cast<const __half*>(source);
    for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
        const int64_t i02 = i0203 % ne02_fd.z;
        const int64_t i03 = i0203 / ne02_fd.z;
        const int64_t in = i03 * s03 + i02 * s02 + i01 * s01 + i00;
        const int64_t out = (i0203 * ne01 + i01) * ne00 + i00;
        dst[out] = __half2float(src[in]);
    }
}

namespace {

bool near(float actual, float expected, float tolerance = 2.0e-3f) {
    return std::fabs(actual - expected) <=
           tolerance * std::max(1.0f, std::fabs(expected));
}

bool check_cuda(const char* operation) {
    const cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s returned CUDA error %d\n", operation,
                     static_cast<int>(status));
        return false;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s synchronization\n", operation);
        return false;
    }
    return true;
}

bool test_rms_norm() {
    constexpr int ncols = 7;
    constexpr int nrows = 3;
    constexpr int stride_row = 11;
    std::vector<float> input(stride_row * nrows, -99.0f);
    std::vector<float> mul(ncols);
    std::vector<float> output(ncols * nrows, 0.0f);
    for (int row = 0; row < nrows; ++row) {
        for (int col = 0; col < ncols; ++col) {
            input[row * stride_row + col] =
                0.25f * static_cast<float>(1 + row * ncols + col);
        }
    }
    for (int col = 0; col < ncols; ++col) {
        mul[col] = 0.75f + 0.1f * static_cast<float>(col);
    }

    float *d_input = nullptr, *d_mul = nullptr, *d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_mul), mul.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_mul, mul.data(), mul.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    const uint3 ncols_packed = make_uint3(0, 0, ncols);
    const uint3 one_packed = make_uint3(0, 0, 1);
    const uint3 zero_packed = make_uint3(0, 0, 0);
    rms_norm_f32_probe<<<dim3(nrows, 1, 1), dim3(256, 1, 1)>>>(
        d_input, d_output, ncols, stride_row, 0, 0, 1.0e-5f,
        d_mul, 0, 0, 0, ncols_packed, one_packed, one_packed, one_packed,
        nullptr, 0, 0, 0, zero_packed, zero_packed, zero_packed, zero_packed);
    if (!check_cuda("rms_norm_f32_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < ncols; ++col) {
            const float value = input[row * stride_row + col];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / ncols + 1.0e-5f);
        for (int col = 0; col < ncols; ++col) {
            const float expected =
                input[row * stride_row + col] * scale * mul[col];
            if (!near(output[row * ncols + col], expected)) {
                std::fprintf(stderr,
                             "FAIL: rms row=%d col=%d got=%f expected=%f\n",
                             row, col, output[row * ncols + col], expected);
                return false;
            }
        }
    }
    cudaFree(d_input);
    cudaFree(d_mul);
    cudaFree(d_output);
    return true;
}

bool test_rms_norm_3d() {
    constexpr int ncols = 64;
    constexpr int nrows = 5;
    constexpr int nchannels = 4;
    constexpr int nsamples = 3;
    constexpr int stride_row = 71;
    constexpr int stride_channel = stride_row * nrows + 13;
    constexpr int stride_sample = stride_channel * nchannels + 17;
    constexpr int dense_values =
        ncols * nrows * nchannels * nsamples;
    std::vector<float> input(stride_sample * nsamples, -99.0f);
    std::vector<float> output(dense_values, 0.0f);
    for (int sample = 0; sample < nsamples; ++sample) {
        for (int channel = 0; channel < nchannels; ++channel) {
            for (int row = 0; row < nrows; ++row) {
                for (int col = 0; col < ncols; ++col) {
                    const int source =
                        sample * stride_sample + channel * stride_channel +
                        row * stride_row + col;
                    input[source] =
                        0.03125f *
                        static_cast<float>(1 + col + 3 * row + 7 * channel +
                                           11 * sample);
                }
            }
        }
    }

    constexpr std::size_t input_offset = 1u * 1024u * 1024u;
    constexpr std::size_t output_offset = 6u * 1024u * 1024u;
    constexpr std::size_t arena_bytes = 9u * 1024u * 1024u;
    void* d_arena = nullptr;
    cudaMalloc(&d_arena, arena_bytes);
    float* d_input = reinterpret_cast<float*>(
        static_cast<unsigned char*>(d_arena) + input_offset);
    float* d_output = reinterpret_cast<float*>(
        static_cast<unsigned char*>(d_arena) + output_offset);
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    rms_norm_f32_plain_probe<<<dim3(nrows, nchannels, nsamples),
                               dim3(256, 1, 1), 32 * sizeof(float)>>>(
        d_input, d_output, ncols, stride_row, stride_channel, stride_sample,
        1.0e-6f);
    if (!check_cuda("rms_norm_f32_plain_probe 3d")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (int sample = 0; sample < nsamples; ++sample) {
        for (int channel = 0; channel < nchannels; ++channel) {
            for (int row = 0; row < nrows; ++row) {
                float sum = 0.0f;
                for (int col = 0; col < ncols; ++col) {
                    const int source =
                        sample * stride_sample + channel * stride_channel +
                        row * stride_row + col;
                    sum += input[source] * input[source];
                }
                const float scale =
                    1.0f / std::sqrt(sum / ncols + 1.0e-6f);
                for (int col = 0; col < ncols; ++col) {
                    const int source =
                        sample * stride_sample + channel * stride_channel +
                        row * stride_row + col;
                    const int dense =
                        (((sample * nchannels + channel) * nrows + row) *
                             ncols +
                         col);
                    const float expected = input[source] * scale;
                    if (!near(output[dense], expected)) {
                        std::fprintf(
                            stderr,
                            "FAIL: rms3d s=%d c=%d r=%d x=%d got=%f "
                            "expected=%f\n",
                            sample, channel, row, col, output[dense],
                            expected);
                        return false;
                    }
                }
            }
        }
    }
    cudaFree(d_arena);
    return true;
}

bool test_rms_norm_ping_pong() {
    constexpr int ncols = 576;
    constexpr int nrows = 35;
    constexpr int values = ncols * nrows;
    constexpr int iterations = 64;
    constexpr std::size_t tensor_bytes = values * sizeof(float);
    std::vector<float> current(values);
    std::vector<float> output(values);
    std::vector<float> expected(values);
    for (int i = 0; i < values; ++i) {
        current[i] =
            std::sin(0.017f * static_cast<float>(i + 1)) * 7.0f +
            0.03125f * static_cast<float>(i % 23);
    }

    void* d_arena = nullptr;
    cudaMalloc(&d_arena, 2 * tensor_bytes);
    float* buffers[2] = {
        reinterpret_cast<float*>(d_arena),
        reinterpret_cast<float*>(
            static_cast<unsigned char*>(d_arena) + tensor_bytes),
    };
    cudaMemcpy(buffers[0], current.data(), tensor_bytes, cudaMemcpyHostToDevice);

    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int src = iteration & 1;
        const int dst = src ^ 1;
        rms_norm_f32_plain_probe<<<dim3(nrows, 1, 1),
                                   dim3(256, 1, 1), 32 * sizeof(float)>>>(
            buffers[src], buffers[dst], ncols, ncols, values, values,
            1.0e-6f);
        if (!check_cuda("rms_norm_f32_plain_probe ping-pong")) return false;
        cudaMemcpy(output.data(), buffers[dst], tensor_bytes,
                   cudaMemcpyDeviceToHost);

        for (int row = 0; row < nrows; ++row) {
            double sum = 0.0;
            for (int col = 0; col < ncols; ++col) {
                const float value = current[row * ncols + col];
                sum += static_cast<double>(value) * value;
            }
            const float scale =
                1.0f /
                std::sqrt(static_cast<float>(sum / ncols) + 1.0e-6f);
            for (int col = 0; col < ncols; ++col) {
                expected[row * ncols + col] =
                    current[row * ncols + col] * scale;
            }
        }
        for (int i = 0; i < values; ++i) {
            if (!near(output[i], expected[i], 4.0e-5f)) {
                std::fprintf(
                    stderr,
                    "FAIL: rms ping-pong iteration=%d i=%d got=%f "
                    "expected=%f\n",
                    iteration, i, output[i], expected[i]);
                return false;
            }
        }
        current.swap(output);
    }

    cudaFree(d_arena);
    return true;
}

bool test_bin_bcast() {
    constexpr int ne0 = 7;
    constexpr int ne1 = 3;
    std::vector<float> src0(ne0 * ne1);
    std::vector<float> extra(ne0);
    std::vector<float> output(ne0 * ne1, 0.0f);
    for (int i = 0; i < ne0 * ne1; ++i) src0[i] = 0.25f * i;
    for (int i = 0; i < ne0; ++i) extra[i] = 10.0f + i;

    float *d_src0 = nullptr, *d_src1 = nullptr, *d_extra = nullptr,
          *d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_src0), src0.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_src1), extra.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_extra), extra.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(float));
    cudaMemcpy(d_src0, src0.data(), src0.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_src1, extra.data(), extra.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_extra, extra.data(), extra.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    const uint3 one = make_uint3(0, 0, 1);
    const uint3 seven = make_uint3(0, 0, ne0);
    k_bin_bcast_op_addff_probe<<<dim3(1, ne1, 1), dim3(128, 1, 1)>>>(
        d_src0, d_src1, d_output, ne0, ne1, 1, one, seven, one, one, one,
        ne0, ne0 * ne1, ne0 * ne1, 1, ne0, ne0 * ne1, ne0 * ne1,
        1, 0, 0, 0, d_extra);
    if (!check_cuda("k_bin_bcast_op_addff_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < ne0 * ne1; ++i) {
        const float expected = src0[i] + extra[i % ne0];
        if (!near(output[i], expected, 1.0e-6f)) {
            std::fprintf(stderr, "FAIL: bcast i=%d got=%f expected=%f\n", i,
                         output[i], expected);
            return false;
        }
    }
    cudaFree(d_src0);
    cudaFree(d_src1);
    cudaFree(d_extra);
    cudaFree(d_output);
    return true;
}

#pragma pack(push, 1)
struct BlockQ8 {
    __half scale;
    int8_t quant[32];
};
struct BlockQ6K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    __half delta;
};
#pragma pack(pop)
static_assert(sizeof(BlockQ8) == 34, "Q8_0 block layout");
static_assert(sizeof(BlockQ6K) == 210, "Q6_K block layout");

bool test_q8_dequant() {
    constexpr int blocks = 64;
    constexpr int values = blocks * 32;
    std::vector<BlockQ8> packed(blocks);
    std::vector<__half> output(values);
    for (int block = 0; block < blocks; ++block) {
        packed[block].scale = static_cast<__half>(0.125f * (1 + block % 5));
        for (int lane = 0; lane < 32; ++lane) {
            packed[block].quant[lane] =
                static_cast<int8_t>((lane * 7 + block) % 31 - 15);
        }
    }
    BlockQ8* d_packed = nullptr;
    __half* d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_packed), packed.size() * sizeof(BlockQ8));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(__half));
    cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(BlockQ8),
               cudaMemcpyHostToDevice);
    dequantize_block_q8_0_f16_probe<<<1, 32>>>(d_packed, d_output, values);
    if (!check_cuda("dequantize_block_q8_0_f16_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < values; ++i) {
        const float expected = static_cast<float>(packed[i / 32].scale) *
                               packed[i / 32].quant[i % 32];
        if (!near(static_cast<float>(output[i]), expected)) {
            std::fprintf(stderr, "FAIL: q8 i=%d got=%f expected=%f\n", i,
                         static_cast<float>(output[i]), expected);
            return false;
        }
    }
    cudaFree(d_packed);
    cudaFree(d_output);
    return true;
}

bool test_gated_silu() {
    constexpr int64_t n = 5;
    constexpr int64_t rows = 3;
    constexpr int64_t k = n * rows;
    constexpr int64_t o0 = 8;
    constexpr int64_t o1 = 9;
    std::vector<float> x(o0 * rows, -99.0f);
    std::vector<float> gate(o1 * rows, -77.0f);
    std::vector<float> output(k, 0.0f);
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < n; ++col) {
            x[row * o0 + col] =
                0.25f * static_cast<float>(col - 2 + row);
            gate[row * o1 + col] =
                0.125f * static_cast<float>(1 + col + 2 * row);
        }
    }

    float *d_x = nullptr, *d_gate = nullptr, *d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_x), x.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_gate), gate.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    unary_gated_op_kernel_op_silu_EEfEv_probe<<<1, 256>>>(
        d_x, d_gate, d_output, k, n, o0, o1);
    if (!check_cuda("unary_gated_op_kernel_op_silu_EEfEv_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int64_t i = 0; i < k; ++i) {
        const int64_t row = i / n;
        const int64_t col = i % n;
        const float value = x[row * o0 + col];
        const float expected =
            value / (1.0f + std::exp(-value)) * gate[row * o1 + col];
        if (!near(output[i], expected, 1.0e-5f)) {
            std::fprintf(stderr,
                         "FAIL: gated silu i=%lld got=%f expected=%f\n",
                         static_cast<long long>(i), output[i], expected);
            return false;
        }
    }
    cudaFree(d_x);
    cudaFree(d_gate);
    cudaFree(d_output);
    return true;
}

bool test_rope_norm_f32_f16() {
    constexpr int ne00 = 6;
    constexpr int ne01 = 2;
    constexpr int ne02 = 2;
    constexpr int s01 = 8;
    constexpr int s02 = 20;
    constexpr int s03 = 40;
    constexpr int s1 = ne00;
    constexpr int s2 = ne00 * ne01;
    constexpr int s3 = ne00 * ne01 * ne02;
    constexpr int rows = ne01 * ne02;
    std::vector<float> input(s03, -99.0f);
    std::vector<__half> output(rows * ne00);
    const int32_t positions[ne02] = {1, 3};
    for (int i2 = 0; i2 < ne02; ++i2) {
        for (int i1 = 0; i1 < ne01; ++i1) {
            for (int i0 = 0; i0 < ne00; ++i0) {
                input[i2 * s02 + i1 * s01 + i0] =
                    0.125f * static_cast<float>(1 + i0 + 7 * i1 + 13 * i2);
            }
        }
    }
    float* d_input = nullptr;
    __half* d_output = nullptr;
    int32_t* d_pos = nullptr;
    float* d_freq = nullptr;
    int64_t* d_rows = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_pos), sizeof(positions));
    cudaMalloc(reinterpret_cast<void**>(&d_freq), sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_rows), ne02 * sizeof(int64_t));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, positions, sizeof(positions), cudaMemcpyHostToDevice);
    const RopeCorrDims corr{{0.0f, 1.0f}};
    constexpr float freq_scale = 1.0f;
    constexpr float ext_factor = 0.0f;
    constexpr float attn_factor = 1.0f;
    constexpr float theta_scale = 0.25f;
    rope_normILb1ELb0EfDF16_Ev_probe<<<dim3(rows, 1, 1), dim3(1, 256, 1)>>>(
        d_input, d_output, ne00, ne01, ne02, s01, s02, s03, s1, s2, s3,
        ne00, d_pos, freq_scale, ext_factor, attn_factor, corr, theta_scale,
        d_freq, d_rows, 0);
    if (!check_cuda("rope_normILb1ELb0EfDF16_Ev_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    for (int i2 = 0; i2 < ne02; ++i2) {
        for (int i1 = 0; i1 < ne01; ++i1) {
            for (int i0 = 0; i0 < ne00; i0 += 2) {
                const float x0 = input[i2 * s02 + i1 * s01 + i0];
                const float x1 = input[i2 * s02 + i1 * s01 + i0 + 1];
                const float theta =
                    static_cast<float>(positions[i2]) *
                    std::pow(theta_scale, 0.5f * static_cast<float>(i0));
                const float expected0 =
                    x0 * std::cos(theta) - x1 * std::sin(theta);
                const float expected1 =
                    x0 * std::sin(theta) + x1 * std::cos(theta);
                const int out = (i2 * ne01 + i1) * ne00 + i0;
                if (!near(static_cast<float>(output[out]), expected0) ||
                    !near(static_cast<float>(output[out + 1]), expected1)) {
                    std::fprintf(stderr,
                                 "FAIL: rope i=%d got=(%f,%f) expected=(%f,%f)\n",
                                 out, static_cast<float>(output[out]),
                                 static_cast<float>(output[out + 1]),
                                 expected0, expected1);
                    return false;
                }
            }
        }
    }
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_pos);
    cudaFree(d_freq);
    cudaFree(d_rows);
    return true;
}

bool test_q6_k_dequant() {
    constexpr int blocks = 3;
    constexpr int values_per_block = 256;
    std::vector<BlockQ6K> packed(blocks);
    std::vector<__half> output(blocks * values_per_block);
    std::vector<float> expected(output.size(), 0.0f);
    for (int block = 0; block < blocks; ++block) {
        BlockQ6K& value = packed[block];
        value.delta = static_cast<__half>(0.03125f * (block + 1));
        for (int i = 0; i < 128; ++i) {
            value.ql[i] = static_cast<uint8_t>((13 * i + 17 * block) & 0xff);
        }
        for (int i = 0; i < 64; ++i) {
            value.qh[i] = static_cast<uint8_t>((29 * i + 7 * block) & 0xff);
        }
        for (int i = 0; i < 16; ++i) {
            value.scales[i] = static_cast<int8_t>((5 * i + 3 * block) % 17 - 8);
        }
        const float delta = static_cast<float>(value.delta);
        for (int tid = 0; tid < 64; ++tid) {
            const int ip = tid / 32;
            const int il = tid % 32;
            const int is = 8 * ip + il / 16;
            const int ql_index = 64 * ip + il;
            const uint8_t qh = value.qh[32 * ip + il];
            const uint8_t ql0 = value.ql[ql_index];
            const uint8_t ql32 = value.ql[ql_index + 32];
            const int quant[4] = {
                static_cast<int>((ql0 & 0x0f) | (((qh >> 0) & 3) << 4)) - 32,
                static_cast<int>((ql32 & 0x0f) | (((qh >> 2) & 3) << 4)) - 32,
                static_cast<int>((ql0 >> 4) | (((qh >> 4) & 3) << 4)) - 32,
                static_cast<int>((ql32 >> 4) | (((qh >> 6) & 3) << 4)) - 32,
            };
            const int out = block * values_per_block + 128 * ip + il;
            for (int segment = 0; segment < 4; ++segment) {
                expected[out + 32 * segment] =
                    delta * value.scales[is + 2 * segment] * quant[segment];
            }
        }
    }

    BlockQ6K* d_packed = nullptr;
    __half* d_output = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_packed), packed.size() * sizeof(BlockQ6K));
    cudaMalloc(reinterpret_cast<void**>(&d_output), output.size() * sizeof(__half));
    cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(BlockQ6K),
               cudaMemcpyHostToDevice);
    dequantize_block_q6_KIDF16_E_probe<<<blocks, 64>>>(d_packed, d_output);
    if (!check_cuda("dequantize_block_q6_KIDF16_E_probe")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    for (std::size_t i = 0; i < output.size(); ++i) {
        const float rounded = static_cast<float>(static_cast<__half>(expected[i]));
        if (!near(static_cast<float>(output[i]), rounded, 1.0e-6f)) {
            std::fprintf(stderr, "FAIL: q6_K i=%zu got=%f expected=%f\n", i,
                         static_cast<float>(output[i]), rounded);
            return false;
        }
    }
    cudaFree(d_packed);
    cudaFree(d_output);
    return true;
}

bool test_convert() {
    constexpr int ne00 = 9;
    constexpr int ne01 = 2;
    constexpr int ne02 = 2;
    constexpr int ne03 = 2;
    constexpr int s01 = 13;
    constexpr int s02 = 31;
    constexpr int s03 = 67;
    constexpr int total = ne00 * ne01 * ne02 * ne03;
    std::vector<float> input(s03 * ne03, -123.0f);
    std::vector<float> output(total, 0.0f);
    std::vector<__half> half_values(total);
    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            for (int i01 = 0; i01 < ne01; ++i01) {
                for (int i00 = 0; i00 < ne00; ++i00) {
                    input[i03 * s03 + i02 * s02 + i01 * s01 + i00] =
                        0.03125f * (1 + i00 + 10 * i01 + 20 * i02 + 40 * i03);
                }
            }
        }
    }
    float *d_input = nullptr, *d_output = nullptr;
    __half* d_half = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_input), input.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_half), total * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_output), total * sizeof(float));
    cudaMemcpy(d_input, input.data(), input.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    const uint3 divisor = make_uint3(0, 0, ne02);
    convert_unaryIfDF16_E_probe<<<dim3(1, ne01, ne02 * ne03), 256>>>(
        d_input, d_half, ne00, ne01, ne02 * ne03, divisor, s01, s02, s03);
    if (!check_cuda("convert f32-to-f16")) return false;
    convert_unaryIDF16_fE_probe<<<dim3(1, ne01, ne02 * ne03), 256>>>(
        d_half, d_output, ne00, ne01, ne02 * ne03, divisor,
        ne00, ne00 * ne01, ne00 * ne01 * ne02);
    if (!check_cuda("convert f16-to-f32")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    for (int i03 = 0; i03 < ne03; ++i03) {
        for (int i02 = 0; i02 < ne02; ++i02) {
            for (int i01 = 0; i01 < ne01; ++i01) {
                for (int i00 = 0; i00 < ne00; ++i00) {
                    const int dense =
                        ((i03 * ne02 + i02) * ne01 + i01) * ne00 + i00;
                    const float expected =
                        input[i03 * s03 + i02 * s02 + i01 * s01 + i00];
                    if (!near(output[dense], expected)) {
                        std::fprintf(stderr,
                                     "FAIL: convert i=%d got=%f expected=%f\n",
                                     dense, output[dense], expected);
                        return false;
                    }
                }
            }
        }
    }
    cudaFree(d_input);
    cudaFree(d_half);
    cudaFree(d_output);
    return true;
}

bool test_integrated_output_head() {
    // Exact SmolLM2 decode-time output projection shape. This deliberately
    // exercises every packed Q8_0 weight and every output logit; sparse or
    // toy-sized GEMM checks can miss large-shape indexing/orientation bugs.
    constexpr int m = 49152;
    constexpr int n = 1;
    constexpr int k = 576;
    constexpr int weight_values = m * k;
    constexpr int weight_blocks = weight_values / 32;
    static_assert(weight_values % 2048 == 0, "exercise full Q8 groups");

    std::vector<BlockQ8> packed(weight_blocks);
    std::vector<float> activation(k * n);
    std::vector<float> output(m * n, 0.0f);
    std::vector<float> expected(m * n, 0.0f);
    for (int block = 0; block < weight_blocks; ++block) {
        packed[block].scale =
            static_cast<__half>(0.00390625f * (1 + block % 4));
        for (int lane = 0; lane < 32; ++lane) {
            packed[block].quant[lane] =
                static_cast<int8_t>(1 + (block + 5 * lane) % 7);
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            activation[row + col * k] =
                0.00390625f *
                static_cast<float>(1 + row % 13 + 2 * col);
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int inner = 0; inner < k; ++inner) {
                const int weight_index = inner + row * k;
                const BlockQ8& block = packed[weight_index / 32];
                const float weight =
                    static_cast<float>(block.scale) *
                    static_cast<float>(block.quant[weight_index % 32]);
                // The GPU activation is rounded to half before GEMM.
                const float value = static_cast<float>(
                    static_cast<__half>(activation[inner + col * k]));
                sum += weight * value;
            }
            expected[row + col * m] = sum;
        }
    }

    BlockQ8* d_packed = nullptr;
    float *d_activation = nullptr, *d_output = nullptr;
    __half *d_weights = nullptr, *d_activation_half = nullptr,
           *d_output_half = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&d_packed),
               packed.size() * sizeof(BlockQ8));
    cudaMalloc(reinterpret_cast<void**>(&d_weights),
               weight_values * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_activation),
               activation.size() * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&d_activation_half),
               activation.size() * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_output_half),
               output.size() * sizeof(__half));
    cudaMalloc(reinterpret_cast<void**>(&d_output),
               output.size() * sizeof(float));
    cudaMemcpy(d_packed, packed.data(), packed.size() * sizeof(BlockQ8),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_activation, activation.data(), activation.size() * sizeof(float),
               cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;
    cudaStreamCreate(&stream);
    cublasCreate(&handle);
    cublasSetStream(handle, stream);

    dequantize_block_q8_0_f16_probe<<<weight_values / 2048, 32, 0, stream>>>(
        d_packed, d_weights, weight_values);
    const uint3 one = make_uint3(0, 0, 1);
    convert_unaryIfDF16_E_probe<<<dim3((k + 255) / 256, n, 1), 256, 0,
                                        stream>>>(
        d_activation, d_activation_half, k, n, 1, one, k, k * n, k * n);
    cudaStreamSynchronize(stream);
    std::vector<__half> first_weight_row(k);
    std::vector<__half> activation_half(k);
    cudaMemcpy(first_weight_row.data(), d_weights,
               first_weight_row.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(activation_half.data(), d_activation_half,
               activation_half.size() * sizeof(__half),
               cudaMemcpyDeviceToHost);
    for (int inner = 0; inner < k; ++inner) {
        const BlockQ8& block = packed[inner / 32];
        const float expected_weight = static_cast<float>(
            static_cast<__half>(
                static_cast<float>(block.scale) *
                static_cast<float>(block.quant[inner % 32])));
        if (!near(static_cast<float>(first_weight_row[inner]),
                  expected_weight, 1.0e-6f)) {
            std::fprintf(stderr,
                         "FAIL: integrated dequant i=%d got=%f expected=%f\n",
                         inner, static_cast<float>(first_weight_row[inner]),
                         expected_weight);
            return false;
        }
        if (!near(static_cast<float>(activation_half[inner]),
                  activation[inner], 1.0e-6f)) {
            std::fprintf(stderr,
                         "FAIL: integrated activation i=%d got=%f "
                         "expected=%f\n",
                         inner, static_cast<float>(activation_half[inner]),
                         activation[inner]);
            return false;
        }
    }

    const __half alpha = static_cast<__half>(1.0f);
    const __half beta = static_cast<__half>(0.0f);
    const cublasStatus_t gemm_status = cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k,
        &alpha, d_weights, CUDA_R_16F, k,
                d_activation_half, CUDA_R_16F, k,
        &beta,  d_output_half, CUDA_R_16F, m,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (gemm_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: integrated GemmEx status=%d\n",
                     static_cast<int>(gemm_status));
        return false;
    }
    convert_unaryIDF16_fE_probe<<<dim3((m + 255) / 256, n, 1), 256, 0,
                                      stream>>>(
        d_output_half, d_output, m, n, 1, one, m, m * n, m * n);
    if (!check_cuda("integrated output-head chain")) return false;
    cudaMemcpy(output.data(), d_output, output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    for (std::size_t i = 0; i < output.size(); ++i) {
        // GemmEx writes half, so compare with the same final rounding.
        const float rounded =
            static_cast<float>(static_cast<__half>(expected[i]));
        if (!near(output[i], rounded, 3.0e-3f)) {
            std::fprintf(stderr,
                         "FAIL: integrated output i=%zu got=%f expected=%f\n",
                         i, output[i], rounded);
            return false;
        }
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(d_packed);
    cudaFree(d_weights);
    cudaFree(d_activation);
    cudaFree(d_activation_half);
    cudaFree(d_output_half);
    cudaFree(d_output);
    return true;
}

}  // namespace

int main() {
    if (!test_rms_norm() || !test_rms_norm_3d() ||
        !test_rms_norm_ping_pong() || !test_bin_bcast() ||
        !test_q8_dequant() || !test_gated_silu() ||
        !test_rope_norm_f32_f16() || !test_q6_k_dequant() ||
        !test_convert() || !test_integrated_output_head()) {
        return 1;
    }
    std::puts("PASS: GGML output-head kernels match CPU references on Apple GPU");
    return 0;
}
