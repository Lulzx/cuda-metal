#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
}

namespace {

struct CorrDims {
    float low;
    float high;
};

constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;

struct FatbinWrapper {
    std::uint32_t magic = kFatbinWrapperMagic;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinBlobHeader {
    std::uint32_t magic = kFatbinBlobMagic;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

void rope_neox_float_host_stub() {}
void rope_neox_half_host_stub() {}

float half_to_float(std::uint16_t bits) {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000u) << 16u;
    std::uint32_t exponent = (bits >> 10u) & 0x1fu;
    std::uint32_t mantissa = bits & 0x03ffu;
    std::uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1u;
                ++shift;
            }
            mantissa &= 0x03ffu;
            result = sign |
                     (static_cast<std::uint32_t>(127 - 15 - shift) << 23u) |
                     (mantissa << 13u);
        }
    } else if (exponent == 0x1fu) {
        result = sign | 0x7f800000u | (mantissa << 13u);
    } else {
        result = sign | ((exponent + (127u - 15u)) << 23u) | (mantissa << 13u);
    }
    float value = 0.0f;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

std::vector<float> reference_rope_neox(const std::vector<float>& input,
                                       const std::vector<int>& positions,
                                       int ne00,
                                       int ne01,
                                       int ne02,
                                       int n_dims,
                                       float freq_scale,
                                       float ext_factor,
                                       float attn_factor,
                                       CorrDims corr_dims,
                                       float theta_scale) {
    std::vector<float> output(input.size(), 0.0f);
    const int row_count = static_cast<int>(input.size()) / ne00;
    for (int row = 0; row < row_count; ++row) {
        const int i2 = (row % (ne01 * ne02)) / ne01;
        const int base = row * ne00;
        for (int i0 = 0; i0 < ne00; i0 += 2) {
            if (i0 >= n_dims) {
                output[base + i0] = input[base + i0];
                output[base + i0 + 1] = input[base + i0 + 1];
                continue;
            }
            const float theta_extrap =
                static_cast<float>(positions[i2]) *
                std::pow(theta_scale, static_cast<float>(i0) * 0.5f);
            const float theta_interp = freq_scale * theta_extrap;
            float theta = theta_interp;
            float mscale = attn_factor;
            if (ext_factor != 0.0f) {
                const float y =
                    (static_cast<float>(i0 / 2) - corr_dims.low) /
                    std::max(0.001f, corr_dims.high - corr_dims.low);
                const float ramp = 1.0f - std::clamp(y, 0.0f, 1.0f);
                const float mix = ramp * ext_factor;
                theta = theta_interp * (1.0f - mix) + theta_extrap * mix;
                mscale *= 1.0f + 0.1f * std::log(1.0f / freq_scale);
            }
            const float cosine = std::cos(theta) * mscale;
            const float sine = std::sin(theta) * mscale;
            const float x0 = input[base + i0 / 2];
            const float x1 = input[base + i0 / 2 + n_dims / 2];
            output[base + i0 / 2] = x0 * cosine - x1 * sine;
            output[base + i0 / 2 + n_dims / 2] = x0 * sine + x1 * cosine;
        }
    }
    return output;
}

bool launch_variant(const void* kernel,
                    void* input,
                    void* output,
                    void* positions,
                    void* freq_factors,
                    void* row_indices) {
    int ne00 = 8;
    int ne01 = 2;
    int ne02 = 2;
    int s01 = 8;
    int s02 = 16;
    int s03 = 32;
    int s1 = 8;
    int s2 = 16;
    int s3 = 32;
    int n_dims = 4;
    float freq_scale = 0.5f;
    float ext_factor = 0.6f;
    float attn_factor = 1.1f;
    CorrDims corr_dims{0.0f, 2.0f};
    float theta_scale = 0.5f;
    int set_rows_stride = 0;
    void* params[] = {
        &input,          &output,         &ne00,          &ne01,
        &ne02,           &s01,            &s02,           &s03,
        &s1,             &s2,             &s3,            &n_dims,
        &positions,      &freq_scale,      &ext_factor,    &attn_factor,
        &corr_dims,      &theta_scale,     &freq_factors,  &row_indices,
        &set_rows_stride, nullptr,
    };
    const cudaError_t status =
        cudaLaunchKernel(kernel, dim3(4, 1, 1), dim3(1, 4, 1), params, 0, nullptr);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel returned %d\n", status);
    }
    return status == cudaSuccess;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "usage: %s <rope-neox.ptx>\n", argv[0]);
        return argc < 2 ? 64 : 77;
    }
    std::ifstream input_file(argv[1], std::ios::binary);
    std::vector<char> ptx((std::istreambuf_iterator<char>(input_file)),
                          std::istreambuf_iterator<char>());
    if (ptx.empty()) {
        std::fprintf(stderr, "FAIL: PTX fixture is empty\n");
        return 1;
    }

    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx.size(), 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx.size());
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx.data(), ptx.size());
    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: runtime initialization failed\n");
        return 1;
    }
    void** fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: PTX fatbinary registration failed\n");
        return 1;
    }
    char float_name[] =
        "_ZL9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiiiiPKifff"
        "14rope_corr_dimsfPKfPKxi";
    char half_name[] =
        "_ZL9rope_neoxILb1ELb0EfDF16_EvPKT1_PT2_iiiiiiiiiiPKifff"
        "14rope_corr_dimsfPKfPKxi";
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&rope_neox_float_host_stub),
        float_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&rope_neox_half_host_stub),
        half_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);

    constexpr int kElementCount = 32;
    std::vector<float> host_input(kElementCount);
    for (int i = 0; i < kElementCount; ++i) {
        host_input[i] = 0.125f * static_cast<float>(i - 11);
    }
    const std::vector<int> host_positions = {1, 3};
    const auto expected = reference_rope_neox(
        host_input, host_positions, 8, 2, 2, 4, 0.5f, 0.6f, 1.1f,
        CorrDims{0.0f, 2.0f}, 0.5f);

    void* device_input = nullptr;
    void* device_float_output = nullptr;
    void* device_half_output = nullptr;
    void* device_positions = nullptr;
    void* device_freq_factors = nullptr;
    void* device_row_indices = nullptr;
    if (cudaMalloc(&device_input, host_input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_float_output, expected.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_half_output, expected.size() * sizeof(std::uint16_t)) != cudaSuccess ||
        cudaMalloc(&device_positions, host_positions.size() * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&device_freq_factors, 2 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_row_indices, 2 * sizeof(std::int64_t)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input.data(), host_input.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_positions, host_positions.data(),
                   host_positions.size() * sizeof(int),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device allocation/copy failed\n");
        return 1;
    }

    if (!launch_variant(reinterpret_cast<const void*>(&rope_neox_float_host_stub),
                        device_input, device_float_output,
                        device_positions, device_freq_factors, device_row_indices) ||
        !launch_variant(reinterpret_cast<const void*>(&rope_neox_half_host_stub),
                        device_input, device_half_output,
                        device_positions, device_freq_factors, device_row_indices) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: rope_neox launch failed\n");
        return 1;
    }

    std::vector<float> float_output(expected.size());
    std::vector<std::uint16_t> half_output(expected.size());
    if (cudaMemcpy(float_output.data(), device_float_output,
                   float_output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(half_output.data(), device_half_output,
                   half_output.size() * sizeof(std::uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: output copy failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(float_output[i] - expected[i]) > 2e-5f) {
            std::fprintf(stderr, "FAIL: float output[%zu]=%g expected=%g\n",
                         i, float_output[i], expected[i]);
            return 1;
        }
        const float half_value = half_to_float(half_output[i]);
        if (std::fabs(half_value - expected[i]) > 2e-3f) {
            std::fprintf(stderr, "FAIL: half output[%zu]=%g expected=%g\n",
                         i, half_value, expected[i]);
            return 1;
        }
    }

    cudaFree(device_input);
    cudaFree(device_float_output);
    cudaFree(device_half_output);
    cudaFree(device_positions);
    cudaFree(device_freq_factors);
    cudaFree(device_row_indices);
    __cudaUnregisterFatBinary(fatbin_handle);
    std::printf("PASS: exact forward no-FF GPT-NeoX RoPE float/half outputs\n");
    return 0;
}
