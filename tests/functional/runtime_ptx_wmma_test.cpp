#include "cuda_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::uint16_t float_to_half(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 112;
    const std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        return static_cast<std::uint16_t>(sign);
    }
    if (exponent >= 31) {
        return static_cast<std::uint16_t>(sign | 0x7bffu);
    }
    return static_cast<std::uint16_t>(sign |
                                      (static_cast<std::uint32_t>(exponent) << 10) |
                                      (mantissa >> 13));
}

float a_value(int row, int k) { return static_cast<float>((row * 3 + k) % 5) * 0.5f; }
float b_value(int k, int col) { return static_cast<float>((k + col * 2) % 7) * 0.25f; }
float c_value(int row, int col) { return static_cast<float>((row + col) % 3) * 0.5f; }

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-msl-source>\n", argv[0]);
        return 64;
    }
    const std::string msl_path = argv[1];
    if (!std::filesystem::exists(msl_path)) {
        std::fprintf(stderr, "SKIP: MSL source not found at %s\n", msl_path.c_str());
        return 77;
    }
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<std::uint16_t> a(256), b(256);
    std::vector<float> c(256), out(256, -99.0f);
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            a[row * 16 + col] = float_to_half(a_value(row, col));
            b[col * 16 + row] = float_to_half(b_value(row, col));
            c[row * 16 + col] = c_value(row, col);
        }
    }

    void* da = nullptr;
    void* db = nullptr;
    void* dc = nullptr;
    if (cudaMalloc(&da, a.size() * sizeof(std::uint16_t)) != cudaSuccess ||
        cudaMalloc(&db, b.size() * sizeof(std::uint16_t)) != cudaSuccess ||
        cudaMalloc(&dc, c.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }
    if (cudaMemcpy(da, a.data(), a.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(db, b.data(), b.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dc, c.data(), c.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy H2D failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = msl_path.c_str(),
        .kernel_name = "wm",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };
    void* arg_a = da;
    void* arg_b = db;
    void* arg_c = dc;
    void* args[] = {&arg_a, &arg_b, &arg_c};
    const dim3 grid_dim(1, 1, 1);
    const dim3 block_dim(32, 1, 1);
    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel failed\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }
    if (cudaMemcpy(out.data(), dc, out.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy D2H failed\n");
        return 1;
    }

    float max_error = 0.0f;
    for (int row = 0; row < 16; ++row) {
        for (int col = 0; col < 16; ++col) {
            float expected = c_value(row, col);
            for (int k = 0; k < 16; ++k) {
                expected += a_value(row, k) * b_value(k, col);
            }
            max_error = std::fmax(max_error, std::fabs(out[row * 16 + col] - expected));
        }
    }
    std::printf("PTX wmma m16n16k16 max_err=%g\n", max_error);
    if (max_error > 0.01f) {
        std::fprintf(stderr, "FAIL: PTX wmma result mismatch\n");
        return 1;
    }
    std::printf("PASS: PTX wmma m16n16k16\n");
    return 0;
}
