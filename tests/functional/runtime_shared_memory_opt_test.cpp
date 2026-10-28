#include "cuda_runtime.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
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

void quad_copy_host_stub() {}
void transpose32_host_stub() {}

bool run_kernel(const void* kernel, dim3 grid, dim3 block, void** params) {
    return cudaLaunchKernel(kernel, grid, block, params, 0, nullptr) ==
               cudaSuccess &&
           cudaDeviceSynchronize() == cudaSuccess;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "usage: %s <shared-memory-opt.ptx>\n", argv[0]);
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
    char quad_name[] = "_Z9quad_copyPKfPf";
    char transpose_name[] = "_Z11transpose32PKfPf";
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&quad_copy_host_stub),
        quad_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&transpose32_host_stub),
        transpose_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr,
        nullptr);

    // quad_copy: out[t] = in[4t] + in[4t+1] + in[4t+2] + in[4t+3].
    {
        std::vector<float> input(64);
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = 0.25f * static_cast<float>(i) - 3.0f;
        void* device_input = nullptr;
        void* device_output = nullptr;
        if (cudaMalloc(&device_input, input.size() * sizeof(float)) !=
                cudaSuccess ||
            cudaMalloc(&device_output, 16 * sizeof(float)) != cudaSuccess ||
            cudaMemcpy(device_input, input.data(),
                       input.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: quad_copy setup failed\n");
            return 1;
        }
        void* params[] = {&device_input, &device_output};
        const bool launched =
            run_kernel(reinterpret_cast<const void*>(&quad_copy_host_stub),
                       dim3(1, 1, 1), dim3(16, 1, 1), params);
        std::vector<float> output(16, 0.0f);
        const bool copied =
            launched &&
            cudaMemcpy(output.data(), device_output,
                       output.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(device_input);
        cudaFree(device_output);
        if (!copied) {
            std::fprintf(stderr, "FAIL: quad_copy launch/copy failed\n");
            return 1;
        }
        for (std::size_t t = 0; t < output.size(); ++t) {
            const float expected = input[4 * t] + input[4 * t + 1] +
                                   input[4 * t + 2] + input[4 * t + 3];
            if (std::fabs(output[t] - expected) > 1e-5f) {
                std::fprintf(stderr, "FAIL: quad_copy out[%zu]=%g expected=%g\n",
                             t, output[t], expected);
                return 1;
            }
        }
        std::printf("PASS: quad_copy shared float4 staging exact sums\n");
    }

    // transpose32: out[tx*32+ty] = in[ty*32+tx].
    {
        std::vector<float> input(32 * 32);
        for (std::size_t i = 0; i < input.size(); ++i)
            input[i] = static_cast<float>(static_cast<int>(i % 97) - 48) * 0.5f;
        void* device_input = nullptr;
        void* device_output = nullptr;
        if (cudaMalloc(&device_input, input.size() * sizeof(float)) !=
                cudaSuccess ||
            cudaMalloc(&device_output, input.size() * sizeof(float)) !=
                cudaSuccess ||
            cudaMemcpy(device_input, input.data(),
                       input.size() * sizeof(float),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: transpose32 setup failed\n");
            return 1;
        }
        void* params[] = {&device_input, &device_output};
        const bool launched =
            run_kernel(reinterpret_cast<const void*>(&transpose32_host_stub),
                       dim3(1, 1, 1), dim3(32, 32, 1), params);
        std::vector<float> output(input.size(), 0.0f);
        const bool copied =
            launched &&
            cudaMemcpy(output.data(), device_output,
                       output.size() * sizeof(float),
                       cudaMemcpyDeviceToHost) == cudaSuccess;
        cudaFree(device_input);
        cudaFree(device_output);
        if (!copied) {
            std::fprintf(stderr, "FAIL: transpose32 launch/copy failed\n");
            return 1;
        }
        for (int ty = 0; ty < 32; ++ty) {
            for (int tx = 0; tx < 32; ++tx) {
                const float expected =
                    input[static_cast<std::size_t>(tx * 32 + ty)];
                const float actual =
                    output[static_cast<std::size_t>(ty * 32 + tx)];
                if (actual != expected) {
                    std::fprintf(stderr,
                                 "FAIL: transpose32 out[%d][%d]=%g expected=%g\n",
                                 tx, ty, actual, expected);
                    return 1;
                }
            }
        }
        std::printf("PASS: transpose32 padded shared tile exact transpose\n");
    }

    __cudaUnregisterFatBinary(fatbin_handle);
    return 0;
}
