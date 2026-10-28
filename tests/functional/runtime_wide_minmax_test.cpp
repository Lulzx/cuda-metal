#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
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

void wide_minmax_host_stub() {}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "usage: %s <wide-minmax.ptx>\n", argv[0]);
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
    char kernel_name[] = "_Z11wide_minmaxPyS_S_PKy";
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&wide_minmax_host_stub),
        kernel_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);

    constexpr int kThreads = 64;
    std::vector<unsigned long long> values(kThreads);
    unsigned long long expected_min =
        std::numeric_limits<unsigned long long>::max();
    unsigned long long expected_max = 0;
    for (int i = 0; i < kThreads; ++i) {
        values[static_cast<std::size_t>(i)] =
            static_cast<unsigned long long>((i * 37 + 11) % 61) + 4;
        if (values[static_cast<std::size_t>(i)] < expected_min)
            expected_min = values[static_cast<std::size_t>(i)];
        if (values[static_cast<std::size_t>(i)] > expected_max)
            expected_max = values[static_cast<std::size_t>(i)];
    }
    const unsigned long long old_seed = 50ull;
    const unsigned long long expected_old_final =
        old_seed < expected_min ? old_seed : expected_min;

    void* device_lo = nullptr;
    void* device_hi = nullptr;
    void* device_old = nullptr;
    void* device_values = nullptr;
    const unsigned long long lo_init =
        std::numeric_limits<unsigned long long>::max();
    const unsigned long long hi_init = 0;
    if (cudaMalloc(&device_lo, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&device_hi, sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&device_old, 2 * sizeof(unsigned long long)) != cudaSuccess ||
        cudaMalloc(&device_values, values.size() * sizeof(unsigned long long)) !=
            cudaSuccess ||
        cudaMemcpy(device_lo, &lo_init, sizeof(lo_init),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_hi, &hi_init, sizeof(hi_init),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_old, &old_seed, sizeof(old_seed),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(device_values, values.data(),
                   values.size() * sizeof(unsigned long long),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device allocation/copy failed\n");
        return 1;
    }

    void* params[] = {&device_lo, &device_hi, &device_old, &device_values};
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&wide_minmax_host_stub),
                         dim3(1, 1, 1), dim3(kThreads, 1, 1), params, 0,
                         nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: wide_minmax launch failed\n");
        return 1;
    }

    unsigned long long lo = 0;
    unsigned long long hi = 0;
    unsigned long long old_pair[2] = {0, 0};
    if (cudaMemcpy(&lo, device_lo, sizeof(lo), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(&hi, device_hi, sizeof(hi), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(old_pair, device_old, sizeof(old_pair),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: output copy failed\n");
        return 1;
    }
    if (lo != expected_min || hi != expected_max) {
        std::fprintf(stderr,
                     "FAIL: native min/max lo=%llu hi=%llu expected=%llu/%llu\n",
                     lo, hi, expected_min, expected_max);
        return 1;
    }
    // The live-result atomicMin (lock-bank path) must leave the true
    // minimum and report an old value that is the seed or one of the
    // racing values, never below the final minimum.
    if (old_pair[0] != expected_old_final || old_pair[1] < old_pair[0] ||
        (old_pair[1] != old_seed &&
         [&] {
             for (const unsigned long long value : values)
                 if (value == old_pair[1]) return false;
             return true;
         }())) {
        std::fprintf(stderr,
                     "FAIL: lock-bank old value final=%llu old=%llu expected final=%llu\n",
                     old_pair[0], old_pair[1], expected_old_final);
        return 1;
    }

    cudaFree(device_lo);
    cudaFree(device_hi);
    cudaFree(device_old);
    cudaFree(device_values);
    __cudaUnregisterFatBinary(fatbin_handle);
    std::printf("PASS: native u64 min/max and lock-bank old value exact\n");
    return 0;
}
