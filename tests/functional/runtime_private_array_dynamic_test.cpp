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

void dyn_hist_host_stub() {}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || !std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "usage: %s <dyn-hist.ptx>\n", argv[0]);
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
    char kernel_name[] = "_Z8dyn_histPKfPfii";
    __cudaRegisterFunction(
        fatbin_handle, reinterpret_cast<const void*>(&dyn_hist_host_stub),
        kernel_name, nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);

    // Histogram over a dynamically indexed private array: acc[b] += in[i]
    // with b derived from the data, so the typed backend must scalarize
    // the private array into select chains (or keep it indexed) and still
    // produce exact sums.
    const std::vector<float> host_input = {0.5f, 1.5f, 2.5f, 3.5f,
                                           0.25f, 1.25f, -0.5f, -1.5f};
    constexpr int kBins = 4;
    std::vector<float> expected(kBins, 0.0f);
    for (const float value : host_input) {
        int bin = static_cast<int>(value) % kBins;
        if (bin < 0) bin += kBins;
        expected[static_cast<std::size_t>(bin)] += value;
    }

    void* device_input = nullptr;
    void* device_output = nullptr;
    if (cudaMalloc(&device_input, host_input.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&device_output, expected.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(device_input, host_input.data(),
                   host_input.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device allocation/copy failed\n");
        return 1;
    }

    int n = static_cast<int>(host_input.size());
    int bins = kBins;
    void* params[] = {&device_input, &device_output, &n, &bins};
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&dyn_hist_host_stub),
                         dim3(1, 1, 1), dim3(1, 1, 1), params, 0,
                         nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: dyn_hist launch failed\n");
        return 1;
    }

    std::vector<float> output(expected.size(), 0.0f);
    if (cudaMemcpy(output.data(), device_output,
                   output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: output copy failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (std::fabs(output[i] - expected[i]) > 1e-6f) {
            std::fprintf(stderr, "FAIL: bin[%zu]=%g expected=%g\n", i,
                         output[i], expected[i]);
            return 1;
        }
    }

    cudaFree(device_input);
    cudaFree(device_output);
    __cudaUnregisterFatBinary(fatbin_handle);
    std::printf("PASS: dynamic private-array histogram exact bins\n");
    return 0;
}
