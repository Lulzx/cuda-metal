#include "cuda_runtime.h"
#include "compressed_fatbin_fixture.h"
#include "elf_fatbin_fixture.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void** __cudaRegisterFatBinary3(const void* fat_cubin, ...);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);

// A failed launch is retained as a pending error and consumed by the next device
// synchronization, whether or not the caller read the launch's own return value:
// generated <<<...>>> host stubs discard it, so that is the only place such a
// failure would otherwise become visible. Every deliberately failed launch below
// therefore asserts that propagation and consumes it, so that a later
// synchronization cannot report it and be mistaken for a different failure.
bool consume_expected_launch_failure(const char* what) {
    if (cudaDeviceSynchronize() != cudaErrorInvalidValue) {
        std::fprintf(stderr,
                     "FAIL: cudaDeviceSynchronize should report the failed launch (%s)\n",
                     what);
        return false;
    }
    return true;
}

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

constexpr std::size_t kElementCount = 1u << 13;
constexpr std::size_t kThreadsPerBlock = 256;
constexpr std::uint32_t kFatbinWrapperMagic = 0x466243b1u;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;

struct FatbinWrapper {
    std::uint32_t magic = kFatbinWrapperMagic;
    std::uint32_t version = 1;
    const void* data = nullptr;
    const void* unknown = nullptr;
};

struct FatbinWrapperPrefixed {
    std::uint64_t prefix0 = 0;
    std::uint64_t prefix1 = 0;
    FatbinWrapper wrapper{};
};

struct FatbinBlobHeader {
    std::uint32_t magic = kFatbinBlobMagic;
    std::uint16_t version = 1;
    std::uint16_t header_size = 16;
    std::uint64_t fat_size = 0;
};

void vector_add_host_stub() {}

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-ptx>\n", argv[0]);
        return 64;
    }

    const std::string ptx_path = argv[1];
    const bool concurrent_first_use =
        argc > 2 && std::string(argv[2]) == "--concurrent-first-use";
    const std::string fallback_metallib =
        argc > 2 && !concurrent_first_use ? argv[2] : "";
    if (!std::filesystem::exists(ptx_path)) {
        std::fprintf(stderr, "SKIP: PTX not found at %s\n", ptx_path.c_str());
        return 77;
    }

    std::ifstream ptx_in(ptx_path, std::ios::binary);
    std::vector<char> ptx_file_bytes((std::istreambuf_iterator<char>(ptx_in)),
                                     std::istreambuf_iterator<char>());
    if (ptx_file_bytes.empty()) {
        std::fprintf(stderr, "FAIL: failed to read PTX bytes\n");
        return 1;
    }

    std::vector<char> ptx_bytes = ptx_file_bytes;
    ptx_bytes.push_back('\0');

    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx_file_bytes.size(), 0);
    FatbinBlobHeader header{};
    header.fat_size = static_cast<std::uint64_t>(ptx_file_bytes.size());
    std::memcpy(fatbin_blob.data(), &header, sizeof(header));
    std::memcpy(fatbin_blob.data() + sizeof(header), ptx_file_bytes.data(), ptx_file_bytes.size());
    const std::vector<std::uint8_t> elf_fatbin =
        cumetal::test::make_elf64_image(".nv_fatbin", fatbin_blob);
    const std::vector<std::uint8_t> elf32_fatbin =
        cumetal::test::make_elf32_image(".nv_fatbin", fatbin_blob);
    const std::vector<std::uint8_t> elf32_extended_indexes =
        cumetal::test::make_elf32_image(".nv_fatbin", fatbin_blob, true);
    const std::vector<std::uint8_t> elf_extended_indexes =
        cumetal::test::make_elf64_image(".nv_fatbin", fatbin_blob, true);
    const std::vector<std::uint8_t> raw_ptx_payload(
        ptx_file_bytes.begin(), ptx_file_bytes.end());
    const std::vector<std::uint8_t> elf_raw_ptx =
        cumetal::test::make_elf64_image(".ptx", raw_ptx_payload);
    const std::vector<std::uint8_t> elf32_raw_ptx =
        cumetal::test::make_elf32_image(".ptx", raw_ptx_payload);
    const std::vector<std::uint8_t> lz4_fatbin =
        cumetal::test::make_compressed_fatbin(
            ptx_file_bytes, cumetal::test::FatbinCompression::kLz4);
    const std::vector<std::uint8_t> zstd_fatbin =
        cumetal::test::make_compressed_fatbin(
            ptx_file_bytes, cumetal::test::FatbinCompression::kZstd);
    if (lz4_fatbin.empty() || zstd_fatbin.empty()) {
        std::fprintf(stderr, "FAIL: compressed fatbin fixture creation failed\n");
        return 1;
    }
    const std::vector<std::uint8_t> elf_lz4_fatbin =
        cumetal::test::make_elf64_image(".nv_fatbin", lz4_fatbin);
    const std::vector<std::uint8_t> elf_zstd_fatbin =
        cumetal::test::make_elf64_image(".nv_fatbin", zstd_fatbin);

    FatbinBlobHeader padded_header{};
    padded_header.header_size = 64;
    padded_header.fat_size = static_cast<std::uint64_t>(ptx_file_bytes.size());
    std::vector<std::uint8_t> fatbin_blob_padded(padded_header.header_size + ptx_file_bytes.size(), 0);
    std::memcpy(fatbin_blob_padded.data(), &padded_header, sizeof(padded_header));
    std::memcpy(fatbin_blob_padded.data() + padded_header.header_size,
                ptx_file_bytes.data(),
                ptx_file_bytes.size());

    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    void** fatbin_handle = __cudaRegisterFatBinary(&wrapper);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary returned null\n");
        return 1;
    }

    char device_function[] = "vector_add";
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 11) % 37) * 0.25f;
        host_b[i] = static_cast<float>((i * 7) % 29) * 1.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_c = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_c, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    void* arg_a = dev_a;
    void* arg_b = dev_b;
    void* arg_c = dev_c;
    void* args[] = {&arg_a, &arg_b, &arg_c, nullptr};

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock),
                        1,
                        1);

    if (concurrent_first_use) {
        constexpr int kLaunchThreads = 8;
        std::atomic<int> ready{0};
        std::atomic<bool> start{false};
        std::atomic<int> launch_failures{0};
        std::vector<std::thread> threads;
        threads.reserve(kLaunchThreads);
        for (int i = 0; i < kLaunchThreads; ++i) {
            threads.emplace_back([&]() {
                ready.fetch_add(1, std::memory_order_release);
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }
                if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                                     grid_dim,
                                     block_dim,
                                     args,
                                     0,
                                     nullptr) != cudaSuccess) {
                    launch_failures.fetch_add(1, std::memory_order_relaxed);
                }
            });
        }
        while (ready.load(std::memory_order_acquire) != kLaunchThreads) {
            std::this_thread::yield();
        }
        start.store(true, std::memory_order_release);
        for (auto& thread : threads) thread.join();
        if (launch_failures.load(std::memory_order_relaxed) != 0) {
            std::fprintf(stderr, "FAIL: concurrent first-use registration launch failed\n");
            return 1;
        }
    } else if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                                grid_dim,
                                block_dim,
                                args,
                                0,
                                nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through fatbin PTX registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: launch should fail after __cudaUnregisterFatBinary\n");
        return 1;
    }

    if (!consume_expected_launch_failure("after __cudaUnregisterFatBinary")) {
        return 1;
    }

    const auto run_compressed_registration =
        [&](const std::vector<std::uint8_t>& image, const char* name) -> bool {
        FatbinWrapper compressed_wrapper{};
        compressed_wrapper.data = image.data();
        void** compressed_handle =
            __cudaRegisterFatBinary(&compressed_wrapper);
        if (compressed_handle == nullptr) {
            std::fprintf(stderr, "FAIL: %s registration returned null\n", name);
            return false;
        }
        __cudaRegisterFunction(
            compressed_handle,
            reinterpret_cast<const void*>(&vector_add_host_stub),
            device_function, nullptr, 0, nullptr, nullptr, nullptr, nullptr,
            nullptr);
        if (cudaMemset(dev_c, 0, bytes) != cudaSuccess ||
            cudaLaunchKernel(
                reinterpret_cast<const void*>(&vector_add_host_stub), grid_dim,
                block_dim, args, 0, nullptr) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(host_c.data(), dev_c, bytes,
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: %s registration launch failed\n", name);
            __cudaUnregisterFatBinary(compressed_handle);
            return false;
        }
        for (std::size_t i = 0; i < kElementCount; ++i) {
            const float expected = host_a[i] + host_b[i];
            if (!nearly_equal(host_c[i], expected)) {
                std::fprintf(stderr,
                             "FAIL: %s mismatch at %zu (got=%f expected=%f)\n",
                             name, i, static_cast<double>(host_c[i]),
                             static_cast<double>(expected));
                __cudaUnregisterFatBinary(compressed_handle);
                return false;
            }
        }
        __cudaUnregisterFatBinary(compressed_handle);
        std::printf("COMPRESSED_REGISTRATION_OK %s\n", name);
        return true;
    };

    if (!run_compressed_registration(lz4_fatbin, "LZ4 fatbin") ||
        !run_compressed_registration(zstd_fatbin, "Zstd fatbin") ||
        !run_compressed_registration(elf_lz4_fatbin, "ELF LZ4 fatbin") ||
        !run_compressed_registration(elf_zstd_fatbin, "ELF Zstd fatbin")) {
        return 1;
    }

    std::vector<std::uint8_t> malformed_compressed = lz4_fatbin;
    cumetal::test::write_value<std::uint32_t>(
        &malformed_compressed, 16u + 16u, 1u);
    FatbinWrapper malformed_compressed_wrapper{};
    malformed_compressed_wrapper.data = malformed_compressed.data();
    void** malformed_compressed_handle =
        __cudaRegisterFatBinary(&malformed_compressed_wrapper);
    if (malformed_compressed_handle == nullptr) {
        std::fprintf(stderr,
                     "FAIL: malformed compressed registration returned null handle\n");
        return 1;
    }
    __cudaRegisterFunction(
        malformed_compressed_handle,
        reinterpret_cast<const void*>(&vector_add_host_stub), device_function,
        nullptr, 0, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (cudaLaunchKernel(
            reinterpret_cast<const void*>(&vector_add_host_stub), grid_dim,
            block_dim, args, 0, nullptr) != cudaErrorInvalidValue ||
        !consume_expected_launch_failure("malformed compressed fatbin")) {
        std::fprintf(stderr,
                     "FAIL: malformed compressed registration was launchable\n");
        __cudaUnregisterFatBinary(malformed_compressed_handle);
        return 1;
    }
    __cudaUnregisterFatBinary(malformed_compressed_handle);

    if (!fallback_metallib.empty()) {
        std::vector<std::uint8_t> unsupported_entry_version = lz4_fatbin;
        cumetal::test::write_value<std::uint16_t>(
            &unsupported_entry_version, 16u + 2u, 0x0102u);
        if (setenv("CUMETAL_FATBIN_METALLIB",
                   fallback_metallib.c_str(), 1) != 0) {
            std::fprintf(stderr,
                         "FAIL: could not configure environment fallback test\n");
            return 1;
        }
        FatbinWrapper unsupported_wrapper{};
        unsupported_wrapper.data = unsupported_entry_version.data();
        void** unsupported_handle =
            __cudaRegisterFatBinary(&unsupported_wrapper);
        __cudaRegisterFunction(
            unsupported_handle,
            reinterpret_cast<const void*>(&vector_add_host_stub),
            device_function, nullptr, 0, nullptr, nullptr, nullptr, nullptr,
            nullptr);
        const cudaError_t unsupported_launch = cudaLaunchKernel(
            reinterpret_cast<const void*>(&vector_add_host_stub), grid_dim,
            block_dim, args, 0, nullptr);
        const cudaError_t unsupported_sync = cudaDeviceSynchronize();
        unsetenv("CUMETAL_FATBIN_METALLIB");
        if (unsupported_launch != cudaErrorInvalidValue ||
            unsupported_sync != cudaErrorInvalidValue) {
            std::fprintf(stderr,
                         "FAIL: unsupported fatbin version used the environment metallib fallback "
                         "(launch=%d sync=%d)\n",
                         static_cast<int>(unsupported_launch),
                         static_cast<int>(unsupported_sync));
            __cudaUnregisterFatBinary(unsupported_handle);
            return 1;
        }
        __cudaUnregisterFatBinary(unsupported_handle);
        std::printf("UNSUPPORTED_FATBIN_FALLBACK_REFUSED\n");
    }

    FatbinWrapperPrefixed wrapper_prefixed{};
    wrapper_prefixed.prefix0 = 0x1111222233334444ull;
    wrapper_prefixed.prefix1 = 0x5555666677778888ull;
    wrapper_prefixed.wrapper.data = fatbin_blob.data();
    fatbin_handle = __cudaRegisterFatBinary(&wrapper_prefixed);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (prefixed wrapper) returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through prefixed fatbin wrapper failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after prefixed fatbin wrapper launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr,
                     "FAIL: cudaMemcpy device->host after prefixed fatbin wrapper launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: prefixed fatbin wrapper mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary(fatbin_blob.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (direct fatbin blob) returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through direct fatbin blob registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after direct fatbin blob launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after direct fatbin blob launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: direct fatbin blob mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary(fatbin_blob_padded.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (padded fatbin blob) returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through padded fatbin blob registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after padded fatbin blob launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after padded fatbin blob launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: padded fatbin blob mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary(elf_fatbin.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (ELF .nv_fatbin) returned null\n");
        return 1;
    }
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch through ELF .nv_fatbin registration failed\n");
        return 1;
    }
    __cudaUnregisterFatBinary(fatbin_handle);

    for (const auto* elf32 :
         {&elf32_fatbin, &elf32_extended_indexes, &elf32_raw_ptx}) {
        fatbin_handle = __cudaRegisterFatBinary(elf32->data());
        if (fatbin_handle == nullptr) {
            std::fprintf(stderr,
                         "FAIL: __cudaRegisterFatBinary (ELF32) returned null\n");
            return 1;
        }
        __cudaRegisterFunction(fatbin_handle,
                               reinterpret_cast<const void*>(&vector_add_host_stub),
                               device_function,
                               nullptr,
                               0,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr);
        if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                             grid_dim,
                             block_dim,
                             args,
                             0,
                             nullptr) != cudaSuccess ||
            cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "FAIL: launch through ELF32 registration failed\n");
            return 1;
        }
        __cudaUnregisterFatBinary(fatbin_handle);
    }

    fatbin_handle = __cudaRegisterFatBinary(elf_raw_ptx.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary (ELF raw PTX) returned null\n");
        return 1;
    }
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: launch through ELF raw PTX registration failed\n");
        return 1;
    }
    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary(elf_extended_indexes.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(
            stderr,
            "FAIL: __cudaRegisterFatBinary (ELF extended indexes) returned null\n");
        return 1;
    }
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(
            stderr,
            "FAIL: launch through ELF extended-index registration failed\n");
        return 1;
    }
    __cudaUnregisterFatBinary(fatbin_handle);

    std::vector<std::uint8_t> malformed_elf = elf_fatbin;
    cumetal::test::write_value<std::uint16_t>(&malformed_elf, 62, 7);
    fatbin_handle = __cudaRegisterFatBinary(malformed_elf.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: malformed ELF registration did not return a handle\n");
        return 1;
    }
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr,
                     "FAIL: malformed ELF registration should not produce a launchable kernel\n");
        return 1;
    }
    if (!consume_expected_launch_failure("malformed ELF registration")) {
        return 1;
    }
    __cudaUnregisterFatBinary(fatbin_handle);

    std::vector<std::uint8_t> malformed_extended_count =
        elf_extended_indexes;
    const std::size_t extended_section_table_offset =
        malformed_extended_count.size() - 3 * 64;
    cumetal::test::write_value<std::uint64_t>(
        &malformed_extended_count,
        extended_section_table_offset + 32,
        64ull * 1024ull * 1024ull);
    fatbin_handle =
        __cudaRegisterFatBinary(malformed_extended_count.data());
    if (fatbin_handle == nullptr) {
        std::fprintf(
            stderr,
            "FAIL: malformed extended-count ELF registration returned null\n");
        return 1;
    }
    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaErrorInvalidValue) {
        std::fprintf(
            stderr,
            "FAIL: out-of-range extended section count should not launch\n");
        return 1;
    }
    if (!consume_expected_launch_failure("out-of-range extended section count")) {
        return 1;
    }
    __cudaUnregisterFatBinary(fatbin_handle);

    fatbin_handle = __cudaRegisterFatBinary3(&wrapper, 0, nullptr, nullptr);
    if (fatbin_handle == nullptr) {
        std::fprintf(stderr, "FAIL: __cudaRegisterFatBinary3 returned null\n");
        return 1;
    }

    __cudaRegisterFunction(fatbin_handle,
                           reinterpret_cast<const void*>(&vector_add_host_stub),
                           device_function,
                           nullptr,
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    if (cudaLaunchKernel(reinterpret_cast<const void*>(&vector_add_host_stub),
                         grid_dim,
                         block_dim,
                         args,
                         0,
                         nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaLaunchKernel through fatbinary3 registration failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize after fatbinary3 launch failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host after fatbinary3 launch failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: fatbinary3 mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    __cudaUnregisterFatBinary(fatbin_handle);

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf(
        "PASS: runtime registration supports bounded LZ4/Zstd fatbins, ELF32/ELF64 extended indexes, and FatBinary3 PTX paths\n");
    return 0;
}
