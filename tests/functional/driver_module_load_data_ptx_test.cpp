#include "cuda.h"
#include "compressed_fatbin_fixture.h"
#include "elf_fatbin_fixture.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

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

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

bool run_vector_add(CUmodule module) {
    CUfunction vector_add = nullptr;
    if (cuModuleGetFunction(&vector_add, module, "vector_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction failed\n");
        return false;
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_c(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 7) % 31) * 0.5f;
        host_b[i] = static_cast<float>((i * 5) % 29) * 1.25f;
    }

    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
    CUdeviceptr dev_c = 0;
    if (cuMemAlloc(&dev_a, kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_b, kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_c, kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return false;
    }

    if (cuMemcpyHtoD(dev_a, host_a.data(), kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dev_b, host_b.data(), kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyHtoD failed\n");
        return false;
    }

    CUdeviceptr arg_a = dev_a;
    CUdeviceptr arg_b = dev_b;
    CUdeviceptr arg_c = dev_c;
    void* params[] = {&arg_a, &arg_b, &arg_c, nullptr};

    const unsigned int grid_x =
        static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    CUresult launch_res = cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       nullptr,
                       params,
                       nullptr);
    if (launch_res != CUDA_SUCCESS) {
        const char* err_name = nullptr;
        const char* err_str = nullptr;
        cuGetErrorName(launch_res, &err_name);
        cuGetErrorString(launch_res, &err_str);
        std::fprintf(stderr, "FAIL: cuLaunchKernel failed: %s (%s)\n",
                     err_name ? err_name : "?", err_str ? err_str : "?");
        return false;
    }

    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize failed\n");
        return false;
    }

    if (cuMemcpyDtoH(host_c.data(), dev_c, kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH failed\n");
        return false;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i];
        if (!nearly_equal(host_c[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected));
            return false;
        }
    }

    if (cuMemFree(dev_a) != CUDA_SUCCESS || cuMemFree(dev_b) != CUDA_SUCCESS ||
        cuMemFree(dev_c) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return false;
    }

    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-ptx>\n", argv[0]);
        return 64;
    }

    const std::string ptx_path = argv[1];
    if (!std::filesystem::exists(ptx_path)) {
        std::fprintf(stderr, "SKIP: PTX not found at %s\n", ptx_path.c_str());
        return 77;
    }

    std::ifstream in(ptx_path, std::ios::binary);
    std::vector<char> ptx_file_bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    if (ptx_file_bytes.empty()) {
        std::fprintf(stderr, "FAIL: failed to read PTX bytes\n");
        return 1;
    }

    std::vector<char> ptx_bytes = ptx_file_bytes;
    ptx_bytes.push_back('\0');

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }

    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    CUmodule module = nullptr;
    if (cuModuleLoadData(&module, ptx_bytes.data()) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(PTX text) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after PTX text load failed\n");
        return 1;
    }

    std::vector<std::uint8_t> fatbin_blob(sizeof(FatbinBlobHeader) + ptx_file_bytes.size(), 0);
    FatbinBlobHeader header{};
    header.fat_size = ptx_file_bytes.size();
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
    padded_header.fat_size = ptx_file_bytes.size();
    std::vector<std::uint8_t> fatbin_blob_padded(padded_header.header_size + ptx_file_bytes.size(), 0);
    std::memcpy(fatbin_blob_padded.data(), &padded_header, sizeof(padded_header));
    std::memcpy(fatbin_blob_padded.data() + padded_header.header_size,
                ptx_file_bytes.data(),
                ptx_file_bytes.size());

    FatbinWrapper wrapper{};
    wrapper.data = fatbin_blob.data();

    if (cuModuleLoadData(&module, &wrapper) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(fatbin wrapper) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after fatbin wrapper load failed\n");
        return 1;
    }

    FatbinWrapperPrefixed wrapper_prefixed{};
    wrapper_prefixed.prefix0 = 0x1111222233334444ull;
    wrapper_prefixed.prefix1 = 0x5555666677778888ull;
    wrapper_prefixed.wrapper.data = fatbin_blob.data();
    if (cuModuleLoadData(&module, &wrapper_prefixed) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(prefixed fatbin wrapper) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after prefixed fatbin wrapper load failed\n");
        return 1;
    }

    if (cuModuleLoadData(&module, fatbin_blob.data()) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(fatbin blob) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after fatbin blob load failed\n");
        return 1;
    }

    const std::vector<std::uint8_t>* compressed_images[] = {
        &lz4_fatbin, &zstd_fatbin, &elf_lz4_fatbin, &elf_zstd_fatbin};
    const char* compressed_names[] = {
        "LZ4 fatbin", "Zstd fatbin", "ELF LZ4 fatbin", "ELF Zstd fatbin"};
    for (std::size_t i = 0; i < 4; ++i) {
        module = nullptr;
        if (cuModuleLoadData(&module, compressed_images[i]->data()) !=
                CUDA_SUCCESS ||
            module == nullptr) {
            std::fprintf(stderr, "FAIL: cuModuleLoadData(%s) failed\n",
                         compressed_names[i]);
            return 1;
        }
        if (!run_vector_add(module)) return 1;
        if (cuModuleUnload(module) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: cuModuleUnload after %s failed\n",
                         compressed_names[i]);
            return 1;
        }
        std::printf("COMPRESSED_DRIVER_OK %s\n", compressed_names[i]);
    }

    if (cuModuleLoadData(&module, elf_fatbin.data()) != CUDA_SUCCESS ||
        module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(ELF .nv_fatbin) failed\n");
        return 1;
    }
    if (!run_vector_add(module)) {
        return 1;
    }
    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after ELF .nv_fatbin load failed\n");
        return 1;
    }

    for (const auto* elf32 :
         {&elf32_fatbin, &elf32_extended_indexes, &elf32_raw_ptx}) {
        if (cuModuleLoadData(&module, elf32->data()) != CUDA_SUCCESS ||
            module == nullptr) {
            std::fprintf(stderr, "FAIL: cuModuleLoadData(ELF32) failed\n");
            return 1;
        }
        if (!run_vector_add(module)) {
            return 1;
        }
        if (cuModuleUnload(module) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: cuModuleUnload after ELF32 load failed\n");
            return 1;
        }
    }

    if (cuModuleLoadData(&module, elf_extended_indexes.data()) !=
            CUDA_SUCCESS ||
        module == nullptr) {
        std::fprintf(
            stderr,
            "FAIL: cuModuleLoadData(ELF extended indexes) failed\n");
        return 1;
    }
    if (!run_vector_add(module)) {
        return 1;
    }
    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(
            stderr,
            "FAIL: cuModuleUnload after ELF extended-index load failed\n");
        return 1;
    }

    if (cuModuleLoadData(&module, elf_raw_ptx.data()) != CUDA_SUCCESS ||
        module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(ELF raw PTX section) failed\n");
        return 1;
    }
    if (!run_vector_add(module)) {
        return 1;
    }
    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after ELF raw PTX load failed\n");
        return 1;
    }

    FatbinWrapper wrapper_padded{};
    wrapper_padded.data = fatbin_blob_padded.data();

    if (cuModuleLoadData(&module, &wrapper_padded) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(fatbin wrapper padded header) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after padded-header fatbin wrapper load failed\n");
        return 1;
    }

    FatbinWrapper wrapper_direct_ptx{};
    wrapper_direct_ptx.data = ptx_bytes.data();

    if (cuModuleLoadData(&module, &wrapper_direct_ptx) != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData(fatbin wrapper direct PTX) failed\n");
        return 1;
    }

    if (!run_vector_add(module)) {
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload after fatbin wrapper direct PTX load failed\n");
        return 1;
    }

    FatbinWrapper invalid_wrapper{};
    invalid_wrapper.magic = 0x12345678u;
    invalid_wrapper.data = fatbin_blob.data();
    const CUresult invalid_wrapper_status = cuModuleLoadData(&module, &invalid_wrapper);
    if (invalid_wrapper_status != CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr,
                     "FAIL: invalid fatbin wrapper magic expected CUDA_ERROR_INVALID_IMAGE got %d\n",
                     static_cast<int>(invalid_wrapper_status));
        return 1;
    }

    FatbinBlobHeader invalid_zero_size{};
    invalid_zero_size.fat_size = 0;
    std::vector<std::uint8_t> invalid_zero_blob(sizeof(FatbinBlobHeader), 0);
    std::memcpy(invalid_zero_blob.data(), &invalid_zero_size, sizeof(invalid_zero_size));
    if (cuModuleLoadData(&module, invalid_zero_blob.data()) != CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr, "FAIL: zero-size fatbin blob should return CUDA_ERROR_INVALID_IMAGE\n");
        return 1;
    }

    FatbinBlobHeader invalid_huge_size{};
    invalid_huge_size.fat_size = ~static_cast<std::uint64_t>(0);
    std::vector<std::uint8_t> invalid_huge_blob(sizeof(FatbinBlobHeader), 0);
    std::memcpy(invalid_huge_blob.data(), &invalid_huge_size, sizeof(invalid_huge_size));
    if (cuModuleLoadData(&module, invalid_huge_blob.data()) != CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr, "FAIL: huge-size fatbin blob should return CUDA_ERROR_INVALID_IMAGE\n");
        return 1;
    }

    std::vector<std::uint8_t> malformed_compressed_size = lz4_fatbin;
    cumetal::test::write_value<std::uint32_t>(
        &malformed_compressed_size, 16u + 16u, 1u);
    std::vector<std::uint8_t> oversized_decompression = lz4_fatbin;
    cumetal::test::write_value<std::uint64_t>(
        &oversized_decompression, 16u + 56u,
        64ull * 1024ull * 1024ull + 1ull);
    std::vector<std::uint8_t> ambiguous_compression = lz4_fatbin;
    cumetal::test::write_value<std::uint64_t>(
        &ambiguous_compression, 16u + 40u, 0x2000ull | 0x8000ull);
    std::vector<std::uint8_t> truncated_compressed_entry = lz4_fatbin;
    cumetal::test::write_value<std::uint32_t>(
        &truncated_compressed_entry, 16u + 8u,
        static_cast<std::uint32_t>(truncated_compressed_entry.size()));
    std::vector<std::uint8_t> unsupported_entry_version = lz4_fatbin;
    cumetal::test::write_value<std::uint16_t>(
        &unsupported_entry_version, 16u + 2u, 0x0102u);
    std::vector<std::uint8_t> unsupported_entry_kind = lz4_fatbin;
    cumetal::test::write_value<std::uint16_t>(
        &unsupported_entry_kind, 16u, 3u);
    const std::vector<std::uint8_t>* invalid_compressed[] = {
        &malformed_compressed_size,
        &oversized_decompression,
        &ambiguous_compression,
        &truncated_compressed_entry,
        &unsupported_entry_version,
        &unsupported_entry_kind,
    };
    for (const auto* invalid : invalid_compressed) {
        module = nullptr;
        if (cuModuleLoadData(&module, invalid->data()) !=
            CUDA_ERROR_INVALID_IMAGE) {
            std::fprintf(stderr,
                         "FAIL: malformed compressed fatbin was accepted\n");
            return 1;
        }
    }

    std::vector<std::uint8_t> malformed_elf = elf_fatbin;
    cumetal::test::write_value<std::uint64_t>(
        &malformed_elf, 40, 64ull * 1024ull * 1024ull - 32ull);
    module = nullptr;
    if (cuModuleLoadData(&module, malformed_elf.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr, "FAIL: out-of-range ELF section table should be rejected\n");
        return 1;
    }

    std::vector<std::uint8_t> unsupported_elf = elf_fatbin;
    unsupported_elf[5] = 2;  // ELFDATA2MSB
    module = nullptr;
    if (cuModuleLoadData(&module, unsupported_elf.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr, "FAIL: unsupported ELF byte order should be rejected\n");
        return 1;
    }

    std::vector<std::uint8_t> malformed_nested_fatbin = elf_fatbin;
    cumetal::test::write_value<std::uint64_t>(
        &malformed_nested_fatbin, 64 + 8, ~std::uint64_t{0});
    module = nullptr;
    if (cuModuleLoadData(&module, malformed_nested_fatbin.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(stderr, "FAIL: out-of-range nested fatbin should be rejected\n");
        return 1;
    }

    const std::size_t extended_section_table_offset =
        elf_extended_indexes.size() - 3 * 64;
    std::vector<std::uint8_t> malformed_extended_count =
        elf_extended_indexes;
    cumetal::test::write_value<std::uint64_t>(
        &malformed_extended_count,
        extended_section_table_offset + 32,
        64ull * 1024ull * 1024ull);
    module = nullptr;
    if (cuModuleLoadData(&module, malformed_extended_count.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(
            stderr,
            "FAIL: out-of-range extended section count should be rejected\n");
        return 1;
    }

    std::vector<std::uint8_t> malformed_extended_string_index =
        elf_extended_indexes;
    cumetal::test::write_value<std::uint32_t>(
        &malformed_extended_string_index,
        extended_section_table_offset + 40,
        3);
    module = nullptr;
    if (cuModuleLoadData(&module,
                         malformed_extended_string_index.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(
            stderr,
            "FAIL: out-of-range extended string-table index should be rejected\n");
        return 1;
    }

    const std::size_t elf32_extended_section_table_offset =
        elf32_extended_indexes.size() - 3 * 40;
    std::vector<std::uint8_t> malformed_elf32_extended_count =
        elf32_extended_indexes;
    cumetal::test::write_value<std::uint32_t>(
        &malformed_elf32_extended_count,
        elf32_extended_section_table_offset + 20,
        64u * 1024u * 1024u);
    module = nullptr;
    if (cuModuleLoadData(&module,
                         malformed_elf32_extended_count.data()) !=
        CUDA_ERROR_INVALID_IMAGE) {
        std::fprintf(
            stderr,
            "FAIL: out-of-range ELF32 extended section count should be rejected\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuModuleLoadData supports PTX text, bounded LZ4/Zstd fatbins, and ELF32/ELF64 variants\n");
    return 0;
}
