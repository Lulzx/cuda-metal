#pragma once

#include "elf_fatbin_fixture.h"

#include <lz4.h>
#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cumetal::test {

enum class FatbinCompression {
    kLz4,
    kZstd,
};

inline std::vector<std::uint8_t> make_compressed_fatbin(
    const std::vector<char>& ptx,
    FatbinCompression compression) {
    constexpr std::size_t kOuterHeaderSize = 16u;
    constexpr std::size_t kEntryHeaderSize = 64u;
    constexpr std::uint64_t kLz4Flag = 0x2000u;
    constexpr std::uint64_t kZstdFlag = 0x8000u;

    std::vector<std::uint8_t> source(ptx.begin(), ptx.end());
    source.push_back(0);
    while (source.size() % 8u != 0) source.push_back(0);

    std::vector<std::uint8_t> compressed;
    std::uint64_t flags = 0;
    if (compression == FatbinCompression::kLz4) {
        const int bound = LZ4_compressBound(static_cast<int>(source.size()));
        if (bound <= 0) return {};
        compressed.resize(static_cast<std::size_t>(bound));
        const int size = LZ4_compress_default(
            reinterpret_cast<const char*>(source.data()),
            reinterpret_cast<char*>(compressed.data()),
            static_cast<int>(source.size()), bound);
        if (size <= 0) return {};
        compressed.resize(static_cast<std::size_t>(size));
        flags = kLz4Flag;
    } else {
        compressed.resize(ZSTD_compressBound(source.size()));
        const std::size_t size = ZSTD_compress(
            compressed.data(), compressed.size(), source.data(), source.size(), 1);
        if (ZSTD_isError(size) != 0) return {};
        compressed.resize(size);
        flags = kZstdFlag;
    }

    const std::size_t stored_size = std::max(source.size(), compressed.size());
    std::vector<std::uint8_t> result(
        kOuterHeaderSize + kEntryHeaderSize + stored_size, 0);
    write_value<std::uint32_t>(&result, 0, 0xBA55ED50u);
    write_value<std::uint16_t>(&result, 4, 1u);
    write_value<std::uint16_t>(&result, 6,
                               static_cast<std::uint16_t>(kOuterHeaderSize));
    write_value<std::uint64_t>(
        &result, 8,
        static_cast<std::uint64_t>(kEntryHeaderSize + stored_size));

    const std::size_t entry = kOuterHeaderSize;
    write_value<std::uint16_t>(&result, entry + 0, 1u);       // PTX
    write_value<std::uint16_t>(&result, entry + 2, 0x0101u);  // current entry
    write_value<std::uint32_t>(&result, entry + 4,
                               static_cast<std::uint32_t>(kEntryHeaderSize));
    write_value<std::uint32_t>(&result, entry + 8,
                               static_cast<std::uint32_t>(source.size()));
    write_value<std::uint32_t>(&result, entry + 16,
                               static_cast<std::uint32_t>(compressed.size()));
    write_value<std::uint32_t>(&result, entry + 20,
                               static_cast<std::uint32_t>(kEntryHeaderSize - 8u));
    write_value<std::uint32_t>(&result, entry + 28, 80u);  // sm_80 fixture
    write_value<std::uint32_t>(&result, entry + 32, 64u);
    write_value<std::uint64_t>(&result, entry + 40, flags);
    write_value<std::uint64_t>(&result, entry + 56,
                               static_cast<std::uint64_t>(source.size()));
    std::memcpy(result.data() + kOuterHeaderSize + kEntryHeaderSize,
                compressed.data(), compressed.size());
    return result;
}

}  // namespace cumetal::test
