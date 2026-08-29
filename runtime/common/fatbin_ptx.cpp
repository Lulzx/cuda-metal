#include "fatbin_ptx.h"

#include <lz4.h>
#include <zstd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace cumetal::fatbin {
namespace {

constexpr std::uint32_t kFatbinMagic = 0xBA55ED50u;
constexpr std::uint16_t kFatbinVersion = 1u;
constexpr std::size_t kOuterHeaderSize = 16u;
constexpr std::uint16_t kPtxEntryKind = 1u;
constexpr std::uint16_t kEntryVersion = 0x0101u;
constexpr std::size_t kMinimumEntryHeaderSize = 64u;
constexpr std::uint64_t kFlagCompressedLz4 = 0x2000u;
constexpr std::uint64_t kFlagCompressedZstd = 0x8000u;
constexpr std::size_t kMaximumModuleBytes = 64u * 1024u * 1024u;

template <typename T>
T read_value(const std::uint8_t* bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

bool checked_range(std::size_t offset, std::size_t size, std::size_t limit) {
    return offset <= limit && size <= limit - offset;
}

bool all_zero(const std::uint8_t* bytes, std::size_t size) {
    return std::all_of(bytes, bytes + size,
                       [](std::uint8_t value) { return value == 0; });
}

PtxExtractStatus decompress_entry(const std::uint8_t* payload,
                                  std::size_t compressed_size,
                                  std::size_t uncompressed_size,
                                  std::uint64_t flags,
                                  std::string* out_ptx) {
    if (payload == nullptr || out_ptx == nullptr || compressed_size == 0 ||
        uncompressed_size == 0 || uncompressed_size > kMaximumModuleBytes ||
        compressed_size > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        uncompressed_size > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return PtxExtractStatus::kMalformed;
    }

    const bool lz4 = (flags & kFlagCompressedLz4) != 0;
    const bool zstd = (flags & kFlagCompressedZstd) != 0;
    if (lz4 == zstd) {
        return PtxExtractStatus::kUnsupported;
    }

    std::vector<std::uint8_t> decompressed(uncompressed_size);
    std::size_t actual_size = 0;
    if (lz4) {
        const int result = LZ4_decompress_safe(
            reinterpret_cast<const char*>(payload),
            reinterpret_cast<char*>(decompressed.data()),
            static_cast<int>(compressed_size),
            static_cast<int>(uncompressed_size));
        if (result < 0) return PtxExtractStatus::kMalformed;
        actual_size = static_cast<std::size_t>(result);
    } else {
        const std::size_t result = ZSTD_decompress(
            decompressed.data(), decompressed.size(), payload, compressed_size);
        if (ZSTD_isError(result) != 0) return PtxExtractStatus::kMalformed;
        actual_size = result;
    }
    if (actual_size != uncompressed_size) {
        return PtxExtractStatus::kMalformed;
    }
    return extract_ptx_bytes(decompressed.data(), decompressed.size(), out_ptx)
               ? PtxExtractStatus::kFound
               : PtxExtractStatus::kMalformed;
}

}  // namespace

bool extract_ptx_bytes(const std::uint8_t* bytes,
                       std::size_t size,
                       std::string* out_ptx) {
    constexpr std::string_view kVersion = ".version";
    constexpr std::string_view kEntry = ".entry";
    if (bytes == nullptr || out_ptx == nullptr || size < kVersion.size() ||
        size > kMaximumModuleBytes) {
        return false;
    }

    for (std::size_t offset = 0; offset + kVersion.size() <= size; ++offset) {
        if (std::memcmp(bytes + offset, kVersion.data(), kVersion.size()) != 0) {
            continue;
        }
        const char* begin = reinterpret_cast<const char*>(bytes + offset);
        std::size_t length = size - offset;
        if (const void* terminator = std::memchr(begin, '\0', length);
            terminator != nullptr) {
            length = static_cast<const char*>(terminator) - begin;
        }
        // Normalize framed and legacy raw payloads to the same program bytes:
        // compressed frames are commonly NUL/padding terminated, while raw
        // frames commonly end in a newline.  Keeping that packaging-only
        // difference would create distinct JIT-cache keys for identical PTX.
        for (std::size_t i = length; i > 0; --i) {
            if (begin[i - 1] == '}') {
                length = i;
                break;
            }
        }
        if (length == 0) continue;
        const std::string candidate(begin, length);
        if (candidate.find(kEntry) == std::string::npos) continue;
        *out_ptx = candidate;
        return true;
    }
    return false;
}

PtxExtractStatus extract_fatbin_ptx(const void* image,
                                    std::size_t max_image_bytes,
                                    std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr ||
        max_image_bytes < kOuterHeaderSize) {
        return PtxExtractStatus::kNotFatbin;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (read_value<std::uint32_t>(bytes, 0) != kFatbinMagic) {
        return PtxExtractStatus::kNotFatbin;
    }

    const std::uint16_t version = read_value<std::uint16_t>(bytes, 4);
    const std::size_t header_size = read_value<std::uint16_t>(bytes, 6);
    const std::uint64_t encoded_size = read_value<std::uint64_t>(bytes, 8);
    if (version != kFatbinVersion || header_size < kOuterHeaderSize ||
        header_size > max_image_bytes || encoded_size == 0 ||
        encoded_size > kMaximumModuleBytes ||
        encoded_size > max_image_bytes - header_size) {
        return PtxExtractStatus::kMalformed;
    }

    const std::size_t data_size = static_cast<std::size_t>(encoded_size);
    const std::uint8_t* data = bytes + header_size;

    // Early toolchains and CuMetal's original fixtures place raw PTX directly
    // inside the bounded outer frame. Preserve that accepted compatibility
    // form while preferring the structured entry parser below.
    // A known file kind identifies the framed format independently of its
    // version.  In particular, do not reinterpret an unsupported framed
    // version as the legacy raw-PTX form and then scan through its payload.
    const std::uint16_t first_kind =
        data_size >= sizeof(std::uint16_t)
            ? read_value<std::uint16_t>(data, 0)
            : 0;
    const bool structured = data_size >= kMinimumEntryHeaderSize &&
        (first_kind == kPtxEntryKind || first_kind == 2u);
    if (!structured) {
        return extract_ptx_bytes(data, data_size, out_ptx)
                   ? PtxExtractStatus::kFound
                   : PtxExtractStatus::kNoPtx;
    }

    std::size_t cursor = 0;
    while (cursor < data_size) {
        if (data_size - cursor < kMinimumEntryHeaderSize) {
            return all_zero(data + cursor, data_size - cursor)
                       ? PtxExtractStatus::kNoPtx
                       : PtxExtractStatus::kMalformed;
        }
        const std::uint8_t* entry = data + cursor;
        const std::uint16_t kind = read_value<std::uint16_t>(entry, 0);
        const std::uint16_t entry_version = read_value<std::uint16_t>(entry, 2);
        const std::size_t entry_header_size = read_value<std::uint32_t>(entry, 4);
        const std::size_t payload_size = read_value<std::uint32_t>(entry, 8);
        const std::size_t compressed_size = read_value<std::uint32_t>(entry, 16);
        const std::uint64_t flags = read_value<std::uint64_t>(entry, 40);
        const std::uint64_t encoded_uncompressed = read_value<std::uint64_t>(entry, 56);

        if (entry_version != kEntryVersion) {
            return PtxExtractStatus::kUnsupported;
        }
        if (entry_header_size < kMinimumEntryHeaderSize ||
            entry_header_size > data_size - cursor) {
            return PtxExtractStatus::kMalformed;
        }
        const std::size_t stored_size = std::max(payload_size, compressed_size);
        if (!checked_range(cursor + entry_header_size, stored_size, data_size)) {
            return PtxExtractStatus::kMalformed;
        }

        if (kind == kPtxEntryKind) {
            const std::uint8_t* payload = entry + entry_header_size;
            const std::uint64_t compression_flags =
                flags & (kFlagCompressedLz4 | kFlagCompressedZstd);
            if (compression_flags != 0) {
                if (encoded_uncompressed > kMaximumModuleBytes) {
                    return PtxExtractStatus::kMalformed;
                }
                const PtxExtractStatus status = decompress_entry(
                    payload, compressed_size,
                    static_cast<std::size_t>(encoded_uncompressed), flags,
                    out_ptx);
                if (status != PtxExtractStatus::kNoPtx) return status;
            } else if (compressed_size != 0) {
                return PtxExtractStatus::kUnsupported;
            } else if (payload_size > kMaximumModuleBytes) {
                return PtxExtractStatus::kMalformed;
            } else if (extract_ptx_bytes(payload, payload_size, out_ptx)) {
                return PtxExtractStatus::kFound;
            }
        }

        cursor += entry_header_size + stored_size;
    }
    return PtxExtractStatus::kNoPtx;
}

}  // namespace cumetal::fatbin
