#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace cumetal::fatbin {

enum class PtxExtractStatus {
    kNotFatbin,
    kFound,
    kNoPtx,
    kMalformed,
    kUnsupported,
};

// Extract PTX from a bounded CUDA fatbinary payload beginning with the
// 0xba55ed50 outer header. Both legacy raw-PTX payloads and version-0x0101
// framed entries are accepted. Framed LZ4/Zstd payloads are decompressed only
// within max_image_bytes and the 64 MiB module ceiling.
PtxExtractStatus extract_fatbin_ptx(const void* image,
                                    std::size_t max_image_bytes,
                                    std::string* out_ptx);

// Validate and extract one PTX program from a bounded byte range.
bool extract_ptx_bytes(const std::uint8_t* bytes,
                       std::size_t size,
                       std::string* out_ptx);

}  // namespace cumetal::fatbin
