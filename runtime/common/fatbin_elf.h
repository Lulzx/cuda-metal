#pragma once

#include <cstddef>
#include <string>

namespace cumetal::fatbin {

enum class ElfPtxStatus {
    kNotElf,
    kFound,
    kNoPtx,
    kMalformed,
    kUnsupported,
};

// Extract PTX from named sections in a little-endian ELF64 image. The ELF
// header and section table determine every inspected range; max_image_bytes is
// a hard upper bound for all offsets and sizes read from the image.
ElfPtxStatus extract_elf64_ptx(const void* image,
                               std::size_t max_image_bytes,
                               std::string* out_ptx);

}  // namespace cumetal::fatbin
