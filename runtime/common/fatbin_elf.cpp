#include "fatbin_elf.h"

#include <cstdint>
#include <cstring>
#include <string_view>

namespace cumetal::fatbin {
namespace {

constexpr std::size_t kElf64HeaderSize = 64;
constexpr std::size_t kElf64SectionHeaderSize = 64;
constexpr std::uint32_t kFatbinBlobMagic = 0xBA55ED50u;
constexpr std::uint16_t kFatbinHeaderMinSize = 16u;

struct FatbinBlobHeader {
    std::uint32_t magic = 0;
    std::uint16_t version = 0;
    std::uint16_t header_size = 0;
    std::uint64_t fat_size = 0;
};

struct SectionView {
    std::uint32_t name_offset = 0;
    std::uint32_t link = 0;
    std::uint64_t offset = 0;
    std::uint64_t size = 0;
};

template <typename T>
T read_value(const std::uint8_t* bytes, std::size_t offset) {
    T value{};
    std::memcpy(&value, bytes + offset, sizeof(value));
    return value;
}

bool checked_range(std::uint64_t offset,
                   std::uint64_t size,
                   std::size_t limit) {
    return offset <= limit && size <= limit - static_cast<std::size_t>(offset);
}

bool checked_table_range(std::uint64_t offset,
                         std::uint16_t entry_size,
                         std::uint64_t entry_count,
                         std::size_t limit) {
    if (entry_count == 0 || entry_size == 0 || offset > limit) {
        return false;
    }
    return entry_count <=
           (limit - static_cast<std::size_t>(offset)) / entry_size;
}

SectionView read_section(const std::uint8_t* bytes,
                         std::uint64_t section_table_offset,
                         std::uint16_t section_entry_size,
                         std::uint64_t index) {
    const std::size_t base =
        static_cast<std::size_t>(section_table_offset) +
        static_cast<std::size_t>(section_entry_size) * index;
    SectionView section;
    section.name_offset = read_value<std::uint32_t>(bytes, base);
    section.offset = read_value<std::uint64_t>(bytes, base + 24);
    section.size = read_value<std::uint64_t>(bytes, base + 32);
    section.link = read_value<std::uint32_t>(bytes, base + 40);
    return section;
}

bool extract_ptx(const std::uint8_t* bytes,
                 std::size_t size,
                 std::string* out_ptx) {
    constexpr std::string_view kMarker = ".version";
    if (bytes == nullptr || out_ptx == nullptr || size < kMarker.size()) {
        return false;
    }
    for (std::size_t i = 0; i + kMarker.size() <= size; ++i) {
        if (std::memcmp(bytes + i, kMarker.data(), kMarker.size()) != 0) {
            continue;
        }
        const char* start = reinterpret_cast<const char*>(bytes + i);
        std::size_t candidate_size = size - i;
        if (const void* terminator = std::memchr(start, '\0', candidate_size);
            terminator != nullptr) {
            candidate_size = static_cast<const char*>(terminator) - start;
        } else {
            for (std::size_t j = candidate_size; j > 0; --j) {
                if (start[j - 1] == '}') {
                    candidate_size = j;
                    break;
                }
            }
        }
        if (candidate_size == 0) {
            continue;
        }
        std::string candidate(start, candidate_size);
        if (candidate.find(".entry") == std::string::npos) {
            continue;
        }
        *out_ptx = std::move(candidate);
        return true;
    }
    return false;
}

bool is_ptx_section_name(std::string_view name) {
    return name == ".nv_fatbin" || name == ".nvFatBinSegment" ||
           name == ".ptx" || name == ".nv_ptx";
}

ElfPtxStatus extract_from_section(const std::uint8_t* bytes,
                                  const SectionView& section,
                                  std::string* out_ptx) {
    const auto* section_bytes = bytes + static_cast<std::size_t>(section.offset);
    const std::size_t section_size = static_cast<std::size_t>(section.size);

    for (std::size_t i = 0; i + sizeof(FatbinBlobHeader) <= section_size; ++i) {
        if (read_value<std::uint32_t>(section_bytes, i) != kFatbinBlobMagic) {
            continue;
        }
        FatbinBlobHeader header{};
        std::memcpy(&header, section_bytes + i, sizeof(header));
        if (header.header_size < kFatbinHeaderMinSize ||
            !checked_range(i + header.header_size, header.fat_size, section_size)) {
            return ElfPtxStatus::kMalformed;
        }
        if (extract_ptx(section_bytes + i + header.header_size,
                        static_cast<std::size_t>(header.fat_size),
                        out_ptx)) {
            return ElfPtxStatus::kFound;
        }
    }

    return extract_ptx(section_bytes, section_size, out_ptx)
               ? ElfPtxStatus::kFound
               : ElfPtxStatus::kNoPtx;
}

}  // namespace

ElfPtxStatus extract_elf64_ptx(const void* image,
                               std::size_t max_image_bytes,
                               std::string* out_ptx) {
    if (image == nullptr || out_ptx == nullptr ||
        max_image_bytes < kElf64HeaderSize) {
        return ElfPtxStatus::kNotElf;
    }
    const auto* bytes = static_cast<const std::uint8_t*>(image);
    if (bytes[0] != 0x7f || bytes[1] != 'E' || bytes[2] != 'L' ||
        bytes[3] != 'F') {
        return ElfPtxStatus::kNotElf;
    }
    // ELFCLASS64, ELFDATA2LSB, and EV_CURRENT.
    if (bytes[4] != 2 || bytes[5] != 1 || bytes[6] != 1) {
        return ElfPtxStatus::kUnsupported;
    }

    const std::uint64_t section_table_offset =
        read_value<std::uint64_t>(bytes, 40);
    const std::uint16_t elf_header_size =
        read_value<std::uint16_t>(bytes, 52);
    const std::uint16_t section_entry_size =
        read_value<std::uint16_t>(bytes, 58);
    const std::uint16_t encoded_section_count =
        read_value<std::uint16_t>(bytes, 60);
    const std::uint16_t encoded_string_table_index =
        read_value<std::uint16_t>(bytes, 62);

    if (elf_header_size < kElf64HeaderSize ||
        section_table_offset < elf_header_size ||
        section_entry_size < kElf64SectionHeaderSize ||
        !checked_range(section_table_offset, section_entry_size,
                       max_image_bytes)) {
        return ElfPtxStatus::kMalformed;
    }

    // ELF64 uses section header 0 to carry values that do not fit in the ELF
    // header: sh_size is the real section count when e_shnum is zero, and
    // sh_link is the real string-table index when e_shstrndx is SHN_XINDEX.
    const SectionView section_zero =
        read_section(bytes, section_table_offset, section_entry_size, 0);
    const std::uint64_t section_count =
        encoded_section_count == 0 ? section_zero.size
                                   : encoded_section_count;
    const std::uint64_t string_table_index =
        encoded_string_table_index == 0xffff ? section_zero.link
                                             : encoded_string_table_index;

    if (string_table_index == 0 ||
        string_table_index >= section_count ||
        !checked_table_range(section_table_offset, section_entry_size,
                             section_count, max_image_bytes)) {
        return ElfPtxStatus::kMalformed;
    }

    const SectionView string_table =
        read_section(bytes, section_table_offset, section_entry_size,
                     string_table_index);
    if (!checked_range(string_table.offset, string_table.size,
                       max_image_bytes)) {
        return ElfPtxStatus::kMalformed;
    }
    const char* names =
        reinterpret_cast<const char*>(bytes + string_table.offset);
    const std::size_t names_size = static_cast<std::size_t>(string_table.size);

    for (std::uint64_t index = 0; index < section_count; ++index) {
        const SectionView section =
            read_section(bytes, section_table_offset, section_entry_size, index);
        if (section.name_offset >= names_size ||
            !checked_range(section.offset, section.size, max_image_bytes)) {
            return ElfPtxStatus::kMalformed;
        }
        const char* name = names + section.name_offset;
        const std::size_t remaining = names_size - section.name_offset;
        const void* terminator = std::memchr(name, '\0', remaining);
        if (terminator == nullptr) {
            return ElfPtxStatus::kMalformed;
        }
        const std::string_view section_name(
            name, static_cast<const char*>(terminator) - name);
        if (!is_ptx_section_name(section_name)) {
            continue;
        }
        const ElfPtxStatus result =
            extract_from_section(bytes, section, out_ptx);
        if (result != ElfPtxStatus::kNoPtx) {
            return result;
        }
    }
    return ElfPtxStatus::kNoPtx;
}

}  // namespace cumetal::fatbin
