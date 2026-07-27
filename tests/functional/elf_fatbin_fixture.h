#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace cumetal::test {

inline std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
inline void write_value(std::vector<std::uint8_t>* bytes,
                        std::size_t offset,
                        T value) {
    std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

// Minimal little-endian ELF64 image with a null section, .shstrtab, and one
// named payload section.
inline std::vector<std::uint8_t> make_elf64_image(
    std::string_view section_name,
    const std::vector<std::uint8_t>& payload) {
    constexpr std::size_t kHeaderSize = 64;
    constexpr std::size_t kSectionHeaderSize = 64;
    const std::string names =
        std::string("\0.shstrtab\0", 11) + std::string(section_name) + '\0';
    constexpr std::uint32_t kStringTableNameOffset = 1;
    constexpr std::uint32_t kPayloadNameOffset = 11;

    const std::size_t payload_offset = kHeaderSize;
    const std::size_t names_offset =
        align_up(payload_offset + payload.size(), 8);
    const std::size_t section_table_offset =
        align_up(names_offset + names.size(), 8);
    std::vector<std::uint8_t> image(
        section_table_offset + 3 * kSectionHeaderSize, 0);

    image[0] = 0x7f;
    image[1] = 'E';
    image[2] = 'L';
    image[3] = 'F';
    image[4] = 2;  // ELFCLASS64
    image[5] = 1;  // ELFDATA2LSB
    image[6] = 1;  // EV_CURRENT
    write_value<std::uint16_t>(&image, 16, 1);  // ET_REL
    write_value<std::uint32_t>(&image, 20, 1);
    write_value<std::uint64_t>(&image, 40, section_table_offset);
    write_value<std::uint16_t>(&image, 52, kHeaderSize);
    write_value<std::uint16_t>(&image, 58, kSectionHeaderSize);
    write_value<std::uint16_t>(&image, 60, 3);
    write_value<std::uint16_t>(&image, 62, 1);

    std::memcpy(image.data() + payload_offset, payload.data(), payload.size());
    std::memcpy(image.data() + names_offset, names.data(), names.size());

    const std::size_t strings = section_table_offset + kSectionHeaderSize;
    write_value<std::uint32_t>(&image, strings, kStringTableNameOffset);
    write_value<std::uint32_t>(&image, strings + 4, 3);  // SHT_STRTAB
    write_value<std::uint64_t>(&image, strings + 24, names_offset);
    write_value<std::uint64_t>(&image, strings + 32, names.size());

    const std::size_t data = section_table_offset + 2 * kSectionHeaderSize;
    write_value<std::uint32_t>(&image, data, kPayloadNameOffset);
    write_value<std::uint32_t>(&image, data + 4, 1);  // SHT_PROGBITS
    write_value<std::uint64_t>(&image, data + 24, payload_offset);
    write_value<std::uint64_t>(&image, data + 32, payload.size());
    write_value<std::uint64_t>(&image, data + 48, 1);
    return image;
}

}  // namespace cumetal::test
