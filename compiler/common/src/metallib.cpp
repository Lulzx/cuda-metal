#include "cumetal/common/metallib.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

namespace cumetal::common {
namespace {

constexpr std::array<std::uint8_t, 4> kRawBitcodeSig{0x42, 0x43, 0xC0, 0xDE};
constexpr std::array<std::uint8_t, 4> kWrappedBitcodeSig{0xDE, 0xC0, 0x17, 0x0B};
constexpr std::array<std::uint8_t, 4> kCumetalMagic{static_cast<std::uint8_t>('M'),
                                                    static_cast<std::uint8_t>('T'),
                                                    static_cast<std::uint8_t>('L'),
                                                    static_cast<std::uint8_t>('B')};
constexpr std::size_t kCumetalHeaderSize = 40;
constexpr std::size_t kCumetalFunctionEntrySize = 32;
constexpr std::size_t kMetallibHeaderSize = 88;

std::optional<std::uint16_t> read_u16_le(const std::vector<std::uint8_t>& bytes,
                                         std::size_t offset) {
    if (offset + 2 > bytes.size()) {
        return std::nullopt;
    }
    std::uint16_t value = 0;
    value |= static_cast<std::uint16_t>(bytes[offset + 0]);
    value |= static_cast<std::uint16_t>(bytes[offset + 1]) << 8;
    return value;
}

std::optional<std::uint32_t> read_u32_le(const std::vector<std::uint8_t>& bytes,
                                         std::size_t offset) {
    if (offset + 4 > bytes.size()) {
        return std::nullopt;
    }
    std::uint32_t value = 0;
    value |= static_cast<std::uint32_t>(bytes[offset + 0]);
    value |= static_cast<std::uint32_t>(bytes[offset + 1]) << 8;
    value |= static_cast<std::uint32_t>(bytes[offset + 2]) << 16;
    value |= static_cast<std::uint32_t>(bytes[offset + 3]) << 24;
    return value;
}

std::optional<std::uint64_t> read_u64_le(const std::vector<std::uint8_t>& bytes,
                                         std::size_t offset) {
    if (offset + 8 > bytes.size()) {
        return std::nullopt;
    }
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < 8; ++i) {
        value |= static_cast<std::uint64_t>(bytes[offset + i]) << (8 * i);
    }
    return value;
}

bool matches_sig_at(const std::vector<std::uint8_t>& bytes,
                    std::size_t offset,
                    const std::array<std::uint8_t, 4>& sig) {
    return offset + sig.size() <= bytes.size() &&
           std::memcmp(bytes.data() + offset, sig.data(), sig.size()) == 0;
}

bool is_visible_ascii(std::uint8_t c) {
    return c >= 32 && c <= 126;
}

bool is_valid_symbol_char(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.' || c == '$' ||
           c == '+' || c == '-';
}

std::string sanitize_magic(const std::array<std::uint8_t, 4>& magic) {
    std::string out;
    out.reserve(4);
    for (std::uint8_t c : magic) {
        out.push_back(is_visible_ascii(c) ? static_cast<char>(c) : '.');
    }
    return out;
}

bool parse_cstring(const std::vector<std::uint8_t>& bytes,
                   std::size_t offset,
                   std::size_t end,
                   std::string* out) {
    if (offset >= end || end > bytes.size() || out == nullptr) {
        return false;
    }

    std::size_t i = offset;
    std::string value;
    value.reserve(32);
    while (i < end && bytes[i] != 0) {
        const char c = static_cast<char>(bytes[i]);
        if (!is_visible_ascii(bytes[i]) || (!is_valid_symbol_char(c) && c != ' ')) {
            return false;
        }
        value.push_back(c);
        ++i;
    }

    if (i >= end || bytes[i] != 0 || value.empty()) {
        return false;
    }

    *out = value;
    return true;
}

bool parse_cstring_from_blob(const std::vector<std::uint8_t>& bytes,
                             std::size_t offset,
                             std::size_t size,
                             std::string* out) {
    if (offset + size > bytes.size() || size == 0 || out == nullptr) {
        return false;
    }
    std::size_t end = offset + size;
    while (end > offset && bytes[end - 1] == 0) {
        --end;
    }
    if (end <= offset) {
        return false;
    }
    for (std::size_t i = offset; i < end; ++i) {
        if (!is_visible_ascii(bytes[i])) {
            return false;
        }
    }
    *out = std::string(reinterpret_cast<const char*>(bytes.data() + offset), end - offset);
    return true;
}

std::string to_hex_prefix(const std::vector<std::uint8_t>& bytes,
                          std::size_t offset,
                          std::size_t size,
                          std::size_t max_bytes) {
    const std::size_t n = std::min(size, max_bytes);
    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < n; ++i) {
        out << std::setw(2) << static_cast<int>(bytes[offset + i]);
    }
    return out.str();
}

bool looks_like_bitcode_signature(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return matches_sig_at(bytes, offset, kRawBitcodeSig) ||
           matches_sig_at(bytes, offset, kWrappedBitcodeSig);
}

std::vector<KernelMetadataField> parse_metadata_blob(const std::vector<std::uint8_t>& bytes,
                                                     std::size_t offset,
                                                     std::size_t size) {
    std::vector<KernelMetadataField> fields;
    if (size == 0 || offset + size > bytes.size()) {
        return fields;
    }

    const std::string blob(reinterpret_cast<const char*>(bytes.data() + offset), size);
    std::size_t start = 0;
    while (start < blob.size()) {
        const std::size_t nl = blob.find('\n', start);
        const std::size_t end = (nl == std::string::npos) ? blob.size() : nl;
        if (end > start) {
            const std::string line = blob.substr(start, end - start);
            const std::size_t eq = line.find('=');
            if (eq != std::string::npos && eq > 0 && eq + 1 < line.size()) {
                fields.push_back({.key = line.substr(0, eq), .value = line.substr(eq + 1)});
            }
        }
        if (nl == std::string::npos) {
            break;
        }
        start = nl + 1;
    }
    return fields;
}

CumetalContainerInfo parse_cumetal_container(const std::vector<std::uint8_t>& bytes) {
    CumetalContainerInfo info;
    if (bytes.size() < kCumetalHeaderSize) {
        return info;
    }

    if (!matches_sig_at(bytes, 0, kCumetalMagic)) {
        return info;
    }

    const auto version = read_u32_le(bytes, 4);
    const auto total_size = read_u32_le(bytes, 8);
    const auto function_table_offset = read_u32_le(bytes, 12);
    const auto function_count = read_u32_le(bytes, 16);
    const auto string_table_offset = read_u32_le(bytes, 20);
    const auto string_table_size = read_u32_le(bytes, 24);
    const auto payload_offset = read_u32_le(bytes, 28);
    const auto payload_size = read_u32_le(bytes, 32);
    const auto flags = read_u32_le(bytes, 36);

    if (!version || !total_size || !function_table_offset || !function_count || !string_table_offset ||
        !string_table_size || !payload_offset || !payload_size || !flags) {
        return info;
    }

    if (*version < 2) {
        return info;
    }
    if (*total_size > bytes.size()) {
        return info;
    }
    if (*function_table_offset + *function_count * kCumetalFunctionEntrySize > bytes.size()) {
        return info;
    }
    if (*string_table_offset + *string_table_size > bytes.size()) {
        return info;
    }
    if (*payload_offset + *payload_size > bytes.size()) {
        return info;
    }

    info.parsed = true;
    info.version = *version;
    info.flags = *flags;
    info.function_table_offset = *function_table_offset;
    info.function_count = *function_count;
    info.string_table_offset = *string_table_offset;
    info.string_table_size = *string_table_size;
    info.payload_offset = *payload_offset;
    info.payload_size = *payload_size;

    const std::size_t strings_end = static_cast<std::size_t>(*string_table_offset + *string_table_size);

    for (std::uint32_t i = 0; i < *function_count; ++i) {
        const std::size_t entry_offset =
            static_cast<std::size_t>(*function_table_offset) + i * kCumetalFunctionEntrySize;
        const auto name_rel = read_u32_le(bytes, entry_offset + 0);
        const auto bitcode_rel = read_u32_le(bytes, entry_offset + 4);
        const auto bitcode_size = read_u32_le(bytes, entry_offset + 8);
        const auto metadata_rel = read_u32_le(bytes, entry_offset + 12);
        const auto metadata_size = read_u32_le(bytes, entry_offset + 16);

        if (!name_rel || !bitcode_rel || !bitcode_size || !metadata_rel || !metadata_size) {
            info.parsed = false;
            info.kernels.clear();
            return info;
        }

        const std::size_t name_offset = static_cast<std::size_t>(*string_table_offset + *name_rel);
        std::string name;
        if (!parse_cstring(bytes, name_offset, strings_end, &name)) {
            info.parsed = false;
            info.kernels.clear();
            return info;
        }

        const std::uint64_t bitcode_abs = static_cast<std::uint64_t>(*payload_offset) + *bitcode_rel;
        const std::uint64_t metadata_abs = static_cast<std::uint64_t>(*payload_offset) + *metadata_rel;

        if (bitcode_abs + *bitcode_size > bytes.size() || metadata_abs + *metadata_size > bytes.size()) {
            info.parsed = false;
            info.kernels.clear();
            return info;
        }

        KernelRecord record;
        record.name = std::move(name);
        record.bitcode_offset = bitcode_abs;
        record.bitcode_size = *bitcode_size;
        record.bitcode_signature_ok =
            looks_like_bitcode_signature(bytes, static_cast<std::size_t>(bitcode_abs));
        record.metadata =
            parse_metadata_blob(bytes, static_cast<std::size_t>(metadata_abs), *metadata_size);

        info.kernels.push_back(std::move(record));
    }

    return info;
}

bool parse_real_metallib(MetallibSummary* summary, const std::vector<std::uint8_t>& bytes) {
    if (summary == nullptr || bytes.size() < kMetallibHeaderSize) {
        return false;
    }
    if (!matches_sig_at(bytes, 0, kCumetalMagic)) {
        return false;
    }

    const auto function_list_offset = read_u64_le(bytes, 24);
    const auto function_list_size = read_u64_le(bytes, 32);
    const auto bitcode_section_offset = read_u64_le(bytes, 72);
    const auto bitcode_section_size = read_u64_le(bytes, 80);

    if (!function_list_offset || !function_list_size || !bitcode_section_offset || !bitcode_section_size) {
        return false;
    }

    if (*bitcode_section_offset + *bitcode_section_size > bytes.size()) {
        return false;
    }

    std::vector<std::size_t> function_list_end_candidates;
    const std::uint64_t function_list_end_base = *function_list_offset + *function_list_size;
    if (function_list_end_base <= bytes.size()) {
        function_list_end_candidates.push_back(static_cast<std::size_t>(function_list_end_base));
    }

    // Some metallib variants report function-list size excluding the leading entry-count word.
    const std::uint64_t function_list_end_with_count = function_list_end_base + 4;
    if (function_list_end_with_count <= bytes.size() &&
        function_list_end_with_count != function_list_end_base) {
        function_list_end_candidates.push_back(static_cast<std::size_t>(function_list_end_with_count));
    }

    for (const std::size_t function_list_end : function_list_end_candidates) {
        std::size_t cursor = static_cast<std::size_t>(*function_list_offset);

        const auto entry_count = read_u32_le(bytes, cursor);
        if (!entry_count) {
            continue;
        }
        cursor += 4;

        std::vector<KernelRecord> kernels;
        kernels.reserve(*entry_count);
        bool parse_failed = false;

        for (std::uint32_t entry = 0; entry < *entry_count; ++entry) {
            if (cursor + 4 > function_list_end) {
                parse_failed = true;
                break;
            }

            const auto group_size = read_u32_le(bytes, cursor);
            if (!group_size || *group_size < 4 || cursor + *group_size > function_list_end) {
                parse_failed = true;
                break;
            }

            const std::size_t group_begin = cursor + 4;
            const std::size_t group_end = cursor + *group_size;
            std::size_t tag_cursor = group_begin;

            KernelRecord kernel;
            kernel.name = "function_" + std::to_string(entry);

            std::uint64_t bitcode_rel = 0;
            std::uint64_t bitcode_size = 0;
            bool have_offt = false;
            bool have_mdsz = false;

            while (tag_cursor + 6 <= group_end) {
                const std::string tag(reinterpret_cast<const char*>(bytes.data() + tag_cursor), 4);
                const auto tag_size = read_u16_le(bytes, tag_cursor + 4);
                if (!tag_size) {
                    parse_failed = true;
                    break;
                }

                const std::size_t data_offset = tag_cursor + 6;
                const std::size_t data_end = data_offset + *tag_size;
                if (data_end > group_end) {
                    parse_failed = true;
                    break;
                }

                if (tag == "NAME") {
                    std::string name;
                    if (parse_cstring_from_blob(bytes, data_offset, *tag_size, &name)) {
                        kernel.name = std::move(name);
                    }
                } else if (tag == "MDSZ") {
                    const auto mdsz = read_u64_le(bytes, data_offset);
                    if (mdsz) {
                        bitcode_size = *mdsz;
                        have_mdsz = true;
                    }
                } else if (tag == "OFFT") {
                    const auto public_off = read_u64_le(bytes, data_offset);
                    const auto private_off = read_u64_le(bytes, data_offset + 8);
                    const auto bcode_off = read_u64_le(bytes, data_offset + 16);
                    if (public_off && private_off && bcode_off) {
                        bitcode_rel = *bcode_off;
                        have_offt = true;
                        kernel.metadata.push_back(
                            {.key = "offt.public", .value = std::to_string(*public_off)});
                        kernel.metadata.push_back(
                            {.key = "offt.private", .value = std::to_string(*private_off)});
                        kernel.metadata.push_back(
                            {.key = "offt.bitcode", .value = std::to_string(*bcode_off)});
                    }
                } else if (tag == "TYPE" && *tag_size >= 1) {
                    const std::uint8_t type = bytes[data_offset];
                    kernel.metadata.push_back({.key = "function.type", .value = std::to_string(type)});
                    if (type == 2) {
                        kernel.metadata.push_back({.key = "air.kernel", .value = "true"});
                    }
                } else if (tag == "VERS" && *tag_size >= 8) {
                    const auto air_major = read_u16_le(bytes, data_offset + 0);
                    const auto air_minor = read_u16_le(bytes, data_offset + 2);
                    const auto lang_major = read_u16_le(bytes, data_offset + 4);
                    const auto lang_minor = read_u16_le(bytes, data_offset + 6);
                    if (air_major && air_minor && lang_major && lang_minor) {
                        kernel.metadata.push_back(
                            {.key = "air.version", .value = std::to_string(*air_major) + "." +
                                                            std::to_string(*air_minor)});
                        kernel.metadata.push_back(
                            {.key = "language.version", .value = std::to_string(*lang_major) + "." +
                                                                 std::to_string(*lang_minor)});
                    }
                } else if (tag == "HASH" && *tag_size > 0) {
                    kernel.metadata.push_back({.key = "function.hash.prefix",
                                               .value = to_hex_prefix(bytes, data_offset, *tag_size, 8)});
                } else if (tag == "ENDT") {
                    tag_cursor = group_end;
                    break;
                }

                tag_cursor = data_end;
            }

            if (parse_failed) {
                break;
            }

            if (have_offt && have_mdsz) {
                kernel.bitcode_offset = *bitcode_section_offset + bitcode_rel;
                kernel.bitcode_size = bitcode_size;
                if (kernel.bitcode_offset + kernel.bitcode_size <= bytes.size()) {
                    kernel.bitcode_signature_ok =
                        looks_like_bitcode_signature(bytes, static_cast<std::size_t>(kernel.bitcode_offset));
                }
            }

            kernels.push_back(std::move(kernel));
            cursor += *group_size;
        }

        if (parse_failed) {
            continue;
        }

        summary->function_list_parsed = true;
        summary->function_list_parser = "metallib-function-list";
        summary->kernels = std::move(kernels);
        return true;
    }

    return false;
}

void scan_bitcode_sections(MetallibSummary* summary, const std::vector<std::uint8_t>& bytes) {
    if (summary == nullptr) {
        return;
    }

    if (!summary->kernels.empty()) {
        for (const auto& kernel : summary->kernels) {
            if (kernel.bitcode_size == 0) {
                continue;
            }
            BitcodeSection section;
            section.kind = "kernel-bitcode";
            section.offset = kernel.bitcode_offset;
            section.size = kernel.bitcode_size;
            section.reliable_size = (kernel.bitcode_offset + kernel.bitcode_size <= bytes.size());
            summary->bitcode_sections.push_back(section);
        }
        return;
    }

    std::vector<std::size_t> all_sigs;

    for (std::size_t i = 0; i + 4 <= bytes.size(); ++i) {
        if (matches_sig_at(bytes, i, kWrappedBitcodeSig)) {
            BitcodeSection section;
            section.kind = "llvm-bitcode-wrapper";
            section.offset = i;

            const auto version = read_u32_le(bytes, i + 4);
            const auto inner_offset = read_u32_le(bytes, i + 8);
            const auto inner_size = read_u32_le(bytes, i + 12);
            const auto cpu_type = read_u32_le(bytes, i + 16);

            if (version && inner_offset && inner_size && cpu_type) {
                section.wrapper_version = *version;
                section.wrapper_offset = *inner_offset;
                section.wrapper_size = *inner_size;
                section.wrapper_cpu_type = *cpu_type;

                const std::uint64_t total_size =
                    static_cast<std::uint64_t>(*inner_offset) + static_cast<std::uint64_t>(*inner_size);
                section.size = total_size;
                section.reliable_size = (section.offset + section.size <= bytes.size());
            } else {
                section.size = 4;
                section.reliable_size = false;
            }

            summary->bitcode_sections.push_back(section);
            all_sigs.push_back(i);
        } else if (matches_sig_at(bytes, i, kRawBitcodeSig)) {
            all_sigs.push_back(i);
        }
    }

    std::sort(all_sigs.begin(), all_sigs.end());
    all_sigs.erase(std::unique(all_sigs.begin(), all_sigs.end()), all_sigs.end());

    for (std::size_t i = 0; i + 4 <= bytes.size(); ++i) {
        if (!matches_sig_at(bytes, i, kRawBitcodeSig)) {
            continue;
        }

        std::uint64_t next = bytes.size();
        for (std::size_t sig : all_sigs) {
            if (sig > i) {
                next = sig;
                break;
            }
        }

        BitcodeSection section;
        section.kind = "llvm-bitcode-raw";
        section.offset = i;
        section.size = next >= i ? next - i : 0;
        section.reliable_size = next > i;
        summary->bitcode_sections.push_back(section);
    }
}

void scan_string_candidates(MetallibSummary* summary,
                            const std::vector<std::uint8_t>& bytes,
                            std::size_t max_strings) {
    if (summary == nullptr || max_strings == 0) {
        return;
    }

    for (std::size_t i = 0; i < bytes.size() && summary->strings.size() < max_strings; ++i) {
        if (i > 0 && bytes[i - 1] != 0) {
            continue;
        }
        if (!is_visible_ascii(bytes[i])) {
            continue;
        }

        std::size_t j = i;
        bool seen_alpha = false;
        std::string value;
        value.reserve(64);

        while (j < bytes.size() && bytes[j] != 0 && value.size() <= 96) {
            const char c = static_cast<char>(bytes[j]);
            if (!is_visible_ascii(bytes[j]) || !is_valid_symbol_char(c)) {
                value.clear();
                break;
            }
            seen_alpha = seen_alpha || (std::isalpha(static_cast<unsigned char>(c)) != 0);
            value.push_back(c);
            ++j;
        }

        if (value.size() < 4 || value.size() > 96 || !seen_alpha) {
            continue;
        }
        if (j >= bytes.size() || bytes[j] != 0) {
            continue;
        }

        summary->strings.push_back({.offset = i, .value = value});
        i = j;
    }
}

void infer_air_reflection_argument_counts(MetallibSummary* summary,
                                          const std::vector<std::uint8_t>& bytes) {
    if (summary == nullptr || summary->cumetal.parsed) {
        return;
    }

    for (auto& kernel : summary->kernels) {
        if (kernel.name.empty()) {
            continue;
        }

        const std::string prefix = kernel.name + "_param_";
        std::array<bool, 32> seen{};
        for (std::size_t i = 0; i + prefix.size() < bytes.size(); ++i) {
            if (std::memcmp(bytes.data() + i, prefix.data(), prefix.size()) != 0) {
                continue;
            }
            std::size_t cursor = i + prefix.size();
            std::size_t index = 0;
            bool have_digit = false;
            while (cursor < bytes.size() &&
                   std::isdigit(static_cast<unsigned char>(bytes[cursor])) != 0) {
                have_digit = true;
                index = index * 10 + static_cast<std::size_t>(bytes[cursor] - '0');
                ++cursor;
            }
            // Aggregate source arguments are flattened in AIR and reflected
            // as <kernel>_param_<source-index>_<byte-offset>_. They remain one
            // CUDA Driver API kernelParams slot, so count the source index
            // rather than the flattened AIR fields.
            if (have_digit && index < seen.size() && cursor < bytes.size() &&
                (bytes[cursor] == 0 || bytes[cursor] == '_')) {
                seen[index] = true;
            }
        }

        if (!seen[0]) {
            continue;
        }
        std::size_t count = 0;
        while (count < seen.size() && seen[count]) {
            ++count;
        }
        bool has_gap = false;
        for (std::size_t i = count; i < seen.size(); ++i) {
            has_gap = has_gap || seen[i];
        }
        if (!has_gap) {
            kernel.metadata.push_back(
                {.key = "kernel.arg_count", .value = std::to_string(count)});
        }
    }
}

}  // namespace

std::vector<std::uint8_t> read_file_bytes(const std::filesystem::path& path, std::string* error) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to open file: " + path.string();
        }
        return {};
    }

    const std::ifstream::pos_type end = stream.tellg();
    if (end < 0) {
        if (error != nullptr) {
            *error = "failed to determine file size: " + path.string();
        }
        return {};
    }

    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            if (error != nullptr) {
                *error = "failed to read file: " + path.string();
            }
            return {};
        }
    }

    return bytes;
}

bool write_file_bytes(const std::filesystem::path& path,
                      const std::vector<std::uint8_t>& data,
                      std::string* error) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to open output file: " + path.string();
        }
        return false;
    }

    if (!data.empty()) {
        stream.write(reinterpret_cast<const char*>(data.data()),
                     static_cast<std::streamsize>(data.size()));
    }
    if (!stream) {
        if (error != nullptr) {
            *error = "failed to write output file: " + path.string();
        }
        return false;
    }
    return true;
}

MetallibSummary inspect_metallib_bytes(const std::string& path,
                                       const std::vector<std::uint8_t>& bytes,
                                       std::size_t max_strings) {
    MetallibSummary summary;
    summary.path = path;
    summary.file_size = bytes.size();

    for (std::size_t i = 0; i < 4 && i < bytes.size(); ++i) {
        summary.magic[i] = bytes[i];
    }
    summary.magic_ascii = sanitize_magic(summary.magic);

    if (const auto word = read_u32_le(bytes, 4)) {
        summary.word1_le = *word;
    }
    if (const auto word = read_u32_le(bytes, 8)) {
        summary.word2_le = *word;
    }

    summary.cumetal = parse_cumetal_container(bytes);
    if (summary.cumetal.parsed) {
        summary.function_list_parsed = true;
        summary.function_list_parser = "cumetal-experimental";
        summary.kernels = summary.cumetal.kernels;
    } else {
        (void)parse_real_metallib(&summary, bytes);
    }

    // CuMetal's PTX lowering names explicit AIR arguments
    // <kernel>_param_<index>. Apple's production metallib preserves these
    // reflection strings even though it does not preserve CuMetal's custom
    // metadata fields. Recover the exact Driver API kernelParams bound without
    // relying on CUDA's non-existent null terminator.
    infer_air_reflection_argument_counts(&summary, bytes);

    scan_bitcode_sections(&summary, bytes);
    std::sort(summary.bitcode_sections.begin(), summary.bitcode_sections.end(),
              [](const BitcodeSection& a, const BitcodeSection& b) {
                  if (a.offset == b.offset) {
                      return a.kind < b.kind;
                  }
                  return a.offset < b.offset;
              });

    scan_string_candidates(&summary, bytes, max_strings);
    return summary;
}

bool looks_like_metallib(const MetallibSummary& summary) {
    if (summary.file_size < 16) {
        return false;
    }
    const bool magic_looks_valid = summary.magic_ascii.rfind("MTL", 0) == 0;
    return magic_looks_valid;
}

std::string hex_u64(std::uint64_t value, std::size_t width) {
    std::ostringstream out;
    out << "0x" << std::hex << std::setfill('0');
    if (width > 0) {
        out << std::setw(static_cast<int>(width));
    }
    out << value;
    return out.str();
}

}  // namespace cumetal::common
