#pragma once

#include <cctype>
#include <charconv>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace cumetal::ir::detail {

// Parse one PTX integer literal as its exact 64-bit bit pattern. Expressions,
// malformed suffixes and overflow are not integer literals.
inline std::optional<std::uint64_t> integer_literal_bits(std::string_view text) {
    bool negative = false;
    if (!text.empty() && (text.front() == '-' || text.front() == '+')) {
        negative = text.front() == '-';
        text.remove_prefix(1);
    }
    if (!text.empty() && (text.back() == 'U' || text.back() == 'u')) text.remove_suffix(1);
    int base = 10;
    if (text.size() > 1 && text.front() == '0') {
        base = 8;
        if (text[1] == 'x' || text[1] == 'X') {
            base = 16;
            text.remove_prefix(2);
        } else if (text[1] == 'b' || text[1] == 'B') {
            base = 2;
            text.remove_prefix(2);
        }
    }
    if (text.empty()) return std::nullopt;
    std::uint64_t value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value, base);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) return std::nullopt;
    return negative ? std::uint64_t{0} - value : value;
}

inline std::string trim(std::string_view input) {
    std::size_t begin = 0;
    while (begin < input.size() &&
           std::isspace(static_cast<unsigned char>(input[begin])) != 0) {
        ++begin;
    }
    std::size_t end = input.size();
    while (end > begin &&
           std::isspace(static_cast<unsigned char>(input[end - 1])) != 0) {
        --end;
    }
    return std::string(input.substr(begin, end - begin));
}

inline bool starts_with(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

}  // namespace cumetal::ir::detail
