#include "cumetal/passes/printf_lower.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cumetal::passes {
namespace {

std::string trim(std::string_view text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool is_printf_symbol(const std::string& token) {
    std::string lowered = lowercase(trim(token));
    if (!lowered.empty() && lowered.front() == '@') lowered.erase(lowered.begin());
    return lowered == "vprintf" || lowered == "printf";
}

std::size_t find_printf_callee_index(const std::vector<std::string>& operands) {
    for (std::size_t i = 0; i < operands.size(); ++i) {
        if (is_printf_symbol(trim(operands[i]))) {
            return i;
        }
    }
    return operands.size();
}

std::vector<std::string> split_call_args(std::string text) {
    text = trim(text);
    if (text.size() >= 2 && text.front() == '(' && text.back() == ')') {
        text = text.substr(1, text.size() - 2);
    }

    std::vector<std::string> args;
    std::string current;
    int bracket_depth = 0;
    bool in_quote = false;
    bool escaped = false;

    for (char c : text) {
        if (in_quote) {
            current.push_back(c);
            if (escaped) {
                escaped = false;
            } else if (c == '\\') {
                escaped = true;
            } else if (c == '"') {
                in_quote = false;
            }
            continue;
        }

        if (c == '"') {
            in_quote = true;
            current.push_back(c);
            continue;
        }
        if (c == '[' || c == '(' || c == '{') {
            ++bracket_depth;
            current.push_back(c);
            continue;
        }
        if (c == ']' || c == ')' || c == '}') {
            if (bracket_depth > 0) {
                --bracket_depth;
            }
            current.push_back(c);
            continue;
        }
        if (c == ',' && bracket_depth == 0) {
            const std::string arg = trim(current);
            if (!arg.empty()) {
                args.push_back(arg);
            }
            current.clear();
            continue;
        }

        current.push_back(c);
    }

    const std::string tail = trim(current);
    if (!tail.empty()) {
        args.push_back(tail);
    }
    return args;
}

bool is_quoted_string(const std::string& token) {
    return token.size() >= 2 && token.front() == '"' && token.back() == '"';
}

std::string unescape_string_literal(const std::string& token) {
    if (!is_quoted_string(token)) {
        return token;
    }

    std::string out;
    out.reserve(token.size());
    for (std::size_t i = 1; i + 1 < token.size(); ++i) {
        char c = token[i];
        if (c == '\\' && i + 2 < token.size()) {
            const char escaped = token[++i];
            switch (escaped) {
                case 'n':
                    out.push_back('\n');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                case '\\':
                    out.push_back('\\');
                    break;
                case '"':
                    out.push_back('"');
                    break;
                default:
                    out.push_back('\\');
                    out.push_back(escaped);
                    break;
            }
            continue;
        }
        out.push_back(c);
    }
    return out;
}

bool fail_or_warn(bool strict,
                  const std::string& message,
                  std::vector<std::string>* warnings,
                  std::string* error) {
    if (strict) {
        if (error != nullptr) {
            *error = message;
        }
        return true;
    }
    if (warnings != nullptr) {
        warnings->push_back(message);
    }
    return false;
}

std::string extract_register(std::string_view text) {
    const std::size_t begin = text.find('%');
    if (begin == std::string_view::npos) return {};
    std::size_t end = begin + 1;
    while (end < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[end]);
        if (std::isalnum(c) == 0 && c != '_' && c != '.' && c != '$') break;
        ++end;
    }
    return end > begin + 1 ? std::string(text.substr(begin, end - begin)) : std::string{};
}

std::string bracket_contents(const std::string& operand) {
    const std::size_t open = operand.find('[');
    const std::size_t close = operand.find(']', open == std::string::npos ? 0 : open + 1);
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) return {};
    return trim(std::string_view(operand).substr(open + 1, close - open - 1));
}

std::optional<std::size_t> decimal_offset(std::string_view token) {
    const std::string value = trim(token);
    if (value.empty()) return std::size_t{0};
    char* end = nullptr;
    const long long parsed = std::strtoll(value.c_str(), &end, 10);
    if (end == value.c_str() || *end != '\0' || parsed < 0) return std::nullopt;
    return static_cast<std::size_t>(parsed);
}

struct AddressRef {
    std::string reg;
    std::size_t offset = 0;
};

std::optional<AddressRef> parse_address(const std::string& operand) {
    const std::string contents = bracket_contents(operand);
    const std::string reg = extract_register(contents);
    if (reg.empty()) return std::nullopt;
    const std::size_t reg_end = contents.find(reg) + reg.size();
    std::string_view suffix(contents.data() + reg_end, contents.size() - reg_end);
    const std::size_t plus = suffix.find('+');
    if (plus == std::string_view::npos) return AddressRef{reg, 0};
    auto offset = decimal_offset(suffix.substr(plus + 1));
    if (!offset) return std::nullopt;
    return AddressRef{reg, *offset};
}

struct GlobalFormat {
    std::string bytes;
    bool truncated = false;
};

std::unordered_map<std::string, GlobalFormat> parse_initialized_b8_globals(
    std::string_view ptx,
    std::size_t max_format_length) {
    std::unordered_map<std::string, GlobalFormat> globals;
    const std::regex declaration(
        R"(\.global\s+(?:\.align\s+\d+\s+)?\.b8\s+([^\s\[]+)\s*\[\s*\d+\s*\]\s*=\s*\{([^}]*)\})");
    const std::string source(ptx);
    for (std::sregex_iterator it(source.begin(), source.end(), declaration), end; it != end; ++it) {
        GlobalFormat format;
        std::string values = (*it)[2].str();
        std::size_t begin = 0;
        bool valid = true;
        while (begin <= values.size()) {
            const std::size_t comma = values.find(',', begin);
            const std::string token = trim(std::string_view(values).substr(
                begin, comma == std::string::npos ? std::string::npos : comma - begin));
            if (!token.empty()) {
                char* parse_end = nullptr;
                const long value = std::strtol(token.c_str(), &parse_end, 0);
                if (parse_end == token.c_str() || *parse_end != '\0' || value < 0 || value > 255) {
                    valid = false;
                    break;
                }
                if (value == 0) break;
                if (format.bytes.size() < max_format_length) {
                    format.bytes.push_back(static_cast<char>(value));
                } else {
                    format.truncated = true;
                }
            }
            if (comma == std::string::npos) break;
            begin = comma + 1;
        }
        if (valid && !format.bytes.empty()) {
            globals.emplace((*it)[1].str(), std::move(format));
        }
    }
    return globals;
}

struct PackedArgument {
    std::size_t offset = 0;
    std::string value;
    int bits = 32;
    int source_line = 0;
};

bool format_uses_argument(std::string_view format) {
    for (std::size_t i = 0; i < format.size(); ++i) {
        if (format[i] != '%') continue;
        if (i + 1 < format.size() && format[i + 1] == '%') {
            ++i;
            continue;
        }
        return true;
    }
    return false;
}

// Decode the ABI emitted by Clang for CUDA device printf:
//   vprintf(pointer-to-initialized-global-format, pointer-to-packed-local-values).
// This deliberately accepts only unambiguous 32-bit tuple stores.  Wider values
// need a wider ring-buffer record before they can be represented without loss.
std::optional<PrintfLowerResult> lower_clang_vprintf_abi(
    const cumetal::ptx::EntryFunction& entry,
    const PrintfLowerOptions& options) {
    if (options.ptx_source.empty()) return std::nullopt;
    const auto globals = parse_initialized_b8_globals(options.ptx_source, options.max_format_length);
    if (globals.empty()) return std::nullopt;

    std::unordered_map<std::string, std::size_t> local_pointer;
    std::unordered_map<std::string, std::string> global_pointer;
    std::unordered_map<std::string, std::string> call_param_value;
    std::vector<PackedArgument> packed;
    std::set<int> scaffold_lines;
    PrintfLowerResult result;
    std::map<std::string, std::uint32_t> format_ids;
    std::size_t printf_calls_seen = 0;

    for (const auto& instruction : entry.instructions) {
        const auto& op = instruction.opcode;
        const auto& operands = instruction.operands;

        if (op == "mov.b64" && operands.size() == 2) {
            const std::string dest = extract_register(operands[0]);
            const std::string src_reg = extract_register(operands[1]);
            if (!dest.empty() && operands[1].find("__local_depot") != std::string::npos) {
                local_pointer[dest] = 0;
                scaffold_lines.insert(instruction.line);
            } else if (!dest.empty() && globals.contains(trim(operands[1]))) {
                global_pointer[dest] = trim(operands[1]);
                scaffold_lines.insert(instruction.line);
            } else if (!dest.empty() && !src_reg.empty() && local_pointer.contains(src_reg)) {
                local_pointer[dest] = local_pointer.at(src_reg);
                scaffold_lines.insert(instruction.line);
            }
        } else if (op.rfind("cvta.local", 0) == 0 && operands.size() == 2) {
            const std::string dest = extract_register(operands[0]);
            const std::string src = extract_register(operands[1]);
            if (!dest.empty() && local_pointer.contains(src)) {
                local_pointer[dest] = local_pointer.at(src);
                scaffold_lines.insert(instruction.line);
            }
        } else if (op.rfind("cvta.global", 0) == 0 && operands.size() == 2) {
            const std::string dest = extract_register(operands[0]);
            const std::string src = extract_register(operands[1]);
            if (!dest.empty() && global_pointer.contains(src)) {
                global_pointer[dest] = global_pointer.at(src);
                scaffold_lines.insert(instruction.line);
            }
        } else if (op == "add.u64" && operands.size() == 3) {
            const std::string dest = extract_register(operands[0]);
            const std::string base = extract_register(operands[1]);
            auto offset = decimal_offset(operands[2]);
            if (!dest.empty() && offset && local_pointer.contains(base)) {
                local_pointer[dest] = local_pointer.at(base) + *offset;
                scaffold_lines.insert(instruction.line);
            }
        } else if (op.rfind("st.local", 0) == 0 && operands.size() >= 2) {
            const auto address = parse_address(operands[0]);
            if (address && local_pointer.contains(address->reg)) {
                const int lane_bits = op.find(".b64") != std::string::npos
                                          ? 64
                                          : (op.find(".b32") != std::string::npos ? 32 : 0);
                // Clang may build a thread-local string (typically indentation
                // passed to `%s`) in the same local depot as the packed scalar
                // tuple. Byte stores are not tuple arguments; the pointer to
                // that string is represented by a later b64 tuple store.
                if (lane_bits == 0) continue;
                // The PTX parser splits operands at commas even inside the
                // brace tuple of st.local.v2/v4. Reassemble every value
                // operand before decoding the packed vprintf ABI tuple.
                std::string values;
                for (std::size_t operand_index = 1;
                     operand_index < operands.size(); ++operand_index) {
                    if (!values.empty()) values += ",";
                    values += operands[operand_index];
                }
                values = trim(values);
                if (values.size() >= 2 && values.front() == '{' && values.back() == '}') {
                    values = values.substr(1, values.size() - 2);
                }
                const auto lanes = split_call_args(values);
                if (lanes.empty()) return std::nullopt;
                for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
                    std::string value = extract_register(lanes[lane]);
                    if (value.empty()) {
                        value = trim(lanes[lane]);
                        char* parse_end = nullptr;
                        (void)std::strtoll(value.c_str(), &parse_end, 0);
                        if (parse_end == value.c_str() || *parse_end != '\0') {
                            return std::nullopt;
                        }
                    }
                    packed.push_back({local_pointer.at(address->reg) + address->offset +
                                          lane * static_cast<std::size_t>(lane_bits / 8),
                                      value,
                                      lane_bits,
                                      instruction.line});
                }
                scaffold_lines.insert(instruction.line);
            }
        } else if (op.rfind("st.param", 0) == 0 && operands.size() == 2) {
            const std::string param = bracket_contents(operands[0]);
            if (!param.empty()) {
                call_param_value[param] = trim(operands[1]);
                scaffold_lines.insert(instruction.line);
            }
        }

        if (op.rfind("call", 0) != 0) continue;
        const std::size_t callee_index = find_printf_callee_index(operands);
        if (callee_index == operands.size()) continue;
        ++printf_calls_seen;
        if (callee_index + 1 >= operands.size()) return std::nullopt;
        const auto args = split_call_args(operands[callee_index + 1]);
        if (args.size() != 2 || is_quoted_string(trim(args[0]))) return std::nullopt;
        const auto format_value = call_param_value.find(trim(args[0]));
        const auto tuple_value = call_param_value.find(trim(args[1]));
        if (format_value == call_param_value.end() || tuple_value == call_param_value.end()) {
            return std::nullopt;
        }
        const std::string format_reg = extract_register(format_value->second);
        const std::string tuple_reg = extract_register(tuple_value->second);
        const bool null_tuple = trim(tuple_value->second) == "0";
        if (!global_pointer.contains(format_reg) ||
            (!null_tuple && !local_pointer.contains(tuple_reg))) {
            return std::nullopt;
        }
        const auto format_it = globals.find(global_pointer.at(format_reg));
        if (format_it == globals.end()) return std::nullopt;

        std::vector<PackedArgument> call_args;
        if (null_tuple) {
            if (format_uses_argument(format_it->second.bytes)) return std::nullopt;
        } else {
            const std::size_t tuple_base = local_pointer.at(tuple_reg);
            for (const auto& arg : packed) {
                if (arg.offset >= tuple_base) call_args.push_back(arg);
            }
            std::sort(call_args.begin(), call_args.end(), [](const auto& lhs, const auto& rhs) {
                return lhs.offset < rhs.offset;
            });
            if (call_args.empty()) return std::nullopt;
            std::size_t expected_offset = tuple_base;
            for (const auto& arg : call_args) {
                if (arg.offset != expected_offset) return std::nullopt;
                expected_offset += static_cast<std::size_t>(arg.bits / 8);
            }
        }

        std::uint32_t format_id = 0;
        const auto existing = format_ids.find(format_it->second.bytes);
        if (existing == format_ids.end()) {
            format_id = static_cast<std::uint32_t>(result.formats.size());
            format_ids.emplace(format_it->second.bytes, format_id);
            result.formats.push_back({.id = format_id,
                                      .token = format_it->second.bytes,
                                      .literal = true,
                                      .truncated = format_it->second.truncated});
            if (format_it->second.truncated) {
                result.warnings.push_back(
                    "printf_lower: module-global format truncated to " +
                    std::to_string(options.max_format_length) + " bytes at line " +
                    std::to_string(instruction.line));
            }
        } else {
            format_id = existing->second;
        }
        PrintfLoweredCall call;
        call.source_line = instruction.line;
        call.source_opcode = instruction.opcode;
        call.format_id = format_id;
        call.format_token = format_it->second.bytes;
        for (const auto& arg : call_args) {
            const auto global = global_pointer.find(arg.value);
            call.arguments.push_back(global == global_pointer.end()
                                         ? arg.value
                                         : global->second);
            call.argument_bits.push_back(arg.bits);
        }
        call.abi_scaffold_lines.assign(scaffold_lines.begin(), scaffold_lines.end());
        result.calls.push_back(std::move(call));
        packed.clear();
    }

    if (printf_calls_seen == 0 || result.calls.size() != printf_calls_seen) return std::nullopt;
    result.ok = true;
    return result;
}

}  // namespace

PrintfLowerResult lower_printf_calls(const cumetal::ptx::EntryFunction& entry,
                                     const PrintfLowerOptions& options) {
    if (auto clang_abi = lower_clang_vprintf_abi(entry, options)) {
        return std::move(*clang_abi);
    }
    PrintfLowerResult result;

    std::map<std::string, std::uint32_t> format_ids;

    for (const auto& instruction : entry.instructions) {
        if (instruction.opcode.rfind("call", 0) != 0) {
            continue;
        }

        const std::size_t callee_index = find_printf_callee_index(instruction.operands);
        if (callee_index == instruction.operands.size()) {
            continue;
        }

        if (callee_index + 1 >= instruction.operands.size()) {
            const std::string message = "printf_lower: missing argument tuple at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        const std::vector<std::string> args = split_call_args(instruction.operands[callee_index + 1]);
        if (args.empty()) {
            const std::string message = "printf_lower: empty argument tuple at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        const std::string format_token_raw = trim(args.front());
        if (format_token_raw.empty()) {
            const std::string message = "printf_lower: empty format token at line " +
                                        std::to_string(instruction.line);
            if (fail_or_warn(options.strict, message, &result.warnings, &result.error)) {
                return result;
            }
            continue;
        }

        bool literal = false;
        bool truncated = false;
        std::string canonical_token = format_token_raw;
        if (is_quoted_string(format_token_raw)) {
            literal = true;
            canonical_token = unescape_string_literal(format_token_raw);
            if (canonical_token.size() > options.max_format_length) {
                canonical_token.resize(options.max_format_length);
                truncated = true;
                result.warnings.push_back(
                    "printf_lower: format literal truncated to " +
                    std::to_string(options.max_format_length) + " bytes at line " +
                    std::to_string(instruction.line));
            }
        }

        std::uint32_t format_id = 0;
        const auto existing = format_ids.find(canonical_token);
        if (existing == format_ids.end()) {
            format_id = static_cast<std::uint32_t>(result.formats.size());
            format_ids[canonical_token] = format_id;
            result.formats.push_back(
                {.id = format_id, .token = canonical_token, .literal = literal, .truncated = truncated});
        } else {
            format_id = existing->second;
            if (truncated && format_id < result.formats.size()) {
                result.formats[format_id].truncated = true;
            }
        }

        PrintfLoweredCall call;
        call.source_line = instruction.line;
        call.source_opcode = instruction.opcode;
        call.format_id = format_id;
        call.format_token = canonical_token;
        call.arguments.assign(args.begin() + 1, args.end());
        call.argument_bits.assign(call.arguments.size(), 32);
        result.calls.push_back(std::move(call));
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::passes
