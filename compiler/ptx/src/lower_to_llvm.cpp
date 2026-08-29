#include "cumetal/ptx/lower_to_llvm.h"

#include "cumetal/passes/phase1_pipeline.h"
#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cumetal::ptx {
namespace {

struct ParamInfo {
    std::string ptx_type;
    std::string llvm_type;
    std::string name;
    std::string raw_name;
    std::string builtin_air_key;
    std::string builtin_air_type_name;
};

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

bool parse_major_minor(const std::string& value, int* major, int* minor) {
    if (major == nullptr || minor == nullptr) {
        return false;
    }
    const std::size_t dot = value.find('.');
    if (dot == std::string::npos) {
        return false;
    }
    const std::string major_text = trim(value.substr(0, dot));
    const std::string minor_text = trim(value.substr(dot + 1));
    if (major_text.empty() || minor_text.empty()) {
        return false;
    }
    for (char c : major_text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    for (char c : minor_text) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    *major = std::stoi(major_text);
    *minor = std::stoi(minor_text);
    return true;
}

std::string map_param_type_to_llvm(const std::string& ptx_type, bool is_pointer) {
    if (is_pointer) {
        return "float addrspace(1)*";
    }
    if (ptx_type == ".u64" || ptx_type == ".s64" || ptx_type == ".b64") {
        return "i64";
    }
    if (ptx_type == ".u32" || ptx_type == ".s32" || ptx_type == ".b32") {
        return "i32";
    }
    if (ptx_type == ".u16" || ptx_type == ".s16" || ptx_type == ".b16") {
        return "i16";
    }
    if (ptx_type == ".u8" || ptx_type == ".s8" || ptx_type == ".b8") {
        return "i8";
    }
    if (ptx_type == ".f32") {
        return "float";
    }
    if (ptx_type == ".f64") {
        return "double";
    }
    return "i32";
}

std::string sanitize_llvm_identifier(std::string value, const std::string& fallback) {
    if (value.empty()) {
        value = fallback;
    }
    for (char& c : value) {
        const bool ok =
            std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.' || c == '$';
        if (!ok) {
            c = '_';
        }
    }
    if (value.empty() || std::isdigit(static_cast<unsigned char>(value.front())) != 0) {
        value.insert(value.begin(), '_');
    }
    return value;
}

std::map<std::string, std::string> to_field_map(const cumetal::passes::KernelMetadata& metadata) {
    std::map<std::string, std::string> map;
    for (const auto& field : metadata.fields) {
        if (!field.key.empty()) {
            map[field.key] = field.value;
        }
    }
    return map;
}

bool is_pointer_type(const std::string& llvm_type) {
    return llvm_type.find('*') != std::string::npos;
}

bool is_builtin_param(const ParamInfo& param) {
    return !param.builtin_air_key.empty();
}

bool is_device_buffer_pointer(const std::string& llvm_type) {
    return llvm_type.find("addrspace(1)*") != std::string::npos;
}

bool is_constant_buffer_pointer(const std::string& llvm_type) {
    return llvm_type.find("addrspace(2)*") != std::string::npos;
}

bool is_threadgroup_buffer_pointer(const std::string& llvm_type) {
    return llvm_type.find("addrspace(3)*") != std::string::npos;
}

std::string pointee_type_from_pointer(const std::string& llvm_type) {
    const std::size_t star = llvm_type.find('*');
    if (star == std::string::npos) {
        return "i8";
    }
    return trim(llvm_type.substr(0, star));
}

std::string air_type_name_for_param(const ParamInfo& param, bool is_thread_position) {
    if (is_builtin_param(param) && !param.builtin_air_type_name.empty()) {
        return param.builtin_air_type_name;
    }
    if (is_thread_position) {
        return "uint";
    }
    if (param.llvm_type == "<3 x i32>") {
        return "uint3";
    }
    if (is_device_buffer_pointer(param.llvm_type)) {
        return pointee_type_from_pointer(param.llvm_type) == "double" ? "double" : "float";
    }
    if (is_threadgroup_buffer_pointer(param.llvm_type)) {
        const std::string pointee = pointee_type_from_pointer(param.llvm_type);
        if (pointee == "i8") return "uchar";
        if (pointee == "i16") return "ushort";
        if (pointee == "i32") return "uint";
        if (pointee == "float") return "float";
        return "uint";
    }
    if (is_constant_buffer_pointer(param.llvm_type)) {
        // Use pointee type to pick the correct AIR type name.
        const std::string p = pointee_type_from_pointer(param.llvm_type);
        if (p.find("i64") != std::string::npos || p == "double") return "ulong";
        if (p.find("i16") != std::string::npos || p == "half")   return "ushort";
        return "uint";
    }
    if (param.llvm_type == "float") {
        return "float";
    }
    if (param.llvm_type == "double") {
        return "double";
    }
    if (param.llvm_type == "i64") {
        return "ulong";
    }
    if (param.llvm_type == "i32") {
        return "uint";
    }
    return "uint";
}

int byte_size_for_llvm_type(const std::string& llvm_type) {
    if (llvm_type.find('*') != std::string::npos) {
        const std::string pointee = pointee_type_from_pointer(llvm_type);
        if (pointee == "i8") return 1;
        if (pointee == "i16" || pointee == "half") return 2;
        if (pointee == "i32" || pointee == "float") return 4;
        if (pointee == "i64" || pointee == "double") return 8;
    }
    if (llvm_type == "<3 x i32>") {
        return 12;
    }
    if (llvm_type.find("double") != std::string::npos || llvm_type.find("i64") != std::string::npos) {
        return 8;
    }
    return 4;
}

std::optional<int> parse_trailing_array_size_bytes(const std::string& raw_name) {
    const std::size_t open = raw_name.rfind('[');
    const std::size_t close = raw_name.rfind(']');
    if (open == std::string::npos || close == std::string::npos || close <= open + 1) {
        return std::nullopt;
    }
    int value = 0;
    for (std::size_t i = open + 1; i < close; ++i) {
        const unsigned char c = static_cast<unsigned char>(raw_name[i]);
        if (std::isdigit(c) == 0) {
            return std::nullopt;
        }
        value = value * 10 + static_cast<int>(c - '0');
    }
    if (value <= 0) {
        return std::nullopt;
    }
    return value;
}

int param_type_size_bytes_from_ptx(const std::string& ptx_type) {
    if (ptx_type == ".u64" || ptx_type == ".s64" || ptx_type == ".b64" || ptx_type == ".f64") {
        return 8;
    }
    if (ptx_type == ".u32" || ptx_type == ".s32" || ptx_type == ".b32" || ptx_type == ".f32") {
        return 4;
    }
    if (ptx_type == ".u16" || ptx_type == ".s16" || ptx_type == ".b16") {
        return 2;
    }
    if (ptx_type == ".u8" || ptx_type == ".s8" || ptx_type == ".b8") {
        return 1;
    }
    return 4;
}

int byte_size_for_param_metadata(const ParamInfo& param) {
    if (const auto arr = parse_trailing_array_size_bytes(param.raw_name)) {
        return *arr;
    }
    return byte_size_for_llvm_type(param.llvm_type);
}

struct GenericLlvmBodyResult {
    bool ok = false;
    bool uses_atomic_lock_bank = false;
    bool uses_device_heap = false;
    bool uses_device_launch_queue = false;
    bool uses_device_clock = false;
    std::string body_ir;
    std::string helper_ir;
    std::vector<ParamInfo> builtin_params;
    std::vector<std::string> declarations;
    std::unordered_map<std::string, std::vector<int>> device_function_param_address_spaces;
    std::vector<std::string> warnings;
    std::string error;
};

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

std::string opcode_root(std::string_view opcode) {
    const std::size_t dot = opcode.find('.');
    std::string root = dot == std::string::npos ? std::string(opcode)
                                                : std::string(opcode.substr(0, dot));
    while (!root.empty() && root.back() == ';') root.pop_back();
    return root;
}

std::vector<std::string> split_comma_list(std::string text) {
    text = trim(text);
    if (!text.empty() && text.front() == '{' && text.back() == '}') {
        text = text.substr(1, text.size() - 2);
    }
    std::vector<std::string> out;
    std::string current;
    int depth = 0;
    for (char c : text) {
        if (c == '{' || c == '[' || c == '(') {
            ++depth;
            current.push_back(c);
            continue;
        }
        if (c == '}' || c == ']' || c == ')') {
            if (depth > 0) {
                --depth;
            }
            current.push_back(c);
            continue;
        }
        if (c == ',' && depth == 0) {
            const std::string token = trim(current);
            if (!token.empty()) {
                out.push_back(token);
            }
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    const std::string tail = trim(current);
    if (!tail.empty()) {
        out.push_back(tail);
    }
    return out;
}

std::optional<std::int64_t> parse_signed_immediate(std::string token) {
    token = trim(token);
    if (token.empty()) {
        return std::nullopt;
    }
    // Clang PTX retains C/C++ integer literal suffixes on some immediates
    // (for example prmt selectors such as 0x3340U). They do not change the
    // encoded integer value at this lowering boundary.
    while (!token.empty() &&
           (token.back() == 'u' || token.back() == 'U' ||
            token.back() == 'l' || token.back() == 'L')) {
        token.pop_back();
    }
    if (token.empty()) {
        return std::nullopt;
    }
    try {
        std::size_t idx = 0;
        long long value = 0;
        if (token.size() > 2 && token[0] == '0' && (token[1] == 'x' || token[1] == 'X')) {
            value = std::stoll(token, &idx, 16);
        } else {
            value = std::stoll(token, &idx, 10);
        }
        if (idx != token.size()) {
            return std::nullopt;
        }
        return static_cast<std::int64_t>(value);
    } catch (...) {
        return std::nullopt;
    }
}

struct ParsedConstB8Array {
    std::string symbol;
    std::vector<std::uint8_t> bytes;
    int align = 1;
};

struct ParsedConstU64Array {
    std::string symbol;
    std::vector<std::uint64_t> values;
    int align = 8;
};

struct ConstSymbolInfo {
    std::string llvm_global_name;
    std::string llvm_param_name;
    std::size_t byte_offset = 0;
    std::size_t byte_count = 0;
};

struct GlobalSymbolInfo {
    std::string llvm_param_name;
    std::size_t byte_count = 0;
};

struct SharedSymbolInfo {
    std::size_t offset_bytes = 0;  // byte offset within the threadgroup buffer
    std::size_t size_bytes   = 0;  // size of this symbol in bytes
};

std::string quote_llvm_global_name(const std::string& symbol) {
    std::string out;
    out.reserve(symbol.size() + 4);
    out.push_back('@');
    out.push_back('"');
    for (const char c : symbol) {
        if (c == '"' || c == '\\') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    out.push_back('"');
    return out;
}

std::vector<ParsedConstB8Array> parse_ptx_initialized_b8_arrays(std::string_view ptx_text) {
    std::vector<ParsedConstB8Array> out;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        const bool read_only_initializer =
            t.find(".const") != std::string::npos || t.find(".global") != std::string::npos;
        if (!read_only_initializer || t.find(".b8") == std::string::npos ||
            t.find('{') == std::string::npos || t.find('}') == std::string::npos) {
            continue;
        }

        int align = 1;
        if (const std::size_t align_pos = t.find(".align"); align_pos != std::string::npos) {
            std::size_t pos = align_pos + 6;
            while (pos < t.size() && std::isspace(static_cast<unsigned char>(t[pos])) != 0) ++pos;
            std::size_t end = pos;
            while (end < t.size() && std::isdigit(static_cast<unsigned char>(t[end])) != 0) ++end;
            if (end > pos) {
                try {
                    align = std::max(1, std::stoi(t.substr(pos, end - pos)));
                } catch (...) {
                    align = 1;
                }
            }
        }

        const std::size_t b8_pos = t.find(".b8");
        if (b8_pos == std::string::npos) {
            continue;
        }
        std::size_t sym_begin = b8_pos + 3;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        const std::size_t bracket_close =
            (bracket_open == std::string::npos) ? std::string::npos : t.find(']', bracket_open + 1);
        if (bracket_open == std::string::npos || bracket_close == std::string::npos) {
            continue;
        }
        const std::string symbol = trim(t.substr(sym_begin, bracket_open - sym_begin));
        if (symbol.empty()) {
            continue;
        }

        std::size_t declared_count = 0;
        try {
            declared_count = static_cast<std::size_t>(
                std::stoull(trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1))));
        } catch (...) {
            continue;
        }

        const std::size_t brace_open = t.find('{', bracket_close);
        const std::size_t brace_close = t.find('}', brace_open == std::string::npos ? 0 : brace_open + 1);
        if (brace_open == std::string::npos || brace_close == std::string::npos || brace_close <= brace_open) {
            continue;
        }

        const std::string init_text = t.substr(brace_open + 1, brace_close - brace_open - 1);
        const std::vector<std::string> items = split_comma_list(init_text);
        std::vector<std::uint8_t> bytes;
        bytes.reserve(std::max(declared_count, items.size()));
        for (const std::string& item : items) {
            if (const auto v = parse_signed_immediate(item)) {
                bytes.push_back(static_cast<std::uint8_t>(*v & 0xff));
            }
        }
        if (bytes.size() < declared_count) {
            bytes.resize(declared_count, 0);
        } else if (declared_count > 0 && bytes.size() > declared_count) {
            bytes.resize(declared_count);
        }
        if (bytes.empty()) {
            continue;
        }

        out.push_back(ParsedConstB8Array{
            .symbol = symbol,
            .bytes = std::move(bytes),
            .align = align,
        });
    }

    return out;
}

std::uint64_t stable_device_function_token(std::string_view symbol) {
    // Device function pointers cannot be represented by public Metal/AIR. Keep
    // PTX vtable identity instead: indirect-call lowering dispatches these stable
    // nonzero tokens to direct, inlined device-function bodies. Restrict the hash
    // to signed i64 range so textual LLVM constants are portable across parsers.
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char c : symbol) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    hash &= 0x7fffffffffffffffull;
    return hash == 0 ? 1 : hash;
}

std::vector<ParsedConstU64Array> parse_ptx_initialized_u64_arrays(
    std::string_view ptx_text) {
    std::vector<ParsedConstU64Array> out;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if ((t.find(".const") == std::string::npos &&
             t.find(".global") == std::string::npos) ||
            t.find(".u64") == std::string::npos ||
            t.find('{') == std::string::npos || t.find('}') == std::string::npos) {
            continue;
        }
        int align = 8;
        if (const std::size_t align_pos = t.find(".align");
            align_pos != std::string::npos) {
            std::size_t pos = align_pos + 6;
            while (pos < t.size() &&
                   std::isspace(static_cast<unsigned char>(t[pos])) != 0) ++pos;
            std::size_t end = pos;
            while (end < t.size() &&
                   std::isdigit(static_cast<unsigned char>(t[end])) != 0) ++end;
            if (end > pos) {
                try { align = std::max(1, std::stoi(t.substr(pos, end - pos))); }
                catch (...) { align = 8; }
            }
        }
        const std::size_t u64_pos = t.find(".u64");
        std::size_t sym_begin = u64_pos + 4;
        while (sym_begin < t.size() &&
               std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        const std::size_t bracket_close =
            bracket_open == std::string::npos ? std::string::npos
                                              : t.find(']', bracket_open + 1);
        if (bracket_open == std::string::npos || bracket_close == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, bracket_open - sym_begin));
        std::size_t declared_count = 0;
        try {
            declared_count = static_cast<std::size_t>(std::stoull(
                trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1))));
        } catch (...) { continue; }
        const std::size_t brace_open = t.find('{', bracket_close);
        const std::size_t brace_close =
            brace_open == std::string::npos ? std::string::npos
                                            : t.find('}', brace_open + 1);
        if (symbol.empty() || brace_open == std::string::npos ||
            brace_close == std::string::npos) continue;
        std::vector<std::uint64_t> values;
        for (const std::string& raw_item : split_comma_list(
                 t.substr(brace_open + 1, brace_close - brace_open - 1))) {
            const std::string item = trim(raw_item);
            if (const auto immediate = parse_signed_immediate(item)) {
                values.push_back(static_cast<std::uint64_t>(*immediate));
            } else if (!item.empty()) {
                values.push_back(stable_device_function_token(item));
            }
        }
        values.resize(declared_count, 0);
        if (!values.empty()) {
            out.push_back({.symbol = symbol, .values = std::move(values), .align = align});
        }
    }
    return out;
}

// Extract the body text (between the outermost braces) of a named PTX callable.
// Device functions need this as well as kernels: clang gives each callable its
// own `.local` depot, and using only the entry's declarations leaves helper
// stack frames without a known size.
std::string extract_callable_body(std::string_view ptx_text,
                                  std::string_view callable_name) {
    if (callable_name.empty()) {
        return {};
    }
    std::size_t search_from = 0;
    while (true) {
        const std::size_t entry_pos = ptx_text.find(".entry", search_from);
        const std::size_t function_pos = ptx_text.find(".func", search_from);
        const std::size_t callable_pos =
            entry_pos == std::string_view::npos ? function_pos
            : function_pos == std::string_view::npos ? entry_pos
            : std::min(entry_pos, function_pos);
        if (callable_pos == std::string_view::npos) break;

        const std::size_t body_begin = ptx_text.find('{', callable_pos);
        const std::size_t declaration_end = ptx_text.find(';', callable_pos);
        if (body_begin == std::string_view::npos ||
            (declaration_end != std::string_view::npos && declaration_end < body_begin)) {
            search_from = callable_pos + 5;
            continue;
        }
        const std::string_view header =
            ptx_text.substr(callable_pos, body_begin - callable_pos);
        std::size_t name_pos = header.find(callable_name);
        bool exact_name = false;
        while (name_pos != std::string_view::npos) {
            const auto is_name_char = [](char c) {
                return std::isalnum(static_cast<unsigned char>(c)) != 0 ||
                       c == '_' || c == '$' || c == '.';
            };
            const bool left_ok = name_pos == 0 || !is_name_char(header[name_pos - 1]);
            const std::size_t after = name_pos + callable_name.size();
            const bool right_ok = after == header.size() || !is_name_char(header[after]);
            if (left_ok && right_ok) {
                exact_name = true;
                break;
            }
            name_pos = header.find(callable_name, name_pos + 1);
        }
        if (!exact_name) {
            search_from = body_begin + 1;
            continue;
        }
        std::size_t depth = 1;
        std::size_t body_end = body_begin + 1;
        for (; body_end < ptx_text.size() && depth != 0; ++body_end) {
            if (ptx_text[body_end] == '{') ++depth;
            else if (ptx_text[body_end] == '}') --depth;
        }
        if (depth == 0) {
            return std::string(ptx_text.substr(body_begin + 1, body_end - body_begin - 2));
        }
        break;
    }
    return {};
}

// Retain the entry-specific spelling at call sites that intentionally operate
// only on the selected kernel body (call discovery and entry metadata).
std::string extract_entry_body(std::string_view ptx_text,
                               std::string_view entry_name) {
    return extract_callable_body(ptx_text, entry_name);
}

struct LocalDepotInfo {
    std::size_t size_bytes = 0;
    std::size_t align_bytes = 16;
};

// Parse `.local` stack-depot declarations from the selected entry's body.
//
// Clang emits one depot per kernel (e.g. `.local .align 4 .b8 __local_depot0[288];`)
// holding every address-taken local object, addressed through %SPL. The depot size
// must come from the declaration: allocating a fixed guess silently truncates the
// frame, and every slot past the end reads as zero rather than faulting. That turns
// a register-tiled kernel into one that quietly computes zeros.
std::unordered_map<std::string, LocalDepotInfo> parse_ptx_local_depots(
    std::string_view ptx_text,
    std::string_view callable_name) {
    std::unordered_map<std::string, LocalDepotInfo> out;
    const std::string body = extract_callable_body(ptx_text, callable_name);
    if (body.empty()) {
        return out;
    }

    std::istringstream lines{body};
    std::string line;
    std::string declaration;
    while (std::getline(lines, line)) {
        std::string t = trim(line);
        if (const std::size_t comment = t.find("//"); comment != std::string::npos) {
            t = trim(t.substr(0, comment));
        }
        if (declaration.empty()) {
            // Only declarations start with `.local`; `ld.local`/`st.local` do not.
            if (!starts_with(t, ".local")) continue;
            declaration = t;
        } else if (!t.empty()) {
            declaration += " " + t;
        }
        // LLVM's NVPTX printer may wrap the symbol and extent onto the next
        // line after the type token. Parse the complete semicolon-terminated
        // declaration instead of silently losing the stack frame.
        if (declaration.find(';') == std::string::npos) continue;
        t = std::move(declaration);
        declaration.clear();

        std::size_t type_pos = std::string::npos;
        std::size_t type_len = 0;
        std::size_t elem_bytes = 1;
        for (const auto& p : std::vector<std::pair<std::string, std::size_t>>{
                {".b64", 8}, {".u64", 8}, {".s64", 8}, {".f64", 8},
                {".b32", 4}, {".u32", 4}, {".s32", 4}, {".f32", 4},
                {".b16", 2}, {".u16", 2}, {".s16", 2}, {".f16", 2},
                {".b8", 1}, {".u8", 1}, {".s8", 1}}) {
            const auto pos = t.find(p.first);
            if (pos != std::string::npos) {
                type_pos = pos;
                type_len = p.first.size();
                elem_bytes = p.second;
                break;
            }
        }
        if (type_pos == std::string::npos) continue;

        std::size_t align_bytes = 16;
        if (const std::size_t ap = t.find(".align"); ap != std::string::npos) {
            std::size_t pos = ap + 6;
            while (pos < t.size() && std::isspace(static_cast<unsigned char>(t[pos])) != 0) ++pos;
            std::size_t end = pos;
            while (end < t.size() && std::isdigit(static_cast<unsigned char>(t[end])) != 0) ++end;
            if (end > pos) {
                try { align_bytes = static_cast<std::size_t>(std::max(1, std::stoi(t.substr(pos, end - pos)))); }
                catch (...) {}
            }
        }

        std::size_t sym_begin = type_pos + type_len;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        const std::size_t symbol_end = bracket_open != std::string::npos
                                           ? bracket_open
                                           : t.find(';', sym_begin);
        if (symbol_end == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, symbol_end - sym_begin));
        if (symbol.empty()) continue;

        std::size_t elem_count = bracket_open == std::string::npos ? 1 : 0;
        const std::size_t bracket_close = bracket_open == std::string::npos
                                              ? std::string::npos
                                              : t.find(']', bracket_open + 1);
        if (bracket_open != std::string::npos && bracket_close != std::string::npos) {
            const std::string cnt = trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1));
            if (!cnt.empty()) {
                try { elem_count = static_cast<std::size_t>(std::stoull(cnt)); } catch (...) {}
            }
        }
        const std::size_t size_bytes = elem_count * elem_bytes;
        if (size_bytes == 0) continue;
        out[symbol] = LocalDepotInfo{.size_bytes = size_bytes, .align_bytes = align_bytes};
    }
    return out;
}

// Parse .shared declarations from PTX to build a symbol→byte-offset map.
// Each static __shared__ variable gets a contiguous region in the threadgroup buffer.
// Explicitly retain .extern .shared symbols with size zero at offset zero so address
// resolution can distinguish dynamic shared memory from unrelated module symbols.
std::unordered_map<std::string, SharedSymbolInfo> parse_ptx_shared_symbols(
    std::string_view ptx_text,
    std::string_view entry_name) {
    const std::string selected_entry_body = extract_entry_body(ptx_text, entry_name);

    struct Entry { std::size_t size_bytes; std::size_t align_bytes; };
    std::vector<std::pair<std::string, Entry>> ordered;
    std::vector<std::string> extern_symbols;
    std::unordered_set<std::string> seen;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if (t.find(".shared") == std::string::npos) continue;
        const bool is_extern = t.find(".extern") != std::string::npos;

        // Determine element byte-size from the PTX declaration type. Clang
        // commonly emits byte arrays as .b8 but preserves scalar source types
        // such as .u32 for function-local __shared__ objects.
        std::size_t type_pos = std::string::npos;
        std::size_t type_len = 0;
        int elem_bytes = 1;
        for (const auto& p : std::vector<std::pair<std::string, int>>{
                {".b64", 8}, {".u64", 8}, {".s64", 8}, {".f64", 8},
                {".b32", 4}, {".u32", 4}, {".s32", 4}, {".f32", 4},
                {".b16", 2}, {".u16", 2}, {".s16", 2}, {".f16", 2},
                {".b8", 1}, {".u8", 1}, {".s8", 1}, {".pred", 1}}) {
            const auto pos = t.find(p.first);
            if (pos != std::string::npos) {
                type_pos = pos;
                type_len = p.first.size();
                elem_bytes = p.second;
                break;
            }
        }
        if (type_pos == std::string::npos) continue;

        // Parse .align N
        std::size_t align_bytes = static_cast<std::size_t>(elem_bytes);
        if (const std::size_t ap = t.find(".align"); ap != std::string::npos) {
            std::size_t pos = ap + 6;
            while (pos < t.size() && std::isspace(static_cast<unsigned char>(t[pos])) != 0) ++pos;
            std::size_t end = pos;
            while (end < t.size() && std::isdigit(static_cast<unsigned char>(t[end])) != 0) ++end;
            if (end > pos) {
                try { align_bytes = static_cast<std::size_t>(std::max(1, std::stoi(t.substr(pos, end - pos)))); }
                catch (...) {}
            }
        }

        // Symbol name: from the end of the type token to '[' for arrays or ';'
        // for scalar shared objects.
        std::size_t sym_begin = type_pos + type_len;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        const std::size_t symbol_end = bracket_open != std::string::npos
                                           ? bracket_open
                                           : t.find(';', sym_begin);
        if (symbol_end == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, symbol_end - sym_begin));
        if (symbol.empty() || seen.count(symbol) != 0) continue;
        // Static shared objects are module-scoped in Clang PTX even when they
        // belong to different kernels. Layout only the objects used by the
        // selected entry, matching compute_static_shared_bytes(). Otherwise a
        // selected kernel's first object can be placed beyond the threadgroup
        // allocation reserved for that kernel.
        if (!entry_name.empty() && selected_entry_body.find(symbol) == std::string::npos) {
            continue;
        }
        seen.insert(symbol);

        // All extern shared declarations alias the launch-provided dynamic
        // threadgroup allocation. Pointer arithmetic in the kernel selects
        // sub-regions, so their symbolic base is exactly byte offset zero.
        if (is_extern) {
            extern_symbols.push_back(symbol);
            continue;
        }

        // Element count inside brackets.
        std::size_t elem_count = bracket_open == std::string::npos ? 1 : 0;
        const std::size_t bracket_close = bracket_open == std::string::npos
                                              ? std::string::npos
                                              : t.find(']', bracket_open + 1);
        if (bracket_open != std::string::npos && bracket_close != std::string::npos) {
            const std::string cnt = trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1));
            if (!cnt.empty()) {
                try { elem_count = static_cast<std::size_t>(std::stoull(cnt)); } catch (...) {}
            }
        }
        const std::size_t size_bytes = elem_count * static_cast<std::size_t>(elem_bytes);
        if (size_bytes == 0) continue;
        ordered.push_back({symbol, Entry{size_bytes, align_bytes}});
    }

    // Assign contiguous byte offsets, respecting alignment.
    std::unordered_map<std::string, SharedSymbolInfo> out;
    std::size_t cursor = 0;
    for (const auto& [sym, e] : ordered) {
        if (e.align_bytes > 1) cursor = (cursor + e.align_bytes - 1) & ~(e.align_bytes - 1);
        out.emplace(sym, SharedSymbolInfo{.offset_bytes = cursor, .size_bytes = e.size_bytes});
        cursor += e.size_bytes;
    }
    for (const std::string& sym : extern_symbols) {
        out.emplace(sym, SharedSymbolInfo{.offset_bytes = 0, .size_bytes = 0});
    }
    return out;
}

struct ParsedMemOperand {
    bool ok = false;
    std::string base;
    std::int64_t offset = 0;
};

ParsedMemOperand parse_memory_operand(std::string operand) {
    ParsedMemOperand out;
    operand = trim(operand);
    if (operand.size() < 2 || operand.front() != '[' || operand.back() != ']') {
        return out;
    }
    std::string inner = trim(operand.substr(1, operand.size() - 2));
    if (inner.empty()) {
        return out;
    }

    std::size_t split = std::string::npos;
    for (std::size_t i = 1; i < inner.size(); ++i) {
        if (inner[i] == '+' || inner[i] == '-') {
            split = i;
            break;
        }
    }

    if (split == std::string::npos) {
        out.ok = true;
        out.base = trim(inner);
        out.offset = 0;
        return out;
    }

    out.base = trim(inner.substr(0, split));
    const std::string off = trim(inner.substr(split));
    std::string normalized_off = off;
    if (starts_with(normalized_off, "+-")) {
        normalized_off = normalized_off.substr(1);
    } else if (starts_with(normalized_off, "--")) {
        normalized_off = normalized_off.substr(1);
    }
    const auto parsed_off = parse_signed_immediate(normalized_off);
    if (out.base.empty() || !parsed_off.has_value()) {
        return ParsedMemOperand{};
    }
    out.ok = true;
    out.offset = *parsed_off;
    return out;
}

bool is_register_name(std::string_view token) {
    return !token.empty() && token.front() == '%';
}

// is_register_name() accepts any '%'-prefixed token, so an unhandled PTX special
// register looks exactly like an undeclared virtual register: the generic path
// mints a fresh slot and reads zero, which is a silent wrong answer rather than
// a diagnostic. `%activemask` did that for a long time. Anything named here that
// emit_special_register_value() does not lower must refuse to lower instead.
bool is_ptx_special_register(std::string_view token) {
    static const std::unordered_set<std::string_view> kSpecial = {
        "%tid", "%ntid", "%ctaid", "%nctaid", "%gridid", "%laneid", "%warpid",
        "%nwarpid", "%warpsize", "%activemask", "%smid", "%nsmid",
        "%lanemask_eq", "%lanemask_le", "%lanemask_lt", "%lanemask_ge",
        "%lanemask_gt", "%clock", "%clock_hi", "%clock64", "%globaltimer",
        "%globaltimer_lo", "%globaltimer_hi", "%total_smem_size",
        "%dynamic_smem_size", "%reserved_smem_offset_begin",
        "%reserved_smem_offset_end", "%reserved_smem_offset_cap",
        "%current_graph_exec", "%is_explicit_cluster", "%nclusterid",
        "%clusterid", "%cluster_ctaid", "%cluster_nctaid", "%cluster_ctarank",
        "%cluster_nctarank",
    };
    if (token.empty() || token.front() != '%') {
        return false;
    }
    // Strip a dimensional suffix (%tid.x -> %tid) before matching.
    const std::size_t dot = token.find('.');
    const std::string_view base = dot == std::string_view::npos ? token : token.substr(0, dot);
    if (kSpecial.count(base) != 0) {
        return true;
    }
    return kSpecial.count(token) != 0;
}

std::string extract_register_name(std::string_view text) {
    const std::size_t percent = text.find('%');
    if (percent == std::string::npos) {
        return {};
    }
    std::size_t end = percent + 1;
    while (end < text.size()) {
        const unsigned char c = static_cast<unsigned char>(text[end]);
        if (std::isalnum(c) != 0 || c == '_' || c == '.' || c == '$') {
            ++end;
            continue;
        }
        break;
    }
    if (end <= percent + 1) {
        return {};
    }
    return std::string(text.substr(percent, end - percent));
}

int register_bit_width_from_name(const std::string& reg) {
    if (reg.empty()) {
        return 0;
    }
    if (reg[0] == '%') {
        if (reg.rfind("%p", 0) == 0) return 1;
        if (reg.rfind("%rd", 0) == 0) return 64;
        if (reg.rfind("%fd", 0) == 0) return 64;
        if (reg.rfind("%rs", 0) == 0) return 16;
        if (reg.rfind("%r", 0) == 0) return 32;
        if (reg.rfind("%f", 0) == 0) return 32;
    }
    return 0;
}

std::string sanitize_block_name(std::string name) {
    if (name.empty()) {
        return "lbl";
    }
    for (char& c : name) {
        const bool ok = std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_' || c == '.';
        if (!ok) {
            c = '_';
        }
    }
    if (std::isdigit(static_cast<unsigned char>(name.front())) != 0) {
        name.insert(name.begin(), '_');
    }
    return name;
}

std::vector<std::string> split_opcode_tokens(std::string_view opcode) {
    std::vector<std::string> out;
    std::string current;
    for (char c : opcode) {
        if (c == '.') {
            if (!current.empty()) {
                out.push_back(current);
                current.clear();
            }
            continue;
        }
        current.push_back(c);
    }
    if (!current.empty()) {
        out.push_back(current);
    }
    return out;
}

struct PtxTypeSpec {
    enum class Kind {
        kInvalid,
        kPred,
        kInt,
        kFloat,
    };
    Kind kind = Kind::kInvalid;
    int bits = 0;
    bool is_signed = false;
};

PtxTypeSpec parse_type_suffix_token(const std::string& token) {
    PtxTypeSpec spec;
    if (token == "pred") {
        spec.kind = PtxTypeSpec::Kind::kPred;
        spec.bits = 1;
        return spec;
    }
    if (token.size() < 2) {
        return spec;
    }
    const char family = token[0];
    int bits = 0;
    try {
        bits = std::stoi(token.substr(1));
    } catch (...) {
        return spec;
    }
    if (family == 'f') {
        spec.kind = PtxTypeSpec::Kind::kFloat;
        spec.bits = bits;
        return spec;
    }
    if (family == 's' || family == 'u' || family == 'b') {
        spec.kind = PtxTypeSpec::Kind::kInt;
        spec.bits = bits;
        spec.is_signed = (family == 's');
        return spec;
    }
    return spec;
}

PtxTypeSpec parse_primary_type_from_opcode(const std::string& opcode) {
    const std::vector<std::string> toks = split_opcode_tokens(opcode);
    for (auto it = toks.rbegin(); it != toks.rend(); ++it) {
        const PtxTypeSpec spec = parse_type_suffix_token(*it);
        if (spec.kind != PtxTypeSpec::Kind::kInvalid) {
            return spec;
        }
    }
    return {};
}

struct ParsedCvtTypes {
    bool ok = false;
    PtxTypeSpec dst;
    PtxTypeSpec src;
};

ParsedCvtTypes parse_cvt_types(const std::string& opcode) {
    ParsedCvtTypes out;
    const std::vector<std::string> toks = split_opcode_tokens(opcode);
    std::vector<PtxTypeSpec> specs;
    for (const std::string& t : toks) {
        PtxTypeSpec spec = parse_type_suffix_token(t);
        if (spec.kind != PtxTypeSpec::Kind::kInvalid) {
            specs.push_back(spec);
        }
    }
    if (specs.size() >= 2) {
        out.ok = true;
        out.dst = specs[specs.size() - 2];
        out.src = specs[specs.size() - 1];
    }
    return out;
}

bool opcode_uses_float_math(const std::string& opcode) {
    return parse_primary_type_from_opcode(opcode).kind == PtxTypeSpec::Kind::kFloat;
}

class GenericLlvmEmitter {
  public:
    GenericLlvmEmitter(const cumetal::ptx::EntryFunction& entry,
                      std::vector<ParamInfo>* params,
                      std::vector<std::string>* arg_decls,
                      const std::unordered_map<std::string, GlobalSymbolInfo>* global_symbols,
                      const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols,
                      const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols,
                      const std::unordered_map<std::string, LocalDepotInfo>* local_depots,
                      const std::vector<cumetal::passes::PrintfLoweredCall>* printf_calls,
                      cumetal::ptx::Fp64Mode fp64_mode = cumetal::ptx::Fp64Mode::kNative,
                      const std::vector<cumetal::ptx::EntryFunction>* device_functions = nullptr,
                      bool kernel_mode = true,
                      std::string return_param_name = {},
                      int return_bits = 0,
                      bool module_uses_device_heap = false,
                      bool module_uses_device_printf = false,
                      bool module_uses_device_launch_queue = false,
                      bool module_uses_device_clock = false,
                      bool module_uses_grid_barrier = false,
                      bool module_uses_grid_y_offset = false,
                      const std::vector<std::string>* device_kernel_names = nullptr,
                      const std::vector<int>* parameter_address_spaces = nullptr)
        : entry_(entry), params_(params), arg_decls_(arg_decls), global_symbols_(global_symbols),
          const_symbols_(const_symbols),
          shared_symbols_(shared_symbols), local_depots_(local_depots), fp64_mode_(fp64_mode),
          device_functions_(device_functions), kernel_mode_(kernel_mode),
          return_param_name_(std::move(return_param_name)), return_bits_(return_bits),
          module_uses_device_heap_(module_uses_device_heap),
          module_uses_device_printf_(module_uses_device_printf),
          module_uses_device_launch_queue_(module_uses_device_launch_queue),
          module_uses_device_clock_(module_uses_device_clock),
          module_uses_grid_barrier_(module_uses_grid_barrier),
          module_uses_grid_y_offset_(module_uses_grid_y_offset),
          device_kernel_names_(device_kernel_names) {
        if (params_ != nullptr) {
            for (std::size_t i = 0; i < params_->size(); ++i) {
                param_by_raw_[(*params_)[i].raw_name] = static_cast<int>(i);
                if (const std::size_t open = (*params_)[i].raw_name.find('[');
                    open != std::string::npos) {
                    const std::string base = (*params_)[i].raw_name.substr(0, open);
                    if (!base.empty() && !param_by_raw_.count(base)) {
                        param_by_raw_[base] = static_cast<int>(i);
                    }
                }
                if (parameter_address_spaces != nullptr &&
                    i < parameter_address_spaces->size()) {
                    const int encoded = (*parameter_address_spaces)[i];
                    if (encoded >= static_cast<int>(PointerAs::kUnknown) &&
                        encoded <= static_cast<int>(PointerAs::kLocal)) {
                        param_pointer_as_by_raw_[(*params_)[i].raw_name] =
                            static_cast<PointerAs>(encoded);
                    }
                }
            }
        }
        if (printf_calls != nullptr) {
            for (const auto& call : *printf_calls) {
                printf_call_by_line_[call.source_line] = &call;
            }
        }
    }

    GenericLlvmBodyResult run() {
        GenericLlvmBodyResult result;
        if (params_ == nullptr || arg_decls_ == nullptr) {
            result.error = "internal error: missing param vectors";
            return result;
        }
        if (kernel_mode_ && !append_required_builtin_params()) {
            result.error = error_;
            return result;
        }
        // Scalar params (i64, i32, float, etc.) must be constant-buffer pointers in Metal AIR.
        // Metal passes them via setBytes:length:atIndex:, which creates a small constant buffer
        // and gives the kernel a pointer to it. The function parameter type must be T addrspace(2)*
        // so the kernel can load the actual value from the buffer.
        for (std::size_t i = 0; i < params_->size(); ++i) {
            ParamInfo& p = (*params_)[i];
            if (!kernel_mode_) continue;
            if (is_builtin_param(p)) continue;
            if (is_pointer_type(p.llvm_type)) continue;
            if (p.llvm_type == "<3 x i32>") continue;
            // Convert plain scalar type to constant-buffer pointer
            const std::string new_type = p.llvm_type + " addrspace(2)*";
            p.llvm_type = new_type;
            (*arg_decls_)[i] = new_type + " %" + p.name;
        }
        if (!index_control_flow()) {
            result.error = error_;
            return result;
        }
        if (!emit_body()) {
            result.error = error_;
            return result;
        }

        result.ok = true;
        result.uses_atomic_lock_bank = !lock_bank_param_.empty();
        result.uses_device_heap = uses_device_heap_;
        result.uses_device_launch_queue = module_uses_device_launch_queue_;
        result.uses_device_clock = module_uses_device_clock_;
        result.body_ir = body_.str();
        result.declarations.assign(declarations_.begin(), declarations_.end());
        result.builtin_params = builtin_params_added_;
        result.warnings = warnings_;
        for (const auto& [name, spaces] : device_function_param_address_spaces_) {
            auto& encoded = result.device_function_param_address_spaces[name];
            encoded.reserve(spaces.size());
            for (const PointerAs space : spaces) encoded.push_back(static_cast<int>(space));
        }
        return result;
    }

  private:
    struct RegSlot {
        int bits = 0;
        std::string slot_name;
    };

    struct Value {
        std::string ir;
        PtxTypeSpec type;
        int bits = 0;
    };

    struct Fp64Pair {
        std::string hi;
        std::string lo;
    };

    enum class PointerAs {
        kUnknown = 0,
        kGlobal = 1,
        kParam = 2,
        kShared = 3,
        kLocal = 4,
    };

    struct LocalSymbolInfo {
        std::string alloca_name;
        std::string base_ptr_name;
        std::size_t size_bytes = 0;
        std::size_t align_bytes = 16;
    };

    const cumetal::ptx::EntryFunction& entry_;
    std::vector<ParamInfo>* params_ = nullptr;
    std::vector<std::string>* arg_decls_ = nullptr;
    const std::unordered_map<std::string, GlobalSymbolInfo>* global_symbols_ = nullptr;
    const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols_ = nullptr;
    const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols_ = nullptr;
    const std::unordered_map<std::string, LocalDepotInfo>* local_depots_ = nullptr;
    cumetal::ptx::Fp64Mode fp64_mode_ = cumetal::ptx::Fp64Mode::kNative;

    std::unordered_map<std::string, int> param_by_raw_;
    std::unordered_map<std::string, PointerAs> param_pointer_as_by_raw_;
    std::unordered_map<std::string, std::string> builtin_vector_arg_name_;
    std::unordered_map<std::string, std::string> builtin_scalar_arg_name_;
    std::vector<ParamInfo> builtin_params_added_;
    bool has_threadgroup_buffer_param_ = false;
    std::string threadgroup_buffer_arg_name_ = "__air_tg0";

    std::vector<int> exec_indices_;
    std::unordered_map<int, int> exec_pos_by_instr_index_;
    std::unordered_map<std::string, int> label_to_exec_pos_;
    std::unordered_map<std::string, std::vector<std::string>> branch_tables_;
    std::unordered_map<int, int> next_exec_pos_by_exec_pos_;

    std::unordered_map<std::string, RegSlot> reg_slots_;
    std::unordered_map<std::string, PointerAs> reg_pointer_as_;
    // Preserve the pointee address space when PTX spills generic pointers into
    // a `.local` depot and reloads one through a computed index.
    std::unordered_map<std::string, std::string> reg_local_origin_;
    std::unordered_map<std::string, PointerAs> local_pointer_payload_as_;
    std::unordered_map<std::string, LocalSymbolInfo> local_symbols_;
    std::unordered_map<std::string, std::string> call_param_slots_;
    std::unordered_map<std::string, PointerAs> call_param_pointer_as_;
    std::unordered_map<std::string, std::vector<PointerAs>>
        device_function_param_address_spaces_;
    std::unordered_map<int, const cumetal::passes::PrintfLoweredCall*> printf_call_by_line_;
    bool uses_device_heap_ = false;
    const std::vector<cumetal::ptx::EntryFunction>* device_functions_ = nullptr;
    bool kernel_mode_ = true;
    std::string return_param_name_;
    int return_bits_ = 0;
    bool module_uses_device_heap_ = false;
    bool module_uses_device_printf_ = false;
    bool module_uses_device_launch_queue_ = false;
    bool module_uses_device_clock_ = false;
    bool module_uses_grid_barrier_ = false;
    bool module_uses_grid_y_offset_ = false;
    const std::vector<std::string>* device_kernel_names_ = nullptr;

    std::unordered_set<std::string> declarations_;
    std::vector<std::string> warnings_;
    std::string error_;

    std::ostringstream entry_allocas_;
    std::ostringstream body_;
    int tmp_id_ = 0;
    int slot_id_ = 0;

    std::string next_tmp(const std::string& prefix) {
        return "%" + prefix + "_" + std::to_string(tmp_id_++);
    }

    std::string llvm_int_type(int bits) const {
        if (bits <= 1) return "i1";
        return "i" + std::to_string(bits);
    }

    std::string llvm_float_type(int bits) const {
        if (bits == 16) return "half";
        if (bits == 64) return "double";
        return "float";
    }

    std::string slot_name_for_reg(const std::string& reg) {
        std::string base = reg;
        if (!base.empty() && base.front() == '%') {
            base.erase(base.begin());
        }
        base = sanitize_llvm_identifier(base, "reg");
        return "%cm_reg_" + base + "_" + std::to_string(slot_id_++);
    }

    RegSlot& ensure_reg_slot(const std::string& reg, int bits_hint = 0) {
        auto it = reg_slots_.find(reg);
        if (it != reg_slots_.end()) {
            return it->second;
        }
        int bits = bits_hint;
        if (bits <= 0) {
            bits = register_bit_width_from_name(reg);
        }
        if (bits <= 0) {
            bits = 32;
        }
        RegSlot slot;
        slot.bits = bits;
        slot.slot_name = slot_name_for_reg(reg);
        entry_allocas_ << "  " << slot.slot_name << " = alloca " << llvm_int_type(bits)
                       << ", align " << std::max(1, bits / 8) << "\n";
        if (bits == 1) {
            entry_allocas_ << "  store i1 false, i1* " << slot.slot_name << ", align 1\n";
        } else {
            entry_allocas_ << "  store " << llvm_int_type(bits) << " 0, " << llvm_int_type(bits)
                           << "* " << slot.slot_name << ", align " << std::max(1, bits / 8) << "\n";
        }
        auto [inserted, _] = reg_slots_.emplace(reg, std::move(slot));
        return inserted->second;
    }

    std::string emit_load_reg_bits(std::ostringstream& os, const std::string& reg, int bits_hint = 0) {
        RegSlot& slot = ensure_reg_slot(reg, bits_hint);
        const std::string tmp = next_tmp("ld");
        os << "  " << tmp << " = load " << llvm_int_type(slot.bits) << ", " << llvm_int_type(slot.bits)
           << "* " << slot.slot_name << ", align " << std::max(1, slot.bits / 8) << "\n";
        return tmp;
    }

    bool emit_store_reg_bits(std::ostringstream& os,
                             const std::string& reg,
                             int bits_hint,
                             std::string value,
                             int value_bits,
                             bool sign_extend = false) {
        RegSlot& slot = ensure_reg_slot(reg, bits_hint);
        if (value_bits <= 0) {
            value_bits = slot.bits;
        }
        if (value_bits != slot.bits) {
            const std::string cast = next_tmp("cast");
            if (value_bits < slot.bits) {
                os << "  " << cast << " = " << (sign_extend ? "sext " : "zext ")
                   << llvm_int_type(value_bits) << " " << value
                   << " to " << llvm_int_type(slot.bits) << "\n";
            } else {
                os << "  " << cast << " = trunc " << llvm_int_type(value_bits) << " " << value
                   << " to " << llvm_int_type(slot.bits) << "\n";
            }
            value = cast;
        }
        os << "  store " << llvm_int_type(slot.bits) << " " << value << ", " << llvm_int_type(slot.bits)
           << "* " << slot.slot_name << ", align " << std::max(1, slot.bits / 8) << "\n";
        return true;
    }

    std::optional<Value> decode_integer_operand(std::ostringstream& os,
                                                const std::string& operand,
                                                int bits,
                                                bool is_signed) {
        if (bits <= 0) {
            bits = 32;
        }
        if (is_register_name(operand)) {
            const std::string raw = emit_load_reg_bits(os, operand);
            const int src_bits = ensure_reg_slot(operand).bits;
            std::string v = raw;
            if (src_bits < bits) {
                const std::string ext = next_tmp("ext");
                os << "  " << ext << " = " << (is_signed ? "sext " : "zext ")
                   << llvm_int_type(src_bits) << " " << raw << " to " << llvm_int_type(bits) << "\n";
                v = ext;
            } else if (src_bits > bits) {
                const std::string tr = next_tmp("tr");
                os << "  " << tr << " = trunc " << llvm_int_type(src_bits) << " " << raw << " to "
                   << llvm_int_type(bits) << "\n";
                v = tr;
            }
            Value out;
            out.ir = v;
            out.type = {.kind = PtxTypeSpec::Kind::kInt, .bits = bits, .is_signed = is_signed};
            out.bits = bits;
            return out;
        }
        if (const auto imm = parse_signed_immediate(operand)) {
            Value out;
            out.ir = std::to_string(*imm);
            out.type = {.kind = PtxTypeSpec::Kind::kInt, .bits = bits, .is_signed = is_signed};
            out.bits = bits;
            return out;
        }
        return std::nullopt;
    }

    std::optional<Value> decode_float_operand(std::ostringstream& os,
                                              const std::string& operand,
                                              int bits) {
        if (bits != 16 && bits != 32 && bits != 64) {
            return std::nullopt;
        }
        if (is_register_name(operand)) {
            const int reg_bits = ensure_reg_slot(operand).bits;
            if (reg_bits != bits) {
                return std::nullopt;
            }
            const std::string raw = emit_load_reg_bits(os, operand, bits);
            const std::string cast = next_tmp("bitcastf");
            const std::string fty = (bits == 16) ? "half" : (bits == 32 ? "float" : "double");
            os << "  " << cast << " = bitcast " << llvm_int_type(bits) << " " << raw << " to "
               << fty << "\n";
            Value out;
            out.ir = cast;
            out.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = bits, .is_signed = false};
            out.bits = bits;
            return out;
        }
        if (operand.size() == 10 && operand[0] == '0' && operand[1] == 'f' && bits == 32) {
            // Use decimal integer constant to avoid LLVM 20 "float constant invalid for type" error
            uint32_t bit_pattern = 0;
            try { bit_pattern = static_cast<uint32_t>(std::stoul(operand.substr(2), nullptr, 16)); } catch (...) {}
            const std::string int_bits = next_tmp("fimm");
            os << "  " << int_bits << " = or i32 0, " << static_cast<int32_t>(bit_pattern) << "\n";
            const std::string cast = next_tmp("fimmbc");
            os << "  " << cast << " = bitcast i32 " << int_bits << " to float\n";
            Value out;
            out.ir = cast;
            out.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 32, .is_signed = false};
            out.bits = 32;
            return out;
        }
        if (operand.size() == 18 && operand[0] == '0' && operand[1] == 'd' && bits == 64) {
            uint64_t bit_pattern64 = 0;
            try { bit_pattern64 = std::stoull(operand.substr(2), nullptr, 16); } catch (...) {}
            const std::string int_bits = next_tmp("dimm");
            os << "  " << int_bits << " = or i64 0, " << static_cast<int64_t>(bit_pattern64) << "\n";
            const std::string cast = next_tmp("dimmbc");
            os << "  " << cast << " = bitcast i64 " << int_bits << " to double\n";
            Value out;
            out.ir = cast;
            out.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 64, .is_signed = false};
            out.bits = 64;
            return out;
        }
        // Decimal float literal (e.g. "0.0", "1.0", "-2.5") — convert via bit pattern
        {
            char* end_ptr = nullptr;
            const float fval = std::strtof(operand.c_str(), &end_ptr);
            if (end_ptr != operand.c_str() && (*end_ptr == '\0' || *end_ptr == 'f' || *end_ptr == 'F')) {
                if (bits == 32) {
                    uint32_t bp = 0;
                    std::memcpy(&bp, &fval, 4);
                    const std::string int_bits = next_tmp("dfimm");
                    os << "  " << int_bits << " = or i32 0, " << static_cast<int32_t>(bp) << "\n";
                    const std::string cast = next_tmp("dfimmbc");
                    os << "  " << cast << " = bitcast i32 " << int_bits << " to float\n";
                    Value out;
                    out.ir = cast;
                    out.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 32, .is_signed = false};
                    out.bits = 32;
                    return out;
                }
                if (bits == 64) {
                    const double dval = static_cast<double>(fval);
                    uint64_t bp64 = 0;
                    std::memcpy(&bp64, &dval, 8);
                    const std::string int_bits = next_tmp("dfimm64");
                    os << "  " << int_bits << " = or i64 0, " << static_cast<int64_t>(bp64) << "\n";
                    const std::string cast = next_tmp("dfimmbc64");
                    os << "  " << cast << " = bitcast i64 " << int_bits << " to double\n";
                    Value out;
                    out.ir = cast;
                    out.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 64, .is_signed = false};
                    out.bits = 64;
                    return out;
                }
            }
        }
        return std::nullopt;
    }

    std::string emit_float_constant(std::ostringstream& os, float value,
                                    std::string_view label) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        const std::string raw = next_tmp(std::string(label) + "_bits");
        os << "  " << raw << " = or i32 0, " << static_cast<std::int32_t>(bits) << "\n";
        const std::string result = next_tmp(std::string(label));
        os << "  " << result << " = bitcast i32 " << raw << " to float\n";
        return result;
    }

    // Soft f32→f64 bit conversion (no native double). Handles 0, normals, inf/nan;
    // subnormals flush to signed zero (acceptable for emulate path).
    std::string emit_soft_f32_bits_to_f64_bits(std::ostringstream& os,
                                               const std::string& f32_bits) {
        const std::string sign = next_tmp("fp64_pack_f32_sign");
        const std::string exp_raw = next_tmp("fp64_pack_f32_exp_raw");
        const std::string exp_m = next_tmp("fp64_pack_f32_expm");
        const std::string mant_m = next_tmp("fp64_pack_f32_mantm");
        const std::string zexp = next_tmp("fp64_pack_f32_zexp");
        const std::string aexp = next_tmp("fp64_pack_f32_aexp");
        const std::string zmant = next_tmp("fp64_pack_f32_zmant");
        const std::string is_z = next_tmp("fp64_pack_f32_isz");
        const std::string de = next_tmp("fp64_pack_f32_de");
        const std::string de64 = next_tmp("fp64_pack_f32_de64");
        const std::string m64 = next_tmp("fp64_pack_f32_m64");
        const std::string s64 = next_tmp("fp64_pack_f32_s64");
        const std::string ssh = next_tmp("fp64_pack_f32_ssh");
        const std::string esh = next_tmp("fp64_pack_f32_esh");
        const std::string msh = next_tmp("fp64_pack_f32_msh");
        const std::string se_or = next_tmp("fp64_pack_f32_se");
        const std::string norm = next_tmp("fp64_pack_f32_norm");
        const std::string pay = next_tmp("fp64_pack_f32_pay");
        const std::string paysh = next_tmp("fp64_pack_f32_paysh");
        const std::string inf_exp = next_tmp("fp64_pack_f32_infexp");
        const std::string spec = next_tmp("fp64_pack_f32_spec");
        const std::string zbits = next_tmp("fp64_pack_f32_zb");
        const std::string pick0 = next_tmp("fp64_pack_f32_p0");
        const std::string not_zmant = next_tmp("fp64_pack_f32_nzm");
        const std::string is_sub = next_tmp("fp64_pack_f32_sub");
        const std::string pick1 = next_tmp("fp64_pack_f32_p1");
        const std::string out = next_tmp("fp64_pack");

        os << "  " << sign << " = lshr i32 " << f32_bits << ", 31\n";
        os << "  " << exp_raw << " = lshr i32 " << f32_bits << ", 23\n";
        os << "  " << exp_m << " = and i32 " << exp_raw << ", 255\n";
        os << "  " << mant_m << " = and i32 " << f32_bits << ", 8388607\n";
        os << "  " << zexp << " = icmp eq i32 " << exp_m << ", 0\n";
        os << "  " << aexp << " = icmp eq i32 " << exp_m << ", 255\n";
        os << "  " << zmant << " = icmp eq i32 " << mant_m << ", 0\n";
        os << "  " << is_z << " = and i1 " << zexp << ", " << zmant << "\n";
        os << "  " << de << " = add i32 " << exp_m << ", 896\n"; // 1023-127
        os << "  " << de64 << " = zext i32 " << de << " to i64\n";
        os << "  " << m64 << " = zext i32 " << mant_m << " to i64\n";
        os << "  " << s64 << " = zext i32 " << sign << " to i64\n";
        os << "  " << ssh << " = shl i64 " << s64 << ", 63\n";
        os << "  " << esh << " = shl i64 " << de64 << ", 52\n";
        os << "  " << msh << " = shl i64 " << m64 << ", 29\n";
        os << "  " << se_or << " = or i64 " << ssh << ", " << esh << "\n";
        os << "  " << norm << " = or i64 " << se_or << ", " << msh << "\n";
        os << "  " << pay << " = zext i32 " << mant_m << " to i64\n";
        os << "  " << paysh << " = shl i64 " << pay << ", 29\n";
        os << "  " << inf_exp << " = or i64 " << ssh << ", 9218868437227405312\n"; // 0x7ff<<52
        os << "  " << spec << " = or i64 " << inf_exp << ", " << paysh << "\n";
        os << "  " << zbits << " = or i64 " << ssh << ", 0\n";
        os << "  " << pick0 << " = select i1 " << aexp << ", i64 " << spec << ", i64 " << norm
           << "\n";
        os << "  " << not_zmant << " = xor i1 " << zmant << ", true\n";
        os << "  " << is_sub << " = and i1 " << zexp << ", " << not_zmant << "\n";
        os << "  " << pick1 << " = select i1 " << is_z << ", i64 " << zbits << ", i64 " << pick0
           << "\n";
        os << "  " << out << " = select i1 " << is_sub << ", i64 " << zbits << ", i64 " << pick1
           << "\n";
        return out;
    }

    // Soft f64→f32 bit conversion (no native double). Truncates mantissa; overflow → inf;
    // subnormals/underflow → signed zero.
    std::string emit_soft_f64_bits_to_f32_bits(std::ostringstream& os,
                                               const std::string& f64_bits) {
        const std::string sign = next_tmp("fp64_hi_sign");
        const std::string exp_raw = next_tmp("fp64_hi_exp_raw");
        const std::string expm = next_tmp("fp64_hi_expm");
        const std::string mant64 = next_tmp("fp64_hi_mant64");
        const std::string sign32 = next_tmp("fp64_hi_sign32");
        const std::string ssh = next_tmp("fp64_hi_ssh");
        const std::string zexp = next_tmp("fp64_hi_zexp");
        const std::string aexp = next_tmp("fp64_hi_aexp");
        const std::string de64 = next_tmp("fp64_hi_de64");
        const std::string under = next_tmp("fp64_hi_under");
        const std::string over = next_tmp("fp64_hi_over");
        const std::string mant_sh = next_tmp("fp64_hi_mantsh");
        const std::string mant32 = next_tmp("fp64_hi_mant32");
        const std::string de32 = next_tmp("fp64_hi_de32");
        const std::string exp_sh = next_tmp("fp64_hi_expsh");
        const std::string se = next_tmp("fp64_hi_se");
        const std::string norm = next_tmp("fp64_hi_norm");
        const std::string inf = next_tmp("fp64_hi_inf");
        const std::string nanbits = next_tmp("fp64_hi_nan");
        const std::string zmant = next_tmp("fp64_hi_zmant");
        const std::string use_nan = next_tmp("fp64_hi_usenan");
        const std::string special = next_tmp("fp64_hi_special");
        const std::string pick0 = next_tmp("fp64_hi_p0");
        const std::string signed_zero = next_tmp("fp64_hi_sz");
        const std::string pick1 = next_tmp("fp64_hi_p1");
        const std::string pick2 = next_tmp("fp64_hi_p2");
        const std::string out = next_tmp("fp64_hi_bits");

        os << "  " << sign << " = lshr i64 " << f64_bits << ", 63\n";
        os << "  " << exp_raw << " = lshr i64 " << f64_bits << ", 52\n";
        os << "  " << expm << " = and i64 " << exp_raw << ", 2047\n";
        os << "  " << mant64 << " = and i64 " << f64_bits << ", 4503599627370495\n"; // 52 bits
        os << "  " << sign32 << " = trunc i64 " << sign << " to i32\n";
        os << "  " << ssh << " = shl i32 " << sign32 << ", 31\n";
        os << "  " << zexp << " = icmp eq i64 " << expm << ", 0\n";
        os << "  " << aexp << " = icmp eq i64 " << expm << ", 2047\n";
        os << "  " << de64 << " = sub i64 " << expm << ", 896\n"; // exp - (1023-127)
        const std::string under_raw = next_tmp("fp64_hi_under_raw");
        const std::string over_raw = next_tmp("fp64_hi_over_raw");
        os << "  " << under_raw << " = icmp slt i64 " << de64 << ", 1\n";
        os << "  " << over_raw << " = icmp sgt i64 " << de64 << ", 254\n";
        // under/over apply only to finite normals (not zero/subnormal/inf/nan).
        const std::string not_zexp = next_tmp("fp64_hi_nzexp");
        const std::string not_aexp = next_tmp("fp64_hi_naexp");
        const std::string finite = next_tmp("fp64_hi_finite");
        os << "  " << not_zexp << " = xor i1 " << zexp << ", true\n";
        os << "  " << not_aexp << " = xor i1 " << aexp << ", true\n";
        os << "  " << finite << " = and i1 " << not_zexp << ", " << not_aexp << "\n";
        os << "  " << under << " = and i1 " << under_raw << ", " << finite << "\n";
        os << "  " << over << " = and i1 " << over_raw << ", " << finite << "\n";
        os << "  " << mant_sh << " = lshr i64 " << mant64 << ", 29\n";
        os << "  " << mant32 << " = trunc i64 " << mant_sh << " to i32\n";
        os << "  " << de32 << " = trunc i64 " << de64 << " to i32\n";
        os << "  " << exp_sh << " = shl i32 " << de32 << ", 23\n";
        os << "  " << se << " = or i32 " << ssh << ", " << exp_sh << "\n";
        os << "  " << norm << " = or i32 " << se << ", " << mant32 << "\n";
        os << "  " << inf << " = or i32 " << ssh << ", 2139095040\n"; // 0x7f800000
        os << "  " << nanbits << " = or i32 " << inf << ", " << mant32 << "\n";
        os << "  " << zmant << " = icmp eq i64 " << mant64 << ", 0\n";
        os << "  " << use_nan << " = xor i1 " << zmant << ", true\n";
        os << "  " << special << " = select i1 " << use_nan << ", i32 " << nanbits << ", i32 " << inf
           << "\n";
        os << "  " << pick0 << " = select i1 " << aexp << ", i32 " << special << ", i32 " << norm
           << "\n";
        os << "  " << signed_zero << " = or i32 " << ssh << ", 0\n";
        os << "  " << pick1 << " = select i1 " << under << ", i32 " << signed_zero << ", i32 "
           << pick0 << "\n";
        os << "  " << pick2 << " = select i1 " << over << ", i32 " << inf << ", i32 " << pick1
           << "\n";
        os << "  " << out << " = select i1 " << zexp << ", i32 " << signed_zero << ", i32 " << pick2
           << "\n";
        return out;
    }

    // Branch-free 32-bit leading-zero count; returns 32 for v == 0.
    // Emitted inline rather than via llvm.ctlz so the result does not depend on
    // whether the AIR backend lowers that intrinsic.
    std::string emit_clz_i32(std::ostringstream& os, const std::string& v) {
        // Each step asks whether the top `width` bits are all zero; if so the
        // value is shifted up by `width` and the count advances.
        static const struct { int width; const char* mask; } kSteps[] = {
            {16, "-65536"},       // 0xFFFF0000
            {8,  "-16777216"},    // 0xFF000000
            {4,  "-268435456"},   // 0xF0000000
            {2,  "-1073741824"},  // 0xC0000000
            {1,  "-2147483648"},  // 0x80000000
        };
        std::string t = v;
        std::string n = "0";
        for (const auto& step : kSteps) {
            const std::string masked = next_tmp("clz_mask");
            const std::string has_top = next_tmp("clz_has");
            const std::string shifted = next_tmp("clz_sh");
            const std::string t_next = next_tmp("clz_t");
            const std::string n_add = next_tmp("clz_nadd");
            const std::string n_next = next_tmp("clz_n");
            os << "  " << masked << " = and i32 " << t << ", " << step.mask << "\n";
            os << "  " << has_top << " = icmp ne i32 " << masked << ", 0\n";
            os << "  " << shifted << " = shl i32 " << t << ", " << step.width << "\n";
            os << "  " << t_next << " = select i1 " << has_top << ", i32 " << t << ", i32 "
               << shifted << "\n";
            os << "  " << n_add << " = add i32 " << n << ", " << step.width << "\n";
            os << "  " << n_next << " = select i1 " << has_top << ", i32 " << n << ", i32 "
               << n_add << "\n";
            t = t_next;
            n = n_next;
        }
        // v == 0 leaves every step taking the shift branch, giving n == 31; the
        // true count is 32.
        const std::string is_zero = next_tmp("clz_isz");
        const std::string out = next_tmp("clz");
        os << "  " << is_zero << " = icmp eq i32 " << v << ", 0\n";
        os << "  " << out << " = select i1 " << is_zero << ", i32 32, i32 " << n << "\n";
        return out;
    }

    // Split IEEE-754 binary64 bits into a Dekker FP32 pair, keeping the residual.
    //
    //   hi = roundToNearestEven_f32(x)
    //   lo = roundToNearestEven_f32(x - hi)     (computed from the binary64
    //                                            significand with integer math,
    //                                            never from an already-rounded hi)
    //
    // The pair carries 48 of binary64's 53 significand bits, so the split is
    // where precision is lost; because it is deterministic, re-splitting a value
    // this pair produced returns the same pair, and error does not compound
    // across a chain of instructions.
    //
    // Documented deviations from IEEE-754: binary64 subnormal inputs flush to
    // signed zero (inherited from emit_soft_f64_bits_to_f32_bits), and a residual
    // whose own exponent falls below the binary32 normal range is dropped rather
    // than represented as a binary32 subnormal. Signed zero, infinity and NaN
    // pass through hi unchanged with lo set to a signed zero.
    Fp64Pair emit_soft_f64_bits_to_f32_pair(std::ostringstream& os,
                                            const std::string& f64_bits) {
        const std::string sign64 = next_tmp("fp64_split_s64");
        const std::string sign = next_tmp("fp64_split_sign");
        const std::string sign_sh = next_tmp("fp64_split_signsh");
        const std::string exp_raw = next_tmp("fp64_split_eraw");
        const std::string exp = next_tmp("fp64_split_e");
        const std::string exp32 = next_tmp("fp64_split_e32");
        const std::string mant = next_tmp("fp64_split_M");
        const std::string keep = next_tmp("fp64_split_keep");
        const std::string res = next_tmp("fp64_split_R");
        os << "  " << sign64 << " = lshr i64 " << f64_bits << ", 63\n";
        os << "  " << sign << " = trunc i64 " << sign64 << " to i32\n";
        os << "  " << sign_sh << " = shl i32 " << sign << ", 31\n";
        os << "  " << exp_raw << " = lshr i64 " << f64_bits << ", 52\n";
        os << "  " << exp << " = and i64 " << exp_raw << ", 2047\n";
        os << "  " << exp32 << " = trunc i64 " << exp << " to i32\n";
        os << "  " << mant << " = and i64 " << f64_bits << ", 4503599627370495\n";
        os << "  " << keep << " = lshr i64 " << mant << ", 29\n";   // bits hi will keep
        os << "  " << res << " = and i64 " << mant << ", 536870911\n"; // bits 28..0

        // Round-to-nearest-even at bit 29.
        const std::string gt_half = next_tmp("fp64_split_gt");
        const std::string eq_half = next_tmp("fp64_split_eq");
        const std::string keep_odd_bit = next_tmp("fp64_split_kob");
        const std::string keep_odd = next_tmp("fp64_split_ko");
        const std::string tie_up = next_tmp("fp64_split_tie");
        const std::string round_up_raw = next_tmp("fp64_split_upraw");
        const std::string finite_lo = next_tmp("fp64_split_fina");
        const std::string finite_hi = next_tmp("fp64_split_finb");
        const std::string finite = next_tmp("fp64_split_fin");
        const std::string round_up = next_tmp("fp64_split_up");
        os << "  " << gt_half << " = icmp ugt i64 " << res << ", 268435456\n";
        os << "  " << eq_half << " = icmp eq i64 " << res << ", 268435456\n";
        os << "  " << keep_odd_bit << " = and i64 " << keep << ", 1\n";
        os << "  " << keep_odd << " = icmp ne i64 " << keep_odd_bit << ", 0\n";
        os << "  " << tie_up << " = and i1 " << eq_half << ", " << keep_odd << "\n";
        os << "  " << round_up_raw << " = or i1 " << gt_half << ", " << tie_up << "\n";
        // Only round finite normals: adding to a NaN significand can carry the
        // exponent field into the sign bit, and zero/subnormal inputs flush.
        os << "  " << finite_lo << " = icmp ne i64 " << exp << ", 0\n";
        os << "  " << finite_hi << " = icmp ne i64 " << exp << ", 2047\n";
        os << "  " << finite << " = and i1 " << finite_lo << ", " << finite_hi << "\n";
        os << "  " << round_up << " = and i1 " << round_up_raw << ", " << finite << "\n";

        // Adding one ulp-of-hi to the raw bit pattern carries out of the
        // significand into the exponent for free, and saturates to infinity if
        // the exponent runs out -- both exactly what rounding up should do.
        const std::string bumped = next_tmp("fp64_split_bumped");
        const std::string rounded = next_tmp("fp64_split_rounded");
        os << "  " << bumped << " = add i64 " << f64_bits << ", 536870912\n";
        os << "  " << rounded << " = select i1 " << round_up << ", i64 " << bumped << ", i64 "
           << f64_bits << "\n";
        // The low 29 bits of `rounded` are now the discarded ones, so the
        // truncating conversion below loses nothing further.
        const std::string hi_bits = emit_soft_f64_bits_to_f32_bits(os, rounded);
        const std::string hi = next_tmp("fp64_split_hi");
        os << "  " << hi << " = bitcast i32 " << hi_bits << " to float\n";

        // Signed residual x - hi, in units of 2^(e-52) and bounded by 2^28 in
        // magnitude. Rounding up makes it negative, which flips lo's sign
        // relative to x -- a Dekker pair permits that.
        const std::string res_signed = next_tmp("fp64_split_rs");
        const std::string res_neg = next_tmp("fp64_split_rneg");
        const std::string res_negated = next_tmp("fp64_split_rnegd");
        const std::string res_mag64 = next_tmp("fp64_split_rmag64");
        const std::string res_mag = next_tmp("fp64_split_rmag");
        const std::string lowered = next_tmp("fp64_split_low");
        os << "  " << lowered << " = sub i64 " << res << ", 536870912\n";
        os << "  " << res_signed << " = select i1 " << round_up << ", i64 " << lowered << ", i64 "
           << res << "\n";
        os << "  " << res_neg << " = icmp slt i64 " << res_signed << ", 0\n";
        os << "  " << res_negated << " = sub i64 0, " << res_signed << "\n";
        os << "  " << res_mag64 << " = select i1 " << res_neg << ", i64 " << res_negated
           << ", i64 " << res_signed << "\n";
        os << "  " << res_mag << " = trunc i64 " << res_mag64 << " to i32\n";

        const std::string clz = emit_clz_i32(os, res_mag);
        const std::string norm = next_tmp("fp64_split_norm");
        const std::string mant_sh = next_tmp("fp64_split_mantsh");
        const std::string lo_mant = next_tmp("fp64_split_lomant");
        os << "  " << norm << " = shl i32 " << res_mag << ", " << clz << "\n";
        os << "  " << mant_sh << " = lshr i32 " << norm << ", 8\n";
        os << "  " << lo_mant << " = and i32 " << mant_sh << ", 8388607\n";

        // Leading residual bit sits at position 31-clz of a value scaled by
        // 2^(e-52), so the binary32 biased exponent is E - 917 - clz.
        const std::string e_tmp = next_tmp("fp64_split_etmp");
        const std::string e_lo = next_tmp("fp64_split_elo");
        os << "  " << e_tmp << " = sub i32 " << exp32 << ", " << clz << "\n";
        os << "  " << e_lo << " = sub i32 " << e_tmp << ", 917\n";

        const std::string res_nz = next_tmp("fp64_split_rnz");
        const std::string e_lo_ok_lo = next_tmp("fp64_split_eloa");
        const std::string e_lo_ok_hi = next_tmp("fp64_split_elob");
        const std::string e_lo_ok = next_tmp("fp64_split_eloc");
        const std::string ok0 = next_tmp("fp64_split_ok0");
        const std::string ok = next_tmp("fp64_split_ok");
        os << "  " << res_nz << " = icmp ne i32 " << res_mag << ", 0\n";
        os << "  " << e_lo_ok_lo << " = icmp sgt i32 " << e_lo << ", 0\n";
        os << "  " << e_lo_ok_hi << " = icmp slt i32 " << e_lo << ", 255\n";
        os << "  " << e_lo_ok << " = and i1 " << e_lo_ok_lo << ", " << e_lo_ok_hi << "\n";
        os << "  " << ok0 << " = and i1 " << res_nz << ", " << finite << "\n";
        os << "  " << ok << " = and i1 " << ok0 << ", " << e_lo_ok << "\n";

        const std::string lo_sign_flip = next_tmp("fp64_split_lsf");
        const std::string lo_sign = next_tmp("fp64_split_lsign");
        const std::string e_sh = next_tmp("fp64_split_esh");
        const std::string packed0 = next_tmp("fp64_split_p0");
        const std::string packed = next_tmp("fp64_split_p1");
        const std::string lo_bits = next_tmp("fp64_split_lobits");
        const std::string lo = next_tmp("fp64_split_lo");
        os << "  " << lo_sign_flip << " = select i1 " << res_neg << ", i32 -2147483648, i32 0\n";
        os << "  " << lo_sign << " = xor i32 " << sign_sh << ", " << lo_sign_flip << "\n";
        os << "  " << e_sh << " = shl i32 " << e_lo << ", 23\n";
        os << "  " << packed0 << " = or i32 " << lo_sign << ", " << e_sh << "\n";
        os << "  " << packed << " = or i32 " << packed0 << ", " << lo_mant << "\n";
        // Dropped residual keeps x's sign as a signed zero.
        os << "  " << lo_bits << " = select i1 " << ok << ", i32 " << packed << ", i32 "
           << sign_sh << "\n";
        os << "  " << lo << " = bitcast i32 " << lo_bits << " to float\n";
        return Fp64Pair{hi, lo};
    }

    // Combine a Dekker FP32 pair back into IEEE-754 binary64 bits.
    //
    // Deliberately does NOT compute hi + lo in FP32 first: that is what collapsed
    // every value to 24 bits. Instead both limbs are widened exactly, the smaller
    // is aligned into the larger's significand, and the 53-bit result is packed
    // directly. The join is exact whenever both limbs' significant bits fit
    // binary64's 53-bit window, which a normalized pair usually satisfies; an
    // unusually small lo can sit farther below hi than that, so the alignment
    // rounds to nearest even rather than assuming exact representability.
    std::string emit_soft_f32_pair_to_f64_bits(std::ostringstream& os,
                                               const std::string& hi,
                                               const std::string& lo) {
        const std::string hi_bits = next_tmp("fp64_join_hib");
        const std::string lo_bits = next_tmp("fp64_join_lob");
        os << "  " << hi_bits << " = bitcast float " << hi << " to i32\n";
        os << "  " << lo_bits << " = bitcast float " << lo << " to i32\n";
        const std::string hi64 = emit_soft_f32_bits_to_f64_bits(os, hi_bits);
        const std::string lo64 = emit_soft_f32_bits_to_f64_bits(os, lo_bits);

        // Fallback for anything the exact path cannot handle (zero/inf/nan hi, an
        // unnormalized pair, cancellation). This is the old collapse, so the
        // result is never worse than before the split path existed.
        const std::string collapsed = next_tmp("fp64_join_sum");
        const std::string collapsed_bits = next_tmp("fp64_join_sumb");
        os << "  " << collapsed << " = fadd float " << hi << ", " << lo << "\n";
        os << "  " << collapsed_bits << " = bitcast float " << collapsed << " to i32\n";
        const std::string fallback = emit_soft_f32_bits_to_f64_bits(os, collapsed_bits);

        const std::string hi_exp_raw = next_tmp("fp64_join_hieraw");
        const std::string hi_exp = next_tmp("fp64_join_hie");
        const std::string lo_exp_raw = next_tmp("fp64_join_loeraw");
        const std::string lo_exp = next_tmp("fp64_join_loe");
        os << "  " << hi_exp_raw << " = lshr i64 " << hi64 << ", 52\n";
        os << "  " << hi_exp << " = and i64 " << hi_exp_raw << ", 2047\n";
        os << "  " << lo_exp_raw << " = lshr i64 " << lo64 << ", 52\n";
        os << "  " << lo_exp << " = and i64 " << lo_exp_raw << ", 2047\n";

        const std::string mant_mask = "4503599627370495";       // (1<<52)-1
        const std::string implicit = "4503599627370496";        // 1<<52
        const std::string overflow_bit = "9007199254740992";    // 1<<53

        const std::string hi_mant = next_tmp("fp64_join_him");
        const std::string m_hi = next_tmp("fp64_join_mhi");
        const std::string lo_mant = next_tmp("fp64_join_lom");
        const std::string m_lo = next_tmp("fp64_join_mlo");
        os << "  " << hi_mant << " = and i64 " << hi64 << ", " << mant_mask << "\n";
        os << "  " << m_hi << " = or i64 " << hi_mant << ", " << implicit << "\n";
        os << "  " << lo_mant << " = and i64 " << lo64 << ", " << mant_mask << "\n";
        os << "  " << m_lo << " = or i64 " << lo_mant << ", " << implicit << "\n";

        const std::string shift = next_tmp("fp64_join_shift");
        const std::string shift_ok_lo = next_tmp("fp64_join_shoka");
        const std::string shift_ok_hi = next_tmp("fp64_join_shokb");
        const std::string shift_ok = next_tmp("fp64_join_shok");
        const std::string shift_safe = next_tmp("fp64_join_shsafe");
        os << "  " << shift << " = sub i64 " << hi_exp << ", " << lo_exp << "\n";
        os << "  " << shift_ok_lo << " = icmp sgt i64 " << shift << ", 0\n";
        os << "  " << shift_ok_hi << " = icmp slt i64 " << shift << ", 64\n";
        os << "  " << shift_ok << " = and i1 " << shift_ok_lo << ", " << shift_ok_hi << "\n";
        // Keep the shift in range even when unused; an out-of-range shl/lshr is
        // poison in LLVM and would contaminate the select.
        os << "  " << shift_safe << " = select i1 " << shift_ok << ", i64 " << shift << ", i64 1\n";

        // Align lo into hi's significand. For shift <= 29 nothing is dropped --
        // lo is a float, so its binary64 significand has 29 trailing zeros --
        // and the join is exact. Beyond that the discarded bits are rounded to
        // nearest even rather than truncated, so the packed result is the
        // correctly rounded sum instead of assuming exact representability.
        const std::string delta_trunc = next_tmp("fp64_join_dtrunc");
        const std::string keep_mask = next_tmp("fp64_join_kmask");
        const std::string dropped = next_tmp("fp64_join_dropped");
        const std::string half_sh = next_tmp("fp64_join_halfsh");
        const std::string half = next_tmp("fp64_join_half");
        const std::string drop_gt = next_tmp("fp64_join_dgt");
        const std::string drop_eq = next_tmp("fp64_join_deq");
        const std::string dt_odd_bit = next_tmp("fp64_join_dob");
        const std::string dt_odd = next_tmp("fp64_join_dodd");
        const std::string drop_tie = next_tmp("fp64_join_dtie");
        const std::string drop_up = next_tmp("fp64_join_dup");
        const std::string delta_inc = next_tmp("fp64_join_dinc");
        const std::string delta = next_tmp("fp64_join_delta");
        os << "  " << delta_trunc << " = lshr i64 " << m_lo << ", " << shift_safe << "\n";
        os << "  " << half_sh << " = sub i64 " << shift_safe << ", 1\n";
        os << "  " << half << " = shl i64 1, " << half_sh << "\n";
        os << "  " << keep_mask << " = sub i64 " << half << ", 1\n";
        // dropped = m_lo & ((1 << shift) - 1); (half << 1) - 1 == (1 << shift) - 1
        const std::string full_mask = next_tmp("fp64_join_fmask");
        const std::string half2 = next_tmp("fp64_join_half2");
        os << "  " << half2 << " = shl i64 " << half << ", 1\n";
        os << "  " << full_mask << " = sub i64 " << half2 << ", 1\n";
        os << "  " << dropped << " = and i64 " << m_lo << ", " << full_mask << "\n";
        os << "  " << drop_gt << " = icmp ugt i64 " << dropped << ", " << half << "\n";
        os << "  " << drop_eq << " = icmp eq i64 " << dropped << ", " << half << "\n";
        os << "  " << dt_odd_bit << " = and i64 " << delta_trunc << ", 1\n";
        os << "  " << dt_odd << " = icmp ne i64 " << dt_odd_bit << ", 0\n";
        os << "  " << drop_tie << " = and i1 " << drop_eq << ", " << dt_odd << "\n";
        os << "  " << drop_up << " = or i1 " << drop_gt << ", " << drop_tie << "\n";
        os << "  " << delta_inc << " = add i64 " << delta_trunc << ", 1\n";
        os << "  " << delta << " = select i1 " << drop_up << ", i64 " << delta_inc << ", i64 "
           << delta_trunc << "\n";
        (void) keep_mask;

        const std::string hi_sign = next_tmp("fp64_join_hs");
        const std::string lo_sign = next_tmp("fp64_join_ls");
        const std::string same_sign = next_tmp("fp64_join_same");
        const std::string sum = next_tmp("fp64_join_add");
        const std::string dif = next_tmp("fp64_join_sub");
        const std::string m = next_tmp("fp64_join_m");
        os << "  " << hi_sign << " = lshr i64 " << hi64 << ", 63\n";
        os << "  " << lo_sign << " = lshr i64 " << lo64 << ", 63\n";
        os << "  " << same_sign << " = icmp eq i64 " << hi_sign << ", " << lo_sign << "\n";
        os << "  " << sum << " = add i64 " << m_hi << ", " << delta << "\n";
        os << "  " << dif << " = sub i64 " << m_hi << ", " << delta << "\n";
        os << "  " << m << " = select i1 " << same_sign << ", i64 " << sum << ", i64 " << dif
           << "\n";

        // A normalized pair moves the significand by at most one bit either way:
        // adding can carry out of bit 52, and subtracting can only borrow when hi
        // is an exact power of two.
        const std::string carry = next_tmp("fp64_join_carry");
        const std::string m_sh = next_tmp("fp64_join_msh");
        const std::string m1 = next_tmp("fp64_join_m1");
        const std::string e_inc = next_tmp("fp64_join_einc");
        const std::string e1 = next_tmp("fp64_join_e1");
        os << "  " << carry << " = icmp uge i64 " << m << ", " << overflow_bit << "\n";
        // Renormalizing a carry drops one bit, so it needs its own round to
        // nearest even rather than a bare shift: round up when that bit is set
        // and the surviving LSB is odd (a tie with nothing below rounds to even).
        const std::string c_lost = next_tmp("fp64_join_clost");
        const std::string c_sh = next_tmp("fp64_join_csh");
        const std::string c_odd_bit = next_tmp("fp64_join_cob");
        const std::string c_tie = next_tmp("fp64_join_ctie");
        const std::string c_up = next_tmp("fp64_join_cup");
        const std::string c_inc = next_tmp("fp64_join_cinc");
        os << "  " << c_lost << " = and i64 " << m << ", 1\n";
        os << "  " << c_sh << " = lshr i64 " << m << ", 1\n";
        os << "  " << c_odd_bit << " = and i64 " << c_sh << ", 1\n";
        os << "  " << c_tie << " = and i64 " << c_lost << ", " << c_odd_bit << "\n";
        os << "  " << c_up << " = icmp ne i64 " << c_tie << ", 0\n";
        os << "  " << c_inc << " = add i64 " << c_sh << ", 1\n";
        os << "  " << m_sh << " = select i1 " << c_up << ", i64 " << c_inc << ", i64 " << c_sh
           << "\n";
        os << "  " << m1 << " = select i1 " << carry << ", i64 " << m_sh << ", i64 " << m << "\n";
        os << "  " << e_inc << " = add i64 " << hi_exp << ", 1\n";
        os << "  " << e1 << " = select i1 " << carry << ", i64 " << e_inc << ", i64 " << hi_exp
           << "\n";
        // Rounding up can itself carry out again, but only from an all-ones
        // significand, so the value is then a power of two and one more shift
        // is exact.
        const std::string recarry = next_tmp("fp64_join_recarry");
        const std::string m1b_sh = next_tmp("fp64_join_m1bsh");
        const std::string m1b = next_tmp("fp64_join_m1b");
        const std::string e1b_inc = next_tmp("fp64_join_e1binc");
        const std::string e1b = next_tmp("fp64_join_e1b");
        os << "  " << recarry << " = icmp uge i64 " << m1 << ", " << overflow_bit << "\n";
        os << "  " << m1b_sh << " = lshr i64 " << m1 << ", 1\n";
        os << "  " << m1b << " = select i1 " << recarry << ", i64 " << m1b_sh << ", i64 " << m1
           << "\n";
        os << "  " << e1b_inc << " = add i64 " << e1 << ", 1\n";
        os << "  " << e1b << " = select i1 " << recarry << ", i64 " << e1b_inc << ", i64 " << e1
           << "\n";

        const std::string borrow = next_tmp("fp64_join_borrow");
        const std::string m1_sh = next_tmp("fp64_join_m1sh");
        const std::string m2 = next_tmp("fp64_join_m2");
        const std::string e_dec = next_tmp("fp64_join_edec");
        const std::string e2 = next_tmp("fp64_join_e2");
        os << "  " << borrow << " = icmp ult i64 " << m1b << ", " << implicit << "\n";
        os << "  " << m1_sh << " = shl i64 " << m1b << ", 1\n";
        os << "  " << m2 << " = select i1 " << borrow << ", i64 " << m1_sh << ", i64 " << m1b
           << "\n";
        os << "  " << e_dec << " = sub i64 " << e1b << ", 1\n";
        os << "  " << e2 << " = select i1 " << borrow << ", i64 " << e_dec << ", i64 " << e1b
           << "\n";

        const std::string normalized = next_tmp("fp64_join_normok");
        const std::string e_ok_lo = next_tmp("fp64_join_eoka");
        const std::string e_ok_hi = next_tmp("fp64_join_eokb");
        const std::string e_ok = next_tmp("fp64_join_eok");
        const std::string hi_finite_lo = next_tmp("fp64_join_hfa");
        const std::string hi_finite_hi = next_tmp("fp64_join_hfb");
        const std::string hi_finite = next_tmp("fp64_join_hf");
        const std::string lo_nz = next_tmp("fp64_join_lonz");
        const std::string lo_mag = next_tmp("fp64_join_lomag");
        os << "  " << normalized << " = icmp uge i64 " << m2 << ", " << implicit << "\n";
        os << "  " << e_ok_lo << " = icmp sgt i64 " << e2 << ", 0\n";
        os << "  " << e_ok_hi << " = icmp slt i64 " << e2 << ", 2047\n";
        os << "  " << e_ok << " = and i1 " << e_ok_lo << ", " << e_ok_hi << "\n";
        os << "  " << hi_finite_lo << " = icmp ne i64 " << hi_exp << ", 0\n";
        os << "  " << hi_finite_hi << " = icmp ne i64 " << hi_exp << ", 2047\n";
        os << "  " << hi_finite << " = and i1 " << hi_finite_lo << ", " << hi_finite_hi << "\n";
        os << "  " << lo_mag << " = and i32 " << lo_bits << ", 2147483647\n";
        os << "  " << lo_nz << " = icmp ne i32 " << lo_mag << ", 0\n";

        const std::string g0 = next_tmp("fp64_join_g0");
        const std::string g1 = next_tmp("fp64_join_g1");
        const std::string g2 = next_tmp("fp64_join_g2");
        const std::string use_exact = next_tmp("fp64_join_use");
        os << "  " << g0 << " = and i1 " << shift_ok << ", " << hi_finite << "\n";
        os << "  " << g1 << " = and i1 " << g0 << ", " << normalized << "\n";
        os << "  " << g2 << " = and i1 " << g1 << ", " << e_ok << "\n";
        os << "  " << use_exact << " = and i1 " << g2 << ", " << lo_nz << "\n";

        const std::string sign_sh = next_tmp("fp64_join_ssh");
        const std::string e_sh = next_tmp("fp64_join_esh");
        const std::string mant_out = next_tmp("fp64_join_mout");
        const std::string packed0 = next_tmp("fp64_join_p0");
        const std::string packed = next_tmp("fp64_join_p1");
        const std::string exact_or_hi = next_tmp("fp64_join_exhi");
        const std::string out = next_tmp("fp64_join");
        os << "  " << sign_sh << " = shl i64 " << hi_sign << ", 63\n";
        os << "  " << e_sh << " = shl i64 " << e2 << ", 52\n";
        os << "  " << mant_out << " = and i64 " << m2 << ", " << mant_mask << "\n";
        os << "  " << packed0 << " = or i64 " << sign_sh << ", " << e_sh << "\n";
        os << "  " << packed << " = or i64 " << packed0 << ", " << mant_out << "\n";
        // lo == 0 is the common case and needs no work: hi widens exactly.
        os << "  " << exact_or_hi << " = select i1 " << lo_nz << ", i64 " << fallback << ", i64 "
           << hi64 << "\n";
        os << "  " << out << " = select i1 " << use_exact << ", i64 " << packed << ", i64 "
           << exact_or_hi << "\n";
        return out;
    }

    std::optional<Fp64Pair> decode_fp64_pair(std::ostringstream& os,
                                             const std::string& operand) {
        if (is_register_name(operand)) {
            if (ensure_reg_slot(operand).bits != 64) return std::nullopt;
            // Register ABI is IEEE binary64 bits (matches ld.global.b64 / st.global.b64
            // and host-side double memory). Convert to a Dekker float pair for ALU,
            // keeping the residual: binary64 carries 53 significand bits and the
            // pair carries 48, so the split is the only place precision is lost
            // and the loss does not compound across a chain of instructions.
            const std::string ieee = emit_load_reg_bits(os, operand, 64);
            return emit_soft_f64_bits_to_f32_pair(os, ieee);
        }

        double value = 0.0;
        bool parsed = false;
        if (operand.size() == 18 && operand[0] == '0' && operand[1] == 'd') {
            try {
                const std::uint64_t bits = std::stoull(operand.substr(2), nullptr, 16);
                std::memcpy(&value, &bits, sizeof(value));
                parsed = true;
            } catch (...) {
            }
        } else {
            char* end = nullptr;
            value = std::strtod(operand.c_str(), &end);
            parsed = end != operand.c_str() && *end == '\0';
        }
        if (!parsed) return std::nullopt;
        const float hi_value = static_cast<float>(value);
        const float lo_value =
            static_cast<float>(value - static_cast<double>(hi_value));
        return Fp64Pair{
            emit_float_constant(os, hi_value, "fp64_imm_hi"),
            emit_float_constant(os, lo_value, "fp64_imm_lo")};
    }

    // Call parameters are untyped bit containers, so libdevice FP64 entry points
    // hand over IEEE binary64 bits rather than a register the pair decoder can
    // read. These two convert at that boundary using the same residual-preserving
    // split/join as decode_fp64_pair/store_fp64_pair.
    Fp64Pair fp64_pair_from_ieee_bits(std::ostringstream& os, const std::string& ieee_bits) {
        return emit_soft_f64_bits_to_f32_pair(os, ieee_bits);
    }

    std::string fp64_ieee_bits_from_pair(std::ostringstream& os, const Fp64Pair& value) {
        return emit_soft_f32_pair_to_f64_bits(os, value.hi, value.lo);
    }

    bool store_fp64_pair(std::ostringstream& os, const std::string& dst,
                         const Fp64Pair& value) {
        // Join both limbs into IEEE binary64 bits, so the register slot always
        // holds the CUDA-visible bit pattern. Global stores, .local spills,
        // mov.b64, warp shuffles and aliasing the same eight bytes as a
        // uint64_t therefore need no special handling: nothing anywhere holds a
        // private packed-pair format.
        const std::string ieee = emit_soft_f32_pair_to_f64_bits(os, value.hi, value.lo);
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, ieee, 64);
    }

    bool uses_vf64_support() const {
        return fp64_mode_ == cumetal::ptx::Fp64Mode::kWide48 ||
               fp64_mode_ == cumetal::ptx::Fp64Mode::kIEEE64;
    }

    std::optional<std::string> decode_fp64_raw_bits(
        std::ostringstream& os, const std::string& operand
    ) {
        if (is_register_name(operand)) {
            if (ensure_reg_slot(operand).bits != 64) return std::nullopt;
            return emit_load_reg_bits(os, operand, 64);
        }
        std::uint64_t bits = 0;
        bool parsed = false;
        if (operand.size() == 18 && operand[0] == '0' && operand[1] == 'd') {
            try {
                bits = std::stoull(operand.substr(2), nullptr, 16);
                parsed = true;
            } catch (...) {
            }
        } else {
            char* end = nullptr;
            const double value = std::strtod(operand.c_str(), &end);
            if (end != operand.c_str() && *end == '\0') {
                std::memcpy(&bits, &value, sizeof(bits));
                parsed = true;
            }
        }
        if (!parsed) return std::nullopt;
        const std::string raw = next_tmp("vf64_imm");
        os << "  " << raw << " = or i64 0, " << static_cast<std::int64_t>(bits) << "\n";
        return raw;
    }

    static int vf64_rounding_mode(const std::string& opcode) {
        if (opcode.find(".rz") != std::string::npos) return 1;
        if (opcode.find(".rm") != std::string::npos) return 2;
        if (opcode.find(".rp") != std::string::npos) return 3;
        return 0;
    }

    bool emit_vf64_binary_call(
        std::ostringstream& os,
        const cumetal::ptx::EntryFunction::Instruction& instr,
        const std::string& operation
    ) {
        auto a = decode_fp64_raw_bits(os, instr.operands[1]);
        auto b = decode_fp64_raw_bits(os, instr.operands[2]);
        if (!a || !b) return fail(instr, "VF64 binary source unsupported");
        const int rounding = vf64_rounding_mode(instr.opcode);
        std::string function;
        std::string arguments;
        if (fp64_mode_ == cumetal::ptx::Fp64Mode::kWide48) {
            if (rounding != 0) {
                return fail(instr, "wide48 supports round-to-nearest-even arithmetic only");
            }
            function = "vf64_wide_" + operation;
            declarations_.insert("declare i64 @" + function + "(i64, i64)");
            arguments = "i64 " + *a + ", i64 " + *b;
        } else {
            function = "vf64_" + operation + "_round";
            declarations_.insert("declare i64 @" + function + "(i64, i64, i32)");
            arguments = "i64 " + *a + ", i64 " + *b + ", i32 " +
                        std::to_string(rounding);
        }
        const std::string result = next_tmp("vf64_" + operation);
        os << "  " << result << " = call i64 @" << function << "(" << arguments << ")\n";
        return emit_store_reg_bits(
            os, instr.operands[0], ensure_reg_slot(instr.operands[0]).bits,
            result, 64
        );
    }

    // A pair renormalization ends in `fadd leading, error`, which mishandles two
    // cases that the leading term itself gets right:
    //
    //   zero      IEEE gives -0.0 + 0.0 == +0.0, so -0.0 * 1.0 came back as +0.0.
    //   infinity  the exact-error term of an infinite product or sum is NaN
    //             (fma(inf, 1.0, -inf)), and inf + NaN poisons the result.
    //
    // Both are repaired by falling back to the leading term, which already
    // carries what IEEE prescribes for the operation. The low limb is dropped
    // whenever the result is non-finite so a NaN residual cannot leak into the
    // packer. Tests use integer bit comparisons rather than fcmp because the
    // module is compiled with air.compile.fast_math_enable, under which the
    // backend may assume no NaN or infinity operands.
    Fp64Pair finalize_pair(std::ostringstream& os,
                           const std::string& hi,
                           const std::string& lo,
                           const std::string& leading) {
        const std::string hi_bits = next_tmp("fp64_fin_hib");
        const std::string hi_mag = next_tmp("fp64_fin_himag");
        const std::string hi_nan = next_tmp("fp64_fin_hinan");
        const std::string lead_bits = next_tmp("fp64_fin_lb");
        const std::string lead_mag = next_tmp("fp64_fin_lmag");
        const std::string lead_nan = next_tmp("fp64_fin_lnan");
        const std::string lead_ok = next_tmp("fp64_fin_lok");
        const std::string use_lead = next_tmp("fp64_fin_usel");
        const std::string hi1 = next_tmp("fp64_fin_hi1");
        os << "  " << hi_bits << " = bitcast float " << hi << " to i32\n";
        os << "  " << hi_mag << " = and i32 " << hi_bits << ", 2147483647\n";
        os << "  " << hi_nan << " = icmp ugt i32 " << hi_mag << ", 2139095040\n";
        os << "  " << lead_bits << " = bitcast float " << leading << " to i32\n";
        os << "  " << lead_mag << " = and i32 " << lead_bits << ", 2147483647\n";
        os << "  " << lead_nan << " = icmp ugt i32 " << lead_mag << ", 2139095040\n";
        os << "  " << lead_ok << " = xor i1 " << lead_nan << ", true\n";
        os << "  " << use_lead << " = and i1 " << hi_nan << ", " << lead_ok << "\n";
        os << "  " << hi1 << " = select i1 " << use_lead << ", float " << leading << ", float "
           << hi << "\n";

        const std::string hi1_bits = next_tmp("fp64_fin_h1b");
        const std::string hi1_mag = next_tmp("fp64_fin_h1mag");
        const std::string hi1_zero = next_tmp("fp64_fin_h1z");
        const std::string hi2 = next_tmp("fp64_fin_hi2");
        os << "  " << hi1_bits << " = bitcast float " << hi1 << " to i32\n";
        os << "  " << hi1_mag << " = and i32 " << hi1_bits << ", 2147483647\n";
        os << "  " << hi1_zero << " = icmp eq i32 " << hi1_mag << ", 0\n";
        os << "  " << hi2 << " = select i1 " << hi1_zero << ", float " << leading << ", float "
           << hi1 << "\n";

        const std::string hi2_bits = next_tmp("fp64_fin_h2b");
        const std::string hi2_mag = next_tmp("fp64_fin_h2mag");
        const std::string finite = next_tmp("fp64_fin_fin");
        const std::string zero = emit_float_constant(os, 0.0f, "fp64_fin_zero");
        const std::string lo_out = next_tmp("fp64_fin_lo");
        os << "  " << hi2_bits << " = bitcast float " << hi2 << " to i32\n";
        os << "  " << hi2_mag << " = and i32 " << hi2_bits << ", 2147483647\n";
        os << "  " << finite << " = icmp ult i32 " << hi2_mag << ", 2139095040\n";
        os << "  " << lo_out << " = select i1 " << finite << ", float " << lo << ", float " << zero
           << "\n";
        return {hi2, lo_out};
    }

    Fp64Pair emit_fp64_pair_add(std::ostringstream& os,
                                const Fp64Pair& a,
                                const Fp64Pair& b,
                                bool subtract = false) {
        std::string b_hi = b.hi;
        std::string b_lo = b.lo;
        if (subtract) {
            b_hi = next_tmp("fp64_sub_hi");
            b_lo = next_tmp("fp64_sub_lo");
            os << "  " << b_hi << " = fneg float " << b.hi << "\n";
            os << "  " << b_lo << " = fneg float " << b.lo << "\n";
        }
        const std::string sum = next_tmp("fp64_sum");
        const std::string b_virtual = next_tmp("fp64_bvirtual");
        const std::string a_round = next_tmp("fp64_around");
        const std::string a_error = next_tmp("fp64_aerror");
        const std::string b_round = next_tmp("fp64_bround");
        const std::string err0 = next_tmp("fp64_err0");
        const std::string err1 = next_tmp("fp64_err1");
        const std::string err2 = next_tmp("fp64_err2");
        os << "  " << sum << " = fadd float " << a.hi << ", " << b_hi << "\n";
        os << "  " << b_virtual << " = fsub float " << sum << ", " << a.hi << "\n";
        os << "  " << a_round << " = fsub float " << sum << ", " << b_virtual << "\n";
        os << "  " << a_error << " = fsub float " << a.hi << ", " << a_round << "\n";
        os << "  " << b_round << " = fsub float " << b_hi << ", " << b_virtual << "\n";
        os << "  " << err0 << " = fadd float " << a_error << ", " << b_round << "\n";
        os << "  " << err1 << " = fadd float " << err0 << ", " << a.lo << "\n";
        os << "  " << err2 << " = fadd float " << err1 << ", " << b_lo << "\n";
        const std::string hi = next_tmp("fp64_add_hi");
        const std::string delta = next_tmp("fp64_add_delta");
        const std::string lo = next_tmp("fp64_add_lo");
        os << "  " << hi << " = fadd float " << sum << ", " << err2 << "\n";
        os << "  " << delta << " = fsub float " << hi << ", " << sum << "\n";
        os << "  " << lo << " = fsub float " << err2 << ", " << delta << "\n";
        return finalize_pair(os, hi, lo, sum);
    }

    Fp64Pair emit_fp64_pair_div(std::ostringstream& os,
                                const Fp64Pair& a,
                                const Fp64Pair& b) {
        const std::string quotient = next_tmp("fp64_quotient");
        os << "  " << quotient << " = fdiv float " << a.hi << ", " << b.hi << "\n";
        const std::string zero = emit_float_constant(os, 0.0f, "fp64_zero");
        const Fp64Pair q{quotient, zero};
        const Fp64Pair product = emit_fp64_pair_mul(os, q, b);
        const Fp64Pair residual = emit_fp64_pair_add(os, a, product, true);
        const std::string residual_sum = next_tmp("fp64_div_residual");
        os << "  " << residual_sum << " = fadd float "
           << residual.hi << ", " << residual.lo << "\n";
        const std::string correction = next_tmp("fp64_div_correction");
        os << "  " << correction << " = fdiv float "
           << residual_sum << ", " << b.hi << "\n";
        const Fp64Pair correction_pair{correction, zero};
        const Fp64Pair result = emit_fp64_pair_add(os, q, correction_pair);
        return finalize_pair(os, result.hi, result.lo, quotient);
    }

    Fp64Pair emit_fp64_pair_mul(std::ostringstream& os,
                                const Fp64Pair& a,
                                const Fp64Pair& b) {
        declarations_.insert("declare float @llvm.fma.f32(float, float, float)");
        const std::string product = next_tmp("fp64_product");
        const std::string neg_product = next_tmp("fp64_neg_product");
        const std::string exact_error = next_tmp("fp64_product_error");
        const std::string cross0 = next_tmp("fp64_cross0");
        const std::string cross1 = next_tmp("fp64_cross1");
        const std::string cross2 = next_tmp("fp64_cross2");
        const std::string error0 = next_tmp("fp64_mul_error0");
        const std::string error1 = next_tmp("fp64_mul_error1");
        os << "  " << product << " = fmul float " << a.hi << ", " << b.hi << "\n";
        os << "  " << neg_product << " = fneg float " << product << "\n";
        os << "  " << exact_error << " = call float @llvm.fma.f32(float "
           << a.hi << ", float " << b.hi << ", float " << neg_product << ")\n";
        os << "  " << cross0 << " = fmul float " << a.hi << ", " << b.lo << "\n";
        os << "  " << cross1 << " = fmul float " << a.lo << ", " << b.hi << "\n";
        os << "  " << cross2 << " = fmul float " << a.lo << ", " << b.lo << "\n";
        os << "  " << error0 << " = fadd float " << exact_error << ", " << cross0 << "\n";
        os << "  " << error1 << " = fadd float " << error0 << ", " << cross1 << "\n";
        const std::string error2 = next_tmp("fp64_mul_error2");
        os << "  " << error2 << " = fadd float " << error1 << ", " << cross2 << "\n";
        const std::string hi = next_tmp("fp64_mul_hi");
        const std::string delta = next_tmp("fp64_mul_delta");
        const std::string lo = next_tmp("fp64_mul_lo");
        os << "  " << hi << " = fadd float " << product << ", " << error2 << "\n";
        os << "  " << delta << " = fsub float " << hi << ", " << product << "\n";
        os << "  " << lo << " = fsub float " << error2 << ", " << delta << "\n";
        return finalize_pair(os, hi, lo, product);
    }

    std::optional<std::string> encode_value_to_reg_bits(std::ostringstream& os,
                                                        const Value& value,
                                                        int dst_bits) {
        if (value.type.kind == PtxTypeSpec::Kind::kPred) {
            if (dst_bits == 1) {
                return value.ir;
            }
            const std::string z = next_tmp("predzext");
            os << "  " << z << " = zext i1 " << value.ir << " to " << llvm_int_type(dst_bits) << "\n";
            return z;
        }
        if (value.type.kind == PtxTypeSpec::Kind::kInt) {
            std::string out = value.ir;
            if (value.bits < dst_bits) {
                const std::string ext = next_tmp("iext");
                os << "  " << ext << " = " << (value.type.is_signed ? "sext " : "zext ")
                   << llvm_int_type(value.bits) << " " << out << " to " << llvm_int_type(dst_bits) << "\n";
                out = ext;
            } else if (value.bits > dst_bits) {
                const std::string tr = next_tmp("itr");
                os << "  " << tr << " = trunc " << llvm_int_type(value.bits) << " " << out << " to "
                   << llvm_int_type(dst_bits) << "\n";
                out = tr;
            }
            return out;
        }
        if (value.type.kind == PtxTypeSpec::Kind::kFloat) {
            if (value.type.bits != dst_bits) {
                return std::nullopt;
            }
            const std::string bc = next_tmp("f2i");
            const std::string fty = (dst_bits == 16) ? "half" : (dst_bits == 32 ? "float" : "double");
            os << "  " << bc << " = bitcast " << fty << " " << value.ir
               << " to " << llvm_int_type(dst_bits) << "\n";
            return bc;
        }
        return std::nullopt;
    }

    bool append_builtin_vec3(const std::string& air_key, const std::string& arg_name) {
        if (builtin_vector_arg_name_.count(air_key)) {
            return true;
        }
        ParamInfo p;
        p.ptx_type = ".builtin." + air_key;
        p.llvm_type = "<3 x i32>";
        p.name = arg_name;
        p.raw_name = arg_name;
        p.builtin_air_key = air_key;
        p.builtin_air_type_name = "uint3";
        params_->push_back(p);
        arg_decls_->push_back("<3 x i32> %" + arg_name);
        builtin_vector_arg_name_[air_key] = arg_name;
        builtin_params_added_.push_back(p);
        return true;
    }

    bool append_builtin_scalar(const std::string& air_key, const std::string& arg_name) {
        if (builtin_scalar_arg_name_.count(air_key)) {
            return true;
        }
        ParamInfo p;
        p.ptx_type = ".builtin." + air_key;
        p.llvm_type = "i32";
        p.name = arg_name;
        p.raw_name = arg_name;
        p.builtin_air_key = air_key;
        p.builtin_air_type_name = "uint";
        params_->push_back(p);
        arg_decls_->push_back("i32 %" + arg_name);
        builtin_scalar_arg_name_[air_key] = arg_name;
        builtin_params_added_.push_back(p);
        return true;
    }

    bool append_threadgroup_buffer_param() {
        if (has_threadgroup_buffer_param_) {
            return true;
        }
        ParamInfo p;
        p.ptx_type = ".builtin.air.threadgroup_buffer.0";
        p.llvm_type = "i8 addrspace(3)*";
        p.name = threadgroup_buffer_arg_name_;
        p.raw_name = threadgroup_buffer_arg_name_;
        p.builtin_air_type_name = "uchar";
        params_->push_back(p);
        arg_decls_->push_back("i8 addrspace(3)* %" + threadgroup_buffer_arg_name_);
        has_threadgroup_buffer_param_ = true;
        return true;
    }

    bool append_required_builtin_params() {
        bool needs_tid = false;
        bool needs_bid = false;
        bool needs_tpg = false;
        bool needs_gpg = false;
        bool needs_lane = false;
        bool needs_threadgroup_buffer = false;
        for (const auto& instr : entry_.instructions) {
            const std::vector<std::string> scan_operands = [&]() {
                std::vector<std::string> o = instr.operands;
                if (!instr.predicate.empty()) {
                    o.push_back(instr.predicate);
                }
                return o;
            }();
            for (const std::string& op : scan_operands) {
                if (op.find("%tid.") != std::string::npos) needs_tid = true;
                if (op.find("%ctaid.") != std::string::npos) needs_bid = true;
                if (op.find("%ntid.") != std::string::npos) needs_tpg = true;
                if (op.find("%nctaid.") != std::string::npos) needs_gpg = true;
                if (op.find("%laneid") != std::string::npos ||
                    op.find("%lanemask_") != std::string::npos) needs_lane = true;
                if (op.find("__cumetal_grid_sync") != std::string::npos) {
                    needs_tid = true;
                    needs_gpg = true;
                }
                if (shared_symbols_ != nullptr &&
                    shared_symbols_->find(trim(op)) != shared_symbols_->end()) {
                    needs_threadgroup_buffer = true;
                }
            }
            if (instr.opcode.find(".shared") != std::string::npos ||
                starts_with(instr.opcode, "bar.sync") ||
                starts_with(instr.opcode, "bar.warp.sync") ||
                starts_with(instr.opcode, "shfl.sync")) {
                needs_threadgroup_buffer = true;
            }
            if (starts_with(instr.opcode, "shfl.sync")) {
                needs_lane = true;
            }
        }
        if (needs_tid && !append_builtin_vec3("air.thread_position_in_threadgroup", "__air_tid3")) return false;
        if (needs_bid && !append_builtin_vec3("air.threadgroup_position_in_grid", "__air_bid3")) return false;
        if (needs_tpg && !append_builtin_vec3("air.threads_per_threadgroup", "__air_tpg3")) return false;
        if (needs_gpg && !append_builtin_vec3("air.threadgroups_per_grid", "__air_gpg3")) return false;
        if (needs_lane && !append_builtin_scalar("air.thread_index_in_simdgroup", "__air_laneid")) return false;
        if (needs_threadgroup_buffer && !append_threadgroup_buffer_param()) return false;
        return true;
    }

    bool index_control_flow() {
        exec_indices_.clear();
        for (int i = 0; i < static_cast<int>(entry_.instructions.size()); ++i) {
            const std::string& opcode =
                entry_.instructions[static_cast<std::size_t>(i)].opcode;
            if (opcode == "ptx.label" || opcode == "ptx.branchtargets") {
                continue;
            }
            exec_pos_by_instr_index_[i] = static_cast<int>(exec_indices_.size());
            exec_indices_.push_back(i);
        }
        for (int pos = 0; pos < static_cast<int>(exec_indices_.size()); ++pos) {
            next_exec_pos_by_exec_pos_[pos] = (pos + 1 < static_cast<int>(exec_indices_.size())) ? (pos + 1) : -1;
        }
        for (int i = 0; i < static_cast<int>(entry_.instructions.size()); ++i) {
            const auto& instr = entry_.instructions[static_cast<std::size_t>(i)];
            if (instr.opcode != "ptx.label" || instr.operands.empty()) {
                continue;
            }
            int target_pos = -1;
            for (int j = i + 1; j < static_cast<int>(entry_.instructions.size()); ++j) {
                auto it = exec_pos_by_instr_index_.find(j);
                if (it != exec_pos_by_instr_index_.end()) {
                    target_pos = it->second;
                    break;
                }
            }
            if (target_pos < 0) {
                target_pos = -1;
            }
            label_to_exec_pos_[instr.operands[0]] = target_pos;
        }
        for (const auto& instr : entry_.instructions) {
            if (instr.opcode != "ptx.branchtargets" || instr.operands.empty()) continue;
            if (instr.operands.size() < 2) {
                return fail(instr, "branch-target table is empty");
            }
            branch_tables_[instr.operands[0]] =
                std::vector<std::string>(instr.operands.begin() + 1,
                                         instr.operands.end());
        }
        return true;
    }

    std::string block_name_for_exec_pos(int exec_pos) const {
        if (exec_pos < 0) {
            return "cm_exit";
        }
        return "cm_bb_" + std::to_string(exec_pos);
    }

    std::optional<std::string> emit_special_register_value(std::ostringstream& os,
                                                           const std::string& token,
                                                           int dst_bits) {
        auto emit_extract = [&](const std::string& air_key, const std::string& reg_name, int idx) -> std::optional<std::string> {
            const auto it = builtin_vector_arg_name_.find(air_key);
            if (it == builtin_vector_arg_name_.end()) {
                return std::nullopt;
            }
            const std::string ex = next_tmp("extract");
            os << "  " << ex << " = extractelement <3 x i32> %" << it->second << ", i64 " << idx << "\n";
            if (dst_bits == 32) {
                return ex;
            }
            const std::string ext = next_tmp("zext");
            os << "  " << ext << " = zext i32 " << ex << " to " << llvm_int_type(dst_bits) << "\n";
            return ext;
        };
        if (token == "%tid.x") return emit_extract("air.thread_position_in_threadgroup", "__air_tid3", 0);
        if (token == "%tid.y") return emit_extract("air.thread_position_in_threadgroup", "__air_tid3", 1);
        if (token == "%tid.z") return emit_extract("air.thread_position_in_threadgroup", "__air_tid3", 2);
        if (token == "%ctaid.x") return emit_extract("air.threadgroup_position_in_grid", "__air_bid3", 0);
        if (token == "%ctaid.y" && module_uses_grid_y_offset_) {
            const auto base = emit_extract("air.threadgroup_position_in_grid",
                                           "__air_bid3", 1);
            if (!base) return std::nullopt;
            const std::string offset = next_tmp("grid_y_offset");
            const std::string adjusted = next_tmp("grid_y_adjusted");
            os << "  " << offset
               << " = load i32, i32 addrspace(2)* %__cumetal_grid_y_offset, align 4\n"
               << "  " << adjusted << " = add i32 " << *base << ", " << offset << "\n";
            if (dst_bits == 32) return adjusted;
            const std::string ext = next_tmp("grid_y_ext");
            os << "  " << ext << " = zext i32 " << adjusted << " to "
               << llvm_int_type(dst_bits) << "\n";
            return ext;
        }
        if (token == "%ctaid.y") return emit_extract("air.threadgroup_position_in_grid", "__air_bid3", 1);
        if (token == "%ctaid.z") return emit_extract("air.threadgroup_position_in_grid", "__air_bid3", 2);
        if (token == "%ntid.x") return emit_extract("air.threads_per_threadgroup", "__air_tpg3", 0);
        if (token == "%ntid.y") return emit_extract("air.threads_per_threadgroup", "__air_tpg3", 1);
        if (token == "%ntid.z") return emit_extract("air.threads_per_threadgroup", "__air_tpg3", 2);
        if (token == "%nctaid.x") return emit_extract("air.threadgroups_per_grid", "__air_gpg3", 0);
        if (token == "%nctaid.y") return emit_extract("air.threadgroups_per_grid", "__air_gpg3", 1);
        if (token == "%nctaid.z") return emit_extract("air.threadgroups_per_grid", "__air_gpg3", 2);
        if (token == "%laneid") {
            const auto it = builtin_scalar_arg_name_.find("air.thread_index_in_simdgroup");
            if (it == builtin_scalar_arg_name_.end()) {
                return std::nullopt;
            }
            if (dst_bits == 32) {
                return "%" + it->second;
            }
            const std::string ext = next_tmp("laneext");
            os << "  " << ext << " = zext i32 %" << it->second << " to " << llvm_int_type(dst_bits) << "\n";
            return ext;
        }
        // The lanemask registers are pure functions of the lane index, so derive
        // them from AIR's simdgroup lane rather than leaving them to the generic
        // register path, which silently read zero.
        if (starts_with(token, "%lanemask_")) {
            const auto it = builtin_scalar_arg_name_.find("air.thread_index_in_simdgroup");
            if (it == builtin_scalar_arg_name_.end()) {
                return std::nullopt;
            }
            const std::string eq = next_tmp("lanemask_eq");
            os << "  " << eq << " = shl i32 1, %" << it->second << "\n";
            std::string value;
            if (token == "%lanemask_eq") {
                value = eq;
            } else {
                const std::string lt = next_tmp("lanemask_lt");
                os << "  " << lt << " = add i32 " << eq << ", -1\n";
                if (token == "%lanemask_lt") {
                    value = lt;
                } else if (token == "%lanemask_ge") {
                    value = next_tmp("lanemask_ge");
                    os << "  " << value << " = xor i32 " << lt << ", -1\n";
                } else {
                    // le = lt | eq, which stays correct for lane 31 where
                    // shifting by laneid + 1 would overflow.
                    const std::string le = next_tmp("lanemask_le");
                    os << "  " << le << " = or i32 " << lt << ", " << eq << "\n";
                    if (token == "%lanemask_le") {
                        value = le;
                    } else if (token == "%lanemask_gt") {
                        value = next_tmp("lanemask_gt");
                        os << "  " << value << " = xor i32 " << le << ", -1\n";
                    } else {
                        return std::nullopt;
                    }
                }
            }
            if (dst_bits == 32) {
                return value;
            }
            const std::string ext = next_tmp("lanemask_ext");
            os << "  " << ext << " = zext i32 " << value << " to " << llvm_int_type(dst_bits)
               << "\n";
            return ext;
        }
        if (token == "%activemask") {
            // clang lowers __activemask()'s inline asm to `mov.u32 %r, %activemask`
            // rather than the standalone `activemask.b32` opcode, so it arrives
            // here as a special register read rather than through
            // emit_activemask(). Without this case it fell through to the generic
            // register path, which minted an uninitialised slot and read zero.
            declarations_.insert("declare i64 @air.simd_ballot.i64(i1)");
            const std::string active64 = next_tmp("sreg_activemask64");
            os << "  " << active64 << " = call i64 @air.simd_ballot.i64(i1 true)\n";
            const std::string active32 = next_tmp("sreg_activemask32");
            os << "  " << active32 << " = trunc i64 " << active64 << " to i32\n";
            if (dst_bits == 32) {
                return active32;
            }
            const std::string ext = next_tmp("sreg_activemask_ext");
            os << "  " << ext << " = zext i32 " << active32 << " to " << llvm_int_type(dst_bits)
               << "\n";
            return ext;
        }
        if (token == "%warpsize") {
            if (dst_bits <= 32) {
                return std::string("32");
            }
            const std::string ext = next_tmp("warpext");
            os << "  " << ext << " = zext i32 32 to " << llvm_int_type(dst_bits) << "\n";
            return ext;
        }
        if (token == "%clock" || token == "%clock64" || token == "%globaltimer" ||
            token == "%globaltimer_lo") {
            if (!module_uses_device_clock_) {
                return std::nullopt;
            }
            // Metal Shading Language has no public device cycle-counter API.
            // A device-wide atomic counter retains CUDA's unsigned monotonic
            // and wraparound behavior, which is what clock-based wait loops
            // and ordering probes rely on. The 1024-unit quantum keeps waits
            // bounded; values are deliberately not advertised as GPU cycles.
            const std::string tick = next_tmp("device_clock");
            os << "  " << tick
               << " = atomicrmw add i32 addrspace(1)* %__cumetal_device_clock, i32 1024 monotonic\n";
            if (dst_bits <= 32) {
                return tick;
            }
            const std::string ext = next_tmp("device_clock_ext");
            os << "  " << ext << " = zext i32 " << tick << " to "
               << llvm_int_type(dst_bits) << "\n";
            return ext;
        }
        if (token == "%clock_hi" || token == "%globaltimer_hi") {
            return std::string("0");
        }
        return std::nullopt;
    }

    std::optional<std::string> resolve_param_symbol_address(std::ostringstream& os,
                                                            const std::string& symbol) {
        const auto pit = param_by_raw_.find(symbol);
        if (pit == param_by_raw_.end()) {
            return std::nullopt;
        }
        const ParamInfo& p = (*params_)[static_cast<std::size_t>(pit->second)];
        if (!is_constant_buffer_pointer(p.llvm_type)) {
            return std::nullopt;
        }
        const std::string tmp = next_tmp("p2i");
        os << "  " << tmp << " = ptrtoint " << p.llvm_type << " %" << p.name << " to i64\n";
        return tmp;
    }

    std::optional<std::string> resolve_threadgroup_symbol_address(std::ostringstream& os,
                                                                  const std::string& symbol) {
        if (starts_with(symbol, "__local_depot")) {
            return std::nullopt;
        }
        if (!has_threadgroup_buffer_param_ || shared_symbols_ == nullptr) {
            return std::nullopt;
        }
        const auto it = shared_symbols_->find(symbol);
        if (it == shared_symbols_->end()) {
            return std::nullopt;
        }
        const std::string tmp = next_tmp("tg_p2i");
        os << "  " << tmp << " = ptrtoint i8 addrspace(3)* %" << threadgroup_buffer_arg_name_ << " to i64\n";
        // Apply per-symbol byte offset for multiple static __shared__ arrays.
        if (it->second.offset_bytes > 0) {
            const std::string off = next_tmp("tg_sym_off");
            os << "  " << off << " = add i64 " << tmp << ", " << it->second.offset_bytes << "\n";
            return off;
        }
        return tmp;
    }

    std::optional<std::string> resolve_local_symbol_address(std::ostringstream& os,
                                                            const std::string& symbol) {
        if (!starts_with(symbol, "__local_depot")) {
            return std::nullopt;
        }
        auto it = local_symbols_.find(symbol);
        if (it == local_symbols_.end()) {
            // The declared depot size is the only correct frame size. Refuse to
            // lower rather than guess: an undersized frame does not fault, it
            // silently reads zeros (see parse_ptx_local_depots).
            if (local_depots_ == nullptr) {
                return std::nullopt;
            }
            const auto dit = local_depots_->find(symbol);
            if (dit == local_depots_->end() || dit->second.size_bytes == 0) {
                return std::nullopt;
            }
            LocalSymbolInfo info;
            info.size_bytes = dit->second.size_bytes;
            info.align_bytes = dit->second.align_bytes;
            const std::string sanitized = sanitize_llvm_identifier(symbol, "local_depot");
            info.alloca_name = "%cm_local_" + sanitized + "_" + std::to_string(slot_id_++);
            info.base_ptr_name = "%cm_local_base_" + sanitized + "_" + std::to_string(slot_id_++);
            entry_allocas_ << "  " << info.alloca_name << " = alloca [" << info.size_bytes << " x i8], align "
                           << info.align_bytes << "\n";
            entry_allocas_ << "  " << info.base_ptr_name << " = getelementptr [" << info.size_bytes
                           << " x i8], [" << info.size_bytes << " x i8]* " << info.alloca_name
                           << ", i32 0, i32 0\n";
            auto inserted = local_symbols_.emplace(symbol, std::move(info));
            it = inserted.first;
        }
        const std::string tmp = next_tmp("loc_p2i");
        os << "  " << tmp << " = ptrtoint i8* " << it->second.base_ptr_name << " to i64\n";
        return tmp;
    }

    std::optional<std::string> resolve_global_symbol_address(std::ostringstream& os,
                                                             const std::string& symbol) {
        if (global_symbols_ == nullptr) {
            return std::nullopt;
        }
        const auto it = global_symbols_->find(symbol);
        if (it == global_symbols_->end() || it->second.byte_count == 0 ||
            it->second.llvm_param_name.empty()) {
            return std::nullopt;
        }
        const std::string p2i = next_tmp("global_arg_p2i");
        os << "  " << p2i << " = ptrtoint i8 addrspace(1)* %"
           << it->second.llvm_param_name << " to i64\n";
        return p2i;
    }

    std::optional<std::string> resolve_const_symbol_address(std::ostringstream& os,
                                                            const std::string& symbol) {
        if (const_symbols_ == nullptr) {
            return std::nullopt;
        }
        const auto it = const_symbols_->find(symbol);
        if (it == const_symbols_->end() || it->second.byte_count == 0) {
            return std::nullopt;
        }
        if (!it->second.llvm_param_name.empty()) {
            const std::string p2i = next_tmp("const_arg_p2i");
            os << "  " << p2i << " = ptrtoint i8 addrspace(2)* %"
               << it->second.llvm_param_name << " to i64\n";
            return pointer_add_bytes(os, p2i,
                                     static_cast<std::int64_t>(it->second.byte_offset));
        }
        const std::string gep = next_tmp("const_gep");
        os << "  " << gep << " = getelementptr inbounds [" << it->second.byte_count << " x i8], ["
           << it->second.byte_count << " x i8] addrspace(2)* " << it->second.llvm_global_name
           << ", i64 0, i64 0\n";
        const std::string p2i = next_tmp("const_p2i");
        os << "  " << p2i << " = ptrtoint i8 addrspace(2)* " << gep << " to i64\n";
        return p2i;
    }

    std::string pointer_add_bytes(std::ostringstream& os, const std::string& base_i64, std::int64_t offset) {
        if (offset == 0) {
            return base_i64;
        }
        const std::string out = next_tmp("ptradd");
        if (offset >= 0) {
            os << "  " << out << " = add i64 " << base_i64 << ", " << offset << "\n";
        } else {
            os << "  " << out << " = sub i64 " << base_i64 << ", " << (-offset) << "\n";
        }
        return out;
    }

    std::optional<std::string> get_param_slot(const std::string& name, int bits, bool create) {
        // PTX call-sequence parameters are lexically scoped, and clang freely
        // reuses names such as `param1` at different widths in one entry. Key
        // by width as well as spelling so a later b64 sequence cannot load from
        // the i32 alloca created by an earlier b32 sequence.
        const std::string slot_key = name + ":" + std::to_string(bits);
        auto it = call_param_slots_.find(slot_key);
        if (it != call_param_slots_.end()) {
            return it->second;
        }
        if (!create) {
            return std::nullopt;
        }
        const std::string slot = "%cm_callslot_" + sanitize_llvm_identifier(name, "slot") + "_" +
                                 std::to_string(slot_id_++);
        entry_allocas_ << "  " << slot << " = alloca " << llvm_int_type(bits) << ", align " << std::max(1, bits / 8) << "\n";
        entry_allocas_ << "  store " << llvm_int_type(bits) << " 0, " << llvm_int_type(bits) << "* " << slot
                       << ", align " << std::max(1, bits / 8) << "\n";
        call_param_slots_[slot_key] = slot;
        return slot;
    }

    std::vector<std::string> parse_paren_tuple(std::string text) {
        text = trim(text);
        if (!text.empty() && text.front() == '(' && text.back() == ')') {
            text = text.substr(1, text.size() - 2);
        }
        return split_comma_list(text);
    }

    std::optional<std::string> load_call_slot_value(std::ostringstream& os,
                                                    const std::string& name,
                                                    int bits) {
        auto slot = get_param_slot(name, bits, false);
        if (!slot) {
            return std::nullopt;
        }
        const std::string ld = next_tmp("ldcall");
        os << "  " << ld << " = load " << llvm_int_type(bits) << ", " << llvm_int_type(bits)
           << "* " << *slot << ", align " << std::max(1, bits / 8) << "\n";
        return ld;
    }

    std::optional<std::string> load_call_slot_value_at(std::ostringstream& os,
                                                       const std::string& name,
                                                       int bits,
                                                       std::int64_t byte_offset) {
        const std::string slot_name = byte_offset == 0
                                          ? name
                                          : name + "@" + std::to_string(byte_offset);
        return load_call_slot_value(os, slot_name, bits);
    }

    std::optional<std::string> emit_integer_from_any(std::ostringstream& os,
                                                     const std::string& operand,
                                                     int bits,
                                                     bool is_signed) {
        if (auto special = emit_special_register_value(os, operand, bits)) {
            return *special;
        }
        if (is_register_name(operand)) {
            if (auto v = decode_integer_operand(os, operand, bits, is_signed)) {
                return v->ir;
            }
        }
        if (const auto imm = parse_signed_immediate(operand)) {
            return std::to_string(*imm);
        }
        if (bits == 32 && operand.size() == 10 && operand[0] == '0' && operand[1] == 'f') {
            // PTX float hex bit-pattern in integer context: convert to decimal
            try {
                const auto v = static_cast<int32_t>(
                    static_cast<uint32_t>(std::stoul(operand.substr(2), nullptr, 16)));
                return std::to_string(v);
            } catch (...) {}
        }
        if (bits == 64 && operand.size() == 18 && operand[0] == '0' && operand[1] == 'd') {
            try {
                const auto v = static_cast<int64_t>(std::stoull(operand.substr(2), nullptr, 16));
                return std::to_string(v);
            } catch (...) {}
        }
        if (bits == 64 && device_kernel_names_ != nullptr) {
            for (const auto& kernel_name : *device_kernel_names_) {
                if (operand == kernel_name) {
                    return std::to_string(stable_device_function_token(kernel_name));
                }
            }
        }
        if (const auto addr = resolve_param_symbol_address(os, operand)) {
            if (bits == 64) {
                return *addr;
            }
            const std::string cast = next_tmp("addrtr");
            os << "  " << cast << " = trunc i64 " << *addr << " to " << llvm_int_type(bits) << "\n";
            return cast;
        }
        if (const auto local = resolve_local_symbol_address(os, operand)) {
            if (bits == 64) {
                return *local;
            }
            const std::string cast = next_tmp("loc_addrtr");
            os << "  " << cast << " = trunc i64 " << *local << " to " << llvm_int_type(bits) << "\n";
            return cast;
        }
        if (const auto tg = resolve_threadgroup_symbol_address(os, operand)) {
            if (bits == 64) {
                return *tg;
            }
            const std::string cast = next_tmp("tg_addrtr");
            os << "  " << cast << " = trunc i64 " << *tg << " to " << llvm_int_type(bits) << "\n";
            return cast;
        }
        if (const auto global = resolve_global_symbol_address(os, operand)) {
            if (bits == 64) {
                return *global;
            }
            const std::string cast = next_tmp("global_addrtr");
            os << "  " << cast << " = trunc i64 " << *global << " to "
               << llvm_int_type(bits) << "\n";
            return cast;
        }
        if (const auto cst = resolve_const_symbol_address(os, operand)) {
            if (bits == 64) {
                return *cst;
            }
            const std::string cast = next_tmp("const_addrtr");
            os << "  " << cast << " = trunc i64 " << *cst << " to " << llvm_int_type(bits) << "\n";
            return cast;
        }
        return std::nullopt;
    }

    bool emit_mov_instruction(std::ostringstream& os,
                             const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2) {
            return fail(instr, "mov requires 2 operands");
        }
        const std::string& dst = instr.operands[0];
        const std::string& src = instr.operands[1];

        if (!dst.empty() && dst.front() == '{') {
            const std::vector<std::string> parts = split_comma_list(dst);
            if (instr.opcode.find(".b64") != std::string::npos) {
                // mov.b64 {%r1, %r2}, %rd — unpack 64-bit into lo/hi 32-bit halves
                if (parts.size() != 2) {
                    return fail(instr, "mov.b64 tuple unpack expects 2 dests");
                }
                auto src_i64 = emit_integer_from_any(os, src, 64, false);
                if (!src_i64.has_value()) {
                    return fail(instr, "mov.b64 tuple unpack source unsupported");
                }
                const std::string lo32 = next_tmp("movlo");
                os << "  " << lo32 << " = trunc i64 " << *src_i64 << " to i32\n";
                const std::string hi_shift = next_tmp("movhi_sh");
                os << "  " << hi_shift << " = lshr i64 " << *src_i64 << ", 32\n";
                const std::string hi32 = next_tmp("movhi");
                os << "  " << hi32 << " = trunc i64 " << hi_shift << " to i32\n";
                if (!emit_store_reg_bits(os, parts[0], 32, lo32, 32)) return false;
                if (!emit_store_reg_bits(os, parts[1], 32, hi32, 32)) return false;
                return true;
            }
            if (instr.opcode.find(".b32") == std::string::npos) {
                return fail(instr, "only mov.b32/mov.b64 tuple unpack supported");
            }
            // mov.b32 {%r1, %r2}, %r3 — unpack 32-bit into lo/hi 16-bit halves
            if (parts.size() != 2) {
                return fail(instr, "mov.b32 tuple unpack expects 2 dests");
            }
            auto src_i32 = emit_integer_from_any(os, src, 32, false);
            if (!src_i32.has_value()) {
                return fail(instr, "mov.b32 tuple unpack source unsupported");
            }
            const std::string lo16 = next_tmp("movlo");
            os << "  " << lo16 << " = trunc i32 " << *src_i32 << " to i16\n";
            const std::string hi_shift = next_tmp("movhi_sh");
            os << "  " << hi_shift << " = lshr i32 " << *src_i32 << ", 16\n";
            const std::string hi16 = next_tmp("movhi");
            os << "  " << hi16 << " = trunc i32 " << hi_shift << " to i16\n";
            if (!emit_store_reg_bits(os, parts[0], 16, lo16, 16)) return false;
            if (!emit_store_reg_bits(os, parts[1], 16, hi16, 16)) return false;
            return true;
        }

        if (!src.empty() && src.front() == '{') {
            if (!is_register_name(dst)) {
                return fail(instr, "mov tuple-pack destination must be register");
            }
            const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
            const int packed_bits = ty.bits;
            const std::vector<std::string> parts = split_comma_list(src);
            if ((packed_bits != 32 && packed_bits != 64) || parts.size() < 2 ||
                packed_bits % static_cast<int>(parts.size()) != 0) {
                return fail(instr, "mov tuple pack requires an evenly sized b32/b64 source tuple");
            }
            const int part_bits = packed_bits / static_cast<int>(parts.size());
            if (part_bits != 8 && part_bits != 16 && part_bits != 32) {
                return fail(instr, "mov tuple pack element width unsupported");
            }

            std::string packed;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                auto part = emit_integer_from_any(os, parts[i], part_bits, false);
                if (!part.has_value()) {
                    return fail(instr, "mov tuple pack source unsupported");
                }
                const std::string extended = next_tmp("movpack_ext");
                os << "  " << extended << " = zext " << llvm_int_type(part_bits) << " "
                   << *part << " to " << llvm_int_type(packed_bits) << "\n";
                std::string positioned = extended;
                const int shift_bits = static_cast<int>(i) * part_bits;
                if (shift_bits != 0) {
                    positioned = next_tmp("movpack_sh");
                    os << "  " << positioned << " = shl " << llvm_int_type(packed_bits)
                       << " " << extended << ", " << shift_bits << "\n";
                }
                if (packed.empty()) {
                    packed = positioned;
                } else {
                    const std::string combined = next_tmp("movpack_or");
                    os << "  " << combined << " = or " << llvm_int_type(packed_bits)
                       << " " << packed << ", " << positioned << "\n";
                    packed = combined;
                }
            }
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, packed, packed_bits);
        }

        if (!is_register_name(dst)) {
            return fail(instr, "mov destination must be register");
        }
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        int dst_bits = ensure_reg_slot(dst).bits;
        if (ty.kind == PtxTypeSpec::Kind::kFloat && (ty.bits == 32 || ty.bits == 64)) {
            if (ty.bits == 64 && uses_vf64_support()) {
                auto raw = decode_fp64_raw_bits(os, src);
                if (!raw) return fail(instr, "VF64 mov.f64 source unsupported");
                return emit_store_reg_bits(os, dst, dst_bits, *raw, 64);
            }
            if (ty.bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto pair = decode_fp64_pair(os, src);
                if (!pair) return fail(instr, "mov.f64 emulation source unsupported");
                return store_fp64_pair(os, dst, *pair);
            }
            if (auto fv = decode_float_operand(os, src, ty.bits)) {
                if (auto bits = encode_value_to_reg_bits(os, *fv, dst_bits)) {
                    return emit_store_reg_bits(os, dst, dst_bits, *bits, dst_bits);
                }
            }
            return fail(instr, "mov float source unsupported");
        }
        auto iv = emit_integer_from_any(os, src, std::max(dst_bits, ty.bits > 0 ? ty.bits : dst_bits),
                                        ty.is_signed);
        if (!iv.has_value()) {
            return fail(instr, "mov source unsupported: '" + src + "'");
        }
        const int src_bits = std::max(dst_bits, ty.bits > 0 ? ty.bits : dst_bits);
        if (resolve_param_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kParam;
        } else if (resolve_local_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kLocal;
            reg_local_origin_[dst] = src;
        } else if (resolve_threadgroup_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kShared;
        } else if (resolve_global_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kGlobal;
        } else if (resolve_const_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kParam;
        } else if (is_register_name(src) && reg_pointer_as_.count(src)) {
            reg_pointer_as_[dst] = reg_pointer_as_[src];
        }
        if (is_register_name(src) && reg_local_origin_.count(src)) {
            reg_local_origin_[dst] = reg_local_origin_[src];
        }
        return emit_store_reg_bits(os, dst, dst_bits, *iv, src_bits);
    }

    bool emit_cvta_instruction(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "cvta requires dest register and src");
        }
        const std::string& dst = instr.operands[0];
        const std::string& src = instr.operands[1];
        auto src_v = emit_integer_from_any(os, src, 64, false);
        if (!src_v.has_value()) {
            return fail(instr, "cvta source unsupported");
        }
        if (instr.opcode.find(".to.global") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kGlobal;
        } else if (instr.opcode.find(".to.const") != std::string::npos || instr.opcode.find(".to.param") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kParam;
        } else if (instr.opcode.find(".to.local") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kLocal;
        } else if (instr.opcode.find(".global") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kGlobal;
        } else if (instr.opcode.find(".local") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kLocal;
        } else if (instr.opcode.find(".shared") != std::string::npos) {
            reg_pointer_as_[dst] = PointerAs::kShared;
        } else if (is_register_name(src) && reg_pointer_as_.count(src)) {
            reg_pointer_as_[dst] = reg_pointer_as_[src];
        }
        if (is_register_name(src) && reg_local_origin_.count(src)) {
            reg_local_origin_[dst] = reg_local_origin_[src];
        }
        return emit_store_reg_bits(os, dst, 64, *src_v, 64);
    }

    bool emit_binary_int_op(std::ostringstream& os,
                            const cumetal::ptx::EntryFunction::Instruction& instr,
                            const std::string& llvm_op) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "binary op requires dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
        if (!a.has_value() || !b.has_value()) {
            return fail(instr, "binary op source unsupported");
        }
        const std::string out = next_tmp("bin");
        std::string op = llvm_op;
        if (llvm_op == "shr") {
            op = ty.is_signed ? "ashr" : "lshr";
        } else if (llvm_op == "div") {
            op = opcode_uses_float_math(instr.opcode) ? "fdiv" : (ty.is_signed ? "sdiv" : "udiv");
        } else if (llvm_op == "rem") {
            op = opcode_uses_float_math(instr.opcode) ? "frem" : (ty.is_signed ? "srem" : "urem");
        }
        os << "  " << out << " = " << op << " " << llvm_int_type(bits) << " " << *a << ", " << *b << "\n";
        if (reg_pointer_as_.count(instr.operands[1]) && (opcode_root(instr.opcode) == "add" || opcode_root(instr.opcode) == "sub")) {
            reg_pointer_as_[dst] = reg_pointer_as_[instr.operands[1]];
            if (reg_local_origin_.count(instr.operands[1])) {
                reg_local_origin_[dst] = reg_local_origin_[instr.operands[1]];
            }
        } else if (reg_pointer_as_.count(instr.operands[2]) && (opcode_root(instr.opcode) == "add")) {
            reg_pointer_as_[dst] = reg_pointer_as_[instr.operands[2]];
            if (reg_local_origin_.count(instr.operands[2])) {
                reg_local_origin_[dst] = reg_local_origin_[instr.operands[2]];
            }
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, out, bits);
    }

    bool emit_binary_float_op(std::ostringstream& os,
                              const cumetal::ptx::EntryFunction::Instruction& instr,
                              const std::string& llvm_op) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "float binary op requires dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        if (instr.opcode.find(".f16x2") != std::string::npos) {
            auto a = emit_integer_from_any(os, instr.operands[1], 32, false);
            auto b = emit_integer_from_any(os, instr.operands[2], 32, false);
            if (!a || !b) return fail(instr, "f16x2 binary source unsupported");
            const std::string av = next_tmp("f16x2_a");
            const std::string bv = next_tmp("f16x2_b");
            const std::string result = next_tmp("f16x2_result");
            const std::string packed = next_tmp("f16x2_packed");
            os << "  " << av << " = bitcast i32 " << *a << " to <2 x half>\n";
            os << "  " << bv << " = bitcast i32 " << *b << " to <2 x half>\n";
            os << "  " << result << " = " << llvm_op << " <2 x half> "
               << av << ", " << bv << "\n";
            os << "  " << packed << " = bitcast <2 x half> " << result << " to i32\n";
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, packed, 32);
        }
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind != PtxTypeSpec::Kind::kFloat) {
            return fail(instr, "float op without float suffix");
        }

        // FP64 emulation: decompose to FP32 Dekker pairs when --fp64=emulate
        if (ty.bits == 64 && uses_vf64_support()) {
            if (llvm_op == "fadd") return emit_vf64_binary_call(os, instr, "add");
            if (llvm_op == "fsub") return emit_vf64_binary_call(os, instr, "sub");
            if (llvm_op == "fmul") return emit_vf64_binary_call(os, instr, "mul");
            if (llvm_op == "fdiv") return emit_vf64_binary_call(os, instr, "div");
            return fail(instr, "unsupported VF64 binary operation");
        }
        if (ty.bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
            auto a = decode_fp64_pair(os, instr.operands[1]);
            auto b = decode_fp64_pair(os, instr.operands[2]);
            if (!a || !b) return fail(instr, "fp64 emulation source unsupported");
            Fp64Pair result;
            if (llvm_op == "fadd") result = emit_fp64_pair_add(os, *a, *b);
            else if (llvm_op == "fsub") result = emit_fp64_pair_add(os, *a, *b, true);
            else if (llvm_op == "fmul") result = emit_fp64_pair_mul(os, *a, *b);
            else if (llvm_op == "fdiv") result = emit_fp64_pair_div(os, *a, *b);
            else return fail(instr, "unsupported fp64 emulation binary operation");
            return store_fp64_pair(os, dst, result);
        }

        auto a = decode_float_operand(os, instr.operands[1], ty.bits);
        auto b = decode_float_operand(os, instr.operands[2], ty.bits);
        if (!a.has_value() || !b.has_value()) {
            return fail(instr, "float op source unsupported");
        }
        const std::string out = next_tmp("fbin");
        os << "  " << out << " = " << llvm_op << " " << llvm_float_type(ty.bits)
           << " " << a->ir << ", " << b->ir << "\n";
        Value v;
        v.ir = out;
        v.type = ty;
        v.bits = ty.bits;
        auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
        if (!bitsv.has_value()) {
            return fail(instr, "float op result encode failed");
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_mul(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (opcode_uses_float_math(instr.opcode)) {
            return emit_binary_float_op(os, instr, "fmul");
        }
        if (instr.opcode.find(".hi.") == std::string::npos) {
            return emit_binary_int_op(os, instr, "mul");
        }
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "mul.hi requires dst, a, b");
        }

        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        if (bits <= 0 || bits > 64) {
            return fail(instr, "mul.hi only supports <=64-bit integers");
        }

        auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
        if (!a || !b) {
            return fail(instr, "mul.hi source unsupported");
        }

        if (bits == 64) {
            if (ty.is_signed) {
                return fail(instr, "mul.hi.s64 is not yet supported");
            }
            const std::string a_lo = next_tmp("mulhi_a_lo");
            const std::string b_lo = next_tmp("mulhi_b_lo");
            const std::string a_hi_sh = next_tmp("mulhi_a_hi_sh");
            const std::string b_hi_sh = next_tmp("mulhi_b_hi_sh");
            const std::string a_hi = next_tmp("mulhi_a_hi");
            const std::string b_hi = next_tmp("mulhi_b_hi");
            os << "  " << a_lo << " = trunc i64 " << *a << " to i32\n";
            os << "  " << b_lo << " = trunc i64 " << *b << " to i32\n";
            os << "  " << a_hi_sh << " = lshr i64 " << *a << ", 32\n";
            os << "  " << b_hi_sh << " = lshr i64 " << *b << ", 32\n";
            os << "  " << a_hi << " = trunc i64 " << a_hi_sh << " to i32\n";
            os << "  " << b_hi << " = trunc i64 " << b_hi_sh << " to i32\n";

            const std::string a_lo64 = next_tmp("mulhi_a_lo64");
            const std::string b_lo64 = next_tmp("mulhi_b_lo64");
            const std::string a_hi64 = next_tmp("mulhi_a_hi64");
            const std::string b_hi64 = next_tmp("mulhi_b_hi64");
            os << "  " << a_lo64 << " = zext i32 " << a_lo << " to i64\n";
            os << "  " << b_lo64 << " = zext i32 " << b_lo << " to i64\n";
            os << "  " << a_hi64 << " = zext i32 " << a_hi << " to i64\n";
            os << "  " << b_hi64 << " = zext i32 " << b_hi << " to i64\n";

            const std::string p0 = next_tmp("mulhi_p0");
            const std::string p1 = next_tmp("mulhi_p1");
            const std::string p2 = next_tmp("mulhi_p2");
            const std::string p3 = next_tmp("mulhi_p3");
            os << "  " << p0 << " = mul i64 " << a_lo64 << ", " << b_lo64 << "\n";
            os << "  " << p1 << " = mul i64 " << a_lo64 << ", " << b_hi64 << "\n";
            os << "  " << p2 << " = mul i64 " << a_hi64 << ", " << b_lo64 << "\n";
            os << "  " << p3 << " = mul i64 " << a_hi64 << ", " << b_hi64 << "\n";

            const std::string carry0 = next_tmp("mulhi_c0");
            os << "  " << carry0 << " = lshr i64 " << p0 << ", 32\n";
            const std::string sum12 = next_tmp("mulhi_s12");
            os << "  " << sum12 << " = add i64 " << p1 << ", " << p2 << "\n";
            const std::string ov1 = next_tmp("mulhi_ov1");
            os << "  " << ov1 << " = icmp ult i64 " << sum12 << ", " << p1 << "\n";
            const std::string ov1i = next_tmp("mulhi_ov1i");
            os << "  " << ov1i << " = zext i1 " << ov1 << " to i64\n";

            const std::string sum = next_tmp("mulhi_sum");
            os << "  " << sum << " = add i64 " << sum12 << ", " << carry0 << "\n";
            const std::string ov2 = next_tmp("mulhi_ov2");
            os << "  " << ov2 << " = icmp ult i64 " << sum << ", " << sum12 << "\n";
            const std::string ov2i = next_tmp("mulhi_ov2i");
            os << "  " << ov2i << " = zext i1 " << ov2 << " to i64\n";

            const std::string carry = next_tmp("mulhi_carry");
            os << "  " << carry << " = add i64 " << ov1i << ", " << ov2i << "\n";
            const std::string mid_hi = next_tmp("mulhi_mid_hi");
            os << "  " << mid_hi << " = lshr i64 " << sum << ", 32\n";
            const std::string carry_sh = next_tmp("mulhi_carry_sh");
            os << "  " << carry_sh << " = shl i64 " << carry << ", 32\n";

            const std::string hi0 = next_tmp("mulhi_hi0");
            os << "  " << hi0 << " = add i64 " << p3 << ", " << mid_hi << "\n";
            const std::string hi = next_tmp("mulhi_hi");
            os << "  " << hi << " = add i64 " << hi0 << ", " << carry_sh << "\n";
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, hi, 64);
        }

        const int wide_bits = bits * 2;
        const std::string wide_ty = llvm_int_type(wide_bits);
        const std::string a_w = next_tmp("mulhi_a");
        const std::string b_w = next_tmp("mulhi_b");
        os << "  " << a_w << " = " << (ty.is_signed ? "sext " : "zext ")
           << llvm_int_type(bits) << " " << *a << " to " << wide_ty << "\n";
        os << "  " << b_w << " = " << (ty.is_signed ? "sext " : "zext ")
           << llvm_int_type(bits) << " " << *b << " to " << wide_ty << "\n";

        const std::string prod = next_tmp("mulhi_prod");
        os << "  " << prod << " = mul " << wide_ty << " " << a_w << ", " << b_w << "\n";
        const std::string shr = next_tmp("mulhi_shr");
        os << "  " << shr << " = " << (ty.is_signed ? "ashr " : "lshr ")
           << wide_ty << " " << prod << ", " << bits << "\n";
        const std::string hi = next_tmp("mulhi_hi");
        os << "  " << hi << " = trunc " << wide_ty << " " << shr << " to " << llvm_int_type(bits) << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, hi, bits);
    }

    bool emit_mad_or_fma(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "mad/fma requires dst, a, b, c");
        }
        const std::string& dst = instr.operands[0];
        if (instr.opcode.find(".f16x2") != std::string::npos) {
            auto a = emit_integer_from_any(os, instr.operands[1], 32, false);
            auto b = emit_integer_from_any(os, instr.operands[2], 32, false);
            auto c = emit_integer_from_any(os, instr.operands[3], 32, false);
            if (!a || !b || !c) return fail(instr, "f16x2 fma source unsupported");
            const std::string av = next_tmp("f16x2_fma_a");
            const std::string bv = next_tmp("f16x2_fma_b");
            const std::string cv = next_tmp("f16x2_fma_c");
            const std::string product = next_tmp("f16x2_fma_product");
            const std::string result = next_tmp("f16x2_fma_result");
            const std::string packed = next_tmp("f16x2_fma_packed");
            os << "  " << av << " = bitcast i32 " << *a << " to <2 x half>\n";
            os << "  " << bv << " = bitcast i32 " << *b << " to <2 x half>\n";
            os << "  " << cv << " = bitcast i32 " << *c << " to <2 x half>\n";
            os << "  " << product << " = fmul <2 x half> " << av << ", " << bv << "\n";
            os << "  " << result << " = fadd <2 x half> " << product << ", " << cv << "\n";
            os << "  " << packed << " = bitcast <2 x half> " << result << " to i32\n";
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, packed, 32);
        }
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (opcode_uses_float_math(instr.opcode)) {
            const int bits = (ty.bits == 64) ? 64 : 32;
            if (bits == 64 && uses_vf64_support()) {
                auto a = decode_fp64_raw_bits(os, instr.operands[1]);
                auto b = decode_fp64_raw_bits(os, instr.operands[2]);
                auto c = decode_fp64_raw_bits(os, instr.operands[3]);
                if (!a || !b || !c) return fail(instr, "VF64 fma source unsupported");
                const int rounding = vf64_rounding_mode(instr.opcode);
                std::string function;
                std::string arguments;
                if (fp64_mode_ == cumetal::ptx::Fp64Mode::kWide48) {
                    if (rounding != 0) {
                        return fail(instr, "wide48 supports round-to-nearest-even fma only");
                    }
                    function = "vf64_wide_fma";
                    declarations_.insert("declare i64 @vf64_wide_fma(i64, i64, i64)");
                    arguments = "i64 " + *a + ", i64 " + *b + ", i64 " + *c;
                } else {
                    function = "vf64_fma_round";
                    declarations_.insert("declare i64 @vf64_fma_round(i64, i64, i64, i32)");
                    arguments = "i64 " + *a + ", i64 " + *b + ", i64 " + *c +
                                ", i32 " + std::to_string(rounding);
                }
                const std::string result = next_tmp("vf64_fma");
                os << "  " << result << " = call i64 @" << function << "("
                   << arguments << ")\n";
                return emit_store_reg_bits(
                    os, dst, ensure_reg_slot(dst).bits, result, 64
                );
            }
            if (bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto a = decode_fp64_pair(os, instr.operands[1]);
                auto b = decode_fp64_pair(os, instr.operands[2]);
                auto c = decode_fp64_pair(os, instr.operands[3]);
                if (!a || !b || !c) return fail(instr, "fp64 fma emulation source unsupported");
                const Fp64Pair product = emit_fp64_pair_mul(os, *a, *b);
                return store_fp64_pair(os, dst, emit_fp64_pair_add(os, product, *c));
            }
            auto a = decode_float_operand(os, instr.operands[1], bits);
            auto b = decode_float_operand(os, instr.operands[2], bits);
            auto c = decode_float_operand(os, instr.operands[3], bits);
            if (!a || !b || !c) return fail(instr, "mad/fma float source unsupported");

            const std::string mul = next_tmp("fmul");
            const std::string add = next_tmp("fadd");
            os << "  " << mul << " = fmul " << llvm_float_type(bits) << " " << a->ir << ", " << b->ir << "\n";
            os << "  " << add << " = fadd " << llvm_float_type(bits) << " " << mul << ", " << c->ir << "\n";
            Value v{.ir = add, .type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = bits}, .bits = bits};
            auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
            if (!bitsv) return fail(instr, "mad/fma float encode failed");
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
        }
        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
        auto c = emit_integer_from_any(os, instr.operands[3], bits, ty.is_signed);
        if (!a || !b || !c) return fail(instr, "mad int source unsupported");
        const std::string mul = next_tmp("imul");
        const std::string add = next_tmp("iadd");
        os << "  " << mul << " = mul " << llvm_int_type(bits) << " " << *a << ", " << *b << "\n";
        os << "  " << add << " = add " << llvm_int_type(bits) << " " << mul << ", " << *c << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, add, bits);
    }

    bool emit_minmax(std::ostringstream& os,
                     const cumetal::ptx::EntryFunction::Instruction& instr,
                     bool is_min) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "min/max requires dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);

        if (opcode_uses_float_math(instr.opcode)) {
            const int bits = (ty.bits > 0) ? ty.bits : 32;
            if (bits == 64 && uses_vf64_support()) {
                auto raw = decode_fp64_raw_bits(os, instr.operands[1]);
                if (!raw) return fail(instr, "VF64 neg source unsupported");
                const std::string result = next_tmp("vf64_neg");
                os << "  " << result << " = xor i64 " << *raw
                   << ", -9223372036854775808\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, 64);
            }
            if (bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto a = decode_fp64_pair(os, instr.operands[1]);
                auto b = decode_fp64_pair(os, instr.operands[2]);
                if (!a || !b) return fail(instr, "fp64 min/max emulation source unsupported");
                const std::string hi_cmp = next_tmp("fp64_minmax_hi");
                const std::string hi_eq = next_tmp("fp64_minmax_hieq");
                const std::string lo_cmp = next_tmp("fp64_minmax_lo");
                const std::string tie_cmp = next_tmp("fp64_minmax_tie");
                const std::string choose_a = next_tmp("fp64_minmax_choose");
                os << "  " << hi_cmp << " = fcmp " << (is_min ? "olt" : "ogt")
                   << " float " << a->hi << ", " << b->hi << "\n";
                os << "  " << hi_eq << " = fcmp oeq float " << a->hi << ", " << b->hi << "\n";
                os << "  " << lo_cmp << " = fcmp " << (is_min ? "olt" : "ogt")
                   << " float " << a->lo << ", " << b->lo << "\n";
                os << "  " << tie_cmp << " = and i1 " << hi_eq << ", " << lo_cmp << "\n";
                os << "  " << choose_a << " = or i1 " << hi_cmp << ", " << tie_cmp << "\n";
                const std::string hi = next_tmp("fp64_minmax_sel_hi");
                const std::string lo = next_tmp("fp64_minmax_sel_lo");
                os << "  " << hi << " = select i1 " << choose_a << ", float "
                   << a->hi << ", float " << b->hi << "\n";
                os << "  " << lo << " = select i1 " << choose_a << ", float "
                   << a->lo << ", float " << b->lo << "\n";
                return store_fp64_pair(os, dst, Fp64Pair{hi, lo});
            }
            auto a = decode_float_operand(os, instr.operands[1], bits);
            auto b = decode_float_operand(os, instr.operands[2], bits);
            if (!a || !b) return fail(instr, "min/max float source unsupported");
            const std::string fty = (bits == 64) ? "double" : (bits == 16 ? "half" : "float");
            const std::string cmp = next_tmp(is_min ? "fmin_cmp" : "fmax_cmp");
            const std::string sel = next_tmp(is_min ? "fmin_sel" : "fmax_sel");
            os << "  " << cmp << " = fcmp " << (is_min ? "olt" : "ogt") << " "
               << fty << " " << a->ir << ", " << b->ir << "\n";
            os << "  " << sel << " = select i1 " << cmp << ", " << fty << " " << a->ir
               << ", " << fty << " " << b->ir << "\n";
            Value v{.ir = sel, .type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = bits}, .bits = bits};
            auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
            if (!bitsv) return fail(instr, "min/max float encode failed");
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
        }

        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
        if (!a || !b) return fail(instr, "min/max int source unsupported");
        const std::string cmp = next_tmp(is_min ? "imin_cmp" : "imax_cmp");
        const std::string sel = next_tmp(is_min ? "imin_sel" : "imax_sel");
        const std::string cc = is_min ? (ty.is_signed ? "slt" : "ult")
                                      : (ty.is_signed ? "sgt" : "ugt");
        os << "  " << cmp << " = icmp " << cc << " " << llvm_int_type(bits)
           << " " << *a << ", " << *b << "\n";
        os << "  " << sel << " = select i1 " << cmp << ", " << llvm_int_type(bits) << " "
           << *a << ", " << llvm_int_type(bits) << " " << *b << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, sel, bits);
    }

    bool emit_neg(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "neg requires dst, src");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (opcode_uses_float_math(instr.opcode)) {
            const int bits = (ty.bits > 0) ? ty.bits : 32;
            if (bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto pair = decode_fp64_pair(os, instr.operands[1]);
                if (!pair) return fail(instr, "fp64 neg emulation source unsupported");
                const std::string hi = next_tmp("fp64_neg_hi");
                const std::string lo = next_tmp("fp64_neg_lo");
                os << "  " << hi << " = fneg float " << pair->hi << "\n";
                os << "  " << lo << " = fneg float " << pair->lo << "\n";
                return store_fp64_pair(os, dst, Fp64Pair{hi, lo});
            }
            auto a = decode_float_operand(os, instr.operands[1], bits);
            if (!a) return fail(instr, "neg float source unsupported");
            const std::string out = next_tmp("fneg");
            os << "  " << out << " = fneg " << llvm_float_type(a->type.bits) << " " << a->ir << "\n";
            Value v{.ir = out, .type = a->type, .bits = a->bits};
            auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
            if (!bitsv) return fail(instr, "neg float encode failed");
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
        }
        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, true);
        if (!a) return fail(instr, "neg int source unsupported");
        const std::string out = next_tmp("ineg");
        os << "  " << out << " = sub " << llvm_int_type(bits) << " 0, " << *a << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, out, bits);
    }

    bool emit_not(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "not requires dst, src");
        }
        const std::string& dst = instr.operands[0];
        const int bits = ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, false);
        if (!a) return fail(instr, "not source unsupported");
        // PTX `not` is bitwise complement: dst = ~src = xor(src, all-ones)
        const std::string out = next_tmp("bnot");
        os << "  " << out << " = xor " << llvm_int_type(bits) << " " << *a << ", -1\n";
        return emit_store_reg_bits(os, dst, bits, out, bits);
    }

    bool emit_rcp(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "rcp requires dst, src");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind != PtxTypeSpec::Kind::kFloat || ty.bits != 32) {
            return fail(instr, "only rcp.f32 currently supported");
        }
        auto a = decode_float_operand(os, instr.operands[1], 32);
        if (!a) return fail(instr, "rcp source unsupported");
        const std::string out = next_tmp("rcp");
        os << "  " << out << " = fdiv float 1.000000e+00, " << a->ir << "\n";
        Value v{.ir = out, .type = ty, .bits = 32};
        auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
        if (!bitsv) return fail(instr, "rcp encode failed");
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_cvt(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "cvt requires dst, src");
        }
        const std::string& dst = instr.operands[0];

        // PTX: cvt.rn.f16x2.f32 dst, a, b
        // Pack two f32 values into one 32-bit register carrying two IEEE fp16 lanes.
        if (instr.opcode.find("f16x2.f32") != std::string::npos) {
            if (instr.operands.size() < 3) {
                return fail(instr, "cvt.f16x2.f32 requires dst, a, b");
            }
            auto a = decode_float_operand(os, instr.operands[1], 32);
            auto b = decode_float_operand(os, instr.operands[2], 32);
            if (!a || !b) {
                return fail(instr, "cvt.f16x2.f32 sources unsupported");
            }
            const std::string a_h = next_tmp("cvtf16x2_a_h");
            const std::string b_h = next_tmp("cvtf16x2_b_h");
            os << "  " << a_h << " = fptrunc float " << a->ir << " to half\n";
            os << "  " << b_h << " = fptrunc float " << b->ir << " to half\n";
            const std::string a_i16 = next_tmp("cvtf16x2_a_i16");
            const std::string b_i16 = next_tmp("cvtf16x2_b_i16");
            os << "  " << a_i16 << " = bitcast half " << a_h << " to i16\n";
            os << "  " << b_i16 << " = bitcast half " << b_h << " to i16\n";
            const std::string lo_i32 = next_tmp("cvtf16x2_lo");
            const std::string hi_i32 = next_tmp("cvtf16x2_hi");
            // PTX f16x2 packs operand 1 into high lane and operand 2 into low lane.
            os << "  " << lo_i32 << " = zext i16 " << b_i16 << " to i32\n";
            os << "  " << hi_i32 << " = zext i16 " << a_i16 << " to i32\n";
            const std::string hi_sh = next_tmp("cvtf16x2_hish");
            os << "  " << hi_sh << " = shl i32 " << hi_i32 << ", 16\n";
            const std::string packed = next_tmp("cvtf16x2_pack");
            os << "  " << packed << " = or i32 " << lo_i32 << ", " << hi_sh << "\n";
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, packed, 32);
        }

        const ParsedCvtTypes cvt = parse_cvt_types(instr.opcode);
        if (!cvt.ok) {
            return fail(instr, "unable to parse cvt types");
        }
        const std::string& src = instr.operands[1];

        const bool converts_fp64 =
            (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64) ||
            (cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64);
        if (uses_vf64_support() && converts_fp64) {
            const int rounding = vf64_rounding_mode(instr.opcode);
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64 &&
                cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64) {
                auto raw = decode_fp64_raw_bits(os, src);
                if (!raw) return fail(instr, "VF64 conversion source unsupported");
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *raw, 64);
            }
            if (cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64 &&
                cvt.src.kind == PtxTypeSpec::Kind::kFloat &&
                (cvt.src.bits == 16 || cvt.src.bits == 32)) {
                auto value = decode_float_operand(os, src, cvt.src.bits);
                if (!value) return fail(instr, "float-to-VF64 conversion source unsupported");
                const std::string raw = next_tmp("vf64_from_float_raw");
                os << "  " << raw << " = bitcast "
                   << (cvt.src.bits == 16 ? "half" : "float") << " " << value->ir
                   << " to " << llvm_int_type(cvt.src.bits) << "\n";
                const std::string function = cvt.src.bits == 16
                                                 ? "vf64_f16_to_f64"
                                                 : "vf64_f32_to_f64";
                declarations_.insert("declare i64 @" + function + "(" +
                                     llvm_int_type(cvt.src.bits) + ")");
                const std::string converted = next_tmp("vf64_from_float");
                os << "  " << converted << " = call i64 @" << function << "("
                   << llvm_int_type(cvt.src.bits) << " " << raw << ")\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits,
                                           converted, 64);
            }
            if (cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64 &&
                cvt.src.kind == PtxTypeSpec::Kind::kInt &&
                (cvt.src.bits == 32 || cvt.src.bits == 64)) {
                auto value = emit_integer_from_any(os, src, cvt.src.bits,
                                                   cvt.src.is_signed);
                if (!value) return fail(instr, "integer-to-VF64 conversion source unsupported");
                const std::string function = std::string("vf64_") +
                    (cvt.src.is_signed ? "i" : "ui") +
                    std::to_string(cvt.src.bits) + "_to_f64";
                declarations_.insert("declare i64 @" + function + "(" +
                                     llvm_int_type(cvt.src.bits) + ", i32)");
                const std::string converted = next_tmp("vf64_from_int");
                os << "  " << converted << " = call i64 @" << function << "("
                   << llvm_int_type(cvt.src.bits) << " " << *value << ", i32 "
                   << rounding << ")\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits,
                                           converted, 64);
            }
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64 &&
                cvt.dst.kind == PtxTypeSpec::Kind::kFloat &&
                (cvt.dst.bits == 16 || cvt.dst.bits == 32)) {
                auto raw = decode_fp64_raw_bits(os, src);
                if (!raw) return fail(instr, "VF64-to-float conversion source unsupported");
                const std::string function = cvt.dst.bits == 16
                                                 ? "vf64_f64_to_f16"
                                                 : "vf64_f64_to_f32";
                declarations_.insert("declare " + llvm_int_type(cvt.dst.bits) +
                                     " @" + function + "(i64, i32)");
                const std::string converted = next_tmp("vf64_to_float");
                os << "  " << converted << " = call " << llvm_int_type(cvt.dst.bits)
                   << " @" << function << "(i64 " << *raw << ", i32 " << rounding
                   << ")\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits,
                                           converted, cvt.dst.bits);
            }
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64 &&
                cvt.dst.kind == PtxTypeSpec::Kind::kInt &&
                (cvt.dst.bits == 32 || cvt.dst.bits == 64)) {
                auto raw = decode_fp64_raw_bits(os, src);
                if (!raw) return fail(instr, "VF64-to-integer conversion source unsupported");
                const std::string function = std::string("vf64_f64_to_") +
                    (cvt.dst.is_signed ? "i" : "ui") + std::to_string(cvt.dst.bits);
                declarations_.insert("declare " + llvm_int_type(cvt.dst.bits) +
                                     " @" + function + "(i64, i32, i1)");
                const std::string converted = next_tmp("vf64_to_int");
                os << "  " << converted << " = call " << llvm_int_type(cvt.dst.bits)
                   << " @" << function << "(i64 " << *raw << ", i32 " << rounding
                   << ", i1 true)\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits,
                                           converted, cvt.dst.bits);
            }
            return fail(instr, "this VF64 conversion is not supported");
        }
        if (fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate && converts_fp64) {
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64 &&
                cvt.dst.kind == PtxTypeSpec::Kind::kFloat) {
                auto pair = decode_fp64_pair(os, src);
                if (!pair) return fail(instr, "fp64 conversion source unsupported");
                if (cvt.dst.bits == 64) return store_fp64_pair(os, dst, *pair);
                if (cvt.dst.bits != 32) {
                    return fail(instr, "FP32-pair emulation currently converts fp64 only to fp32");
                }
                const std::string rounded = next_tmp("fp64_to_f32");
                os << "  " << rounded << " = fadd float " << pair->hi << ", " << pair->lo << "\n";
                Value result{.ir = rounded,
                             .type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 32},
                             .bits = 32};
                auto bits = encode_value_to_reg_bits(os, result, ensure_reg_slot(dst).bits);
                if (!bits) return fail(instr, "fp64-to-f32 conversion encode failed");
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bits,
                                           ensure_reg_slot(dst).bits);
            }
            if (cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64 &&
                cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 32) {
                auto value = decode_float_operand(os, src, 32);
                if (!value) return fail(instr, "f32-to-fp64 conversion source unsupported");
                const std::string zero = emit_float_constant(os, 0.0f, "fp64_cvt_zero");
                return store_fp64_pair(os, dst, Fp64Pair{value->ir, zero});
            }
            if (cvt.dst.kind == PtxTypeSpec::Kind::kFloat && cvt.dst.bits == 64 &&
                cvt.src.kind == PtxTypeSpec::Kind::kInt && cvt.src.bits <= 32) {
                auto value = decode_integer_operand(os, src, cvt.src.bits,
                                                    cvt.src.is_signed);
                if (!value) {
                    auto special = emit_integer_from_any(os, src, cvt.src.bits,
                                                         cvt.src.is_signed);
                    if (!special) {
                        return fail(instr, "integer-to-fp64 conversion source unsupported");
                    }
                    value = Value{.ir = *special, .type = cvt.src,
                                  .bits = cvt.src.bits};
                }
                const std::string converted = next_tmp("int_to_fp64_hi");
                os << "  " << converted << " = "
                   << (cvt.src.is_signed ? "sitofp " : "uitofp ")
                   << llvm_int_type(cvt.src.bits) << " " << value->ir
                   << " to float\n";
                const std::string zero = emit_float_constant(os, 0.0f, "fp64_cvt_zero");
                return store_fp64_pair(os, dst, Fp64Pair{converted, zero});
            }
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && cvt.src.bits == 64 &&
                cvt.dst.kind == PtxTypeSpec::Kind::kInt && cvt.dst.bits <= 32) {
                auto pair = decode_fp64_pair(os, src);
                if (!pair) return fail(instr, "fp64-to-integer conversion source unsupported");
                const std::string combined = next_tmp("fp64_to_int_value");
                os << "  " << combined << " = fadd float " << pair->hi << ", "
                   << pair->lo << "\n";
                const std::string converted = next_tmp("fp64_to_int");
                os << "  " << converted << " = "
                   << (cvt.dst.is_signed ? "fptosi float " : "fptoui float ")
                   << combined << " to " << llvm_int_type(cvt.dst.bits) << "\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits,
                                           converted, cvt.dst.bits);
            }
            return fail(instr,
                        "this fp64 conversion is not supported by FP32-pair emulation");
        }

        Value src_value;
        if (cvt.src.kind == PtxTypeSpec::Kind::kFloat) {
            auto fv = decode_float_operand(os, src, cvt.src.bits);
            if (!fv && is_register_name(src)) {
                // Float source in a wider integer register (e.g., cvt.f32.f16 with %r src).
                // Load the raw bits, truncate to the source width, then bitcast to float.
                const std::string raw = emit_load_reg_bits(os, src);
                const int slot_bits = ensure_reg_slot(src).bits;
                std::string trunc_bits = raw;
                if (slot_bits > cvt.src.bits) {
                    trunc_bits = next_tmp("cvt_srct");
                    os << "  " << trunc_bits << " = trunc " << llvm_int_type(slot_bits) << " "
                       << raw << " to " << llvm_int_type(cvt.src.bits) << "\n";
                }
                const std::string fty_src = (cvt.src.bits == 16) ? "half"
                                          : (cvt.src.bits == 32) ? "float" : "double";
                const std::string bc = next_tmp("cvt_srcbc");
                os << "  " << bc << " = bitcast " << llvm_int_type(cvt.src.bits) << " "
                   << trunc_bits << " to " << fty_src << "\n";
                src_value.ir = bc;
                src_value.type = cvt.src;
                src_value.bits = cvt.src.bits;
            } else {
                if (!fv) return fail(instr, "cvt float source unsupported");
                src_value = *fv;
            }
        } else if (cvt.src.kind == PtxTypeSpec::Kind::kInt) {
            auto iv = decode_integer_operand(os, src, cvt.src.bits, cvt.src.is_signed);
            if (!iv) {
                if (auto special = emit_integer_from_any(os, src, cvt.src.bits, cvt.src.is_signed)) {
                    src_value.ir = *special;
                    src_value.type = cvt.src;
                    src_value.bits = cvt.src.bits;
                } else {
                    return fail(instr, "cvt int source unsupported");
                }
            } else {
                src_value = *iv;
            }
        } else if (cvt.src.kind == PtxTypeSpec::Kind::kPred) {
            const std::string raw = emit_load_reg_bits(os, src, 1);
            src_value.ir = raw;
            src_value.type = cvt.src;
            src_value.bits = 1;
        } else {
            return fail(instr, "unsupported cvt source kind");
        }

        Value dst_value;
        dst_value.type = cvt.dst;
        dst_value.bits = cvt.dst.bits;

        // PTX integer-rounding modes (the ones ending in `i`) round a float to
        // an integral value. Ignoring them made `cvt.rni.f32.f32` -- what clang
        // emits for rintf -- a plain register copy, so rint/floor/ceil silently
        // returned their argument unrounded. Apply the rounding to the source
        // before any width conversion; for a float->int destination the later
        // fptosi/fptoui truncation is then already correct.
        if (cvt.src.kind == PtxTypeSpec::Kind::kFloat && !src_value.ir.empty()) {
            const std::string fty_src = (cvt.src.bits == 16)   ? "half"
                                        : (cvt.src.bits == 32) ? "float"
                                                               : "double";
            const char* round_fn = nullptr;
            if (instr.opcode.find(".rni.") != std::string::npos) round_fn = "llvm.rint";
            else if (instr.opcode.find(".rmi.") != std::string::npos) round_fn = "llvm.floor";
            else if (instr.opcode.find(".rpi.") != std::string::npos) round_fn = "llvm.ceil";
            else if (instr.opcode.find(".rzi.") != std::string::npos) round_fn = "llvm.trunc";
            if (round_fn != nullptr) {
                const std::string suffix =
                    (cvt.src.bits == 16) ? ".f16" : (cvt.src.bits == 32 ? ".f32" : ".f64");
                declarations_.insert("declare " + fty_src + " @" + round_fn + suffix + "(" +
                                     fty_src + ")");
                const std::string r = next_tmp("cvtrnd");
                os << "  " << r << " = call " << fty_src << " @" << round_fn << suffix << "("
                   << fty_src << " " << src_value.ir << ")\n";
                src_value.ir = r;
            }
        }

        if (cvt.dst.kind == PtxTypeSpec::Kind::kFloat) {
            const std::string fty =
                (cvt.dst.bits == 16) ? "half" : (cvt.dst.bits == 32 ? "float" : (cvt.dst.bits == 64 ? "double" : ""));
            if (fty.empty()) return fail(instr, "unsupported cvt float dst width");
            if (cvt.src.kind == PtxTypeSpec::Kind::kFloat) {
                if (cvt.src.bits == cvt.dst.bits) {
                    dst_value.ir = src_value.ir;
                } else if (cvt.src.bits < cvt.dst.bits) {
                    const std::string t = next_tmp("fpext");
                    os << "  " << t << " = fpext " << (cvt.src.bits == 16 ? "half" : "float") << " " << src_value.ir
                       << " to " << fty << "\n";
                    dst_value.ir = t;
                } else {
                    const std::string t = next_tmp("fptrunc");
                    os << "  " << t << " = fptrunc " << (cvt.src.bits == 64 ? "double" : "float") << " "
                       << src_value.ir << " to " << fty << "\n";
                    dst_value.ir = t;
                }
            } else if (cvt.src.kind == PtxTypeSpec::Kind::kInt) {
                const std::string t = next_tmp("itofp");
                os << "  " << t << " = " << (cvt.src.is_signed ? "sitofp " : "uitofp ")
                   << llvm_int_type(cvt.src.bits) << " " << src_value.ir << " to " << fty << "\n";
                dst_value.ir = t;
            } else {
                return fail(instr, "unsupported cvt to float");
            }
        } else if (cvt.dst.kind == PtxTypeSpec::Kind::kInt) {
            if (cvt.src.kind == PtxTypeSpec::Kind::kInt) {
                std::string t = src_value.ir;
                if (cvt.src.bits < cvt.dst.bits) {
                    const std::string ext = next_tmp("intcvt_ext");
                    os << "  " << ext << " = " << (cvt.src.is_signed ? "sext " : "zext ")
                       << llvm_int_type(cvt.src.bits) << " " << t << " to " << llvm_int_type(cvt.dst.bits) << "\n";
                    t = ext;
                } else if (cvt.src.bits > cvt.dst.bits) {
                    const std::string tr = next_tmp("intcvt_tr");
                    os << "  " << tr << " = trunc " << llvm_int_type(cvt.src.bits) << " " << t << " to "
                       << llvm_int_type(cvt.dst.bits) << "\n";
                    t = tr;
                }
                dst_value.ir = t;
            } else if (cvt.src.kind == PtxTypeSpec::Kind::kFloat) {
                const std::string t = next_tmp("fptoi");
                os << "  " << t << " = " << (cvt.dst.is_signed ? "fptosi " : "fptoui ")
                   << (cvt.src.bits == 32 ? "float " : "double ") << src_value.ir << " to "
                   << llvm_int_type(cvt.dst.bits) << "\n";
                dst_value.ir = t;
            } else if (cvt.src.kind == PtxTypeSpec::Kind::kPred) {
                const std::string t = next_tmp("pred2int");
                os << "  " << t << " = zext i1 " << src_value.ir << " to " << llvm_int_type(cvt.dst.bits) << "\n";
                dst_value.ir = t;
            } else {
                return fail(instr, "unsupported cvt to int");
            }
        } else {
            return fail(instr, "unsupported cvt destination kind");
        }

        auto bitsv = encode_value_to_reg_bits(os, dst_value, ensure_reg_slot(dst).bits);
        if (!bitsv) return fail(instr, "cvt encode failed");
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_setp(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "setp requires pred dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        std::string cmp;
        if (instr.opcode.find(".eq") != std::string::npos) cmp = "eq";
        else if (instr.opcode.find(".ne") != std::string::npos) cmp = "ne";
        else if (instr.opcode.find(".lt") != std::string::npos) cmp = "lt";
        else if (instr.opcode.find(".le") != std::string::npos) cmp = "le";
        else if (instr.opcode.find(".gt") != std::string::npos) cmp = "gt";
        else if (instr.opcode.find(".ge") != std::string::npos) cmp = "ge";
        else if (instr.opcode.find(".num") != std::string::npos) cmp = "num";
        else if (instr.opcode.find(".nan") != std::string::npos) cmp = "nan";
        else return fail(instr, "unsupported setp comparison");

        std::string pred_value;
        if (ty.kind == PtxTypeSpec::Kind::kFloat) {
            if (ty.bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto a = decode_fp64_pair(os, instr.operands[1]);
                auto b = decode_fp64_pair(os, instr.operands[2]);
                if (!a || !b) return fail(instr, "fp64 setp emulation source unsupported");
                if (cmp == "num" || cmp == "nan") {
                    const std::string out = next_tmp("fp64_cmp_ordered");
                    os << "  " << out << " = fcmp " << (cmp == "num" ? "ord" : "uno")
                       << " float " << a->hi << ", " << b->hi << "\n";
                    return emit_store_reg_bits(os, dst, 1, out, 1);
                }
                const std::string hi_eq = next_tmp("fp64_cmp_hieq");
                const std::string lo_eq = next_tmp("fp64_cmp_loeq");
                const std::string equal = next_tmp("fp64_cmp_eq");
                os << "  " << hi_eq << " = fcmp oeq float " << a->hi << ", " << b->hi << "\n";
                os << "  " << lo_eq << " = fcmp oeq float " << a->lo << ", " << b->lo << "\n";
                os << "  " << equal << " = and i1 " << hi_eq << ", " << lo_eq << "\n";
                if (cmp == "eq") {
                    pred_value = equal;
                } else if (cmp == "ne") {
                    const std::string out = next_tmp("fp64_cmp_ne");
                    os << "  " << out << " = xor i1 " << equal << ", true\n";
                    pred_value = out;
                } else {
                    const bool greater = cmp == "gt" || cmp == "ge";
                    const std::string hi_order = next_tmp("fp64_cmp_hiorder");
                    const std::string lo_order = next_tmp("fp64_cmp_loorder");
                    const std::string tie_order = next_tmp("fp64_cmp_tie");
                    const std::string strict = next_tmp("fp64_cmp_strict");
                    os << "  " << hi_order << " = fcmp " << (greater ? "ogt" : "olt")
                       << " float " << a->hi << ", " << b->hi << "\n";
                    os << "  " << lo_order << " = fcmp " << (greater ? "ogt" : "olt")
                       << " float " << a->lo << ", " << b->lo << "\n";
                    os << "  " << tie_order << " = and i1 " << hi_eq << ", " << lo_order << "\n";
                    os << "  " << strict << " = or i1 " << hi_order << ", " << tie_order << "\n";
                    if (cmp == "le" || cmp == "ge") {
                        const std::string inclusive = next_tmp("fp64_cmp_inclusive");
                        os << "  " << inclusive << " = or i1 " << strict << ", " << equal << "\n";
                        pred_value = inclusive;
                    } else {
                        pred_value = strict;
                    }
                }
                return emit_store_reg_bits(os, dst, 1, pred_value, 1);
            }
            auto a = decode_float_operand(os, instr.operands[1], ty.bits);
            auto b = decode_float_operand(os, instr.operands[2], ty.bits);
            if (!a || !b) return fail(instr, "setp float source unsupported");
            const std::string out = next_tmp("fcmp");
            std::string cc;
            if (cmp == "eq") cc = "oeq";
            else if (cmp == "ne") cc = "one";
            else if (cmp == "lt") cc = "olt";
            else if (cmp == "le") cc = "ole";
            else if (cmp == "gt") cc = "ogt";
            else if (cmp == "ge") cc = "oge";
            else if (cmp == "num") cc = "ord";
            else cc = "uno";
            os << "  " << out << " = fcmp " << cc << " " << llvm_float_type(ty.bits)
               << " " << a->ir << ", " << b->ir << "\n";
            pred_value = out;
        } else {
            const int bits = (ty.bits > 0) ? ty.bits : 32;
            auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
            auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
            if (!a || !b) return fail(instr, "setp int source unsupported");
            const std::string out = next_tmp("icmp");
            std::string cc;
            if (cmp == "eq") cc = "eq";
            else if (cmp == "ne") cc = "ne";
            else if (cmp == "lt") cc = ty.is_signed ? "slt" : "ult";
            else if (cmp == "le") cc = ty.is_signed ? "sle" : "ule";
            else if (cmp == "gt") cc = ty.is_signed ? "sgt" : "ugt";
            else cc = ty.is_signed ? "sge" : "uge";
            os << "  " << out << " = icmp " << cc << " " << llvm_int_type(bits) << " " << *a << ", " << *b << "\n";
            pred_value = out;
        }
        return emit_store_reg_bits(os, dst, 1, pred_value, 1);
    }

    bool emit_ld_st(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        const std::string root = opcode_root(instr.opcode);
        const bool is_load = (root == "ld");
        const bool is_store = (root == "st");
        if (!is_load && !is_store) {
            return false;
        }
        if (instr.operands.size() < 2) {
            return fail(instr, "ld/st requires 2 operands");
        }

        int vector_width = 0;
        std::size_t vector_pos = instr.opcode.find(".v2.");
        if (vector_pos != std::string::npos) {
            vector_width = 2;
        } else {
            vector_pos = instr.opcode.find(".v4.");
            if (vector_pos != std::string::npos) {
                vector_width = 4;
            }
        }
        if (vector_width != 0) {
            const std::size_t data_operand_index = is_load ? 0 : 1;
            const std::size_t memory_operand_index = is_load ? 1 : 0;
            const std::vector<std::string> data_parts =
                split_comma_list(instr.operands[data_operand_index]);
            if (data_parts.size() != static_cast<std::size_t>(vector_width)) {
                return fail(instr,
                            std::string("vector ") + (is_load ? "load" : "store") +
                                " expects " + std::to_string(vector_width) + " registers");
            }
            const PtxTypeSpec elem_ty = parse_primary_type_from_opcode(instr.opcode);
            if (elem_ty.kind == PtxTypeSpec::Kind::kInvalid) {
                return fail(instr, "unable to parse vector memory element type");
            }
            const ParsedMemOperand mem = parse_memory_operand(instr.operands[memory_operand_index]);
            if (!mem.ok) {
                return fail(instr, "unable to parse vector memory operand");
            }

            for (int lane = 0; lane < vector_width; ++lane) {
                cumetal::ptx::EntryFunction::Instruction scalar = instr;
                scalar.opcode.replace(vector_pos, 4, ".");
                std::ostringstream mem_op;
                const std::int64_t lane_offset =
                    mem.offset + static_cast<std::int64_t>(lane) * std::max(1, elem_ty.bits / 8);
                mem_op << "[" << mem.base;
                if (lane_offset > 0) {
                    mem_op << "+" << lane_offset;
                } else if (lane_offset < 0) {
                    mem_op << lane_offset;
                }
                mem_op << "]";
                const std::string data_part = trim(data_parts[static_cast<std::size_t>(lane)]);
                scalar.operands =
                    is_load ? std::vector<std::string>{data_part, mem_op.str()}
                            : std::vector<std::string>{mem_op.str(), data_part};
                if (!emit_ld_st(os, scalar)) {
                    return false;
                }
            }
            return true;
        }

        const std::string data_token = is_load ? instr.operands[0] : instr.operands[1];
        const std::string mem_token = is_load ? instr.operands[1] : instr.operands[0];
        const ParsedMemOperand mem = parse_memory_operand(mem_token);
        if (!mem.ok) {
            return fail(instr, "unable to parse memory operand");
        }
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind == PtxTypeSpec::Kind::kInvalid) {
            return fail(instr, "unable to parse memory element type");
        }
        if (ty.kind == PtxTypeSpec::Kind::kFloat && ty.bits == 64 &&
            fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
            return fail(instr,
                        "fp64 memory load/store is not supported by FP32-pair emulation; "
                        "convert at an fp32 boundary");
        }

        auto emit_ptr_from_i64 = [&](const std::string& addr_i64, int as, int elem_bits, bool float_elem) -> std::string {
            const std::string ptr_i8 = next_tmp("i2p");
            if (as == 0) {
                os << "  " << ptr_i8 << " = inttoptr i64 " << addr_i64 << " to i8*\n";
            } else {
                os << "  " << ptr_i8 << " = inttoptr i64 " << addr_i64 << " to i8 addrspace(" << as << ")*\n";
            }
            const std::string ptr_t = next_tmp("bcptr");
            const std::string elem_ty = float_elem ? (elem_bits == 32 ? "float" : (elem_bits == 64 ? "double" : "half"))
                                                   : llvm_int_type(elem_bits);
            if (as == 0) {
                os << "  " << ptr_t << " = bitcast i8* " << ptr_i8 << " to " << elem_ty << "*\n";
            } else {
                os << "  " << ptr_t << " = bitcast i8 addrspace(" << as << ")* " << ptr_i8 << " to "
                   << elem_ty << " addrspace(" << as << ")*\n";
            }
            return ptr_t;
        };

        if (starts_with(instr.opcode, "ld.param") || starts_with(instr.opcode, "st.param")) {
            if (mem.base.empty()) return fail(instr, "param mem base missing");
            if (param_by_raw_.count(mem.base) && mem.offset == 0 && starts_with(instr.opcode, "ld.param")) {
                const ParamInfo& p = (*params_)[static_cast<std::size_t>(param_by_raw_.at(mem.base))];
                if (!is_constant_buffer_pointer(p.llvm_type)) {
                    if (!is_load || !is_register_name(data_token)) return fail(instr, "ld.param scalar form unsupported");
                    const std::string& dst = data_token;
                    if (ty.kind == PtxTypeSpec::Kind::kFloat) {
                        std::string fv;
                        if ((p.llvm_type == "float" && ty.bits == 32) || (p.llvm_type == "double" && ty.bits == 64) ||
                            (p.llvm_type == "half" && ty.bits == 16)) {
                            fv = "%" + p.name;
                        } else if ((p.llvm_type == "i32" && ty.bits == 32) || (p.llvm_type == "i64" && ty.bits == 64) ||
                                   (p.llvm_type == "i16" && ty.bits == 16)) {
                            const std::string bc = next_tmp("param_i2f");
                            const std::string fty = (ty.bits == 16) ? "half" : (ty.bits == 32 ? "float" : "double");
                            os << "  " << bc << " = bitcast " << p.llvm_type << " %" << p.name
                               << " to " << fty << "\n";
                            fv = bc;
                        } else {
                            return fail(instr, "ld.param scalar float type mismatch");
                        }
                        Value v{.ir = fv, .type = ty, .bits = ty.bits};
                        const int ldp_slot_bits = ensure_reg_slot(dst).bits;
                        auto bitsv = encode_value_to_reg_bits(os, v, ldp_slot_bits);
                        if (!bitsv) return fail(instr, "ld.param scalar float encode failed");
                        return emit_store_reg_bits(os, dst, ldp_slot_bits, *bitsv, ldp_slot_bits);
                    }
                    if (ty.kind == PtxTypeSpec::Kind::kInt) {
                        if (const auto pointer_space =
                                param_pointer_as_by_raw_.find(p.raw_name);
                            pointer_space != param_pointer_as_by_raw_.end()) {
                            reg_pointer_as_[dst] = pointer_space->second;
                        }
                        std::string srcv;
                        int src_bits = 0;
                        if (is_pointer_type(p.llvm_type)) {
                            const std::string p2i = next_tmp("paramptr2i");
                            os << "  " << p2i << " = ptrtoint " << p.llvm_type << " %" << p.name << " to i64\n";
                            srcv = p2i;
                            src_bits = 64;
                        } else if (p.llvm_type == "float" || p.llvm_type == "double" || p.llvm_type == "half") {
                            // Float param loaded as integer (bitcast preserving bits)
                            const int float_src_bits = (p.llvm_type == "double") ? 64 : (p.llvm_type == "half" ? 16 : 32);
                            const std::string bc = next_tmp("paramfbc");
                            os << "  " << bc << " = bitcast " << p.llvm_type << " %" << p.name
                               << " to " << llvm_int_type(float_src_bits) << "\n";
                            srcv = bc;
                            src_bits = float_src_bits;
                        } else {
                            srcv = "%" + p.name;
                            src_bits = byte_size_for_param_metadata(p) * 8;
                        }
                        if (src_bits != ty.bits) {
                            const std::string cast = next_tmp("paramcast");
                            if (src_bits < ty.bits) {
                                os << "  " << cast << " = zext " << llvm_int_type(src_bits) << " " << srcv
                                   << " to " << llvm_int_type(ty.bits) << "\n";
                            } else {
                                os << "  " << cast << " = trunc " << llvm_int_type(src_bits) << " " << srcv
                                   << " to " << llvm_int_type(ty.bits) << "\n";
                            }
                            srcv = cast;
                        }
                        return emit_store_reg_bits(os,
                                                   dst,
                                                   ensure_reg_slot(dst).bits,
                                                   srcv,
                                                   ty.bits,
                                                   ty.kind == PtxTypeSpec::Kind::kInt && ty.is_signed);
                    }
                }
            }

            // Param aggregate / call-sequence slots.
            if (is_register_name(mem.base)) {
                const auto as_it = reg_pointer_as_.find(mem.base);
                if (as_it != reg_pointer_as_.end() && as_it->second == PointerAs::kParam) {
                    const std::string base_i64 = emit_load_reg_bits(os, mem.base, 64);
                    const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);
                    if (is_load) {
                        if (!is_register_name(data_token)) return fail(instr, "ld.param register-base dst must be register");
                        const std::string ptr = emit_ptr_from_i64(addr_i64, 2, ty.bits, ty.kind == PtxTypeSpec::Kind::kFloat);
                        const std::string ld = next_tmp("ldparamrb");
                        const std::string elem_ty = (ty.kind == PtxTypeSpec::Kind::kFloat)
                                                        ? (ty.bits == 16 ? "half" : ty.bits == 32 ? "float" : "double")
                                                        : llvm_int_type(ty.bits);
                        os << "  " << ld << " = load " << elem_ty << ", " << elem_ty << " addrspace(2)* " << ptr
                           << ", align " << std::max(1, ty.bits / 8) << "\n";
                        Value v{.ir = ld, .type = ty, .bits = ty.bits};
                        auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(data_token).bits);
                        if (!bitsv) return fail(instr, "ld.param register-base encode failed");
                        return emit_store_reg_bits(os, data_token, ensure_reg_slot(data_token).bits, *bitsv, ensure_reg_slot(data_token).bits);
                    }
                    return fail(instr, "st.param register-base unsupported");
                }
            }

            if (param_by_raw_.count(mem.base)) {
                const ParamInfo& p = (*params_)[static_cast<std::size_t>(param_by_raw_.at(mem.base))];
                if (!is_constant_buffer_pointer(p.llvm_type)) {
                    return fail(instr, "param offset load on scalar param unsupported");
                }
                const std::string base_i64 = next_tmp("p2i_param");
                os << "  " << base_i64 << " = ptrtoint " << p.llvm_type << " %" << p.name << " to i64\n";
                const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);
                if (is_load) {
                    if (!is_register_name(data_token)) return fail(instr, "ld.param destination must be register");
                    const std::string ptr = emit_ptr_from_i64(addr_i64, 2, ty.bits, ty.kind == PtxTypeSpec::Kind::kFloat);
                    const std::string ld = next_tmp("ldparam");
                    const std::string elem_ty = (ty.kind == PtxTypeSpec::Kind::kFloat)
                                                    ? (ty.bits == 32 ? "float" : ty.bits == 64 ? "double" : "half")
                                                    : llvm_int_type(ty.bits);
                    os << "  " << ld << " = load " << elem_ty << ", " << elem_ty << " addrspace(2)* " << ptr
                       << ", align " << std::max(1, ty.bits / 8) << "\n";
                    Value v{.ir = ld, .type = ty, .bits = ty.bits};
                    auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(data_token).bits);
                    if (!bitsv) return fail(instr, "ld.param aggregate encode failed");
                    return emit_store_reg_bits(os, data_token, ensure_reg_slot(data_token).bits, *bitsv, ensure_reg_slot(data_token).bits);
                }
                return fail(instr, "st.param to kernel param unsupported");
            }

            if (starts_with(instr.opcode, "st.param")) {
                const std::string slot_name = mem.offset == 0
                                                  ? mem.base
                                                  : mem.base + "@" + std::to_string(mem.offset);
                auto slot = get_param_slot(slot_name, ty.bits, true);
                if (!slot) return fail(instr, "unable to allocate call param slot");
                if (is_register_name(data_token)) {
                    if (const auto provenance = reg_pointer_as_.find(data_token);
                        provenance != reg_pointer_as_.end()) {
                        call_param_pointer_as_[mem.base] = provenance->second;
                    }
                }
                if (ty.kind == PtxTypeSpec::Kind::kFloat) {
                    auto fv = decode_float_operand(os, data_token, ty.bits);
                    if (!fv) return fail(instr, "st.param float source unsupported");
                    auto bitsv = encode_value_to_reg_bits(os, *fv, ty.bits);
                    if (!bitsv) return fail(instr, "st.param float encode failed");
                    os << "  store " << llvm_int_type(ty.bits) << " " << *bitsv << ", "
                       << llvm_int_type(ty.bits) << "* " << *slot << ", align " << std::max(1, ty.bits / 8) << "\n";
                    return true;
                }
                auto iv = emit_integer_from_any(os, data_token, ty.bits, ty.is_signed);
                if (!iv) return fail(instr, "st.param int source unsupported");
                os << "  store " << llvm_int_type(ty.bits) << " " << *iv << ", "
                   << llvm_int_type(ty.bits) << "* " << *slot << ", align " << std::max(1, ty.bits / 8) << "\n";
                return true;
            }

            if (starts_with(instr.opcode, "ld.param")) {
                if (!is_register_name(data_token)) return fail(instr, "ld.param dst must be register");
                const std::string slot_name = mem.offset == 0
                                                  ? mem.base
                                                  : mem.base + "@" + std::to_string(mem.offset);
                auto slot = get_param_slot(slot_name, ty.bits, false);
                if (!slot) return fail(instr, "ld.param unknown param slot");
                const std::string ld = next_tmp("ldcallp");
                os << "  " << ld << " = load " << llvm_int_type(ty.bits) << ", " << llvm_int_type(ty.bits)
                   << "* " << *slot << ", align " << std::max(1, ty.bits / 8) << "\n";
                return emit_store_reg_bits(os,
                                           data_token,
                                           ensure_reg_slot(data_token).bits,
                                           ld,
                                           ty.bits,
                                           ty.kind == PtxTypeSpec::Kind::kInt && ty.is_signed);
            }
        }

        int addr_space = -1;
        if (instr.opcode.find(".global") != std::string::npos) {
            addr_space = 1;
        } else if (instr.opcode.find(".shared") != std::string::npos) {
            addr_space = 3;
        } else if (instr.opcode.find(".const") != std::string::npos) {
            addr_space = 2;
        } else if (instr.opcode.find(".local") != std::string::npos) {
            addr_space = 0;
        } else if (root == "ld" || root == "st") {
            // Clang emits generic ld/st for pointers whose PTX state space is
            // not encoded in the instruction. CUDA kernel pointer parameters
            // and pointers loaded from descriptors refer to device/global
            // memory; shared/local accesses retain explicit state-space
            // opcodes in generated PTX.
            addr_space = 1;
        } else {
            return fail(instr, "only global/const/param/shared/local ld/st supported in generic LLVM path");
        }
        if (instr.opcode.find(".global") == std::string::npos &&
            instr.opcode.find(".shared") == std::string::npos &&
            instr.opcode.find(".const") == std::string::npos &&
            instr.opcode.find(".local") == std::string::npos &&
            is_register_name(mem.base)) {
            if (const auto provenance = reg_pointer_as_.find(mem.base);
                provenance != reg_pointer_as_.end()) {
                if (provenance->second == PointerAs::kShared) addr_space = 3;
                else if (provenance->second == PointerAs::kLocal) addr_space = 0;
                else if (provenance->second == PointerAs::kParam) addr_space = 2;
                else if (provenance->second == PointerAs::kGlobal) addr_space = 1;
            }
        }

        std::string base_i64;
        if (is_register_name(mem.base)) {
            base_i64 = emit_load_reg_bits(os, mem.base, 64);
        } else if (const auto sym = resolve_param_symbol_address(os, mem.base)) {
            base_i64 = *sym;
        } else if (const auto global = resolve_global_symbol_address(os, mem.base)) {
            base_i64 = *global;
        } else if (const auto cst = resolve_const_symbol_address(os, mem.base)) {
            base_i64 = *cst;
        } else if (const auto local = resolve_local_symbol_address(os, mem.base)) {
            base_i64 = *local;
        } else if (addr_space == 3) {
            // Named .shared/.extern .shared symbol (e.g. extern __shared__ float sdata[]).
            // All shared symbols map to offset 0 within the threadgroup buffer.
            if (const auto tg = resolve_threadgroup_symbol_address(os, mem.base)) {
                base_i64 = *tg;
            } else {
                return fail(instr,
                            "ld/st.shared: cannot resolve shared symbol address '" +
                                mem.base + "'");
            }
        } else {
            return fail(instr, "memory base must be register or param/local/global/const symbol");
        }
        const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);

        if (is_load) {
            if (!is_register_name(data_token)) return fail(instr, "load dst must be register");
            const std::string ptr =
                emit_ptr_from_i64(addr_i64, addr_space, ty.bits, ty.kind == PtxTypeSpec::Kind::kFloat);
            const std::string elem_ty = (ty.kind == PtxTypeSpec::Kind::kFloat)
                                            ? (ty.bits == 32 ? "float" : ty.bits == 64 ? "double" : "half")
                                            : llvm_int_type(ty.bits);
            const std::string ld = next_tmp("ld");
            if (addr_space == 0) {
                os << "  " << ld << " = load " << elem_ty << ", " << elem_ty << "* " << ptr
                   << ", align " << std::max(1, ty.bits / 8) << "\n";
            } else {
                os << "  " << ld << " = load " << elem_ty << ", " << elem_ty << " addrspace(" << addr_space << ")* "
                   << ptr << ", align " << std::max(1, ty.bits / 8) << "\n";
            }
            Value v{.ir = ld, .type = ty, .bits = ty.bits};
            const int slot_bits = ensure_reg_slot(data_token).bits;
            auto bitsv = encode_value_to_reg_bits(os, v, slot_bits);
            if (!bitsv) return fail(instr, "load encode failed");
            // bitsv has already been extended/truncated to slot_bits by encode_value_to_reg_bits
            const bool stored =
                emit_store_reg_bits(os, data_token, slot_bits, *bitsv, slot_bits);
            if (stored && ty.bits == 64) {
                if (addr_space == 0 && is_register_name(mem.base) &&
                    reg_local_origin_.count(mem.base)) {
                    const std::string& origin = reg_local_origin_.at(mem.base);
                    const auto payload = local_pointer_payload_as_.find(origin);
                    if (payload != local_pointer_payload_as_.end() &&
                        payload->second != PointerAs::kUnknown) {
                        reg_pointer_as_[data_token] = payload->second;
                    } else {
                        reg_pointer_as_.erase(data_token);
                    }
                } else {
                    // The address space of the load describes where the pointer
                    // value is stored, not what the loaded value points at. CUDA
                    // code commonly stages global pointers in shared memory
                    // (PhysX does this for contact-preparation descriptors), so
                    // treating every ld.shared.u64 result as a shared pointer
                    // redirects later generic stores into threadgroup memory.
                    // Explicit shared pointers retain their provenance through
                    // mov/cvta and the local-spill payload tracking above. In the
                    // absence of such evidence, a loaded device pointer is global.
                    reg_pointer_as_[data_token] =
                        addr_space == 2 ? PointerAs::kParam : PointerAs::kGlobal;
                }
            }
            return stored;
        }

        if (ty.kind == PtxTypeSpec::Kind::kFloat) {
            auto fv = decode_float_operand(os, data_token, ty.bits);
            if (!fv) return fail(instr, "store float source unsupported");
            const std::string ptr =
                emit_ptr_from_i64(addr_i64, addr_space, ty.bits, true);
            const std::string elem_ty = (ty.bits == 32 ? "float" : ty.bits == 64 ? "double" : "half");
            if (addr_space == 0) {
                os << "  store " << elem_ty << " " << fv->ir << ", " << elem_ty << "* " << ptr
                   << ", align " << std::max(1, ty.bits / 8) << "\n";
            } else {
                os << "  store " << elem_ty << " " << fv->ir << ", " << elem_ty << " addrspace(" << addr_space << ")* "
                   << ptr << ", align " << std::max(1, ty.bits / 8) << "\n";
            }
            return true;
        }
        auto iv = emit_integer_from_any(os, data_token, ty.bits, ty.is_signed);
        if (!iv) return fail(instr, "store int source unsupported");
        if (addr_space == 0 && ty.bits == 64 && is_register_name(mem.base) &&
            reg_local_origin_.count(mem.base) && is_register_name(data_token) &&
            reg_pointer_as_.count(data_token)) {
            const std::string& origin = reg_local_origin_.at(mem.base);
            const PointerAs payload = reg_pointer_as_.at(data_token);
            const auto previous = local_pointer_payload_as_.find(origin);
            if (previous == local_pointer_payload_as_.end()) {
                local_pointer_payload_as_[origin] = payload;
            } else if (previous->second != payload) {
                previous->second = PointerAs::kUnknown;
            }
        }
        const std::string ptr =
            emit_ptr_from_i64(addr_i64, addr_space, ty.bits, false);
        if (addr_space == 0) {
            os << "  store " << llvm_int_type(ty.bits) << " " << *iv << ", " << llvm_int_type(ty.bits)
               << "* " << ptr << ", align " << std::max(1, ty.bits / 8) << "\n";
        } else {
            os << "  store " << llvm_int_type(ty.bits) << " " << *iv << ", " << llvm_int_type(ty.bits)
               << " addrspace(" << addr_space << ")* " << ptr << ", align " << std::max(1, ty.bits / 8) << "\n";
        }
        return true;
    }

    bool emit_mul_wide(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "mul.wide requires dst,a,b");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind != PtxTypeSpec::Kind::kInt ||
            (ty.bits != 16 && ty.bits != 32)) {
            return fail(instr, "mul.wide supports 16- and 32-bit integer operands");
        }
        const int result_bits = ty.bits * 2;
        auto a = emit_integer_from_any(os, instr.operands[1], ty.bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], ty.bits, ty.is_signed);
        if (!a || !b) return fail(instr, "mul.wide source unsupported");
        const std::string wide_ty = llvm_int_type(result_bits);
        const std::string narrow_ty = llvm_int_type(ty.bits);
        const std::string a_wide = next_tmp("mw_a");
        const std::string b_wide = next_tmp("mw_b");
        os << "  " << a_wide << " = " << (ty.is_signed ? "sext " : "zext ")
           << narrow_ty << " " << *a << " to " << wide_ty << "\n";
        os << "  " << b_wide << " = " << (ty.is_signed ? "sext " : "zext ")
           << narrow_ty << " " << *b << " to " << wide_ty << "\n";
        const std::string prod = next_tmp("mw");
        os << "  " << prod << " = mul " << wide_ty << " " << a_wide << ", " << b_wide << "\n";
        return emit_store_reg_bits(os, dst, result_bits, prod, result_bits);
    }

    std::optional<std::string> emit_device_malloc(
        std::ostringstream& os, const std::string& arg_name) {
        auto requested64 = load_call_slot_value(os, arg_name, 64);
        if (!requested64) return std::nullopt;
        uses_device_heap_ = true;

        const std::string requested32 = next_tmp("heap_requested");
        const std::string with_header = next_tmp("heap_with_header");
        const std::string total = next_tmp("heap_total");
        os << "  " << requested32 << " = trunc i64 " << *requested64 << " to i32\n";
        os << "  " << with_header << " = add i32 " << requested32 << ", 31\n";
        os << "  " << total << " = and i32 " << with_header << ", -16\n";

        const std::string free_head_ptr = next_tmp("heap_free_head_ptr");
        const std::string capacity_ptr = next_tmp("heap_capacity_ptr");
        const std::string capacity = next_tmp("heap_capacity");
        const std::string heap_i8 = next_tmp("heap_base_i8");
        os << "  " << heap_i8
           << " = bitcast i32 addrspace(1)* %__cumetal_device_heap to i8 addrspace(1)*\n";
        os << "  " << free_head_ptr
           << " = getelementptr inbounds i32, i32 addrspace(1)* %__cumetal_device_heap, i64 1\n";
        os << "  " << capacity_ptr
           << " = getelementptr inbounds i32, i32 addrspace(1)* %__cumetal_device_heap, i64 2\n";
        os << "  " << capacity << " = load i32, i32 addrspace(1)* "
           << capacity_ptr << ", align 4\n";
        const std::string capacity64 = next_tmp("heap_capacity64");
        const std::string size_valid = next_tmp("heap_size_valid");
        os << "  " << capacity64 << " = zext i32 " << capacity << " to i64\n";
        os << "  " << size_valid << " = icmp ule i64 " << *requested64 << ", "
           << capacity64 << "\n";

        const std::string pop_label = "heap_pop_" + std::to_string(tmp_id_++);
        const std::string reuse_label = "heap_reuse_" + std::to_string(tmp_id_++);
        const std::string claim_label = "heap_claim_" + std::to_string(tmp_id_++);
        const std::string bump_label = "heap_bump_" + std::to_string(tmp_id_++);
        const std::string ready_label = "heap_ready_" + std::to_string(tmp_id_++);
        const std::string null_label = "heap_null_" + std::to_string(tmp_id_++);
        const std::string done_label = "heap_done_" + std::to_string(tmp_id_++);
        os << "  br i1 " << size_valid << ", label %" << pop_label
           << ", label %" << null_label << "\n\n";

        os << pop_label << ":\n";
        const std::string head = next_tmp("heap_head");
        const std::string no_free = next_tmp("heap_no_free");
        os << "  " << head << " = load atomic i32, i32 addrspace(1)* "
           << free_head_ptr << " monotonic, align 4\n";
        os << "  " << no_free << " = icmp eq i32 " << head << ", 0\n";
        os << "  br i1 " << no_free << ", label %" << bump_label
           << ", label %" << reuse_label << "\n\n";

        os << reuse_label << ":\n";
        const std::string head64 = next_tmp("heap_head64");
        const std::string block_i8 = next_tmp("heap_block_i8");
        const std::string block_i32 = next_tmp("heap_block_i32");
        const std::string block_size = next_tmp("heap_block_size");
        const std::string fits_free = next_tmp("heap_fits_free");
        os << "  " << head64 << " = zext i32 " << head << " to i64\n";
        os << "  " << block_i8
           << " = getelementptr inbounds i8, i8 addrspace(1)* " << heap_i8 << ", i64 "
           << head64 << "\n";
        os << "  " << block_i32 << " = bitcast i8 addrspace(1)* " << block_i8
           << " to i32 addrspace(1)*\n";
        os << "  " << block_size << " = load i32, i32 addrspace(1)* "
           << block_i32 << ", align 4\n";
        os << "  " << fits_free << " = icmp uge i32 " << block_size << ", "
           << total << "\n";
        os << "  br i1 " << fits_free << ", label %" << claim_label
           << ", label %" << bump_label << "\n\n";

        os << claim_label << ":\n";
        const std::string next_ptr = next_tmp("heap_next_ptr");
        const std::string next = next_tmp("heap_next");
        const std::string popped = next_tmp("heap_popped");
        const std::string pop_ok = next_tmp("heap_pop_ok");
        os << "  " << next_ptr
           << " = getelementptr inbounds i32, i32 addrspace(1)* " << block_i32
           << ", i64 1\n";
        os << "  " << next << " = load i32, i32 addrspace(1)* " << next_ptr
           << ", align 4\n";
        os << "  " << popped << " = cmpxchg i32 addrspace(1)* " << free_head_ptr
           << ", i32 " << head << ", i32 " << next << " monotonic monotonic\n";
        os << "  " << pop_ok << " = extractvalue { i32, i1 } " << popped << ", 1\n";
        os << "  br i1 " << pop_ok << ", label %" << ready_label
           << ", label %" << pop_label << "\n\n";

        os << bump_label << ":\n";
        const std::string bump = next_tmp("heap_bump");
        const std::string end = next_tmp("heap_end");
        const std::string fits_capacity = next_tmp("heap_fits_capacity");
        os << "  " << bump
           << " = atomicrmw add i32 addrspace(1)* %__cumetal_device_heap, i32 "
           << total << " monotonic\n";
        os << "  " << end << " = add i32 " << bump << ", " << total << "\n";
        os << "  " << fits_capacity << " = icmp ule i32 " << end << ", "
           << capacity << "\n";
        os << "  br i1 " << fits_capacity << ", label %" << ready_label
           << ", label %" << null_label << "\n\n";

        os << ready_label << ":\n";
        const std::string chosen = next_tmp("heap_chosen");
        const std::string chosen64 = next_tmp("heap_chosen64");
        const std::string chosen_i8 = next_tmp("heap_chosen_i8");
        const std::string chosen_i32 = next_tmp("heap_chosen_i32");
        const std::string payload = next_tmp("heap_payload");
        const std::string payload_bits = next_tmp("heap_payload_bits");
        os << "  " << chosen << " = phi i32 [ " << head << ", %" << claim_label
           << " ], [ " << bump << ", %" << bump_label << " ]\n";
        os << "  " << chosen64 << " = zext i32 " << chosen << " to i64\n";
        os << "  " << chosen_i8
           << " = getelementptr inbounds i8, i8 addrspace(1)* " << heap_i8 << ", i64 "
           << chosen64 << "\n";
        os << "  " << chosen_i32 << " = bitcast i8 addrspace(1)* " << chosen_i8
           << " to i32 addrspace(1)*\n";
        os << "  store i32 " << total << ", i32 addrspace(1)* " << chosen_i32
           << ", align 4\n";
        os << "  " << payload << " = getelementptr inbounds i8, i8 addrspace(1)* "
           << chosen_i8 << ", i64 16\n";
        os << "  " << payload_bits << " = ptrtoint i8 addrspace(1)* " << payload
           << " to i64\n";
        os << "  br label %" << done_label << "\n\n";

        os << null_label << ":\n";
        os << "  br label %" << done_label << "\n\n";
        os << done_label << ":\n";
        const std::string result = next_tmp("heap_result");
        os << "  " << result << " = phi i64 [ " << payload_bits << ", %"
           << ready_label << " ], [ 0, %" << null_label << " ]\n";
        return result;
    }

    bool emit_device_free(std::ostringstream& os, const std::string& arg_name) {
        auto pointer_bits = load_call_slot_value(os, arg_name, 64);
        if (!pointer_bits) return false;
        uses_device_heap_ = true;
        const std::string is_null = next_tmp("heap_free_null");
        os << "  " << is_null << " = icmp eq i64 " << *pointer_bits << ", 0\n";
        const std::string push_label = "heap_free_push_" + std::to_string(tmp_id_++);
        const std::string loop_label = "heap_free_loop_" + std::to_string(tmp_id_++);
        const std::string done_label = "heap_free_done_" + std::to_string(tmp_id_++);
        os << "  br i1 " << is_null << ", label %" << done_label
           << ", label %" << push_label << "\n\n";
        os << push_label << ":\n";
        const std::string pointer = next_tmp("heap_free_pointer");
        const std::string header = next_tmp("heap_free_header");
        const std::string header_bits = next_tmp("heap_free_header_bits");
        const std::string base_bits = next_tmp("heap_free_base_bits");
        const std::string offset64 = next_tmp("heap_free_offset64");
        const std::string offset = next_tmp("heap_free_offset");
        const std::string header_i32 = next_tmp("heap_free_header_i32");
        const std::string next_ptr = next_tmp("heap_free_next_ptr");
        const std::string free_head_ptr = next_tmp("heap_free_head_ptr");
        os << "  " << pointer << " = inttoptr i64 " << *pointer_bits
           << " to i8 addrspace(1)*\n";
        os << "  " << header << " = getelementptr inbounds i8, i8 addrspace(1)* "
           << pointer << ", i64 -16\n";
        os << "  " << header_bits << " = ptrtoint i8 addrspace(1)* " << header
           << " to i64\n";
        os << "  " << base_bits
           << " = ptrtoint i32 addrspace(1)* %__cumetal_device_heap to i64\n";
        os << "  " << offset64 << " = sub i64 " << header_bits << ", " << base_bits
           << "\n";
        os << "  " << offset << " = trunc i64 " << offset64 << " to i32\n";
        os << "  " << header_i32 << " = bitcast i8 addrspace(1)* " << header
           << " to i32 addrspace(1)*\n";
        os << "  " << next_ptr
           << " = getelementptr inbounds i32, i32 addrspace(1)* " << header_i32
           << ", i64 1\n";
        os << "  " << free_head_ptr
           << " = getelementptr inbounds i32, i32 addrspace(1)* %__cumetal_device_heap, i64 1\n";
        os << "  br label %" << loop_label << "\n\n";
        os << loop_label << ":\n";
        const std::string old_head = next_tmp("heap_free_old_head");
        const std::string pushed = next_tmp("heap_free_pushed");
        const std::string push_ok = next_tmp("heap_free_push_ok");
        os << "  " << old_head << " = load atomic i32, i32 addrspace(1)* "
           << free_head_ptr << " monotonic, align 4\n";
        os << "  store i32 " << old_head << ", i32 addrspace(1)* " << next_ptr
           << ", align 4\n";
        os << "  " << pushed << " = cmpxchg i32 addrspace(1)* " << free_head_ptr
           << ", i32 " << old_head << ", i32 " << offset
           << " monotonic monotonic\n";
        os << "  " << push_ok << " = extractvalue { i32, i1 } " << pushed << ", 1\n";
        os << "  br i1 " << push_ok << ", label %" << done_label
           << ", label %" << loop_label << "\n\n";
        os << done_label << ":\n";
        return true;
    }

    bool emit_call(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2) {
            return fail(instr, "call requires callee and argument tuple");
        }
        const bool has_destination = !trim(instr.operands[0]).empty() &&
                                     trim(instr.operands[0]).front() == '(';
        const std::string dest_token = has_destination ? instr.operands[0] : std::string{};
        const std::size_t callee_index = has_destination ? 1 : 0;
        const std::size_t args_index = callee_index + 1;
        const std::string callee = trim(instr.operands[callee_index]);
        const std::vector<std::string> arg_names =
            (instr.operands.size() > args_index) ? parse_paren_tuple(instr.operands[args_index])
                                                : std::vector<std::string>{};
        const std::vector<std::string> dest_names =
            has_destination ? parse_paren_tuple(dest_token) : std::vector<std::string>{};

        auto store_ret_bits = [&](const std::string& bits_value, int bits) -> bool {
            if (!dest_names.empty()) {
                for (const std::string& dn : dest_names) {
                    const std::string d = trim(dn);
                    if (d.empty()) continue;
                    if (is_register_name(d)) {
                        if (!emit_store_reg_bits(os, d, ensure_reg_slot(d).bits, bits_value, bits)) {
                            return false;
                        }
                    } else {
                        auto slot = get_param_slot(d, bits, true);
                        if (!slot) return false;
                        os << "  store " << llvm_int_type(bits) << " " << bits_value << ", "
                           << llvm_int_type(bits) << "* " << *slot << ", align " << std::max(1, bits / 8) << "\n";
                    }
                }
            }
            return true;
        };

        auto load_call_slot_f32 = [&](const std::string& arg_name) -> std::optional<std::string> {
            auto bits = load_call_slot_value(os, arg_name, 32);
            if (!bits) return std::nullopt;
            const std::string f = next_tmp("callf");
            os << "  " << f << " = bitcast i32 " << *bits << " to float\n";
            return f;
        };

        auto store_ret_f32 = [&](const std::string& fval) -> bool {
            const std::string bits = next_tmp("callf2i");
            os << "  " << bits << " = bitcast float " << fval << " to i32\n";
            return store_ret_bits(bits, 32);
        };

        if (callee == "__cumetal_grid_sync") {
            if (!module_uses_grid_barrier_ || !arg_names.empty()) {
                return fail(instr, "grid sync requires the cooperative-grid barrier ABI");
            }
            const auto tid_it = builtin_vector_arg_name_.find(
                "air.thread_position_in_threadgroup");
            const auto gpg_it = builtin_vector_arg_name_.find(
                "air.threadgroups_per_grid");
            if (tid_it == builtin_vector_arg_name_.end() ||
                gpg_it == builtin_vector_arg_name_.end()) {
                return fail(instr, "grid sync AIR builtins are unavailable");
            }
            declarations_.insert("declare void @air.wg.barrier(i32, i32)");
            os << "  call void @air.wg.barrier(i32 3, i32 1)\n";
            const std::string tidx = next_tmp("grid_sync_tidx");
            const std::string tidy = next_tmp("grid_sync_tidy");
            const std::string tidz = next_tmp("grid_sync_tidz");
            os << "  " << tidx << " = extractelement <3 x i32> %" << tid_it->second
               << ", i64 0\n";
            os << "  " << tidy << " = extractelement <3 x i32> %" << tid_it->second
               << ", i64 1\n";
            os << "  " << tidz << " = extractelement <3 x i32> %" << tid_it->second
               << ", i64 2\n";
            const std::string xy = next_tmp("grid_sync_tid_xy");
            const std::string xyz = next_tmp("grid_sync_tid_xyz");
            const std::string is_leader = next_tmp("grid_sync_is_leader");
            os << "  " << xy << " = or i32 " << tidx << ", " << tidy << "\n";
            os << "  " << xyz << " = or i32 " << xy << ", " << tidz << "\n";
            os << "  " << is_leader << " = icmp eq i32 " << xyz << ", 0\n";

            const std::string leader_label = "grid_sync_leader_" + std::to_string(tmp_id_++);
            const std::string wait_label = "grid_sync_wait_" + std::to_string(tmp_id_++);
            const std::string last_label = "grid_sync_last_" + std::to_string(tmp_id_++);
            const std::string spin_label = "grid_sync_spin_" + std::to_string(tmp_id_++);
            const std::string arrive_done_label =
                "grid_sync_arrive_done_" + std::to_string(tmp_id_++);
            const std::string done_label = "grid_sync_done_" + std::to_string(tmp_id_++);
            os << "  br i1 " << is_leader << ", label %" << leader_label
               << ", label %" << wait_label << "\n\n";

            os << leader_label << ":\n";
            const std::string gen_ptr = next_tmp("grid_sync_gen_ptr");
            const std::string generation = next_tmp("grid_sync_generation");
            const std::string old_count = next_tmp("grid_sync_old_count");
            const std::string arrived = next_tmp("grid_sync_arrived");
            os << "  " << gen_ptr
               << " = getelementptr inbounds i32, i32 addrspace(1)* %__cumetal_grid_barrier, i64 1\n";
            os << "  " << generation << " = load atomic i32, i32 addrspace(1)* "
               << gen_ptr << " acquire, align 4\n";
            os << "  " << old_count
               << " = atomicrmw add i32 addrspace(1)* %__cumetal_grid_barrier, i32 1 acq_rel\n";
            os << "  " << arrived << " = add i32 " << old_count << ", 1\n";
            const std::string gpx = next_tmp("grid_sync_gpx");
            const std::string gpy = next_tmp("grid_sync_gpy");
            const std::string gpz = next_tmp("grid_sync_gpz");
            const std::string gp_xy = next_tmp("grid_sync_gp_xy");
            const std::string gp_xyz = next_tmp("grid_sync_gp_xyz");
            const std::string is_last = next_tmp("grid_sync_is_last");
            os << "  " << gpx << " = extractelement <3 x i32> %" << gpg_it->second
               << ", i64 0\n";
            os << "  " << gpy << " = extractelement <3 x i32> %" << gpg_it->second
               << ", i64 1\n";
            os << "  " << gpz << " = extractelement <3 x i32> %" << gpg_it->second
               << ", i64 2\n";
            os << "  " << gp_xy << " = mul i32 " << gpx << ", " << gpy << "\n";
            os << "  " << gp_xyz << " = mul i32 " << gp_xy << ", " << gpz << "\n";
            os << "  " << is_last << " = icmp eq i32 " << arrived << ", " << gp_xyz
               << "\n";
            os << "  br i1 " << is_last << ", label %" << last_label
               << ", label %" << spin_label << "\n\n";

            os << last_label << ":\n";
            os << "  store atomic i32 0, i32 addrspace(1)* %__cumetal_grid_barrier release, align 4\n";
            const std::string advanced = next_tmp("grid_sync_advanced");
            os << "  " << advanced << " = atomicrmw add i32 addrspace(1)* " << gen_ptr
               << ", i32 1 release\n";
            os << "  br label %" << arrive_done_label << "\n\n";

            os << spin_label << ":\n";
            const std::string observed = next_tmp("grid_sync_observed");
            const std::string released = next_tmp("grid_sync_released");
            os << "  " << observed << " = load atomic volatile i32, i32 addrspace(1)* "
               << gen_ptr << " acquire, align 4\n";
            os << "  " << released << " = icmp ne i32 " << observed << ", "
               << generation << "\n";
            os << "  br i1 " << released << ", label %" << arrive_done_label
               << ", label %" << spin_label << "\n\n";

            os << arrive_done_label << ":\n";
            os << "  br label %" << done_label << "\n\n";
            os << wait_label << ":\n";
            os << "  br label %" << done_label << "\n\n";
            os << done_label << ":\n";
            os << "  call void @air.wg.barrier(i32 3, i32 1)\n";
            return true;
        }

        if (callee == "cudaStreamCreateWithFlags") {
            if (arg_names.size() != 2) {
                return fail(instr, "cudaStreamCreateWithFlags expects stream pointer and flags");
            }
            auto stream_pointer = load_call_slot_value(os, arg_names[0], 64);
            auto flags = load_call_slot_value(os, arg_names[1], 32);
            if (!stream_pointer || !flags) {
                return fail(instr, "cudaStreamCreateWithFlags arguments unavailable");
            }
            int address_space = 0;
            if (const auto provenance = call_param_pointer_as_.find(arg_names[0]);
                provenance != call_param_pointer_as_.end()) {
                switch (provenance->second) {
                    case PointerAs::kGlobal: address_space = 1; break;
                    case PointerAs::kParam: address_space = 2; break;
                    case PointerAs::kShared: address_space = 3; break;
                    case PointerAs::kLocal:
                    case PointerAs::kUnknown: address_space = 0; break;
                }
            }
            const std::string stream_slot = next_tmp("device_stream_slot");
            os << "  " << stream_slot << " = inttoptr i64 " << *stream_pointer
               << " to i64";
            if (address_space != 0) os << " addrspace(" << address_space << ")";
            os << "*\n";
            // Host queue draining serializes child records in issue order. Use
            // the address of the caller's stream variable as a stable nonzero
            // device-stream token; distinct variables remain distinct handles.
            os << "  store i64 " << *stream_pointer << ", i64";
            if (address_space != 0) os << " addrspace(" << address_space << ")";
            os << "* " << stream_slot << ", align 8\n";
            return store_ret_bits("0", 32);
        }
        if (callee == "cudaStreamDestroy") {
            if (arg_names.size() != 1 ||
                !load_call_slot_value(os, arg_names[0], 64)) {
                return fail(instr, "cudaStreamDestroy stream argument unavailable");
            }
            return store_ret_bits("0", 32);
        }
        if (callee == "cudaGetLastError" || callee == "cudaPeekAtLastError") {
            if (!arg_names.empty()) {
                return fail(instr, callee + " expects no arguments");
            }
            return store_ret_bits("0", 32);
        }
        if (callee == "cudaGetErrorString") {
            if (arg_names.size() != 1 ||
                !load_call_slot_value(os, arg_names[0], 32)) {
                return fail(instr, "cudaGetErrorString error argument unavailable");
            }
            // Device-side callers use this only on an error branch. Successful
            // child-queue operations keep that branch inactive; return a null
            // pointer rather than exposing a host string address to Metal.
            return store_ret_bits("0", 64);
        }

        if (callee == "cudaMemcpyAsync") {
            if (!module_uses_device_launch_queue_ || arg_names.size() != 5) {
                return fail(instr, "device cudaMemcpyAsync requires five arguments and the device queue ABI");
            }
            auto destination = load_call_slot_value(os, arg_names[0], 64);
            auto source = load_call_slot_value(os, arg_names[1], 64);
            auto count = load_call_slot_value(os, arg_names[2], 64);
            auto kind = load_call_slot_value(os, arg_names[3], 32);
            auto stream = load_call_slot_value(os, arg_names[4], 64);
            if (!destination || !source || !count || !kind || !stream) {
                return fail(instr, "device cudaMemcpyAsync argument slots unavailable");
            }

            const std::string record_count_ptr = next_tmp("cdp_copy_record_count_ptr");
            const std::string record_index = next_tmp("cdp_copy_record_index");
            const std::string has_record = next_tmp("cdp_copy_has_record");
            const std::string write_label = "cdp_copy_write_" + std::to_string(tmp_id_++);
            const std::string overflow_label = "cdp_copy_overflow_" + std::to_string(tmp_id_++);
            const std::string done_label = "cdp_copy_done_" + std::to_string(tmp_id_++);
            os << "  " << record_count_ptr
               << " = getelementptr inbounds i32, i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue, i64 1\n"
               << "  " << record_index << " = atomicrmw add i32 addrspace(1)* "
               << record_count_ptr << ", i32 1 monotonic\n"
               << "  " << has_record << " = icmp ult i32 " << record_index << ", 1023\n"
               << "  br i1 " << has_record << ", label %" << write_label
               << ", label %" << overflow_label << "\n\n"
               << write_label << ":\n";
            const std::string record_word = next_tmp("cdp_copy_record_word");
            const std::string record_base = next_tmp("cdp_copy_record_base");
            os << "  " << record_word << " = mul i32 " << record_index << ", 16\n"
               << "  " << record_word << "_base = add i32 " << record_word << ", 4\n"
               << "  " << record_base
               << " = getelementptr inbounds i32, i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue, i32 " << record_word << "_base\n";
            const auto store_copy_word = [&](int offset, const std::string& value) {
                const std::string pointer = next_tmp("cdp_copy_field");
                os << "  " << pointer
                   << " = getelementptr inbounds i32, i32 addrspace(1)* "
                   << record_base << ", i32 " << offset << "\n"
                   << "  store i32 " << value << ", i32 addrspace(1)* " << pointer
                   << ", align 4\n";
            };
            const auto store_copy_i64 = [&](int offset, const std::string& value,
                                             const std::string& stem) {
                const std::string low = next_tmp(stem + "_low");
                const std::string shift = next_tmp(stem + "_shift");
                const std::string high = next_tmp(stem + "_high");
                os << "  " << low << " = trunc i64 " << value << " to i32\n"
                   << "  " << shift << " = lshr i64 " << value << ", 32\n"
                   << "  " << high << " = trunc i64 " << shift << " to i32\n";
                store_copy_word(offset, low);
                store_copy_word(offset + 1, high);
            };
            store_copy_i64(0, *destination, "cdp_copy_destination");
            store_copy_i64(2, *source, "cdp_copy_source");
            store_copy_i64(4, *count, "cdp_copy_count");
            store_copy_word(6, *kind);
            store_copy_i64(7, *stream, "cdp_copy_stream");
            for (int offset = 9; offset < 15; ++offset) store_copy_word(offset, "0");
            store_copy_word(15, "1");
            os << "  br label %" << done_label << "\n\n"
               << overflow_label << ":\n  br label %" << done_label << "\n\n"
               << done_label << ":\n";
            const std::string copy_status = next_tmp("cdp_copy_status");
            os << "  " << copy_status << " = phi i32 [ 0, %" << write_label
               << " ], [ 1, %" << overflow_label << " ]\n";
            return store_ret_bits(copy_status, 32);
        }

        if (callee == "cudaGetParameterBuffer") {
            if (!module_uses_device_launch_queue_ || arg_names.size() != 2) {
                return fail(instr, "cudaGetParameterBuffer requires the device launch queue ABI");
            }
            auto alignment64 = load_call_slot_value(os, arg_names[0], 64);
            auto size64 = load_call_slot_value(os, arg_names[1], 64);
            if (!alignment64 || !size64) {
                return fail(instr, "cudaGetParameterBuffer arguments unavailable");
            }
            const std::string alignment32 = next_tmp("cdp_alignment");
            const std::string size32 = next_tmp("cdp_size");
            const std::string alignment_nonzero = next_tmp("cdp_alignment_nonzero");
            const std::string alignment = next_tmp("cdp_alignment_safe");
            const std::string reserve = next_tmp("cdp_reserve");
            const std::string raw = next_tmp("cdp_raw");
            const std::string with_header = next_tmp("cdp_with_header");
            const std::string alignment_minus_one = next_tmp("cdp_alignment_minus_one");
            const std::string rounded = next_tmp("cdp_rounded");
            const std::string alignment_mask = next_tmp("cdp_alignment_mask");
            const std::string aligned = next_tmp("cdp_aligned");
            const std::string end = next_tmp("cdp_end");
            const std::string capacity = next_tmp("cdp_capacity");
            const std::string fits = next_tmp("cdp_fits");
            const std::string ok_label = "cdp_param_ok_" + std::to_string(tmp_id_++);
            const std::string fail_label = "cdp_param_fail_" + std::to_string(tmp_id_++);
            const std::string done_label = "cdp_param_done_" + std::to_string(tmp_id_++);
            os << "  " << alignment32 << " = trunc i64 " << *alignment64 << " to i32\n"
               << "  " << size32 << " = trunc i64 " << *size64 << " to i32\n"
               << "  " << alignment_nonzero << " = icmp ne i32 " << alignment32 << ", 0\n"
               << "  " << alignment << " = select i1 " << alignment_nonzero
               << ", i32 " << alignment32 << ", i32 1\n"
               << "  " << alignment_minus_one << " = sub i32 " << alignment << ", 1\n"
               << "  " << reserve << " = add i32 " << size32 << ", 8\n"
               << "  " << reserve << "_aligned = add i32 " << reserve << ", "
               << alignment_minus_one << "\n"
               << "  " << raw << " = atomicrmw add i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue, i32 " << reserve
               << "_aligned monotonic\n"
               << "  " << with_header << " = add i32 " << raw << ", 8\n"
               << "  " << rounded << " = add i32 " << with_header << ", "
               << alignment_minus_one << "\n"
               << "  " << alignment_mask << " = sub i32 0, " << alignment << "\n"
               << "  " << aligned << " = and i32 " << rounded << ", "
               << alignment_mask << "\n"
               << "  " << end << " = add i32 " << aligned << ", " << size32 << "\n"
               << "  " << capacity
               << " = load i32, i32 addrspace(2)* %__cumetal_device_launch_queue_capacity, align 4\n"
               << "  " << fits << " = icmp ule i32 " << end << ", " << capacity << "\n"
               << "  br i1 " << fits << ", label %" << ok_label << ", label %"
               << fail_label << "\n\n"
               << ok_label << ":\n";
            const std::string queue_i8 = next_tmp("cdp_queue_i8");
            const std::string header_offset = next_tmp("cdp_header_offset");
            const std::string header_ptr_i8 = next_tmp("cdp_header_i8");
            const std::string header_ptr = next_tmp("cdp_header");
            const std::string param_ptr_i8 = next_tmp("cdp_param_i8");
            const std::string param_bits = next_tmp("cdp_param_bits");
            os << "  " << queue_i8 << " = bitcast i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue to i8 addrspace(1)*\n"
               << "  " << header_offset << " = sub i32 " << aligned << ", 8\n"
               << "  " << header_ptr_i8
               << " = getelementptr inbounds i8, i8 addrspace(1)* " << queue_i8
               << ", i32 " << header_offset << "\n"
               << "  " << header_ptr << " = bitcast i8 addrspace(1)* " << header_ptr_i8
               << " to i32 addrspace(1)*\n"
               << "  store i32 " << size32 << ", i32 addrspace(1)* " << header_ptr
               << ", align 4\n"
               << "  " << param_ptr_i8
               << " = getelementptr inbounds i8, i8 addrspace(1)* " << queue_i8
               << ", i32 " << aligned << "\n"
               << "  " << param_bits << " = ptrtoint i8 addrspace(1)* " << param_ptr_i8
               << " to i64\n"
               << "  br label %" << done_label << "\n\n"
               << fail_label << ":\n  br label %" << done_label << "\n\n"
               << done_label << ":\n";
            const std::string result_ptr = next_tmp("cdp_param_result");
            os << "  " << result_ptr << " = phi i64 [ " << param_bits << ", %"
               << ok_label << " ], [ 0, %" << fail_label << " ]\n";
            return store_ret_bits(result_ptr, 64);
        }

        if (callee == "cudaLaunchDevice") {
            if (!module_uses_device_launch_queue_ || arg_names.size() != 6) {
                return fail(instr, "cudaLaunchDevice requires six arguments and the device launch queue ABI");
            }
            auto token = load_call_slot_value(os, arg_names[0], 64);
            auto parameter_buffer = load_call_slot_value(os, arg_names[1], 64);
            auto shared_bytes = load_call_slot_value(os, arg_names[4], 32);
            auto stream = load_call_slot_value(os, arg_names[5], 64);
            std::array<std::optional<std::string>, 3> grid;
            std::array<std::optional<std::string>, 3> block;
            for (int axis = 0; axis < 3; ++axis) {
                grid[static_cast<std::size_t>(axis)] =
                    load_call_slot_value_at(os, arg_names[2], 32, axis * 4);
                block[static_cast<std::size_t>(axis)] =
                    load_call_slot_value_at(os, arg_names[3], 32, axis * 4);
            }
            if (!token || !parameter_buffer || !shared_bytes || !stream ||
                !grid[0] || !grid[1] || !grid[2] ||
                !block[0] || !block[1] || !block[2]) {
                return fail(instr, "cudaLaunchDevice argument slots unavailable");
            }
            const std::string record_count_ptr = next_tmp("cdp_record_count_ptr");
            const std::string record_index = next_tmp("cdp_record_index");
            const std::string has_record = next_tmp("cdp_has_record");
            const std::string write_label = "cdp_launch_write_" + std::to_string(tmp_id_++);
            const std::string overflow_label = "cdp_launch_overflow_" + std::to_string(tmp_id_++);
            const std::string done_label = "cdp_launch_done_" + std::to_string(tmp_id_++);
            os << "  " << record_count_ptr
               << " = getelementptr inbounds i32, i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue, i64 1\n"
               << "  " << record_index << " = atomicrmw add i32 addrspace(1)* "
               << record_count_ptr << ", i32 1 monotonic\n"
               << "  " << has_record << " = icmp ult i32 " << record_index << ", 1023\n"
               << "  br i1 " << has_record << ", label %" << write_label
               << ", label %" << overflow_label << "\n\n"
               << write_label << ":\n";
            const std::string record_word = next_tmp("cdp_record_word");
            const std::string record_base = next_tmp("cdp_record_base");
            os << "  " << record_word << " = mul i32 " << record_index << ", 16\n"
               << "  " << record_word << "_base = add i32 " << record_word << ", 4\n"
               << "  " << record_base
               << " = getelementptr inbounds i32, i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue, i32 " << record_word << "_base\n";
            const auto store_record_word = [&](int offset, const std::string& value) {
                const std::string ptr = next_tmp("cdp_record_field");
                os << "  " << ptr << " = getelementptr inbounds i32, i32 addrspace(1)* "
                   << record_base << ", i32 " << offset << "\n"
                   << "  store i32 " << value << ", i32 addrspace(1)* " << ptr
                   << ", align 4\n";
            };
            const std::string token_low = next_tmp("cdp_token_low");
            const std::string token_shift = next_tmp("cdp_token_shift");
            const std::string token_high = next_tmp("cdp_token_high");
            const std::string queue_bits = next_tmp("cdp_queue_bits");
            const std::string parameter_offset64 = next_tmp("cdp_parameter_offset64");
            const std::string parameter_offset = next_tmp("cdp_parameter_offset");
            const std::string parameter_header_bits = next_tmp("cdp_parameter_header_bits");
            const std::string parameter_header_i8 = next_tmp("cdp_parameter_header_i8");
            const std::string parameter_header = next_tmp("cdp_parameter_header");
            const std::string parameter_size = next_tmp("cdp_parameter_size");
            os << "  " << token_low << " = trunc i64 " << *token << " to i32\n"
               << "  " << token_shift << " = lshr i64 " << *token << ", 32\n"
               << "  " << token_high << " = trunc i64 " << token_shift << " to i32\n"
               << "  " << queue_bits << " = ptrtoint i32 addrspace(1)* "
               << "%__cumetal_device_launch_queue to i64\n"
               << "  " << parameter_offset64 << " = sub i64 " << *parameter_buffer
               << ", " << queue_bits << "\n"
               << "  " << parameter_offset << " = trunc i64 " << parameter_offset64
               << " to i32\n"
               << "  " << parameter_header_bits << " = sub i64 " << *parameter_buffer
               << ", 8\n"
               << "  " << parameter_header_i8 << " = inttoptr i64 "
               << parameter_header_bits << " to i8 addrspace(1)*\n"
               << "  " << parameter_header << " = bitcast i8 addrspace(1)* "
               << parameter_header_i8 << " to i32 addrspace(1)*\n"
               << "  " << parameter_size << " = load i32, i32 addrspace(1)* "
               << parameter_header << ", align 4\n";
            store_record_word(0, token_low);
            store_record_word(1, token_high);
            store_record_word(2, parameter_offset);
            store_record_word(3, parameter_size);
            for (int axis = 0; axis < 3; ++axis) {
                store_record_word(4 + axis, *grid[static_cast<std::size_t>(axis)]);
                store_record_word(7 + axis, *block[static_cast<std::size_t>(axis)]);
            }
            store_record_word(10, *shared_bytes);
            const std::string stream_low = next_tmp("cdp_stream_low");
            const std::string stream_shift = next_tmp("cdp_stream_shift");
            const std::string stream_high = next_tmp("cdp_stream_high");
            os << "  " << stream_low << " = trunc i64 " << *stream << " to i32\n"
               << "  " << stream_shift << " = lshr i64 " << *stream << ", 32\n"
               << "  " << stream_high << " = trunc i64 " << stream_shift << " to i32\n";
            store_record_word(11, stream_low);
            store_record_word(12, stream_high);
            store_record_word(13, "0");
            store_record_word(14, "0");
            store_record_word(15, "0");
            os << "  br label %" << done_label << "\n\n"
               << overflow_label << ":\n  br label %" << done_label << "\n\n"
               << done_label << ":\n";
            const std::string launch_status = next_tmp("cdp_launch_status");
            os << "  " << launch_status << " = phi i32 [ 0, %" << write_label
               << " ], [ 1, %" << overflow_label << " ]\n";
            return store_ret_bits(launch_status, 32);
        }

        const bool is_device_allocation =
            callee == "malloc" || callee == "_Znwm" || callee == "_Znam";
        const bool is_device_deallocation =
            callee == "free" || callee == "_ZdlPv" || callee == "_ZdaPv" ||
            callee == "_ZdlPvm" || callee == "_ZdaPvm";
        if (is_device_allocation) {
            if (arg_names.empty()) return fail(instr, "device allocation expects 1 arg");
            auto result = emit_device_malloc(os, arg_names[0]);
            if (!result) return fail(instr, "device allocation size arg missing");
            return store_ret_bits(*result, 64);
        }
        if (is_device_deallocation) {
            if (arg_names.empty()) return fail(instr, "device deallocation expects 1 arg");
            if (!emit_device_free(os, arg_names[0])) {
                return fail(instr, "device deallocation pointer arg missing");
            }
            return true;
        }

        if (callee == "__cumetal_wmma_bf16_mma_8x8" ||
            callee == "__cumetal_wmma_f32_mma_8x8") {
            const bool bf16_inputs = callee == "__cumetal_wmma_bf16_mma_8x8";
            if (arg_names.size() != 3) {
                return fail(instr, bf16_inputs
                    ? "BF16 WMMA marker expects destination, A, B"
                    : "FP32 WMMA marker expects destination, A, B");
            }
            auto destination_bits = load_call_slot_value(os, arg_names[0], 64);
            auto a_bits = load_call_slot_value(os, arg_names[1], 64);
            auto b_bits = load_call_slot_value(os, arg_names[2], 64);
            if (!destination_bits || !a_bits || !b_bits) {
                return fail(instr, bf16_inputs
                    ? "BF16 WMMA marker pointer arguments unavailable"
                    : "FP32 WMMA marker pointer arguments unavailable");
            }

            const std::string destination_i8 = next_tmp("wmma_dst_i8");
            const std::string a_i8 = next_tmp("wmma_a_i8");
            const std::string b_i8 = next_tmp("wmma_b_i8");
            const std::string destination = next_tmp("wmma_dst");
            const std::string a = next_tmp("wmma_a");
            const std::string b = next_tmp("wmma_b");
            os << "  " << destination_i8 << " = inttoptr i64 " << *destination_bits
               << " to i8 addrspace(3)*\n"
               << "  " << a_i8 << " = inttoptr i64 " << *a_bits
               << " to i8 addrspace(3)*\n"
               << "  " << b_i8 << " = inttoptr i64 " << *b_bits
               << " to i8 addrspace(3)*\n"
               << "  " << destination << " = bitcast i8 addrspace(3)* "
               << destination_i8 << " to float addrspace(3)*\n"
               << "  " << a << " = bitcast i8 addrspace(3)* " << a_i8
               << " to " << (bf16_inputs ? "bfloat" : "float") << " addrspace(3)*\n"
               << "  " << b << " = bitcast i8 addrspace(3)* " << b_i8
               << " to " << (bf16_inputs ? "bfloat" : "float") << " addrspace(3)*\n";

            const std::string matrix_shape = "<2 x i64> <i64 8, i64 8>";
            const std::string matrix_stride = "<2 x i64> <i64 1, i64 8>";
            const std::string matrix_origin = "<2 x i64> zeroinitializer";
            const std::string input_type = bf16_inputs ? "bfloat" : "float";
            const std::string input_vector = bf16_inputs ? "v64bf16" : "v64f32";
            declarations_.insert(
                "declare <64 x " + input_type + "> "
                "@air.simdgroup_matrix_8x8_load." + input_vector + ".p3" +
                (bf16_inputs ? "bf16" : "f32") + "(" + input_type +
                " addrspace(3)*, <2 x i64>, <2 x i64>, <2 x i64>)");
            declarations_.insert(
                "declare <64 x float> "
                "@air.simdgroup_matrix_8x8_load.v64f32.p3f32("
                "float addrspace(3)*, <2 x i64>, <2 x i64>, <2 x i64>)");
            const std::string mma_suffix = bf16_inputs
                ? "v64f32.v64bf16.v64bf16.v64f32"
                : "v64f32.v64f32.v64f32.v64f32";
            declarations_.insert(
                "declare <64 x float> "
                "@air.simdgroup_matrix_8x8_multiply_accumulate." + mma_suffix +
                "(<64 x " + input_type + ">, <64 x " + input_type +
                ">, <64 x float>)");
            declarations_.insert(
                "declare void @air.simdgroup_matrix_8x8_store.v64f32.p3f32("
                "<64 x float>, float addrspace(3)*, <2 x i64>, <2 x i64>, "
                "<2 x i64>)");

            const std::string a_matrix = next_tmp("wmma_a_matrix");
            const std::string b_matrix = next_tmp("wmma_b_matrix");
            const std::string c_matrix = next_tmp("wmma_c_matrix");
            const std::string d_matrix = next_tmp("wmma_d_matrix");
            os << "  " << a_matrix << " = call fast <64 x " << input_type << "> "
               << "@air.simdgroup_matrix_8x8_load." << input_vector << ".p3"
               << (bf16_inputs ? "bf16" : "f32") << "("
               << input_type << " addrspace(3)* " << a << ", " << matrix_shape << ", "
               << matrix_stride << ", " << matrix_origin << ")\n"
               << "  " << b_matrix << " = call fast <64 x " << input_type << "> "
               << "@air.simdgroup_matrix_8x8_load." << input_vector << ".p3"
               << (bf16_inputs ? "bf16" : "f32") << "("
               << input_type << " addrspace(3)* " << b << ", " << matrix_shape << ", "
               << matrix_stride << ", " << matrix_origin << ")\n"
               << "  " << c_matrix << " = call fast <64 x float> "
               << "@air.simdgroup_matrix_8x8_load.v64f32.p3f32("
               << "float addrspace(3)* " << destination << ", " << matrix_shape
               << ", " << matrix_stride << ", " << matrix_origin << ")\n"
               << "  " << d_matrix << " = call fast <64 x float> "
               << "@air.simdgroup_matrix_8x8_multiply_accumulate."
               << mma_suffix << "(<64 x " << input_type << "> " << a_matrix
               << ", <64 x " << input_type << "> " << b_matrix
               << ", <64 x float> " << c_matrix
               << ")\n"
               << "  call void @air.simdgroup_matrix_8x8_store.v64f32.p3f32("
               << "<64 x float> " << d_matrix << ", float addrspace(3)* "
               << destination << ", " << matrix_shape << ", " << matrix_stride
               << ", " << matrix_origin << ")\n";
            return true;
        }

        const auto ptx_parameter_bits = [](const cumetal::ptx::Parameter& parameter) {
            const PtxTypeSpec type = parse_primary_type_from_opcode(parameter.type);
            return type.bits > 0 ? type.bits : 0;
        };
        const auto find_device_function = [&](const std::string& name)
            -> const cumetal::ptx::EntryFunction* {
            if (device_functions_ == nullptr) return nullptr;
            for (const auto& function : *device_functions_) {
                if (function.name == name) return &function;
            }
            return nullptr;
        };
        const auto compatible_device_function = [&](
            const cumetal::ptx::EntryFunction& function) {
            if (function.params.size() != arg_names.size() ||
                function.return_params.size() != dest_names.size()) return false;
            for (const auto& parameter : function.params) {
                // Aggregate-by-value call slots need a byte-addressable ABI;
                // scalar virtual calls are lowered here first.
                if (ptx_parameter_bits(parameter) < 16) return false;
            }
            return function.return_params.empty() ||
                   ptx_parameter_bits(function.return_params.front()) > 0;
        };
        const auto emit_direct_device_call = [&](
            const cumetal::ptx::EntryFunction& function,
            std::optional<std::string>* returned) -> bool {
            std::vector<std::pair<int, std::string>> arguments;
            auto& parameter_spaces =
                device_function_param_address_spaces_[function.name];
            if (parameter_spaces.size() < function.params.size()) {
                parameter_spaces.resize(function.params.size(), PointerAs::kUnknown);
            }
            for (std::size_t i = 0; i < function.params.size(); ++i) {
                const int bits = ptx_parameter_bits(function.params[i]);
                if (bits <= 0) return false;
                if (const auto provenance =
                        call_param_pointer_as_.find(arg_names[i]);
                    provenance != call_param_pointer_as_.end()) {
                    if (parameter_spaces[i] == PointerAs::kUnknown ||
                        parameter_spaces[i] == provenance->second) {
                        parameter_spaces[i] = provenance->second;
                    } else {
                        return false;
                    }
                }
                auto value = load_call_slot_value(os, arg_names[i], bits);
                if (!value) return false;
                arguments.emplace_back(bits, *value);
            }
            const int result_bits = function.return_params.empty()
                                        ? 0
                                        : ptx_parameter_bits(function.return_params.front());
            const std::string call_result =
                result_bits > 0 ? next_tmp("device_call") : std::string{};
            os << "  ";
            if (result_bits > 0) {
                os << call_result << " = ";
            }
            os << "call " << (result_bits > 0 ? llvm_int_type(result_bits) : "void")
               << " @" << function.name << "(";
            for (std::size_t i = 0; i < arguments.size(); ++i) {
                if (i > 0) os << ", ";
                os << llvm_int_type(arguments[i].first) << " " << arguments[i].second;
            }
            if (module_uses_device_heap_) {
                if (!arguments.empty()) os << ", ";
                os << "i32 addrspace(1)* %__cumetal_device_heap";
            }
            if (module_uses_device_printf_) {
                if (!arguments.empty() || module_uses_device_heap_) os << ", ";
                os << "i32 addrspace(1)* %__cumetal_printf_buffer, "
                      "i32 addrspace(2)* %__cumetal_printf_capacity";
            }
            if (module_uses_device_launch_queue_) {
                if (!arguments.empty() || module_uses_device_heap_ ||
                    module_uses_device_printf_) {
                    os << ", ";
                }
                os << "i32 addrspace(1)* %__cumetal_device_launch_queue, "
                      "i32 addrspace(2)* %__cumetal_device_launch_queue_capacity";
            }
            if (module_uses_device_clock_) {
                if (!arguments.empty() || module_uses_device_heap_ ||
                    module_uses_device_printf_ || module_uses_device_launch_queue_) {
                    os << ", ";
                }
                os << "i32 addrspace(1)* %__cumetal_device_clock";
            }
            if (module_uses_grid_barrier_) {
                if (!arguments.empty() || module_uses_device_heap_ ||
                    module_uses_device_printf_ || module_uses_device_launch_queue_ ||
                    module_uses_device_clock_) {
                    os << ", ";
                }
                os << "i32 addrspace(1)* %__cumetal_grid_barrier";
            }
            if (module_uses_grid_y_offset_) {
                if (!arguments.empty() || module_uses_device_heap_ ||
                    module_uses_device_printf_ || module_uses_device_launch_queue_ ||
                    module_uses_device_clock_ || module_uses_grid_barrier_) {
                    os << ", ";
                }
                os << "i32 addrspace(2)* %__cumetal_grid_y_offset";
            }
            os << ")\n";
            if (returned != nullptr && result_bits > 0) *returned = call_result;
            return true;
        };

        if (const auto* function = find_device_function(callee);
            function != nullptr && compatible_device_function(*function)) {
            std::optional<std::string> returned;
            if (!emit_direct_device_call(*function, &returned)) {
                return fail(instr, "device function call arguments are unavailable");
            }
            if (returned) {
                return store_ret_bits(*returned,
                                      ptx_parameter_bits(function->return_params.front()));
            }
            return true;
        }

        if (is_register_name(callee) && device_functions_ != nullptr) {
            std::vector<const cumetal::ptx::EntryFunction*> candidates;
            for (const auto& function : *device_functions_) {
                if (compatible_device_function(function)) candidates.push_back(&function);
            }
            if (candidates.empty()) {
                return fail(instr, "indirect call has no ABI-compatible device target");
            }
            const std::string token = emit_load_reg_bits(os, callee, 64);
            const std::string merge_label =
                "device_dispatch_merge_" + std::to_string(tmp_id_++);
            const std::string miss_label =
                "device_dispatch_miss_" + std::to_string(tmp_id_++);
            std::vector<std::string> check_labels;
            std::vector<std::string> call_labels;
            check_labels.reserve(candidates.size());
            call_labels.reserve(candidates.size());
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                check_labels.push_back(
                    "device_dispatch_check_" + std::to_string(tmp_id_++));
                call_labels.push_back(
                    "device_dispatch_call_" + std::to_string(tmp_id_++));
            }
            std::vector<std::pair<std::string, std::string>> returned_values;
            for (std::size_t i = 0; i < candidates.size(); ++i) {
                const std::string& check_label = check_labels[i];
                const std::string& call_label = call_labels[i];
                if (i == 0) os << "  br label %" << check_label << "\n\n";
                os << check_label << ":\n";
                const std::string matches = next_tmp("device_dispatch_match");
                os << "  " << matches << " = icmp eq i64 " << token << ", "
                   << stable_device_function_token(candidates[i]->name) << "\n";
                const std::string next_label =
                    i + 1 < candidates.size()
                        ? check_labels[i + 1]
                        : miss_label;
                os << "  br i1 " << matches << ", label %" << call_label
                   << ", label %" << next_label << "\n\n";
                os << call_label << ":\n";
                std::optional<std::string> returned;
                if (!emit_direct_device_call(*candidates[i], &returned)) {
                    return fail(instr, "indirect device call arguments are unavailable");
                }
                if (returned) returned_values.emplace_back(*returned, call_label);
                os << "  br label %" << merge_label << "\n\n";
            }
            os << miss_label << ":\n";
            declarations_.insert("declare void @llvm.trap()");
            os << "  call void @llvm.trap()\n  br label %" << merge_label << "\n\n";
            os << merge_label << ":\n";
            if (!dest_names.empty()) {
                const int bits = ptx_parameter_bits(candidates.front()->return_params.front());
                const std::string result = next_tmp("device_dispatch_result");
                os << "  " << result << " = phi " << llvm_int_type(bits) << " ";
                for (std::size_t i = 0; i < returned_values.size(); ++i) {
                    if (i > 0) os << ", ";
                    os << "[ " << returned_values[i].first << ", %"
                       << returned_values[i].second << " ]";
                }
                if (!returned_values.empty()) os << ", ";
                os << "[ 0, %" << miss_label << " ]\n";
                return store_ret_bits(result, bits);
            }
            return true;
        }

        if (callee == "vprintf") {
            const auto printf_it = printf_call_by_line_.find(instr.line);
            if (printf_it == printf_call_by_line_.end()) {
                return fail(instr, "vprintf ABI was not decoded by printf lowering");
            }
            const auto& call = *printf_it->second;
            if (call.arguments.size() > 1024) {
                return fail(instr, "vprintf argument tuple is unreasonably large");
            }

            std::uint32_t payload_words = 0;
            for (std::size_t i = 0; i < call.arguments.size(); ++i) {
                const int bits = i < call.argument_bits.size()
                                     ? call.argument_bits[i]
                                     : 32;
                if (bits != 32 && bits != 64) {
                    return fail(instr, "vprintf argument width is not 32 or 64 bits");
                }
                payload_words += static_cast<std::uint32_t>(bits / 32);
            }
            const std::uint32_t record_words = 2u + payload_words;
            const std::string old_pos = next_tmp("printf_pos");
            os << "  " << old_pos
               << " = atomicrmw add i32 addrspace(1)* %__cumetal_printf_buffer, i32 "
               << record_words << " monotonic\n";
            const std::string end_pos = next_tmp("printf_end");
            os << "  " << end_pos << " = add i32 " << old_pos << ", " << record_words << "\n";
            const std::string cap = next_tmp("printf_cap");
            os << "  " << cap
               << " = load i32, i32 addrspace(2)* %__cumetal_printf_capacity, align 4\n";
            const std::string fits = next_tmp("printf_fits");
            os << "  " << fits << " = icmp ult i32 " << end_pos << ", " << cap << "\n";

            const std::string write_label = "printf_write_" + std::to_string(tmp_id_++);
            const std::string done_label = "printf_done_" + std::to_string(tmp_id_++);
            os << "  br i1 " << fits << ", label %" << write_label
               << ", label %" << done_label << "\n\n";
            os << write_label << ":\n";

            auto emit_record_word = [&](std::uint32_t relative_index,
                                        const std::string& value) {
                const std::string index32 = next_tmp("printf_index");
                os << "  " << index32 << " = add i32 " << old_pos << ", "
                   << relative_index << "\n";
                const std::string index64 = next_tmp("printf_index64");
                os << "  " << index64 << " = zext i32 " << index32 << " to i64\n";
                const std::string word_ptr = next_tmp("printf_word");
                os << "  " << word_ptr
                   << " = getelementptr inbounds i32, i32 addrspace(1)* "
                      "%__cumetal_printf_buffer, i64 "
                   << index64 << "\n";
                os << "  store i32 " << value << ", i32 addrspace(1)* " << word_ptr
                   << ", align 4\n";
            };

            emit_record_word(1u, std::to_string(call.format_id));
            emit_record_word(2u, std::to_string(payload_words));
            std::uint32_t payload_offset = 0;
            for (std::size_t i = 0; i < call.arguments.size(); ++i) {
                const std::string arg = trim(call.arguments[i]);
                const int arg_bits = i < call.argument_bits.size()
                                         ? call.argument_bits[i]
                                         : 32;
                // Clang may pack an immediate directly into a v2.b32 tuple
                // (for example an expected tile size), so accept both register
                // values and integer literals as their raw ABI bits.
                auto bits = emit_integer_from_any(os, arg, arg_bits, false);
                if (!bits) {
                    return fail(instr, "vprintf decoded unsupported " +
                                           std::to_string(arg_bits) +
                                           "-bit argument '" + arg + "'");
                }
                if (arg_bits == 32) {
                    emit_record_word(3u + payload_offset, *bits);
                    ++payload_offset;
                } else {
                    const std::string low = next_tmp("printf_arg_low");
                    os << "  " << low << " = trunc i64 " << *bits << " to i32\n";
                    const std::string shifted = next_tmp("printf_arg_shift");
                    os << "  " << shifted << " = lshr i64 " << *bits << ", 32\n";
                    const std::string high = next_tmp("printf_arg_high");
                    os << "  " << high << " = trunc i64 " << shifted << " to i32\n";
                    emit_record_word(3u + payload_offset, low);
                    emit_record_word(4u + payload_offset, high);
                    payload_offset += 2;
                }
            }
            os << "  br label %" << done_label << "\n\n";
            os << done_label << ":\n";
            return store_ret_bits("0", 32);
        }

        if (callee == "__nv_frexp") {
            if (arg_names.size() < 2) return fail(instr, "__nv_frexp expects 2 args");
            auto input = load_call_slot_value(os, arg_names[0], 64);
            auto exponent_ptr_bits = load_call_slot_value(os, arg_names[1], 64);
            if (!input || !exponent_ptr_bits) {
                return fail(instr, "__nv_frexp args missing call slots");
            }

            // Implement binary64 frexp entirely in integer IR. Public Metal/AIR
            // rejects native double ALU, but CUDA's register/call ABI still uses
            // IEEE-754 bits. This preserves normals, signed zero, infinities and
            // NaNs exactly, and normalizes subnormals with ctlz.
            const std::string sign = next_tmp("frexp_sign");
            const std::string exp_shift = next_tmp("frexp_exp_shift");
            const std::string exp_raw = next_tmp("frexp_exp_raw");
            const std::string mantissa = next_tmp("frexp_mantissa");
            os << "  " << sign << " = and i64 " << *input
               << ", -9223372036854775808\n";
            os << "  " << exp_shift << " = lshr i64 " << *input << ", 52\n";
            os << "  " << exp_raw << " = and i64 " << exp_shift << ", 2047\n";
            os << "  " << mantissa << " = and i64 " << *input
               << ", 4503599627370495\n";

            const std::string zero_exp = next_tmp("frexp_zero_exp");
            const std::string zero_mantissa = next_tmp("frexp_zero_mantissa");
            const std::string is_zero = next_tmp("frexp_zero");
            const std::string is_special = next_tmp("frexp_special");
            const std::string nonzero_mantissa = next_tmp("frexp_nonzero_mantissa");
            const std::string is_subnormal = next_tmp("frexp_subnormal");
            os << "  " << zero_exp << " = icmp eq i64 " << exp_raw << ", 0\n";
            os << "  " << zero_mantissa << " = icmp eq i64 " << mantissa << ", 0\n";
            os << "  " << is_zero << " = and i1 " << zero_exp << ", " << zero_mantissa << "\n";
            os << "  " << is_special << " = icmp eq i64 " << exp_raw << ", 2047\n";
            os << "  " << nonzero_mantissa << " = xor i1 " << zero_mantissa << ", true\n";
            os << "  " << is_subnormal << " = and i1 " << zero_exp << ", "
               << nonzero_mantissa << "\n";

            declarations_.insert("declare i64 @llvm.ctlz.i64(i64, i1 immarg)");
            const std::string leading = next_tmp("frexp_leading");
            const std::string sub_shift = next_tmp("frexp_sub_shift");
            const std::string normalized_full = next_tmp("frexp_normalized_full");
            const std::string normalized_mantissa = next_tmp("frexp_normalized_mantissa");
            os << "  " << leading << " = call i64 @llvm.ctlz.i64(i64 " << mantissa
               << ", i1 false)\n";
            os << "  " << sub_shift << " = sub i64 " << leading << ", 11\n";
            os << "  " << normalized_full << " = shl i64 " << mantissa << ", "
               << sub_shift << "\n";
            os << "  " << normalized_mantissa << " = and i64 " << normalized_full
               << ", 4503599627370495\n";

            const std::string normal_result0 = next_tmp("frexp_normal_result0");
            const std::string normal_result = next_tmp("frexp_normal_result");
            const std::string sub_result0 = next_tmp("frexp_sub_result0");
            const std::string sub_result = next_tmp("frexp_sub_result");
            os << "  " << normal_result0 << " = or i64 " << sign
               << ", 4602678819172646912\n"; // sign | (1022 << 52)
            os << "  " << normal_result << " = or i64 " << normal_result0 << ", "
               << mantissa << "\n";
            os << "  " << sub_result0 << " = or i64 " << sign
               << ", 4602678819172646912\n";
            os << "  " << sub_result << " = or i64 " << sub_result0 << ", "
               << normalized_mantissa << "\n";
            const std::string finite_result = next_tmp("frexp_finite_result");
            const std::string zero_or_special = next_tmp("frexp_zero_or_special");
            const std::string result_bits = next_tmp("frexp_result");
            os << "  " << finite_result << " = select i1 " << is_subnormal << ", i64 "
               << sub_result << ", i64 " << normal_result << "\n";
            os << "  " << zero_or_special << " = or i1 " << is_zero << ", "
               << is_special << "\n";
            os << "  " << result_bits << " = select i1 " << zero_or_special << ", i64 "
               << *input << ", i64 " << finite_result << "\n";

            const std::string normal_exp64 = next_tmp("frexp_normal_exp64");
            const std::string sub_exp64 = next_tmp("frexp_sub_exp64");
            const std::string finite_exp64 = next_tmp("frexp_finite_exp64");
            const std::string exponent64 = next_tmp("frexp_exponent64");
            const std::string exponent32 = next_tmp("frexp_exponent");
            os << "  " << normal_exp64 << " = sub i64 " << exp_raw << ", 1022\n";
            os << "  " << sub_exp64 << " = sub i64 -1010, " << leading << "\n";
            os << "  " << finite_exp64 << " = select i1 " << is_subnormal << ", i64 "
               << sub_exp64 << ", i64 " << normal_exp64 << "\n";
            os << "  " << exponent64 << " = select i1 " << zero_or_special
               << ", i64 0, i64 " << finite_exp64 << "\n";
            os << "  " << exponent32 << " = trunc i64 " << exponent64 << " to i32\n";
            const std::string exponent_ptr = next_tmp("frexp_exponent_ptr");
            os << "  " << exponent_ptr << " = inttoptr i64 " << *exponent_ptr_bits
               << " to i32*\n";
            os << "  store i32 " << exponent32 << ", i32* " << exponent_ptr
               << ", align 4\n";
            return store_ret_bits(result_bits, 64);
        }

        if (callee == "__nv_abs") {
            if (arg_names.empty()) return fail(instr, "__nv_abs expects 1 arg");
            auto value = load_call_slot_value(os, arg_names[0], 32);
            if (!value) return fail(instr, "__nv_abs arg missing");
            const std::string is_negative = next_tmp("abs_negative");
            os << "  " << is_negative << " = icmp slt i32 " << *value << ", 0\n";
            const std::string negated = next_tmp("abs_negated");
            // Deliberately omit nsw: CUDA's integer abs preserves INT_MIN's
            // two's-complement bit pattern rather than introducing LLVM poison.
            os << "  " << negated << " = sub i32 0, " << *value << "\n";
            const std::string result = next_tmp("abs");
            os << "  " << result << " = select i1 " << is_negative << ", i32 "
               << negated << ", i32 " << *value << "\n";
            return store_ret_bits(result, 32);
        }

        if (callee == "__nv_clz" || callee == "__nv_clzll") {
            if (arg_names.empty()) return fail(instr, callee + " expects 1 arg");
            const int input_bits = callee == "__nv_clzll" ? 64 : 32;
            auto value = load_call_slot_value(os, arg_names[0], input_bits);
            if (!value) return fail(instr, callee + " arg missing");
            const std::string type = llvm_int_type(input_bits);
            declarations_.insert("declare " + type + " @llvm.ctlz." + type +
                                 "(" + type + ", i1)");
            const std::string count = next_tmp("clz");
            os << "  " << count << " = call " << type << " @llvm.ctlz." << type
               << "(" << type << " " << *value << ", i1 false)\n";
            if (input_bits == 32) return store_ret_bits(count, 32);
            const std::string narrowed = next_tmp("clz_i32");
            os << "  " << narrowed << " = trunc i64 " << count << " to i32\n";
            return store_ret_bits(narrowed, 32);
        }

        if (callee == "__nv_umulhi") {
            if (arg_names.size() < 2) return fail(instr, "__nv_umulhi expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, "__nv_umulhi args missing call slots");
            const std::string a64 = next_tmp("umulhi_a");
            const std::string b64 = next_tmp("umulhi_b");
            os << "  " << a64 << " = zext i32 " << *a << " to i64\n";
            os << "  " << b64 << " = zext i32 " << *b << " to i64\n";
            const std::string prod = next_tmp("umulhi_mul");
            os << "  " << prod << " = mul i64 " << a64 << ", " << b64 << "\n";
            const std::string shr = next_tmp("umulhi_shr");
            os << "  " << shr << " = lshr i64 " << prod << ", 32\n";
            const std::string hi = next_tmp("umulhi_hi");
            os << "  " << hi << " = trunc i64 " << shr << " to i32\n";
            return store_ret_bits(hi, 32);
        }

        if (callee == "__nv_mul24" || callee == "__nv_umul24") {
            if (arg_names.size() < 2) return fail(instr, callee + " expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, callee + " args missing call slots");

            std::string a24;
            std::string b24;
            if (callee == "__nv_mul24") {
                // CUDA __mul24 multiplies the sign-extended low 24 bits of
                // each operand and returns the low 32 bits of the product.
                const std::string a_shifted = next_tmp("mul24_a_shifted");
                const std::string b_shifted = next_tmp("mul24_b_shifted");
                a24 = next_tmp("mul24_a");
                b24 = next_tmp("mul24_b");
                os << "  " << a_shifted << " = shl i32 " << *a << ", 8\n";
                os << "  " << b_shifted << " = shl i32 " << *b << ", 8\n";
                os << "  " << a24 << " = ashr i32 " << a_shifted << ", 8\n";
                os << "  " << b24 << " = ashr i32 " << b_shifted << ", 8\n";
            } else {
                a24 = next_tmp("umul24_a");
                b24 = next_tmp("umul24_b");
                os << "  " << a24 << " = and i32 " << *a << ", 16777215\n";
                os << "  " << b24 << " = and i32 " << *b << ", 16777215\n";
            }
            const std::string product = next_tmp("mul24");
            os << "  " << product << " = mul i32 " << a24 << ", " << b24 << "\n";
            return store_ret_bits(product, 32);
        }

        if (callee == "__nv_popc") {
            if (arg_names.empty()) return fail(instr, "__nv_popc expects 1 arg");
            auto value = load_call_slot_value(os, arg_names[0], 32);
            if (!value) return fail(instr, "__nv_popc arg missing");
            declarations_.insert("declare i32 @llvm.ctpop.i32(i32)");
            const std::string count = next_tmp("popc");
            os << "  " << count << " = call i32 @llvm.ctpop.i32(i32 " << *value << ")\n";
            return store_ret_bits(count, 32);
        }

        if (callee == "__nv_ffs") {
            if (arg_names.empty()) return fail(instr, "__nv_ffs expects 1 arg");
            auto value = load_call_slot_value(os, arg_names[0], 32);
            if (!value) return fail(instr, "__nv_ffs arg missing");
            declarations_.insert("declare i32 @llvm.cttz.i32(i32, i1 immarg)");
            const std::string trailing = next_tmp("ffs_cttz");
            os << "  " << trailing << " = call i32 @llvm.cttz.i32(i32 " << *value
               << ", i1 false)\n";
            const std::string one_based = next_tmp("ffs_one_based");
            os << "  " << one_based << " = add i32 " << trailing << ", 1\n";
            const std::string is_zero = next_tmp("ffs_zero");
            os << "  " << is_zero << " = icmp eq i32 " << *value << ", 0\n";
            const std::string result = next_tmp("ffs");
            os << "  " << result << " = select i1 " << is_zero << ", i32 0, i32 "
               << one_based << "\n";
            return store_ret_bits(result, 32);
        }

        if (callee == "__nv_rsqrtf") {
            if (arg_names.empty()) return fail(instr, "__nv_rsqrtf expects 1 arg");
            auto bits = load_call_slot_value(os, arg_names[0], 32);
            if (!bits) return fail(instr, "__nv_rsqrtf arg missing");
            const std::string f = next_tmp("rsqrtf_bc");
            os << "  " << f << " = bitcast i32 " << *bits << " to float\n";
            declarations_.insert("declare float @air.fast_sqrt.f32(float)");
            const std::string s = next_tmp("rsqrtf_sqrt");
            os << "  " << s << " = call float @air.fast_sqrt.f32(float " << f << ")\n";
            const std::string r = next_tmp("rsqrtf_div");
            os << "  " << r << " = fdiv float 1.000000e+00, " << s << "\n";
            const std::string rbits = next_tmp("rsqrtf_i");
            os << "  " << rbits << " = bitcast float " << r << " to i32\n";
            return store_ret_bits(rbits, 32);
        }

        if (callee == "__nv_fabsf") {
            if (arg_names.empty()) return fail(instr, "__nv_fabsf expects 1 arg");
            auto bits = load_call_slot_value(os, arg_names[0], 32);
            if (!bits) return fail(instr, "__nv_fabsf arg missing");
            const std::string out = next_tmp("fabsf_bits");
            os << "  " << out << " = and i32 " << *bits << ", 2147483647\n";
            return store_ret_bits(out, 32);
        }

        if (callee == "__nv_fabs") {
            if (arg_names.empty()) return fail(instr, "__nv_fabs expects 1 arg");
            auto bits = load_call_slot_value(os, arg_names[0], 64);
            if (!bits) return fail(instr, "__nv_fabs arg missing");
            const std::string out = next_tmp("fabs_bits");
            os << "  " << out << " = and i64 " << *bits
               << ", 9223372036854775807\n";
            return store_ret_bits(out, 64);
        }

        // CUDA's fma(double,double,double) -- __nv_fma from fma(), __nv_fma_rn
        // from __fma_rn(). The PTX instruction form
        // (fma.rn.f64) is handled in emit_mad_or_fma; clang emits this call form
        // whenever the source calls fma()/__fma_rn() rather than relying on
        // contraction, which is what cuPDLP-C's gradient-step kernels do.
        if (callee == "__nv_fma_rn" || callee == "__nv_fma") {
            if (arg_names.size() < 3) return fail(instr, callee + " expects 3 args");
            auto a_bits = load_call_slot_value(os, arg_names[0], 64);
            auto b_bits = load_call_slot_value(os, arg_names[1], 64);
            auto c_bits = load_call_slot_value(os, arg_names[2], 64);
            if (!a_bits || !b_bits || !c_bits) return fail(instr, callee + " args missing");
            if (fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                const Fp64Pair a = fp64_pair_from_ieee_bits(os, *a_bits);
                const Fp64Pair b = fp64_pair_from_ieee_bits(os, *b_bits);
                const Fp64Pair c = fp64_pair_from_ieee_bits(os, *c_bits);
                const Fp64Pair product = emit_fp64_pair_mul(os, a, b);
                const Fp64Pair sum = emit_fp64_pair_add(os, product, c);
                return store_ret_bits(fp64_ieee_bits_from_pair(os, sum), 64);
            }
            // Native FP64: multiply-then-add, matching the non-emulated path in
            // emit_mad_or_fma (no llvm.fma intrinsic is declared in this module).
            const std::string a_val = next_tmp("fma_a");
            const std::string b_val = next_tmp("fma_b");
            const std::string c_val = next_tmp("fma_c");
            const std::string mul = next_tmp("fma_mul");
            const std::string add = next_tmp("fma_add");
            const std::string out = next_tmp("fma_bits");
            os << "  " << a_val << " = bitcast i64 " << *a_bits << " to double\n";
            os << "  " << b_val << " = bitcast i64 " << *b_bits << " to double\n";
            os << "  " << c_val << " = bitcast i64 " << *c_bits << " to double\n";
            os << "  " << mul << " = fmul double " << a_val << ", " << b_val << "\n";
            os << "  " << add << " = fadd double " << mul << ", " << c_val << "\n";
            os << "  " << out << " = bitcast double " << add << " to i64\n";
            return store_ret_bits(out, 64);
        }

        // CUDA interval arithmetic spells directed binary64 operations as
        // libdevice calls. Apple GPUs have no public FP64 ALU, so evaluate in
        // CuMetal's normalized FP32 pair and widen the low component by a
        // conservative pair-precision error bound in the requested direction.
        // The padding is intentionally larger than one pair ULP: preserving an
        // enclosure matters more here than reproducing round-to-nearest bits.
        const bool directed_double =
            callee == "__nv_dadd_rd" || callee == "__nv_dadd_ru" ||
            callee == "__nv_dmul_rd" || callee == "__nv_dmul_ru" ||
            callee == "__nv_ddiv_rd" || callee == "__nv_ddiv_ru";
        if (directed_double) {
            if (arg_names.size() < 2) return fail(instr, callee + " expects 2 args");
            auto a_bits = load_call_slot_value(os, arg_names[0], 64);
            auto b_bits = load_call_slot_value(os, arg_names[1], 64);
            if (!a_bits || !b_bits) return fail(instr, callee + " args missing");
            const Fp64Pair a = fp64_pair_from_ieee_bits(os, *a_bits);
            const Fp64Pair b = fp64_pair_from_ieee_bits(os, *b_bits);
            Fp64Pair result;
            if (callee.find("dadd") != std::string::npos) {
                result = emit_fp64_pair_add(os, a, b);
            } else if (callee.find("dmul") != std::string::npos) {
                result = emit_fp64_pair_mul(os, a, b);
            } else {
                result = emit_fp64_pair_div(os, a, b);
            }

            declarations_.insert("declare float @llvm.fabs.f32(float)");
            const std::string magnitude = next_tmp("directed_fp64_abs");
            os << "  " << magnitude << " = call float @llvm.fabs.f32(float "
               << result.hi << ")\n";
            const std::string relative = next_tmp("directed_fp64_relative");
            const std::string scale = emit_float_constant(
                os, 5.684341886080802e-14f, "directed_fp64_scale");
            os << "  " << relative << " = fmul float " << magnitude << ", "
               << scale << "\n";
            const std::string minimum = emit_float_constant(
                os, 1.1754943508222875e-38f, "directed_fp64_minimum");
            const std::string padding = next_tmp("directed_fp64_padding");
            os << "  " << padding << " = fadd float " << relative << ", "
               << minimum << "\n";
            std::string signed_padding = padding;
            if (callee.size() >= 3 &&
                callee.compare(callee.size() - 3, 3, "_rd") == 0) {
                signed_padding = next_tmp("directed_fp64_negative_padding");
                os << "  " << signed_padding << " = fneg float " << padding << "\n";
            }
            const std::string zero = emit_float_constant(os, 0.0f,
                                                         "directed_fp64_zero");
            result = emit_fp64_pair_add(os, result, Fp64Pair{zero, signed_padding});
            return store_ret_bits(fp64_ieee_bits_from_pair(os, result), 64);
        }

        if (callee == "__nv_longlong_as_double") {
            if (arg_names.empty()) return fail(instr, callee + " expects 1 arg");
            auto bits = load_call_slot_value(os, arg_names[0], 64);
            if (!bits) return fail(instr, callee + " arg missing");
            return store_ret_bits(*bits, 64);
        }

        if (callee == "__nv_float_as_int" || callee == "__nv_float_as_uint" ||
            callee == "__nv_int_as_float" || callee == "__nv_uint_as_float") {
            if (arg_names.empty()) return fail(instr, callee + " expects 1 arg");
            auto bits = load_call_slot_value(os, arg_names[0], 32);
            if (!bits) return fail(instr, callee + " arg missing");
            // PTX call parameters are untyped bit containers. CUDA's scalar
            // reinterpretation helpers therefore require no LLVM instruction.
            return store_ret_bits(*bits, 32);
        }

        // fmin/fmax on doubles. clang emits these whenever the source calls
        // fmin()/fmax() on a double, which HiGHS's PDLP convergence kernels do
        // for every constraint row. Without them those two kernels refuse to
        // lower, and a solver that ignores the launch error then reads an
        // untouched result buffer and "converges" on zeros.
        //
        // These are order comparisons, not arithmetic, so they are exact under
        // emulation: a normalized pair orders by hi and then by lo, which is
        // the same comparison emit_setp already makes for setp.lt.f64. The
        // selected operand is returned as its original IEEE bits, so nothing
        // round-trips through the pair.
        if (callee == "__nv_fmin" || callee == "__nv_fmax") {
            if (arg_names.size() < 2) return fail(instr, callee + " expects 2 args");
            auto a_bits = load_call_slot_value(os, arg_names[0], 64);
            auto b_bits = load_call_slot_value(os, arg_names[1], 64);
            if (!a_bits || !b_bits) return fail(instr, callee + " args missing");
            const bool want_max = (callee == "__nv_fmax");
            const char* strict_op = want_max ? "ogt" : "olt";
            std::string pick_a;
            if (fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                const Fp64Pair a = fp64_pair_from_ieee_bits(os, *a_bits);
                const Fp64Pair b = fp64_pair_from_ieee_bits(os, *b_bits);
                const std::string hi_ord = next_tmp("fmm_hiord");
                const std::string hi_eq = next_tmp("fmm_hieq");
                const std::string lo_ord = next_tmp("fmm_loord");
                const std::string tie = next_tmp("fmm_tie");
                const std::string ord = next_tmp("fmm_ord");
                os << "  " << hi_ord << " = fcmp " << strict_op << " float " << a.hi << ", "
                   << b.hi << "\n";
                os << "  " << hi_eq << " = fcmp oeq float " << a.hi << ", " << b.hi << "\n";
                os << "  " << lo_ord << " = fcmp " << strict_op << " float " << a.lo << ", "
                   << b.lo << "\n";
                os << "  " << tie << " = and i1 " << hi_eq << ", " << lo_ord << "\n";
                os << "  " << ord << " = or i1 " << hi_ord << ", " << tie << "\n";
                // CUDA returns the operand that is not NaN when exactly one is.
                // Every comparison above is false against a NaN, which already
                // gives fmin(NaN, b) = b; the other direction has to be said.
                const std::string b_nan = next_tmp("fmm_bnan");
                const std::string out_pred = next_tmp("fmm_pick");
                os << "  " << b_nan << " = fcmp uno float " << b.hi << ", " << b.hi << "\n";
                os << "  " << out_pred << " = or i1 " << ord << ", " << b_nan << "\n";
                pick_a = out_pred;
            } else {
                const std::string a = next_tmp("fmm_a");
                const std::string b = next_tmp("fmm_b");
                const std::string ord = next_tmp("fmm_ord");
                const std::string b_nan = next_tmp("fmm_bnan");
                const std::string out_pred = next_tmp("fmm_pick");
                os << "  " << a << " = bitcast i64 " << *a_bits << " to double\n";
                os << "  " << b << " = bitcast i64 " << *b_bits << " to double\n";
                os << "  " << ord << " = fcmp " << strict_op << " double " << a << ", " << b
                   << "\n";
                os << "  " << b_nan << " = fcmp uno double " << b << ", " << b << "\n";
                os << "  " << out_pred << " = or i1 " << ord << ", " << b_nan << "\n";
                pick_a = out_pred;
            }
            const std::string out = next_tmp("fmm_bits");
            os << "  " << out << " = select i1 " << pick_a << ", i64 " << *a_bits << ", i64 "
               << *b_bits << "\n";
            return store_ret_bits(out, 64);
        }

        if (callee == "__nv_fmaxf") {
            if (arg_names.size() < 2) return fail(instr, "__nv_fmaxf expects 2 args");
            auto a_bits = load_call_slot_value(os, arg_names[0], 32);
            auto b_bits = load_call_slot_value(os, arg_names[1], 32);
            if (!a_bits || !b_bits) return fail(instr, "__nv_fmaxf args missing");
            const std::string a = next_tmp("fmaxf_a");
            const std::string b = next_tmp("fmaxf_b");
            os << "  " << a << " = bitcast i32 " << *a_bits << " to float\n";
            os << "  " << b << " = bitcast i32 " << *b_bits << " to float\n";
            const std::string cmp = next_tmp("fmaxf_cmp");
            os << "  " << cmp << " = fcmp ogt float " << a << ", " << b << "\n";
            const std::string sel = next_tmp("fmaxf_sel");
            os << "  " << sel << " = select i1 " << cmp << ", float " << a << ", float " << b << "\n";
            const std::string bits = next_tmp("fmaxf_i");
            os << "  " << bits << " = bitcast float " << sel << " to i32\n";
            return store_ret_bits(bits, 32);
        }
        if (callee == "__nv_fminf") {
            if (arg_names.size() < 2) return fail(instr, "__nv_fminf expects 2 args");
            auto a_bits = load_call_slot_value(os, arg_names[0], 32);
            auto b_bits = load_call_slot_value(os, arg_names[1], 32);
            if (!a_bits || !b_bits) return fail(instr, "__nv_fminf args missing");
            const std::string a = next_tmp("fminf_a");
            const std::string b = next_tmp("fminf_b");
            os << "  " << a << " = bitcast i32 " << *a_bits << " to float\n";
            os << "  " << b << " = bitcast i32 " << *b_bits << " to float\n";
            const std::string cmp = next_tmp("fminf_cmp");
            os << "  " << cmp << " = fcmp olt float " << a << ", " << b << "\n";
            const std::string sel = next_tmp("fminf_sel");
            os << "  " << sel << " = select i1 " << cmp << ", float " << a << ", float " << b << "\n";
            const std::string bits = next_tmp("fminf_i");
            os << "  " << bits << " = bitcast float " << sel << " to i32\n";
            return store_ret_bits(bits, 32);
        }

        if (callee == "__nv_min" || callee == "__nv_umin") {
            if (arg_names.size() < 2) return fail(instr, callee + " expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, callee + " args missing");
            const std::string cmp = next_tmp("min_cmp");
            os << "  " << cmp << " = icmp " << (callee == "__nv_umin" ? "ult" : "slt")
               << " i32 " << *a << ", " << *b << "\n";
            const std::string sel = next_tmp("min_sel");
            os << "  " << sel << " = select i1 " << cmp << ", i32 " << *a << ", i32 " << *b << "\n";
            return store_ret_bits(sel, 32);
        }
        if (callee == "__nv_max" || callee == "__nv_umax") {
            if (arg_names.size() < 2) return fail(instr, callee + " expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, callee + " args missing");
            const std::string cmp = next_tmp("max_cmp");
            os << "  " << cmp << " = icmp " << (callee == "__nv_umax" ? "ugt" : "sgt")
               << " i32 " << *a << ", " << *b << "\n";
            const std::string sel = next_tmp("max_sel");
            os << "  " << sel << " = select i1 " << cmp << ", i32 " << *a << ", i32 " << *b << "\n";
            return store_ret_bits(sel, 32);
        }
        if (callee == "__nv_fast_sincosf") {
            if (arg_names.size() < 3) return fail(instr, "__nv_fast_sincosf expects 3 args");
            auto x = load_call_slot_f32(arg_names[0]);
            auto sin_ptr_bits = load_call_slot_value(os, arg_names[1], 64);
            auto cos_ptr_bits = load_call_slot_value(os, arg_names[2], 64);
            if (!x || !sin_ptr_bits || !cos_ptr_bits) {
                return fail(instr, "__nv_fast_sincosf args missing");
            }
            declarations_.insert("declare float @air.fast_sin.f32(float)");
            declarations_.insert("declare float @air.fast_cos.f32(float)");
            const std::string sin_value = next_tmp("sincos_sin");
            const std::string cos_value = next_tmp("sincos_cos");
            os << "  " << sin_value << " = call float @air.fast_sin.f32(float " << *x << ")\n";
            os << "  " << cos_value << " = call float @air.fast_cos.f32(float " << *x << ")\n";
            const std::string sin_ptr = next_tmp("sincos_sin_ptr");
            const std::string cos_ptr = next_tmp("sincos_cos_ptr");
            os << "  " << sin_ptr << " = inttoptr i64 " << *sin_ptr_bits << " to float*\n";
            os << "  " << cos_ptr << " = inttoptr i64 " << *cos_ptr_bits << " to float*\n";
            os << "  store float " << sin_value << ", float* " << sin_ptr << ", align 4\n";
            os << "  store float " << cos_value << ", float* " << cos_ptr << ", align 4\n";
            return true;
        }
        if (callee == "__nv_fast_fdividef") {
            if (arg_names.size() < 2) return fail(instr, "__nv_fast_fdividef expects 2 args");
            auto numerator = load_call_slot_f32(arg_names[0]);
            auto denominator = load_call_slot_f32(arg_names[1]);
            if (!numerator || !denominator) return fail(instr, "__nv_fast_fdividef args missing");
            const std::string out = next_tmp("fast_fdividef");
            os << "  " << out << " = fdiv fast float " << *numerator << ", " << *denominator << "\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_sqrt") {
            if (arg_names.empty()) return fail(instr, "__nv_sqrt expects 1 arg");
            auto input_bits = load_call_slot_value(os, arg_names[0], 64);
            if (!input_bits) return fail(instr, "__nv_sqrt arg missing");
            if (fp64_mode_ != cumetal::ptx::Fp64Mode::kEmulate) {
                declarations_.insert("declare double @llvm.sqrt.f64(double)");
                const std::string input = next_tmp("sqrt_f64_input");
                const std::string root = next_tmp("sqrt_f64");
                const std::string bits = next_tmp("sqrt_f64_bits");
                os << "  " << input << " = bitcast i64 " << *input_bits << " to double\n";
                os << "  " << root << " = call double @llvm.sqrt.f64(double " << input << ")\n";
                os << "  " << bits << " = bitcast double " << root << " to i64\n";
                return store_ret_bits(bits, 64);
            }
            declarations_.insert("declare float @air.fast_sqrt.f32(float)");
            const Fp64Pair input = fp64_pair_from_ieee_bits(os, *input_bits);
            const std::string approximate = next_tmp("sqrt_pair_approximate");
            const std::string root = next_tmp("sqrt_pair_root");
            const std::string zero = next_tmp("sqrt_pair_zero");
            os << "  " << approximate << " = fadd float " << input.hi << ", "
               << input.lo << "\n";
            os << "  " << root << " = call float @air.fast_sqrt.f32(float "
               << approximate << ")\n";
            os << "  " << zero << " = fsub float " << root << ", " << root << "\n";
            const Fp64Pair initial{root, zero};
            const Fp64Pair square = emit_fp64_pair_mul(os, initial, initial);
            const Fp64Pair residual = emit_fp64_pair_add(os, input, square, true);
            const Fp64Pair twice_root = emit_fp64_pair_add(os, initial, initial);
            Fp64Pair correction = emit_fp64_pair_div(os, residual, twice_root);
            const std::string is_zero = next_tmp("sqrt_pair_is_zero");
            os << "  " << is_zero << " = fcmp oeq float " << root
               << ", 0.000000e+00\n";
            const std::string correction_hi = next_tmp("sqrt_pair_correction_hi");
            const std::string correction_lo = next_tmp("sqrt_pair_correction_lo");
            os << "  " << correction_hi << " = select i1 " << is_zero
               << ", float 0.000000e+00, float " << correction.hi << "\n";
            os << "  " << correction_lo << " = select i1 " << is_zero
               << ", float 0.000000e+00, float " << correction.lo << "\n";
            const Fp64Pair refined = emit_fp64_pair_add(
                os, initial, Fp64Pair{correction_hi, correction_lo});
            return store_ret_bits(fp64_ieee_bits_from_pair(os, refined), 64);
        }
        // ---- libdevice float surface ---------------------------------------
        //
        // Every libdevice function used to need its own hand-written block, so
        // anything not explicitly listed aborted the entire kernel -- one
        // missing entry (__nv_tanf) was enough to make a whole ray tracer
        // unlowerable. The mapping is declarative instead, and every entry is
        // probed against the host libm by tests/cuda_projects/libdevice so a
        // wrong symbol is caught by measurement rather than assumed correct.
        struct FloatBuiltin {
            const char* nv;
            const char* sym;
            int arity;
        };
        static const FloatBuiltin kFloatBuiltins[] = {
            {"__nv_sqrtf", "air.fast_sqrt.f32", 1},
            {"__nv_expf", "air.fast_exp.f32", 1},
            {"__nv_fast_expf", "air.fast_exp.f32", 1},
            {"__nv_exp2f", "air.fast_exp2.f32", 1},
            {"__nv_exp10f", "air.fast_exp10.f32", 1},
            {"__nv_logf", "air.fast_log.f32", 1},
            {"__nv_log2f", "air.fast_log2.f32", 1},
            {"__nv_log10f", "air.fast_log10.f32", 1},
            {"__nv_sinf", "air.fast_sin.f32", 1},
            {"__nv_cosf", "air.fast_cos.f32", 1},
            {"__nv_tanf", "air.fast_tan.f32", 1},
            {"__nv_asinf", "air.fast_asin.f32", 1},
            {"__nv_acosf", "air.fast_acos.f32", 1},
            {"__nv_atanf", "air.fast_atan.f32", 1},
            {"__nv_sinhf", "air.fast_sinh.f32", 1},
            {"__nv_coshf", "air.fast_cosh.f32", 1},
            {"__nv_tanhf", "air.fast_tanh.f32", 1},
            {"__nv_asinhf", "air.fast_asinh.f32", 1},
            {"__nv_acoshf", "air.fast_acosh.f32", 1},
            {"__nv_atanhf", "air.fast_atanh.f32", 1},
            {"__nv_floorf", "llvm.floor.f32", 1},
            {"__nv_ceilf", "llvm.ceil.f32", 1},
            {"__nv_truncf", "llvm.trunc.f32", 1},
            {"__nv_roundf", "llvm.round.f32", 1},
            {"__nv_rintf", "llvm.rint.f32", 1},
            {"__nv_nearbyintf", "llvm.nearbyint.f32", 1},
            {"__nv_powf", "air.fast_pow.f32", 2},
            {"__nv_atan2f", "air.fast_atan2.f32", 2},
            {"__nv_fmodf", "air.fmod.f32", 2},
            {"__nv_copysignf", "llvm.copysign.f32", 2},
            {"__nv_fmaf", "llvm.fma.f32", 3},
        };
        for (const FloatBuiltin& fb : kFloatBuiltins) {
            if (callee != fb.nv) continue;
            if (static_cast<int>(arg_names.size()) < fb.arity) {
                return fail(instr, callee + " expects " + std::to_string(fb.arity) + " arg(s)");
            }
            std::vector<std::string> values;
            for (int i = 0; i < fb.arity; ++i) {
                auto v = load_call_slot_f32(arg_names[static_cast<std::size_t>(i)]);
                if (!v) return fail(instr, callee + " arg missing");
                values.push_back(*v);
            }
            std::string decl = "declare float @" + std::string(fb.sym) + "(float";
            for (int i = 1; i < fb.arity; ++i) decl += ", float";
            decl += ")";
            declarations_.insert(decl);
            const std::string out = next_tmp("nvmath");
            os << "  " << out << " = call float @" << fb.sym << "(";
            for (int i = 0; i < fb.arity; ++i) {
                if (i != 0) os << ", ";
                os << "float " << values[static_cast<std::size_t>(i)];
            }
            os << ")\n";
            return store_ret_f32(out);
        }

        // Functions with no direct Metal builtin, expressed exactly in terms of
        // ones that do. Kept separate from the table so the substitution is
        // visible rather than hidden behind a symbol name.
        if (callee == "__nv_expm1f" || callee == "__nv_log1pf") {
            if (arg_names.empty()) return fail(instr, callee + " expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, callee + " arg missing");
            const bool is_expm1 = (callee == "__nv_expm1f");
            declarations_.insert(is_expm1 ? "declare float @air.fast_exp.f32(float)"
                                          : "declare float @air.fast_log.f32(float)");
            const std::string adj = next_tmp(is_expm1 ? "expm1_e" : "log1p_a");
            if (is_expm1) {
                os << "  " << adj << " = call float @air.fast_exp.f32(float " << *x << ")\n";
                const std::string out = next_tmp("expm1");
                os << "  " << out << " = fsub float " << adj << ", 1.000000e+00\n";
                return store_ret_f32(out);
            }
            os << "  " << adj << " = fadd float " << *x << ", 1.000000e+00\n";
            const std::string out = next_tmp("log1p");
            os << "  " << out << " = call float @air.fast_log.f32(float " << adj << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_fdimf") {
            // fdim(x,y) = (x > y) ? x - y : +0
            if (arg_names.size() < 2) return fail(instr, "__nv_fdimf expects 2 args");
            auto x = load_call_slot_f32(arg_names[0]);
            auto y = load_call_slot_f32(arg_names[1]);
            if (!x || !y) return fail(instr, "__nv_fdimf args missing");
            const std::string cmp = next_tmp("fdim_cmp");
            os << "  " << cmp << " = fcmp ogt float " << *x << ", " << *y << "\n";
            const std::string diff = next_tmp("fdim_sub");
            os << "  " << diff << " = fsub float " << *x << ", " << *y << "\n";
            const std::string out = next_tmp("fdim");
            os << "  " << out << " = select i1 " << cmp << ", float " << diff
               << ", float 0.000000e+00\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_hypotf") {
            // Metal has no hypot builtin. sqrt(x*x + y*y) matches it except for
            // intermediate overflow when |x| or |y| exceeds ~1e19, where the
            // true hypot is still finite; that is the documented limit here.
            if (arg_names.size() < 2) return fail(instr, "__nv_hypotf expects 2 args");
            auto x = load_call_slot_f32(arg_names[0]);
            auto y = load_call_slot_f32(arg_names[1]);
            if (!x || !y) return fail(instr, "__nv_hypotf args missing");
            declarations_.insert("declare float @air.fast_sqrt.f32(float)");
            const std::string xx = next_tmp("hypot_xx");
            const std::string yy = next_tmp("hypot_yy");
            const std::string sum = next_tmp("hypot_sum");
            const std::string out = next_tmp("hypot");
            os << "  " << xx << " = fmul float " << *x << ", " << *x << "\n";
            os << "  " << yy << " = fmul float " << *y << ", " << *y << "\n";
            os << "  " << sum << " = fadd float " << xx << ", " << yy << "\n";
            os << "  " << out << " = call float @air.fast_sqrt.f32(float " << sum << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_cbrtf") {
            // pow() is undefined for a negative base, so fold the sign out and
            // put it back: cbrt(x) = copysign(pow(|x|, 1/3), x).
            if (arg_names.empty()) return fail(instr, "__nv_cbrtf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_cbrtf arg missing");
            declarations_.insert("declare float @air.fast_pow.f32(float, float)");
            declarations_.insert("declare float @llvm.fabs.f32(float)");
            declarations_.insert("declare float @llvm.copysign.f32(float, float)");
            const std::string ax = next_tmp("cbrt_abs");
            const std::string root = next_tmp("cbrt_pow");
            const std::string out = next_tmp("cbrt");
            os << "  " << ax << " = call float @llvm.fabs.f32(float " << *x << ")\n";
            os << "  " << root << " = call float @air.fast_pow.f32(float " << ax
               << ", float 0x3FD5555560000000)\n";
            os << "  " << out << " = call float @llvm.copysign.f32(float " << root
               << ", float " << *x << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_remainderf") {
            // IEEE remainder: x - y * rint(x/y), with rint's round-half-to-even
            // being exactly what distinguishes it from fmod's truncation.
            if (arg_names.size() < 2) return fail(instr, "__nv_remainderf expects 2 args");
            auto x = load_call_slot_f32(arg_names[0]);
            auto y = load_call_slot_f32(arg_names[1]);
            if (!x || !y) return fail(instr, "__nv_remainderf args missing");
            declarations_.insert("declare float @llvm.rint.f32(float)");
            const std::string q = next_tmp("rem_q");
            const std::string n = next_tmp("rem_n");
            const std::string prod = next_tmp("rem_p");
            const std::string out = next_tmp("rem");
            os << "  " << q << " = fdiv float " << *x << ", " << *y << "\n";
            os << "  " << n << " = call float @llvm.rint.f32(float " << q << ")\n";
            os << "  " << prod << " = fmul float " << n << ", " << *y << "\n";
            os << "  " << out << " = fsub float " << *x << ", " << prod << "\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_erff" || callee == "__nv_erfcf") {
            // Metal has no erf/erfc. Abramowitz & Stegun 7.1.26 gives
            //   erfc(|x|) ~ P(t) * exp(-x^2),  t = 1/(1 + 0.3275911|x|)
            // with |absolute error| <= 1.5e-7.
            //
            // erfc is computed from that product directly rather than as
            // 1 - erf(x): the subtraction cancels catastrophically for large x
            // (erfc(5) is ~1.5e-12, so 1 - erf would return a flat 0), and a
            // silently-zero tail is exactly the kind of wrong answer that must
            // not ship. erf keeps the 1 - P*exp form, which is accurate for
            // small x and saturates correctly to +-1 for large x.
            if (arg_names.empty()) return fail(instr, callee + " expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, callee + " arg missing");
            declarations_.insert("declare float @llvm.fabs.f32(float)");
            declarations_.insert("declare float @llvm.copysign.f32(float, float)");
            declarations_.insert("declare float @air.fast_exp.f32(float)");

            const std::string ax = next_tmp("erf_ax");
            os << "  " << ax << " = call float @llvm.fabs.f32(float " << *x << ")\n";
            const std::string td = next_tmp("erf_td");
            const std::string td1 = next_tmp("erf_td1");
            os << "  " << td << " = fmul float " << ax << ", 0x3FD4F740A0000000\n";
            os << "  " << td1 << " = fadd float " << td << ", 1.000000e+00\n";
            const std::string t = next_tmp("erf_t");
            os << "  " << t << " = fdiv float 1.000000e+00, " << td1 << "\n";
            // Horner: ((((a5 t + a4) t + a3) t + a2) t + a1) t
            static const char* kA[5] = {
                "0x3FF0FB8440000000",   // a5 =  1.061405429
                "0xBFF7401C60000000",   // a4 = -1.453152027
                "0x3FF6BE1C60000000",   // a3 =  1.421413741
                "0xBFD23531C0000000",   // a2 = -0.284496736
                "0x3FD04F20C0000000",   // a1 =  0.254829592
            };
            std::string acc = kA[0];
            for (int i = 1; i < 5; ++i) {
                const std::string m = next_tmp("erf_m");
                const std::string a = next_tmp("erf_a");
                os << "  " << m << " = fmul float " << acc << ", " << t << "\n";
                os << "  " << a << " = fadd float " << m << ", " << kA[i] << "\n";
                acc = a;
            }
            const std::string poly = next_tmp("erf_poly");
            os << "  " << poly << " = fmul float " << acc << ", " << t << "\n";
            const std::string x2 = next_tmp("erf_x2");
            const std::string nx2 = next_tmp("erf_nx2");
            os << "  " << x2 << " = fmul float " << ax << ", " << ax << "\n";
            os << "  " << nx2 << " = fsub float -0.000000e+00, " << x2 << "\n";
            const std::string ex = next_tmp("erf_exp");
            os << "  " << ex << " = call float @air.fast_exp.f32(float " << nx2 << ")\n";
            const std::string tail = next_tmp("erf_tail");  // == erfc(|x|)
            os << "  " << tail << " = fmul float " << poly << ", " << ex << "\n";

            if (callee == "__nv_erff") {
                const std::string mag = next_tmp("erf_mag");
                os << "  " << mag << " = fsub float 1.000000e+00, " << tail << "\n";
                const std::string out = next_tmp("erf");
                os << "  " << out << " = call float @llvm.copysign.f32(float " << mag
                   << ", float " << *x << ")\n";
                return store_ret_f32(out);
            }
            // erfc(x) = tail for x >= 0, and 2 - tail for x < 0.
            const std::string neg = next_tmp("erfc_neg");
            os << "  " << neg << " = fcmp olt float " << *x << ", 0.000000e+00\n";
            const std::string refl = next_tmp("erfc_refl");
            os << "  " << refl << " = fsub float 2.000000e+00, " << tail << "\n";
            const std::string out = next_tmp("erfc");
            os << "  " << out << " = select i1 " << neg << ", float " << refl << ", float "
               << tail << "\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_saturatef") {
            if (arg_names.empty()) return fail(instr, "__nv_saturatef expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_saturatef arg missing");
            const std::string lo = next_tmp("sat_lo");
            os << "  " << lo << " = fcmp ogt float " << *x << ", 0.000000e+00\n";
            const std::string c0 = next_tmp("sat_c0");
            os << "  " << c0 << " = select i1 " << lo << ", float " << *x << ", float 0.000000e+00\n";
            const std::string hi = next_tmp("sat_hi");
            os << "  " << hi << " = fcmp olt float " << c0 << ", 1.000000e+00\n";
            const std::string out = next_tmp("sat");
            os << "  " << out << " = select i1 " << hi << ", float " << c0
               << ", float 1.000000e+00\n";
            return store_ret_f32(out);
        }

        return fail(instr, "unsupported call target '" + callee + "'");
    }

    bool emit_bfe(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "bfe requires dst, a, b, c");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind != PtxTypeSpec::Kind::kInt || (ty.bits != 32 && ty.bits != 64)) {
            return fail(instr, "unsupported bfe type");
        }
        const int bits = ty.bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, false);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, false);
        auto c = emit_integer_from_any(os, instr.operands[3], bits, false);
        if (!a || !b || !c) {
            return fail(instr, "bfe operands unsupported");
        }

        const std::string shifted = next_tmp("bfe_sh");
        os << "  " << shifted << " = lshr " << llvm_int_type(bits) << " " << *a << ", " << *b << "\n";

        const std::string width_is_zero = next_tmp("bfe_w0");
        os << "  " << width_is_zero << " = icmp eq " << llvm_int_type(bits) << " " << *c << ", 0\n";
        const std::string width_is_full = next_tmp("bfe_wfull");
        os << "  " << width_is_full << " = icmp uge " << llvm_int_type(bits) << " " << *c << ", " << bits << "\n";
        const std::string width_nz = next_tmp("bfe_wnz");
        os << "  " << width_nz << " = select i1 " << width_is_zero << ", " << llvm_int_type(bits) << " 1, "
           << llvm_int_type(bits) << " " << *c << "\n";
        const std::string width_shift_safe = next_tmp("bfe_wsafe");
        os << "  " << width_shift_safe << " = select i1 " << width_is_full << ", " << llvm_int_type(bits)
           << " " << (bits - 1) << ", " << llvm_int_type(bits) << " " << width_nz << "\n";
        const std::string one_sh = next_tmp("bfe_onesh");
        os << "  " << one_sh << " = shl " << llvm_int_type(bits) << " 1, " << width_shift_safe << "\n";
        const std::string mask_nz = next_tmp("bfe_masknz");
        os << "  " << mask_nz << " = sub " << llvm_int_type(bits) << " " << one_sh << ", 1\n";
        const std::string mask_full = next_tmp("bfe_maskfull");
        os << "  " << mask_full << " = xor " << llvm_int_type(bits) << " 0, -1\n";
        const std::string mask = next_tmp("bfe_mask");
        os << "  " << mask << " = select i1 " << width_is_full << ", " << llvm_int_type(bits) << " " << mask_full
           << ", " << llvm_int_type(bits) << " " << mask_nz << "\n";
        const std::string extracted = next_tmp("bfe_ext");
        os << "  " << extracted << " = and " << llvm_int_type(bits) << " " << shifted << ", " << mask << "\n";

        std::string result_bits = extracted;
        if (ty.is_signed) {
            const std::string sign_shift = next_tmp("bfe_ss");
            os << "  " << sign_shift << " = sub " << llvm_int_type(bits) << " " << bits << ", " << width_nz << "\n";
            const std::string left = next_tmp("bfe_left");
            os << "  " << left << " = shl " << llvm_int_type(bits) << " " << extracted << ", " << sign_shift << "\n";
            const std::string ashr = next_tmp("bfe_ashr");
            os << "  " << ashr << " = ashr " << llvm_int_type(bits) << " " << left << ", " << sign_shift << "\n";
            const std::string zero_val = "0";
            const std::string sel = next_tmp("bfe_sel");
            os << "  " << sel << " = select i1 " << width_is_zero << ", " << llvm_int_type(bits) << " " << zero_val
               << ", " << llvm_int_type(bits) << " " << ashr << "\n";
            result_bits = sel;
        } else {
            const std::string sel = next_tmp("bfe_usel");
            os << "  " << sel << " = select i1 " << width_is_zero << ", " << llvm_int_type(bits) << " 0, "
               << llvm_int_type(bits) << " " << extracted << "\n";
            result_bits = sel;
        }

        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result_bits, bits);
    }

    // ── 64-bit atomics ────────────────────────────────────────────────────
    // Metal has no 64-bit atomic of any width, so a .b64/.f64 atomic is
    // serialized behind a lock taken from a bank of the 32-bit atomics Metal
    // does have, hashed on the target address. The bank is one device buffer
    // the runtime binds at reserved index 29 and shares across every kernel
    // and stream, which is what makes the exclusion device-wide rather than
    // per-launch. Two addresses colliding in the hash cost serialization, not
    // correctness.
    static constexpr std::size_t kLockBankBindingIndex =
        cumetal::ptx::kAtomicLockBankBindingIndex;

    std::string lock_bank_param_;

    // Appended lazily: only a kernel that actually performs a 64-bit atomic
    // pays for the binding. entry_requires_atomic_lock_bank() answers the same
    // question for the runtime from the same PTX, so the two cannot disagree.
    const std::string& ensure_lock_bank_param() {
        static const std::string kNone;
        if (!lock_bank_param_.empty()) return lock_bank_param_;
        if (params_ == nullptr || arg_decls_ == nullptr ||
            params_->size() > kLockBankBindingIndex) {
            return kNone;
        }
        lock_bank_param_ = "__cumetal_atomic_lock_bank";
        arg_decls_->push_back("i8 addrspace(1)* %" + lock_bank_param_);
        params_->push_back({.ptx_type = ".b8",
                            .llvm_type = "i8 addrspace(1)*",
                            .name = lock_bank_param_,
                            .raw_name = lock_bank_param_});
        return lock_bank_param_;
    }

    // One 64-bit read-modify-write under a bank lock. `src_operand` is the
    // value operand (for cas, the *new* value; `cmp_operand` is the comparand),
    // and `dst_reg` receives the pre-operation value when the PTX asked for it.
    //
    // The loop shape matters more than the hash does. A lane must never wait on
    // a lock held by a lane it is executing in lockstep with, so the critical
    // section is a straight-line if-body inside the retry loop: whoever wins an
    // iteration finishes and releases within that same iteration, and only then
    // does the group go round again for the losers. Spinning *inside* the
    // acquire, the shape a naive lock takes, deadlocks a SIMD group instead.
    bool emit_wide_atomic(std::ostringstream& os,
                          const cumetal::ptx::EntryFunction::Instruction& instr,
                          const std::string& operation,
                          const PtxTypeSpec& ty,
                          int addr_space,
                          const std::string& addr_i64,
                          const std::string& src_operand,
                          const std::string* cmp_operand,
                          const std::string* dst_reg) {
        const bool is_float = (ty.kind == PtxTypeSpec::Kind::kFloat);
        if (is_float && operation != "add" && operation != "exch") {
            return fail(instr, "atom: 64-bit float '" + operation + "' is not supported");
        }
        if (operation != "add" && operation != "and" && operation != "or" &&
            operation != "xor" && operation != "min" && operation != "max" &&
            operation != "exch" && operation != "cas") {
            return fail(instr, "atom: 64-bit '" + operation + "' is not supported");
        }

        const std::string bank = ensure_lock_bank_param();
        if (bank.empty()) {
            return fail(instr,
                        "atom: kernel argument ABI conflicts with the 64-bit atomic "
                        "lock bank at reserved buffer index 29");
        }

        // Both operands are loop-invariant; decode them before the retry loop.
        auto src_bits = emit_integer_from_any(os, src_operand, 64, false);
        if (!src_bits) return fail(instr, "atom: 64-bit source operand unsupported");
        std::optional<std::string> cmp_bits;
        if (cmp_operand != nullptr) {
            cmp_bits = emit_integer_from_any(os, *cmp_operand, 64, false);
            if (!cmp_bits) return fail(instr, "atom.cas: 64-bit comparand unsupported");
        }

        // Hash the 8-byte-aligned address so adjacent words do not all queue on
        // one lock. Knuth's multiplicative constant, keeping the high bits.
        const std::string word = next_tmp("wa_word");
        os << "  " << word << " = lshr i64 " << addr_i64 << ", 3\n";
        const std::string word32 = next_tmp("wa_word32");
        os << "  " << word32 << " = trunc i64 " << word << " to i32\n";
        const std::string mixed = next_tmp("wa_mix");
        os << "  " << mixed << " = mul i32 " << word32 << ", -1640531527\n";
        int slot_shift = 32;
        for (std::size_t n = cumetal::ptx::kAtomicLockBankSlots; n > 1; n >>= 1) --slot_shift;
        const std::string slot = next_tmp("wa_slot");
        os << "  " << slot << " = lshr i32 " << mixed << ", " << slot_shift << "\n";
        const std::string bank32 = next_tmp("wa_bank");
        os << "  " << bank32 << " = bitcast i8 addrspace(1)* %" << bank
           << " to i32 addrspace(1)*\n";
        const std::string lock = next_tmp("wa_lock");
        os << "  " << lock << " = getelementptr inbounds i32, i32 addrspace(1)* " << bank32
           << ", i32 " << slot << "\n";

        const std::string payload_i8 = next_tmp("wa_i2p");
        os << "  " << payload_i8 << " = inttoptr i64 " << addr_i64 << " to i8 addrspace("
           << addr_space << ")*\n";
        const std::string payload = next_tmp("wa_ptr");
        os << "  " << payload << " = bitcast i8 addrspace(" << addr_space << ")* " << payload_i8
           << " to i64 addrspace(" << addr_space << ")*\n";

        declarations_.insert("declare void @air.atomic.fence(i32, i32, i32)");
        // An LLVM `atomicrmw` inside a loop is not what the AIR backend expects,
        // and it does not say so: `xcrun metallib` either bus-errored outright
        // or emitted a kernel whose spin never made progress -- two threads on
        // one address hung the GPU. Metal spells an atomic as a call, the same
        // way it spells a fence (see emit_membar_or_fence), and the loop below
        // mirrors the CFG `xcrun metal` itself produces for a spin lock:
        // attempt at the header, test at the latch.
        //   air.atomic.global.xchg.i32(ptr, value, memory_order, scope, volatile)
        declarations_.insert(
            "declare i32 @air.atomic.global.xchg.i32(i32 addrspace(1)* nocapture, i32, i32, "
            "i32, i1)");

        const std::string tag = next_tmp("wa").substr(1);
        const std::string pre = tag + "_pre";
        const std::string head = tag + "_head";
        const std::string crit = tag + "_crit";
        const std::string latch = tag + "_latch";
        const std::string done_lbl = tag + "_done";

        const std::string done = next_tmp("wa_done");
        const std::string oldp = next_tmp("wa_oldphi");
        const std::string done2 = next_tmp("wa_done2");
        const std::string old2 = next_tmp("wa_old2");

        // The enclosing block's label is not known here, so branch into one we
        // name ourselves and use that as the loop's entry predecessor.
        os << "  br label %" << pre << "\n";
        os << pre << ":\n";
        os << "  br label %" << head << "\n";

        os << head << ":\n";
        os << "  " << done << " = phi i1 [ false, %" << pre << " ], [ " << done2 << ", %"
           << latch << " ]\n";
        os << "  " << oldp << " = phi i64 [ 0, %" << pre << " ], [ " << old2 << ", %"
           << latch << " ]\n";
        const std::string got = next_tmp("wa_got");
        os << "  " << got << " = call i32 @air.atomic.global.xchg.i32(i32 addrspace(1)* "
           << lock << ", i32 1, i32 0, i32 2, i1 true)\n";
        const std::string acq = next_tmp("wa_acq");
        os << "  " << acq << " = icmp eq i32 " << got << ", 0\n";
        os << "  br i1 " << acq << ", label %" << crit << ", label %" << latch << "\n";

        os << crit << ":\n";
        // AIR spells a fence as a call; the `fence` instruction crashes the
        // Metal compiler service (see emit_membar_or_fence).
        os << "  call void @air.atomic.fence(i32 3, i32 5, i32 2)\n";
        const std::string old = next_tmp("wa_old");
        os << "  " << old << " = load i64, i64 addrspace(" << addr_space << ")* " << payload
           << ", align 8\n";

        std::string newval;
        if (operation == "exch") {
            newval = *src_bits;
        } else if (operation == "cas") {
            const std::string eq = next_tmp("wa_caseq");
            os << "  " << eq << " = icmp eq i64 " << old << ", " << *cmp_bits << "\n";
            newval = next_tmp("wa_casnew");
            os << "  " << newval << " = select i1 " << eq << ", i64 " << *src_bits << ", i64 "
               << old << "\n";
        } else if (is_float) {
            if (fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                const Fp64Pair a = fp64_pair_from_ieee_bits(os, old);
                const Fp64Pair b = fp64_pair_from_ieee_bits(os, *src_bits);
                newval = fp64_ieee_bits_from_pair(os, emit_fp64_pair_add(os, a, b));
            } else {
                const std::string ad = next_tmp("wa_ad");
                const std::string bd = next_tmp("wa_bd");
                const std::string sd = next_tmp("wa_sd");
                os << "  " << ad << " = bitcast i64 " << old << " to double\n";
                os << "  " << bd << " = bitcast i64 " << *src_bits << " to double\n";
                os << "  " << sd << " = fadd double " << ad << ", " << bd << "\n";
                newval = next_tmp("wa_sb");
                os << "  " << newval << " = bitcast double " << sd << " to i64\n";
            }
        } else if (operation == "min" || operation == "max") {
            const std::string cmp = next_tmp("wa_cmp");
            const std::string pred = ty.is_signed ? (operation == "min" ? "slt" : "sgt")
                                                  : (operation == "min" ? "ult" : "ugt");
            os << "  " << cmp << " = icmp " << pred << " i64 " << old << ", " << *src_bits << "\n";
            newval = next_tmp("wa_sel");
            os << "  " << newval << " = select i1 " << cmp << ", i64 " << old << ", i64 "
               << *src_bits << "\n";
        } else {
            newval = next_tmp("wa_rmw");
            os << "  " << newval << " = " << operation << " i64 " << old << ", " << *src_bits
               << "\n";
        }

        os << "  store i64 " << newval << ", i64 addrspace(" << addr_space << ")* " << payload
           << ", align 8\n";
        os << "  call void @air.atomic.fence(i32 3, i32 5, i32 2)\n";
        const std::string rel = next_tmp("wa_rel");
        os << "  " << rel << " = call i32 @air.atomic.global.xchg.i32(i32 addrspace(1)* "
           << lock << ", i32 0, i32 0, i32 2, i1 true)\n";
        os << "  br label %" << latch << "\n";

        os << latch << ":\n";
        os << "  " << done2 << " = phi i1 [ true, %" << crit << " ], [ " << done << ", %"
           << head << " ]\n";
        os << "  " << old2 << " = phi i64 [ " << old << ", %" << crit << " ], [ " << oldp
           << ", %" << head << " ]\n";
        os << "  br i1 " << done2 << ", label %" << done_lbl << ", label %" << head << "\n";

        os << done_lbl << ":\n";

        if (dst_reg == nullptr) return true;
        return emit_store_reg_bits(os, *dst_reg, ensure_reg_slot(*dst_reg).bits, old2, 64);
    }

    bool emit_atom(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        // atom[.scope].operation.type dst, [ptr], src[, cmp]
        if (instr.operands.size() < 3) {
            return fail(instr, "atom requires at least 3 operands");
        }
        const int addr_space = (instr.opcode.find(".shared") != std::string::npos) ? 3 : 1;

        std::string operation;
        for (const std::string& tok : split_opcode_tokens(instr.opcode)) {
            if (tok == "add" || tok == "and" || tok == "or" || tok == "xor" ||
                tok == "cas" || tok == "min" || tok == "max" || tok == "exch") {
                operation = tok;
                break;
            }
        }
        if (operation.empty()) {
            return fail(instr, "atom: unrecognized atomic operation in opcode");
        }

        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind == PtxTypeSpec::Kind::kInvalid) {
            return fail(instr, "atom: unrecognized element type in opcode");
        }
        const bool is_float = (ty.kind == PtxTypeSpec::Kind::kFloat);
        const int bits = ty.bits;

        const std::string elem_ty_str = is_float
            ? (bits == 32 ? "float" : (bits == 64 ? "double" : "half"))
            : llvm_int_type(bits);

        const ParsedMemOperand mem = parse_memory_operand(instr.operands[1]);
        if (!mem.ok) {
            return fail(instr, "atom: cannot parse memory operand");
        }
        std::optional<std::string> resolved_base;
        if (is_register_name(mem.base)) {
            resolved_base = emit_load_reg_bits(os, mem.base, 64);
        } else if (addr_space == 3) {
            resolved_base = resolve_threadgroup_symbol_address(os, mem.base);
        } else {
            resolved_base = resolve_global_symbol_address(os, mem.base);
        }
        if (!resolved_base.has_value()) {
            return fail(instr, "atom: cannot resolve memory base '" + mem.base + "'");
        }
        const std::string base_i64 = *resolved_base;
        const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);

        // A CUDA atomicAdd(double*) does not arrive as a float add: clang
        // lowers it to a CAS loop over the IEEE bits, so it reaches here as
        // atom.cas.b64. Both forms take the lock path.
        if (bits == 64) {
            if (operation == "cas" && instr.operands.size() < 4) {
                return fail(instr, "atom.cas requires 4 operands");
            }
            const std::string& src_operand =
                (operation == "cas") ? instr.operands[3] : instr.operands[2];
            const std::string* cmp_operand =
                (operation == "cas") ? &instr.operands[2] : nullptr;
            const std::string* dst_reg =
                is_register_name(instr.operands[0]) ? &instr.operands[0] : nullptr;
            return emit_wide_atomic(os, instr, operation, ty, addr_space, addr_i64,
                                    src_operand, cmp_operand, dst_reg);
        }

        const std::string ptr_i8 = next_tmp("atom_i2p");
        os << "  " << ptr_i8 << " = inttoptr i64 " << addr_i64
           << " to i8 addrspace(" << addr_space << ")*\n";
        const std::string ptr_t = next_tmp("atom_ptr");
        os << "  " << ptr_t << " = bitcast i8 addrspace(" << addr_space << ")* " << ptr_i8
           << " to " << elem_ty_str << " addrspace(" << addr_space << ")*\n";

        const std::string old_val = next_tmp("atom_old");
        if (operation == "cas") {
            if (instr.operands.size() < 4) return fail(instr, "atom.cas requires 4 operands");
            std::optional<std::string> cmp_v, new_v;
            if (is_float) {
                auto cmp_f = decode_float_operand(os, instr.operands[2], bits);
                auto new_f = decode_float_operand(os, instr.operands[3], bits);
                if (!cmp_f || !new_f) return fail(instr, "atom.cas float operands unsupported");
                cmp_v = cmp_f->ir;
                new_v = new_f->ir;
            } else {
                cmp_v = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
                new_v = emit_integer_from_any(os, instr.operands[3], bits, ty.is_signed);
                if (!cmp_v || !new_v) return fail(instr, "atom.cas int operands unsupported");
            }
            const std::string cx = next_tmp("atom_cx");
            os << "  " << cx << " = cmpxchg " << elem_ty_str << " addrspace(" << addr_space << ")* "
               << ptr_t << ", " << elem_ty_str << " " << *cmp_v << ", " << elem_ty_str << " " << *new_v
               << " monotonic monotonic\n";
            os << "  " << old_val << " = extractvalue { " << elem_ty_str << ", i1 } " << cx << ", 0\n";
        } else {
            std::string llvm_op;
            if (operation == "add")       llvm_op = is_float ? "fadd" : "add";
            else if (operation == "and")  llvm_op = "and";
            else if (operation == "or")   llvm_op = "or";
            else if (operation == "xor")  llvm_op = "xor";
            else if (operation == "min")  llvm_op = ty.is_signed ? "min" : "umin";
            else if (operation == "max")  llvm_op = ty.is_signed ? "max" : "umax";
            else if (operation == "exch") llvm_op = "xchg";
            else return fail(instr, "atom: operation '" + operation + "' not supported");

            std::optional<std::string> src_v;
            if (is_float) {
                auto fv = decode_float_operand(os, instr.operands[2], bits);
                if (!fv) return fail(instr, "atom float source operand unsupported");
                src_v = fv->ir;
            } else {
                src_v = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
                if (!src_v) return fail(instr, "atom int source operand unsupported");
            }
            os << "  " << old_val << " = atomicrmw " << llvm_op << " " << elem_ty_str
               << " addrspace(" << addr_space << ")* " << ptr_t << ", " << elem_ty_str << " " << *src_v
               << " monotonic\n";
        }

        if (!is_register_name(instr.operands[0])) return true;
        const std::string& dst = instr.operands[0];
        const int slot_bits = ensure_reg_slot(dst).bits;
        if (is_float) {
            const Value v{old_val, ty, bits};
            auto bitsv = encode_value_to_reg_bits(os, v, slot_bits);
            if (!bitsv) return fail(instr, "atom float result encode failed");
            return emit_store_reg_bits(os, dst, slot_bits, *bitsv, slot_bits);
        }
        return emit_store_reg_bits(os, dst, slot_bits, old_val, bits);
    }

    bool emit_vote(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        const bool is_ballot = instr.opcode.find("ballot") != std::string::npos;
        const bool is_any    = instr.opcode.find(".any.") != std::string::npos;
        const bool is_all    = instr.opcode.find(".all.") != std::string::npos;
        const bool is_uni    = instr.opcode.find(".uni.") != std::string::npos;
        if (instr.operands.size() < 2) return fail(instr, "vote requires at least 2 operands");
        const std::string& dst = instr.operands[0];
        if (!is_register_name(dst)) return fail(instr, "vote dst must be register");
        const std::string src_bits = emit_load_reg_bits(os, instr.operands[1], 1);
        declarations_.insert("declare i64 @air.simd_ballot.i64(i1)");
        const std::string ballot64 = next_tmp("vote_ballot64");
        os << "  " << ballot64 << " = call i64 @air.simd_ballot.i64(i1 " << src_bits << ")\n";
        const std::string ballot32 = next_tmp("vote_ballot32");
        os << "  " << ballot32 << " = trunc i64 " << ballot64 << " to i32\n";

        // The sync forms name the participating lanes explicitly.  AIR's ballot
        // covers the currently active SIMD lanes, so intersecting it with the
        // PTX member mask gives CUDA's defined vote set.  Legacy vote forms have
        // no member-mask operand and therefore use all currently active lanes.
        std::string member_mask = "-1";
        if (instr.opcode.find(".sync.") != std::string::npos) {
            if (instr.operands.size() < 3) return fail(instr, "vote.sync requires member mask");
            auto parsed_mask = emit_integer_from_any(os, instr.operands[2], 32, false);
            if (!parsed_mask) return fail(instr, "vote member mask unsupported");
            member_mask = *parsed_mask;
        }
        const std::string masked_ballot = next_tmp("vote_masked");
        os << "  " << masked_ballot << " = and i32 " << ballot32 << ", " << member_mask << "\n";
        if (is_ballot) {
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, masked_ballot, 32);
        }
        std::string pred_res;
        if (is_any) {
            pred_res = next_tmp("vote_any");
            os << "  " << pred_res << " = icmp ne i32 " << masked_ballot << ", 0\n";
        } else if (is_all || is_uni) {
            const std::string active64 = next_tmp("vote_active64");
            os << "  " << active64 << " = call i64 @air.simd_ballot.i64(i1 true)\n";
            const std::string active32 = next_tmp("vote_active32");
            os << "  " << active32 << " = trunc i64 " << active64 << " to i32\n";
            const std::string expected = next_tmp("vote_expected");
            os << "  " << expected << " = and i32 " << active32 << ", " << member_mask << "\n";
            pred_res = next_tmp("vote_all");
            if (is_uni) {
                const std::string none = next_tmp("vote_none");
                os << "  " << none << " = icmp eq i32 " << masked_ballot << ", 0\n";
                const std::string all = next_tmp("vote_uniform_all");
                os << "  " << all << " = icmp eq i32 " << masked_ballot << ", " << expected << "\n";
                os << "  " << pred_res << " = or i1 " << none << ", " << all << "\n";
            } else {
                os << "  " << pred_res << " = icmp eq i32 " << masked_ballot << ", " << expected << "\n";
            }
        } else {
            return fail(instr, "unrecognized vote form");
        }
        return emit_store_reg_bits(os, dst, 1, pred_res, 1);
    }

    bool emit_membar_or_fence(std::ostringstream& os,
                              const cumetal::ptx::EntryFunction::Instruction& instr) {
        // Apple's AIR backend cannot lower LLVM's `fence` instruction. Emitting one
        // crashes the Metal compiler service when the pipeline state is created
        // (XPC_ERROR_CONNECTION_INTERRUPTED, "after multiple retries"), long after the
        // metallib has been written and validated -- so the kernel simply never runs.
        // Metal spells a fence as a call, which is what its own compiler emits for
        // atomic_thread_fence:
        //
        //   air.atomic.fence(mem_flags, memory_order, scope)
        //     mem_flags    0 none, 1 device, 2 threadgroup, 3 device|threadgroup
        //     memory_order 0 relaxed, 5 seq_cst
        //     scope        2
        //
        // This is not only about explicit __threadfence(): clang plants a membar next
        // to atomicCAS, so every CAS-bearing kernel -- and anything built on one, such
        // as atomicInc/atomicDec -- was silently doing nothing.
        //
        // membar.cta / fence...cta is threadgroup scope; .gl and .sys are device, and
        // a bare membar is treated as the wider one.
        const bool cta_scope = instr.opcode.find(".cta") != std::string::npos;
        const int mem_flags = cta_scope ? 2 : 3;
        declarations_.insert("declare void @air.atomic.fence(i32, i32, i32)");
        os << "  call void @air.atomic.fence(i32 " << mem_flags << ", i32 5, i32 2)\n";
        return true;
    }

    bool emit_cp_async(std::ostringstream& os,
                       const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.opcode.find(".bulk") != std::string::npos ||
            instr.opcode.find(".tensor") != std::string::npos) {
            return fail(instr, "TMA/bulk cp.async is unsupported");
        }
        if (instr.opcode.find("commit_group") != std::string::npos ||
            instr.opcode.find("wait_group") != std::string::npos ||
            instr.opcode.find("wait_all") != std::string::npos) {
            declarations_.insert("declare void @air.wg.barrier(i32, i32)");
            os << "  call void @air.wg.barrier(i32 2, i32 1)\n";
            return true;
        }
        if (instr.operands.size() != 3) {
            return fail(instr,
                        "cp.async requires dst, src, and a fixed copy size; "
                        "source-size/zero-fill forms are unsupported");
        }
        const ParsedMemOperand dst = parse_memory_operand(instr.operands[0]);
        const ParsedMemOperand src = parse_memory_operand(instr.operands[1]);
        const auto bytes = parse_signed_immediate(instr.operands[2]);
        if (!dst.ok || !src.ok || !is_register_name(dst.base) ||
            !is_register_name(src.base) || !bytes ||
            (*bytes != 4 && *bytes != 8 && *bytes != 16)) {
            return fail(instr, "cp.async supports fixed 4, 8, or 16-byte register-addressed copies");
        }
        const std::string dst_base = emit_load_reg_bits(os, dst.base, 64);
        const std::string src_base = emit_load_reg_bits(os, src.base, 64);
        const std::string dst_addr = pointer_add_bytes(os, dst_base, dst.offset);
        const std::string src_addr = pointer_add_bytes(os, src_base, src.offset);
        for (std::int64_t offset = 0; offset < *bytes; offset += 4) {
            const std::string src_word_addr = pointer_add_bytes(os, src_addr, offset);
            const std::string dst_word_addr = pointer_add_bytes(os, dst_addr, offset);
            const std::string src_ptr = next_tmp("cp_src_ptr");
            const std::string dst_ptr = next_tmp("cp_dst_ptr");
            os << "  " << src_ptr << " = inttoptr i64 " << src_word_addr
               << " to i32 addrspace(1)*\n";
            os << "  " << dst_ptr << " = inttoptr i64 " << dst_word_addr
               << " to i32 addrspace(3)*\n";
            const std::string word = next_tmp("cp_word");
            os << "  " << word << " = load i32, i32 addrspace(1)* " << src_ptr
               << ", align 4\n";
            os << "  store i32 " << word << ", i32 addrspace(3)* " << dst_ptr
               << ", align 4\n";
        }
        return true;
    }

    bool emit_brev(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "brev requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const int bits = (instr.opcode.find(".b64") != std::string::npos) ? 64 : 32;
        const std::string ty = llvm_int_type(bits);
        declarations_.insert("declare " + ty + " @llvm.bitreverse." + ty + "(" + ty + ")");
        auto src = emit_integer_from_any(os, instr.operands[1], bits, false);
        if (!src) return fail(instr, "brev source unsupported");
        const std::string res = next_tmp("brev");
        os << "  " << res << " = call " << ty << " @llvm.bitreverse." << ty << "(" << ty << " " << *src << ")\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, res, bits);
    }

    bool emit_red(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2) return fail(instr, "red requires 2 operands");
        const int addr_space = (instr.opcode.find(".shared") != std::string::npos) ? 3 : 1;
        std::string operation;
        for (const std::string& tok : split_opcode_tokens(instr.opcode)) {
            if (tok == "add" || tok == "and" || tok == "or" || tok == "xor" ||
                tok == "min" || tok == "max" || tok == "exch") {
                operation = tok;
                break;
            }
        }
        if (operation.empty()) return fail(instr, "red: unrecognized operation");
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind == PtxTypeSpec::Kind::kInvalid) return fail(instr, "red: unrecognized type");
        const bool is_float = (ty.kind == PtxTypeSpec::Kind::kFloat);
        const int bits = ty.bits;
        const std::string elem_ty_str = is_float
            ? (bits == 32 ? "float" : (bits == 64 ? "double" : "half"))
            : llvm_int_type(bits);
        const ParsedMemOperand mem = parse_memory_operand(instr.operands[0]);
        if (!mem.ok || !is_register_name(mem.base)) return fail(instr, "red: cannot parse memory operand");
        const std::string base_i64 = emit_load_reg_bits(os, mem.base, 64);
        const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);

        // red is atom without a result. It had no 64-bit guard at all, so a
        // red.global.add.f64 emitted `atomicrmw fadd double` straight into AIR
        // and took the Metal compiler service down at pipeline creation.
        if (bits == 64) {
            return emit_wide_atomic(os, instr, operation, ty, addr_space, addr_i64,
                                    instr.operands[1], nullptr, nullptr);
        }

        const std::string ptr_i8 = next_tmp("red_i2p");
        os << "  " << ptr_i8 << " = inttoptr i64 " << addr_i64 << " to i8 addrspace(" << addr_space << ")*\n";
        const std::string ptr_t = next_tmp("red_ptr");
        os << "  " << ptr_t << " = bitcast i8 addrspace(" << addr_space << ")* " << ptr_i8
           << " to " << elem_ty_str << " addrspace(" << addr_space << ")*\n";
        std::string llvm_op;
        if      (operation == "add")  llvm_op = is_float ? "fadd" : "add";
        else if (operation == "and")  llvm_op = "and";
        else if (operation == "or")   llvm_op = "or";
        else if (operation == "xor")  llvm_op = "xor";
        else if (operation == "min")  llvm_op = ty.is_signed ? "min" : "umin";
        else if (operation == "max")  llvm_op = ty.is_signed ? "max" : "umax";
        else if (operation == "exch") llvm_op = "xchg";
        else return fail(instr, "red: operation not supported");
        std::optional<std::string> src_v;
        if (is_float) {
            auto fv = decode_float_operand(os, instr.operands[1], bits);
            if (!fv) return fail(instr, "red float source unsupported");
            src_v = fv->ir;
        } else {
            src_v = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
            if (!src_v) return fail(instr, "red int source unsupported");
        }
        const std::string unused = next_tmp("red_old");
        os << "  " << unused << " = atomicrmw " << llvm_op << " " << elem_ty_str
           << " addrspace(" << addr_space << ")* " << ptr_t << ", " << elem_ty_str << " " << *src_v
           << " monotonic\n";
        return true;
    }

    bool emit_activemask(std::ostringstream& os,
                         const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.empty() || !is_register_name(instr.operands[0])) {
            return fail(instr, "activemask requires dst register");
        }
        declarations_.insert("declare i64 @air.simd_ballot.i64(i1)");
        const std::string active64 = next_tmp("activemask64");
        os << "  " << active64 << " = call i64 @air.simd_ballot.i64(i1 true)\n";
        const std::string active32 = next_tmp("activemask32");
        os << "  " << active32 << " = trunc i64 " << active64 << " to i32\n";
        return emit_store_reg_bits(os, instr.operands[0], ensure_reg_slot(instr.operands[0]).bits, active32, 32);
    }

    bool emit_bfind(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "bfind requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".u64") != std::string::npos ||
                          instr.opcode.find(".s64") != std::string::npos;
        const bool shiftamt = instr.opcode.find("shiftamt") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string ty = llvm_int_type(bits);
        auto src = emit_integer_from_any(os, instr.operands[1], bits, false);
        if (!src) return fail(instr, "bfind source unsupported");
        declarations_.insert("declare " + ty + " @llvm.ctlz." + ty + "(" + ty + ", i1)");
        const std::string ctlz = next_tmp("bfind_ctlz");
        os << "  " << ctlz << " = call " << ty << " @llvm.ctlz." << ty << "(" << ty << " " << *src << ", i1 false)\n";
        const std::string msb = next_tmp("bfind_msb");
        os << "  " << msb << " = sub " << ty << " " << (bits - 1) << ", " << ctlz << "\n";
        std::string result = msb;
        if (shiftamt) {
            const std::string sa = next_tmp("bfind_sa");
            os << "  " << sa << " = sub " << ty << " " << (bits - 1) << ", " << msb << "\n";
            result = sa;
        }
        std::string result32 = result;
        if (is64) {
            const std::string tr = next_tmp("bfind_tr");
            os << "  " << tr << " = trunc i64 " << result << " to i32\n";
            result32 = tr;
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result32, 32);
    }

    bool emit_lop3(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 5 || !is_register_name(instr.operands[0])) {
            return fail(instr, "lop3 requires dst, a, b, c, lut");
        }
        const std::string& dst = instr.operands[0];
        auto a = emit_integer_from_any(os, instr.operands[1], 32, false);
        auto b = emit_integer_from_any(os, instr.operands[2], 32, false);
        auto c = emit_integer_from_any(os, instr.operands[3], 32, false);
        if (!a || !b || !c) return fail(instr, "lop3 sources unsupported");
        const auto lut_opt = parse_signed_immediate(instr.operands[4]);
        const uint8_t lut = lut_opt ? static_cast<uint8_t>(*lut_opt & 0xff) : 0xf0;
        std::string accumulated = next_tmp("lop3_acc");
        os << "  " << accumulated << " = or i32 0, 0\n";
        for (int i = 0; i < 8; ++i) {
            if (!(lut & (1 << i))) continue;
            const bool use_a = ((i >> 2) & 1) != 0;
            const bool use_b = ((i >> 1) & 1) != 0;
            const bool use_c = ((i >> 0) & 1) != 0;
            const std::string ta = next_tmp("lop3_ta");
            const std::string tb = next_tmp("lop3_tb");
            const std::string tc = next_tmp("lop3_tc");
            os << "  " << ta << " = " << (use_a ? "or i32 " + *a + ", 0" : "xor i32 " + *a + ", -1") << "\n";
            os << "  " << tb << " = " << (use_b ? "or i32 " + *b + ", 0" : "xor i32 " + *b + ", -1") << "\n";
            os << "  " << tc << " = " << (use_c ? "or i32 " + *c + ", 0" : "xor i32 " + *c + ", -1") << "\n";
            const std::string t_ab = next_tmp("lop3_ab");
            os << "  " << t_ab << " = and i32 " << ta << ", " << tb << "\n";
            const std::string t_abc = next_tmp("lop3_abc");
            os << "  " << t_abc << " = and i32 " << t_ab << ", " << tc << "\n";
            const std::string new_acc = next_tmp("lop3_acc");
            os << "  " << new_acc << " = or i32 " << accumulated << ", " << t_abc << "\n";
            accumulated = new_acc;
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, accumulated, 32);
    }

    bool emit_sad(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "sad requires dst, a, b, c");
        }
        const bool is_signed = instr.opcode.find(".s32") != std::string::npos;
        const std::string& dst = instr.operands[0];
        auto a = emit_integer_from_any(os, instr.operands[1], 32, is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], 32, is_signed);
        auto c = emit_integer_from_any(os, instr.operands[3], 32, is_signed);
        if (!a || !b || !c) return fail(instr, "sad sources unsupported");
        const std::string diff = next_tmp("sad_diff");
        os << "  " << diff << " = sub i32 " << *a << ", " << *b << "\n";
        const std::string neg = next_tmp("sad_neg");
        os << "  " << neg << " = sub i32 0, " << diff << "\n";
        const std::string positive = next_tmp("sad_pos");
        os << "  " << positive << " = icmp sgt i32 " << diff << ", -1\n";
        const std::string absdiff = next_tmp("sad_abs");
        os << "  " << absdiff << " = select i1 " << positive << ", i32 " << diff << ", i32 " << neg << "\n";
        const std::string result = next_tmp("sad_result");
        os << "  " << result << " = add i32 " << absdiff << ", " << *c << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, 32);
    }

    bool emit_match(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.empty() || !is_register_name(instr.operands[0])) {
            return fail(instr, "match requires dst register");
        }
        const std::string raw_dest = instr.operands[0];
        std::string dst_token = raw_dest;
        std::string pred_token;
        if (const std::size_t pipe = raw_dest.find('|'); pipe != std::string::npos) {
            dst_token = trim(raw_dest.substr(0, pipe));
            pred_token = trim(raw_dest.substr(pipe + 1));
        }
        if (!emit_store_reg_bits(os, dst_token, ensure_reg_slot(dst_token).bits, "-1", 32)) return false;
        if (!pred_token.empty() && is_register_name(pred_token)) {
            if (!emit_store_reg_bits(os, pred_token, 1, "true", 1)) return false;
        }
        return true;
    }

    bool emit_fns(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.empty() || !is_register_name(instr.operands[0])) {
            return fail(instr, "fns requires dst register");
        }
        return emit_store_reg_bits(os, instr.operands[0], ensure_reg_slot(instr.operands[0]).bits, "0", 32);
    }

    bool emit_testp(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "testp requires pred and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".f64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string raw = emit_load_reg_bits(os, instr.operands[1], bits);
        const std::string exp_shr = next_tmp("tp_eshr");
        const std::string exp    = next_tmp("tp_exp");
        const std::string man    = next_tmp("tp_man");
        const std::string exp_ff = next_tmp("tp_expff");
        const std::string has_man = next_tmp("tp_hman");
        const std::string result  = next_tmp("testp_r");
        if (is64) {
            os << "  " << exp_shr << " = lshr i64 " << raw << ", 52\n";
            os << "  " << exp    << " = and i64 " << exp_shr << ", 2047\n";
            os << "  " << man    << " = and i64 " << raw << ", 4503599627370495\n";
            os << "  " << exp_ff << " = icmp eq i64 " << exp << ", 2047\n";
            os << "  " << has_man << " = icmp ne i64 " << man << ", 0\n";
        } else {
            os << "  " << exp_shr << " = lshr i32 " << raw << ", 23\n";
            os << "  " << exp    << " = and i32 " << exp_shr << ", 255\n";
            os << "  " << man    << " = and i32 " << raw << ", 8388607\n";
            os << "  " << exp_ff << " = icmp eq i32 " << exp << ", 255\n";
            os << "  " << has_man << " = icmp ne i32 " << man << ", 0\n";
        }
        if (instr.opcode.find("nan") != std::string::npos) {
            os << "  " << result << " = and i1 " << exp_ff << ", " << has_man << "\n";
        } else if (instr.opcode.find("infinite") != std::string::npos) {
            const std::string no_man = next_tmp("tp_noman");
            os << "  " << no_man << " = xor i1 " << has_man << ", true\n";
            os << "  " << result << " = and i1 " << exp_ff << ", " << no_man << "\n";
        } else if (instr.opcode.find("finite") != std::string::npos) {
            os << "  " << result << " = xor i1 " << exp_ff << ", true\n";
        } else if (instr.opcode.find("number") != std::string::npos) {
            const std::string is_nan = next_tmp("tp_isnan");
            os << "  " << is_nan << " = and i1 " << exp_ff << ", " << has_man << "\n";
            os << "  " << result << " = xor i1 " << is_nan << ", true\n";
        } else if (instr.opcode.find("subnormal") != std::string::npos) {
            const std::string exp_0 = next_tmp("tp_exp0");
            const std::string is_i0_expr = is64 ? "icmp eq i64 " : "icmp eq i32 ";
            os << "  " << exp_0 << " = " << is_i0_expr << exp << ", 0\n";
            os << "  " << result << " = and i1 " << exp_0 << ", " << has_man << "\n";
        } else {
            // normal = exp not 0 and not FF
            const std::string exp_0 = next_tmp("tp_exp0");
            const std::string is_i0_expr = is64 ? "icmp eq i64 " : "icmp eq i32 ";
            os << "  " << exp_0 << " = " << is_i0_expr << exp << ", 0\n";
            const std::string not_spec = next_tmp("tp_nosp");
            os << "  " << not_spec << " = xor i1 " << exp_ff << ", true\n";
            const std::string not_zero = next_tmp("tp_noz");
            os << "  " << not_zero << " = xor i1 " << exp_0 << ", true\n";
            os << "  " << result << " = and i1 " << not_spec << ", " << not_zero << "\n";
        }
        return emit_store_reg_bits(os, dst, 1, result, 1);
    }

    bool emit_prmt(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "prmt requires dst, a, b, sel");
        }
        const std::string& dst = instr.operands[0];
        auto a = emit_integer_from_any(os, instr.operands[1], 32, false);
        auto b = emit_integer_from_any(os, instr.operands[2], 32, false);
        auto sel = emit_integer_from_any(os, instr.operands[3], 32, false);
        if (!a || !b || !sel) return fail(instr, "prmt sources unsupported");
        const std::string a64 = next_tmp("prmt_a64");
        os << "  " << a64 << " = zext i32 " << *a << " to i64\n";
        const std::string b64 = next_tmp("prmt_b64");
        os << "  " << b64 << " = zext i32 " << *b << " to i64\n";
        const std::string bshift = next_tmp("prmt_bsh");
        os << "  " << bshift << " = shl i64 " << b64 << ", 32\n";
        const std::string src64 = next_tmp("prmt_src");
        os << "  " << src64 << " = or i64 " << a64 << ", " << bshift << "\n";
        std::string result = next_tmp("prmt_res");
        os << "  " << result << " = or i32 0, 0\n";
        for (int lane = 0; lane < 4; ++lane) {
            const std::string sel_shr = next_tmp("prmt_ss");
            os << "  " << sel_shr << " = lshr i32 " << *sel << ", " << (lane * 4) << "\n";
            const std::string byte_idx = next_tmp("prmt_bi");
            os << "  " << byte_idx << " = and i32 " << sel_shr << ", 7\n";
            const std::string byte_idx64 = next_tmp("prmt_bi64");
            os << "  " << byte_idx64 << " = zext i32 " << byte_idx << " to i64\n";
            const std::string bit_off = next_tmp("prmt_boff");
            os << "  " << bit_off << " = mul i64 " << byte_idx64 << ", 8\n";
            const std::string src_shr = next_tmp("prmt_srcsh");
            os << "  " << src_shr << " = lshr i64 " << src64 << ", " << bit_off << "\n";
            const std::string byte_val64 = next_tmp("prmt_bv64");
            os << "  " << byte_val64 << " = and i64 " << src_shr << ", 255\n";
            const std::string byte_val = next_tmp("prmt_bv");
            os << "  " << byte_val << " = trunc i64 " << byte_val64 << " to i32\n";
            const std::string placed = next_tmp("prmt_pl");
            os << "  " << placed << " = shl i32 " << byte_val << ", " << (lane * 8) << "\n";
            const std::string new_result = next_tmp("prmt_or");
            os << "  " << new_result << " = or i32 " << result << ", " << placed << "\n";
            result = new_result;
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, 32);
    }

    bool emit_bfi(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 5 || !is_register_name(instr.operands[0])) {
            return fail(instr, "bfi requires dst, insert, base, offset, len");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".b64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string ty = llvm_int_type(bits);
        auto ins  = emit_integer_from_any(os, instr.operands[1], bits, false);
        auto base = emit_integer_from_any(os, instr.operands[2], bits, false);
        auto off  = emit_integer_from_any(os, instr.operands[3], 32, false);
        auto len  = emit_integer_from_any(os, instr.operands[4], 32, false);
        if (!ins || !base || !off || !len) return fail(instr, "bfi sources unsupported");
        const std::string len_ext = next_tmp("bfi_lext");
        const std::string off_ext = next_tmp("bfi_oext");
        if (is64) {
            os << "  " << len_ext << " = zext i32 " << *len << " to i64\n";
            os << "  " << off_ext << " = zext i32 " << *off << " to i64\n";
        } else {
            os << "  " << len_ext << " = or i32 " << *len << ", 0\n";
            os << "  " << off_ext << " = or i32 " << *off << ", 0\n";
        }
        const std::string one = next_tmp("bfi_one");
        os << "  " << one << " = or " << ty << " 0, 1\n";
        const std::string shifted1 = next_tmp("bfi_s1");
        os << "  " << shifted1 << " = shl " << ty << " " << one << ", " << len_ext << "\n";
        const std::string mask_raw = next_tmp("bfi_mr");
        os << "  " << mask_raw << " = sub " << ty << " " << shifted1 << ", 1\n";
        const std::string mask = next_tmp("bfi_mask");
        os << "  " << mask << " = shl " << ty << " " << mask_raw << ", " << off_ext << "\n";
        const std::string not_mask = next_tmp("bfi_nmask");
        os << "  " << not_mask << " = xor " << ty << " " << mask << ", -1\n";
        const std::string base_clear = next_tmp("bfi_bc");
        os << "  " << base_clear << " = and " << ty << " " << *base << ", " << not_mask << "\n";
        const std::string ins_masked = next_tmp("bfi_im");
        os << "  " << ins_masked << " = and " << ty << " " << *ins << ", " << mask_raw << "\n";
        const std::string ins_placed = next_tmp("bfi_ip");
        os << "  " << ins_placed << " = shl " << ty << " " << ins_masked << ", " << off_ext << "\n";
        const std::string result = next_tmp("bfi_res");
        os << "  " << result << " = or " << ty << " " << base_clear << ", " << ins_placed << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, bits);
    }

    bool emit_isspacep(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.empty() || !is_register_name(instr.operands[0])) {
            return fail(instr, "isspacep requires pred register");
        }
        return emit_store_reg_bits(os, instr.operands[0], 1, "true", 1);
    }

    bool emit_float_math_unary(std::ostringstream& os,
                               const cumetal::ptx::EntryFunction::Instruction& instr,
                               const std::string& air_intrinsic) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "unary float math requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".f64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string fty = is64 ? "double" : "float";
        const std::string air_name = air_intrinsic + (is64 ? ".f64" : ".f32");
        const std::string air_decl = "declare " + fty + " @" + air_name + "(" + fty + ")";
        declarations_.insert(air_decl);
        auto fv = decode_float_operand(os, instr.operands[1], bits);
        if (!fv) return fail(instr, "float unary source unsupported");
        const std::string res = next_tmp("fmath");
        os << "  " << res << " = call " << fty << " @" << air_name << "(" << fty << " " << fv->ir << ")\n";
        const Value rv{res, PtxTypeSpec{PtxTypeSpec::Kind::kFloat, bits, false}, bits};
        auto bitsv = encode_value_to_reg_bits(os, rv, ensure_reg_slot(dst).bits);
        if (!bitsv) return fail(instr, "float unary encode failed");
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_sqrt(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.opcode.find(".f64") != std::string::npos && uses_vf64_support()) {
            if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
                return fail(instr, "VF64 sqrt requires dst and src");
            }
            auto raw = decode_fp64_raw_bits(os, instr.operands[1]);
            if (!raw) return fail(instr, "VF64 sqrt source unsupported");
            const int rounding = vf64_rounding_mode(instr.opcode);
            std::string function;
            std::string arguments;
            if (fp64_mode_ == cumetal::ptx::Fp64Mode::kWide48) {
                if (rounding != 0) {
                    return fail(instr, "wide48 supports round-to-nearest-even sqrt only");
                }
                function = "vf64_wide_sqrt";
                declarations_.insert("declare i64 @vf64_wide_sqrt(i64)");
                arguments = "i64 " + *raw;
            } else {
                function = "vf64_sqrt_round";
                declarations_.insert("declare i64 @vf64_sqrt_round(i64, i32)");
                arguments = "i64 " + *raw + ", i32 " + std::to_string(rounding);
            }
            const std::string result = next_tmp("vf64_sqrt");
            os << "  " << result << " = call i64 @" << function << "("
               << arguments << ")\n";
            return emit_store_reg_bits(
                os, instr.operands[0], ensure_reg_slot(instr.operands[0]).bits,
                result, 64
            );
        }
        return emit_float_math_unary(os, instr, "air.fast_sqrt");
    }

    bool emit_rsqrt(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "rsqrt requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".f64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string fty = is64 ? "double" : "float";
        const std::string sqrt_name = "air.fast_sqrt." + fty;
        declarations_.insert("declare " + fty + " @" + sqrt_name + "(" + fty + ")");
        auto fv = decode_float_operand(os, instr.operands[1], bits);
        if (!fv) return fail(instr, "rsqrt source unsupported");
        const std::string sq = next_tmp("rsqrt_sq");
        os << "  " << sq << " = call " << fty << " @" << sqrt_name << "(" << fty << " " << fv->ir << ")\n";
        const std::string one = next_tmp("rsqrt_one");
        os << "  " << one << " = " << (is64 ? "or i64 0, " : "or i32 0, ")
           << (is64 ? "4607182418800017408" : "1065353216") << "\n";  // 1.0 bits
        const std::string one_f = next_tmp("rsqrt_onef");
        os << "  " << one_f << " = bitcast " << (is64 ? "i64" : "i32") << " " << one << " to " << fty << "\n";
        const std::string res = next_tmp("rsqrt_res");
        os << "  " << res << " = fdiv " << fty << " " << one_f << ", " << sq << "\n";
        const Value rv{res, PtxTypeSpec{PtxTypeSpec::Kind::kFloat, bits, false}, bits};
        auto bitsv = encode_value_to_reg_bits(os, rv, ensure_reg_slot(dst).bits);
        if (!bitsv) return fail(instr, "rsqrt encode failed");
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_abs(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "abs requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        if (ty.kind == PtxTypeSpec::Kind::kFloat) {
            if (bits == 64 && uses_vf64_support()) {
                auto raw = decode_fp64_raw_bits(os, instr.operands[1]);
                if (!raw) return fail(instr, "VF64 abs source unsupported");
                const std::string result = next_tmp("vf64_abs");
                os << "  " << result << " = and i64 " << *raw
                   << ", 9223372036854775807\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, 64);
            }
            if (bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto pair = decode_fp64_pair(os, instr.operands[1]);
                if (!pair) return fail(instr, "fp64 abs emulation source unsupported");
                const std::string hi_bits = next_tmp("fp64_abs_hi_bits");
                const std::string negative = next_tmp("fp64_abs_negative");
                const std::string abs_hi_bits = next_tmp("fp64_abs_hi_clear_sign");
                const std::string abs_hi = next_tmp("fp64_abs_hi");
                const std::string neg_lo = next_tmp("fp64_abs_neg_lo");
                const std::string abs_lo = next_tmp("fp64_abs_lo");
                os << "  " << hi_bits << " = bitcast float " << pair->hi << " to i32\n"
                   << "  " << negative << " = icmp slt i32 " << hi_bits << ", 0\n"
                   << "  " << abs_hi_bits << " = and i32 " << hi_bits << ", 2147483647\n"
                   << "  " << abs_hi << " = bitcast i32 " << abs_hi_bits << " to float\n"
                   << "  " << neg_lo << " = fneg float " << pair->lo << "\n"
                   << "  " << abs_lo << " = select i1 " << negative << ", float "
                   << neg_lo << ", float " << pair->lo << "\n";
                return store_fp64_pair(os, dst, Fp64Pair{abs_hi, abs_lo});
            }
            const std::string fty = (bits == 64) ? "double" : "float";
            const std::string intr = "llvm.fabs." + fty;
            declarations_.insert("declare " + fty + " @" + intr + "(" + fty + ")");
            auto fv = decode_float_operand(os, instr.operands[1], bits);
            if (!fv) return fail(instr, "abs float source unsupported");
            const std::string res = next_tmp("fabs");
            os << "  " << res << " = call " << fty << " @" << intr << "(" << fty << " " << fv->ir << ")\n";
            const Value rv{res, PtxTypeSpec{PtxTypeSpec::Kind::kFloat, bits, false}, bits};
            auto bitsv = encode_value_to_reg_bits(os, rv, ensure_reg_slot(dst).bits);
            if (!bitsv) return fail(instr, "abs float encode failed");
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
        }
        const std::string ity = llvm_int_type(bits);
        auto src = emit_integer_from_any(os, instr.operands[1], bits, true);
        if (!src) return fail(instr, "abs int source unsupported");
        const std::string neg = next_tmp("abs_neg");
        os << "  " << neg << " = sub " << ity << " 0, " << *src << "\n";
        const std::string pos = next_tmp("abs_pos");
        os << "  " << pos << " = icmp sgt " << ity << " " << *src << ", -1\n";
        const std::string res = next_tmp("abs_res");
        os << "  " << res << " = select i1 " << pos << ", " << ity << " " << *src << ", " << ity << " " << neg << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, res, bits);
    }

    bool emit_clz(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "clz requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".b64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string ty = llvm_int_type(bits);
        auto src = emit_integer_from_any(os, instr.operands[1], bits, false);
        if (!src) return fail(instr, "clz source unsupported");
        declarations_.insert("declare " + ty + " @llvm.ctlz." + ty + "(" + ty + ", i1)");
        const std::string res = next_tmp("clz");
        os << "  " << res << " = call " << ty << " @llvm.ctlz." + ty + "(" << ty << " " << *src << ", i1 false)\n";
        std::string res32 = res;
        if (is64) {
            res32 = next_tmp("clz32");
            os << "  " << res32 << " = trunc i64 " << res << " to i32\n";
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, res32, 32);
    }

    bool emit_popc(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "popc requires dst and src");
        }
        const std::string& dst = instr.operands[0];
        const bool is64 = instr.opcode.find(".b64") != std::string::npos;
        const int bits = is64 ? 64 : 32;
        const std::string ty = llvm_int_type(bits);
        auto src = emit_integer_from_any(os, instr.operands[1], bits, false);
        if (!src) return fail(instr, "popc source unsupported");
        declarations_.insert("declare " + ty + " @llvm.ctpop." + ty + "(" + ty + ")");
        const std::string res = next_tmp("popc");
        os << "  " << res << " = call " << ty << " @llvm.ctpop." << ty << "(" << ty << " " << *src << ")\n";
        std::string res32 = res;
        if (is64) {
            res32 = next_tmp("popc32");
            os << "  " << res32 << " = trunc i64 " << res << " to i32\n";
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, res32, 32);
    }

    bool emit_set(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        // set.CmpOp.dtype.stype dst, a, b → dst = (a CmpOp b) ? 0xffffffff : 0
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "set requires dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        // Parse source type (last type token in opcode)
        const bool src_float = instr.opcode.find(".f32") != std::string::npos ||
                               instr.opcode.find(".f64") != std::string::npos;
        const bool src64 = instr.opcode.find(".f64") != std::string::npos ||
                           instr.opcode.find(".s64") != std::string::npos ||
                           instr.opcode.find(".u64") != std::string::npos;
        const int src_bits = src64 ? 64 : 32;
        std::string cmp_result;
        if (src_float) {
            const std::string fty = src64 ? "double" : "float";
            auto fa = decode_float_operand(os, instr.operands[1], src_bits);
            auto fb = decode_float_operand(os, instr.operands[2], src_bits);
            if (!fa || !fb) return fail(instr, "set float sources unsupported");
            std::string cmp_op = "oeq";
            if      (instr.opcode.find(".lt.") != std::string::npos) cmp_op = "olt";
            else if (instr.opcode.find(".le.") != std::string::npos) cmp_op = "ole";
            else if (instr.opcode.find(".gt.") != std::string::npos) cmp_op = "ogt";
            else if (instr.opcode.find(".ge.") != std::string::npos) cmp_op = "oge";
            else if (instr.opcode.find(".ne.") != std::string::npos) cmp_op = "one";
            cmp_result = next_tmp("set_cmp");
            os << "  " << cmp_result << " = fcmp " << cmp_op << " " << fty << " " << fa->ir << ", " << fb->ir << "\n";
        } else {
            const bool is_signed = instr.opcode.find(".s32") != std::string::npos ||
                                   instr.opcode.find(".s64") != std::string::npos;
            auto ia = emit_integer_from_any(os, instr.operands[1], src_bits, is_signed);
            auto ib = emit_integer_from_any(os, instr.operands[2], src_bits, is_signed);
            if (!ia || !ib) return fail(instr, "set int sources unsupported");
            std::string cmp_op = "eq";
            if      (instr.opcode.find(".lt.") != std::string::npos) cmp_op = is_signed ? "slt" : "ult";
            else if (instr.opcode.find(".le.") != std::string::npos) cmp_op = is_signed ? "sle" : "ule";
            else if (instr.opcode.find(".gt.") != std::string::npos) cmp_op = is_signed ? "sgt" : "ugt";
            else if (instr.opcode.find(".ge.") != std::string::npos) cmp_op = is_signed ? "sge" : "uge";
            else if (instr.opcode.find(".ne.") != std::string::npos) cmp_op = "ne";
            cmp_result = next_tmp("set_cmp");
            const std::string sty = llvm_int_type(src_bits);
            os << "  " << cmp_result << " = icmp " << cmp_op << " " << sty << " " << *ia << ", " << *ib << "\n";
        }
        // dst = cmp ? -1 : 0 (PTX set result is 0xffffffff for true)
        const std::string result = next_tmp("set_res");
        os << "  " << result << " = select i1 " << cmp_result << ", i32 -1, i32 0\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, result, 32);
    }

    bool emit_selp(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "selp requires dst,a,b,pred");
        }
        const std::string& dst = instr.operands[0];
        const std::string& pred = instr.operands[3];
        if (!is_register_name(pred)) {
            return fail(instr, "selp predicate must be register");
        }
        const std::string p = emit_load_reg_bits(os, pred, 1);
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind == PtxTypeSpec::Kind::kFloat) {
            if (ty.bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                auto a = emit_integer_from_any(os, instr.operands[1], 64, false);
                auto b = emit_integer_from_any(os, instr.operands[2], 64, false);
                if (!a || !b) return fail(instr, "fp64 selp emulation sources unsupported");
                const std::string sel = next_tmp("fp64_sel_bits");
                os << "  " << sel << " = select i1 " << p << ", i64 " << *a
                   << ", i64 " << *b << "\n";
                return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, sel, 64);
            }
            auto a = decode_float_operand(os, instr.operands[1], ty.bits);
            auto b = decode_float_operand(os, instr.operands[2], ty.bits);
            if (!a || !b) return fail(instr, "selp float sources unsupported");
            const std::string sel = next_tmp("selpf");
            const std::string fty = llvm_float_type(ty.bits);
            os << "  " << sel << " = select i1 " << p << ", " << fty << " " << a->ir << ", " << fty
               << " " << b->ir << "\n";
            Value v{.ir = sel, .type = ty, .bits = ty.bits};
            auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
            if (!bitsv) return fail(instr, "selp float encode failed");
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
        }

        const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
        auto a = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
        auto b = emit_integer_from_any(os, instr.operands[2], bits, ty.is_signed);
        if (!a || !b) return fail(instr, "selp integer sources unsupported");
        const std::string sel = next_tmp("selpi");
        os << "  " << sel << " = select i1 " << p << ", " << llvm_int_type(bits) << " " << *a << ", "
           << llvm_int_type(bits) << " " << *b << "\n";
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, sel, bits);
    }

    bool emit_shfl(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (!starts_with(instr.opcode, "shfl.sync") || instr.operands.size() < 4) {
            return fail(instr, "unsupported shfl form");
        }
        const bool shfl_is_b32 = instr.opcode.find(".b32") != std::string::npos;
        const bool shfl_is_f32 = instr.opcode.find(".f32") != std::string::npos;
        if (!shfl_is_b32 && !shfl_is_f32) {
            return fail(instr, "only shfl.*.b32 and shfl.*.f32 supported");
        }

        const std::string raw_dest = instr.operands[0];
        std::string dst_token = raw_dest;
        std::string pred_token;
        if (const std::size_t pipe = raw_dest.find('|'); pipe != std::string::npos) {
            dst_token = trim(raw_dest.substr(0, pipe));
            pred_token = trim(raw_dest.substr(pipe + 1));
        }
        if (!is_register_name(dst_token)) {
            return fail(instr, "shfl destination must be register");
        }
        if (!pred_token.empty() && !is_register_name(pred_token)) {
            return fail(instr, "shfl predicate destination must be register");
        }

        auto src = emit_integer_from_any(os, instr.operands[1], 32, false);
        auto sel = emit_integer_from_any(os, instr.operands[2], 32, false);
        auto clamp = emit_integer_from_any(os, instr.operands[3], 32, false);
        if (!src || !sel || !clamp) {
            return fail(instr, "shfl operands unsupported");
        }
        if (instr.operands.size() < 5) {
            return fail(instr, "shfl.sync requires member mask");
        }
        auto member_mask = emit_integer_from_any(os, instr.operands[4], 32, false);
        if (!member_mask) {
            return fail(instr, "shfl member mask unsupported");
        }

        const auto lane_it = builtin_scalar_arg_name_.find("air.thread_index_in_simdgroup");
        if (lane_it == builtin_scalar_arg_name_.end()) {
            return fail(instr, "shfl requires thread_index_in_simdgroup builtin");
        }
        const std::string lane = "%" + lane_it->second;
        const std::string lane_bit = next_tmp("shfl_lane_bit");
        os << "  " << lane_bit << " = shl i32 1, " << lane << "\n";
        const std::string lane_member_bit = next_tmp("shfl_lane_member_bit");
        os << "  " << lane_member_bit << " = and i32 " << *member_mask << ", " << lane_bit << "\n";
        const std::string lane_participates = next_tmp("shfl_lane_participates");
        os << "  " << lane_participates << " = icmp ne i32 " << lane_member_bit << ", 0\n";

        // PTX clamp encoding: width = 32 - ((clamp >> 8) & 0x1f); 0 means 32.
        const std::string clamp_shr = next_tmp("shfl_cshr");
        os << "  " << clamp_shr << " = lshr i32 " << *clamp << ", 8\n";
        const std::string clamp_wraw = next_tmp("shfl_wraw");
        os << "  " << clamp_wraw << " = and i32 " << clamp_shr << ", 31\n";
        const std::string width0 = next_tmp("shfl_w0");
        os << "  " << width0 << " = sub i32 32, " << clamp_wraw << "\n";
        const std::string width0_is_zero = next_tmp("shfl_w0z");
        os << "  " << width0_is_zero << " = icmp eq i32 " << width0 << ", 0\n";
        const std::string width = next_tmp("shfl_w");
        os << "  " << width << " = select i1 " << width0_is_zero << ", i32 32, i32 " << width0 << "\n";
        const std::string div = next_tmp("shfl_div");
        os << "  " << div << " = udiv i32 " << lane << ", " << width << "\n";
        const std::string base = next_tmp("shfl_base");
        os << "  " << base << " = mul i32 " << div << ", " << width << "\n";
        const std::string local = next_tmp("shfl_local");
        os << "  " << local << " = sub i32 " << lane << ", " << base << "\n";

        const bool is_down = instr.opcode.find(".down.") != std::string::npos;
        const bool is_up = instr.opcode.find(".up.") != std::string::npos;
        const bool is_bfly = instr.opcode.find(".bfly.") != std::string::npos;
        std::string target;
        std::string valid;
        if (is_down) {
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = add i32 " << lane << ", " << *sel << "\n";
            const std::string limit = next_tmp("shfl_limit");
            os << "  " << limit << " = add i32 " << base << ", " << width << "\n";
            const std::string ok = next_tmp("shfl_ok");
            os << "  " << ok << " = icmp ult i32 " << t << ", " << limit << "\n";
            target = t;
            valid = ok;
            declarations_.insert("declare i32 @air.simd_shuffle_down.u.i32(i32, i16)");
        } else if (is_up) {
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = sub i32 " << lane << ", " << *sel << "\n";
            const std::string ok = next_tmp("shfl_ok");
            os << "  " << ok << " = icmp uge i32 " << local << ", " << *sel << "\n";
            target = t;
            valid = ok;
            declarations_.insert("declare i32 @air.simd_shuffle_up.u.i32(i32, i16)");
        } else if (is_bfly) {
            const std::string tlocal = next_tmp("shfl_tlocal");
            os << "  " << tlocal << " = xor i32 " << local << ", " << *sel << "\n";
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = add i32 " << base << ", " << tlocal << "\n";
            const std::string ok = next_tmp("shfl_ok");
            os << "  " << ok << " = icmp ult i32 " << tlocal << ", " << width << "\n";
            target = t;
            valid = ok;
            declarations_.insert("declare i32 @air.simd_shuffle_xor.u.i32(i32, i16)");
        } else {
            // idx (default)
            const std::string width_minus_1 = next_tmp("shfl_wm1");
            os << "  " << width_minus_1 << " = sub i32 " << width << ", 1\n";
            const std::string src_local = next_tmp("shfl_src_local");
            os << "  " << src_local << " = and i32 " << *sel << ", " << width_minus_1 << "\n";
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = add i32 " << base << ", " << src_local << "\n";
            target = t;
            valid = "true";
            declarations_.insert("declare i32 @air.simd_shuffle.u.i32(i32, i16)");
        }

        // Apple's AIR intrinsics take the original relative delta / XOR mask.
        // Only the index form takes an absolute lane. Passing the PTX target
        // lane here double-applies the current lane in AIR (for example,
        // shfl.down delta=1 returned lane 3 to lane 0 in a 16-lane tile).
        const std::string air_selector = (is_down || is_up || is_bfly) ? *sel : target;
        const std::string selector16 = next_tmp("shfl_s16");
        os << "  " << selector16 << " = trunc i32 " << air_selector << " to i16\n";
        const std::string call = next_tmp("shfl_call");
        if (is_down) {
            os << "  " << call << " = call i32 @air.simd_shuffle_down.u.i32(i32 " << *src
               << ", i16 " << selector16 << ")\n";
        } else if (is_up) {
            os << "  " << call << " = call i32 @air.simd_shuffle_up.u.i32(i32 " << *src
               << ", i16 " << selector16 << ")\n";
        } else if (is_bfly) {
            os << "  " << call << " = call i32 @air.simd_shuffle_xor.u.i32(i32 " << *src
               << ", i16 " << selector16 << ")\n";
        } else {
            os << "  " << call << " = call i32 @air.simd_shuffle.u.i32(i32 " << *src
               << ", i16 " << selector16 << ")\n";
        }

        const std::string defined = next_tmp("shfl_defined");
        os << "  " << defined << " = and i1 " << valid << ", " << lane_participates << "\n";
        const std::string result = next_tmp("shfl_res");
        os << "  " << result << " = select i1 " << defined << ", i32 " << call << ", i32 " << *src << "\n";
        if (!emit_store_reg_bits(os, dst_token, ensure_reg_slot(dst_token).bits, result, 32)) {
            return false;
        }
        if (!pred_token.empty()) {
            if (!emit_store_reg_bits(os, pred_token, 1, defined, 1)) {
                return false;
            }
        }
        return true;
    }

    bool emit_barrier(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (starts_with(instr.opcode, "bar.warp.sync")) {
            if (instr.operands.empty()) {
                return fail(instr, "bar.warp.sync requires member mask");
            }
            // Validate/load a dynamic member mask even though AIR derives
            // participation from the currently active SIMD lanes.  Extra
            // active lanes at the same instruction only strengthen ordering;
            // divergent execution naturally restricts the barrier to the
            // active member lanes.
            auto member_mask = emit_integer_from_any(os, instr.operands[0], 32, false);
            if (!member_mask) {
                return fail(instr, "bar.warp.sync member mask unsupported");
            }
            declarations_.insert("declare void @air.simdgroup.barrier(i32, i32)");
            // From Metal 4 AIR for
            // simdgroup_barrier(mem_flags::mem_threadgroup): flags=2,
            // thread_scope_simdgroup=4.
            os << "  call void @air.simdgroup.barrier(i32 2, i32 4)\n";
            return true;
        }
        if (starts_with(instr.opcode, "bar.sync")) {
            declarations_.insert("declare void @air.wg.barrier(i32, i32)");
            // From xcrun AIR LLVM for threadgroup_barrier(mem_flags::mem_threadgroup):
            //   @air.wg.barrier(i32 2, i32 1)
            os << "  call void @air.wg.barrier(i32 2, i32 1)\n";
            return true;
        }
        if (starts_with(instr.opcode, "bar.arrive")) {
            declarations_.insert("declare void @air.wg.barrier(i32, i32)");
            os << "  call void @air.wg.barrier(i32 2, i32 1)\n";
            return true;
        }
        if (starts_with(instr.opcode, "bar.red")) {
            declarations_.insert("declare void @air.wg.barrier(i32, i32)");
            os << "  call void @air.wg.barrier(i32 2, i32 1)\n";
            if (!instr.operands.empty() && is_register_name(instr.operands[0])) {
                const std::string& dst = instr.operands[0];
                const int slot_bits = ensure_reg_slot(dst).bits;
                emit_store_reg_bits(os, dst, slot_bits, "0", std::min(slot_bits, 32));
            }
            return true;
        }
        return fail(instr, "unsupported barrier opcode");
    }

    bool emit_branch(std::ostringstream& os,
                     const cumetal::ptx::EntryFunction::Instruction& instr,
                     int current_exec_pos,
                     bool* out_terminated) {
        if (instr.operands.empty()) return fail(instr, "bra missing target");
        const std::string target_label = instr.operands[0];
        auto it = label_to_exec_pos_.find(target_label);
        if (it == label_to_exec_pos_.end()) {
            return fail(instr, "unknown branch target '" + target_label + "'");
        }
        const int target_pos = it->second;
        const int fallthrough_pos = next_exec_pos_by_exec_pos_.count(current_exec_pos)
                                        ? next_exec_pos_by_exec_pos_.at(current_exec_pos)
                                        : -1;
        if (!instr.predicate.empty()) {
            std::string pred_tok = trim(instr.predicate);
            bool invert = false;
            if (starts_with(pred_tok, "@!")) {
                invert = true;
                pred_tok = pred_tok.substr(2);
            } else if (starts_with(pred_tok, "@")) {
                pred_tok = pred_tok.substr(1);
            }
            if (!is_register_name(pred_tok)) {
                return fail(instr, "predicated bra expects predicate register");
            }
            const std::string p = emit_load_reg_bits(os, pred_tok, 1);
            std::string cond = p;
            if (invert) {
                const std::string n = next_tmp("notp");
                os << "  " << n << " = xor i1 " << p << ", true\n";
                cond = n;
            }
            os << "  br i1 " << cond << ", label %" << block_name_for_exec_pos(target_pos)
               << ", label %" << block_name_for_exec_pos(fallthrough_pos) << "\n";
        } else {
            os << "  br label %" << block_name_for_exec_pos(target_pos) << "\n";
        }
        *out_terminated = true;
        return true;
    }

    bool emit_indexed_branch(std::ostringstream& os,
                             const cumetal::ptx::EntryFunction::Instruction& instr,
                             int current_exec_pos,
                             bool* out_terminated) {
        if (!instr.predicate.empty()) return fail(instr, "predicated brx.idx is unsupported");
        if (instr.operands.size() != 2 || !is_register_name(instr.operands[0])) {
            return fail(instr, "brx.idx expects an index register and target table");
        }
        const auto table = branch_tables_.find(instr.operands[1]);
        if (table == branch_tables_.end()) {
            return fail(instr, "unknown branch-target table '" + instr.operands[1] + "'");
        }
        const std::string index = emit_load_reg_bits(os, instr.operands[0], 32);
        const int fallthrough_pos = next_exec_pos_by_exec_pos_.count(current_exec_pos)
                                        ? next_exec_pos_by_exec_pos_.at(current_exec_pos)
                                        : -1;
        os << "  switch i32 " << index << ", label %"
           << block_name_for_exec_pos(fallthrough_pos) << " [\n";
        for (std::size_t i = 0; i < table->second.size(); ++i) {
            const auto target = label_to_exec_pos_.find(table->second[i]);
            if (target == label_to_exec_pos_.end()) {
                return fail(instr, "unknown indexed branch target '" + table->second[i] + "'");
            }
            os << "    i32 " << i << ", label %"
               << block_name_for_exec_pos(target->second) << "\n";
        }
        os << "  ]\n";
        *out_terminated = true;
        return true;
    }

    bool fail(const cumetal::ptx::EntryFunction::Instruction& instr, const std::string& msg) {
        error_ = "generic llvm lowering: line " + std::to_string(instr.line) + " opcode '" + instr.opcode + "': " + msg;
        return false;
    }

    void emit_function_return(std::ostringstream& os) {
        if (kernel_mode_ || return_bits_ <= 0 || return_param_name_.empty()) {
            os << "  ret void\n";
            return;
        }
        const auto slot = get_param_slot(return_param_name_, return_bits_, false);
        if (!slot) {
            os << "  ret " << llvm_int_type(return_bits_) << " 0\n";
            return;
        }
        const std::string value = next_tmp("function_return");
        os << "  " << value << " = load " << llvm_int_type(return_bits_) << ", "
           << llvm_int_type(return_bits_) << "* " << *slot << ", align "
           << std::max(1, return_bits_ / 8) << "\n";
        os << "  ret " << llvm_int_type(return_bits_) << " " << value << "\n";
    }

    bool emit_instruction_block(const cumetal::ptx::EntryFunction::Instruction& instr,
                                int exec_pos,
                                std::ostringstream& os,
                                bool* out_terminated) {
        *out_terminated = false;

        // Refuse rather than silently reading zero for a special register we do
        // not lower. Probe with 32 bits into a scratch stream: the handled ones
        // are pure reads, so discarding the emitted IR here is safe.
        for (std::size_t i = 1; i < instr.operands.size(); ++i) {
            const std::string& operand = instr.operands[i];
            if (!is_ptx_special_register(operand)) {
                continue;
            }
            std::ostringstream probe;
            if (!emit_special_register_value(probe, operand, 32)) {
                return fail(instr, "unsupported PTX special register '" + operand + "'");
            }
        }

        if (!instr.predicate.empty() && opcode_root(instr.opcode) != "bra") {
            return fail(instr, "predicated non-branch instructions not yet supported");
        }

        const std::string root = opcode_root(instr.opcode);
        if (root == "ret" || root == "exit") {
            emit_function_return(os);
            *out_terminated = true;
            return true;
        }
        if (root == "bra") {
            return emit_branch(os, instr, exec_pos, out_terminated);
        }
        if (root == "brx") {
            return emit_indexed_branch(os, instr, exec_pos, out_terminated);
        }
        if (root == "mov") {
            return emit_mov_instruction(os, instr);
        }
        if (root == "cvta") {
            return emit_cvta_instruction(os, instr);
        }
        if (root == "ld" || root == "st") {
            return emit_ld_st(os, instr);
        }
        if (starts_with(instr.opcode, "mul.wide")) {
            return emit_mul_wide(os, instr);
        }
        if (root == "cvt") {
            return emit_cvt(os, instr);
        }
        if (root == "setp") {
            return emit_setp(os, instr);
        }
        if (root == "selp") return emit_selp(os, instr);
        if (root == "shfl") return emit_shfl(os, instr);
        if (root == "bar") return emit_barrier(os, instr);
        if (root == "add") return opcode_uses_float_math(instr.opcode) ? emit_binary_float_op(os, instr, "fadd") : emit_binary_int_op(os, instr, "add");
        if (root == "sub") return opcode_uses_float_math(instr.opcode) ? emit_binary_float_op(os, instr, "fsub") : emit_binary_int_op(os, instr, "sub");
        if (root == "mul") return emit_mul(os, instr);
        if (root == "div") return opcode_uses_float_math(instr.opcode) ? emit_binary_float_op(os, instr, "fdiv") : emit_binary_int_op(os, instr, "div");
        if (root == "min") return emit_minmax(os, instr, true);
        if (root == "max") return emit_minmax(os, instr, false);
        if (root == "rem") return opcode_uses_float_math(instr.opcode) ? fail(instr, "frem not implemented") : emit_binary_int_op(os, instr, "rem");
        if (root == "and") return emit_binary_int_op(os, instr, "and");
        if (root == "or") return emit_binary_int_op(os, instr, "or");
        if (root == "xor") return emit_binary_int_op(os, instr, "xor");
        if (root == "not") return emit_not(os, instr);
        if (root == "shl") return emit_binary_int_op(os, instr, "shl");
        if (root == "shr") return emit_binary_int_op(os, instr, "shr");
        if (root == "mad" || root == "fma") return emit_mad_or_fma(os, instr);
        if (root == "neg") return emit_neg(os, instr);
        if (root == "rcp") return emit_rcp(os, instr);
        if (root == "call") return emit_call(os, instr);
        if (root == "bfe") return emit_bfe(os, instr);
        if (root == "atom") return emit_atom(os, instr);
        if (root == "vote") return emit_vote(os, instr);
        if (root == "membar" || root == "fence") return emit_membar_or_fence(os, instr);
        if (root == "cp") return emit_cp_async(os, instr);
        if (root == "brev") return emit_brev(os, instr);
        if (root == "red") return emit_red(os, instr);
        if (root == "activemask") return emit_activemask(os, instr);
        if (root == "bfind") return emit_bfind(os, instr);
        if (root == "lop3") return emit_lop3(os, instr);
        if (root == "sad") return emit_sad(os, instr);
        if (root == "match") return emit_match(os, instr);
        if (root == "fns") return emit_fns(os, instr);
        if (root == "testp") return emit_testp(os, instr);
        if (root == "prmt") return emit_prmt(os, instr);
        if (root == "bfi") return emit_bfi(os, instr);
        if (root == "isspacep") return emit_isspacep(os, instr);
        if (root == "trap") {
            declarations_.insert("declare void @llvm.trap()");
            os << "  call void @llvm.trap()\n";
            return true;
        }
        if (root == "nanosleep" || root == "prefetch" || root == "prefetchu") {
            return true;
        }
        if (root == "redux") {
            return fail(instr,
                        "redux.sync is not implemented by the LLVM PTX backend; "
                        "refusing the former per-lane identity placeholder");
        }
        if (root == "sqrt") return emit_sqrt(os, instr);
        if (root == "rsqrt") return emit_rsqrt(os, instr);
        if (root == "sin") return emit_float_math_unary(os, instr, "air.sin");
        if (root == "cos") return emit_float_math_unary(os, instr, "air.cos");
        if (root == "ex2") return emit_float_math_unary(os, instr, "llvm.exp2");
        if (root == "lg2") return emit_float_math_unary(os, instr, "llvm.log2");
        if (root == "abs") return emit_abs(os, instr);
        if (root == "clz") return emit_clz(os, instr);
        if (root == "popc") return emit_popc(os, instr);
        if (root == "set") return emit_set(os, instr);

        return fail(instr, "unsupported opcode in generic llvm path");
    }

    bool emit_body() {
        if (exec_indices_.empty()) {
            emit_function_return(body_);
            return true;
        }
        // Allocas are gathered lazily while lowering; they will be inserted before the first branch.

        std::ostringstream blocks;
        for (int pos = 0; pos < static_cast<int>(exec_indices_.size()); ++pos) {
            const auto& instr = entry_.instructions[static_cast<std::size_t>(exec_indices_[static_cast<std::size_t>(pos)])];
            std::ostringstream bb;
            bool terminated = false;
            if (!emit_instruction_block(instr, pos, bb, &terminated)) {
                return false;
            }
            if (!terminated) {
                const int next_pos =
                    next_exec_pos_by_exec_pos_.count(pos) ? next_exec_pos_by_exec_pos_.at(pos) : -1;
                bb << "  br label %" << block_name_for_exec_pos(next_pos) << "\n";
            }
            blocks << block_name_for_exec_pos(pos) << ":\n" << bb.str();
        }
        blocks << "cm_exit:\n";
        emit_function_return(blocks);

        body_.str("");
        body_.clear();
        body_ << entry_allocas_.str();
        body_ << "  br label %" << block_name_for_exec_pos(0) << "\n\n";
        body_ << blocks.str();
        return true;
    }
};

GenericLlvmBodyResult try_emit_generic_llvm_body(std::string_view ptx_source,
                                                 const std::string& entry_name,
                                                 std::vector<ParamInfo>* params,
                                                 std::vector<std::string>* arg_decls,
                                                 const std::unordered_map<std::string, GlobalSymbolInfo>* global_symbols,
                                                 const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols,
                                                 const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols,
                                                 const std::vector<cumetal::passes::PrintfLoweredCall>* printf_calls,
                                                 const std::unordered_map<std::string,
                                                     std::vector<cumetal::passes::PrintfLoweredCall>>*
                                                     device_printf_calls,
                                                 cumetal::ptx::Fp64Mode fp64_mode = cumetal::ptx::Fp64Mode::kNative,
                                                 bool module_uses_device_heap = false,
                                                 bool module_uses_device_printf = false,
                                                 bool module_uses_device_launch_queue = false,
                                                 bool module_uses_device_clock = false,
                                                 bool module_uses_grid_barrier = false,
                                                 bool module_uses_grid_y_offset = false) {
    GenericLlvmBodyResult out;

    cumetal::ptx::ParseOptions parse_opts;
    parse_opts.strict = false;
    const auto parsed = cumetal::ptx::parse_ptx(ptx_source, parse_opts);
    if (!parsed.ok) {
        out.error = "generic parse failed: " + parsed.error;
        return out;
    }

    const cumetal::ptx::EntryFunction* entry = nullptr;
    for (const auto& candidate : parsed.module.entries) {
        if (candidate.name == entry_name) {
            entry = &candidate;
            break;
        }
    }
    if (entry == nullptr) {
        out.error = "generic entry not found: " + entry_name;
        return out;
    }

    const auto parameter_bits = [](const cumetal::ptx::Parameter& parameter) {
        const PtxTypeSpec type = parse_primary_type_from_opcode(parameter.type);
        return type.bits > 0 ? type.bits : 0;
    };
    const auto integer_type = [](int bits) {
        return "i" + std::to_string(bits);
    };

    // Establish the scalar indirect-call target set. Helper bodies are emitted
    // after the kernel pass records each call argument's actual address space.
    std::vector<cumetal::ptx::EntryFunction> lowered_functions;
    bool entry_needs_device_functions = false;
    for (const auto& instruction : entry->instructions) {
        if (opcode_root(instruction.opcode) != "call") continue;
        for (const std::string& operand : instruction.operands) {
            if (is_register_name(operand)) entry_needs_device_functions = true;
            for (const auto& function : parsed.module.functions) {
                if (operand == function.name) entry_needs_device_functions = true;
            }
        }
    }
    for (const auto& function : parsed.module.functions) {
        if (!entry_needs_device_functions) break;
        bool scalar_abi = function.return_params.size() <= 1;
        for (const auto& parameter : function.params) {
            scalar_abi = scalar_abi && parameter_bits(parameter) >= 16;
        }
        for (const auto& parameter : function.return_params) {
            scalar_abi = scalar_abi && parameter_bits(parameter) > 0;
        }
        if (scalar_abi) lowered_functions.push_back(function);
    }
    std::vector<std::string> device_kernel_names;
    device_kernel_names.reserve(parsed.module.entries.size());
    for (const auto& kernel : parsed.module.entries) {
        device_kernel_names.push_back(kernel.name);
    }
    // RDC PTX may reference a child kernel that is defined in another fatbin
    // module and appears here only as an `.extern .entry` declaration. It still
    // needs the same stable token as a locally defined launch target.
    const std::string ptx_text(ptx_source);
    const auto collect_declared_symbols = [&](const std::regex& declaration) {
        for (std::sregex_iterator it(ptx_text.begin(), ptx_text.end(), declaration), end;
             it != end; ++it) {
            const std::string name = (*it)[1].str();
            if (std::find(device_kernel_names.begin(), device_kernel_names.end(), name) ==
                device_kernel_names.end()) {
                device_kernel_names.push_back(name);
            }
        }
    };
    collect_declared_symbols(std::regex(
        R"(\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*))"));
    // Clang's RDC PTX declares a kernel defined in another translation unit as
    // `.extern .func child(...)`, even though its address is passed to
    // cudaLaunchDevice as a kernel token. Include those declarations as stable
    // callable identities too. Ordinary extern device functions are harmless:
    // the token is only materialized when PTX takes the symbol's address.
    collect_declared_symbols(std::regex(
        R"(\.extern\s+\.func(?:\s+\([^)]*\))?\s+([A-Za-z_.$][A-Za-z0-9_.$]*))"));

    const auto local_depots = parse_ptx_local_depots(ptx_source, entry_name);
    GenericLlvmEmitter emitter(*entry, params, arg_decls, global_symbols, const_symbols,
                               shared_symbols, &local_depots, printf_calls, fp64_mode,
                               &lowered_functions, true, {}, 0,
                               module_uses_device_heap, module_uses_device_printf,
                               module_uses_device_launch_queue, module_uses_device_clock,
                               module_uses_grid_barrier, module_uses_grid_y_offset,
                               &device_kernel_names);
    out = emitter.run();
    if (out.ok) {
        std::ostringstream helper_ir;
        std::unordered_set<std::string> helper_declarations;
        for (const auto& function : lowered_functions) {
            std::vector<ParamInfo> function_params;
            std::vector<std::string> function_arg_decls;
            for (const auto& parameter : function.params) {
                const int bits = parameter_bits(parameter);
                const std::string type = integer_type(bits);
                const std::string name =
                    sanitize_llvm_identifier(parameter.name, "arg");
                function_params.push_back({.ptx_type = parameter.type,
                                           .llvm_type = type,
                                           .name = name,
                                           .raw_name = parameter.name});
                function_arg_decls.push_back(type + " %" + name);
            }
            if (module_uses_device_heap) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(1)*",
                                           .name = "__cumetal_device_heap",
                                           .raw_name = "__cumetal_device_heap"});
                function_arg_decls.push_back(
                    "i32 addrspace(1)* %__cumetal_device_heap");
            }
            if (module_uses_device_printf) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(1)*",
                                           .name = "__cumetal_printf_buffer",
                                           .raw_name = "__cumetal_printf_buffer"});
                function_arg_decls.push_back(
                    "i32 addrspace(1)* %__cumetal_printf_buffer");
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(2)*",
                                           .name = "__cumetal_printf_capacity",
                                           .raw_name = "__cumetal_printf_capacity"});
                function_arg_decls.push_back(
                    "i32 addrspace(2)* %__cumetal_printf_capacity");
            }
            if (module_uses_device_launch_queue) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(1)*",
                                           .name = "__cumetal_device_launch_queue",
                                           .raw_name = "__cumetal_device_launch_queue"});
                function_arg_decls.push_back(
                    "i32 addrspace(1)* %__cumetal_device_launch_queue");
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(2)*",
                                           .name = "__cumetal_device_launch_queue_capacity",
                                           .raw_name = "__cumetal_device_launch_queue_capacity"});
                function_arg_decls.push_back(
                    "i32 addrspace(2)* %__cumetal_device_launch_queue_capacity");
            }
            if (module_uses_device_clock) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(1)*",
                                           .name = "__cumetal_device_clock",
                                           .raw_name = "__cumetal_device_clock"});
                function_arg_decls.push_back(
                    "i32 addrspace(1)* %__cumetal_device_clock");
            }
            if (module_uses_grid_barrier) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(1)*",
                                           .name = "__cumetal_grid_barrier",
                                           .raw_name = "__cumetal_grid_barrier"});
                function_arg_decls.push_back(
                    "i32 addrspace(1)* %__cumetal_grid_barrier");
            }
            if (module_uses_grid_y_offset) {
                function_params.push_back({.ptx_type = ".u32",
                                           .llvm_type = "i32 addrspace(2)*",
                                           .name = "__cumetal_grid_y_offset",
                                           .raw_name = "__cumetal_grid_y_offset"});
                function_arg_decls.push_back(
                    "i32 addrspace(2)* %__cumetal_grid_y_offset");
            }
            const int return_bits = function.return_params.empty()
                                        ? 0
                                        : parameter_bits(function.return_params.front());
            const std::string return_name = function.return_params.empty()
                                                ? std::string{}
                                                : function.return_params.front().name;
            const auto function_local_depots =
                parse_ptx_local_depots(ptx_source, function.name);
            const auto spaces_it =
                out.device_function_param_address_spaces.find(function.name);
            const std::vector<int>* spaces =
                spaces_it == out.device_function_param_address_spaces.end()
                    ? nullptr
                    : &spaces_it->second;
            const std::vector<cumetal::passes::PrintfLoweredCall>* function_printf_calls =
                nullptr;
            if (device_printf_calls != nullptr) {
                const auto calls_it = device_printf_calls->find(function.name);
                if (calls_it != device_printf_calls->end()) {
                    function_printf_calls = &calls_it->second;
                }
            }
            GenericLlvmEmitter function_emitter(
                function, &function_params, &function_arg_decls, global_symbols,
                const_symbols, shared_symbols, &function_local_depots,
                function_printf_calls, fp64_mode,
                &lowered_functions, false, return_name, return_bits,
                module_uses_device_heap, module_uses_device_printf,
                module_uses_device_launch_queue, module_uses_device_clock,
                module_uses_grid_barrier, module_uses_grid_y_offset,
                &device_kernel_names, spaces);
            GenericLlvmBodyResult function_body = function_emitter.run();
            if (!function_body.ok) {
                out.ok = false;
                out.error = "device function '" + function.name +
                            "' lowering failed: " + function_body.error;
                return out;
            }
            helper_ir << "; PTX device function " << function.name << "\n"
                      << "define internal "
                      << (return_bits > 0 ? integer_type(return_bits) : "void")
                      << " @" << function.name << "(";
            for (std::size_t i = 0; i < function_arg_decls.size(); ++i) {
                if (i > 0) helper_ir << ", ";
                helper_ir << function_arg_decls[i];
            }
            helper_ir << ") {\nentry:\n" << function_body.body_ir << "}\n\n";
            helper_declarations.insert(function_body.declarations.begin(),
                                       function_body.declarations.end());
        }
        out.helper_ir = helper_ir.str();
        std::unordered_set<std::string> all_declarations(out.declarations.begin(),
                                                         out.declarations.end());
        all_declarations.insert(helper_declarations.begin(), helper_declarations.end());
        out.declarations.assign(all_declarations.begin(), all_declarations.end());
        out.uses_device_heap = out.uses_device_heap || module_uses_device_heap;
        out.uses_device_launch_queue = module_uses_device_launch_queue;
        out.uses_device_clock = module_uses_device_clock;
    }
    return out;
}






}  // namespace

static std::vector<ExternalConstantSymbol> find_referenced_external_symbols(
    std::string_view ptx,
    std::string_view entry_name,
    std::string_view state_space) {
    std::vector<ExternalConstantSymbol> out;
    const bool include_all = entry_name.empty();
    const std::string body = extract_entry_body(ptx, entry_name);
    if (!include_all && body.empty()) {
        return out;
    }

    const auto is_identifier_char = [](char c) {
        const unsigned char u = static_cast<unsigned char>(c);
        return std::isalnum(u) != 0 || c == '_' || c == '$' || c == '.';
    };
    const auto body_references = [&](const std::string& symbol) {
        std::size_t pos = 0;
        while ((pos = body.find(symbol, pos)) != std::string::npos) {
            const bool left_ok = pos == 0 || !is_identifier_char(body[pos - 1]);
            const std::size_t end = pos + symbol.size();
            const bool right_ok = end == body.size() || !is_identifier_char(body[end]);
            if (left_ok && right_ok) {
                return true;
            }
            pos = end;
        }
        return false;
    };

    const std::vector<std::pair<std::string_view, std::size_t>> types = {
        {".b64", 8}, {".u64", 8}, {".s64", 8}, {".f64", 8},
        {".b32", 4}, {".u32", 4}, {".s32", 4}, {".f32", 4},
        {".b16", 2}, {".u16", 2}, {".s16", 2}, {".f16", 2},
        {".b8", 1},  {".u8", 1},  {".s8", 1},
    };

    std::istringstream lines{std::string(ptx)};
    std::string line;
    std::size_t next_offset = 0;
    while (std::getline(lines, line)) {
        std::string declaration = trim(line);
        if (const std::size_t comment = declaration.find("//");
            comment != std::string::npos) {
            declaration = trim(declaration.substr(0, comment));
        }
        if (declaration.empty() || declaration.front() != '.' ||
            declaration.find(state_space) == std::string::npos ||
            declaration.find(';') == std::string::npos ||
            declaration.find('=') != std::string::npos ||
            declaration.find('{') != std::string::npos) {
            continue;
        }

        std::size_t alignment = 1;
        if (const std::size_t align_pos = declaration.find(".align");
            align_pos != std::string::npos) {
            std::size_t begin = align_pos + 6;
            while (begin < declaration.size() &&
                   std::isspace(static_cast<unsigned char>(declaration[begin])) != 0) {
                ++begin;
            }
            std::size_t end = begin;
            while (end < declaration.size() &&
                   std::isdigit(static_cast<unsigned char>(declaration[end])) != 0) {
                ++end;
            }
            try {
                alignment = std::max<std::size_t>(
                    1, static_cast<std::size_t>(
                           std::stoull(declaration.substr(begin, end - begin))));
            } catch (...) {
                continue;
            }
        }

        std::size_t type_pos = std::string::npos;
        std::size_t type_len = 0;
        std::size_t element_bytes = 0;
        for (const auto& [token, bytes] : types) {
            const std::size_t found = declaration.find(token);
            if (found != std::string::npos) {
                type_pos = found;
                type_len = token.size();
                element_bytes = bytes;
                break;
            }
        }
        if (type_pos == std::string::npos) {
            continue;
        }

        std::size_t name_begin = type_pos + type_len;
        while (name_begin < declaration.size() &&
               std::isspace(static_cast<unsigned char>(declaration[name_begin])) != 0) {
            ++name_begin;
        }
        std::size_t name_end = name_begin;
        while (name_end < declaration.size() &&
               !std::isspace(static_cast<unsigned char>(declaration[name_end])) &&
               declaration[name_end] != '[' && declaration[name_end] != ';') {
            ++name_end;
        }
        const std::string symbol = declaration.substr(name_begin, name_end - name_begin);
        if (symbol.empty()) {
            continue;
        }

        std::size_t count = 1;
        if (name_end < declaration.size() && declaration[name_end] == '[') {
            const std::size_t close = declaration.find(']', name_end + 1);
            if (close == std::string::npos) {
                continue;
            }
            try {
                count = static_cast<std::size_t>(
                    std::stoull(trim(declaration.substr(name_end + 1, close - name_end - 1))));
            } catch (...) {
                continue;
            }
        }
        if (count == 0 || element_bytes == 0 || count > SIZE_MAX / element_bytes) {
            continue;
        }
        const std::size_t size_bytes = count * element_bytes;
        const std::size_t remainder = next_offset % alignment;
        if (remainder != 0) {
            const std::size_t padding = alignment - remainder;
            if (next_offset > SIZE_MAX - padding) {
                continue;
            }
            next_offset += padding;
        }
        const std::size_t symbol_offset = next_offset;
        if (next_offset > SIZE_MAX - size_bytes) {
            continue;
        }
        next_offset += size_bytes;
        if (include_all || body_references(symbol)) {
            out.push_back({.name = symbol,
                           .offset_bytes = symbol_offset,
                           .size_bytes = size_bytes});
        }
    }
    return out;
}

std::vector<ExternalConstantSymbol> find_referenced_external_constant_symbols(
    std::string_view ptx,
    std::string_view entry_name) {
    return find_referenced_external_symbols(ptx, entry_name, ".const");
}

std::size_t compute_external_constant_buffer_bytes(std::string_view ptx) {
    const auto symbols = find_referenced_external_constant_symbols(ptx, {});
    if (symbols.empty()) {
        return 0;
    }
    const ExternalConstantSymbol& last = symbols.back();
    if (last.offset_bytes > SIZE_MAX - last.size_bytes) {
        return SIZE_MAX;
    }
    return last.offset_bytes + last.size_bytes;
}

std::vector<ExternalGlobalSymbol> find_referenced_external_global_symbols(
    std::string_view ptx,
    std::string_view entry_name) {
    return find_referenced_external_symbols(ptx, entry_name, ".global");
}

LowerToLlvmResult lower_ptx_to_llvm_ir(std::string_view ptx, const LowerToLlvmOptions& options) {
    LowerToLlvmResult result;

    cumetal::passes::Phase1PipelineOptions pipeline_options;
    pipeline_options.strict = options.strict;
    pipeline_options.entry_name = options.entry_name;
    const auto pipeline = cumetal::passes::run_phase1_pipeline(ptx, pipeline_options);
    if (!pipeline.ok) {
        result.error = pipeline.error;
        return result;
    }

    // Device printf may live in a helper rather than directly in the selected
    // kernel (Clang's CUDA frontend commonly emits exactly that shape). Decode
    // every device function up front so the kernel ABI can carry the ring
    // buffer through ordinary helper calls and the runtime receives one module-
    // wide format table with stable, non-colliding ids.
    std::unordered_map<std::string, std::vector<cumetal::passes::PrintfLoweredCall>>
        device_printf_calls;
    std::vector<cumetal::passes::PrintfFormatEntry> all_printf_formats =
        pipeline.printf_formats;
    std::vector<std::string> device_printf_warnings;
    {
        cumetal::ptx::ParseOptions parse_options;
        parse_options.strict = false;
        const auto parsed_module = cumetal::ptx::parse_ptx(ptx, parse_options);
        if (!parsed_module.ok) {
            result.error = "generic parse failed while discovering device printf: " +
                           parsed_module.error;
            return result;
        }
        for (const auto& function : parsed_module.module.functions) {
            cumetal::passes::PrintfLowerOptions printf_options;
            printf_options.strict = options.strict;
            printf_options.ptx_source = ptx;
            auto lowered = cumetal::passes::lower_printf_calls(function, printf_options);
            if (!lowered.ok) {
                result.error = "device function '" + function.name +
                               "' printf lowering failed: " + lowered.error;
                return result;
            }
            device_printf_warnings.insert(device_printf_warnings.end(),
                                          lowered.warnings.begin(),
                                          lowered.warnings.end());
            if (lowered.calls.empty()) continue;

            std::unordered_map<std::uint32_t, std::uint32_t> remapped_ids;
            for (auto format : lowered.formats) {
                const std::uint32_t new_id =
                    static_cast<std::uint32_t>(all_printf_formats.size());
                remapped_ids.emplace(format.id, new_id);
                format.id = new_id;
                all_printf_formats.push_back(std::move(format));
            }
            for (auto& call : lowered.calls) {
                const auto id_it = remapped_ids.find(call.format_id);
                if (id_it == remapped_ids.end()) {
                    result.error = "device function '" + function.name +
                                   "' printf call has no format metadata";
                    return result;
                }
                call.format_id = id_it->second;
            }
            device_printf_calls.emplace(function.name, std::move(lowered.calls));
        }
    }
    const bool module_uses_device_printf =
        !pipeline.printf_calls.empty() || !device_printf_calls.empty();

    const auto fields = to_field_map(pipeline.metadata);
    int arg_count = 0;
    if (const auto it = fields.find("kernel.arg_count"); it != fields.end()) {
        arg_count = std::max(0, std::stoi(it->second));
    }

    std::vector<std::string> arg_decls;
    std::vector<ParamInfo> params;
    arg_decls.reserve(static_cast<std::size_t>(arg_count));
    params.reserve(static_cast<std::size_t>(arg_count));
    for (int i = 0; i < arg_count; ++i) {
        const std::string type_key = "kernel.arg." + std::to_string(i) + ".type";
        const std::string name_key = "kernel.arg." + std::to_string(i) + ".name";
        const std::string pointer_key = "kernel.arg." + std::to_string(i) + ".pointer";

        const auto type_it = fields.find(type_key);
        const auto name_it = fields.find(name_key);
        const auto pointer_it = fields.find(pointer_key);
        const std::string ptx_type = (type_it != fields.end()) ? type_it->second : ".u32";
        const bool is_pointer =
            (pointer_it != fields.end()) &&
            (pointer_it->second == "true" || pointer_it->second == "1");
        const std::string llvm_type = map_param_type_to_llvm(ptx_type, is_pointer);
        const std::string raw_arg_name =
            (name_it != fields.end() && !name_it->second.empty())
                ? name_it->second
                : ("arg_" + std::to_string(i));
        const bool is_param_array = parse_trailing_array_size_bytes(raw_arg_name).has_value();
        const std::string arg_name = sanitize_llvm_identifier(raw_arg_name, "arg_" + std::to_string(i));
        std::string final_llvm_type = llvm_type;
        if (is_param_array && !is_pointer) {
            // PTX by-value aggregates appear as `.param .b8 name[N]` and are addressed via
            // `mov.b64 %rdX, name; ld.param.* [%rdX+off]`. Represent them as constant-buffer
            // pointers so the generic LLVM path can load subfields by offset.
            final_llvm_type = "i8 addrspace(2)*";
        }
        arg_decls.push_back(final_llvm_type + " %" + arg_name);
        params.push_back({.ptx_type = ptx_type,
                          .llvm_type = final_llvm_type,
                          .name = arg_name,
                          .raw_name = raw_arg_name});
    }

    // There used to be four more name-matched body templates here -- vector_add, matrix_mul,
    // negate, and reduce_sum. Each replaced the kernel's *actual* PTX body with a canned
    // implementation whenever the entry name matched a substring and the parameters had roughly
    // the right shape, and each was consulted BEFORE generic lowering was even attempted, which
    // also bypassed strict mode.
    //
    // That silently miscompiled any real kernel whose name happened to match. A kernel named
    // `neg_but_actually_triples` whose PTX computed `x*3` was emitted as `fneg`; `neg.s32` came
    // out as a float sign-bit flip, returning 0x80000007 for -(7) instead of 0xFFFFFFF9. The
    // name match also mutated the ABI (retyping parameters, appending a thread-position builtin),
    // which then made generic lowering fail, which fed the fallback that produced the wrong body
    // in the first place. docs/known-gaps.md had claimed such source-pattern templates were
    // removed and were "not a hidden production fallback"; these were exactly that.
    //
    // Deleting them costs nothing: the generic path lowers all of it. Caught by ptx_sweep_numeric.
    //
    int air_major = 2;
    int air_minor = 8;
    if (const auto it = fields.find("air.version"); it != fields.end()) {
        (void)parse_major_minor(it->second, &air_major, &air_minor);
    }

    int language_major = 4;
    int language_minor = 0;
    if (const auto it = fields.find("language.version"); it != fields.end()) {
        (void)parse_major_minor(it->second, &language_major, &language_minor);
    }

    std::unordered_map<std::string, GlobalSymbolInfo> global_symbols;
    const auto external_global_symbols =
        find_referenced_external_global_symbols(ptx, pipeline.entry_name);
    for (const ExternalGlobalSymbol& symbol : external_global_symbols) {
        if (symbol.name.empty() || symbol.size_bytes == 0 ||
            global_symbols.count(symbol.name) != 0) {
            continue;
        }
        if (params.size() >= 31) {
            result.error = "external PTX global exceeds Metal's kernel buffer binding limit";
            return result;
        }
        const std::string param_name =
            sanitize_llvm_identifier("__cumetal_global_" + symbol.name,
                                     "__cumetal_global_arg");
        arg_decls.push_back("i8 addrspace(1)* %" + param_name);
        params.push_back({.ptx_type = ".b8",
                          .llvm_type = "i8 addrspace(1)*",
                          .name = param_name,
                          .raw_name = param_name});
        global_symbols.emplace(symbol.name,
                               GlobalSymbolInfo{
                                   .llvm_param_name = param_name,
                                   .byte_count = symbol.size_bytes,
                               });
    }

    std::unordered_map<std::string, ConstSymbolInfo> const_symbols;
    std::vector<std::string> const_global_defs;
    // Clang emits device string literals as initialized `.global .b8` arrays,
    // even though kernels only read them. Embed all initialized byte arrays in
    // AIR constant space; writable external `.global` declarations still use
    // the registration-backed global-buffer path below.
    for (const ParsedConstB8Array& array : parse_ptx_initialized_b8_arrays(ptx)) {
        if (array.symbol.empty() || array.bytes.empty()) {
            continue;
        }
        if (const_symbols.count(array.symbol) != 0) {
            continue;
        }
        const std::string llvm_name = quote_llvm_global_name(array.symbol);
        const_symbols.emplace(array.symbol,
                              ConstSymbolInfo{
                                  .llvm_global_name = llvm_name,
                                  .llvm_param_name = {},
                                  .byte_offset = 0,
                                  .byte_count = array.bytes.size(),
                              });
        std::ostringstream def;
        def << llvm_name << " = internal addrspace(2) constant ["
            << array.bytes.size() << " x i8] [";
        for (std::size_t i = 0; i < array.bytes.size(); ++i) {
            if (i > 0) {
                def << ", ";
            }
            def << "i8 " << static_cast<unsigned int>(array.bytes[i]);
        }
        def << "], align " << std::max(1, array.align);
        const_global_defs.push_back(def.str());
    }
    // Clang represents C++ vtables as initialized `.global .u64` arrays whose
    // symbolic elements are device-function addresses. AIR has no public
    // function-pointer ABI, so embed stable tokens in read-only storage. The
    // later indirect-call lowering uses the same token mapping for direct
    // dispatch; ordinary numeric u64 initializers retain their exact bits.
    for (const ParsedConstU64Array& array : parse_ptx_initialized_u64_arrays(ptx)) {
        if (array.symbol.empty() || array.values.empty() ||
            const_symbols.count(array.symbol) != 0) {
            continue;
        }
        const std::string llvm_name = quote_llvm_global_name(array.symbol);
        const std::size_t byte_count = array.values.size() * sizeof(std::uint64_t);
        const_symbols.emplace(array.symbol,
                              ConstSymbolInfo{
                                  .llvm_global_name = llvm_name,
                                  .llvm_param_name = {},
                                  .byte_offset = 0,
                                  .byte_count = byte_count,
                              });
        std::ostringstream def;
        def << llvm_name << " = internal addrspace(2) constant ["
            << byte_count << " x i8] [";
        bool first_byte = true;
        for (const std::uint64_t value : array.values) {
            for (unsigned shift = 0; shift < 64; shift += 8) {
                if (!first_byte) def << ", ";
                first_byte = false;
                def << "i8 " << ((value >> shift) & 0xffu);
            }
        }
        def << "], align " << std::max(1, array.align);
        const_global_defs.push_back(def.str());
    }

    const auto external_constant_symbols =
        find_referenced_external_constant_symbols(ptx, pipeline.entry_name);
    const std::size_t external_constant_buffer_bytes =
        compute_external_constant_buffer_bytes(ptx);
    if (external_constant_buffer_bytes > 64u * 1024u) {
        result.error = "external PTX constant buffer exceeds CUDA's 64 KB module limit";
        return result;
    }
    if (!external_constant_symbols.empty()) {
        if (params.size() > 30) {
            result.error = "kernel argument ABI conflicts with reserved constant buffer index 30";
            return result;
        }
        const std::string param_name = "__cumetal_constant_buffer";
        arg_decls.push_back("i8 addrspace(2)* %" + param_name);
        params.push_back({.ptx_type = ".b8",
                          .llvm_type = "i8 addrspace(2)*",
                          .name = param_name,
                          .raw_name = param_name + "[" +
                                      std::to_string(external_constant_buffer_bytes) + "]"});
        for (const ExternalConstantSymbol& symbol : external_constant_symbols) {
            if (symbol.name.empty() || symbol.size_bytes == 0 ||
                const_symbols.count(symbol.name) != 0) {
                continue;
            }
            const_symbols.emplace(symbol.name,
                                  ConstSymbolInfo{
                                      .llvm_global_name = {},
                                      .llvm_param_name = param_name,
                                      .byte_offset = symbol.offset_bytes,
                                      .byte_count = symbol.size_bytes,
                                  });
        }
    }

    const auto shared_symbols = parse_ptx_shared_symbols(ptx, pipeline.entry_name);

    const std::string selected_entry_body =
        extract_entry_body(ptx, pipeline.entry_name);
    const bool entry_directly_uses_device_heap =
        selected_entry_body.find(", malloc,") != std::string::npos ||
        selected_entry_body.find(" malloc,") != std::string::npos ||
        selected_entry_body.find(" free,") != std::string::npos ||
        selected_entry_body.find("_Znwm") != std::string::npos ||
        selected_entry_body.find("_Znam") != std::string::npos ||
        selected_entry_body.find("_ZdlPv") != std::string::npos ||
        selected_entry_body.find("_ZdaPv") != std::string::npos ||
        selected_entry_body.find("_ZdlPvm") != std::string::npos ||
        selected_entry_body.find("_ZdaPvm") != std::string::npos;
    const bool module_uses_device_heap =
        ptx.find(", malloc,") != std::string::npos ||
        ptx.find(" free,") != std::string::npos ||
        ptx.find("_Znwm") != std::string::npos || ptx.find("_Znam") != std::string::npos ||
        ptx.find("_ZdlPv") != std::string::npos || ptx.find("_ZdaPv") != std::string::npos ||
        ptx.find("_ZdlPvm") != std::string::npos || ptx.find("_ZdaPvm") != std::string::npos;
    const bool entry_has_indirect_call =
        selected_entry_body.find("call %") != std::string::npos ||
        selected_entry_body.find("), %") != std::string::npos;
    const bool entry_uses_device_heap =
        entry_directly_uses_device_heap ||
        (module_uses_device_heap && entry_has_indirect_call);
    if (entry_uses_device_heap) {
        if (params.size() >= 31) {
            result.error = "device heap hidden argument exceeds Metal's kernel buffer binding limit";
            return result;
        }
        arg_decls.push_back("i32 addrspace(1)* %__cumetal_device_heap");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32 addrspace(1)*",
                          .name = "__cumetal_device_heap",
                          .raw_name = "__cumetal_device_heap"});
    }

    const bool module_uses_device_launch_queue =
        ptx.find("cudaGetParameterBuffer") != std::string::npos ||
        ptx.find("cudaLaunchDevice") != std::string::npos ||
        ptx.find("cudaMemcpyAsync") != std::string::npos;
    const bool entry_directly_uses_device_launch_queue =
        selected_entry_body.find("cudaGetParameterBuffer") != std::string::npos ||
        selected_entry_body.find("cudaLaunchDevice") != std::string::npos ||
        selected_entry_body.find("cudaMemcpyAsync") != std::string::npos;
    const bool entry_uses_device_launch_queue =
        entry_directly_uses_device_launch_queue ||
        (module_uses_device_launch_queue && entry_has_indirect_call);
    const bool module_uses_device_clock =
        ptx.find("%clock") != std::string::npos ||
        ptx.find("%globaltimer") != std::string::npos;
    const bool module_uses_grid_barrier =
        ptx.find("__cumetal_grid_sync") != std::string::npos;
    const bool module_uses_grid_y_offset =
        ptx.find("__cumetal_wmma_f32_mma_8x8") != std::string::npos;

    if (module_uses_device_printf) {
        if (params.size() > 28) {
            result.error = "device printf hidden arguments exceed Metal's kernel buffer binding limit";
            return result;
        }
        arg_decls.push_back("i32 addrspace(1)* %__cumetal_printf_buffer");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32 addrspace(1)*",
                          .name = "__cumetal_printf_buffer",
                          .raw_name = "__cumetal_printf_buffer"});
        arg_decls.push_back("i32 %__cumetal_printf_capacity");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32",
                          .name = "__cumetal_printf_capacity",
                          .raw_name = "__cumetal_printf_capacity"});
    }

    if (entry_uses_device_launch_queue) {
        if (params.size() > 28) {
            result.error =
                "device launch queue hidden arguments exceed Metal's kernel buffer binding limit";
            return result;
        }
        arg_decls.push_back("i32 addrspace(1)* %__cumetal_device_launch_queue");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32 addrspace(1)*",
                          .name = "__cumetal_device_launch_queue",
                          .raw_name = "__cumetal_device_launch_queue"});
        arg_decls.push_back("i32 %__cumetal_device_launch_queue_capacity");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32",
                          .name = "__cumetal_device_launch_queue_capacity",
                          .raw_name = "__cumetal_device_launch_queue_capacity"});
    }

    if (module_uses_device_clock) {
        if (params.size() > kDeviceClockBindingIndex) {
            result.error =
                "device clock hidden argument conflicts with reserved Metal buffer index 28";
            return result;
        }
        arg_decls.push_back("i32 addrspace(1)* %__cumetal_device_clock");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32 addrspace(1)*",
                          .name = "__cumetal_device_clock",
                          .raw_name = "__cumetal_device_clock"});
    }
    if (module_uses_grid_barrier) {
        if (params.size() > kGridBarrierBindingIndex) {
            result.error =
                "grid barrier hidden argument conflicts with reserved Metal buffer index 27";
            return result;
        }
        arg_decls.push_back("i32 addrspace(1)* %__cumetal_grid_barrier");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32 addrspace(1)*",
                          .name = "__cumetal_grid_barrier",
                          .raw_name = "__cumetal_grid_barrier"});
    }
    if (module_uses_grid_y_offset) {
        if (params.size() > kGridYOffsetBindingIndex) {
            result.error =
                "grid Y offset hidden argument conflicts with reserved Metal buffer index 26";
            return result;
        }
        arg_decls.push_back("i32 %__cumetal_grid_y_offset");
        params.push_back({.ptx_type = ".u32",
                          .llvm_type = "i32",
                          .name = "__cumetal_grid_y_offset",
                          .raw_name = "__cumetal_grid_y_offset"});
    }

    // Always attempt to lower the kernel's real body first. Nothing may pre-empt this on the
    // strength of the entry name: lowering what the PTX actually says is the only behavior that
    // cannot be silently wrong.
    GenericLlvmBodyResult generic_body;
    bool use_generic_body = false;
    {
        std::vector<ParamInfo> generic_params = params;
        std::vector<std::string> generic_arg_decls = arg_decls;
        generic_body = try_emit_generic_llvm_body(ptx,
                                                  pipeline.entry_name,
                                                  &generic_params,
                                                  &generic_arg_decls,
                                                  &global_symbols,
                                                  &const_symbols,
                                                  &shared_symbols,
                                                  &pipeline.printf_calls,
                                                  &device_printf_calls,
                                                  options.fp64_mode,
                                                  entry_uses_device_heap,
                                                  module_uses_device_printf,
                                                  entry_uses_device_launch_queue,
                                                  module_uses_device_clock,
                                                  module_uses_grid_barrier,
                                                  module_uses_grid_y_offset);
        if (generic_body.ok) {
            params = std::move(generic_params);
            arg_decls = std::move(generic_arg_decls);
            use_generic_body = true;
        } else if (options.strict) {
            result.error = generic_body.error.empty()
                               ? "generic llvm lowering failed"
                               : generic_body.error;
            return result;
        }
    }

    std::ostringstream ir;
    ir << "; ModuleID = '" << options.module_id << "'\n";
    ir << "target triple = \"" << options.target_triple << "\"\n\n";
    ir << "define void @" << pipeline.entry_name << "(";
    for (std::size_t i = 0; i < arg_decls.size(); ++i) {
        if (i > 0) {
            ir << ", ";
        }
        ir << arg_decls[i];
    }
    ir << ") #0 {\n";
    ir << "entry:\n";

    if (use_generic_body) {
        ir << generic_body.body_ir;
    } else {
        // Previously this emitted instruction comments followed by a bare `ret void`, producing a
        // kernel that loaded, launched, and silently did nothing -- every output buffer left
        // untouched, no diagnostic. For a translation layer that is the worst possible outcome:
        // the caller reads whatever was already in the buffer and treats it as a result. Refuse
        // instead, matching how unsupported GGML kernels and approximate lowerings behave.
        result.error = generic_body.error.empty()
                           ? ("no lowering available for entry '" + pipeline.entry_name + "'")
                           : generic_body.error;
        return result;
    }
    ir << "}\n\n";
    if (use_generic_body && !generic_body.helper_ir.empty()) {
        ir << generic_body.helper_ir;
    }

    result.uses_atomic_lock_bank = generic_body.uses_atomic_lock_bank;
    result.uses_device_heap = generic_body.uses_device_heap;
    result.uses_device_clock = generic_body.uses_device_clock;

    if (use_generic_body) {
        for (const std::string& decl : generic_body.declarations) {
            // Attribute group #1 mirrors what `xcrun metal -emit-llvm` puts on
            // AIR builtin declarations. `convergent` is the one that matters:
            // the SIMD shuffles read other lanes' registers, so a backend free
            // to treat the call as non-convergent may move or scalarize it.
            ir << decl << " #1\n";
        }
        if (!generic_body.declarations.empty()) {
            ir << "\n";
        }
    }
    for (const std::string& def : const_global_defs) {
        ir << def << "\n";
    }
    if (!const_global_defs.empty()) {
        ir << "\n";
    }

    std::ostringstream kernel_type;
    kernel_type << "void (";
    for (std::size_t i = 0; i < params.size(); ++i) {
        if (i > 0) {
            kernel_type << ", ";
        }
        kernel_type << params[i].llvm_type;
    }
    kernel_type << ")* @" << pipeline.entry_name;

    const int kernel_node_id = 0;
    const int empty_tuple_id = 1;
    const int kernel_args_tuple_id = 2;
    int next_meta_id = 3;

    std::vector<int> arg_meta_ids;
    arg_meta_ids.reserve(params.size());
    for (std::size_t i = 0; i < params.size(); ++i) {
        arg_meta_ids.push_back(next_meta_id++);
    }

    const int compile_opt_denorm_id = next_meta_id++;
    const int compile_opt_fast_math_id = next_meta_id++;
    const int compile_opt_fbfetch_id = next_meta_id++;
    const int air_version_tuple_id = next_meta_id++;
    const int language_version_tuple_id = next_meta_id++;

    ir << "attributes #0 = { \"air.kernel\" \"air.version\"=\"" << air_major << "."
       << air_minor << "\" }\n";
    ir << "attributes #1 = { convergent mustprogress nounwind willreturn }\n\n";
    ir << "!air.kernel = !{!" << kernel_node_id << "}\n";
    ir << "!" << kernel_node_id << " = !{" << kernel_type.str() << ", !" << empty_tuple_id << ", !"
       << kernel_args_tuple_id << "}\n";
    ir << "!" << empty_tuple_id << " = !{}\n";
    ir << "!" << kernel_args_tuple_id << " = !{";
    for (std::size_t i = 0; i < arg_meta_ids.size(); ++i) {
        if (i > 0) {
            ir << ", ";
        }
        ir << "!" << arg_meta_ids[i];
    }
    ir << "}\n";

    for (std::size_t i = 0; i < params.size(); ++i) {
        const ParamInfo& param = params[i];
        const bool is_legacy_thread_position =
            !is_builtin_param(param) && param.ptx_type == ".builtin.air.thread_position_in_grid";
        const std::string air_type_name = air_type_name_for_param(param, is_legacy_thread_position);
        const int arg_size = byte_size_for_param_metadata(param);
        const int arg_align = arg_size;

        if (is_builtin_param(param) || is_legacy_thread_position) {
            const std::string builtin_key =
                is_builtin_param(param) ? param.builtin_air_key : "air.thread_position_in_grid";
            ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
               << ", !\"" << builtin_key << "\", !\"air.arg_type_name\", !\"" << air_type_name
               << "\", !\"air.arg_name\", !\"" << param.name << "\"}\n";
            continue;
        }

        if (is_device_buffer_pointer(param.llvm_type)) {
            const std::size_t location_index =
                param.name == "__cumetal_atomic_lock_bank"
                    ? kAtomicLockBankBindingIndex
                    : (param.name == "__cumetal_device_clock"
                           ? kDeviceClockBindingIndex
                           : (param.name == "__cumetal_grid_barrier"
                                  ? kGridBarrierBindingIndex
                                  : i));
            ir << "!" << arg_meta_ids[i]
               << " = !{i32 " << i << ", !\"air.buffer\", !\"air.location_index\", i32 "
               << location_index
               << ", i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 "
               << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
               << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
               << param.name << "\"}\n";
            continue;
        }

        if (is_threadgroup_buffer_pointer(param.llvm_type)) {
            ir << "!" << arg_meta_ids[i]
               << " = !{i32 " << i << ", !\"air.buffer\", !\"air.location_index\", i32 0"
               << ", i32 1, !\"air.read_write\", !\"air.address_space\", i32 3, !\"air.arg_type_size\", i32 "
               << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
               << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
               << param.name << "\"}\n";
            continue;
        }

        if (is_constant_buffer_pointer(param.llvm_type)) {
            const int pointee_size = byte_size_for_llvm_type(param.llvm_type);
            const std::size_t location_index =
                param.name == "__cumetal_constant_buffer"
                    ? 30u
                    : (param.name == "__cumetal_grid_y_offset"
                           ? kGridYOffsetBindingIndex
                           : i);
            ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
               << ", !\"air.buffer\", !\"air.buffer_size\", i32 " << arg_size
               << ", !\"air.location_index\", i32 " << location_index
               << ", i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 "
               << pointee_size << ", !\"air.arg_type_align_size\", i32 " << pointee_size
               << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
               << param.name << "\"}\n";
            continue;
        }

        ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
           << ", !\"air.buffer\", !\"air.buffer_size\", i32 " << arg_size
           << ", !\"air.location_index\", i32 " << i
           << ", i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 "
           << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
           << ", !\"air.arg_type_name\", !\"" << air_type_name << "\", !\"air.arg_name\", !\""
           << param.name << "\"}\n";
    }

    ir << "!air.compile_options = !{!" << compile_opt_denorm_id << ", !" << compile_opt_fast_math_id
       << ", !" << compile_opt_fbfetch_id << "}\n";
    ir << "!" << compile_opt_denorm_id << " = !{!\"air.compile.denorms_disable\"}\n";
    ir << "!" << compile_opt_fast_math_id << " = !{!\"air.compile.fast_math_enable\"}\n";
    ir << "!" << compile_opt_fbfetch_id << " = !{!\"air.compile.framebuffer_fetch_enable\"}\n";
    ir << "!air.version = !{!" << air_version_tuple_id << "}\n";
    ir << "!air.language_version = !{!" << language_version_tuple_id << "}\n";
    ir << "!" << air_version_tuple_id << " = !{i32 " << air_major << ", i32 " << air_minor << ", i32 0}\n";
    ir << "!" << language_version_tuple_id << " = !{!\"Metal\", i32 " << language_major << ", i32 "
       << language_minor << ", i32 0}\n";

    result.ok = true;
    result.entry_name = pipeline.entry_name;
    result.llvm_ir = ir.str();
    for (const auto& format : all_printf_formats) {
        result.printf_formats.push_back(format.token);
    }
    result.uses_device_launch_queue =
        use_generic_body && generic_body.uses_device_launch_queue;
    result.warnings = pipeline.warnings;
    result.warnings.insert(result.warnings.end(), device_printf_warnings.begin(),
                           device_printf_warnings.end());
    if (use_generic_body) {
        result.warnings.insert(result.warnings.end(), generic_body.warnings.begin(), generic_body.warnings.end());
    } else if (!generic_body.error.empty()) {
        result.warnings.push_back(generic_body.error);
    }

    // --fp64=emulate: generic register arithmetic uses an FP32 pair and never
    // emits native double ALU operations. Unsupported FP64 memory/conversion
    // forms fail lowering explicitly.

    // --fp64=warn: scan PTX source for any .f64 instructions and emit per-line warnings.
    if (options.fp64_mode == Fp64Mode::kWarn) {
        const std::string ptx_str(ptx);
        std::istringstream ptx_stream(ptx_str);
        std::string line;
        int line_no = 0;
        while (std::getline(ptx_stream, line)) {
            ++line_no;
            if (line.find(".f64") != std::string::npos) {
                result.warnings.push_back(
                    "fp64 instruction at line " + std::to_string(line_no) +
                    " (--fp64=warn): " + trim(line));
            }
        }
    }

    return result;
}

std::size_t compute_static_shared_bytes(std::string_view ptx_text,
                                        std::string_view entry_name) {
    std::string selected_text;
    if (!entry_name.empty()) {
        std::size_t search_from = 0;
        bool found_entry = false;
        while (true) {
            const std::size_t entry_pos = ptx_text.find(".entry", search_from);
            if (entry_pos == std::string_view::npos) {
                break;
            }
            std::size_t name_begin = entry_pos + 6;
            while (name_begin < ptx_text.size() &&
                   std::isspace(static_cast<unsigned char>(ptx_text[name_begin])) != 0) {
                ++name_begin;
            }
            std::size_t name_end = name_begin;
            while (name_end < ptx_text.size() &&
                   std::isspace(static_cast<unsigned char>(ptx_text[name_end])) == 0 &&
                   ptx_text[name_end] != '(' && ptx_text[name_end] != '{') {
                ++name_end;
            }
            if (ptx_text.substr(name_begin, name_end - name_begin) != entry_name) {
                search_from = name_end;
                continue;
            }

            const std::size_t body_begin = ptx_text.find('{', name_end);
            if (body_begin == std::string_view::npos) {
                return 0;
            }
            std::size_t depth = 1;
            std::size_t body_end = body_begin + 1;
            for (; body_end < ptx_text.size() && depth != 0; ++body_end) {
                if (ptx_text[body_end] == '{') {
                    ++depth;
                } else if (ptx_text[body_end] == '}') {
                    --depth;
                }
            }
            if (depth != 0) {
                return 0;
            }
            selected_text.assign(ptx_text.substr(body_begin + 1,
                                                 body_end - body_begin - 2));
            found_entry = true;
            break;
        }
        if (!found_entry) {
            return 0;
        }
    }

    // Scan .shared declarations (non-extern) and compute total aligned size.
    std::size_t cursor = 0;
    std::unordered_set<std::string> seen;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if (t.find(".shared") == std::string::npos) continue;
        if (t.find(".extern") != std::string::npos) continue;

        // This table has to match parse_ptx_shared_symbols() above, which lays the
        // objects out. It used to list only .b64/.b32/.b16/.b8, while the layout pass
        // accepted the full set -- so a scalar `__shared__ unsigned x`, which clang
        // emits as `.shared .align 4 .u32 name;`, was given an offset by one pass and
        // counted as zero bytes by this one. The threadgroup allocation then came out
        // at length 0, and every store to it was dropped while every load returned 0:
        // no error, no warning, just a kernel quietly computing nothing.
        int elem_bytes = 1;
        std::size_t b_pos = std::string::npos;
        std::size_t type_len = 0;
        for (const auto& p : std::vector<std::pair<std::string, int>>{
                {".b64", 8}, {".u64", 8}, {".s64", 8}, {".f64", 8},
                {".b32", 4}, {".u32", 4}, {".s32", 4}, {".f32", 4},
                {".b16", 2}, {".u16", 2}, {".s16", 2}, {".f16", 2},
                {".b8", 1}, {".u8", 1}, {".s8", 1}, {".pred", 1}}) {
            const auto pos = t.find(p.first);
            if (pos != std::string::npos) {
                b_pos = pos;
                type_len = p.first.size();
                elem_bytes = p.second;
                break;
            }
        }
        if (b_pos == std::string::npos) continue;

        std::size_t align_bytes = static_cast<std::size_t>(elem_bytes);
        if (const std::size_t ap = t.find(".align"); ap != std::string::npos) {
            std::size_t pos = ap + 6;
            while (pos < t.size() && std::isspace(static_cast<unsigned char>(t[pos])) != 0) ++pos;
            std::size_t end = pos;
            while (end < t.size() && std::isdigit(static_cast<unsigned char>(t[end])) != 0) ++end;
            if (end > pos) {
                try { align_bytes = static_cast<std::size_t>(std::max(1, std::stoi(t.substr(pos, end - pos)))); }
                catch (...) {}
            }
        }

        std::size_t sym_begin = b_pos + type_len;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        // A declaration without '[' is a scalar shared object, not something to skip:
        // it still occupies one element's worth of threadgroup memory.
        const std::size_t bracket_open = t.find('[', sym_begin);
        const bool is_scalar = bracket_open == std::string::npos;
        const std::size_t symbol_end = is_scalar ? t.find(';', sym_begin) : bracket_open;
        if (symbol_end == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, symbol_end - sym_begin));
        if (symbol.empty() || seen.count(symbol) != 0) continue;
        // Clang commonly emits static shared objects at module scope.  For an
        // entry-specific query, count only declarations referenced by the
        // selected entry body instead of assigning every module object to
        // every separately emitted metallib.
        if (!entry_name.empty() && selected_text.find(symbol) == std::string::npos) {
            continue;
        }
        seen.insert(symbol);

        std::size_t elem_count = is_scalar ? 1u : 0u;
        const std::size_t bracket_close = is_scalar ? std::string::npos : t.find(']', bracket_open + 1);
        if (bracket_close != std::string::npos) {
            const std::string cnt = trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1));
            if (!cnt.empty()) {
                try { elem_count = static_cast<std::size_t>(std::stoull(cnt)); } catch (...) {}
            }
        }
        const std::size_t size_bytes = elem_count * static_cast<std::size_t>(elem_bytes);
        if (size_bytes == 0) continue;

        if (align_bytes > 1) cursor = (cursor + align_bytes - 1) & ~(align_bytes - 1);
        cursor += size_bytes;
    }
    return cursor;
}

}  // namespace cumetal::ptx
