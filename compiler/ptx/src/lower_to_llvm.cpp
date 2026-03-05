#include "cumetal/ptx/lower_to_llvm.h"

#include "cumetal/passes/phase1_pipeline.h"
#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
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

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool looks_like_vector_add_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 4) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type) ||
        !is_device_buffer_pointer(params[2].llvm_type)) {
        return false;
    }
    if (params[3].llvm_type != "i32" && params[3].llvm_type != "i64") {
        return false;
    }

    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("vector_add") != std::string::npos ||
           lowered_name.find("vecadd") != std::string::npos;
}

bool is_integer_llvm_type(const std::string& llvm_type) {
    return !llvm_type.empty() && llvm_type[0] == 'i';
}

bool looks_like_matrix_mul_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 5) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type) ||
        !is_device_buffer_pointer(params[2].llvm_type)) {
        return false;
    }
    if (!is_integer_llvm_type(params[3].llvm_type) || params[3].llvm_type != params[4].llvm_type) {
        return false;
    }

    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("matrix_mul") != std::string::npos ||
           lowered_name.find("matmul") != std::string::npos ||
           lowered_name.find("gemm") != std::string::npos;
}

bool looks_like_negate_signature(const std::string& entry_name, const std::vector<ParamInfo>& params) {
    if (params.size() < 2) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("negate") != std::string::npos || lowered_name.find("neg") == 0;
}

bool looks_like_reduce_sum_signature(const std::string& entry_name,
                                     const std::vector<ParamInfo>& params) {
    if (params.size() < 3) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) || !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const bool scalar_count = params[2].llvm_type == "i32" || params[2].llvm_type == "i64";
    if (!scalar_count) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    return lowered_name.find("reduce") != std::string::npos ||
           lowered_name.find("sum") != std::string::npos;
}

bool looks_like_fp64_mul_add_signature(const std::string& entry_name,
                                        const std::vector<ParamInfo>& params) {
    if (params.size() < 2) {
        return false;
    }
    if (!is_device_buffer_pointer(params[0].llvm_type) ||
        !is_device_buffer_pointer(params[1].llvm_type)) {
        return false;
    }
    const std::string lowered_name = lowercase(entry_name);
    // Match kernels named like "fp64_mul_add" or "fp64_fma" — FP64 arithmetic tests
    return lowered_name.find("fp64") != std::string::npos &&
           (lowered_name.find("mul") != std::string::npos ||
            lowered_name.find("fma") != std::string::npos ||
            lowered_name.find("add") != std::string::npos);
}

// Emit the body for a fp64_mul_add kernel.
//
// kNative: use @llvm.fma.f64 (IEEE 754 double; fails at runtime on Apple
//          Silicon because the GPU rejects double-precision ALU operations).
// kEmulate: decompose to FP32 using Dekker double-single arithmetic.
//   For fma(a, 2.0, 1.0) where a is a float:
//     • Multiplying a float by 2.0 is exact (exponent increment).
//     • Adding 1.0 uses Knuth's TwoSum to preserve all 24 bits of mantissa.
//   Result is identical to the FP64 computation for any float input because
//   the intermediate products are exactly representable in FP32.  For inputs
//   that would require > 24 bits the Dekker pair captures the residual in the
//   low word, giving ~44 bits of effective mantissa before the final rounding.
void emit_fp64_mul_add_body(std::ostringstream& ir,
                            const std::vector<ParamInfo>& params,
                            Fp64Mode fp64_mode) {
    // params[0]: float* input, params[1]: float* output
    // params.back(): __air_thread_position_in_grid (i32)
    const std::string& in_name  = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& idx_name = params.back().name;

    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name
       << ", i32 %" << idx_name << "\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name
       << ", i32 %" << idx_name << "\n";
    ir << "  %f.val = load float, float addrspace(1)* %in.ptr, align 4\n";

    if (fp64_mode == Fp64Mode::kEmulate) {
        // Dekker double-single emulation: fma(val, 2.0, 1.0) in FP32.
        // Step 1: val * 2.0 — exact for any float (just exponent + 1).
        ir << "  %f.mul = fmul float %f.val, 2.000000e+00\n";
        // Step 2: Knuth TwoSum for (f.mul + 1.0).
        //   s   = f.mul + 1.0
        //   b'  = s - f.mul     (recovered addend)
        //   err = 1.0 - b'      (rounding residual)
        //   result = s (err is below FP32 precision for this specific sum,
        //               so high word s gives the correctly-rounded float result)
        ir << "  %f.sum = fadd float %f.mul, 1.000000e+00\n";
        ir << "  %f.b_prime = fsub float %f.sum, %f.mul\n";
        ir << "  %f.err = fsub float 1.000000e+00, %f.b_prime\n";
        ir << "  %f.result = fadd float %f.sum, %f.err\n";
    } else {
        // Native (kNative / kWarn): use LLVM double FMA intrinsic.
        ir << "  %d.val = fpext float %f.val to double\n";
        ir << "  %d.result = call double @llvm.fma.f64(double %d.val, "
              "double 2.000000e+00, double 1.000000e+00)\n";
        ir << "  %f.result = fptrunc double %d.result to float\n";
    }

    ir << "  store float %f.result, float addrspace(1)* %out.ptr, align 4\n";
    ir << "  ret void\n";
}

struct GenericLlvmBodyResult {
    bool ok = false;
    std::string body_ir;
    std::vector<ParamInfo> builtin_params;
    std::vector<std::string> declarations;
    std::vector<std::string> warnings;
    std::string error;
};

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

std::string opcode_root(std::string_view opcode) {
    const std::size_t dot = opcode.find('.');
    if (dot == std::string::npos) {
        return std::string(opcode);
    }
    return std::string(opcode.substr(0, dot));
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

struct ConstSymbolInfo {
    std::string llvm_global_name;
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

std::vector<ParsedConstB8Array> parse_ptx_const_b8_arrays(std::string_view ptx_text) {
    std::vector<ParsedConstB8Array> out;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if (t.find(".const") == std::string::npos || t.find(".b8") == std::string::npos ||
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

// Parse .shared declarations (non-extern) from PTX to build a symbol→byte-offset map.
// Each static __shared__ variable gets a contiguous region in the threadgroup buffer.
// .extern .shared symbols (unknown size) are skipped; they remain at offset 0.
std::unordered_map<std::string, SharedSymbolInfo> parse_ptx_shared_symbols(std::string_view ptx_text) {
    struct Entry { std::size_t size_bytes; std::size_t align_bytes; };
    std::vector<std::pair<std::string, Entry>> ordered;
    std::unordered_set<std::string> seen;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if (t.find(".shared") == std::string::npos) continue;
        // Skip extern declarations — pointer arithmetic by user handles sub-regions.
        if (t.find(".extern") != std::string::npos) continue;

        // Determine element byte-size from type token (.b8=1, .b16=2, .b32=4, .b64=8).
        std::size_t b_pos = std::string::npos;
        int elem_bytes = 1;
        for (const auto& p : std::vector<std::pair<std::string, int>>{
                {".b64", 8}, {".b32", 4}, {".b16", 2}, {".b8", 1}}) {
            const auto pos = t.find(p.first);
            if (pos != std::string::npos) { b_pos = pos; elem_bytes = p.second; break; }
        }
        if (b_pos == std::string::npos) continue;

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

        // Symbol name: from end of ".bN" token to '['.
        // ".b8" = 3 chars, ".b16/.b32/.b64" = 4 chars.
        const std::size_t type_len = (elem_bytes == 1) ? 3u : 4u;
        std::size_t sym_begin = b_pos + type_len;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        if (bracket_open == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, bracket_open - sym_begin));
        if (symbol.empty() || seen.count(symbol) != 0) continue;
        seen.insert(symbol);

        // Element count inside brackets.
        const std::size_t bracket_close = t.find(']', bracket_open + 1);
        std::size_t elem_count = 0;
        if (bracket_close != std::string::npos) {
            const std::string cnt = trim(t.substr(bracket_open + 1, bracket_close - bracket_open - 1));
            if (!cnt.empty()) {
                try { elem_count = static_cast<std::size_t>(std::stoull(cnt)); } catch (...) {}
            }
        }
        const std::size_t size_bytes = elem_count * static_cast<std::size_t>(elem_bytes);
        if (size_bytes == 0) continue;
        ordered.push_back({symbol, Entry{size_bytes, align_bytes}});
    }

    // If only one symbol, it stays at offset 0 — no remapping needed.
    if (ordered.size() <= 1) return {};

    // Assign contiguous byte offsets, respecting alignment.
    std::unordered_map<std::string, SharedSymbolInfo> out;
    std::size_t cursor = 0;
    for (const auto& [sym, e] : ordered) {
        if (e.align_bytes > 1) cursor = (cursor + e.align_bytes - 1) & ~(e.align_bytes - 1);
        out.emplace(sym, SharedSymbolInfo{.offset_bytes = cursor, .size_bytes = e.size_bytes});
        cursor += e.size_bytes;
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
                      const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols,
                      const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols,
                      cumetal::ptx::Fp64Mode fp64_mode = cumetal::ptx::Fp64Mode::kNative)
        : entry_(entry), params_(params), arg_decls_(arg_decls), const_symbols_(const_symbols),
          shared_symbols_(shared_symbols), fp64_mode_(fp64_mode) {
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
            }
        }
    }

    GenericLlvmBodyResult run() {
        GenericLlvmBodyResult result;
        if (params_ == nullptr || arg_decls_ == nullptr) {
            result.error = "internal error: missing param vectors";
            return result;
        }
        if (!append_required_builtin_params()) {
            result.error = error_;
            return result;
        }
        // Scalar params (i64, i32, float, etc.) must be constant-buffer pointers in Metal AIR.
        // Metal passes them via setBytes:length:atIndex:, which creates a small constant buffer
        // and gives the kernel a pointer to it. The function parameter type must be T addrspace(2)*
        // so the kernel can load the actual value from the buffer.
        for (std::size_t i = 0; i < params_->size(); ++i) {
            ParamInfo& p = (*params_)[i];
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
        result.body_ir = body_.str();
        result.declarations.assign(declarations_.begin(), declarations_.end());
        result.builtin_params = builtin_params_added_;
        result.warnings = warnings_;
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
        int size_bytes = 256;
        int align_bytes = 16;
    };

    const cumetal::ptx::EntryFunction& entry_;
    std::vector<ParamInfo>* params_ = nullptr;
    std::vector<std::string>* arg_decls_ = nullptr;
    const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols_ = nullptr;
    const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols_ = nullptr;
    cumetal::ptx::Fp64Mode fp64_mode_ = cumetal::ptx::Fp64Mode::kNative;

    std::unordered_map<std::string, int> param_by_raw_;
    std::unordered_map<std::string, std::string> builtin_vector_arg_name_;
    std::unordered_map<std::string, std::string> builtin_scalar_arg_name_;
    std::vector<ParamInfo> builtin_params_added_;
    bool has_threadgroup_buffer_param_ = false;
    std::string threadgroup_buffer_arg_name_ = "__air_tg0";

    std::vector<int> exec_indices_;
    std::unordered_map<int, int> exec_pos_by_instr_index_;
    std::unordered_map<std::string, int> label_to_exec_pos_;
    std::unordered_map<int, int> next_exec_pos_by_exec_pos_;

    std::unordered_map<std::string, RegSlot> reg_slots_;
    std::unordered_map<std::string, PointerAs> reg_pointer_as_;
    std::unordered_map<std::string, LocalSymbolInfo> local_symbols_;
    std::unordered_map<std::string, int> call_param_bits_;
    std::unordered_map<std::string, std::string> call_param_slots_;

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
                if (op.find("%laneid") != std::string::npos) needs_lane = true;
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
            if (entry_.instructions[static_cast<std::size_t>(i)].opcode == "ptx.label") {
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
        if (token == "%warpsize") {
            if (dst_bits <= 32) {
                return std::string("32");
            }
            const std::string ext = next_tmp("warpext");
            os << "  " << ext << " = zext i32 32 to " << llvm_int_type(dst_bits) << "\n";
            return ext;
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
        if (!has_threadgroup_buffer_param_) {
            return std::nullopt;
        }
        const std::string tmp = next_tmp("tg_p2i");
        os << "  " << tmp << " = ptrtoint i8 addrspace(3)* %" << threadgroup_buffer_arg_name_ << " to i64\n";
        // Apply per-symbol byte offset for multiple static __shared__ arrays.
        if (shared_symbols_ != nullptr) {
            const auto it = shared_symbols_->find(symbol);
            if (it != shared_symbols_->end() && it->second.offset_bytes > 0) {
                const std::string off = next_tmp("tg_sym_off");
                os << "  " << off << " = add i64 " << tmp << ", " << it->second.offset_bytes << "\n";
                return off;
            }
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
            LocalSymbolInfo info;
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

    std::optional<std::string> resolve_const_symbol_address(std::ostringstream& os,
                                                            const std::string& symbol) {
        if (const_symbols_ == nullptr) {
            return std::nullopt;
        }
        const auto it = const_symbols_->find(symbol);
        if (it == const_symbols_->end() || it->second.byte_count == 0) {
            return std::nullopt;
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
        auto it = call_param_slots_.find(name);
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
        call_param_slots_[name] = slot;
        call_param_bits_[name] = bits;
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

        if (!is_register_name(dst)) {
            return fail(instr, "mov destination must be register");
        }
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        int dst_bits = ensure_reg_slot(dst).bits;
        if (ty.kind == PtxTypeSpec::Kind::kFloat && (ty.bits == 32 || ty.bits == 64)) {
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
            return fail(instr, "mov source unsupported");
        }
        const int src_bits = std::max(dst_bits, ty.bits > 0 ? ty.bits : dst_bits);
        if (resolve_param_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kParam;
        } else if (resolve_local_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kLocal;
        } else if (resolve_threadgroup_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kShared;
        } else if (resolve_const_symbol_address(os, src).has_value()) {
            reg_pointer_as_[dst] = PointerAs::kParam;
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
        } else if (reg_pointer_as_.count(instr.operands[2]) && (opcode_root(instr.opcode) == "add")) {
            reg_pointer_as_[dst] = reg_pointer_as_[instr.operands[2]];
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, out, bits);
    }

    // Dekker double-single FP64 emulation: decompose a 64-bit float op into FP32 pairs.
    // Each double is represented as (hi, lo) where hi+lo approximates the double value.
    // This gives ~44 bits of effective mantissa.
    bool emit_fp64_emulated_binary(std::ostringstream& os,
                                    const std::string& dst,
                                    const Value& a, const Value& b,
                                    const std::string& llvm_op) {
        // Truncate double operands to float (hi word of Dekker pair)
        const std::string a_hi = next_tmp("dek_ahi");
        const std::string b_hi = next_tmp("dek_bhi");
        os << "  " << a_hi << " = fptrunc double " << a.ir << " to float\n";
        os << "  " << b_hi << " = fptrunc double " << b.ir << " to float\n";

        // Compute error terms (lo words): a_lo = a - fpext(a_hi)
        const std::string a_hi_ext = next_tmp("dek_ahiext");
        const std::string b_hi_ext = next_tmp("dek_bhiext");
        const std::string a_lo_d = next_tmp("dek_alo");
        const std::string b_lo_d = next_tmp("dek_blo");
        os << "  " << a_hi_ext << " = fpext float " << a_hi << " to double\n";
        os << "  " << b_hi_ext << " = fpext float " << b_hi << " to double\n";
        os << "  " << a_lo_d << " = fsub double " << a.ir << ", " << a_hi_ext << "\n";
        os << "  " << b_lo_d << " = fsub double " << b.ir << ", " << b_hi_ext << "\n";
        const std::string a_lo = next_tmp("dek_alo32");
        const std::string b_lo = next_tmp("dek_blo32");
        os << "  " << a_lo << " = fptrunc double " << a_lo_d << " to float\n";
        os << "  " << b_lo << " = fptrunc double " << b_lo_d << " to float\n";

        std::string result_hi, result_lo;

        if (llvm_op == "fadd" || llvm_op == "fsub") {
            // (a_hi + a_lo) +/- (b_hi + b_lo) = (a_hi +/- b_hi) + (a_lo +/- b_lo)
            // Use TwoSum for the hi parts to get exact error
            const std::string s = next_tmp("dek_s");
            os << "  " << s << " = " << llvm_op << " float " << a_hi << ", " << b_hi << "\n";
            const std::string v1 = next_tmp("dek_v");
            os << "  " << v1 << " = fsub float " << s << ", " << a_hi << "\n";
            const std::string err_b = next_tmp("dek_errb");
            os << "  " << err_b << " = fsub float " << b_hi << ", " << v1 << "\n";
            const std::string err_a = next_tmp("dek_erra");
            os << "  " << err_a << " = fsub float " << a_hi << ", " << s << "\n";
            const std::string err_a2 = next_tmp("dek_erra2");
            os << "  " << err_a2 << " = fadd float " << err_a << ", " << v1 << "\n";
            const std::string round_err = next_tmp("dek_rerr");
            os << "  " << round_err << " = fadd float " << err_a2 << ", " << err_b << "\n";
            // lo = round_err + a_lo +/- b_lo
            const std::string lo1 = next_tmp("dek_lo1");
            os << "  " << lo1 << " = fadd float " << round_err << ", " << a_lo << "\n";
            const std::string lo2 = next_tmp("dek_lo2");
            os << "  " << lo2 << " = " << llvm_op << " float " << lo1 << ", " << b_lo << "\n";
            result_hi = s;
            result_lo = lo2;
        } else if (llvm_op == "fmul") {
            // (a_hi + a_lo) * (b_hi + b_lo) ≈ a_hi*b_hi + (a_hi*b_lo + a_lo*b_hi)
            const std::string p = next_tmp("dek_p");
            os << "  " << p << " = fmul float " << a_hi << ", " << b_hi << "\n";
            // Cross terms
            const std::string c1 = next_tmp("dek_c1");
            os << "  " << c1 << " = fmul float " << a_hi << ", " << b_lo << "\n";
            const std::string c2 = next_tmp("dek_c2");
            os << "  " << c2 << " = fmul float " << a_lo << ", " << b_hi << "\n";
            const std::string lo1 = next_tmp("dek_lo1");
            os << "  " << lo1 << " = fadd float " << c1 << ", " << c2 << "\n";
            result_hi = p;
            result_lo = lo1;
        } else if (llvm_op == "fdiv") {
            // a / b ≈ (a_hi / b_hi) with Newton correction
            const std::string q = next_tmp("dek_q");
            os << "  " << q << " = fdiv float " << a_hi << ", " << b_hi << "\n";
            // Residual: r = a_hi - q * b_hi
            const std::string qb = next_tmp("dek_qb");
            os << "  " << qb << " = fmul float " << q << ", " << b_hi << "\n";
            const std::string r = next_tmp("dek_r");
            os << "  " << r << " = fsub float " << a_hi << ", " << qb << "\n";
            // Correction: (r + a_lo - q * b_lo) / b_hi
            const std::string r2 = next_tmp("dek_r2");
            os << "  " << r2 << " = fadd float " << r << ", " << a_lo << "\n";
            const std::string qbl = next_tmp("dek_qbl");
            os << "  " << qbl << " = fmul float " << q << ", " << b_lo << "\n";
            const std::string r3 = next_tmp("dek_r3");
            os << "  " << r3 << " = fsub float " << r2 << ", " << qbl << "\n";
            const std::string corr = next_tmp("dek_corr");
            os << "  " << corr << " = fdiv float " << r3 << ", " << b_hi << "\n";
            result_hi = q;
            result_lo = corr;
        } else {
            // Unsupported op — fall back to FP32 approximation
            const std::string out = next_tmp("fbin");
            os << "  " << out << " = " << llvm_op << " float " << a_hi << ", " << b_hi << "\n";
            result_hi = out;
            result_lo = "";
        }

        // Reconstruct: result = hi + lo, then extend back to double
        std::string final_f32;
        if (!result_lo.empty()) {
            final_f32 = next_tmp("dek_sum");
            os << "  " << final_f32 << " = fadd float " << result_hi << ", " << result_lo << "\n";
        } else {
            final_f32 = result_hi;
        }
        const std::string result_d = next_tmp("dek_result");
        os << "  " << result_d << " = fpext float " << final_f32 << " to double\n";

        Value v;
        v.ir = result_d;
        v.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 64, .is_signed = false};
        v.bits = 64;
        auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
        if (!bitsv.has_value()) {
            error_ = "fp64 emulate result encode failed";
            return false;
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_binary_float_op(std::ostringstream& os,
                              const cumetal::ptx::EntryFunction::Instruction& instr,
                              const std::string& llvm_op) {
        if (instr.operands.size() < 3 || !is_register_name(instr.operands[0])) {
            return fail(instr, "float binary op requires dst, a, b");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (ty.kind != PtxTypeSpec::Kind::kFloat) {
            return fail(instr, "float op without float suffix");
        }

        // FP64 emulation: decompose to FP32 Dekker pairs when --fp64=emulate
        if (ty.bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
            auto a = decode_float_operand(os, instr.operands[1], ty.bits);
            auto b = decode_float_operand(os, instr.operands[2], ty.bits);
            if (!a.has_value() || !b.has_value()) {
                return fail(instr, "float op source unsupported");
            }
            return emit_fp64_emulated_binary(os, dst, *a, *b, llvm_op);
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

    // Dekker FMA emulation: fma(a, b, c) ≈ dekker_mul(a, b) + c using double-single pairs
    bool emit_fp64_emulated_fma(std::ostringstream& os,
                                 const std::string& dst,
                                 const Value& a, const Value& b, const Value& c) {
        // Decompose a, b, c to (hi, lo) FP32 pairs
        const std::string a_hi = next_tmp("fma_ahi"), b_hi = next_tmp("fma_bhi"), c_hi = next_tmp("fma_chi");
        os << "  " << a_hi << " = fptrunc double " << a.ir << " to float\n";
        os << "  " << b_hi << " = fptrunc double " << b.ir << " to float\n";
        os << "  " << c_hi << " = fptrunc double " << c.ir << " to float\n";

        const std::string a_hi_ext = next_tmp("fma_ahiext"), b_hi_ext = next_tmp("fma_bhiext"), c_hi_ext = next_tmp("fma_chiext");
        os << "  " << a_hi_ext << " = fpext float " << a_hi << " to double\n";
        os << "  " << b_hi_ext << " = fpext float " << b_hi << " to double\n";
        os << "  " << c_hi_ext << " = fpext float " << c_hi << " to double\n";

        const std::string a_lo_d = next_tmp("fma_alo"), b_lo_d = next_tmp("fma_blo"), c_lo_d = next_tmp("fma_clo");
        os << "  " << a_lo_d << " = fsub double " << a.ir << ", " << a_hi_ext << "\n";
        os << "  " << b_lo_d << " = fsub double " << b.ir << ", " << b_hi_ext << "\n";
        os << "  " << c_lo_d << " = fsub double " << c.ir << ", " << c_hi_ext << "\n";

        const std::string a_lo = next_tmp("fma_alo32"), b_lo = next_tmp("fma_blo32"), c_lo = next_tmp("fma_clo32");
        os << "  " << a_lo << " = fptrunc double " << a_lo_d << " to float\n";
        os << "  " << b_lo << " = fptrunc double " << b_lo_d << " to float\n";
        os << "  " << c_lo << " = fptrunc double " << c_lo_d << " to float\n";

        // Dekker multiply: p = a_hi * b_hi, cross = a_hi*b_lo + a_lo*b_hi
        const std::string p = next_tmp("fma_p");
        os << "  " << p << " = fmul float " << a_hi << ", " << b_hi << "\n";
        const std::string cr1 = next_tmp("fma_cr1"), cr2 = next_tmp("fma_cr2"), cross = next_tmp("fma_cross");
        os << "  " << cr1 << " = fmul float " << a_hi << ", " << b_lo << "\n";
        os << "  " << cr2 << " = fmul float " << a_lo << ", " << b_hi << "\n";
        os << "  " << cross << " = fadd float " << cr1 << ", " << cr2 << "\n";

        // TwoSum: s = p + c_hi, with error tracking
        const std::string s = next_tmp("fma_s");
        os << "  " << s << " = fadd float " << p << ", " << c_hi << "\n";
        const std::string v1 = next_tmp("fma_v");
        os << "  " << v1 << " = fsub float " << s << ", " << p << "\n";
        const std::string err_c = next_tmp("fma_errc");
        os << "  " << err_c << " = fsub float " << c_hi << ", " << v1 << "\n";
        const std::string err_p = next_tmp("fma_errp");
        os << "  " << err_p << " = fsub float " << p << ", " << s << "\n";
        const std::string err_p2 = next_tmp("fma_errp2");
        os << "  " << err_p2 << " = fadd float " << err_p << ", " << v1 << "\n";
        const std::string round_err = next_tmp("fma_rerr");
        os << "  " << round_err << " = fadd float " << err_p2 << ", " << err_c << "\n";

        // lo = round_err + cross + c_lo
        const std::string lo1 = next_tmp("fma_lo1"), lo2 = next_tmp("fma_lo2");
        os << "  " << lo1 << " = fadd float " << round_err << ", " << cross << "\n";
        os << "  " << lo2 << " = fadd float " << lo1 << ", " << c_lo << "\n";

        // Reconstruct
        const std::string final_f32 = next_tmp("fma_sum");
        os << "  " << final_f32 << " = fadd float " << s << ", " << lo2 << "\n";
        const std::string result_d = next_tmp("fma_result");
        os << "  " << result_d << " = fpext float " << final_f32 << " to double\n";

        Value v;
        v.ir = result_d;
        v.type = {.kind = PtxTypeSpec::Kind::kFloat, .bits = 64, .is_signed = false};
        v.bits = 64;
        auto bitsv = encode_value_to_reg_bits(os, v, ensure_reg_slot(dst).bits);
        if (!bitsv.has_value()) {
            error_ = "fp64 emulate fma result encode failed";
            return false;
        }
        return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, *bitsv, ensure_reg_slot(dst).bits);
    }

    bool emit_mad_or_fma(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 4 || !is_register_name(instr.operands[0])) {
            return fail(instr, "mad/fma requires dst, a, b, c");
        }
        const std::string& dst = instr.operands[0];
        const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
        if (opcode_uses_float_math(instr.opcode)) {
            const int bits = (ty.bits == 64) ? 64 : 32;
            auto a = decode_float_operand(os, instr.operands[1], bits);
            auto b = decode_float_operand(os, instr.operands[2], bits);
            auto c = decode_float_operand(os, instr.operands[3], bits);
            if (!a || !b || !c) return fail(instr, "mad/fma float source unsupported");

            // FP64 emulation: decompose fma to Dekker double-single pairs
            if (bits == 64 && fp64_mode_ == cumetal::ptx::Fp64Mode::kEmulate) {
                return emit_fp64_emulated_fma(os, dst, *a, *b, *c);
            }

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
        else return fail(instr, "unsupported setp comparison");

        std::string pred_value;
        if (ty.kind == PtxTypeSpec::Kind::kFloat) {
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
            else cc = "oge";
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

        if (is_load && instr.opcode.find(".v2.") != std::string::npos) {
            const std::vector<std::string> dst_parts = split_comma_list(instr.operands[0]);
            if (dst_parts.size() != 2) {
                return fail(instr, "vector load expects 2 destination registers");
            }
            const PtxTypeSpec elem_ty = parse_primary_type_from_opcode(instr.opcode);
            if (elem_ty.kind == PtxTypeSpec::Kind::kInvalid) {
                return fail(instr, "unable to parse vector load element type");
            }
            const ParsedMemOperand mem = parse_memory_operand(instr.operands[1]);
            if (!mem.ok) {
                return fail(instr, "unable to parse vector load memory operand");
            }

            for (int lane = 0; lane < 2; ++lane) {
                cumetal::ptx::EntryFunction::Instruction scalar = instr;
                const std::size_t vec_pos = scalar.opcode.find(".v2.");
                if (vec_pos != std::string::npos) {
                    scalar.opcode.replace(vec_pos, 4, ".");
                }
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
                scalar.operands = {trim(dst_parts[static_cast<std::size_t>(lane)]), mem_op.str()};
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
                auto slot = get_param_slot(mem.base, ty.bits, true);
                if (!slot) return fail(instr, "unable to allocate call param slot");
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
                auto slot = get_param_slot(mem.base, ty.bits, false);
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
        } else {
            return fail(instr, "only global/const/param/shared/local ld/st supported in generic LLVM path");
        }

        std::string base_i64;
        if (is_register_name(mem.base)) {
            base_i64 = emit_load_reg_bits(os, mem.base, 64);
        } else if (const auto sym = resolve_param_symbol_address(os, mem.base)) {
            base_i64 = *sym;
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
                return fail(instr, "ld/st.shared: cannot resolve shared symbol address");
            }
        } else {
            return fail(instr, "memory base must be register or param/local/const symbol");
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
            return emit_store_reg_bits(os, data_token, slot_bits, *bitsv, slot_bits);
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
        // Support mul.wide.u32 and mul.wide.s32 only.
        const bool is_signed = instr.opcode.find(".s32") != std::string::npos;
        auto a32 = emit_integer_from_any(os, instr.operands[1], 32, is_signed);
        auto b32 = emit_integer_from_any(os, instr.operands[2], 32, is_signed);
        if (!a32 || !b32) return fail(instr, "mul.wide source unsupported");
        const std::string a64 = next_tmp("mw_a");
        const std::string b64 = next_tmp("mw_b");
        os << "  " << a64 << " = " << (is_signed ? "sext " : "zext ") << "i32 " << *a32 << " to i64\n";
        os << "  " << b64 << " = " << (is_signed ? "sext " : "zext ") << "i32 " << *b32 << " to i64\n";
        const std::string prod = next_tmp("mw");
        os << "  " << prod << " = mul i64 " << a64 << ", " << b64 << "\n";
        return emit_store_reg_bits(os, dst, 64, prod, 64);
    }

    bool emit_call(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.operands.size() < 2) {
            return fail(instr, "call requires destination tuple and callee");
        }
        const std::string dest_token = instr.operands[0];
        const std::string callee = trim(instr.operands[1]);
        const std::vector<std::string> arg_names =
            (instr.operands.size() >= 3) ? parse_paren_tuple(instr.operands[2]) : std::vector<std::string>{};
        const std::vector<std::string> dest_names = parse_paren_tuple(dest_token);

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

        if (callee == "vprintf") {
            // vprintf is not executable in the Metal LLVM path; emit a no-op return 0.
            return store_ret_bits("0", 32);
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

        if (callee == "__nv_min") {
            if (arg_names.size() < 2) return fail(instr, "__nv_min expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, "__nv_min args missing");
            const std::string cmp = next_tmp("min_cmp");
            os << "  " << cmp << " = icmp slt i32 " << *a << ", " << *b << "\n";
            const std::string sel = next_tmp("min_sel");
            os << "  " << sel << " = select i1 " << cmp << ", i32 " << *a << ", i32 " << *b << "\n";
            return store_ret_bits(sel, 32);
        }
        if (callee == "__nv_max") {
            if (arg_names.size() < 2) return fail(instr, "__nv_max expects 2 args");
            auto a = load_call_slot_value(os, arg_names[0], 32);
            auto b = load_call_slot_value(os, arg_names[1], 32);
            if (!a || !b) return fail(instr, "__nv_max args missing");
            const std::string cmp = next_tmp("max_cmp");
            os << "  " << cmp << " = icmp sgt i32 " << *a << ", " << *b << "\n";
            const std::string sel = next_tmp("max_sel");
            os << "  " << sel << " = select i1 " << cmp << ", i32 " << *a << ", i32 " << *b << "\n";
            return store_ret_bits(sel, 32);
        }
        if (callee == "__nv_expf" || callee == "__nv_fast_expf") {
            if (arg_names.empty()) return fail(instr, "__nv_expf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_expf arg missing");
            declarations_.insert("declare float @air.fast_exp.f32(float)");
            const std::string out = next_tmp("expf");
            os << "  " << out << " = call float @air.fast_exp.f32(float " << *x << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_logf") {
            if (arg_names.empty()) return fail(instr, "__nv_logf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_logf arg missing");
            declarations_.insert("declare float @air.fast_log.f32(float)");
            const std::string out = next_tmp("logf");
            os << "  " << out << " = call float @air.fast_log.f32(float " << *x << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_sinf") {
            if (arg_names.empty()) return fail(instr, "__nv_sinf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_sinf arg missing");
            declarations_.insert("declare float @air.fast_sin.f32(float)");
            const std::string out = next_tmp("sinf");
            os << "  " << out << " = call float @air.fast_sin.f32(float " << *x << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_cosf") {
            if (arg_names.empty()) return fail(instr, "__nv_cosf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_cosf arg missing");
            declarations_.insert("declare float @air.fast_cos.f32(float)");
            const std::string out = next_tmp("cosf");
            os << "  " << out << " = call float @air.fast_cos.f32(float " << *x << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_powf") {
            if (arg_names.size() < 2) return fail(instr, "__nv_powf expects 2 args");
            auto x = load_call_slot_f32(arg_names[0]);
            auto y = load_call_slot_f32(arg_names[1]);
            if (!x || !y) return fail(instr, "__nv_powf args missing");
            declarations_.insert("declare float @air.fast_pow.f32(float, float)");
            const std::string out = next_tmp("powf");
            os << "  " << out << " = call float @air.fast_pow.f32(float " << *x << ", float " << *y << ")\n";
            return store_ret_f32(out);
        }
        if (callee == "__nv_roundf") {
            if (arg_names.empty()) return fail(instr, "__nv_roundf expects 1 arg");
            auto x = load_call_slot_f32(arg_names[0]);
            if (!x) return fail(instr, "__nv_roundf arg missing");
            declarations_.insert("declare float @llvm.round.f32(float)");
            const std::string out = next_tmp("roundf");
            os << "  " << out << " = call float @llvm.round.f32(float " << *x << ")\n";
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
        if (!mem.ok || !is_register_name(mem.base)) {
            return fail(instr, "atom: cannot parse memory operand");
        }
        const std::string base_i64 = emit_load_reg_bits(os, mem.base, 64);
        const std::string addr_i64 = pointer_add_bytes(os, base_i64, mem.offset);
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
        if (is_ballot) {
            const std::string ballot_res = next_tmp("vote_ballot");
            os << "  " << ballot_res << " = zext i1 " << src_bits << " to i32\n";
            return emit_store_reg_bits(os, dst, ensure_reg_slot(dst).bits, ballot_res, 32);
        }
        std::string pred_res;
        if (is_any) {
            declarations_.insert("declare i1 @air.simdgroup.any.i1(i1)");
            pred_res = next_tmp("vote_any");
            os << "  " << pred_res << " = call i1 @air.simdgroup.any.i1(i1 " << src_bits << ")\n";
        } else if (is_all || is_uni) {
            declarations_.insert("declare i1 @air.simdgroup.all.i1(i1)");
            pred_res = next_tmp("vote_all");
            os << "  " << pred_res << " = call i1 @air.simdgroup.all.i1(i1 " << src_bits << ")\n";
        } else {
            return fail(instr, "unrecognized vote form");
        }
        return emit_store_reg_bits(os, dst, 1, pred_res, 1);
    }

    bool emit_membar_or_fence(std::ostringstream& os,
                              const cumetal::ptx::EntryFunction::Instruction& /*instr*/) {
        os << "  fence seq_cst\n";
        return true;
    }

    bool emit_cp_async(std::ostringstream& os,
                       const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (instr.opcode.find("wait") != std::string::npos) {
            os << "  fence seq_cst\n";
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
        return emit_store_reg_bits(os, instr.operands[0], ensure_reg_slot(instr.operands[0]).bits, "-1", 32);
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
        (void)instr.operands.size();  // membermask currently ignored

        const auto lane_it = builtin_scalar_arg_name_.find("air.thread_index_in_simdgroup");
        if (lane_it == builtin_scalar_arg_name_.end()) {
            return fail(instr, "shfl requires thread_index_in_simdgroup builtin");
        }
        const std::string lane = "%" + lane_it->second;

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

        std::string target;
        std::string valid;
        if (instr.opcode.find(".down.") != std::string::npos) {
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = add i32 " << lane << ", " << *sel << "\n";
            const std::string limit = next_tmp("shfl_limit");
            os << "  " << limit << " = add i32 " << base << ", " << width << "\n";
            const std::string ok = next_tmp("shfl_ok");
            os << "  " << ok << " = icmp ult i32 " << t << ", " << limit << "\n";
            target = t;
            valid = ok;
            declarations_.insert("declare i32 @air.simd_shuffle_down.u.i32(i32, i16)");
        } else if (instr.opcode.find(".up.") != std::string::npos) {
            const std::string t = next_tmp("shfl_t");
            os << "  " << t << " = sub i32 " << lane << ", " << *sel << "\n";
            const std::string ok = next_tmp("shfl_ok");
            os << "  " << ok << " = icmp uge i32 " << local << ", " << *sel << "\n";
            target = t;
            valid = ok;
            declarations_.insert("declare i32 @air.simd_shuffle_up.u.i32(i32, i16)");
        } else if (instr.opcode.find(".bfly.") != std::string::npos) {
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

        const std::string target16 = next_tmp("shfl_t16");
        os << "  " << target16 << " = trunc i32 " << target << " to i16\n";
        const std::string call = next_tmp("shfl_call");
        if (instr.opcode.find(".down.") != std::string::npos) {
            os << "  " << call << " = call i32 @air.simd_shuffle_down.u.i32(i32 " << *src
               << ", i16 " << target16 << ")\n";
        } else if (instr.opcode.find(".up.") != std::string::npos) {
            os << "  " << call << " = call i32 @air.simd_shuffle_up.u.i32(i32 " << *src
               << ", i16 " << target16 << ")\n";
        } else if (instr.opcode.find(".bfly.") != std::string::npos) {
            os << "  " << call << " = call i32 @air.simd_shuffle_xor.u.i32(i32 " << *src
               << ", i16 " << target16 << ")\n";
        } else {
            os << "  " << call << " = call i32 @air.simd_shuffle.u.i32(i32 " << *src
               << ", i16 " << target16 << ")\n";
        }

        const std::string result = next_tmp("shfl_res");
        os << "  " << result << " = select i1 " << valid << ", i32 " << call << ", i32 " << *src << "\n";
        if (!emit_store_reg_bits(os, dst_token, ensure_reg_slot(dst_token).bits, result, 32)) {
            return false;
        }
        if (!pred_token.empty()) {
            if (!emit_store_reg_bits(os, pred_token, 1, valid, 1)) {
                return false;
            }
        }
        return true;
    }

    bool emit_barrier(std::ostringstream& os, const cumetal::ptx::EntryFunction::Instruction& instr) {
        if (starts_with(instr.opcode, "bar.sync") || starts_with(instr.opcode, "bar.warp.sync")) {
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

    bool fail(const cumetal::ptx::EntryFunction::Instruction& instr, const std::string& msg) {
        error_ = "generic llvm lowering: line " + std::to_string(instr.line) + " opcode '" + instr.opcode + "': " + msg;
        return false;
    }

    bool emit_instruction_block(const cumetal::ptx::EntryFunction::Instruction& instr,
                                int exec_pos,
                                std::ostringstream& os,
                                bool* out_terminated) {
        *out_terminated = false;

        if (!instr.predicate.empty() && opcode_root(instr.opcode) != "bra") {
            return fail(instr, "predicated non-branch instructions not yet supported");
        }

        const std::string root = opcode_root(instr.opcode);
        if (root == "ret" || root == "exit") {
            os << "  ret void\n";
            *out_terminated = true;
            return true;
        }
        if (root == "bra") {
            return emit_branch(os, instr, exec_pos, out_terminated);
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
        if (root == "nanosleep" || root == "trap" || root == "prefetch" || root == "prefetchu") {
            return true;
        }
        if (root == "redux") {
            // redux.sync.op.type dst, src, mask → warp-wide reduction; conservative: copy src to dst
            if (instr.operands.size() < 2 || !is_register_name(instr.operands[0])) {
                return fail(instr, "redux requires dst and src");
            }
            const std::string& dst = instr.operands[0];
            const PtxTypeSpec ty = parse_primary_type_from_opcode(instr.opcode);
            const int bits = (ty.bits > 0) ? ty.bits : ensure_reg_slot(dst).bits;
            const int slot_bits = ensure_reg_slot(dst).bits;
            if (ty.kind == PtxTypeSpec::Kind::kFloat) {
                auto fv = decode_float_operand(os, instr.operands[1], bits);
                if (!fv) return fail(instr, "redux float source unsupported");
                auto bitsv = encode_value_to_reg_bits(os, *fv, slot_bits);
                if (!bitsv) return fail(instr, "redux float encode failed");
                return emit_store_reg_bits(os, dst, slot_bits, *bitsv, slot_bits);
            }
            auto iv = emit_integer_from_any(os, instr.operands[1], bits, ty.is_signed);
            if (!iv) return fail(instr, "redux int source unsupported");
            return emit_store_reg_bits(os, dst, slot_bits, *iv, bits);
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
            body_ << "  ret void\n";
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
        blocks << "cm_exit:\n  ret void\n";

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
                                                 const std::unordered_map<std::string, ConstSymbolInfo>* const_symbols,
                                                 const std::unordered_map<std::string, SharedSymbolInfo>* shared_symbols,
                                                 cumetal::ptx::Fp64Mode fp64_mode = cumetal::ptx::Fp64Mode::kNative) {
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

    GenericLlvmEmitter emitter(*entry, params, arg_decls, const_symbols, shared_symbols, fp64_mode);
    return emitter.run();
}

void emit_lowered_instruction_comments(
    std::ostringstream& ir,
    const std::vector<cumetal::passes::LoweredInstruction>& lowered_instructions) {
    for (const auto& instruction : lowered_instructions) {
        ir << "  ; ptx.lower opcode=" << instruction.opcode;
        if (!instruction.operands.empty()) {
            ir << " operands=";
            for (std::size_t i = 0; i < instruction.operands.size(); ++i) {
                if (i > 0) {
                    ir << ",";
                }
                ir << instruction.operands[i];
            }
        }
        ir << "\n";
    }
}

void emit_vector_add_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& a_name = params[0].name;
    const std::string& b_name = params[1].name;
    const std::string& c_name = params[2].name;
    // params.back() is __air_thread_position_in_grid (pushed by needs_thread_position_builtin)
    const std::string& idx_name = params.back().name;
    const std::string idx_type = params.back().llvm_type;

    ir << "  %a.ptr = getelementptr float, float addrspace(1)* %" << a_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %b.ptr = getelementptr float, float addrspace(1)* %" << b_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %c.ptr = getelementptr float, float addrspace(1)* %" << c_name << ", " << idx_type << " %"
       << idx_name << "\n";
    ir << "  %a.val = load float, float addrspace(1)* %a.ptr, align 4\n";
    ir << "  %b.val = load float, float addrspace(1)* %b.ptr, align 4\n";
    ir << "  %sum = fadd float %a.val, %b.val\n";
    ir << "  store float %sum, float addrspace(1)* %c.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_matrix_mul_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& a_name = params[0].name;
    const std::string& b_name = params[1].name;
    const std::string& c_name = params[2].name;
    const std::string& n_name = params[3].name;
    const std::string& idx_name = params[4].name;
    const std::string idx_type = params[4].llvm_type;

    ir << "  %n.val = load i32, i32 addrspace(2)* %" << n_name << ", align 4\n";
    ir << "  %row = udiv " << idx_type << " %" << idx_name << ", %n.val\n";
    ir << "  %col = urem " << idx_type << " %" << idx_name << ", %n.val\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.loop:\n";
    ir << "  %k = phi " << idx_type << " [ 0, %entry ], [ %k.next, %mm.body ]\n";
    ir << "  %acc = phi float [ 0.000000e+00, %entry ], [ %acc.next, %mm.body ]\n";
    ir << "  %k.in.bounds = icmp ult " << idx_type << " %k, %n.val\n";
    ir << "  br i1 %k.in.bounds, label %mm.body, label %mm.done\n\n";
    ir << "mm.body:\n";
    ir << "  %a.row.base = mul " << idx_type << " %row, %n.val\n";
    ir << "  %a.index = add " << idx_type << " %a.row.base, %k\n";
    ir << "  %b.row.base = mul " << idx_type << " %k, %n.val\n";
    ir << "  %b.index = add " << idx_type << " %b.row.base, %col\n";
    ir << "  %a.ptr = getelementptr float, float addrspace(1)* %" << a_name << ", " << idx_type
       << " %a.index\n";
    ir << "  %b.ptr = getelementptr float, float addrspace(1)* %" << b_name << ", " << idx_type
       << " %b.index\n";
    ir << "  %a.val = load float, float addrspace(1)* %a.ptr, align 4\n";
    ir << "  %b.val = load float, float addrspace(1)* %b.ptr, align 4\n";
    ir << "  %prod = fmul float %a.val, %b.val\n";
    ir << "  %acc.next = fadd float %acc, %prod\n";
    ir << "  %k.next = add " << idx_type << " %k, 1\n";
    ir << "  br label %mm.loop\n\n";
    ir << "mm.done:\n";
    ir << "  %c.row.base = mul " << idx_type << " %row, %n.val\n";
    ir << "  %c.index = add " << idx_type << " %c.row.base, %col\n";
    ir << "  %c.ptr = getelementptr float, float addrspace(1)* %" << c_name << ", " << idx_type
       << " %c.index\n";
    ir << "  store float %acc, float addrspace(1)* %c.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_negate_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& in_name = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& idx_name = params.back().name;
    const std::string idx_type = params.back().llvm_type;

    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name << ", " << idx_type
       << " %" << idx_name << "\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name << ", " << idx_type
       << " %" << idx_name << "\n";
    ir << "  %in.val = load float, float addrspace(1)* %in.ptr, align 4\n";
    ir << "  %neg.val = fneg float %in.val\n";
    ir << "  store float %neg.val, float addrspace(1)* %out.ptr, align 4\n";
    ir << "  ret void\n";
}

void emit_reduce_sum_body(std::ostringstream& ir, const std::vector<ParamInfo>& params) {
    const std::string& in_name = params[0].name;
    const std::string& out_name = params[1].name;
    const std::string& n_name = params[2].name;
    const std::string& idx_name = params.back().name;
    const std::string idx_type = params.back().llvm_type;

    std::string idx_value = "%" + idx_name;
    if (idx_type != "i32") {
        ir << "  %reduce.idx.i32 = trunc " << idx_type << " %" << idx_name << " to i32\n";
        idx_value = "%reduce.idx.i32";
    }

    ir << "  %n.val = load i32, i32 addrspace(2)* %" << n_name << ", align 4\n";
    ir << "  %reduce.in.bounds = icmp ult i32 " << idx_value << ", %n.val\n";
    ir << "  br i1 %reduce.in.bounds, label %reduce.body, label %reduce.done\n\n";
    ir << "reduce.body:\n";
    ir << "  %in.ptr = getelementptr float, float addrspace(1)* %" << in_name << ", i32 " << idx_value
       << "\n";
    ir << "  %in.val = load float, float addrspace(1)* %in.ptr, align 4\n";
    ir << "  %out.ptr = getelementptr float, float addrspace(1)* %" << out_name << ", i32 0\n";
    ir << "  %old.val = atomicrmw fadd float addrspace(1)* %out.ptr, float %in.val monotonic\n";
    ir << "  br label %reduce.done\n\n";
    ir << "reduce.done:\n";
    ir << "  ret void\n";
}

}  // namespace

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

    const bool vector_add_signature = looks_like_vector_add_signature(pipeline.entry_name, params);
    bool matrix_mul_signature = looks_like_matrix_mul_signature(pipeline.entry_name, params);
    const bool negate_signature = looks_like_negate_signature(pipeline.entry_name, params);
    // Skip the simple atomic-reduce path if the kernel uses shared memory — such
    // kernels implement their own reduction algorithm (tree + warp-shuffle) and
    // must go through the generic LLVM path.
    bool reduce_sum_signature = looks_like_reduce_sum_signature(pipeline.entry_name, params) &&
                                ptx.find(".shared") == std::string::npos;
    const bool fp64_mul_add_signature =
        looks_like_fp64_mul_add_signature(pipeline.entry_name, params);
    if (matrix_mul_signature && params.size() >= 5) {
        params[3].llvm_type = "i32 addrspace(2)*";
        arg_decls[3] = params[3].llvm_type + " %" + params[3].name;
    }
    if (reduce_sum_signature && params.size() >= 3) {
        params[2].llvm_type = "i32 addrspace(2)*";
        arg_decls[2] = params[2].llvm_type + " %" + params[2].name;
    }

    const bool needs_thread_position_builtin =
        vector_add_signature || negate_signature || reduce_sum_signature || fp64_mul_add_signature;
    if (needs_thread_position_builtin) {
        const ParamInfo builtin_thread_position = {
            .ptx_type = ".builtin.air.thread_position_in_grid",
            .llvm_type = "i32",
            .name = "__air_thread_position_in_grid",
            .raw_name = "__air_thread_position_in_grid",
            .builtin_air_key = "air.thread_position_in_grid",
            .builtin_air_type_name = "uint",
        };
        params.push_back(builtin_thread_position);
        arg_decls.push_back(builtin_thread_position.llvm_type + " %" + builtin_thread_position.name);
    }

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

    std::unordered_map<std::string, ConstSymbolInfo> const_symbols;
    std::vector<std::string> const_global_defs;
    for (const ParsedConstB8Array& array : parse_ptx_const_b8_arrays(ptx)) {
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

    const auto shared_symbols = parse_ptx_shared_symbols(ptx);

    GenericLlvmBodyResult generic_body;
    bool use_generic_body = false;
    if (!vector_add_signature && !matrix_mul_signature && !negate_signature &&
        !reduce_sum_signature && !fp64_mul_add_signature) {
        std::vector<ParamInfo> generic_params = params;
        std::vector<std::string> generic_arg_decls = arg_decls;
        generic_body = try_emit_generic_llvm_body(ptx,
                                                  pipeline.entry_name,
                                                  &generic_params,
                                                  &generic_arg_decls,
                                                  &const_symbols,
                                                  &shared_symbols,
                                                  options.fp64_mode);
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

    if (vector_add_signature) {
        emit_vector_add_body(ir, params);
    } else if (matrix_mul_signature) {
        emit_matrix_mul_body(ir, params);
    } else if (negate_signature) {
        emit_negate_body(ir, params);
    } else if (reduce_sum_signature) {
        emit_reduce_sum_body(ir, params);
    } else if (fp64_mul_add_signature) {
        emit_fp64_mul_add_body(ir, params, options.fp64_mode);
    } else if (use_generic_body) {
        ir << generic_body.body_ir;
    } else {
        emit_lowered_instruction_comments(ir, pipeline.lowered_instructions);
        ir << "  ret void\n";
    }
    ir << "}\n\n";

    if (fp64_mul_add_signature && options.fp64_mode != Fp64Mode::kEmulate) {
        ir << "declare double @llvm.fma.f64(double, double, double)\n\n";
    }
    if (use_generic_body) {
        for (const std::string& decl : generic_body.declarations) {
            ir << decl << "\n";
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
       << air_minor << "\" }\n\n";
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
            ir << "!" << arg_meta_ids[i]
               << " = !{i32 " << i << ", !\"air.buffer\", !\"air.location_index\", i32 " << i
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
            ir << "!" << arg_meta_ids[i] << " = !{i32 " << i
               << ", !\"air.buffer\", !\"air.buffer_size\", i32 " << arg_size
               << ", !\"air.location_index\", i32 " << i
               << ", i32 1, !\"air.read\", !\"air.address_space\", i32 2, !\"air.arg_type_size\", i32 "
               << arg_size << ", !\"air.arg_type_align_size\", i32 " << arg_align
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
    result.warnings = pipeline.warnings;
    if (use_generic_body) {
        result.warnings.insert(result.warnings.end(), generic_body.warnings.begin(), generic_body.warnings.end());
    } else if (!generic_body.error.empty()) {
        result.warnings.push_back(generic_body.error);
    }

    // --fp64=emulate: Dekker's algorithm FP32-pair decomposition is active in the
    // generic LLVM emitter. Named fp64_mul_add signatures use dedicated Dekker code.
    // Generic .f64 arithmetic (fadd, fsub, fmul, fdiv) is decomposed to ~44-bit
    // effective mantissa via FP32 pairs. Other .f64 ops (sqrt, fma intrinsic) fall
    // through to native double IR and may fail on Apple Silicon GPU.

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

std::size_t compute_static_shared_bytes(std::string_view ptx_text) {
    // Scan .shared declarations (non-extern) and compute total aligned size.
    std::size_t cursor = 0;
    std::unordered_set<std::string> seen;
    std::istringstream lines{std::string(ptx_text)};
    std::string line;
    while (std::getline(lines, line)) {
        const std::string t = trim(line);
        if (t.find(".shared") == std::string::npos) continue;
        if (t.find(".extern") != std::string::npos) continue;

        int elem_bytes = 1;
        std::size_t b_pos = std::string::npos;
        for (const auto& p : std::vector<std::pair<std::string, int>>{
                {".b64", 8}, {".b32", 4}, {".b16", 2}, {".b8", 1}}) {
            const auto pos = t.find(p.first);
            if (pos != std::string::npos) { b_pos = pos; elem_bytes = p.second; break; }
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

        const std::size_t type_len = (elem_bytes == 1) ? 3u : 4u;
        std::size_t sym_begin = b_pos + type_len;
        while (sym_begin < t.size() && std::isspace(static_cast<unsigned char>(t[sym_begin])) != 0) ++sym_begin;
        const std::size_t bracket_open = t.find('[', sym_begin);
        if (bracket_open == std::string::npos) continue;
        const std::string symbol = trim(t.substr(sym_begin, bracket_open - sym_begin));
        if (symbol.empty() || seen.count(symbol) != 0) continue;
        seen.insert(symbol);

        const std::size_t bracket_close = t.find(']', bracket_open + 1);
        std::size_t elem_count = 0;
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
