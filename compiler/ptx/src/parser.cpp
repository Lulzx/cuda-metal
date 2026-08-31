#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cumetal::ptx {
namespace {

std::string strip_comments(std::string_view text) {
    std::string out;
    out.reserve(text.size());

    enum class State { kNormal, kLineComment, kBlockComment };
    State state = State::kNormal;

    for (std::size_t i = 0; i < text.size(); ++i) {
        const char c = text[i];
        const char next = (i + 1 < text.size()) ? text[i + 1] : '\0';

        if (state == State::kLineComment) {
            if (c == '\n') {
                state = State::kNormal;
                out.push_back(c);
            }
            continue;
        }

        if (state == State::kBlockComment) {
            if (c == '*' && next == '/') {
                state = State::kNormal;
                ++i;
            }
            continue;
        }

        if (c == '/' && next == '/') {
            state = State::kLineComment;
            ++i;
            continue;
        }

        if (c == '/' && next == '*') {
            state = State::kBlockComment;
            ++i;
            continue;
        }

        out.push_back(c);
    }

    return out;
}

bool parse_number(const std::string& token, int* out) {
    if (out == nullptr || token.empty()) {
        return false;
    }
    int value = 0;
    for (const char c : token) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
        value = value * 10 + (c - '0');
    }
    *out = value;
    return true;
}

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

bool extract_balanced_body(const std::string& text,
                           std::size_t open_brace_index,
                           std::string* body,
                           std::size_t* body_start,
                           std::size_t* body_end) {
    if (body == nullptr || body_start == nullptr || body_end == nullptr ||
        open_brace_index >= text.size() || text[open_brace_index] != '{') {
        return false;
    }

    int depth = 1;
    const std::size_t start = open_brace_index + 1;
    std::size_t end = std::string::npos;
    for (std::size_t i = start; i < text.size(); ++i) {
        const char c = text[i];
        if (c == '{') {
            ++depth;
        } else if (c == '}') {
            --depth;
            if (depth == 0) {
                end = i;
                break;
            }
        }
    }

    if (end == std::string::npos || end < start) {
        return false;
    }

    *body = text.substr(start, end - start);
    *body_start = start;
    *body_end = end;
    return true;
}

bool is_supported_opcode(const std::string& opcode) {
    if (opcode.empty()) {
        return false;
    }
    const std::size_t dot = opcode.find('.');
    const std::string root = (dot == std::string::npos) ? opcode : opcode.substr(0, dot);

    static const std::unordered_set<std::string> kSupportedRoots = {
        "abs",       "activemask", "add",       "and",   "atom",  "bar",      "bfe",   "bfi",
        "bfind",     "bra",        "brev",      "brx",   "call",  "clz",      "cos",   "cp",    "cvt",
        "cvta",      "div",        "ex2",       "exit",  "fma",   "fns",      "isspacep",
        "ld",        "lg2",        "lop3",      "mad",   "match", "max",      "membar","min",
        "mov",       "mul",        "nanosleep", "neg",   "not",   "or",       "popc",  "prmt",
        "rcp",       "redux",      "rem",       "ret",   "rsqrt", "sad",      "selp",  "set",
        "setp",      "shl",        "shr",       "shfl",  "sin",   "sqrt",     "st",    "sub",
        "fence",     "prefetch",   "prefetchu", "red",   "suq",   "testp",    "trap",  "txq",
        "vote",      "xor",
    };
    return kSupportedRoots.contains(root);
}

// Returns true for opcodes that must be treated as unsupported regardless of root membership.
// This overrides kSupportedRoots for opcodes whose root is shared with supported variants
// (e.g., cp.async.bulk.tensor shares root "cp" with supported cp.async).
bool is_explicitly_unsupported(const std::string& opcode) {
    return opcode.rfind("cluster.", 0) == 0 ||
           opcode.rfind("mbarrier.", 0) == 0 ||
           opcode.rfind("cp.async.bulk.tensor", 0) == 0 ||
           opcode.rfind("cvt.rn.f8x2", 0) == 0 ||
           opcode.rfind("cvt.rn.f8.", 0) == 0 ||
           opcode.rfind("wmma.", 0) == 0 ||
           opcode.rfind("mma.sync", 0) == 0 ||
           opcode.rfind("ldmatrix.", 0) == 0 ||
           opcode.rfind("tex.", 0) == 0 ||
           opcode.rfind("tld4.", 0) == 0 ||
           opcode.rfind("sust.", 0) == 0 ||
           opcode.rfind("suld.", 0) == 0 ||
           opcode.rfind("sured.", 0) == 0;
}

// Returns a targeted diagnostic for opcodes that are unsupported with a known reason.
// Returns empty string for generic unsupported opcodes.
std::string targeted_unsupported_message(const std::string& opcode) {
    if (opcode.rfind("cluster.", 0) == 0 || opcode.rfind("mbarrier.", 0) == 0) {
        return "unsupported opcode '" + opcode +
               "' (Hopper cluster / distributed shared memory ops have no Metal equivalent)";
    }
    if (opcode.rfind("cp.async.bulk.tensor", 0) == 0) {
        return "unsupported opcode '" + opcode +
               "' (Tensor Memory Accelerator (TMA) has no Metal equivalent)";
    }
    if (opcode.rfind("cvt.rn.f8x2", 0) == 0 || opcode.rfind("cvt.rn.f8.", 0) == 0) {
        return "unsupported opcode '" + opcode +
               "' (FP8 / Transformer Engine types have no Metal equivalent)";
    }
    if (opcode.rfind("wmma.", 0) == 0 || opcode.rfind("mma.sync", 0) == 0 ||
        opcode.rfind("ldmatrix.", 0) == 0) {
        return "unsupported opcode '" + opcode +
               "' (tensor core ops (wmma/mma/ldmatrix) have no Metal equivalent; "
               "use MPSMatrixMultiplication for GEMM workloads)";
    }
    if (opcode.rfind("tex.", 0) == 0 || opcode.rfind("tld4.", 0) == 0 ||
        opcode.rfind("sust.", 0) == 0 ||
        opcode.rfind("suld.", 0) == 0 || opcode.rfind("sured.", 0) == 0 ||
        opcode.rfind("suq.", 0) == 0 || opcode.rfind("txq.", 0) == 0) {
        return "unsupported opcode '" + opcode +
               "' (texture/surface instructions not yet supported; "
               "deferred to v2 — use global memory buffers instead)";
    }
    return "";
}

std::vector<std::string> split_operands(const std::string& text) {
    std::vector<std::string> operands;
    std::string current;
    int bracket_depth = 0;
    for (const char c : text) {
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
            const std::string token = trim(current);
            if (!token.empty()) {
                operands.push_back(token);
            }
            current.clear();
            continue;
        }
        current.push_back(c);
    }
    const std::string tail = trim(current);
    if (!tail.empty()) {
        operands.push_back(tail);
    }
    return operands;
}

std::vector<std::string> split_tokens_ws(const std::string& text) {
    std::istringstream stream(text);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

bool is_ptx_type_token(const std::string& token) {
    if (token.size() < 2 || token[0] != '.') {
        return false;
    }

    static const std::unordered_set<std::string> kNonTypeQualifiers = {
        ".param",
        ".ptr",
        ".align",
    };
    return !kNonTypeQualifiers.contains(token);
}

std::string sanitize_param_name(std::string name) {
    while (!name.empty() && (name.back() == ',' || name.back() == ';' || name.back() == ')')) {
        name.pop_back();
    }
    if (const std::size_t array = name.find('['); array != std::string::npos) {
        name.resize(array);
    }
    return name;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    if (text.size() < prefix.size()) {
        return false;
    }
    return text.substr(0, prefix.size()) == prefix;
}

std::string opcode_root(const std::string& opcode) {
    const std::size_t dot = opcode.find('.');
    if (dot == std::string::npos) {
        return opcode;
    }
    return opcode.substr(0, dot);
}

std::string extract_bracket_contents(const std::string& operand) {
    const std::size_t open = operand.find('[');
    if (open == std::string::npos) {
        return {};
    }
    const std::size_t close = operand.find(']', open + 1);
    if (close == std::string::npos || close <= open + 1) {
        return {};
    }
    return trim(operand.substr(open + 1, close - open - 1));
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

std::string extract_param_name_from_operand(const std::string& operand) {
    std::string token = extract_bracket_contents(operand);
    if (token.empty()) {
        return {};
    }
    if (!token.empty() && token[0] == '%') {
        return {};
    }

    const std::size_t cutoff = token.find_first_of(" +-");
    if (cutoff != std::string::npos) {
        token = token.substr(0, cutoff);
    }
    return sanitize_param_name(trim(token));
}

bool is_memory_opcode(const std::string& opcode) {
    const std::string root = opcode_root(opcode);
    if (root != "ld" && root != "st" && root != "atom") {
        return false;
    }
    if (starts_with(opcode, "ld.param") || starts_with(opcode, "st.param")) {
        return false;
    }
    return true;
}

bool is_64bit_param_type(const std::string& type) {
    return type == ".u64" || type == ".s64" || type == ".b64";
}

void infer_pointer_parameters(EntryFunction* entry) {
    if (entry == nullptr) {
        return;
    }

    struct Usage {
        bool address_use = false;
        bool value_use = false;
    };

    std::unordered_set<std::string> param_names;
    std::unordered_map<std::string, Usage> usage;
    for (const auto& param : entry->params) {
        param_names.insert(param.name);
        usage[param.name] = Usage{};
    }

    std::unordered_map<std::string, std::string> register_to_param;
    for (const auto& instruction : entry->instructions) {
        if (starts_with(instruction.opcode, "ld.param") && instruction.operands.size() >= 2) {
            const std::string dest_register = extract_register_name(instruction.operands[0]);
            const std::string param_name = extract_param_name_from_operand(instruction.operands[1]);
            if (!dest_register.empty() && param_names.contains(param_name)) {
                register_to_param[dest_register] = param_name;
            }
        }

        // ld.param brackets denote PTX parameter syntax, not pointer dereferences.
        // Skipping them prevents incorrectly marking params as address_use=true.
        if (is_memory_opcode(instruction.opcode) &&
            !starts_with(instruction.opcode, "ld.param") &&
            !starts_with(instruction.opcode, "st.param")) {
            for (const std::string& operand : instruction.operands) {
                const std::string bracket = extract_bracket_contents(operand);
                if (bracket.empty()) {
                    continue;
                }

                const std::string register_name = extract_register_name(bracket);
                if (!register_name.empty()) {
                    const auto reg_it = register_to_param.find(register_name);
                    if (reg_it != register_to_param.end()) {
                        usage[reg_it->second].address_use = true;
                    }
                    continue;
                }

                const std::string param_name = sanitize_param_name(trim(bracket));
                if (param_names.contains(param_name)) {
                    usage[param_name].address_use = true;
                }
            }
        }

        if (!starts_with(instruction.opcode, "ld.param") && !instruction.operands.empty()) {
            const std::string dest_register = extract_register_name(instruction.operands[0]);
            if (!dest_register.empty()) {
                std::string mapped_param;
                bool ambiguous = false;
                for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                    const std::string src_register = extract_register_name(instruction.operands[i]);
                    if (src_register.empty()) {
                        continue;
                    }
                    const auto reg_it = register_to_param.find(src_register);
                    if (reg_it == register_to_param.end()) {
                        continue;
                    }
                    if (mapped_param.empty()) {
                        mapped_param = reg_it->second;
                    } else if (mapped_param != reg_it->second) {
                        ambiguous = true;
                        break;
                    }
                }

                if (ambiguous || mapped_param.empty()) {
                    register_to_param.erase(dest_register);
                } else {
                    register_to_param[dest_register] = mapped_param;
                }
            }
        }

        const std::string root = opcode_root(instruction.opcode);
        const bool marks_scalar_use =
            root == "set" || root == "setp" || root == "bra" || root == "call" || root == "div" ||
            root == "rem" || root == "mul" || root == "mad" || root == "fma" || root == "max" ||
            root == "min" || root == "neg" || root == "not" || root == "and" || root == "or" ||
            root == "xor" || root == "rcp" || root == "selp";
        if (marks_scalar_use) {
            const std::size_t first_source_index = (root == "set" || root == "setp") ? 1 : 0;
            for (std::size_t i = first_source_index; i < instruction.operands.size(); ++i) {
                if (!extract_bracket_contents(instruction.operands[i]).empty()) {
                    continue;
                }
                const std::string register_name = extract_register_name(instruction.operands[i]);
                if (register_name.empty()) {
                    continue;
                }
                const auto reg_it = register_to_param.find(register_name);
                if (reg_it != register_to_param.end()) {
                    usage[reg_it->second].value_use = true;
                }
            }
        }
    }

    for (auto& param : entry->params) {
        if (!is_64bit_param_type(param.type)) {
            continue;
        }
        if (param.is_pointer) {
            continue;
        }

        const auto usage_it = usage.find(param.name);
        if (usage_it == usage.end()) {
            continue;
        }
        if (usage_it->second.address_use) {
            param.is_pointer = true;
            continue;
        }
        if (usage_it->second.value_use) {
            param.is_pointer = false;
            continue;
        }

        // Preserve historical behavior for unannotated .u64 params unless proven scalar.
        param.is_pointer = true;
    }
}

int count_line_number_at_offset(const std::string& text, std::size_t offset) {
    int line = 1;
    const std::size_t clamped = std::min(offset, text.size());
    for (std::size_t i = 0; i < clamped; ++i) {
        if (text[i] == '\n') {
            ++line;
        }
    }
    return line;
}

void parse_instructions(const std::string& body,
                        int start_line,
                        EntryFunction* entry,
                        std::vector<std::string>* warnings) {
    if (entry == nullptr || warnings == nullptr) {
        return;
    }

    int line = start_line;
    bool skipping_multiline_directive = false;
    int branch_table_instruction = -1;

    std::istringstream stream(body);
    std::string raw_line;
    while (std::getline(stream, raw_line)) {
        const int current_line = line++;
        std::string line_text = trim(raw_line);
        if (line_text.empty()) {
            continue;
        }

        if (branch_table_instruction >= 0) {
            const bool table_complete = line_text.find(';') != std::string::npos;
            while (!line_text.empty() &&
                   (line_text.back() == ',' || line_text.back() == ';')) {
                line_text.pop_back();
                line_text = trim(line_text);
            }
            if (!line_text.empty()) {
                entry->instructions[static_cast<std::size_t>(branch_table_instruction)]
                    .operands.push_back(line_text);
            }
            if (table_complete) branch_table_instruction = -1;
            continue;
        }
        // PTX allows lexical scope braces to appear on the same line as
        // declarations/instructions. Strip standalone leading/trailing braces so
        // lines like "{ .reg .b32 %r<4>;" parse correctly.
        bool stripped_brace = false;
        do {
            stripped_brace = false;
            if (!line_text.empty() && (line_text.front() == '{' || line_text.front() == '}')) {
                line_text = trim(line_text.substr(1));
                stripped_brace = true;
            }
            if (!line_text.empty() && (line_text.back() == '{' || line_text.back() == '}')) {
                line_text.pop_back();
                line_text = trim(line_text);
                stripped_brace = true;
            }
        } while (stripped_brace);
        if (line_text.empty()) {
            continue;
        }
        if (line_text == "{" || line_text == "}") {
            continue;
        }

        if (skipping_multiline_directive) {
            const std::size_t semi = line_text.find(';');
            if (semi == std::string::npos) {
                continue;
            }
            skipping_multiline_directive = false;
            line_text = trim(line_text.substr(semi + 1));
            if (line_text.empty()) {
                continue;
            }
        }

        // Debug-enabled Clang PTX emits source-location metadata without a
        // terminating semicolon. These are complete, single-line directives;
        // treating them like multiline declarations drops the next real
        // instruction (for example the dynamic-shared-memory address move).
        if (starts_with(line_text, ".loc ") || line_text == ".loc" ||
            starts_with(line_text, ".file ") || line_text == ".file" ||
            starts_with(line_text, ".pragma ") || line_text == ".pragma") {
            continue;
        }

        // PTX sometimes emits inline lexical declarations followed by a real
        // instruction on the same source line:
        //   { .reg .b16 tmp; mov.b32 {tmp, %rs35}, %r1; }
        // Consume leading directives and keep the trailing instruction.
        while (!line_text.empty() && line_text[0] == '.') {
            const std::size_t semi = line_text.find(';');
            if (semi == std::string::npos) {
                skipping_multiline_directive = true;
                line_text.clear();
                break;
            }
            line_text = trim(line_text.substr(semi + 1));
        }
        if (line_text.empty()) {
            continue;
        }

        // `name : .callprototype (...) _ (...);` declares the signature of an
        // indirect-call target. It is a declaration, not an instruction, but the
        // leading label is not followed directly by ':' so the generic path read
        // `name` as an opcode and rejected the entire module -- including entries
        // that never make an indirect call. Skip it. An entry that really does
        // call through the prototype still fails later, at the call site, where
        // indirect device calls are refused.
        if (line_text.find(".callprototype") != std::string::npos ||
            line_text.find(".calltargets") != std::string::npos) {
            continue;
        }


        if (const std::size_t table_marker = line_text.find(": .branchtargets");
            table_marker != std::string::npos) {
            EntryFunction::Instruction table;
            table.line = current_line;
            table.opcode = "ptx.branchtargets";
            table.operands.push_back(trim(line_text.substr(0, table_marker)));
            table.supported = true;
            entry->instructions.push_back(std::move(table));
            if (line_text.find(';', table_marker) == std::string::npos) {
                branch_table_instruction =
                    static_cast<int>(entry->instructions.size()) - 1;
            }
            continue;
        }

        if (line_text.back() == ':') {
            EntryFunction::Instruction label;
            label.line = current_line;
            label.opcode = "ptx.label";
            label.operands.push_back(trim(line_text.substr(0, line_text.size() - 1)));
            label.supported = true;
            entry->instructions.push_back(std::move(label));
            continue;
        }
        if (!line_text.empty() && line_text.back() == ';') {
            line_text.pop_back();
            line_text = trim(line_text);
        }
        if (line_text.empty()) {
            continue;
        }

        EntryFunction::Instruction instruction;
        instruction.line = current_line;

        if (line_text[0] == '@') {
            const std::size_t ws = line_text.find_first_of(" \t");
            if (ws == std::string::npos) {
                continue;
            }
            instruction.predicate = line_text.substr(0, ws);
            line_text = trim(line_text.substr(ws + 1));
        }

        if (line_text.empty()) {
            continue;
        }

        const std::size_t ws = line_text.find_first_of(" \t");
        if (ws == std::string::npos) {
            instruction.opcode = line_text;
        } else {
            instruction.opcode = line_text.substr(0, ws);
            instruction.operands = split_operands(line_text.substr(ws + 1));
        }

        instruction.supported = !is_explicitly_unsupported(instruction.opcode) &&
                                is_supported_opcode(instruction.opcode);
        if (!instruction.supported) {
            const std::string targeted = targeted_unsupported_message(instruction.opcode);
            warnings->push_back(!targeted.empty()
                                     ? targeted
                                     : "unsupported opcode '" + instruction.opcode + "' at line " +
                                           std::to_string(instruction.line));
        }
        entry->instructions.push_back(std::move(instruction));
    }
}

}  // namespace

ParseResult parse_ptx(std::string_view text) {
    return parse_ptx(text, ParseOptions{});
}

ParseResult parse_ptx(std::string_view text, const ParseOptions& options) {
    ParseResult result;
    const std::string source = strip_comments(text);

    std::smatch match;
    const std::regex version_regex(R"(\.version\s+([0-9]+)\.([0-9]+))");
    if (std::regex_search(source, match, version_regex) && match.size() >= 3) {
        (void)parse_number(match[1].str(), &result.module.version_major);
        (void)parse_number(match[2].str(), &result.module.version_minor);
    }

    const std::regex target_regex(R"(\.target\s+([A-Za-z0-9_\.]+))");
    if (std::regex_search(source, match, target_regex) && match.size() >= 2) {
        result.module.target = match[1].str();
    }

    const std::regex entry_regex(
        R"(\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(([^;{}]*?)\)\s*(?:\.[^\n{};]*\s*)*\{)");
    const std::regex param_decl_regex(R"(\.param\s+([^,\n\)]+))");

    const auto parse_parameters = [&](const std::string& params_blob) {
        std::vector<Parameter> params;
        std::sregex_iterator param_iter(params_blob.begin(), params_blob.end(),
                                        param_decl_regex);
        const std::sregex_iterator param_end;
        for (; param_iter != param_end; ++param_iter) {
            if (param_iter->size() < 2) continue;
            const std::string decl = trim((*param_iter)[1].str());
            if (decl.empty()) continue;
            const std::vector<std::string> tokens = split_tokens_ws(decl);
            if (tokens.empty()) continue;
            std::string type = ".u32";
            for (const std::string& token : tokens) {
                if (is_ptx_type_token(token)) {
                    type = token;
                    break;
                }
            }
            const bool is_pointer =
                std::find(tokens.begin(), tokens.end(), ".ptr") != tokens.end();
            std::size_t alignment = 1;
            for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
                if (tokens[i] == ".align") {
                    try {
                        alignment = std::max<std::size_t>(
                            1, static_cast<std::size_t>(std::stoull(tokens[i + 1])));
                    } catch (...) {
                        alignment = 1;
                    }
                }
            }
            std::size_t scalar_bytes = 4;
            if (type.size() >= 3) {
                try {
                    scalar_bytes = std::max<std::size_t>(
                        1, static_cast<std::size_t>(
                               std::stoul(type.substr(2))) /
                               8);
                } catch (...) {
                    scalar_bytes = 4;
                }
            }
            std::size_t byte_size = is_pointer ? 8 : scalar_bytes;
            const std::string raw_name = tokens.back();
            if (const std::size_t open = raw_name.find('[');
                open != std::string::npos) {
                const std::size_t close = raw_name.find(']', open + 1);
                if (close != std::string::npos) {
                    try {
                        byte_size = scalar_bytes * static_cast<std::size_t>(
                            std::stoull(raw_name.substr(open + 1,
                                                        close - open - 1)));
                    } catch (...) {
                        byte_size = scalar_bytes;
                    }
                }
            }
            std::string name = sanitize_param_name(tokens.back());
            if (!name.empty()) {
                params.push_back({.type = type, .name = std::move(name),
                                  .is_pointer = is_pointer,
                                  .byte_size = byte_size,
                                  .alignment = alignment});
            }
        }
        return params;
    };

    std::sregex_iterator iter(source.begin(), source.end(), entry_regex);
    const std::sregex_iterator end;

    for (; iter != end; ++iter) {
        if (iter->size() < 3) {
            continue;
        }

        EntryFunction entry;
        entry.name = (*iter)[1].str();

        entry.params = parse_parameters((*iter)[2].str());

        const std::size_t open_brace = static_cast<std::size_t>(iter->position(0) + iter->length(0) - 1);
        std::string body;
        std::size_t body_start = 0;
        std::size_t body_end = 0;
        if (!extract_balanced_body(source, open_brace, &body, &body_start, &body_end)) {
            result.error = "malformed entry body for '" + entry.name + "'";
            return result;
        }
        const int start_line = count_line_number_at_offset(source, body_start);
        parse_instructions(body, start_line, &entry, &result.warnings);
        infer_pointer_parameters(&entry);

        result.module.entries.push_back(std::move(entry));
    }

    // Device functions use the same instruction grammar as entries but may
    // carry PTX return parameters before their symbol name. Ignore forward
    // declarations: requiring the opening body brace keeps only definitions.
    const std::regex function_regex(
        R"(\.func\s+(?:\(\s*([^;{}]*?)\)\s+)?([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(([^;{}]*?)\)\s*(?:\.[^\n{};]*\s*)*\{)");
    std::sregex_iterator function_iter(source.begin(), source.end(), function_regex);
    for (; function_iter != end; ++function_iter) {
        if (function_iter->size() < 4) continue;
        const std::size_t function_start =
            static_cast<std::size_t>(function_iter->position(0));
        const std::size_t open_brace = static_cast<std::size_t>(
            function_iter->position(0) + function_iter->length(0) - 1);
        // A declaration ends in `;` and has no body. The permissive multiline
        // regex can otherwise begin at an `.extern .func` declaration and keep
        // scanning until the opening brace of the next entry/function, making
        // that unrelated body appear to define the extern symbol.
        if (source.find(';', function_start) < open_brace) {
            continue;
        }
        EntryFunction function;
        function.name = (*function_iter)[2].str();
        function.return_params = parse_parameters((*function_iter)[1].str());
        function.params = parse_parameters((*function_iter)[3].str());
        std::string body;
        std::size_t body_start = 0;
        std::size_t body_end = 0;
        if (!extract_balanced_body(source, open_brace, &body, &body_start, &body_end)) {
            result.error = "malformed device function body for '" + function.name + "'";
            return result;
        }
        const int start_line = count_line_number_at_offset(source, body_start);
        // Function-body coverage is consumed lazily by the selected kernel's
        // call graph. Do not make an unrelated unsupported helper reject every
        // entry in the module during the phase-1 kernel parse.
        std::vector<std::string> function_warnings;
        parse_instructions(body, start_line, &function, &function_warnings);
        infer_pointer_parameters(&function);
        result.module.functions.push_back(std::move(function));
    }

    if (result.module.entries.empty()) {
        result.error = "no .entry definitions found";
        return result;
    }

    if (options.strict && !result.warnings.empty()) {
        result.error = result.warnings.front();
        return result;
    }

    result.ok = true;
    return result;
}

}  // namespace cumetal::ptx
