#include "cumetal/ptx/parser.h"

#include <algorithm>
#include <cctype>
#include <charconv>
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
            if (c == '\n') out.push_back(c);
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
            out.push_back(' '); // Comments separate tokens rather than joining them.
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
        "setp",      "shl",        "shr",       "shf",   "shfl",  "sin",   "sqrt",     "st",    "sub",
        "fence",     "prefetch",   "prefetchu", "red",   "suq",   "testp",    "trap",  "txq",
        "vote",      "xor",        "wmma",  "mma",  "ldmatrix",
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
    // `txq`/`suq` dereference their object operand: a texture or surface
    // handle is a device pointer to the descriptor whose dimensions they
    // read. Leaving them out of the address-use scan made the only pointer
    // use of such a parameter invisible, and once width alone stopped
    // implying a pointer the ABI sidecar labelled the buffer as eight
    // by-value bytes -- every query then returned zero.
    if (root == "txq" || root == "suq") {
        return true;
    }
    // `wmma.load` / `wmma.store` dereference their bracketed address
    // operand; without this a wmma-only pointer parameter is invisible to
    // the address-use scan and misclassified as a by-value scalar.
    if (root == "wmma" && (opcode.find(".load.") != std::string::npos ||
                           opcode.find(".store.") != std::string::npos)) {
        return true;
    }
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
    std::unordered_set<std::string> non_parameter_addresses;
    // A 64-bit parameter spilled to a private stack slot and reloaded (a
    // cudaTextureObject_t kept in a struct on the stack, as COLMAP's PatchMatch
    // cost computer does) is still that parameter. Slots are keyed by their
    // address text; this only nominates the parameter as a pointer, and the
    // typed importer separately proves the reload sees this store.
    std::unordered_map<std::string, std::string> spilled_params;
    const auto slot_key = [](const std::string& operand) {
        std::string key;
        for (const char c : extract_bracket_contents(operand))
            if (c != ' ' && c != '\t') key.push_back(c);
        return key;
    };
    const auto is_wide = [](const std::string& opcode) {
        return opcode.ends_with(".b64") || opcode.ends_with(".u64") || opcode.ends_with(".s64");
    };
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

        // A store reads its address; it does not define that register.
        if (!starts_with(instruction.opcode, "ld.param") &&
            opcode_root(instruction.opcode) != "st" && !instruction.operands.empty()) {
            const std::string dest_register = extract_register_name(instruction.operands[0]);
            if (!dest_register.empty()) {
                const auto root = opcode_root(instruction.opcode);
                bool address = root == "cvta";
                if (root == "mov" && instruction.operands.size() == 2) {
                    const auto source = trim(instruction.operands[1]);
                    address |= !source.empty() &&
                        (std::isalpha(static_cast<unsigned char>(source.front())) ||
                         source.front() == '_' || source.front() == '$') &&
                        !param_names.contains(source);
                }
                if (root == "mov" || root == "add" || root == "sub") {
                    for (std::size_t i = 1; i < instruction.operands.size(); ++i)
                        address |= non_parameter_addresses.contains(extract_register_name(instruction.operands[i]));
                }
                if (address) non_parameter_addresses.insert(dest_register);
                else if (instruction.predicate.empty()) non_parameter_addresses.erase(dest_register);
                std::string mapped_param;
                bool ambiguous = false;
                for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                    const std::string src_register = extract_register_name(instruction.operands[i]);
                    if (src_register.empty()) {
                        continue;
                    }
                    const auto reg_it = register_to_param.find(src_register);
                    if (reg_it == register_to_param.end()) {
                        // A known symbol-derived address must not attribute a
                        // combined address solely to its scalar offset. Keep
                        // ordinary untracked thread-index arithmetic compatible.
                        if (non_parameter_addresses.contains(src_register)) {
                            ambiguous = true;
                            break;
                        }
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
                // A redefined base register moves every slot addressed through it.
                for (auto it = spilled_params.begin(); it != spilled_params.end();) {
                    if (it->first == dest_register || it->first.starts_with(dest_register + "+") ||
                        it->first.starts_with(dest_register + "-"))
                        it = spilled_params.erase(it);
                    else
                        ++it;
                }
                if (starts_with(instruction.opcode, "ld.local") && is_wide(instruction.opcode) &&
                    instruction.predicate.empty() && instruction.operands.size() == 2) {
                    const auto spilled = spilled_params.find(slot_key(instruction.operands[1]));
                    if (spilled != spilled_params.end()) register_to_param[dest_register] = spilled->second;
                }
            }
        }
        if (starts_with(instruction.opcode, "st.local") && instruction.operands.size() == 2) {
            const std::string key = slot_key(instruction.operands[0]);
            const auto source = register_to_param.find(extract_register_name(instruction.operands[1]));
            if (!key.empty() && is_wide(instruction.opcode) && instruction.predicate.empty() &&
                source != register_to_param.end())
                spilled_params[key] = source->second;
            else
                spilled_params.erase(key);
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

        // Integer width alone is not pointer evidence. The typed importer can
        // recover omitted annotations from address uses beyond this classifier.
        param.is_pointer = false;
    }
}

// A 64-bit kernel parameter spilled into the kernel's stack frame and only
// dereferenced by a device helper that receives a pointer to that frame (a
// cudaTextureObject_t kept in a struct whose member function reads it, as
// COLMAP's PatchMatch cost computer does) is a pointer too, although the
// kernel itself never uses it as an address. Recognised shape:
//   kernel: st.local.b64 [frame + S], %param_value
//           st.param.b64 [arg_k], frame + A   (then a direct call to H)
//   H:      ld.param.b64 %p, [H_param_k];  ld.b64 %v, [%p + (S - A)]
//           %v (or one add/mov of it) used as a memory address.
// Frame addresses are tracked from the depot setup (%SPL/%SP plus literal
// offsets); anything less direct is left alone. This only nominates the
// parameter as a pointer; the typed importer still proves every reload.
void infer_pointer_parameters_through_calls(ModuleInfo* module) {
    std::unordered_map<std::string, const EntryFunction*> helpers;
    for (const auto& function : module->functions) helpers.emplace(function.name, &function);
    const auto is_wide = [](const std::string& opcode) {
        return opcode.ends_with(".b64") || opcode.ends_with(".u64") || opcode.ends_with(".s64");
    };
    const auto bracket_parts = [](const std::string& operand, std::string* base, std::int64_t* offset) {
        const std::string inner = trim(extract_bracket_contents(operand));
        if (inner.empty()) return false;
        const std::size_t plus = inner.find('+');
        *base = trim(inner.substr(0, plus));
        *offset = 0;
        if (plus != std::string::npos) {
            try {
                *offset = std::stoll(trim(inner.substr(plus + 1)));
            } catch (...) {
                return false;
            }
        }
        return !base->empty() && base->front() == '%';
    };
    // Byte offsets (relative to the parameter) that a helper dereferences
    // after loading them from its pointer parameter `index`.
    std::unordered_map<std::string, std::unordered_set<std::int64_t>> helper_pointer_fields;
    const auto pointer_fields = [&](const EntryFunction& helper, std::size_t index) {
        const std::string key = helper.name + "#" + std::to_string(index);
        if (const auto found = helper_pointer_fields.find(key); found != helper_pointer_fields.end())
            return found->second;
        std::unordered_set<std::int64_t> fields;
        if (index < helper.params.size()) {
            std::unordered_set<std::string> carriers;
            std::unordered_map<std::string, std::int64_t> loaded;  // register -> field offset
            for (const auto& instruction : helper.instructions) {
                if (starts_with(instruction.opcode, "ld.param") && instruction.operands.size() == 2 &&
                    extract_param_name_from_operand(instruction.operands[1]) == helper.params[index].name)
                    carriers.insert(extract_register_name(instruction.operands[0]));
            }
            for (const auto& instruction : helper.instructions) {
                if (opcode_root(instruction.opcode) == "ld" && !starts_with(instruction.opcode, "ld.param") &&
                    is_wide(instruction.opcode) && instruction.operands.size() == 2) {
                    std::string base;
                    std::int64_t offset = 0;
                    if (bracket_parts(instruction.operands[1], &base, &offset) && carriers.contains(base))
                        loaded[extract_register_name(instruction.operands[0])] = offset;
                }
            }
            // One add/mov hop: a field plus an offset is still that field's pointer.
            for (const auto& instruction : helper.instructions) {
                const std::string root = opcode_root(instruction.opcode);
                if ((root == "add" || root == "mov") && instruction.operands.size() >= 2) {
                    for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                        const auto source = loaded.find(extract_register_name(instruction.operands[i]));
                        if (source != loaded.end()) {
                            loaded.emplace(extract_register_name(instruction.operands[0]), source->second);
                            break;
                        }
                    }
                }
            }
            for (const auto& instruction : helper.instructions) {
                if (!is_memory_opcode(instruction.opcode) || starts_with(instruction.opcode, "ld.param") ||
                    starts_with(instruction.opcode, "st.param"))
                    continue;
                for (const auto& operand : instruction.operands) {
                    std::string base;
                    std::int64_t offset = 0;
                    if (!bracket_parts(operand, &base, &offset)) continue;
                    if (const auto field = loaded.find(base); field != loaded.end()) fields.insert(field->second);
                }
            }
        }
        return helper_pointer_fields[key] = fields;
    };
    // Whether a helper uses its parameter `index` itself as a memory address
    // (directly, or after one add/mov/cvta hop).
    std::unordered_map<std::string, bool> helper_dereferences;
    const auto dereferences_param = [&](const EntryFunction& helper, std::size_t index) {
        const std::string key = helper.name + "#" + std::to_string(index);
        if (const auto found = helper_dereferences.find(key); found != helper_dereferences.end())
            return found->second;
        bool used = false;
        if (index < helper.params.size()) {
            std::unordered_set<std::string> carriers;
            for (const auto& instruction : helper.instructions) {
                if (starts_with(instruction.opcode, "ld.param") && is_wide(instruction.opcode) &&
                    instruction.operands.size() == 2 &&
                    extract_param_name_from_operand(instruction.operands[1]) == helper.params[index].name)
                    carriers.insert(extract_register_name(instruction.operands[0]));
            }
            std::unordered_set<std::string> addresses = carriers;
            for (const auto& instruction : helper.instructions) {
                const std::string root = opcode_root(instruction.opcode);
                if ((root == "add" || root == "mov" || root == "cvta") && instruction.operands.size() >= 2 &&
                    is_wide(instruction.opcode)) {
                    for (std::size_t i = 1; i < instruction.operands.size(); ++i) {
                        if (carriers.contains(extract_register_name(instruction.operands[i]))) {
                            addresses.insert(extract_register_name(instruction.operands[0]));
                            break;
                        }
                    }
                }
            }
            for (const auto& instruction : helper.instructions) {
                if (used) break;
                if (!is_memory_opcode(instruction.opcode) || starts_with(instruction.opcode, "ld.param") ||
                    starts_with(instruction.opcode, "st.param"))
                    continue;
                for (const auto& operand : instruction.operands) {
                    std::string base;
                    std::int64_t offset = 0;
                    if (bracket_parts(operand, &base, &offset) && addresses.contains(base)) {
                        used = true;
                        break;
                    }
                }
            }
        }
        return helper_dereferences[key] = used;
    };

    for (auto& entry : module->entries) {
        std::unordered_map<std::string, std::int64_t> frame;   // register -> depot offset
        std::unordered_map<std::string, std::size_t> param_value;  // register -> param index
        std::unordered_map<std::int64_t, std::size_t> spilled;     // depot offset -> param index
        std::unordered_map<std::string, std::int64_t> staged;      // call arg name -> depot offset
        std::unordered_map<std::string, std::size_t> staged_param;  // call arg name -> param index
        std::unordered_set<std::size_t> pointers;
        for (const auto& instruction : entry.instructions) {
            const std::string root = opcode_root(instruction.opcode);
            const std::string dest = instruction.operands.empty() ? std::string()
                                                                  : extract_register_name(instruction.operands[0]);
            if (root == "mov" && instruction.operands.size() == 2 &&
                trim(instruction.operands[1]).starts_with("__local_depot")) {
                frame[dest] = 0;
                continue;
            }
            if ((root == "cvta" || root == "mov") && instruction.operands.size() == 2) {
                if (const auto source = frame.find(extract_register_name(instruction.operands[1]));
                    source != frame.end()) {
                    frame[dest] = source->second;
                    continue;
                }
                if (const auto source = param_value.find(extract_register_name(instruction.operands[1]));
                    source != param_value.end() && is_wide(instruction.opcode)) {
                    const std::size_t index = source->second;
                    frame.erase(dest);
                    param_value[dest] = index;
                    continue;
                }
            }
            if (root == "add" && instruction.operands.size() == 3) {
                const auto source = frame.find(extract_register_name(instruction.operands[1]));
                std::int64_t literal = 0;
                bool numeric = true;
                try {
                    literal = std::stoll(trim(instruction.operands[2]));
                } catch (...) {
                    numeric = false;
                }
                if (source != frame.end() && numeric) {
                    frame[dest] = source->second + literal;
                    continue;
                }
            }
            if (starts_with(instruction.opcode, "ld.param") && instruction.operands.size() == 2 &&
                is_wide(instruction.opcode)) {
                const std::string name = extract_param_name_from_operand(instruction.operands[1]);
                for (std::size_t i = 0; i < entry.params.size(); ++i)
                    if (entry.params[i].name == name) param_value[dest] = i;
                continue;
            }
            if (root == "st" && starts_with(instruction.opcode, "st.param") && instruction.operands.size() == 2 &&
                is_wide(instruction.opcode)) {
                const std::string name = extract_param_name_from_operand(instruction.operands[0]);
                if (const auto value = frame.find(extract_register_name(instruction.operands[1]));
                    value != frame.end())
                    staged[name] = value->second;
                else
                    staged.erase(name);
                if (const auto value = param_value.find(extract_register_name(instruction.operands[1]));
                    value != param_value.end())
                    staged_param[name] = value->second;
                else
                    staged_param.erase(name);
                continue;
            }
            if (root == "st" && instruction.operands.size() == 2 && is_wide(instruction.opcode) &&
                instruction.predicate.empty()) {
                std::string base;
                std::int64_t offset = 0;
                if (bracket_parts(instruction.operands[0], &base, &offset)) {
                    if (const auto address = frame.find(base); address != frame.end()) {
                        const auto value = param_value.find(extract_register_name(instruction.operands[1]));
                        if (value != param_value.end()) spilled[address->second + offset] = value->second;
                        else spilled.erase(address->second + offset);
                    }
                }
                continue;
            }
            if (root == "call" && instruction.operands.size() >= 2) {
                std::string callee;
                std::vector<std::string> arguments;
                for (const auto& operand : instruction.operands) {
                    const std::string text = trim(operand);
                    if (text.starts_with("(")) {
                        std::string list = text.substr(1, text.size() >= 2 ? text.size() - 2 : 0);
                        std::size_t at = 0;
                        arguments.clear();
                        while (at <= list.size()) {
                            const std::size_t comma = list.find(',', at);
                            arguments.push_back(trim(list.substr(at, comma == std::string::npos ? std::string::npos : comma - at)));
                            if (comma == std::string::npos) break;
                            at = comma + 1;
                        }
                    } else if (!text.empty() && text.front() != '%') {
                        callee = text;
                    }
                }
                const auto helper = helpers.find(callee);
                if (helper != helpers.end()) {
                    for (std::size_t k = 0; k < arguments.size(); ++k) {
                        // The parameter's value itself is the argument, and
                        // the helper dereferences it: a texture object or
                        // pointer the kernel only forwards (COLMAP passes its
                        // depth-map texture straight to an out-of-line
                        // ComputeGeomConsistencyCost).
                        if (const auto direct = staged_param.find(arguments[k]);
                            direct != staged_param.end() && dereferences_param(*helper->second, k))
                            pointers.insert(direct->second);
                        const auto base = staged.find(arguments[k]);
                        if (base == staged.end()) continue;
                        for (const auto field : pointer_fields(*helper->second, k)) {
                            const auto param = spilled.find(base->second + field);
                            if (param != spilled.end()) pointers.insert(param->second);
                        }
                    }
                }
                continue;
            }
            if (!dest.empty() && root != "st") {
                frame.erase(dest);
                param_value.erase(dest);
            }
        }
        for (const auto index : pointers)
            if (is_64bit_param_type(entry.params[index].type)) entry.params[index].is_pointer = true;
    }
}

// 1-based line number of a byte offset. Entries and functions are visited in
// source order, so the counter resumes from its last query instead of
// rescanning the module for each one (which was quadratic in module size).
class LineCounter {
public:
    explicit LineCounter(const std::string& text) : text_(text) {}

    int at(std::size_t offset) {
        const std::size_t clamped = std::min(offset, text_.size());
        if (clamped < offset_) {
            offset_ = 0;
            line_ = 1;
        }
        for (; offset_ < clamped; ++offset_) {
            if (text_[offset_] == '\n') ++line_;
        }
        return line_;
    }

private:
    const std::string& text_;
    std::size_t offset_ = 0;
    int line_ = 1;
};

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
    // Bare register names declared by `.reg` in this body, mapped to the
    // `%`-prefixed spelling the rest of the pipeline recognises.
    // Each entry remembers the brace depth of its declaration: PTX scopes such
    // names lexically, and a kernel routinely has both `{ .reg .b32 tmp; ... }`
    // and a module symbol called `tmp` (the dynamic shared array), so the
    // mapping must end with its block.
    struct BareRegister {
        std::string name;
        std::string renamed;
        int depth = 0;
    };
    std::vector<BareRegister> bare_registers;
    int scope_depth = 0;
    int pending_opens = 0;
    int pending_closes = 0;
    const auto close_scopes = [&](int count) {
        for (int i = 0; i < count; ++i) {
            if (scope_depth > 0) --scope_depth;
            bare_registers.erase(
                std::remove_if(bare_registers.begin(), bare_registers.end(),
                               [&](const BareRegister& r) { return r.depth > scope_depth; }),
                bare_registers.end());
        }
    };
    const auto record_register_directive = [&](std::string_view directive) {
        // `.reg .<type> a, b, %c;` -- ranges such as `%r<8>` already carry `%`.
        std::string text(directive);
        if (!starts_with(text, ".reg")) return;
        const std::size_t type_begin = text.find('.', 4);
        if (type_begin == std::string::npos) return;
        std::size_t type_end = type_begin + 1;
        while (type_end < text.size() &&
               (std::isalnum(static_cast<unsigned char>(text[type_end])) != 0 ||
                text[type_end] == '_')) {
            ++type_end;
        }
        const std::string type = text.substr(type_begin + 1, type_end - type_begin - 1);
        std::string names = text.substr(type_end);
        const std::size_t semi = names.find(';');
        if (semi != std::string::npos) names = names.substr(0, semi);
        std::size_t start = 0;
        while (start <= names.size()) {
            const std::size_t comma = names.find(',', start);
            const std::string name = trim(names.substr(
                start, comma == std::string::npos ? std::string::npos : comma - start));
            const auto open = name.find('<');
            if (!name.empty() && name[0] == '%' && open > 1 &&
                open != std::string::npos && name.back() == '>') {
                const auto digits = std::string_view(name).substr(open + 1, name.size() - open - 2);
                std::size_t count = 0;
                const auto parsed = std::from_chars(digits.data(), digits.data() + digits.size(), count);
                const auto prefix = name.substr(0, open);
                if (parsed.ec == std::errc{} && parsed.ptr == digits.data() + digits.size() &&
                    count != 0 && std::all_of(prefix.begin() + 1, prefix.end(), [](unsigned char c) {
                        return std::isalnum(c) || c == '_' || c == '$';
                    })) {
                    entry->register_ranges.push_back({prefix, type, count, scope_depth == 0});
                }
            }
            if (!name.empty() && name.find('<') == std::string::npos) {
                if (name[0] == '%') {
                    entry->register_declarations.push_back({name, type, scope_depth == 0});
                } else {
                    // The legacy backend types a register by its NVPTX name
                    // prefix (%p predicate, %rs 16-bit, %r/%f 32-bit, %rd/%fd
                    // 64-bit), so the synthetic name must carry the declared
                    // type's prefix. `%cm_` alone would be an unknown width.
                    const std::string prefix =
                        type == "pred"                                        ? "%p_cm_"
                        : (type == "b64" || type == "u64" || type == "s64")  ? "%rd_cm_"
                        : type == "f64"                                       ? "%fd_cm_"
                        : (type == "b16" || type == "u16" || type == "s16" ||
                           type == "f16" || type == "b8" || type == "u8" || type == "s8")
                                                                              ? "%rs_cm_"
                        : type == "f32"                                       ? "%f_cm_"
                                                                              : "%r_cm_";
                    const std::string renamed = prefix + name;
                    bare_registers.push_back({name, renamed, scope_depth});
                    entry->register_declarations.push_back({renamed, type, scope_depth == 0});
                }
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    };
    const auto rename_bare_registers = [&](const std::string& text) {
        if (bare_registers.empty()) return text;
        std::string out;
        for (std::size_t i = 0; i < text.size();) {
            bool replaced = false;
            const unsigned char before = i == 0 ? ' ' : static_cast<unsigned char>(text[i - 1]);
            const bool boundary_before = std::isalnum(before) == 0 && before != '_' &&
                                         before != '%' && before != '$' && before != '.';
            if (boundary_before) {
                for (const BareRegister& reg : bare_registers) {
                    const std::string& name = reg.name;
                    const std::string& renamed = reg.renamed;
                    if (text.compare(i, name.size(), name) != 0) continue;
                    const std::size_t after = i + name.size();
                    if (after < text.size() &&
                        (std::isalnum(static_cast<unsigned char>(text[after])) != 0 ||
                         text[after] == '_')) {
                        continue;
                    }
                    out += renamed;
                    i = after;
                    replaced = true;
                    break;
                }
            }
            if (!replaced) out += text[i++];
        }
        return out;
    };

    std::istringstream stream(body);
    std::string raw_line;
    std::string remainder;
    int remainder_line = 0;
    while (true) {
        int current_line;
        if (!remainder.empty()) {
            raw_line = std::move(remainder);
            remainder.clear();
            current_line = remainder_line;
        } else {
            if (!std::getline(stream, raw_line)) break;
            current_line = line++;
        }
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
        // Scope bookkeeping for bare register names: braces stripped from the
        // front of a line take effect before its content, braces stripped from
        // the back after it (i.e. before the next line).
        scope_depth += pending_opens;
        close_scopes(pending_closes);
        pending_opens = 0;
        pending_closes = 0;
        bool stripped_brace = false;
        do {
            stripped_brace = false;
            if (!line_text.empty() && (line_text.front() == '{' || line_text.front() == '}')) {
                if (line_text.front() == '{') ++scope_depth; else close_scopes(1);
                line_text = trim(line_text.substr(1));
                stripped_brace = true;
            }
            if (!line_text.empty() && (line_text.back() == '{' || line_text.back() == '}')) {
                if (line_text.back() == '{') ++pending_opens; else ++pending_closes;
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
            record_register_directive(line_text.substr(0, semi + 1));
            line_text = trim(line_text.substr(semi + 1));
        }
        if (line_text.empty()) {
            continue;
        }
        // A physical newline is whitespace inside a PTX call. Rust/LLVM emits
        // the return tuple, callee and argument tuple on separate lines. Assemble
        // through the semicolon before operand splitting or register renaming.
        std::string call_head = line_text;
        if (call_head.front() == '@') {
            const auto predicate_end = call_head.find_first_of(" \t");
            if (predicate_end != std::string::npos) {
                call_head = trim(call_head.substr(predicate_end + 1));
            }
        }
        const std::string candidate_opcode = call_head.substr(0, call_head.find_first_of(" \t"));
        std::string assembly_error;
        if (candidate_opcode == "call" || starts_with(candidate_opcode, "call.")) {
            int last_line = current_line;
            std::size_t semi = line_text.find(';');
            while (semi == std::string::npos && std::getline(stream, raw_line)) {
                last_line = line++;
                line_text += " " + trim(raw_line);
                semi = line_text.find(';');
            }
            if (semi == std::string::npos) {
                assembly_error = "unterminated PTX call at line " + std::to_string(current_line);
            } else {
                remainder = trim(line_text.substr(semi + 1));
                remainder_line = last_line;
                if (!remainder.empty()) {
                    // Braces stripped from the original physical line close
                    // only after its remaining instructions have been parsed.
                    remainder += std::string(pending_closes, '}');
                    pending_closes = 0;
                }
                line_text.resize(semi + 1);
                int parentheses = 0;
                for (char ch : line_text) {
                    if (ch == '(') ++parentheses;
                    if (ch == ')' && --parentheses < 0) break;
                }
                if (parentheses != 0) {
                    assembly_error = "unbalanced PTX call parentheses at line " + std::to_string(current_line);
                }
            }
        } else if (const std::size_t semi = line_text.find(';');
                   semi != std::string::npos && semi + 1 < line_text.size()) {
            // Several statements on one line -- CUB's inline-asm warp reductions
            // emit `{ .reg .f32 r0; shfl.sync.down.b32 r0|p, ...; @p add.f32 r0,
            // r0, %r1; mov.f32 %r2, r0;}`. Parsing the line as one instruction
            // folded the add and mov into the shuffle's operand list: the typed
            // backend then saw %r2 used before definition, and the legacy one
            // compiled the kernel without them. Parse the rest as its own line.
            std::string rest = trim(line_text.substr(semi + 1));
            if (!rest.empty()) {
                remainder = std::move(rest) + std::string(pending_closes, '}');
                remainder_line = current_line;
                pending_closes = 0;
                line_text.resize(semi + 1);
            }
        }
        line_text = rename_bare_registers(line_text);

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

        if (opcode_root(instruction.opcode) == "call" && assembly_error.empty()) {
            // Do not absorb a following instruction when a call's semicolon
            // is missing: top-level operands are single names or complete tuples.
            for (const auto& operand : instruction.operands) {
                if ((operand.front() == '(' && operand.back() != ')') ||
                    (operand.front() != '(' && operand.find_first_of(" \t") != std::string::npos)) {
                    assembly_error = "malformed PTX call operand at line " + std::to_string(current_line);
                    break;
                }
            }
            if (instruction.operands.empty()) {
                assembly_error = "missing PTX call target at line " + std::to_string(current_line);
            }
        }
        instruction.supported = assembly_error.empty() &&
                                !is_explicitly_unsupported(instruction.opcode) &&
                                is_supported_opcode(instruction.opcode);
        if (!instruction.supported) {
            const std::string targeted = targeted_unsupported_message(instruction.opcode);
            warnings->push_back(!assembly_error.empty()
                                     ? assembly_error
                                     : !targeted.empty() ? targeted
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

EntryFunction parse_instruction_block(std::string_view body, int start_line,
                                      std::vector<std::string>* warnings) {
    EntryFunction entry;
    entry.name = "cm_inline_asm";
    std::vector<std::string> local_warnings;
    parse_instructions(strip_comments(body), start_line, &entry,
                       warnings != nullptr ? warnings : &local_warnings);
    return entry;
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
        R"(\.entry\s+([A-Za-z_.$][A-Za-z0-9_.$]*)\s*\(([^;{}]*?)\)\s*((?:\.[^\n{};]*\s*)*)\{)");
    const std::regex maxntid_regex(R"(\.maxntid\s+([0-9]+)(?:\s*,\s*([0-9]+))?(?:\s*,\s*([0-9]+))?)");
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

    LineCounter line_counter(source);
    std::sregex_iterator iter(source.begin(), source.end(), entry_regex);
    const std::sregex_iterator end;

    for (; iter != end; ++iter) {
        if (iter->size() < 4) {
            continue;
        }

        EntryFunction entry;
        entry.name = (*iter)[1].str();

        // .maxntid X[,Y[,Z]] entry property (in the blob before the body).
        {
            const std::string properties = (*iter)[3].str();
            std::smatch maxntid_match;
            if (std::regex_search(properties, maxntid_match, maxntid_regex)) {
                std::uint64_t product = 1;
                bool valid = true;
                for (std::size_t axis = 1; axis < maxntid_match.size(); ++axis) {
                    if (!maxntid_match[axis].matched) continue;
                    int dim = 0;
                    if (!parse_number(maxntid_match[axis].str(), &dim) || dim <= 0) {
                        valid = false;
                        break;
                    }
                    product *= static_cast<std::uint64_t>(dim);
                }
                if (valid && product <= 1024) {
                    entry.max_threads_per_threadgroup =
                        static_cast<std::uint32_t>(product);
                }
            }
        }

        entry.params = parse_parameters((*iter)[2].str());

        const std::size_t open_brace = static_cast<std::size_t>(iter->position(0) + iter->length(0) - 1);
        std::string body;
        std::size_t body_start = 0;
        std::size_t body_end = 0;
        if (!extract_balanced_body(source, open_brace, &body, &body_start, &body_end)) {
            result.error = "malformed entry body for '" + entry.name + "'";
            return result;
        }
        const int start_line = line_counter.at(body_start);
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
        const int start_line = line_counter.at(body_start);
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

    infer_pointer_parameters_through_calls(&result.module);
    result.ok = true;
    return result;
}

}  // namespace cumetal::ptx
