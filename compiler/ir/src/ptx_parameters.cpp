#include "ptx_parameters.h"
#include "ptx_instruction.h"
#include "ptx_text.h"

#include <algorithm>
#include <cctype>
#include <limits>

namespace cumetal::ir::detail {

// Parameter slots require a literal byte offset; never interpret malformed
// offsets as zero (the permissive address helper is used by other paths).
std::optional<std::int64_t> parameter_slot_offset(std::string operand, const std::string& name) {
    operand.erase(std::remove_if(operand.begin(), operand.end(),
        [](unsigned char c) { return std::isspace(c); }), operand.end());
    const std::string prefix = "[" + name;
    if (!starts_with(operand, prefix) || operand.back() != ']') return std::nullopt;
    const std::string suffix = operand.substr(prefix.size(), operand.size() - prefix.size() - 1);
    if (suffix.empty()) return 0;
    if (suffix.front() != '+' && suffix.front() != '-') return std::nullopt;
    try {
        std::size_t consumed = 0;
        const auto offset = std::stoll(suffix, &consumed, 0);
        if (consumed == suffix.size()) return offset;
    } catch (...) {}
    return std::nullopt;
}

// Parameter slots are compiler-managed byte storage, not observable memory.
// Expand exact, aligned v2/v4.b32 and v2.b64 transfers before SSA so all lanes participate
// in normal definition tracking and the existing aggregate ABI checks.
// Register-addressed loads use the same scalar parameter-pointer provenance
// and bounds checks as their non-vector counterparts.
void normalize_vector_parameter_transfers(cumetal::ptx::EntryFunction* function) {
    std::vector<Instruction> rewritten;
    for (const auto& instruction : function->instructions) {
        const bool load = starts_with(instruction.opcode, "ld.param.");
        const std::string suffix = instruction.opcode.substr(std::min(instruction.opcode.size(), std::size_t{9}));
        const bool supported_vector = suffix == "v2.b64" || suffix == "v2.b32" || suffix == "v4.b32";
        const std::size_t width = suffix == "v4.b32" ? 4 : 2;
        const unsigned bits = suffix == "v2.b64" ? 64 : 32;
        const unsigned bytes = bits / 8;
        const unsigned vector_bytes = bytes * width;
        if ((!load && !starts_with(instruction.opcode, "st.param.")) || !supported_vector ||
            !instruction.predicate.empty() || instruction.operands.size() != 2) {
            rewritten.push_back(instruction);
            continue;
        }
        const auto& address = instruction.operands[load ? 1 : 0];
        const std::string name = parameter_name_from_operand(address);
        const auto offset = parameter_slot_offset(address, name);
        const std::string tuple = trim(instruction.operands[load ? 0 : 1]);
        if (name.empty() || !offset || *offset < 0 || *offset % vector_bytes != 0 ||
            *offset > std::numeric_limits<std::int64_t>::max() - bytes * (width - 1) ||
            tuple.size() < 2 || tuple.front() != '{' || tuple.back() != '}') {
            rewritten.push_back(instruction);
            continue;
        }
        const auto lanes = grouped_names(tuple.substr(1, tuple.size() - 2));
        bool valid = lanes.size() == width;
        if (load) for (std::size_t i = 0; i < lanes.size(); ++i)
            for (std::size_t j = 0; j < i; ++j) valid &= lanes[i] != lanes[j];
        for (const auto& lane : lanes) {
            if (first_register(lane) == lane && ptx_register_container_bits(lane) == bits) continue;
            if (load) { valid = false; break; }
            try {
                std::size_t consumed = 0;
                (void)std::stoll(lane, &consumed, 0);
                valid &= consumed == lane.size();
            } catch (...) { valid = false; }
        }
        if (!valid) { rewritten.push_back(instruction); continue; }
        for (std::size_t i = 0; i < width; ++i) {
            Instruction scalar = instruction;
            scalar.opcode = (load ? "ld.param.b" : "st.param.b") + std::to_string(bits);
            const std::string slot = "[" + name + "+" + std::to_string(*offset + bytes*i) + "]";
            scalar.operands = load ? std::vector<std::string>{lanes[i], slot} :
                                     std::vector<std::string>{slot, lanes[i]};
            rewritten.push_back(std::move(scalar));
        }
    }
    function->instructions = std::move(rewritten);
}

}  // namespace cumetal::ir::detail
