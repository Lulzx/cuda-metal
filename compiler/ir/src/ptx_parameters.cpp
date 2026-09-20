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
// Expand exact, aligned v2.b64 transfers before SSA so both lanes participate
// in normal definition tracking and the existing aggregate ABI checks.
void normalize_vector_parameter_transfers(cumetal::ptx::EntryFunction* function) {
    std::vector<Instruction> rewritten;
    for (const auto& instruction : function->instructions) {
        const bool load = instruction.opcode == "ld.param.v2.b64";
        if ((!load && instruction.opcode != "st.param.v2.b64") ||
            !instruction.predicate.empty() || instruction.operands.size() != 2) {
            rewritten.push_back(instruction);
            continue;
        }
        const auto& address = instruction.operands[load ? 1 : 0];
        const std::string name = parameter_name_from_operand(address);
        const auto offset = parameter_slot_offset(address, name);
        const std::string tuple = trim(instruction.operands[load ? 0 : 1]);
        if (name.empty() || !registers_in(name).empty() || !offset || *offset < 0 || *offset % 16 != 0 ||
            *offset > std::numeric_limits<std::int64_t>::max() - 8 ||
            tuple.size() < 2 || tuple.front() != '{' || tuple.back() != '}') {
            rewritten.push_back(instruction);
            continue;
        }
        const auto lanes = grouped_names(tuple.substr(1, tuple.size() - 2));
        bool valid = lanes.size() == 2 && (!load || lanes[0] != lanes[1]);
        for (const auto& lane : lanes) {
            if (first_register(lane) == lane && ptx_register_container_bits(lane) == 64) continue;
            if (load) { valid = false; break; }
            try {
                std::size_t consumed = 0;
                (void)std::stoll(lane, &consumed, 0);
                valid &= consumed == lane.size();
            } catch (...) { valid = false; }
        }
        if (!valid) { rewritten.push_back(instruction); continue; }
        for (int i = 0; i < 2; ++i) {
            Instruction scalar = instruction;
            scalar.opcode = load ? "ld.param.b64" : "st.param.b64";
            const std::string slot = "[" + name + "+" + std::to_string(*offset + 8*i) + "]";
            scalar.operands = load ? std::vector<std::string>{lanes[i], slot} :
                                     std::vector<std::string>{slot, lanes[i]};
            rewritten.push_back(std::move(scalar));
        }
    }
    function->instructions = std::move(rewritten);
}

}  // namespace cumetal::ir::detail
