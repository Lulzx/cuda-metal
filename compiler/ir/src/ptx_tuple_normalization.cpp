#include "ptx_tuple_normalization.h"
#include "ptx_text.h"
#include <algorithm>

namespace cumetal::ir::detail {

// A scalar pack used only by a same-block, single-lane extraction need
// not read its discarded lane. Restrict this to one definition/use across
// the function and an unchanged selected source; otherwise keep normal SSA
// validation. No undefined bits are materialized or initialized.
void remove_discarded_pack_halves(std::vector<RawBlock>& raw_blocks,
                                  std::deque<Instruction>& storage,
                                  InstructionOrigins* origins) {
    const auto scalar32 = [](const std::string& operand) {
        return !operand.empty() && first_register(operand) == operand &&
               ptx_register_container_bits(operand) == 32;
    };
    const auto tuple = [](const std::string& operand) -> std::vector<std::string> {
        const auto text = trim(operand);
        if (text.size() < 5 || text.front() != '{' || text.back() != '}') return {};
        const auto comma = text.find(',');
        if (comma == std::string::npos || text.find(',', comma + 1) != std::string::npos) return {};
        return {trim(text.substr(1, comma - 1)), trim(text.substr(comma + 1, text.size() - comma - 2))};
    };
    std::unordered_map<std::string, std::size_t> definitions, uses;
    std::unordered_map<std::string, const Instruction*> consumer;
    for (const auto& block : raw_blocks)
        for (const auto* instruction : block.instructions) {
            for (const auto& reg : destination_registers(*instruction)) ++definitions[reg];
            for (const auto& reg : source_registers(*instruction)) {
                ++uses[reg];
                consumer[reg] = instruction;
            }
        }
    for (auto& block : raw_blocks) {
        for (std::size_t i = 0; i < block.instructions.size(); ++i) {
            const auto* pack = block.instructions[i];
            if (pack->opcode != "mov.b64" || !pack->predicate.empty() || pack->operands.size() != 2) continue;
            const auto packed = trim(pack->operands[0]);
            const auto halves = tuple(pack->operands[1]);
            if (packed.empty() || first_register(packed) != packed ||
                ptx_register_container_bits(packed) != 64 || halves.size() != 2 ||
                !scalar32(halves[0]) || !scalar32(halves[1])) continue;
            if (definitions[packed] != 1 || uses[packed] != 1) continue;
            const auto* extract = consumer[packed];
            if (!extract || extract->opcode != "mov.b64" ||
                !extract->predicate.empty() || extract->operands.size() != 2 ||
                trim(extract->operands[1]) != packed) continue;
            const auto lanes = tuple(extract->operands[0]);
            if (lanes.size() != 2) continue;
            const auto unobserved = [&](const std::string& lane) {
                return lane == "_" || (scalar32(lane) && uses[lane] == 0);
            };
            const bool discard_low = unobserved(lanes[0]);
            const bool discard_high = unobserved(lanes[1]);
            const int selected = discard_low != discard_high ? (discard_low ? 1 : 0) : -1;
            if (selected < 0 || !scalar32(lanes[selected])) continue;
            auto end = std::find(block.instructions.begin() + i + 1, block.instructions.end(), extract);
            if (end == block.instructions.end()) continue;
            bool stable = true;
            for (auto it = block.instructions.begin() + i + 1; it != end; ++it) {
                const auto written = destination_registers(**it);
                if (root_opcode((*it)->opcode) == "call" ||
                    std::find(written.begin(), written.end(), halves[selected]) != written.end()) stable = false;
            }
            if (!stable) continue;
            Instruction replacement = *extract;
            replacement.opcode = "mov.b32";
            replacement.operands = {lanes[selected], halves[selected]};
            storage.push_back(std::move(replacement));
            record_instruction_origin(origins, &storage.back(), extract);
            *end = &storage.back();
            block.instructions.erase(block.instructions.begin() + i);
            --i;
        }
    }
}

}  // namespace cumetal::ir::detail
