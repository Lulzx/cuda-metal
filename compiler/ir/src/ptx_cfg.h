#pragma once

#include "cumetal/ir/ir.h"
#include "ptx_instruction.h"
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::ir::detail {

// Normalized instructions retain their original source identity, including
// across repeated copies. Origin alone does not preserve a result type: a
// rewrite may change the opcode, operands, or result layout.
using InstructionOrigins = std::unordered_map<const Instruction*, const Instruction*>;

inline void record_instruction_origin(InstructionOrigins* origins,
                                      const Instruction* instruction,
                                      const Instruction* source) {
    if (origins == nullptr) return;
    for (;;) {
        const auto previous = origins->find(source);
        if (previous == origins->end()) break;
        source = previous->second;
    }
    origins->emplace(instruction, source);
}

struct RawBlock {
    BlockId id = kInvalidBlock;
    std::string name;
    std::vector<const Instruction*> instructions;
    std::vector<std::size_t> successors;
    std::vector<std::size_t> predecessors;
    std::unordered_map<std::string, ValueId> last_definitions;
    std::unordered_set<std::string> uses_before_definition;
};

void remove_unreachable_blocks(std::vector<RawBlock>& blocks);

// Called before SSA allocation. Storage owns rewritten instructions until the
// importer finishes materializing the function; deque preserves their addresses.
void simplify_guarded_paths(std::vector<RawBlock>& blocks, Builder& builder,
                            std::deque<Instruction>& storage,
                            const cumetal::ptx::EntryFunction* function = nullptr,
                            InstructionOrigins* origins = nullptr);

}  // namespace cumetal::ir::detail
