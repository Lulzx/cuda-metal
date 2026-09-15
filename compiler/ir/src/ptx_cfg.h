#pragma once

#include "cumetal/ir/ir.h"
#include "ptx_instruction.h"
#include <deque>
#include <unordered_set>

namespace cumetal::ir::detail {

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
                            const cumetal::ptx::EntryFunction* function = nullptr);

}  // namespace cumetal::ir::detail
