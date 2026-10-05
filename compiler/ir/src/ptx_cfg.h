#pragma once

#include "cumetal/ir/ir.h"
#include "ptx_instruction.h"
#include <deque>
#include <map>
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

// Per-instruction scalar facts, never pointer type evidence. The lane index
// refers to the unchanged load's destination tuple; the value is its bit width.
using ScalarZeroLoads = std::unordered_map<const Instruction*, std::map<std::size_t, unsigned>>;

// Use independently proven local-load zero bits to remove impossible branch
// edges. Instructions other than those branches retain their values/order.
// The caller must rebuild SSA and type evidence when this returns nonzero.
std::size_t simplify_local_zero_guards(std::vector<RawBlock>& blocks, Builder& builder,
    std::deque<Instruction>& storage, const cumetal::ptx::EntryFunction& function,
    InstructionOrigins* origins, const ScalarZeroLoads& zero_loads);

// Called before SSA allocation. Storage owns rewritten instructions until the
// importer finishes materializing the function; deque preserves their addresses.
void simplify_guarded_paths(std::vector<RawBlock>& blocks, Builder& builder,
                            std::deque<Instruction>& storage,
                            const cumetal::ptx::EntryFunction* function = nullptr,
                            InstructionOrigins* origins = nullptr,
                            const std::unordered_set<std::string>* synchronizing_callees = nullptr);

// Give every predecessor group after the first its own copy of `block`, so a
// register the groups feed with pointers in different address spaces no longer
// joins. Refuses (returns false) for a loop header, which would gain a second
// entry, and for a block that synchronizes, whose copies Metal would pair.
// Refreshes predecessors; the caller must rebuild SSA.
bool split_join_block(std::vector<RawBlock>& blocks, Builder& builder,
                      std::deque<Instruction>& storage, InstructionOrigins* origins,
                      const std::unordered_set<std::string>& synchronizing_callees,
                      std::size_t block, const std::vector<std::vector<std::size_t>>& groups);

// Turn `selp d, a, b, p` at `index` of `block` into `@p bra A; B: mov d, b`
// / `A: mov d, a`, both continuing into a new block with the rest of the
// original. A pointer select whose operands live in different address spaces
// thereby becomes a join split_join_block can take apart. Refreshes
// predecessors; the caller must rebuild SSA.
bool split_pointer_select(std::vector<RawBlock>& blocks, Builder& builder,
                          std::deque<Instruction>& storage, InstructionOrigins* origins,
                          std::size_t block, std::size_t index);

// Device functions that reach a barrier or warp/CTA collective, directly or
// through calls. Edge threading must not clone calls into them.
std::unordered_set<std::string> synchronizing_functions(
    const std::unordered_map<std::string, const cumetal::ptx::EntryFunction*>& functions);

}  // namespace cumetal::ir::detail
