#pragma once
#include "ptx_cfg.h"

namespace cumetal::ir::detail {

// Private proof limits; tests can exercise exhaustion without enormous inputs.
struct TupleNormalizationLimits {
    std::size_t instruction_visits = 4194304;
    std::size_t register_occurrences = 16777216;
    std::size_t text_bytes = 67108864;
    std::size_t tracked_registers = 1048576;
    std::size_t candidates = 65536;
    std::size_t declaration_checks = 4194304;
    std::size_t declaration_entries = 65536;
};

// Pre-SSA: discard only an unobserved lane of a single-use, same-block pack
// followed by an extraction or exact unsigned 16/32-bit narrowing conversion.
// Declarations establish storage widths; no register spelling implies a type.
// Storage owns replacements. Exhausted proofs leave the function unchanged.
void remove_discarded_pack_halves(std::vector<RawBlock>& blocks, std::deque<Instruction>& storage,
                                  const cumetal::ptx::EntryFunction* function,
                                  InstructionOrigins* origins = nullptr,
                                  TupleNormalizationLimits limits = {});

} // namespace cumetal::ir::detail
