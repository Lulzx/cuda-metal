#pragma once

// Inline PTX assembly in CUDA source, lowered by the PTX instruction importer.
//
// Clang hands the NVVM importer an `asm` call: a template with `$N`
// placeholders, a constraint string, and LLVM operands. The PTX frontend
// already lowers every instruction such templates contain, so instead of a
// second implementation the NVVM importer binds the operands to synthetic
// registers and runs the template through that lowering in place.

#include "cumetal/ir/ir.h"

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cumetal::ir::detail {

struct InlineAsmBinding {
    // PTX register type implied by the constraint code ("r", "h", "l", "f", "d", ...).
    Type type = Type::integer(32);
    bool is_output = false;
    // Inputs: the LLVM operand, or an immediate spelled the way PTX reads it.
    std::optional<Operand> input;
    bool is_immediate = false;
    std::string immediate_text;
    // An input tied to an output ("0" constraint): the output register starts
    // with the input's value.
    std::optional<std::size_t> tied_output;
};

struct InlineAsmRequest {
    std::string text;
    // Indexed by `$N`: outputs first, then inputs, as LLVM numbers them.
    std::vector<InlineAsmBinding> bindings;
    std::string fp64_mode;
    std::string source_name;
    std::uint32_t line = 0;
};

struct InlineAsmResult {
    bool ok = false;
    std::string error;
    // One operand per output binding, in binding order.
    std::vector<Operand> outputs;
    std::vector<std::string> caveats;
};

// Appends the lowered operations to `block`. New SSA values are allocated from
// `builder` and typed in `value_types`, exactly as the caller's own values.
InlineAsmResult lower_inline_ptx_asm(const InlineAsmRequest& request, Builder* builder,
                                     std::unordered_map<ValueId, Type>* value_types,
                                     Function* function, BasicBlock* block);

}  // namespace cumetal::ir::detail
