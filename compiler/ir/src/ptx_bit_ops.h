#pragma once

#include "ptx_value_builder.h"

#include <string_view>

namespace cumetal::ir::detail {

// The caller validates opcode/operand forms and supplies the destination value.
// These helpers append typed temporary expressions and fill its final operation.
void lower_bit_permutation(PtxValueBuilder& values, Operation& operation,
                           std::string_view opcode, Operand a, Operand b, Operand count);
void lower_bit_insert(PtxValueBuilder& values, Operation& operation, const Type& type,
                      Operand a, Operand b, Operand position, Operand length);
// lop3.b32 with an immediate truth table over (a, b, c) = (0xf0, 0xcc, 0xaa).
void lower_logic3(PtxValueBuilder& values, Operation& operation,
                  const Operand inputs[3], std::uint8_t table);

}  // namespace cumetal::ir::detail
