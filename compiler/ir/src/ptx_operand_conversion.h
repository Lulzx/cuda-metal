#pragma once

#include "cumetal/ir/ir.h"

#include <unordered_map>
#include <utility>

namespace cumetal::ir::detail {

// PTX register containers, instruction operands, and memory accesses can have
// different widths. These conversions establish an operand's bits; signedness
// and load extension remain properties of the consuming operation.
class PtxOperandConversion {
public:
    PtxOperandConversion(Builder& builder, BasicBlock& block,
                         std::unordered_map<ValueId, Type>& types,
                         SourceLocation location)
        : builder_(builder), block_(block), types_(types), location_(std::move(location)) {}

    // Reinterpret equal-width float/integer storage, never a numeric conversion.
    Operand bit_container(Operand input, const Type& expected) {
        const bool pair = (input.type.kind == TypeKind::kFloat && expected.kind == TypeKind::kInteger) ||
                          (input.type.kind == TypeKind::kInteger && expected.kind == TypeKind::kFloat);
        if (!pair || input.type.bit_width != expected.bit_width) return input;
        return emit(std::move(input), expected, true);
    }

    // Discard high register bits only when the instruction consumes fewer bits.
    // In particular, narrowing before a signed cvt must not sign-extend yet.
    Operand low_integer_bits(Operand input, const Type& instruction_type) {
        if (input.type.kind != TypeKind::kInteger || instruction_type.kind != TypeKind::kInteger ||
            input.type.bit_width <= instruction_type.bit_width) return input;
        return emit(std::move(input), instruction_type, false);
    }

private:
    Operand emit(Operand input, const Type& type, bool bitcast) {
        const ValueId value = builder_.next_value();
        Operation conversion;
        conversion.opcode = OpCode::kConvert;
        conversion.location = location_;
        conversion.operands = {std::move(input)};
        conversion.results = {value};
        conversion.result_types = {type};
        if (bitcast) conversion.attributes["bitcast"] = "true";
        types_[value] = type;
        block_.operations.push_back(std::move(conversion));
        return Operand::value_ref(value, type);
    }

    Builder& builder_;
    BasicBlock& block_;
    std::unordered_map<ValueId, Type>& types_;
    SourceLocation location_;
};

}  // namespace cumetal::ir::detail
