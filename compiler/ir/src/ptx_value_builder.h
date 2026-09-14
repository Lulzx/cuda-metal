#pragma once

#include "cumetal/ir/ir.h"

#include <unordered_map>
#include <utility>

namespace cumetal::ir::detail {

// PTX register containers, instruction operands, and memory accesses can have
// different widths. These conversions establish an operand's bits; signedness
// and load extension remain properties of the consuming operation.
class PtxValueBuilder {
public:
    PtxValueBuilder(Builder& builder, BasicBlock& block,
                    std::unordered_map<ValueId, Type>& types, SourceLocation location)
        : builder_(builder), block_(block), types_(types), location_(std::move(location)) {}

    Operand emit(OpCode opcode, const Type& type, std::vector<Operand> inputs,
                 std::unordered_map<std::string, std::string> attributes = {}) {
        const ValueId value = builder_.next_value();
        Operation operation;
        operation.opcode = opcode;
        operation.location = location_;
        operation.operands = std::move(inputs);
        operation.results = {value};
        operation.result_types = {type};
        operation.attributes = std::move(attributes);
        types_[value] = type;
        block_.operations.push_back(std::move(operation));
        return Operand::value_ref(value, type);
    }

    // Reinterpret equal-width float/integer storage, never a numeric conversion.
    Operand bit_container(Operand input, const Type& expected) {
        const bool pair = (input.type.kind == TypeKind::kFloat && expected.kind == TypeKind::kInteger) ||
                          (input.type.kind == TypeKind::kInteger && expected.kind == TypeKind::kFloat);
        if (!pair || input.type.bit_width != expected.bit_width) return input;
        return emit(OpCode::kConvert, expected, {std::move(input)}, {{"bitcast", "true"}});
    }

    // Discard high register bits only when the instruction consumes fewer bits.
    // In particular, narrowing before a signed cvt must not sign-extend yet.
    Operand low_integer_bits(Operand input, const Type& instruction_type) {
        if (input.type.kind != TypeKind::kInteger || instruction_type.kind != TypeKind::kInteger ||
            input.type.bit_width <= instruction_type.bit_width) return input;
        return emit(OpCode::kConvert, instruction_type, {std::move(input)});
    }

private:
    Builder& builder_;
    BasicBlock& block_;
    std::unordered_map<ValueId, Type>& types_;
    SourceLocation location_;
};

}  // namespace cumetal::ir::detail
