#include "ptx_bit_ops.h"

namespace cumetal::ir::detail {

void lower_bit_permutation(PtxValueBuilder& values, Operation& operation,
                           std::string_view opcode, Operand a, Operand b, Operand count) {
    // Concatenate [b:a] in 64 bits so a wrapped zero shift is always defined.
    const bool left = opcode == "shf.l.wrap.b32";
    const bool permute = opcode == "prmt.b32";
    const Type u32 = Type::integer(32), u64 = Type::integer(64);
    const Operand low = values.emit(OpCode::kConvert, u64, {a});
    const Operand high = values.emit(OpCode::kConvert, u64, {b});
    const Operand high_bits = values.emit(OpCode::kShiftLeft, u64,
        {high, Operand::immediate("32", u64)});
    const Operand packed = values.emit(OpCode::kBitOr, u64, {high_bits, low});
    Operand shifted;
    if (permute) {
        const Operand selector = values.emit(OpCode::kConvert, u64, {count});
        const auto imm = [&](unsigned n) { return Operand::immediate(std::to_string(n), u64); };
        shifted = imm(0);
        for (unsigned lane = 0; lane < 4; ++lane) {
            const Operand nibble = values.emit(OpCode::kShiftRight, u64, {selector, imm(lane * 4)});
            const Operand index = values.emit(OpCode::kBitAnd, u64, {nibble, imm(7)});
            const Operand distance = values.emit(OpCode::kMul, u64, {index, imm(8)});
            const Operand source = values.emit(OpCode::kShiftRight, u64, {packed, distance});
            const Operand byte = values.emit(OpCode::kBitAnd, u64, {source, imm(255)});
            const Operand sign = values.emit(OpCode::kShiftRight, u64, {byte, imm(7)});
            const Operand replicated = values.emit(OpCode::kMul, u64, {sign, imm(255)});
            const Operand flag_bits = values.emit(OpCode::kShiftRight, u64, {nibble, imm(3)});
            const Operand flag = values.emit(OpCode::kBitAnd, u64, {flag_bits, imm(1)});
            const Operand mask = values.emit(OpCode::kSub, u64, {imm(0), flag});
            const Operand difference = values.emit(OpCode::kBitXor, u64, {byte, replicated});
            const Operand selected_difference = values.emit(OpCode::kBitAnd, u64, {difference, mask});
            const Operand selected = values.emit(OpCode::kBitXor, u64, {byte, selected_difference});
            const Operand positioned = values.emit(OpCode::kShiftLeft, u64, {selected, imm(lane * 8)});
            shifted = values.emit(OpCode::kBitOr, u64, {shifted, positioned});
        }
    } else {
        const Operand masked = values.emit(OpCode::kBitAnd, u32,
            {count, Operand::immediate("31", u32)});
        const Operand shift = values.emit(OpCode::kConvert, u64, {masked});
        shifted = values.emit(left ? OpCode::kShiftLeft : OpCode::kShiftRight,
                       u64, {packed, shift});
        if (left) {
            shifted = values.emit(OpCode::kShiftRight, u64,
                {shifted, Operand::immediate("32", u64)});
        }
    }
    operation.opcode = OpCode::kConvert;
    operation.result_types = {u32};
    operation.operands = {shifted};
}

void lower_bit_insert(PtxValueBuilder& values, Operation& operation, const Type& type,
                      Operand a, Operand b, Operand position, Operand length) {
    const Type u32 = Type::integer(32);
    const auto imm32 = [&](unsigned n) {
        return Operand::immediate(std::to_string(n), u32);
    };
    const Operand all = Operand::immediate(
        type.bit_width == 64 ? "18446744073709551615" : "4294967295", type);
    const Operand pos = values.emit(OpCode::kBitAnd, u32, {position, imm32(255)});
    const Operand len = values.emit(OpCode::kBitAnd, u32, {length, imm32(255)});
    // Even discarded select arms must avoid an undefined full-width
    // shift. Clamp shift counts by masking, then select the PTX result.
    const Operand safe_pos = values.emit(OpCode::kBitAnd, u32, {pos, imm32(type.bit_width - 1)});
    const Operand safe_len = values.emit(OpCode::kBitAnd, u32, {len, imm32(type.bit_width - 1)});
    const Operand shifted_ones = values.emit(OpCode::kShiftLeft, type, {all, safe_len});
    const Operand short_mask = values.emit(OpCode::kBitXor, type, {shifted_ones, all});
    const Operand full_length = values.emit(OpCode::kCompare, Type::predicate(),
        {len, imm32(type.bit_width)}, {{"predicate", "ge"}});
    const Operand low_mask = values.emit(OpCode::kSelect, type, {full_length, all, short_mask});
    const Operand mask = values.emit(OpCode::kShiftLeft, type, {low_mask, safe_pos});
    const Operand inverse = values.emit(OpCode::kBitXor, type, {mask, all});
    const Operand retained = values.emit(OpCode::kBitAnd, type, {b, inverse});
    const Operand shifted_a = values.emit(OpCode::kShiftLeft, type, {a, safe_pos});
    const Operand inserted = values.emit(OpCode::kBitAnd, type, {shifted_a, mask});
    const Operand merged = values.emit(OpCode::kBitOr, type, {retained, inserted});
    const Operand in_range = values.emit(OpCode::kCompare, Type::predicate(),
        {pos, imm32(type.bit_width)}, {{"predicate", "lt"}});
    operation.opcode = OpCode::kSelect;
    operation.result_types = {type};
    operation.operands = {in_range, merged, b};
}

}  // namespace cumetal::ir::detail
