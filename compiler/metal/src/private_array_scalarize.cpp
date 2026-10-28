#include "private_array_scalarize.h"

#include <algorithm>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cumetal::metal::detail {
namespace {

bool parse_immediate_i64(const ir::Operand& operand, std::int64_t* out) {
    if (operand.kind != ir::OperandKind::kImmediate) return false;
    try {
        std::size_t consumed = 0;
        const long long value = std::stoll(operand.text, &consumed, 0);
        if (consumed != operand.text.size()) return false;
        *out = static_cast<std::int64_t>(value);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::optional<std::uint64_t> scalar_byte_size(const ir::Type& type) {
    if (type.kind != ir::TypeKind::kInteger && type.kind != ir::TypeKind::kFloat)
        return std::nullopt;
    if (type.bit_width != 8 && type.bit_width != 16 && type.bit_width != 32 &&
        type.bit_width != 64)
        return std::nullopt;
    return type.bit_width / 8;
}

// Byte offset expressed as const_bytes + index * index_scale.
struct OffsetDesc {
    std::int64_t const_bytes = 0;
    std::optional<ir::ValueId> index;
    std::int64_t index_scale = 1;
};

struct Access {
    std::size_t block = 0;
    std::size_t operation = 0;
    bool is_load = false;
    OffsetDesc desc;
};

struct Plan {
    ir::ValueId alloca_value = ir::kInvalidValue;
    ir::Type element_type;
    std::uint64_t element_bytes = 0;
    std::uint64_t count = 0;
    std::unordered_map<ir::ValueId, OffsetDesc> derived;
    std::vector<Access> accesses;
    bool has_dynamic = false;
};

class Analyzer {
  public:
    explicit Analyzer(const ir::Function& function) : function_(function) {
        for (const ir::FunctionArgument& argument : function.arguments)
            value_types_[argument.value] = argument.type;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments)
                value_types_[argument.value] = argument.type;
            for (std::size_t i = 0; i < block.operations.size(); ++i) {
                const ir::Operation& operation = block.operations[i];
                for (std::size_t r = 0; r < operation.results.size(); ++r) {
                    definers_[operation.results[r]] = {&block, i};
                    if (r < operation.result_types.size())
                        value_types_[operation.results[r]] = operation.result_types[r];
                }
            }
        }
    }

    // Immediate value of an operand, folding through constants and integer
    // converts defined anywhere in the function.
    std::optional<std::int64_t> fold_immediate(const ir::Operand& operand,
                                               int depth = 6) const {
        std::int64_t immediate = 0;
        if (parse_immediate_i64(operand, &immediate)) return immediate;
        if (operand.kind != ir::OperandKind::kValue || depth <= 0)
            return std::nullopt;
        const auto found = definers_.find(operand.value);
        if (found == definers_.end()) return std::nullopt;
        const ir::Operation& definition =
            found->second.first->operations[found->second.second];
        if (definition.operands.size() != 1) return std::nullopt;
        if (definition.opcode == ir::OpCode::kConstant ||
            definition.opcode == ir::OpCode::kConvert)
            return fold_immediate(definition.operands.front(), depth - 1);
        return std::nullopt;
    }

    // Decompose a byte-offset operand into const + index * scale.
    std::optional<OffsetDesc> decompose_offset(const ir::Operand& operand,
                                               int depth = 6) const {
        if (const auto immediate = fold_immediate(operand))
            return OffsetDesc{.const_bytes = *immediate};
        if (operand.kind != ir::OperandKind::kValue || depth <= 0)
            return std::nullopt;
        const auto found = definers_.find(operand.value);
        if (found == definers_.end()) return std::nullopt;
        const ir::Operation& definition =
            found->second.first->operations[found->second.second];
        if (definition.opcode == ir::OpCode::kConvert &&
            definition.operands.size() == 1)
            return decompose_offset(definition.operands.front(), depth - 1);
        if (definition.opcode == ir::OpCode::kMul &&
            definition.operands.size() == 2) {
            for (const int side : {0, 1}) {
                const auto scale =
                    fold_immediate(definition.operands[1 - side]);
                if (scale && *scale > 0 &&
                    definition.operands[side].kind == ir::OperandKind::kValue)
                    return OffsetDesc{.const_bytes = 0,
                                      .index = definition.operands[side].value,
                                      .index_scale = *scale};
            }
            return std::nullopt;
        }
        if (definition.opcode == ir::OpCode::kAdd &&
            definition.operands.size() == 2) {
            const auto lhs = decompose_offset(definition.operands[0], depth - 1);
            const auto rhs = decompose_offset(definition.operands[1], depth - 1);
            if (lhs && rhs) return combine(*lhs, *rhs);
            return std::nullopt;
        }
        return std::nullopt;
    }

    static std::optional<OffsetDesc> combine(const OffsetDesc& lhs,
                                             const OffsetDesc& rhs) {
        OffsetDesc result;
        result.const_bytes = lhs.const_bytes + rhs.const_bytes;
        if (lhs.index && rhs.index) {
            if (*lhs.index != *rhs.index || lhs.index_scale != rhs.index_scale)
                return std::nullopt;
            result.index = lhs.index;
            result.index_scale = lhs.index_scale;
        } else if (lhs.index) {
            result = lhs;
            result.const_bytes += rhs.const_bytes;
        } else if (rhs.index) {
            result = rhs;
            result.const_bytes += lhs.const_bytes;
        }
        return result;
    }

    std::optional<ir::Type> type_of(ir::ValueId value) const {
        const auto found = value_types_.find(value);
        if (found == value_types_.end()) return std::nullopt;
        return found->second;
    }

    std::optional<Plan> analyze(ir::ValueId alloca_value,
                                std::uint64_t byte_size) const {
        Plan plan;
        plan.alloca_value = alloca_value;
        plan.derived[alloca_value] = OffsetDesc{};

        // Propagate offset descriptors through pointer arithmetic chains.
        bool changed = true;
        while (changed) {
            changed = false;
            for (const ir::BasicBlock& block : function_.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.results.empty()) continue;
                    const ir::ValueId result = operation.results.front();
                    std::optional<OffsetDesc> desc;
                    if (operation.opcode == ir::OpCode::kPointerOffset &&
                        operation.operands.size() == 2 &&
                        operation.operands.front().kind ==
                            ir::OperandKind::kValue) {
                        const auto base =
                            plan.derived.find(operation.operands.front().value);
                        if (base != plan.derived.end()) {
                            const auto offset =
                                decompose_offset(operation.operands[1]);
                            if (offset) desc = combine(base->second, *offset);
                        }
                    } else if ((operation.opcode == ir::OpCode::kConvert ||
                                operation.opcode == ir::OpCode::kAddressSpaceCast) &&
                               operation.operands.size() == 1 &&
                               operation.operands.front().kind ==
                                   ir::OperandKind::kValue &&
                               !operation.result_types.empty() &&
                               operation.result_types.front().is_pointer() &&
                               operation.result_types.front().address_space ==
                                   ir::AddressSpace::kPrivate) {
                        const auto base =
                            plan.derived.find(operation.operands.front().value);
                        if (base != plan.derived.end()) desc = base->second;
                    }
                    if (!desc) continue;
                    const auto existing = plan.derived.find(result);
                    if (existing == plan.derived.end()) {
                        plan.derived[result] = *desc;
                        changed = true;
                    } else if (existing->second.const_bytes != desc->const_bytes ||
                               existing->second.index != desc->index ||
                               existing->second.index_scale != desc->index_scale) {
                        return std::nullopt;
                    }
                }
            }
        }

        // Audit every use of a derived pointer and collect accesses.
        std::optional<ir::Type> element_type;
        for (std::size_t b = 0; b < function_.blocks.size(); ++b) {
            const ir::BasicBlock& block = function_.blocks[b];
            for (std::size_t i = 0; i < block.operations.size(); ++i) {
                const ir::Operation& operation = block.operations[i];
                for (const ir::Successor& successor : operation.successors)
                    for (const ir::ValueId argument : successor.arguments)
                        if (plan.derived.contains(argument))
                            return std::nullopt; // crosses a block argument
                for (const ir::Operand& operand : operation.operands) {
                    if (operand.kind != ir::OperandKind::kValue ||
                        !plan.derived.contains(operand.value))
                        continue;
                    const bool chain =
                        (operation.opcode == ir::OpCode::kPointerOffset ||
                         operation.opcode == ir::OpCode::kConvert ||
                         operation.opcode == ir::OpCode::kAddressSpaceCast) &&
                        !operation.operands.empty() &&
                        operand.value == operation.operands.front().value &&
                        !operation.results.empty() &&
                        plan.derived.contains(operation.results.front());
                    if (chain) continue;
                    const bool memory = operation.opcode == ir::OpCode::kLoad ||
                                        operation.opcode == ir::OpCode::kStore;
                    if (!memory || operation.operands.empty() ||
                        operand.value != operation.operands.front().value)
                        return std::nullopt;
                    if (operation.attributes.contains("guard_operand") ||
                        operation.attributes.contains("volatile"))
                        return std::nullopt;
                    ir::Type access_type;
                    if (operation.opcode == ir::OpCode::kLoad) {
                        if (operation.results.size() != 1 ||
                            operation.result_types.size() != 1)
                            return std::nullopt;
                        access_type = operation.result_types.front();
                    } else {
                        if (operation.operands.size() != 2)
                            return std::nullopt;
                        access_type = operation.operands[1].type;
                    }
                    if (!scalar_byte_size(access_type)) return std::nullopt;
                    if (element_type && *element_type != access_type)
                        return std::nullopt;
                    element_type = access_type;
                    Access access;
                    access.block = b;
                    access.operation = i;
                    access.is_load = operation.opcode == ir::OpCode::kLoad;
                    access.desc = plan.derived.at(operand.value);
                    plan.has_dynamic |= access.desc.index.has_value();
                    plan.accesses.push_back(access);
                }
            }
        }
        if (!element_type || !plan.has_dynamic) return std::nullopt;
        plan.element_type = *element_type;
        plan.element_bytes = *scalar_byte_size(*element_type);
        if (byte_size % plan.element_bytes != 0) return std::nullopt;
        plan.count = byte_size / plan.element_bytes;
        if (plan.count < 2 || plan.count > 16) return std::nullopt;
        for (const Access& access : plan.accesses) {
            if (access.desc.const_bytes %
                    static_cast<std::int64_t>(plan.element_bytes) != 0)
                return std::nullopt;
            const std::int64_t base_element =
                access.desc.const_bytes /
                static_cast<std::int64_t>(plan.element_bytes);
            if (base_element < 0 ||
                base_element >= static_cast<std::int64_t>(plan.count))
                return std::nullopt;
            if (access.desc.index &&
                access.desc.index_scale !=
                    static_cast<std::int64_t>(plan.element_bytes))
                return std::nullopt;
        }
        // Derived values must not participate in generic-pointer machinery.
        for (const auto& [value, desc] : plan.derived) {
            if (function_.generic_pointer_values.contains(value) ||
                function_.mixed_pointer_address_spaces.contains(value))
                return std::nullopt;
        }
        return plan;
    }

  private:
    const ir::Function& function_;
    std::unordered_map<ir::ValueId, std::pair<const ir::BasicBlock*, std::size_t>>
        definers_;
    std::unordered_map<ir::ValueId, ir::Type> value_types_;
};

ir::Operation make_operation(ir::OpCode opcode, ir::SourceLocation location) {
    ir::Operation operation;
    operation.opcode = opcode;
    operation.location = location;
    return operation;
}

// Emit ops materializing an operand as a value of the element type and
// return its id. Integers use a plain constant; floats go through a
// bit-pattern convert so the constant op stays integer-typed.
ir::ValueId materialize(std::vector<ir::Operation>* ops,
                        const ir::Operand& operand, const ir::Type& type,
                        ir::SourceLocation location, ir::ValueId* fresh) {
    if (operand.kind == ir::OperandKind::kValue) return operand.value;
    ir::Operation constant =
        make_operation(ir::OpCode::kConstant, location);
    const ir::ValueId constant_id = (*fresh)++;
    constant.results = {constant_id};
    if (type.kind == ir::TypeKind::kFloat) {
        const ir::Type int_type = ir::Type::integer(type.bit_width);
        std::uint64_t bits = 0;
        try {
            if (type.bit_width == 32) {
                const float parsed = std::stof(operand.text);
                std::uint32_t word = 0;
                std::memcpy(&word, &parsed, sizeof(word));
                bits = word;
            } else {
                const double parsed = std::stod(operand.text);
                std::memcpy(&bits, &parsed, sizeof(bits));
            }
        } catch (const std::exception&) {
            bits = 0;
        }
        constant.operands = {ir::Operand::immediate(std::to_string(bits), int_type)};
        constant.result_types = {int_type};
        ops->push_back(std::move(constant));
        ir::Operation convert =
            make_operation(ir::OpCode::kConvert, location);
        const ir::ValueId result = (*fresh)++;
        convert.operands = {ir::Operand::value_ref(constant_id, int_type)};
        convert.results = {result};
        convert.result_types = {type};
        convert.attributes["bitcast"] = "true";
        ops->push_back(std::move(convert));
        return result;
    }
    constant.operands = {ir::Operand::immediate(operand.text, type)};
    constant.result_types = {type};
    ops->push_back(std::move(constant));
    return constant_id;
}

bool rewrite_plan(ir::Function& function, const Plan& plan,
                  const Analyzer& analyzer) {
    ir::ValueId fresh = ir::kInvalidValue + 1;
    const auto consider = [&fresh](ir::ValueId value) {
        fresh = std::max(fresh, value + 1);
    };
    for (const ir::FunctionArgument& argument : function.arguments)
        consider(argument.value);
    for (const ir::BasicBlock& block : function.blocks) {
        for (const ir::BlockArgument& argument : block.arguments)
            consider(argument.value);
        for (const ir::Operation& operation : block.operations)
            for (const ir::ValueId result : operation.results) consider(result);
    }

    const std::size_t count = static_cast<std::size_t>(plan.count);
    const ir::Type type = plan.element_type;

    // Element entry values: appended block arguments everywhere except the
    // entry block, which starts from zero-initialized elements.
    std::vector<std::vector<ir::ValueId>> entry_elements(function.blocks.size());
    for (std::size_t b = 1; b < function.blocks.size(); ++b) {
        for (std::size_t k = 0; k < count; ++k) {
            const ir::ValueId value = fresh++;
            function.blocks[b].arguments.push_back(
                {.value = value, .type = type, .name = "cm_priv_elem"});
            entry_elements[b].push_back(value);
        }
    }

    std::vector<ir::Operation> init_ops;
    {
        std::vector<ir::ValueId> zeros;
        for (std::size_t k = 0; k < count; ++k) {
            const ir::Operand zero = ir::Operand::immediate("0", type);
            zeros.push_back(materialize(&init_ops, zero, type, {}, &fresh));
        }
        entry_elements[0] = zeros;
    }

    std::unordered_map<std::size_t, std::unordered_map<std::size_t, const Access*>>
        access_at;
    for (const Access& access : plan.accesses)
        access_at[access.block][access.operation] = &access;

    for (std::size_t b = 0; b < function.blocks.size(); ++b) {
        ir::BasicBlock& block = function.blocks[b];
        std::vector<ir::ValueId> elements = entry_elements[b];
        std::vector<ir::Operation> rewritten;
        if (b == 0) {
            for (auto& operation : init_ops) rewritten.push_back(std::move(operation));
        }
        for (std::size_t i = 0; i < block.operations.size(); ++i) {
            const ir::Operation& operation = block.operations[i];
            // Drop the allocation and its pointer-arithmetic chain.
            if (!operation.results.empty() &&
                plan.derived.contains(operation.results.front()) &&
                (operation.opcode == ir::OpCode::kAlloca ||
                 operation.opcode == ir::OpCode::kPointerOffset ||
                 operation.opcode == ir::OpCode::kConvert ||
                 operation.opcode == ir::OpCode::kAddressSpaceCast)) {
                continue;
            }
            const auto block_accesses = access_at.find(b);
            const Access* access = nullptr;
            if (block_accesses != access_at.end()) {
                const auto found = block_accesses->second.find(i);
                if (found != block_accesses->second.end()) access = found->second;
            }
            if (!access) {
                ir::Operation copy = operation;
                for (ir::Successor& successor : copy.successors)
                    for (const ir::ValueId element : elements)
                        successor.arguments.push_back(element);
                rewritten.push_back(std::move(copy));
                continue;
            }
            const std::int64_t base_element =
                access->desc.const_bytes /
                static_cast<std::int64_t>(plan.element_bytes);
            if (access->is_load) {
                const ir::ValueId load_result = operation.results.front();
                if (!access->desc.index) {
                    ir::Operation convert =
                        make_operation(ir::OpCode::kConvert, operation.location);
                    convert.operands = {ir::Operand::value_ref(
                        elements[static_cast<std::size_t>(base_element)], type)};
                    convert.results = {load_result};
                    convert.result_types = {type};
                    rewritten.push_back(std::move(convert));
                } else {
                    const ir::ValueId index_value = *access->desc.index;
                    const ir::Type resolved_index_type =
                        analyzer.type_of(index_value)
                            .value_or(ir::Type::integer(64));
                    ir::ValueId accumulator =
                        elements[static_cast<std::size_t>(plan.count) - 1];
                    for (std::int64_t k = static_cast<std::int64_t>(plan.count) - 2;
                         k >= 0; --k) {
                        ir::Operation compare =
                            make_operation(ir::OpCode::kCompare, operation.location);
                        const ir::ValueId predicate = fresh++;
                        compare.operands = {
                            ir::Operand::value_ref(index_value, resolved_index_type),
                            ir::Operand::immediate(
                                std::to_string(k - base_element),
                                resolved_index_type)};
                        compare.results = {predicate};
                        compare.result_types = {ir::Type::predicate()};
                        compare.attributes["predicate"] = "eq";
                        rewritten.push_back(std::move(compare));
                        ir::Operation select =
                            make_operation(ir::OpCode::kSelect, operation.location);
                        const ir::ValueId selected =
                            k == 0 ? load_result : fresh++;
                        select.operands = {
                            ir::Operand::value_ref(predicate, ir::Type::predicate()),
                            ir::Operand::value_ref(
                                elements[static_cast<std::size_t>(k)], type),
                            ir::Operand::value_ref(accumulator, type)};
                        select.results = {selected};
                        select.result_types = {type};
                        rewritten.push_back(std::move(select));
                        accumulator = selected;
                    }
                }
                continue;
            }
            // Store: conditional per-element update.
            const ir::ValueId stored = materialize(
                &rewritten, operation.operands[1], type, operation.location,
                &fresh);
            if (!access->desc.index) {
                elements[static_cast<std::size_t>(base_element)] = stored;
                continue;
            }
            const ir::ValueId index_value = *access->desc.index;
            const ir::Type resolved_index_type =
                analyzer.type_of(index_value).value_or(ir::Type::integer(64));
            for (std::size_t k = 0; k < count; ++k) {
                ir::Operation compare =
                    make_operation(ir::OpCode::kCompare, operation.location);
                const ir::ValueId predicate = fresh++;
                compare.operands = {
                    ir::Operand::value_ref(index_value, resolved_index_type),
                    ir::Operand::immediate(
                        std::to_string(static_cast<std::int64_t>(k) - base_element),
                        resolved_index_type)};
                compare.results = {predicate};
                compare.result_types = {ir::Type::predicate()};
                compare.attributes["predicate"] = "eq";
                rewritten.push_back(std::move(compare));
                ir::Operation select =
                    make_operation(ir::OpCode::kSelect, operation.location);
                const ir::ValueId selected = fresh++;
                select.operands = {
                    ir::Operand::value_ref(predicate, ir::Type::predicate()),
                    ir::Operand::value_ref(stored, type),
                    ir::Operand::value_ref(elements[k], type)};
                select.results = {selected};
                select.result_types = {type};
                rewritten.push_back(std::move(select));
                elements[k] = selected;
            }
        }
        block.operations = std::move(rewritten);
    }

    for (const auto& [value, desc] : plan.derived)
        function.pointer_provenance.erase(value);
    return true;
}

} // namespace

bool scalarize_private_arrays(ir::Function& function) {
    if (function.blocks.empty()) return false;
    // The entry block must not be a branch target: element values start
    // there and are threaded through every other block's arguments.
    const ir::BlockId entry_id = function.blocks.front().id;
    for (const ir::BasicBlock& block : function.blocks)
        for (const ir::Operation& operation : block.operations)
            for (const ir::Successor& successor : operation.successors)
                if (successor.block == entry_id) return false;

    bool changed_any = false;
    for (int attempt = 0; attempt < 8; ++attempt) {
        const Analyzer analyzer(function);
        bool rewrote = false;
        for (const ir::Operation& operation :
             function.blocks.front().operations) {
            if (operation.opcode != ir::OpCode::kAlloca ||
                operation.results.empty() ||
                !operation.attributes.contains("byte_size"))
                continue;
            std::uint64_t byte_size = 0;
            try {
                byte_size = std::stoull(operation.attributes.at("byte_size"));
            } catch (const std::exception&) {
                continue;
            }
            const std::optional<Plan> plan =
                analyzer.analyze(operation.results.front(), byte_size);
            if (!plan) continue;
            if (rewrite_plan(function, *plan, analyzer)) {
                rewrote = true;
                changed_any = true;
                break;
            }
        }
        if (!rewrote) break;
    }
    return changed_any;
}

} // namespace cumetal::metal::detail
