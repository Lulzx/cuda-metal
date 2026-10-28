#include "shared_memory_opt.h"

#include <algorithm>
#include <cstring>
#include <map>
#include <optional>
#include <set>
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

// Byte offset expressed as const_bytes + index * index_scale.
struct OffsetDesc {
    std::int64_t const_bytes = 0;
    std::optional<ir::ValueId> index;
    std::int64_t index_scale = 1;
};

struct TgDesc {
    ir::ValueId root = ir::kInvalidValue;
    std::string allocation;
    OffsetDesc offset;
};

class FunctionView {
  public:
    explicit FunctionView(const ir::Function& function) : function_(function) {
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
        build_descriptors();
    }

    const ir::Function& function() const { return function_; }
    const std::unordered_map<ir::ValueId, TgDesc>& descriptors() const {
        return descriptors_;
    }
    std::optional<ir::Type> type_of(ir::ValueId value) const {
        const auto found = value_types_.find(value);
        if (found == value_types_.end()) return std::nullopt;
        return found->second;
    }
    const ir::Operation* definer_of(ir::ValueId value) const {
        const auto found = definers_.find(value);
        if (found == definers_.end()) return nullptr;
        return &found->second.first->operations[found->second.second];
    }

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
                const auto scale = fold_immediate(definition.operands[1 - side]);
                if (scale && *scale > 0 &&
                    definition.operands[side].kind == ir::OperandKind::kValue) {
                    // Push constant factors out of the index expression
                    // (clang factors t*16 as (t<<2)*4) so the scale is the
                    // true byte stride per index step.
                    ir::ValueId index = definition.operands[side].value;
                    std::int64_t total_scale = *scale;
                    for (int step = 0; step < 4; ++step) {
                        const auto index_def = definers_.find(index);
                        if (index_def == definers_.end()) break;
                        const ir::Operation& index_op = index_def->second.first
                            ->operations[index_def->second.second];
                        if (index_op.opcode == ir::OpCode::kConvert &&
                            index_op.operands.size() == 1 &&
                            index_op.operands.front().kind ==
                                ir::OperandKind::kValue) {
                            index = index_op.operands.front().value;
                            continue;
                        }
                        if (index_op.opcode == ir::OpCode::kMul &&
                            index_op.operands.size() == 2) {
                            bool pushed = false;
                            for (const int inner : {0, 1}) {
                                const auto factor =
                                    fold_immediate(index_op.operands[1 - inner]);
                                if (factor && *factor > 0 &&
                                    index_op.operands[inner].kind ==
                                        ir::OperandKind::kValue) {
                                    index = index_op.operands[inner].value;
                                    total_scale *= *factor;
                                    pushed = true;
                                    break;
                                }
                            }
                            if (pushed) continue;
                        }
                        if (index_op.opcode == ir::OpCode::kShiftLeft &&
                            index_op.operands.size() == 2) {
                            const auto shift_amount =
                                fold_immediate(index_op.operands[1]);
                            if (shift_amount && *shift_amount >= 0 &&
                                *shift_amount < 32 &&
                                index_op.operands.front().kind ==
                                    ir::OperandKind::kValue) {
                                index = index_op.operands.front().value;
                                total_scale *= (std::int64_t{1} << *shift_amount);
                                continue;
                            }
                        }
                        break;
                    }
                    return OffsetDesc{.const_bytes = 0,
                                      .index = index,
                                      .index_scale = total_scale};
                }
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

    // Search an offset expression tree for a power-of-two row stride
    // (64..4096 bytes): either a direct row * row_bytes multiply, or the
    // (row * cols + col) * element_size form where cols * element_size is
    // the row stride.
    std::optional<std::int64_t> find_row_stride(const ir::Operand& operand,
                                                int depth = 8) const {
        if (operand.kind != ir::OperandKind::kValue || depth <= 0)
            return std::nullopt;
        const auto found = definers_.find(operand.value);
        if (found == definers_.end()) return std::nullopt;
        const ir::Operation& definition =
            found->second.first->operations[found->second.second];
        const auto is_stride = [](std::int64_t value) {
            return value >= 64 && value <= 4096 && (value & (value - 1)) == 0;
        };
        if (definition.opcode == ir::OpCode::kMul &&
            definition.operands.size() == 2) {
            for (const int side : {0, 1}) {
                const auto scale = fold_immediate(definition.operands[1 - side]);
                if (!scale) continue;
                if (is_stride(*scale)) return scale;
                // (row * cols + col) * elem: the other side is an add with
                // a row * cols term.
                if (*scale != 4 && *scale != 8) continue;
                const ir::Operand& sum_operand = definition.operands[side];
                if (sum_operand.kind != ir::OperandKind::kValue) continue;
                const auto sum_found = definers_.find(sum_operand.value);
                if (sum_found == definers_.end()) continue;
                const ir::Operation& sum =
                    sum_found->second.first->operations[sum_found->second.second];
                if (sum.opcode != ir::OpCode::kAdd || sum.operands.size() != 2)
                    continue;
                for (const ir::Operand& term : sum.operands) {
                    if (term.kind != ir::OperandKind::kValue) continue;
                    const auto term_found = definers_.find(term.value);
                    if (term_found == definers_.end()) continue;
                    const ir::Operation& product = term_found->second.first
                        ->operations[term_found->second.second];
                    if (product.opcode != ir::OpCode::kMul ||
                        product.operands.size() != 2)
                        continue;
                    for (const int product_side : {0, 1}) {
                        const auto cols =
                            fold_immediate(product.operands[1 - product_side]);
                        if (cols && is_stride(*cols * *scale))
                            return *cols * *scale;
                    }
                }
            }
        }
        for (const ir::Operand& child : definition.operands) {
            if (const auto stride = find_row_stride(child, depth - 1))
                return stride;
        }
        return std::nullopt;
    }

  private:
    void build_descriptors() {
        for (const ir::BasicBlock& block : function_.blocks) {
            for (const ir::Operation& operation : block.operations) {
                if (operation.results.empty() || operation.operands.empty())
                    continue;
                if (operation.operands.front().kind != ir::OperandKind::kSymbol)
                    continue;
                if (operation.result_types.empty() ||
                    !operation.result_types.front().is_pointer() ||
                    operation.result_types.front().address_space !=
                        ir::AddressSpace::kThreadgroup)
                    continue;
                descriptors_[operation.results.front()] = TgDesc{
                    .root = operation.results.front(),
                    .allocation = operation.operands.front().text,
                    .offset = OffsetDesc{}};
            }
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (const ir::BasicBlock& block : function_.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.results.empty() || operation.operands.empty())
                        continue;
                    const ir::ValueId result = operation.results.front();
                    std::optional<TgDesc> desc;
                    if (operation.opcode == ir::OpCode::kPointerOffset &&
                        operation.operands.size() == 2 &&
                        operation.operands.front().kind ==
                            ir::OperandKind::kValue) {
                        const auto base =
                            descriptors_.find(operation.operands.front().value);
                        if (base != descriptors_.end()) {
                            const auto offset =
                                decompose_offset(operation.operands[1]);
                            if (offset) {
                                const auto combined =
                                    combine(base->second.offset, *offset);
                                if (combined) {
                                    desc = base->second;
                                    desc->offset = *combined;
                                }
                            }
                        }
                    } else if ((operation.opcode == ir::OpCode::kConvert ||
                                operation.opcode ==
                                    ir::OpCode::kAddressSpaceCast) &&
                               operation.operands.size() == 1 &&
                               operation.operands.front().kind ==
                                   ir::OperandKind::kValue &&
                               !operation.result_types.empty() &&
                               operation.result_types.front().is_pointer() &&
                               operation.result_types.front().address_space ==
                                   ir::AddressSpace::kThreadgroup) {
                        const auto base =
                            descriptors_.find(operation.operands.front().value);
                        if (base != descriptors_.end()) desc = base->second;
                    }
                    if (!desc) continue;
                    const auto existing = descriptors_.find(result);
                    if (existing == descriptors_.end()) {
                        descriptors_[result] = *desc;
                        changed = true;
                    }
                }
            }
        }
    }

    const ir::Function& function_;
    std::unordered_map<ir::ValueId, std::pair<const ir::BasicBlock*, std::size_t>>
        definers_;
    std::unordered_map<ir::ValueId, ir::Type> value_types_;
    std::unordered_map<ir::ValueId, TgDesc> descriptors_;
};

// An intervening operation blocks merging only if it can observe or
// modify threadgroup memory: barriers, calls, and threadgroup (or
// unresolved generic) memory accesses. Device/global accesses cannot
// alias a threadgroup allocation.
bool is_blocking_op(const ir::Operation& operation) {
    switch (operation.opcode) {
        case ir::OpCode::kBarrier:
        case ir::OpCode::kMetalBarrier:
        case ir::OpCode::kCall:
        case ir::OpCode::kIndirectCall:
        case ir::OpCode::kDeviceLaunch:
        case ir::OpCode::kDeviceMemcpyAsync:
        case ir::OpCode::kDeviceMalloc:
        case ir::OpCode::kDeviceFree:
            return true;
        case ir::OpCode::kLoad:
        case ir::OpCode::kStore:
        case ir::OpCode::kAtomic: {
            if (operation.operands.empty() ||
                !operation.operands.front().type.is_pointer())
                return true;
            const ir::AddressSpace space =
                operation.operands.front().type.address_space;
            return space == ir::AddressSpace::kThreadgroup ||
                   space == ir::AddressSpace::kNone;
        }
        default:
            return false;
    }
}

bool is_access_clean(const ir::Operation& operation) {
    return !operation.attributes.contains("guard_operand") &&
           !operation.attributes.contains("volatile") &&
           operation.memory_ordering == ir::MemoryOrdering::kNone;
}

ir::ValueId fresh_start(const ir::Function& function) {
    ir::ValueId fresh = ir::kInvalidValue + 1;
    for (const ir::FunctionArgument& argument : function.arguments)
        fresh = std::max(fresh, argument.value + 1);
    for (const ir::BasicBlock& block : function.blocks) {
        for (const ir::BlockArgument& argument : block.arguments)
            fresh = std::max(fresh, argument.value + 1);
        for (const ir::Operation& operation : block.operations)
            for (const ir::ValueId result : operation.results)
                fresh = std::max(fresh, result + 1);
    }
    return fresh;
}

ir::Operation make_operation(ir::OpCode opcode, ir::SourceLocation location) {
    ir::Operation operation;
    operation.opcode = opcode;
    operation.location = location;
    return operation;
}

// Materialize a (possibly immediate) store operand as a value.
ir::ValueId materialize(std::vector<ir::Operation>* ops,
                        const ir::Operand& operand, const ir::Type& type,
                        ir::SourceLocation location, ir::ValueId* fresh) {
    if (operand.kind == ir::OperandKind::kValue) return operand.value;
    ir::Operation constant = make_operation(ir::OpCode::kConstant, location);
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
        ir::Operation convert = make_operation(ir::OpCode::kConvert, location);
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

struct LaneGroup {
    std::array<std::size_t, 4> ops{};
    ir::Type element_type;
    ir::Operand base_address;
    std::string allocation;
};

// Find 4-lane load/store groups in one block and rewrite them. Returns
// the set of allocations whose alignment must be raised.
std::set<std::string> coalesce_block(ir::BasicBlock& block,
                                     const FunctionView& view,
                                     ir::ValueId* fresh) {
    std::set<std::string> touched;
    struct Candidate {
        std::size_t op_index;
        bool is_load;
        TgDesc desc;
        ir::Type element_type;
    };
    std::vector<Candidate> candidates;
    for (std::size_t i = 0; i < block.operations.size(); ++i) {
        const ir::Operation& operation = block.operations[i];
        const bool is_load = operation.opcode == ir::OpCode::kLoad;
        const bool is_store = operation.opcode == ir::OpCode::kStore;
        if (!is_load && !is_store) continue;
        if (!is_access_clean(operation) || operation.operands.empty() ||
            operation.operands.front().kind != ir::OperandKind::kValue)
            continue;
        const auto desc = view.descriptors().find(operation.operands.front().value);
        if (desc == view.descriptors().end()) continue;
        ir::Type element_type;
        if (is_load) {
            if (operation.results.size() != 1 || operation.result_types.size() != 1)
                continue;
            element_type = operation.result_types.front();
        } else {
            if (operation.operands.size() != 2) continue;
            element_type = operation.operands[1].type;
        }
        if ((element_type.kind != ir::TypeKind::kInteger &&
             element_type.kind != ir::TypeKind::kFloat) ||
            element_type.bit_width != 32)
            continue;
        if (desc->second.offset.const_bytes < 0 ||
            desc->second.offset.const_bytes % 4 != 0)
            continue;
        if (desc->second.offset.index &&
            desc->second.offset.index_scale % 16 != 0)
            continue;
        candidates.push_back({i, is_load, desc->second, element_type});
    }

    // Group by (allocation, index, index_scale, kind, element type, and the
    // 16-byte-aligned base offset).
    struct GroupKey {
        std::string allocation;
        std::optional<ir::ValueId> index;
        std::int64_t index_scale = 1;
        bool is_load = false;
        ir::Type element_type;
        std::int64_t base_bytes = 0;
        bool operator<(const GroupKey& other) const {
            const auto discriminator = [](const GroupKey& key) {
                return std::make_tuple(
                    key.allocation, key.index, key.index_scale, key.is_load,
                    key.base_bytes, static_cast<int>(key.element_type.kind),
                    key.element_type.bit_width);
            };
            return discriminator(*this) < discriminator(other);
        }
        bool operator==(const GroupKey& other) const {
            return allocation == other.allocation && index == other.index &&
                   index_scale == other.index_scale && is_load == other.is_load &&
                   element_type == other.element_type &&
                   base_bytes == other.base_bytes;
        }
    };
    std::map<GroupKey, std::map<int, std::size_t>> groups;
    for (const Candidate& candidate : candidates) {
        const std::int64_t lane_offset = candidate.desc.offset.const_bytes;
        GroupKey key;
        key.allocation = candidate.desc.allocation;
        key.index = candidate.desc.offset.index;
        key.index_scale = candidate.desc.offset.index_scale;
        key.is_load = candidate.is_load;
        key.element_type = candidate.element_type;
        key.base_bytes = lane_offset - (lane_offset % 16);
        groups[key][static_cast<int>((lane_offset % 16) / 4)] = candidate.op_index;
    }

    std::unordered_set<std::size_t> removed;
    std::unordered_map<std::size_t, std::vector<ir::Operation>> replace_with;
    for (const auto& [key, lanes] : groups) {
        if (lanes.size() != 4) continue;
        std::array<std::size_t, 4> lane_ops{};
        bool complete = true;
        for (int lane = 0; lane < 4; ++lane) {
            const auto found = lanes.find(lane);
            if (found == lanes.end()) complete = false;
            else lane_ops[lane] = found->second;
        }
        if (!complete) continue;
        const std::size_t first = *std::min_element(lane_ops.begin(), lane_ops.end());
        const std::size_t last = *std::max_element(lane_ops.begin(), lane_ops.end());
        bool blocked = false;
        for (std::size_t i = first; i <= last && !blocked; ++i) {
            if (i == lane_ops[0] || i == lane_ops[1] || i == lane_ops[2] ||
                i == lane_ops[3])
                continue;
            if (is_blocking_op(block.operations[i])) blocked = true;
        }
        if (blocked) continue;

        const ir::Operation& lane0 = block.operations[lane_ops[0]];
        const ir::Type aggregate_type = ir::Type::aggregate(
            {key.element_type, key.element_type, key.element_type,
             key.element_type},
            key.element_type.kind == ir::TypeKind::kFloat ? "float4" : "uint4");
        const ir::SourceLocation location = lane0.location;
        if (key.is_load) {
            std::vector<ir::Operation> emitted;
            ir::Operation load = make_operation(ir::OpCode::kLoad, location);
            const ir::ValueId vector_value = (*fresh)++;
            load.operands = {lane0.operands.front()};
            load.results = {vector_value};
            load.result_types = {aggregate_type};
            load.attributes = lane0.attributes;
            load.attributes.erase("memory_bit_width");
            load.attributes["alignment"] = "16";
            emitted.push_back(std::move(load));
            for (int lane = 0; lane < 4; ++lane) {
                ir::Operation extract =
                    make_operation(ir::OpCode::kAggregateExtract, location);
                extract.operands = {
                    ir::Operand::value_ref(vector_value, aggregate_type),
                    ir::Operand::immediate(std::to_string(lane),
                                           ir::Type::integer(32))};
                extract.results = {block.operations[lane_ops[lane]].results.front()};
                extract.result_types = {key.element_type};
                emitted.push_back(std::move(extract));
            }
            replace_with[lane_ops[0]] = std::move(emitted);
        } else {
            std::vector<ir::Operation> emitted;
            std::array<ir::Operand, 4> values;
            for (int lane = 0; lane < 4; ++lane) {
                const ir::Operand& stored =
                    block.operations[lane_ops[lane]].operands[1];
                const ir::ValueId value = materialize(
                    &emitted, stored, key.element_type, location, fresh);
                values[lane] = ir::Operand::value_ref(value, key.element_type);
            }
            ir::Operation construct =
                make_operation(ir::OpCode::kAggregateConstruct, location);
            const ir::ValueId vector_value = (*fresh)++;
            construct.operands = {values[0], values[1], values[2], values[3]};
            construct.results = {vector_value};
            construct.result_types = {aggregate_type};
            construct.attributes["constructor"] =
                key.element_type.kind == ir::TypeKind::kFloat ? "float4" : "uint4";
            emitted.push_back(std::move(construct));
            ir::Operation store = make_operation(ir::OpCode::kStore, location);
            store.operands = {lane0.operands.front(),
                               ir::Operand::value_ref(vector_value, aggregate_type)};
            emitted.push_back(std::move(store));
            replace_with[last] = std::move(emitted);
        }
        for (std::size_t lane_op : lane_ops) {
            if (!replace_with.contains(lane_op)) removed.insert(lane_op);
        }
        touched.insert(key.allocation);
    }

    if (removed.empty() && replace_with.empty()) return touched;
    std::vector<ir::Operation> rewritten;
    for (std::size_t i = 0; i < block.operations.size(); ++i) {
        if (removed.contains(i)) continue;
        const auto replacement = replace_with.find(i);
        if (replacement != replace_with.end()) {
            for (auto& operation : replacement->second)
                rewritten.push_back(std::move(operation));
            continue;
        }
        rewritten.push_back(std::move(block.operations[i]));
    }
    block.operations = std::move(rewritten);
    return touched;
}

// Padding plan for one allocation in one function. Analysis and rewrite
// are separate so a refusal in any function leaves every function (and
// the allocation size) untouched. Access addresses are followed back
// through their pointer_offset chain to the allocation root; the total
// byte offset is the sum of the chain's offset operands, which handles
// multi-term 2D addresses (row * stride + col * elem) directly.
struct PaddingSite {
    ir::BasicBlock* block;
    std::size_t op_index;
    ir::ValueId root;
    std::vector<ir::Operand> offset_operands;
};

struct PaddingPlan {
    enum class State { kPoisoned, kNoSites, kReady };
    State state = State::kNoSites;
    std::vector<PaddingSite> sites;
    std::int64_t row_bytes = 0;
    std::uint64_t added = 0;
};

struct ChainWalk {
    ir::ValueId root;
    std::string allocation;
    std::vector<ir::Operand> offset_operands;
    std::vector<ir::ValueId> chain_values;
};

// Follow an address value back to its threadgroup symbol root. Returns
// nullopt when any step is not a plain pointer_offset/convert chain.
std::optional<ChainWalk> walk_chain(const FunctionView& view,
                                    const ir::Function& function,
                                    ir::ValueId address) {
    (void)function;
    ChainWalk walk;
    walk.root = ir::kInvalidValue;
    ir::ValueId current = address;
    for (int step = 0; step < 16; ++step) {
        walk.chain_values.push_back(current);
        const ir::Operation* definition = view.definer_of(current);
        if (!definition) return std::nullopt;
        // Root: an op whose first operand is the allocation symbol.
        if (!definition->operands.empty() &&
            definition->operands.front().kind == ir::OperandKind::kSymbol &&
            !definition->result_types.empty() &&
            definition->result_types.front().is_pointer() &&
            definition->result_types.front().address_space ==
                ir::AddressSpace::kThreadgroup) {
            walk.root = current;
            walk.allocation = definition->operands.front().text;
            return walk;
        }
        if (definition->opcode == ir::OpCode::kPointerOffset &&
            definition->operands.size() == 2 &&
            definition->operands.front().kind == ir::OperandKind::kValue) {
            walk.offset_operands.push_back(definition->operands[1]);
            current = definition->operands.front().value;
            continue;
        }
        if ((definition->opcode == ir::OpCode::kConvert ||
             definition->opcode == ir::OpCode::kAddressSpaceCast) &&
            definition->operands.size() == 1 &&
            definition->operands.front().kind == ir::OperandKind::kValue) {
            current = definition->operands.front().value;
            continue;
        }
        return std::nullopt;
    }
    return std::nullopt;
}

PaddingPlan analyze_padding(const ir::Function& function,
                            const std::string& allocation,
                            std::uint64_t byte_size) {
    PaddingPlan plan;
    FunctionView view(function);
    std::unordered_set<ir::ValueId> chain_values;
    for (const ir::BasicBlock& block : function.blocks) {
        for (std::size_t i = 0; i < block.operations.size(); ++i) {
            const ir::Operation& operation = block.operations[i];
            const bool is_load = operation.opcode == ir::OpCode::kLoad;
            const bool is_store = operation.opcode == ir::OpCode::kStore;
            if (!is_load && !is_store) {
                if (operation.opcode == ir::OpCode::kAtomic &&
                    !operation.operands.empty() &&
                    operation.operands.front().kind == ir::OperandKind::kValue &&
                    operation.operands.front().type.is_pointer() &&
                    operation.operands.front().type.address_space ==
                        ir::AddressSpace::kThreadgroup) {
                    plan.state = PaddingPlan::State::kPoisoned;
                    return plan;
                }
                continue;
            }
            if (operation.operands.empty() ||
                operation.operands.front().kind != ir::OperandKind::kValue ||
                !operation.operands.front().type.is_pointer() ||
                operation.operands.front().type.address_space !=
                    ir::AddressSpace::kThreadgroup)
                continue;
            const std::optional<ChainWalk> walk =
                walk_chain(view, function, operation.operands.front().value);
            if (!walk) {
                plan.state = PaddingPlan::State::kPoisoned;
                return plan;
            }
            if (walk->allocation != allocation) continue;
            if (!is_access_clean(operation)) {
                plan.state = PaddingPlan::State::kPoisoned;
                return plan;
            }
            ir::Type element_type;
            if (is_load) {
                if (operation.result_types.size() != 1) {
                    plan.state = PaddingPlan::State::kPoisoned;
                    return plan;
                }
                element_type = operation.result_types.front();
            } else {
                if (operation.operands.size() != 2) {
                    plan.state = PaddingPlan::State::kPoisoned;
                    return plan;
                }
                element_type = operation.operands[1].type;
            }
            if ((element_type.kind != ir::TypeKind::kInteger &&
                 element_type.kind != ir::TypeKind::kFloat) ||
                element_type.bit_width != 32) {
                plan.state = PaddingPlan::State::kPoisoned;
                return plan;
            }
            chain_values.insert(walk->chain_values.begin(),
                                walk->chain_values.end());
            plan.sites.push_back({const_cast<ir::BasicBlock*>(&block), i,
                                  walk->root, walk->offset_operands});
        }
    }
    if (plan.sites.empty()) return plan;

    // Address escapes: chain values may only feed chain steps or the
    // audited memory accesses themselves.
    for (const ir::BasicBlock& block : function.blocks) {
        for (const ir::Operation& operation : block.operations) {
            for (const ir::Operand& operand : operation.operands) {
                if (operand.kind != ir::OperandKind::kValue ||
                    !chain_values.contains(operand.value))
                    continue;
                const bool chain =
                    (operation.opcode == ir::OpCode::kPointerOffset ||
                     operation.opcode == ir::OpCode::kConvert ||
                     operation.opcode == ir::OpCode::kAddressSpaceCast) &&
                    operand.value == operation.operands.front().value;
                const bool memory =
                    (operation.opcode == ir::OpCode::kLoad ||
                     operation.opcode == ir::OpCode::kStore) &&
                    operand.value == operation.operands.front().value;
                if (!chain && !memory) {
                    plan.state = PaddingPlan::State::kPoisoned;
                    return plan;
                }
            }
            for (const ir::Successor& successor : operation.successors)
                for (const ir::ValueId argument : successor.arguments)
                    if (chain_values.contains(argument)) {
                        plan.state = PaddingPlan::State::kPoisoned;
                        return plan;
                    }
        }
    }

    std::set<std::int64_t> strides;
    for (const PaddingSite& site : plan.sites) {
        for (const ir::Operand& operand : site.offset_operands) {
            if (const auto stride = view.find_row_stride(operand))
                strides.insert(*stride);
        }
    }
    if (strides.size() != 1) {
        plan.state = PaddingPlan::State::kPoisoned;
        return plan;
    }
    plan.row_bytes = *strides.begin();
    if (byte_size % static_cast<std::uint64_t>(plan.row_bytes) != 0) {
        plan.state = PaddingPlan::State::kPoisoned;
        return plan;
    }
    plan.added = (byte_size / static_cast<std::uint64_t>(plan.row_bytes)) * 4;
    plan.state = PaddingPlan::State::kReady;
    return plan;
}

void apply_padding(ir::Function& function, PaddingPlan& plan,
                   ir::ValueId* fresh) {
    int shift = 0;
    while ((std::int64_t{1} << shift) < plan.row_bytes) ++shift;
    const ir::Type i64 = ir::Type::integer(64);
    std::unordered_map<ir::BasicBlock*, std::size_t> block_shift;
    for (PaddingSite& site : plan.sites) {
        const std::size_t op_index = site.op_index + block_shift[site.block];
        ir::Operation& access = site.block->operations[op_index];
        const ir::Type pointer_type = access.operands.front().type;
        std::vector<ir::Operation> prefix;
        const auto make_i64 = [&](ir::OpCode opcode,
                                  std::vector<ir::Operand> operands) {
            ir::Operation made = make_operation(opcode, access.location);
            const ir::ValueId result = (*fresh)++;
            made.operands = std::move(operands);
            made.results = {result};
            made.result_types = {i64};
            prefix.push_back(std::move(made));
            return result;
        };
        // Total byte offset = sum of the chain's offset operands.
        std::optional<ir::ValueId> total;
        std::int64_t total_const = 0;
        bool has_dynamic = false;
        for (const ir::Operand& operand : site.offset_operands) {
            std::int64_t immediate = 0;
            if (parse_immediate_i64(operand, &immediate)) {
                total_const += immediate;
                continue;
            }
            if (operand.kind != ir::OperandKind::kValue) continue;
            has_dynamic = true;
            ir::ValueId term = operand.value;
            if (operand.type != i64) {
                const ir::ValueId widened = (*fresh)++;
                ir::Operation convert =
                    make_operation(ir::OpCode::kConvert, access.location);
                convert.operands = {ir::Operand::value_ref(term, operand.type)};
                convert.results = {widened};
                convert.result_types = {i64};
                prefix.push_back(std::move(convert));
                term = widened;
            }
            if (!total) {
                total = term;
            } else {
                total = make_i64(ir::OpCode::kAdd,
                                 {ir::Operand::value_ref(*total, i64),
                                  ir::Operand::value_ref(term, i64)});
            }
        }
        ir::Operand padded_operand =
            ir::Operand::immediate("0", i64);
        if (has_dynamic && total) {
            ir::ValueId total_value = *total;
            if (total_const != 0) {
                total_value = make_i64(
                    ir::OpCode::kAdd,
                    {ir::Operand::value_ref(total_value, i64),
                     ir::Operand::immediate(std::to_string(total_const), i64)});
            }
            const ir::ValueId rows = make_i64(
                ir::OpCode::kShiftRight,
                {ir::Operand::value_ref(total_value, i64),
                 ir::Operand::immediate(std::to_string(shift), i64)});
            const ir::ValueId pad = make_i64(
                ir::OpCode::kMul,
                {ir::Operand::value_ref(rows, i64),
                 ir::Operand::immediate("4", i64)});
            const ir::ValueId padded_value = make_i64(
                ir::OpCode::kAdd,
                {ir::Operand::value_ref(total_value, i64),
                 ir::Operand::value_ref(pad, i64)});
            padded_operand = ir::Operand::value_ref(padded_value, i64);
        } else {
            padded_operand = ir::Operand::immediate(
                std::to_string(total_const +
                               (total_const / plan.row_bytes) * 4),
                i64);
        }
        ir::Operation address_op =
            make_operation(ir::OpCode::kPointerOffset, access.location);
        const ir::ValueId new_address = (*fresh)++;
        address_op.operands = {ir::Operand::value_ref(site.root, pointer_type),
                               padded_operand};
        address_op.results = {new_address};
        address_op.result_types = {pointer_type};
        address_op.attributes["offset_unit"] = "bytes";
        prefix.push_back(std::move(address_op));

        const std::size_t prefix_size = prefix.size();
        site.block->operations.insert(
            site.block->operations.begin() +
                static_cast<std::ptrdiff_t>(op_index),
            std::make_move_iterator(prefix.begin()),
            std::make_move_iterator(prefix.end()));
        site.block->operations[op_index + prefix_size].operands.front() =
            ir::Operand::value_ref(new_address, pointer_type);
        block_shift[site.block] += prefix_size;
    }
}

} // namespace

void optimize_threadgroup_memory(ir::Module& module) {
    // 1. float4 coalescing (per block), then raise allocation alignment.
    std::set<std::string> coalesced_allocations;
    for (ir::Function& function : module.functions) {
        ir::ValueId fresh = fresh_start(function);
        for (ir::BasicBlock& block : function.blocks) {
            FunctionView view(function);
            const std::set<std::string> touched =
                coalesce_block(block, view, &fresh);
            coalesced_allocations.insert(touched.begin(), touched.end());
        }
    }
    for (ir::GlobalThreadgroup& allocation : module.global_threadgroups) {
        if (coalesced_allocations.contains(allocation.name))
            allocation.alignment = std::max<std::uint32_t>(allocation.alignment, 16);
    }

    // 2. Bank-conflict padding for static allocations. Analyze every
    // function first; a refusal anywhere leaves the allocation untouched.
    for (ir::GlobalThreadgroup& allocation : module.global_threadgroups) {
        if (allocation.is_dynamic || allocation.byte_size < 256) continue;
        std::vector<std::pair<ir::Function*, PaddingPlan>> plans;
        std::uint64_t added = 0;
        bool poisoned = false;
        for (ir::Function& function : module.functions) {
            PaddingPlan plan =
                analyze_padding(function, allocation.name, allocation.byte_size);
            if (plan.state == PaddingPlan::State::kPoisoned) {
                poisoned = true;
                break;
            }
            if (plan.state == PaddingPlan::State::kReady) {
                if (added != 0 && plan.added != added) {
                    poisoned = true;
                    break;
                }
                added = plan.added;
                plans.emplace_back(&function, std::move(plan));
            }
        }
        if (poisoned || added == 0) continue;
        for (auto& [function, plan] : plans) {
            ir::ValueId fresh = fresh_start(*function);
            apply_padding(*function, plan, &fresh);
            if (function->is_kernel && function->kernel_abi) {
                // Static threadgroup accounting follows the padded size;
                // that is what Metal will reserve for this kernel.
                function->kernel_abi->static_threadgroup_memory +=
                    static_cast<std::uint32_t>(added);
            }
        }
        allocation.byte_size += added;
    }
}

} // namespace cumetal::metal::detail
