#include "cumetal/metal/lower_to_msl.h"

#include "cumetal/ir/ptx_importer.h"
#include "cumetal/ir/nvvm_importer.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::metal {
namespace {

MslAddressSpace lower_address_space(ir::AddressSpace address_space) {
    switch (address_space) {
        case ir::AddressSpace::kDevice: return MslAddressSpace::kDevice;
        case ir::AddressSpace::kConstant: return MslAddressSpace::kConstant;
        case ir::AddressSpace::kThreadgroup: return MslAddressSpace::kThreadgroup;
        case ir::AddressSpace::kPrivate: return MslAddressSpace::kThread;
        case ir::AddressSpace::kNone: return MslAddressSpace::kNone;
    }
    return MslAddressSpace::kNone;
}

bool is_native_vector_aggregate(const ir::Type& type) {
    if (type.kind != ir::TypeKind::kAggregate ||
        (type.elements.size() != 2 && type.elements.size() != 4) ||
        type.name.empty()) {
        return false;
    }
    if (!std::all_of(type.elements.begin(), type.elements.end(),
                     [&](const ir::Type& element) {
                         return element == type.elements.front();
                     })) {
        return false;
    }
    const std::string_view name = type.name;
    return name.ends_with("float2") || name.ends_with("float4") ||
           name.ends_with("double2") || name.ends_with("double4") ||
           name.ends_with("uint2") || name.ends_with("uint4") ||
           name.ends_with("int2") || name.ends_with("int4") ||
           name.ends_with("uchar2") || name.ends_with("uchar4") ||
           name.ends_with("ushort2") || name.ends_with("ushort4") ||
           name.ends_with("ulong2") || name.ends_with("ulong4") ||
           name.ends_with("ulonglong2") || name.ends_with("longlong2");
}

std::string aggregate_type_name(const ir::Type& type) {
    if (!type.name.empty()) return type.name;
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char byte : type.str()) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    std::ostringstream name;
    name << "CuMetalAggregate_" << std::hex << hash;
    return name.str();
}

MslType lower_type(const ir::Type& type) {
    switch (type.kind) {
        case ir::TypeKind::kVoid:
            return MslType::void_type();
        case ir::TypeKind::kPredicate:
            return MslType::boolean();
        case ir::TypeKind::kInteger:
            return type.bit_width == 1 ? MslType::boolean() : MslType::uint(type.bit_width);
        case ir::TypeKind::kFloat:
            return MslType::floating(type.bit_width);
        case ir::TypeKind::kVector:
            return MslType::vector(
                type.elements.empty() ? MslType::uint() : lower_type(type.elements.front()),
                type.lanes);
        case ir::TypeKind::kPointer:
            return MslType::pointer(
                type.elements.empty() ? MslType::uint(8) : lower_type(type.elements.front()),
                lower_address_space(type.address_space));
        case ir::TypeKind::kAggregate: {
            if (is_native_vector_aggregate(type)) {
                return MslType::vector(lower_type(type.elements.front()),
                                       static_cast<std::uint32_t>(type.elements.size()));
            }
            MslType result;
            result.kind = MslTypeKind::kStruct;
            result.struct_name = aggregate_type_name(type);
            return result;
        }
    }
    return MslType::void_type();
}

std::string value_name(ir::ValueId value) {
    return "v" + std::to_string(value);
}

std::string dimension_member(const ir::Operation& operation) {
    const auto dimension = operation.attributes.find("dimension");
    return dimension == operation.attributes.end() ? "x" : dimension->second;
}

std::string binary_spelling(ir::OpCode opcode) {
    switch (opcode) {
        case ir::OpCode::kAdd:
        case ir::OpCode::kPointerOffset: return "+";
        case ir::OpCode::kSub: return "-";
        case ir::OpCode::kMul: return "*";
        case ir::OpCode::kDiv: return "/";
        case ir::OpCode::kRemainder: return "%";
        case ir::OpCode::kBitAnd: return "&";
        case ir::OpCode::kBitOr: return "|";
        case ir::OpCode::kBitXor: return "^";
        case ir::OpCode::kShiftLeft: return "<<";
        case ir::OpCode::kShiftRight: return ">>";
        default: return "";
    }
}

std::string compare_spelling(const ir::Operation& operation) {
    const auto predicate = operation.attributes.find("predicate");
    const std::string value = predicate == operation.attributes.end() ? "eq" : predicate->second;
    if (value == "eq" || value == "equ") return "==";
    if (value == "ne" || value == "neu") return "!=";
    if (value == "lt" || value == "lo" || value == "ltu" || value == "slt") return "<";
    if (value == "le" || value == "ls" || value == "leu" || value == "sle") return "<=";
    if (value == "gt" || value == "hi" || value == "gtu" || value == "sgt") return ">";
    if (value == "ge" || value == "hs" || value == "geu" || value == "sge") return ">=";
    return "==";
}

bool is_terminal_return_block(const ir::BasicBlock& block) {
    return block.operations.size() == 1 &&
           block.operations.front().opcode == ir::OpCode::kReturn;
}

bool is_inlineable_terminal_return_block(const ir::BasicBlock& block) {
    return block.arguments.empty() && is_terminal_return_block(block);
}

struct AddressSpaceResolution {
    bool ok = false;
    std::string error;
};

class AddressSpaceConstraints {
public:
    std::size_t add_node() {
        const std::size_t node = parents_.size();
        parents_.push_back(node);
        ranks_.push_back(0);
        spaces_.push_back(0);
        polymorphic_.push_back(false);
        return node;
    }

    void flow(std::size_t source, std::size_t target) {
        flows_.push_back({source, target});
    }

    bool solve() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& [source, target] : flows_) {
                const std::uint8_t incoming = spaces_[source];
                if (incoming == 0) continue;
                const std::uint8_t combined = spaces_[target] | incoming;
                if (!polymorphic_[target] && spaces_[target] != 0 &&
                    combined != spaces_[target]) {
                    return false;
                }
                if (combined != spaces_[target]) {
                    spaces_[target] = combined;
                    changed = true;
                }
            }
        }
        return true;
    }

    std::size_t find(std::size_t node) {
        if (parents_[node] != node) parents_[node] = find(parents_[node]);
        return parents_[node];
    }

    bool unite(std::size_t left, std::size_t right) {
        left = find(left);
        right = find(right);
        if (left == right) return true;
        if (spaces_[left] != 0 && spaces_[right] != 0 &&
            spaces_[left] != spaces_[right] &&
            !polymorphic_[left] && !polymorphic_[right]) {
            return false;
        }
        if (ranks_[left] < ranks_[right]) std::swap(left, right);
        parents_[right] = left;
        if (ranks_[left] == ranks_[right]) ++ranks_[left];
        spaces_[left] |= spaces_[right];
        polymorphic_[left] = polymorphic_[left] || polymorphic_[right];
        return true;
    }

    bool seed(std::size_t node, ir::AddressSpace space) {
        node = find(node);
        if (space == ir::AddressSpace::kNone) return true;
        const std::uint8_t bit =
            static_cast<std::uint8_t>(1u << static_cast<unsigned>(space));
        if (spaces_[node] != 0 && (spaces_[node] & bit) == 0 &&
            !polymorphic_[node]) {
            return false;
        }
        spaces_[node] |= bit;
        return true;
    }

    void mark_polymorphic(std::size_t node) {
        polymorphic_[find(node)] = true;
    }

    std::uint8_t mask(std::size_t node) {
        return spaces_[find(node)];
    }

    std::optional<ir::AddressSpace> space(std::size_t node) {
        const std::uint8_t value = mask(node);
        if (value == 0 || (value & (value - 1)) != 0) return std::nullopt;
        for (unsigned bit = 1; bit <= static_cast<unsigned>(ir::AddressSpace::kPrivate);
             ++bit) {
            if (value == (1u << bit)) return static_cast<ir::AddressSpace>(bit);
        }
        return std::nullopt;
    }

private:
    std::vector<std::size_t> parents_;
    std::vector<std::uint8_t> ranks_;
    std::vector<std::uint8_t> spaces_;
    std::vector<bool> polymorphic_;
    std::vector<std::pair<std::size_t, std::size_t>> flows_;
};

AddressSpaceResolution resolve_generic_address_spaces(ir::Module* module) {
    AddressSpaceConstraints constraints;
    std::unordered_map<ir::ValueId, std::size_t> value_nodes;
    std::unordered_map<ir::ValueId, ir::AddressSpace> concrete_value_spaces;
    std::vector<std::optional<std::size_t>> return_nodes(module->functions.size());
    std::unordered_map<std::string, std::size_t> function_indices;

    auto add_value = [&](ir::ValueId value) {
        if (!value_nodes.contains(value)) value_nodes[value] = constraints.add_node();
    };
    for (std::size_t function_index = 0; function_index < module->functions.size();
         ++function_index) {
        ir::Function& function = module->functions[function_index];
        function.mixed_pointer_address_spaces.clear();
        function_indices[function.name] = function_index;
        if (function.return_type.is_pointer()) {
            return_nodes[function_index] = constraints.add_node();
            if (function.generic_pointer_return) {
                constraints.mark_polymorphic(*return_nodes[function_index]);
            }
            if (!function.generic_pointer_return &&
                !constraints.seed(*return_nodes[function_index],
                                  function.return_type.address_space)) {
                return {false, "conflicting concrete return address spaces in '" +
                                   function.name + "'"};
            }
        }
        for (const ir::FunctionArgument& argument : function.arguments) {
            if (!argument.type.is_pointer()) continue;
            add_value(argument.value);
            if (function.generic_pointer_values.contains(argument.value)) {
                constraints.mark_polymorphic(value_nodes.at(argument.value));
            }
            if (!function.generic_pointer_values.contains(argument.value) &&
                !constraints.seed(value_nodes.at(argument.value),
                                  argument.type.address_space)) {
                return {false, "conflicting concrete argument address spaces in '" +
                                   function.name + "'"};
            }
            if (!function.generic_pointer_values.contains(argument.value)) {
                concrete_value_spaces[argument.value] = argument.type.address_space;
            }
        }
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments) {
                if (!argument.type.is_pointer()) continue;
                add_value(argument.value);
                if (function.generic_pointer_values.contains(argument.value)) {
                    constraints.mark_polymorphic(value_nodes.at(argument.value));
                }
                if (!function.generic_pointer_values.contains(argument.value) &&
                    !constraints.seed(value_nodes.at(argument.value),
                                      argument.type.address_space)) {
                    return {false, "conflicting block-argument address spaces in '" +
                                       function.name + "'"};
                }
                if (!function.generic_pointer_values.contains(argument.value)) {
                    concrete_value_spaces[argument.value] = argument.type.address_space;
                }
            }
            for (const ir::Operation& operation : block.operations) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i >= operation.result_types.size() ||
                        !operation.result_types[i].is_pointer()) {
                        continue;
                    }
                    add_value(operation.results[i]);
                    if (function.generic_pointer_values.contains(operation.results[i])) {
                        constraints.mark_polymorphic(value_nodes.at(operation.results[i]));
                    }
                    const bool concrete =
                        !function.generic_pointer_values.contains(operation.results[i]) ||
                        operation.opcode == ir::OpCode::kAlloca ||
                        (operation.attributes.contains("pointer_integer_concrete") &&
                         operation.attributes.at("pointer_integer_concrete") == "true");
                    const ir::AddressSpace seed =
                        operation.opcode == ir::OpCode::kAlloca
                            ? ir::AddressSpace::kPrivate
                            : operation.result_types[i].address_space;
                    if (concrete &&
                        !constraints.seed(value_nodes.at(operation.results[i]), seed)) {
                        return {false, "conflicting result address spaces in '" +
                                           function.name + "'"};
                    }
                    if (concrete) concrete_value_spaces[operation.results[i]] = seed;
                }
            }
        }
    }

    auto constrain_operand = [&](std::size_t node, const ir::Operand& operand) {
        if (operand.kind == ir::OperandKind::kValue &&
            value_nodes.contains(operand.value)) {
            constraints.flow(value_nodes.at(operand.value), node);
            return true;
        }
        if (operand.kind == ir::OperandKind::kSymbol && operand.type.is_pointer()) {
            return constraints.seed(node, operand.type.address_space);
        }
        return true;
    };

    // Preserve singleton provenance through address-preserving pointer operations.
    // Equality constraints alone deliberately merge at a generic PHI; without this
    // directional fact the merge would incorrectly turn its concrete sources into
    // mixed pointers as well.
    bool propagated_concrete_space = true;
    while (propagated_concrete_space) {
        propagated_concrete_space = false;
        for (ir::Function& function : module->functions) {
            for (const ir::BasicBlock& block : function.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.results.empty() || operation.result_types.empty() ||
                        !operation.result_types.front().is_pointer() ||
                        concrete_value_spaces.contains(operation.results.front()) ||
                        (operation.opcode != ir::OpCode::kPointerOffset &&
                         operation.opcode != ir::OpCode::kConvert &&
                         operation.opcode != ir::OpCode::kAddressSpaceCast) ||
                        operation.operands.empty()) {
                        continue;
                    }
                    std::optional<ir::AddressSpace> source_space;
                    const ir::Operand& source = operation.operands.front();
                    if (source.kind == ir::OperandKind::kSymbol &&
                        source.type.is_pointer() &&
                        source.type.address_space != ir::AddressSpace::kNone) {
                        source_space = source.type.address_space;
                    } else if (source.kind == ir::OperandKind::kValue) {
                        const auto concrete = concrete_value_spaces.find(source.value);
                        if (concrete != concrete_value_spaces.end()) {
                            source_space = concrete->second;
                        }
                    }
                    if (!source_space.has_value()) continue;
                    concrete_value_spaces[operation.results.front()] = *source_space;
                    if (!constraints.seed(value_nodes.at(operation.results.front()),
                                          *source_space)) {
                        return AddressSpaceResolution{
                            false,
                            "address-preserving pointer operation changes concrete address space in '" +
                                function.name + "'",
                        };
                    }
                    propagated_concrete_space = true;
                }
            }
        }
    }

    for (std::size_t function_index = 0; function_index < module->functions.size();
         ++function_index) {
        ir::Function& function = module->functions[function_index];
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                if (!operation.results.empty() &&
                    value_nodes.contains(operation.results.front())) {
                    const std::size_t result_node =
                        value_nodes.at(operation.results.front());
                    if ((operation.opcode == ir::OpCode::kPointerOffset ||
                         operation.opcode == ir::OpCode::kConvert ||
                         operation.opcode == ir::OpCode::kAddressSpaceCast) &&
                        !operation.operands.empty() &&
                        operation.operands.front().type.is_pointer() &&
                        !constrain_operand(result_node, operation.operands.front())) {
                        return {false, "generic pointer changes address space in '" +
                                           function.name + "'"};
                    }
                    if (operation.attributes.contains("pointer_source_value")) {
                        const ir::ValueId source = static_cast<ir::ValueId>(
                            std::stoul(operation.attributes.at("pointer_source_value")));
                        if (!value_nodes.contains(source)) {
                            return {false,
                                    "pointer integer round-trip changes address space in '" +
                                        function.name + "'"};
                        }
                        constraints.flow(value_nodes.at(source), result_node);
                    }
                    if (operation.opcode == ir::OpCode::kSelect) {
                        for (std::size_t i = 1; i < operation.operands.size(); ++i) {
                            if (operation.operands[i].type.is_pointer() &&
                                !constrain_operand(result_node, operation.operands[i])) {
                                return {false, "select merges incompatible pointer address spaces in '" +
                                                   function.name + "'"};
                            }
                        }
                    }
                }
                if (operation.opcode == ir::OpCode::kCall) {
                    const auto callee_name = operation.attributes.find("callee");
                    const auto callee_index =
                        callee_name == operation.attributes.end()
                            ? function_indices.end()
                            : function_indices.find(callee_name->second);
                    if (callee_index != function_indices.end()) {
                        ir::Function& callee = module->functions[callee_index->second];
                        const std::size_t count =
                            std::min(operation.operands.size(), callee.arguments.size());
                        for (std::size_t i = 0; i < count; ++i) {
                            if (!callee.arguments[i].type.is_pointer()) continue;
                            if (!constrain_operand(
                                    value_nodes.at(callee.arguments[i].value),
                                    operation.operands[i])) {
                                return {false, "device helper '" + callee.name +
                                                   "' requires address-space specialization"};
                            }
                        }
                        if (!operation.results.empty() &&
                            value_nodes.contains(operation.results.front()) &&
                            return_nodes[callee_index->second].has_value()) {
                            constraints.flow(*return_nodes[callee_index->second],
                                             value_nodes.at(operation.results.front()));
                        }
                    }
                }
                if (operation.opcode == ir::OpCode::kReturn &&
                    return_nodes[function_index].has_value() &&
                    !operation.operands.empty() &&
                    !constrain_operand(*return_nodes[function_index],
                                       operation.operands.front())) {
                    return {false, "return merges incompatible pointer address spaces in '" +
                                       function.name + "'"};
                }
            }
            const ir::Operation& terminator = block.operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                ir::BasicBlock* target = function.find_block(successor.block);
                if (target == nullptr) continue;
                const std::size_t count =
                    std::min(successor.arguments.size(), target->arguments.size());
                for (std::size_t i = 0; i < count; ++i) {
                    if (!target->arguments[i].type.is_pointer() ||
                        !value_nodes.contains(successor.arguments[i])) {
                        continue;
                    }
                    if (function.generic_null_pointer_values.contains(
                            successor.arguments[i])) {
                        constraints.flow(value_nodes.at(target->arguments[i].value),
                                         value_nodes.at(successor.arguments[i]));
                    } else {
                        constraints.flow(value_nodes.at(successor.arguments[i]),
                                         value_nodes.at(target->arguments[i].value));
                    }
                }
            }
        }
    }

    // CUDA permits generic pointers to be stored in ordinary structs. Connect
    // pointer loads and stores through an exact base+constant-offset memory slot.
    // The base uses the already unified interprocedural pointer component, so a
    // field initialized in a constructor is visible to a method called later on
    // the same object without relying on source-level type names.
    std::unordered_map<std::string, std::size_t> pointer_memory_slots;
    for (ir::Function& function : module->functions) {
        auto slot_for = [&](const ir::Operand& address) -> std::optional<std::size_t> {
            if (address.kind != ir::OperandKind::kValue ||
                !value_nodes.contains(address.value)) {
                return std::nullopt;
            }
            const auto provenance = function.pointer_provenance.find(address.value);
            if (provenance == function.pointer_provenance.end() ||
                provenance->second.base_kind == ir::PointerBaseKind::kUnknown ||
                (!provenance->second.known_layout_offset.has_value() &&
                 !provenance->second.known_byte_offset.has_value())) {
                return std::nullopt;
            }
            const std::int64_t slot_offset =
                provenance->second.known_layout_offset.has_value()
                    ? *provenance->second.known_layout_offset
                    : *provenance->second.known_byte_offset;
            const std::string key =
                (provenance->second.memory_layout.empty()
                     ? "root:" + std::to_string(
                                     constraints.find(value_nodes.at(address.value)))
                     : "layout:" + provenance->second.memory_layout) +
                ":" + std::to_string(slot_offset);
            const auto existing = pointer_memory_slots.find(key);
            if (existing != pointer_memory_slots.end()) return existing->second;
            const std::size_t node = constraints.add_node();
            constraints.mark_polymorphic(node);
            pointer_memory_slots.emplace(key, node);
            return node;
        };
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                if (operation.opcode == ir::OpCode::kStore &&
                    operation.operands.size() >= 2 &&
                    operation.operands[1].type.is_pointer()) {
                    const auto slot = slot_for(operation.operands[0]);
                    if (slot.has_value() &&
                        !constrain_operand(*slot, operation.operands[1])) {
                        return {false, "pointer field stores incompatible concrete address spaces in '" +
                                           function.name + "'"};
                    }
                } else if (operation.opcode == ir::OpCode::kLoad &&
                           !operation.results.empty() &&
                           !operation.result_types.empty() &&
                           operation.result_types.front().is_pointer() &&
                           !operation.operands.empty()) {
                    const auto slot = slot_for(operation.operands[0]);
                    if (slot.has_value()) {
                        constraints.flow(*slot,
                                         value_nodes.at(operation.results.front()));
                    }
                }
            }
        }
    }

    if (!constraints.solve()) {
        return {false,
                "directional pointer flow reaches a conflicting concrete address space"};
    }

    auto resolve_type = [&](ir::Function* function, ir::ValueId value, ir::Type* type,
                            std::string_view context) -> std::optional<std::string> {
        if (!type->is_pointer() || !value_nodes.contains(value)) return std::nullopt;
        if (const auto concrete = concrete_value_spaces.find(value);
            concrete != concrete_value_spaces.end()) {
            type->address_space = concrete->second;
            return std::nullopt;
        }
        const std::uint8_t mask = constraints.mask(value_nodes.at(value));
        const auto space = constraints.space(value_nodes.at(value));
        if (!space.has_value()) {
            if (mask != 0 && (mask & (mask - 1)) != 0) {
                type->address_space = ir::AddressSpace::kNone;
                function->mixed_pointer_address_spaces[value] = mask;
                return std::nullopt;
            }
            std::string detail;
            if (const auto provenance = function->pointer_provenance.find(value);
                provenance != function->pointer_provenance.end()) {
                detail = " (layout='" + provenance->second.memory_layout + "', offset=" +
                         (provenance->second.known_byte_offset.has_value()
                              ? std::to_string(*provenance->second.known_byte_offset)
                              : std::string("unknown")) +
                         ")";
            }
            return "unresolved generic pointer address space for " +
                   std::string(context) + " value " + std::to_string(value) + detail;
        }
        type->address_space = *space;
        return std::nullopt;
    };

    for (std::size_t function_index = 0; function_index < module->functions.size();
         ++function_index) {
        ir::Function& function = module->functions[function_index];
        if (return_nodes[function_index].has_value()) {
            const auto space = constraints.space(*return_nodes[function_index]);
            if (!space.has_value()) {
                return {false, "unresolved generic pointer return address space in '" +
                                   function.name + "'"};
            }
            function.return_type.address_space = *space;
        }
        for (ir::FunctionArgument& argument : function.arguments) {
            if (const auto error = resolve_type(&function, argument.value, &argument.type,
                                                function.name + " argument")) {
                return {false, *error};
            }
        }
        for (ir::BasicBlock& block : function.blocks) {
            for (ir::BlockArgument& argument : block.arguments) {
                if (const auto error = resolve_type(&function, argument.value, &argument.type,
                                                    function.name + " block argument")) {
                    return {false, *error};
                }
            }
            for (ir::Operation& operation : block.operations) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i < operation.result_types.size()) {
                        if (const auto error = resolve_type(
                                &function, operation.results[i], &operation.result_types[i],
                                function.name + " result")) {
                            return {false, *error};
                        }
                    }
                }
                for (ir::Operand& operand : operation.operands) {
                    if (operand.kind == ir::OperandKind::kValue &&
                        value_nodes.contains(operand.value) && operand.type.is_pointer()) {
                        if (const auto concrete = concrete_value_spaces.find(operand.value);
                            concrete != concrete_value_spaces.end()) {
                            operand.type.address_space = concrete->second;
                            continue;
                        }
                        const auto space = constraints.space(value_nodes.at(operand.value));
                        if (!space.has_value()) {
                            const std::uint8_t mask =
                                constraints.mask(value_nodes.at(operand.value));
                            if (mask != 0 && (mask & (mask - 1)) != 0) {
                                operand.type.address_space = ir::AddressSpace::kNone;
                                function.mixed_pointer_address_spaces[operand.value] = mask;
                                continue;
                            }
                            return {false, "unresolved pointer operand address space in '" +
                                               function.name + "'"};
                        }
                        operand.type.address_space = *space;
                    }
                }
            }
        }
        if (function.kernel_abi.has_value()) {
            for (std::size_t i = 0;
                 i < function.arguments.size() &&
                 i < function.kernel_abi->arguments.size();
                 ++i) {
                if (function.arguments[i].type.is_pointer()) {
                    function.kernel_abi->arguments[i].type = function.arguments[i].type;
                    function.kernel_abi->arguments[i].address_space =
                        function.arguments[i].type.address_space;
                }
            }
        }
    }
    return {true, {}};
}

struct BuiltinUsage {
    bool thread_position = false;
    bool threadgroup_position = false;
    bool threads_per_threadgroup = false;
    bool threadgroups_per_grid = false;
    bool lane_id = false;

    bool merge(const BuiltinUsage& other) {
        const BuiltinUsage before = *this;
        thread_position = thread_position || other.thread_position;
        threadgroup_position = threadgroup_position || other.threadgroup_position;
        threads_per_threadgroup = threads_per_threadgroup || other.threads_per_threadgroup;
        threadgroups_per_grid = threadgroups_per_grid || other.threadgroups_per_grid;
        lane_id = lane_id || other.lane_id;
        return thread_position != before.thread_position ||
               threadgroup_position != before.threadgroup_position ||
               threads_per_threadgroup != before.threads_per_threadgroup ||
               threadgroups_per_grid != before.threadgroups_per_grid ||
               lane_id != before.lane_id;
    }
};

using BuiltinUsageMap = std::unordered_map<std::string, BuiltinUsage>;
using SharedUsageMap =
    std::unordered_map<std::string, std::vector<std::string>>;
using BarrierUsageMap = std::unordered_map<std::string, bool>;

std::string shared_parameter_name(std::string_view global) {
    return "cm_shared_" + sanitize_identifier(global);
}

void collect_msl_struct(const ir::Type& type, std::unordered_set<std::string>* seen,
                        std::vector<MslStruct>* structs) {
    for (const ir::Type& element : type.elements) {
        collect_msl_struct(element, seen, structs);
    }
    if (type.kind != ir::TypeKind::kAggregate || is_native_vector_aggregate(type)) return;
    const std::string name = aggregate_type_name(type);
    if (!seen->insert(name).second) return;
    MslStruct structure;
    structure.name = name;
    for (std::size_t i = 0; i < type.elements.size(); ++i) {
        structure.fields.push_back({
            .type = lower_type(type.elements[i]),
            .name = "field" + std::to_string(i),
        });
    }
    structs->push_back(std::move(structure));
}

std::vector<MslStruct> collect_msl_structs(const ir::Module& module) {
    std::unordered_set<std::string> seen;
    std::vector<MslStruct> structs;
    for (const ir::Function& function : module.functions) {
        collect_msl_struct(function.return_type, &seen, &structs);
        for (const ir::FunctionArgument& argument : function.arguments) {
            collect_msl_struct(argument.type, &seen, &structs);
        }
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments) {
                collect_msl_struct(argument.type, &seen, &structs);
            }
            for (const ir::Operation& operation : block.operations) {
                for (const ir::Type& type : operation.result_types) {
                    collect_msl_struct(type, &seen, &structs);
                }
                for (const ir::Operand& operand : operation.operands) {
                    collect_msl_struct(operand.type, &seen, &structs);
                }
            }
        }
    }
    return structs;
}

BuiltinUsageMap analyze_builtin_usage(const ir::Module& module) {
    BuiltinUsageMap usage;
    for (const ir::Function& function : module.functions) {
        BuiltinUsage direct;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                direct.thread_position = direct.thread_position ||
                                         operation.opcode == ir::OpCode::kMetalThreadPosition;
                direct.threadgroup_position = direct.threadgroup_position ||
                                              operation.opcode == ir::OpCode::kMetalThreadgroupPosition;
                direct.threads_per_threadgroup = direct.threads_per_threadgroup ||
                                                 operation.opcode == ir::OpCode::kMetalThreadsPerThreadgroup;
                direct.threadgroups_per_grid = direct.threadgroups_per_grid ||
                                               operation.opcode == ir::OpCode::kMetalThreadgroupsPerGrid;
                direct.lane_id = direct.lane_id ||
                                 operation.opcode == ir::OpCode::kMetalLaneId ||
                                 operation.opcode == ir::OpCode::kMetalShuffle ||
                                 operation.opcode == ir::OpCode::kMetalBallot ||
                                 operation.opcode == ir::OpCode::kMetalVote;
            }
        }
        usage[function.name] = direct;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const ir::Function& function : module.functions) {
            for (const ir::BasicBlock& block : function.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kCall ||
                        operation.attributes.contains("builtin")) {
                        continue;
                    }
                    const auto callee = operation.attributes.find("callee");
                    if (callee == operation.attributes.end()) continue;
                    const auto callee_usage = usage.find(callee->second);
                    if (callee_usage != usage.end()) {
                        changed = usage[function.name].merge(callee_usage->second) || changed;
                    }
                }
            }
        }
    }
    return usage;
}

SharedUsageMap analyze_shared_usage(const ir::Module& module) {
    SharedUsageMap usage;
    for (const ir::Function& function : module.functions) {
        std::unordered_set<std::string> direct;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                for (const ir::Operand& operand : operation.operands) {
                    if (operand.kind == ir::OperandKind::kSymbol &&
                        operand.type.is_pointer() &&
                        operand.type.address_space == ir::AddressSpace::kThreadgroup) {
                        direct.insert(operand.text);
                    }
                }
            }
        }
        usage[function.name] = {direct.begin(), direct.end()};
        std::sort(usage[function.name].begin(), usage[function.name].end());
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (const ir::Function& function : module.functions) {
            std::unordered_set<std::string> merged(
                usage[function.name].begin(), usage[function.name].end());
            for (const ir::BasicBlock& block : function.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kCall) continue;
                    const auto callee = operation.attributes.find("callee");
                    if (callee == operation.attributes.end() ||
                        !usage.contains(callee->second)) {
                        continue;
                    }
                    merged.insert(usage[callee->second].begin(),
                                  usage[callee->second].end());
                }
            }
            std::vector<std::string> next(merged.begin(), merged.end());
            std::sort(next.begin(), next.end());
            if (next != usage[function.name]) {
                usage[function.name] = std::move(next);
                changed = true;
            }
        }
    }
    return usage;
}

BarrierUsageMap analyze_barrier_usage(const ir::Module& module) {
    BarrierUsageMap usage;
    for (const ir::Function& function : module.functions) {
        bool direct = false;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                direct = direct || operation.opcode == ir::OpCode::kMetalBarrier;
            }
        }
        usage[function.name] = direct;
    }
    bool changed = true;
    while (changed) {
        changed = false;
        for (const ir::Function& function : module.functions) {
            if (usage[function.name]) continue;
            for (const ir::BasicBlock& block : function.blocks) {
                for (const ir::Operation& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kCall) continue;
                    const auto callee = operation.attributes.find("callee");
                    if (callee != operation.attributes.end() &&
                        usage.contains(callee->second) && usage.at(callee->second)) {
                        usage[function.name] = true;
                        changed = true;
                        break;
                    }
                }
                if (usage[function.name]) break;
            }
        }
    }
    return usage;
}

struct AstLowerer {
    const ir::Module& module;
    const ir::Function& function;
    const BuiltinUsageMap& builtin_usage;
    const SharedUsageMap& shared_usage;
    LowerToMslResult result;
    MslFunction output;
    std::unordered_map<ir::ValueId, MslExpr> values;
    std::unordered_map<ir::ValueId, MslExpr> mixed_pointer_tags;
    std::unordered_set<ir::ValueId> declared_block_arguments;
    std::unordered_map<ir::BlockId, std::size_t> block_indices;
    std::vector<std::vector<std::size_t>> predecessors;
    std::vector<std::vector<bool>> dominators;
    std::vector<std::vector<bool>> postdominators;
    std::unordered_set<ir::BlockId> emitted;
    std::unordered_set<ir::BlockId> region_stack;
    struct LoopEscapeContext {
        std::size_t enclosing_header_index;
        MslExpr continue_enclosing;
    };
    std::vector<LoopEscapeContext> loop_escape_stack;
    bool needs_thread_position = false;
    bool needs_threadgroup_position = false;
    bool needs_threads_per_threadgroup = false;
    bool needs_threadgroups_per_grid = false;
    bool needs_lane_id = false;
    bool cfg_dispatcher_mode = false;
    bool predeclared_ssa_storage = false;
    bool force_cfg_dispatcher = false;
    std::size_t edge_temporary_index = 0;
    std::size_t loop_escape_index = 0;
    std::optional<ir::AddressSpace> pointer_specialization;

    AstLowerer(const ir::Module& input_module, const ir::Function& input_function,
               const BuiltinUsageMap& input_builtin_usage,
               const SharedUsageMap& input_shared_usage,
               bool force_dispatcher = false,
               std::optional<ir::AddressSpace> specialization = std::nullopt)
        : module(input_module),
          function(input_function),
          builtin_usage(input_builtin_usage),
          shared_usage(input_shared_usage),
          force_cfg_dispatcher(force_dispatcher),
          pointer_specialization(specialization) {
        const BuiltinUsage& required = builtin_usage.at(function.name);
        needs_thread_position = required.thread_position;
        needs_threadgroup_position = required.threadgroup_position;
        needs_threads_per_threadgroup = required.threads_per_threadgroup;
        needs_threadgroups_per_grid = required.threadgroups_per_grid;
        needs_lane_id = required.lane_id;
    }

    bool fail(const ir::Operation* operation, std::string message) {
        if (operation != nullptr && !operation->location.str().empty()) {
            message = operation->location.str() + ": " + message;
        }
        result.error = std::move(message);
        return false;
    }

    bool is_mixed_pointer(ir::ValueId value) const {
        return function.mixed_pointer_address_spaces.contains(value);
    }

    MslType lower_value_type(ir::ValueId value, const ir::Type& type) const {
        if (!type.is_pointer() || !is_mixed_pointer(value)) return lower_type(type);
        if (pointer_specialization.has_value()) {
            ir::Type specialized = type;
            specialized.address_space = *pointer_specialization;
            return lower_type(specialized);
        }
        return MslType::uint(64);
    }

    MslType lower_result_type(const ir::Operation& operation,
                              std::size_t index = 0) const {
        if (index >= operation.results.size() || index >= operation.result_types.size()) {
            return MslType::void_type();
        }
        return lower_value_type(operation.results[index], operation.result_types[index]);
    }

    std::string specialized_callee(std::string_view callee,
                                   ir::AddressSpace space) const {
        std::string suffix;
        switch (space) {
            case ir::AddressSpace::kDevice: suffix = "__cm_device"; break;
            case ir::AddressSpace::kConstant: suffix = "__cm_constant"; break;
            case ir::AddressSpace::kThreadgroup: suffix = "__cm_threadgroup"; break;
            case ir::AddressSpace::kPrivate: suffix = "__cm_thread"; break;
            case ir::AddressSpace::kNone: suffix = "__cm_generic"; break;
        }
        return std::string(callee) + suffix;
    }

    MslExpr expression_for(const ir::Operand& operand) {
        if (operand.kind == ir::OperandKind::kValue) {
            const auto value = values.find(operand.value);
            if (value != values.end()) return value->second;
            return MslExpression::identifier(
                value_name(operand.value),
                lower_value_type(operand.value, operand.type));
        }
        if (operand.kind == ir::OperandKind::kSymbol) {
            if (operand.type.is_pointer() &&
                operand.type.address_space == ir::AddressSpace::kThreadgroup) {
                return MslExpression::identifier(shared_parameter_name(operand.text),
                                                 lower_type(operand.type));
            }
            return MslExpression::identifier(operand.text, lower_type(operand.type));
        }
        return MslExpression::literal(operand.text == "null" ? "nullptr" : operand.text,
                                      lower_type(operand.type));
    }

    MslStmt declare_result(const ir::Operation& operation, MslExpr initializer) {
        const ir::ValueId value = operation.results.front();
        const MslType type = lower_result_type(operation);
        values[value] = MslExpression::identifier(value_name(value), type);
        if (cfg_dispatcher_mode || predeclared_ssa_storage) {
            return MslStatement::assignment(values.at(value), std::move(initializer));
        }
        return MslStatement::variable(type, value_name(value), std::move(initializer), true);
    }

    std::optional<MslStmt> lower_operation(const ir::Operation& operation) {
        if (operation.opcode == ir::OpCode::kReturn ||
            operation.opcode == ir::OpCode::kBranch ||
            operation.opcode == ir::OpCode::kCondBranch) {
            return std::nullopt;
        }
        if (operation.attributes.contains("guard_operand")) {
            fail(&operation, "predicated non-branch operations require structured guard lowering");
            return std::nullopt;
        }

        if (operation.opcode == ir::OpCode::kParameter) {
            if (operation.results.size() != 1 || operation.operands.size() != 1) {
                fail(&operation, "malformed parameter operation");
                return std::nullopt;
            }
            values[operation.results.front()] = expression_for(operation.operands.front());
            return std::nullopt;
        }

        if (operation.opcode == ir::OpCode::kMetalThreadPosition ||
            operation.opcode == ir::OpCode::kMetalThreadgroupPosition ||
            operation.opcode == ir::OpCode::kMetalThreadsPerThreadgroup ||
            operation.opcode == ir::OpCode::kMetalThreadgroupsPerGrid) {
            std::string builtin;
            if (operation.opcode == ir::OpCode::kMetalThreadPosition) {
                builtin = "cm_thread_position";
                needs_thread_position = true;
            } else if (operation.opcode == ir::OpCode::kMetalThreadgroupPosition) {
                builtin = "cm_threadgroup_position";
                needs_threadgroup_position = true;
            } else if (operation.opcode == ir::OpCode::kMetalThreadsPerThreadgroup) {
                builtin = "cm_threads_per_threadgroup";
                needs_threads_per_threadgroup = true;
            } else {
                builtin = "cm_threadgroups_per_grid";
                needs_threadgroups_per_grid = true;
            }
            const MslExpr base =
                MslExpression::identifier(builtin, MslType::vector(MslType::uint(), 3));
            const MslExpr member =
                MslExpression::member(base, dimension_member(operation), MslType::uint());
            return declare_result(operation, member);
        }

        if (operation.opcode == ir::OpCode::kMetalLaneId) {
            needs_lane_id = true;
            return declare_result(
                operation,
                MslExpression::identifier("cm_lane_id", MslType::uint()));
        }

        const std::string binary = binary_spelling(operation.opcode);
        if (!binary.empty()) {
            if (operation.results.size() != 1 || operation.operands.size() < 2) {
                fail(&operation, "malformed binary operation");
                return std::nullopt;
            }
            MslExpr left = expression_for(operation.operands[0]);
            MslExpr right = expression_for(operation.operands[1]);
            MslType expression_type = lower_result_type(operation);
            if (operation.attributes.contains("signed") &&
                operation.attributes.at("signed") == "true" &&
                operation.operands[0].type.kind == ir::TypeKind::kInteger) {
                const MslType signed_type =
                    MslType::sint(operation.operands[0].type.bit_width);
                left = MslExpression::cast(signed_type, left);
                right = MslExpression::cast(signed_type, right);
                expression_type = signed_type;
            }
            MslExpr expression;
            if (operation.opcode == ir::OpCode::kPointerOffset &&
                operation.attributes.contains("offset_unit") &&
                operation.attributes.at("offset_unit") == "bytes" &&
                !is_mixed_pointer(operation.results.front())) {
                // CuMetal pointer offsets are byte offsets even when the source
                // pointer originated from an aggregate alloca. Cast before the
                // addition so C++/MSL cannot scale the offset by the aggregate's
                // sizeof (for example, `&vec3_storage + 4`).
                const MslAddressSpace address_space =
                    expression_type.kind == MslTypeKind::kPointer
                        ? expression_type.address_space
                        : left->type.address_space;
                const MslType byte_pointer = MslType::pointer(
                    MslType::uint(8), address_space);
                const MslExpr byte_base = MslExpression::cast(
                    byte_pointer, left, true);
                const MslExpr byte_offset = MslExpression::binary(
                    binary, byte_base, right, byte_pointer);
                expression = MslExpression::cast(
                    expression_type, byte_offset, true);
            } else {
                expression = MslExpression::binary(
                    binary, left, right, expression_type);
            }
            if (operation.attributes.contains("combined") &&
                operation.attributes.at("combined") == "mul_add" &&
                operation.operands.size() >= 3) {
                expression = MslExpression::binary(
                    "+", expression, expression_for(operation.operands[2]),
                    lower_result_type(operation));
            }
            return declare_result(operation, expression);
        }

        if (operation.opcode == ir::OpCode::kFma) {
            std::vector<MslExpr> arguments;
            for (const ir::Operand& operand : operation.operands) {
                arguments.push_back(expression_for(operand));
            }
            return declare_result(
                operation,
                MslExpression::call("fma", std::move(arguments),
                                    lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kAggregateExtract) {
            if (operation.results.empty() || operation.operands.size() != 2) {
                fail(&operation, "malformed aggregate extraction");
                return std::nullopt;
            }
            return declare_result(
                operation,
                MslExpression::subscript(expression_for(operation.operands[0]),
                                         expression_for(operation.operands[1]),
                                         lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kAggregateConstruct) {
            const auto constructor = operation.attributes.find("constructor");
            if (operation.results.empty() || constructor == operation.attributes.end()) {
                fail(&operation, "malformed aggregate construction");
                return std::nullopt;
            }
            std::vector<MslExpr> elements;
            elements.reserve(operation.operands.size());
            for (const ir::Operand& operand : operation.operands) {
                elements.push_back(expression_for(operand));
            }
            return declare_result(
                operation,
                MslExpression::call(constructor->second, std::move(elements),
                                    lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kCall) {
            const auto callee = operation.attributes.find("callee");
            if (callee == operation.attributes.end()) {
                fail(&operation, "direct call is missing a callee");
                return std::nullopt;
            }
            if (callee->second == "__cumetal_signed_abs") {
                if (operation.results.empty() || operation.operands.size() != 1) {
                    fail(&operation, "malformed CUDA signed abs builtin");
                    return std::nullopt;
                }
                const MslType signed_type =
                    MslType::sint(operation.operands.front().type.bit_width);
                const MslExpr signed_input = MslExpression::cast(
                    signed_type, expression_for(operation.operands.front()));
                const MslExpr absolute =
                    MslExpression::call("abs", {signed_input}, signed_type);
                return declare_result(
                    operation,
                    MslExpression::cast(lower_result_type(operation), absolute));
            }
            if (callee->second == "__cumetal_ffs") {
                if (operation.results.empty() || operation.operands.size() != 1) {
                    fail(&operation, "malformed CUDA ffs builtin");
                    return std::nullopt;
                }
                const MslType result_type = lower_result_type(operation);
                const MslExpr input = expression_for(operation.operands.front());
                const MslExpr zero = MslExpression::literal("0", input->type);
                const MslExpr is_zero =
                    MslExpression::binary("==", input, zero, MslType::boolean());
                const MslExpr trailing =
                    MslExpression::call("ctz", {input}, result_type);
                const MslExpr one = MslExpression::literal("1", result_type);
                const MslExpr one_based =
                    MslExpression::binary("+", trailing, one, result_type);
                return declare_result(
                    operation,
                    MslExpression::conditional(is_zero, zero, one_based, result_type));
            }
            std::vector<MslExpr> arguments;
            for (const ir::Operand& operand : operation.operands) {
                arguments.push_back(expression_for(operand));
            }
            const auto callee_function = std::find_if(
                module.functions.begin(), module.functions.end(),
                [&](const ir::Function& candidate) {
                    return candidate.name == callee->second;
                });
            if (callee_function != module.functions.end()) {
                const std::size_t count =
                    std::min(arguments.size(), callee_function->arguments.size());
                for (std::size_t i = 0; i < count; ++i) {
                    if (callee_function->mixed_pointer_address_spaces.contains(
                            callee_function->arguments[i].value)) {
                        continue;
                    }
                    const MslType expected =
                        lower_type(callee_function->arguments[i].type);
                    if (arguments[i]->type.kind == MslTypeKind::kPointer &&
                        expected.kind == MslTypeKind::kPointer &&
                        arguments[i]->type != expected) {
                        arguments[i] =
                            MslExpression::cast(expected, arguments[i], true);
                    }
                }
            }
            const auto callee_shared = shared_usage.find(callee->second);
            if (callee_shared != shared_usage.end()) {
                for (const std::string& global : callee_shared->second) {
                    arguments.push_back(MslExpression::identifier(
                        shared_parameter_name(global),
                        MslType::pointer(MslType::uint(8),
                                         MslAddressSpace::kThreadgroup)));
                }
            }
            const auto callee_usage = builtin_usage.find(callee->second);
            if (callee_usage != builtin_usage.end()) {
                const BuiltinUsage& required = callee_usage->second;
                if (required.thread_position) {
                    arguments.push_back(MslExpression::identifier(
                        "cm_thread_position", MslType::vector(MslType::uint(), 3)));
                }
                if (required.threadgroup_position) {
                    arguments.push_back(MslExpression::identifier(
                        "cm_threadgroup_position", MslType::vector(MslType::uint(), 3)));
                }
                if (required.threads_per_threadgroup) {
                    arguments.push_back(MslExpression::identifier(
                        "cm_threads_per_threadgroup", MslType::vector(MslType::uint(), 3)));
                }
                if (required.threadgroups_per_grid) {
                    arguments.push_back(MslExpression::identifier(
                        "cm_threadgroups_per_grid", MslType::vector(MslType::uint(), 3)));
                }
                if (required.lane_id) {
                    arguments.push_back(MslExpression::identifier(
                        "cm_lane_id", MslType::uint()));
                }
            }
            const MslType return_type = operation.result_types.empty()
                                            ? MslType::void_type()
                                            : lower_result_type(operation);
            const bool polymorphic_callee =
                callee_function != module.functions.end() &&
                !callee_function->mixed_pointer_address_spaces.empty();
            std::optional<std::size_t> mixed_argument;
            if (!pointer_specialization.has_value()) {
                for (std::size_t i = 0; i < operation.operands.size(); ++i) {
                    if (operation.operands[i].kind == ir::OperandKind::kValue &&
                        is_mixed_pointer(operation.operands[i].value)) {
                        mixed_argument = i;
                        break;
                    }
                }
            }
            if (polymorphic_callee && mixed_argument.has_value()) {
                if (operation.results.empty()) {
                    fail(&operation,
                         "void calls through mixed CUDA pointers require statement dispatch");
                    return std::nullopt;
                }
                auto specialized_arguments = [&](ir::AddressSpace space) {
                    std::vector<MslExpr> result = arguments;
                    const std::size_t count = std::min(
                        operation.operands.size(), callee_function->arguments.size());
                    for (std::size_t i = 0; i < count; ++i) {
                        if (operation.operands[i].kind != ir::OperandKind::kValue ||
                            !is_mixed_pointer(operation.operands[i].value)) {
                            continue;
                        }
                        ir::Type pointer_type = callee_function->arguments[i].type;
                        pointer_type.address_space = space;
                        result[i] = MslExpression::cast(
                            lower_type(pointer_type), arguments[i], true);
                    }
                    return result;
                };
                const MslExpr device_call = MslExpression::call(
                    specialized_callee(callee->second, ir::AddressSpace::kDevice),
                    specialized_arguments(ir::AddressSpace::kDevice), return_type);
                const MslExpr threadgroup_call = MslExpression::call(
                    specialized_callee(callee->second, ir::AddressSpace::kThreadgroup),
                    specialized_arguments(ir::AddressSpace::kThreadgroup), return_type);
                const ir::ValueId mixed_value =
                    operation.operands[*mixed_argument].value;
                const MslExpr is_device = MslExpression::binary(
                    "==", mixed_pointer_tags.at(mixed_value),
                    MslExpression::literal(
                        std::to_string(static_cast<unsigned>(ir::AddressSpace::kDevice)) +
                            "u",
                        MslType::uint()),
                    MslType::boolean());
                return declare_result(
                    operation,
                    MslExpression::conditional(is_device, device_call,
                                               threadgroup_call, return_type));
            }
            if (polymorphic_callee) {
                std::optional<ir::AddressSpace> concrete_specialization;
                const std::size_t count = std::min(
                    operation.operands.size(), callee_function->arguments.size());
                for (std::size_t i = 0; i < count; ++i) {
                    if (!callee_function->mixed_pointer_address_spaces.contains(
                            callee_function->arguments[i].value) ||
                        arguments[i]->type.kind != MslTypeKind::kPointer) {
                        continue;
                    }
                    ir::AddressSpace space = ir::AddressSpace::kNone;
                    if (arguments[i]->type.address_space == MslAddressSpace::kDevice) {
                        space = ir::AddressSpace::kDevice;
                    } else if (arguments[i]->type.address_space ==
                               MslAddressSpace::kConstant) {
                        space = ir::AddressSpace::kConstant;
                    } else if (arguments[i]->type.address_space ==
                               MslAddressSpace::kThreadgroup) {
                        space = ir::AddressSpace::kThreadgroup;
                    } else if (arguments[i]->type.address_space ==
                               MslAddressSpace::kThread) {
                        space = ir::AddressSpace::kPrivate;
                    }
                    if (space == ir::AddressSpace::kNone) continue;
                    if (concrete_specialization.has_value() &&
                        *concrete_specialization != space) {
                        fail(&operation,
                             "polymorphic helper call has multiple concrete address spaces");
                        return std::nullopt;
                    }
                    concrete_specialization = space;
                }
                if (concrete_specialization.has_value()) {
                    for (std::size_t i = 0; i < count; ++i) {
                        if (!callee_function->mixed_pointer_address_spaces.contains(
                                callee_function->arguments[i].value)) {
                            continue;
                        }
                        ir::Type expected_type = callee_function->arguments[i].type;
                        expected_type.address_space = *concrete_specialization;
                        arguments[i] = MslExpression::cast(
                            lower_type(expected_type), arguments[i], true);
                    }
                    MslExpr call = MslExpression::call(
                        specialized_callee(callee->second,
                                           *concrete_specialization),
                        std::move(arguments), return_type);
                    if (operation.results.empty()) {
                        return MslStatement::expression(std::move(call));
                    }
                    return declare_result(operation, std::move(call));
                }
            }
            MslExpr call = MslExpression::call(callee->second, std::move(arguments), return_type);
            if (operation.results.empty()) return MslStatement::expression(std::move(call));
            return declare_result(operation, std::move(call));
        }

        if (operation.opcode == ir::OpCode::kNegate) {
            return declare_result(
                operation,
                MslExpression::unary("-", expression_for(operation.operands.front()),
                                     lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kCompare) {
            MslExpr left = expression_for(operation.operands[0]);
            MslExpr right = expression_for(operation.operands[1]);
            if (operation.attributes.contains("signed") &&
                operation.attributes.at("signed") == "true" &&
                operation.operands[0].type.kind == ir::TypeKind::kInteger) {
                const MslType signed_type =
                    MslType::sint(operation.operands[0].type.bit_width);
                left = MslExpression::cast(signed_type, left);
                right = MslExpression::cast(signed_type, right);
            }
            return declare_result(
                operation,
                MslExpression::binary(
                    compare_spelling(operation), left, right, MslType::boolean()));
        }

        if (operation.opcode == ir::OpCode::kSelect) {
            return declare_result(
                operation,
                MslExpression::conditional(
                    expression_for(operation.operands[0]),
                    expression_for(operation.operands[1]),
                    expression_for(operation.operands[2]),
                    lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kConvert ||
            operation.opcode == ir::OpCode::kAddressSpaceCast) {
            if (operation.results.empty() || operation.operands.empty()) {
                fail(&operation, "malformed conversion");
                return std::nullopt;
            }
            const bool reinterpret =
                operation.opcode == ir::OpCode::kAddressSpaceCast &&
                operation.operands.front().type.is_pointer();
            if (operation.operands.front().kind == ir::OperandKind::kImmediate &&
                operation.operands.front().text == "null" &&
                operation.result_types.front().is_pointer()) {
                return declare_result(
                    operation,
                    MslExpression::literal("nullptr", lower_result_type(operation)));
            }
            if (operation.operands.front().type == operation.result_types.front()) {
                return declare_result(operation,
                                      expression_for(operation.operands.front()));
            }
            if (operation.attributes.contains("bitcast") &&
                operation.attributes.at("bitcast") == "true") {
                return declare_result(
                    operation,
                    MslExpression::bitcast(lower_result_type(operation),
                                           expression_for(operation.operands.front())));
            }
            if (operation.attributes.contains("pointer_integer") &&
                operation.attributes.at("pointer_integer") == "true") {
                return declare_result(
                    operation,
                    MslExpression::cast(lower_result_type(operation),
                                        expression_for(operation.operands.front()), true));
            }
            return declare_result(
                operation,
                MslExpression::cast(lower_result_type(operation),
                                    expression_for(operation.operands.front()), reinterpret));
        }

        if (operation.opcode == ir::OpCode::kAlloca) {
            if (operation.results.size() != 1 || operation.result_types.size() != 1 ||
                !operation.result_types.front().is_pointer() ||
                operation.result_types.front().pointee() == nullptr) {
                fail(&operation, "malformed thread-local allocation");
                return std::nullopt;
            }
            const ir::ValueId result_value = operation.results.front();
            const MslType storage_type =
                lower_type(*operation.result_types.front().pointee());
            const std::string storage_name = value_name(result_value) + "_storage";
            const MslExpr storage =
                MslExpression::identifier(storage_name, storage_type);
            values[result_value] = MslExpression::unary(
                "&", storage, lower_result_type(operation));
            if (cfg_dispatcher_mode || predeclared_ssa_storage) {
                return std::nullopt;
            }
            return MslStatement::variable(storage_type, storage_name, std::nullopt, false);
        }

        if (operation.opcode == ir::OpCode::kLoad) {
            if (operation.results.empty() || operation.operands.empty()) {
                fail(&operation, "malformed load");
                return std::nullopt;
            }
            const MslType value_type = lower_result_type(operation);
            const MslExpr source_pointer = expression_for(operation.operands.front());
            const MslAddressSpace address_space =
                source_pointer->type.kind == MslTypeKind::kPointer
                    ? source_pointer->type.address_space
                    : lower_address_space(operation.operands.front().type.address_space);
            const MslType pointer_type = MslType::pointer(value_type, address_space);
            const MslExpr pointer =
                MslExpression::cast(pointer_type, source_pointer, true);
            return declare_result(
                operation,
                MslExpression::unary("*", pointer, value_type));
        }

        if (operation.opcode == ir::OpCode::kStore) {
            if (operation.operands.size() < 2) {
                fail(&operation, "malformed store");
                return std::nullopt;
            }
            const MslExpr stored_value = expression_for(operation.operands[1]);
            const MslExpr destination_pointer = expression_for(operation.operands.front());
            const MslAddressSpace address_space =
                destination_pointer->type.kind == MslTypeKind::kPointer
                    ? destination_pointer->type.address_space
                    : lower_address_space(operation.operands.front().type.address_space);
            const MslType pointer_type =
                MslType::pointer(stored_value->type, address_space);
            const MslExpr pointer =
                MslExpression::cast(pointer_type, destination_pointer, true);
            return MslStatement::assignment(
                MslExpression::unary("*", pointer, stored_value->type), stored_value);
        }

        if (operation.opcode == ir::OpCode::kMetalBarrier) {
            const std::string barrier =
                operation.memory_scope == ir::MemoryScope::kSimdgroup
                    ? "simdgroup_barrier"
                    : "threadgroup_barrier";
            return MslStatement::expression(
                MslExpression::call(
                    barrier,
                    {MslExpression::literal("mem_flags::mem_threadgroup", MslType::uint())},
                    MslType::void_type()));
        }

        if (operation.opcode == ir::OpCode::kMetalFence) {
            std::string flag = operation.memory_scope == ir::MemoryScope::kThreadgroup
                                   ? "mem_flags::mem_threadgroup"
                                   : "mem_flags::mem_device";
            return MslStatement::expression(
                MslExpression::call(
                    "threadgroup_barrier",
                    {MslExpression::literal(flag, MslType::uint())},
                    MslType::void_type()));
        }

        if (operation.opcode == ir::OpCode::kMetalShuffle) {
            if (operation.operands.size() < 2 || operation.results.empty()) {
                fail(&operation, "malformed SIMD shuffle");
                return std::nullopt;
            }
            const std::string intrinsic = "simd_shuffle";
            MslExpr shuffle_index = expression_for(operation.operands[1]);
            if (operation.attributes.contains("kind")) {
                const std::string& kind = operation.attributes.at("kind");
                if ((kind == "index" || kind == "down" || kind == "up") &&
                    operation.operands.size() < 3) {
                    fail(&operation, "PTX SIMD shuffle is missing its clamp/control operand");
                    return std::nullopt;
                }
                if (kind == "index" || kind == "down" || kind == "up") {
                    needs_lane_id = true;
                    const MslType uint_type = MslType::uint();
                    const MslExpr lane = MslExpression::identifier(
                        "cm_lane_id", uint_type);
                    const MslExpr source_or_delta = MslExpression::binary(
                        "&",
                        MslExpression::cast(
                            uint_type, expression_for(operation.operands[1])),
                        MslExpression::literal("31u", uint_type), uint_type);
                    const MslExpr control = MslExpression::cast(
                        uint_type, expression_for(operation.operands[2]));
                    const MslExpr five_bits = MslExpression::literal(
                        "31u", uint_type);
                    const MslExpr segment_mask = MslExpression::binary(
                        "&",
                        MslExpression::binary(
                            ">>", control,
                            MslExpression::literal("8u", uint_type), uint_type),
                        five_bits, uint_type);
                    const MslExpr minimum_lane = MslExpression::binary(
                        "&", lane, segment_mask, uint_type);
                    const MslExpr maximum_lane = MslExpression::binary(
                        "|", minimum_lane,
                        MslExpression::binary("&", control, five_bits, uint_type),
                        uint_type);
                    MslExpr requested_lane;
                    MslExpr valid_lane;
                    if (kind == "index") {
                        requested_lane = MslExpression::binary(
                            "|",
                            MslExpression::binary(
                                "&", source_or_delta,
                                MslExpression::unary("~", segment_mask, uint_type),
                                uint_type),
                            minimum_lane, uint_type);
                        valid_lane = MslExpression::binary(
                            "<=", requested_lane, maximum_lane,
                            MslType::boolean());
                    } else if (kind == "down") {
                        requested_lane = MslExpression::binary(
                            "+", lane, source_or_delta, uint_type);
                        valid_lane = MslExpression::binary(
                            "<=", requested_lane, maximum_lane,
                            MslType::boolean());
                    } else {
                        requested_lane = MslExpression::binary(
                            "-", lane, source_or_delta, uint_type);
                        valid_lane = MslExpression::binary(
                            ">=", lane,
                            MslExpression::binary(
                                "+", minimum_lane, source_or_delta, uint_type),
                            MslType::boolean());
                    }
                    shuffle_index = MslExpression::conditional(
                        std::move(valid_lane),
                        requested_lane, lane, uint_type);
                }
            }
            return declare_result(
                operation,
                MslExpression::call(
                    intrinsic,
                    {expression_for(operation.operands[0]),
                     std::move(shuffle_index)},
                    lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kMetalBallot) {
            if (operation.results.empty()) {
                fail(&operation, "malformed SIMD ballot");
                return std::nullopt;
            }
            const bool active_mask = operation.attributes.contains("kind") &&
                                     operation.attributes.at("kind") == "active_mask";
            MslExpr vote;
            if (active_mask) {
                vote = MslExpression::call("simd_active_threads_mask", {}, MslType{
                    .kind = MslTypeKind::kStruct, .struct_name = "simd_vote"});
            } else {
                if (operation.operands.size() < 2) {
                    fail(&operation, "SIMD ballot requires mask and predicate operands");
                    return std::nullopt;
                }
                const MslExpr lane = MslExpression::identifier("cm_lane_id", MslType::uint());
                const MslExpr lane_bit = MslExpression::binary(
                    "<<", MslExpression::literal("1u", MslType::uint()), lane,
                    MslType::uint());
                const MslExpr selected = MslExpression::binary(
                    "&", expression_for(operation.operands[0]), lane_bit,
                    MslType::uint());
                const MslExpr participates = MslExpression::binary(
                    "!=", selected, MslExpression::literal("0u", MslType::uint()),
                    MslType::boolean());
                const MslExpr predicate = MslExpression::binary(
                    "&&", participates, expression_for(operation.operands[1]),
                    MslType::boolean());
                vote = MslExpression::call("simd_ballot", {predicate}, MslType{
                    .kind = MslTypeKind::kStruct, .struct_name = "simd_vote"});
            }
            return declare_result(operation, MslExpression::vote_mask(vote));
        }

        if (operation.opcode == ir::OpCode::kMetalVote) {
            if (operation.results.empty() || operation.operands.size() < 2) {
                fail(&operation, "malformed SIMD vote");
                return std::nullopt;
            }
            const MslExpr lane = MslExpression::identifier("cm_lane_id", MslType::uint());
            const MslExpr lane_bit = MslExpression::binary(
                "<<", MslExpression::literal("1u", MslType::uint()), lane,
                MslType::uint());
            const MslExpr selected = MslExpression::binary(
                "&", expression_for(operation.operands[0]), lane_bit, MslType::uint());
            const MslExpr participates = MslExpression::binary(
                "!=", selected, MslExpression::literal("0u", MslType::uint()),
                MslType::boolean());
            const bool all = operation.attributes.contains("kind") &&
                             operation.attributes.at("kind") == "all";
            const MslExpr predicate = all
                                          ? MslExpression::binary(
                                                "||",
                                                MslExpression::unary(
                                                    "!", participates, MslType::boolean()),
                                                expression_for(operation.operands[1]),
                                                MslType::boolean())
                                          : MslExpression::binary(
                                                "&&", participates,
                                                expression_for(operation.operands[1]),
                                                MslType::boolean());
            const MslExpr voted = MslExpression::call(
                all ? "simd_all" : "simd_any", {predicate}, MslType::boolean());
            return declare_result(
                operation,
                MslExpression::cast(lower_result_type(operation), voted));
        }

        if (operation.opcode == ir::OpCode::kMetalAtomic) {
            if (operation.results.size() != 1 || operation.result_types.size() != 1 ||
                operation.operands.size() != 2 ||
                operation.result_types.front().kind != ir::TypeKind::kInteger ||
                operation.result_types.front().bit_width != 32) {
                fail(&operation,
                     "Metal atomic lowering currently requires one 32-bit integer result");
                return std::nullopt;
            }
            const auto atomic_op = operation.attributes.find("atomic_op");
            const std::string callee =
                atomic_op != operation.attributes.end() && atomic_op->second == "add"
                    ? "atomic_fetch_add_explicit"
                    : atomic_op != operation.attributes.end() && atomic_op->second == "or"
                          ? "atomic_fetch_or_explicit"
                          : std::string{};
            if (callee.empty()) {
                fail(&operation, "unsupported Metal atomic operation '" +
                                     (atomic_op == operation.attributes.end()
                                          ? std::string("<missing>")
                                          : atomic_op->second) +
                                     "'");
                return std::nullopt;
            }
            if (operation.memory_ordering != ir::MemoryOrdering::kRelaxed) {
                fail(&operation,
                     "Metal atomic lowering currently requires relaxed CUDA ordering");
                return std::nullopt;
            }
            const MslExpr raw_pointer = expression_for(operation.operands.front());
            const MslAddressSpace address_space =
                raw_pointer->type.kind == MslTypeKind::kPointer
                    ? raw_pointer->type.address_space
                    : lower_address_space(operation.operands.front().type.address_space);
            const MslType atomic_uint = {
                .kind = MslTypeKind::kStruct,
                .struct_name = "atomic_uint",
            };
            const MslType atomic_pointer =
                MslType::pointer(atomic_uint, address_space);
            const MslExpr pointer =
                MslExpression::cast(atomic_pointer, raw_pointer, true);
            const MslExpr ordering = MslExpression::identifier(
                "memory_order_relaxed", MslType{
                                            .kind = MslTypeKind::kStruct,
                                            .struct_name = "memory_order",
                                        });
            return declare_result(
                operation,
                MslExpression::call(callee,
                                    {pointer, expression_for(operation.operands[1]),
                                     ordering},
                                    lower_result_type(operation)));
        }

        if (operation.opcode == ir::OpCode::kMetalReduction) {
            fail(&operation, "Metal semantic operation '" +
                                 std::string(ir::opcode_name(operation.opcode)) +
                                 "' is represented but not yet MSL-emittable");
            return std::nullopt;
        }

        fail(&operation, "operation '" + std::string(ir::opcode_name(operation.opcode)) +
                             "' is not representable in the typed MSL backend");
        return std::nullopt;
    }

    bool emit_operations(const ir::BasicBlock& block, std::vector<MslStmt>* statements) {
        for (const ir::Operation& operation : block.operations) {
            if (operation.is_terminator()) continue;
            const std::optional<MslStmt> lowered = lower_operation(operation);
            if (!result.error.empty()) return false;
            if (lowered.has_value()) statements->push_back(*lowered);
        }
        return true;
    }

    MslStmt lower_return(const ir::Operation& operation) {
        if (operation.operands.empty()) return MslStatement::return_statement();
        return MslStatement::return_statement(expression_for(operation.operands.front()));
    }

    MslExpr pointer_tag_for(const MslExpr& pointer) const {
        unsigned tag = 0;
        if (pointer->type.kind == MslTypeKind::kPointer) {
            if (pointer->type.address_space == MslAddressSpace::kDevice) {
                tag = static_cast<unsigned>(ir::AddressSpace::kDevice);
            } else if (pointer->type.address_space == MslAddressSpace::kThreadgroup) {
                tag = static_cast<unsigned>(ir::AddressSpace::kThreadgroup);
            } else if (pointer->type.address_space == MslAddressSpace::kThread) {
                tag = static_cast<unsigned>(ir::AddressSpace::kPrivate);
            } else if (pointer->type.address_space == MslAddressSpace::kConstant) {
                tag = static_cast<unsigned>(ir::AddressSpace::kConstant);
            }
        }
        return MslExpression::literal(std::to_string(tag) + "u", MslType::uint());
    }

    std::pair<MslExpr, MslExpr> mixed_pointer_parts(ir::ValueId source,
                                                    const ir::Type& target_type) {
        if (is_mixed_pointer(source) && !pointer_specialization.has_value()) {
            return {expression_for(ir::Operand::value_ref(source, target_type)),
                    mixed_pointer_tags.at(source)};
        }
        const MslExpr pointer = expression_for(ir::Operand::value_ref(source, target_type));
        return {MslExpression::cast(MslType::uint(64), pointer, true),
                pointer_tag_for(pointer)};
    }

    void declare_block_argument(const ir::BlockArgument& argument,
                                std::vector<MslStmt>* statements) {
        if (!declared_block_arguments.insert(argument.value).second) return;
        const MslType type = lower_value_type(argument.value, argument.type);
        const std::string name = value_name(argument.value);
        values[argument.value] = MslExpression::identifier(name, type);
        statements->push_back(
            MslStatement::variable(type, name, std::nullopt, false));
        if (is_mixed_pointer(argument.value) &&
            !pointer_specialization.has_value()) {
            const std::string tag_name = name + "_space";
            mixed_pointer_tags[argument.value] =
                MslExpression::identifier(tag_name, MslType::uint());
            statements->push_back(MslStatement::variable(
                MslType::uint(), tag_name, std::nullopt, false));
        }
    }

    bool bind_block_arguments(const ir::BasicBlock& block,
                              const ir::Successor* incoming) {
        if (incoming == nullptr) return true;
        if (incoming->arguments.size() != block.arguments.size()) {
            return fail(nullptr, "branch argument count does not match block arguments for '" +
                                     block.name + "'");
        }
        std::vector<MslExpr> incoming_values;
        std::vector<std::optional<MslExpr>> incoming_tags;
        incoming_values.reserve(incoming->arguments.size());
        incoming_tags.reserve(incoming->arguments.size());
        for (std::size_t i = 0; i < incoming->arguments.size(); ++i) {
            if (is_mixed_pointer(block.arguments[i].value) &&
                !pointer_specialization.has_value()) {
                auto [address, tag] = mixed_pointer_parts(
                    incoming->arguments[i], block.arguments[i].type);
                incoming_values.push_back(std::move(address));
                incoming_tags.push_back(std::move(tag));
            } else {
                incoming_values.push_back(expression_for(ir::Operand::value_ref(
                    incoming->arguments[i], block.arguments[i].type)));
                incoming_tags.push_back(std::nullopt);
            }
        }
        for (std::size_t i = 0; i < block.arguments.size(); ++i) {
            values[block.arguments[i].value] = std::move(incoming_values[i]);
            if (incoming_tags[i].has_value()) {
                mixed_pointer_tags[block.arguments[i].value] =
                    std::move(*incoming_tags[i]);
            }
        }
        return true;
    }

    std::optional<std::size_t> find_nearest_common_successor(std::size_t first,
                                                              std::size_t second) const {
        if (first >= postdominators.size() || second >= postdominators.size()) {
            return std::nullopt;
        }
        std::vector<std::size_t> common;
        for (std::size_t candidate = 0; candidate < function.blocks.size();
             ++candidate) {
            if (postdominators[first][candidate] &&
                postdominators[second][candidate]) {
                common.push_back(candidate);
            }
        }
        for (const std::size_t candidate : common) {
            const bool postdominates_another_common = std::any_of(
                common.begin(), common.end(), [&](std::size_t other) {
                    return other != candidate &&
                           postdominators[other][candidate];
                });
            if (!postdominates_another_common) return candidate;
        }
        return std::nullopt;
    }

    bool assign_join_arguments(const ir::BasicBlock& join,
                               const ir::Successor& successor,
                               std::vector<MslStmt>* statements) {
        if (successor.arguments.size() != join.arguments.size()) {
            return fail(nullptr, "branch argument count does not match join block '" +
                                     join.name + "'");
        }
        for (std::size_t i = 0; i < join.arguments.size(); ++i) {
            if (is_mixed_pointer(join.arguments[i].value) &&
                !pointer_specialization.has_value()) {
                auto [address, tag] = mixed_pointer_parts(
                    successor.arguments[i], join.arguments[i].type);
                statements->push_back(MslStatement::assignment(
                    values.at(join.arguments[i].value), std::move(address)));
                statements->push_back(MslStatement::assignment(
                    mixed_pointer_tags.at(join.arguments[i].value), std::move(tag)));
                continue;
            }
            statements->push_back(MslStatement::assignment(
                values.at(join.arguments[i].value),
                expression_for(ir::Operand::value_ref(successor.arguments[i],
                                                       join.arguments[i].type))));
        }
        return true;
    }

    bool can_reach(std::size_t start, std::size_t target) const {
        std::queue<std::size_t> pending;
        std::unordered_set<std::size_t> visited;
        pending.push(start);
        visited.insert(start);
        while (!pending.empty()) {
            const std::size_t index = pending.front();
            pending.pop();
            if (index == target) return true;
            const ir::Operation& terminator = function.blocks[index].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                const std::size_t next = block_indices.at(successor.block);
                if (visited.insert(next).second) pending.push(next);
            }
        }
        return false;
    }

    void analyze_cfg() {
        const std::size_t count = function.blocks.size();
        predecessors.assign(count, {});
        for (std::size_t source = 0; source < count; ++source) {
            const ir::Operation& terminator =
                function.blocks[source].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                predecessors[block_indices.at(successor.block)].push_back(source);
            }
        }

        dominators.assign(count, std::vector<bool>(count, true));
        if (count == 0) return;
        std::fill(dominators[0].begin(), dominators[0].end(), false);
        dominators[0][0] = true;
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t block = 1; block < count; ++block) {
                std::vector<bool> next(count, true);
                if (predecessors[block].empty()) {
                    std::fill(next.begin(), next.end(), false);
                } else {
                    for (const std::size_t predecessor : predecessors[block]) {
                        for (std::size_t candidate = 0; candidate < count;
                             ++candidate) {
                            next[candidate] =
                                next[candidate] && dominators[predecessor][candidate];
                        }
                    }
                }
                next[block] = true;
                if (next != dominators[block]) {
                    dominators[block] = std::move(next);
                    changed = true;
                }
            }
        }

        postdominators.assign(count, std::vector<bool>(count, true));
        for (std::size_t block = 0; block < count; ++block) {
            if (!function.blocks[block].operations.back().successors.empty()) continue;
            std::fill(postdominators[block].begin(),
                      postdominators[block].end(), false);
            postdominators[block][block] = true;
        }
        changed = true;
        while (changed) {
            changed = false;
            for (std::size_t block = 0; block < count; ++block) {
                const ir::Operation& terminator =
                    function.blocks[block].operations.back();
                if (terminator.successors.empty()) continue;
                std::vector<bool> next(count, true);
                for (const ir::Successor& successor : terminator.successors) {
                    const std::size_t target = block_indices.at(successor.block);
                    for (std::size_t candidate = 0; candidate < count;
                         ++candidate) {
                        next[candidate] =
                            next[candidate] && postdominators[target][candidate];
                    }
                }
                next[block] = true;
                if (next != postdominators[block]) {
                    postdominators[block] = std::move(next);
                    changed = true;
                }
            }
        }
    }

    bool dominates(std::size_t dominator, std::size_t block) const {
        return block < dominators.size() && dominator < dominators[block].size() &&
               dominators[block][dominator];
    }

    std::unordered_set<std::size_t> natural_loop_nodes(
        std::size_t header_index) const {
        std::unordered_set<std::size_t> nodes = {header_index};
        std::vector<std::size_t> pending;
        for (std::size_t source = 0; source < function.blocks.size(); ++source) {
            const ir::Operation& terminator =
                function.blocks[source].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                if (block_indices.at(successor.block) == header_index &&
                    source != header_index && dominates(header_index, source) &&
                    nodes.insert(source).second) {
                    pending.push_back(source);
                }
            }
        }
        while (!pending.empty()) {
            const std::size_t block = pending.back();
            pending.pop_back();
            for (const std::size_t predecessor : predecessors[block]) {
                if (predecessor != header_index &&
                    !dominates(header_index, predecessor)) {
                    continue;
                }
                if (nodes.insert(predecessor).second &&
                    predecessor != header_index) {
                    pending.push_back(predecessor);
                }
            }
        }
        return nodes;
    }

    std::optional<std::pair<std::size_t, std::size_t>> loop_body_and_exit(
        std::size_t header_index) const {
        const ir::Operation& terminator =
            function.blocks[header_index].operations.back();
        if (terminator.opcode != ir::OpCode::kCondBranch ||
            terminator.successors.size() != 2) {
            return std::nullopt;
        }
        const std::size_t first = block_indices.at(terminator.successors[0].block);
        const std::size_t second = block_indices.at(terminator.successors[1].block);
        const std::unordered_set<std::size_t> loop =
            natural_loop_nodes(header_index);
        if (loop.size() == 1) return std::nullopt;
        const bool first_in_loop = loop.contains(first);
        const bool second_in_loop = loop.contains(second);
        if (first_in_loop == second_in_loop) return std::nullopt;
        return first_in_loop ? std::pair{first, second}
                             : std::pair{second, first};
    }

    std::optional<std::size_t> natural_loop_exit_index(
        std::size_t header_index) const {
        if (const auto canonical = loop_body_and_exit(header_index)) {
            return canonical->second;
        }
        const std::unordered_set<std::size_t> loop =
            natural_loop_nodes(header_index);
        if (loop.size() <= 1) return std::nullopt;
        std::vector<std::size_t> exits;
        for (const std::size_t source : loop) {
            const ir::Operation& terminator =
                function.blocks[source].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                const std::size_t target = block_indices.at(successor.block);
                if (loop.contains(target)) continue;
                if (std::find(exits.begin(), exits.end(), target) == exits.end()) {
                    exits.push_back(target);
                }
            }
        }
        if (exits.empty()) return std::nullopt;
        if (exits.size() == 1) return exits.front();
        std::vector<std::size_t> common;
        for (std::size_t candidate = 0; candidate < function.blocks.size();
             ++candidate) {
            if (loop.contains(candidate)) continue;
            const bool postdominates_all = std::all_of(
                exits.begin(), exits.end(), [&](std::size_t exit) {
                    return postdominators[exit][candidate];
                });
            if (postdominates_all) common.push_back(candidate);
        }
        for (const std::size_t candidate : common) {
            const bool postdominates_another_common = std::any_of(
                common.begin(), common.end(), [&](std::size_t other) {
                    return other != candidate &&
                           postdominators[other][candidate];
                });
            if (!postdominates_another_common) return candidate;
        }
        return std::nullopt;
    }

    bool assign_loop_arguments(const ir::BasicBlock& header,
                               const ir::Successor& backedge,
                               std::vector<MslStmt>* statements) {
        if (backedge.arguments.size() != header.arguments.size()) {
            return fail(nullptr, "loop backedge argument count does not match header '" +
                                     header.name + "'");
        }
        std::vector<MslExpr> temporaries;
        std::vector<std::optional<MslExpr>> tag_temporaries;
        temporaries.reserve(header.arguments.size());
        tag_temporaries.reserve(header.arguments.size());
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            if (is_mixed_pointer(header.arguments[i].value) &&
                !pointer_specialization.has_value()) {
                auto [address, tag] = mixed_pointer_parts(
                    backedge.arguments[i], header.arguments[i].type);
                const std::string name = value_name(header.arguments[i].value) + "_next";
                const std::string tag_name = name + "_space";
                statements->push_back(MslStatement::variable(
                    MslType::uint(64), name, std::move(address), true));
                statements->push_back(MslStatement::variable(
                    MslType::uint(), tag_name, std::move(tag), true));
                temporaries.push_back(
                    MslExpression::identifier(name, MslType::uint(64)));
                tag_temporaries.push_back(
                    MslExpression::identifier(tag_name, MslType::uint()));
                continue;
            }
            const MslType type = lower_value_type(header.arguments[i].value,
                                                  header.arguments[i].type);
            const std::string name = value_name(header.arguments[i].value) + "_next";
            statements->push_back(MslStatement::variable(
                type, name,
                expression_for(ir::Operand::value_ref(backedge.arguments[i],
                                                       header.arguments[i].type)),
                true));
            temporaries.push_back(MslExpression::identifier(name, type));
            tag_temporaries.push_back(std::nullopt);
        }
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            statements->push_back(MslStatement::assignment(
                values.at(header.arguments[i].value), temporaries[i]));
            if (tag_temporaries[i].has_value()) {
                statements->push_back(MslStatement::assignment(
                    mixed_pointer_tags.at(header.arguments[i].value),
                    *tag_temporaries[i]));
            }
        }
        return true;
    }

    bool emit_loop_region(
        std::size_t block_index, std::size_t stop_index,
        std::optional<std::size_t> active_loop_header_index,
        const ir::Successor* incoming,
                          std::vector<MslStmt>* statements) {
        if (!loop_escape_stack.empty() &&
            block_index == loop_escape_stack.back().enclosing_header_index) {
            if (incoming == nullptr) {
                return fail(nullptr,
                            "nested loop reaches its enclosing header without an edge");
            }
            if (!assign_loop_arguments(function.blocks[block_index], *incoming,
                                       statements)) {
                return false;
            }
            statements->push_back(MslStatement::assignment(
                loop_escape_stack.back().continue_enclosing,
                MslExpression::literal("true", MslType::boolean())));
            statements->push_back(MslStatement::break_statement());
            return true;
        }
        if (active_loop_header_index.has_value() &&
            block_index == *active_loop_header_index) {
            if (incoming == nullptr) {
                return fail(nullptr, "loop region reaches header '" +
                                         function.blocks[block_index].name +
                                         "' without a backedge");
            }
            return assign_loop_arguments(function.blocks[block_index], *incoming,
                                         statements);
        }
        if (block_index == stop_index) {
            if (incoming == nullptr) return true;
            return assign_loop_arguments(function.blocks[block_index], *incoming,
                                         statements);
        }
        if (natural_loop_exit_index(block_index).has_value()) {
            return emit_natural_loop(block_index, incoming, statements,
                                     active_loop_header_index, stop_index);
        }
        const ir::BasicBlock& block = function.blocks[block_index];
        if (!region_stack.insert(block.id).second) {
            const std::string owner = active_loop_header_index.has_value()
                                          ? function.blocks[*active_loop_header_index].name
                                          : function.blocks[stop_index].name;
            return fail(nullptr, "structured region for '" + owner +
                                     "' revisits block '" + block.name + "'");
        }
        struct RegionStackGuard {
            std::unordered_set<ir::BlockId>* stack;
            ir::BlockId block;
            ~RegionStackGuard() { stack->erase(block); }
        } guard{&region_stack, block.id};
        if (!bind_block_arguments(block, incoming) ||
            !emit_operations(block, statements)) {
            return false;
        }
        const ir::Operation& terminator = block.operations.back();
        if (terminator.opcode == ir::OpCode::kReturn) {
            statements->push_back(lower_return(terminator));
            return true;
        }
        if (terminator.opcode == ir::OpCode::kBranch) {
            if (terminator.successors.size() != 1) {
                return fail(&terminator, "malformed loop-body branch");
            }
            const ir::Successor& successor = terminator.successors.front();
            return emit_loop_region(block_indices.at(successor.block), stop_index,
                                    active_loop_header_index, &successor,
                                    statements);
        }
        if (terminator.opcode != ir::OpCode::kCondBranch ||
            terminator.successors.size() != 2 || terminator.operands.empty()) {
            return fail(&terminator, "unsupported loop-body terminator");
        }

        const std::size_t first = block_indices.at(terminator.successors[0].block);
        const std::size_t second = block_indices.at(terminator.successors[1].block);
        const bool first_returns =
            is_inlineable_terminal_return_block(function.blocks[first]);
        const bool second_returns =
            is_inlineable_terminal_return_block(function.blocks[second]);
        if (first_returns || second_returns) {
            if (first_returns && second_returns) {
                statements->push_back(MslStatement::if_statement(
                    expression_for(terminator.operands.front()),
                    {lower_return(function.blocks[first].operations.front())},
                    {lower_return(function.blocks[second].operations.front())}));
                return true;
            }
            const std::size_t return_index = first_returns ? first : second;
            const std::size_t continuation_index = first_returns ? second : first;
            MslExpr condition = expression_for(terminator.operands.front());
            if (second_returns) {
                condition = MslExpression::unary("!", condition, MslType::boolean());
            }
            statements->push_back(MslStatement::if_statement(
                condition, {lower_return(function.blocks[return_index].operations.front())}));
            const std::size_t successor_index = first_returns ? 1 : 0;
            return emit_loop_region(
                continuation_index, stop_index, active_loop_header_index,
                &terminator.successors[successor_index], statements);
        }

        const auto enclosing_loop_exit =
            active_loop_header_index.has_value()
                ? natural_loop_exit_index(*active_loop_header_index)
                : std::nullopt;
        const std::unordered_set<std::size_t> loop_nodes =
            enclosing_loop_exit.has_value()
                ? natural_loop_nodes(*active_loop_header_index)
                : std::unordered_set<std::size_t>{};
        const bool first_in_loop =
            enclosing_loop_exit.has_value() && loop_nodes.contains(first);
        const bool second_in_loop =
            enclosing_loop_exit.has_value() && loop_nodes.contains(second);
        if (enclosing_loop_exit.has_value() && first_in_loop != second_in_loop) {
            const std::size_t exit_successor_index = first_in_loop ? 1 : 0;
            const std::size_t continuation_successor_index = 1 - exit_successor_index;
            const std::size_t exit_index =
                block_indices.at(terminator.successors[exit_successor_index].block);
            std::vector<MslStmt> exit_statements;
            if (exit_index == *enclosing_loop_exit) {
                const ir::BasicBlock& exit_block = function.blocks[exit_index];
                if (!assign_join_arguments(
                        exit_block, terminator.successors[exit_successor_index],
                        &exit_statements)) {
                    return false;
                }
            } else {
                if (loop_nodes.contains(exit_index)) {
                    return fail(&terminator,
                                "natural loop secondary exit does not reconverge");
                }
                if (!emit_loop_region(
                        exit_index, *enclosing_loop_exit,
                        active_loop_header_index,
                        &terminator.successors[exit_successor_index],
                        &exit_statements)) {
                    return false;
                }
            }
            exit_statements.push_back(MslStatement::break_statement());
            MslExpr exit_condition = expression_for(terminator.operands.front());
            if (exit_successor_index == 1) {
                exit_condition = MslExpression::unary(
                    "!", exit_condition, MslType::boolean());
            }
            statements->push_back(MslStatement::if_statement(
                std::move(exit_condition), std::move(exit_statements)));
            const ir::Successor& continuation =
                terminator.successors[continuation_successor_index];
            return emit_loop_region(block_indices.at(continuation.block),
                                    stop_index, active_loop_header_index,
                                    &continuation, statements);
        }

        const auto join = find_nearest_common_successor(first, second);
        if (!join || *join == block_index) {
            return fail(&terminator, "nested loop conditional has no reconvergence");
        }
        if (*join != stop_index) {
            const ir::BasicBlock& join_block = function.blocks[*join];
            for (const ir::BlockArgument& argument : join_block.arguments) {
                declare_block_argument(argument, statements);
            }
        }
        std::vector<MslStmt> first_statements;
        std::vector<MslStmt> second_statements;
        if (!emit_loop_region(first, *join, active_loop_header_index,
                              &terminator.successors[0],
                              &first_statements) ||
            !emit_loop_region(second, *join, active_loop_header_index,
                              &terminator.successors[1],
                              &second_statements)) {
            return false;
        }
        statements->push_back(MslStatement::if_statement(
            expression_for(terminator.operands.front()),
            std::move(first_statements), std::move(second_statements)));
        if (*join == stop_index) return true;
        return emit_loop_region(*join, stop_index, active_loop_header_index,
                                nullptr, statements);
    }

    bool emit_natural_loop(
        std::size_t header_index, const ir::Successor* incoming,
        std::vector<MslStmt>* statements,
        std::optional<std::size_t> enclosing_header_index = std::nullopt,
        std::optional<std::size_t> continuation_stop_index = std::nullopt) {
        const ir::BasicBlock& header = function.blocks[header_index];
        const auto body_and_exit = loop_body_and_exit(header_index);
        const auto loop_exit = natural_loop_exit_index(header_index);
        if (!loop_exit || incoming == nullptr ||
            incoming->arguments.size() != header.arguments.size()) {
            return fail(nullptr, "malformed natural loop header '" + header.name + "'");
        }

        if (!assign_loop_arguments(header, *incoming, statements)) return false;

        const std::size_t exit_index = *loop_exit;
        const ir::BasicBlock& exit_block = function.blocks[exit_index];
        const bool exits_to_enclosing_header =
            enclosing_header_index.has_value() &&
            exit_index == *enclosing_header_index;
        const bool exits_enclosing_loop =
            enclosing_header_index.has_value() &&
            natural_loop_exit_index(*enclosing_header_index) == loop_exit;
        if (!exits_to_enclosing_header && !exits_enclosing_loop) {
            for (const ir::BlockArgument& argument : exit_block.arguments) {
                declare_block_argument(argument, statements);
            }
        }

        std::optional<MslExpr> continue_enclosing;
        if (exits_enclosing_loop) {
            const std::string name =
                "cm_continue_enclosing_" + std::to_string(loop_escape_index++);
            statements->push_back(MslStatement::variable(
                MslType::boolean(), name,
                MslExpression::literal("false", MslType::boolean()), false));
            continue_enclosing =
                MslExpression::identifier(name, MslType::boolean());
            loop_escape_stack.push_back(
                {*enclosing_header_index, *continue_enclosing});
        }
        struct LoopEscapeGuard {
            std::vector<LoopEscapeContext>* stack;
            bool active;
            ~LoopEscapeGuard() {
                if (active) stack->pop_back();
            }
        } escape_guard{&loop_escape_stack, exits_enclosing_loop};

        std::vector<MslStmt> loop_statements;
        if (!emit_operations(header, &loop_statements)) return false;
        const ir::Operation& terminator = header.operations.back();
        if (!body_and_exit) {
            if (terminator.opcode != ir::OpCode::kCondBranch ||
                terminator.successors.size() != 2 || terminator.operands.empty()) {
                return fail(&terminator,
                            "general natural loop requires a conditional header");
            }
            const std::size_t first =
                block_indices.at(terminator.successors[0].block);
            const std::size_t second =
                block_indices.at(terminator.successors[1].block);
            const std::unordered_set<std::size_t> loop_nodes =
                natural_loop_nodes(header_index);
            if (!loop_nodes.contains(first) || !loop_nodes.contains(second)) {
                return fail(&terminator,
                            "general natural loop header has an unrecognized exit");
            }
            const auto join = find_nearest_common_successor(first, second);
            if (!join || !loop_nodes.contains(*join)) {
                return fail(&terminator,
                            "general natural loop header paths do not reconverge in-loop");
            }
            const ir::BasicBlock& join_block = function.blocks[*join];
            for (const ir::BlockArgument& argument : join_block.arguments) {
                declare_block_argument(argument, &loop_statements);
            }
            std::vector<MslStmt> first_statements;
            std::vector<MslStmt> second_statements;
            if (!emit_loop_region(first, *join, header_index,
                                  &terminator.successors[0],
                                  &first_statements) ||
                !emit_loop_region(second, *join, header_index,
                                  &terminator.successors[1],
                                  &second_statements)) {
                return false;
            }
            loop_statements.push_back(MslStatement::if_statement(
                expression_for(terminator.operands.front()),
                std::move(first_statements), std::move(second_statements)));
            if (!emit_loop_region(*join, header_index, header_index, nullptr,
                                  &loop_statements)) {
                return false;
            }
            statements->push_back(MslStatement::while_statement(
                MslExpression::literal("true", MslType::boolean()),
                std::move(loop_statements)));
            if (exits_enclosing_loop) {
                statements->push_back(MslStatement::if_statement(
                    *continue_enclosing,
                    {MslStatement::continue_statement()}));
                statements->push_back(MslStatement::break_statement());
                return true;
            }
            if (exits_to_enclosing_header) return true;
            if (continuation_stop_index.has_value()) {
                return emit_loop_region(exit_index, *continuation_stop_index,
                                        enclosing_header_index, nullptr,
                                        statements);
            }
            return emit_from(exit_index, statements);
        }
        const std::size_t body_index = body_and_exit->first;
        const std::size_t body_successor_index =
            block_indices.at(terminator.successors[0].block) == body_index ? 0 : 1;
        const std::size_t exit_successor_index = 1 - body_successor_index;
        MslExpr continue_condition = expression_for(terminator.operands.front());
        if (body_successor_index == 1) {
            continue_condition =
                MslExpression::unary("!", continue_condition, MslType::boolean());
        }
        MslExpr exit_condition =
            MslExpression::unary("!", continue_condition, MslType::boolean());
        std::vector<MslStmt> exit_statements;
        const bool assigned_exit = exits_to_enclosing_header
                                       ? assign_loop_arguments(
                                             exit_block,
                                             terminator.successors[exit_successor_index],
                                             &exit_statements)
                                       : assign_join_arguments(
                                             exit_block,
                                             terminator.successors[exit_successor_index],
                                             &exit_statements);
        if (!assigned_exit) {
            return false;
        }
        exit_statements.push_back(MslStatement::break_statement());
        loop_statements.push_back(MslStatement::if_statement(
            exit_condition, std::move(exit_statements)));
        if (!emit_loop_region(body_index, header_index, header_index,
                              &terminator.successors[body_successor_index],
                              &loop_statements)) {
            return false;
        }
        statements->push_back(MslStatement::while_statement(
            MslExpression::literal("true", MslType::boolean()),
            std::move(loop_statements)));
        if (exits_enclosing_loop) {
            statements->push_back(MslStatement::if_statement(
                *continue_enclosing, {MslStatement::continue_statement()}));
            statements->push_back(MslStatement::break_statement());
            return true;
        }
        if (exits_to_enclosing_header) return true;
        if (continuation_stop_index.has_value()) {
            return emit_loop_region(exit_index, *continuation_stop_index,
                                    enclosing_header_index, nullptr, statements);
        }
        return emit_from(exit_index, statements);
    }

    bool requires_cfg_dispatcher() const {
        for (std::size_t source = 0; source < function.blocks.size(); ++source) {
            const ir::Operation& terminator = function.blocks[source].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                const std::size_t target = block_indices.at(successor.block);
                if (target != source && dominates(target, source) &&
                    !natural_loop_exit_index(target).has_value()) {
                    return true;
                }
            }
        }
        return false;
    }

    bool emit_dispatch_transition(const ir::Successor& successor,
                                  const MslExpr& state,
                                  std::vector<MslStmt>* statements) {
        const std::size_t target_index = block_indices.at(successor.block);
        const ir::BasicBlock& target = function.blocks[target_index];
        if (successor.arguments.size() != target.arguments.size()) {
            return fail(nullptr, "branch argument count does not match dispatcher target '" +
                                     target.name + "'");
        }
        std::vector<MslExpr> temporaries;
        std::vector<std::optional<MslExpr>> tag_temporaries;
        temporaries.reserve(target.arguments.size());
        tag_temporaries.reserve(target.arguments.size());
        for (std::size_t i = 0; i < target.arguments.size(); ++i) {
            const bool mixed = is_mixed_pointer(target.arguments[i].value) &&
                               !pointer_specialization.has_value();
            const MslType type = lower_value_type(target.arguments[i].value,
                                                  target.arguments[i].type);
            const std::string temporary =
                "cm_edge_" + std::to_string(edge_temporary_index++) + "_" +
                std::to_string(i);
            MslExpr incoming;
            std::optional<MslExpr> incoming_tag;
            if (mixed) {
                auto parts = mixed_pointer_parts(successor.arguments[i],
                                                 target.arguments[i].type);
                incoming = std::move(parts.first);
                incoming_tag = std::move(parts.second);
            } else {
                incoming = expression_for(ir::Operand::value_ref(
                    successor.arguments[i], target.arguments[i].type));
            }
            statements->push_back(MslStatement::variable(
                type, temporary, std::move(incoming), true));
            temporaries.push_back(MslExpression::identifier(temporary, type));
            if (incoming_tag.has_value()) {
                const std::string tag_temporary = temporary + "_space";
                statements->push_back(MslStatement::variable(
                    MslType::uint(), tag_temporary, std::move(*incoming_tag), true));
                tag_temporaries.push_back(MslExpression::identifier(
                    tag_temporary, MslType::uint()));
            } else {
                tag_temporaries.push_back(std::nullopt);
            }
        }
        for (std::size_t i = 0; i < target.arguments.size(); ++i) {
            statements->push_back(MslStatement::assignment(
                values.at(target.arguments[i].value), temporaries[i]));
            if (tag_temporaries[i].has_value()) {
                statements->push_back(MslStatement::assignment(
                    mixed_pointer_tags.at(target.arguments[i].value),
                    *tag_temporaries[i]));
            }
        }
        statements->push_back(MslStatement::assignment(
            state, MslExpression::literal(std::to_string(target_index) + "u",
                                          MslType::uint())));
        return true;
    }

    bool emit_cfg_dispatcher(std::vector<MslStmt>* statements) {
        cfg_dispatcher_mode = true;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments) {
                declare_block_argument(argument, statements);
            }
            for (const ir::Operation& operation : block.operations) {
                if (operation.opcode == ir::OpCode::kParameter ||
                    operation.results.empty()) {
                    continue;
                }
                if (operation.opcode == ir::OpCode::kAlloca) {
                    if (operation.result_types.empty() ||
                        !operation.result_types.front().is_pointer() ||
                        operation.result_types.front().pointee() == nullptr) {
                        return fail(&operation, "malformed thread-local allocation");
                    }
                    const ir::ValueId value = operation.results.front();
                    const MslType storage_type =
                        lower_type(*operation.result_types.front().pointee());
                    const std::string storage_name = value_name(value) + "_storage";
                    statements->push_back(
                        MslStatement::variable(storage_type, storage_name));
                    values[value] = MslExpression::unary(
                        "&", MslExpression::identifier(storage_name, storage_type),
                        lower_result_type(operation));
                    continue;
                }
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    const ir::ValueId value = operation.results[i];
                    const MslType type = lower_result_type(operation, i);
                    const std::string name = value_name(value);
                    values[value] = MslExpression::identifier(name, type);
                    statements->push_back(MslStatement::variable(type, name));
                }
            }
        }

        const MslExpr state = MslExpression::identifier("cm_block_state", MslType::uint());
        statements->push_back(MslStatement::variable(
            MslType::uint(), "cm_block_state",
            MslExpression::literal("0u", MslType::uint())));

        std::vector<MslSwitchCase> cases;
        cases.reserve(function.blocks.size());
        for (std::size_t block_index = 0; block_index < function.blocks.size();
             ++block_index) {
            const ir::BasicBlock& block = function.blocks[block_index];
            std::vector<MslStmt> body;
            if (!emit_operations(block, &body)) return false;
            const ir::Operation& terminator = block.operations.back();
            if (terminator.opcode == ir::OpCode::kReturn) {
                body.push_back(lower_return(terminator));
            } else if (terminator.opcode == ir::OpCode::kBranch &&
                       terminator.successors.size() == 1) {
                if (!emit_dispatch_transition(terminator.successors.front(), state,
                                              &body)) {
                    return false;
                }
                body.push_back(MslStatement::break_statement());
            } else if (terminator.opcode == ir::OpCode::kCondBranch &&
                       terminator.successors.size() == 2 &&
                       !terminator.operands.empty()) {
                std::vector<MslStmt> first;
                std::vector<MslStmt> second;
                if (!emit_dispatch_transition(terminator.successors[0], state, &first) ||
                    !emit_dispatch_transition(terminator.successors[1], state, &second)) {
                    return false;
                }
                MslExpr condition = expression_for(terminator.operands.front());
                if (terminator.attributes.contains("inverted") &&
                    terminator.attributes.at("inverted") == "true") {
                    condition =
                        MslExpression::unary("!", condition, MslType::boolean());
                }
                body.push_back(MslStatement::if_statement(
                    condition, std::move(first), std::move(second)));
                body.push_back(MslStatement::break_statement());
            } else if (terminator.opcode == ir::OpCode::kTrap) {
                return fail(&terminator,
                            "trap has no faithful MSL source representation");
            } else {
                return fail(&terminator, "malformed dispatcher terminator");
            }
            cases.push_back({
                .value = MslExpression::literal(std::to_string(block_index) + "u",
                                                MslType::uint()),
                .statements = std::move(body),
            });
        }
        statements->push_back(MslStatement::while_statement(
            MslExpression::literal("true", MslType::boolean()),
            {MslStatement::switch_statement(state, std::move(cases))}));
        return true;
    }

    bool emit_from(std::size_t block_index, std::vector<MslStmt>* statements,
                   const ir::Successor* incoming = nullptr) {
        if (block_index >= function.blocks.size()) return false;
        const ir::BasicBlock& block = function.blocks[block_index];
        if (!emitted.insert(block.id).second) {
            return fail(nullptr, "loop structurization is not implemented for block '" + block.name + "'");
        }
        if (natural_loop_exit_index(block_index)) {
            return emit_natural_loop(block_index, incoming, statements);
        }
        if (!bind_block_arguments(block, incoming)) return false;
        if (!emit_operations(block, statements)) return false;
        const ir::Operation& terminator = block.operations.back();
        if (terminator.opcode == ir::OpCode::kReturn) {
            statements->push_back(lower_return(terminator));
            return true;
        }
        if (terminator.opcode == ir::OpCode::kBranch) {
            if (terminator.successors.size() != 1) {
                return fail(&terminator, "malformed unconditional branch");
            }
            const std::size_t target =
                block_indices.at(terminator.successors.front().block);
            if (emitted.contains(function.blocks[target].id) &&
                is_inlineable_terminal_return_block(function.blocks[target])) {
                const ir::BasicBlock& return_block = function.blocks[target];
                const ir::Successor& successor = terminator.successors.front();
                if (successor.arguments.size() != return_block.arguments.size()) {
                    return fail(&terminator, "branch argument count does not match return block");
                }
                for (std::size_t i = 0; i < return_block.arguments.size(); ++i) {
                    values[return_block.arguments[i].value] = expression_for(
                        ir::Operand::value_ref(successor.arguments[i],
                                               return_block.arguments[i].type));
                }
                statements->push_back(lower_return(return_block.operations.front()));
                return true;
            }
            return emit_from(target, statements, &terminator.successors.front());
        }
        if (terminator.opcode == ir::OpCode::kCondBranch) {
            if (terminator.successors.size() != 2 || terminator.operands.empty()) {
                return fail(&terminator, "malformed conditional branch");
            }
            const std::size_t first = block_indices.at(terminator.successors[0].block);
            const std::size_t second = block_indices.at(terminator.successors[1].block);
            const bool first_returns =
                is_inlineable_terminal_return_block(function.blocks[first]);
            const bool second_returns =
                is_inlineable_terminal_return_block(function.blocks[second]);
            if (first_returns == second_returns) {
                if (first_returns) {
                    statements->push_back(MslStatement::if_statement(
                        expression_for(terminator.operands.front()),
                        {lower_return(function.blocks[first].operations.front())},
                        {lower_return(function.blocks[second].operations.front())}));
                    return true;
                }
                const auto join = find_nearest_common_successor(first, second);
                if (!join || *join == block_index) {
                    return fail(&terminator, "conditional CFG has no forward reconvergence");
                }
                ir::BasicBlock const& join_block = function.blocks[*join];
                for (const ir::BlockArgument& argument : join_block.arguments) {
                    declare_block_argument(argument, statements);
                }
                std::vector<MslStmt> first_statements;
                std::vector<MslStmt> second_statements;
                if (!emit_loop_region(first, *join, std::nullopt,
                                      &terminator.successors[0],
                                      &first_statements) ||
                    !emit_loop_region(second, *join, std::nullopt,
                                      &terminator.successors[1],
                                      &second_statements)) {
                    return false;
                }
                statements->push_back(MslStatement::if_statement(
                    expression_for(terminator.operands.front()),
                    std::move(first_statements), std::move(second_statements)));
                return emit_from(*join, statements);
            }
            MslExpr condition = expression_for(terminator.operands.front());
            const bool inverted =
                terminator.attributes.contains("inverted") &&
                terminator.attributes.at("inverted") == "true";
            if (inverted) {
                condition = MslExpression::unary("!", condition, MslType::boolean());
            }
            if (second_returns) {
                condition = MslExpression::unary("!", condition, MslType::boolean());
            }
            const std::size_t return_index = first_returns ? first : second;
            const std::size_t return_successor_index = first_returns ? 0 : 1;
            const ir::BasicBlock& return_block = function.blocks[return_index];
            const ir::Successor& return_successor =
                terminator.successors[return_successor_index];
            if (return_successor.arguments.size() != return_block.arguments.size()) {
                return fail(&terminator, "branch argument count does not match return block");
            }
            for (std::size_t i = 0; i < return_block.arguments.size(); ++i) {
                values[return_block.arguments[i].value] = expression_for(
                    ir::Operand::value_ref(return_successor.arguments[i],
                                           return_block.arguments[i].type));
            }
            statements->push_back(MslStatement::if_statement(
                condition, {lower_return(return_block.operations.front())}));
            const std::size_t continuation_successor_index = first_returns ? 1 : 0;
            return emit_from(first_returns ? second : first, statements,
                             &terminator.successors[continuation_successor_index]);
        }
        if (terminator.opcode == ir::OpCode::kTrap) {
            return fail(&terminator, "trap has no faithful MSL source representation");
        }
        return fail(&terminator, "unsupported CFG terminator");
    }

    LowerToMslResult run() {
        output.name = pointer_specialization.has_value()
                          ? specialized_callee(function.name, *pointer_specialization)
                          : function.name;
        output.return_type = lower_type(function.return_type);
        output.is_kernel = function.is_kernel;
        for (std::size_t i = 0; i < function.arguments.size(); ++i) {
            const ir::FunctionArgument& argument = function.arguments[i];
            MslType type;
            std::vector<MslAttribute> attributes;
            if (!function.is_kernel) {
                type = lower_value_type(argument.value, argument.type);
            } else if (argument.type.is_pointer()) {
                type = lower_value_type(argument.value, argument.type);
            } else {
                type = MslType::reference(lower_type(argument.type), MslAddressSpace::kConstant);
            }
            if (function.is_kernel) {
                attributes.push_back(MslAttribute{
                    .name = "buffer",
                    .index = static_cast<std::uint32_t>(i),
                });
            }
            output.parameters.push_back({
                .type = type,
                .name = argument.name,
                .attributes = std::move(attributes),
            });
            values[argument.value] =
                MslExpression::identifier(argument.name, type);
        }
        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            block_indices[function.blocks[i].id] = i;
            for (const ir::BlockArgument& argument : function.blocks[i].arguments) {
                values[argument.value] =
                    MslExpression::identifier(value_name(argument.value),
                                              lower_value_type(argument.value,
                                                               argument.type));
                if (is_mixed_pointer(argument.value) &&
                    !pointer_specialization.has_value()) {
                    mixed_pointer_tags[argument.value] = MslExpression::identifier(
                        value_name(argument.value) + "_space", MslType::uint());
                }
            }
        }
        analyze_cfg();

        const auto required_shared = shared_usage.find(function.name);
        if (required_shared != shared_usage.end()) {
            for (const std::string& global : required_shared->second) {
                const std::string name = shared_parameter_name(global);
                if (function.is_kernel) {
                    const auto declaration = std::find_if(
                        module.global_threadgroups.begin(),
                        module.global_threadgroups.end(),
                        [&](const ir::GlobalThreadgroup& candidate) {
                            return candidate.name == global;
                        });
                    if (declaration == module.global_threadgroups.end() ||
                        declaration->byte_size == 0) {
                        return LowerToMslResult{
                            .error = "missing static threadgroup declaration for '" +
                                     global + "'",
                        };
                    }
                    output.statements.push_back(
                        MslStatement::threadgroup_byte_array(name,
                                                             declaration->byte_size));
                } else {
                    output.parameters.push_back({
                        .type = MslType::pointer(MslType::uint(8),
                                                 MslAddressSpace::kThreadgroup),
                        .name = name,
                    });
                }
            }
        }

        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments) {
                declare_block_argument(argument, &output.statements);
            }
        }
        predeclared_ssa_storage = true;
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::Operation& operation : block.operations) {
                if (operation.opcode == ir::OpCode::kParameter) continue;
                if (operation.opcode == ir::OpCode::kAlloca) {
                    if (operation.results.size() != 1 ||
                        operation.result_types.size() != 1 ||
                        !operation.result_types.front().is_pointer() ||
                        operation.result_types.front().pointee() == nullptr) {
                        return LowerToMslResult{
                            .error = "malformed thread-local allocation",
                        };
                    }
                    const ir::ValueId value = operation.results.front();
                    const MslType storage_type =
                        lower_type(*operation.result_types.front().pointee());
                    const std::string storage_name = value_name(value) + "_storage";
                    output.statements.push_back(MslStatement::variable(
                        storage_type, storage_name, std::nullopt, false));
                    values[value] = MslExpression::unary(
                        "&", MslExpression::identifier(storage_name, storage_type),
                        lower_result_type(operation));
                    continue;
                }
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    const ir::ValueId value = operation.results[i];
                    const MslType type = lower_result_type(operation, i);
                    const std::string name = value_name(value);
                    values[value] = MslExpression::identifier(name, type);
                    output.statements.push_back(MslStatement::variable(
                        type, name, std::nullopt, false));
                }
            }
        }

        if (force_cfg_dispatcher || requires_cfg_dispatcher()) {
            if (!emit_cfg_dispatcher(&output.statements)) return result;
        } else if (!emit_from(0, &output.statements)) {
            return result;
        }

        auto builtin_attributes = [&](std::string name) {
            std::vector<MslAttribute> attributes;
            if (function.is_kernel) {
                attributes.push_back(MslAttribute{.name = std::move(name)});
            }
            return attributes;
        };

        if (needs_thread_position) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_thread_position",
                .attributes = builtin_attributes("thread_position_in_threadgroup"),
            });
        }
        if (needs_threadgroup_position) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threadgroup_position",
                .attributes = builtin_attributes("threadgroup_position_in_grid"),
            });
        }
        if (needs_threads_per_threadgroup) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threads_per_threadgroup",
                .attributes = builtin_attributes("threads_per_threadgroup"),
            });
        }
        if (needs_threadgroups_per_grid) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threadgroups_per_grid",
                .attributes = builtin_attributes("threadgroups_per_grid"),
            });
        }
        if (needs_lane_id) {
            output.parameters.push_back({
                .type = MslType::uint(),
                .name = "cm_lane_id",
                .attributes = builtin_attributes("thread_index_in_simdgroup"),
            });
        }

        result.ast.functions.push_back(std::move(output));
        const MslPrintResult printed = print_msl(result.ast);
        if (!printed.ok) {
            std::ostringstream error;
            error << "typed MSL printer rejected the module";
            for (const std::string& item : printed.errors) error << "\n" << item;
            result.error = error.str();
            return result;
        }
        result.source = printed.source;
        result.ok = true;
        return result;
    }
};

}  // namespace

MetalLegalizeResult legalize_for_metal(const ir::Module& module) {
    MetalLegalizeResult result;
    const ir::VerifyResult input_verification = ir::verify(module);
    if (!input_verification.ok) {
        result.error = "cannot legalize invalid CuMetal GPU IR";
        return result;
    }
    result.module = module;
    const AddressSpaceResolution address_spaces =
        resolve_generic_address_spaces(&result.module);
    if (!address_spaces.ok) {
        result.error = "cannot legalize CUDA generic pointers: " +
                       address_spaces.error;
        return result;
    }
    result.module.stage = ir::IrStage::kMetalLegalized;

    for (ir::Function& function : result.module.functions) {
        for (ir::BasicBlock& block : function.blocks) {
            for (ir::Operation& operation : block.operations) {
                switch (operation.opcode) {
                    case ir::OpCode::kThreadId:
                        operation.opcode = ir::OpCode::kMetalThreadPosition;
                        break;
                    case ir::OpCode::kThreadgroupId:
                        operation.opcode = ir::OpCode::kMetalThreadgroupPosition;
                        break;
                    case ir::OpCode::kThreadgroupSize:
                        operation.opcode = ir::OpCode::kMetalThreadsPerThreadgroup;
                        break;
                    case ir::OpCode::kGridSize:
                        operation.opcode = ir::OpCode::kMetalThreadgroupsPerGrid;
                        break;
                    case ir::OpCode::kLaneId:
                        operation.opcode = ir::OpCode::kMetalLaneId;
                        break;
                    case ir::OpCode::kBarrier:
                        if (operation.memory_scope != ir::MemoryScope::kThreadgroup &&
                            operation.memory_scope != ir::MemoryScope::kSimdgroup) {
                            result.error = operation.location.str() +
                                           ": barrier scope cannot be represented by Metal";
                            return result;
                        }
                        operation.opcode = ir::OpCode::kMetalBarrier;
                        break;
                    case ir::OpCode::kFence:
                        if (operation.memory_scope == ir::MemoryScope::kSystem) {
                            result.error = operation.location.str() +
                                           ": system-scope fences are unsupported";
                            return result;
                        }
                        if (operation.memory_ordering ==
                            ir::MemoryOrdering::kSequentiallyConsistent) {
                            result.error = operation.location.str() +
                                           ": sequentially-consistent fences are unsupported";
                            return result;
                        }
                        operation.opcode = ir::OpCode::kMetalFence;
                        break;
                    case ir::OpCode::kAtomic:
                        if (operation.memory_scope == ir::MemoryScope::kSystem ||
                            operation.memory_ordering ==
                                ir::MemoryOrdering::kSequentiallyConsistent) {
                            result.error = operation.location.str() +
                                           ": atomic scope/ordering has no faithful Metal mapping";
                            return result;
                        }
                        operation.opcode = ir::OpCode::kMetalAtomic;
                        break;
                    case ir::OpCode::kShuffle:
                        operation.opcode = ir::OpCode::kMetalShuffle;
                        break;
                    case ir::OpCode::kBallot:
                        operation.opcode = ir::OpCode::kMetalBallot;
                        break;
                    case ir::OpCode::kVote:
                        operation.opcode = ir::OpCode::kMetalVote;
                        break;
                    case ir::OpCode::kReduction:
                        operation.opcode = ir::OpCode::kMetalReduction;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    const ir::VerifyResult output_verification = ir::verify(result.module);
    if (!output_verification.ok) {
        std::ostringstream error;
        error << "Metal legalization produced invalid IR";
        for (const ir::Diagnostic& diagnostic : output_verification.diagnostics) {
            error << "\n" << diagnostic.location.str() << ": " << diagnostic.message;
        }
        result.error = error.str();
        return result;
    }
    result.ok = true;
    return result;
}

StructurizeResult check_structurizable(const ir::Function& function) {
    StructurizeResult result;
    for (const ir::BasicBlock& block : function.blocks) {
        const ir::Operation& terminator = block.operations.back();
        if (terminator.opcode == ir::OpCode::kBranch) {
            if (terminator.successors.size() != 1) {
                result.error = "block '" + block.name + "' has malformed branch";
                return result;
            }
        } else if (terminator.opcode == ir::OpCode::kCondBranch) {
            if (terminator.successors.size() != 2) {
                result.error = "block '" + block.name + "' has malformed conditional branch";
                return result;
            }
        } else if (terminator.opcode != ir::OpCode::kReturn &&
                   terminator.opcode != ir::OpCode::kTrap) {
            result.error = "block '" + block.name + "' has unsupported terminator";
            return result;
        }
    }
    result.ok = true;
    return result;
}

LowerToMslResult lower_to_msl(const ir::Module& metal_module) {
    LowerToMslResult result;
    if (metal_module.stage != ir::IrStage::kMetalLegalized) {
        result.error = "MSL lowering requires Metal-legalized CuMetal IR";
        return result;
    }
    if (metal_module.functions.empty()) {
        result.error = "MSL lowering requires at least one function";
        return result;
    }
    const auto provenance = metal_module.attributes.find("provenance");
    if (provenance != metal_module.attributes.end()) {
        result.ast.comments.push_back("cumetal-provenance: " + provenance->second);
    }
    switch (metal_module.semantic_quality) {
        case ir::SemanticQuality::kExact:
            result.ast.comments.push_back("cumetal-semantic-quality: exact");
            break;
        case ir::SemanticQuality::kToleranceBounded:
            result.ast.comments.push_back("cumetal-semantic-quality: tolerance_bounded");
            break;
        case ir::SemanticQuality::kSemanticEmulation:
            result.ast.comments.push_back("cumetal-semantic-quality: semantic_emulation");
            break;
        case ir::SemanticQuality::kPerformanceDegraded:
            result.ast.comments.push_back("cumetal-semantic-quality: performance_degraded");
            break;
        case ir::SemanticQuality::kCpuFallback:
            result.ast.comments.push_back("cumetal-semantic-quality: cpu_fallback");
            break;
        case ir::SemanticQuality::kUnsupported:
            result.ast.comments.push_back("cumetal-semantic-quality: unsupported");
            break;
    }
    for (const std::string& caveat : metal_module.semantic_caveats) {
        result.ast.comments.push_back("cumetal-semantic-caveat: " + caveat);
    }
    result.ast.structs = collect_msl_structs(metal_module);
    for (const ir::GlobalConstant& global : metal_module.global_constants) {
        result.ast.global_byte_arrays.push_back({
            .name = global.name,
            .bytes = global.bytes,
        });
    }
    const BuiltinUsageMap builtin_usage = analyze_builtin_usage(metal_module);
    const SharedUsageMap shared_usage = analyze_shared_usage(metal_module);
    const BarrierUsageMap barrier_usage = analyze_barrier_usage(metal_module);
    for (const ir::Function& function : metal_module.functions) {
        const StructurizeResult structurized = check_structurizable(function);
        if (!structurized.ok) {
            result.error = "cannot structurize function '" + function.name + "': " +
                           structurized.error;
            return result;
        }
        std::vector<std::optional<ir::AddressSpace>> specializations = {std::nullopt};
        std::uint8_t argument_space_mask = 0;
        if (!function.is_kernel) {
            for (const ir::FunctionArgument& argument : function.arguments) {
                const auto mixed =
                    function.mixed_pointer_address_spaces.find(argument.value);
                if (mixed != function.mixed_pointer_address_spaces.end()) {
                    argument_space_mask |= mixed->second;
                }
            }
        }
        if (argument_space_mask != 0) {
            specializations.clear();
            for (unsigned bit = 1;
                 bit <= static_cast<unsigned>(ir::AddressSpace::kPrivate); ++bit) {
                if ((argument_space_mask & (1u << bit)) != 0) {
                    specializations.push_back(static_cast<ir::AddressSpace>(bit));
                }
            }
        }
        for (const std::optional<ir::AddressSpace> specialization : specializations) {
            AstLowerer lowerer(metal_module, function, builtin_usage, shared_usage,
                               false, specialization);
            LowerToMslResult function_result = lowerer.run();
            const bool structurization_failure =
                !function_result.ok &&
                (function_result.error.find("revisits block") != std::string::npos ||
                 function_result.error.find("no forward reconvergence") !=
                     std::string::npos ||
                 function_result.error.find("nested loop conditional") !=
                     std::string::npos ||
                 function_result.error.find("loop structurization") !=
                     std::string::npos);
            if (structurization_failure) {
                const std::string structurization_error = function_result.error;
                if (barrier_usage.at(function.name)) {
                    function_result.error =
                        "cannot lower function '" + function.name +
                        "': structured CFG lowering failed for a barrier-containing "
                        "call graph: " + structurization_error;
                    return function_result;
                }
                AstLowerer dispatcher_lowerer(metal_module, function, builtin_usage,
                                              shared_usage, true, specialization);
                function_result = dispatcher_lowerer.run();
                if (!function_result.ok) {
                    function_result.error =
                        "structured CFG lowering failed: " + structurization_error +
                        "; CFG dispatcher fallback failed" +
                        (function_result.error.empty()
                             ? std::string{}
                             : ": " + function_result.error);
                }
            }
            if (!function_result.ok) {
                function_result.error =
                    "cannot lower function '" + function.name + "': " +
                    function_result.error;
                return function_result;
            }
            result.ast.functions.insert(result.ast.functions.end(),
                                        function_result.ast.functions.begin(),
                                        function_result.ast.functions.end());
        }
    }
    const MslPrintResult printed = print_msl(result.ast);
    if (!printed.ok) {
        result.error = "typed MSL printer rejected the legalized module";
        return result;
    }
    result.source = printed.source;
    result.ok = true;
    return result;
}

PtxToMslResult compile_ptx_to_msl(std::string_view ptx, const PtxToMslOptions& options) {
    PtxToMslResult result;
    ir::PtxImportOptions import_options;
    import_options.strict = options.strict;
    import_options.entry_name = options.entry_name;
    import_options.source_name = options.source_name;
    ir::PtxImportResult imported = ir::import_ptx(ptx, import_options);
    result.warnings = imported.warnings;
    if (!imported.ok) {
        result.error = imported.error;
        return result;
    }
    result.gpu_ir = imported.module;
    result.gpu_ir.attributes["provenance"] = "generic_ptx_lowering";
    MetalLegalizeResult legalized = legalize_for_metal(result.gpu_ir);
    result.warnings.insert(result.warnings.end(), legalized.warnings.begin(), legalized.warnings.end());
    if (!legalized.ok) {
        result.error = legalized.error;
        return result;
    }
    result.metal_ir = legalized.module;
    LowerToMslResult lowered = lower_to_msl(result.metal_ir);
    result.warnings.insert(result.warnings.end(), lowered.warnings.begin(), lowered.warnings.end());
    if (!lowered.ok) {
        result.error = lowered.error;
        return result;
    }
    result.ast = std::move(lowered.ast);
    result.source = std::move(lowered.source);
    result.ok = true;
    return result;
}

NvvmToMslResult compile_nvvm_to_msl(std::string_view llvm_ir,
                                     std::string_view source_name,
                                     std::string_view entry_name) {
    NvvmToMslResult result;
    ir::NvvmImportOptions options;
    options.source_name = std::string(source_name);
    options.entry_name = std::string(entry_name);
    ir::NvvmImportResult imported = ir::import_nvvm_llvm_ir(llvm_ir, options);
    result.warnings = imported.warnings;
    if (!imported.ok) {
        result.error = imported.error;
        return result;
    }
    result.gpu_ir = imported.module;
    result.gpu_ir.attributes["provenance"] = "generic_nvvm_lowering";
    MetalLegalizeResult legalized = legalize_for_metal(result.gpu_ir);
    result.warnings.insert(result.warnings.end(), legalized.warnings.begin(), legalized.warnings.end());
    if (!legalized.ok) {
        result.error = legalized.error;
        return result;
    }
    result.metal_ir = legalized.module;
    LowerToMslResult lowered = lower_to_msl(result.metal_ir);
    result.warnings.insert(result.warnings.end(), lowered.warnings.begin(), lowered.warnings.end());
    if (!lowered.ok) {
        result.error = lowered.error;
        return result;
    }
    result.ast = std::move(lowered.ast);
    result.source = std::move(lowered.source);
    result.ok = true;
    return result;
}

}  // namespace cumetal::metal
