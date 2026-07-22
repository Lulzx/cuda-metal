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
        spaces_.push_back(std::nullopt);
        return node;
    }

    std::size_t find(std::size_t node) {
        if (parents_[node] != node) parents_[node] = find(parents_[node]);
        return parents_[node];
    }

    bool unite(std::size_t left, std::size_t right) {
        left = find(left);
        right = find(right);
        if (left == right) return true;
        if (spaces_[left].has_value() && spaces_[right].has_value() &&
            spaces_[left] != spaces_[right]) {
            return false;
        }
        if (ranks_[left] < ranks_[right]) std::swap(left, right);
        parents_[right] = left;
        if (ranks_[left] == ranks_[right]) ++ranks_[left];
        if (!spaces_[left].has_value()) spaces_[left] = spaces_[right];
        return true;
    }

    bool seed(std::size_t node, ir::AddressSpace space) {
        node = find(node);
        if (space == ir::AddressSpace::kNone) return true;
        if (spaces_[node].has_value() && spaces_[node] != space) return false;
        spaces_[node] = space;
        return true;
    }

    std::optional<ir::AddressSpace> space(std::size_t node) {
        return spaces_[find(node)];
    }

private:
    std::vector<std::size_t> parents_;
    std::vector<std::uint8_t> ranks_;
    std::vector<std::optional<ir::AddressSpace>> spaces_;
};

AddressSpaceResolution resolve_generic_address_spaces(ir::Module* module) {
    AddressSpaceConstraints constraints;
    std::unordered_map<ir::ValueId, std::size_t> value_nodes;
    std::vector<std::optional<std::size_t>> return_nodes(module->functions.size());
    std::unordered_map<std::string, std::size_t> function_indices;

    auto add_value = [&](ir::ValueId value) {
        if (!value_nodes.contains(value)) value_nodes[value] = constraints.add_node();
    };
    for (std::size_t function_index = 0; function_index < module->functions.size();
         ++function_index) {
        ir::Function& function = module->functions[function_index];
        function_indices[function.name] = function_index;
        if (function.return_type.is_pointer()) {
            return_nodes[function_index] = constraints.add_node();
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
            if (!function.generic_pointer_values.contains(argument.value) &&
                !constraints.seed(value_nodes.at(argument.value),
                                  argument.type.address_space)) {
                return {false, "conflicting concrete argument address spaces in '" +
                                   function.name + "'"};
            }
        }
        for (const ir::BasicBlock& block : function.blocks) {
            for (const ir::BlockArgument& argument : block.arguments) {
                if (!argument.type.is_pointer()) continue;
                add_value(argument.value);
                if (!function.generic_pointer_values.contains(argument.value) &&
                    !constraints.seed(value_nodes.at(argument.value),
                                      argument.type.address_space)) {
                    return {false, "conflicting block-argument address spaces in '" +
                                       function.name + "'"};
                }
            }
            for (const ir::Operation& operation : block.operations) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i >= operation.result_types.size() ||
                        !operation.result_types[i].is_pointer()) {
                        continue;
                    }
                    add_value(operation.results[i]);
                    const bool concrete =
                        !function.generic_pointer_values.contains(operation.results[i]) ||
                        operation.opcode == ir::OpCode::kAlloca ||
                        (operation.attributes.contains("pointer_integer") &&
                         operation.attributes.at("pointer_integer") == "true");
                    const ir::AddressSpace seed =
                        operation.opcode == ir::OpCode::kAlloca
                            ? ir::AddressSpace::kPrivate
                            : operation.result_types[i].address_space;
                    if (concrete &&
                        !constraints.seed(value_nodes.at(operation.results[i]), seed)) {
                        return {false, "conflicting result address spaces in '" +
                                           function.name + "'"};
                    }
                }
            }
        }
    }

    auto constrain_operand = [&](std::size_t node, const ir::Operand& operand) {
        if (operand.kind == ir::OperandKind::kValue &&
            value_nodes.contains(operand.value)) {
            return constraints.unite(node, value_nodes.at(operand.value));
        }
        if (operand.kind == ir::OperandKind::kSymbol && operand.type.is_pointer()) {
            return constraints.seed(node, operand.type.address_space);
        }
        return true;
    };

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
                            return_nodes[callee_index->second].has_value() &&
                            !constraints.unite(
                                value_nodes.at(operation.results.front()),
                                *return_nodes[callee_index->second])) {
                            return {false, "device helper return requires address-space specialization: " +
                                               callee.name};
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
                    if (!constraints.unite(
                            value_nodes.at(target->arguments[i].value),
                            value_nodes.at(successor.arguments[i]))) {
                        const auto target_space = constraints.space(
                            value_nodes.at(target->arguments[i].value));
                        const auto incoming_space = constraints.space(
                            value_nodes.at(successor.arguments[i]));
                        return {false,
                                "PHI merges incompatible pointer address spaces in '" +
                                    function.name + "' at block '" + target->name +
                                    "' (argument " +
                                    std::to_string(target->arguments[i].value) + " is " +
                                    (target_space.has_value()
                                         ? std::string(ir::address_space_name(*target_space))
                                         : "unresolved") +
                                    ", incoming " +
                                    std::to_string(successor.arguments[i]) + " is " +
                                    (incoming_space.has_value()
                                         ? std::string(ir::address_space_name(*incoming_space))
                                         : "unresolved") +
                                    ")"};
                    }
                }
            }
        }
    }

    auto resolve_type = [&](ir::ValueId value, ir::Type* type,
                            std::string_view context) -> std::optional<std::string> {
        if (!type->is_pointer() || !value_nodes.contains(value)) return std::nullopt;
        const auto space = constraints.space(value_nodes.at(value));
        if (!space.has_value()) {
            return "unresolved generic pointer address space for " +
                   std::string(context) + " value " + std::to_string(value);
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
            if (const auto error = resolve_type(argument.value, &argument.type,
                                                function.name + " argument")) {
                return {false, *error};
            }
        }
        for (ir::BasicBlock& block : function.blocks) {
            for (ir::BlockArgument& argument : block.arguments) {
                if (const auto error = resolve_type(argument.value, &argument.type,
                                                    function.name + " block argument")) {
                    return {false, *error};
                }
            }
            for (ir::Operation& operation : block.operations) {
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i < operation.result_types.size()) {
                        if (const auto error = resolve_type(
                                operation.results[i], &operation.result_types[i],
                                function.name + " result")) {
                            return {false, *error};
                        }
                    }
                }
                for (ir::Operand& operand : operation.operands) {
                    if (operand.kind == ir::OperandKind::kValue &&
                        value_nodes.contains(operand.value) && operand.type.is_pointer()) {
                        const auto space = constraints.space(value_nodes.at(operand.value));
                        if (!space.has_value()) {
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

struct AstLowerer {
    const ir::Module& module;
    const ir::Function& function;
    const BuiltinUsageMap& builtin_usage;
    LowerToMslResult result;
    MslFunction output;
    std::unordered_map<ir::ValueId, MslExpr> values;
    std::unordered_map<ir::BlockId, std::size_t> block_indices;
    std::unordered_set<ir::BlockId> emitted;
    bool needs_thread_position = false;
    bool needs_threadgroup_position = false;
    bool needs_threads_per_threadgroup = false;
    bool needs_threadgroups_per_grid = false;
    bool needs_lane_id = false;
    bool cfg_dispatcher_mode = false;
    bool force_cfg_dispatcher = false;
    std::size_t edge_temporary_index = 0;

    AstLowerer(const ir::Module& input_module, const ir::Function& input_function,
               const BuiltinUsageMap& input_builtin_usage,
               bool force_dispatcher = false)
        : module(input_module),
          function(input_function),
          builtin_usage(input_builtin_usage),
          force_cfg_dispatcher(force_dispatcher) {
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

    MslExpr expression_for(const ir::Operand& operand) {
        if (operand.kind == ir::OperandKind::kValue) {
            const auto value = values.find(operand.value);
            if (value != values.end()) return value->second;
            return MslExpression::identifier(value_name(operand.value), lower_type(operand.type));
        }
        if (operand.kind == ir::OperandKind::kSymbol) {
            return MslExpression::identifier(operand.text, lower_type(operand.type));
        }
        return MslExpression::literal(operand.text == "null" ? "nullptr" : operand.text,
                                      lower_type(operand.type));
    }

    MslStmt declare_result(const ir::Operation& operation, MslExpr initializer) {
        const ir::ValueId value = operation.results.front();
        const MslType type = lower_type(operation.result_types.front());
        values[value] = MslExpression::identifier(value_name(value), type);
        if (cfg_dispatcher_mode) {
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
            MslType expression_type = lower_type(operation.result_types.front());
            if (operation.attributes.contains("signed") &&
                operation.attributes.at("signed") == "true" &&
                operation.operands[0].type.kind == ir::TypeKind::kInteger) {
                const MslType signed_type =
                    MslType::sint(operation.operands[0].type.bit_width);
                left = MslExpression::cast(signed_type, left);
                right = MslExpression::cast(signed_type, right);
                expression_type = signed_type;
            }
            MslExpr expression = MslExpression::binary(
                binary, left, right, expression_type);
            if (operation.attributes.contains("combined") &&
                operation.attributes.at("combined") == "mul_add" &&
                operation.operands.size() >= 3) {
                expression = MslExpression::binary(
                    "+", expression, expression_for(operation.operands[2]),
                    lower_type(operation.result_types.front()));
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
                                    lower_type(operation.result_types.front())));
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
                                         lower_type(operation.result_types.front())));
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
                                    lower_type(operation.result_types.front())));
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
                    MslExpression::cast(lower_type(operation.result_types.front()), absolute));
            }
            if (callee->second == "__cumetal_ffs") {
                if (operation.results.empty() || operation.operands.size() != 1) {
                    fail(&operation, "malformed CUDA ffs builtin");
                    return std::nullopt;
                }
                const MslType result_type = lower_type(operation.result_types.front());
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
                                            : lower_type(operation.result_types.front());
            MslExpr call = MslExpression::call(callee->second, std::move(arguments), return_type);
            if (operation.results.empty()) return MslStatement::expression(std::move(call));
            return declare_result(operation, std::move(call));
        }

        if (operation.opcode == ir::OpCode::kNegate) {
            return declare_result(
                operation,
                MslExpression::unary("-", expression_for(operation.operands.front()),
                                     lower_type(operation.result_types.front())));
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
                    lower_type(operation.result_types.front())));
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
            if (operation.operands.front().type == operation.result_types.front()) {
                return declare_result(operation,
                                      expression_for(operation.operands.front()));
            }
            if (operation.attributes.contains("bitcast") &&
                operation.attributes.at("bitcast") == "true") {
                return declare_result(
                    operation,
                    MslExpression::bitcast(lower_type(operation.result_types.front()),
                                           expression_for(operation.operands.front())));
            }
            if (operation.attributes.contains("pointer_integer") &&
                operation.attributes.at("pointer_integer") == "true") {
                return declare_result(
                    operation,
                    MslExpression::cast(lower_type(operation.result_types.front()),
                                        expression_for(operation.operands.front()), true));
            }
            return declare_result(
                operation,
                MslExpression::cast(lower_type(operation.result_types.front()),
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
                "&", storage, lower_type(operation.result_types.front()));
            if (cfg_dispatcher_mode) return std::nullopt;
            return MslStatement::variable(storage_type, storage_name, std::nullopt, false);
        }

        if (operation.opcode == ir::OpCode::kLoad) {
            if (operation.results.empty() || operation.operands.empty()) {
                fail(&operation, "malformed load");
                return std::nullopt;
            }
            const MslType value_type = lower_type(operation.result_types.front());
            const ir::AddressSpace address_space =
                operation.operands.front().type.is_pointer()
                    ? operation.operands.front().type.address_space
                    : ir::AddressSpace::kDevice;
            const MslType pointer_type =
                MslType::pointer(value_type, lower_address_space(address_space));
            const MslExpr pointer =
                MslExpression::cast(pointer_type, expression_for(operation.operands.front()), true);
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
            const ir::AddressSpace address_space =
                operation.operands.front().type.is_pointer()
                    ? operation.operands.front().type.address_space
                    : ir::AddressSpace::kDevice;
            const MslType pointer_type =
                MslType::pointer(stored_value->type, lower_address_space(address_space));
            const MslExpr pointer =
                MslExpression::cast(pointer_type, expression_for(operation.operands[0]), true);
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
            std::string intrinsic = "simd_shuffle";
            if (operation.attributes.contains("kind")) {
                const std::string& kind = operation.attributes.at("kind");
                if (kind == "down") intrinsic = "simd_shuffle_down";
                else if (kind == "up") intrinsic = "simd_shuffle_up";
            }
            return declare_result(
                operation,
                MslExpression::call(
                    intrinsic,
                    {expression_for(operation.operands[0]),
                     expression_for(operation.operands[1])},
                    lower_type(operation.result_types.front())));
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
                MslExpression::cast(lower_type(operation.result_types.front()), voted));
        }

        if (operation.opcode == ir::OpCode::kMetalReduction ||
            operation.opcode == ir::OpCode::kMetalAtomic) {
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

    bool bind_block_arguments(const ir::BasicBlock& block,
                              const ir::Successor* incoming) {
        if (incoming == nullptr) return true;
        if (incoming->arguments.size() != block.arguments.size()) {
            return fail(nullptr, "branch argument count does not match block arguments for '" +
                                     block.name + "'");
        }
        std::vector<MslExpr> incoming_values;
        incoming_values.reserve(incoming->arguments.size());
        for (std::size_t i = 0; i < incoming->arguments.size(); ++i) {
            incoming_values.push_back(expression_for(ir::Operand::value_ref(
                incoming->arguments[i], block.arguments[i].type)));
        }
        for (std::size_t i = 0; i < block.arguments.size(); ++i) {
            values[block.arguments[i].value] = std::move(incoming_values[i]);
        }
        return true;
    }

    std::optional<std::size_t> find_nearest_common_successor(std::size_t first,
                                                              std::size_t second) const {
        auto distances = [&](std::size_t start) {
            std::unordered_map<std::size_t, std::size_t> result;
            std::queue<std::size_t> pending;
            result[start] = 0;
            pending.push(start);
            while (!pending.empty()) {
                const std::size_t index = pending.front();
                pending.pop();
                const ir::Operation& terminator = function.blocks[index].operations.back();
                for (const ir::Successor& successor : terminator.successors) {
                    const std::size_t target = block_indices.at(successor.block);
                    if (result.emplace(target, result.at(index) + 1).second) {
                        pending.push(target);
                    }
                }
            }
            return result;
        };
        const auto first_distances = distances(first);
        const auto second_distances = distances(second);
        std::optional<std::size_t> best;
        std::size_t best_max = std::numeric_limits<std::size_t>::max();
        std::size_t best_sum = std::numeric_limits<std::size_t>::max();
        for (const auto& [candidate, first_distance] : first_distances) {
            const auto other = second_distances.find(candidate);
            if (other == second_distances.end()) continue;
            const std::size_t maximum = std::max(first_distance, other->second);
            const std::size_t sum = first_distance + other->second;
            if (maximum < best_max || (maximum == best_max && sum < best_sum)) {
                best = candidate;
                best_max = maximum;
                best_sum = sum;
            }
        }
        return best;
    }

    bool assign_join_arguments(const ir::BasicBlock& join,
                               const ir::Successor& successor,
                               std::vector<MslStmt>* statements) {
        if (successor.arguments.size() != join.arguments.size()) {
            return fail(nullptr, "branch argument count does not match join block '" +
                                     join.name + "'");
        }
        for (std::size_t i = 0; i < join.arguments.size(); ++i) {
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
        const bool first_backedge = can_reach(first, header_index);
        const bool second_backedge = can_reach(second, header_index);
        if (first_backedge == second_backedge) return std::nullopt;
        return first_backedge ? std::pair{first, second} : std::pair{second, first};
    }

    bool assign_loop_arguments(const ir::BasicBlock& header,
                               const ir::Successor& backedge,
                               std::vector<MslStmt>* statements) {
        if (backedge.arguments.size() != header.arguments.size()) {
            return fail(nullptr, "loop backedge argument count does not match header '" +
                                     header.name + "'");
        }
        std::vector<MslExpr> temporaries;
        temporaries.reserve(header.arguments.size());
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            const MslType type = lower_type(header.arguments[i].type);
            const std::string name = value_name(header.arguments[i].value) + "_next";
            statements->push_back(MslStatement::variable(
                type, name,
                expression_for(ir::Operand::value_ref(backedge.arguments[i],
                                                       header.arguments[i].type)),
                true));
            temporaries.push_back(MslExpression::identifier(name, type));
        }
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            statements->push_back(MslStatement::assignment(
                values.at(header.arguments[i].value), temporaries[i]));
        }
        return true;
    }

    bool emit_loop_region(std::size_t block_index, std::size_t header_index,
                          const ir::Successor* incoming,
                          std::vector<MslStmt>* statements) {
        const ir::BasicBlock& header = function.blocks[header_index];
        if (block_index == header_index) {
            return incoming != nullptr &&
                   assign_loop_arguments(header, *incoming, statements);
        }
        const ir::BasicBlock& block = function.blocks[block_index];
        if (!emitted.insert(block.id).second) {
            return fail(nullptr, "loop body revisits block '" + block.name + "'");
        }
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
            return emit_loop_region(block_indices.at(successor.block), header_index,
                                    &successor, statements);
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
            return emit_loop_region(continuation_index, header_index,
                                    &terminator.successors[successor_index], statements);
        }

        const auto join = find_nearest_common_successor(first, second);
        if (!join || *join == block_index) {
            return fail(&terminator, "nested loop conditional has no reconvergence");
        }
        if (*join != header_index) {
            const ir::BasicBlock& join_block = function.blocks[*join];
            for (const ir::BlockArgument& argument : join_block.arguments) {
                const MslType type = lower_type(argument.type);
                const std::string name = value_name(argument.value);
                values[argument.value] = MslExpression::identifier(name, type);
                statements->push_back(
                    MslStatement::variable(type, name, std::nullopt, false));
            }
        }
        std::vector<MslStmt> first_statements;
        std::vector<MslStmt> second_statements;
        if (!emit_loop_region(first, *join, &terminator.successors[0],
                              &first_statements) ||
            !emit_loop_region(second, *join, &terminator.successors[1],
                              &second_statements)) {
            return false;
        }
        statements->push_back(MslStatement::if_statement(
            expression_for(terminator.operands.front()),
            std::move(first_statements), std::move(second_statements)));
        if (*join == header_index) return true;
        return emit_loop_region(*join, header_index, nullptr, statements);
    }

    bool emit_natural_loop(std::size_t header_index, const ir::Successor* incoming,
                           std::vector<MslStmt>* statements) {
        const ir::BasicBlock& header = function.blocks[header_index];
        const auto body_and_exit = loop_body_and_exit(header_index);
        if (!body_and_exit || incoming == nullptr ||
            incoming->arguments.size() != header.arguments.size()) {
            return fail(nullptr, "malformed natural loop header '" + header.name + "'");
        }

        std::vector<MslExpr> initial_values;
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            initial_values.push_back(expression_for(ir::Operand::value_ref(
                incoming->arguments[i], header.arguments[i].type)));
        }
        for (std::size_t i = 0; i < header.arguments.size(); ++i) {
            const MslType type = lower_type(header.arguments[i].type);
            const std::string name = value_name(header.arguments[i].value);
            values[header.arguments[i].value] = MslExpression::identifier(name, type);
            statements->push_back(
                MslStatement::variable(type, name, initial_values[i], false));
        }

        const std::size_t body_index = body_and_exit->first;
        const std::size_t exit_index = body_and_exit->second;
        const ir::BasicBlock& exit_block = function.blocks[exit_index];
        for (const ir::BlockArgument& argument : exit_block.arguments) {
            const MslType type = lower_type(argument.type);
            const std::string name = value_name(argument.value);
            values[argument.value] = MslExpression::identifier(name, type);
            statements->push_back(MslStatement::variable(type, name, std::nullopt, false));
        }

        std::vector<MslStmt> loop_statements;
        if (!emit_operations(header, &loop_statements)) return false;
        const ir::Operation& terminator = header.operations.back();
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
        if (!assign_join_arguments(exit_block,
                                   terminator.successors[exit_successor_index],
                                   &exit_statements)) {
            return false;
        }
        exit_statements.push_back(MslStatement::break_statement());
        loop_statements.push_back(MslStatement::if_statement(
            exit_condition, std::move(exit_statements)));
        if (!emit_loop_region(body_index, header_index,
                              &terminator.successors[body_successor_index],
                              &loop_statements)) {
            return false;
        }
        statements->push_back(MslStatement::while_statement(
            MslExpression::literal("true", MslType::boolean()),
            std::move(loop_statements)));
        return emit_from(exit_index, statements);
    }

    bool requires_cfg_dispatcher() const {
        for (std::size_t source = 0; source < function.blocks.size(); ++source) {
            const ir::Operation& terminator = function.blocks[source].operations.back();
            for (const ir::Successor& successor : terminator.successors) {
                const std::size_t target = block_indices.at(successor.block);
                if (target <= source && can_reach(target, source) &&
                    !loop_body_and_exit(target).has_value()) {
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
        temporaries.reserve(target.arguments.size());
        for (std::size_t i = 0; i < target.arguments.size(); ++i) {
            const MslType type = lower_type(target.arguments[i].type);
            const std::string temporary =
                "cm_edge_" + std::to_string(edge_temporary_index++) + "_" +
                std::to_string(i);
            statements->push_back(MslStatement::variable(
                type, temporary,
                expression_for(ir::Operand::value_ref(successor.arguments[i],
                                                       target.arguments[i].type)),
                true));
            temporaries.push_back(MslExpression::identifier(temporary, type));
        }
        for (std::size_t i = 0; i < target.arguments.size(); ++i) {
            statements->push_back(MslStatement::assignment(
                values.at(target.arguments[i].value), temporaries[i]));
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
                const MslType type = lower_type(argument.type);
                const std::string name = value_name(argument.value);
                values[argument.value] = MslExpression::identifier(name, type);
                statements->push_back(MslStatement::variable(type, name));
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
                        lower_type(operation.result_types.front()));
                    continue;
                }
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    const ir::ValueId value = operation.results[i];
                    const MslType type = lower_type(operation.result_types[i]);
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
        if (loop_body_and_exit(block_index)) {
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
                    const MslType type = lower_type(argument.type);
                    const MslExpr identifier =
                        MslExpression::identifier(value_name(argument.value), type);
                    values[argument.value] = identifier;
                    statements->push_back(MslStatement::variable(
                        type, value_name(argument.value), std::nullopt, false));
                }
                std::vector<MslStmt> first_statements;
                std::vector<MslStmt> second_statements;
                if (!emit_loop_region(first, *join, &terminator.successors[0],
                                      &first_statements) ||
                    !emit_loop_region(second, *join, &terminator.successors[1],
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
        output.name = function.name;
        output.return_type = lower_type(function.return_type);
        output.is_kernel = function.is_kernel;
        for (std::size_t i = 0; i < function.arguments.size(); ++i) {
            const ir::FunctionArgument& argument = function.arguments[i];
            MslType type;
            std::vector<MslAttribute> attributes;
            if (!function.is_kernel) {
                type = lower_type(argument.type);
            } else if (argument.type.is_pointer()) {
                type = lower_type(argument.type);
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
                    MslExpression::identifier(value_name(argument.value), lower_type(argument.type));
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
    for (const ir::Function& function : metal_module.functions) {
        const StructurizeResult structurized = check_structurizable(function);
        if (!structurized.ok) {
            result.error = "cannot structurize function '" + function.name + "': " +
                           structurized.error;
            return result;
        }
        AstLowerer lowerer(metal_module, function, builtin_usage);
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
            AstLowerer dispatcher_lowerer(metal_module, function, builtin_usage, true);
            function_result = dispatcher_lowerer.run();
        }
        if (!function_result.ok) {
            function_result.error =
                "cannot lower function '" + function.name + "': " + function_result.error;
            return function_result;
        }
        result.ast.functions.insert(result.ast.functions.end(),
                                    function_result.ast.functions.begin(),
                                    function_result.ast.functions.end());
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
