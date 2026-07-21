#include "cumetal/metal/lower_to_msl.h"

#include "cumetal/ir/ptx_importer.h"
#include "cumetal/ir/nvvm_importer.h"

#include <algorithm>
#include <functional>
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
            MslType result;
            result.kind = MslTypeKind::kStruct;
            result.struct_name = "CuMetalAggregate";
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

struct AstLowerer {
    const ir::Module& module;
    const ir::Function& function;
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

    AstLowerer(const ir::Module& input_module, const ir::Function& input_function)
        : module(input_module), function(input_function) {}

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
        return MslExpression::literal(operand.text, lower_type(operand.type));
    }

    MslStmt declare_result(const ir::Operation& operation, MslExpr initializer) {
        const ir::ValueId value = operation.results.front();
        const MslType type = lower_type(operation.result_types.front());
        values[value] = MslExpression::identifier(value_name(value), type);
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

        if (operation.opcode == ir::OpCode::kCall) {
            const auto callee = operation.attributes.find("callee");
            if (callee == operation.attributes.end()) {
                fail(&operation, "direct call is missing a callee");
                return std::nullopt;
            }
            std::vector<MslExpr> arguments;
            for (const ir::Operand& operand : operation.operands) {
                arguments.push_back(expression_for(operand));
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
            return declare_result(
                operation,
                MslExpression::cast(lower_type(operation.result_types.front()),
                                    expression_for(operation.operands.front()), reinterpret));
        }

        if (operation.opcode == ir::OpCode::kLoad) {
            if (operation.results.empty() || operation.operands.empty()) {
                fail(&operation, "malformed load");
                return std::nullopt;
            }
            const MslType value_type = lower_type(operation.result_types.front());
            const MslType pointer_type =
                MslType::pointer(value_type, MslAddressSpace::kDevice);
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
            const MslType pointer_type =
                MslType::pointer(stored_value->type, MslAddressSpace::kDevice);
            const MslExpr pointer =
                MslExpression::cast(pointer_type, expression_for(operation.operands[0]), true);
            return MslStatement::assignment(
                MslExpression::unary("*", pointer, stored_value->type), stored_value);
        }

        if (operation.opcode == ir::OpCode::kMetalBarrier) {
            return MslStatement::expression(
                MslExpression::call(
                    "threadgroup_barrier",
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
            return declare_result(
                operation,
                MslExpression::call(
                    "simd_shuffle",
                    {expression_for(operation.operands[0]),
                     expression_for(operation.operands[1])},
                    lower_type(operation.result_types.front())));
        }

        if (operation.opcode == ir::OpCode::kMetalBallot ||
            operation.opcode == ir::OpCode::kMetalVote ||
            operation.opcode == ir::OpCode::kMetalReduction ||
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

    bool emit_from(std::size_t block_index, std::vector<MslStmt>* statements) {
        if (block_index >= function.blocks.size()) return false;
        const ir::BasicBlock& block = function.blocks[block_index];
        if (!emitted.insert(block.id).second) {
            return fail(nullptr, "loop structurization is not implemented for block '" + block.name + "'");
        }
        if (!emit_operations(block, statements)) return false;
        const ir::Operation& terminator = block.operations.back();
        if (terminator.opcode == ir::OpCode::kReturn) {
            statements->push_back(MslStatement::return_statement());
            return true;
        }
        if (terminator.opcode == ir::OpCode::kBranch) {
            if (terminator.successors.size() != 1) {
                return fail(&terminator, "malformed unconditional branch");
            }
            const std::size_t target =
                block_indices.at(terminator.successors.front().block);
            if (emitted.contains(function.blocks[target].id) &&
                is_terminal_return_block(function.blocks[target])) {
                statements->push_back(MslStatement::return_statement());
                return true;
            }
            return emit_from(target, statements);
        }
        if (terminator.opcode == ir::OpCode::kCondBranch) {
            if (terminator.successors.size() != 2 || terminator.operands.empty()) {
                return fail(&terminator, "malformed conditional branch");
            }
            const std::size_t first = block_indices.at(terminator.successors[0].block);
            const std::size_t second = block_indices.at(terminator.successors[1].block);
            const bool first_returns = is_terminal_return_block(function.blocks[first]);
            const bool second_returns = is_terminal_return_block(function.blocks[second]);
            if (first_returns == second_returns) {
                return fail(&terminator,
                            "conditional CFG requires if/else region structurization");
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
            statements->push_back(
                MslStatement::if_statement(
                    condition, {MslStatement::return_statement()}));
            emitted.insert(function.blocks[first_returns ? first : second].id);
            return emit_from(first_returns ? second : first, statements);
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

        if (!emit_from(0, &output.statements)) return result;

        if (!function.is_kernel &&
            (needs_thread_position || needs_threadgroup_position ||
             needs_threads_per_threadgroup || needs_threadgroups_per_grid ||
             needs_lane_id)) {
            fail(nullptr,
                 "device functions using GPU builtins require explicit builtin "
                 "parameter threading");
            return result;
        }

        if (needs_thread_position) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_thread_position",
                .attributes = {MslAttribute{.name = "thread_position_in_threadgroup"}},
            });
        }
        if (needs_threadgroup_position) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threadgroup_position",
                .attributes = {MslAttribute{.name = "threadgroup_position_in_grid"}},
            });
        }
        if (needs_threads_per_threadgroup) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threads_per_threadgroup",
                .attributes = {MslAttribute{.name = "threads_per_threadgroup"}},
            });
        }
        if (needs_threadgroups_per_grid) {
            output.parameters.push_back({
                .type = MslType::vector(MslType::uint(), 3),
                .name = "cm_threadgroups_per_grid",
                .attributes = {MslAttribute{.name = "threadgroups_per_grid"}},
            });
        }
        if (needs_lane_id) {
            output.parameters.push_back({
                .type = MslType::uint(),
                .name = "cm_lane_id",
                .attributes = {MslAttribute{.name = "thread_index_in_simdgroup"}},
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
    std::unordered_map<ir::BlockId, std::size_t> indices;
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        indices[function.blocks[i].id] = i;
    }
    for (std::size_t i = 0; i < function.blocks.size(); ++i) {
        const ir::BasicBlock& block = function.blocks[i];
        const ir::Operation& terminator = block.operations.back();
        if (terminator.opcode == ir::OpCode::kBranch) {
            if (terminator.successors.size() != 1) {
                result.error = "block '" + block.name + "' has malformed branch";
                return result;
            }
            const std::size_t target = indices.at(terminator.successors.front().block);
            if (target <= i) {
                result.error = "loop CFG requires the loop-discovery/region structurizer";
                return result;
            }
        } else if (terminator.opcode == ir::OpCode::kCondBranch) {
            if (terminator.successors.size() != 2) {
                result.error = "block '" + block.name + "' has malformed conditional branch";
                return result;
            }
            const ir::BasicBlock& first =
                function.blocks[indices.at(terminator.successors[0].block)];
            const ir::BasicBlock& second =
                function.blocks[indices.at(terminator.successors[1].block)];
            if (!is_terminal_return_block(first) && !is_terminal_return_block(second)) {
                result.error = "if/else region discovery is not implemented for block '" +
                               block.name + "'";
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
    for (const ir::Function& function : metal_module.functions) {
        const StructurizeResult structurized = check_structurizable(function);
        if (!structurized.ok) {
            result.error = "cannot structurize function '" + function.name + "': " +
                           structurized.error;
            return result;
        }
        AstLowerer lowerer(metal_module, function);
        LowerToMslResult function_result = lowerer.run();
        if (!function_result.ok) return function_result;
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
                                     std::string_view source_name) {
    NvvmToMslResult result;
    ir::NvvmImportOptions options;
    options.source_name = std::string(source_name);
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
