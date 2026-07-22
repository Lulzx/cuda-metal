#include "cumetal/ir/ir.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <unordered_set>

namespace cumetal::ir {
namespace {

std::uint32_t natural_size(const Type& type) {
    if (type.kind == TypeKind::kPointer) {
        return 8;
    }
    if (type.kind == TypeKind::kPredicate) {
        return 1;
    }
    if (type.kind == TypeKind::kInteger || type.kind == TypeKind::kFloat) {
        return std::max<std::uint32_t>(1, type.bit_width / 8);
    }
    if (type.kind == TypeKind::kVector && !type.elements.empty()) {
        return natural_size(type.elements.front()) * type.lanes;
    }
    if (type.kind == TypeKind::kAggregate) {
        std::uint32_t size = 0;
        for (const Type& element : type.elements) {
            size += natural_size(element);
        }
        return size;
    }
    return 0;
}

void add_diagnostic(VerifyResult* result, const SourceLocation& location, std::string message) {
    result->diagnostics.push_back({.location = location, .message = std::move(message)});
}

struct ValueDefinition {
    Type type;
    BlockId block = kInvalidBlock;
    std::size_t operation_index = 0;
    bool function_argument = false;
    bool block_argument = false;
};

}  // namespace

Type Type::void_type() {
    return {};
}

Type Type::predicate() {
    // PTX predicate registers are canonicalized to ordinary i1 SSA values.
    return integer(1);
}

Type Type::integer(std::uint32_t bits) {
    return {.kind = TypeKind::kInteger, .bit_width = bits};
}

Type Type::floating(std::uint32_t bits) {
    return {.kind = TypeKind::kFloat, .bit_width = bits};
}

Type Type::vector(Type element, std::uint32_t vector_lanes) {
    return {
        .kind = TypeKind::kVector,
        .bit_width = element.bit_width,
        .lanes = vector_lanes,
        .elements = {std::move(element)},
    };
}

Type Type::pointer(Type pointee_type, AddressSpace pointer_address_space) {
    return {
        .kind = TypeKind::kPointer,
        .bit_width = 64,
        .address_space = pointer_address_space,
        .elements = {std::move(pointee_type)},
    };
}

Type Type::aggregate(std::vector<Type> aggregate_elements, std::string aggregate_name) {
    return {
        .kind = TypeKind::kAggregate,
        .elements = std::move(aggregate_elements),
        .name = std::move(aggregate_name),
    };
}

bool Type::is_scalar() const {
    return kind == TypeKind::kPredicate || kind == TypeKind::kInteger || kind == TypeKind::kFloat;
}

bool Type::is_pointer() const {
    return kind == TypeKind::kPointer;
}

const Type* Type::pointee() const {
    return is_pointer() && !elements.empty() ? &elements.front() : nullptr;
}

std::string Type::str() const {
    switch (kind) {
        case TypeKind::kVoid:
            return "void";
        case TypeKind::kPredicate:
            return "pred";
        case TypeKind::kInteger:
            return "i" + std::to_string(bit_width);
        case TypeKind::kFloat:
            return "f" + std::to_string(bit_width);
        case TypeKind::kVector:
            return "vector<" + std::to_string(lanes) + "x" +
                   (elements.empty() ? std::string("?") : elements.front().str()) + ">";
        case TypeKind::kPointer:
            return "ptr<" + std::string(address_space_name(address_space)) + ", " +
                   (elements.empty() ? std::string("?") : elements.front().str()) + ">";
        case TypeKind::kAggregate: {
            std::string out = "struct<";
            for (std::size_t i = 0; i < elements.size(); ++i) {
                if (i != 0) {
                    out += ", ";
                }
                out += elements[i].str();
            }
            return out + ">";
        }
    }
    return "?";
}

std::string SourceLocation::str() const {
    if (file.empty()) {
        return line == 0 ? std::string{} : ("line " + std::to_string(line));
    }
    std::string out = file;
    if (line != 0) {
        out += ":" + std::to_string(line);
        if (column != 0) {
            out += ":" + std::to_string(column);
        }
    }
    return out;
}

Operand Operand::value_ref(ValueId operand_value, Type operand_type) {
    return {.kind = OperandKind::kValue, .value = operand_value, .type = std::move(operand_type)};
}

Operand Operand::immediate(std::string operand_text, Type operand_type) {
    return {
        .kind = OperandKind::kImmediate,
        .type = std::move(operand_type),
        .text = std::move(operand_text),
    };
}

Operand Operand::symbol(std::string operand_text, Type operand_type) {
    return {
        .kind = OperandKind::kSymbol,
        .type = std::move(operand_type),
        .text = std::move(operand_text),
    };
}

bool Operation::is_terminator() const {
    return opcode == OpCode::kBranch || opcode == OpCode::kCondBranch ||
           opcode == OpCode::kReturn || opcode == OpCode::kTrap;
}

const BasicBlock* Function::find_block(BlockId id) const {
    const auto it = std::find_if(blocks.begin(), blocks.end(), [id](const BasicBlock& block) {
        return block.id == id;
    });
    return it == blocks.end() ? nullptr : &*it;
}

BasicBlock* Function::find_block(BlockId id) {
    const auto it = std::find_if(blocks.begin(), blocks.end(), [id](const BasicBlock& block) {
        return block.id == id;
    });
    return it == blocks.end() ? nullptr : &*it;
}

ValueId Builder::next_value() {
    return next_value_++;
}

BlockId Builder::next_block() {
    return next_block_++;
}

std::string_view address_space_name(AddressSpace address_space) {
    switch (address_space) {
        case AddressSpace::kNone:
            return "none";
        case AddressSpace::kDevice:
            return "device";
        case AddressSpace::kConstant:
            return "constant";
        case AddressSpace::kThreadgroup:
            return "threadgroup";
        case AddressSpace::kPrivate:
            return "private";
    }
    return "unknown";
}

std::string_view memory_scope_name(MemoryScope scope) {
    switch (scope) {
        case MemoryScope::kNone: return "none";
        case MemoryScope::kSimdgroup: return "simdgroup";
        case MemoryScope::kThreadgroup: return "threadgroup";
        case MemoryScope::kDevice: return "device";
        case MemoryScope::kSystem: return "system";
    }
    return "unknown";
}

std::string_view memory_ordering_name(MemoryOrdering ordering) {
    switch (ordering) {
        case MemoryOrdering::kNone: return "none";
        case MemoryOrdering::kRelaxed: return "relaxed";
        case MemoryOrdering::kAcquire: return "acquire";
        case MemoryOrdering::kRelease: return "release";
        case MemoryOrdering::kAcquireRelease: return "acq_rel";
        case MemoryOrdering::kSequentiallyConsistent: return "seq_cst";
    }
    return "unknown";
}

bool is_gpu_semantic_opcode(OpCode opcode) {
    return opcode >= OpCode::kThreadId && opcode <= OpCode::kReduction;
}

bool is_metal_opcode(OpCode opcode) {
    return opcode >= OpCode::kMetalThreadPosition &&
           opcode <= OpCode::kMetalThreadgroupArgument;
}

std::string_view opcode_name(OpCode opcode) {
    switch (opcode) {
        case OpCode::kInvalid: return "invalid";
        case OpCode::kConstant: return "constant";
        case OpCode::kParameter: return "parameter";
        case OpCode::kAdd: return "add";
        case OpCode::kSub: return "sub";
        case OpCode::kMul: return "mul";
        case OpCode::kDiv: return "div";
        case OpCode::kRemainder: return "rem";
        case OpCode::kFma: return "fma";
        case OpCode::kNegate: return "neg";
        case OpCode::kBitAnd: return "and";
        case OpCode::kBitOr: return "or";
        case OpCode::kBitXor: return "xor";
        case OpCode::kShiftLeft: return "shl";
        case OpCode::kShiftRight: return "shr";
        case OpCode::kCompare: return "compare";
        case OpCode::kSelect: return "select";
        case OpCode::kAggregateConstruct: return "aggregate_construct";
        case OpCode::kAggregateExtract: return "aggregate_extract";
        case OpCode::kConvert: return "convert";
        case OpCode::kAddressSpaceCast: return "addrspace_cast";
        case OpCode::kAlloca: return "alloca";
        case OpCode::kPointerOffset: return "pointer_offset";
        case OpCode::kLoad: return "load";
        case OpCode::kStore: return "store";
        case OpCode::kCall: return "call";
        case OpCode::kThreadId: return "gpu.thread_id";
        case OpCode::kThreadgroupId: return "gpu.threadgroup_id";
        case OpCode::kThreadgroupSize: return "gpu.threadgroup_size";
        case OpCode::kGridSize: return "gpu.grid_size";
        case OpCode::kLaneId: return "gpu.lane_id";
        case OpCode::kSimdgroupSize: return "gpu.simdgroup_size";
        case OpCode::kBarrier: return "gpu.barrier";
        case OpCode::kFence: return "gpu.fence";
        case OpCode::kAtomic: return "gpu.atomic";
        case OpCode::kShuffle: return "gpu.shuffle";
        case OpCode::kBallot: return "gpu.ballot";
        case OpCode::kVote: return "gpu.vote";
        case OpCode::kReduction: return "gpu.reduction";
        case OpCode::kBranch: return "branch";
        case OpCode::kCondBranch: return "cond_branch";
        case OpCode::kReturn: return "return";
        case OpCode::kTrap: return "trap";
        case OpCode::kMetalThreadPosition: return "metal.thread_position";
        case OpCode::kMetalThreadgroupPosition: return "metal.threadgroup_position";
        case OpCode::kMetalThreadsPerThreadgroup: return "metal.threads_per_threadgroup";
        case OpCode::kMetalThreadgroupsPerGrid: return "metal.threadgroups_per_grid";
        case OpCode::kMetalLaneId: return "metal.lane_id";
        case OpCode::kMetalBarrier: return "metal.barrier";
        case OpCode::kMetalFence: return "metal.fence";
        case OpCode::kMetalAtomic: return "metal.atomic";
        case OpCode::kMetalShuffle: return "metal.shuffle";
        case OpCode::kMetalBallot: return "metal.ballot";
        case OpCode::kMetalVote: return "metal.vote";
        case OpCode::kMetalReduction: return "metal.reduction";
        case OpCode::kMetalBufferArgument: return "metal.buffer_argument";
        case OpCode::kMetalThreadgroupArgument: return "metal.threadgroup_argument";
    }
    return "unknown";
}

VerifyResult verify(const Module& module) {
    VerifyResult result;
    std::unordered_set<std::string> function_names;

    for (const Function& function : module.functions) {
        if (function.name.empty() || !function_names.insert(function.name).second) {
            add_diagnostic(&result, {}, "function names must be non-empty and unique");
        }
    }

    std::unordered_map<std::string, std::unordered_set<std::string>> call_graph;
    for (const Function& function : module.functions) {
        if (function.name.empty()) continue;
        if (function.blocks.empty()) {
            add_diagnostic(&result, {}, "function '" + function.name + "' has no basic blocks");
            continue;
        }
        if (function.is_kernel && !function.kernel_abi.has_value()) {
            add_diagnostic(&result, {}, "kernel '" + function.name + "' is missing ABI metadata");
        }
        if (function.kernel_abi.has_value() && function.kernel_abi->required_simd_width != 32) {
            add_diagnostic(&result, {}, "kernel '" + function.name + "' must require SIMD width 32");
        }

        for (const FunctionArgument& argument : function.arguments) {
            if (argument.type.is_pointer() &&
                !function.pointer_provenance.contains(argument.value)) {
                add_diagnostic(&result, {},
                               "pointer argument '" + argument.name +
                                   "' is missing pointer provenance");
            }
        }

        std::unordered_map<BlockId, std::size_t> block_indices;
        std::unordered_map<std::string, BlockId> block_names;
        for (std::size_t i = 0; i < function.blocks.size(); ++i) {
            const BasicBlock& block = function.blocks[i];
            if (block.id == kInvalidBlock || !block_indices.emplace(block.id, i).second) {
                add_diagnostic(&result, {}, "function '" + function.name + "' has duplicate/invalid block IDs");
            }
            if (block.name.empty() || !block_names.emplace(block.name, block.id).second) {
                add_diagnostic(&result, {}, "function '" + function.name + "' has duplicate/empty block names");
            }
            if (block.operations.empty() || !block.operations.back().is_terminator()) {
                add_diagnostic(&result, {}, "block '" + block.name + "' is missing a terminator");
            }
            for (std::size_t op_index = 0; op_index + 1 < block.operations.size(); ++op_index) {
                if (block.operations[op_index].is_terminator()) {
                    add_diagnostic(&result, block.operations[op_index].location,
                                   "terminator must be the final operation in block '" + block.name + "'");
                }
            }
        }

        const std::size_t block_count = function.blocks.size();
        std::vector<std::vector<std::size_t>> predecessors(block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
            const BasicBlock& block = function.blocks[i];
            for (const Operation& operation : block.operations) {
                for (const Successor& successor : operation.successors) {
                    const auto target = block_indices.find(successor.block);
                    if (target == block_indices.end()) {
                        add_diagnostic(&result, operation.location,
                                       "operation in block '" + block.name + "' targets an unknown block");
                        continue;
                    }
                    predecessors[target->second].push_back(i);
                    const BasicBlock& target_block = function.blocks[target->second];
                    if (successor.arguments.size() != target_block.arguments.size()) {
                        add_diagnostic(&result, operation.location,
                                       "edge to block '" + target_block.name +
                                           "' supplies the wrong number of block arguments");
                    }
                }
            }
        }

        std::vector<std::unordered_set<std::size_t>> dominators(block_count);
        for (std::size_t i = 0; i < block_count; ++i) {
            if (i == 0) {
                dominators[i].insert(0);
            } else {
                for (std::size_t j = 0; j < block_count; ++j) {
                    dominators[i].insert(j);
                }
            }
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (std::size_t i = 1; i < block_count; ++i) {
                std::unordered_set<std::size_t> next;
                if (!predecessors[i].empty()) {
                    next = dominators[predecessors[i].front()];
                    for (std::size_t predecessor : predecessors[i]) {
                        std::erase_if(next, [&](std::size_t candidate) {
                            return !dominators[predecessor].contains(candidate);
                        });
                    }
                }
                next.insert(i);
                if (next != dominators[i]) {
                    dominators[i] = std::move(next);
                    changed = true;
                }
            }
        }

        std::unordered_map<ValueId, ValueDefinition> definitions;
        for (const FunctionArgument& argument : function.arguments) {
            if (argument.value == kInvalidValue ||
                !definitions.emplace(argument.value,
                                     ValueDefinition{.type = argument.type, .function_argument = true})
                     .second) {
                add_diagnostic(&result, {}, "function '" + function.name + "' has duplicate/invalid argument values");
            }
        }
        for (const BasicBlock& block : function.blocks) {
            for (const BlockArgument& argument : block.arguments) {
                if (argument.value == kInvalidValue ||
                    !definitions.emplace(argument.value,
                                         ValueDefinition{
                                             .type = argument.type,
                                             .block = block.id,
                                             .block_argument = true,
                                         })
                         .second) {
                    add_diagnostic(&result, {}, "block '" + block.name + "' has duplicate/invalid argument values");
                }
            }
            for (std::size_t op_index = 0; op_index < block.operations.size(); ++op_index) {
                const Operation& operation = block.operations[op_index];
                if (operation.results.size() != operation.result_types.size()) {
                    add_diagnostic(&result, operation.location, "operation result/type arity mismatch");
                }
                for (std::size_t result_index = 0; result_index < operation.results.size(); ++result_index) {
                    const ValueId value = operation.results[result_index];
                    const Type type = result_index < operation.result_types.size()
                                          ? operation.result_types[result_index]
                                          : Type::void_type();
                    if (value == kInvalidValue ||
                        !definitions.emplace(value,
                                             ValueDefinition{
                                                 .type = type,
                                                 .block = block.id,
                                                 .operation_index = op_index,
                                             })
                             .second) {
                        add_diagnostic(&result, operation.location, "operation defines a duplicate/invalid value");
                    }
                }
            }
        }

        for (std::size_t block_index = 0; block_index < block_count; ++block_index) {
            const BasicBlock& block = function.blocks[block_index];
            for (std::size_t op_index = 0; op_index < block.operations.size(); ++op_index) {
                const Operation& operation = block.operations[op_index];
                for (const Operand& operand : operation.operands) {
                    if (operand.kind != OperandKind::kValue) {
                        continue;
                    }
                    const auto definition = definitions.find(operand.value);
                    if (definition == definitions.end()) {
                        add_diagnostic(&result, operation.location,
                                       "operation uses undefined value %" + std::to_string(operand.value));
                        continue;
                    }
                    if (!(operand.type == definition->second.type)) {
                        add_diagnostic(&result, operation.location,
                                       "operand type does not match value %" + std::to_string(operand.value));
                    }
                    if (definition->second.function_argument) {
                        continue;
                    }
                    const auto definition_block = block_indices.find(definition->second.block);
                    if (definition_block == block_indices.end()) {
                        continue;
                    }
                    if (definition_block->second == block_index) {
                        if (!definition->second.block_argument &&
                            definition->second.operation_index >= op_index) {
                            add_diagnostic(&result, operation.location,
                                           "value %" + std::to_string(operand.value) +
                                               " is used before its definition");
                        }
                    } else if (!dominators[block_index].contains(definition_block->second)) {
                        add_diagnostic(&result, operation.location,
                                       "value %" + std::to_string(operand.value) +
                                           " does not dominate its use");
                    }
                }

                if (operation.opcode == OpCode::kAddressSpaceCast &&
                    (!operation.operands.empty() && !operation.result_types.empty())) {
                    const Type& source = operation.operands.front().type;
                    const Type& target = operation.result_types.front();
                    if (!source.is_pointer() || !target.is_pointer() ||
                        source.address_space == AddressSpace::kNone ||
                        target.address_space == AddressSpace::kNone) {
                        add_diagnostic(&result, operation.location,
                                       "address-space casts require explicit pointer address spaces");
                    }
                }
                if (operation.opcode == OpCode::kBarrier &&
                    operation.attributes.contains("predicate")) {
                    add_diagnostic(&result, operation.location,
                                   "predicated barriers are not legal in CuMetal GPU IR");
                }
                if (operation.opcode == OpCode::kAtomic) {
                    if (operation.memory_ordering == MemoryOrdering::kNone ||
                        operation.memory_scope == MemoryScope::kNone) {
                        add_diagnostic(&result, operation.location,
                                       "atomic operation requires typed memory ordering and scope");
                    }
                    if (operation.memory_scope == MemoryScope::kSystem) {
                        add_diagnostic(&result, operation.location,
                                       "system-scope atomics are unsupported by the Metal backend");
                    }
                    if (operation.memory_ordering == MemoryOrdering::kSequentiallyConsistent) {
                        add_diagnostic(&result, operation.location,
                                       "sequentially-consistent atomics require an explicit backend policy");
                    }
                }
                if (module.stage == IrStage::kGpuSemantic && is_metal_opcode(operation.opcode)) {
                    add_diagnostic(&result, operation.location,
                                   "Metal operation is illegal before Metal legalization");
                }
                if (module.stage == IrStage::kMetalLegalized &&
                    is_gpu_semantic_opcode(operation.opcode)) {
                    add_diagnostic(&result, operation.location,
                                   "GPU semantic operation survived Metal legalization");
                }
                if (operation.opcode == OpCode::kCall) {
                    const auto callee = operation.attributes.find("callee");
                    const auto indirect = operation.attributes.find("indirect");
                    if (callee == operation.attributes.end() ||
                        (indirect != operation.attributes.end() && indirect->second == "true")) {
                        add_diagnostic(&result, operation.location,
                                       "device calls must be direct and resolve within the module");
                    } else if ((!operation.attributes.contains("builtin") ||
                                operation.attributes.at("builtin") != "true") &&
                               !function_names.contains(callee->second)) {
                        add_diagnostic(&result, operation.location,
                                       "device call target '" + callee->second +
                                           "' is not defined in the module");
                    } else if (!operation.attributes.contains("builtin") ||
                               operation.attributes.at("builtin") != "true") {
                        call_graph[function.name].insert(callee->second);
                    }
                }
            }
        }

        if (function.kernel_abi.has_value()) {
            for (const ArgumentDescriptor& argument : function.kernel_abi->arguments) {
                if (argument.size == 0 || argument.alignment == 0) {
                    add_diagnostic(&result, {},
                                   "kernel argument '" + argument.name + "' has invalid size/alignment");
                }
                if (argument.type.is_pointer() &&
                    argument.address_space == AddressSpace::kNone) {
                    add_diagnostic(&result, {},
                                   "pointer kernel argument '" + argument.name +
                                       "' is missing an address space");
                }
                if (argument.size < natural_size(argument.type) && !argument.type.is_pointer()) {
                    add_diagnostic(&result, {},
                                   "kernel argument '" + argument.name + "' is smaller than its type");
                }
            }
        }
    }

    enum class VisitState : std::uint8_t { kUnvisited, kVisiting, kVisited };
    std::unordered_map<std::string, VisitState> visit_state;
    std::function<void(const std::string&)> visit = [&](const std::string& function) {
        visit_state[function] = VisitState::kVisiting;
        for (const std::string& callee : call_graph[function]) {
            if (visit_state[callee] == VisitState::kVisiting) {
                add_diagnostic(&result, {},
                               "recursive device-call cycle involving '" + callee +
                                   "' is unsupported");
            } else if (visit_state[callee] == VisitState::kUnvisited) {
                visit(callee);
            }
        }
        visit_state[function] = VisitState::kVisited;
    };
    for (const std::string& function : function_names) {
        if (visit_state[function] == VisitState::kUnvisited) visit(function);
    }

    result.ok = result.diagnostics.empty();
    return result;
}

std::string print(const Module& module) {
    std::ostringstream out;
    out << "module";
    if (!module.source_name.empty()) {
        out << " source=\"" << module.source_name << "\"";
    }
    out << " {\n";
    for (const Function& function : module.functions) {
        out << "  " << (function.is_kernel ? "kernel" : "func") << " @" << function.name << "(";
        for (std::size_t i = 0; i < function.arguments.size(); ++i) {
            if (i != 0) {
                out << ", ";
            }
            out << "%" << function.arguments[i].value << " " << function.arguments[i].name
                << ": " << function.arguments[i].type.str();
        }
        out << ") -> " << function.return_type.str() << " {\n";
        for (const BasicBlock& block : function.blocks) {
            out << "  ^" << block.name;
            if (!block.arguments.empty()) {
                out << "(";
                for (std::size_t i = 0; i < block.arguments.size(); ++i) {
                    if (i != 0) {
                        out << ", ";
                    }
                    out << "%" << block.arguments[i].value << " " << block.arguments[i].name
                        << ": " << block.arguments[i].type.str();
                }
                out << ")";
            }
            out << ":\n";
            for (const Operation& operation : block.operations) {
                out << "    ";
                for (std::size_t i = 0; i < operation.results.size(); ++i) {
                    if (i != 0) {
                        out << ", ";
                    }
                    out << "%" << operation.results[i];
                }
                if (!operation.results.empty()) {
                    out << " = ";
                }
                out << opcode_name(operation.opcode);
                if (!operation.operands.empty()) {
                    out << " ";
                    for (std::size_t i = 0; i < operation.operands.size(); ++i) {
                        if (i != 0) {
                            out << ", ";
                        }
                        const Operand& operand = operation.operands[i];
                        if (operand.kind == OperandKind::kValue) {
                            out << "%" << operand.value;
                        } else if (operand.kind == OperandKind::kSymbol) {
                            out << "@" << operand.text;
                        } else {
                            out << operand.text;
                        }
                        if (operand.kind != OperandKind::kSymbol) {
                            out << ": " << operand.type.str();
                        }
                    }
                }
                for (const Successor& successor : operation.successors) {
                    const BasicBlock* target = function.find_block(successor.block);
                    out << " ^" << (target == nullptr ? "?" : target->name) << "(";
                    for (std::size_t i = 0; i < successor.arguments.size(); ++i) {
                        if (i != 0) {
                            out << ", ";
                        }
                        out << "%" << successor.arguments[i];
                    }
                    out << ")";
                }
                if (!operation.attributes.empty()) {
                    std::vector<std::pair<std::string, std::string>> attributes(
                        operation.attributes.begin(), operation.attributes.end());
                    std::sort(attributes.begin(), attributes.end());
                    out << " {";
                    for (std::size_t i = 0; i < attributes.size(); ++i) {
                        if (i != 0) {
                            out << ", ";
                        }
                        out << attributes[i].first << "=\"" << attributes[i].second << "\"";
                    }
                    out << "}";
                }
                if (operation.memory_scope != MemoryScope::kNone) {
                    out << " scope(" << memory_scope_name(operation.memory_scope) << ")";
                }
                if (operation.memory_ordering != MemoryOrdering::kNone) {
                    out << " ordering(" << memory_ordering_name(operation.memory_ordering) << ")";
                }
                if (!operation.result_types.empty()) {
                    out << " -> ";
                    for (std::size_t i = 0; i < operation.result_types.size(); ++i) {
                        if (i != 0) {
                            out << ", ";
                        }
                        out << operation.result_types[i].str();
                    }
                }
                if (operation.location.line != 0) {
                    out << " loc(" << operation.location.str() << ")";
                }
                out << "\n";
            }
        }
        out << "  }\n";
    }
    out << "}\n";
    return out.str();
}

}  // namespace cumetal::ir
