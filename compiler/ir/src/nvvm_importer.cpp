#include "cumetal/ir/nvvm_importer.h"

#ifndef CUMETAL_HAVE_LLVM
#define CUMETAL_HAVE_LLVM 0
#endif

#if CUMETAL_HAVE_LLVM

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <map>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace cumetal::ir {
namespace {

AddressSpace import_address_space(unsigned address_space, bool kernel_pointer) {
    switch (address_space) {
        case 1: return AddressSpace::kDevice;
        case 3: return AddressSpace::kThreadgroup;
        case 4: return AddressSpace::kConstant;
        case 5: return AddressSpace::kPrivate;
        default: return kernel_pointer ? AddressSpace::kDevice : AddressSpace::kPrivate;
    }
}

Type import_type(llvm::Type* type, bool kernel_pointer = false) {
    if (type->isVoidTy()) return Type::void_type();
    if (type->isIntegerTy()) return Type::integer(type->getIntegerBitWidth());
    if (type->isHalfTy()) return Type::floating(16);
    if (type->isFloatTy()) return Type::floating(32);
    if (type->isDoubleTy()) return Type::floating(64);
    if (auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(type)) {
        return Type::vector(import_type(vector->getElementType()),
                            vector->getNumElements());
    }
    if (auto* pointer = llvm::dyn_cast<llvm::PointerType>(type)) {
        return Type::pointer(Type::integer(8),
                             import_address_space(pointer->getAddressSpace(), kernel_pointer));
    }
    if (auto* structure = llvm::dyn_cast<llvm::StructType>(type)) {
        std::vector<Type> elements;
        for (llvm::Type* element : structure->elements()) {
            elements.push_back(import_type(element));
        }
        return Type::aggregate(std::move(elements));
    }
    if (auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        std::vector<Type> elements;
        elements.reserve(array->getNumElements());
        for (std::uint64_t i = 0; i < array->getNumElements(); ++i) {
            elements.push_back(import_type(array->getElementType()));
        }
        return Type::aggregate(std::move(elements));
    }
    return Type::void_type();
}

std::uint32_t type_size(const Type& type) {
    if (type.is_pointer()) return 8;
    if (type.kind == TypeKind::kInteger || type.kind == TypeKind::kFloat) {
        return std::max<std::uint32_t>(1, type.bit_width / 8);
    }
    if (type.kind == TypeKind::kVector && !type.elements.empty()) {
        return type_size(type.elements.front()) * type.lanes;
    }
    std::uint32_t size = 0;
    for (const Type& element : type.elements) size += type_size(element);
    return size;
}

SourceLocation import_location(const llvm::Instruction& instruction,
                               std::string_view fallback_file) {
    SourceLocation location;
    location.file = std::string(fallback_file);
    if (const llvm::DebugLoc& debug = instruction.getDebugLoc()) {
        location.line = debug.getLine();
        location.column = debug.getCol();
        if (const llvm::DIScope* scope = llvm::dyn_cast_or_null<llvm::DIScope>(debug.getScope())) {
            if (!scope->getFilename().empty()) {
                location.file = scope->getFilename().str();
            }
        }
    }
    return location;
}

std::string constant_spelling(const llvm::Constant& constant) {
    std::string spelling;
    llvm::raw_string_ostream stream(spelling);
    constant.printAsOperand(stream, false);
    stream.flush();
    return spelling;
}

std::string value_name(ValueId value) {
    return "v" + std::to_string(value);
}

std::string comparison_predicate(llvm::CmpInst::Predicate predicate) {
    switch (predicate) {
        case llvm::CmpInst::ICMP_EQ:
        case llvm::CmpInst::FCMP_OEQ:
        case llvm::CmpInst::FCMP_UEQ: return "eq";
        case llvm::CmpInst::ICMP_NE:
        case llvm::CmpInst::FCMP_ONE:
        case llvm::CmpInst::FCMP_UNE: return "ne";
        case llvm::CmpInst::ICMP_SLT: return "slt";
        case llvm::CmpInst::ICMP_ULT:
        case llvm::CmpInst::FCMP_OLT:
        case llvm::CmpInst::FCMP_ULT: return "lt";
        case llvm::CmpInst::ICMP_SLE: return "sle";
        case llvm::CmpInst::ICMP_ULE:
        case llvm::CmpInst::FCMP_OLE:
        case llvm::CmpInst::FCMP_ULE: return "le";
        case llvm::CmpInst::ICMP_SGT: return "sgt";
        case llvm::CmpInst::ICMP_UGT:
        case llvm::CmpInst::FCMP_OGT:
        case llvm::CmpInst::FCMP_UGT: return "gt";
        case llvm::CmpInst::ICMP_SGE: return "sge";
        case llvm::CmpInst::ICMP_UGE:
        case llvm::CmpInst::FCMP_OGE:
        case llvm::CmpInst::FCMP_UGE: return "ge";
        default: return "unsupported";
    }
}

OpCode binary_opcode(unsigned opcode) {
    switch (opcode) {
        case llvm::Instruction::Add:
        case llvm::Instruction::FAdd: return OpCode::kAdd;
        case llvm::Instruction::Sub:
        case llvm::Instruction::FSub: return OpCode::kSub;
        case llvm::Instruction::Mul:
        case llvm::Instruction::FMul: return OpCode::kMul;
        case llvm::Instruction::UDiv:
        case llvm::Instruction::SDiv:
        case llvm::Instruction::FDiv: return OpCode::kDiv;
        case llvm::Instruction::URem:
        case llvm::Instruction::SRem:
        case llvm::Instruction::FRem: return OpCode::kRemainder;
        case llvm::Instruction::And: return OpCode::kBitAnd;
        case llvm::Instruction::Or: return OpCode::kBitOr;
        case llvm::Instruction::Xor: return OpCode::kBitXor;
        case llvm::Instruction::Shl: return OpCode::kShiftLeft;
        case llvm::Instruction::LShr:
        case llvm::Instruction::AShr: return OpCode::kShiftRight;
        default: return OpCode::kInvalid;
    }
}

MemoryOrdering import_ordering(llvm::AtomicOrdering ordering) {
    switch (ordering) {
        case llvm::AtomicOrdering::Unordered:
        case llvm::AtomicOrdering::Monotonic: return MemoryOrdering::kRelaxed;
        case llvm::AtomicOrdering::Acquire: return MemoryOrdering::kAcquire;
        case llvm::AtomicOrdering::Release: return MemoryOrdering::kRelease;
        case llvm::AtomicOrdering::AcquireRelease: return MemoryOrdering::kAcquireRelease;
        case llvm::AtomicOrdering::SequentiallyConsistent:
            return MemoryOrdering::kSequentiallyConsistent;
        case llvm::AtomicOrdering::NotAtomic: return MemoryOrdering::kNone;
    }
    return MemoryOrdering::kNone;
}

struct FunctionState {
    Function output;
    std::unordered_map<const llvm::Value*, ValueId> values;
    std::unordered_map<const llvm::BasicBlock*, BlockId> blocks;
    std::unordered_map<ValueId, Type> value_types;
};

struct Importer {
    Builder builder;
    NvvmImportResult result;
    llvm::Module* input = nullptr;
    std::string fallback_source;

    bool fail(const llvm::Instruction* instruction, std::string message) {
        if (instruction != nullptr) {
            const SourceLocation location = import_location(*instruction, fallback_source);
            if (!location.str().empty()) message = location.str() + ": " + message;
        }
        result.error = std::move(message);
        return false;
    }

    Operand import_operand(const llvm::Value& value, const FunctionState& state) {
        if (const auto* constant = llvm::dyn_cast<llvm::Constant>(&value)) {
            return Operand::immediate(constant_spelling(*constant), import_type(value.getType()));
        }
        const auto found = state.values.find(&value);
        if (found == state.values.end()) {
            return Operand::symbol("<undefined>", import_type(value.getType()));
        }
        return Operand::value_ref(found->second, state.value_types.at(found->second));
    }

    bool allocate_function(const llvm::Function& function, FunctionState* state) {
        state->output.name = function.getName().str();
        state->output.is_kernel =
            function.getCallingConv() == llvm::CallingConv::PTX_Kernel;
        state->output.return_type = import_type(function.getReturnType());
        if (state->output.is_kernel) state->output.kernel_abi = KernelAbi{};

        std::uint32_t argument_index = 0;
        for (const llvm::Argument& argument : function.args()) {
            const Type type = import_type(argument.getType(), state->output.is_kernel);
            const ValueId value = builder.next_value();
            const std::string name =
                argument.hasName() ? argument.getName().str() : ("arg" + std::to_string(argument_index));
            state->values[&argument] = value;
            state->value_types[value] = type;
            state->output.arguments.push_back({.value = value, .name = name, .type = type});
            if (type.is_pointer()) {
                state->output.pointer_provenance[value] = {
                    .base_kind = PointerBaseKind::kKernelArgument,
                    .base_name = name,
                    .known_byte_offset = 0,
                    .alignment = 1,
                };
            }
            if (state->output.kernel_abi.has_value()) {
                const std::uint32_t size = type_size(type);
                state->output.kernel_abi->arguments.push_back({
                    .name = name,
                    .kind = type.is_pointer() ? ArgumentKind::kPointer : ArgumentKind::kScalar,
                    .type = type,
                    .size = size,
                    .alignment = std::min<std::uint32_t>(size, 8),
                    .address_space = type.is_pointer() ? type.address_space : AddressSpace::kConstant,
                    .binding_indices = {argument_index},
                });
                state->output.kernel_abi->bindings.push_back({
                    .kind = type.is_pointer() ? BindingKind::kBuffer : BindingKind::kBytes,
                    .binding_index = argument_index,
                    .logical_argument_index = argument_index,
                    .type = type,
                    .size = size,
                    .alignment = std::min<std::uint32_t>(size, 8),
                });
            }
            ++argument_index;
        }

        std::uint32_t unnamed_block = 0;
        for (const llvm::BasicBlock& block : function) {
            BasicBlock output_block;
            output_block.id = builder.next_block();
            output_block.name =
                block.hasName() ? block.getName().str() : ("bb" + std::to_string(unnamed_block++));
            state->blocks[&block] = output_block.id;
            state->output.blocks.push_back(std::move(output_block));
        }

        std::size_t block_index = 0;
        for (const llvm::BasicBlock& block : function) {
            BasicBlock& output_block = state->output.blocks[block_index++];
            for (const llvm::Instruction& instruction : block) {
                if (instruction.getType()->isVoidTy()) continue;
                const ValueId value = builder.next_value();
                Type type = import_type(instruction.getType());
                if (llvm::isa<llvm::GetElementPtrInst>(instruction)) {
                    type = Type::pointer(Type::integer(8), AddressSpace::kDevice);
                }
                state->values[&instruction] = value;
                state->value_types[value] = type;
                if (const auto* phi = llvm::dyn_cast<llvm::PHINode>(&instruction)) {
                    output_block.arguments.push_back({
                        .value = value,
                        .type = type,
                        .name = phi->hasName() ? phi->getName().str() : value_name(value),
                    });
                    if (type.is_pointer()) {
                        state->output.pointer_provenance[value] = {
                            .base_kind = PointerBaseKind::kUnknown,
                            .base_name = output_block.name,
                        };
                    }
                }
            }
        }
        return true;
    }

    Successor import_successor(const llvm::BasicBlock& source,
                               const llvm::BasicBlock& target,
                               const FunctionState& state) {
        Successor successor;
        successor.block = state.blocks.at(&target);
        for (const llvm::Instruction& instruction : target) {
            const auto* phi = llvm::dyn_cast<llvm::PHINode>(&instruction);
            if (phi == nullptr) break;
            const llvm::Value* incoming = phi->getIncomingValueForBlock(&source);
            successor.arguments.push_back(state.values.at(incoming));
        }
        return successor;
    }

    bool import_call(const llvm::CallBase& call, FunctionState* state, Operation* operation) {
        const llvm::Function* callee = call.getCalledFunction();
        if (callee == nullptr) return fail(&call, "indirect device calls are unsupported");
        const std::string name = callee->getName().str();
        operation->attributes["llvm_intrinsic"] = name;

        auto dimension = [&](std::string_view prefix, OpCode opcode) {
            if (!name.starts_with(prefix)) return false;
            operation->opcode = opcode;
            operation->attributes["dimension"] =
                name.ends_with(".y") ? "y" : (name.ends_with(".z") ? "z" : "x");
            return true;
        };
        if (dimension("llvm.nvvm.read.ptx.sreg.tid.", OpCode::kThreadId) ||
            dimension("llvm.nvvm.read.ptx.sreg.ctaid.", OpCode::kThreadgroupId) ||
            dimension("llvm.nvvm.read.ptx.sreg.ntid.", OpCode::kThreadgroupSize) ||
            dimension("llvm.nvvm.read.ptx.sreg.nctaid.", OpCode::kGridSize)) {
            return true;
        }
        if (name == "llvm.nvvm.read.ptx.sreg.laneid") {
            operation->opcode = OpCode::kLaneId;
            return true;
        }
        if (name == "llvm.nvvm.barrier0") {
            operation->opcode = OpCode::kBarrier;
            operation->memory_scope = MemoryScope::kThreadgroup;
            return true;
        }
        if (name.find("llvm.nvvm.shfl") == 0) operation->opcode = OpCode::kShuffle;
        else if (name.find("llvm.nvvm.vote.ballot") == 0) operation->opcode = OpCode::kBallot;
        else if (name.find("llvm.nvvm.vote") == 0) operation->opcode = OpCode::kVote;
        else if (name.find("llvm.fma.") == 0) operation->opcode = OpCode::kFma;
        else if (name.find("llvm.sqrt.") == 0 || name.find("llvm.sin.") == 0 ||
                 name.find("llvm.cos.") == 0 || name.find("llvm.exp.") == 0 ||
                 name.find("llvm.log.") == 0) {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] =
                name.substr(name.find('.') + 1, name.find('.', 5) - name.find('.') - 1);
            operation->attributes["builtin"] = "true";
        } else if (!callee->isDeclaration()) {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] = name;
        } else {
            return fail(&call, "unsupported LLVM/NVVM intrinsic '" + name + "'");
        }
        for (const llvm::Use& argument : call.args()) {
            operation->operands.push_back(import_operand(*argument.get(), *state));
        }
        return true;
    }

    bool import_instruction(const llvm::Instruction& instruction,
                            const llvm::BasicBlock& source_block,
                            FunctionState* state, BasicBlock* output_block) {
        if (llvm::isa<llvm::PHINode>(instruction)) return true;
        Operation operation;
        operation.location = import_location(instruction, fallback_source);
        if (!instruction.getType()->isVoidTy()) {
            operation.results.push_back(state->values.at(&instruction));
            operation.result_types.push_back(state->value_types.at(operation.results.front()));
        }

        if (const auto* binary = llvm::dyn_cast<llvm::BinaryOperator>(&instruction)) {
            operation.opcode = binary_opcode(binary->getOpcode());
            if (operation.opcode == OpCode::kInvalid) {
                return fail(&instruction, "unsupported LLVM binary operation");
            }
            operation.operands.push_back(import_operand(*binary->getOperand(0), *state));
            operation.operands.push_back(import_operand(*binary->getOperand(1), *state));
            if (binary->getOpcode() == llvm::Instruction::SDiv ||
                binary->getOpcode() == llvm::Instruction::SRem ||
                binary->getOpcode() == llvm::Instruction::AShr) {
                operation.attributes["signed"] = "true";
            }
        } else if (const auto* compare = llvm::dyn_cast<llvm::CmpInst>(&instruction)) {
            const std::string predicate = comparison_predicate(compare->getPredicate());
            if (predicate == "unsupported") {
                return fail(&instruction, "unsupported LLVM comparison predicate");
            }
            operation.opcode = OpCode::kCompare;
            operation.attributes["predicate"] = predicate;
            if (!predicate.empty() && predicate.front() == 's') {
                operation.attributes["signed"] = "true";
            }
            operation.operands.push_back(import_operand(*compare->getOperand(0), *state));
            operation.operands.push_back(import_operand(*compare->getOperand(1), *state));
        } else if (const auto* select = llvm::dyn_cast<llvm::SelectInst>(&instruction)) {
            operation.opcode = OpCode::kSelect;
            operation.operands.push_back(import_operand(*select->getCondition(), *state));
            operation.operands.push_back(import_operand(*select->getTrueValue(), *state));
            operation.operands.push_back(import_operand(*select->getFalseValue(), *state));
        } else if (const auto* cast = llvm::dyn_cast<llvm::CastInst>(&instruction)) {
            if (llvm::isa<llvm::PtrToIntInst>(cast) || llvm::isa<llvm::IntToPtrInst>(cast)) {
                return fail(&instruction, "integer/pointer round trips are unsupported");
            }
            operation.opcode = llvm::isa<llvm::AddrSpaceCastInst>(cast)
                                   ? OpCode::kAddressSpaceCast
                                   : OpCode::kConvert;
            operation.operands.push_back(import_operand(*cast->getOperand(0), *state));
        } else if (const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
            if (gep->getNumIndices() != 1) {
                return fail(&instruction, "only single-index LLVM GEP is supported initially");
            }
            llvm::Value* index = gep->idx_begin()->get();
            const std::uint64_t stride =
                input->getDataLayout().getTypeAllocSize(gep->getSourceElementType()).getFixedValue();
            Operand byte_offset;
            if (const auto* constant = llvm::dyn_cast<llvm::ConstantInt>(index)) {
                byte_offset = Operand::immediate(
                    std::to_string(constant->getSExtValue() * static_cast<std::int64_t>(stride)),
                    Type::integer(64));
            } else {
                const Operand imported_index = import_operand(*index, *state);
                const ValueId scaled = builder.next_value();
                state->value_types[scaled] = Type::integer(64);
                Operation multiply;
                multiply.opcode = OpCode::kMul;
                multiply.results = {scaled};
                multiply.result_types = {Type::integer(64)};
                multiply.operands = {
                    imported_index,
                    Operand::immediate(std::to_string(stride), Type::integer(64)),
                };
                multiply.location = operation.location;
                output_block->operations.push_back(std::move(multiply));
                byte_offset = Operand::value_ref(scaled, Type::integer(64));
            }
            operation.opcode = OpCode::kPointerOffset;
            operation.operands.push_back(import_operand(*gep->getPointerOperand(), *state));
            operation.operands.push_back(std::move(byte_offset));
            operation.attributes["element_stride"] = std::to_string(stride);
            const auto source = state->output.pointer_provenance.find(
                operation.operands.front().value);
            if (source != state->output.pointer_provenance.end()) {
                state->output.pointer_provenance[operation.results.front()] = source->second;
            }
        } else if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
            operation.opcode = OpCode::kLoad;
            operation.operands.push_back(import_operand(*load->getPointerOperand(), *state));
            operation.attributes["alignment"] = std::to_string(load->getAlign().value());
        } else if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
            operation.opcode = OpCode::kStore;
            operation.operands.push_back(import_operand(*store->getPointerOperand(), *state));
            operation.operands.push_back(import_operand(*store->getValueOperand(), *state));
            operation.attributes["alignment"] = std::to_string(store->getAlign().value());
        } else if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
            if (!import_call(*call, state, &operation)) return false;
        } else if (const auto* atomic = llvm::dyn_cast<llvm::AtomicRMWInst>(&instruction)) {
            operation.opcode = OpCode::kAtomic;
            operation.operands.push_back(import_operand(*atomic->getPointerOperand(), *state));
            operation.operands.push_back(import_operand(*atomic->getValOperand(), *state));
            operation.attributes["atomic_op"] =
                llvm::AtomicRMWInst::getOperationName(atomic->getOperation()).str();
            operation.memory_scope = MemoryScope::kDevice;
            operation.memory_ordering = import_ordering(atomic->getOrdering());
        } else if (const auto* fence = llvm::dyn_cast<llvm::FenceInst>(&instruction)) {
            operation.opcode = OpCode::kFence;
            operation.memory_scope = MemoryScope::kDevice;
            operation.memory_ordering = import_ordering(fence->getOrdering());
        } else if (const auto* branch = llvm::dyn_cast<llvm::BranchInst>(&instruction)) {
            if (branch->isConditional()) {
                operation.opcode = OpCode::kCondBranch;
                operation.operands.push_back(import_operand(*branch->getCondition(), *state));
                operation.successors.push_back(
                    import_successor(source_block, *branch->getSuccessor(0), *state));
                operation.successors.push_back(
                    import_successor(source_block, *branch->getSuccessor(1), *state));
            } else {
                operation.opcode = OpCode::kBranch;
                operation.successors.push_back(
                    import_successor(source_block, *branch->getSuccessor(0), *state));
            }
        } else if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
            operation.opcode = OpCode::kReturn;
            if (ret->getReturnValue() != nullptr) {
                operation.operands.push_back(import_operand(*ret->getReturnValue(), *state));
            }
        } else if (llvm::isa<llvm::UnreachableInst>(instruction)) {
            operation.opcode = OpCode::kTrap;
        } else if (llvm::isa<llvm::AllocaInst>(instruction)) {
            operation.opcode = OpCode::kAlloca;
            operation.attributes["alignment"] =
                std::to_string(llvm::cast<llvm::AllocaInst>(instruction).getAlign().value());
            state->output.pointer_provenance[operation.results.front()] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = value_name(operation.results.front()),
                .known_byte_offset = 0,
                .alignment = static_cast<std::uint32_t>(
                    llvm::cast<llvm::AllocaInst>(instruction).getAlign().value()),
            };
        } else {
            return fail(&instruction, "unsupported LLVM instruction '" +
                                          std::string(instruction.getOpcodeName()) + "'");
        }
        output_block->operations.push_back(std::move(operation));
        return true;
    }

    bool import_function(const llvm::Function& function) {
        FunctionState state;
        if (!allocate_function(function, &state)) return false;
        std::size_t block_index = 0;
        for (const llvm::BasicBlock& block : function) {
            BasicBlock& output_block = state.output.blocks[block_index++];
            for (const llvm::Instruction& instruction : block) {
                if (!import_instruction(instruction, block, &state, &output_block)) return false;
            }
        }
        result.module.functions.push_back(std::move(state.output));
        return true;
    }

    NvvmImportResult run(llvm::Module* module, const NvvmImportOptions& options) {
        input = module;
        fallback_source = options.source_name.empty()
                              ? module->getSourceFileName()
                              : options.source_name;
        result.module.source_name = fallback_source;
        result.module.stage = IrStage::kGpuSemantic;
        result.module.attributes["frontend"] = "nvvm";
        result.module.attributes["ir_schema"] = "1";
        result.module.attributes["target_triple"] = module->getTargetTriple().str();

        for (const llvm::Function& function : *module) {
            if (!function.isDeclaration() && !import_function(function)) return std::move(result);
        }
        if (result.module.functions.empty()) {
            result.error = "LLVM module contains no device function definitions";
            return std::move(result);
        }
        const VerifyResult verification = verify(result.module);
        if (!verification.ok) {
            std::ostringstream error;
            error << "imported NVVM module failed CuMetal IR verification";
            for (const Diagnostic& diagnostic : verification.diagnostics) {
                error << "\n" << diagnostic.location.str() << ": " << diagnostic.message;
            }
            result.error = error.str();
            return std::move(result);
        }
        result.ok = true;
        return std::move(result);
    }
};

NvvmImportResult parse_module(std::unique_ptr<llvm::MemoryBuffer> buffer,
                              const NvvmImportOptions& options) {
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    std::unique_ptr<llvm::Module> module =
        llvm::parseIR(buffer->getMemBufferRef(), diagnostic, context);
    if (module == nullptr) {
        std::string message;
        llvm::raw_string_ostream stream(message);
        diagnostic.print("cumetalc", stream);
        stream.flush();
        NvvmImportResult result;
        result.error = "failed to parse LLVM/NVVM IR: " + message;
        return result;
    }
    return Importer{}.run(module.get(), options);
}

}  // namespace

bool llvm_frontend_available() {
    return true;
}

NvvmImportResult import_nvvm_llvm_ir(std::string_view llvm_ir,
                                     const NvvmImportOptions& options) {
    return parse_module(
        llvm::MemoryBuffer::getMemBufferCopy(
            llvm::StringRef(llvm_ir.data(), llvm_ir.size()), options.source_name),
        options);
}

NvvmImportResult import_nvvm_bitcode_file(const std::filesystem::path& input,
                                          const NvvmImportOptions& options) {
    auto buffer = llvm::MemoryBuffer::getFile(input.string());
    if (!buffer) {
        NvvmImportResult result;
        result.error = "failed to read LLVM/NVVM input: " + input.string();
        return result;
    }
    return parse_module(std::move(*buffer), options);
}

}  // namespace cumetal::ir

#else

namespace cumetal::ir {

bool llvm_frontend_available() {
    return false;
}

NvvmImportResult import_nvvm_llvm_ir(std::string_view,
                                     const NvvmImportOptions&) {
    NvvmImportResult result;
    result.error =
        "CuMetal was built without LLVM IRReader support; install LLVM 18+ and reconfigure";
    return result;
}

NvvmImportResult import_nvvm_bitcode_file(const std::filesystem::path&,
                                          const NvvmImportOptions&) {
    NvvmImportResult result;
    result.error =
        "CuMetal was built without LLVM IRReader support; install LLVM 18+ and reconfigure";
    return result;
}

}  // namespace cumetal::ir

#endif
