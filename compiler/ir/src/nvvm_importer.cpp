#include "cumetal/ir/nvvm_importer.h"

#ifndef CUMETAL_HAVE_LLVM
#define CUMETAL_HAVE_LLVM 0
#endif

#if CUMETAL_HAVE_LLVM

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Verifier.h>
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
        return Type::aggregate(std::move(elements),
                               structure->hasName() ? structure->getName().str() : std::string{});
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

std::optional<std::string> homogeneous_aggregate_constructor(const Type& type) {
    if (type.kind != TypeKind::kAggregate || type.elements.size() < 2 ||
        type.elements.size() > 4) {
        return std::nullopt;
    }
    const std::string_view name = type.name;
    const bool cuda_vector_name =
        name.ends_with("float2") || name.ends_with("float4") ||
        name.ends_with("double2") || name.ends_with("double4") ||
        name.ends_with("uint2") || name.ends_with("uint4") ||
        name.ends_with("int2") || name.ends_with("int4") ||
        name.ends_with("uchar2") || name.ends_with("uchar4") ||
        name.ends_with("ushort2") || name.ends_with("ushort4") ||
        name.ends_with("ulong2") || name.ends_with("ulong4") ||
        name.ends_with("ulonglong2") || name.ends_with("longlong2");
    if (!cuda_vector_name) return std::nullopt;
    const Type& element = type.elements.front();
    if (!std::all_of(type.elements.begin(), type.elements.end(),
                     [&](const Type& candidate) { return candidate == element; })) {
        return std::nullopt;
    }
    std::string scalar;
    if (element.kind == TypeKind::kFloat) {
        scalar = element.bit_width == 16 ? "half" :
                 (element.bit_width == 32 ? "float" : "double");
    } else if (element.kind == TypeKind::kInteger) {
        scalar = element.bit_width == 8 ? "uchar" :
                 (element.bit_width == 16 ? "ushort" :
                  (element.bit_width == 32 ? "uint" : "ulong"));
    } else {
        return std::nullopt;
    }
    return scalar + std::to_string(type.elements.size());
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

bool write_constant_bytes(const llvm::Constant& constant, std::uint64_t offset,
                          std::vector<std::uint8_t>* bytes,
                          const llvm::DataLayout& layout) {
    const std::uint64_t size = layout.getTypeAllocSize(constant.getType());
    if (offset + size > bytes->size()) return false;
    if (constant.isNullValue() || llvm::isa<llvm::UndefValue>(constant)) return true;
    if (const auto* integer = llvm::dyn_cast<llvm::ConstantInt>(&constant)) {
        const llvm::APInt& value = integer->getValue();
        for (std::uint64_t i = 0; i < size; ++i) {
            (*bytes)[offset + i] = static_cast<std::uint8_t>(
                value.extractBitsAsZExtValue(8, static_cast<unsigned>(i * 8)));
        }
        return true;
    }
    if (const auto* floating = llvm::dyn_cast<llvm::ConstantFP>(&constant)) {
        const llvm::APInt value = floating->getValueAPF().bitcastToAPInt();
        for (std::uint64_t i = 0; i < size; ++i) {
            (*bytes)[offset + i] = static_cast<std::uint8_t>(
                value.extractBitsAsZExtValue(8, static_cast<unsigned>(i * 8)));
        }
        return true;
    }
    if (const auto* sequential =
            llvm::dyn_cast<llvm::ConstantDataSequential>(&constant)) {
        llvm::Type* element_type = sequential->getElementType();
        const std::uint64_t stride = layout.getTypeAllocSize(element_type);
        for (unsigned i = 0; i < sequential->getNumElements(); ++i) {
            llvm::APInt value = element_type->isIntegerTy()
                                    ? sequential->getElementAsAPInt(i)
                                    : sequential->getElementAsAPFloat(i).bitcastToAPInt();
            for (std::uint64_t byte = 0; byte < stride; ++byte) {
                (*bytes)[offset + i * stride + byte] =
                    static_cast<std::uint8_t>(value.extractBitsAsZExtValue(
                        8, static_cast<unsigned>(byte * 8)));
            }
        }
        return true;
    }
    if (const auto* structure = llvm::dyn_cast<llvm::StructType>(constant.getType())) {
        const llvm::StructLayout* structure_layout =
            layout.getStructLayout(const_cast<llvm::StructType*>(structure));
        for (unsigned i = 0; i < constant.getNumOperands(); ++i) {
            const auto* element = llvm::dyn_cast<llvm::Constant>(constant.getOperand(i));
            if (element == nullptr ||
                !write_constant_bytes(*element,
                                      offset + structure_layout->getElementOffset(i),
                                      bytes, layout)) {
                return false;
            }
        }
        return true;
    }
    if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(constant.getType())) {
        const std::uint64_t stride = layout.getTypeAllocSize(array->getElementType());
        for (unsigned i = 0; i < constant.getNumOperands(); ++i) {
            const auto* element = llvm::dyn_cast<llvm::Constant>(constant.getOperand(i));
            if (element == nullptr ||
                !write_constant_bytes(*element, offset + i * stride, bytes, layout)) {
                return false;
            }
        }
        return true;
    }
    if (const auto* vector = llvm::dyn_cast<llvm::FixedVectorType>(constant.getType())) {
        const std::uint64_t stride = layout.getTypeStoreSize(vector->getElementType());
        for (unsigned i = 0; i < constant.getNumOperands(); ++i) {
            const auto* element = llvm::dyn_cast<llvm::Constant>(constant.getOperand(i));
            if (element == nullptr ||
                !write_constant_bytes(*element, offset + i * stride, bytes, layout)) {
                return false;
            }
        }
        return true;
    }
    return false;
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
    std::unordered_map<ValueId, AddressSpace> integer_pointer_address_spaces;
    std::unordered_map<const llvm::Value*, std::vector<std::optional<Operand>>>
        aggregate_components;
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
            const llvm::GlobalVariable* global =
                llvm::dyn_cast<llvm::GlobalVariable>(constant);
            if (const auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(constant)) {
                if (expression->isCast()) {
                    global = llvm::dyn_cast<llvm::GlobalVariable>(
                        expression->getOperand(0));
                }
            }
            if (global != nullptr) {
                return Operand::symbol(
                    global->getName().str(),
                    Type::pointer(Type::integer(8),
                                  import_address_space(global->getAddressSpace(), false)));
            }
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
        state->output.generic_pointer_return =
            function.getReturnType()->isPointerTy() &&
            function.getReturnType()->getPointerAddressSpace() == 0;
        if (state->output.is_kernel) state->output.kernel_abi = KernelAbi{};

        std::uint32_t argument_index = 0;
        for (const llvm::Argument& argument : function.args()) {
            const Type type = import_type(argument.getType(), state->output.is_kernel);
            const ValueId value = builder.next_value();
            const std::string name =
                argument.hasName() ? argument.getName().str() : ("arg" + std::to_string(argument_index));
            state->values[&argument] = value;
            state->value_types[value] = type;
            if (!state->output.is_kernel && argument.getType()->isPointerTy() &&
                argument.getType()->getPointerAddressSpace() == 0) {
                state->output.generic_pointer_values.insert(value);
            }
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
                if (instruction.getType()->isPointerTy() &&
                    instruction.getType()->getPointerAddressSpace() == 0) {
                    state->output.generic_pointer_values.insert(value);
                }
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

    std::optional<Successor> import_successor(const llvm::Instruction& branch,
                                              const llvm::BasicBlock& source,
                                              const llvm::BasicBlock& target,
                                              FunctionState* state,
                                              BasicBlock* output_block) {
        Successor successor;
        successor.block = state->blocks.at(&target);
        for (const llvm::Instruction& instruction : target) {
            const auto* phi = llvm::dyn_cast<llvm::PHINode>(&instruction);
            if (phi == nullptr) break;
            const llvm::Value* incoming = phi->getIncomingValueForBlock(&source);
            const auto existing = state->values.find(incoming);
            if (existing != state->values.end()) {
                successor.arguments.push_back(existing->second);
                continue;
            }
            const auto* constant = llvm::dyn_cast<llvm::Constant>(incoming);
            if (constant == nullptr || llvm::isa<llvm::PoisonValue>(constant)) {
                fail(&branch, "phi incoming value is not representable");
                return std::nullopt;
            }

            // CuMetal block successors carry SSA value ids rather than general
            // operands. Materialize constant phi inputs in the predecessor so
            // the edge remains explicit and verified.
            const Type type = import_type(phi->getType());
            const ValueId materialized = builder.next_value();
            state->value_types[materialized] = type;
            Operation convert;
            convert.opcode = OpCode::kConvert;
            convert.results = {materialized};
            convert.result_types = {type};
            // LLVM undef may take any value independently at each use. Choosing
            // zero here is a valid refinement and is important for CUDA warp
            // idioms that intentionally leave non-source lanes uninitialized
            // before a shuffle. Poison remains rejected above because silently
            // refining poison would hide genuinely invalid IR.
            convert.operands = {
                llvm::isa<llvm::UndefValue>(constant)
                    ? Operand::immediate("0", type)
                    : import_operand(*constant, *state),
            };
            convert.location = import_location(branch, fallback_source);
            output_block->operations.push_back(std::move(convert));
            successor.arguments.push_back(materialized);
        }
        return successor;
    }

    bool import_call(const llvm::CallBase& call, FunctionState* state, Operation* operation) {
        if (call.isInlineAsm()) {
            const auto* assembly = llvm::dyn_cast<llvm::InlineAsm>(call.getCalledOperand());
            if (assembly == nullptr) {
                return fail(&call, "malformed LLVM inline assembly call");
            }
            const std::string text = assembly->getAsmString().str();
            if (text.find("mov.u32 $0, %laneid") != std::string::npos) {
                operation->opcode = OpCode::kLaneId;
            } else if (text.find("shfl.sync.idx.b32") != std::string::npos) {
                operation->opcode = OpCode::kShuffle;
                operation->attributes["kind"] = "index";
            } else if (text.find("shfl.sync.down.b32") != std::string::npos) {
                operation->opcode = OpCode::kShuffle;
                operation->attributes["kind"] = "down";
            } else if (text.find("shfl.sync.up.b32") != std::string::npos) {
                operation->opcode = OpCode::kShuffle;
                operation->attributes["kind"] = "up";
            } else {
                return fail(&call, "unsupported LLVM inline assembly '" + text + "'");
            }
            for (const llvm::Use& argument : call.args()) {
                operation->operands.push_back(import_operand(*argument.get(), *state));
            }
            return true;
        }
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
        if (name == "llvm.nvvm.barrier0" ||
            name.starts_with("llvm.nvvm.barrier.cta.sync")) {
            operation->opcode = OpCode::kBarrier;
            operation->memory_scope = MemoryScope::kThreadgroup;
            return true;
        }
        if (name == "llvm.nvvm.bar.warp.sync") {
            operation->opcode = OpCode::kBarrier;
            operation->memory_scope = MemoryScope::kSimdgroup;
            return true;
        }
        if (name == "__nv_float_as_int" || name == "__nv_float_as_uint" ||
            name == "__nv_int_as_float" || name == "__nv_uint_as_float" ||
            name == "__nv_double_as_longlong" || name == "__nv_longlong_as_double") {
            operation->opcode = OpCode::kConvert;
            operation->attributes["bitcast"] = "true";
            for (const llvm::Use& argument : call.args()) {
                operation->operands.push_back(import_operand(*argument.get(), *state));
            }
            return true;
        }
        static const std::unordered_map<std::string, std::string> kCudaBuiltins = {
            {"__nv_fminf", "fmin"},
            {"__nv_fmaxf", "fmax"},
            {"__nv_sqrtf", "sqrt"},
            {"__nv_fabsf", "fabs"},
            {"__nv_acosf", "acos"},
            {"__nv_popc", "popcount"},
            {"__nv_clz", "clz"},
            {"__nv_abs", "__cumetal_signed_abs"},
            {"__nv_ffs", "__cumetal_ffs"},
        };
        const auto cuda_builtin = kCudaBuiltins.find(name);
        if (cuda_builtin != kCudaBuiltins.end()) {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] = cuda_builtin->second;
            operation->attributes["builtin"] = "true";
        } else if (name.find("llvm.nvvm.shfl") == 0) {
            operation->opcode = OpCode::kShuffle;
            operation->attributes["kind"] =
                name.find(".down.") != std::string::npos
                    ? "down"
                    : (name.find(".up.") != std::string::npos ? "up" : "index");
        } else if (name.find("llvm.nvvm.vote.ballot") == 0) {
            operation->opcode = OpCode::kBallot;
            operation->attributes["kind"] = "ballot";
        } else if (name == "llvm.nvvm.activemask") {
            operation->opcode = OpCode::kBallot;
            operation->attributes["kind"] = "active_mask";
        } else if (name.find("llvm.nvvm.vote") == 0) {
            operation->opcode = OpCode::kVote;
            operation->attributes["kind"] =
                name.find(".all.") != std::string::npos ? "all" : "any";
        }
        else if (name.find("llvm.fma.") == 0) operation->opcode = OpCode::kFma;
        else if (name.find("llvm.sqrt.") == 0 || name.find("llvm.sin.") == 0 ||
                 name.find("llvm.cos.") == 0 || name.find("llvm.exp.") == 0 ||
                 name.find("llvm.log.") == 0 || name.find("llvm.fabs.") == 0 ||
                 name.find("llvm.acos.") == 0) {
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

    bool import_memcpy(const llvm::MemCpyInst& copy, FunctionState* state,
                       BasicBlock* output_block) {
        const auto* length = llvm::dyn_cast<llvm::ConstantInt>(copy.getLength());
        if (length == nullptr) {
            return fail(&copy, "dynamic-length LLVM memcpy is unsupported");
        }
        if (copy.isVolatile()) {
            return fail(&copy, "volatile LLVM memcpy is unsupported");
        }

        const std::uint64_t byte_count = length->getZExtValue();
        const std::uint64_t destination_alignment = copy.getDestAlign().value().value();
        const std::uint64_t source_alignment = copy.getSourceAlign().value().value();
        const SourceLocation location = import_location(copy, fallback_source);
        const Operand destination = import_operand(*copy.getDest(), *state);
        const Operand source = import_operand(*copy.getSource(), *state);

        auto offset_pointer = [&](const Operand& base, std::uint64_t offset) {
            if (offset == 0) return base;
            const ValueId value = builder.next_value();
            state->value_types[value] = base.type;
            Operation pointer_offset;
            pointer_offset.opcode = OpCode::kPointerOffset;
            pointer_offset.results = {value};
            pointer_offset.result_types = {base.type};
            pointer_offset.operands = {
                base,
                Operand::immediate(std::to_string(offset), Type::integer(64)),
            };
            pointer_offset.attributes["offset_unit"] = "bytes";
            pointer_offset.location = location;
            output_block->operations.push_back(std::move(pointer_offset));
            return Operand::value_ref(value, base.type);
        };

        for (std::uint64_t offset = 0; offset < byte_count;) {
            const bool word_aligned = destination_alignment >= 4 && source_alignment >= 4 &&
                                      offset % 4 == 0 && byte_count - offset >= 4;
            const std::uint64_t width = word_aligned ? 4 : 1;
            const Type value_type = Type::integer(static_cast<std::uint32_t>(width * 8));
            const Operand source_pointer = offset_pointer(source, offset);
            const Operand destination_pointer = offset_pointer(destination, offset);

            const ValueId loaded = builder.next_value();
            state->value_types[loaded] = value_type;
            Operation load;
            load.opcode = OpCode::kLoad;
            load.results = {loaded};
            load.result_types = {value_type};
            load.operands = {source_pointer};
            load.attributes["alignment"] = std::to_string(width);
            load.location = location;
            output_block->operations.push_back(std::move(load));

            Operation store;
            store.opcode = OpCode::kStore;
            store.operands = {
                destination_pointer,
                Operand::value_ref(loaded, value_type),
            };
            store.attributes["alignment"] = std::to_string(width);
            store.location = location;
            output_block->operations.push_back(std::move(store));
            offset += width;
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
            std::optional<AddressSpace> pointer_address_space;
            for (const Operand& operand : operation.operands) {
                if (operand.kind != OperandKind::kValue) continue;
                const auto provenance =
                    state->integer_pointer_address_spaces.find(operand.value);
                if (provenance == state->integer_pointer_address_spaces.end()) continue;
                if (pointer_address_space && *pointer_address_space != provenance->second) {
                    return fail(&instruction,
                                "integer arithmetic mixes incompatible pointer address spaces");
                }
                pointer_address_space = provenance->second;
            }
            if (pointer_address_space) {
                state->integer_pointer_address_spaces[operation.results.front()] =
                    *pointer_address_space;
            }
            if (binary->getOpcode() == llvm::Instruction::SDiv ||
                binary->getOpcode() == llvm::Instruction::SRem ||
                binary->getOpcode() == llvm::Instruction::AShr) {
                operation.attributes["signed"] = "true";
            }
        } else if (const auto* unary = llvm::dyn_cast<llvm::UnaryOperator>(&instruction)) {
            if (unary->getOpcode() != llvm::Instruction::FNeg) {
                return fail(&instruction, "unsupported LLVM unary operation");
            }
            operation.opcode = OpCode::kNegate;
            operation.operands.push_back(import_operand(*unary->getOperand(0), *state));
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
        } else if (const auto* insert = llvm::dyn_cast<llvm::InsertValueInst>(&instruction)) {
            if (insert->getNumIndices() != 1) {
                return fail(&instruction, "nested LLVM insertvalue is unsupported");
            }
            const Type aggregate_type = import_type(insert->getType());
            const auto constructor = homogeneous_aggregate_constructor(aggregate_type);
            if (!constructor) {
                return fail(&instruction,
                            "LLVM insertvalue requires a homogeneous 2-4 element aggregate");
            }
            std::vector<std::optional<Operand>> components(aggregate_type.elements.size());
            const llvm::Value* aggregate = insert->getAggregateOperand();
            const auto previous = state->aggregate_components.find(aggregate);
            if (previous != state->aggregate_components.end()) {
                components = previous->second;
            } else if (!llvm::isa<llvm::PoisonValue>(aggregate) &&
                       !llvm::isa<llvm::UndefValue>(aggregate)) {
                return fail(&instruction,
                            "inserting into an already-materialized aggregate is unsupported");
            }
            const unsigned index = *insert->idx_begin();
            if (index >= components.size()) {
                return fail(&instruction, "LLVM insertvalue index is out of bounds");
            }
            components[index] = import_operand(*insert->getInsertedValueOperand(), *state);
            state->aggregate_components[insert] = components;
            if (!std::all_of(components.begin(), components.end(),
                             [](const auto& component) { return component.has_value(); })) {
                return true;
            }
            operation.opcode = OpCode::kAggregateConstruct;
            operation.attributes["constructor"] = *constructor;
            for (const auto& component : components) {
                operation.operands.push_back(*component);
            }
        } else if (const auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)) {
            if (extract->getNumIndices() != 1) {
                return fail(&instruction, "nested LLVM extractvalue is unsupported");
            }
            const unsigned index = *extract->idx_begin();
            const auto components =
                state->aggregate_components.find(extract->getAggregateOperand());
            if (components != state->aggregate_components.end()) {
                if (index >= components->second.size() || !components->second[index]) {
                    return fail(&instruction,
                                "LLVM extractvalue reads an uninitialized aggregate element");
                }
                operation.opcode = OpCode::kConvert;
                operation.operands.push_back(*components->second[index]);
            } else {
                operation.opcode = OpCode::kAggregateExtract;
                operation.operands.push_back(
                    import_operand(*extract->getAggregateOperand(), *state));
                operation.operands.push_back(
                    Operand::immediate(std::to_string(index), Type::integer(32)));
            }
        } else if (const auto* cast = llvm::dyn_cast<llvm::CastInst>(&instruction)) {
            operation.operands.push_back(import_operand(*cast->getOperand(0), *state));
            if (llvm::isa<llvm::PtrToIntInst>(cast)) {
                if (cast->getType()->getIntegerBitWidth() != 64) {
                    return fail(&instruction, "pointer-to-integer conversion must target i64");
                }
                operation.opcode = OpCode::kConvert;
                operation.attributes["pointer_integer"] = "true";
                state->integer_pointer_address_spaces[operation.results.front()] =
                    operation.operands.front().type.address_space;
                state->output.pointer_provenance[operation.results.front()] = {
                    .base_kind = PointerBaseKind::kIntegerRoundTrip,
                    .base_name = value_name(operation.operands.front().value),
                };
            } else if (llvm::isa<llvm::IntToPtrInst>(cast)) {
                if (cast->getOperand(0)->getType()->getIntegerBitWidth() != 64) {
                    return fail(&instruction, "integer-to-pointer conversion must originate from i64");
                }
                operation.opcode = OpCode::kConvert;
                operation.attributes["pointer_integer"] = "true";
                AddressSpace address_space = AddressSpace::kDevice;
                if (operation.operands.front().kind == OperandKind::kValue) {
                    const auto provenance = state->integer_pointer_address_spaces.find(
                        operation.operands.front().value);
                    if (provenance != state->integer_pointer_address_spaces.end()) {
                        address_space = provenance->second;
                    }
                }
                const Type pointer_type = Type::pointer(Type::integer(8), address_space);
                operation.result_types.front() = pointer_type;
                state->value_types[operation.results.front()] = pointer_type;
                state->output.pointer_provenance[operation.results.front()] = {
                    .base_kind = PointerBaseKind::kIntegerRoundTrip,
                    .base_name = value_name(operation.operands.front().value),
                };
            } else {
                operation.opcode = llvm::isa<llvm::AddrSpaceCastInst>(cast)
                                       ? OpCode::kAddressSpaceCast
                                       : OpCode::kConvert;
                if (llvm::isa<llvm::BitCastInst>(cast)) {
                    operation.attributes["bitcast"] = "true";
                }
            }
        } else if (const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
            constexpr unsigned kPointerBits = 64;
            llvm::SmallMapVector<llvm::Value*, llvm::APInt, 4> variable_offsets;
            llvm::APInt constant_offset(kPointerBits, 0, true);
            if (!gep->collectOffset(input->getDataLayout(), kPointerBits,
                                    variable_offsets, constant_offset)) {
                return fail(&instruction, "LLVM GEP offset is not statically representable");
            }

            Operand byte_offset = Operand::immediate(
                std::to_string(constant_offset.getSExtValue()), Type::integer(kPointerBits));
            bool has_dynamic_offset = false;
            for (const auto& [index, multiplier] : variable_offsets) {
                Operand imported_index = import_operand(*index, *state);
                if (imported_index.type.kind != TypeKind::kInteger) {
                    return fail(&instruction, "LLVM GEP index is not an integer");
                }
                if (imported_index.type.bit_width != kPointerBits) {
                    const ValueId widened = builder.next_value();
                    state->value_types[widened] = Type::integer(kPointerBits);
                    Operation convert;
                    convert.opcode = OpCode::kConvert;
                    convert.results = {widened};
                    convert.result_types = {Type::integer(kPointerBits)};
                    convert.operands = {imported_index};
                    convert.attributes["signed"] = "true";
                    convert.location = operation.location;
                    output_block->operations.push_back(std::move(convert));
                    imported_index =
                        Operand::value_ref(widened, Type::integer(kPointerBits));
                }

                const ValueId scaled = builder.next_value();
                state->value_types[scaled] = Type::integer(kPointerBits);
                Operation multiply;
                multiply.opcode = OpCode::kMul;
                multiply.results = {scaled};
                multiply.result_types = {Type::integer(kPointerBits)};
                multiply.operands = {
                    imported_index,
                    Operand::immediate(std::to_string(multiplier.getSExtValue()),
                                       Type::integer(kPointerBits)),
                };
                multiply.location = operation.location;
                output_block->operations.push_back(std::move(multiply));

                if (!has_dynamic_offset && constant_offset.isZero()) {
                    byte_offset = Operand::value_ref(scaled, Type::integer(kPointerBits));
                } else {
                    const ValueId sum = builder.next_value();
                    state->value_types[sum] = Type::integer(kPointerBits);
                    Operation add;
                    add.opcode = OpCode::kAdd;
                    add.results = {sum};
                    add.result_types = {Type::integer(kPointerBits)};
                    add.operands = {
                        byte_offset,
                        Operand::value_ref(scaled, Type::integer(kPointerBits)),
                    };
                    add.location = operation.location;
                    output_block->operations.push_back(std::move(add));
                    byte_offset = Operand::value_ref(sum, Type::integer(kPointerBits));
                }
                has_dynamic_offset = true;
            }
            operation.opcode = OpCode::kPointerOffset;
            operation.operands.push_back(import_operand(*gep->getPointerOperand(), *state));
            operation.operands.push_back(std::move(byte_offset));
            if (operation.operands.front().type.is_pointer()) {
                operation.result_types.front() = operation.operands.front().type;
                state->value_types[operation.results.front()] = operation.operands.front().type;
            }
            operation.attributes["offset_unit"] = "bytes";
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
        } else if (const auto* copy = llvm::dyn_cast<llvm::MemCpyInst>(&instruction)) {
            return import_memcpy(*copy, state, output_block);
        } else if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
            if (const llvm::Function* callee = call->getCalledFunction();
                callee != nullptr &&
                (callee->getName() == "llvm.experimental.noalias.scope.decl" ||
                 callee->getName() == "llvm.assume")) {
                // This intrinsic only communicates alias-analysis metadata to
                // LLVM optimization passes and has no runtime GPU semantics.
                return true;
            }
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
                auto true_successor = import_successor(
                    instruction, source_block, *branch->getSuccessor(0), state, output_block);
                auto false_successor = import_successor(
                    instruction, source_block, *branch->getSuccessor(1), state, output_block);
                if (!true_successor || !false_successor) return false;
                operation.successors.push_back(std::move(*true_successor));
                operation.successors.push_back(std::move(*false_successor));
            } else {
                operation.opcode = OpCode::kBranch;
                auto successor = import_successor(
                    instruction, source_block, *branch->getSuccessor(0), state, output_block);
                if (!successor) return false;
                operation.successors.push_back(std::move(*successor));
            }
        } else if (const auto* ret = llvm::dyn_cast<llvm::ReturnInst>(&instruction)) {
            operation.opcode = OpCode::kReturn;
            if (ret->getReturnValue() != nullptr) {
                operation.operands.push_back(import_operand(*ret->getReturnValue(), *state));
            }
        } else if (llvm::isa<llvm::UnreachableInst>(instruction)) {
            operation.opcode = OpCode::kTrap;
        } else if (llvm::isa<llvm::AllocaInst>(instruction)) {
            const auto& alloca = llvm::cast<llvm::AllocaInst>(instruction);
            const auto* count = llvm::dyn_cast<llvm::ConstantInt>(alloca.getArraySize());
            if (count == nullptr || !count->isOne()) {
                return fail(&instruction,
                            "dynamic or multi-element LLVM alloca is unsupported");
            }
            const Type pointer_type = Type::pointer(
                import_type(alloca.getAllocatedType()), AddressSpace::kPrivate);
            operation.result_types.front() = pointer_type;
            state->value_types[operation.results.front()] = pointer_type;
            operation.opcode = OpCode::kAlloca;
            operation.attributes["alignment"] =
                std::to_string(alloca.getAlign().value());
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

        for (const llvm::GlobalVariable& global : module->globals()) {
            if (global.getAddressSpace() == 3 && global.hasInitializer() &&
                !global.getName().starts_with("llvm.")) {
                result.module.global_threadgroups.push_back({
                    .name = global.getName().str(),
                    .byte_size = module->getDataLayout().getTypeAllocSize(
                        global.getValueType()),
                    .alignment = static_cast<std::uint32_t>(
                        global.getAlign().has_value()
                            ? global.getAlign()->value()
                            : 1),
                });
                continue;
            }
            if (!global.isConstant() || !global.hasInitializer() ||
                global.getAddressSpace() != 4 ||
                global.getName().starts_with("llvm.")) {
                continue;
            }
            const std::uint64_t size =
                module->getDataLayout().getTypeAllocSize(global.getValueType());
            GlobalConstant imported;
            imported.name = global.getName().str();
            imported.bytes.assign(size, 0);
            imported.alignment = global.getAlign().has_value()
                                     ? global.getAlign()->value()
                                     : 1;
            if (!write_constant_bytes(*global.getInitializer(), 0, &imported.bytes,
                                      module->getDataLayout())) {
                result.error = "unsupported constant global initializer: " +
                               imported.name;
                return std::move(result);
            }
            result.module.global_constants.push_back(std::move(imported));
        }

        std::vector<const llvm::Function*> functions_to_import;
        if (options.entry_name.empty()) {
            for (const llvm::Function& function : *module) {
                if (!function.isDeclaration()) functions_to_import.push_back(&function);
            }
        } else {
            const llvm::Function* root = module->getFunction(options.entry_name);
            if (root == nullptr || root->isDeclaration()) {
                result.error = "NVVM kernel not found: " + options.entry_name;
                return std::move(result);
            }
            if (root->getCallingConv() != llvm::CallingConv::PTX_Kernel) {
                result.error = "selected NVVM entry is not a kernel: " + options.entry_name;
                return std::move(result);
            }

            std::unordered_set<const llvm::Function*> visited;
            const auto collect_reachable = [&](const auto& self,
                                               const llvm::Function* function) -> void {
                if (!visited.insert(function).second) return;
                for (const llvm::BasicBlock& block : *function) {
                    for (const llvm::Instruction& instruction : block) {
                        const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                        if (call == nullptr) continue;
                        const llvm::Function* callee = call->getCalledFunction();
                        if (callee != nullptr && !callee->isDeclaration()) {
                            self(self, callee);
                        }
                    }
                }
                // Callees precede callers so emitted MSL never needs an
                // implicit forward declaration.
                functions_to_import.push_back(function);
            };
            collect_reachable(collect_reachable, root);
        }

        for (const llvm::Function* function : functions_to_import) {
            if (!import_function(*function)) return std::move(result);
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
    std::string verification_message;
    llvm::raw_string_ostream verification_stream(verification_message);
    if (llvm::verifyModule(*module, &verification_stream)) {
        verification_stream.flush();
        NvvmImportResult result;
        result.error = "invalid LLVM/NVVM module: " + verification_message;
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
