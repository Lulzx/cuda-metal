#include "cumetal/ir/nvvm_importer.h"

#ifndef CUMETAL_HAVE_LLVM
#define CUMETAL_HAVE_LLVM 0
#endif

#if CUMETAL_HAVE_LLVM

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <algorithm>
#include <iomanip>
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

bool contains_fp64(llvm::Type* type) {
    if (type->isDoubleTy()) return true;
    if (const auto* vector = llvm::dyn_cast<llvm::VectorType>(type)) {
        return contains_fp64(vector->getElementType());
    }
    if (const auto* array = llvm::dyn_cast<llvm::ArrayType>(type)) {
        return contains_fp64(array->getElementType());
    }
    if (const auto* structure = llvm::dyn_cast<llvm::StructType>(type)) {
        return std::any_of(structure->element_begin(), structure->element_end(),
                           [](llvm::Type* element) { return contains_fp64(element); });
    }
    return false;
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
    if (const auto* floating = llvm::dyn_cast<llvm::ConstantFP>(&constant)) {
        llvm::SmallVector<char, 32> text;
        floating->getValueAPF().toString(text);
        std::string spelling(text.begin(), text.end());
        if (spelling.find_first_of(".eE") == std::string::npos &&
            spelling != "NaN" && spelling != "+Inf" && spelling != "-Inf") {
            spelling += ".0";
        }
        return spelling;
    }
    std::string spelling;
    llvm::raw_string_ostream stream(spelling);
    constant.printAsOperand(stream, false);
    stream.flush();
    return spelling;
}

struct DemotedFloatMultiply {
    const llvm::FPExtInst* extension = nullptr;
    const llvm::BinaryOperator* multiply = nullptr;
    const llvm::ConstantFP* constant = nullptr;
};

std::optional<DemotedFloatMultiply> match_demotable_float_multiply(
    const llvm::Instruction& instruction) {
    const auto* truncation = llvm::dyn_cast<llvm::FPTruncInst>(&instruction);
    if (truncation == nullptr || !truncation->getType()->isFloatTy()) {
        return std::nullopt;
    }
    const auto* multiply = llvm::dyn_cast<llvm::BinaryOperator>(
        truncation->getOperand(0));
    if (multiply == nullptr || multiply->getOpcode() != llvm::Instruction::FMul ||
        !multiply->getType()->isDoubleTy() || !multiply->hasOneUse()) {
        return std::nullopt;
    }
    const llvm::FPExtInst* extension = nullptr;
    const llvm::ConstantFP* constant = nullptr;
    for (unsigned i = 0; i < 2; ++i) {
        extension = llvm::dyn_cast<llvm::FPExtInst>(multiply->getOperand(i));
        constant = llvm::dyn_cast<llvm::ConstantFP>(multiply->getOperand(1 - i));
        if (extension != nullptr && constant != nullptr &&
            extension->getOperand(0)->getType()->isFloatTy() &&
            extension->hasOneUse()) {
            return DemotedFloatMultiply{extension, multiply, constant};
        }
    }
    return std::nullopt;
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

bool is_nvvm_kernel(const llvm::Function& function) {
    if (function.getCallingConv() == llvm::CallingConv::PTX_Kernel) {
        return true;
    }
    const llvm::Module* module = function.getParent();
    const llvm::NamedMDNode* annotations =
        module != nullptr ? module->getNamedMetadata("nvvm.annotations") : nullptr;
    if (annotations == nullptr) return false;

    for (const llvm::MDNode* annotation : annotations->operands()) {
        if (annotation == nullptr || annotation->getNumOperands() < 2) continue;
        const auto* annotated_value =
            llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(
                annotation->getOperand(0).get());
        const auto* property =
            llvm::dyn_cast_or_null<llvm::MDString>(
                annotation->getOperand(1).get());
        if (annotated_value != nullptr &&
            annotated_value->getValue() == &function &&
            property != nullptr && property->getString() == "kernel") {
            return true;
        }
    }
    return false;
}

struct FunctionState {
    Function output;
    std::unordered_map<const llvm::Value*, ValueId> values;
    std::unordered_map<const llvm::BasicBlock*, BlockId> blocks;
    std::unordered_map<ValueId, Type> value_types;
    std::unordered_map<ValueId, AddressSpace> integer_pointer_address_spaces;
    std::unordered_map<ValueId, ValueId> integer_pointer_sources;
    struct AggregateState {
        Type type;
        std::map<std::vector<unsigned>, Operand> leaves;
    };
    std::unordered_map<const llvm::Value*, AggregateState> aggregate_components;
    struct ExternalGlobalBinding {
        Operand base;
        std::uint64_t byte_offset = 0;
    };
    std::unordered_map<const llvm::GlobalVariable*, ExternalGlobalBinding>
        external_globals;
    std::unordered_set<const llvm::AllocaInst*> printf_argument_allocas;
    std::optional<Operand> printf_buffer;
    std::optional<Operand> printf_capacity;
};

struct Importer {
    Builder builder;
    NvvmImportResult result;
    llvm::Module* input = nullptr;
    std::string fallback_source;
    struct ExternalGlobalInfo {
        const llvm::GlobalVariable* global = nullptr;
        std::uint64_t byte_size = 0;
        std::uint32_t alignment = 1;
        std::uint64_t constant_offset = 0;
        bool constant = false;
    };
    std::vector<ExternalGlobalInfo> external_globals;
    std::uint64_t external_constant_buffer_size = 0;

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
            if (const auto* floating = llvm::dyn_cast<llvm::ConstantFP>(constant);
                floating != nullptr && floating->getType()->isDoubleTy()) {
                std::ostringstream bits;
                bits << "0x" << std::hex << std::setw(16) << std::setfill('0')
                     << floating->getValueAPF().bitcastToAPInt().getZExtValue()
                     << "ul";
                return Operand::immediate(bits.str(), Type::floating(64));
            }
            return Operand::immediate(constant_spelling(*constant), import_type(value.getType()));
        }
        const auto found = state.values.find(&value);
        if (found == state.values.end()) {
            return Operand::symbol("<undefined>", import_type(value.getType()));
        }
        return Operand::value_ref(found->second, state.value_types.at(found->second));
    }

    static const llvm::GlobalVariable* referenced_global(const llvm::Value& value) {
        if (const auto* global = llvm::dyn_cast<llvm::GlobalVariable>(&value)) {
            return global;
        }
        const auto* constant = llvm::dyn_cast<llvm::ConstantExpr>(&value);
        if (constant == nullptr || constant->getNumOperands() == 0) return nullptr;
        if (constant->isCast() || constant->getOpcode() == llvm::Instruction::GetElementPtr) {
            return referenced_global(*constant->getOperand(0));
        }
        return nullptr;
    }

    static bool is_empty_device_assert(const llvm::Function* function) {
        return function != nullptr &&
               (function->getName() == "_Z12__assert_rtnPKcS0_iS0_" ||
                function->getName() == "__assert_fail") &&
               !function->isDeclaration() && function->size() == 1 &&
               function->front().size() == 1 &&
               llvm::isa<llvm::ReturnInst>(function->front().front());
    }

    bool function_references_global(const llvm::Function& function,
                                    const llvm::GlobalVariable& global) const {
        std::unordered_set<const llvm::Function*> visited;
        const auto references = [&](const auto& self,
                                    const llvm::Function& candidate) -> bool {
            if (!visited.insert(&candidate).second) return false;
            for (const llvm::BasicBlock& block : candidate) {
                for (const llvm::Instruction& instruction : block) {
                    for (const llvm::Use& operand : instruction.operands()) {
                        if (referenced_global(*operand.get()) == &global) return true;
                    }
                    const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                    const llvm::Function* callee =
                        call == nullptr ? nullptr : call->getCalledFunction();
                    if (callee != nullptr && !callee->isDeclaration() &&
                        self(self, *callee)) {
                        return true;
                    }
                }
            }
            return false;
        };
        return references(references, function);
    }

    static bool function_uses_vprintf(const llvm::Function& function) {
        std::unordered_set<const llvm::Function*> visited;
        const auto uses_vprintf = [&](const auto& self,
                                      const llvm::Function& candidate) -> bool {
            if (!visited.insert(&candidate).second) return false;
            for (const llvm::BasicBlock& block : candidate) {
                for (const llvm::Instruction& instruction : block) {
                    const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                    const llvm::Function* callee =
                        call == nullptr ? nullptr : call->getCalledFunction();
                    if (callee == nullptr) continue;
                    if (callee->getName() == "vprintf") return true;
                    if (!callee->isDeclaration() && self(self, *callee)) return true;
                }
            }
            return false;
        };
        return uses_vprintf(uses_vprintf, function);
    }

    bool constant_pointer_base_and_offset(const llvm::Value& value,
                                          const llvm::Value** base,
                                          std::int64_t* byte_offset) const {
        if (const auto* gep = llvm::dyn_cast<llvm::GEPOperator>(&value)) {
            llvm::APInt offset(64, 0, true);
            if (!gep->accumulateConstantOffset(input->getDataLayout(), offset) ||
                !constant_pointer_base_and_offset(*gep->getPointerOperand(), base,
                                                  byte_offset)) {
                return false;
            }
            *byte_offset += offset.getSExtValue();
            return true;
        }
        if (const auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(&value);
            expression != nullptr && expression->isCast()) {
            return constant_pointer_base_and_offset(*expression->getOperand(0), base,
                                                    byte_offset);
        }
        if (const auto* cast = llvm::dyn_cast<llvm::CastInst>(&value)) {
            return constant_pointer_base_and_offset(*cast->getOperand(0), base,
                                                    byte_offset);
        }
        *base = &value;
        return true;
    }

    std::optional<std::uint32_t> printf_format_id(const llvm::Value& value) {
        const llvm::GlobalVariable* global = referenced_global(value);
        if (global == nullptr || !global->hasInitializer()) return std::nullopt;
        const auto* data = llvm::dyn_cast<llvm::ConstantDataArray>(global->getInitializer());
        if (data == nullptr || !data->isString()) return std::nullopt;
        std::string format = data->getAsString().str();
        if (!format.empty() && format.back() == '\0') format.pop_back();
        const auto found = std::find(result.printf_formats.begin(),
                                     result.printf_formats.end(), format);
        if (found != result.printf_formats.end()) {
            return static_cast<std::uint32_t>(
                std::distance(result.printf_formats.begin(), found));
        }
        result.printf_formats.push_back(std::move(format));
        return static_cast<std::uint32_t>(result.printf_formats.size() - 1);
    }

    bool import_vprintf(const llvm::CallBase& call, FunctionState* state,
                        Operation* operation) {
        if (!state->printf_buffer.has_value() ||
            !state->printf_capacity.has_value() || call.arg_size() != 2) {
            return fail(&call, "typed vprintf is missing its ring-buffer ABI");
        }
        const std::optional<std::uint32_t> format_id =
            printf_format_id(*call.getArgOperand(0));
        if (!format_id.has_value()) {
            return fail(&call, "typed vprintf requires a constant format string");
        }

        const bool no_arguments =
            llvm::isa<llvm::ConstantPointerNull>(call.getArgOperand(1));
        const llvm::Value* argument_base = nullptr;
        if (!no_arguments) {
            std::int64_t ignored_offset = 0;
            if (!constant_pointer_base_and_offset(*call.getArgOperand(1),
                                                  &argument_base,
                                                  &ignored_offset) ||
                ignored_offset != 0 ||
                !llvm::isa<llvm::AllocaInst>(argument_base)) {
                return fail(
                    &call,
                    "typed vprintf requires a null or statically packed argument tuple");
            }
        }

        std::map<std::int64_t, const llvm::Value*> packed_arguments;
        for (const llvm::BasicBlock& block : *call.getFunction()) {
            for (const llvm::Instruction& instruction : block) {
                const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction);
                if (store == nullptr) continue;
                const llvm::Value* store_base = nullptr;
                std::int64_t store_offset = 0;
                if (!constant_pointer_base_and_offset(*store->getPointerOperand(),
                                                      &store_base, &store_offset) ||
                    store_base != argument_base) {
                    continue;
                }
                if (store_offset < 0 ||
                    !packed_arguments.emplace(store_offset,
                                              store->getValueOperand()).second) {
                    return fail(&call,
                                "typed vprintf argument tuple has overlapping or negative fields");
                }
            }
        }
        if (!no_arguments && packed_arguments.empty()) {
            return fail(&call, "typed vprintf argument tuple is empty");
        }

        operation->opcode = OpCode::kPrintf;
        operation->operands = {*state->printf_buffer, *state->printf_capacity};
        operation->attributes["format_id"] = std::to_string(*format_id);
        std::ostringstream widths;
        std::int64_t expected_offset = 0;
        bool first = true;
        for (const auto& [offset, value] : packed_arguments) {
            llvm::Type* type = value->getType();
            const std::uint64_t bits = input->getDataLayout().getTypeSizeInBits(type);
            if ((!type->isIntegerTy() && !type->isFloatingPointTy() &&
                 !type->isPointerTy()) ||
                (bits != 32 && bits != 64) || offset != expected_offset) {
                return fail(&call,
                            "typed vprintf supports tightly packed 32/64-bit scalar and pointer arguments");
            }
            if (!first) widths << ',';
            widths << bits;
            first = false;
            if (type->isPointerTy()) {
                const llvm::GlobalVariable* global = referenced_global(*value);
                const auto external = global == nullptr
                    ? state->external_globals.end()
                    : state->external_globals.find(global);
                if (external != state->external_globals.end()) {
                    const llvm::Value* pointer_base = nullptr;
                    std::int64_t pointer_offset = 0;
                    if (!constant_pointer_base_and_offset(
                            *value, &pointer_base, &pointer_offset) ||
                        pointer_base != global || pointer_offset != 0 ||
                        external->second.byte_offset != 0) {
                        return fail(
                            &call,
                            "typed vprintf module-string pointers require zero offset");
                    }
                    operation->operands.push_back(external->second.base);
                    expected_offset += static_cast<std::int64_t>(bits / 8);
                    continue;
                }
                if (const auto* expression =
                        llvm::dyn_cast<llvm::ConstantExpr>(value);
                    expression != nullptr &&
                    expression->getOpcode() == llvm::Instruction::IntToPtr) {
                    const auto* address = llvm::dyn_cast<llvm::ConstantInt>(
                        expression->getOperand(0));
                    if (address == nullptr) {
                        return fail(
                            &call,
                            "typed vprintf inttoptr arguments require a constant integer address");
                    }
                    operation->operands.push_back(Operand::immediate(
                        constant_spelling(*address), Type::integer(64)));
                    expected_offset += static_cast<std::int64_t>(bits / 8);
                    continue;
                }
            }
            operation->operands.push_back(import_operand(*value, *state));
            expected_offset += static_cast<std::int64_t>(bits / 8);
        }
        operation->attributes["argument_bits"] = widths.str();
        return true;
    }

    std::optional<Operand> import_external_pointer(
        const llvm::Value& value, FunctionState* state, BasicBlock* output_block,
        const SourceLocation& location) {
        const llvm::GlobalVariable* global = referenced_global(value);
        if (global == nullptr) return std::nullopt;
        const auto found = state->external_globals.find(global);
        if (found == state->external_globals.end()) return std::nullopt;

        std::int64_t expression_offset = 0;
        const llvm::Value* cursor = &value;
        while (const auto* expression = llvm::dyn_cast<llvm::ConstantExpr>(cursor)) {
            if (expression->getOpcode() == llvm::Instruction::GetElementPtr) {
                llvm::APInt offset(64, 0, true);
                const auto* gep = llvm::cast<llvm::GEPOperator>(expression);
                if (!gep->accumulateConstantOffset(input->getDataLayout(), offset)) {
                    return std::nullopt;
                }
                expression_offset += offset.getSExtValue();
            } else if (!expression->isCast()) {
                return std::nullopt;
            }
            cursor = expression->getOperand(0);
        }

        const std::uint64_t base_offset = found->second.byte_offset;
        if (expression_offset < 0 &&
            static_cast<std::uint64_t>(-expression_offset) > base_offset) {
            return std::nullopt;
        }
        const std::int64_t total_offset =
            static_cast<std::int64_t>(base_offset) + expression_offset;
        if (total_offset == 0) return found->second.base;

        const ValueId pointer = builder.next_value();
        state->value_types[pointer] = found->second.base.type;
        Operation offset;
        offset.opcode = OpCode::kPointerOffset;
        offset.results = {pointer};
        offset.result_types = {found->second.base.type};
        offset.operands = {
            found->second.base,
            Operand::immediate(std::to_string(total_offset), Type::integer(64)),
        };
        offset.attributes["offset_unit"] = "bytes";
        offset.location = location;
        output_block->operations.push_back(std::move(offset));
        return Operand::value_ref(pointer, found->second.base.type);
    }

    Operand import_pointer_operand(const llvm::Value& value, FunctionState* state,
                                   BasicBlock* output_block,
                                   const SourceLocation& location) {
        if (auto external = import_external_pointer(value, state, output_block, location)) {
            return *external;
        }
        return import_operand(value, *state);
    }

    bool validate_aggregate_shape(const Type& type, std::size_t depth,
                                  std::size_t* leaf_count) const {
        if (type.kind != TypeKind::kAggregate) {
            ++*leaf_count;
            return *leaf_count <= 64;
        }
        if (type.elements.empty() || type.elements.size() > 16 || depth >= 8) {
            return false;
        }
        for (const Type& element : type.elements) {
            if (!validate_aggregate_shape(element, depth + 1, leaf_count)) return false;
        }
        return true;
    }

    static void collect_leaf_paths(const Type& type, std::vector<unsigned>* prefix,
                                   std::vector<std::vector<unsigned>>* paths) {
        if (type.kind != TypeKind::kAggregate) {
            paths->push_back(*prefix);
            return;
        }
        for (unsigned index = 0; index < type.elements.size(); ++index) {
            prefix->push_back(index);
            collect_leaf_paths(type.elements[index], prefix, paths);
            prefix->pop_back();
        }
    }

    static const Type* type_at_path(const Type& root,
                                    const std::vector<unsigned>& path) {
        const Type* current = &root;
        for (unsigned index : path) {
            if (current->kind != TypeKind::kAggregate ||
                index >= current->elements.size()) {
                return nullptr;
            }
            current = &current->elements[index];
        }
        return current;
    }

    std::optional<Operand> emit_aggregate_extract(
        Operand aggregate, const Type& aggregate_type,
        const std::vector<unsigned>& path, std::optional<ValueId> final_result,
        FunctionState* state, BasicBlock* output_block,
        const SourceLocation& location) {
        Operand current = std::move(aggregate);
        const Type* current_type = &aggregate_type;
        for (std::size_t depth = 0; depth < path.size(); ++depth) {
            const unsigned index = path[depth];
            if (current_type->kind != TypeKind::kAggregate ||
                index >= current_type->elements.size()) {
                return std::nullopt;
            }
            const Type next_type = current_type->elements[index];
            const bool last = depth + 1 == path.size();
            const ValueId result_value =
                last && final_result.has_value() ? *final_result : builder.next_value();
            state->value_types[result_value] = next_type;

            Operation extract;
            extract.opcode = OpCode::kAggregateExtract;
            extract.results = {result_value};
            extract.result_types = {next_type};
            extract.operands = {
                current,
                Operand::immediate(std::to_string(index), Type::integer(32)),
            };
            extract.location = location;
            output_block->operations.push_back(std::move(extract));
            current = Operand::value_ref(result_value, next_type);
            current_type = &current_type->elements[index];
        }
        return current;
    }

    bool decompose_aggregate_operand(
        const Operand& aggregate, const Type& aggregate_type,
        const std::vector<unsigned>& destination_prefix,
        std::map<std::vector<unsigned>, Operand>* leaves, FunctionState* state,
        BasicBlock* output_block, const SourceLocation& location) {
        std::vector<std::vector<unsigned>> relative_paths;
        std::vector<unsigned> prefix;
        collect_leaf_paths(aggregate_type, &prefix, &relative_paths);
        for (const std::vector<unsigned>& relative_path : relative_paths) {
            const auto extracted = emit_aggregate_extract(
                aggregate, aggregate_type, relative_path, std::nullopt,
                state, output_block, location);
            if (!extracted.has_value()) return false;
            std::vector<unsigned> destination = destination_prefix;
            destination.insert(destination.end(), relative_path.begin(),
                               relative_path.end());
            (*leaves)[std::move(destination)] = *extracted;
        }
        return true;
    }

    std::optional<Operand> materialize_aggregate(
        const Type& type, const std::vector<unsigned>& prefix,
        const std::map<std::vector<unsigned>, Operand>& leaves,
        std::optional<ValueId> result_value, FunctionState* state,
        BasicBlock* output_block, const SourceLocation& location) {
        if (type.kind != TypeKind::kAggregate) {
            const auto found = leaves.find(prefix);
            return found == leaves.end() ? std::nullopt
                                         : std::optional<Operand>(found->second);
        }

        std::vector<Operand> elements;
        elements.reserve(type.elements.size());
        for (unsigned index = 0; index < type.elements.size(); ++index) {
            std::vector<unsigned> child_prefix = prefix;
            child_prefix.push_back(index);
            const auto child = materialize_aggregate(
                type.elements[index], child_prefix, leaves, std::nullopt,
                state, output_block, location);
            if (!child.has_value()) return std::nullopt;
            elements.push_back(*child);
        }

        const ValueId value = result_value.value_or(builder.next_value());
        state->value_types[value] = type;
        Operation construct;
        construct.opcode = OpCode::kAggregateConstruct;
        construct.results = {value};
        construct.result_types = {type};
        construct.operands = std::move(elements);
        if (const auto constructor = homogeneous_aggregate_constructor(type)) {
            construct.attributes["constructor"] = *constructor;
        } else {
            construct.attributes["aggregate_init"] = "true";
        }
        construct.location = location;
        output_block->operations.push_back(std::move(construct));
        return Operand::value_ref(value, type);
    }

    bool allocate_function(const llvm::Function& function, FunctionState* state) {
        state->output.name = function.getName().str();
        state->output.is_kernel = is_nvvm_kernel(function);
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

        for (const ExternalGlobalInfo& info : external_globals) {
            if (!function_references_global(function, *info.global)) continue;
            const std::uint32_t binding_index = info.constant ? 30u : argument_index++;
            if (state->output.is_kernel && !info.constant && binding_index >= 29u) {
                return fail(nullptr, "CUDA device global conflicts with reserved Metal bindings");
            }
            const Type pointer_type = Type::pointer(
                Type::integer(8),
                info.constant ? AddressSpace::kConstant : AddressSpace::kDevice);
            const ValueId value = builder.next_value();
            const std::string name = info.constant
                ? "__cumetal_constant_symbols"
                : "__cumetal_global_" + info.global->getName().str();
            Operand base = Operand::value_ref(value, pointer_type);
            if (info.constant) {
                const auto existing = std::find_if(
                    state->output.arguments.begin(), state->output.arguments.end(),
                    [](const FunctionArgument& argument) {
                        return argument.name == "__cumetal_constant_symbols";
                    });
                if (existing != state->output.arguments.end()) {
                    base = Operand::value_ref(existing->value, existing->type);
                    state->external_globals[info.global] = {
                        .base = base, .byte_offset = info.constant_offset};
                    continue;
                }
            }
            state->value_types[value] = pointer_type;
            state->output.arguments.push_back({.value = value, .name = name, .type = pointer_type});
            state->output.pointer_provenance[value] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = name,
                .known_byte_offset = 0,
                .alignment = info.alignment,
            };
            state->external_globals[info.global] = {
                .base = base, .byte_offset = info.constant ? info.constant_offset : 0};
            if (!state->output.is_kernel) continue;
            const std::uint32_t logical_index =
                static_cast<std::uint32_t>(state->output.kernel_abi->arguments.size());
            const std::string hidden_role =
                info.constant ? "constant_symbols" : "global_symbol:" + info.global->getName().str();
            state->output.kernel_abi->arguments.push_back({
                .name = name,
                .kind = ArgumentKind::kPointer,
                .type = pointer_type,
                .size = 8,
                .alignment = 8,
                .address_space = pointer_type.address_space,
                .binding_indices = {binding_index},
                .hidden_role = hidden_role,
            });
            state->output.kernel_abi->bindings.push_back({
                .kind = BindingKind::kBuffer,
                .binding_index = binding_index,
                .logical_argument_index = logical_index,
                .type = pointer_type,
                .size = static_cast<std::uint32_t>(
                    info.constant ? external_constant_buffer_size : info.byte_size),
                .alignment = info.alignment,
                .hidden_role = hidden_role,
            });
        }

        if (function_uses_vprintf(function)) {
            if (state->output.is_kernel && argument_index + 1 >= 29u) {
                return fail(nullptr,
                            "typed vprintf hidden arguments conflict with reserved Metal bindings");
            }
            const Type buffer_type =
                Type::pointer(Type::integer(32), AddressSpace::kDevice);
            const Type capacity_type = Type::integer(32);
            const ValueId buffer = builder.next_value();
            const ValueId capacity = builder.next_value();
            state->value_types[buffer] = buffer_type;
            state->value_types[capacity] = capacity_type;
            state->printf_buffer = Operand::value_ref(buffer, buffer_type);
            state->printf_capacity = Operand::value_ref(capacity, capacity_type);
            state->output.arguments.push_back({
                .value = buffer, .name = "__cumetal_printf_buffer", .type = buffer_type});
            state->output.arguments.push_back({
                .value = capacity, .name = "__cumetal_printf_capacity", .type = capacity_type});
            state->output.pointer_provenance[buffer] = {
                .base_kind = PointerBaseKind::kAllocation,
                .base_name = "__cumetal_printf_buffer",
                .known_byte_offset = 0,
                .alignment = 4,
            };
            if (state->output.is_kernel) {
                const std::uint32_t buffer_logical =
                    static_cast<std::uint32_t>(state->output.kernel_abi->arguments.size());
                state->output.kernel_abi->arguments.push_back({
                    .name = "__cumetal_printf_buffer",
                    .kind = ArgumentKind::kPointer,
                    .type = buffer_type,
                    .size = 8,
                    .alignment = 8,
                    .address_space = AddressSpace::kDevice,
                    .binding_indices = {argument_index},
                    .hidden_role = "printf_buffer",
                });
                state->output.kernel_abi->bindings.push_back({
                    .kind = BindingKind::kBuffer,
                    .binding_index = argument_index++,
                    .logical_argument_index = buffer_logical,
                    .type = buffer_type,
                    .size = 0,
                    .alignment = 4,
                    .hidden_role = "printf_buffer",
                });
                const std::uint32_t capacity_logical =
                    static_cast<std::uint32_t>(state->output.kernel_abi->arguments.size());
                state->output.kernel_abi->arguments.push_back({
                    .name = "__cumetal_printf_capacity",
                    .kind = ArgumentKind::kScalar,
                    .type = capacity_type,
                    .size = 4,
                    .alignment = 4,
                    .address_space = AddressSpace::kConstant,
                    .binding_indices = {argument_index},
                    .hidden_role = "printf_capacity",
                });
                state->output.kernel_abi->bindings.push_back({
                    .kind = BindingKind::kBytes,
                    .binding_index = argument_index++,
                    .logical_argument_index = capacity_logical,
                    .type = capacity_type,
                    .size = 4,
                    .alignment = 4,
                    .hidden_role = "printf_capacity",
                });
            }
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
            if (phi->getType()->isPointerTy() &&
                phi->getType()->getPointerAddressSpace() == 0) {
                state->output.generic_pointer_values.insert(materialized);
                if (llvm::isa<llvm::ConstantPointerNull>(constant)) {
                    state->output.generic_null_pointer_values.insert(materialized);
                }
            }
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
            const std::string text = llvm::StringRef(assembly->getAsmString()).str();
            if (text.find("mov.u32 $0, %activemask") != std::string::npos) {
                operation->opcode = OpCode::kBallot;
                operation->attributes["kind"] = "active_mask";
            } else if (text.find("mov.u32 $0, %laneid") != std::string::npos) {
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

        if (name == "vprintf") return import_vprintf(call, state, operation);

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
        if (name == "llvm.nvvm.read.ptx.sreg.clock" ||
            name == "llvm.nvvm.read.ptx.sreg.clock64" ||
            name == "llvm.nvvm.read.ptx.sreg.globaltimer") {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] = "cm_device_clock";
            operation->attributes["builtin"] = "true";
            if (result.module.semantic_quality == SemanticQuality::kExact) {
                result.module.semantic_quality = SemanticQuality::kSemanticEmulation;
            }
            const std::string caveat =
                "CUDA device clock uses a monotonic atomic quantum, not GPU cycles";
            if (std::find(result.module.semantic_caveats.begin(),
                          result.module.semantic_caveats.end(), caveat) ==
                result.module.semantic_caveats.end()) {
                result.module.semantic_caveats.push_back(caveat);
            }
            return true;
        }
        if (name == "__cumetal_grid_sync") {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] = "cm_grid_sync";
            operation->attributes["builtin"] = "true";
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
        if (name == "llvm.nvvm.membar.cta" ||
            name == "llvm.nvvm.membar.gl" ||
            name == "llvm.nvvm.membar.sys") {
            operation->opcode = OpCode::kFence;
            operation->memory_scope =
                name.ends_with(".cta")
                    ? MemoryScope::kThreadgroup
                    : name.ends_with(".sys") ? MemoryScope::kSystem
                                               : MemoryScope::kDevice;
            operation->memory_ordering = MemoryOrdering::kRelaxed;
            operation->attributes["cuda_membar"] = "true";
            if (operation->memory_scope == MemoryScope::kSystem) {
                operation->attributes["metal_uma_system_scope"] = "true";
            }
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
        // Integer -> float conversions are a plain numeric cast in MSL; the
        // rounding-mode suffix is not separable in binary32 anyway, so every
        // variant lowers to the same conversion the IR already models.
        static const std::unordered_set<std::string> kIntToFloatConversions = {
            "__nv_int2float_rn",  "__nv_int2float_rz",  "__nv_int2float_ru",
            "__nv_int2float_rd",  "__nv_uint2float_rn", "__nv_uint2float_rz",
            "__nv_uint2float_ru", "__nv_uint2float_rd", "__nv_ll2float_rn",
            "__nv_ll2float_rz",   "__nv_ll2float_ru",   "__nv_ll2float_rd",
            "__nv_ull2float_rn",  "__nv_ull2float_rz",  "__nv_ull2float_ru",
            "__nv_ull2float_rd",
        };
        if (kIntToFloatConversions.contains(name)) {
            operation->opcode = OpCode::kConvert;
            for (const llvm::Use& argument : call.args()) {
                operation->operands.push_back(import_operand(*argument.get(), *state));
            }
            return true;
        }

        static const std::unordered_map<std::string, std::string> kCudaBuiltins = {
            {"__nv_fminf", "fmin"},
            {"__nv_fmaxf", "fmax"},
            {"__nv_sqrtf", "sqrt"},
            {"__nv_rsqrtf", "rsqrt"},
            {"__nv_fabsf", "fabs"},
            {"__nv_acosf", "acos"},
            {"__nv_expf", "exp"},
            {"__nv_fast_expf", "exp"},
            {"__nv_exp2f", "exp2"},
            {"__nv_exp10f", "exp10"},
            {"__nv_expm1f", "expm1"},
            {"__nv_logf", "log"},
            {"__nv_log2f", "log2"},
            {"__nv_log10f", "log10"},
            {"__nv_log1pf", "log1p"},
            {"__nv_sinf", "sin"},
            {"__nv_cosf", "cos"},
            {"__nv_tanf", "tan"},
            {"__nv_sinhf", "sinh"},
            {"__nv_coshf", "cosh"},
            {"__nv_tanhf", "tanh"},
            {"__nv_asinf", "asin"},
            {"__nv_atanf", "atan"},
            {"__nv_atan2f", "atan2"},
            {"__nv_asinhf", "asinh"},
            {"__nv_acoshf", "acosh"},
            {"__nv_atanhf", "atanh"},
            {"__nv_cbrtf", "cbrt"},
            {"__nv_erff", "erf"},
            {"__nv_erfcf", "erfc"},
            {"__nv_floorf", "floor"},
            {"__nv_ceilf", "ceil"},
            {"__nv_truncf", "trunc"},
            {"__nv_roundf", "round"},
            {"__nv_rintf", "rint"},
            {"__nv_powf", "pow"},
            {"__nv_hypotf", "hypot"},
            {"__nv_fmodf", "fmod"},
            {"__nv_copysignf", "copysign"},
            {"__nv_fdimf", "fdim"},
            {"__nv_remainderf", "remainder"},
            {"__nv_fmaf", "fma"},
            {"__nv_fma", "fma"},
            {"__nv_sqrt", "sqrt"},
            {"__nv_rsqrt", "rsqrt"},
            {"__nv_fmin", "fmin"},
            {"__nv_fmax", "fmax"},
            {"__nv_remainder", "remainder"},
            {"__nv_floor", "floor"},
            {"__nv_ceil", "ceil"},
            {"__nv_trunc", "trunc"},
            {"__nv_round", "round"},
            {"__nv_rint", "rint"},
            {"__nv_popc", "popcount"},
            {"__nv_clz", "clz"},
            {"__nv_abs", "__cumetal_signed_abs"},
            {"__nv_ffs", "__cumetal_ffs"},
            // Float -> integer. The rounding mode in the name applies to the
            // float *before* the cast, and the cast itself truncates; folding it
            // to a bare cast would silently drop the rounding, which is how
            // cvt.rni came to truncate. Each maps to a helper that lower_to_msl
            // expands into the matching round-then-cast.
            {"__nv_float2int_rn", "__cumetal_float2int_rne"},
            {"__nv_float2int_rz", "__cumetal_float2int_rtz"},
            {"__nv_float2int_ru", "__cumetal_float2int_rtp"},
            {"__nv_float2int_rd", "__cumetal_float2int_rtn"},
            {"__nv_float2uint_rn", "__cumetal_float2uint_rne"},
            {"__nv_float2uint_rz", "__cumetal_float2uint_rtz"},
            {"__nv_float2uint_ru", "__cumetal_float2uint_rtp"},
            {"__nv_float2uint_rd", "__cumetal_float2uint_rtn"},
            {"__nv_float2ll_rn", "__cumetal_float2int_rne"},
            {"__nv_float2ll_rz", "__cumetal_float2int_rtz"},
            {"__nv_float2ll_ru", "__cumetal_float2int_rtp"},
            {"__nv_float2ll_rd", "__cumetal_float2int_rtn"},
            {"__nv_float2ull_rn", "__cumetal_float2uint_rne"},
            {"__nv_float2ull_rz", "__cumetal_float2uint_rtz"},
            {"__nv_float2ull_ru", "__cumetal_float2uint_rtp"},
            {"__nv_float2ull_rd", "__cumetal_float2uint_rtn"},
        };
        const auto cuda_builtin = kCudaBuiltins.find(name);
        if (cuda_builtin != kCudaBuiltins.end()) {
            operation->opcode = OpCode::kCall;
            operation->attributes["callee"] = cuda_builtin->second;
            operation->attributes["builtin"] = "true";
            static const std::unordered_set<std::string> kExpandedMath = {
                "__nv_expm1f", "__nv_log1pf", "__nv_cbrtf", "__nv_erff",
                "__nv_erfcf", "__nv_hypotf", "__nv_remainderf",
            };
            if (kExpandedMath.contains(name)) {
                if (result.module.semantic_quality == SemanticQuality::kExact) {
                    result.module.semantic_quality = SemanticQuality::kToleranceBounded;
                }
                const std::string caveat =
                    "Metal-missing float math functions use numerically tested typed expansions";
                if (std::find(result.module.semantic_caveats.begin(),
                              result.module.semantic_caveats.end(), caveat) ==
                    result.module.semantic_caveats.end()) {
                    result.module.semantic_caveats.push_back(caveat);
                }
            }
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
                 name.find("llvm.acos.") == 0 || name.find("llvm.rint.") == 0) {
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
        if (!callee->isDeclaration()) {
            bool passed_constant_symbols = false;
            for (const ExternalGlobalInfo& info : external_globals) {
                if (!function_references_global(*callee, *info.global)) continue;
                const auto binding = state->external_globals.find(info.global);
                if (binding == state->external_globals.end()) {
                    return fail(&call, "missing transitive CUDA global binding for device helper");
                }
                if (info.constant && passed_constant_symbols) continue;
                operation->operands.push_back(binding->second.base);
                passed_constant_symbols |= info.constant;
            }
            if (function_uses_vprintf(*callee)) {
                if (!state->printf_buffer.has_value() ||
                    !state->printf_capacity.has_value()) {
                    return fail(&call,
                                "missing transitive printf binding for device helper");
                }
                operation->operands.push_back(*state->printf_buffer);
                operation->operands.push_back(*state->printf_capacity);
            }
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
        const Operand destination = import_pointer_operand(
            *copy.getDest(), state, output_block, location);
        const Operand source = import_pointer_operand(
            *copy.getSource(), state, output_block, location);

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

    bool import_memset(const llvm::MemSetInst& set, FunctionState* state,
                       BasicBlock* output_block) {
        const auto* length = llvm::dyn_cast<llvm::ConstantInt>(set.getLength());
        if (length == nullptr) {
            return fail(&set, "dynamic-length LLVM memset is unsupported");
        }
        if (set.isVolatile()) {
            return fail(&set, "volatile LLVM memset is unsupported");
        }
        const auto* byte = llvm::dyn_cast<llvm::ConstantInt>(set.getValue());
        if (byte == nullptr) {
            return fail(&set, "dynamic-byte LLVM memset is unsupported");
        }

        const std::uint64_t byte_count = length->getZExtValue();
        const std::uint64_t destination_alignment = set.getDestAlign().value().value();
        const std::uint64_t byte_value = byte->getZExtValue() & 0xffu;
        const SourceLocation location = import_location(set, fallback_source);
        const Operand destination = import_pointer_operand(
            *set.getDest(), state, output_block, location);

        auto offset_pointer = [&](std::uint64_t offset) {
            if (offset == 0) return destination;
            const ValueId value = builder.next_value();
            state->value_types[value] = destination.type;
            Operation pointer_offset;
            pointer_offset.opcode = OpCode::kPointerOffset;
            pointer_offset.results = {value};
            pointer_offset.result_types = {destination.type};
            pointer_offset.operands = {
                destination,
                Operand::immediate(std::to_string(offset), Type::integer(64)),
            };
            pointer_offset.attributes["offset_unit"] = "bytes";
            pointer_offset.location = location;
            output_block->operations.push_back(std::move(pointer_offset));
            return Operand::value_ref(value, destination.type);
        };

        for (std::uint64_t offset = 0; offset < byte_count;) {
            const bool word_aligned = destination_alignment >= 4 && offset % 4 == 0 &&
                                      byte_count - offset >= 4;
            const std::uint64_t width = word_aligned ? 4 : 1;
            const std::uint64_t stored =
                word_aligned ? byte_value * 0x01010101ull : byte_value;
            const Type value_type =
                Type::integer(static_cast<std::uint32_t>(width * 8));
            Operation store;
            store.opcode = OpCode::kStore;
            store.operands = {
                offset_pointer(offset),
                Operand::immediate(std::to_string(stored), value_type),
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
        auto float_frexp_truncation = [](const llvm::CallBase& call)
            -> const llvm::FPTruncInst* {
            const llvm::Function* callee = call.getCalledFunction();
            if (callee == nullptr || callee->getName() != "__nv_frexp" ||
                call.arg_size() != 2 || !call.hasOneUse()) {
                return nullptr;
            }
            const auto* extension =
                llvm::dyn_cast<llvm::FPExtInst>(call.getArgOperand(0));
            const auto* truncation =
                llvm::dyn_cast<llvm::FPTruncInst>(*call.user_begin());
            if (extension == nullptr || !extension->getSrcTy()->isFloatTy() ||
                !extension->getDestTy()->isDoubleTy() || truncation == nullptr ||
                !truncation->getDestTy()->isFloatTy() ||
                !llvm::isa<llvm::AllocaInst>(call.getArgOperand(1))) {
                return nullptr;
            }
            return truncation;
        };
        if (const auto* extension = llvm::dyn_cast<llvm::FPExtInst>(&instruction);
            extension != nullptr && extension->hasOneUse()) {
            const auto* call =
                llvm::dyn_cast<llvm::CallBase>(*extension->user_begin());
            if (call != nullptr && float_frexp_truncation(*call) != nullptr) {
                return true;
            }
        }
        if (const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction)) {
            if (is_empty_device_assert(call->getCalledFunction())) {
                // runtime/api/cuda_runtime.h deliberately supplies an empty
                // device assert overload because Metal has no trap-and-report
                // ABI. Erase that proven no-op before its diagnostic string
                // globals can leak into the generated MSL call expression.
                return true;
            }
            if (const llvm::FPTruncInst* truncation = float_frexp_truncation(*call)) {
                Operation frexp;
                frexp.opcode = OpCode::kCall;
                frexp.results = {state->values.at(truncation)};
                frexp.result_types = {Type::floating(32)};
                const auto* extension =
                    llvm::cast<llvm::FPExtInst>(call->getArgOperand(0));
                frexp.operands = {
                    import_operand(*extension->getOperand(0), *state),
                    import_operand(*call->getArgOperand(1), *state),
                };
                frexp.attributes["callee"] = "frexp";
                frexp.attributes["builtin"] = "true";
                frexp.location = import_location(instruction, fallback_source);
                output_block->operations.push_back(std::move(frexp));
                state->value_types[state->values.at(truncation)] = Type::floating(32);
                return true;
            }
        }
        if (const auto* truncation = llvm::dyn_cast<llvm::FPTruncInst>(&instruction)) {
            const auto* call = llvm::dyn_cast<llvm::CallBase>(truncation->getOperand(0));
            if (call != nullptr && float_frexp_truncation(*call) == truncation) {
                return true;
            }
        }
        if (const auto demoted = match_demotable_float_multiply(instruction)) {
            llvm::APFloat constant = demoted->constant->getValueAPF();
            bool loses_info = false;
            constant.convert(llvm::APFloat::IEEEsingle(),
                             llvm::APFloat::rmNearestTiesToEven, &loses_info);
            const llvm::Constant* float_constant =
                llvm::ConstantFP::get(input->getContext(), constant);
            Operation multiply;
            multiply.opcode = OpCode::kMul;
            multiply.results = {state->values.at(&instruction)};
            multiply.result_types = {Type::floating(32)};
            multiply.operands = {
                import_operand(*demoted->extension->getOperand(0), *state),
                Operand::immediate(constant_spelling(*float_constant),
                                   Type::floating(32)),
            };
            multiply.location = import_location(instruction, fallback_source);
            output_block->operations.push_back(std::move(multiply));
            result.module.semantic_quality = SemanticQuality::kPerformanceDegraded;
            const std::string caveat =
                "Metal lacks FP64; demoted isolated float-to-double multiply-to-float chains";
            if (std::find(result.module.semantic_caveats.begin(),
                          result.module.semantic_caveats.end(), caveat) ==
                result.module.semantic_caveats.end()) {
                result.module.semantic_caveats.push_back(caveat);
            }
            return true;
        }
        if (const auto* extension = llvm::dyn_cast<llvm::FPExtInst>(&instruction);
            extension != nullptr && extension->hasOneUse()) {
            const auto* multiply = llvm::dyn_cast<llvm::Instruction>(*extension->user_begin());
            if (multiply != nullptr && multiply->hasOneUse()) {
                const auto* truncation =
                    llvm::dyn_cast<llvm::Instruction>(*multiply->user_begin());
                if (truncation != nullptr &&
                    match_demotable_float_multiply(*truncation).has_value()) {
                    return true;
                }
            }
        }
        if (const auto* multiply = llvm::dyn_cast<llvm::BinaryOperator>(&instruction);
            multiply != nullptr && multiply->hasOneUse()) {
            const auto* truncation =
                llvm::dyn_cast<llvm::Instruction>(*multiply->user_begin());
            if (truncation != nullptr &&
                match_demotable_float_multiply(*truncation).has_value()) {
                return true;
            }
        }
        const bool uses_fp64 = contains_fp64(instruction.getType()) ||
            std::any_of(instruction.op_begin(), instruction.op_end(),
                        [](const llvm::Use& operand) {
                            return contains_fp64(operand->getType());
                        });
        Operation operation;
        operation.location = import_location(instruction, fallback_source);
        if (uses_fp64) {
            operation.attributes["fp64_mode"] =
                result.module.attributes.at("fp64_mode");
            result.module.semantic_quality = SemanticQuality::kSemanticEmulation;
            const std::string caveat =
                "FP64 uses raw binary64 storage with the selected software ALU mode";
            if (std::find(result.module.semantic_caveats.begin(),
                          result.module.semantic_caveats.end(), caveat) ==
                result.module.semantic_caveats.end()) {
                result.module.semantic_caveats.push_back(caveat);
            }
        }
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
            std::optional<ValueId> pointer_source;
            for (const Operand& operand : operation.operands) {
                if (operand.kind != OperandKind::kValue) continue;
                const auto source = state->integer_pointer_sources.find(operand.value);
                if (source == state->integer_pointer_sources.end()) continue;
                if (pointer_source.has_value() && *pointer_source != source->second) {
                    return fail(&instruction,
                                "integer arithmetic mixes distinct pointer identities");
                }
                pointer_source = source->second;
            }
            if (pointer_source.has_value()) {
                state->integer_pointer_sources[operation.results.front()] = *pointer_source;
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
            const Type aggregate_type = import_type(insert->getType());
            std::size_t leaf_count = 0;
            if (aggregate_type.kind != TypeKind::kAggregate ||
                !validate_aggregate_shape(aggregate_type, 0, &leaf_count)) {
                return fail(&instruction,
                            "LLVM insertvalue requires a bounded aggregate "
                            "(depth <= 8, width <= 16, leaves <= 64)");
            }
            const std::vector<unsigned> path(insert->idx_begin(), insert->idx_end());
            const Type* inserted_type = type_at_path(aggregate_type, path);
            if (path.empty() || inserted_type == nullptr ||
                *inserted_type != import_type(insert->getInsertedValueOperand()->getType())) {
                return fail(&instruction, "LLVM insertvalue path is invalid for its aggregate");
            }

            FunctionState::AggregateState aggregate_state{.type = aggregate_type};
            const llvm::Value* aggregate = insert->getAggregateOperand();
            const auto previous = state->aggregate_components.find(aggregate);
            if (previous != state->aggregate_components.end()) {
                if (previous->second.type != aggregate_type) {
                    return fail(&instruction,
                                "LLVM insertvalue base aggregate type changed");
                }
                aggregate_state = previous->second;
            } else if (!llvm::isa<llvm::PoisonValue>(aggregate) &&
                       !llvm::isa<llvm::UndefValue>(aggregate)) {
                if (!decompose_aggregate_operand(
                        import_operand(*aggregate, *state), aggregate_type, {},
                        &aggregate_state.leaves, state, output_block,
                        operation.location)) {
                    return fail(&instruction,
                                "LLVM insertvalue base aggregate cannot be decomposed");
                }
            }

            const llvm::Value* inserted = insert->getInsertedValueOperand();
            if (llvm::isa<llvm::PoisonValue>(inserted) ||
                llvm::isa<llvm::UndefValue>(inserted)) {
                return fail(&instruction,
                            "LLVM insertvalue cannot materialize poison or undef fields");
            }
            if (inserted_type->kind == TypeKind::kAggregate) {
                if (!decompose_aggregate_operand(
                        import_operand(*inserted, *state), *inserted_type, path,
                        &aggregate_state.leaves, state, output_block,
                        operation.location)) {
                    return fail(&instruction,
                                "LLVM insertvalue aggregate field cannot be decomposed");
                }
            } else {
                aggregate_state.leaves[path] = import_operand(*inserted, *state);
            }
            state->aggregate_components[insert] = aggregate_state;
            if (aggregate_state.leaves.size() != leaf_count) {
                return true;
            }

            if (!materialize_aggregate(
                    aggregate_type, {}, aggregate_state.leaves,
                    operation.results.front(), state, output_block,
                    operation.location)) {
                return fail(&instruction,
                            "LLVM insertvalue leaves an aggregate field uninitialized");
            }
            return true;
        } else if (const auto* extract = llvm::dyn_cast<llvm::ExtractValueInst>(&instruction)) {
            const Type aggregate_type = import_type(extract->getAggregateOperand()->getType());
            std::size_t leaf_count = 0;
            if (aggregate_type.kind != TypeKind::kAggregate ||
                !validate_aggregate_shape(aggregate_type, 0, &leaf_count)) {
                return fail(&instruction,
                            "LLVM extractvalue requires a bounded aggregate "
                            "(depth <= 8, width <= 16, leaves <= 64)");
            }
            const std::vector<unsigned> path(extract->idx_begin(), extract->idx_end());
            const Type* extracted_type = type_at_path(aggregate_type, path);
            if (path.empty() || extracted_type == nullptr ||
                *extracted_type != import_type(extract->getType())) {
                return fail(&instruction, "LLVM extractvalue path is invalid for its aggregate");
            }
            const auto components =
                state->aggregate_components.find(extract->getAggregateOperand());
            if (components != state->aggregate_components.end()) {
                if (extracted_type->kind == TypeKind::kAggregate) {
                    if (!materialize_aggregate(
                            *extracted_type, path, components->second.leaves,
                            operation.results.front(), state, output_block,
                            operation.location)) {
                        return fail(
                            &instruction,
                            "LLVM extractvalue reads an uninitialized aggregate element");
                    }
                    return true;
                }
                const auto component = components->second.leaves.find(path);
                if (component == components->second.leaves.end()) {
                    return fail(
                        &instruction,
                        "LLVM extractvalue reads an uninitialized aggregate element");
                }
                operation.opcode = OpCode::kConvert;
                operation.operands.push_back(component->second);
            } else {
                if (!emit_aggregate_extract(
                        import_operand(*extract->getAggregateOperand(), *state),
                        aggregate_type, path, operation.results.front(), state,
                        output_block, operation.location)) {
                    return fail(&instruction,
                                "LLVM extractvalue path cannot be lowered");
                }
                return true;
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
                if (operation.operands.front().kind == OperandKind::kValue) {
                    state->integer_pointer_sources[operation.results.front()] =
                        operation.operands.front().value;
                }
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
                bool has_pointer_source = false;
                if (operation.operands.front().kind == OperandKind::kValue) {
                    const auto provenance = state->integer_pointer_address_spaces.find(
                        operation.operands.front().value);
                    if (provenance != state->integer_pointer_address_spaces.end()) {
                        address_space = provenance->second;
                    }
                    const auto source = state->integer_pointer_sources.find(
                        operation.operands.front().value);
                    if (source != state->integer_pointer_sources.end()) {
                        has_pointer_source = true;
                        operation.attributes["pointer_source_value"] =
                            std::to_string(source->second);
                    }
                }
                if (!has_pointer_source) {
                    operation.attributes["pointer_integer_concrete"] = "true";
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
                } else if (llvm::isa<llvm::SIToFPInst>(cast) &&
                           !cast->getType()->isDoubleTy()) {
                    // CuMetal integer types are signless bit containers.  LLVM
                    // carries the signed interpretation on sitofp itself, so
                    // retain it for MSL instead of numerically converting an
                    // i8/i16 payload through uchar/ushort.
                    operation.attributes["signed_input"] = "true";
                } else if (cast->getOperand(0)->getType()->isFloatTy() &&
                           cast->getType()->isDoubleTy()) {
                    operation.attributes["fp64_conversion"] = "f32_to_f64";
                } else if (cast->getOperand(0)->getType()->isDoubleTy() &&
                           cast->getType()->isFloatTy()) {
                    operation.attributes["fp64_conversion"] = "f64_to_f32";
                } else if (cast->getType()->isDoubleTy() &&
                           cast->getOperand(0)->getType()->isIntegerTy()) {
                    operation.attributes["fp64_conversion"] =
                        llvm::isa<llvm::SIToFPInst>(cast) ? "signed_to_f64"
                                                        : "unsigned_to_f64";
                } else if (cast->getOperand(0)->getType()->isDoubleTy() &&
                           cast->getType()->isIntegerTy()) {
                    operation.attributes["fp64_conversion"] =
                        llvm::isa<llvm::FPToSIInst>(cast) ? "f64_to_signed"
                                                        : "f64_to_unsigned";
                }
            }
        } else if (const auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&instruction)) {
            constexpr unsigned kPointerBits = 64;
#if LLVM_VERSION_MAJOR >= 20
            llvm::SmallMapVector<llvm::Value*, llvm::APInt, 4> variable_offsets;
#else
            llvm::MapVector<llvm::Value*, llvm::APInt> variable_offsets;
#endif
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
            operation.operands.push_back(import_pointer_operand(
                *gep->getPointerOperand(), state, output_block, operation.location));
            operation.operands.push_back(std::move(byte_offset));
            if (operation.operands.front().type.is_pointer()) {
                operation.result_types.front() = operation.operands.front().type;
                state->value_types[operation.results.front()] = operation.operands.front().type;
            }
            operation.attributes["offset_unit"] = "bytes";
            std::optional<PointerProvenance> source_provenance;
            if (operation.operands.front().kind == OperandKind::kValue) {
                const auto source = state->output.pointer_provenance.find(
                    operation.operands.front().value);
                if (source != state->output.pointer_provenance.end()) {
                    source_provenance = source->second;
                }
            } else if (operation.operands.front().kind == OperandKind::kSymbol &&
                       operation.operands.front().type.is_pointer()) {
                source_provenance = PointerProvenance{
                    .base_kind = operation.operands.front().type.address_space ==
                                         AddressSpace::kThreadgroup
                                     ? PointerBaseKind::kDynamicThreadgroupMemory
                                     : PointerBaseKind::kAllocation,
                    .base_name = operation.operands.front().text,
                    .known_byte_offset = 0,
                };
            }
            if (source_provenance.has_value()) {
                PointerProvenance provenance = std::move(*source_provenance);
                if (provenance.memory_layout.empty()) {
                    llvm::raw_string_ostream layout(provenance.memory_layout);
                    gep->getSourceElementType()->print(layout);
                    layout.flush();
                    provenance.known_layout_offset = 0;
                }
                if (provenance.known_layout_offset.has_value()) {
                    provenance.known_layout_offset =
                        *provenance.known_layout_offset +
                        constant_offset.getSExtValue();
                }
                if (!has_dynamic_offset && provenance.known_byte_offset.has_value()) {
                    provenance.known_byte_offset =
                        *provenance.known_byte_offset + constant_offset.getSExtValue();
                } else {
                    provenance.known_byte_offset = std::nullopt;
                }
                state->output.pointer_provenance[operation.results.front()] =
                    std::move(provenance);
            }
        } else if (const auto* load = llvm::dyn_cast<llvm::LoadInst>(&instruction)) {
            operation.opcode = OpCode::kLoad;
            operation.operands.push_back(import_pointer_operand(
                *load->getPointerOperand(), state, output_block, operation.location));
            operation.attributes["alignment"] = std::to_string(load->getAlign().value());
        } else if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&instruction)) {
            const llvm::Value* store_base = nullptr;
            std::int64_t store_offset = 0;
            if (constant_pointer_base_and_offset(*store->getPointerOperand(),
                                                 &store_base, &store_offset) &&
                state->printf_argument_allocas.contains(
                    llvm::dyn_cast_or_null<llvm::AllocaInst>(store_base))) {
                // import_vprintf serializes this tuple directly into the
                // CuMetal ring. Retaining the original private-memory store is
                // redundant and can leave module-symbol operands referring to
                // their pre-import LLVM spelling.
                return true;
            }
            operation.opcode = OpCode::kStore;
            operation.operands.push_back(import_pointer_operand(
                *store->getPointerOperand(), state, output_block, operation.location));
            operation.operands.push_back(import_operand(*store->getValueOperand(), *state));
            operation.attributes["alignment"] = std::to_string(store->getAlign().value());
        } else if (const auto* set = llvm::dyn_cast<llvm::MemSetInst>(&instruction)) {
            return import_memset(*set, state, output_block);
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
            operation.operands.push_back(import_pointer_operand(
                *atomic->getPointerOperand(), state, output_block, operation.location));
            operation.operands.push_back(import_operand(*atomic->getValOperand(), *state));
            operation.attributes["atomic_op"] =
                llvm::AtomicRMWInst::getOperationName(atomic->getOperation()).str();
            if (atomic->getOperation() == llvm::AtomicRMWInst::Max ||
                atomic->getOperation() == llvm::AtomicRMWInst::Min) {
                operation.attributes["signed"] = "true";
            }
            operation.memory_scope = MemoryScope::kDevice;
            operation.memory_ordering = import_ordering(atomic->getOrdering());
            // LLVM's NVPTX frontend spells legacy CUDA atomic intrinsics as
            // seq_cst atomicrmw even though CUDA specifies these operations as
            // atomic but relaxed (they are not memory fences). Preserve CUDA's
            // source semantics and map the operation to Metal's relaxed order.
            if (operation.memory_ordering == MemoryOrdering::kSequentiallyConsistent &&
                llvm::Triple(input->getTargetTriple()).isNVPTX()) {
                operation.memory_ordering = MemoryOrdering::kRelaxed;
                operation.attributes["cuda_legacy_atomic"] = "true";
            }
        } else if (const auto* compare_exchange =
                       llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&instruction)) {
            const Type value_type = import_type(compare_exchange->getCompareOperand()->getType());
            if (value_type.kind != TypeKind::kInteger ||
                (value_type.bit_width != 32 && value_type.bit_width != 64)) {
                return fail(&instruction,
                            "typed Metal cmpxchg requires a 32-bit or lock-backed 64-bit integer payload");
            }
            const ValueId old_value = builder.next_value();
            state->value_types[old_value] = value_type;
            operation.results = {old_value};
            operation.result_types = {value_type};
            operation.opcode = OpCode::kAtomic;
            operation.operands = {
                import_pointer_operand(*compare_exchange->getPointerOperand(), state,
                                       output_block, operation.location),
                import_operand(*compare_exchange->getCompareOperand(), *state),
                import_operand(*compare_exchange->getNewValOperand(), *state),
            };
            operation.attributes["atomic_op"] = "cas";
            operation.memory_scope = MemoryScope::kDevice;
            operation.memory_ordering =
                import_ordering(compare_exchange->getSuccessOrdering());
            if (operation.memory_ordering == MemoryOrdering::kSequentiallyConsistent &&
                llvm::Triple(input->getTargetTriple()).isNVPTX()) {
                operation.memory_ordering = MemoryOrdering::kRelaxed;
                operation.attributes["cuda_legacy_atomic"] = "true";
            }
            output_block->operations.push_back(std::move(operation));

            const ValueId succeeded = builder.next_value();
            state->value_types[succeeded] = Type::predicate();
            Operation comparison;
            comparison.opcode = OpCode::kCompare;
            comparison.results = {succeeded};
            comparison.result_types = {Type::predicate()};
            comparison.operands = {
                Operand::value_ref(old_value, value_type),
                import_operand(*compare_exchange->getCompareOperand(), *state),
            };
            comparison.attributes["predicate"] = "eq";
            comparison.location = import_location(instruction, fallback_source);
            output_block->operations.push_back(std::move(comparison));
            FunctionState::AggregateState compare_exchange_state;
            compare_exchange_state.type = import_type(compare_exchange->getType());
            compare_exchange_state.leaves[{0}] =
                Operand::value_ref(old_value, value_type);
            compare_exchange_state.leaves[{1}] =
                Operand::value_ref(succeeded, Type::predicate());
            state->aggregate_components[compare_exchange] =
                std::move(compare_exchange_state);
            return true;
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
        for (const llvm::BasicBlock& block : function) {
            for (const llvm::Instruction& instruction : block) {
                const auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
                const llvm::Function* callee =
                    call == nullptr ? nullptr : call->getCalledFunction();
                if (callee == nullptr || callee->getName() != "vprintf" ||
                    call->arg_size() != 2) {
                    continue;
                }
                const llvm::Value* argument_base = nullptr;
                std::int64_t argument_offset = 0;
                if (constant_pointer_base_and_offset(*call->getArgOperand(1),
                                                     &argument_base,
                                                     &argument_offset) &&
                    argument_offset == 0) {
                    if (const auto* alloca =
                            llvm::dyn_cast<llvm::AllocaInst>(argument_base)) {
                        state.printf_argument_allocas.insert(alloca);
                    }
                }
            }
        }
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
        std::string rendered_module;
        llvm::raw_string_ostream rendered_stream(rendered_module);
        module->print(rendered_stream, nullptr);
        rendered_stream.flush();
        const bool uses_fp64 = rendered_module.find("double") != std::string::npos;
        if (uses_fp64 && options.fp64_mode != "fast48" &&
            options.fp64_mode != "wide48" && options.fp64_mode != "ieee64") {
            result.error = "typed NVVM FP64 mode must be fast48, wide48, or ieee64";
            return std::move(result);
        }
        result.module.attributes["fp64_mode"] = options.fp64_mode;
        result.module.attributes["target_triple"] =
            llvm::Triple(module->getTargetTriple()).str();

        for (const llvm::GlobalVariable& global : module->globals()) {
            const std::uint64_t global_size =
                module->getDataLayout().getTypeAllocSize(global.getValueType());
            const bool is_dynamic_threadgroup =
                global.getAddressSpace() == 3 && global.isDeclaration() &&
                global_size == 0;
            if (global.getAddressSpace() == 3 &&
                (global.hasInitializer() || is_dynamic_threadgroup) &&
                !global.getName().starts_with("llvm.")) {
                result.module.global_threadgroups.push_back({
                    .name = global.getName().str(),
                    .byte_size = global_size,
                    .alignment = static_cast<std::uint32_t>(
                        global.getAlign().has_value()
                            ? global.getAlign()->value()
                            : 1),
                    .is_dynamic = is_dynamic_threadgroup,
                });
                continue;
            }
            // Clang marks source-initialized CUDA constant tables as
            // externally_initialized too. A standalone precompiled metallib
            // has no registration record from which to populate a hidden
            // symbol buffer, so preserve non-zero read-only initializers as
            // embedded MSL constants. Zero-initialized __constant__ storage
            // and externally visible writable __device__ globals remain
            // registration-backed. Translation-unit-private writable globals
            // use the same persistent hidden-buffer ABI, with generated native
            // registration metadata owning their initializer and storage.
            const bool registration_backed_constant =
                global.getAddressSpace() == 4 &&
                global.getInitializer() != nullptr &&
                global.getInitializer()->isNullValue();
            const bool module_private_writable =
                global.getAddressSpace() == 1 && global.hasInitializer() &&
                !global.isConstant() && global.hasLocalLinkage();
            if (global.hasInitializer() &&
                ((global.isExternallyInitialized() &&
                  (global.getAddressSpace() == 1 || registration_backed_constant)) ||
                 module_private_writable) &&
                !global.getName().starts_with("llvm.")) {
                const std::uint32_t alignment = static_cast<std::uint32_t>(
                    global.getAlign().has_value() ? global.getAlign()->value() : 1);
                std::uint64_t constant_offset = 0;
                if (global.getAddressSpace() == 4) {
                    const std::uint64_t mask = alignment - 1;
                    external_constant_buffer_size =
                        (external_constant_buffer_size + mask) & ~mask;
                    constant_offset = external_constant_buffer_size;
                    external_constant_buffer_size += global_size;
                    if (external_constant_buffer_size > 64u * 1024u) {
                        result.error = "externally initialized CUDA constant storage exceeds 64 KiB";
                        return std::move(result);
                    }
                }
                external_globals.push_back({
                    .global = &global,
                    .byte_size = global_size,
                    .alignment = alignment,
                    .constant_offset = constant_offset,
                    .constant = global.getAddressSpace() == 4,
                });
                ExternalSymbol external_symbol{
                    .name = global.getName().str(),
                    .byte_size = global_size,
                    .alignment = alignment,
                    .constant_offset = constant_offset,
                    .constant = global.getAddressSpace() == 4,
                    .module_private = module_private_writable,
                };
                if (!global.getInitializer()->isNullValue()) {
                    external_symbol.initializer.assign(global_size, 0);
                    if (!write_constant_bytes(*global.getInitializer(), 0,
                                              &external_symbol.initializer,
                                              module->getDataLayout())) {
                        result.error =
                            "unsupported registration-backed global initializer: " +
                            global.getName().str();
                        return std::move(result);
                    }
                }
                result.module.external_symbols.push_back(
                    std::move(external_symbol));
                continue;
            }
            if (!global.isConstant() || !global.hasInitializer() ||
                global.getAddressSpace() != 4 ||
                global.getName().starts_with("llvm.")) {
                continue;
            }
            const std::uint64_t size = global_size;
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
                if (!function.isDeclaration() && !is_empty_device_assert(&function)) {
                    functions_to_import.push_back(&function);
                }
            }
        } else {
            const llvm::Function* root = module->getFunction(options.entry_name);
            if (root == nullptr || root->isDeclaration()) {
                result.error = "NVVM kernel not found: " + options.entry_name;
                return std::move(result);
            }
            if (!is_nvvm_kernel(*root)) {
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
                        if (callee != nullptr && !callee->isDeclaration() &&
                            !is_empty_device_assert(callee)) {
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
