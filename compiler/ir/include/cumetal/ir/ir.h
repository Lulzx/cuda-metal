#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cumetal::ir {

using ValueId = std::uint32_t;
using BlockId = std::uint32_t;

constexpr ValueId kInvalidValue = 0;
constexpr BlockId kInvalidBlock = 0;

enum class AddressSpace : std::uint8_t {
    kNone,
    kDevice,
    kConstant,
    kThreadgroup,
    kPrivate,
};

enum class MemoryScope : std::uint8_t {
    kNone,
    kSimdgroup,
    kThreadgroup,
    kDevice,
    kSystem,
};

enum class MemoryOrdering : std::uint8_t {
    kNone,
    kRelaxed,
    kAcquire,
    kRelease,
    kAcquireRelease,
    kSequentiallyConsistent,
};

enum class TypeKind : std::uint8_t {
    kVoid,
    kPredicate,
    kInteger,
    kFloat,
    kVector,
    kPointer,
    kAggregate,
};

struct Type {
    TypeKind kind = TypeKind::kVoid;
    std::uint32_t bit_width = 0;
    std::uint32_t lanes = 1;
    AddressSpace address_space = AddressSpace::kNone;
    std::vector<Type> elements;
    std::string name;

    static Type void_type();
    static Type predicate();
    static Type integer(std::uint32_t bits);
    static Type floating(std::uint32_t bits);
    static Type vector(Type element, std::uint32_t lanes);
    static Type pointer(Type pointee, AddressSpace address_space);
    static Type aggregate(std::vector<Type> elements, std::string name = {});

    [[nodiscard]] bool is_scalar() const;
    [[nodiscard]] bool is_pointer() const;
    [[nodiscard]] const Type* pointee() const;
    [[nodiscard]] std::string str() const;

    friend bool operator==(const Type&, const Type&) = default;
};

struct SourceLocation {
    std::string file;
    std::uint32_t line = 0;
    std::uint32_t column = 0;

    [[nodiscard]] std::string str() const;
};

enum class ArgumentKind : std::uint8_t {
    kPointer,
    kScalar,
    kAggregate,
    kDynamicThreadgroupMemory,
};

struct ArgumentDescriptor {
    std::string name;
    ArgumentKind kind = ArgumentKind::kScalar;
    Type type;
    std::uint32_t size = 0;
    std::uint32_t alignment = 0;
    AddressSpace address_space = AddressSpace::kNone;
    std::vector<std::uint32_t> binding_indices;
};

enum class BindingKind : std::uint8_t {
    kBuffer,
    kBytes,
    kThreadgroupMemory,
    kBuiltin,
};

struct BindingDescriptor {
    BindingKind kind = BindingKind::kBuffer;
    std::uint32_t binding_index = 0;
    std::uint32_t logical_argument_index = 0;
    Type type;
    std::uint32_t size = 0;
    std::uint32_t alignment = 0;
    std::optional<std::string> hidden_role;
};

struct KernelAbi {
    std::vector<ArgumentDescriptor> arguments;
    std::vector<BindingDescriptor> bindings;
    std::uint32_t static_threadgroup_memory = 0;
    std::uint32_t required_simd_width = 32;
    std::vector<std::string> required_metal_features;
};

enum class PointerBaseKind : std::uint8_t {
    kUnknown,
    kKernelArgument,
    kAllocation,
    kDynamicThreadgroupMemory,
    kIntegerRoundTrip,
};

struct PointerProvenance {
    PointerBaseKind base_kind = PointerBaseKind::kUnknown;
    std::string base_name;
    std::optional<std::int64_t> known_byte_offset;
    std::uint32_t alignment = 1;
    bool no_alias = false;
    bool escaped = false;
};

enum class SemanticQuality : std::uint8_t {
    kExact,
    kToleranceBounded,
    kSemanticEmulation,
    kPerformanceDegraded,
    kCpuFallback,
    kUnsupported,
};

enum class IrStage : std::uint8_t {
    kGpuSemantic,
    kMetalLegalized,
};

enum class OpCode : std::uint16_t {
    kInvalid,
    kConstant,
    kParameter,
    kAdd,
    kSub,
    kMul,
    kDiv,
    kRemainder,
    kFma,
    kNegate,
    kBitAnd,
    kBitOr,
    kBitXor,
    kShiftLeft,
    kShiftRight,
    kCompare,
    kSelect,
    kAggregateConstruct,
    kAggregateExtract,
    kConvert,
    kAddressSpaceCast,
    kAlloca,
    kPointerOffset,
    kLoad,
    kStore,
    kCall,
    kThreadId,
    kThreadgroupId,
    kThreadgroupSize,
    kGridSize,
    kLaneId,
    kSimdgroupSize,
    kBarrier,
    kFence,
    kAtomic,
    kShuffle,
    kBallot,
    kVote,
    kReduction,
    kBranch,
    kCondBranch,
    kReturn,
    kTrap,
    kMetalThreadPosition,
    kMetalThreadgroupPosition,
    kMetalThreadsPerThreadgroup,
    kMetalThreadgroupsPerGrid,
    kMetalLaneId,
    kMetalBarrier,
    kMetalFence,
    kMetalAtomic,
    kMetalShuffle,
    kMetalBallot,
    kMetalVote,
    kMetalReduction,
    kMetalBufferArgument,
    kMetalThreadgroupArgument,
};

enum class OperandKind : std::uint8_t {
    kValue,
    kImmediate,
    kSymbol,
};

struct Operand {
    OperandKind kind = OperandKind::kImmediate;
    ValueId value = kInvalidValue;
    Type type;
    std::string text;

    static Operand value_ref(ValueId value, Type type);
    static Operand immediate(std::string text, Type type);
    static Operand symbol(std::string text, Type type = Type::void_type());
};

struct Successor {
    BlockId block = kInvalidBlock;
    std::vector<ValueId> arguments;
};

struct Operation {
    OpCode opcode = OpCode::kInvalid;
    std::vector<ValueId> results;
    std::vector<Type> result_types;
    std::vector<Operand> operands;
    std::vector<Successor> successors;
    std::unordered_map<std::string, std::string> attributes;
    MemoryScope memory_scope = MemoryScope::kNone;
    MemoryOrdering memory_ordering = MemoryOrdering::kNone;
    SourceLocation location;

    [[nodiscard]] bool is_terminator() const;
};

struct BlockArgument {
    ValueId value = kInvalidValue;
    Type type;
    std::string name;
};

struct BasicBlock {
    BlockId id = kInvalidBlock;
    std::string name;
    std::vector<BlockArgument> arguments;
    std::vector<Operation> operations;
};

struct FunctionArgument {
    ValueId value = kInvalidValue;
    std::string name;
    Type type;
};

struct Function {
    std::string name;
    bool is_kernel = false;
    Type return_type = Type::void_type();
    std::vector<FunctionArgument> arguments;
    std::vector<BasicBlock> blocks;
    std::optional<KernelAbi> kernel_abi;
    std::unordered_map<ValueId, PointerProvenance> pointer_provenance;
    std::unordered_set<ValueId> generic_pointer_values;
    bool generic_pointer_return = false;

    [[nodiscard]] const BasicBlock* find_block(BlockId id) const;
    [[nodiscard]] BasicBlock* find_block(BlockId id);
};

struct GlobalConstant {
    std::string name;
    std::vector<std::uint8_t> bytes;
    std::uint32_t alignment = 1;
};

struct Module {
    std::string source_name;
    IrStage stage = IrStage::kGpuSemantic;
    SemanticQuality semantic_quality = SemanticQuality::kExact;
    std::vector<std::string> semantic_caveats;
    std::vector<GlobalConstant> global_constants;
    std::vector<Function> functions;
    std::unordered_map<std::string, std::string> attributes;
};

struct Diagnostic {
    SourceLocation location;
    std::string message;
};

struct VerifyResult {
    bool ok = false;
    std::vector<Diagnostic> diagnostics;
};

class Builder {
public:
    [[nodiscard]] ValueId next_value();
    [[nodiscard]] BlockId next_block();

private:
    ValueId next_value_ = 1;
    BlockId next_block_ = 1;
};

[[nodiscard]] VerifyResult verify(const Module& module);
[[nodiscard]] std::string print(const Module& module);
[[nodiscard]] std::string_view opcode_name(OpCode opcode);
[[nodiscard]] std::string_view address_space_name(AddressSpace address_space);
[[nodiscard]] std::string_view memory_scope_name(MemoryScope scope);
[[nodiscard]] std::string_view memory_ordering_name(MemoryOrdering ordering);
[[nodiscard]] bool is_gpu_semantic_opcode(OpCode opcode);
[[nodiscard]] bool is_metal_opcode(OpCode opcode);

}  // namespace cumetal::ir
