#include "cumetal/ir/ir.h"
#include "cumetal/metal/lower_to_msl.h"

#include <array>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {

using namespace cumetal;

bool expect(bool condition, const std::string& message) {
    if (!condition) std::cerr << "FAIL: " << message << '\n';
    return condition;
}

const std::string kPrefix = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry arithmetic_probe(
    .param .u64 .ptr .global output,
    .param .u64 offset,
    .param .u64 length
) {
    .local .align 8 .b8 scratch[64];
    .local .align 8 .b8 other[64];
    .shared .align 8 .b8 shared_scratch[64];
    .reg .b64 %rd<16>;
    .reg .pred %p1;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [offset];
    ld.param.u64 %rd2, [length];
    mov.u64 %rd3, scratch;
)ptx";

const std::string kCancellation = R"ptx(
    mov.u64 %rd5, 1;
    sub.s64 %rd6, %rd5, %rd4;
    sub.s64 %rd7, %rd6, %rd2;
    add.s64 %rd8, %rd3, %rd7;
    add.s64 %rd9, %rd8, 30;
)ptx";

const std::string kSuffix = R"ptx(
    st.global.u64 [%rd0], %rd9;
    ret;
}
)ptx";

std::string replace_once(std::string source, const std::string& from,
                         const std::string& to) {
    const auto at = source.find(from);
    if (at == std::string::npos) throw std::logic_error("missing test replacement: " + from);
    source.replace(at, from.size(), to);
    return source;
}

// Evaluate only the scalar dependency graph of a straight-line output store.
// This deliberately has no numeric representation of an allocation: a residual
// pointer dependency must fail instead of being assigned a convenient address.
// CFG and actual Apple-GPU numerical coverage live in the functional fixture.
std::optional<std::uint64_t> evaluate_scalar_store(const ir::Function& function,
                                                  std::uint64_t offset,
                                                  std::uint64_t length) {
    std::unordered_map<ir::ValueId, const ir::Operation*> definitions;
    std::unordered_map<ir::ValueId, std::uint64_t> values;
    for (const auto& argument : function.arguments) {
        if (argument.name == "offset") values[argument.value] = offset;
        if (argument.name == "length") values[argument.value] = length;
    }
    const ir::Operand* stored = nullptr;
    for (const auto& block : function.blocks) {
        for (const auto& operation : block.operations) {
            for (const auto result : operation.results) definitions[result] = &operation;
            if (operation.opcode == ir::OpCode::kStore) {
                if (stored != nullptr || operation.operands.size() != 2) return std::nullopt;
                stored = &operation.operands[1];
            }
        }
    }
    std::function<std::optional<std::uint64_t>(const ir::Operand&, unsigned)> evaluate;
    evaluate = [&](const ir::Operand& operand, unsigned depth) -> std::optional<std::uint64_t> {
        if (depth > 128 || operand.type != ir::Type::integer(64)) return std::nullopt;
        if (operand.kind == ir::OperandKind::kImmediate) {
            std::size_t consumed = 0;
            try {
                const auto value = std::stoull(operand.text, &consumed, 0);
                return consumed == operand.text.size() ? std::optional<std::uint64_t>(value)
                                                      : std::nullopt;
            } catch (...) { return std::nullopt; }
        }
        if (operand.kind != ir::OperandKind::kValue) return std::nullopt;
        if (const auto found = values.find(operand.value); found != values.end()) return found->second;
        const auto found = definitions.find(operand.value);
        if (found == definitions.end()) return std::nullopt;
        const auto& operation = *found->second;
        if (operation.operands.empty()) return std::nullopt;
        const auto left = evaluate(operation.operands.front(), depth + 1);
        if (!left) return std::nullopt;
        if (operation.opcode == ir::OpCode::kParameter ||
            operation.opcode == ir::OpCode::kConvert ||
            operation.opcode == ir::OpCode::kConstant)
            return operation.operands.size() == 1 ? left : std::nullopt;
        if (operation.operands.size() != 2) return std::nullopt;
        const auto right = evaluate(operation.operands[1], depth + 1);
        if (!right) return std::nullopt;
        if (operation.opcode == ir::OpCode::kAdd) return *left + *right;
        if (operation.opcode == ir::OpCode::kSub) return *left - *right;
        return std::nullopt;
    };
    return stored ? evaluate(*stored, 0) : std::nullopt;
}

bool accepts(const std::string& source, const std::string& label, bool numerical = false) {
    const auto compiled = metal::compile_ptx_to_msl(source);
    bool ok = expect(compiled.ok, label + ": " + compiled.error);
    if (!compiled.ok) return false;
    ok &= expect(ir::verify(compiled.gpu_ir).ok, label + " GPU IR verifies");
    ok &= expect(ir::verify(compiled.metal_ir).ok, label + " Metal IR verifies");
    bool found_store = false;
    for (const auto& function : compiled.gpu_ir.functions) {
        if (!function.is_kernel) continue;
        for (const auto& block : function.blocks)
            for (const auto& operation : block.operations)
                if (operation.opcode == ir::OpCode::kStore) {
                    found_store = true;
                    ok &= expect(operation.operands.size() == 2 &&
                                     operation.operands[1].type == ir::Type::integer(64),
                                 label + " stores the scalar difference, not address bits");
                }
        if (!numerical) continue;
        constexpr auto max = std::numeric_limits<std::uint64_t>::max();
        for (const auto inputs : std::array<std::array<std::uint64_t, 2>, 7>{{
                 {0, 0}, {7, 9}, {31, 1}, {32, 0}, {max, 0}, {0, max},
                 {0x8000000000000000ULL, 0x8000000000000001ULL}}}) {
            const auto actual = evaluate_scalar_store(function, inputs[0], inputs[1]);
            const auto expected = std::uint64_t{31} - inputs[0] - inputs[1];
            ok &= expect(actual && *actual == expected,
                         label + " computes 31 - offset - length modulo 2^64 for " +
                             std::to_string(inputs[0]) + ", " + std::to_string(inputs[1]));
        }
    }
    return expect(found_store, label + " retains its output store") && ok;
}

bool rejects(const std::string& source, const std::string& label) {
    const auto compiled = metal::compile_ptx_to_msl(source);
    // Require a pointer/provenance failure, so an unrelated malformed fixture
    // cannot silently satisfy a negative regression.
    return expect(!compiled.ok &&
                      (compiled.error.find("pointer") != std::string::npos ||
                       compiled.error.find("address") != std::string::npos ||
                       compiled.error.find("provenance") != std::string::npos),
                  label + " must reject unresolved address arithmetic: " + compiled.error);
}

}  // namespace

int main() {
    bool ok = true;
    const auto cursor = std::string("    add.u64 %rd4, %rd3, %rd1;\n");
    const auto source = kPrefix + cursor + kCancellation + kSuffix;
    ok &= accepts(source, "same allocation cancels across four integer instructions", true);
    ok &= accepts(replace_once(source, "add.u64 %rd4, %rd3, %rd1;",
                                      "add.u64 %rd4, %rd1, %rd3;"),
                  "commuted cursor addition", true);
    const auto in_place = replace_once(source, "add.u64 %rd4, %rd3, %rd1;", R"ptx(
    add.u64 %rd1, %rd3, %rd1;
    mov.u64 %rd4, %rd1;
)ptx");
    ok &= accepts(in_place, "in-place address add captures its scalar operand before overwrite", true);
    const auto retained_memory = replace_once(in_place, "mov.u64 %rd5, 1;",
        "st.volatile.local.u64 [%rd4], 7;\nmov.u64 %rd5, 1;");
    const auto stored = metal::compile_ptx_to_msl(retained_memory);
    ok &= expect(stored.ok, "an actual cursor store retains the original address add: " + stored.error);
    if (stored.ok) {
        ok &= expect(ir::verify(stored.gpu_ir).ok && ir::verify(stored.metal_ir).ok,
                     "retained cursor store has valid GPU and Metal IR");
        unsigned local_stores = 0, output_stores = 0;
        for (const auto& function : stored.gpu_ir.functions)
            for (const auto& block : function.blocks)
                for (const auto& operation : block.operations) {
                    if (operation.opcode != ir::OpCode::kStore || operation.operands.empty()) continue;
                    local_stores += operation.operands[0].type.address_space == ir::AddressSpace::kPrivate;
                    output_stores += operation.operands[0].type.address_space == ir::AddressSpace::kDevice &&
                        operation.operands.size() == 2 && operation.operands[1].type == ir::Type::integer(64);
                }
        ok &= expect(local_stores == 1 && output_stores == 1,
                     "address cancellation preserves both the private cursor store and scalar output");
    }
    ok &= accepts(replace_once(source, "add.s64 %rd8, %rd3, %rd7;",
                                      "add.s64 %rd8, %rd7, %rd3;"),
                  "commuted cancellation addition", true);
    ok &= accepts(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;",
                                      "mov.b64 %rd10, %rd6;\nsub.s64 %rd7, %rd10, %rd2;"),
                  "copy of partially cancelled expression", true);
    ok &= accepts(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;",
                                      "mov.u64 %rd4, 7;\nsub.s64 %rd7, %rd6, %rd2;"),
                  "overwritten register cannot change an earlier reaching definition", true);
    ok &= accepts(replace_once(source, "mov.u64 %rd3, scratch;",
                                      "mov.u64 %rd10, scratch;\nmov.u64 %rd3, %rd10;"),
                  "base allocation copied through another register", true);
    ok &= accepts(replace_once(source, "mov.u64 %rd3, scratch;",
                                      "mov.u64 %rd10, scratch;\ncvta.local.u64 %rd3, %rd10;"),
                  "base and cursor share one converted pointer representation", true);
    ok &= accepts(replace_once(source, "mov.u64 %rd3, scratch;",
                                      "mov.u64 %rd3, shared_scratch;"),
                  "same shared allocation cancels without defaulting to device", true);
    ok &= accepts(replace_once(source, ".reg .pred %p1;",
        ".reg .pred %p1;\n.reg .b32 %__cumetal_address_offset_0;"),
        "generated offsets avoid an unused scalar register declaration", true);
    ok &= accepts(replace_once(source, ".reg .pred %p1;",
        ".reg .pred %p1;\n.reg .b32 %__cumetal_address_offset_<64>;"),
        "generated offsets avoid every member of an unused register range", true);
    ok &= accepts(kPrefix + cursor + R"ptx(
    sub.s64 %rd6, %rd3, %rd4;
    add.s64 %rd7, %rd6, 31;
    sub.s64 %rd9, %rd7, %rd2;
)ptx" + kSuffix, "direct same-base pointer difference yields a modular scalar", true);
    const auto rebased_offset = kPrefix + cursor + kCancellation + R"ptx(
    add.u64 %rd10, %rd3, %rd9;
    st.volatile.local.u64 [%rd10], %rd2;
)ptx" + kSuffix;
    ok &= accepts(rebased_offset,
                  "cancelled scalar remains valid as pointer offset and separately stored data");
    const auto helper_prefix = replace_once(kPrefix, ".visible .entry", R"ptx(
.func (.param .b64 retval) pointer_read(.param .u64 .ptr input) {
    .reg .b64 %address, %value;
    ld.param.u64 %address, [input];
    ld.u64 %value, [%address];
    st.param.b64 [retval], %value;
    ret;
}
.visible .entry)ptx");
    const std::string pointer_call = R"ptx(
    .param .b64 argument;
    .param .b64 returned;
    st.param.b64 [argument], %rd9;
    call.uni (returned), pointer_read, (argument);
    ld.param.b64 %rd11, [returned];
    st.global.u64 [%rd0+8], %rd11;
)ptx";
    ok &= rejects(helper_prefix + cursor + kCancellation + pointer_call + kSuffix,
                  "a pointer helper cannot reinterpret a cancelled scalar argument");
    const auto rebased_call = replace_once(pointer_call, "st.param.b64 [argument], %rd9;", R"ptx(
    add.u64 %rd10, %rd3, %rd9;
    st.local.u64 [%rd10], 7;
    st.param.b64 [argument], %rd10;
)ptx");
    ok &= accepts(helper_prefix + cursor + kCancellation + rebased_call + kSuffix,
                  "a helper accepts the proven pointer after adding back its base");

    const std::string loop = R"ptx(
    mov.u64 %rd4, %rd3;
    mov.u64 %rd10, 0;
    setp.eq.u64 %p1, %rd1, 0;
    @%p1 bra DONE;
LOOP:
    add.u64 %rd4, %rd4, 1;
    add.u64 %rd10, %rd10, 1;
    setp.lt.u64 %p1, %rd10, %rd1;
    @%p1 bra LOOP;
DONE:
)ptx";
    ok &= accepts(kPrefix + loop + kCancellation + kSuffix,
                  "loop-carried cursor retains its allocation and scalar offset");
    const std::string joined = R"ptx(
    setp.eq.u64 %p1, %rd1, 0;
    @%p1 bra ALTERNATE;
    add.u64 %rd4, %rd3, %rd1;
    bra JOIN;
ALTERNATE:
    add.u64 %rd4, %rd3, 7;
JOIN:
)ptx";
    ok &= accepts(kPrefix + joined + kCancellation + kSuffix,
                  "same-allocation branch join retains edge-specific offsets");

    ok &= rejects(replace_once(source, "add.s64 %rd8, %rd3, %rd7;",
                                      "mov.u64 %rd8, %rd7;"),
                  "raw integer minus pointer never reaches a cancelling base");
    ok &= rejects(kPrefix + cursor + kCancellation + R"ptx(
    ld.local.u64 %rd11, [%rd9];
    st.global.u64 [%rd0+8], %rd11;
)ptx" + kSuffix, "a cancelled scalar cannot become an unproven local address");
    ok &= rejects(kPrefix + cursor + kCancellation + R"ptx(
    mov.u64 %rd10, %rd9;
    ld.local.u64 %rd11, [%rd10];
    st.global.u64 [%rd0+8], %rd11;
)ptx" + kSuffix, "copying a cancelled scalar does not restore pointer provenance");
    ok &= rejects(kPrefix + cursor + kCancellation + R"ptx(
    mov.u64 %rd10, other;
    st.local.u64 [%rd10], %rd9;
    ld.local.u64 %rd11, [%rd10];
    ld.local.u64 %rd12, [%rd11];
    st.global.u64 [%rd0+8], %rd12;
)ptx" + kSuffix, "a scalar spill cannot reuse a provisional pointer-load proof after cancellation");
    ok &= rejects(kPrefix + cursor + kCancellation + R"ptx(
    setp.eq.u64 %p1, %rd2, 0;
    @%p1 bra LITERAL;
    mov.u64 %rd10, %rd9;
    bra SCALAR_JOIN;
LITERAL:
    mov.u64 %rd10, 7;
SCALAR_JOIN:
    ld.local.u64 %rd11, [%rd10];
    st.global.u64 [%rd0+8], %rd11;
)ptx" + kSuffix, "a scalar phi does not restore cancelled pointer provenance");
    ok &= rejects(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;",
                                      "st.global.u64 [%rd0+8], %rd6;\nsub.s64 %rd7, %rd6, %rd2;"),
                  "observing a partially cancelled expression in memory");
    auto narrowed = replace_once(source, ".reg .pred %p1;",
                                ".reg .pred %p1;\n.reg .b32 %r0;");
    narrowed = replace_once(narrowed, "sub.s64 %rd7, %rd6, %rd2;", R"ptx(
    cvt.u32.u64 %r0, %rd6;
    cvt.u64.u32 %rd6, %r0;
    sub.s64 %rd7, %rd6, %rd2;
)ptx");
    ok &= rejects(narrowed, "narrowing a partial expression destroys 64-bit cancellation");
    ok &= rejects(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;", R"ptx(
    setp.eq.u64 %p1, %rd6, 0;
    @%p1 bra OBSERVED;
    st.global.u64 [%rd0+8], 9;
OBSERVED:
    sub.s64 %rd7, %rd6, %rd2;
)ptx"), "observing a partially cancelled expression in a predicate");
    ok &= rejects(replace_once(source, "add.s64 %rd8, %rd3, %rd7;",
                                      "mov.u64 %rd10, other;\nadd.s64 %rd8, %rd10, %rd7;"),
                  "unrelated allocations in the same address space do not cancel");
    ok &= rejects(replace_once(source, "add.s64 %rd8, %rd3, %rd7;",
                                      "mov.u64 %rd10, shared_scratch;\nadd.s64 %rd8, %rd10, %rd7;"),
                  "allocations in different address spaces do not cancel");
    ok &= rejects(replace_once(source, "add.u64 %rd4, %rd3, %rd1;",
                                      "cvta.local.u64 %rd10, %rd3;\nadd.u64 %rd4, %rd10, %rd1;"),
                  "local and generic representations do not cancel across cvta");
    ok &= rejects(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;",
                                      "mov.u64 %rd3, other;\nsub.s64 %rd7, %rd6, %rd2;"),
                  "overwriting a base register invalidates name-based cancellation");
    ok &= rejects(replace_once(source, "sub.s64 %rd7, %rd6, %rd2;", R"ptx(
    mov.u64 %rd10, other;
    setp.eq.u64 %p1, %rd2, 0;
    @%p1 mov.u64 %rd3, %rd10;
    sub.s64 %rd7, %rd6, %rd2;
)ptx"), "a predicated base overwrite cannot borrow the other edge's provenance");
    ok &= rejects(replace_once(kPrefix + joined + kCancellation + kSuffix,
                              "add.u64 %rd4, %rd3, 7;",
                              "mov.u64 %rd10, other;\nadd.u64 %rd4, %rd10, 7;"),
                  "one unrelated incoming branch prevents cancellation at a join");

    if (!ok) return 1;
    std::cout << "PTX address cancellation tests passed\n";
    return 0;
}
