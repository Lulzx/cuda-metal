#include "cumetal/ptx/parser.h"

#include <cstdio>
#include <string>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    // Explicit scalar use isolates store provenance from width-based defaults.
    const auto stores = cumetal::ptx::parse_ptx(R"PTX(
.version 7.0
.target sm_80
.visible .entry stores(.param .u64 output, .param .u64 length) {
ld.param.u64 %rd1, [output];
ld.param.u64 %rd2, [length];
mul.lo.u64 %rd3, %rd2, 3;
st.global.u64 [%rd1], %rd3;
st.global.u64 [%rd1+8], %rd2;
st.global.u8 [%rd1+16], 7;
ret;
}
)PTX");
    if (!expect(stores.ok && stores.module.entries.size() == 1 &&
                stores.module.entries[0].params[0].is_pointer &&
                !stores.module.entries[0].params[1].is_pointer,
                "stores preserve address provenance without promoting stored scalars")) return 1;

    const auto classification = cumetal::ptx::parse_ptx(R"PTX(
.version 7.0
.target sm_80
.visible .entry classify(.param .u64 unused, .param .u64 length,
                        .param .u64 output, .param .u64 .ptr explicit_pointer) {
ld.param.u64 %rd1, [output];
ld.param.u64 %rd2, [length];
sub.u64 %rd3, 31, %rd2;
st.global.u64 [%rd1], %rd3;
ret;
}
)PTX");
    if (!expect(classification.ok && classification.module.entries.size() == 1,
                "parse scalar and pointer classification")) return 1;
    const auto& parameters = classification.module.entries[0].params;
    if (!expect(!parameters[0].is_pointer && !parameters[1].is_pointer &&
                parameters[2].is_pointer && parameters[3].is_pointer,
                "width alone is scalar; actual address uses and explicit annotations remain pointers")) return 1;

    const std::string sample_ptx = R"PTX(
// vector ops
.version 8.0
.target sm_90
.address_size 64

.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2
)
{
    .reg .pred %p1;
    ld.param.u64 %rd1, [vector_add_param_0];
    ld.param.u64 %rd2, [vector_add_param_1];
    add.s32 %r1, %r2, %r3;
    @%p1 bra L_done;
L_done:
    ret;
}

/* second entry */
.visible .entry scale(
    .param .u64 scale_param_0,
    .param .f32 scale_param_1
)
{
    mul.f32 %f2, %f0, %f1;
    ret;
}
)PTX";

    const auto parsed = cumetal::ptx::parse_ptx(sample_ptx);
    if (!expect(parsed.ok, "parse PTX sample")) {
        return 1;
    }
    if (!expect(parsed.module.version_major == 8 && parsed.module.version_minor == 0,
                "parse .version 8.0")) {
        return 1;
    }
    if (!expect(parsed.module.target == "sm_90", "parse .target sm_90")) {
        return 1;
    }
    if (!expect(parsed.module.entries.size() == 2, "parse two entries")) {
        return 1;
    }

    if (!expect(parsed.module.entries[0].name == "vector_add", "first entry name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params.size() == 3, "vector_add parameter count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params[0].type == ".u64", "vector_add param0 type")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].params[0].name == "vector_add_param_0",
                "vector_add param0 name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions.size() == 6,
                "vector_add instruction count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[0].opcode == "ld.param.u64",
                "first instruction opcode")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[0].supported, "first instruction supported")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[3].predicate == "@%p1",
                "predicate captured for branch")) {
        return 1;
    }
    if (!expect(parsed.module.entries[0].instructions[3].opcode == "bra",
                "branch opcode captured")) {
        return 1;
    }

    if (!expect(parsed.module.entries[1].name == "scale", "second entry name")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].params.size() == 2, "scale parameter count")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].params[1].type == ".f32", "scale param1 type")) {
        return 1;
    }
    if (!expect(parsed.module.entries[1].instructions.size() == 2, "scale instruction count")) {
        return 1;
    }

    const auto no_entry = cumetal::ptx::parse_ptx(".version 8.0\n.target sm_90\n");
    if (!expect(!no_entry.ok, "reject module without entry")) {
        return 1;
    }
    if (!expect(no_entry.error.find("no .entry") != std::string::npos, "error message for missing entry")) {
        return 1;
    }

    const std::string debug_metadata_ptx = R"PTX(
.version 8.0
.target sm_90
.file 1 "debug_metadata.cu"
.visible .entry debug_metadata()
{
    .reg .b64 %rd1;
    .loc 1 7 5
    mov.u64 %rd1, 42;
    .pragma "nounroll"
    add.u64 %rd1, %rd1, 1;
    ret;
}
)PTX";
    const auto debug_metadata = cumetal::ptx::parse_ptx(debug_metadata_ptx);
    if (!expect(debug_metadata.ok, "parse debug metadata PTX") ||
        !expect(debug_metadata.module.entries[0].instructions.size() == 3,
                "single-line metadata does not consume following instruction") ||
        !expect(debug_metadata.module.entries[0].instructions[0].opcode == "mov.u64",
                "instruction after .loc is retained") ||
        !expect(debug_metadata.module.entries[0].instructions[1].opcode == "add.u64",
                "instruction after .pragma is retained")) {
        return 1;
    }

    const std::string indexed_branch_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry indexed_branch()
{
    .reg .b32 %r1;
    $L_table: .branchtargets
        $L_zero,
        $L_one;
    brx.idx %r1, $L_table;
$L_zero:
    ret;
$L_one:
    ret;
}
)PTX";
    const auto indexed_branch = cumetal::ptx::parse_ptx(indexed_branch_ptx);
    if (!expect(indexed_branch.ok, "parse indexed branch target table") ||
        !expect(indexed_branch.module.entries[0].instructions.size() == 6,
                "indexed branch instruction count") ||
        !expect(indexed_branch.module.entries[0].instructions[0].opcode ==
                    "ptx.branchtargets",
                "branch-target table pseudo opcode") ||
        !expect(indexed_branch.module.entries[0].instructions[0].operands.size() == 3,
                "branch-target table operands") ||
        !expect(indexed_branch.module.entries[0].instructions[1].opcode == "brx.idx" &&
                    indexed_branch.module.entries[0].instructions[1].supported,
                "indexed branch opcode supported")) {
        return 1;
    }

    const std::string unsupported_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry unsupported(
    .param .u64 p0
)
{
    foo.bar %r1, %r2;
    ret;
}
)PTX";

    const auto tolerant = cumetal::ptx::parse_ptx(unsupported_ptx);
    if (!expect(tolerant.ok, "tolerant parse accepts unknown opcode")) {
        return 1;
    }
    if (!expect(!tolerant.warnings.empty(), "tolerant parse emits warning")) {
        return 1;
    }
    if (!expect(tolerant.module.entries[0].instructions.size() == 2,
                "unsupported entry instruction count")) {
        return 1;
    }
    if (!expect(!tolerant.module.entries[0].instructions[0].supported,
                "unknown opcode marked unsupported")) {
        return 1;
    }

    cumetal::ptx::ParseOptions strict_options;
    strict_options.strict = true;
    const auto strict = cumetal::ptx::parse_ptx(unsupported_ptx, strict_options);
    if (!expect(!strict.ok, "strict parse rejects unknown opcode")) {
        return 1;
    }
    if (!expect(strict.error.find("unsupported opcode") != std::string::npos,
                "strict parse error mentions unsupported opcode")) {
        return 1;
    }

    // Test: .u64 parameter used in mul.lo.u64 is inferred as scalar (non-pointer)
    const std::string scalar_mul_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry step_mul_test(
    .param .u64 data_ptr,
    .param .u64 step_num
)
{
    ld.param.u64 %rd0, [data_ptr];
    ld.param.u64 %rd1, [step_num];
    ld.global.f32 %f0, [%rd0];
    mul.lo.u64 %rd2, %rd1, 4;
    ret;
}
)PTX";

    const auto scalar_mul = cumetal::ptx::parse_ptx(scalar_mul_ptx);
    if (!expect(scalar_mul.ok, "parse scalar_mul_test")) {
        return 1;
    }
    if (!expect(scalar_mul.module.entries.size() == 1, "scalar_mul_test entry count")) {
        return 1;
    }
    if (!expect(scalar_mul.module.entries[0].params.size() == 2,
                "scalar_mul_test param count")) {
        return 1;
    }
    if (!expect(scalar_mul.module.entries[0].params[0].is_pointer,
                "data_ptr inferred as pointer")) {
        return 1;
    }
    if (!expect(!scalar_mul.module.entries[0].params[1].is_pointer,
                "step_num inferred as scalar (not pointer) via mul.lo.u64")) {
        return 1;
    }

    // Test: .u64 parameter used in div.u64 is inferred as scalar (non-pointer)
    const std::string scalar_div_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry step_div_test(
    .param .u64 buf_ptr,
    .param .u64 divisor
)
{
    ld.param.u64 %rd0, [buf_ptr];
    ld.param.u64 %rd1, [divisor];
    ld.global.u32 %r0, [%rd0];
    div.u64 %rd2, %rd1, 2;
    ret;
}
)PTX";

    const auto scalar_div = cumetal::ptx::parse_ptx(scalar_div_ptx);
    if (!expect(scalar_div.ok, "parse scalar_div_test")) {
        return 1;
    }
    if (!expect(scalar_div.module.entries[0].params[0].is_pointer,
                "buf_ptr inferred as pointer in div test")) {
        return 1;
    }
    if (!expect(!scalar_div.module.entries[0].params[1].is_pointer,
                "divisor inferred as scalar (not pointer) via div.u64")) {
        return 1;
    }

    // Test: mixed .u64 params — one pointer, one scalar (models the adamw step_num case)
    const std::string mixed_u64_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry mixed_u64_test(
    .param .u64 weights,
    .param .u64 gradients,
    .param .u64 step_count
)
{
    ld.param.u64 %rd0, [weights];
    ld.param.u64 %rd1, [gradients];
    ld.param.u64 %rd2, [step_count];
    ld.global.f32 %f0, [%rd0];
    ld.global.f32 %f1, [%rd1];
    mul.lo.u64 %rd3, %rd2, %rd2;
    ret;
}
)PTX";

    const auto mixed_u64 = cumetal::ptx::parse_ptx(mixed_u64_ptx);
    if (!expect(mixed_u64.ok, "parse mixed_u64_test")) {
        return 1;
    }
    if (!expect(mixed_u64.module.entries[0].params.size() == 3,
                "mixed_u64_test param count")) {
        return 1;
    }
    if (!expect(mixed_u64.module.entries[0].params[0].is_pointer,
                "weights inferred as pointer")) {
        return 1;
    }
    if (!expect(mixed_u64.module.entries[0].params[1].is_pointer,
                "gradients inferred as pointer")) {
        return 1;
    }
    if (!expect(!mixed_u64.module.entries[0].params[2].is_pointer,
                "step_count inferred as scalar via mul.lo.u64 self-multiply")) {
        return 1;
    }

    // ── Test: targeted diagnostics for cluster/TMA/FP8 unsupported opcodes ──
    const std::string cluster_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry cluster_test(
    .param .u64 p0
)
{
    cluster.sync.aligned;
    ret;
}
)PTX";

    const auto cluster_parse = cumetal::ptx::parse_ptx(cluster_ptx);
    if (!expect(cluster_parse.ok, "cluster opcode parses tolerantly")) return 1;
    if (!expect(!cluster_parse.warnings.empty(), "cluster opcode emits warning")) return 1;
    if (!expect(cluster_parse.warnings[0].find("cluster") != std::string::npos &&
                    cluster_parse.warnings[0].find("Metal equivalent") != std::string::npos,
                "cluster warning mentions Metal equivalent gap"))
        return 1;

    const std::string tma_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry tma_test(
    .param .u64 p0
)
{
    cp.async.bulk.tensor.1d.global.shared [p0], [p0], 16;
    ret;
}
)PTX";

    const auto tma_parse = cumetal::ptx::parse_ptx(tma_ptx);
    if (!expect(tma_parse.ok, "TMA opcode parses tolerantly")) return 1;
    if (!expect(!tma_parse.warnings.empty(), "TMA opcode emits warning")) return 1;
    if (!expect(tma_parse.warnings[0].find("TMA") != std::string::npos ||
                    tma_parse.warnings[0].find("Tensor Memory") != std::string::npos,
                "TMA warning identifies TMA opcode"))
        return 1;

    const std::string extern_before_entry_ptx = R"PTX(
.version 7.0
.target sm_80
.weak .entry allocation_kernel(
    .param .u64 output
);
.extern .func (.param .b64 retval) _Znam(
    .param .b64 size
);
.visible .entry allocation_kernel(
    .param .u64 output
)
{
    mov.u32 %r1, %tid.x;
    ret;
}
.func (.param .b32 retval) helper(
    .param .b32 value
)
{
    ld.param.b32 %r1, [value];
    st.param.b32 [retval], %r1;
    ret;
}
)PTX";
    const auto extern_before_entry = cumetal::ptx::parse_ptx(extern_before_entry_ptx);
    if (!expect(extern_before_entry.ok, "extern function before entry parses") ||
        !expect(extern_before_entry.module.entries.size() == 1 &&
                    extern_before_entry.module.entries[0].params.size() == 1,
                "forward entry declaration does not absorb later ABI parameters") ||
        !expect(extern_before_entry.module.functions.size() == 1,
                "extern function declaration is not mistaken for a definition") ||
        !expect(extern_before_entry.module.functions[0].name == "helper",
                "real device function definition remains visible")) {
        return 1;
    }

    const std::string aggregate_param_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry aggregate_param(
    .param .align 4 .b8 aggregate_param_0[12]
) {
    .reg .b32 %r1;
    ld.param.b32 %r1, [aggregate_param_0+8];
    ret;
}
)PTX";
    const auto aggregate_param = cumetal::ptx::parse_ptx(aggregate_param_ptx);
    if (!expect(aggregate_param.ok &&
                    aggregate_param.module.entries.size() == 1 &&
                    aggregate_param.module.entries[0].params.size() == 1 &&
                    aggregate_param.module.entries[0].params[0].name ==
                        "aggregate_param_0" &&
                    aggregate_param.module.entries[0].params[0].byte_size == 12 &&
                    aggregate_param.module.entries[0].params[0].alignment == 4,
                "aggregate PTX parameters retain base name, byte size, and alignment")) {
        return 1;
    }

    // A bare register declared inside a `{ ... }` block is renamed only within
    // that block. cuda-samples kernels declare `.extern .shared .b8 tmp[]` AND
    // `{ .reg .b32 tmp; mov.b64 {tmp, %r2}, %rd1; }`; a rename that leaked out
    // of the block rewrote the later `mov.b64 %rd2, tmp;` (the shared symbol).
    const std::string scoped_register_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .shared .align 16 .b8 tmp[];
.visible .entry scoped(
    .param .u64 scoped_0
) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [scoped_0];
    {
    .reg .b32 tmp;
    mov.b64 {tmp, %r2}, %rd1;
    mov.b32 %r1, tmp;
    }
    { .reg .b32 tmp; mov.b64 {tmp, %r3}, %rd1; }
    mov.b64 %rd2, tmp;
    ret;
}
)PTX";
    const auto scoped = cumetal::ptx::parse_ptx(scoped_register_ptx);
    bool scoped_ok = scoped.ok && scoped.module.entries.size() == 1;
    std::size_t renamed_uses = 0;
    std::size_t bare_uses = 0;
    if (scoped_ok) {
        for (const auto& instruction : scoped.module.entries[0].instructions) {
            for (const auto& operand : instruction.operands) {
                if (operand.find("%r_cm_tmp") != std::string::npos) ++renamed_uses;
                if (operand == "tmp") ++bare_uses;
            }
        }
    }
    if (!expect(scoped_ok && renamed_uses == 3 && bare_uses == 1,
                "bare .reg names are renamed only inside their brace scope")) {
        std::fprintf(stderr, "  renamed_uses=%zu bare_uses=%zu\n", renamed_uses, bare_uses);
        return 1;
    }


    std::vector<std::string> call_warnings;
    const auto calls = cumetal::ptx::parse_instruction_block(
        ".reg .b32 arg;\n"
        "@%p1 call.uni (result), // return slot\n"
        "helper, /* comment\n spanning lines */\n"
        "(arg,\n 7); ret;\n"
        "call\n no_args;\n", 100, &call_warnings);
    if (!expect(call_warnings.empty() && calls.instructions.size() == 3,
                "multiline calls assemble once and retain trailing instructions") ||
        !expect(calls.instructions[0].opcode == "call.uni" &&
                calls.instructions[0].predicate == "@%p1" &&
                calls.instructions[0].line == 101 &&
                calls.instructions[0].operands == std::vector<std::string>({"(result)", "helper", "(%r_cm_arg, 7)"}),
                "call preserves predicate, return slot, callee, argument tuple and start line") ||
        !expect(calls.instructions[1].opcode == "ret" && calls.instructions[1].line == 105 &&
                calls.instructions[2].opcode == "call" && calls.instructions[2].line == 106 &&
                calls.instructions[2].operands == std::vector<std::string>({"no_args"}),
                "trailing and subsequent statements retain physical source lines")) return 1;

    call_warnings.clear();
    const auto call_scope = cumetal::ptx::parse_instruction_block(
        "{ .reg .b32 arg; call helper, (arg); call helper, (arg); }\n"
        "call helper, (arg);\n", 1, &call_warnings);
    if (!expect(call_warnings.empty() && call_scope.instructions.size() == 3 &&
                call_scope.instructions[0].operands[1] == "(%r_cm_arg)" &&
                call_scope.instructions[1].operands[1] == "(%r_cm_arg)" &&
                call_scope.instructions[2].operands[1] == "(arg)",
                "call remainders keep bare register scopes until the closing brace")) return 1;
    for (const auto* text : {"call.uni (r),\nhelper,\n(a)", "call helper,\n(a;", "call helper,\na);", "call helper,\n(a)\nret;", "call;"}) {
        call_warnings.clear();
        const auto invalid = cumetal::ptx::parse_instruction_block(text, 20, &call_warnings);
        if (!expect(!call_warnings.empty() && invalid.instructions.size() == 1 &&
                    !invalid.instructions[0].supported && invalid.instructions[0].line == 20,
                    "unterminated/unbalanced calls are explicit unsupported instructions")) return 1;
    }
    std::printf("PASS: ptx parser unit tests\n");
    return 0;
}
