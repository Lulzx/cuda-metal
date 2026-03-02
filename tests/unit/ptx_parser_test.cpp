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

    std::printf("PASS: ptx parser unit tests\n");
    return 0;
}
