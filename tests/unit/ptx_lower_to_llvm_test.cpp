#include "cumetal/ptx/lower_to_llvm.h"

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

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    mov.u32 %r0, %tid.x;
    add.s32 %r1, %r0, %r0;
    ld.shared.u32 %r2, [%rd1];
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions options;
    options.entry_name = "vector_add";
    options.module_id = "unit.ptx.vector_add";
    const auto lowered = cumetal::ptx::lower_ptx_to_llvm_ir(ptx, options);
    if (!expect(lowered.ok, "lower_ptx_to_llvm_ir succeeds")) {
        return 1;
    }
    if (!expect(lowered.entry_name == "vector_add", "entry name propagated")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "; ModuleID = 'unit.ptx.vector_add'"), "module id emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "define void @vector_add("), "kernel definition emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "float addrspace(1)* %vector_add_param_0"),
                "u64 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "i32 %vector_add_param_3"), "u32 param mapped")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "%sum = fadd float %a.val, %b.val"),
                "vector-add floating add body emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "store float %sum, float addrspace(1)* %c.ptr"),
                "vector-add store emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.kernel\""), "air.kernel attribute emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "\"air.version\"=\"2.8\""), "air.version emitted")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "!air.language_version = !{!"),
                "air language version metadata emitted")) {
        return 1;
    }
    if (!expect(lowered.warnings.empty(), "no warnings for supported vector-add lowering path")) {
        return 1;
    }

    const std::string matrix_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry matrix_mul(
    .param .u64 matrix_mul_param_0,
    .param .u64 matrix_mul_param_1,
    .param .u64 matrix_mul_param_2,
    .param .u32 matrix_mul_param_3,
    .param .u32 matrix_mul_param_4
)
{
    mov.u32 %r0, %tid.x;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions matrix_options;
    matrix_options.entry_name = "matrix_mul";
    matrix_options.module_id = "unit.ptx.matrix_mul";
    const auto matrix_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(matrix_ptx, matrix_options);
    if (!expect(matrix_lowered.ok, "matrix multiply lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir,
                         "define void @matrix_mul(float addrspace(1)* %matrix_mul_param_0"),
                "matrix multiply kernel definition emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir,
                         "%row = udiv i32 %matrix_mul_param_4, %n.val"),
                "matrix row index derivation emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "%prod = fmul float %a.val, %b.val"),
                "matrix multiply fmul emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "%acc.next = fadd float %acc, %prod"),
                "matrix multiply accumulation emitted")) {
        return 1;
    }
    if (!expect(contains(matrix_lowered.llvm_ir, "store float %acc, float addrspace(1)* %c.ptr"),
                "matrix multiply store emitted")) {
        return 1;
    }
    if (!expect(matrix_lowered.warnings.empty(), "no warnings for matrix multiply lowering path")) {
        return 1;
    }

    const std::string negate_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry negate(
    .param .u64 negate_param_0,
    .param .u64 negate_param_1
)
{
    mov.u32 %r0, %tid.x;
    neg.f32 %f1, %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions negate_options;
    negate_options.entry_name = "negate";
    const auto negate_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(negate_ptx, negate_options);
    if (!expect(negate_lowered.ok, "negate lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(negate_lowered.llvm_ir,
                         "define void @negate(float addrspace(1)* %negate_param_0"),
                "negate kernel definition emitted")) {
        return 1;
    }
    if (!expect(contains(negate_lowered.llvm_ir, "i32 %__air_thread_position_in_grid"),
                "negate includes implicit thread position argument")) {
        return 1;
    }
    if (!expect(contains(negate_lowered.llvm_ir, "%neg.val = fneg float %in.val"),
                "negate emits fneg body")) {
        return 1;
    }

    const std::string reduce_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry reduce_sum(
    .param .u64 reduce_param_0,
    .param .u64 reduce_param_1,
    .param .u32 reduce_param_2
)
{
    mov.u32 %r0, %tid.x;
    atom.global.add.f32 %f1, [%rd1], %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions reduce_options;
    reduce_options.entry_name = "reduce_sum";
    const auto reduce_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(reduce_ptx, reduce_options);
    if (!expect(reduce_lowered.ok, "reduce_sum lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(reduce_lowered.llvm_ir, "i32 addrspace(2)* %reduce_param_2"),
                "reduce_sum scalar count lowered as constant buffer pointer")) {
        return 1;
    }
    if (!expect(contains(reduce_lowered.llvm_ir,
                         "atomicrmw fadd float addrspace(1)* %out.ptr, float %in.val monotonic"),
                "reduce_sum emits atomic add")) {
        return 1;
    }
    if (!expect(reduce_lowered.warnings.empty(), "reduce_sum path should not emit warnings")) {
        return 1;
    }

    const std::string unsupported_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2,
    .param .u32 vector_add_param_3
)
{
    foo.shared.u32 %r3, %r2;
    ret;
}
)PTX";

    const auto tolerant = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, options);
    if (!expect(tolerant.ok, "tolerant lowering accepts unsupported opcode")) {
        return 1;
    }
    if (!expect(!tolerant.warnings.empty(), "warnings propagated for unsupported opcode")) {
        return 1;
    }

    cumetal::ptx::LowerToLlvmOptions strict_options;
    strict_options.entry_name = "vector_add";
    strict_options.strict = true;
    const auto strict = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, strict_options);
    if (!expect(!strict.ok, "strict lowering fails on unsupported opcode set")) {
        return 1;
    }

    // Test: .u64 parameter used in arithmetic is inferred as non-pointer scalar,
    // lowered to i64 in LLVM IR rather than float addrspace(1)*.
    // This exercises the ld.param erase-bug fix end-to-end: without the fix,
    // the register-to-param mapping for %rd1 would be immediately erased by the
    // propagation block processing the ld.param instruction itself, causing
    // scale_step_param_1 to default to is_pointer=true → float addrspace(1)*.
    const std::string scale_step_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry scale_step(
    .param .u64 scale_step_param_0,
    .param .u64 scale_step_param_1
)
{
    ld.param.u64 %rd0, [scale_step_param_0];
    ld.param.u64 %rd1, [scale_step_param_1];
    ld.global.f32 %f0, [%rd0];
    mul.lo.u64 %rd2, %rd1, 4;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions scale_step_options;
    scale_step_options.entry_name = "scale_step";
    const auto scale_step_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(scale_step_ptx, scale_step_options);
    if (!expect(scale_step_lowered.ok, "scale_step lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(scale_step_lowered.llvm_ir,
                         "float addrspace(1)* %scale_step_param_0"),
                "scale_step pointer param lowered as device buffer pointer")) {
        return 1;
    }
    if (!expect(contains(scale_step_lowered.llvm_ir, "i64 %scale_step_param_1"),
                "scale_step scalar .u64 param lowered as i64 (not pointer)")) {
        return 1;
    }

    // Regression coverage for the real generic PTX→LLVM path:
    // - parser preserves labels as control-flow targets
    // - inline `.reg ...; mov...` on one line keeps the trailing instruction
    // - `.param .b8 name[N]` aggregate symbols can be addressed via mov.b64 + ld.param
    const std::string generic_branch_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry branchy_generic(
    .param .u64 branchy_param_0,
    .param .u64 branchy_param_1,
    .param .align 4 .b8 branchy_param_2[12]
)
{
    .reg .pred %p<2>;
    .reg .b16  %rs<4>;
    .reg .b32  %r<8>;
    .reg .b64  %rd<4>;
    mov.u32 %r1, %tid.x;
    setp.gt.u32 %p1, %r1, 15;
    @%p1 bra $L1;
    { .reg .b16 tmp; mov.b32 {tmp, %rs1}, %r1; }
$L1:
    mov.b64 %rd1, branchy_param_2;
    ld.param.b32 %r2, [%rd1+4];
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions generic_branch_options;
    generic_branch_options.entry_name = "branchy_generic";
    generic_branch_options.strict = true;
    generic_branch_options.module_id = "unit.ptx.branchy_generic";
    const auto generic_branch_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(generic_branch_ptx, generic_branch_options);
    if (!expect(generic_branch_lowered.ok, "generic branchy PTX lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(generic_branch_lowered.llvm_ir,
                         "air.thread_position_in_threadgroup"),
                "generic PTX lowering injects threadgroup builtin metadata")) {
        return 1;
    }
    if (!expect(contains(generic_branch_lowered.llvm_ir, "cm_bb_"),
                "generic PTX lowering emits structured control-flow blocks")) {
        return 1;
    }
    if (!expect(!contains(generic_branch_lowered.llvm_ir, "ptx.lower opcode="),
                "generic PTX lowering should not fall back to comment-only stub body")) {
        return 1;
    }
    if (!expect(generic_branch_lowered.warnings.empty(),
                "generic branchy PTX lowering should not emit warnings")) {
        return 1;
    }

    std::printf("PASS: ptx lower-to-llvm unit tests\n");
    return 0;
}
