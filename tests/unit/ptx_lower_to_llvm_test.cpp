#include "cumetal/ptx/lower_to_llvm.h"

#include <cstdint>
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

std::size_t count_occurrences(const std::string& haystack, const std::string& needle) {
    std::size_t count = 0;
    std::size_t offset = 0;
    while ((offset = haystack.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += needle.size();
    }
    return count;
}

std::uint64_t stable_device_function_token(const std::string& symbol) {
    std::uint64_t hash = 1469598103934665603ull;
    for (const unsigned char c : symbol) {
        hash ^= c;
        hash *= 1099511628211ull;
    }
    hash &= 0x7fffffffffffffffull;
    return hash == 0 ? 1 : hash;
}

}  // namespace

int main() {
    // Every fixture in this file used to be a stub -- `mov.u32 %r0, %tid.x; ret;` -- while the
    // assertions below checked for a fully computed body. They passed because lower_to_llvm.cpp
    // carried name-matched templates that substituted a canned implementation for any kernel
    // called vector_add / matrix_mul / negate / reduce_sum with roughly the right parameters,
    // discarding the real PTX. The tests were verifying those templates, so they could not have
    // caught the templates miscompiling real kernels -- which they did; see ptx_sweep_numeric.
    //
    // The templates are gone. These fixtures are now real kernels, and the assertions check what
    // the compiler actually emitted for them.
    const std::string ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry vector_add(
    .param .u64 vector_add_param_0,
    .param .u64 vector_add_param_1,
    .param .u64 vector_add_param_2
)
{
    .reg .b64 %rd<8>;
    .reg .b32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [vector_add_param_0];
    ld.param.u64 %rd2, [vector_add_param_1];
    ld.param.u64 %rd3, [vector_add_param_2];
    cvta.to.global.u64 %rd4, %rd1;
    cvta.to.global.u64 %rd5, %rd2;
    cvta.to.global.u64 %rd6, %rd3;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd7, %r1, 4;
    add.s64 %rd4, %rd4, %rd7;
    add.s64 %rd5, %rd5, %rd7;
    add.s64 %rd6, %rd6, %rd7;
    ld.global.f32 %f1, [%rd4];
    ld.global.f32 %f2, [%rd5];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd6], %f3;
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
    // Assert on the operations the PTX actually asked for, not on template-generated SSA names.
    if (!expect(contains(lowered.llvm_ir, "fadd float"),
                "vector-add floating add emitted from the real add.f32")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "load float"),
                "vector-add loads its operands from device memory")) {
        return 1;
    }
    if (!expect(contains(lowered.llvm_ir, "store float"),
                "vector-add stores its result to device memory")) {
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

    // A kernel named `negate` whose body multiplies instead. Under the old name-matched templates
    // this lowered to `fneg` regardless -- the entry name decided the semantics and the PTX was
    // discarded. Guarding against that specifically, because the name still matches.
    const std::string misleading_name_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry negate(
    .param .u64 negate_param_0,
    .param .u64 negate_param_1
)
{
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [negate_param_0];
    ld.param.u64 %rd2, [negate_param_1];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    ld.global.f32 %f1, [%rd3];
    mul.f32 %f2, %f1, %f1;
    st.global.f32 [%rd4], %f2;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions misleading_options;
    misleading_options.entry_name = "negate";
    const auto misleading_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(misleading_name_ptx, misleading_options);
    if (!expect(misleading_lowered.ok, "kernel named negate lowers")) {
        return 1;
    }
    if (!expect(contains(misleading_lowered.llvm_ir, "fmul float"),
                "kernel named negate emits the multiply its PTX actually specifies")) {
        return 1;
    }
    if (!expect(!contains(misleading_lowered.llvm_ir, "fneg"),
                "entry name must not substitute a negate body for unrelated PTX")) {
        return 1;
    }

    // Same guard for the reduce_sum template: a matching name plus an atomic body must lower the
    // real atomic, not a canned reduction.
    const std::string reduce_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry reduce_sum(
    .param .u64 reduce_param_0,
    .param .u64 reduce_param_1
)
{
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd1, [reduce_param_0];
    ld.param.u64 %rd2, [reduce_param_1];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    ld.global.f32 %f1, [%rd3];
    atom.global.add.f32 %f2, [%rd4], %f1;
    ret;
}
)PTX";

    cumetal::ptx::LowerToLlvmOptions reduce_options;
    reduce_options.entry_name = "reduce_sum";
    const auto reduce_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(reduce_ptx, reduce_options);
    if (!expect(reduce_lowered.ok, "reduce_sum lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(reduce_lowered.llvm_ir, "atomicrmw fadd"),
                "reduce_sum emits the atomic add its PTX specifies")) {
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

    // Tolerant (non-strict) mode used to "accept" an unsupported opcode by emitting the kernel
    // signature with a bare `ret void` body. That kernel loaded and launched successfully and
    // wrote nothing, so the caller read back whatever was already in the output buffer and had no
    // way to tell. For a translation layer that is worse than failing: it is an unsupported
    // opcode reported as a successful run. Both modes now refuse.
    const auto tolerant = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, options);
    if (!expect(!tolerant.ok, "tolerant lowering refuses an unsupported opcode")) {
        return 1;
    }
    if (!expect(!tolerant.error.empty(), "refusal carries a diagnostic")) {
        return 1;
    }

    cumetal::ptx::LowerToLlvmOptions strict_options;
    strict_options.entry_name = "vector_add";
    strict_options.strict = true;
    const auto strict = cumetal::ptx::lower_ptx_to_llvm_ir(unsupported_ptx, strict_options);
    if (!expect(!strict.ok, "strict lowering fails on unsupported opcode set")) {
        return 1;
    }

    const std::string llvm_printf_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry llvm_printf()
{
    .reg .b32 %r<2>;
    mov.u32 %r1, 7;
    call.uni (%r0), vprintf, ("value=%d", %r1);
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions printf_options;
    printf_options.entry_name = "llvm_printf";
    const auto llvm_printf =
        cumetal::ptx::lower_ptx_to_llvm_ir(llvm_printf_ptx, printf_options);
    if (!expect(llvm_printf.ok,
                "LLVM PTX backend lowers vprintf to the runtime ring-buffer ABI")) {
        return 1;
    }
    if (!expect(contains(llvm_printf.llvm_ir,
                         "atomicrmw add i32 addrspace(1)* %__cumetal_printf_buffer") &&
                    contains(llvm_printf.llvm_ir,
                             "i32 addrspace(2)* %__cumetal_printf_capacity") &&
                    contains(llvm_printf.llvm_ir, "store i32 0") &&
                    llvm_printf.printf_formats.size() == 1 &&
                    llvm_printf.printf_formats[0] == "value=%d",
                "LLVM vprintf carries hidden args, record writer, and format metadata")) {
        return 1;
    }

    // Clang commonly moves printf into a device helper and passes vprintf a
    // module-global format pointer plus a packed local tuple. The printf ring
    // ABI and format ids must be propagated through the helper call rather than
    // being inferred only from the selected kernel entry.
    const std::string helper_printf_ptx = R"PTX(
.version 8.0
.target sm_80
.address_size 64
.global .align 1 .b8 helper_fmt[16] = {104, 101, 108, 112, 101, 114, 61, 37, 100, 44, 37, 100, 10, 0, 0, 0};
.func helper_printf(.param .b32 helper_a, .param .b32 helper_b)
{
    .local .align 8 .b8 __local_depot_helper[8];
    .reg .b32 %r<3>;
    .reg .b64 %SP;
    .reg .b64 %SPL;
    .reg .b64 %rd<6>;
    mov.b64 %SPL, __local_depot_helper;
    cvta.local.u64 %SP, %SPL;
    add.u64 %rd1, %SP, 0;
    add.u64 %rd2, %SPL, 0;
    ld.param.b32 %r1, [helper_a];
    ld.param.b32 %r2, [helper_b];
    st.local.v2.b32 [%rd2], {%r1, %r2};
    {
    .param .b64 format_arg;
    .param .b64 tuple_arg;
    .param .b32 retval0;
    st.param.b64 [tuple_arg], %rd1;
    mov.b64 %rd3, helper_fmt;
    cvta.global.u64 %rd4, %rd3;
    st.param.b64 [format_arg], %rd4;
    call.uni (retval0), vprintf, (format_arg, tuple_arg);
    }
    ret;
}
.visible .entry calls_helper_printf(.param .u32 input_a, .param .u32 input_b)
{
    .reg .b32 %r<3>;
    ld.param.u32 %r1, [input_a];
    ld.param.u32 %r2, [input_b];
    {
    .param .b32 helper_a;
    .param .b32 helper_b;
    st.param.b32 [helper_a], %r1;
    st.param.b32 [helper_b], %r2;
    call.uni helper_printf, (helper_a, helper_b);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions helper_printf_options;
    helper_printf_options.entry_name = "calls_helper_printf";
    helper_printf_options.strict = true;
    const auto helper_printf =
        cumetal::ptx::lower_ptx_to_llvm_ir(helper_printf_ptx, helper_printf_options);
    if (!expect(helper_printf.ok,
                "device-helper packed-tuple vprintf lowers")) {
        std::fprintf(stderr, "  error: %s\n", helper_printf.error.c_str());
        return 1;
    }
    if (!expect(helper_printf.printf_formats.size() == 1 &&
                    helper_printf.printf_formats[0] == "helper=%d,%d\n" &&
                    contains(helper_printf.llvm_ir,
                             "call void @helper_printf(i32") &&
                    contains(helper_printf.llvm_ir,
                             "i32 addrspace(1)* %__cumetal_printf_buffer, i32 addrspace(2)* %__cumetal_printf_capacity") &&
                    contains(helper_printf.llvm_ir,
                             "define internal void @helper_printf") &&
                    contains(helper_printf.llvm_ir,
                             "atomicrmw add i32 addrspace(1)* %__cumetal_printf_buffer"),
                "device-helper printf propagates hidden args, ring writes, and format metadata")) {
        return 1;
    }

    const std::string mul_wide_s16_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry mul_wide_s16(.param .u64 output) {
    .reg .b64 %rd<2>;
    .reg .b32 %r<2>;
    .reg .b16 %rs<2>;
    ld.param.u64 %rd1, [output];
    mov.b16 %rs1, 65535;
    mul.wide.s16 %r1, %rs1, 4;
    st.global.b32 [%rd1], %r1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions mul_wide_s16_options;
    mul_wide_s16_options.entry_name = "mul_wide_s16";
    const auto mul_wide_s16 =
        cumetal::ptx::lower_ptx_to_llvm_ir(mul_wide_s16_ptx, mul_wide_s16_options);
    if (!expect(mul_wide_s16.ok &&
                    contains(mul_wide_s16.llvm_ir, "sext i16") &&
                    contains(mul_wide_s16.llvm_ir, "mul i32"),
                "mul.wide.s16 sign-extends 16-bit operands and produces i32")) {
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
    if (!expect(contains(scale_step_lowered.llvm_ir, "i64 addrspace(2)* %scale_step_param_1"),
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

    const std::string indexed_branch_ptx = R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry indexed_branch(
    .param .u32 indexed_branch_param_0,
    .param .u64 indexed_branch_param_1
)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd1;
    ld.param.u32 %r1, [indexed_branch_param_0];
    ld.param.u64 %rd1, [indexed_branch_param_1];
    $L_table: .branchtargets
        $L_zero,
        $L_one;
    brx.idx %r1, $L_table;
$L_zero:
    mov.u32 %r2, 10;
    bra $L_done;
$L_one:
    mov.u32 %r2, 20;
$L_done:
    st.global.u32 [%rd1], %r2;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions indexed_branch_options;
    indexed_branch_options.entry_name = "indexed_branch";
    indexed_branch_options.strict = true;
    indexed_branch_options.module_id = "unit.ptx.indexed_branch";
    const auto indexed_branch_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(indexed_branch_ptx,
                                            indexed_branch_options);
    if (!indexed_branch_lowered.ok) {
        std::fprintf(stderr, "indexed branch lowering error: %s\n",
                     indexed_branch_lowered.error.c_str());
    }
    if (!expect(indexed_branch_lowered.ok, "brx.idx PTX lowering succeeds") ||
        !expect(contains(indexed_branch_lowered.llvm_ir, "switch i32"),
                "brx.idx lowers to LLVM switch") ||
        !expect(contains(indexed_branch_lowered.llvm_ir, "i32 0, label %cm_bb_"),
                "brx.idx emits first target") ||
        !expect(contains(indexed_branch_lowered.llvm_ir, "i32 1, label %cm_bb_"),
                "brx.idx emits second target")) {
        return 1;
    }

    // CUDA frontends use vectorized memory operations for ordinary struct
    // copies. Scalarize both v2 and v4 forms so large CUDA projects do not
    // require source changes merely to express the same contiguous accesses.
    const std::string vector_memory_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry vector_memory_generic(
    .param .u64 vector_memory_param_0,
    .param .u64 vector_memory_param_1
)
{
    .reg .b32 %r<7>;
    .reg .b64 %rd<3>;
    .param .b32 call_arg;
    .param .b32 call_arg2;
    .param .b32 call_ret;
    .param .b64 call_ret64;
    .param .b64 call_arg64;
    .param .b64 sin_ptr_arg;
    .param .b64 cos_ptr_arg;
    ld.param.u64 %rd1, [vector_memory_param_0];
    ld.param.u64 %rd2, [vector_memory_param_1];
    ld.global.v4.b32 {%r1, %r2, %r3, %r4}, [%rd1];
    st.global.v4.b32 [%rd2], {%r1, %r2, %r3, %r4};
    ld.global.v2.b32 {%r5, %r6}, [%rd1+16];
    st.global.v2.b32 [%rd2+16], {%r5, %r6};
    ld.b32 %r1, [%rd1];
    st.b32 [%rd2], %r1;
    st.param.b32 [call_arg], %r1;
    call.uni (call_ret), __nv_sqrtf, (call_arg);
    call.uni (call_ret), __nv_acosf, (call_arg);
    ld.param.b32 %r1, [call_ret];
    call.uni (call_ret), __nv_float_as_int, (call_arg);
    call.uni (call_ret), __nv_abs, (call_arg);
    call.uni (call_ret), __nv_clz, (call_arg);
    st.param.b64 [call_arg64], %rd1;
    call.uni (call_ret), __nv_clzll, (call_arg64);
    call.uni (call_ret), __nv_popc, (call_arg);
    call.uni (call_ret), __nv_ffs, (call_arg);
    st.param.b32 [call_arg2], %r2;
    call.uni (call_ret), __nv_mul24, (call_arg, call_arg2);
    call.uni (call_ret), __nv_umul24, (call_arg, call_arg2);
    call.uni (call_ret), __nv_umin, (call_arg, call_arg2);
    call.uni (call_ret), __nv_umax, (call_arg, call_arg2);
    call.uni (call_ret64), __nv_fabs, (call_arg64);
    call.uni (call_ret64), __nv_sqrt, (call_arg64);
    call.uni (call_ret), __nv_fast_fdividef, (call_arg, call_arg);
    st.param.b64 [sin_ptr_arg], %rd1;
    st.param.b64 [cos_ptr_arg], %rd2;
    call.uni (call_ret64), __nv_frexp, (call_arg64, cos_ptr_arg);
    call.uni __nv_fast_sincosf, (call_arg, sin_ptr_arg, cos_ptr_arg);
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions vector_memory_options;
    vector_memory_options.entry_name = "vector_memory_generic";
    vector_memory_options.strict = true;
    vector_memory_options.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const auto vector_memory_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(vector_memory_ptx, vector_memory_options);
    if (!vector_memory_lowered.ok) {
        std::fprintf(stderr, "vector memory lowering error: %s\n",
                     vector_memory_lowered.error.c_str());
    }
    if (!expect(vector_memory_lowered.ok, "v2/v4 vector memory lowering succeeds")) {
        return 1;
    }
    if (!expect(vector_memory_lowered.warnings.empty(),
                "v2/v4 vector memory lowering emits no warnings")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "@air.fast_sqrt.f32"),
                "__nv_sqrtf lowers to Metal sqrt intrinsic")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "sqrt_pair_correction") &&
                    contains(vector_memory_lowered.llvm_ir, "sqrt_pair_is_zero"),
                "emulated __nv_sqrt applies FP32-pair Newton refinement")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "@air.fast_acos.f32"),
                "__nv_acosf lowers to Metal inverse-cosine intrinsic")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "abs_negative") &&
                    contains(vector_memory_lowered.llvm_ir, "abs_negated") &&
                    !contains(vector_memory_lowered.llvm_ir, "sub nsw i32"),
                "__nv_abs lowers with wrapping INT_MIN semantics")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "@llvm.ctlz.i32") &&
                    contains(vector_memory_lowered.llvm_ir, "@llvm.ctlz.i64") &&
                    contains(vector_memory_lowered.llvm_ir, "clz_i32"),
                "__nv_clz and __nv_clzll lower with defined zero semantics")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "@llvm.cttz.i32") &&
                    contains(vector_memory_lowered.llvm_ir, "ffs_zero"),
                "__nv_ffs lowers to count-trailing-zeros plus one")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "mul24_a_shifted") &&
                    contains(vector_memory_lowered.llvm_ir, "ashr i32") &&
                    contains(vector_memory_lowered.llvm_ir, "umul24_a") &&
                    contains(vector_memory_lowered.llvm_ir, "and i32") &&
                    contains(vector_memory_lowered.llvm_ir, " = mul i32"),
                "__nv_mul24 and __nv_umul24 preserve signed and unsigned low-24-bit semantics")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "icmp ult i32") &&
                    contains(vector_memory_lowered.llvm_ir, "icmp ugt i32"),
                "__nv_umin and __nv_umax use unsigned comparisons")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir,
                         "and i64") &&
                    contains(vector_memory_lowered.llvm_ir,
                             "9223372036854775807"),
                "__nv_fabs clears only the binary64 sign bit")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "frexp_subnormal") &&
                    contains(vector_memory_lowered.llvm_ir, "@llvm.ctlz.i64") &&
                    contains(vector_memory_lowered.llvm_ir, "frexp_exponent_ptr") &&
                    contains(vector_memory_lowered.llvm_ir, "4602678819172646912"),
                "__nv_frexp decomposes binary64 bits and stores the exponent")) {
        return 1;
    }
    if (!expect(contains(vector_memory_lowered.llvm_ir, "@air.fast_sin.f32") &&
                    contains(vector_memory_lowered.llvm_ir, "@air.fast_cos.f32"),
                "destination-less __nv_fast_sincosf call lowers to Metal trig intrinsics")) {
        return 1;
    }

    const std::string malformed_abs_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_abs()
{
    .param .b32 call_ret;
    call.uni (call_ret), __nv_abs, ();
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_abs_options;
    malformed_abs_options.entry_name = "malformed_abs";
    malformed_abs_options.strict = true;
    const auto malformed_abs_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_abs_ptx, malformed_abs_options);
    if (!expect(!malformed_abs_lowered.ok &&
                    contains(malformed_abs_lowered.error, "__nv_abs expects 1 arg"),
                "strict lowering rejects malformed __nv_abs calls")) {
        return 1;
    }

    const std::string malformed_clz_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_clz()
{
    .param .b32 call_ret;
    call.uni (call_ret), __nv_clz, ();
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_clz_options;
    malformed_clz_options.entry_name = "malformed_clz";
    malformed_clz_options.strict = true;
    const auto malformed_clz_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_clz_ptx, malformed_clz_options);
    if (!expect(!malformed_clz_lowered.ok &&
                    contains(malformed_clz_lowered.error, "__nv_clz expects 1 arg"),
                "strict lowering rejects malformed __nv_clz calls")) {
        return 1;
    }

    const std::string malformed_mul24_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_mul24()
{
    .param .b32 call_arg;
    .param .b32 call_ret;
    call.uni (call_ret), __nv_mul24, (call_arg);
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_mul24_options;
    malformed_mul24_options.entry_name = "malformed_mul24";
    malformed_mul24_options.strict = true;
    const auto malformed_mul24_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_mul24_ptx, malformed_mul24_options);
    if (!expect(!malformed_mul24_lowered.ok &&
                    contains(malformed_mul24_lowered.error, "__nv_mul24 expects 2 args"),
                "strict lowering rejects malformed __nv_mul24 calls")) {
        return 1;
    }

    const std::string malformed_fabs_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_fabs()
{
    .param .b64 call_ret;
    call.uni (call_ret), __nv_fabs, ();
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_fabs_options;
    malformed_fabs_options.entry_name = "malformed_fabs";
    malformed_fabs_options.strict = true;
    const auto malformed_fabs_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_fabs_ptx, malformed_fabs_options);
    if (!expect(!malformed_fabs_lowered.ok &&
                    contains(malformed_fabs_lowered.error, "__nv_fabs expects 1 arg"),
                "strict lowering rejects malformed __nv_fabs calls")) {
        return 1;
    }

    const std::string malformed_frexp_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_frexp()
{
    .param .b64 call_arg;
    .param .b64 call_ret;
    call.uni (call_ret), __nv_frexp, (call_arg);
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_frexp_options;
    malformed_frexp_options.entry_name = "malformed_frexp";
    malformed_frexp_options.strict = true;
    const auto malformed_frexp_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_frexp_ptx, malformed_frexp_options);
    if (!expect(!malformed_frexp_lowered.ok &&
                    contains(malformed_frexp_lowered.error, "__nv_frexp expects 2 args"),
                "strict lowering rejects malformed __nv_frexp calls")) {
        return 1;
    }

    const std::string reused_call_slot_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry reused_call_slot()
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    {
        .param .b32 param1;
        .param .b32 retval;
        st.param.b32 [param1], %r1;
        call.uni (retval), __nv_abs, (param1);
    }
    {
        .param .b64 param1;
        .param .b32 retval;
        st.param.b64 [param1], %rd1;
        call.uni (retval), __nv_clzll, (param1);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions reused_call_slot_options;
    reused_call_slot_options.entry_name = "reused_call_slot";
    reused_call_slot_options.strict = true;
    const auto reused_call_slot_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(reused_call_slot_ptx, reused_call_slot_options);
    if (!expect(reused_call_slot_lowered.ok &&
                    contains(reused_call_slot_lowered.llvm_ir, "alloca i32") &&
                    contains(reused_call_slot_lowered.llvm_ir, "alloca i64"),
                "lexically reused PTX call-slot names keep distinct integer widths")) {
        return 1;
    }

    const std::string masked_vote_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry masked_vote()
{
    .reg .pred %p<5>;
    .reg .b32 %r<9>;
    mov.u32 %r1, %laneid;
    and.b32 %r2, %r1, 1;
    setp.eq.u32 %p1, %r2, 0;
    vote.sync.ballot.b32 %r3, %p1, 0x0000ffff;
    vote.sync.any.pred %p2, %p1, 0x000000ff;
    vote.sync.all.pred %p3, %p1, 0x00000055;
    activemask.b32 %r4;
    shfl.sync.idx.b32 %r5|%p4, %r1, 0, 0x1f, 0x0000ffff;
    shfl.sync.down.b32 %r6, %r1, 1, 0x101f, 0x0000ffff;
    shfl.sync.up.b32 %r7, %r1, 1, 0x1000, 0x0000ffff;
    shfl.sync.bfly.b32 %r8, %r1, 1, 0x101f, 0x0000ffff;
    bar.warp.sync 0x0000ffff;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions masked_vote_options;
    masked_vote_options.entry_name = "masked_vote";
    masked_vote_options.strict = true;
    const auto masked_vote_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(masked_vote_ptx, masked_vote_options);
    if (!expect(masked_vote_lowered.ok, "masked vote/shuffle lowering succeeds")) {
        return 1;
    }
    if (!expect(contains(masked_vote_lowered.llvm_ir,
                         "declare i64 @air.simd_ballot.i64(i1)"),
                "vote and activemask use AIR SIMD ballot")) {
        return 1;
    }
    if (!expect(contains(masked_vote_lowered.llvm_ir, "and i32") &&
                    contains(masked_vote_lowered.llvm_ir, "65535"),
                "partial vote member mask is retained")) {
        return 1;
    }
    if (!expect(contains(masked_vote_lowered.llvm_ir, "shfl_lane_participates") &&
                    contains(masked_vote_lowered.llvm_ir, "shfl_defined"),
                "partial shuffle predicates non-member lanes")) {
        return 1;
    }
    if (!expect(contains(masked_vote_lowered.llvm_ir,
                         "call i32 @air.simd_shuffle_down.u.i32") &&
                    contains(masked_vote_lowered.llvm_ir,
                             "call i32 @air.simd_shuffle_up.u.i32") &&
                    contains(masked_vote_lowered.llvm_ir,
                             "call i32 @air.simd_shuffle_xor.u.i32") &&
                    count_occurrences(masked_vote_lowered.llvm_ir,
                                      " = trunc i32 1 to i16") == 3,
                "directional AIR shuffles receive the PTX delta or XOR mask")) {
        return 1;
    }
    if (!expect(contains(masked_vote_lowered.llvm_ir,
                         "call void @air.simdgroup.barrier(i32 2, i32 4)"),
                "bar.warp.sync uses AIR simdgroup barrier scope")) {
        return 1;
    }
    if (!expect(!contains(masked_vote_lowered.llvm_ir,
                          "call void @air.wg.barrier(i32 2, i32 1)"),
                "bar.warp.sync does not use a threadgroup barrier")) {
        return 1;
    }
    if (!expect(!contains(masked_vote_lowered.llvm_ir, "zext i1") ||
                    contains(masked_vote_lowered.llvm_ir, "vote_ballot64"),
                "ballot is not lowered to only the caller predicate")) {
        return 1;
    }

    const std::string multi_entry_shared_ptx = R"PTX(
.version 8.0
.target sm_80
.shared .align 16 .b8 shared_for_second[128];
.visible .entry no_shared()
{
    ret;
}
.visible .entry with_shared()
{
    .reg .b64 %rd<2>;
    mov.u64 %rd1, shared_for_second;
    ret;
}
)PTX";
    if (!expect(cumetal::ptx::compute_static_shared_bytes(multi_entry_shared_ptx,
                                                          "no_shared") == 0,
                "entry-specific shared accounting excludes other kernels")) {
        return 1;
    }
    if (!expect(cumetal::ptx::compute_static_shared_bytes(multi_entry_shared_ptx,
                                                          "with_shared") == 128,
                "entry-specific shared accounting includes selected kernel")) {
        return 1;
    }
    if (!expect(cumetal::ptx::compute_static_shared_bytes(multi_entry_shared_ptx) == 128,
                "module-wide shared accounting remains available for registration")) {
        return 1;
    }

    const std::string selected_shared_layout_ptx = R"PTX(
.version 8.0
.target sm_80
.shared .align 16 .b8 unrelated_shared[64];
.shared .align 4 .b8 selected_first[12];
.shared .align 16 .b8 selected_second[32];
.visible .entry other_shared()
{
    .reg .b64 %rd<2>;
    mov.u64 %rd1, unrelated_shared;
    ret;
}
.visible .entry selected_shared()
{
    .reg .b64 %rd<3>;
    mov.u64 %rd1, selected_first;
    mov.u64 %rd2, selected_second;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions selected_shared_options;
    selected_shared_options.entry_name = "selected_shared";
    selected_shared_options.strict = true;
    const auto selected_shared_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(selected_shared_layout_ptx,
                                           selected_shared_options);
    if (!selected_shared_lowered.ok) {
        std::fprintf(stderr, "selected shared lowering error: %s\n",
                     selected_shared_lowered.error.c_str());
    }
    if (!expect(selected_shared_lowered.ok,
                "multiple selected static shared symbols lower")) {
        return 1;
    }
    if (!expect(cumetal::ptx::compute_static_shared_bytes(selected_shared_layout_ptx,
                                                          "selected_shared") == 48,
                "selected static shared allocation includes alignment padding")) {
        return 1;
    }
    if (!expect(contains(selected_shared_lowered.llvm_ir, "tg_sym_off") &&
                    contains(selected_shared_lowered.llvm_ir, ", 16\n") &&
                    !contains(selected_shared_lowered.llvm_ir, ", 80\n"),
                "selected shared symbols start at zero and ignore other entries")) {
        return 1;
    }

    const std::string mixed_shared_const_ptx = R"PTX(
.version 8.0
.target sm_80
.const .align 8 .b8 constant_table[16] = {1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0};
.global .align 1 .b8 _$_str[4] = {79, 75, 10, 0};
.shared .align 16 .b8 shared_scratch[16];
.visible .entry mixed_shared_const()
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    .reg .b32 %r<2>;
    mov.u64 %rd1, shared_scratch;
    mov.u64 %rd2, constant_table;
    mov.u64 %rd3, _$_str;
    ld.const.u32 %r1, [%rd2];
    st.shared.u32 [%rd1], %r1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions mixed_shared_const_options;
    mixed_shared_const_options.entry_name = "mixed_shared_const";
    mixed_shared_const_options.strict = true;
    const auto mixed_shared_const_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(mixed_shared_const_ptx,
                                           mixed_shared_const_options);
    if (!mixed_shared_const_lowered.ok) {
        std::fprintf(stderr, "mixed shared/const lowering error: %s\n",
                     mixed_shared_const_lowered.error.c_str());
    }
    if (!expect(mixed_shared_const_lowered.ok,
                "mixed static shared and constant symbols lower")) {
        return 1;
    }
    if (!expect(contains(mixed_shared_const_lowered.llvm_ir,
                         "getelementptr inbounds [16 x i8], [16 x i8] addrspace(2)* @\"constant_table\"") &&
                    contains(mixed_shared_const_lowered.llvm_ir,
                             "getelementptr inbounds [4 x i8], [4 x i8] addrspace(2)* @\"_$_str\"") &&
                    contains(mixed_shared_const_lowered.llvm_ir,
                             "ptrtoint i8 addrspace(3)* %__air_tg0 to i64"),
                "constant and shared symbols keep distinct address-space bases")) {
        return 1;
    }

    const std::string external_const_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .const .align 1 .b8 unrelated_table[3];
.visible .const .align 16 .b8 external_table[27904];
.visible .entry read_external_const(
    .param .u64 read_external_const_param_0
)
{
    .reg .b64 %rd<4>;
    .reg .b32 %r<2>;
    ld.param.u64 %rd1, [read_external_const_param_0];
    mov.b64 %rd2, external_table;
    add.s64 %rd3, %rd2, 16384;
    ld.const.u32 %r1, [%rd3];
    st.global.u32 [%rd1], %r1;
    ret;
}
)PTX";
    const auto external_symbols =
        cumetal::ptx::find_referenced_external_constant_symbols(
            external_const_ptx, "read_external_const");
    if (external_symbols.size() != 1 ||
        external_symbols.front().name != "external_table" ||
        external_symbols.front().offset_bytes != 16 ||
        external_symbols.front().size_bytes != 27904) {
        std::fprintf(stderr, "external constant scan returned %zu symbols", external_symbols.size());
        for (const auto& symbol : external_symbols) {
            std::fprintf(stderr, " [%s:%zu+%zu]", symbol.name.c_str(),
                         symbol.offset_bytes, symbol.size_bytes);
        }
        std::fprintf(stderr, "\n");
    }
    if (!expect(external_symbols.size() == 1 &&
                    external_symbols.front().name == "external_table" &&
                    external_symbols.front().offset_bytes == 16 &&
                    external_symbols.front().size_bytes == 27904,
                "external constant scan is entry-specific and preserves byte size")) {
        return 1;
    }
    if (!expect(cumetal::ptx::compute_external_constant_buffer_bytes(
                    external_const_ptx) == 27920,
                "module constant layout includes aligned unreferenced declarations")) {
        return 1;
    }
    cumetal::ptx::LowerToLlvmOptions external_const_options;
    external_const_options.entry_name = "read_external_const";
    external_const_options.strict = true;
    const auto external_const_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(external_const_ptx,
                                           external_const_options);
    if (!external_const_lowered.ok) {
        std::fprintf(stderr, "external const lowering error: %s\n",
                     external_const_lowered.error.c_str());
    }
    const bool external_const_shape_ok =
        external_const_lowered.ok &&
                    contains(external_const_lowered.llvm_ir,
                             "i8 addrspace(2)* %__cumetal_constant_buffer") &&
                    contains(external_const_lowered.llvm_ir,
                             "ptrtoint i8 addrspace(2)* %__cumetal_constant_buffer to i64") &&
                    contains(external_const_lowered.llvm_ir,
                             "add i64 %const_arg_p2i_") &&
                    contains(external_const_lowered.llvm_ir, ", 16\n") &&
                    contains(external_const_lowered.llvm_ir,
                             "!\"air.buffer_size\", i32 27920") &&
                    contains(external_const_lowered.llvm_ir,
                             "!\"air.location_index\", i32 30") &&
                    !contains(external_const_lowered.llvm_ir,
                              "__cumetal_const_unrelated_table");
    if (!external_const_shape_ok) {
        std::fprintf(stderr, "external const IR:\n%s\n",
                     external_const_lowered.llvm_ir.c_str());
    }
    if (!expect(external_const_shape_ok,
                "referenced external constant lowers to the module buffer at index 30")) {
        return 1;
    }

    const std::string oversized_external_const_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .const .align 1 .b8 too_large[65537];
.visible .entry reject_large_const()
{
    .reg .b64 %rd<2>;
    mov.b64 %rd1, too_large;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions oversized_const_options;
    oversized_const_options.entry_name = "reject_large_const";
    oversized_const_options.strict = true;
    const auto oversized_const_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(oversized_external_const_ptx,
                                           oversized_const_options);
    if (!expect(!oversized_const_lowered.ok &&
                    contains(oversized_const_lowered.error, "exceeds CUDA's 64 KB"),
                "external constant modules above 64 KB fail explicitly")) {
        return 1;
    }

    const std::string external_global_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .global .align 4 .u32 persistent_counter;
.visible .global .align 4 .u32 unrelated_counter;
.visible .entry increment_external_global(
    .param .u64 increment_external_global_param_0
)
{
    .reg .b64 %rd<3>;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [increment_external_global_param_0];
    mov.b64 %rd2, persistent_counter;
    ld.global.u32 %r1, [%rd2];
    add.u32 %r2, %r1, 1;
    st.global.u32 [%rd2], %r2;
    atom.acquire.sys.global.cas.b32 %r3, [persistent_counter], %r1, %r2;
    st.global.u32 [%rd1], %r2;
    ret;
}
)PTX";
    const auto external_globals =
        cumetal::ptx::find_referenced_external_global_symbols(
            external_global_ptx, "increment_external_global");
    if (!expect(external_globals.size() == 1 &&
                    external_globals.front().name == "persistent_counter" &&
                    external_globals.front().size_bytes == 4,
                "external global scan is entry-specific")) {
        return 1;
    }
    cumetal::ptx::LowerToLlvmOptions external_global_options;
    external_global_options.entry_name = "increment_external_global";
    external_global_options.strict = true;
    const auto external_global_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(external_global_ptx,
                                           external_global_options);
    if (!external_global_lowered.ok) {
        std::fprintf(stderr, "external global lowering error: %s\n",
                     external_global_lowered.error.c_str());
    }
    if (!expect(external_global_lowered.ok &&
                    contains(external_global_lowered.llvm_ir,
                             "i8 addrspace(1)* %__cumetal_global_persistent_counter") &&
                    contains(external_global_lowered.llvm_ir,
                             "ptrtoint i8 addrspace(1)* %__cumetal_global_persistent_counter to i64") &&
                    contains(external_global_lowered.llvm_ir, "cmpxchg") &&
                    !contains(external_global_lowered.llvm_ir,
                              "__cumetal_global_unrelated_counter"),
                "referenced external global lowers to a writable hidden buffer")) {
        return 1;
    }

    const std::string extern_shared_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .shared .align 16 .b8 dynamic_smem[];
.visible .entry use_extern_shared()
{
    .reg .b64 %rd<2>;
    mov.b64 %rd1, dynamic_smem;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions extern_shared_options;
    extern_shared_options.entry_name = "use_extern_shared";
    extern_shared_options.strict = true;
    const auto extern_shared_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(extern_shared_ptx,
                                           extern_shared_options);
    if (!expect(extern_shared_lowered.ok &&
                    contains(extern_shared_lowered.llvm_ir,
                             "ptrtoint i8 addrspace(3)* %__air_tg0 to i64"),
                "declared extern shared symbol resolves to dynamic threadgroup memory")) {
        return 1;
    }

    const std::string spilled_shared_pointer_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .shared .align 16 .b8 dynamic_smem[];
.visible .entry use_spilled_shared_pointer()
{
    .local .align 8 .b8 __local_depot0[16];
    .reg .b32 %r<2>;
    .reg .b64 %SPL, %SP, %rd<4>;
    mov.b64 %SPL, __local_depot0;
    cvta.local.u64 %SP, %SPL;
    mov.b64 %rd1, dynamic_smem;
    cvta.shared.u64 %rd2, %rd1;
    st.local.b64 [%SP], %rd2;
    ld.local.b64 %rd3, [%SP];
    ld.b32 %r1, [%rd3];
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions spilled_shared_pointer_options;
    spilled_shared_pointer_options.entry_name = "use_spilled_shared_pointer";
    spilled_shared_pointer_options.strict = true;
    const auto spilled_shared_pointer_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(spilled_shared_pointer_ptx,
                                           spilled_shared_pointer_options);
    if (!expect(spilled_shared_pointer_lowered.ok &&
                    contains(spilled_shared_pointer_lowered.llvm_ir,
                             "load i32, i32 addrspace(3)*"),
                "generic load through a locally spilled shared pointer stays in threadgroup memory")) {
        return 1;
    }

    const std::string shared_staged_global_pointer_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .shared .align 8 .b8 dynamic_smem[];
.visible .entry use_shared_staged_global_pointer(
    .param .u64 output
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [output];
    cvta.to.global.u64 %rd2, %rd1;
    mov.b64 %rd3, dynamic_smem;
    st.shared.u64 [%rd3], %rd2;
    ld.shared.u64 %rd4, [%rd3];
    mov.u32 %r1, 7;
    st.u32 [%rd4], %r1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions shared_staged_global_pointer_options;
    shared_staged_global_pointer_options.entry_name =
        "use_shared_staged_global_pointer";
    shared_staged_global_pointer_options.strict = true;
    const auto shared_staged_global_pointer_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(
            shared_staged_global_pointer_ptx,
            shared_staged_global_pointer_options);
    if (!expect(shared_staged_global_pointer_lowered.ok &&
                    contains(shared_staged_global_pointer_lowered.llvm_ir,
                             ", i32 addrspace(1)*"),
                "global pointer staged in shared memory retains global pointee address space")) {
        return 1;
    }

    const std::string scalar_shared_ptx = R"PTX(
.version 8.0
.target sm_80
.shared .align 4 .u32 scalar_shared;
.visible .entry use_scalar_shared()
{
    .reg .b32 %r<3>;
    mov.u32 %r1, 7;
    st.shared.u32 [scalar_shared], %r1;
    atom.shared.exch.b32 %r2, [scalar_shared], 0;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions scalar_shared_options;
    scalar_shared_options.entry_name = "use_scalar_shared";
    scalar_shared_options.strict = true;
    const auto scalar_shared_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(scalar_shared_ptx,
                                           scalar_shared_options);
    if (!expect(scalar_shared_lowered.ok &&
                    contains(scalar_shared_lowered.llvm_ir,
                             "ptrtoint i8 addrspace(3)* %__air_tg0 to i64") &&
                    contains(scalar_shared_lowered.llvm_ir,
                             "atomicrmw xchg i32 addrspace(3)*"),
                "declared scalar shared symbol resolves for ordinary and atomic memory access")) {
        return 1;
    }

    const std::string undeclared_symbol_ptx = R"PTX(
.version 8.0
.target sm_80
.shared .align 16 .b8 declared_shared[16];
.visible .entry reject_undeclared_symbol()
{
    .reg .b64 %rd<3>;
    mov.u64 %rd1, declared_shared;
    mov.u64 %rd2, undeclared_symbol;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions undeclared_symbol_options;
    undeclared_symbol_options.entry_name = "reject_undeclared_symbol";
    undeclared_symbol_options.strict = true;
    const auto undeclared_symbol_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(undeclared_symbol_ptx,
                                           undeclared_symbol_options);
    if (!expect(!undeclared_symbol_lowered.ok &&
                    contains(undeclared_symbol_lowered.error,
                             "mov source unsupported"),
                "strict lowering rejects undeclared symbols instead of aliasing shared memory")) {
        return 1;
    }

    const std::string suffixed_immediate_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry suffixed_immediate()
{
    .reg .b32 %r<4>;
    mov.u32 %r1, 287454020U;
    prmt.b32 %r2, %r1, 0, 0x3340U;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions suffixed_immediate_options;
    suffixed_immediate_options.entry_name = "suffixed_immediate";
    suffixed_immediate_options.strict = true;
    const auto suffixed_immediate_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(suffixed_immediate_ptx,
                                           suffixed_immediate_options);
    if (!expect(suffixed_immediate_lowered.ok &&
                    contains(suffixed_immediate_lowered.llvm_ir, "prmt_src"),
                "Clang-style unsigned PTX immediates lower")) {
        return 1;
    }

    const std::string malformed_suffixed_immediate_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_suffixed_immediate()
{
    .reg .b32 %r<3>;
    prmt.b32 %r1, 0, 0, 0x33G0U;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_suffixed_immediate_options;
    malformed_suffixed_immediate_options.entry_name =
        "malformed_suffixed_immediate";
    malformed_suffixed_immediate_options.strict = true;
    const auto malformed_suffixed_immediate_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_suffixed_immediate_ptx,
                                           malformed_suffixed_immediate_options);
    if (!expect(!malformed_suffixed_immediate_lowered.ok &&
                    contains(malformed_suffixed_immediate_lowered.error,
                             "prmt sources unsupported"),
                "malformed suffixed PTX immediate is rejected")) {
        return 1;
    }

    const std::string tuple_pack_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry tuple_pack()
{
    .reg .b8 %b<5>;
    .reg .b16 %rs<4>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    mov.b16 %rs1, 1;
    mov.b16 %rs2, 2;
    mov.b32 %r1, {%rs1, %rs2};
    mov.b8 %b1, 1;
    mov.b8 %b2, 2;
    mov.b8 %b3, 3;
    mov.b8 %b4, 4;
    mov.b32 %r2, {%b1, %b2, %b3, %b4};
    mov.b64 %rd1, {%r1, %r2};
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions tuple_pack_options;
    tuple_pack_options.entry_name = "tuple_pack";
    tuple_pack_options.strict = true;
    const auto tuple_pack_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(tuple_pack_ptx, tuple_pack_options);
    if (!expect(tuple_pack_lowered.ok &&
                    contains(tuple_pack_lowered.llvm_ir, "movpack_sh") &&
                    contains(tuple_pack_lowered.llvm_ir, "movpack_or"),
                "mov.b32/mov.b64 pack evenly sized source tuples")) {
        return 1;
    }

    const std::string malformed_tuple_pack_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_tuple_pack()
{
    .reg .b8 %b<4>;
    .reg .b32 %r<2>;
    mov.b32 %r1, {%b1, %b2, %b3};
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_tuple_pack_options;
    malformed_tuple_pack_options.entry_name = "malformed_tuple_pack";
    malformed_tuple_pack_options.strict = true;
    const auto malformed_tuple_pack_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_tuple_pack_ptx,
                                           malformed_tuple_pack_options);
    if (!expect(!malformed_tuple_pack_lowered.ok &&
                    contains(malformed_tuple_pack_lowered.error,
                             "evenly sized b32/b64 source tuple"),
                "malformed mov tuple pack is rejected")) {
        return 1;
    }

    const std::string malformed_masked_vote_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_masked_vote()
{
    .reg .pred %p<2>;
    .reg .b32 %r<2>;
    vote.sync.ballot.b32 %r1, %p1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_masked_vote_options;
    malformed_masked_vote_options.entry_name = "malformed_masked_vote";
    malformed_masked_vote_options.strict = true;
    const auto malformed_masked_vote_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_masked_vote_ptx,
                                           malformed_masked_vote_options);
    if (!expect(!malformed_masked_vote_lowered.ok,
                "strict lowering rejects vote.sync without member mask")) {
        return 1;
    }

    const std::string malformed_masked_shuffle_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_masked_shuffle()
{
    .reg .b32 %r<3>;
    shfl.sync.idx.b32 %r1, %r2, 0, 0x1f;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_masked_shuffle_options;
    malformed_masked_shuffle_options.entry_name = "malformed_masked_shuffle";
    malformed_masked_shuffle_options.strict = true;
    const auto malformed_masked_shuffle_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_masked_shuffle_ptx,
                                           malformed_masked_shuffle_options);
    if (!expect(!malformed_masked_shuffle_lowered.ok,
                "strict lowering rejects shfl.sync without member mask")) {
        return 1;
    }

    const std::string malformed_warp_barrier_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_warp_barrier()
{
    bar.warp.sync;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_warp_barrier_options;
    malformed_warp_barrier_options.entry_name = "malformed_warp_barrier";
    malformed_warp_barrier_options.strict = true;
    const auto malformed_warp_barrier_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_warp_barrier_ptx,
                                           malformed_warp_barrier_options);
    if (!expect(!malformed_warp_barrier_lowered.ok,
                "strict lowering rejects bar.warp.sync without member mask")) {
        return 1;
    }

    // A `.local` stack depot must be allocated at its declared size. Guessing a
    // fixed size silently truncates the frame: out-of-range slots read as zero
    // instead of faulting, so a register-tiled kernel quietly computes zeros.
    const std::string local_depot_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry local_depot_frame(
    .param .u64 local_depot_frame_param_0
)
{
    .local .align 4 .b8
    __local_depot0[288];
    .reg .b64 %SP;
    .reg .b64 %SPL;
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;

    mov.b64 %SPL, __local_depot0;
    ld.param.b64 %rd1, [local_depot_frame_param_0];
    add.u64 %rd2, %SPL, 256;
    mov.b32 %r1, 0;
    st.local.b32 [%rd2], %r1;
    ld.local.b32 %r1, [%rd2];
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions local_depot_options;
    local_depot_options.entry_name = "local_depot_frame";
    local_depot_options.strict = true;
    const auto local_depot_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(local_depot_ptx, local_depot_options);
    if (!expect(local_depot_lowered.ok, "local depot kernel lowers")) {
        std::fprintf(stderr, "  error: %s\n", local_depot_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(local_depot_lowered.llvm_ir, "alloca [288 x i8]"),
                "local depot alloca uses the declared frame size")) {
        return 1;
    }
    if (!expect(!contains(local_depot_lowered.llvm_ir, "alloca [256 x i8]"),
                "local depot alloca is not a fixed-size guess")) {
        return 1;
    }

    // A device function owns a separate local frame. Its depot declaration is
    // outside the selected entry body and must be associated with the helper,
    // rather than discarded or confused with the kernel's frame.
    const std::string helper_local_depot_ptx = R"PTX(
.version 8.0
.target sm_80
.func helper_with_frame(.param .b64 helper_arg)
{
    .local .align 8 .b8 __local_depot1[32];
    .reg .b64 %SPL;
    .reg .b64 %rd<3>;
    mov.b64 %SPL, __local_depot1;
    ld.param.b64 %rd1, [helper_arg];
    st.local.b64 [%SPL+24], %rd1;
    ret;
}
.visible .entry calls_helper(.param .b64 input)
{
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [input];
    {
    .param .b64 helper_arg;
    st.param.b64 [helper_arg], %rd1;
    call.uni helper_with_frame, (helper_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions helper_local_depot_options;
    helper_local_depot_options.entry_name = "calls_helper";
    helper_local_depot_options.strict = true;
    const auto helper_local_depot_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(helper_local_depot_ptx,
                                           helper_local_depot_options);
    if (!expect(helper_local_depot_lowered.ok,
                "device-function local depot lowers")) {
        std::fprintf(stderr, "  error: %s\n", helper_local_depot_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(helper_local_depot_lowered.llvm_ir, "alloca [32 x i8]"),
                "device-function depot uses its declared frame size")) {
        return 1;
    }

    // Without a parseable depot declaration the frame size is unknown; refuse to
    // lower rather than emit an under-sized frame that reads zeros.
    const std::string undeclared_depot_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry undeclared_depot()
{
    .reg .b64 %SP;
    .reg .b64 %SPL;
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;

    mov.b64 %SPL, __local_depot0;
    add.u64 %rd2, %SPL, 16;
    mov.b32 %r1, 0;
    st.local.b32 [%rd2], %r1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions undeclared_depot_options;
    undeclared_depot_options.entry_name = "undeclared_depot";
    undeclared_depot_options.strict = true;
    const auto undeclared_depot_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(undeclared_depot_ptx, undeclared_depot_options);
    if (!expect(!undeclared_depot_lowered.ok,
                "strict lowering refuses a local depot with no declared size")) {
        return 1;
    }

    // FP64 emulation is selected by instruction type, not by a magic kernel
    // name. Its register representation is two packed FP32 values, so the AIR
    // module must contain no native double ALU operations.
    const std::string generic_fp64_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry arbitrary_precision_work()
{
    .reg .f32 %f<3>;
    .reg .f64 %fd<8>;
    .reg .s32 %r<3>;
    .reg .pred %p<3>;
    mov.f32 %f1, 1.25;
    cvt.rn.f64.f32 %fd1, %f1;
    mov.s32 %r1, -17;
    cvt.rn.f64.s32 %fd1, %r1;
    mov.f64 %fd2, 0d4000000000000000;
    add.f64 %fd3, %fd1, %fd2;
    mul.f64 %fd4, %fd3, %fd2;
    div.f64 %fd5, %fd4, %fd2;
    fma.rn.f64 %fd6, %fd5, %fd2, %fd1;
    neg.f64 %fd7, %fd6;
    abs.f64 %fd3, %fd7;
    selp.f64 %fd4, %fd3, %fd2, %p1;
    cvt.rn.f32.f64 %f2, %fd7;
    cvt.rzi.s32.f64 %r2, %fd7;
    setp.num.f64 %p1, %fd1, %fd7;
    setp.nan.f64 %p2, %fd1, %fd7;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions generic_fp64_options;
    generic_fp64_options.entry_name = "arbitrary_precision_work";
    generic_fp64_options.strict = true;
    generic_fp64_options.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const auto generic_fp64_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(generic_fp64_ptx, generic_fp64_options);
    if (!expect(generic_fp64_lowered.ok,
                "generic non-name-matched fp64 register arithmetic lowers")) {
        std::fprintf(stderr, "  error: %s\n", generic_fp64_lowered.error.c_str());
        return 1;
    }
    if (!expect(!contains(generic_fp64_lowered.llvm_ir, "double"),
                "fp64 emulation emits no native double operations")) {
        return 1;
    }
    if (!expect(contains(generic_fp64_lowered.llvm_ir, "fp64_pack"),
                "fp64 emulation stores packed FP32 pairs")) {
        return 1;
    }
    if (!expect(contains(generic_fp64_lowered.llvm_ir, "sitofp i32") &&
                    contains(generic_fp64_lowered.llvm_ir, "fptosi float"),
                "fp64 emulation converts signed 32-bit integers in both directions")) {
        return 1;
    }
    if (!expect(contains(generic_fp64_lowered.llvm_ir, "fcmp ord float") &&
                    contains(generic_fp64_lowered.llvm_ir, "fcmp uno float"),
                "fp64 emulation lowers ordered and NaN predicates")) {
        return 1;
    }

    const std::string directed_fp64_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry directed_fp64_calls()
{
    .reg .b64 %rd<5>;
    .param .b64 arg0;
    .param .b64 arg1;
    .param .b64 retval;
    mov.b64 %rd1, 0d3FF0000000000000;
    mov.b64 %rd2, 0d4000000000000000;
    st.param.b64 [arg0], %rd1;
    st.param.b64 [arg1], %rd2;
    call.uni (retval), __nv_dadd_ru, (arg0, arg1);
    call.uni (retval), __nv_dmul_rd, (arg0, arg1);
    call.uni (retval), __nv_ddiv_ru, (arg0, arg1);
    call.uni (retval), __nv_longlong_as_double, (arg0);
    ld.param.b64 %rd3, [retval];
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions directed_fp64_options;
    directed_fp64_options.entry_name = "directed_fp64_calls";
    directed_fp64_options.strict = true;
    directed_fp64_options.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const auto directed_fp64_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(directed_fp64_ptx,
                                            directed_fp64_options);
    if (!expect(directed_fp64_lowered.ok,
                "directed fp64 libdevice calls lower under pair emulation")) {
        std::fprintf(stderr, "  error: %s\n", directed_fp64_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(directed_fp64_lowered.llvm_ir,
                         "directed_fp64_padding") &&
                    contains(directed_fp64_lowered.llvm_ir,
                             "directed_fp64_negative_padding"),
                "directed fp64 calls widen results in the requested direction")) {
        return 1;
    }

    const std::string integer_width_conversion_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry integer_width_conversion()
{
    .reg .u32 %r1;
    .reg .u64 %rd1;
    mov.u32 %r1, 7;
    cvt.u64.u32 %rd1, %r1;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions integer_width_conversion_options;
    integer_width_conversion_options.entry_name = "integer_width_conversion";
    integer_width_conversion_options.strict = true;
    integer_width_conversion_options.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const auto integer_width_conversion_lowered = cumetal::ptx::lower_ptx_to_llvm_ir(
        integer_width_conversion_ptx, integer_width_conversion_options);
    if (!expect(integer_width_conversion_lowered.ok,
                "FP64 emulation does not intercept 64-bit integer conversions")) {
        std::fprintf(stderr, "  error: %s\n", integer_width_conversion_lowered.error.c_str());
        return 1;
    }

    const std::string fp64_memory_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry unsupported_fp64_memory(.param .u64 p)
{
    .reg .b64 %rd1;
    .reg .f64 %fd1;
    ld.param.u64 %rd1, [p];
    ld.global.f64 %fd1, [%rd1];
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions fp64_memory_options;
    fp64_memory_options.entry_name = "unsupported_fp64_memory";
    fp64_memory_options.strict = true;
    fp64_memory_options.fp64_mode = cumetal::ptx::Fp64Mode::kEmulate;
    const auto fp64_memory_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(fp64_memory_ptx, fp64_memory_options);
    if (!expect(!fp64_memory_lowered.ok,
                "unsupported fp64 memory representation is rejected explicitly")) {
        return 1;
    }
    if (!expect(contains(fp64_memory_lowered.error, "fp64 memory load/store"),
                "fp64 memory rejection identifies the unsupported boundary")) {
        return 1;
    }

    const std::string packed_half_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry packed_half_arithmetic()
{
    .reg .b32 %r<5>;
    mov.b32 %r1, 0x3c003c00;
    mov.b32 %r2, 0x40004000;
    add.f16x2 %r3, %r1, %r2;
    fma.rn.f16x2 %r4, %r1, %r2, %r3;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions packed_half_options;
    packed_half_options.entry_name = "packed_half_arithmetic";
    packed_half_options.strict = true;
    const auto packed_half_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(packed_half_ptx, packed_half_options);
    if (!expect(packed_half_lowered.ok, "packed f16x2 add and fma lower")) {
        std::fprintf(stderr, "  error: %s\n", packed_half_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(packed_half_lowered.llvm_ir, "fadd <2 x half>") &&
                contains(packed_half_lowered.llvm_ir, "fmul <2 x half>"),
                "packed half arithmetic preserves both lanes")) {
        return 1;
    }

    const std::string device_heap_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry device_heap_calls(.param .u64 output)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    {
    .param .b64 size_arg;
    .param .b64 malloc_result;
    st.param.b64 [size_arg], 64;
    call.uni (malloc_result), malloc, (size_arg);
    ld.param.b64 %rd2, [malloc_result];
    }
    st.global.b64 [%rd1], %rd2;
    {
    .param .b64 pointer_arg;
    st.param.b64 [pointer_arg], %rd2;
    call.uni free, (pointer_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions device_heap_options;
    device_heap_options.entry_name = "device_heap_calls";
    device_heap_options.strict = true;
    const auto device_heap_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(device_heap_ptx, device_heap_options);
    if (!expect(device_heap_lowered.ok, "device malloc/free calls lower")) {
        std::fprintf(stderr, "  error: %s\n", device_heap_lowered.error.c_str());
        return 1;
    }
    if (!expect(device_heap_lowered.uses_device_heap &&
                    contains(device_heap_lowered.llvm_ir,
                             "i32 addrspace(1)* %__cumetal_device_heap") &&
                    contains(device_heap_lowered.llvm_ir, "atomicrmw add") &&
                    contains(device_heap_lowered.llvm_ir, "cmpxchg"),
                "device heap lowering exposes its hidden ABI and atomic allocator")) {
        return 1;
    }

    const std::string device_vtable_ptx = R"PTX(
.version 8.0
.target sm_80
.weak .func target_fn(.param .b64 object);
.weak .global .align 8 .u64 test_vtable[3] = {0, target_fn, 7};
.visible .entry store_vtable_address(.param .u64 output)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, test_vtable;
    st.global.b64 [%rd1], %rd2;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions device_vtable_options;
    device_vtable_options.entry_name = "store_vtable_address";
    device_vtable_options.strict = true;
    const auto device_vtable_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(device_vtable_ptx, device_vtable_options);
    if (!expect(device_vtable_lowered.ok,
                "initialized symbolic u64 device tables lower")) {
        std::fprintf(stderr, "  error: %s\n", device_vtable_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(device_vtable_lowered.llvm_ir,
                         "@\"test_vtable\" = internal addrspace(2) constant [24 x i8]") &&
                    contains(device_vtable_lowered.llvm_ir, "const_p2i"),
                "device vtables preserve addressable storage and symbolic identity")) {
        return 1;
    }

    const std::string bf16_wmma_marker_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry bf16_wmma_marker(.param .u64 destination,
                                 .param .u64 matrix_a,
                                 .param .u64 matrix_b)
{
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [destination];
    ld.param.u64 %rd2, [matrix_a];
    ld.param.u64 %rd3, [matrix_b];
    {
    .param .b64 destination_arg;
    .param .b64 matrix_a_arg;
    .param .b64 matrix_b_arg;
    st.param.b64 [destination_arg], %rd1;
    st.param.b64 [matrix_a_arg], %rd2;
    st.param.b64 [matrix_b_arg], %rd3;
    call.uni __cumetal_wmma_bf16_mma_8x8, (destination_arg, matrix_a_arg, matrix_b_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions bf16_wmma_marker_options;
    bf16_wmma_marker_options.entry_name = "bf16_wmma_marker";
    bf16_wmma_marker_options.strict = true;
    const auto bf16_wmma_marker_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(bf16_wmma_marker_ptx,
                                            bf16_wmma_marker_options);
    if (!expect(bf16_wmma_marker_lowered.ok,
                "BF16 WMMA marker lowers to public Metal matrix operations")) {
        std::fprintf(stderr, "  error: %s\n", bf16_wmma_marker_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(bf16_wmma_marker_lowered.llvm_ir,
                         "air.simdgroup_matrix_8x8_load.v64bf16.p3bf16") &&
                    contains(bf16_wmma_marker_lowered.llvm_ir,
                             "air.simdgroup_matrix_8x8_multiply_accumulate") &&
                    contains(bf16_wmma_marker_lowered.llvm_ir,
                             "air.simdgroup_matrix_8x8_store.v64f32.p3f32"),
                "BF16 WMMA marker emits load, multiply-accumulate, and store intrinsics")) {
        return 1;
    }

    const std::string malformed_bf16_wmma_marker_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_bf16_wmma_marker(.param .u64 destination,
                                           .param .u64 matrix_a)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [destination];
    ld.param.u64 %rd2, [matrix_a];
    {
    .param .b64 destination_arg;
    .param .b64 matrix_a_arg;
    st.param.b64 [destination_arg], %rd1;
    st.param.b64 [matrix_a_arg], %rd2;
    call.uni __cumetal_wmma_bf16_mma_8x8, (destination_arg, matrix_a_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_bf16_wmma_marker_options;
    malformed_bf16_wmma_marker_options.entry_name = "malformed_bf16_wmma_marker";
    malformed_bf16_wmma_marker_options.strict = true;
    const auto malformed_bf16_wmma_marker_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_bf16_wmma_marker_ptx,
                                            malformed_bf16_wmma_marker_options);
    if (!expect(!malformed_bf16_wmma_marker_lowered.ok &&
                    contains(malformed_bf16_wmma_marker_lowered.error,
                             "BF16 WMMA marker expects destination, A, B"),
                "BF16 WMMA marker rejects the wrong argument count")) {
        return 1;
    }

    const std::string f32_wmma_marker_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry f32_wmma_marker(.param .u64 destination,
                                .param .u64 matrix_a,
                                .param .u64 matrix_b)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [destination];
    ld.param.u64 %rd2, [matrix_a];
    ld.param.u64 %rd3, [matrix_b];
    mov.u32 %r1, %ctaid.y;
    st.global.u32 [%rd1], %r1;
    {
    .param .b64 destination_arg;
    .param .b64 matrix_a_arg;
    .param .b64 matrix_b_arg;
    st.param.b64 [destination_arg], %rd1;
    st.param.b64 [matrix_a_arg], %rd2;
    st.param.b64 [matrix_b_arg], %rd3;
    call.uni __cumetal_wmma_f32_mma_8x8, (destination_arg, matrix_a_arg, matrix_b_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions f32_wmma_marker_options;
    f32_wmma_marker_options.entry_name = "f32_wmma_marker";
    f32_wmma_marker_options.strict = true;
    const auto f32_wmma_marker_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(f32_wmma_marker_ptx,
                                           f32_wmma_marker_options);
    if (!expect(f32_wmma_marker_lowered.ok &&
                    contains(f32_wmma_marker_lowered.llvm_ir,
                             "air.simdgroup_matrix_8x8_load.v64f32.p3f32") &&
                    contains(f32_wmma_marker_lowered.llvm_ir,
                             "v64f32.v64f32.v64f32.v64f32") &&
                    contains(f32_wmma_marker_lowered.llvm_ir,
                             "grid_y_adjusted") &&
                    contains(f32_wmma_marker_lowered.llvm_ir,
                             "!\"air.location_index\", i32 26"),
                "FP32 WMMA marker lowers to public Metal matrix operations")) {
        if (!f32_wmma_marker_lowered.ok) {
            std::fprintf(stderr, "  error: %s\n",
                         f32_wmma_marker_lowered.error.c_str());
        }
        return 1;
    }

    const std::string rdc_external_kernel_token_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .func external_child(.param .b64 child_arg);
.visible .entry launch_external_child(.param .u64 output)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, external_child;
    st.global.u64 [%rd1], %rd2;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions rdc_external_kernel_token_options;
    rdc_external_kernel_token_options.entry_name = "launch_external_child";
    rdc_external_kernel_token_options.strict = true;
    const auto rdc_external_kernel_token_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(rdc_external_kernel_token_ptx,
                                            rdc_external_kernel_token_options);
    if (!expect(rdc_external_kernel_token_lowered.ok,
                "RDC external kernel symbols lower to stable device-launch tokens")) {
        std::fprintf(stderr, "  error: %s\n",
                     rdc_external_kernel_token_lowered.error.c_str());
        return 1;
    }
    if (!expect(contains(rdc_external_kernel_token_lowered.llvm_ir,
                         "store i64 " + std::to_string(
                             stable_device_function_token("external_child"))),
                "RDC external kernel address is materialized without a Metal function pointer")) {
        return 1;
    }

    const std::string device_memcpy_async_ptx = R"PTX(
.version 8.0
.target sm_80
.extern .func (.param .b32 result) cudaMemcpyAsync(
    .param .b64 destination, .param .b64 source, .param .b64 count,
    .param .b32 kind, .param .b64 stream);
.visible .entry enqueue_device_copy(.param .u64 destination,
                                    .param .u64 source)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [destination];
    ld.param.u64 %rd2, [source];
    {
    .param .b64 destination_arg;
    .param .b64 source_arg;
    .param .b64 count_arg;
    .param .b32 kind_arg;
    .param .b64 stream_arg;
    .param .b32 result_arg;
    st.param.b64 [destination_arg], %rd1;
    st.param.b64 [source_arg], %rd2;
    st.param.b64 [count_arg], 64;
    st.param.b32 [kind_arg], 3;
    st.param.b64 [stream_arg], 0;
    call.uni (result_arg), cudaMemcpyAsync, (destination_arg, source_arg, count_arg, kind_arg, stream_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions device_memcpy_async_options;
    device_memcpy_async_options.entry_name = "enqueue_device_copy";
    device_memcpy_async_options.strict = true;
    const auto device_memcpy_async_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(device_memcpy_async_ptx,
                                            device_memcpy_async_options);
    if (!expect(device_memcpy_async_lowered.ok &&
                    device_memcpy_async_lowered.uses_device_launch_queue &&
                    contains(device_memcpy_async_lowered.llvm_ir,
                             "%__cumetal_device_launch_queue") &&
                    contains(device_memcpy_async_lowered.llvm_ir,
                             "cdp_copy_record_index") &&
                    contains(device_memcpy_async_lowered.llvm_ir,
                             "atomicrmw add"),
                "device cudaMemcpyAsync emits an ordered queue record")) {
        if (!device_memcpy_async_lowered.ok) {
            std::fprintf(stderr, "  error: %s\n", device_memcpy_async_lowered.error.c_str());
        }
        return 1;
    }

    const std::string malformed_device_memcpy_async_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry malformed_device_copy()
{
    {
    .param .b64 destination_arg;
    .param .b64 source_arg;
    .param .b64 count_arg;
    .param .b32 kind_arg;
    .param .b32 result_arg;
    st.param.b64 [destination_arg], 0;
    st.param.b64 [source_arg], 0;
    st.param.b64 [count_arg], 4;
    st.param.b32 [kind_arg], 3;
    call.uni (result_arg), cudaMemcpyAsync, (destination_arg, source_arg, count_arg, kind_arg);
    }
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions malformed_device_memcpy_async_options;
    malformed_device_memcpy_async_options.entry_name = "malformed_device_copy";
    malformed_device_memcpy_async_options.strict = true;
    const auto malformed_device_memcpy_async_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(malformed_device_memcpy_async_ptx,
                                            malformed_device_memcpy_async_options);
    if (!expect(!malformed_device_memcpy_async_lowered.ok &&
                    contains(malformed_device_memcpy_async_lowered.error,
                             "requires five arguments"),
                "device cudaMemcpyAsync rejects a malformed call ABI")) {
        return 1;
    }

    const std::string device_clock_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry read_device_clock(.param .u64 output)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %clock;
    mov.u32 %r2, %clock;
    sub.u32 %r2, %r2, %r1;
    st.global.u32 [%rd1], %r2;
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions device_clock_options;
    device_clock_options.entry_name = "read_device_clock";
    device_clock_options.strict = true;
    const auto device_clock_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(device_clock_ptx,
                                            device_clock_options);
    if (!expect(device_clock_lowered.ok && device_clock_lowered.uses_device_clock &&
                    contains(device_clock_lowered.llvm_ir,
                             "%__cumetal_device_clock") &&
                    contains(device_clock_lowered.llvm_ir,
                             "atomicrmw add i32 addrspace(1)") &&
                    contains(device_clock_lowered.llvm_ir,
                             "!\"air.location_index\", i32 28"),
                "clock special register uses the reserved monotonic counter")) {
        if (!device_clock_lowered.ok) {
            std::fprintf(stderr, "  error: %s\n", device_clock_lowered.error.c_str());
        }
        return 1;
    }

    const std::string grid_sync_ptx = R"PTX(
.version 8.0
.target sm_80
.visible .entry grid_sync_probe()
{
    call.uni __cumetal_grid_sync, ();
    ret;
}
)PTX";
    cumetal::ptx::LowerToLlvmOptions grid_sync_options;
    grid_sync_options.entry_name = "grid_sync_probe";
    grid_sync_options.strict = true;
    const auto grid_sync_lowered =
        cumetal::ptx::lower_ptx_to_llvm_ir(grid_sync_ptx, grid_sync_options);
    if (!expect(grid_sync_lowered.ok &&
                    contains(grid_sync_lowered.llvm_ir,
                             "%__cumetal_grid_barrier") &&
                    contains(grid_sync_lowered.llvm_ir,
                             "atomicrmw add i32 addrspace(1)") &&
                    contains(grid_sync_lowered.llvm_ir,
                             "!\"air.location_index\", i32 27"),
                "grid sync lowers to the reserved resident-grid barrier ABI")) {
        if (!grid_sync_lowered.ok) {
            std::fprintf(stderr, "  error: %s\n", grid_sync_lowered.error.c_str());
        }
        return 1;
    }

    std::printf("PASS: ptx lower-to-llvm unit tests\n");
    return 0;
}
