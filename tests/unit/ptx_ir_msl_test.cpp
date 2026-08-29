#include "cumetal/ir/ir.h"
#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <string>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

constexpr const char* kVectorAddPtx = R"ptx(
.version 7.0
.target sm_80
.address_size 64

.visible .entry vector_add(
    .param .u64 a,
    .param .u64 b,
    .param .u64 c,
    .param .u32 n
)
{
    .reg .pred %p1;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    .reg .f32 %f<4>;

    ld.param.u64 %rd1, [a];
    ld.param.u64 %rd2, [b];
    ld.param.u64 %rd3, [c];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd4, %r5, 4;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    add.u64 %rd7, %rd3, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    st.global.f32 [%rd7], %f3;
DONE:
    ret;
}
)ptx";

}  // namespace

int main() {
    using namespace cumetal;
    bool ok = true;

    metal::PtxToMslOptions options;
    options.entry_name = "vector_add";
    options.source_name = "vector_add.ptx";
    const metal::PtxToMslResult result =
        metal::compile_ptx_to_msl(kVectorAddPtx, options);

    if (!result.ok) {
        std::cerr << result.error << "\n";
        return 1;
    }
    ok &= expect(ir::print(result.gpu_ir).find("gpu.thread_id") != std::string::npos,
                 "PTX importer normalizes thread identity");
    ok &= expect(ir::print(result.gpu_ir).find("cond_branch") != std::string::npos,
                 "PTX importer constructs typed CFG");
    ok &= expect(ir::print(result.metal_ir).find("metal.thread_position") !=
                     std::string::npos,
                 "Metal legalization removes generic GPU builtin");
    ok &= expect(result.source.find("kernel void vector_add") != std::string::npos,
                 "typed backend emits a kernel");
    ok &= expect(result.source.find("[[buffer(0)]]") != std::string::npos,
                 "typed backend emits explicit bindings");
    ok &= expect(result.source.find("threadgroup_position_in_grid") != std::string::npos,
                 "typed backend emits Metal threadgroup builtin");
    ok &= expect(result.source.find("if (") != std::string::npos,
                 "simple forward branch is structurized");
    ok &= expect(result.source.find("reinterpret_cast<device float*>") !=
                     std::string::npos,
                 "typed backend emits checked pointer casts");

    const std::string undefined_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry bad() {
    add.u32 %r1, %r2, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult undefined =
        metal::compile_ptx_to_msl(undefined_ptx);
    ok &= expect(!undefined.ok &&
                     undefined.error.find("used before definition") != std::string::npos,
                 "undefined PTX registers fail before MSL emission");

    const std::string explicit_implicit_def_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry explicit_implicit_def() {
    .reg .b32 %r<3>;
    // implicit-def: %r1
    add.u32 %r2, %r1, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult explicit_implicit_def =
        metal::compile_ptx_to_msl(explicit_implicit_def_ptx);
    ok &= expect(explicit_implicit_def.ok,
                 "compiler-emitted PTX implicit-def markers receive a valid refinement");

    const std::string loop_join_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry loop_join(
    .param .u64 output,
    .param .u32 count
) {
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [count];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra DONE;
    mov.u32 %r2, 0;
LOOP:
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p2, %r2, %r1;
    @%p2 bra LOOP;
DONE:
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";
    metal::PtxToMslOptions loop_options;
    loop_options.entry_name = "loop_join";
    loop_options.source_name = "loop_join.ptx";
    const metal::PtxToMslResult loop_join =
        metal::compile_ptx_to_msl(loop_join_ptx, loop_options);
    ok &= expect(loop_join.ok,
                 "SSA construction carries dominating values through loop and exit joins");
    if (!loop_join.ok) {
        std::cerr << loop_join.error << "\n";
    }

    const std::string barrier_self_loop_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .shared .align 4 .b8 scratch[];
.visible .entry barrier_self_loop(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    .reg .b64 %rd1;
    ld.param.u32 %r1, [count];
    mov.b64 %rd1, scratch;
    mov.u32 %r2, 0;
LOOP:
    st.shared.b32 [%rd1], %r2;
    bar.sync 0;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra LOOP;
    ret;
}
)ptx";
    const metal::PtxToMslResult barrier_self_loop =
        metal::compile_ptx_to_msl(barrier_self_loop_ptx);
    ok &= expect(barrier_self_loop.ok &&
                     barrier_self_loop.source.find("while (true)") != std::string::npos &&
                     barrier_self_loop.source.find("threadgroup_barrier") != std::string::npos &&
                     barrier_self_loop.source.find("[[threadgroup(0)]]") != std::string::npos,
                 "single-block barrier loops preserve dynamic shared memory and structure");
    if (!barrier_self_loop.ok) std::cerr << barrier_self_loop.error << "\n";

    const std::string unconditional_loop_header_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry unconditional_loop_header(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    ld.param.u32 %r1, [count];
    mov.u32 %r2, 0;
HEADER:
    bra BODY;
BODY:
    bar.sync 0;
    add.u32 %r2, %r2, 1;
    setp.lt.u32 %p1, %r2, %r1;
    @%p1 bra HEADER;
    ret;
}
)ptx";
    const metal::PtxToMslResult unconditional_loop_header =
        metal::compile_ptx_to_msl(unconditional_loop_header_ptx);
    ok &= expect(unconditional_loop_header.ok &&
                     unconditional_loop_header.source.find("while (true)") !=
                         std::string::npos,
                 "unconditional natural-loop headers structurize through their latch");
    if (!unconditional_loop_header.ok) {
        std::cerr << unconditional_loop_header.error << "\n";
    }

    const std::string local_depot_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry local_depot() {
    .local .align 8 .b8 depot[32];
    .reg .b64 %rd1;
    mov.b64 %rd1, depot;
    st.local.b32 [%rd1], 7;
    ret;
}
)ptx";
    const metal::PtxToMslResult local_depot =
        metal::compile_ptx_to_msl(local_depot_ptx);
    ok &= expect(local_depot.ok &&
                     local_depot.source.find("thread uchar") != std::string::npos &&
                     local_depot.source.find("reinterpret_cast<thread uint*>") !=
                         std::string::npos,
                 "PTX local depots retain bounded private byte-array storage");
    if (!local_depot.ok) std::cerr << local_depot.error << "\n";

    const std::string module_constant_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .const .align 16 .b8 table[32];
.visible .entry module_constant(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.const.b32 %r1, [table+16];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult module_constant =
        metal::compile_ptx_to_msl(module_constant_ptx);
    ok &= expect(module_constant.ok &&
                     module_constant.source.find("[[buffer(30)]]") != std::string::npos &&
                     module_constant.source.find(" + 16") != std::string::npos,
                 "PTX module constants use reserved binding 30 and byte offsets");
    if (!module_constant.ok) std::cerr << module_constant.error << "\n";

    const std::string frexp_abi_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b64 result) __nv_frexp(
    .param .b64 value,
    .param .b64 exponent
);
.visible .entry frexp_abi(.param .f32 input) {
    .local .align 4 .b8 exponent_slot[4];
    .reg .b32 %r1;
    .reg .b64 %rd<4>;
    ld.param.f32 %r1, [input];
    mov.b64 %rd1, exponent_slot;
    cvt.f64.f32 %rd2, %r1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd1;
    call.uni (retval0), __nv_frexp, (param0, param1);
    ld.param.b64 %rd3, [retval0];
    cvt.rn.f32.f64 %r1, %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult frexp_abi =
        metal::compile_ptx_to_msl(frexp_abi_ptx);
    ok &= expect(frexp_abi.ok && frexp_abi.source.find("frexp(") != std::string::npos &&
                     frexp_abi.source.find("\n    double") == std::string::npos,
                 "proven float frexp pattern normalizes its PTX double ABI boundary");
    if (!frexp_abi.ok) {
        std::cerr << frexp_abi.error << "\n";
    } else if (frexp_abi.source.find("frexp(") == std::string::npos ||
               frexp_abi.source.find("\n    double") != std::string::npos) {
        std::cerr << frexp_abi.source << "\n";
    }

    const std::string hex_float_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry hex_float(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    st.global.f32 [%rd1], %f1;
    ret;
}
)ptx";
    const metal::PtxToMslResult hex_float =
        metal::compile_ptx_to_msl(hex_float_ptx);
    ok &= expect(hex_float.ok &&
                     hex_float.source.find(
                         "as_type<float>(0x3f800000u)") != std::string::npos,
                 "PTX hexadecimal float bit patterns become valid exact MSL literals");

    const std::string signed_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry signed_div(.param .s32 value) {
    .reg .s32 %r<3>;
    ld.param.s32 %r1, [value];
    div.s32 %r2, %r1, -2;
    ret;
}
)ptx";
    const metal::PtxToMslResult signed_result =
        metal::compile_ptx_to_msl(signed_ptx);
    ok &= expect(signed_result.ok &&
                     signed_result.source.find("int(") != std::string::npos &&
                     signed_result.source.find(" / int(-2)") != std::string::npos,
                 "signed PTX division preserves signed semantics in MSL");

    const std::string call_slots_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b32 result) __nv_fmaxf(
    .param .b32 lhs,
    .param .b32 rhs
);
.visible .entry call_slots(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [output];
    mov.b32 %r1, 0f3f800000;
    mov.b32 %r2, 0f40000000;
    .param .b32 param0;
    .param .b32 param1;
    .param .b32 retval0;
    st.param.b32 [param0], %r1;
    st.param.b32 [param1], %r2;
    call.uni (retval0), __nv_fmaxf, (param0, param1);
    ld.param.b32 %r3, [retval0];
    st.global.b32 [%rd1], %r3;
    ret;
}
)ptx";
    const metal::PtxToMslResult call_slots =
        metal::compile_ptx_to_msl(call_slots_ptx);
    ok &= expect(call_slots.ok &&
                     call_slots.source.find("fmax(") != std::string::npos &&
                     call_slots.source.find("as_type<float>") != std::string::npos &&
                     call_slots.source.find("as_type<uint>") != std::string::npos,
                 "PTX call slots preserve float bits across a typed libdevice call");
    if (!call_slots.ok) std::cerr << call_slots.error << "\n";

    const std::string missing_call_slot_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry missing_call_slot() {
    .param .b32 retval0;
    call.uni (retval0), __nv_expf, (param0);
    ret;
}
)ptx";
    const metal::PtxToMslResult missing_call_slot =
        metal::compile_ptx_to_msl(missing_call_slot_ptx);
    ok &= expect(!missing_call_slot.ok &&
                     missing_call_slot.error.find("was not initialized") != std::string::npos,
                 "uninitialized PTX call slots fail explicitly");

    const std::string reciprocal_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry reciprocal(.param .f32 value) {
    .reg .f32 %f<3>;
    ld.param.f32 %f1, [value];
    rcp.rn.f32 %f2, %f1;
    ret;
}
)ptx";
    const metal::PtxToMslResult reciprocal =
        metal::compile_ptx_to_msl(reciprocal_ptx);
    ok &= expect(reciprocal.ok && reciprocal.source.find(" / ") != std::string::npos,
                 "PTX reciprocal lowers to a typed floating division");

    const std::string atomic32_ptx = R"ptx(
.version 8.8
.target sm_80
.address_size 64
.visible .entry atomic32(.param .u64 output) {
    .reg .b32 %r<12>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    atom.relaxed.sys.global.add.u32 %r1, [%rd1], 1;
    atom.relaxed.sys.global.and.b32 %r2, [%rd1+4], 255;
    atom.relaxed.sys.global.or.b32 %r3, [%rd1+8], 2;
    atom.relaxed.sys.global.xor.b32 %r4, [%rd1+12], 4;
    atom.relaxed.sys.global.exch.b32 %r5, [%rd1+16], 7;
    atom.relaxed.sys.global.max.s32 %r6, [%rd1+20], 8;
    atom.relaxed.sys.global.min.s32 %r7, [%rd1+24], -1;
    atom.relaxed.sys.global.cas.b32 %r8, [%rd1+28], 0, 9;
    membar.gl;
    ret;
}
)ptx";
    const metal::PtxToMslResult atomic32 =
        metal::compile_ptx_to_msl(atomic32_ptx);
    ok &= expect(atomic32.ok &&
                     atomic32.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_and_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_or_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_xor_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_exchange_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_max_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_fetch_min_explicit") !=
                         std::string::npos &&
                     atomic32.source.find("cm_atomic_cas_device_u32") !=
                         std::string::npos &&
                     atomic32.source.find("atomic_compare_exchange_weak_explicit") !=
                         std::string::npos,
                 "typed PTX lowers the complete 32-bit CUDA atomic family with explicit UMA policy");
    ok &= expect(atomic32.ok &&
                     atomic32.source.find(" + 28;") != std::string::npos,
                 "PTX atomic memory operands retain literal byte displacements");
    if (!atomic32.ok) std::cerr << atomic32.error << "\n";

    const std::string unfenced_acquire_atomic_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry unfenced_acquire(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    atom.acquire.global.cas.b32 %r1, [%rd1], 0, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult unfenced_acquire_atomic =
        metal::compile_ptx_to_msl(unfenced_acquire_atomic_ptx);
    ok &= expect(!unfenced_acquire_atomic.ok &&
                     unfenced_acquire_atomic.error.find(
                         "requires relaxed CUDA ordering") != std::string::npos,
                 "an acquire atomic without Clang's preceding system fence is not silently weakened");

    const std::string wide_atomic_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry wide_atomic(.param .u64 output) {
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    atom.relaxed.sys.global.add.u64 %rd2, [%rd1], 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult wide_atomic =
        metal::compile_ptx_to_msl(wide_atomic_ptx);
    ok &= expect(!wide_atomic.ok &&
                     wide_atomic.error.find("requires one 32-bit integer result") !=
                         std::string::npos,
                 "typed PTX keeps wide atomics explicit until the lock-bank ABI is implemented");

    if (!ok) return 1;
    std::cout << "PTX -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
