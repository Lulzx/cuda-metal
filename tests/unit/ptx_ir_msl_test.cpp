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

    const std::string inverted_loop_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry inverted_loop(.param .u32 count) {
    .reg .pred %p1;
    .reg .b32 %r<3>;
    ld.param.u32 %r1, [count];
    mov.u32 %r2, 0;
LOOP:
    add.u32 %r2, %r2, 1;
    setp.eq.u32 %p1, %r2, %r1;
    @!%p1 bra LOOP;
    ret;
}
)ptx";
    const metal::PtxToMslResult inverted_loop =
        metal::compile_ptx_to_msl(inverted_loop_ptx);
    ok &= expect(inverted_loop.ok &&
                     inverted_loop.source.find("if (!!") != std::string::npos,
                 "inverted PTX loop predicates preserve the back-edge condition");

    const std::string b64_shuffle_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry b64_shuffle(.param .u64 input, .param .u64 output) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.b64 %rd3, [%rd1];
    cvt.u32.u64 %r1, %rd3;
    { .reg .b32 tmp; mov.b64 {tmp, %r2}, %rd3; }
    shfl.sync.down.b32 %r3, %r2, 1, 31, -1;
    cvt.u64.u32 %rd4, %r3;
    st.global.b64 [%rd2], %rd4;
    ret;
}
)ptx";
    const metal::PtxToMslResult b64_shuffle =
        metal::compile_ptx_to_msl(b64_shuffle_ptx);
    const std::string b64_shuffle_ir = ir::print(b64_shuffle.gpu_ir);
    ok &= expect(b64_shuffle.ok &&
                     b64_shuffle_ir.find("shr") != std::string::npos &&
                     b64_shuffle_ir.find("-> i64") != std::string::npos &&
                     b64_shuffle.source.find("cm_lane_id +") != std::string::npos &&
                     b64_shuffle.source.find("simd_shuffle(") != std::string::npos,
                 "partial mov.b64 tuples and cvt.u64 preserve 32-bit shuffle halves");

    const std::string float_shuffle_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry float_shuffle(.param .u64 output) {
    .reg .f32 %f1;
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    shfl.sync.down.b32 %r1, %f1, 1, 31, -1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult float_shuffle =
        metal::compile_ptx_to_msl(float_shuffle_ptx);
    ok &= expect(float_shuffle.ok &&
                     float_shuffle.source.find("as_type<uint>(") !=
                         std::string::npos &&
                     float_shuffle.source.find("simd_shuffle(") !=
                         std::string::npos,
                 "PTX b32 shuffle preserves float register bits before typed SIMD exchange");
    if (!float_shuffle.ok) std::cerr << float_shuffle.error << "\n";

    const std::string masked_vote_mul_hi_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry masked_vote_mul_hi(.param .u64 output) {
    .reg .pred %p<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %laneid;
    mul.hi.u32 %r2, %r1, 1431655766;
    setp.eq.u32 %p1, %r2, 0;
    vote.sync.any.pred %p2, %p1, 255;
    vote.sync.all.pred %p3, %p1, 255;
    vote.sync.ballot.b32 %r3, %p1, 255;
    selp.u32 %r4, 1, 0, %p2;
    selp.u32 %r5, 1, 0, %p3;
    add.u32 %r6, %r4, %r5;
    add.u32 %r7, %r6, %r3;
    st.global.u32 [%rd1], %r7;
    ret;
}
)ptx";
    const metal::PtxToMslResult masked_vote_mul_hi =
        metal::compile_ptx_to_msl(masked_vote_mul_hi_ptx);
    const std::string masked_vote_mul_hi_ir =
        ir::print(masked_vote_mul_hi.gpu_ir);
    const bool masked_vote_mul_hi_valid = masked_vote_mul_hi.ok &&
                     masked_vote_mul_hi_ir.find("high_half=\"true\"") !=
                         std::string::npos &&
                     masked_vote_mul_hi_ir.find("kind=\"all\"") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("ulong(") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find(">> 32u") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_ballot(") !=
                         std::string::npos &&
                     masked_vote_mul_hi.source.find(
                         "simd_active_threads_mask()") != std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_any(") ==
                         std::string::npos &&
                     masked_vote_mul_hi.source.find("simd_all(") ==
                         std::string::npos;
    ok &= expect(masked_vote_mul_hi_valid,
                 "PTX mul.hi and masked vote operands retain CUDA semantics");
    if (!masked_vote_mul_hi_valid) {
        std::cerr << masked_vote_mul_hi.error << "\n"
                  << masked_vote_mul_hi_ir << "\n"
                  << masked_vote_mul_hi.source << "\n";
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

    const std::string generic_shared_helper_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func shared_barrier_helper(.param .b64 ptr) {
    .reg .b64 %rd1;
    ld.param.b64 %rd1, [ptr];
    st.b32 [%rd1], 7;
    bar.sync 0;
    ret;
}
.visible .entry shared_barrier_call(.param .u64 output) {
    .shared .align 4 .b8 tile[4];
    .reg .b32 %r1;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, tile;
    cvta.shared.u64 %rd3, %rd2;
    .param .b64 param0;
    st.param.b64 [param0], %rd3;
    call.uni shared_barrier_helper, (param0);
    ld.shared.b32 %r1, [tile];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult generic_shared_helper =
        metal::compile_ptx_to_msl(generic_shared_helper_ptx);
    const bool generic_shared_helper_valid =
        generic_shared_helper.ok &&
        generic_shared_helper.source.find(
            "shared_barrier_helper(threadgroup uchar*") != std::string::npos &&
        generic_shared_helper.source.find("threadgroup_barrier") !=
            std::string::npos;
    ok &= expect(generic_shared_helper_valid,
                 "generic PTX helper pointers specialize to shared memory at call sites");
    if (!generic_shared_helper_valid) {
        std::cerr << generic_shared_helper.error << "\n"
                  << generic_shared_helper.source << "\n";
    }

    const std::string predicated_barrier_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry predicated_barrier(.param .u32 enabled) {
    .reg .pred %p1;
    .reg .b32 %r1;
    ld.param.u32 %r1, [enabled];
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bar.sync 0;
    ret;
}
)ptx";
    const metal::PtxToMslResult predicated_barrier =
        metal::compile_ptx_to_msl(predicated_barrier_ptx);
    ok &= expect(!predicated_barrier.ok &&
                     predicated_barrier.error.find("predicated barriers") !=
                         std::string::npos,
                 "predicated PTX barriers fail with an explicit diagnostic");

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

    const std::string module_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 state[32];
.visible .entry module_global(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [state+28];
    add.u32 %r1, %r1, 3;
    st.global.b32 [state+28], %r1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult module_global =
        metal::compile_ptx_to_msl(module_global_ptx);
    const bool module_global_valid = module_global.ok &&
                     module_global.source.find(
                         "device uchar* cm___cumetal_global_state [[buffer(1)]]") !=
                         std::string::npos &&
                     module_global.source.find(
                         "cm___cumetal_global_state + 28") !=
                         std::string::npos &&
                     module_global.source.find("[state+28]") == std::string::npos;
    ok &= expect(module_global_valid,
                 "PTX writable module globals use ordered hidden persistent buffers");
    if (!module_global_valid) {
        std::cerr << module_global.error << "\n" << module_global.source << "\n";
    }

    const std::string promoted_aggregate_literal_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 4 .b8 __const_$record[12] = {3, 0, 0, 0, 0, 0, 32, 64, 4};
.visible .entry promoted_aggregate_literal(.param .u64 output) {
    .reg .b32 %r<5>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [__const_$record];
    ld.global.b32 %r2, [__const_$record+4];
    ld.global.b32 %r3, [__const_$record+8];
    add.u32 %r4, %r1, %r2;
    add.u32 %r4, %r4, %r3;
    st.global.b32 [%rd1], %r4;
    ret;
}
)ptx";
    const metal::PtxToMslResult promoted_aggregate_literal =
        metal::compile_ptx_to_msl(promoted_aggregate_literal_ptx);
    ok &= expect(
        promoted_aggregate_literal.ok &&
            promoted_aggregate_literal.source.find(
                "constant uchar cm___const__record[12]") != std::string::npos &&
            promoted_aggregate_literal.source.find(
                "0x00, 0x00, 0x20, 0x40, 0x04, 0x00, 0x00, 0x00") !=
                std::string::npos &&
            promoted_aggregate_literal.source.find("global_symbol:") ==
                std::string::npos,
        "Clang-promoted aggregate literals embed exact zero-filled bytes without a runtime global binding");
    if (!promoted_aggregate_literal.ok) {
        std::cerr << promoted_aggregate_literal.error << "\n";
    }

    const std::string initialized_writable_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 mutable_state[4] = {1};
.visible .entry initialized_writable_global(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [mutable_state];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult initialized_writable_global =
        metal::compile_ptx_to_msl(initialized_writable_global_ptx);
    const bool initialized_writable_global_valid =
        initialized_writable_global.ok &&
        initialized_writable_global.source.find(
            "device uchar* cm___cumetal_global_mutable_state [[buffer(1)]]") !=
            std::string::npos &&
        initialized_writable_global.source.find(
            "constant uchar cm_mutable_state") == std::string::npos;
    ok &= expect(
        initialized_writable_global_valid,
        "initialized writable PTX globals retain registration-backed mutable storage");
    if (!initialized_writable_global_valid) {
        std::cerr << initialized_writable_global.error << "\n"
                  << initialized_writable_global.source << "\n";
    }

    const std::string oversized_writable_initializer_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .global .align 4 .b8 bad_state[4] = {1, 2, 3, 4, 5};
.visible .entry oversized_writable_initializer(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [bad_state];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult oversized_writable_initializer =
        metal::compile_ptx_to_msl(oversized_writable_initializer_ptx);
    ok &= expect(
        !oversized_writable_initializer.ok &&
            oversized_writable_initializer.error.find(
                "more elements than its declaration") != std::string::npos,
        "oversized writable PTX global initializers fail explicitly");

    const std::string aggregate_param_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry aggregate_param(
    .param .align 4 .b8 packed[12],
    .param .u64 output
) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.b32 %r1, [packed+8];
    ld.param.u64 %rd1, [output];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult aggregate_param =
        metal::compile_ptx_to_msl(aggregate_param_ptx);
    ok &= expect(aggregate_param.ok &&
                     aggregate_param.source.find(
                         "struct CuMetalPackedParam12") != std::string::npos &&
                     aggregate_param.source.find("packed.field2") !=
                         std::string::npos,
                 "typed PTX reads aligned fields from by-value aggregate arguments");
    if (!aggregate_param.ok) std::cerr << aggregate_param.error << "\n";

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

    const std::string rint_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry rint_probe(.param .f32 input, .param .u64 output) {
    .reg .f32 %f<3>;
    .reg .b64 %rd1;
    ld.param.f32 %f1, [input];
    ld.param.u64 %rd1, [output];
    cvt.rni.f32.f32 %f2, %f1;
    st.global.f32 [%rd1], %f2;
    ret;
}
)ptx";
    const metal::PtxToMslResult rint_probe =
        metal::compile_ptx_to_msl(rint_ptx);
    ok &= expect(rint_probe.ok &&
                     rint_probe.source.find("floor(") != std::string::npos &&
                     rint_probe.source.find("fmod(") != std::string::npos &&
                     rint_probe.source.find("copysign(") != std::string::npos &&
                     rint_probe.source.find("rint(") == std::string::npos,
                 "PTX rintf spells deterministic round-to-nearest-even MSL");
    if (!rint_probe.ok) std::cerr << rint_probe.error << "\n";

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

    const std::string direct_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .b32 add_one_ret) add_one(
    .param .b32 add_one_value
) {
    .reg .b32 %r<3>;
    ld.param.b32 %r1, [add_one_value];
    add.u32 %r2, %r1, 1;
    st.param.b32 [add_one_ret], %r2;
    ret;
}
.visible .entry direct_device_call(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, 41;
    .param .b32 param0;
    .param .b32 retval0;
    st.param.b32 [param0], %r1;
    call.uni (retval0), add_one, (param0);
    ld.param.b32 %r2, [retval0];
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult direct_device_call =
        metal::compile_ptx_to_msl(direct_device_call_ptx);
    ok &= expect(direct_device_call.ok &&
                     direct_device_call.source.find("uint add_one(uint") !=
                         std::string::npos &&
                     direct_device_call.source.find("add_one(") !=
                         direct_device_call.source.rfind("add_one("),
                 "typed PTX materializes direct scalar device helpers and return slots");
    if (!direct_device_call.ok) std::cerr << direct_device_call.error << "\n";

    const std::string inferred_pointer_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .b32 pointer_load_ret) pointer_load(
    .param .b64 pointer_load_address
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [pointer_load_address];
    ld.global.b32 %r1, [%rd1];
    st.param.b32 [pointer_load_ret], %r1;
    ret;
}
.visible .entry inferred_pointer_device_call(
    .param .u64 .ptr .align 1 output
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [output];
    .param .b64 argument0;
    st.param.b64 [argument0], %rd1;
    .param .b32 retval0;
    call.uni (retval0), pointer_load, (argument0);
    ld.param.b32 %r1, [retval0];
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult inferred_pointer_device_call =
        metal::compile_ptx_to_msl(inferred_pointer_device_call_ptx);
    ok &= expect(inferred_pointer_device_call.ok &&
                     inferred_pointer_device_call.source.find(
                         "uint pointer_load(device uchar*") != std::string::npos,
                 "typed PTX infers pointer-valued device parameters when older Clang omits .ptr");
    if (!inferred_pointer_device_call.ok) {
        std::cerr << inferred_pointer_device_call.error << "\n";
    }

    const std::string aggregate_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .func (.param .align 4 .b8 make_ret[12]) make_record(
    .param .b32 make_first,
    .param .b32 make_second,
    .param .b32 make_third
) {
    .reg .b32 %r<4>;
    ld.param.b32 %r1, [make_first];
    ld.param.b32 %r2, [make_second];
    ld.param.b32 %r3, [make_third];
    st.param.b32 [make_ret], %r1;
    st.param.b32 [make_ret+4], %r2;
    st.param.b32 [make_ret+8], %r3;
    ret;
}
.visible .func (.param .b32 consume_ret) consume_record(
    .param .align 4 .b8 consume_value[12]
) {
    .reg .b32 %r<6>;
    ld.param.b32 %r1, [consume_value];
    ld.param.b32 %r2, [consume_value+4];
    ld.param.b32 %r3, [consume_value+8];
    add.u32 %r4, %r1, %r2;
    add.u32 %r5, %r4, %r3;
    st.param.b32 [consume_ret], %r5;
    ret;
}
.visible .entry aggregate_device_call(
    .param .u64 .ptr .align 1 output
) {
    .reg .b32 %r<8>;
    .reg .b64 %rd<2>;
    ld.param.b64 %rd1, [output];
    .param .b32 make_arg0;
    .param .b32 make_arg1;
    .param .b32 make_arg2;
    st.param.b32 [make_arg0], 3;
    st.param.b32 [make_arg1], 7;
    st.param.b32 [make_arg2], 11;
    .param .align 4 .b8 make_result[12];
    call.uni (make_result), make_record, (make_arg0, make_arg1, make_arg2);
    ld.param.b32 %r1, [make_result];
    ld.param.b32 %r2, [make_result+4];
    ld.param.b32 %r3, [make_result+8];
    .param .align 4 .b8 consume_arg[12];
    st.param.b32 [consume_arg], %r1;
    st.param.b32 [consume_arg+4], %r2;
    st.param.b32 [consume_arg+8], %r3;
    .param .b32 consume_result;
    call.uni (consume_result), consume_record, (consume_arg);
    ld.param.b32 %r4, [consume_result];
    st.global.b32 [%rd1], %r4;
    ret;
}
)ptx";
    const metal::PtxToMslResult aggregate_device_call =
        metal::compile_ptx_to_msl(aggregate_device_call_ptx);
    ok &= expect(aggregate_device_call.ok &&
                     aggregate_device_call.source.find(
                         "CuMetalPackedParam12 make_record(") !=
                         std::string::npos &&
                     aggregate_device_call.source.find(
                         "consume_record(CuMetalPackedParam12") !=
                         std::string::npos,
                 "typed PTX materializes aggregate device arguments and returns");
    if (!aggregate_device_call.ok) {
        std::cerr << aggregate_device_call.error << "\n";
    }

    const std::string incomplete_aggregate_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .func consume_incomplete(
    .param .align 4 .b8 consume_value[12]
) {
    ret;
}
.visible .entry incomplete_aggregate_device_call() {
    .reg .b32 %r<2>;
    mov.u32 %r1, 1;
    .param .align 4 .b8 consume_arg[12];
    st.param.b32 [consume_arg], %r1;
    st.param.b32 [consume_arg+8], %r1;
    call.uni consume_incomplete, (consume_arg);
    ret;
}
)ptx";
    const metal::PtxToMslResult incomplete_aggregate_device_call =
        metal::compile_ptx_to_msl(incomplete_aggregate_device_call_ptx);
    ok &= expect(!incomplete_aggregate_device_call.ok &&
                     incomplete_aggregate_device_call.error.find(
                         "missing, partial, or overlapping fields") !=
                         std::string::npos,
                 "incomplete aggregate PTX device-call slots fail explicitly");

    const std::string recursive_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .func recursive_helper() {
    call.uni recursive_helper, ();
    ret;
}
.visible .entry recursive_device_call() {
    call.uni recursive_helper, ();
    ret;
}
)ptx";
    const metal::PtxToMslResult recursive_device_call =
        metal::compile_ptx_to_msl(recursive_device_call_ptx);
    ok &= expect(!recursive_device_call.ok &&
                     recursive_device_call.error.find(
                         "recursive PTX device-call cycle") != std::string::npos,
                 "recursive typed PTX device-call graphs fail explicitly");

    const std::string missing_device_call_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry missing_device_call() {
    call.uni absent_helper, ();
    ret;
}
)ptx";
    const metal::PtxToMslResult missing_device_call =
        metal::compile_ptx_to_msl(missing_device_call_ptx);
    ok &= expect(!missing_device_call.ok &&
                     missing_device_call.error.find(
                         "has no typed PTX definition") != std::string::npos,
                 "undefined typed PTX device-call targets fail explicitly");

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
.shared .align 8 .f64 block_sum;
.visible .entry wide_atomic(.param .u64 output) {
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [output];
    atom.relaxed.sys.global.add.u64 %rd2, [%rd1], 1;
    st.shared.b64 [block_sum], 0;
    atom.relaxed.sys.shared.cas.b64 %rd2, [block_sum], 0, 1;
    ret;
}
)ptx";
    const metal::PtxToMslResult wide_atomic =
        metal::compile_ptx_to_msl(wide_atomic_ptx);
    ok &= expect(wide_atomic.ok &&
                     wide_atomic.source.find(
                         "cm_atomic_lock_bank [[buffer(29)]]") !=
                         std::string::npos &&
                     wide_atomic.source.find("cm_wide_atomic_add_device_u64") !=
                         std::string::npos &&
                     wide_atomic.source.find(
                         "cm_wide_atomic_cas_threadgroup_u64") !=
                         std::string::npos &&
                     wide_atomic.source.find("atomic_exchange_explicit") !=
                         std::string::npos &&
                     wide_atomic.source.find("threadgroup uchar cm_shared_block_sum[8]") !=
                         std::string::npos,
                 "typed PTX lowers device and static-shared 64-bit atomics through the lock-bank ABI");
    if (!wide_atomic.ok) std::cerr << wide_atomic.error << "\n";

    const std::string clang_printf_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.extern .func (.param .b32 func_retval0) vprintf(
    .param .b64 vprintf_param_0, .param .b64 vprintf_param_1);
.global .align 1 .b8 format[18] = {80, 82, 73, 78, 84, 70, 91, 37, 100, 44, 37, 100, 93, 61, 37, 100, 10};
.visible .entry typed_printf(.param .u32 value) {
    .local .align 8 .b8 __local_depot0[16];
    .reg .b32 %r<4>;
    .reg .b64 %rd<5>;
    mov.b64 %rd1, __local_depot0;
    cvta.local.u64 %rd2, %rd1;
    ld.param.u32 %r1, [value];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %tid.x;
    st.local.v2.b32 [%rd1], {%r2, %r3};
    st.local.b32 [%rd1+8], %r1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b32 retval0;
    st.param.b64 [param1], %rd2;
    mov.b64 %rd3, format;
    cvta.global.u64 %rd4, %rd3;
    st.param.b64 [param0], %rd4;
    call.uni (retval0), vprintf, (param0, param1);
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_printf =
        metal::compile_ptx_to_msl(clang_printf_ptx);
    ok &= expect(clang_printf.ok && clang_printf.printf_formats.size() == 1 &&
                     clang_printf.printf_formats.front() == "PRINTF[%d,%d]=%d\n" &&
                     clang_printf.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     clang_printf.source.find(" = 3;") != std::string::npos &&
                     clang_printf.source.find("vprintf(") == std::string::npos,
                 "typed PTX decodes Clang vprintf into the bounded ring and parsed-count return ABI");
    if (!clang_printf.ok) std::cerr << clang_printf.error << "\n";

    const std::string clang_null_cvta_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry clang_null_cvta(.param .u64 output) {
    .reg .pred %p<2>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, 0;
    cvta.to.global.u64 %rd3, %rd2;
    setp.eq.b64 %p1, %rd1, %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_null_cvta =
        metal::compile_ptx_to_msl(clang_null_cvta_ptx);
    ok &= expect(clang_null_cvta.ok &&
                     clang_null_cvta.source.find("nullptr") != std::string::npos,
                 "typed PTX preserves Clang mov-zero plus cvta null pointers");
    if (!clang_null_cvta.ok) std::cerr << clang_null_cvta.error << "\n";

    const std::string clang_call_slot_rcp_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry clang_call_slot_rcp(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.b32 %r1, 0f40000000;
    rcp.rn.f32 %r2, %r1;
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult clang_call_slot_rcp =
        metal::compile_ptx_to_msl(clang_call_slot_rcp_ptx);
    ok &= expect(clang_call_slot_rcp.ok &&
                     clang_call_slot_rcp.source.find("as_type<float>") !=
                         std::string::npos &&
                     clang_call_slot_rcp.source.find("1.0 /") != std::string::npos,
                 "typed PTX reciprocal reinterprets b32 call-slot containers as f32");
    if (!clang_call_slot_rcp.ok) std::cerr << clang_call_slot_rcp.error << "\n";

    const std::string pointer_select_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry pointer_select(
    .param .u64 lhs, .param .u64 rhs, .param .u64 output, .param .u32 pick_lhs) {
    .reg .pred %p<2>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<7>;
    ld.param.u64 %rd1, [lhs];
    ld.param.u64 %rd2, [rhs];
    ld.param.u64 %rd3, [output];
    ld.param.u32 %r1, [pick_lhs];
    setp.ne.u32 %p1, %r1, 0;
    selp.b64 %rd4, %rd1, %rd2, %p1;
    mov.u64 %rd6, 1;
    mad.lo.s64 %rd5, %rd6, 4, %rd4;
    ld.global.b32 %r2, [%rd5];
    st.global.b32 [%rd3], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult pointer_select =
        metal::compile_ptx_to_msl(pointer_select_ptx);
    ok &= expect(pointer_select.ok &&
                     pointer_select.source.find("device uchar*") !=
                         std::string::npos &&
                     pointer_select.source.find("reinterpret_cast<device uint*>") !=
                         std::string::npos,
                 "typed PTX propagates device pointers through selp and pointer arithmetic");
    if (!pointer_select.ok) std::cerr << pointer_select.error << "\n";

    const std::string signed_narrow_load_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry signed_narrow_load(.param .u64 input, .param .u64 output) {
    .reg .b16 %rs<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.s8 %rs1, [%rd1];
    cvt.rn.f16.s16 %rs2, %rs1;
    st.global.b16 [%rd2], %rs2;
    ret;
}
)ptx";
    const metal::PtxToMslResult signed_narrow_load =
        metal::compile_ptx_to_msl(signed_narrow_load_ptx);
    ok &= expect(signed_narrow_load.ok &&
                     signed_narrow_load.source.find(
                         "reinterpret_cast<device char*>") != std::string::npos &&
                     signed_narrow_load.source.find("short(") != std::string::npos,
                 "typed PTX sign-extends narrow signed loads before numeric conversion");
    if (!signed_narrow_load.ok) std::cerr << signed_narrow_load.error << "\n";

    const std::string float_bit_container_cvt_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry float_bit_container_cvt(.param .u64 input, .param .u64 output) {
    .reg .b16 %rs<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.b32 %r1, [%rd1];
    cvt.rn.f16.f32 %rs1, %r1;
    st.global.b16 [%rd2], %rs1;
    ret;
}
)ptx";
    const metal::PtxToMslResult float_bit_container_cvt =
        metal::compile_ptx_to_msl(float_bit_container_cvt_ptx);
    ok &= expect(float_bit_container_cvt.ok &&
                     float_bit_container_cvt.source.find("as_type<float>") !=
                         std::string::npos &&
                     float_bit_container_cvt.source.find("half(") !=
                         std::string::npos,
                 "typed PTX reinterprets b32 containers before f32-to-f16 conversion");
    if (!float_bit_container_cvt.ok) std::cerr << float_bit_container_cvt.error << "\n";

    const std::string immediate_bfe_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry immediate_bfe(.param .u64 output, .param .u32 input) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [input];
    bfe.u32 %r2, %r1, 4, 1;
    st.global.u32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult immediate_bfe =
        metal::compile_ptx_to_msl(immediate_bfe_ptx);
    ok &= expect(immediate_bfe.ok &&
                     immediate_bfe.source.find(">> 4") != std::string::npos &&
                     immediate_bfe.source.find("& 1") != std::string::npos,
                 "typed PTX lowers bounded unsigned immediate bfe exactly");
    if (!immediate_bfe.ok) std::cerr << immediate_bfe.error << "\n";

    const std::string predicate_not_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry predicate_not(.param .u64 output, .param .u32 input) {
    .reg .pred %p<3>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    ld.param.u32 %r1, [input];
    setp.ne.u32 %p1, %r1, 0;
    not.pred %p2, %p1;
    @!%p2 bra done;
    st.global.u32 [%rd1], 1;
done:
    ret;
}
)ptx";
    const metal::PtxToMslResult predicate_not =
        metal::compile_ptx_to_msl(predicate_not_ptx);
    ok &= expect(predicate_not.ok &&
                     predicate_not.source.find("4294967295") == std::string::npos,
                 "typed PTX lowers not.pred as boolean negation");
    if (!predicate_not.ok) std::cerr << predicate_not.error << "\n";

    const std::string mov_bit_container_ptx = R"ptx(
.version 8.0
.target sm_80
.address_size 64
.visible .entry mov_bit_container(.param .u64 output) {
    .reg .b32 %r<3>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    add.f32 %r1, 0f3F800000, 0f3F800000;
    mov.b32 %r2, %r1;
    st.global.b32 [%rd1], %r2;
    ret;
}
)ptx";
    const metal::PtxToMslResult mov_bit_container =
        metal::compile_ptx_to_msl(mov_bit_container_ptx);
    ok &= expect(mov_bit_container.ok &&
                     mov_bit_container.source.find("as_type<uint>") !=
                         std::string::npos,
                 "typed PTX mov.b32 preserves float register bit containers");
    if (!mov_bit_container.ok) std::cerr << mov_bit_container.error << "\n";

    const std::string indirect_aggregate_param_ptx = R"ptx(
.version 7.0
.target sm_70
.address_size 64
.visible .entry indirect_aggregate_param(
    .param .u64 output,
    .param .align 4 .b8 dims[12]
) {
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    .reg .pred %p<2>;
    ld.param.u64 %rd1, [output];
    ld.b32 %r1, [%rd1];
    st.b32 [%rd1], %r1;
    mov.b64 %rd2, dims;
    mov.pred %p1, 1;
    @%p1 bra indirect_join;
    mov.u32 %r1, 0;
indirect_join:
    ld.param.b32 %r1, [%rd2+8];
    st.global.u32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult indirect_aggregate_param =
        metal::compile_ptx_to_msl(indirect_aggregate_param_ptx);
    ok &= expect(indirect_aggregate_param.ok &&
                     indirect_aggregate_param.source.find(".field2") !=
                         std::string::npos,
                 "typed PTX resolves CUDA Clang 21 generic memory and indirect ld.param addresses");
    if (!indirect_aggregate_param.ok) {
        std::cerr << indirect_aggregate_param.error << "\n";
    }

    const std::string symbolic_printf_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry symbolic_printf() {
    call.uni vprintf, (format_symbol, 7);
    ret;
}
)ptx";
    const metal::PtxToMslResult symbolic_printf =
        metal::compile_ptx_to_msl(symbolic_printf_ptx);
    ok &= expect(!symbolic_printf.ok &&
                     symbolic_printf.error.find("literal format") !=
                         std::string::npos,
                 "typed PTX rejects unresolved printf formats instead of emitting a fallback");

    if (!ok) return 1;
    std::cout << "PTX -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
