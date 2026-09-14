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

    // Strict mode checks the selected entry and every reachable helper.
    const std::string entry_scope_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry good() {
    ret;
}
.visible .entry bad() {
    unknown.b32 %r1, 0;
    ret;
}
.func helper() {
    unknown.b32 %r1, 0;
    ret;
}
.visible .entry caller() {
    call helper, ();
    ret;
}
)ptx";
    metal::PtxToMslOptions strict_options;
    strict_options.strict = true;
    strict_options.entry_name = "good";
    ok &= expect(metal::compile_ptx_to_msl(entry_scope_ptx, strict_options).ok,
                 "unrelated unsupported entry does not fail strict compilation");
    for (const auto* name : {"bad", "caller"}) {
        strict_options.entry_name = name;
        const auto invalid = metal::compile_ptx_to_msl(entry_scope_ptx, strict_options);
        ok &= expect(!invalid.ok && invalid.error.find("unknown.b32") != std::string::npos,
                     "strict compilation rejects unsupported reachable instructions");
    }

    const std::string funnel_ptx = R"ptx(
.version 7.0
.target sm_80
.visible .entry funnel(.param .u64 output) {
    .reg .b64 %rd1;
    .reg .b32 %r<4>;
    ld.param.u64 %rd1, [output];
    mov.u32 %r1, %tid.x;
    shf.l.wrap.b32 %r2, 305419896, 2596069104, %r1;
    shf.r.wrap.b32 %r3, 305419896, 2596069104, %r1;
    st.global.v4.b32 [%rd1], {%r2, -1, %r3, 0};
    st.global.b8 [%rd1+15], %r2;
    st.global.b16 [%rd1+12], %r3;
    ret;
}
)ptx";
    strict_options.entry_name = "funnel";
    const auto funnel = metal::compile_ptx_to_msl(funnel_ptx, strict_options);
    ok &= expect(funnel.ok, "wrapped funnel shifts and mixed literal vector stores compile: " + funnel.error);
    ok &= expect(funnel.source.find("uchar(") != std::string::npos &&
                 funnel.source.find("ushort(") != std::string::npos,
                 "narrow stores truncate wider PTX source registers");
    for (const auto* invalid_opcode : {"shf.l.clamp.b32", "shf.l.wrap.b64", "shf.l.extra.wrap.b32"}) {
        auto invalid_ptx = funnel_ptx;
        invalid_ptx.replace(invalid_ptx.find("shf.l.wrap.b32"), 14, invalid_opcode);
        ok &= expect(!metal::compile_ptx_to_msl(invalid_ptx, strict_options).ok,
                     "unsupported funnel shift variants fail");
    }
    for (const auto* tuple : {"{%r2, 0}", "{%r2, , %r3, 0}", "{%r2, 0, %r3,}"}) {
        auto invalid_ptx = funnel_ptx;
        const std::string original = "{%r2, -1, %r3, 0}";
        invalid_ptx.replace(invalid_ptx.find(original), original.size(), tuple);
        ok &= expect(!metal::compile_ptx_to_msl(invalid_ptx, strict_options).ok,
                     "malformed vector store tuples fail");
    }
    for (const auto* opcode : {"prmt.b32", "prmt.b32.f4e", "prmt.b64"}) {
        auto permute_ptx = funnel_ptx;
        permute_ptx.replace(permute_ptx.find("shf.l.wrap.b32"), 14, opcode);
        const auto compiled = metal::compile_ptx_to_msl(permute_ptx, strict_options);
        ok &= expect(compiled.ok == (std::string(opcode) == "prmt.b32"),
                     "only generic prmt.b32 is supported");
    }

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
    ok &= expect(result.source.find("reinterpret_cast<device cm_alias_float*>") !=
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
                     local_depot.source.find("reinterpret_cast<thread cm_alias_uint*>") !=
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
                         "reinterpret_cast<device cm_alias_uchar*>") !=
                         std::string::npos &&
                     module_global.source.find(" + 28)") != std::string::npos &&
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

    const std::string private_initialized_global_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.global .align 4 .b8 private_state[8] = {13, 0, 0, 0, 254, 255, 255, 255};
.visible .entry private_initialized_global(.param .u64 output) {
    .reg .b32 %r1;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    ld.global.b32 %r1, [private_state];
    add.u32 %r1, %r1, 4;
    st.global.b32 [private_state], %r1;
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult private_initialized_global =
        metal::compile_ptx_to_msl(private_initialized_global_ptx);
    const bool private_initialized_global_valid =
        private_initialized_global.ok &&
        private_initialized_global.source.find(
            "device uchar* cm___cumetal_global_private_state [[buffer(1)]]") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "cumetal-native-symbol: private-global private_state 8 4 0") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "cumetal-native-symbol-initializer: private_state 0d000000feffffff") !=
            std::string::npos &&
        private_initialized_global.source.find(
            "constant uchar cm_private_state") == std::string::npos;
    ok &= expect(
        private_initialized_global_valid,
        "module-private initialized PTX globals use persistent module-owned storage");
    if (!private_initialized_global_valid) {
        std::cerr << private_initialized_global.error << "\n"
                  << private_initialized_global.source << "\n";
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

    const std::string double_exp_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry double_exp(.param .u64 output) {
    .reg .b64 %rd<4>;
    .reg .f32 %f1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;
    cvt.f64.f32 %rd2, %f1;
    .param .b64 param0;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    call.uni (retval0), __nv_exp, (param0);
    ld.param.b64 %rd3, [retval0];
    st.global.b64 [%rd1], %rd3;
    ret;
}
)ptx";
    const metal::PtxToMslResult double_exp =
        metal::compile_ptx_to_msl(double_exp_ptx);
    ok &= expect(double_exp.ok &&
                     double_exp.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     double_exp.source.find("exp(as_type<float>") !=
                         std::string::npos &&
                     double_exp.source.find("cm_fp64_fast_f32_to_f64(") !=
                         std::string::npos &&
                     double_exp.source.find(
                         "cumetal-semantic-caveat: FP64 libdevice calls "
                         "evaluate through binary32") != std::string::npos,
                 "double-precision __nv_exp evaluates through binary32 with an "
                 "explicit caveat: " + double_exp.error);
    if (!double_exp.ok) std::cerr << double_exp.error << "\n";

    const std::string sincos_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry sincos_probe(.param .f32 input, .param .u64 output) {
    .local .align 4 .b8 sin_slot[4];
    .local .align 4 .b8 cos_slot[4];
    .reg .b32 %r1;
    .reg .b64 %rd<5>;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    mov.b64 %rd2, sin_slot;
    mov.b64 %rd3, cos_slot;
    .param .b32 param0;
    .param .b64 param1;
    .param .b64 param2;
    st.param.b32 [param0], %r1;
    st.param.b64 [param1], %rd2;
    st.param.b64 [param2], %rd3;
    call.uni __nv_sincosf, (param0, param1, param2);
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult sincos_probe =
        metal::compile_ptx_to_msl(sincos_ptx);
    ok &= expect(sincos_probe.ok &&
                     sincos_probe.source.find("= sin(") != std::string::npos &&
                     sincos_probe.source.find("= cos(") != std::string::npos,
                 "void __nv_sincosf writes both results through its output "
                 "pointers: " + sincos_probe.error);
    if (!sincos_probe.ok) std::cerr << sincos_probe.error << "\n";

    const std::string sincos_double_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry sincos_double(.param .f32 input, .param .u64 output) {
    .local .align 8 .b8 sin_slot[8];
    .local .align 8 .b8 cos_slot[8];
    .reg .b32 %r1;
    .reg .b64 %rd<6>;
    .reg .f32 %f1;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %r1;
    mov.b64 %rd3, sin_slot;
    mov.b64 %rd4, cos_slot;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 param2;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd3;
    st.param.b64 [param2], %rd4;
    call.uni __nv_sincos, (param0, param1, param2);
    st.global.b32 [%rd1], %r1;
    ret;
}
)ptx";
    const metal::PtxToMslResult sincos_double_probe =
        metal::compile_ptx_to_msl(sincos_double_ptx);
    ok &= expect(sincos_double_probe.ok &&
                     sincos_double_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     sincos_double_probe.source.find("sin(as_type<float>") !=
                         std::string::npos &&
                     sincos_double_probe.source.find("cos(as_type<float>") !=
                         std::string::npos &&
                     sincos_double_probe.source.find(
                         "cm_fp64_fast_f32_to_f64(") != std::string::npos,
                 "void __nv_sincos evaluates through binary32 and writes "
                 "binary64 storage through both output pointers: " +
                     sincos_double_probe.error);
    if (!sincos_double_probe.ok) {
        std::cerr << sincos_double_probe.error << "\n";
    } else if (sincos_double_probe.source.find("vf64_f64_to_f32(") ==
                   std::string::npos ||
               sincos_double_probe.source.find("sin(as_type<float>") ==
                   std::string::npos ||
               sincos_double_probe.source.find("cos(as_type<float>") ==
                   std::string::npos ||
               sincos_double_probe.source.find("cm_fp64_fast_f32_to_f64(") ==
                   std::string::npos) {
        std::cerr << sincos_double_probe.source << "\n";
    }

    const std::string modf_double_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry modf_double(.param .f32 input, .param .u64 output) {
    .local .align 8 .b8 integral_slot[8];
    .reg .b32 %r1;
    .reg .b64 %rd<6>;
    ld.param.f32 %r1, [input];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %r1;
    mov.b64 %rd3, integral_slot;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd3;
    call.uni (retval0), __nv_modf, (param0, param1);
    ld.param.b64 %rd4, [retval0];
    st.global.b64 [%rd1], %rd4;
    ret;
}
)ptx";
    const metal::PtxToMslResult modf_double_probe =
        metal::compile_ptx_to_msl(modf_double_ptx);
    ok &= expect(modf_double_probe.ok &&
                     modf_double_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     modf_double_probe.source.find("trunc(") !=
                         std::string::npos &&
                     modf_double_probe.source.find(
                         "cm_fp64_fast_f32_to_f64(") != std::string::npos,
                 "double __nv_modf stores a binary64 integral part and "
                 "returns the fraction through binary32: " +
                     modf_double_probe.error);
    if (!modf_double_probe.ok) {
        std::cerr << modf_double_probe.error << "\n";
    } else if (!ok) {
        std::cerr << modf_double_probe.source << "\n";
    }

    const std::string directed_ptx = R"ptx(
.version 7.0
.target sm_80
.address_size 64
.visible .entry directed_probe(.param .f32 input, .param .f32 input2,
                               .param .u64 output) {
    .reg .f32 %f<4>;
    .reg .b64 %rd<4>;
    ld.param.f32 %f1, [input];
    ld.param.f32 %f2, [input2];
    ld.param.u64 %rd1, [output];
    cvt.f64.f32 %rd2, %f1;
    .param .b64 param0;
    .param .b64 param1;
    .param .b64 retval0;
    .param .b32 param2;
    .param .b32 param3;
    .param .b32 retval1;
    st.param.b64 [param0], %rd2;
    st.param.b64 [param1], %rd2;
    call.uni (retval0), __nv_dadd_rd, (param0, param1);
    ld.param.b64 %rd3, [retval0];
    st.global.b64 [%rd1], %rd3;
    st.param.b32 [param2], %f1;
    st.param.b32 [param3], %f2;
    call.uni (retval1), __nv_fmul_ru, (param2, param3);
    ret;
}
)ptx";
    const metal::PtxToMslResult directed_probe =
        metal::compile_ptx_to_msl(directed_ptx);
    ok &= expect(directed_probe.ok &&
                     directed_probe.source.find("vf64_add_round(") !=
                         std::string::npos &&
                     directed_probe.source.find("vf64_mul_round(") !=
                         std::string::npos &&
                     directed_probe.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     directed_probe.source.find("cm_fp64_fast_f32_to_f64(") !=
                         std::string::npos,
                 "directed-rounding intrinsics lower through the "
                 "correctly-rounded vf64 ALU: " + directed_probe.error);
    if (!directed_probe.ok) {
        std::cerr << directed_probe.error << "\n";
    } else if (!ok) {
        std::cerr << directed_probe.source << "\n";
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


    std::string multiline_device_call_ptx = direct_device_call_ptx;
    const std::string one_line_call = "call.uni (retval0), add_one, (param0);";
    multiline_device_call_ptx.replace(multiline_device_call_ptx.find(one_line_call),
        one_line_call.size(), "call.uni (retval0),\n add_one,\n (\n param0\n );");
    const auto multiline_device_call = metal::compile_ptx_to_msl(multiline_device_call_ptx);
    ok &= expect(multiline_device_call.ok && multiline_device_call.source == direct_device_call.source,
                 "multiline direct device call produces identical MSL to the one-line form");
    if (!multiline_device_call.ok) std::cerr << multiline_device_call.error << "\n";


    std::string byte_return_ptx = direct_device_call_ptx;
    const std::string addition = "    add.u32 %r2, %r1, 1;";
    byte_return_ptx.replace(byte_return_ptx.find(addition), addition.size(),
        "    .local .align 1 .b8 scratch[1];\n"
        "    .reg .b64 %rd1;\n"
        "    mov.u64 %rd1, scratch;\n"
        "    st.local.u8 [%rd1], %r1;\n"
        "    ld.volatile.local.u8 %r2, [%rd1];");
    const auto byte_return = metal::compile_ptx_to_msl(byte_return_ptx);
    ok &= expect(byte_return.ok,
                 "unsigned byte load into a 32-bit register satisfies a 32-bit helper return");
    if (!byte_return.ok) std::cerr << byte_return.error << "\n";


    std::string mismatched_return_ptx = byte_return_ptx;
    const auto return_decl = mismatched_return_ptx.find(".param .b32 add_one_ret");
    mismatched_return_ptx.replace(return_decl, std::string(".param .b32 add_one_ret").size(),
                                  ".param .b64 add_one_ret");
    const auto mismatched_return = metal::compile_ptx_to_msl(mismatched_return_ptx);
    ok &= expect(!mismatched_return.ok &&
                 mismatched_return.error.find("does not fit its declared return type") != std::string::npos,
                 "load widening does not weaken the return ABI width check");


    const auto mixed_load = metal::compile_ptx_to_msl(R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry mixed_load(.param .u64 input) {
 .reg .b64 %rd<3>;
 .reg .b16 %rs1;
 .reg .b32 %r1;
 ld.param.u64 %rd1, [input];
 ld.global.v2.u8 {%rs1, %r1}, [%rd1];
 ld.global.u16 %rd2, [%rd1];
 ret;
}
)ptx");
    bool byte16 = false, byte32 = false, half64 = false;
    if (mixed_load.ok) {
        for (const auto& fn : mixed_load.gpu_ir.functions)
            for (const auto& block : fn.blocks)
                for (const auto& op : block.operations) {
                    if (op.opcode != ir::OpCode::kLoad || op.result_types.size() != 1 ||
                        !op.attributes.contains("memory_bit_width")) continue;
                    const auto bits = op.result_types[0].bit_width;
                    const auto memory = op.attributes.at("memory_bit_width");
                    byte16 |= memory == "8" && bits == 16;
                    byte32 |= memory == "8" && bits == 32;
                    half64 |= memory == "16" && bits == 64;
                }
    }
    ok &= expect(mixed_load.ok && byte16 && byte32 && half64,
                 "vector lanes and 64-bit destinations retain independent register and memory widths");


    const auto bfi_module = [](const std::string& instruction) {
        return ".version 7.1\n.target sm_80\n.address_size 64\n"
               ".visible .entry insert_bits(.param .u64 input) {\n"
               ".reg .b32 %r<3>;\n.reg .b64 %rd<3>;\n.reg .pred %p1;\n"
               "ld.param.u64 %rd1, [input];\n"
               "ld.global.u32 %r1, [%rd1];\nld.global.u64 %rd2, [%rd1];\n"
               "setp.eq.u32 %p1, %r1, 0;\n" + instruction + "\nret;\n}\n";
    };
    for (const auto* instruction : {"bfi.b32 %r2, %r1, 42, 3, 13;",
                                    "bfi.b64 %rd2, %rd2, 42, %r1, 64;"}) {
        const auto result = metal::compile_ptx_to_msl(bfi_module(instruction));
        ok &= expect(result.ok, "32/64-bit bit insertion supports register and immediate operands");
        if (!result.ok) std::cerr << result.error << "\n";
    }
    for (const auto* instruction : {"bfi.b16 %r2, %r1, 0, 3, 13;",
                                    "bfi.u32 %r2, %r1, 0, 3, 13;",
                                    "bfi.b32.extra %r2, %r1, 0, 3, 13;",
                                    "bfi.b32 %r2, %r1, 0, 3;",
                                    "bfi.b32 %r2, %rd2, 0, 3, 13;",
                                    "bfi.b32 %r2, %r1, 0, %rd2, 13;",
                                    "@%p1 bfi.b32 %r2, %r1, 0, 3, 13;"}) {
        const auto result = metal::compile_ptx_to_msl(bfi_module(instruction));
        ok &= expect(!result.ok && result.error.find("bfi") != std::string::npos,
                     "unsupported bit-insertion forms and mismatched operands fail explicitly");
    }


    const auto tuple_module = [](const std::string& instruction) {
        return ".version 7.1\n.target sm_80\n.address_size 64\n"
               ".visible .entry tuple_move(.param .u64 input) {\n"
               ".reg .b64 %rd1;\n.reg .b32 %r<3>;\n.reg .b16 %rs<5>;\n.reg .pred %p1;\n"
               "ld.param.u64 %rd1, [input];\nld.global.u32 %r1, [%rd1];\n"
               "cvt.u16.u32 %rs1, %r1;\nmov.u16 %rs2, 43981;\n"
               "setp.eq.u32 %p1, %r1, 0;\n" + instruction + "\nret;\n}\n";
    };
    for (const auto* instruction : {"mov.b32 %r2, {%rs1, %rs2};",
                                    "mov.b32 {%rs3, %rs4}, %r1;",
                                    "mov.b32 {_, %rs4}, %r1;",
                                    "mov.b32 {%rs3, _}, %r1;"}) {
        const auto result = metal::compile_ptx_to_msl(tuple_module(instruction));
        ok &= expect(result.ok, "mov.b32 halfword packing/unpacking and sink lanes compile");
        if (!result.ok) std::cerr << result.error << "\n";
    }
    for (const auto* instruction : {"mov.b32 %r2, {%rs1};",
                                    "mov.b32 %r2, {%rs1, %rs2, %rs1, %rs2};",
                                    "mov.b32 %r2, {%r1, %rs2};",
                                    "mov.b32 %r2, {%rs1, _};",
                                    "mov.b32 %r2, {%rs1, 7};",
                                    "mov.b32 {_, _}, %r1;",
                                    "mov.b32 {%rs3, %rs3}, %r1;",
                                    "mov.b32 {%rs3, %rs4}, %rd1;",
                                    "@%p1 mov.b32 %r2, {%rs1, %rs2};"}) {
        const auto result = metal::compile_ptx_to_msl(tuple_module(instruction));
        ok &= expect(!result.ok && result.error.find("mov.b32") != std::string::npos,
                     "unsupported/malformed mov.b32 tuples are rejected rather than scalarized");
    }

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
    call.uni
    recursive_helper,
    (
    );
    ret;
}
.visible .entry recursive_device_call() {
    call.uni
    recursive_helper,
    (
    );
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

    const std::string atomic_f32_ptx = R"ptx(
.version 8.8
.target sm_80
.address_size 64
.visible .entry atomic_f32(.param .u64 output) {
    .reg .f32 %f<3>;
    .reg .b64 %rd1;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3F800000;
    atom.global.add.f32 %f2, [%rd1], %f1;
    st.global.f32 [%rd1+4], %f2;
    ret;
}
)ptx";
    const metal::PtxToMslResult atomic_f32 =
        metal::compile_ptx_to_msl(atomic_f32_ptx);
    ok &= expect(atomic_f32.ok &&
                     atomic_f32.source.find(
                         "atomic_fetch_add_explicit(reinterpret_cast<device atomic_float*>") !=
                         std::string::npos,
                 "PTX atom.add.f32 lowers to Metal's native device atomic_float add: " +
                     atomic_f32.error);

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
                     atomic32.source.find(
                         "reinterpret_cast<device cm_alias_uchar*>") !=
                         std::string::npos &&
                     atomic32.source.find(" + 28)") != std::string::npos,
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
                     pointer_select.source.find("reinterpret_cast<device cm_alias_uint*>") !=
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
                         "reinterpret_cast<device cm_alias_char*>") != std::string::npos &&
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

    const std::string guarded_select = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry guarded_select(.param .u64 input, .param .u64 output, .param .u32 count) {
.reg .b64 %rd<6>;
.reg .b32 %r<12>;
.reg .pred %p<4>;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 4;
add.u64 %rd4, %rd1, %rd3;
add.u64 %rd5, %rd2, %rd3;
ld.global.u32 %r5, [%rd4];
and.b32 %r5, %r5, 7;
mov.u32 %r6, 0;
mov.u32 %r7, 0;
HEAD:
setp.lt.u32 %p2, %r6, %r5;
@%p2 bra SELECT;
st.global.u32 [%rd5], %r7;
bra DONE;
SELECT:
and.b32 %r8, %r6, 1;
setp.ne.u32 %p3, %r8, 0;
selp.b32 %r9, %r6, %r9, %p3;
add.u32 %r6, %r6, 1;
@%p3 bra USE;
bra HEAD;
USE:
add.u32 %r7, %r7, %r9;
bra HEAD;
DONE:
ret;
}
)ptx";
    const auto selected = metal::compile_ptx_to_msl(guarded_select);
    ok &= expect(selected.ok, "unobserved loop-carried select arm is eliminated: " + selected.error);
    std::string observable_select = guarded_select;
    const auto false_edge = observable_select.find("@%p3 bra USE;\nbra HEAD;");
    observable_select.replace(false_edge, std::string("@%p3 bra USE;\nbra HEAD;").size(),
        "@%p3 bra USE;\nst.global.u32 [%rd5], %r9;\nbra HEAD;");
    const auto observable = metal::compile_ptx_to_msl(observable_select);
    ok &= expect(!observable.ok && observable.error.find("undefined") != std::string::npos,
                 "observable undefined false arm remains rejected");
    for (const std::string replacement : {
        "@%p2 bra USE;", "@!%p3 bra USE;",
        "st.global.u32 [%rd5], %r9;\n@%p3 bra USE;",
        "setp.eq.u32 %p3, %r6, 0;\n@%p3 bra USE;"}) {
        std::string unsafe_select = guarded_select;
        unsafe_select.replace(unsafe_select.find("@%p3 bra USE;"),
                              std::string("@%p3 bra USE;").size(), replacement);
        const auto rejected = metal::compile_ptx_to_msl(unsafe_select);
        ok &= expect(!rejected.ok, "select rewrite requires matching unchanged predicate and no intervening use");
    }
    std::string tuple_select = guarded_select;
    tuple_select.insert(tuple_select.find(".reg .pred"), ".reg .b16 %rs1;\n.reg .b16 %rs2;\n");
    tuple_select.insert(tuple_select.find("selp.b32 %r9"), "mov.u16 %rs1, 1;\nmov.u16 %rs2, 2;\n");
    tuple_select.replace(tuple_select.find("%r6, %r9, %p3"), std::string("%r6, %r9, %p3").size(), "{%rs1,%rs2}, %r9, %p3");
    ok &= expect(!metal::compile_ptx_to_msl(tuple_select).ok, "malformed tuple cannot become a valid mov");
    for (const std::string false_path : {
        "bra USE;", "@%p3 mov.u32 %r9, 7;\nst.global.u32 [%rd5], %r9;\nbra HEAD;"}) {
        std::string crossing = guarded_select;
        crossing.replace(crossing.find("@%p3 bra USE;\nbra HEAD;"),
                         std::string("@%p3 bra USE;\nbra HEAD;").size(), "@%p3 bra USE;\n" + false_path);
        ok &= expect(!metal::compile_ptx_to_msl(crossing).ok,
                     "false-path joins and predicated kills cannot hide an undefined read");
    }
    std::string initialized = observable_select;
    initialized.insert(initialized.find("HEAD:"), "mov.u32 %r9, 99;\n");
    const auto initialized_result = metal::compile_ptx_to_msl(initialized);
    ok &= expect(initialized_result.ok && initialized_result.source.find(" ? ") != std::string::npos,
                 "observable but initialized false arm remains valid");
    std::string wide_select = guarded_select;
    wide_select.insert(wide_select.find(".reg .pred"), ".reg .b64 %rd90;\n.reg .b64 %rd91;\n");
    const std::string narrow = "selp.b32 %r9, %r6, %r9, %p3;";
    wide_select.replace(wide_select.find(narrow), narrow.size(),
        "cvt.u64.u32 %rd90, %r6;\nselp.b64 %rd91, %rd90, %rd91, %p3;");
    wide_select.insert(wide_select.find("add.u32 %r7, %r7, %r9;"), "cvt.u32.u64 %r9, %rd91;\n");
    ok &= expect(metal::compile_ptx_to_msl(wide_select).ok, "64-bit guarded self-select compiles");
    const std::string dead_pointer_cast = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.visible .entry dead_pointer_cast(.param .u64 input, .param .u64 output, .param .u32 count) {
.local .align 8 .b8 depot[8];
.reg .b64 %rd<10>;
.reg .b32 %r<7>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
add.u64 %rd4, %rd1, %rd3;
add.u64 %rd5, %rd2, %rd3;
ld.global.u64 %rd6, [%rd4];
mov.u64 %rd7, depot;
st.local.u64 [%rd7], %rd6;
cvt.u32.u64 %r5, %rd7;
ld.volatile.local.u64 %rd8, [%rd7];
st.global.u64 [%rd5], %rd8;
DONE:
ret;
}
)ptx";
    ok &= expect(metal::compile_ptx_to_msl(dead_pointer_cast).ok,
                 "unused pointer truncation does not require numeric Metal pointer semantics");
    for (const std::string use : {
        "st.global.u32 [%rd5], %r5;",
        "bra OBSERVE;\nOBSERVE:\nst.global.u32 [%rd5], %r5;",
        "setp.eq.u32 %p1, %r5, 0;\n@%p1 bra DONE;"}) {
        std::string observed = dead_pointer_cast;
        observed.insert(observed.find("ld.volatile.local.u64"), use + "\n");
        const auto rejected = metal::compile_ptx_to_msl(observed);
        ok &= expect(!rejected.ok && rejected.error.find("observable pointer-to-integer") != std::string::npos,
                     "observable truncation, cross-block and predicate uses stay rejected: " + rejected.error);
    }
    std::string merged_pointer_cast = dead_pointer_cast;
    merged_pointer_cast.insert(merged_pointer_cast.find("ld.volatile.local.u64"),
        "@%p1 bra ZERO;\nbra MERGE;\nZERO:\nmov.u32 %r5, 0;\nMERGE:\nst.global.u32 [%rd5], %r5;\n");
    const auto merged_cast = metal::compile_ptx_to_msl(merged_pointer_cast);
    ok &= expect(!merged_cast.ok && merged_cast.error.find("observable pointer-to-integer") != std::string::npos,
                 "pointer truncation passed as a CFG block argument remains observable");
    const std::string local_helper_probe = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.func (.param .b64 retvalue) local_helper(.param .b64 address) {
.reg .b64 %rd<4>;
ld.param.b64 %rd1, [address];
cvta.to.local.u64 %rd2, %rd1;
ld.local.u64 %rd3, [%rd2];
st.param.b64 [retvalue], %rd3;
ret;
}
.visible .entry local_helper_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.local .align 8 .b8 depot[8];
.reg .b64 %rd<10>;
.reg .b32 %r<7>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
add.u64 %rd4, %rd1, %rd3;
add.u64 %rd5, %rd2, %rd3;
ld.global.u64 %rd6, [%rd4];
mov.u64 %rd7, depot;
st.local.u64 [%rd7], %rd6;
cvta.local.u64 %rd9, %rd7;
.param .b64 arg;
.param .b64 value;
st.param.b64 [arg], %rd9;
call.uni (value), local_helper, (arg);
ld.param.b64 %rd8, [value];
st.global.u64 [%rd5], %rd8;
DONE:
ret;
}
)ptx";
    const auto local_helper_result = metal::compile_ptx_to_msl(local_helper_probe);
    ok &= expect(local_helper_result.ok, "cvta.to.local proves helper parameter pointer-ness: " + local_helper_result.error);
    for (const std::string argument : {"7", "%rd1"}) {
        std::string invalid = local_helper_probe;
        invalid.replace(invalid.find("st.param.b64 [arg], %rd9;"), std::string("st.param.b64 [arg], %rd9;").size(),
                        "st.param.b64 [arg], " + argument + ";");
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "integer and device pointers cannot be silently treated as local helper pointers");
    }
    const std::string wide_return = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.func (.param .align 8 .b8 retval[16]) tail_count(.param .b64 seed) {
st.param.b64 [retval], 1;
st.param.b64 [retval+8], 2;
ret;
}
.visible .entry tail_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.reg .b64 %rd<10>;
.reg .b32 %r<5>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
mul.wide.u32 %rd4, %r2, 16;
add.u64 %rd5, %rd1, %rd3;
add.u64 %rd6, %rd2, %rd4;
ld.global.u64 %rd7, [%rd5];
.param .b64 arg;
.param .align 8 .b8 result[16];
st.param.b64 [arg], %rd7;
call.uni (result), tail_count, (arg);
ld.param.b64 %rd8, [result];
ld.param.b64 %rd9, [result+8];
st.global.u64 [%rd6], %rd8;
st.global.u64 [%rd6+8], %rd9;
DONE:
ret;
}
)ptx";
    const auto wide_result = metal::compile_ptx_to_msl(wide_return);
    ok &= expect(wide_result.ok, "64-bit immediate stores populate byte-array returns: " + wide_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"[retval+8]", "[retval+4]"},
        {"[retval+8]", "[retval+16]"},
        {"[retval+8]", "[retval+unknown]"},
        {"[retval+8]", "[retval+8junk]"},
        {"[result+8]", "[result+unknown]"},
        {"[result+8]", "[result+8junk]"},
        {"[result+8]", "[result+4]"},
        {"[result+8]", "[result+16]"}}) {
        std::string invalid = wide_return;
        invalid.replace(invalid.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "malformed, overlapping, missing and out-of-bounds wide return fields rejected");
    }
    std::string vector_return = wide_return;
    for (std::size_t pos = 0; (pos = vector_return.find(".align 8 .b8", pos)) != std::string::npos; pos += 13)
        vector_return.replace(pos, 12, ".align 16 .b8");
    const std::string scalar_stores = "st.param.b64 [retval], 1;\nst.param.b64 [retval+8], 2;";
    vector_return.replace(vector_return.find(scalar_stores), scalar_stores.size(),
                          "st.param.v2.b64 [retval], {1, 2};");
    const std::string scalar_loads = "ld.param.b64 %rd8, [result];\nld.param.b64 %rd9, [result+8];";
    vector_return.replace(vector_return.find(scalar_loads), scalar_loads.size(),
                          "ld.param.v2.b64 {%rd8, %rd9}, [result];");
    const auto vector_result = metal::compile_ptx_to_msl(vector_return);
    ok &= expect(vector_result.ok, "both vector parameter lanes participate in aggregate ABI: " + vector_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"{1, 2}", "{1}"}, {"{1, 2}", "{1, unknown}"},
        {"{%rd8, %rd9}", "{%rd8, %rd8}"},
        {"{%rd8, %rd9}", "{%rd8, %r1}"},
        {"[result];", "[result+8];"},
        {"[result];", "[result+unknown];"},
        {"[result];", "[%rd8];"},
        {"st.param.v2.b64", "@%p1 st.param.v2.b64"}}) {
        std::string invalid = vector_return;
        invalid.replace(invalid.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "malformed, misaligned and predicated vector parameter transfers rejected");
    }
    const std::string scalar_tail = R"ptx(
.version 7.1
.target sm_80
.address_size 64
.func (.param .align 8 .b8 retval[16]) tail_count(.param .b64 seed) {
.reg .b64 %rd<7>;
.reg .pred %p1;
ld.param.u64 %rd1, [seed];
setp.eq.u64 %p1, %rd1, 0;
@%p1 bra BASE;
sub.u64 %rd2, %rd1, 1;
bra AGAIN;
BASE:
mov.u64 %rd3, 0;
mov.u64 %rd4, 81985529216486895;
bra RETURN;
AGAIN:
.param .b64 arg;
.param .align 8 .b8 result[16];
st.param.b64 [arg], %rd2;
call.uni (result), tail_count, (arg);
ld.param.b64 %rd3, [result];
ld.param.b64 %rd4, [result+8];
RETURN:
st.param.b64 [retval], %rd3;
st.param.b64 [retval+8], %rd4;
ret;
}
.visible .entry tail_probe(.param .u64 input, .param .u64 output, .param .u32 count) {
.reg .b64 %rd<10>;
.reg .b32 %r<5>;
.reg .pred %p1;
ld.param.u64 %rd1, [input];
ld.param.u64 %rd2, [output];
ld.param.u32 %r1, [count];
mov.u32 %r2, %ctaid.x;
mov.u32 %r3, %ntid.x;
mov.u32 %r4, %tid.x;
mad.lo.u32 %r2, %r2, %r3, %r4;
setp.ge.u32 %p1, %r2, %r1;
@%p1 bra DONE;
mul.wide.u32 %rd3, %r2, 8;
mul.wide.u32 %rd4, %r2, 16;
add.u64 %rd5, %rd1, %rd3;
add.u64 %rd6, %rd2, %rd4;
ld.global.u64 %rd7, [%rd5];
.param .b64 arg;
.param .align 8 .b8 result[16];
st.param.b64 [arg], %rd7;
call.uni (result), tail_count, (arg);
ld.param.b64 %rd8, [result];
ld.param.b64 %rd9, [result+8];
st.global.u64 [%rd6], %rd8;
st.global.u64 [%rd6+8], %rd9;
DONE:
ret;
}
)ptx";
    const auto tail_result = metal::compile_ptx_to_msl(scalar_tail);
    ok &= expect(tail_result.ok, "scalar aggregate-return tail call becomes a loop: " + tail_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"RETURN:\nst.param.b64 [retval], %rd3;", "RETURN:\nadd.u64 %rd3, %rd3, 1;\nst.param.b64 [retval], %rd3;"},
        {"ld.param.b64 %rd4, [result+8];", "ld.param.b64 %rd4, [result];"},
        {"st.param.b64 [retval+8], %rd4;", "st.param.b64 [retval+8], %rd3;"},
        {"call.uni (result), tail_count, (arg);", "@%p1 call.uni (result), tail_count, (arg);"},
        {"sub.u64 %rd2, %rd1, 1;", "ld.local.u64 %rd2, [%rd1];"},
        {"sub.u64 %rd2, %rd1, 1;", "add.u64 %rd2, depot, 0;"},
        {"st.param.b64 [arg], %rd2;", "st.param.b64 [arg+unknown], %rd2;"}}) {
        std::string unsupported = scalar_tail;
        unsupported.replace(unsupported.find(from), from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(unsupported);
        ok &= expect(!rejected.ok && rejected.error.find("recursive PTX") != std::string::npos,
                     "non-tail, malformed, predicated and memory-dependent recursion remain rejected: " + rejected.error);
    }
    const std::string trap_probe = R"ptx(
.version 7.1
.target sm_80
.visible .entry trap_probe(.param .u64 output, .param .u32 mode) {
.reg .b64 %rd1;
.reg .b32 %r1;
.reg .pred %p1;
ld.param.u64 %rd1, [output];
ld.param.u32 %r1, [mode];
setp.eq.u32 %p1, %r1, 1;
@%p1 bra FAIL;
st.global.u32 [%rd1], 1;
ret;
FAIL:
trap;
}
)ptx";
    const auto trap_result = metal::compile_ptx_to_msl(trap_probe);
    ok &= expect(trap_result.ok && trap_result.source.find("atomic_fetch_or_explicit(cm_trap_status") != std::string::npos &&
                     trap_result.source.find("atomic_load_explicit(cm_trap_status") != std::string::npos,
                 "kernel traps report failure and poll cancellation: " + trap_result.error);
    for (const auto& [from, to] : std::vector<std::pair<std::string, std::string>>{
        {"FAIL:\ntrap;", "FAIL:\nbar.sync 0;\ntrap;"},
        {"output", "cm_trap_status"}}) {
        std::string invalid = trap_probe;
        for (std::size_t at = 0; (at = invalid.find(from, at)) != std::string::npos; at += to.size())
            invalid.replace(at, from.size(), to);
        const auto rejected = metal::compile_ptx_to_msl(invalid);
        ok &= expect(!rejected.ok, "trap barriers and hidden argument collisions rejected");
    }
    const auto helper_trap = metal::compile_ptx_to_msl(R"ptx(
.version 7.1
.target sm_80
.func helper() {
trap;
}
.visible .entry helper_trap() {
call.uni helper, ();
ret;
}
)ptx");
    ok &= expect(helper_trap.ok && helper_trap.source.find("atomic_fetch_or_explicit(cm_trap_status") != std::string::npos,
                 "helper trap expands into kernel cancellation CFG: " + helper_trap.error);
    const std::string nested_barrier = R"ptx(
.version 7.1
.target sm_80
.func barrier_helper() {
bar.sync 0;
ret;
}
.func caller() {
call.uni barrier_helper, ();
trap;
}
.visible .entry nested_barrier() {
call.uni caller, ();
ret;
}
)ptx";
    const auto rejected_barrier = metal::compile_ptx_to_msl(nested_barrier);
    ok &= expect(!rejected_barrier.ok && rejected_barrier.error.find("without barriers or collectives") != std::string::npos,
                 "barriers in expanded call graphs remain rejected: " + rejected_barrier.error);
    std::string oversized = R"ptx(
.version 7.1
.target sm_80
.func noop(.param .u32 flag) {
.reg .b32 %r1;
.reg .pred %p1;
ld.param.u32 %r1, [flag];
setp.eq.u32 %p1, %r1, 1;
@%p1 bra FAIL;
ret;
FAIL:
trap;
}
.visible .entry oversized() {
.param .u32 flag;

)ptx";
    for (int i = 0; i < 1025; ++i) oversized += "st.param.u32 [flag], 0;\ncall.uni noop, (flag);\n";
    oversized += "trap;\n}\n";
    const auto rejected_size = metal::compile_ptx_to_msl(oversized);
    ok &= expect(!rejected_size.ok && rejected_size.error.find("bounded CFG size") != std::string::npos,
                 "trap call expansion refuses excessive code growth: " + rejected_size.error);
    if (!ok) return 1;
    std::cout << "PTX -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
