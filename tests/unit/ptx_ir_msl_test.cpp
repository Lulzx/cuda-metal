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

    if (!ok) return 1;
    std::cout << "PTX -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
