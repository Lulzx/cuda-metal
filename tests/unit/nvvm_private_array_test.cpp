#include "cumetal/ir/nvvm_importer.h"
#include "cumetal/metal/lower_to_msl.h"
#include <iostream>
#include <string>

namespace {
std::string fixture(const std::string &hint, const std::string &bound,
                    const std::string &extra = "", bool vol = false) {
    return R"(target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
define ptx_kernel void @private_array(ptr %input, ptr %output, i32 %count) {
entry:
  %array = alloca [64 x float], align 4
  br label %loop
loop:
  %i = phi i32 [0, %entry], [%next, %body]
  %more = icmp ult i32 %i, )" +
           bound + R"(
  br i1 %more, label %body, label %exit
body:
  %src = getelementptr float, ptr %input, i32 %i
  %value = load float, ptr %src
  %dst = getelementptr [64 x float], ptr %array, i32 0, i32 %i
  store )" +
           (vol ? "volatile " : "") + R"(float %value, ptr %dst
)" + extra +
           R"(
  %next = add nuw nsw i32 %i, 1
  br label %loop, !llvm.loop !0
exit:
  %p0 = getelementptr [64 x float], ptr %array, i32 0, i32 0
  %v0 = load float, ptr %p0
  %p1 = getelementptr [64 x float], ptr %array, i32 0, i32 1
  %v1 = load float, ptr %p1
  %sum = fadd float %v0, %v1
  store float %sum, ptr %output
  ret void
}
declare void @llvm.nvvm.barrier0()
!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.)" +
           hint + "\"}\n";
}

bool check(const std::string &source, bool expect_array, const char *label) {
    auto compiled = cumetal::metal::compile_nvvm_to_msl(source, label);
    if (!compiled.ok) {
        std::cerr << "FAIL " << label << ": " << compiled.error << '\n';
        return false;
    }
    bool array = false;
    for (const auto &function : compiled.gpu_ir.functions)
        for (const auto &block : function.blocks)
            for (const auto &op : block.operations)
                array |= op.opcode == cumetal::ir::OpCode::kAlloca;
    if (array != expect_array) {
        std::cerr << "FAIL " << label << ": unexpected private storage\n";
        return false;
    }
    return true;
}
} // namespace

int main() {
    if (!cumetal::ir::llvm_frontend_available())
        return 77;
    bool ok = true;
    ok &= check(fixture("enable", "4"), false, "hinted-four");
    ok &= check(fixture("full", "8"), false, "hinted-eight");
    std::string small = fixture("enable", "4");
    for (std::size_t pos = 0;
         (pos = small.find("[64 x float]", pos)) != std::string::npos;)
        small.replace(pos, 12, "[16 x float]");
    ok &= check(small, true, "small-array-left-to-metal");
    ok &= check(fixture("disable", "4"), true, "disabled");
    ok &= check(fixture("enable", "%count"), true, "dynamic");
    ok &= check(fixture("enable", "32"), true, "too-large");
    ok &= check(fixture("enable", "4", "", true), true, "volatile");
    ok &= check(fixture("enable", "4", "  call void @llvm.nvvm.barrier0()"),
                true, "barrier");
    std::string counted = fixture("enable", "4");
    const auto hint_pos = counted.find("!1 = !{");
    counted.replace(hint_pos, counted.size() - hint_pos,
                    "!1 = !{!\"llvm.loop.unroll.count\", i32 2}\n");
    ok &= check(counted, true, "explicit-unroll-count");
    std::string expensive;
    for (int i = 0; i < 260; ++i)
        expensive += "  %pad" + std::to_string(i) + " = add i32 %i, 1\n";
    ok &= check(fixture("enable", "4", expensive), true, "growth-budget");
    std::string mixed = fixture("enable", "4");
    const auto exit_pos = mixed.find("exit:\n");
    mixed.insert(exit_pos + 6, R"(
  br label %second
second:
  %j = phi i32 [0, %exit], [%jnext, %secondbody]
  %jmore = icmp ult i32 %j, 4
  br i1 %jmore, label %secondbody, label %finish
secondbody:
  %jd = getelementptr [64 x float], ptr %array, i32 0, i32 %j
  store float 2.0, ptr %jd
  %jnext = add nuw nsw i32 %j, 1
  br label %second, !llvm.loop !2
finish:
)");
    mixed += "!2 = distinct !{!2, !3}\n!3 = !{!\"llvm.loop.unroll.disable\"}\n";
    ok &= check(mixed, true, "unrelated-disable-survives");
    const auto shared =
        cumetal::metal::compile_nvvm_to_msl(R"(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@tile = internal addrspace(3) global [16 x float] undef

define ptx_kernel void @shared_offset(ptr %out, i64 %index) {
  %p = getelementptr float, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(3) @tile to ptr), i64 16), i64 %index
  store float 3.0, ptr %p
  %v = load float, ptr %p
  store float %v, ptr %out
  ret void
}
)",
                                            "shared-constant-gep");
    if (!shared.ok ||
        shared.source.find("getelementptr") != std::string::npos ||
        shared.source.find("threadgroup") == std::string::npos) {
        std::cerr << "FAIL shared constant GEP: " << shared.error << '\n';
        ok = false;
    }
    if (ok)
        std::cout << "PASS: bounded private-array unrolling and shared "
                     "constant GEP\n";
    return ok ? 0 : 1;
}
