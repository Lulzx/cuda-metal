#include "cumetal/ir/nvvm_importer.h"
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

constexpr const char* kNvvm = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @vector_add(ptr %a, ptr %b, ptr %out, i32 %count) {
entry:
  %block = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
  %width = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
  %base = mul i32 %block, %width
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %index = add i32 %base, %thread
  %in_bounds = icmp slt i32 %index, %count
  br i1 %in_bounds, label %body, label %done

body:
  %wide = sext i32 %index to i64
  %pa = getelementptr float, ptr %a, i64 %wide
  %va = load float, ptr %pa, align 4
  %pb = getelementptr float, ptr %b, i64 %wide
  %vb = load float, ptr %pb, align 4
  %sum = fadd float %va, %vb
  %po = getelementptr float, ptr %out, i64 %wide
  store float %sum, ptr %po, align 4
  br label %done

done:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
)llvm";

}  // namespace

int main() {
    using namespace cumetal;
    if (!ir::llvm_frontend_available()) {
        std::cout << "LLVM frontend unavailable; stub behavior verified\n";
        const metal::NvvmToMslResult unavailable =
            metal::compile_nvvm_to_msl(kNvvm, "vector_add.ll");
        return unavailable.ok ? 1 : 0;
    }

    const metal::NvvmToMslResult result =
        metal::compile_nvvm_to_msl(kNvvm, "vector_add.ll");
    if (!result.ok) {
        std::cerr << result.error << "\n";
        return 1;
    }

    bool ok = true;
    ok &= expect(result.gpu_ir.attributes.at("frontend") == "nvvm",
                 "NVVM frontend is recorded in module metadata");
    ok &= expect(ir::print(result.gpu_ir).find("gpu.threadgroup_id") !=
                     std::string::npos,
                 "NVVM intrinsic normalizes into GPU semantics");
    ok &= expect(ir::print(result.gpu_ir).find("pointer_offset") !=
                     std::string::npos,
                 "LLVM GEP normalizes into typed pointer offsets");
    ok &= expect(result.source.find("kernel void vector_add") !=
                     std::string::npos,
                 "NVVM path reaches typed MSL");
    ok &= expect(result.source.find("if (!") != std::string::npos,
                 "LLVM branch is structurized as an early return");
    ok &= expect(result.source.find("[[thread_position_in_threadgroup]]") !=
                     std::string::npos,
                 "NVVM thread intrinsic becomes a Metal builtin");
    ok &= expect(result.source.find("int(") != std::string::npos &&
                     result.source.find(" < int(") != std::string::npos,
                 "signed LLVM comparisons preserve signed semantics in MSL");

    if (!ok) return 1;
    std::cout << "NVVM -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
