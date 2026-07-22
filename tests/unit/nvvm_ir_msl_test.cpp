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

constexpr const char* kNvvmSelectedCall = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @add_one(i32 %value) {
entry:
  %thread = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %incremented = add i32 %value, 1
  %result = add i32 %incremented, %thread
  ret i32 %result
}

define ptx_kernel void @selected(ptr %out) {
entry:
  call void @llvm.nvvm.bar.warp.sync(i32 -1)
  %value = call i32 @add_one(i32 41)
  store i32 %value, ptr %out, align 4
  ret void
}

define ptx_kernel void @unused(ptr %out) {
entry:
  store i32 7, ptr %out, align 4
  ret void
}

declare void @llvm.nvvm.bar.warp.sync(i32)
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
)llvm";

constexpr const char* kNvvmAggregateGep = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%row = type { i32, [4 x float] }

define ptx_kernel void @aggregate_gep(ptr %rows, i64 %row_index, i64 %column) {
entry:
  %element = getelementptr %row, ptr %rows, i64 %row_index, i32 1, i64 %column
  store float 1.0, ptr %element, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmConstantPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @choose(i1 %condition) {
entry:
  br i1 %condition, label %left, label %right
left:
  br label %merge
right:
  br label %merge
merge:
  %value = phi i32 [ 11, %left ], [ 22, %right ]
  ret i32 %value
}

define ptx_kernel void @constant_phi(ptr %out, i1 %condition) {
entry:
  %value = call i32 @choose(i1 %condition)
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmSharedReturnPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i32 @nested_returns(i1 %a, i1 %b) {
entry:
  br i1 %a, label %join, label %cont
cont:
  br i1 %b, label %left, label %right
left:
  br label %join
right:
  br label %join
join:
  %result = phi i32 [ 99, %entry ], [ 11, %left ], [ 22, %right ]
  ret i32 %result
}

define ptx_kernel void @shared_return_phi(ptr %out, i1 %a, i1 %b) {
entry:
  %result = call i32 @nested_returns(i1 %a, i1 %b)
  store i32 %result, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmConstantGlobal = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@table = addrspace(4) constant [2 x i32] [i32 287454020, i32 1432778632], align 4

define ptx_kernel void @constant_global(ptr %out, i64 %index) {
entry:
  %element = getelementptr [2 x i32], ptr addrspacecast (ptr addrspace(4) @table to ptr), i64 0, i64 %index
  %value = load i32, ptr %element, align 4
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmNoncanonicalLoop = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @noncanonical_loop(ptr %out, i32 %count) {
entry:
  br label %header
header:
  %index = phi i32 [ 0, %entry ], [ %next, %latch ]
  %odd = and i32 %index, 1
  %choose = icmp ne i32 %odd, 0
  br i1 %choose, label %left, label %right
left:
  %done = icmp uge i32 %index, %count
  br i1 %done, label %exit, label %latch
right:
  br label %latch
latch:
  %next = add i32 %index, 1
  br label %header
exit:
  store i32 %index, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmGenericDevicePointer = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @store_helper(ptr %out, i32 %value) {
entry:
  store i32 %value, ptr %out, align 4
  ret void
}

define ptx_kernel void @generic_device_pointer(ptr %out) {
entry:
  call void @store_helper(ptr %out, i32 42)
  ret void
}
)llvm";

constexpr const char* kNvvmMalformedPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @malformed_phi(ptr %out, i1 %condition) {
entry:
  br i1 %condition, label %left, label %right
left:
  br label %merge
right:
  br label %merge
merge:
  %value = phi i32 [ 11, %left ], [ 22, %entry ]
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmUndefPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @undef_phi(ptr %out, i1 %condition) {
entry:
  br i1 %condition, label %initialized, label %uninitialized
initialized:
  br label %merge
uninitialized:
  br label %merge
merge:
  %value = phi float [ 1.0, %initialized ], [ undef, %uninitialized ]
  store float %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmPoisonPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @poison_phi(ptr %out, i1 %condition) {
entry:
  br i1 %condition, label %initialized, label %invalid
initialized:
  br label %merge
invalid:
  br label %merge
merge:
  %value = phi i32 [ 1, %initialized ], [ poison, %invalid ]
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmInlineShuffle = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @inline_shuffle(ptr %out, i32 %value, i32 %lane) {
entry:
  %lane_id = call i32 asm "mov.u32 $0, %laneid;", "=r"()
  %source = xor i32 %lane, %lane_id
  %indexed = call i32 asm sideeffect "shfl.sync.idx.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,r"(i32 %value, i32 %source, i32 31, i32 -1)
  %down = call i32 asm sideeffect "shfl.sync.down.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,r"(i32 %indexed, i32 1, i32 31, i32 -1)
  %up = call i32 asm sideeffect "shfl.sync.up.b32 $0, $1, $2, $3, $4;", "=r,r,r,r,r"(i32 %down, i32 1, i32 31, i32 -1)
  store i32 %up, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmBitcast = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @__nv_float_as_uint(float)

define ptx_kernel void @bitcast_kernel(ptr %out, float %value) {
entry:
  %native = bitcast float %value to i32
  %cuda = call i32 @__nv_float_as_uint(float %value)
  %combined = xor i32 %native, %cuda
  store i32 %combined, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmMemcpy = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @memcpy_kernel(ptr %destination, ptr %source) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %destination, ptr align 4 %source, i64 12, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
)llvm";

constexpr const char* kNvvmValueReturningHelper = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define float @scale_add(float %value, float %scale, float %bias) {
entry:
  %scaled = fmul float %value, %scale
  %result = fadd float %scaled, %bias
  ret float %result
}

define ptx_kernel void @scale_add_kernel(ptr %out, float %value, float %scale, float %bias) {
entry:
  %result = call float @scale_add(float %value, float %scale, float %bias)
  store float %result, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmDynamicMemcpy = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @dynamic_memcpy(ptr %destination, ptr %source, i64 %size) {
entry:
  call void @llvm.memcpy.p0.p0.i64(ptr align 1 %destination, ptr align 1 %source, i64 %size, i1 false)
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
)llvm";

constexpr const char* kNvvmPointerAlignment = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @align_pointer(ptr %out, ptr %base) {
entry:
  %address = ptrtoint ptr %base to i64
  %biased = add i64 %address, 15
  %aligned = and i64 %biased, -16
  %pointer = inttoptr i64 %aligned to ptr
  store ptr %pointer, ptr %out, align 8
  ret void
}
)llvm";

constexpr const char* kNvvmHomogeneousAggregate = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%float4 = type { float, float, float, float }

define %float4 @make_float4(float %x, float %y, float %z, float %w) {
entry:
  %v0 = insertvalue %float4 poison, float %x, 0
  %v1 = insertvalue %float4 %v0, float %y, 1
  %v2 = insertvalue %float4 %v1, float %z, 2
  %v3 = insertvalue %float4 %v2, float %w, 3
  ret %float4 %v3
}

define ptx_kernel void @aggregate_kernel(ptr %out, float %x) {
entry:
  %vector = call %float4 @make_float4(float %x, float 2.0, float 3.0, float 4.0)
  %z = extractvalue %float4 %vector, 2
  store float %z, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmCudaMathBuiltins = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare float @__nv_fminf(float, float)
declare float @__nv_fmaxf(float, float)
declare float @__nv_sqrtf(float)
declare float @__nv_fabsf(float)
declare float @__nv_acosf(float)
declare i32 @__nv_popc(i32)
declare i32 @__nv_clz(i32)
declare i32 @__nv_ffs(i32)
declare i32 @__nv_abs(i32)
declare float @llvm.fabs.f32(float)

define ptx_kernel void @cuda_math(ptr %out, float %x, i32 %bits) {
entry:
  %minimum = call float @__nv_fminf(float %x, float 1.0)
  %maximum = call float @__nv_fmaxf(float %minimum, float 0.0)
  %root = call float @__nv_sqrtf(float %maximum)
  %magnitude = call float @__nv_fabsf(float %root)
  %intrinsic_magnitude = call float @llvm.fabs.f32(float %magnitude)
  %angle = call float @__nv_acosf(float %intrinsic_magnitude)
  %negative = fneg float %angle
  %population = call i32 @__nv_popc(i32 %bits)
  %leading = call i32 @__nv_clz(i32 %bits)
  %first = call i32 @__nv_ffs(i32 %bits)
  %absolute = call i32 @__nv_abs(i32 %bits)
  %sum0 = add i32 %population, %leading
  %sum1 = add i32 %first, %absolute
  %sum = add i32 %sum0, %sum1
  %wide = sitofp i32 %sum to float
  %result = fadd float %negative, %wide
  store float %result, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmNoaliasScope = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @noalias_scope(ptr %out) {
entry:
  call void @llvm.experimental.noalias.scope.decl(metadata !0)
  store i32 7, ptr %out, align 4
  ret void
}

declare void @llvm.experimental.noalias.scope.decl(metadata)
!0 = !{!1}
!1 = distinct !{!1, !2, !"scope"}
!2 = distinct !{!2, !"domain"}
)llvm";

constexpr const char* kNvvmThreadAlloca = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%Pair = type { i32, float }

define ptx_kernel void @thread_alloca(ptr %out, float %value) {
entry:
  %local = alloca %Pair, align 4
  %integer = getelementptr %Pair, ptr %local, i32 0, i32 0
  store i32 9, ptr %integer, align 4
  %floating = getelementptr %Pair, ptr %local, i32 0, i32 1
  store float %value, ptr %floating, align 4
  %loaded = load float, ptr %floating, align 4
  store float %loaded, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmNaturalLoop = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @natural_loop(ptr %out, i32 %count) {
entry:
  br label %header
header:
  %index = phi i32 [ 0, %entry ], [ %next, %store ]
  %active = icmp ult i32 %index, %count
  br i1 %active, label %body, label %exit
body:
  %low_bit = and i32 %index, 1
  %is_even = icmp eq i32 %low_bit, 0
  br i1 %is_even, label %even, label %odd
even:
  %doubled = mul i32 %index, 2
  br label %store
odd:
  %tripled = mul i32 %index, 3
  br label %store
store:
  %selected = phi i32 [ %doubled, %even ], [ %tripled, %odd ]
  %slot = getelementptr i32, ptr %out, i32 %index
  store i32 %selected, ptr %slot, align 4
  %next = add i32 %index, 1
  br label %header
exit:
  ret void
}
)llvm";

constexpr const char* kNvvmWarpVotes = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.vote.ballot.sync(i32, i1)
declare i1 @llvm.nvvm.vote.any.sync(i32, i1)
declare i1 @llvm.nvvm.vote.all.sync(i32, i1)
declare i32 @llvm.nvvm.activemask()

define ptx_kernel void @warp_votes(ptr %out, i32 %mask, i1 %predicate) {
entry:
  %ballot = call i32 @llvm.nvvm.vote.ballot.sync(i32 %mask, i1 %predicate)
  %any = call i1 @llvm.nvvm.vote.any.sync(i32 %mask, i1 %predicate)
  %all = call i1 @llvm.nvvm.vote.all.sync(i32 %mask, i1 %predicate)
  %active = call i32 @llvm.nvvm.activemask()
  %any_i32 = zext i1 %any to i32
  %all_i32 = zext i1 %all to i32
  %sum0 = add i32 %ballot, %active
  %sum1 = add i32 %any_i32, %all_i32
  %sum = add i32 %sum0, %sum1
  store i32 %sum, ptr %out, align 4
  ret void
}
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
    ok &= expect(result.source.find("device uchar* const") != std::string::npos &&
                     result.source.find("const device uchar*") == std::string::npos,
                 "SSA pointer bindings are const without making device pointees read-only");

    const metal::NvvmToMslResult selected =
        metal::compile_nvvm_to_msl(kNvvmSelectedCall, "selected.ll", "selected");
    ok &= expect(selected.ok, "selected NVVM kernel with a device call lowers");
    if (selected.ok) {
        ok &= expect(selected.gpu_ir.functions.size() == 2,
                     "only the selected kernel and reachable helper are imported");
        ok &= expect(selected.source.find("add_one") < selected.source.find("selected"),
                     "device helper is emitted before its caller");
        ok &= expect(selected.source.find("kernel void unused") == std::string::npos,
                     "unselected kernels are excluded from MSL");
        ok &= expect(selected.source.find("simdgroup_barrier") != std::string::npos,
                     "NVVM warp synchronization becomes a SIMD-group barrier");
        ok &= expect(selected.source.find("add_one(41, cm_thread_position)") !=
                         std::string::npos &&
                         selected.source.find("thread_position_in_threadgroup") !=
                             std::string::npos,
                     "transitive GPU builtins are threaded through device helper calls");
    }

    const metal::NvvmToMslResult missing =
        metal::compile_nvvm_to_msl(kNvvmSelectedCall, "selected.ll", "missing");
    ok &= expect(!missing.ok && missing.error.find("not found") != std::string::npos,
                 "missing selected NVVM kernels fail explicitly");

    const metal::NvvmToMslResult aggregate_gep =
        metal::compile_nvvm_to_msl(kNvvmAggregateGep, "aggregate-gep.ll",
                                    "aggregate_gep");
    ok &= expect(aggregate_gep.ok,
                 "nested aggregate GEP lowers through data-layout byte offsets");
    if (aggregate_gep.ok) {
        const std::string printed = ir::print(aggregate_gep.gpu_ir);
        ok &= expect(printed.find("pointer_offset") != std::string::npos &&
                         printed.find("4: i64") != std::string::npos,
                     "aggregate field offset is preserved in CuMetal IR");
    }

    ir::NvvmImportOptions phi_options;
    phi_options.source_name = "constant-phi.ll";
    phi_options.entry_name = "constant_phi";
    const ir::NvvmImportResult constant_phi =
        ir::import_nvvm_llvm_ir(kNvvmConstantPhi, phi_options);
    ok &= expect(constant_phi.ok,
                 "constant phi inputs are materialized on predecessor edges");

    const metal::NvvmToMslResult shared_return_phi =
        metal::compile_nvvm_to_msl(kNvvmSharedReturnPhi, "shared-return-phi.ll",
                                   "shared_return_phi");
    ok &= expect(shared_return_phi.ok &&
                     shared_return_phi.source.find("int nested_returns(") !=
                         std::string::npos &&
                     shared_return_phi.source.find("return v") != std::string::npos,
                 "shared value-return joins reconverge before emitting their PHI return");

    const metal::NvvmToMslResult constant_global =
        metal::compile_nvvm_to_msl(kNvvmConstantGlobal, "constant-global.ll",
                                   "constant_global");
    ok &= expect(constant_global.ok &&
                     constant_global.source.find("constant uchar table[8]") !=
                         std::string::npos &&
                     constant_global.source.find("0x44, 0x33, 0x22, 0x11") !=
                         std::string::npos,
                 "NVVM constant globals preserve their exact data-layout bytes in MSL");

    const metal::NvvmToMslResult noncanonical_loop =
        metal::compile_nvvm_to_msl(kNvvmNoncanonicalLoop,
                                   "noncanonical-loop.ll",
                                   "noncanonical_loop");
    ok &= expect(noncanonical_loop.ok &&
                     noncanonical_loop.source.find("switch (cm_block_state)") !=
                         std::string::npos &&
                     noncanonical_loop.source.find("while (true)") !=
                         std::string::npos,
                 "noncanonical nested loop exits lower through the typed CFG dispatcher");

    const metal::NvvmToMslResult generic_device_pointer =
        metal::compile_nvvm_to_msl(kNvvmGenericDevicePointer,
                                   "generic-device-pointer.ll",
                                   "generic_device_pointer");
    ok &= expect(generic_device_pointer.ok &&
                     generic_device_pointer.source.find(
                         "void store_helper(\n    device uchar* out") !=
                         std::string::npos &&
                     generic_device_pointer.source.find(
                         "reinterpret_cast<thread uchar*>(arg0)") ==
                         std::string::npos,
                 "generic helper pointers inherit concrete device address spaces from callers");

    const metal::NvvmToMslResult malformed_phi =
        metal::compile_nvvm_to_msl(kNvvmMalformedPhi, "malformed-phi.ll",
                                   "malformed_phi");
    ok &= expect(!malformed_phi.ok &&
                     malformed_phi.error.find("invalid LLVM/NVVM module") !=
                         std::string::npos,
                 "malformed LLVM PHIs fail verification instead of reaching importer assertions");

    const metal::NvvmToMslResult undef_phi =
        metal::compile_nvvm_to_msl(kNvvmUndefPhi, "undef-phi.ll", "undef_phi");
    ok &= expect(undef_phi.ok && undef_phi.source.find("if (") != std::string::npos &&
                     (undef_phi.source.find("float(0)") != std::string::npos ||
                      undef_phi.source.find("= 0;") != std::string::npos),
                 "diamond CFGs hoist PHIs and refine undef inputs to typed zero");

    const metal::NvvmToMslResult poison_phi =
        metal::compile_nvvm_to_msl(kNvvmPoisonPhi, "poison-phi.ll", "poison_phi");
    ok &= expect(!poison_phi.ok &&
                     poison_phi.error.find("phi incoming value is not representable") !=
                         std::string::npos,
                 "poison phi inputs remain an explicit diagnostic");

    const metal::NvvmToMslResult inline_shuffle = metal::compile_nvvm_to_msl(
        kNvvmInlineShuffle, "inline-shuffle.ll", "inline_shuffle");
    ok &= expect(inline_shuffle.ok &&
                     inline_shuffle.source.find("thread_index_in_simdgroup") !=
                         std::string::npos &&
                     inline_shuffle.source.find("simd_shuffle(") != std::string::npos &&
                     inline_shuffle.source.find("simd_shuffle_down(") != std::string::npos &&
                     inline_shuffle.source.find("simd_shuffle_up(") != std::string::npos,
                 "CUDA shuffle inline assembly lowers to direction-correct Metal SIMD intrinsics");

    const metal::NvvmToMslResult bitcast =
        metal::compile_nvvm_to_msl(kNvvmBitcast, "bitcast.ll", "bitcast_kernel");
    ok &= expect(bitcast.ok && bitcast.source.find("as_type<uint>(") != std::string::npos,
                 "LLVM and CUDA scalar bit reinterpretation lower through Metal as_type");

    const metal::NvvmToMslResult memcpy =
        metal::compile_nvvm_to_msl(kNvvmMemcpy, "memcpy.ll", "memcpy_kernel");
    ok &= expect(memcpy.ok &&
                     memcpy.source.find("reinterpret_cast<device uint*>") != std::string::npos,
                 "constant-length aligned LLVM memcpy expands into typed Metal loads and stores");

    const metal::NvvmToMslResult value_returning_helper = metal::compile_nvvm_to_msl(
        kNvvmValueReturningHelper, "scale_add.ll", "scale_add_kernel");
    ok &= expect(value_returning_helper.ok &&
                     value_returning_helper.source.find("float scale_add(") !=
                         std::string::npos &&
                     value_returning_helper.source.find("return v") != std::string::npos &&
                     value_returning_helper.source.find("= scale_add(") != std::string::npos,
                 "value-returning device helpers preserve their return operand in MSL");

    const metal::NvvmToMslResult dynamic_memcpy = metal::compile_nvvm_to_msl(
        kNvvmDynamicMemcpy, "dynamic-memcpy.ll", "dynamic_memcpy");
    ok &= expect(!dynamic_memcpy.ok &&
                     dynamic_memcpy.error.find("dynamic-length LLVM memcpy") != std::string::npos,
                 "dynamic-length LLVM memcpy remains an explicit diagnostic");

    const metal::NvvmToMslResult pointer_alignment = metal::compile_nvvm_to_msl(
        kNvvmPointerAlignment, "pointer-alignment.ll", "align_pointer");
    ok &= expect(pointer_alignment.ok &&
                     pointer_alignment.source.find("reinterpret_cast<ulong>") !=
                         std::string::npos &&
                     pointer_alignment.source.find("reinterpret_cast<device uchar*>") !=
                         std::string::npos,
                 "64-bit pointer alignment arithmetic preserves the Metal address space");

    const metal::NvvmToMslResult homogeneous_aggregate = metal::compile_nvvm_to_msl(
        kNvvmHomogeneousAggregate, "homogeneous-aggregate.ll", "aggregate_kernel");
    ok &= expect(homogeneous_aggregate.ok &&
                     homogeneous_aggregate.source.find("float4 make_float4(") !=
                         std::string::npos &&
                     homogeneous_aggregate.source.find("float4(") != std::string::npos &&
                     homogeneous_aggregate.source.find("[2]") != std::string::npos,
                 "homogeneous CUDA aggregates lower as native Metal vectors");

    const metal::NvvmToMslResult cuda_math =
        metal::compile_nvvm_to_msl(kNvvmCudaMathBuiltins, "cuda-math.ll", "cuda_math");
    ok &= expect(cuda_math.ok && cuda_math.source.find("fmin(") != std::string::npos &&
                     cuda_math.source.find("popcount(") != std::string::npos &&
                     cuda_math.source.find("ctz(") != std::string::npos &&
                     cuda_math.source.find(" ? ") != std::string::npos,
                 "CUDA math and bit-count declarations map to semantics-correct Metal builtins");

    const metal::NvvmToMslResult noalias_scope = metal::compile_nvvm_to_msl(
        kNvvmNoaliasScope, "noalias-scope.ll", "noalias_scope");
    ok &= expect(noalias_scope.ok &&
                     noalias_scope.source.find("noalias.scope") == std::string::npos,
                 "LLVM alias-analysis scope markers erase before Metal lowering");

    const metal::NvvmToMslResult thread_alloca = metal::compile_nvvm_to_msl(
        kNvvmThreadAlloca, "thread-alloca.ll", "thread_alloca");
    ok &= expect(thread_alloca.ok &&
                     thread_alloca.source.find("struct Pair") != std::string::npos &&
                     thread_alloca.source.find("Pair v") != std::string::npos &&
                     thread_alloca.source.find("reinterpret_cast<thread float*>") !=
                         std::string::npos,
                 "LLVM allocas lower to addressable thread-local Metal storage");

    const metal::NvvmToMslResult natural_loop = metal::compile_nvvm_to_msl(
        kNvvmNaturalLoop, "natural-loop.ll", "natural_loop");
    ok &= expect(natural_loop.ok &&
                     natural_loop.source.find("while (true)") != std::string::npos &&
                     natural_loop.source.find("break;") != std::string::npos &&
                     natural_loop.source.find("_next") != std::string::npos,
                 "natural loops preserve loop-carried PHIs through explicit Metal updates");

    const metal::NvvmToMslResult warp_votes =
        metal::compile_nvvm_to_msl(kNvvmWarpVotes, "warp-votes.ll", "warp_votes");
    ok &= expect(warp_votes.ok &&
                     warp_votes.source.find("simd_ballot(") != std::string::npos &&
                     warp_votes.source.find("simd_vote::vote_t(") != std::string::npos &&
                     warp_votes.source.find("simd_any(") != std::string::npos &&
                     warp_votes.source.find("simd_all(") != std::string::npos &&
                     warp_votes.source.find("thread_index_in_simdgroup") !=
                         std::string::npos,
                 "masked CUDA warp votes lower to Metal SIMD vote semantics");

    if (!ok) return 1;
    std::cout << "NVVM -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
