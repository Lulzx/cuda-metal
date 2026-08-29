#include "cumetal/ir/nvvm_importer.h"
#include "cumetal/metal/lower_to_msl.h"

#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool has_duplicate_ssa_declaration(const std::string& source) {
    std::istringstream lines(source);
    std::unordered_set<std::string> declarations;
    std::string line;
    while (std::getline(lines, line)) {
        const bool top_level_statement = line.starts_with("    ") &&
                                         !line.starts_with("        ");
        const bool ssa_declaration = top_level_statement && line.ends_with(';') &&
                                     line.find(" v") != std::string::npos &&
                                     line.find('=') == std::string::npos;
        if (ssa_declaration && !declarations.insert(line).second) return true;
    }
    return false;
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

constexpr const char* kNvvmPrintf = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%printf_args = type { i32, i32, i32 }
@.str = private unnamed_addr constant [18 x i8] c"PRINTF[%d,%d]=%d\0A\00"

define ptx_kernel void @typed_printf(i32 %value) {
entry:
  %args = alloca %printf_args, align 8
  %a = getelementptr %printf_args, ptr %args, i32 0, i32 0
  store i32 2, ptr %a, align 8
  %b = getelementptr %printf_args, ptr %args, i32 0, i32 1
  store i32 7, ptr %b, align 4
  %c = getelementptr %printf_args, ptr %args, i32 0, i32 2
  store i32 %value, ptr %c, align 8
  %result = call i32 @vprintf(ptr @.str, ptr %args)
  ret void
}

declare i32 @vprintf(ptr, ptr)
)llvm";

constexpr const char* kNvvmMalformedPrintf = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@.str = private unnamed_addr constant [10 x i8] c"value=%d\0A\00"

define ptx_kernel void @bad_printf() {
entry:
  %args = alloca i16, align 2
  store i16 7, ptr %args, align 2
  %result = call i32 @vprintf(ptr @.str, ptr %args)
  ret void
}

declare i32 @vprintf(ptr, ptr)
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

constexpr const char* kNvvmNestedMultiExitLoop = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.bar.warp.sync(i32)

define ptx_kernel void @nested_multi_exit_loop(ptr %out, i32 %count) {
entry:
  br label %outer.header
outer.header:
  %outer = phi i32 [ 0, %entry ], [ %outer.next, %inner.escape ]
  %outer.active = icmp ult i32 %outer, %count
  br i1 %outer.active, label %inner.preheader, label %exit
inner.preheader:
  br label %inner.header
inner.header:
  %inner = phi i32 [ 0, %inner.preheader ], [ %inner.next, %inner.latch ]
  call void @llvm.nvvm.bar.warp.sync(i32 -1)
  %odd = and i32 %inner, 1
  %choose = icmp ne i32 %odd, 0
  br i1 %choose, label %left, label %right
left:
  br label %decision
right:
  br label %decision
decision:
  %escape = icmp eq i32 %inner, 3
  br i1 %escape, label %inner.escape, label %inner.latch
inner.latch:
  %inner.next = add i32 %inner, 1
  %repeat = icmp ult i32 %inner.next, 5
  br i1 %repeat, label %inner.header, label %exit
inner.escape:
  %outer.next = add i32 %outer, 1
  %continue.outer = icmp ult i32 %outer.next, %count
  br i1 %continue.outer, label %outer.header, label %exit
exit:
  store i32 %outer, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmIrreducibleBarrier = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare void @llvm.nvvm.bar.warp.sync(i32)

define internal void @barrier_helper() {
entry:
  call void @llvm.nvvm.bar.warp.sync(i32 -1)
  ret void
}

define ptx_kernel void @irreducible_barrier(ptr %out, i1 %first, i1 %next) {
entry:
  br i1 %first, label %left, label %right
left:
  br label %cycle
right:
  br label %cycle
cycle:
  call void @barrier_helper()
  br i1 %next, label %left, label %right
}
)llvm";

constexpr const char* kNvvmIrreducibleDispatcher = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @irreducible_dispatcher(ptr %out, i1 %first, i1 %next) {
entry:
  %seed = add i32 1, 2
  br i1 %first, label %left, label %right
left:
  %left.value = add i32 %seed, 3
  br label %cycle
right:
  %right.value = add i32 %seed, 4
  br label %cycle
cycle:
  %value = phi i32 [ %left.value, %left ], [ %right.value, %right ]
  store i32 %value, ptr %out, align 4
  br i1 %next, label %left, label %right
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

constexpr const char* kNvvmKernelDescriptorPointer = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%descriptor = type { ptr, i32 }

define ptr @descriptor_element(ptr %descriptor, i64 %index) {
entry:
  %field = getelementptr %descriptor, ptr %descriptor, i64 0, i32 0
  %data = load ptr, ptr %field, align 8
  %element = getelementptr float, ptr %data, i64 %index
  ret ptr %element
}

define ptx_kernel void @kernel_descriptor_pointer(ptr byval(%descriptor) %descriptor) {
entry:
  %element = call ptr @descriptor_element(ptr %descriptor, i64 0)
  store float 1.0, ptr %element, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmStaticThreadgroupGlobal = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@shared_bytes = internal addrspace(3) global [32 x i8] undef, align 16

define void @shared_helper(i32 %value) {
entry:
  %slot = getelementptr [32 x i8], ptr addrspacecast (ptr addrspace(3) @shared_bytes to ptr), i64 0, i64 4
  store i32 %value, ptr %slot, align 4
  ret void
}

define ptx_kernel void @static_threadgroup_global(i32 %value) {
entry:
  call void @shared_helper(i32 %value)
  ret void
}
)llvm";

constexpr const char* kNvvmDynamicThreadgroupGlobal = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@dynamic_bytes = external addrspace(3) global [0 x i8], align 16

define void @dynamic_helper(i32 %value) {
entry:
  %slot = getelementptr [0 x i8], ptr addrspacecast (ptr addrspace(3) @dynamic_bytes to ptr), i64 0, i64 4
  store i32 %value, ptr %slot, align 4
  ret void
}

define ptx_kernel void @dynamic_threadgroup_global(i32 %value) {
entry:
  call void @dynamic_helper(i32 %value)
  ret void
}
)llvm";

constexpr const char* kNvvmMultipleDynamicThreadgroupGlobals = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@dynamic_a = external addrspace(3) global [0 x i8], align 4
@dynamic_b = external addrspace(3) global [0 x i8], align 4

define ptx_kernel void @multiple_dynamic_threadgroup_globals() {
entry:
  %a = getelementptr [0 x i8], ptr addrspacecast (ptr addrspace(3) @dynamic_a to ptr), i64 0, i64 0
  %b = getelementptr [0 x i8], ptr addrspacecast (ptr addrspace(3) @dynamic_b to ptr), i64 0, i64 0
  store i8 1, ptr %a, align 1
  store i8 2, ptr %b, align 1
  ret void
}
)llvm";

constexpr const char* kNvvmHelperDefinedAfterKernel = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @kernel_before_helper(ptr %output) {
entry:
  %value = call i32 @later_helper(i32 41)
  store i32 %value, ptr %output, align 4
  ret void
}

define i32 @later_helper(i32 %value) {
entry:
  %result = add i32 %value, 1
  ret i32 %result
}
)llvm";

constexpr const char* kNvvmMixedDeviceThreadgroupPhi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@mixed_shared = internal addrspace(3) global [16 x i8] undef, align 4

define i32 @read_generic_word(ptr %input) {
entry:
  %value = load i32, ptr %input, align 4
  ret i32 %value
}

define ptx_kernel void @mixed_device_threadgroup_phi(ptr %device_input, ptr %output,
                                                      i1 %use_device) {
entry:
  br i1 %use_device, label %from_device, label %from_threadgroup

from_device:
  br label %join

from_threadgroup:
  %shared = getelementptr [16 x i8], ptr addrspacecast (ptr addrspace(3) @mixed_shared to ptr), i64 0, i64 0
  br label %join

join:
  %input = phi ptr [ %device_input, %from_device ], [ %shared, %from_threadgroup ]
  %value = call i32 @read_generic_word(ptr %input)
  store i32 %value, ptr %output, align 4
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

constexpr const char* kNvvmMemset = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @memset_kernel(ptr %destination) {
entry:
  call void @llvm.memset.p0.i64(ptr align 4 %destination, i8 90, i64 12, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
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

constexpr const char* kNvvmIsolatedFp64Multiply = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @isolated_fp64_multiply(ptr %out, float %value) {
entry:
  %wide = fpext float %value to double
  %scaled = fmul double %wide, 1.000000e-02
  %result = fptrunc double %scaled to float
  store float %result, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmUnsupportedFp64 = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @unsupported_fp64(ptr %out, float %value) {
entry:
  %wide = fpext float %value to double
  %sum = fadd double %wide, 1.000000e+00
  %result = fptrunc double %sum to float
  store float %result, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmFloatFrexpViaDoubleAbi = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare double @__nv_frexp(double, ptr)

define ptx_kernel void @float_frexp_via_double_abi(ptr %out, ptr %exponent_out, float %value) {
entry:
  %exponent = alloca i32, align 4
  %wide = fpext float %value to double
  %mantissa.wide = call double @__nv_frexp(double %wide, ptr %exponent)
  %mantissa = fptrunc double %mantissa.wide to float
  %exponent.value = load i32, ptr %exponent, align 4
  store float %mantissa, ptr %out, align 4
  store i32 %exponent.value, ptr %exponent_out, align 4
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

constexpr const char* kNvvmDynamicMemset = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @dynamic_memset(ptr %destination, i64 %size) {
entry:
  call void @llvm.memset.p0.i64(ptr align 1 %destination, i8 0, i64 %size, i1 false)
  ret void
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
)llvm";

constexpr const char* kNvvmCudaLegacyAtomic = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cuda_legacy_atomic(ptr %counter, ptr %old_value, i32 %increment) {
entry:
  %old = atomicrmw add ptr %counter, i32 %increment seq_cst, align 4
  store i32 %old, ptr %old_value, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmUnsupportedAtomic = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @unsupported_atomic(ptr %counter) {
entry:
  %old = atomicrmw xor ptr %counter, i32 1 monotonic, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmCudaCmpXchg = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @cuda_cmpxchg(ptr %counter, ptr %old_out, ptr %success_out) {
entry:
  %pair = cmpxchg ptr %counter, i32 7, i32 9 seq_cst monotonic, align 4
  %old = extractvalue { i32, i1 } %pair, 0
  %success = extractvalue { i32, i1 } %pair, 1
  store i32 %old, ptr %old_out, align 4
  store i1 %success, ptr %success_out, align 1
  ret void
}
)llvm";

constexpr const char* kNvvmCudaCmpXchg64Helper = R"llvm(
target datalayout = "e-p6:32:32-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define i64 @cas64_helper(ptr %counter, i64 %compare, i64 %desired) {
entry:
  %pair = cmpxchg ptr %counter, i64 %compare, i64 %desired seq_cst monotonic, align 8
  %old = extractvalue { i64, i1 } %pair, 0
  ret i64 %old
}

define ptx_kernel void @cuda_cmpxchg64_helper(ptr %counter, ptr %old_out) {
entry:
  %old = call i64 @cas64_helper(ptr %counter, i64 7, i64 9)
  store i64 %old, ptr %old_out, align 8
  ret void
}
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
declare float @__nv_rsqrtf(float)
declare float @__nv_expf(float)
declare float @__nv_fast_expf(float)
declare float @__nv_exp2f(float)
declare float @__nv_exp10f(float)
declare float @__nv_expm1f(float)
declare float @__nv_logf(float)
declare float @__nv_log2f(float)
declare float @__nv_sinf(float)
declare float @__nv_cosf(float)
declare float @__nv_tanhf(float)
declare float @__nv_floorf(float)
declare float @__nv_ceilf(float)
declare float @__nv_truncf(float)
declare float @__nv_roundf(float)
declare float @__nv_powf(float, float)
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
  %inverse_root = call float @__nv_rsqrtf(float %maximum)
  %exponential = call float @__nv_expf(float %angle)
  %fast_exponential = call float @__nv_fast_expf(float %exponential)
  %binary_exponential = call float @__nv_exp2f(float %fast_exponential)
  %decimal_exponential = call float @__nv_exp10f(float %binary_exponential)
  %unit_exponential = call float @__nv_expm1f(float %decimal_exponential)
  %logarithm = call float @__nv_logf(float %unit_exponential)
  %binary_logarithm = call float @__nv_log2f(float %binary_exponential)
  %sine = call float @__nv_sinf(float %binary_logarithm)
  %cosine = call float @__nv_cosf(float %sine)
  %hyperbolic = call float @__nv_tanhf(float %cosine)
  %floored = call float @__nv_floorf(float %hyperbolic)
  %ceiled = call float @__nv_ceilf(float %floored)
  %truncated = call float @__nv_truncf(float %ceiled)
  %rounded = call float @__nv_roundf(float %truncated)
  %powered = call float @__nv_powf(float %rounded, float %inverse_root)
  %negative = fneg float %powered
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

constexpr const char* kNvvmHeterogeneousAggregate = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
%mixed = type { i32, float, i64 }

define %mixed @make_mixed(i32 %integer, float %real) {
entry:
  %v0 = insertvalue %mixed poison, i32 %integer, 0
  %v1 = insertvalue %mixed %v0, float %real, 1
  %v2 = insertvalue %mixed %v1, i64 7, 2
  ret %mixed %v2
}

define ptx_kernel void @heterogeneous_aggregate(ptr %out, i32 %integer, float %real) {
entry:
  %value = call %mixed @make_mixed(i32 %integer, float %real)
  %field = extractvalue %mixed %value, 1
  store float %field, ptr %out, align 4
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

constexpr const char* kNvvmSequentialLoops = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @_Z12__assert_rtnPKcS0_iS0_(ptr %function, ptr %file, i32 %line, ptr %expression) {
entry:
  ret void
}

define ptx_kernel void @sequential_loops(ptr %out, i32 %count) {
entry:
  %invalid = icmp slt i32 %count, 0
  br i1 %invalid, label %assert, label %loop1
assert:
  call void @_Z12__assert_rtnPKcS0_iS0_(ptr null, ptr null, i32 1, ptr null)
  br label %loop1
loop1:
  %i = phi i32 [ 0, %entry ], [ 0, %assert ], [ %next_i, %body1 ]
  %more_i = icmp slt i32 %i, 2
  br i1 %more_i, label %body1, label %loop2
body1:
  %next_i = add i32 %i, 1
  br label %loop1
loop2:
  %j = phi i32 [ 0, %loop1 ], [ %next_j, %body2 ]
  %more_j = icmp slt i32 %j, 3
  br i1 %more_j, label %body2, label %done
body2:
  %next_j = add i32 %j, 1
  br label %loop2
done:
  store i32 %j, ptr %out, align 4
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

constexpr const char* kNvvmInlineActiveMask = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @inline_active_mask(ptr %out) {
entry:
  %active = call i32 asm "mov.u32 $0, %activemask;", "=r"()
  store i32 %active, ptr %out, align 4
  ret void
}

!nvvm.annotations = !{!0}
!0 = !{ptr @inline_active_mask, !"kernel", i32 1}
)llvm";

}  // namespace

constexpr const char* kNvvmExternalSymbols = R"llvm(
target datalayout = "e-p6:32:32-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

@constants = dso_local addrspace(4) externally_initialized constant [8 x i32] zeroinitializer, align 4
@writable = dso_local addrspace(1) externally_initialized global [8 x i32] zeroinitializer, align 4
@embedded = dso_local addrspace(4) externally_initialized constant [2 x i32] [i32 287454020, i32 1432778632], align 4

define ptx_kernel void @read_constants(ptr %out) {
entry:
  %value = load i32, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(4) @constants to ptr), i64 16), align 4
  store i32 %value, ptr %out, align 4
  ret void
}

define ptx_kernel void @write_global(ptr %out) {
entry:
  %value = load i32, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(1) @writable to ptr), i64 12), align 4
  %next = add i32 %value, 1
  store i32 %next, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(1) @writable to ptr), i64 12), align 4
  store i32 %next, ptr %out, align 4
  ret void
}

define i32 @read_symbols_helper() {
entry:
  %constant = load i32, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(4) @constants to ptr), i64 16), align 4
  %global = load i32, ptr getelementptr (i8, ptr addrspacecast (ptr addrspace(1) @writable to ptr), i64 12), align 4
  %table = load i32, ptr addrspacecast (ptr addrspace(4) @embedded to ptr), align 4
  %sum0 = add i32 %constant, %global
  %sum1 = add i32 %sum0, %table
  ret i32 %sum1
}

define ptx_kernel void @read_symbols_through_helper(ptr %out) {
entry:
  %value = call i32 @read_symbols_helper()
  store i32 %value, ptr %out, align 4
  ret void
}
)llvm";

constexpr const char* kNvvmOversizedExternalConstant = R"llvm(
target datalayout = "e-p6:32:32-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"
@too_large = addrspace(4) externally_initialized constant [65537 x i8] zeroinitializer, align 1
define ptx_kernel void @read_large(ptr %out) {
entry:
  %value = load i8, ptr addrspacecast (ptr addrspace(4) @too_large to ptr), align 1
  store i8 %value, ptr %out, align 1
  ret void
}
)llvm";

constexpr const char* kNvvmClockAndGridSyncHelper = R"llvm(
target datalayout = "e-p:64:64-i64:64-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @clock_and_grid_sync(ptr %out) {
entry:
  %tick = call i32 @llvm.nvvm.read.ptx.sreg.clock()
  store i32 %tick, ptr %out, align 4
  call void @__cumetal_grid_sync()
  ret void
}

define ptx_kernel void @clock_grid_kernel(ptr %out) {
entry:
  call void @clock_and_grid_sync(ptr %out)
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.clock()
declare void @__cumetal_grid_sync()
)llvm";

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
    ok &= expect(result.source.find("device uchar*") != std::string::npos &&
                     result.source.find("const device uchar*") == std::string::npos,
                 "hoisted SSA pointer storage does not make device pointees read-only");

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
                     noncanonical_loop.source.find("switch (cm_block_state)") ==
                         std::string::npos &&
                     noncanonical_loop.source.find("while (true)") !=
                         std::string::npos,
                 "noncanonical loop exits lower as structured control flow");

    const metal::NvvmToMslResult nested_multi_exit_loop =
        metal::compile_nvvm_to_msl(kNvvmNestedMultiExitLoop,
                                   "nested-multi-exit-loop.ll",
                                   "nested_multi_exit_loop");
    ok &= expect(nested_multi_exit_loop.ok,
                 "nested multi-exit loop lowers: " +
                     nested_multi_exit_loop.error);
    if (nested_multi_exit_loop.ok) {
        ok &= expect(
            nested_multi_exit_loop.source.find("switch (cm_block_state)") ==
                    std::string::npos &&
                nested_multi_exit_loop.source.find("simdgroup_barrier") !=
                    std::string::npos,
            "nested multi-exit loops preserve barriers without per-lane dispatch");
        ok &= expect(nested_multi_exit_loop.source.find("continue;") !=
                         std::string::npos,
                     "non-local nested-loop exits continue the enclosing loop");
    }

    const metal::NvvmToMslResult irreducible_barrier =
        metal::compile_nvvm_to_msl(kNvvmIrreducibleBarrier,
                                   "irreducible-barrier.ll",
                                   "irreducible_barrier");
    ok &= expect(!irreducible_barrier.ok &&
                     irreducible_barrier.error.find("barrier-containing call graph") !=
                         std::string::npos,
                 "irreducible barrier CFGs fail instead of using a per-lane dispatcher");

    const metal::NvvmToMslResult irreducible_dispatcher =
        metal::compile_nvvm_to_msl(kNvvmIrreducibleDispatcher,
                                   "irreducible-dispatcher.ll",
                                   "irreducible_dispatcher");
    ok &= expect(irreducible_dispatcher.ok,
                 "barrier-free irreducible CFG lowers through dispatcher: " +
                     irreducible_dispatcher.error);
    if (irreducible_dispatcher.ok) {
        ok &= expect(
            irreducible_dispatcher.source.find("switch (cm_block_state)") !=
                std::string::npos,
            "irreducible CFG selects dispatcher fallback");
        ok &= expect(!has_duplicate_ssa_declaration(
                         irreducible_dispatcher.source),
                     "dispatcher predeclares each SSA local exactly once");
    }

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

    const metal::NvvmToMslResult kernel_descriptor_pointer =
        metal::compile_nvvm_to_msl(kNvvmKernelDescriptorPointer,
                                   "kernel-descriptor-pointer.ll",
                                   "kernel_descriptor_pointer");
    ok &= expect(kernel_descriptor_pointer.ok &&
                     kernel_descriptor_pointer.source.find(
                         "device uchar* descriptor_element(") != std::string::npos &&
                     kernel_descriptor_pointer.source.find(
                         "reinterpret_cast<device uchar*>") != std::string::npos,
                 "host-populated pointer fields in kernel descriptors resolve as device pointers");

    const metal::NvvmToMslResult static_threadgroup_global =
        metal::compile_nvvm_to_msl(kNvvmStaticThreadgroupGlobal,
                                   "static-threadgroup-global.ll",
                                   "static_threadgroup_global");
    ok &= expect(static_threadgroup_global.ok &&
                     static_threadgroup_global.source.find(
                         "threadgroup uchar cm_shared_shared_bytes[32]") !=
                         std::string::npos &&
                     static_threadgroup_global.source.find(
                         "threadgroup uchar* cm_shared_shared_bytes") !=
                         std::string::npos &&
                     static_threadgroup_global.source.find(
                         "shared_helper(value, cm_shared_shared_bytes)") !=
                         std::string::npos,
                 "static CUDA shared globals become kernel-local arrays threaded through helpers");

    const metal::NvvmToMslResult dynamic_threadgroup_global =
        metal::compile_nvvm_to_msl(kNvvmDynamicThreadgroupGlobal,
                                   "dynamic-threadgroup-global.ll",
                                   "dynamic_threadgroup_global");
    ok &= expect(dynamic_threadgroup_global.ok &&
                     dynamic_threadgroup_global.source.find(
                         "threadgroup uchar* cm_shared_dynamic_bytes [[threadgroup(0)]]") !=
                         std::string::npos &&
                     dynamic_threadgroup_global.source.find(
                         "dynamic_helper(value, cm_shared_dynamic_bytes)") !=
                         std::string::npos &&
                     dynamic_threadgroup_global.source.find(
                         "threadgroup uchar cm_shared_dynamic_bytes[") ==
                         std::string::npos,
                 "extern CUDA shared globals use runtime-sized Metal threadgroup binding 0");

    const metal::NvvmToMslResult multiple_dynamic_threadgroup_globals =
        metal::compile_nvvm_to_msl(kNvvmMultipleDynamicThreadgroupGlobals,
                                   "multiple-dynamic-threadgroup-globals.ll",
                                   "multiple_dynamic_threadgroup_globals");
    ok &= expect(!multiple_dynamic_threadgroup_globals.ok &&
                     multiple_dynamic_threadgroup_globals.error.find(
                         "multiple dynamic threadgroup globals") != std::string::npos,
                 "multiple dynamic shared symbols fail instead of aliasing silently");

    const metal::NvvmToMslResult helper_defined_after_kernel =
        metal::compile_nvvm_to_msl(kNvvmHelperDefinedAfterKernel,
                                   "helper-defined-after-kernel.ll",
                                   "kernel_before_helper");
    const std::size_t helper_declaration =
        helper_defined_after_kernel.source.find("uint later_helper(uint value);");
    const std::size_t kernel_definition =
        helper_defined_after_kernel.source.find("kernel void kernel_before_helper(");
    ok &= expect(helper_defined_after_kernel.ok &&
                     helper_declaration != std::string::npos &&
                     kernel_definition != std::string::npos &&
                     helper_declaration < kernel_definition,
                 "device helpers are declared before kernels regardless of LLVM order");

    const metal::NvvmToMslResult mixed_device_threadgroup_phi =
        metal::compile_nvvm_to_msl(kNvvmMixedDeviceThreadgroupPhi,
                                   "mixed-device-threadgroup-phi.ll",
                                   "mixed_device_threadgroup_phi");
    if (!mixed_device_threadgroup_phi.ok) {
        std::cerr << "mixed pointer lowering: "
                  << mixed_device_threadgroup_phi.error << "\n";
    }
    ok &= expect(mixed_device_threadgroup_phi.ok &&
                     mixed_device_threadgroup_phi.source.find(
                         "read_generic_word__cm_device") != std::string::npos &&
                     mixed_device_threadgroup_phi.source.find(
                         "read_generic_word__cm_threadgroup") != std::string::npos &&
                     mixed_device_threadgroup_phi.source.find("ulong v7") !=
                         std::string::npos &&
                     mixed_device_threadgroup_phi.source.find("v7_space == 1u") !=
                         std::string::npos &&
                     mixed_device_threadgroup_phi.source.find(
                         "reinterpret_cast<device uchar*>(v7)") != std::string::npos &&
                     mixed_device_threadgroup_phi.source.find(
                         "reinterpret_cast<threadgroup uchar*>(v7)") != std::string::npos,
                 "mixed CUDA generic-pointer PHIs dispatch concrete device and threadgroup helper specializations");

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
                     inline_shuffle.source.find(" <= ") != std::string::npos &&
                     inline_shuffle.source.find(" + ") != std::string::npos &&
                     inline_shuffle.source.find(" - ") != std::string::npos &&
                     inline_shuffle.source.find(" >= ") != std::string::npos &&
                     inline_shuffle.source.find("simd_shuffle_down(") == std::string::npos &&
                     inline_shuffle.source.find("simd_shuffle_up(") == std::string::npos,
                 "CUDA shuffle inline assembly preserves PTX clamp and self-lane fallback semantics");

    const metal::NvvmToMslResult bitcast =
        metal::compile_nvvm_to_msl(kNvvmBitcast, "bitcast.ll", "bitcast_kernel");
    ok &= expect(bitcast.ok && bitcast.source.find("as_type<uint>(") != std::string::npos,
                 "LLVM and CUDA scalar bit reinterpretation lower through Metal as_type");

    const metal::NvvmToMslResult memcpy =
        metal::compile_nvvm_to_msl(kNvvmMemcpy, "memcpy.ll", "memcpy_kernel");
    ok &= expect(memcpy.ok &&
                     memcpy.source.find("reinterpret_cast<device uint*>") != std::string::npos,
                 "constant-length aligned LLVM memcpy expands into typed Metal loads and stores");

    const metal::NvvmToMslResult memset = metal::compile_nvvm_to_msl(
        kNvvmMemset, "memset.ll", "memset_kernel");
    ok &= expect(memset.ok &&
                     memset.source.find("1515870810") != std::string::npos &&
                     memset.source.find("reinterpret_cast<device uint*>") !=
                         std::string::npos,
                 "constant-length aligned LLVM memset expands into repeated typed Metal stores");

    const metal::NvvmToMslResult value_returning_helper = metal::compile_nvvm_to_msl(
        kNvvmValueReturningHelper, "scale_add.ll", "scale_add_kernel");
    ok &= expect(value_returning_helper.ok &&
                     value_returning_helper.source.find("float scale_add(") !=
                         std::string::npos &&
                     value_returning_helper.source.find("return v") != std::string::npos &&
                     value_returning_helper.source.find("= scale_add(") != std::string::npos,
                 "value-returning device helpers preserve their return operand in MSL");

    const metal::NvvmToMslResult isolated_fp64_multiply =
        metal::compile_nvvm_to_msl(kNvvmIsolatedFp64Multiply,
                                   "isolated-fp64-multiply.ll",
                                   "isolated_fp64_multiply");
    ok &= expect(isolated_fp64_multiply.ok &&
                     isolated_fp64_multiply.source.find("double v") ==
                         std::string::npos &&
                     isolated_fp64_multiply.source.find(
                         "cumetal-semantic-quality: performance_degraded") !=
                         std::string::npos &&
                     isolated_fp64_multiply.source.find("0x") ==
                         std::string::npos &&
                     isolated_fp64_multiply.source.find(" * ") !=
                         std::string::npos,
                 "isolated float-to-double multiply chains demote explicitly for Metal");

    const metal::NvvmToMslResult general_fp64 =
        metal::compile_nvvm_to_msl(kNvvmUnsupportedFp64,
                                   "unsupported-fp64.ll", "unsupported_fp64");
    ok &= expect(general_fp64.ok &&
                     general_fp64.source.find("cm_fp64_fast_add(") !=
                         std::string::npos &&
                     general_fp64.source.find("vf64_f64_to_f32(") !=
                         std::string::npos &&
                     general_fp64.source.find("double v") == std::string::npos,
                 "general FP64 arithmetic uses raw binary64 software ALU helpers");

    const metal::NvvmToMslResult float_frexp_via_double_abi =
        metal::compile_nvvm_to_msl(kNvvmFloatFrexpViaDoubleAbi,
                                   "float-frexp-via-double-abi.ll",
                                   "float_frexp_via_double_abi");
    ok &= expect(float_frexp_via_double_abi.ok &&
                     float_frexp_via_double_abi.source.find("double v") ==
                     std::string::npos &&
                     float_frexp_via_double_abi.source.find(
                         "frexp(value, *reinterpret_cast<thread int*>(&v") !=
                         std::string::npos,
                 "float frexp round-trips through CUDA's double ABI without FP64 arithmetic");

    const metal::NvvmToMslResult dynamic_memcpy = metal::compile_nvvm_to_msl(
        kNvvmDynamicMemcpy, "dynamic-memcpy.ll", "dynamic_memcpy");
    ok &= expect(!dynamic_memcpy.ok &&
                     dynamic_memcpy.error.find("dynamic-length LLVM memcpy") != std::string::npos,
                 "dynamic-length LLVM memcpy remains an explicit diagnostic");

    const metal::NvvmToMslResult dynamic_memset = metal::compile_nvvm_to_msl(
        kNvvmDynamicMemset, "dynamic-memset.ll", "dynamic_memset");
    ok &= expect(!dynamic_memset.ok &&
                     dynamic_memset.error.find("dynamic-length LLVM memset") !=
                         std::string::npos,
                 "dynamic-length LLVM memset remains an explicit diagnostic");

    const metal::NvvmToMslResult cuda_legacy_atomic =
        metal::compile_nvvm_to_msl(kNvvmCudaLegacyAtomic,
                                   "cuda-legacy-atomic.ll",
                                   "cuda_legacy_atomic");
    ok &= expect(cuda_legacy_atomic.ok &&
                     cuda_legacy_atomic.source.find(
                         "atomic_fetch_add_explicit") != std::string::npos &&
                     cuda_legacy_atomic.source.find("memory_order_relaxed") !=
                         std::string::npos &&
                     cuda_legacy_atomic.source.find("device atomic_uint*") !=
                         std::string::npos,
                 "legacy CUDA seq_cst atomicrmw spelling lowers with CUDA-relaxed Metal semantics");

    const metal::NvvmToMslResult xor_atomic =
        metal::compile_nvvm_to_msl(kNvvmUnsupportedAtomic,
                                   "unsupported-atomic.ll",
                                   "unsupported_atomic");
    ok &= expect(xor_atomic.ok &&
                     xor_atomic.source.find("atomic_fetch_xor_explicit") !=
                         std::string::npos,
                 "typed NVVM lowering supports 32-bit atomic xor");

    const metal::NvvmToMslResult cuda_cmpxchg =
        metal::compile_nvvm_to_msl(kNvvmCudaCmpXchg,
                                   "cuda-cmpxchg.ll", "cuda_cmpxchg");
    ok &= expect(cuda_cmpxchg.ok &&
                     cuda_cmpxchg.source.find("cm_atomic_cas_device_u32") !=
                         std::string::npos &&
                     cuda_cmpxchg.source.find(
                         "atomic_compare_exchange_weak_explicit") !=
                         std::string::npos,
                 "CUDA cmpxchg imports its old-value and success tuple through a retrying Metal CAS");
    if (!cuda_cmpxchg.ok) std::cerr << cuda_cmpxchg.error << "\n";

    const metal::NvvmToMslResult cuda_cmpxchg64_helper =
        metal::compile_nvvm_to_msl(kNvvmCudaCmpXchg64Helper,
                                   "cuda-cmpxchg64-helper.ll",
                                   "cuda_cmpxchg64_helper");
    ok &= expect(cuda_cmpxchg64_helper.ok &&
                     cuda_cmpxchg64_helper.source.find(
                         "cm_wide_atomic_cas_device_u64") != std::string::npos &&
                     cuda_cmpxchg64_helper.source.find(
                         "cm_atomic_lock_bank [[buffer(29)]]") !=
                         std::string::npos &&
                     cuda_cmpxchg64_helper.source.find("cas64_helper(") !=
                         std::string::npos &&
                     cuda_cmpxchg64_helper.source.find(
                         ", cm_atomic_lock_bank)") != std::string::npos,
                 "64-bit NVVM cmpxchg threads the lock bank through device helpers");
    if (!cuda_cmpxchg64_helper.ok) std::cerr << cuda_cmpxchg64_helper.error << "\n";

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
                     cuda_math.source.find("rsqrt(") != std::string::npos &&
                     cuda_math.source.find("exp2(") != std::string::npos &&
                     cuda_math.source.find("exp10(") != std::string::npos &&
                     cuda_math.source.find(
                         "cumetal-semantic-quality: tolerance_bounded") !=
                         std::string::npos &&
                     cuda_math.source.find("log2(") != std::string::npos &&
                     cuda_math.source.find("tanh(") != std::string::npos &&
                     cuda_math.source.find("round(") != std::string::npos &&
                     cuda_math.source.find("pow(") != std::string::npos &&
                     cuda_math.source.find("popcount(") != std::string::npos &&
                     cuda_math.source.find("ctz(") != std::string::npos &&
                     cuda_math.source.find(" ? ") != std::string::npos,
                 "CUDA math and bit-count declarations map to semantics-correct Metal builtins");

    const metal::NvvmToMslResult heterogeneous_aggregate =
        metal::compile_nvvm_to_msl(kNvvmHeterogeneousAggregate,
                                   "heterogeneous-aggregate.ll",
                                   "heterogeneous_aggregate");
    ok &= expect(heterogeneous_aggregate.ok &&
                     heterogeneous_aggregate.source.find("struct mixed") !=
                         std::string::npos &&
                     heterogeneous_aggregate.source.find(
                         "mixed{integer, real, 7}") !=
                         std::string::npos &&
                     heterogeneous_aggregate.source.find(".field1") !=
                         std::string::npos,
                 "flat heterogeneous LLVM aggregates use typed MSL brace initialization");

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
                     thread_alloca.source.find(
                         "reinterpret_cast<thread uchar*>(&") != std::string::npos &&
                     thread_alloca.source.find("_storage + 4") == std::string::npos &&
                     thread_alloca.source.find("reinterpret_cast<thread float*>") !=
                         std::string::npos,
                 "LLVM aggregate allocas apply byte offsets after byte-pointer casts");

    const metal::NvvmToMslResult natural_loop = metal::compile_nvvm_to_msl(
        kNvvmNaturalLoop, "natural-loop.ll", "natural_loop");
    ok &= expect(natural_loop.ok &&
                     natural_loop.source.find("while (true)") != std::string::npos &&
                     natural_loop.source.find("break;") != std::string::npos &&
                     natural_loop.source.find("_next") != std::string::npos,
                 "natural loops preserve loop-carried PHIs through explicit Metal updates");

    const metal::NvvmToMslResult sequential_loops = metal::compile_nvvm_to_msl(
        kNvvmSequentialLoops, "sequential-loops.ll", "sequential_loops");
    const std::size_t first_loop = sequential_loops.source.find("while (true)");
    const std::size_t second_loop = first_loop == std::string::npos
                                        ? std::string::npos
                                        : sequential_loops.source.find("while (true)", first_loop + 1);
    ok &= expect(sequential_loops.ok && first_loop != std::string::npos &&
                     second_loop != std::string::npos &&
                     sequential_loops.source.find("_Z12__assert_rtn") == std::string::npos,
                 "sequential natural loops accept pre-bound PHIs and erase the empty device assert shim");
    if (!sequential_loops.ok) std::cerr << sequential_loops.error << "\n";

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

    const metal::NvvmToMslResult inline_active_mask =
        metal::compile_nvvm_to_msl(
            kNvvmInlineActiveMask, "inline-active-mask.ll", "inline_active_mask");
    ok &= expect(inline_active_mask.ok &&
                     inline_active_mask.source.find("simd_active_threads_mask()") !=
                         std::string::npos,
                 "Clang-compatible inline activemask lowers to the Metal active-lane mask");

    const metal::NvvmToMslResult external_symbols = metal::compile_nvvm_to_msl(
        kNvvmExternalSymbols, "external-symbols.ll", {});
    ok &= expect(external_symbols.ok &&
                     external_symbols.source.find(
                         "cm___cumetal_constant_symbols [[buffer(30)]]") !=
                         std::string::npos &&
                     external_symbols.source.find(
                         "cm___cumetal_global_writable [[buffer(1)]]") !=
                         std::string::npos &&
                     external_symbols.source.find(" + 16") != std::string::npos &&
                     external_symbols.source.find(" + 12") != std::string::npos &&
                     external_symbols.source.find("constant uchar embedded[8]") !=
                         std::string::npos &&
                     external_symbols.source.find("0x44, 0x33, 0x22, 0x11") !=
                         std::string::npos &&
                     external_symbols.source.find("getelementptr") == std::string::npos &&
                     external_symbols.source.find("addrspacecast") == std::string::npos &&
                     external_symbols.source.find("constant uchar constants[") ==
                         std::string::npos &&
                     external_symbols.source.find(
                         "read_symbols_helper(cm___cumetal_constant_symbols, cm___cumetal_global_writable)") !=
                         std::string::npos,
                 "runtime CUDA symbols stay hidden while initialized read-only tables embed their bytes");
    if (!external_symbols.ok) std::cerr << external_symbols.error << "\n";

    const metal::NvvmToMslResult clock_grid = metal::compile_nvvm_to_msl(
        kNvvmClockAndGridSyncHelper, "clock-grid-helper.ll", "clock_grid_kernel");
    ok &= expect(clock_grid.ok &&
                     clock_grid.source.find(
                         "cm_device_clock_counter [[buffer(28)]]") !=
                         std::string::npos &&
                     clock_grid.source.find(
                         "cm_grid_barrier [[buffer(27)]]") !=
                         std::string::npos &&
                     clock_grid.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     clock_grid.source.find("atomic_thread_fence") !=
                         std::string::npos &&
                     clock_grid.source.find(
                         "clock_and_grid_sync(out, cm_device_clock_counter, cm_grid_barrier, cm_thread_position, cm_threadgroups_per_grid)") !=
                         std::string::npos &&
                     clock_grid.source.find(
                         "cumetal-semantic-quality: semantic_emulation") !=
                         std::string::npos,
                 "typed NVVM threads emulated clock and cooperative-grid state through device helpers");
    if (!clock_grid.ok) {
        std::cerr << clock_grid.error << "\n";
    } else if (!ok) {
        std::cerr << clock_grid.source << "\n";
    }

    const metal::NvvmToMslResult oversized_constant = metal::compile_nvvm_to_msl(
        kNvvmOversizedExternalConstant, "oversized-constant.ll", "read_large");
    ok &= expect(!oversized_constant.ok &&
                     oversized_constant.error.find("exceeds 64 KiB") != std::string::npos,
                 "typed NVVM rejects constant symbol storage beyond Metal's 64 KiB limit");

    const metal::NvvmToMslResult typed_printf = metal::compile_nvvm_to_msl(
        kNvvmPrintf, "typed-printf.ll", "typed_printf");
    ok &= expect(typed_printf.ok && typed_printf.printf_formats.size() == 1 &&
                     typed_printf.printf_formats.front() == "PRINTF[%d,%d]=%d\n" &&
                     typed_printf.source.find("atomic_fetch_add_explicit") !=
                         std::string::npos &&
                     typed_printf.source.find("[[buffer(1)]]") !=
                         std::string::npos &&
                     typed_printf.source.find("[[buffer(2)]]") !=
                         std::string::npos &&
                     typed_printf.source.find("vprintf(") == std::string::npos,
                 "typed NVVM lowers constant-format vprintf to a bounded hidden ring ABI");
    if (!typed_printf.ok) std::cerr << typed_printf.error << "\n";

    const metal::NvvmToMslResult malformed_printf = metal::compile_nvvm_to_msl(
        kNvvmMalformedPrintf, "bad-printf.ll", "bad_printf");
    ok &= expect(!malformed_printf.ok &&
                     malformed_printf.error.find("32/64-bit scalar") !=
                         std::string::npos,
                 "typed NVVM rejects unrepresentable printf tuples explicitly");

    if (!ok) return 1;
    std::cout << "NVVM -> CuMetal IR -> typed MSL tests passed\n";
    return 0;
}
