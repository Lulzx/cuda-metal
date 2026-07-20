; Hand-written AIR-style LLVM IR seed for Phase 0.5 emitter tests.
; This module is intentionally minimal and metadata-focused.

target triple = "air64-apple-macosx14.0.0"

define void @vector_add(ptr addrspace(1) %a, ptr addrspace(1) %b, ptr addrspace(1) %c, i32 %__air_tid, ptr addrspace(1) %__air_printf_buf) #0 {
entry:
  %pa = getelementptr float, ptr addrspace(1) %a, i32 %__air_tid
  %pb = getelementptr float, ptr addrspace(1) %b, i32 %__air_tid
  %pc = getelementptr float, ptr addrspace(1) %c, i32 %__air_tid
  %va = load float, ptr addrspace(1) %pa, align 4
  %vb = load float, ptr addrspace(1) %pb, align 4
  %vc = fadd float %va, %vb
  store float %vc, ptr addrspace(1) %pc, align 4
  ret void
}

attributes #0 = { "air.kernel" "air.version"="2.6" }

!air.kernel = !{!0}
!0 = !{ptr @vector_add, !1, !2, !3, !4}
!1 = !{!"air.arg_type_size", i32 8}
!2 = !{!"air.arg_type_size", i32 8}
!3 = !{!"air.arg_type_size", i32 8}
!4 = !{!"air.arg_name", !"a"}
!air.compile_options = !{!5}
!5 = !{!"air.max_total_threads_per_threadgroup", i32 1024}
!air.language_version = !{!"Metal", i32 3, i32 1, i32 0}
