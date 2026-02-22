// Unit tests for the threadgroup_tiling pass.
//
// Tests cover:
//  1. No shared memory → no hints, no accesses.
//  2. Shared-memory accesses without a conflicting stride → no hints.
//  3. Classic GEMM bank-conflict pattern: mul by 32 before ld.shared.f32 → hint.
//  4. shl.b32 stride-of-32 pattern (2^5) → hint.
//  5. f64 element type with stride 16 → hint (16 × 8B = 128B = 32-bank boundary).
//  6. Stride 16 for f32 → no conflict (16 × 4B = 64B ≠ 128B bank boundary).
//  7. Stride 32 via mul, then atom.shared.f32 → hint detected on atomic too.
//  8. Two distinct conflicting strides → two separate hints.
//  9. Padding recommendation equals element size for f32 and f64.

#include "cumetal/passes/threadgroup_tiling.h"
#include "cumetal/ptx/parser.h"

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

// Parse PTX and return the first entry function.  Aborts on parse failure.
cumetal::ptx::EntryFunction parse_entry(const char* ptx_text) {
    const auto result = cumetal::ptx::parse_ptx(ptx_text);
    if (!result.ok || result.module.entries.empty()) {
        std::fprintf(stderr, "INTERNAL: parse_entry failed: %s\n",
                     result.error.c_str());
        std::abort();
    }
    return result.module.entries[0];
}

bool has_hint_for(const cumetal::passes::TilingAnalysis& a,
                  std::uint32_t stride, std::size_t elem_bytes) {
    for (const auto& h : a.hints) {
        if (h.detected_stride == stride && h.elem_bytes == elem_bytes &&
            h.needs_padding) {
            return true;
        }
    }
    return false;
}

}  // namespace

int main() {
    // ── Test 1: no shared memory ─────────────────────────────────────────────
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    ld.global.f32 %f0, [%rd0];
    st.global.f32 [%rd1], %f0;
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test1: ok")) return 1;
        if (!expect(!a.uses_shared_memory, "test1: no shared memory")) return 1;
        if (!expect(a.accesses.empty(), "test1: no accesses")) return 1;
        if (!expect(a.hints.empty(), "test1: no hints")) return 1;
    }

    // ── Test 2: shared access without a conflicting stride ───────────────────
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    ld.shared.f32 %f0, [%rd0];
    st.shared.f32 [%rd1], %f0;
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test2: ok")) return 1;
        if (!expect(a.uses_shared_memory, "test2: uses shared memory")) return 1;
        if (!expect(a.accesses.size() == 2, "test2: two accesses")) return 1;
        if (!expect(a.hints.empty(), "test2: no hints (no conflicting stride)")) return 1;
    }

    // ── Test 3: mul.lo stride=32 before ld.shared.f32 → hint ─────────────────
    // 32 × 4B = 128B = 32 banks × 4B/bank → conflict.
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry gemm(.param .u64 p0) {
    mul.lo.u32 %stride, %tid, 32;
    ld.shared.f32 %f0, [%shmem+%stride];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test3: ok")) return 1;
        if (!expect(a.uses_shared_memory, "test3: uses shared memory")) return 1;
        if (!expect(!a.hints.empty(), "test3: hint emitted")) return 1;
        if (!expect(has_hint_for(a, 32, 4), "test3: hint stride=32 elem=4")) return 1;
        if (!expect(a.hints[0].padding_bytes == 4, "test3: padding=4")) return 1;
        if (!expect(!a.hints[0].reason.empty(), "test3: reason populated")) return 1;
    }

    // ── Test 4: shl.b32 shift=5 (≡ stride=32) before ld.shared.f32 → hint ───
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    shl.b32 %rowbytes, %row, 5;
    ld.shared.f32 %f0, [%shmem+%rowbytes];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test4: ok")) return 1;
        if (!expect(has_hint_for(a, 32, 4), "test4: shl-stride=32 hint")) return 1;
    }

    // ── Test 5: f64 with stride=16 → hint (16 × 8B = 128B = bank boundary) ──
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    mul.lo.u32 %s, %tid, 16;
    ld.shared.f64 %fd0, [%shmem+%s];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test5: ok")) return 1;
        if (!expect(has_hint_for(a, 16, 8), "test5: stride=16 f64 hint")) return 1;
        if (!expect(a.hints[0].padding_bytes == 8, "test5: padding=8 for f64")) return 1;
    }

    // ── Test 6: stride=16 with f32 → NO conflict (16×4=64B ≠ 128B) ──────────
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    mul.lo.u32 %s, %tid, 16;
    ld.shared.f32 %f0, [%shmem+%s];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test6: ok")) return 1;
        if (!expect(!has_hint_for(a, 16, 4), "test6: stride=16 f32 no hint")) return 1;
    }

    // ── Test 7: atom.shared.f32 with stride=32 → hint ────────────────────────
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    mul.lo.u32 %s, %tid, 32;
    atom.shared.add.f32 %f0, [%shmem+%s], %f1;
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test7: ok")) return 1;
        if (!expect(has_hint_for(a, 32, 4), "test7: atom.shared stride=32 hint")) return 1;
        // The atomic is classified as a write.
        bool saw_write = false;
        for (const auto& acc : a.accesses) {
            if (acc.is_write) { saw_write = true; break; }
        }
        if (!expect(saw_write, "test7: atom.shared classified as write")) return 1;
    }

    // ── Test 8: two distinct conflicting strides → two hints ─────────────────
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k(.param .u64 p0) {
    mul.lo.u32 %s32, %tid, 32;
    ld.shared.f32 %f0, [%shmem+%s32];
    mul.lo.u32 %s64, %tid, 64;
    ld.shared.f32 %f1, [%shmem+%s64];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test8: ok")) return 1;
        if (!expect(a.hints.size() == 2, "test8: two hints")) return 1;
        if (!expect(has_hint_for(a, 32, 4), "test8: stride=32 hint")) return 1;
        if (!expect(has_hint_for(a, 64, 4), "test8: stride=64 hint")) return 1;
    }

    // ── Test 9: padding_bytes equals elem_bytes (one f32, one f64 kernel each) ─
    // Test 9a: f32 with stride=32 → padding_bytes == 4
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k9a(.param .u64 p0) {
    mul.lo.u32 %s, %tid, 32;
    ld.shared.f32 %f0, [%shm+%s];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test9a: ok")) return 1;
        if (!expect(has_hint_for(a, 32, 4), "test9a: f32 hint")) return 1;
        if (!expect(a.hints[0].padding_bytes == 4, "test9a: f32 padding_bytes==4")) return 1;
    }
    // Test 9b: f64 with stride=16 → padding_bytes == 8
    {
        const auto entry = parse_entry(R"PTX(
.version 7.0
.target sm_80
.visible .entry k9b(.param .u64 p0) {
    mul.lo.u32 %s, %tid, 16;
    ld.shared.f64 %fd0, [%shm+%s];
    ret;
}
)PTX");
        const auto a = cumetal::passes::analyse_threadgroup_tiling(entry);
        if (!expect(a.ok, "test9b: ok")) return 1;
        if (!expect(has_hint_for(a, 16, 8), "test9b: f64 hint")) return 1;
        if (!expect(a.hints[0].padding_bytes == 8, "test9b: f64 padding_bytes==8")) return 1;
    }

    std::printf("PASS: threadgroup_tiling pass unit tests\n");
    return 0;
}
