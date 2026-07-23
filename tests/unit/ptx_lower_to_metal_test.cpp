#include "cumetal/ptx/lower_to_metal.h"

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

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

}  // namespace

int main() {
    // ── Test 1: skeleton PTX (no global memory) → matched=false ──────────────
    const std::string skeleton_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry negate(
    .param .u64 param_in,
    .param .u64 param_out
) {
    mov.u32 %r0, %tid.x;
    neg.f32 %f1, %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_negate;
    opts_negate.entry_name = "negate";
    const auto r_skeleton = cumetal::ptx::lower_ptx_to_metal_source(skeleton_ptx, opts_negate);
    if (!expect(r_skeleton.ok, "skeleton PTX lowering returns ok")) return 1;
    if (!expect(!r_skeleton.matched, "skeleton PTX (no global ops) not matched by generic emitter"))
        return 1;

    // ── Test 2: simple element-wise kernel via mad.lo.u32 GID ────────────────
    // clamp_relu: out[gid] = max(0.0f, in[gid])  (bounds-checked)
    const std::string relu_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry clamp_relu(
    .param .u64 clamp_relu_param_0,
    .param .u64 clamp_relu_param_1,
    .param .u32 clamp_relu_param_2
) {
    .reg .u64  %rd<4>;
    .reg .f32  %f<3>;
    .reg .u32  %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [clamp_relu_param_0];
    ld.param.u64 %rd1, [clamp_relu_param_1];
    ld.param.u32 %r0,  [clamp_relu_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra DONE;

    cvt.u64.u32 %rd2, %r4;
    shl.b64     %rd2, %rd2, 2;
    add.u64     %rd2, %rd0, %rd2;
    ld.global.f32 %f0, [%rd2];

    max.f32 %f1, %f0, 0.0;

    cvt.u64.u32 %rd3, %r4;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd1, %rd3;
    st.global.f32 [%rd3], %f1;

DONE:
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_relu;
    opts_relu.entry_name = "clamp_relu";
    const auto r_relu = cumetal::ptx::lower_ptx_to_metal_source(relu_ptx, opts_relu);
    if (!expect(r_relu.ok, "clamp_relu lowering ok")) return 1;
    if (!expect(r_relu.matched, "clamp_relu matched by generic emitter")) return 1;
    if (!expect(contains(r_relu.metal_source, "kernel void clamp_relu("),
                "clamp_relu kernel signature emitted"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "device float* clamp_relu_param_0"),
                "clamp_relu input pointer mapped"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "device float* clamp_relu_param_1"),
                "clamp_relu output pointer mapped"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "constant uint& clamp_relu_param_2"),
                "clamp_relu scalar count mapped as constant ref"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "uint gid [[thread_position_in_grid]]"),
                "clamp_relu thread position arg present"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "if (gid >= (uint)clamp_relu_param_2) return;"),
                "clamp_relu bounds guard emitted"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "clamp_relu_param_0[gid]"),
                "clamp_relu global load from param_0"))
        return 1;
    if (!expect(contains(r_relu.metal_source, "clamp_relu_param_1[gid]"),
                "clamp_relu global store to param_1"))
        return 1;

    // ── Test 3: two-instruction GID (mul.lo.u32 + add.u32 pattern) ───────────
    // axpy_twoinstr: out[gid] = a * in[gid] + b
    const std::string twogid_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry axpy_twogid(
    .param .u64 axpy_twogid_param_0,
    .param .u64 axpy_twogid_param_1,
    .param .u32 axpy_twogid_param_2
) {
    .reg .u64  %rd<4>;
    .reg .f32  %f<4>;
    .reg .u32  %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [axpy_twogid_param_0];
    ld.param.u64 %rd1, [axpy_twogid_param_1];
    ld.param.u32 %r0,  [axpy_twogid_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    mov.u32 %r4, %tid.x;
    add.u32 %r5, %r3, %r4;

    setp.ge.u32 %p0, %r5, %r0;
    @%p0 bra DONE;

    cvt.u64.u32 %rd2, %r5;
    shl.b64     %rd2, %rd2, 2;
    add.u64     %rd2, %rd0, %rd2;
    ld.global.f32 %f0, [%rd2];

    fma.rn.f32 %f2, %f0, 2.0, 1.0;

    cvt.u64.u32 %rd3, %r5;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd1, %rd3;
    st.global.f32 [%rd3], %f2;

DONE:
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_twogid;
    opts_twogid.entry_name = "axpy_twogid";
    const auto r_twogid = cumetal::ptx::lower_ptx_to_metal_source(twogid_ptx, opts_twogid);
    if (!expect(r_twogid.ok, "axpy_twogid lowering ok")) return 1;
    if (!expect(r_twogid.matched,
                "axpy_twogid matched by generic emitter (mul.lo.u32+add.u32 GID pattern)"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "kernel void axpy_twogid("),
                "axpy_twogid kernel signature emitted"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "if (gid >= (uint)axpy_twogid_param_2) return;"),
                "axpy_twogid bounds guard emitted"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "axpy_twogid_param_0[gid]"),
                "axpy_twogid global load from param_0"))
        return 1;
    if (!expect(contains(r_twogid.metal_source, "axpy_twogid_param_1[gid]"),
                "axpy_twogid global store to param_1"))
        return 1;

    // ── Test 4: unsupported instruction → matched=false (no crash) ───────────
    const std::string unsup_ptx = R"PTX(
.version 8.0
.target sm_90
.visible .entry unsup_generic(
    .param .u64 unsup_generic_param_0,
    .param .u64 unsup_generic_param_1,
    .param .u32 unsup_generic_param_2
) {
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    cvt.u64.u32 %rd0, %r3;
    shl.b64 %rd0, %rd0, 2;
    ld.global.f32 %f0, [%rd0];
    foo.bar.unknown %f1, %f0;
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_unsup;
    opts_unsup.entry_name = "unsup_generic";
    const auto r_unsup = cumetal::ptx::lower_ptx_to_metal_source(unsup_ptx, opts_unsup);
    if (!expect(r_unsup.ok, "unsupported instruction lowering returns ok=true")) return 1;
    if (!expect(!r_unsup.matched, "unsupported instruction not matched by generic emitter")) return 1;

    // Regression: pointer-provenance registers are not MSL variables. A
    // general arithmetic instruction that consumes one must force the LLVM
    // fallback instead of emitting an undeclared vrd* identifier.
    const std::string pointer_arithmetic_ptx = R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry pointer_distance(
    .param .u64 pointer_distance_param_0,
    .param .u64 pointer_distance_param_1
) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<8>;
    .reg .f32 %f<2>;

    ld.param.u64 %rd0, [pointer_distance_param_0];
    ld.param.u64 %rd1, [pointer_distance_param_1];
    sub.s64 %rd2, %rd1, %rd0;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    cvt.u64.u32 %rd3, %r3;
    shl.b64 %rd3, %rd3, 2;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_pointer_arithmetic;
    opts_pointer_arithmetic.entry_name = "pointer_distance";
    const auto r_pointer_arithmetic =
        cumetal::ptx::lower_ptx_to_metal_source(pointer_arithmetic_ptx,
                                                 opts_pointer_arithmetic);
    if (!expect(r_pointer_arithmetic.ok,
                "pointer arithmetic lowering returns ok for LLVM fallback"))
        return 1;
    if (!expect(!r_pointer_arithmetic.matched,
                "pointer arithmetic does not emit undeclared MSL registers"))
        return 1;

    // ── Test 5: unary math intrinsics (sqrt, ex2, lg2, rsqrt) in generic emitter ──
    const std::string math_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry math_kernel(
    .param .u64 math_kernel_param_0,
    .param .u64 math_kernel_param_1,
    .param .u32 math_kernel_param_2
) {
    .reg .u64  %rd<4>;
    .reg .f32  %f<4>;
    .reg .u32  %r<8>;
    .reg .pred %p<2>;

    ld.param.u64 %rd0, [math_kernel_param_0];
    ld.param.u64 %rd1, [math_kernel_param_1];
    ld.param.u32 %r0,  [math_kernel_param_2];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;

    setp.ge.u32 %p0, %r4, %r0;
    @%p0 bra DONE;

    cvt.u64.u32 %rd2, %r4;
    shl.b64     %rd2, %rd2, 2;
    add.u64     %rd2, %rd0, %rd2;
    ld.global.f32 %f0, [%rd2];

    sqrt.rn.f32      %f1, %f0;
    rsqrt.approx.f32 %f2, %f1;
    ex2.approx.f32   %f3, %f0;

    cvt.u64.u32 %rd3, %r4;
    shl.b64     %rd3, %rd3, 2;
    add.u64     %rd3, %rd1, %rd3;
    st.global.f32 [%rd3], %f3;

DONE:
    ret;
}
)PTX";

    cumetal::ptx::LowerToMetalOptions opts_math;
    opts_math.entry_name = "math_kernel";
    const auto r_math = cumetal::ptx::lower_ptx_to_metal_source(math_ptx, opts_math);
    if (!expect(r_math.ok, "math_kernel lowering ok")) return 1;
    if (!expect(r_math.matched, "math_kernel matched by generic emitter")) return 1;
    if (!expect(contains(r_math.metal_source, "sqrt(vf0)"), "sqrt emitted")) return 1;
    if (!expect(contains(r_math.metal_source, "rsqrt(vf1)"), "rsqrt emitted")) return 1;
    if (!expect(contains(r_math.metal_source, "exp2(vf0)"), "ex2 → exp2 emitted")) return 1;
    if (!expect(contains(r_math.metal_source, "math_kernel_param_1[gid]"),
                "math_kernel global store to param_1"))
        return 1;

    // ── Test: passthru stub is flagged approximate ──────────────────────────
    // GGML rope is lowered to a passthru copy (no real rotary embedding), so its
    // output is numerically wrong. It must be matched AND flagged approximate so
    // the runtime can refuse it by default instead of silently emitting garbage.
    const std::string rope_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry rope_norm_f32(
    .param .u64 rope_norm_f32_param_0,
    .param .u64 rope_norm_f32_param_1
) {
    mov.u32 %r0, %tid.x;
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_rope;
    opts_rope.entry_name = "rope_norm_f32";
    const auto r_rope = cumetal::ptx::lower_ptx_to_metal_source(rope_ptx, opts_rope);
    if (!expect(r_rope.ok, "rope stub lowering ok")) return 1;
    if (!expect(r_rope.matched, "rope stub matched by direct-MSL emitter")) return 1;
    if (!expect(r_rope.approximate, "rope passthru stub flagged approximate")) return 1;
    if (!expect(r_rope.lowering_kind == cumetal::ptx::MetalLoweringKind::kApproximateStub,
                "rope provenance is approximate_stub"))
        return 1;

    // ── Test: a genuine kernel is NOT flagged approximate ────────────────────
    // encoder_forward_kernel3 is a real llm.c lowering — it must match without
    // the approximate flag so the runtime uses it normally.
    const std::string encoder_ptx = R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry encoder_forward_kernel3(
    .param .u64 encoder_forward_kernel3_param_0,
    .param .u64 encoder_forward_kernel3_param_1
) {
    mov.u32 %r0, %tid.x;
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_enc;
    opts_enc.entry_name = "encoder_forward_kernel3";
    const auto r_enc = cumetal::ptx::lower_ptx_to_metal_source(encoder_ptx, opts_enc);
    if (!expect(r_enc.ok, "encoder kernel lowering ok")) return 1;
    if (!expect(r_enc.matched, "encoder kernel matched by direct-MSL emitter")) return 1;
    if (!expect(!r_enc.approximate, "real encoder kernel not flagged approximate")) return 1;
    if (!expect(r_enc.lowering_kind == cumetal::ptx::MetalLoweringKind::kSpecializedMsl,
                "encoder provenance is specialized_msl"))
        return 1;

    // ── Regression: k_bin_bcast op_addff must ADD, op_mulff must MUL ──────────
    // The op tag lives in the PTX .entry name (the parser derives entry_name from
    // it, not from opts). A single shared template once hardcoded `*` for both,
    // turning every offloaded residual add into a multiply (token salad).
    const auto bcast_ptx = [](const char* op) {
        return std::string(R"PTX(
.version 8.0
.target sm_90
.address_size 64
.visible .entry k_bin_bcast_)PTX") + op + R"PTX((
    .param .u64 src0,
    .param .u64 src1
) {
    mov.u32 %r0, %tid.x;
    ret;
}
)PTX";
    };

    cumetal::ptx::LowerToMetalOptions opts_add;
    opts_add.entry_name = "k_bin_bcast_op_addff";
    const auto r_add = cumetal::ptx::lower_ptx_to_metal_source(bcast_ptx("op_addff"), opts_add);
    if (!expect(r_add.ok, "op_addff lowering ok")) return 1;
    if (!expect(r_add.matched, "op_addff matched by direct-MSL emitter")) return 1;
    if (!expect(!r_add.approximate, "op_addff is a real kernel, not approximate")) return 1;
    if (!expect(contains(r_add.metal_source, "result = result + v1"),
                "op_addff emits ADD (result + v1), not multiply"))
        return 1;

    cumetal::ptx::LowerToMetalOptions opts_mul;
    opts_mul.entry_name = "k_bin_bcast_op_mulff";
    const auto r_mul = cumetal::ptx::lower_ptx_to_metal_source(bcast_ptx("op_mulff"), opts_mul);
    if (!expect(r_mul.ok, "op_mulff lowering ok")) return 1;
    if (!expect(r_mul.matched, "op_mulff matched by direct-MSL emitter")) return 1;
    if (!expect(!r_mul.approximate, "op_mulff is a real kernel, not approximate")) return 1;
    if (!expect(contains(r_mul.metal_source, "result = result * v1"),
                "op_mulff emits MUL (result * v1)"))
        return 1;

    // ── Regression: GGML RMS norm preserves its 3D/strided ABI ─────────────
    const std::string rms_ptx = R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry rms_norm_f32(
    .param .u64 src,
    .param .u64 dst
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_rms;
    opts_rms.entry_name = "rms_norm_f32";
    const auto r_rms =
        cumetal::ptx::lower_ptx_to_metal_source(rms_ptx, opts_rms);
    if (!expect(r_rms.ok && r_rms.matched, "GGML RMS norm matched")) return 1;
    if (!expect(!r_rms.approximate, "GGML RMS norm is exact, not approximate"))
        return 1;
    if (!expect(contains(r_rms.metal_source,
                         "(size_t)sample * stride_sample"),
                "RMS source uses sample/channel/row strides"))
        return 1;
    if (!expect(contains(r_rms.metal_source,
                         "* nrows + row)"),
                "RMS destination uses dense 3D row indexing"))
        return 1;
    if (!expect(!contains(r_rms.metal_source,
                          "dst + (size_t)row * stride_row"),
                "RMS destination does not reuse a non-contiguous source stride"))
        return 1;
    if (!expect(contains(r_rms.metal_source,
                         "row % mul_nrows_packed.z"),
                "fused RMS multiply honors broadcast row shape"))
        return 1;
    if (!expect(contains(r_rms.metal_source,
                         "(uint)i % mul_ncols_packed.z"),
                "fused RMS multiply honors broadcast column shape"))
        return 1;
    if (!expect(contains(r_rms.metal_source, "simd_sum(partial)"),
                "RMS uses native 32-lane SIMD reduction"))
        return 1;
    if (!expect(contains(r_rms.metal_source,
                         "threadgroup float shared[32]"),
                "RMS stores one subtotal per SIMD group"))
        return 1;

    // ── Regression: GGML float gated-SiLU is lowered exactly ───────────────
    const std::string silu_name =
        "_ZL21unary_gated_op_kernelIXadL_ZL7op_silufEEfEvPKT0_S2_PS0_xxxx";
    const std::string silu_ptx = std::string(R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry )PTX") + silu_name + R"PTX((
    .param .u64 x,
    .param .u64 gate,
    .param .u64 dst,
    .param .u64 k,
    .param .u64 n,
    .param .u64 o0,
    .param .u64 o1
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_silu;
    opts_silu.entry_name = silu_name;
    const auto r_silu =
        cumetal::ptx::lower_ptx_to_metal_source(silu_ptx, opts_silu);
    if (!expect(r_silu.ok && r_silu.matched, "gated-SiLU lowering matched"))
        return 1;
    if (!expect(!r_silu.approximate, "gated-SiLU lowering is exact"))
        return 1;
    if (!expect(contains(r_silu.metal_source, "row * o0 + lane"),
                "gated-SiLU honors the source row stride"))
        return 1;
    if (!expect(contains(r_silu.metal_source, "exp(-value)"),
                "gated-SiLU applies the sigmoid"))
        return 1;
    if (!expect(contains(r_silu.metal_source, "gate[j1]"),
                "gated-SiLU multiplies by the gate"))
        return 1;

    // ── Regression: observed SmolLM2 forward RoPE is exact ─────────────────
    const std::string rope_exact_name =
        "_ZL9rope_normILb1ELb0EfDF16_EvPKT1_PT2_iiiiiiiiiiPKifff"
        "14rope_corr_dimsfPKfPKxi";
    const std::string rope_exact_ptx = std::string(R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry )PTX") + rope_exact_name + R"PTX((
    .param .u64 src,
    .param .u64 dst
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_rope_exact;
    opts_rope_exact.entry_name = rope_exact_name;
    const auto r_rope_exact =
        cumetal::ptx::lower_ptx_to_metal_source(rope_exact_ptx,
                                                opts_rope_exact);
    if (!expect(r_rope_exact.ok && r_rope_exact.matched,
                "observed forward RoPE lowering matched"))
        return 1;
    if (!expect(!r_rope_exact.approximate,
                "observed forward RoPE lowering is exact"))
        return 1;
    if (!expect(contains(r_rope_exact.metal_source, "pow(theta_scale"),
                "RoPE applies dimension-dependent frequency"))
        return 1;
    if (!expect(contains(r_rope_exact.metal_source,
                         "x0 * cosine - x1 * sine"),
                "RoPE rotates each adjacent pair"))
        return 1;
    if (!expect(contains(r_rope_exact.metal_source, "device half* dst"),
                "RoPE selects the half-output ABI"))
        return 1;

    // ── Regression: GGML Q8_0 conversion is a real dequantizer ──────────────
    const std::string q8_ptx = R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry dequantize_block_q8_0_f16(
    .param .u64 src,
    .param .u64 dst,
    .param .u64 k
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_q8;
    opts_q8.entry_name = "dequantize_block_q8_0_f16";
    const auto r_q8 = cumetal::ptx::lower_ptx_to_metal_source(q8_ptx, opts_q8);
    if (!expect(r_q8.ok && r_q8.matched, "Q8_0 f16 dequantizer matched")) return 1;
    if (!expect(!r_q8.approximate, "Q8_0 f16 dequantizer is not a passthru stub")) return 1;
    if (!expect(contains(r_q8.metal_source, "bytes_per_block = 34"),
                "Q8_0 dequantizer uses the packed 34-byte block layout"))
        return 1;
    if (!expect(contains(r_q8.metal_source, "float(scale) * float(quant)"),
                "Q8_0 dequantizer applies the block scale"))
        return 1;

    // ── Regression: GGML Q6_K packed layout is dequantized exactly ────────
    const std::string q6_name =
        "_ZL21dequantize_block_q6_KIDF16_EvPKvPT_";
    const std::string q6_ptx = std::string(R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry )PTX") + q6_name + R"PTX((
    .param .u64 src,
    .param .u64 dst
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_q6;
    opts_q6.entry_name = q6_name;
    const auto r_q6 = cumetal::ptx::lower_ptx_to_metal_source(q6_ptx, opts_q6);
    if (!(r_q6.ok && r_q6.matched)) {
        std::fprintf(stderr,
                     "Q6_K diagnostic: ok=%d matched=%d entry='%s' error='%s'\n",
                     r_q6.ok ? 1 : 0, r_q6.matched ? 1 : 0,
                     r_q6.entry_name.c_str(), r_q6.error.c_str());
        if (!expect(false, "Q6_K f16 dequantizer matched")) return 1;
    }
    if (!expect(!r_q6.approximate, "Q6_K f16 dequantizer is exact")) return 1;
    if (!expect(contains(r_q6.metal_source, "bytes_per_block = 210"),
                "Q6_K dequantizer uses the packed 210-byte block layout"))
        return 1;
    if (!expect(contains(r_q6.metal_source, "scales_offset = 192"),
                "Q6_K dequantizer reads signed per-group scales"))
        return 1;
    if (!expect(contains(r_q6.metal_source, "delta_offset = 208"),
                "Q6_K dequantizer reads the trailing half super-scale"))
        return 1;
    if (!expect(contains(r_q6.metal_source, "float(q3)"),
                "Q6_K dequantizer reconstructs all four 6-bit values per lane"))
        return 1;

    // ── Regression: GGML strided f32/f16 conversion keeps its ABI ──────────
    const auto convert_ptx = [](const std::string& entry) {
        return std::string(R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry )PTX") + entry + R"PTX((
    .param .u64 src,
    .param .u64 dst,
    .param .u64 ne00,
    .param .u64 ne01,
    .param .u64 ne0203,
    .param .align 4 .b8 ne02_fd[12],
    .param .u64 s01,
    .param .u64 s02,
    .param .u64 s03
) {
    ret;
}
)PTX";
    };
    const std::string f32_f16_name =
        "_ZL13convert_unaryIfDF16_EvPKvPT0_xxx5uint3xxx";
    cumetal::ptx::LowerToMetalOptions opts_f32_f16;
    opts_f32_f16.entry_name = f32_f16_name;
    const auto r_f32_f16 =
        cumetal::ptx::lower_ptx_to_metal_source(convert_ptx(f32_f16_name),
                                                 opts_f32_f16);
    if (!expect(r_f32_f16.ok && r_f32_f16.matched,
                "GGML float-to-half convert matched"))
        return 1;
    if (!expect(!r_f32_f16.approximate,
                "GGML float-to-half convert is not approximate"))
        return 1;
    if (!expect(contains(r_f32_f16.metal_source,
                         "const device float* typed_src"),
                "float-to-half convert reads float"))
        return 1;
    if (!expect(contains(r_f32_f16.metal_source,
                         "device half* typed_dst"),
                "float-to-half convert writes half"))
        return 1;
    if (!expect(contains(r_f32_f16.metal_source,
                         "i03 * s03 + i02 * s02 + i01 * s01 + i00"),
                "convert preserves GGML source strides"))
        return 1;

    const std::string f16_f32_name =
        "_ZL13convert_unaryIDF16_fEvPKvPT0_xxx5uint3xxx";
    cumetal::ptx::LowerToMetalOptions opts_f16_f32;
    opts_f16_f32.entry_name = f16_f32_name;
    const auto r_f16_f32 =
        cumetal::ptx::lower_ptx_to_metal_source(convert_ptx(f16_f32_name),
                                                 opts_f16_f32);
    if (!expect(r_f16_f32.ok && r_f16_f32.matched,
                "GGML half-to-float convert matched"))
        return 1;
    if (!expect(!r_f16_f32.approximate,
                "GGML half-to-float convert is not approximate"))
        return 1;
    if (!expect(contains(r_f16_f32.metal_source,
                         "const device half* typed_src"),
                "half-to-float convert reads half"))
        return 1;
    if (!expect(contains(r_f16_f32.metal_source,
                         "device float* typed_dst"),
                "half-to-float convert writes float"))
        return 1;

    // GGML materializes transposed attention-cache views with the general
    // strided scalar-copy kernel. This must preserve all four byte strides;
    // treating it as a flat memcpy corrupts every row after the first.
    const std::string cpy_f16_f16_name =
        "_ZL10cpy_scalarIXadL_ZL12cpy_1_scalarIDF16_DF16_EvPKcPcEEEvS2_S3_"
        "xxxxxxxxxxxxxxx";
    const std::string cpy_f16_f16_ptx = std::string(R"PTX(
.version 8.0
.target sm_80
.address_size 64
.visible .entry )PTX") + cpy_f16_f16_name + R"PTX((
    .param .u64 src,
    .param .u64 dst,
    .param .u64 ne,
    .param .u64 ne00,
    .param .u64 ne01,
    .param .u64 ne02,
    .param .u64 nb00,
    .param .u64 nb01,
    .param .u64 nb02,
    .param .u64 nb03,
    .param .u64 ne10,
    .param .u64 ne11,
    .param .u64 ne12,
    .param .u64 nb10,
    .param .u64 nb11,
    .param .u64 nb12,
    .param .u64 nb13
) {
    ret;
}
)PTX";
    cumetal::ptx::LowerToMetalOptions opts_cpy_f16_f16;
    opts_cpy_f16_f16.entry_name = cpy_f16_f16_name;
    const auto r_cpy_f16_f16 = cumetal::ptx::lower_ptx_to_metal_source(
        cpy_f16_f16_ptx, opts_cpy_f16_f16);
    if (!expect(r_cpy_f16_f16.ok && r_cpy_f16_f16.matched,
                "GGML half scalar copy matched"))
        return 1;
    if (!expect(!r_cpy_f16_f16.approximate,
                "GGML half scalar copy is exact"))
        return 1;
    if (!expect(contains(r_cpy_f16_f16.metal_source,
                         "i00 * nb00 + i01 * nb01 + i02 * nb02 + i03 * nb03"),
                "scalar copy preserves source byte strides"))
        return 1;
    if (!expect(contains(r_cpy_f16_f16.metal_source,
                         "i10 * nb10 + i11 * nb11 + i12 * nb12 + i13 * nb13"),
                "scalar copy preserves destination byte strides"))
        return 1;
    if (!expect(contains(r_cpy_f16_f16.metal_source,
                         "reinterpret_cast<device half*>"),
                "half scalar copy uses the encoded template type"))
        return 1;

    if (!expect(r_math.lowering_kind == cumetal::ptx::MetalLoweringKind::kGenericPtx,
                "ordinary math kernel provenance is generic_ptx"))
        return 1;

    std::printf("PASS: ptx lower-to-metal unit tests\n");
    return 0;
}
