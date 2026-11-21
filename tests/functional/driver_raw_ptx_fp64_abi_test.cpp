// #231 / #232: run-time PTX with binary64 math, loaded through
// cuModuleLoadDataEx, launched with a kernelParams array that is exactly as
// long as the kernel's parameter list (no NULL terminator; the next word is a
// non-NULL poison pointer) and a 16-byte struct passed by value. The module
// has the shape of xgrid-gpu's biot_savart.ptx: four pointers plus a 16-byte
// .b8 array parameter, and every entry -- even the fp32 one -- touches f64.
//
// It also covers the binary64 opcodes the typed importer used to refuse:
// sqrt.{rn,rz,rm,rp}.f64, rsqrt.approx.f64 and min/max.f64.
//
// usage: driver_raw_ptx_fp64_abi_test [--expect-refusal]
//   CUMETAL_FP64_MODE selects the mode under test (fast48 when unset).
//   --expect-refusal loads a module the typed backend refuses and expects
//   CUDA_ERROR_INVALID_PTX (run with CUMETAL_PTX_BACKEND=cumetal-ir).

#include "cuda.h"

#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#pragma STDC FENV_ACCESS ON

namespace {

constexpr const char* kModule = R"PTX(
.version 7.0
.target sm_60
.address_size 64

.visible .entry fp64_ops(
	.param .u64 .ptr .align 1 fp64_ops_param_0,
	.param .u64 .ptr .align 1 fp64_ops_param_1,
	.param .u64 .ptr .align 1 fp64_ops_param_2,
	.param .u64 .ptr .align 1 fp64_ops_param_3,
	.param .align 4 .b8 fp64_ops_param_4[16]
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<12>;
	.reg .f32 %f<2>;
	.reg .b64 %rd<16>;
	.reg .f64 %fd<16>;

	ld.param.u64 %rd1, [fp64_ops_param_0];
	ld.param.u64 %rd2, [fp64_ops_param_1];
	ld.param.u64 %rd3, [fp64_ops_param_2];
	ld.param.u64 %rd4, [fp64_ops_param_3];
	ld.param.u32 %r1, [fp64_ops_param_4];
	ld.param.u32 %r2, [fp64_ops_param_4+4];
	ld.param.f32 %f1, [fp64_ops_param_4+8];
	ld.param.u32 %r3, [fp64_ops_param_4+12];
	mov.u32 %r4, %ctaid.x;
	mov.u32 %r5, %ntid.x;
	mov.u32 %r6, %tid.x;
	mad.lo.s32 %r7, %r4, %r5, %r6;
	setp.ge.u32 %p1, %r7, %r2;
	@%p1 bra $L__done;
	add.s32 %r8, %r7, %r1;
	cvta.to.global.u64 %rd5, %rd1;
	cvta.to.global.u64 %rd6, %rd2;
	cvta.to.global.u64 %rd7, %rd3;
	cvta.to.global.u64 %rd8, %rd4;
	mul.wide.u32 %rd9, %r8, 8;
	add.s64 %rd10, %rd5, %rd9;
	ld.global.f64 %fd1, [%rd10];
	add.s64 %rd11, %rd6, %rd9;
	ld.global.f64 %fd2, [%rd11];
	add.s64 %rd12, %rd7, %rd9;
	ld.global.f64 %fd3, [%rd12];
	sqrt.rn.f64 %fd4, %fd1;
	sqrt.rz.f64 %fd5, %fd1;
	sqrt.rm.f64 %fd6, %fd1;
	sqrt.rp.f64 %fd7, %fd1;
	rsqrt.approx.f64 %fd8, %fd1;
	max.f64 %fd9, %fd2, %fd3;
	min.f64 %fd10, %fd2, %fd3;
	cvt.f64.f32 %fd11, %f1;
	mul.rn.f64 %fd12, %fd4, %fd11;
	add.rn.f64 %fd13, %fd12, %fd9;
	mul.lo.s32 %r9, %r8, %r3;
	mul.wide.u32 %rd13, %r9, 8;
	add.s64 %rd14, %rd8, %rd13;
	st.global.f64 [%rd14], %fd4;
	st.global.f64 [%rd14+8], %fd5;
	st.global.f64 [%rd14+16], %fd6;
	st.global.f64 [%rd14+24], %fd7;
	st.global.f64 [%rd14+32], %fd8;
	st.global.f64 [%rd14+40], %fd9;
	st.global.f64 [%rd14+48], %fd10;
	st.global.f64 [%rd14+56], %fd13;
$L__done:
	ret;
}

.visible .entry fp32_widen(
	.param .u64 .ptr .align 1 fp32_widen_param_0,
	.param .u64 .ptr .align 1 fp32_widen_param_1,
	.param .u64 .ptr .align 1 fp32_widen_param_2,
	.param .u64 .ptr .align 1 fp32_widen_param_3,
	.param .align 4 .b8 fp32_widen_param_4[16]
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<10>;
	.reg .f32 %f<4>;
	.reg .b64 %rd<10>;
	.reg .f64 %fd<2>;

	ld.param.u64 %rd1, [fp32_widen_param_0];
	ld.param.u64 %rd4, [fp32_widen_param_3];
	ld.param.u32 %r1, [fp32_widen_param_4];
	ld.param.u32 %r2, [fp32_widen_param_4+4];
	ld.param.f32 %f1, [fp32_widen_param_4+8];
	mov.u32 %r4, %ctaid.x;
	mov.u32 %r5, %ntid.x;
	mov.u32 %r6, %tid.x;
	mad.lo.s32 %r7, %r4, %r5, %r6;
	setp.ge.u32 %p1, %r7, %r2;
	@%p1 bra $L__end;
	add.s32 %r8, %r7, %r1;
	cvta.to.global.u64 %rd5, %rd1;
	cvta.to.global.u64 %rd6, %rd4;
	mul.wide.u32 %rd7, %r8, 4;
	add.s64 %rd8, %rd5, %rd7;
	ld.global.f32 %f2, [%rd8];
	mul.rn.f32 %f3, %f2, %f1;
	cvt.f64.f32 %fd1, %f3;
	mul.wide.u32 %rd9, %r8, 8;
	add.s64 %rd9, %rd6, %rd9;
	st.global.f64 [%rd9], %fd1;
$L__end:
	ret;
}
)PTX";

// Typed lowering refuses brkpt, so strict typed selection must fail the load.
constexpr const char* kRefusedModule = R"PTX(
.version 7.0
.target sm_70
.address_size 64
.visible .entry refused(.param .u64 p) {
	brkpt;
	ret;
}
)PTX";

struct Params {
    std::uint32_t offset;
    std::uint32_t count;
    float scale;
    std::uint32_t stride;
};
static_assert(sizeof(Params) == 16);

constexpr std::uint32_t kElements = 160;
constexpr std::uint32_t kOffset = 7;
constexpr std::uint32_t kCount = 133;
constexpr std::uint32_t kStride = 8;
constexpr double kSentinel = -12345.5;

int failures = 0;

void check(bool ok, const char* what, std::uint32_t i, double got, double want) {
    if (ok) return;
    if (failures < 12) {
        std::fprintf(stderr, "FAIL: %s[%u] got %.17g (0x%016llx) want %.17g\n", what, i, got,
                     static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(got)), want);
    }
    ++failures;
}

double rounded_sqrt(double x, int mode) {
    const int saved = std::fegetround();
    std::fesetround(mode);
    volatile double v = x;
    const double r = std::sqrt(v);
    std::fesetround(saved);
    return r;
}

double rel(double got, double want) {
    return std::fabs(got - want) / std::fabs(want);
}

}  // namespace

int main(int argc, char** argv) {
    const bool expect_refusal = argc > 1 && std::strcmp(argv[1], "--expect-refusal") == 0;
    const char* mode_env = std::getenv("CUMETAL_FP64_MODE");
    const std::string mode = mode_env == nullptr || *mode_env == '\0' ? "fast48" : mode_env;
    const bool exact_mode = mode == "ieee64";

    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "SKIP: cuInit failed\n");
        return 77;
    }
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: context\n");
        return 1;
    }

    if (expect_refusal) {
        CUmodule refused = nullptr;
        const CUresult status = cuModuleLoadDataEx(&refused, kRefusedModule, 0, nullptr, nullptr);
        if (status != CUDA_ERROR_INVALID_PTX) {
            std::fprintf(stderr, "FAIL: refused module returned %d, want CUDA_ERROR_INVALID_PTX\n",
                         static_cast<int>(status));
            return 1;
        }
        std::printf("PASS: strict typed selection refuses an unsupported module\n");
        return 0;
    }

    CUmodule module = nullptr;
    const CUresult load = cuModuleLoadDataEx(&module, kModule, 0, nullptr, nullptr);
    if (load == CUDA_ERROR_JIT_COMPILER_NOT_FOUND) {
        std::fprintf(stderr, "SKIP: Metal toolchain unavailable\n");
        return 77;
    }
    if (load != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleLoadDataEx returned %d\n", static_cast<int>(load));
        return 1;
    }
    CUfunction fp64_ops = nullptr;
    CUfunction fp32_widen = nullptr;
    if (cuModuleGetFunction(&fp64_ops, module, "fp64_ops") != CUDA_SUCCESS ||
        cuModuleGetFunction(&fp32_widen, module, "fp32_widen") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: both entries must load\n");
        return 1;
    }

    // Inputs: a in [1e-3, 1e6) (inside every mode's exponent range), b and c
    // with NaNs in either operand to exercise min/max's NaN rule.
    std::vector<double> a(kElements), b(kElements), c(kElements);
    std::vector<float> af(kElements);
    std::uint64_t state = 0x9E3779B97F4A7C15ull;
    const auto next = [&]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<double>(state >> 11) * 0x1.0p-53;
    };
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::uint32_t i = 0; i < kElements; ++i) {
        a[i] = std::exp(std::log(1e-3) + next() * std::log(1e9));
        b[i] = (next() - 0.5) * 1e4;
        c[i] = (next() - 0.5) * 1e4;
        if (i % 11 == 3) b[i] = nan;
        if (i % 13 == 5) c[i] = nan;
        af[i] = static_cast<float>((next() - 0.5) * 1e3);
    }
    const Params params{kOffset, kCount, 1.75f, kStride};

    CUdeviceptr da = 0, db = 0, dc = 0, dout = 0, dfa = 0, dwide = 0;
    const std::size_t out_count = static_cast<std::size_t>(kElements) * kStride;
    if (cuMemAlloc(&da, kElements * sizeof(double)) != CUDA_SUCCESS ||
        cuMemAlloc(&db, kElements * sizeof(double)) != CUDA_SUCCESS ||
        cuMemAlloc(&dc, kElements * sizeof(double)) != CUDA_SUCCESS ||
        cuMemAlloc(&dout, out_count * sizeof(double)) != CUDA_SUCCESS ||
        cuMemAlloc(&dfa, kElements * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dwide, kElements * sizeof(double)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc\n");
        return 1;
    }
    std::vector<double> sentinel(out_count, kSentinel);
    cuMemcpyHtoD(da, a.data(), kElements * sizeof(double));
    cuMemcpyHtoD(db, b.data(), kElements * sizeof(double));
    cuMemcpyHtoD(dc, c.data(), kElements * sizeof(double));
    cuMemcpyHtoD(dout, sentinel.data(), out_count * sizeof(double));
    cuMemcpyHtoD(dfa, af.data(), kElements * sizeof(float));
    cuMemcpyHtoD(dwide, sentinel.data(), kElements * sizeof(double));

    // Exactly five entries. The word after them is non-NULL, so a launch that
    // scans for a NULL terminator reads it and fails (or crashes).
    struct {
        void* args[5];
        void* poison;
    } ops_args = {{&da, &db, &dc, &dout, const_cast<Params*>(&params)},
                  reinterpret_cast<void*>(0x10)};
    struct {
        void* args[5];
        void* poison;
    } widen_args = {{&dfa, &db, &dc, &dwide, const_cast<Params*>(&params)},
                    reinterpret_cast<void*>(0x10)};
    const unsigned blocks = (kCount + 63) / 64;
    CUresult launch = cuLaunchKernel(fp64_ops, blocks, 1, 1, 64, 1, 1, 0, nullptr,
                                     ops_args.args, nullptr);
    if (launch == CUDA_SUCCESS) {
        launch = cuLaunchKernel(fp32_widen, blocks, 1, 1, 64, 1, 1, 0, nullptr,
                                widen_args.args, nullptr);
    }
    if (launch == CUDA_SUCCESS) launch = cuCtxSynchronize();
    if (launch != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: launch returned %d\n", static_cast<int>(launch));
        return 1;
    }
    std::vector<double> out(out_count);
    std::vector<double> wide(kElements);
    cuMemcpyDtoH(out.data(), dout, out_count * sizeof(double));
    cuMemcpyDtoH(wide.data(), dwide, kElements * sizeof(double));

    // fast48 keeps about 48 significand bits; wide48 a little more.
    const double tolerance = exact_mode ? 0.0 : 1e-13;
    for (std::uint32_t i = 0; i < kElements; ++i) {
        const double* row = &out[static_cast<std::size_t>(i) * kStride];
        const bool active = i >= kOffset && i < kOffset + kCount;
        if (!active) {
            for (std::uint32_t k = 0; k < kStride; ++k) {
                check(row[k] == kSentinel, "untouched", i, row[k], kSentinel);
            }
            check(wide[i] == kSentinel, "untouched_wide", i, wide[i], kSentinel);
            continue;
        }
        const double s = std::sqrt(a[i]);
        if (exact_mode) {
            check(row[0] == s, "sqrt.rn", i, row[0], s);
        } else {
            check(rel(row[0], s) <= tolerance, "sqrt.rn", i, row[0], s);
        }
        // Directed sqrt goes through VF64's rounding-mode sqrt in every mode.
        const double rz = rounded_sqrt(a[i], FE_TOWARDZERO);
        const double rm = rounded_sqrt(a[i], FE_DOWNWARD);
        const double rp = rounded_sqrt(a[i], FE_UPWARD);
        check(row[1] == rz, "sqrt.rz", i, row[1], rz);
        check(row[2] == rm, "sqrt.rm", i, row[2], rm);
        check(row[3] == rp, "sqrt.rp", i, row[3], rp);
        const long double rsqrt = 1.0L / std::sqrt(static_cast<long double>(a[i]));
        check(rel(row[4], static_cast<double>(rsqrt)) <= 1e-12, "rsqrt.approx", i, row[4],
              static_cast<double>(rsqrt));
        const double mx = std::fmax(b[i], c[i]);
        const double mn = std::fmin(b[i], c[i]);
        // ieee64 returns an operand bit for bit. fast48/wide48 first
        // canonicalize both operands into their own domain, as every other
        // operation in those modes does, so the selected operand is the
        // canonicalized one. A NaN operand yields the other in every mode.
        const auto selected = [&](double got, double want) {
            if (std::isnan(want)) return std::isnan(got);
            return exact_mode ? got == want : rel(got, want) <= tolerance;
        };
        check(selected(row[5], mx), "max", i, row[5], mx);
        check(selected(row[6], mn), "min", i, row[6], mn);
        volatile double product = s * static_cast<double>(params.scale);
        const double fused = product + mx;
        if (std::isnan(fused)) {
            check(std::isnan(row[7]), "mul_add", i, row[7], fused);
        } else if (exact_mode) {
            check(row[7] == fused, "mul_add", i, row[7], fused);
        } else {
            check(std::fabs(row[7] - fused) <= 1e-12 * (std::fabs(product) + std::fabs(mx)),
                  "mul_add", i, row[7], fused);
        }
        // binary32 multiply then an exact widening, in every mode.
        const double widened = static_cast<double>(af[i] * params.scale);
        check(wide[i] == widened, "fp32_widen", i, wide[i], widened);
    }
    if (failures != 0) {
        std::fprintf(stderr, "FAIL: %d mismatches under fp64 mode %s\n", failures, mode.c_str());
        return 1;
    }
    cuMemFree(da);
    cuMemFree(db);
    cuMemFree(dc);
    cuMemFree(dout);
    cuMemFree(dfa);
    cuMemFree(dwide);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    std::printf("PASS: run-time fp64 PTX with exact argument metadata under %s\n", mode.c_str());
    return 0;
}
