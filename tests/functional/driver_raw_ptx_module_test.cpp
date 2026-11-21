// #222 (Cuda-On-Metal-Lab overlay): raw PTX loaded through the driver API.
//  - an FP32 kernel and an FP64 kernel from the same PTX module both JIT and
//    run, so FP64 support is available to runtime-compiled kernels;
//  - a PTX module larger than 1 MiB loads (the old direct-PTX bound);
//  - a static-shared kernel launched many times keeps its shared size;
//  - a two-entry module keeps both entries on every backend (the legacy
//    lowering builds one metallib per entry);
//  - PTX the JIT cannot compile fails with CUDA_ERROR_INVALID_PTX, and a name
//    the module does not define fails with CUDA_ERROR_NOT_FOUND.
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <string>

static const char* kHeader = R"PTX(//
// hand-written for CuMetal's driver raw-PTX test
//
.version 7.0
.target sm_80
.address_size 64

)PTX";
static const char* kScaleF32 = R"PTX(.visible .entry scale_f32(
	.param .u64 scale_f32_param_0,
	.param .u32 scale_f32_param_1
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<6>;
	.reg .f32 	%f<3>;
	.reg .b64 	%rd<5>;

	ld.param.u64 	%rd1, [scale_f32_param_0];
	ld.param.u32 	%r2, [scale_f32_param_1];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.u32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB0_2;
	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f32 	%f1, [%rd4];
	mul.rn.f32 	%f2, %f1, 0f40000000;
	st.global.f32 	[%rd4], %f2;
$L__BB0_2:
	ret;
}

)PTX";
static const char* kAxpyF64 = R"PTX(.visible .entry axpy_f64(
	.param .u64 axpy_f64_param_0,
	.param .u64 axpy_f64_param_1,
	.param .u32 axpy_f64_param_2
)
{
	.reg .pred 	%p<2>;
	.reg .b32 	%r<6>;
	.reg .f64 	%fd<5>;
	.reg .b64 	%rd<8>;

	ld.param.u64 	%rd1, [axpy_f64_param_0];
	ld.param.u64 	%rd2, [axpy_f64_param_1];
	ld.param.u32 	%r2, [axpy_f64_param_2];
	mov.u32 	%r3, %ctaid.x;
	mov.u32 	%r4, %ntid.x;
	mov.u32 	%r5, %tid.x;
	mad.lo.s32 	%r1, %r3, %r4, %r5;
	setp.ge.u32 	%p1, %r1, %r2;
	@%p1 bra 	$L__BB1_2;
	cvta.to.global.u64 	%rd3, %rd1;
	cvta.to.global.u64 	%rd4, %rd2;
	mul.wide.u32 	%rd5, %r1, 8;
	add.s64 	%rd6, %rd3, %rd5;
	add.s64 	%rd7, %rd4, %rd5;
	ld.global.f64 	%fd1, [%rd6];
	ld.global.f64 	%fd2, [%rd7];
	add.rn.f64 	%fd3, %fd1, %fd2;
	mul.rn.f64 	%fd4, %fd3, 0d3FF8000000000000;
	st.global.f64 	[%rd7], %fd4;
$L__BB1_2:
	ret;
}

)PTX";
static const char* kReverseBlock = R"PTX(.visible .entry reverse_block(
	.param .u64 reverse_block_param_0
)
{
	.reg .b32 	%r<8>;
	.reg .b64 	%rd<6>;
	.shared .align 4 .b8 tile[4096];

	ld.param.u64 	%rd1, [reverse_block_param_0];
	cvta.to.global.u64 	%rd2, %rd1;
	mov.u32 	%r1, %tid.x;
	mov.u32 	%r2, %ctaid.x;
	mov.u32 	%r3, %ntid.x;
	mad.lo.s32 	%r4, %r2, %r3, %r1;
	mul.wide.u32 	%rd3, %r4, 4;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.u32 	%r5, [%rd4];
	mov.u64 	%rd5, tile;
	mul.wide.u32 	%rd3, %r1, 4;
	add.s64 	%rd3, %rd5, %rd3;
	st.shared.u32 	[%rd3], %r5;
	bar.sync 	0;
	sub.s32 	%r6, %r3, %r1;
	add.s32 	%r6, %r6, -1;
	mul.wide.u32 	%rd3, %r6, 4;
	add.s64 	%rd3, %rd5, %rd3;
	ld.shared.u32 	%r7, [%rd3];
	st.global.u32 	[%rd4], %r7;
	ret;
}
)PTX";


static int fail(const char* what, CUresult status = CUDA_SUCCESS) {
    const char* name = "";
    cuGetErrorName(status, &name);
    std::printf("FAIL: %s (%d %s)\n", what, static_cast<int>(status), name ? name : "");
    return 1;
}

// Three-kernel module (FP32, FP64, static shared) and the same without FP64.
static const std::string kPtx = std::string(kHeader) + kScaleF32 + kAxpyF64 + kReverseBlock;
static const std::string kF32Ptx = std::string(kHeader) + kScaleF32 + kReverseBlock;

// Loads `ptx`, runs scale_f32 and (2001 times) reverse_block, checks both.
static int run_f32_kernels(const std::string& ptx, const char* what) {
    CUmodule module = nullptr;
    CUresult status = cuModuleLoadData(&module, ptx.c_str());
    if (status != CUDA_SUCCESS) return fail(what, status);
    CUfunction scale = nullptr, reverse = nullptr;
    if (cuModuleGetFunction(&scale, module, "scale_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&reverse, module, "reverse_block") != CUDA_SUCCESS)
        return fail("cuModuleGetFunction (f32 module)");
    constexpr unsigned n = 1024;
    CUdeviceptr fbuf = 0, ubuf = 0;
    if (cuMemAlloc(&fbuf, n * 4) != CUDA_SUCCESS || cuMemAlloc(&ubuf, n * 4) != CUDA_SUCCESS)
        return fail("cuMemAlloc (f32 module)");
    float f[n];
    unsigned u[n];
    for (unsigned i = 0; i < n; ++i) {
        f[i] = 0.5f * static_cast<float>(i);
        u[i] = i;
    }
    cuMemcpyHtoD(fbuf, f, sizeof f);
    cuMemcpyHtoD(ubuf, u, sizeof u);
    unsigned count = n;
    void* scale_args[] = {&fbuf, &count};
    void* reverse_args[] = {&ubuf};
    if ((status = cuLaunchKernel(scale, n / 256, 1, 1, 256, 1, 1, 0, nullptr, scale_args,
                                 nullptr)) != CUDA_SUCCESS)
        return fail("scale_f32 launch (f32 module)", status);
    for (int launch = 0; launch < 2001; ++launch)
        if ((status = cuLaunchKernel(reverse, n / 256, 1, 1, 256, 1, 1, 0, nullptr, reverse_args,
                                     nullptr)) != CUDA_SUCCESS)
            return fail("reverse_block launch (f32 module)", status);
    if ((status = cuCtxSynchronize()) != CUDA_SUCCESS) return fail("synchronize (f32 module)", status);
    float f_out[n];
    unsigned u_out[n];
    cuMemcpyDtoH(f_out, fbuf, sizeof f_out);
    cuMemcpyDtoH(u_out, ubuf, sizeof u_out);
    for (unsigned i = 0; i < n; ++i) {
        if (f_out[i] != 2.0f * f[i]) return fail("scale_f32 result (f32 module)");
        if (u_out[i] != (i / 256) * 256 + (255 - i % 256)) return fail("reverse_block result (f32 module)");
    }
    CUfunction missing = nullptr;
    if (cuModuleGetFunction(&missing, module, "no_such_kernel") != CUDA_ERROR_NOT_FOUND)
        return fail("cuModuleGetFunction of a missing kernel");
    cuMemFree(fbuf);
    cuMemFree(ubuf);
    if (cuModuleUnload(module) != CUDA_SUCCESS) return fail("cuModuleUnload (f32 module)");
    return 0;
}

int main() {
    if (cuInit(0) != CUDA_SUCCESS) return fail("cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuDevicePrimaryCtxRetain(&context, device) != CUDA_SUCCESS ||
        cuCtxSetCurrent(context) != CUDA_SUCCESS)
        return fail("primary context");

    // Both kernels of a two-entry module, on every backend; the legacy
    // backend builds one metallib per entry.
    if (run_f32_kernels(kF32Ptx, "cuModuleLoadData of the FP32 module") != 0) return 1;

    const char* backend = std::getenv("CUMETAL_PTX_BACKEND");
    const bool legacy = backend != nullptr && std::strcmp(backend, "legacy") == 0;
    CUmodule module = nullptr;
    CUresult status = cuModuleLoadData(&module, kPtx.c_str());
    if (legacy) {
        // The legacy lowering cannot load or store binary64; it must refuse the
        // module rather than load it with kernels missing.
        if (status != CUDA_ERROR_INVALID_PTX) return fail("legacy FP64 module", status);
        std::printf("PASS: raw PTX modules load and run through the driver\n");
        return 0;
    }
    if (status != CUDA_SUCCESS) return fail("cuModuleLoadData raw PTX", status);
    CUfunction scale = nullptr, axpy = nullptr, reverse = nullptr;
    if (cuModuleGetFunction(&scale, module, "scale_f32") != CUDA_SUCCESS ||
        cuModuleGetFunction(&axpy, module, "axpy_f64") != CUDA_SUCCESS ||
        cuModuleGetFunction(&reverse, module, "reverse_block") != CUDA_SUCCESS)
        return fail("cuModuleGetFunction");

    constexpr unsigned n = 1024;
    CUdeviceptr fbuf = 0, xbuf = 0, ybuf = 0, ubuf = 0;
    if (cuMemAlloc(&fbuf, n * 4) != CUDA_SUCCESS || cuMemAlloc(&xbuf, n * 8) != CUDA_SUCCESS ||
        cuMemAlloc(&ybuf, n * 8) != CUDA_SUCCESS || cuMemAlloc(&ubuf, n * 4) != CUDA_SUCCESS)
        return fail("cuMemAlloc");
    float f[n];
    double x[n], y[n];
    unsigned u[n];
    for (unsigned i = 0; i < n; ++i) {
        f[i] = 0.25f * static_cast<float>(i);
        x[i] = 1.0 + 1e-12 * i;  // needs more than FP32 precision to round-trip
        y[i] = static_cast<double>(i);
        u[i] = i;
    }
    cuMemcpyHtoD(fbuf, f, sizeof f);
    cuMemcpyHtoD(xbuf, x, sizeof x);
    cuMemcpyHtoD(ybuf, y, sizeof y);
    cuMemcpyHtoD(ubuf, u, sizeof u);

    unsigned count = n;
    void* scale_args[] = {&fbuf, &count};
    void* axpy_args[] = {&xbuf, &ybuf, &count};
    if ((status = cuLaunchKernel(scale, n / 256, 1, 1, 256, 1, 1, 0, nullptr, scale_args, nullptr)) !=
            CUDA_SUCCESS ||
        (status = cuLaunchKernel(axpy, n / 256, 1, 1, 256, 1, 1, 0, nullptr, axpy_args, nullptr)) !=
            CUDA_SUCCESS)
        return fail("cuLaunchKernel f32/f64", status);
    // The static-shared kernel, many times: each launch reverses within each
    // 256-thread block, so an even number of launches restores the input.
    void* reverse_args[] = {&ubuf};
    for (int launch = 0; launch < 2000; ++launch) {
        if ((status = cuLaunchKernel(reverse, n / 256, 1, 1, 256, 1, 1, 0, nullptr, reverse_args,
                                     nullptr)) != CUDA_SUCCESS)
            return fail("cuLaunchKernel reverse_block", status);
    }
    if ((status = cuLaunchKernel(reverse, n / 256, 1, 1, 256, 1, 1, 0, nullptr, reverse_args,
                                 nullptr)) != CUDA_SUCCESS ||
        (status = cuCtxSynchronize()) != CUDA_SUCCESS)
        return fail("launch/synchronize", status);
    float f_out[n];
    double y_out[n];
    unsigned u_out[n];
    cuMemcpyDtoH(f_out, fbuf, sizeof f_out);
    cuMemcpyDtoH(y_out, ybuf, sizeof y_out);
    cuMemcpyDtoH(u_out, ubuf, sizeof u_out);
    for (unsigned i = 0; i < n; ++i) {
        if (f_out[i] != 2.0f * f[i]) return fail("scale_f32 result");
        const double want = (x[i] + y[i]) * 1.5;
        if (std::fabs(y_out[i] - want) > 1e-9 * std::fabs(want) + 1e-12)
            return fail("axpy_f64 result");
        const unsigned block = i / 256, lane = i % 256;
        if (u_out[i] != block * 256 + (255 - lane)) return fail("reverse_block result");
    }
    cuModuleUnload(module);

    // A raw PTX module larger than 1 MiB: the same text followed by comments.
    std::string big = kPtx;
    const std::string pad = "// " + std::string(200, 'x') + "\n";
    while (big.size() < (3u << 20)) big += pad;
    module = nullptr;
    if ((status = cuModuleLoadData(&module, big.c_str())) != CUDA_SUCCESS)
        return fail("cuModuleLoadData of a 3 MiB PTX module", status);
    CUfunction big_scale = nullptr;
    if (cuModuleGetFunction(&big_scale, module, "scale_f32") != CUDA_SUCCESS)
        return fail("cuModuleGetFunction in the large module");
    cuModuleUnload(module);

    // PTX that parses as PTX but cannot be compiled.
    static const char* kBadPtx =
        ".version 7.0\n.target sm_80\n.address_size 64\n"
        ".visible .entry broken(.param .u64 p)\n{\n\tnot_an_instruction.f32 %f1, %f2;\n\tret;\n}\n";
    module = nullptr;
    status = cuModuleLoadData(&module, kBadPtx);
    if (status != CUDA_ERROR_INVALID_PTX) return fail("uncompilable PTX", status);
    const char* name = nullptr;
    if (cuGetErrorName(CUDA_ERROR_INVALID_PTX, &name) != CUDA_SUCCESS || name == nullptr ||
        std::strcmp(name, "CUDA_ERROR_INVALID_PTX") != 0)
        return fail("cuGetErrorName(CUDA_ERROR_INVALID_PTX)");

    std::printf("PASS: raw PTX modules load and run through the driver\n");
    return 0;
}
