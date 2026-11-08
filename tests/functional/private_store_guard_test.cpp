// A private store the typed PTX backend cannot bound -- the remainder loop of
// an unrolled fill whose trip count is a runtime 64-bit value, as COLMAP's
// GpuMatView::GetSlice compiles -- is guarded at run time instead of refused:
// a launch whose store would land on a proven pointer cell traps and reports
// a launch failure, and every other launch computes the right answer.
#include "cuda.h"

#include <cstdint>
#include <cstdio>

namespace {

// Local depot: [0,8) scratch, [8,16) a pointer cell holding &depot[16],
// [16,32) a float array. The loop stores n%4 floats starting at
// depot + 16 + 4*(n & ~3), then the kernel reloads the pointer cell and adds
// the first two floats it points to.
const char* kPtx = R"PTX(
.version 7.0
.target sm_80
.address_size 64
.visible .entry fill(.param .u64 out, .param .u64 src, .param .u64 n)
{
	.local .align 8 .b8 	__local_depot0[32];
	.reg .b64 	%SP;
	.reg .b64 	%SPL;
	.reg .pred 	%p<3>;
	.reg .b32 	%r<6>;
	.reg .b64 	%rd<20>;
	mov.b64 	%SPL, __local_depot0;
	cvta.local.u64 	%SP, %SPL;
	ld.param.u64 	%rd1, [out];
	ld.param.u64 	%rd2, [src];
	ld.param.u64 	%rd3, [n];
	cvta.to.global.u64 	%rd2, %rd2;
	add.u64 	%rd4, %SP, 16;
	st.local.b64 	[%SPL+8], %rd4;
	add.u64 	%rd5, %SPL, 16;
	and.b64 	%rd11, %rd3, -4;
	and.b64 	%rd6, %rd3, 3;
	setp.eq.s64 	%p1, %rd6, 0;
	@%p1 bra 	$L_done;
	shl.b64 	%rd12, %rd11, 2;
	add.s64 	%rd7, %rd5, %rd12;
	mov.b64 	%rd8, %rd2;
$L_loop:
	ld.global.b32 	%r1, [%rd8];
	st.local.b32 	[%rd7], %r1;
	add.s64 	%rd7, %rd7, 4;
	add.s64 	%rd8, %rd8, 4;
	add.s64 	%rd6, %rd6, -1;
	setp.ne.s64 	%p2, %rd6, 0;
	@%p2 bra 	$L_loop;
$L_done:
	ld.local.b64 	%rd9, [%SPL+8];
	ld.f32 	%r2, [%rd9];
	ld.f32 	%r3, [%rd9+4];
	add.rn.f32 	%r4, %r2, %r3;
	cvta.to.global.u64 	%rd10, %rd1;
	st.global.f32 	[%rd10], %r4;
	ret;
}
)PTX";

bool launch(CUfunction function, CUdeviceptr out, CUdeviceptr src, std::uint64_t n, CUresult* status) {
    void* args[] = {&out, &src, &n};
    if (cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuLaunchKernel rejected the launch itself\n");
        return false;
    }
    *status = cuCtxSynchronize();
    return true;
}

}  // namespace

int main() {
    CUdevice device;
    CUcontext context;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: driver init\n");
        return 1;
    }
    CUmodule module;
    CUfunction function;
    if (cuModuleLoadData(&module, kPtx) != CUDA_SUCCESS ||
        cuModuleGetFunction(&function, module, "fill") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: module load\n");
        return 1;
    }
    const float host_src[4] = {1.5f, 2.25f, 4.0f, 8.0f};
    CUdeviceptr src, out;
    if (cuMemAlloc(&src, sizeof(host_src)) != CUDA_SUCCESS || cuMemAlloc(&out, sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(src, host_src, sizeof(host_src)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: allocation\n");
        return 1;
    }

    // n = 3: three stores at depot offsets 16, 20, 24; no cell is touched.
    CUresult status;
    if (!launch(function, out, src, 3, &status)) return 1;
    float result = 0.0f;
    if (status != CUDA_SUCCESS || cuMemcpyDtoH(&result, out, sizeof(result)) != CUDA_SUCCESS ||
        result != host_src[0] + host_src[1]) {
        std::fprintf(stderr, "FAIL: in-bounds launch status=%d result=%g (want %g)\n",
                     static_cast<int>(status), result, host_src[0] + host_src[1]);
        return 1;
    }

    // n = 2^62 - 1: the cursor starts at 16 + 4*(2^62 - 4) = 0 (mod 2^64), so the
    // third store would overwrite the pointer cell at offset 8. It must trap.
    if (!launch(function, out, src, (std::uint64_t{1} << 62) - 1, &status)) return 1;
    if (status != CUDA_ERROR_LAUNCH_FAILED) {
        std::fprintf(stderr, "FAIL: cell-overwriting launch returned %d, want CUDA_ERROR_LAUNCH_FAILED\n",
                     static_cast<int>(status));
        return 1;
    }
    std::printf("PASS: unbounded private store guarded (correct in bounds, traps on the pointer cell)\n");
    return 0;
}
