// Runs the inline_asm_warp_sum kernel from a cumetalc-built metallib and checks
// PTX semantics: `shfl.sync.down.b32 r0|p` sets p only for an in-range source
// lane, and the guarded `@p add` runs exactly where p is set.
#include <cuda.h>

#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <metallib>\n", argv[0]);
        return 64;
    }
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&function, module, "warp_sum") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: could not load warp_sum from %s\n", argv[1]);
        return 1;
    }
    CUdeviceptr out = 0;
    if (cuMemAlloc(&out, 32 * sizeof(float)) != CUDA_SUCCESS) return 1;
    cuMemsetD32(out, 0, 32);
    void* args[] = {&out};
    if (cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, nullptr, args, nullptr) != CUDA_SUCCESS ||
        cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: launch\n");
        return 1;
    }
    float host[32] = {};
    cuMemcpyDtoH(host, out, sizeof(host));
    int wrong = 0;
    for (int lane = 0; lane < 32; ++lane) {
        const float want = lane < 31 ? static_cast<float>(lane + lane + 1) : 31.0f;
        if (host[lane] != want && wrong++ < 4) {
            std::fprintf(stderr, "FAIL: lane %d = %g, want %g\n", lane, host[lane], want);
        }
    }
    cuMemFree(out);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    if (wrong != 0) return 1;
    std::printf("PASS: inline-asm warp shuffle with predicate output\n");
    return 0;
}
