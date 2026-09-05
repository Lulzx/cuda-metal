#include "cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdio>

// Mirrors the fixture's parameter type exactly; the point of the test is that
// the launch reads all 32 bytes of it, not just the first pointer-sized word.
struct LaunchBounds {
    int shape[4];
    int ndim;
    size_t size;
};

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }

    constexpr int kThreads = 4;

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr out = 0;

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&kernel, module, "byval_aggregate_launch") != CUDA_SUCCESS ||
        cuMemAlloc(&out, kThreads * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval aggregate launch: setup failed\n");
        return 1;
    }

    // shape[3] and size sit past the first eight bytes, so a launch that binds
    // only a pointer-sized word reads them as zero and the kernel writes
    // nothing at all.
    LaunchBounds bounds{};
    bounds.shape[3] = 8;
    bounds.ndim = 1;
    bounds.size = kThreads;
    float scale = 100.0f;

    void* args[] = {&bounds, &scale, &out};
    const CUresult launched =
        cuLaunchKernel(kernel, 1, 1, 1, kThreads, 1, 1, 0, nullptr, args, nullptr);
    if (launched != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval aggregate launch: cuLaunchKernel failed (%d)\n",
                     static_cast<int>(launched));
        return 1;
    }

    float host[kThreads] = {-1.0f, -1.0f, -1.0f, -1.0f};
    if (cuCtxSynchronize() != CUDA_SUCCESS ||
        cuMemcpyDtoH(host, out, sizeof(host)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval aggregate launch: readback failed\n");
        return 1;
    }

    for (int i = 0; i < kThreads; ++i) {
        if (std::fabs(host[i] - 108.0f) > 1.0e-6f) {
            std::fprintf(stderr,
                         "byval aggregate launch: element %d is %.9g, expected 108\n",
                         i, host[i]);
            return 1;
        }
    }

    cuMemFree(out);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
