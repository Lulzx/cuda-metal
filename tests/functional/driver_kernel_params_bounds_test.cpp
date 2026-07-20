#include "cuda.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <sys/mman.h>
#include <unistd.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <experimental-metallib>\n", argv[0]);
        return 64;
    }

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&function, module, "vector_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: driver setup failed\n");
        return 1;
    }

    CUdeviceptr a = 0;
    CUdeviceptr b = 0;
    CUdeviceptr c = 0;
    if (cuMemAlloc(&a, sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&b, sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&c, sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: device allocation failed\n");
        return 1;
    }

    const long page_size_long = sysconf(_SC_PAGESIZE);
    if (page_size_long <= 0) {
        std::fprintf(stderr, "FAIL: sysconf(_SC_PAGESIZE) failed\n");
        return 1;
    }
    const std::size_t page_size = static_cast<std::size_t>(page_size_long);
    void* mapping = mmap(nullptr,
                         page_size * 2,
                         PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS,
                         -1,
                         0);
    if (mapping == MAP_FAILED ||
        mprotect(static_cast<char*>(mapping) + page_size, page_size, PROT_NONE) != 0) {
        std::fprintf(stderr, "FAIL: guarded parameter-array allocation failed\n");
        return 1;
    }

    // A CUDA Driver API kernelParams array has exactly one slot per parameter;
    // it has no null terminator. Put it directly before a protected page so an
    // implementation that scans for a sentinel fails deterministically.
    auto** params = reinterpret_cast<void**>(
        static_cast<char*>(mapping) + page_size - 3 * sizeof(void*));
    params[0] = &a;
    params[1] = &b;
    params[2] = &c;

    (void)cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, nullptr, params, nullptr);

    munmap(mapping, page_size * 2);
    cuMemFree(a);
    cuMemFree(b);
    cuMemFree(c);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    std::printf("PASS: driver kernelParams uses compiler-provided argument count\n");
    return 0;
}
