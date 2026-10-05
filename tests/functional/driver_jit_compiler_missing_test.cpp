#include "cuda.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

// cuModuleLoadData on PTX whose kernel needs Apple's offline Metal compiler
// (FP64 links the VF64 support module). Run with the toolchain hidden, the
// load must fail with CUDA_ERROR_JIT_COMPILER_NOT_FOUND rather than return a
// module whose every launch then fails without naming the cause.
//
// usage: driver_jit_compiler_missing_test <ptx> expect-missing|expect-ok

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s <ptx> expect-missing|expect-ok\n", argv[0]);
        return 64;
    }
    const bool expect_missing = std::strcmp(argv[2], "expect-missing") == 0;

    std::ifstream in(argv[1], std::ios::binary);
    std::vector<char> ptx((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (ptx.empty()) {
        std::fprintf(stderr, "FAIL: failed to read %s\n", argv[1]);
        return 1;
    }
    ptx.push_back('\0');

    CUdevice device = 0;
    CUcontext context = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: context setup failed\n");
        return 1;
    }

    CUmodule module = nullptr;
    const CUresult status = cuModuleLoadData(&module, ptx.data());
    const char* name = nullptr;
    cuGetErrorName(status, &name);
    if (expect_missing) {
        if (status != CUDA_ERROR_JIT_COMPILER_NOT_FOUND) {
            std::fprintf(stderr, "FAIL: cuModuleLoadData returned %d (%s), expected %d\n",
                         static_cast<int>(status), name ? name : "?",
                         static_cast<int>(CUDA_ERROR_JIT_COMPILER_NOT_FOUND));
            return 1;
        }
        if (name == nullptr || std::strcmp(name, "CUDA_ERROR_JIT_COMPILER_NOT_FOUND") != 0) {
            std::fprintf(stderr, "FAIL: cuGetErrorName gave %s\n", name ? name : "null");
            return 1;
        }
    } else if (status != CUDA_SUCCESS || module == nullptr) {
        std::fprintf(stderr, "FAIL: cuModuleLoadData returned %d (%s) with a toolchain\n",
                     static_cast<int>(status), name ? name : "?");
        return 1;
    }
    if (module != nullptr) {
        cuModuleUnload(module);
    }
    cuCtxDestroy(context);
    std::printf("PASS: %s\n", argv[2]);
    return 0;
}
