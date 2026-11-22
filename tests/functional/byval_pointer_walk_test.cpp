#include "cuda.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// Mirrors fixtures/byval_pointer_walk.cu: eight output columns passed by value.
struct Columns {
    CUdeviceptr values[8];
};

int main(int argc, char** argv) {
    if (argc != 2) {
        return 2;
    }

    constexpr unsigned int kRows = 300;
    constexpr int kColumns = 8;

    CUdevice device = 0;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr source = 0;
    Columns columns{};

    if (cuInit(0) != CUDA_SUCCESS || cuDeviceGet(&device, 0) != CUDA_SUCCESS ||
        cuCtxCreate(&context, 0, device) != CUDA_SUCCESS ||
        cuModuleLoad(&module, argv[1]) != CUDA_SUCCESS ||
        cuModuleGetFunction(&kernel, module, "byval_pointer_walk") != CUDA_SUCCESS ||
        cuMemAlloc(&source, kRows * sizeof(std::uint32_t)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval pointer walk: setup failed\n");
        return 1;
    }
    std::vector<std::uint32_t> input(kRows);
    for (unsigned int i = 0; i < kRows; ++i) input[i] = 0x9e3779b9u * (i + 1u);
    const std::vector<std::uint32_t> poison(kRows, 0xdeadbeefu);
    if (cuMemcpyHtoD(source, input.data(), kRows * sizeof(std::uint32_t)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval pointer walk: input copy failed\n");
        return 1;
    }
    for (int c = 0; c < kColumns; ++c) {
        if (cuMemAlloc(&columns.values[c], kRows * sizeof(std::uint32_t)) != CUDA_SUCCESS ||
            cuMemcpyHtoD(columns.values[c], poison.data(), kRows * sizeof(std::uint32_t)) !=
                CUDA_SUCCESS) {
            std::fprintf(stderr, "byval pointer walk: column %d setup failed\n", c);
            return 1;
        }
    }

    unsigned int rows = kRows;
    void* args[] = {&source, &rows, &columns};
    const CUresult launched =
        cuLaunchKernel(kernel, (kRows + 127u) / 128u, 1, 1, 128, 1, 1, 0, nullptr, args, nullptr);
    if (launched != CUDA_SUCCESS || cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "byval pointer walk: launch failed (%d)\n", static_cast<int>(launched));
        return 1;
    }

    std::vector<std::uint32_t> host(kRows);
    for (int c = 0; c < kColumns; ++c) {
        if (cuMemcpyDtoH(host.data(), columns.values[c], kRows * sizeof(std::uint32_t)) !=
            CUDA_SUCCESS) {
            std::fprintf(stderr, "byval pointer walk: readback failed\n");
            return 1;
        }
        for (unsigned int i = 0; i < kRows; ++i) {
            const std::uint32_t expected =
                input[i] * static_cast<std::uint32_t>(c + 1) + static_cast<std::uint32_t>(c);
            if (host[i] != expected) {
                std::fprintf(stderr,
                             "byval pointer walk: column %d row %u is 0x%08x, expected 0x%08x\n",
                             c, i, host[i], expected);
                return 1;
            }
        }
        cuMemFree(columns.values[c]);
    }

    cuMemFree(source);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    std::puts("PASS: by-value pointer aggregate walked through parameter space");
    return 0;
}
