#include "cuda.h"

#include <cstdio>

int main() {
    if (cuInit(0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuInit failed\n");
        return 1;
    }

    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuDeviceGet failed\n");
        return 1;
    }

    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxCreate failed\n");
        return 1;
    }

    CUstream stream = nullptr;

    if (cuStreamCreate(&stream, CU_STREAM_DEFAULT) != CUDA_SUCCESS || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreate(default) failed\n");
        return 1;
    }
    unsigned int observed_flags = ~0u;
    if (cuStreamGetFlags(stream, &observed_flags) != CUDA_SUCCESS ||
        observed_flags != CU_STREAM_DEFAULT) {
        std::fprintf(stderr, "FAIL: default stream flags were not preserved\n");
        return 1;
    }
    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy(default stream) failed\n");
        return 1;
    }

    stream = nullptr;
    if (cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cuStreamCreate(nonblocking) failed\n");
        return 1;
    }
    observed_flags = ~0u;
    if (cuStreamGetFlags(stream, &observed_flags) != CUDA_SUCCESS ||
        observed_flags != CU_STREAM_NON_BLOCKING) {
        std::fprintf(stderr, "FAIL: nonblocking stream flags were not preserved\n");
        return 1;
    }
    if (cuStreamDestroy(stream) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy(nonblocking stream) failed\n");
        return 1;
    }

    if (cuStreamCreate(&stream, 0x100u) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_VALUE for unsupported stream flag\n");
        return 1;
    }

    if (cuStreamSynchronize(CU_STREAM_PER_THREAD) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamSynchronize(CU_STREAM_PER_THREAD) failed\n");
        return 1;
    }
    if (cuStreamQuery(CU_STREAM_LEGACY) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamQuery(CU_STREAM_LEGACY) failed\n");
        return 1;
    }
    if (cuStreamDestroy(CU_STREAM_LEGACY) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: destroying CU_STREAM_LEGACY should be invalid\n");
        return 1;
    }
    if (cuStreamDestroy(CU_STREAM_PER_THREAD) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: destroying CU_STREAM_PER_THREAD should be invalid\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: stream flag validation and special stream handles\n");
    return 0;
}
