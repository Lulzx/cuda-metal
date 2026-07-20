#include "cuda.h"

#include <atomic>
#include <cstdio>
#include <thread>

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

    CUcontext ctx1 = nullptr;
    CUcontext ctx2 = nullptr;
    if (cuCtxCreate(&ctx1, 0, device) != CUDA_SUCCESS || ctx1 == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate(ctx1) failed\n");
        return 1;
    }
    if (cuCtxCreate(&ctx2, 0, device) != CUDA_SUCCESS || ctx2 == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate(ctx2) failed\n");
        return 1;
    }
    CUcontext ctx3 = nullptr;
    const unsigned int create_flags = CU_CTX_SCHED_BLOCKING_SYNC | CU_CTX_MAP_HOST;
    if (cuCtxCreate(&ctx3, create_flags, device) != CUDA_SUCCESS || ctx3 == nullptr) {
        std::fprintf(stderr, "FAIL: cuCtxCreate(ctx3, flags) failed\n");
        return 1;
    }

    std::atomic<bool> worker_ok{true};
    std::thread worker([&]() {
        CUcontext worker_current = reinterpret_cast<CUcontext>(0x1);
        worker_ok = cuCtxGetCurrent(&worker_current) == CUDA_SUCCESS &&
                    worker_current == nullptr;
        worker_ok = worker_ok && cuCtxPushCurrent(ctx2) == CUDA_SUCCESS;
        worker_ok = worker_ok && cuCtxGetCurrent(&worker_current) == CUDA_SUCCESS &&
                    worker_current == ctx2;
        CUcontext popped = nullptr;
        worker_ok = worker_ok && cuCtxPopCurrent(&popped) == CUDA_SUCCESS &&
                    popped == ctx2;
        worker_ok = worker_ok && cuCtxGetCurrent(&worker_current) == CUDA_SUCCESS &&
                    worker_current == nullptr;
    });
    worker.join();
    if (!worker_ok) {
        std::fprintf(stderr, "FAIL: worker thread context stack is not isolated\n");
        return 1;
    }

    CUcontext current = reinterpret_cast<CUcontext>(0x1);
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx1) {
        std::fprintf(stderr, "FAIL: expected ctx1 as initial current context\n");
        return 1;
    }
    unsigned int current_flags = 0xdeadbeefu;
    if (cuCtxGetFlags(&current_flags) != CUDA_SUCCESS || current_flags != 0u) {
        std::fprintf(stderr, "FAIL: expected cuCtxGetFlags to report context creation flags\n");
        return 1;
    }
    const unsigned int updated_flags = CU_CTX_SCHED_YIELD | CU_CTX_MAP_HOST;
    if (cuCtxSetFlags(updated_flags) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetFlags(valid) failed\n");
        return 1;
    }
    if (cuCtxGetFlags(&current_flags) != CUDA_SUCCESS || current_flags != updated_flags) {
        std::fprintf(stderr, "FAIL: cuCtxGetFlags should return updated context flags\n");
        return 1;
    }
    if (cuCtxSetFlags(CU_CTX_SCHED_SPIN | CU_CTX_SCHED_YIELD) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuCtxSetFlags should reject conflicting schedule flags\n");
        return 1;
    }
    if (cuCtxSetFlags(0x80u) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuCtxSetFlags should reject unsupported flags\n");
        return 1;
    }
    CUdevice current_device = -1;
    if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device) {
        std::fprintf(stderr, "FAIL: expected cuCtxGetDevice to report active device\n");
        return 1;
    }

    if (cuCtxSetCurrent(ctx2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx2) failed\n");
        return 1;
    }
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != ctx2) {
        std::fprintf(stderr, "FAIL: expected ctx2 as current context\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx3) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx3) failed\n");
        return 1;
    }
    if (cuCtxGetFlags(&current_flags) != CUDA_SUCCESS || current_flags != create_flags) {
        std::fprintf(stderr, "FAIL: cuCtxGetFlags should reflect flags passed to cuCtxCreate\n");
        return 1;
    }
    if (cuCtxSetCurrent(ctx2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx2) second switch failed\n");
        return 1;
    }

    if (cuCtxSetCurrent(nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(nullptr) failed\n");
        return 1;
    }
    if (cuCtxGetCurrent(&current) != CUDA_SUCCESS || current != nullptr) {
        std::fprintf(stderr, "FAIL: expected null current context\n");
        return 1;
    }
    if (cuCtxGetFlags(&current_flags) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT from cuCtxGetFlags without context\n");
        return 1;
    }
    if (cuCtxSetFlags(CU_CTX_SCHED_AUTO) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT from cuCtxSetFlags without context\n");
        return 1;
    }
    if (cuCtxGetDevice(&current_device) != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT from cuCtxGetDevice without context\n");
        return 1;
    }
    if (cuCtxSynchronize() != CUDA_ERROR_INVALID_CONTEXT) {
        std::fprintf(stderr, "FAIL: expected CUDA_ERROR_INVALID_CONTEXT without current context\n");
        return 1;
    }

    if (cuCtxSetCurrent(ctx1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSetCurrent(ctx1) failed\n");
        return 1;
    }
    if (cuCtxSynchronize() != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxSynchronize with ctx1 failed\n");
        return 1;
    }
    if (cuCtxGetDevice(&current_device) != CUDA_SUCCESS || current_device != device) {
        std::fprintf(stderr, "FAIL: cuCtxGetDevice should succeed after restoring current context\n");
        return 1;
    }
    if (cuCtxGetFlags(&current_flags) != CUDA_SUCCESS || current_flags != updated_flags) {
        std::fprintf(stderr, "FAIL: cuCtxGetFlags should succeed after restoring current context\n");
        return 1;
    }

    if (cuCtxDestroy(ctx2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy(ctx2) failed\n");
        return 1;
    }
    if (cuCtxDestroy(ctx3) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy(ctx3) failed\n");
        return 1;
    }
    if (cuCtxDestroy(ctx1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy(ctx1) failed\n");
        return 1;
    }

    std::printf("PASS: driver context current-context APIs behave correctly\n");
    return 0;
}
