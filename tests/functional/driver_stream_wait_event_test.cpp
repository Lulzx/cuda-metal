#include "cuda.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 15;
constexpr std::size_t kThreadsPerBlock = 256;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

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

    CUmodule module = nullptr;
    if (cuModuleLoad(&module, metallib_path.c_str()) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleLoad failed\n");
        return 1;
    }

    CUfunction vector_add = nullptr;
    if (cuModuleGetFunction(&vector_add, module, "vector_add") != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleGetFunction failed\n");
        return 1;
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);
    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 3) % 41) * 0.5f;
        host_b[i] = static_cast<float>((i * 5) % 29) * 0.75f;
    }

    CUdeviceptr dev_a = 0;
    CUdeviceptr dev_b = 0;
    CUdeviceptr dev_tmp = 0;
    CUdeviceptr dev_out = 0;

    if (cuMemAlloc(&dev_a, kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_b, kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_tmp, kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemAlloc(&dev_out, kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemAlloc failed\n");
        return 1;
    }

    if (cuMemcpyHtoD(dev_a, host_a.data(), kElementCount * sizeof(float)) != CUDA_SUCCESS ||
        cuMemcpyHtoD(dev_b, host_b.data(), kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyHtoD failed\n");
        return 1;
    }

    CUstream stream1 = nullptr;
    CUstream stream2 = nullptr;
    if (cuStreamCreate(&stream1, 0) != CUDA_SUCCESS || cuStreamCreate(&stream2, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamCreate failed\n");
        return 1;
    }

    CUevent stage1_done = nullptr;
    if (cuEventCreate(&stage1_done, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuEventCreate failed\n");
        return 1;
    }

    const unsigned int grid_x =
        static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) / kThreadsPerBlock);

    CUdeviceptr stage1_a = dev_a;
    CUdeviceptr stage1_b = dev_b;
    CUdeviceptr stage1_tmp = dev_tmp;
    void* stage1_params[] = {&stage1_a, &stage1_b, &stage1_tmp, nullptr};

    if (cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       stream1,
                       stage1_params,
                       nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: stage1 cuLaunchKernel failed\n");
        return 1;
    }

    if (cuEventRecord(stage1_done, stream1) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuEventRecord failed\n");
        return 1;
    }

    const CUresult first_query = cuEventQuery(stage1_done);
    if (first_query != CUDA_SUCCESS && first_query != CUDA_ERROR_NOT_READY) {
        std::fprintf(stderr, "FAIL: unexpected cuEventQuery status %d\n", first_query);
        return 1;
    }

    if (cuStreamWaitEvent(stream2, stage1_done, 0) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamWaitEvent failed\n");
        return 1;
    }

    if (cuEventQuery(stage1_done) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: event should be complete after stream wait\n");
        return 1;
    }

    CUdeviceptr stage2_tmp = dev_tmp;
    CUdeviceptr stage2_b = dev_b;
    CUdeviceptr stage2_out = dev_out;
    void* stage2_params[] = {&stage2_tmp, &stage2_b, &stage2_out, nullptr};

    if (cuLaunchKernel(vector_add,
                       grid_x,
                       1,
                       1,
                       static_cast<unsigned int>(kThreadsPerBlock),
                       1,
                       1,
                       0,
                       stream2,
                       stage2_params,
                       nullptr) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: stage2 cuLaunchKernel failed\n");
        return 1;
    }

    if (cuStreamSynchronize(stream2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamSynchronize failed\n");
        return 1;
    }

    if (cuMemcpyDtoH(host_out.data(), dev_out, kElementCount * sizeof(float)) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemcpyDtoH failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + 2.0f * host_b[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    // cuEventElapsedTime answers CUDA_ERROR_INVALID_HANDLE for an event that
    // was never recorded or that was created with CU_EVENT_DISABLE_TIMING.
    {
        CUevent never = nullptr;
        CUevent untimed = nullptr;
        float ms = 0.0f;
        if (cuEventCreate(&never, 0) != CUDA_SUCCESS ||
            cuEventCreate(&untimed, CU_EVENT_DISABLE_TIMING) != CUDA_SUCCESS ||
            cuEventRecord(untimed, stream1) != CUDA_SUCCESS ||
            cuEventSynchronize(untimed) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: elapsed-time event setup failed\n");
            return 1;
        }
        if (cuEventElapsedTime(&ms, stage1_done, never) != CUDA_ERROR_INVALID_HANDLE ||
            cuEventElapsedTime(&ms, stage1_done, untimed) != CUDA_ERROR_INVALID_HANDLE) {
            std::fprintf(stderr,
                         "FAIL: cuEventElapsedTime should reject unrecorded and untimed "
                         "events with CUDA_ERROR_INVALID_HANDLE\n");
            return 1;
        }
        if (cuEventDestroy(never) != CUDA_SUCCESS || cuEventDestroy(untimed) != CUDA_SUCCESS) {
            std::fprintf(stderr, "FAIL: elapsed-time event teardown failed\n");
            return 1;
        }
    }

    if (cuEventDestroy(stage1_done) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuEventDestroy failed\n");
        return 1;
    }

    if (cuStreamDestroy(stream1) != CUDA_SUCCESS || cuStreamDestroy(stream2) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuStreamDestroy failed\n");
        return 1;
    }

    if (cuMemFree(dev_a) != CUDA_SUCCESS || cuMemFree(dev_b) != CUDA_SUCCESS ||
        cuMemFree(dev_tmp) != CUDA_SUCCESS || cuMemFree(dev_out) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuMemFree failed\n");
        return 1;
    }

    if (cuModuleUnload(module) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuModuleUnload failed\n");
        return 1;
    }

    if (cuCtxDestroy(context) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuCtxDestroy failed\n");
        return 1;
    }

    std::printf("PASS: driver stream/event ordering works\n");
    return 0;
}
