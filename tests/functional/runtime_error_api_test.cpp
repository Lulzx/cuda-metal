#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <thread>

extern "C" {
void** __cudaRegisterFatBinary(const void* fat_cubin);
void __cudaUnregisterFatBinary(void** fat_cubin_handle);
void __cudaRegisterFunction(void** fat_cubin_handle,
                            const void* host_function,
                            char* device_function,
                            const char* device_name,
                            int thread_limit,
                            void* thread_id,
                            void* block_id,
                            void* block_dim,
                            void* grid_dim,
                            int* warp_size);
}

namespace {

void missing_registered_kernel_stub() {}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: initial cudaGetLastError should be cudaSuccess\n");
        return 1;
    }

    if (cudaSetDevice(1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaSetDevice(1) should fail\n");
        return 1;
    }
    if (cudaPeekAtLastError() != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaPeekAtLastError should report last failure\n");
        return 1;
    }
    if (cudaGetLastError() != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaGetLastError should report last failure\n");
        return 1;
    }
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGetLastError should clear error state\n");
        return 1;
    }

    if (cudaSetDevice(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaSetDevice(0) failed\n");
        return 1;
    }
    if (cudaPeekAtLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: successful calls should leave success last-error state\n");
        return 1;
    }

    bool thread_ok = true;
    const char* thread_fail = nullptr;
    std::thread worker([&thread_ok, &thread_fail]() {
        if (cudaSetDevice(1) != cudaErrorInvalidValue) {
            thread_ok = false;
            thread_fail = "FAIL: worker cudaSetDevice(1) should fail";
            return;
        }
        if (cudaPeekAtLastError() != cudaErrorInvalidValue) {
            thread_ok = false;
            thread_fail = "FAIL: worker cudaPeekAtLastError should report worker failure";
            return;
        }
        if (cudaGetLastError() != cudaErrorInvalidValue) {
            thread_ok = false;
            thread_fail = "FAIL: worker cudaGetLastError should report worker failure";
            return;
        }
        if (cudaGetLastError() != cudaSuccess) {
            thread_ok = false;
            thread_fail = "FAIL: worker cudaGetLastError should clear worker state";
            return;
        }
    });
    worker.join();
    if (!thread_ok) {
        std::fprintf(stderr, "%s\n", thread_fail);
        return 1;
    }

    if (cudaPeekAtLastError() != cudaSuccess || cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: worker-thread errors should not leak into main thread\n");
        return 1;
    }

    const char* unknown_name = cudaGetErrorName(static_cast<cudaError_t>(12345));
    const char* unknown_string = cudaGetErrorString(static_cast<cudaError_t>(12345));
    if (unknown_name == nullptr || unknown_string == nullptr) {
        std::fprintf(stderr, "FAIL: cudaGetErrorName/String unknown code should not return null\n");
        return 1;
    }
    if (std::strcmp(unknown_name, "cudaErrorUnknown") != 0 ||
        std::strcmp(unknown_string, "cudaErrorUnknown") != 0) {
        std::fprintf(stderr, "FAIL: unknown runtime error should map to cudaErrorUnknown\n");
        return 1;
    }

    if (std::strcmp(cudaGetErrorName(cudaErrorLaunchTimeout), "cudaErrorLaunchTimeout") != 0 ||
        std::strcmp(cudaGetErrorString(cudaErrorLaunchTimeout), "cudaErrorLaunchTimeout") != 0) {
        std::fprintf(stderr, "FAIL: launch-timeout error name/string mismatch\n");
        return 1;
    }

    if (std::strcmp(cudaGetErrorName(cudaErrorIllegalAddress), "cudaErrorIllegalAddress") != 0 ||
        std::strcmp(cudaGetErrorString(cudaErrorIllegalAddress), "cudaErrorIllegalAddress") != 0) {
        std::fprintf(stderr, "FAIL: illegal-address error name/string mismatch\n");
        return 1;
    }

    if (std::strcmp(cudaGetErrorName(cudaErrorDevicesUnavailable), "cudaErrorDevicesUnavailable") !=
            0 ||
        std::strcmp(cudaGetErrorString(cudaErrorDevicesUnavailable),
                    "cudaErrorDevicesUnavailable") != 0) {
        std::fprintf(stderr, "FAIL: devices-unavailable error name/string mismatch\n");
        return 1;
    }

    if (std::strcmp(cudaGetErrorName(cudaErrorPeerAccessAlreadyEnabled),
                    "cudaErrorPeerAccessAlreadyEnabled") != 0 ||
        std::strcmp(cudaGetErrorString(cudaErrorPeerAccessAlreadyEnabled),
                    "cudaErrorPeerAccessAlreadyEnabled") != 0 ||
        std::strcmp(cudaGetErrorName(cudaErrorPeerAccessNotEnabled),
                    "cudaErrorPeerAccessNotEnabled") != 0 ||
        std::strcmp(cudaGetErrorString(cudaErrorPeerAccessNotEnabled),
                    "cudaErrorPeerAccessNotEnabled") != 0) {
        std::fprintf(stderr, "FAIL: peer-access error name/string mismatch\n");
        return 1;
    }

    if (std::strcmp(cudaGetErrorName(cudaErrorGraphExecUpdateFailure),
                    "cudaErrorGraphExecUpdateFailure") != 0 ||
        std::strcmp(cudaGetErrorString(cudaErrorGraphExecUpdateFailure),
                    "graph executable update failure") != 0) {
        std::fprintf(stderr, "FAIL: graph-update error name/string mismatch\n");
        return 1;
    }

    // Use a real metallib but request a function that cannot exist.  This
    // reaches Metal pipeline lookup and models a generated <<<>>> host stub,
    // whose cudaLaunchKernel return value is not available to the caller.
    const cumetalKernel_t missing_kernel{
        .metallib_path = argv[1],
        .kernel_name = "__cumetal_missing_kernel_for_error_test__",
        .arg_count = 0,
        .arg_info = nullptr,
    };
    (void)cudaLaunchKernel(&missing_kernel, dim3(1), dim3(1), nullptr, 0, nullptr);
    if (cudaGetLastError() != cudaErrorInvalidValue ||
        cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr,
                     "FAIL: failed launch should set a get-and-clearable immediate error\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaErrorInvalidValue) {
        std::fprintf(stderr,
                     "FAIL: cudaDeviceSynchronize should propagate a failed launch\n");
        return 1;
    }
    if (cudaGetLastError() != cudaErrorInvalidValue ||
        cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: synchronized launch error should remain get-and-clearable\n");
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize should consume the launch error once\n");
        return 1;
    }

    // Lazy registration/JIT validation fails before Metal submission. This
    // early return used to bypass the pending-launch slot, allowing generated
    // <<<...>>> code to discard the immediate error and observe a false
    // cudaSuccess from cudaDeviceSynchronize.
    void** empty_fatbin = __cudaRegisterFatBinary(nullptr);
    if (empty_fatbin == nullptr) {
        std::fprintf(stderr, "FAIL: empty registration handle creation failed\n");
        return 1;
    }
    __cudaRegisterFunction(empty_fatbin,
                           reinterpret_cast<const void*>(&missing_registered_kernel_stub),
                           nullptr,
                           "missing_registered_kernel",
                           0,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);
    void* no_kernel_arguments[] = {nullptr};
    (void)cudaLaunchKernel(reinterpret_cast<const void*>(&missing_registered_kernel_stub),
                           dim3(1), dim3(1), no_kernel_arguments, 0, nullptr);
    if (cudaGetLastError() != cudaErrorInvalidValue ||
        cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr,
                     "FAIL: registered launch should expose its immediate validation error\n");
        __cudaUnregisterFatBinary(empty_fatbin);
        return 1;
    }
    if (cudaDeviceSynchronize() != cudaErrorInvalidValue) {
        std::fprintf(stderr,
                     "FAIL: synchronization swallowed an early registered-launch failure\n");
        __cudaUnregisterFatBinary(empty_fatbin);
        return 1;
    }
    if (cudaGetLastError() != cudaErrorInvalidValue ||
        cudaGetLastError() != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr,
                     "FAIL: registered launch error was not get-and-clearable exactly once\n");
        __cudaUnregisterFatBinary(empty_fatbin);
        return 1;
    }
    __cudaUnregisterFatBinary(empty_fatbin);

    std::printf("PASS: runtime error APIs behave correctly\n");
    return 0;
}
