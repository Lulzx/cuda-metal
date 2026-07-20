#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <thread>

int main() {
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

    std::printf("PASS: runtime error APIs behave correctly\n");
    return 0;
}
