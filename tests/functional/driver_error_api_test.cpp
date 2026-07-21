#include "cuda.h"

#include <cstdio>
#include <cstring>

int main() {
    if (cuGetErrorName(CUDA_SUCCESS, nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuGetErrorName should reject null output pointer\n");
        return 1;
    }
    if (cuGetErrorString(CUDA_SUCCESS, nullptr) != CUDA_ERROR_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuGetErrorString should reject null output pointer\n");
        return 1;
    }

    const char* name = nullptr;
    const char* description = nullptr;
    if (cuGetErrorName(CUDA_ERROR_INVALID_VALUE, &name) != CUDA_SUCCESS ||
        cuGetErrorString(CUDA_ERROR_INVALID_VALUE, &description) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuGetErrorName/String failed for known code\n");
        return 1;
    }
    if (name == nullptr || description == nullptr) {
        std::fprintf(stderr, "FAIL: cuGetErrorName/String returned null\n");
        return 1;
    }
    if (std::strcmp(name, "CUDA_ERROR_INVALID_VALUE") != 0 ||
        std::strcmp(description, "CUDA_ERROR_INVALID_VALUE") != 0) {
        std::fprintf(stderr, "FAIL: cuGetErrorName/String mismatch for known code\n");
        return 1;
    }

    if (cuGetErrorName(static_cast<CUresult>(12345), &name) != CUDA_SUCCESS ||
        cuGetErrorString(static_cast<CUresult>(12345), &description) != CUDA_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuGetErrorName/String failed for unknown code\n");
        return 1;
    }
    if (name == nullptr || description == nullptr) {
        std::fprintf(stderr, "FAIL: unknown cuGetErrorName/String returned null\n");
        return 1;
    }
    if (std::strcmp(name, "CUDA_ERROR_UNKNOWN") != 0 ||
        std::strcmp(description, "CUDA_ERROR_UNKNOWN") != 0) {
        std::fprintf(stderr, "FAIL: unknown driver error should map to CUDA_ERROR_UNKNOWN\n");
        return 1;
    }

    if (cuGetErrorName(CUDA_ERROR_LAUNCH_TIMEOUT, &name) != CUDA_SUCCESS ||
        std::strcmp(name, "CUDA_ERROR_LAUNCH_TIMEOUT") != 0) {
        std::fprintf(stderr, "FAIL: launch-timeout driver error name mismatch\n");
        return 1;
    }
    if (cuGetErrorName(CUDA_ERROR_ILLEGAL_ADDRESS, &name) != CUDA_SUCCESS ||
        std::strcmp(name, "CUDA_ERROR_ILLEGAL_ADDRESS") != 0) {
        std::fprintf(stderr, "FAIL: illegal-address driver error name mismatch\n");
        return 1;
    }
    if (cuGetErrorName(CUDA_ERROR_DEVICES_UNAVAILABLE, &name) != CUDA_SUCCESS ||
        std::strcmp(name, "CUDA_ERROR_DEVICES_UNAVAILABLE") != 0) {
        std::fprintf(stderr, "FAIL: devices-unavailable driver error name mismatch\n");
        return 1;
    }

    if (cuGetErrorName(CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED, &name) != CUDA_SUCCESS ||
        std::strcmp(name, "CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED") != 0 ||
        cuGetErrorString(CUDA_ERROR_PEER_ACCESS_NOT_ENABLED, &description) != CUDA_SUCCESS ||
        std::strcmp(description, "CUDA_ERROR_PEER_ACCESS_NOT_ENABLED") != 0) {
        std::fprintf(stderr, "FAIL: peer-access driver error name/string mismatch\n");
        return 1;
    }

    std::printf("PASS: driver error APIs behave correctly\n");
    return 0;
}
