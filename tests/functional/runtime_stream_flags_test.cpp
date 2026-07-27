#include "cuda_runtime.h"

#include <cstdio>

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;

    if (cudaStreamCreateWithFlags(&stream, cudaStreamDefault) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithFlags(default) failed\n");
        return 1;
    }
    unsigned int observed_flags = ~0u;
    if (cudaStreamGetFlags(stream, &observed_flags) != cudaSuccess ||
        observed_flags != cudaStreamDefault) {
        std::fprintf(stderr, "FAIL: default stream flags were not preserved\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy(default stream) failed\n");
        return 1;
    }

    stream = nullptr;
    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithFlags(nonblocking) failed\n");
        return 1;
    }
    observed_flags = ~0u;
    if (cudaStreamGetFlags(stream, &observed_flags) != cudaSuccess ||
        observed_flags != cudaStreamNonBlocking) {
        std::fprintf(stderr, "FAIL: nonblocking stream flags were not preserved\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy(nonblocking stream) failed\n");
        return 1;
    }

    if (cudaStreamCreateWithFlags(&stream, 0x100u) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: expected cudaErrorInvalidValue for unsupported stream flag\n");
        return 1;
    }

    if (cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamSynchronize(cudaStreamPerThread) failed\n");
        return 1;
    }
    if (cudaStreamQuery(cudaStreamLegacy) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamQuery(cudaStreamLegacy) failed\n");
        return 1;
    }
    if (cudaStreamDestroy(cudaStreamLegacy) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: destroying cudaStreamLegacy should be invalid\n");
        return 1;
    }
    if (cudaStreamDestroy(cudaStreamPerThread) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: destroying cudaStreamPerThread should be invalid\n");
        return 1;
    }

    std::printf("PASS: stream flag validation and special stream handles\n");
    return 0;
}
