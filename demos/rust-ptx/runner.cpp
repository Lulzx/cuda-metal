#include "cuda.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void check(CUresult result, const char* operation) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(std::string(operation) + ": " +
                             (message ? message : "unknown CUDA error"));
}

struct Buffer {
    CUdeviceptr pointer = 0;
    explicit Buffer(std::size_t size) { check(cuMemAlloc(&pointer, size), "allocate"); }
    ~Buffer() { if (pointer) cuMemFree(pointer); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static void run(CUfunction function, bool sha256, std::uint32_t count) {
    // Allocate tail guards and deliberately dispatch more threads than messages.
    const std::size_t stride = sha256 ? 32 : sizeof(float);
    const std::size_t size = count * stride;
    constexpr std::size_t guard = 64;
    std::vector<unsigned char> a(size), b(size), output(size + guard, 0xa5);
    if (sha256) {
        for (std::size_t i = 0; i < size; ++i)
            a[i] = static_cast<unsigned char>((i * 17 + (i / 32) * 13) & 255);
        std::fill_n(a.begin(), 32, 0); // First message has an independent fixed KAT.
    } else {
        for (std::uint32_t i = 0; i < count; ++i) {
            const float x = static_cast<float>(i % 37) * 0.5f;
            const float y = -static_cast<float>(i % 19) * 0.25f;
            std::memcpy(a.data() + i * stride, &x, sizeof(x));
            std::memcpy(b.data() + i * stride, &y, sizeof(y));
        }
    }
    Buffer da(size), db(size), dout(size + guard);
    check(cuMemcpyHtoD(da.pointer, a.data(), size), "upload a");
    check(cuMemcpyHtoD(db.pointer, b.data(), size), "upload b");
    check(cuMemcpyHtoD(dout.pointer, output.data(), output.size()), "poison output");
    // The ABI describes a u32. CuMetal's current argument classifier reads an
    // eight-byte word, so provide eight bytes of backing storage on this host.
    std::uint64_t count_storage = count;
    void* vector_args[] = {&da.pointer, &db.pointer, &dout.pointer, &count_storage, nullptr};
    void* hash_args[] = {&da.pointer, &dout.pointer, &count_storage, nullptr};
    check(cuLaunchKernel(function, (count + 63) / 64, 1, 1, 64, 1, 1, 0,
                         nullptr, sha256 ? hash_args : vector_args, nullptr), "launch");
    check(cuCtxSynchronize(), "synchronize");
    check(cuMemcpyDtoH(output.data(), dout.pointer, output.size()), "download");
    for (std::uint32_t i = 0; i < count; ++i) {
        if (sha256) {
            std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> expected{};
            CC_SHA256(a.data() + i * stride, 32, expected.data());
            if (std::memcmp(output.data() + i * stride, expected.data(), 32) != 0)
                throw std::runtime_error("SHA-256 mismatch at message " + std::to_string(i));
        } else {
            float x, y, actual;
            std::memcpy(&x, a.data() + i * stride, sizeof(x));
            std::memcpy(&y, b.data() + i * stride, sizeof(y));
            std::memcpy(&actual, output.data() + i * stride, sizeof(actual));
            if (actual != x + y)
                throw std::runtime_error("vector mismatch at element " + std::to_string(i));
        }
    }
    if (sha256) {
        constexpr unsigned char zero_hash[] = {
            0x66,0x68,0x7a,0xad,0xf8,0x62,0xbd,0x77,0x6c,0x8f,0xc1,0x8b,0x8e,0x9f,0x8e,0x20,
            0x08,0x97,0x14,0x85,0x6e,0xe2,0x33,0xb3,0x90,0x2a,0x59,0x1d,0x0d,0x5f,0x29,0x25};
        if (std::memcmp(output.data(), zero_hash, 32) != 0)
            throw std::runtime_error("zero-message known-answer mismatch");
    }
    if (!std::all_of(output.begin() + size, output.end(), [](auto x) { return x == 0xa5; }))
        throw std::runtime_error("output guard overwritten");
    std::printf("NUMERICAL_PASS kernel=%s count=%u\n",
                sha256 ? "rust_sha256_32" : "rust_vecadd", count);
}

int main(int argc, char** argv) {
    if (argc != 3 || (std::string(argv[2]) != "vecadd" && std::string(argv[2]) != "sha256")) {
        std::fprintf(stderr, "usage: rust-ptx-runner kernel.metal vecadd|sha256\n");
        return 2;
    }
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    try {
        const bool sha256 = std::string(argv[2]) == "sha256";
        check(cuInit(0), "initialize");
        CUdevice device;
        check(cuDeviceGet(&device, 0), "device");
        char name[256] = {};
        check(cuDeviceGetName(name, sizeof(name), device), "device name");
        std::printf("DEVICE %s\n", name);
        check(cuCtxCreate(&context, 0, device), "context");
        check(cuModuleLoad(&module, argv[1]), "load generated MSL");
        CUfunction function = nullptr;
        check(cuModuleGetFunction(&function, module, sha256 ? "rust_sha256_32" : "rust_vecadd"), "function");
        for (auto count : {1u, 31u, 32u, 33u, 257u}) run(function, sha256, count);
        check(cuModuleUnload(module), "unload");
        module = nullptr;
        check(cuCtxDestroy(context), "destroy context");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        if (module) cuModuleUnload(module);
        if (context) cuCtxDestroy(context);
        return 1;
    }
}
