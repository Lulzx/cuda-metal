#include "cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Tests the driver entry points NVIDIA Warp resolves through cuGetProcAddress:
// the lookup itself, cuStreamGetCtx, cuEventRecordWithFlags, pitched 2D copies
// and batched async copies. None of these launch a kernel.

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main() {
    if (cuInit(0) != CUDA_SUCCESS) return fail("cuInit");
    CUdevice dev = 0;
    if (cuDeviceGet(&dev, 0) != CUDA_SUCCESS) return fail("cuDeviceGet");
    CUcontext ctx = nullptr;
    if (cuCtxCreate(&ctx, 0, dev) != CUDA_SUCCESS) return fail("cuCtxCreate");

    // --- cuGetProcAddress resolves exported entry points to their real addresses ---
    struct Probe {
        const char* name;
        const void* expected;
        int version;
    };
    const Probe probes[] = {
        {"cuInit", reinterpret_cast<const void*>(&cuInit), 2000},
        {"cuDriverGetVersion", reinterpret_cast<const void*>(&cuDriverGetVersion), 2020},
        {"cuMemcpy2D", reinterpret_cast<const void*>(&cuMemcpy2D), 3020},
        {"cuMemcpy2DAsync", reinterpret_cast<const void*>(&cuMemcpy2DAsync), 3020},
        {"cuMemcpyBatchAsync", reinterpret_cast<const void*>(&cuMemcpyBatchAsync), 12080},
        {"cuEventRecordWithFlags", reinterpret_cast<const void*>(&cuEventRecordWithFlags), 11010},
        {"cuStreamGetCtx", reinterpret_cast<const void*>(&cuStreamGetCtx), 9020},
        {"cuGetProcAddress", reinterpret_cast<const void*>(&cuGetProcAddress), 12000},
    };
    for (const Probe& probe : probes) {
        void* pfn = nullptr;
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
        const CUresult result =
            cuGetProcAddress(probe.name, &pfn, probe.version, CU_GET_PROC_ADDRESS_DEFAULT, &status);
        if (result != CUDA_SUCCESS || status != CU_GET_PROC_ADDRESS_SUCCESS || pfn != probe.expected) {
            std::fprintf(stderr, "FAIL: cuGetProcAddress(%s) result=%d status=%d pfn=%p expected=%p\n",
                         probe.name, static_cast<int>(result), static_cast<int>(status), pfn,
                         probe.expected);
            return 1;
        }
    }
    {
        void* pfn = reinterpret_cast<void*>(0x1);
        CUdriverProcAddressQueryResult status = CU_GET_PROC_ADDRESS_SUCCESS;
        if (cuGetProcAddress("cuGraphicsGLRegisterBuffer", &pfn, 12000, CU_GET_PROC_ADDRESS_DEFAULT,
                             &status) != CUDA_ERROR_NOT_FOUND ||
            status != CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND || pfn != nullptr) {
            return fail("cuGetProcAddress must report an unexported entry point as not found");
        }
        if (cuGetProcAddress("cudaMalloc", &pfn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr) !=
            CUDA_ERROR_NOT_FOUND) {
            return fail("cuGetProcAddress must not hand out runtime-API symbols");
        }
        if (cuGetProcAddress(nullptr, &pfn, 12000, CU_GET_PROC_ADDRESS_DEFAULT, nullptr) !=
            CUDA_ERROR_INVALID_VALUE) {
            return fail("cuGetProcAddress(nullptr) must be an invalid value");
        }
    }

    // --- cuStreamGetCtx ---
    CUstream stream = nullptr;
    if (cuStreamCreate(&stream, 0) != CUDA_SUCCESS) return fail("cuStreamCreate");
    {
        CUcontext stream_ctx = nullptr;
        if (cuStreamGetCtx(stream, &stream_ctx) != CUDA_SUCCESS || stream_ctx != ctx) {
            return fail("cuStreamGetCtx must return the stream's current context");
        }
        if (cuStreamGetCtx(nullptr, &stream_ctx) != CUDA_SUCCESS || stream_ctx != ctx) {
            return fail("cuStreamGetCtx(null stream) must return the current context");
        }
        if (cuStreamGetCtx(stream, nullptr) != CUDA_ERROR_INVALID_VALUE) {
            return fail("cuStreamGetCtx(nullptr out) must be an invalid value");
        }
    }

    // --- cuEventRecordWithFlags ---
    {
        CUevent event = nullptr;
        if (cuEventCreate(&event, 0) != CUDA_SUCCESS) return fail("cuEventCreate");
        if (cuEventRecordWithFlags(event, stream, 0) != CUDA_SUCCESS) {
            return fail("cuEventRecordWithFlags(default)");
        }
        if (cuEventSynchronize(event) != CUDA_SUCCESS) return fail("cuEventSynchronize");
        if (cuEventQuery(event) != CUDA_SUCCESS) return fail("recorded event must be complete");
        cuEventDestroy(event);
    }

    // --- cuMemcpy2D: pitched host -> device -> host round trip with offsets ---
    constexpr size_t kWidth = 7 * sizeof(std::uint32_t);
    constexpr size_t kHeight = 5;
    constexpr size_t kHostPitch = 11 * sizeof(std::uint32_t);
    constexpr size_t kDevicePitch = 16 * sizeof(std::uint32_t);
    std::vector<std::uint32_t> host_src(kHostPitch / 4 * kHeight);
    for (size_t i = 0; i < host_src.size(); ++i) host_src[i] = static_cast<std::uint32_t>(1000 + i);
    CUdeviceptr device_buffer = 0;
    if (cuMemAlloc(&device_buffer, kDevicePitch * (kHeight + 2)) != CUDA_SUCCESS) {
        return fail("cuMemAlloc");
    }
    if (cuMemsetD8(device_buffer, 0xEE, kDevicePitch * (kHeight + 2)) != CUDA_SUCCESS) {
        return fail("cuMemsetD8");
    }
    {
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = host_src.data();
        copy.srcPitch = kHostPitch;
        copy.srcXInBytes = 2 * sizeof(std::uint32_t);
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = device_buffer;
        copy.dstPitch = kDevicePitch;
        copy.dstY = 1;
        copy.dstXInBytes = 3 * sizeof(std::uint32_t);
        copy.WidthInBytes = kWidth;
        copy.Height = kHeight;
        if (cuMemcpy2D(&copy) != CUDA_SUCCESS) return fail("cuMemcpy2D host->device");
    }
    std::vector<std::uint32_t> host_dst(kHostPitch / 4 * kHeight, 0);
    {
        CUDA_MEMCPY2D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = device_buffer;
        copy.srcPitch = kDevicePitch;
        copy.srcY = 1;
        copy.srcXInBytes = 3 * sizeof(std::uint32_t);
        copy.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy.dstHost = host_dst.data();
        copy.dstPitch = kHostPitch;
        copy.dstXInBytes = 2 * sizeof(std::uint32_t);
        copy.WidthInBytes = kWidth;
        copy.Height = kHeight;
        if (cuMemcpy2DAsync(&copy, stream) != CUDA_SUCCESS) return fail("cuMemcpy2DAsync device->host");
        if (cuStreamSynchronize(stream) != CUDA_SUCCESS) return fail("cuStreamSynchronize");
    }
    for (size_t y = 0; y < kHeight; ++y) {
        for (size_t x = 0; x < kHostPitch / 4; ++x) {
            const size_t index = y * (kHostPitch / 4) + x;
            const bool inside = x >= 2 && x < 2 + kWidth / 4;
            const std::uint32_t expected = inside ? host_src[index] : 0u;
            if (host_dst[index] != expected) {
                std::fprintf(stderr, "FAIL: 2D round trip mismatch at (%zu,%zu): %u expected %u\n",
                             x, y, host_dst[index], expected);
                return 1;
            }
        }
    }
    {
        // Rows outside the destination rectangle keep the memset pattern.
        std::vector<std::uint8_t> guard(kDevicePitch);
        if (cuMemcpyDtoH(guard.data(), device_buffer, kDevicePitch) != CUDA_SUCCESS) {
            return fail("cuMemcpyDtoH guard row");
        }
        for (std::uint8_t byte : guard) {
            if (byte != 0xEE) return fail("cuMemcpy2D wrote outside its destination rectangle");
        }
        CUDA_MEMCPY2D bad{};
        bad.srcMemoryType = CU_MEMORYTYPE_HOST;
        bad.srcHost = host_src.data();
        bad.srcPitch = 4;
        bad.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        bad.dstDevice = device_buffer;
        bad.dstPitch = kDevicePitch;
        bad.WidthInBytes = 8;
        bad.Height = 1;
        if (cuMemcpy2D(&bad) != CUDA_ERROR_INVALID_VALUE) {
            return fail("cuMemcpy2D must reject a width wider than its pitch");
        }
        if (cuMemcpy2D(nullptr) != CUDA_ERROR_INVALID_VALUE) {
            return fail("cuMemcpy2D(nullptr) must be an invalid value");
        }
    }

    // --- cuMemcpyBatchAsync ---
    {
        constexpr size_t kCount = 3;
        constexpr size_t kBytes = 64;
        std::vector<std::uint8_t> sources(kCount * kBytes);
        for (size_t i = 0; i < sources.size(); ++i) sources[i] = static_cast<std::uint8_t>(i * 7 + 1);
        CUdeviceptr device_dsts[kCount] = {};
        for (size_t i = 0; i < kCount; ++i) {
            if (cuMemAlloc(&device_dsts[i], kBytes) != CUDA_SUCCESS) return fail("cuMemAlloc batch");
        }
        CUdeviceptr srcs[kCount];
        size_t sizes[kCount];
        for (size_t i = 0; i < kCount; ++i) {
            srcs[i] = reinterpret_cast<CUdeviceptr>(sources.data() + i * kBytes);
            sizes[i] = kBytes;
        }
        CUmemcpyAttributes attrs{};
        attrs.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
        size_t attrs_idxs[1] = {0};
        size_t fail_idx = 0;
        if (cuMemcpyBatchAsync(device_dsts, srcs, sizes, kCount, &attrs, attrs_idxs, 1, &fail_idx,
                               stream) != CUDA_SUCCESS) {
            return fail("cuMemcpyBatchAsync");
        }
        if (fail_idx != SIZE_MAX) return fail("cuMemcpyBatchAsync must leave failIdx untouched on success");
        if (cuStreamSynchronize(stream) != CUDA_SUCCESS) return fail("cuStreamSynchronize batch");
        for (size_t i = 0; i < kCount; ++i) {
            std::vector<std::uint8_t> back(kBytes);
            if (cuMemcpyDtoH(back.data(), device_dsts[i], kBytes) != CUDA_SUCCESS) {
                return fail("cuMemcpyDtoH batch");
            }
            if (std::memcmp(back.data(), sources.data() + i * kBytes, kBytes) != 0) {
                std::fprintf(stderr, "FAIL: batch copy %zu mismatch\n", i);
                return 1;
            }
            cuMemFree(device_dsts[i]);
        }
        size_t bad_idxs[1] = {1};
        if (cuMemcpyBatchAsync(device_dsts, srcs, sizes, kCount, &attrs, bad_idxs, 1, &fail_idx,
                               stream) != CUDA_ERROR_INVALID_VALUE) {
            return fail("cuMemcpyBatchAsync must reject an attribute list that does not start at 0");
        }
        if (cuMemcpyBatchAsync(nullptr, srcs, sizes, kCount, nullptr, nullptr, 0, &fail_idx, stream) !=
            CUDA_ERROR_INVALID_VALUE) {
            return fail("cuMemcpyBatchAsync(null dsts) must be an invalid value");
        }
    }

    // --- cuMemcpy3DAsync used to re-enter cuMemcpy3D from the host-func worker
    // thread, which holds no current context, so the copy silently did nothing.
    {
        constexpr size_t kBytes3D = 32;
        std::vector<std::uint8_t> src3(kBytes3D * 2 * 2), dst3(kBytes3D * 2 * 2, 0);
        for (size_t i = 0; i < src3.size(); ++i) src3[i] = static_cast<std::uint8_t>(i + 3);
        CUDA_MEMCPY3D copy{};
        copy.srcMemoryType = CU_MEMORYTYPE_HOST;
        copy.srcHost = src3.data();
        copy.srcPitch = kBytes3D;
        copy.srcHeight = 2;
        copy.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy.dstHost = dst3.data();
        copy.dstPitch = kBytes3D;
        copy.dstHeight = 2;
        copy.WidthInBytes = kBytes3D;
        copy.Height = 2;
        copy.Depth = 2;
        if (cuMemcpy3DAsync(&copy, stream) != CUDA_SUCCESS) return fail("cuMemcpy3DAsync");
        if (cuStreamSynchronize(stream) != CUDA_SUCCESS) return fail("cuStreamSynchronize 3D");
        if (dst3 != src3) return fail("cuMemcpy3DAsync must perform the copy from the stream worker");
    }

    cuMemFree(device_buffer);
    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
    std::printf("PASS: driver proc-address, stream context, flagged events, 2D and batch copies\n");
    return 0;
}
