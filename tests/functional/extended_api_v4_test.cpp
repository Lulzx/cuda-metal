// extended_api_v4_test.cpp
// Tests: cudaMemset2DAsync, cudaMemset3D, cudaMemset3DAsync,
//        cuMemsetD2D8/16/32 + async, cuMemGetAddressRange, cuPointerGetAttribute,
//        cublasSrotm/Drotm, cublasSrotmg/Drotmg

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"
#include "cublas_v2.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, name)                                            \
    do {                                                             \
        if (cond) { printf("  PASS: %s\n", name); ++g_pass; }       \
        else      { printf("  FAIL: %s\n", name); ++g_fail; }       \
    } while (0)

// ── cudaMemset2DAsync ─────────────────────────────────────────────────────────
static void test_memset2d_async() {
    printf("[cudaMemset2DAsync]\n");
    const size_t width = 16, height = 4, pitch = 32;
    void* ptr = nullptr;
    cudaMalloc(&ptr, pitch * height);
    std::memset(ptr, 0, pitch * height);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaError_t err = cudaMemset2DAsync(ptr, pitch, 0xAB, width, height, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    CHECK(err == cudaSuccess, "cudaMemset2DAsync returns success");

    // Verify each row: first 'width' bytes = 0xAB, remainder = 0x00
    auto* bytes = static_cast<unsigned char*>(ptr);
    bool row_ok = true;
    for (size_t r = 0; r < height && row_ok; ++r) {
        for (size_t c = 0; c < width; ++c)
            if (bytes[r * pitch + c] != 0xAB) { row_ok = false; break; }
        for (size_t c = width; c < pitch && row_ok; ++c)
            if (bytes[r * pitch + c] != 0x00) { row_ok = false; break; }
    }
    CHECK(row_ok, "cudaMemset2DAsync data correct");
    cudaFree(ptr);
}

// ── cudaMemset3D ──────────────────────────────────────────────────────────────
static void test_memset3d() {
    printf("[cudaMemset3D]\n");
    const size_t w = 8, h = 4, d = 2;
    cudaExtent ext = make_cudaExtent(w, h, d);
    cudaPitchedPtr pp;
    cudaMalloc3D(&pp, ext);

    // zero first
    cudaMemset3D(pp, 0, ext);
    cudaError_t err = cudaMemset3D(pp, 0xCC, ext);
    CHECK(err == cudaSuccess, "cudaMemset3D returns success");

    // verify
    const size_t plane_stride = pp.pitch * pp.ysize;
    auto* base = static_cast<unsigned char*>(pp.ptr);
    bool ok = true;
    for (size_t z = 0; z < d && ok; ++z)
        for (size_t y = 0; y < h && ok; ++y)
            for (size_t x = 0; x < w && ok; ++x)
                if (base[z * plane_stride + y * pp.pitch + x] != 0xCC)
                    ok = false;
    CHECK(ok, "cudaMemset3D data correct");
    cudaFree(pp.ptr);
}

// ── cudaMemset3DAsync ─────────────────────────────────────────────────────────
static void test_memset3d_async() {
    printf("[cudaMemset3DAsync]\n");
    const size_t w = 6, h = 3, d = 2;
    cudaExtent ext = make_cudaExtent(w, h, d);
    cudaPitchedPtr pp;
    cudaMalloc3D(&pp, ext);
    cudaMemset3D(pp, 0, ext);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaError_t err = cudaMemset3DAsync(pp, 0x55, ext, stream);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    CHECK(err == cudaSuccess, "cudaMemset3DAsync returns success");

    const size_t plane_stride = pp.pitch * pp.ysize;
    auto* base = static_cast<unsigned char*>(pp.ptr);
    bool ok = true;
    for (size_t z = 0; z < d && ok; ++z)
        for (size_t y = 0; y < h && ok; ++y)
            for (size_t x = 0; x < w && ok; ++x)
                if (base[z * plane_stride + y * pp.pitch + x] != 0x55)
                    ok = false;
    CHECK(ok, "cudaMemset3DAsync data correct");
    cudaFree(pp.ptr);
}

// ── cuMemsetD2D8/16/32 ────────────────────────────────────────────────────────
static void test_cu_memset_d2d() {
    printf("[cuMemsetD2D8/16/32]\n");
    const size_t cols = 4, rows = 3, pitch = 8 * sizeof(unsigned int);
    void* ptr = nullptr;
    cudaMalloc(&ptr, pitch * rows);

    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

    // 8-bit
    std::memset(ptr, 0, pitch * rows);
    CUresult r8 = cuMemsetD2D8(dptr, pitch, 0xAA, cols, rows);
    CHECK(r8 == CUDA_SUCCESS, "cuMemsetD2D8 returns success");
    auto* b = static_cast<unsigned char*>(ptr);
    bool ok8 = true;
    for (size_t r = 0; r < rows && ok8; ++r)
        for (size_t c = 0; c < cols && ok8; ++c)
            if (b[r * pitch + c] != 0xAA) ok8 = false;
    CHECK(ok8, "cuMemsetD2D8 data correct");

    // 16-bit
    std::memset(ptr, 0, pitch * rows);
    CUresult r16 = cuMemsetD2D16(dptr, pitch, 0xBEEF, cols, rows);
    CHECK(r16 == CUDA_SUCCESS, "cuMemsetD2D16 returns success");
    auto* s = static_cast<unsigned short*>(ptr);
    bool ok16 = true;
    for (size_t r = 0; r < rows && ok16; ++r)
        for (size_t c = 0; c < cols && ok16; ++c)
            if (s[r * (pitch / sizeof(unsigned short)) + c] != 0xBEEF) ok16 = false;
    CHECK(ok16, "cuMemsetD2D16 data correct");

    // 32-bit
    std::memset(ptr, 0, pitch * rows);
    CUresult r32 = cuMemsetD2D32(dptr, pitch, 0xDEADBEEFu, cols, rows);
    CHECK(r32 == CUDA_SUCCESS, "cuMemsetD2D32 returns success");
    auto* i = static_cast<unsigned int*>(ptr);
    bool ok32 = true;
    for (size_t r = 0; r < rows && ok32; ++r)
        for (size_t c = 0; c < cols && ok32; ++c)
            if (i[r * (pitch / sizeof(unsigned int)) + c] != 0xDEADBEEFu) ok32 = false;
    CHECK(ok32, "cuMemsetD2D32 data correct");

    cudaFree(ptr);
}

// ── cuMemsetD2D*Async ─────────────────────────────────────────────────────────
static void test_cu_memset_d2d_async() {
    printf("[cuMemsetD2D8/16/32Async]\n");
    const size_t cols = 4, rows = 2, pitch = 8 * sizeof(unsigned int);
    void* ptr = nullptr;
    cudaMalloc(&ptr, pitch * rows);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);
    CUstream stream;
    cuStreamCreate(&stream, 0);

    std::memset(ptr, 0, pitch * rows);
    CUresult r8  = cuMemsetD2D8Async(dptr,  pitch, 0x12,     cols, rows, stream);
    cuStreamSynchronize(stream);
    CHECK(r8 == CUDA_SUCCESS, "cuMemsetD2D8Async returns success");

    std::memset(ptr, 0, pitch * rows);
    CUresult r16 = cuMemsetD2D16Async(dptr, pitch, 0x1234,   cols, rows, stream);
    cuStreamSynchronize(stream);
    CHECK(r16 == CUDA_SUCCESS, "cuMemsetD2D16Async returns success");

    std::memset(ptr, 0, pitch * rows);
    CUresult r32 = cuMemsetD2D32Async(dptr, pitch, 0x12345678u, cols, rows, stream);
    cuStreamSynchronize(stream);
    CHECK(r32 == CUDA_SUCCESS, "cuMemsetD2D32Async returns success");

    cuStreamDestroy(stream);
    cudaFree(ptr);
}

// ── cuMemGetAddressRange ──────────────────────────────────────────────────────
static void test_cu_mem_get_address_range() {
    printf("[cuMemGetAddressRange]\n");
    const size_t alloc_size = 256;
    void* ptr = nullptr;
    cudaMalloc(&ptr, alloc_size);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

    // Query middle of allocation
    CUdeviceptr mid = dptr + 64;
    CUdeviceptr base = 0;
    size_t size = 0;
    CUresult res = cuMemGetAddressRange(&base, &size, mid);
    CHECK(res == CUDA_SUCCESS, "cuMemGetAddressRange returns success");
    CHECK(base == dptr, "cuMemGetAddressRange base correct");
    CHECK(size == alloc_size, "cuMemGetAddressRange size correct");

    cudaFree(ptr);
}

// ── cuPointerGetAttribute ─────────────────────────────────────────────────────
static void test_cu_pointer_get_attribute() {
    printf("[cuPointerGetAttribute]\n");
    void* ptr = nullptr;
    cudaMalloc(&ptr, 128);
    CUdeviceptr dptr = reinterpret_cast<CUdeviceptr>(ptr);

    // MEMORY_TYPE
    unsigned int mem_type = 99;
    CUresult r1 = cuPointerGetAttribute(&mem_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, dptr);
    CHECK(r1 == CUDA_SUCCESS, "cuPointerGetAttribute MEMORY_TYPE success");
    CHECK(mem_type == CU_MEMORYTYPE_UNIFIED, "cuPointerGetAttribute MEMORY_TYPE = UNIFIED");

    // DEVICE_POINTER
    CUdeviceptr dev_ptr = 0;
    CUresult r2 = cuPointerGetAttribute(&dev_ptr, CU_POINTER_ATTRIBUTE_DEVICE_POINTER, dptr);
    CHECK(r2 == CUDA_SUCCESS, "cuPointerGetAttribute DEVICE_POINTER success");
    CHECK(dev_ptr == dptr, "cuPointerGetAttribute DEVICE_POINTER == input");

    // HOST_POINTER
    void* host_ptr = nullptr;
    CUresult r3 = cuPointerGetAttribute(&host_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, dptr);
    CHECK(r3 == CUDA_SUCCESS, "cuPointerGetAttribute HOST_POINTER success");
    CHECK(host_ptr == ptr, "cuPointerGetAttribute HOST_POINTER == malloc ptr");

    // IS_MANAGED
    unsigned int is_managed = 0;
    CUresult r4 = cuPointerGetAttribute(&is_managed, CU_POINTER_ATTRIBUTE_IS_MANAGED, dptr);
    CHECK(r4 == CUDA_SUCCESS, "cuPointerGetAttribute IS_MANAGED success");
    // On UMA: cudaMalloc memory may or may not be managed; just verify the call succeeds

    cudaFree(ptr);
}

// ── cublasSrotm / cublasDrotm ─────────────────────────────────────────────────
static void test_cublas_rotm() {
    printf("[cublasSrotm / cublasDrotm]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Test with flag=-1 (general H): H = [2 1; 3 4]
    // x=[1,2,3], y=[4,5,6]
    // xi' = 2*xi + 1*yi, yi' = 3*xi + 4*yi
    {
        std::vector<float> x = {1.0f, 2.0f, 3.0f};
        std::vector<float> y = {4.0f, 5.0f, 6.0f};
        float param[5] = {-1.0f, 2.0f, 3.0f, 1.0f, 4.0f}; // flag, h11, h21, h12, h22
        cublasStatus_t st = cublasSrotm(handle, 3, x.data(), 1, y.data(), 1, param);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSrotm returns success");
        // x[0] = 2*1 + 1*4 = 6, y[0] = 3*1 + 4*4 = 19
        bool ok = (std::fabsf(x[0] - 6.0f) < 1e-5f &&
                   std::fabsf(x[1] - 9.0f) < 1e-5f &&  // 2*2+1*5=9
                   std::fabsf(x[2] - 12.0f) < 1e-5f && // 2*3+1*6=12
                   std::fabsf(y[0] - 19.0f) < 1e-5f && // 3*1+4*4=19
                   std::fabsf(y[1] - 26.0f) < 1e-5f && // 3*2+4*5=26
                   std::fabsf(y[2] - 33.0f) < 1e-5f);  // 3*3+4*6=33
        CHECK(ok, "cublasSrotm general H correct");
    }

    // Test with flag=-2 (identity): vectors unchanged
    {
        std::vector<float> x = {1.0f, 2.0f};
        std::vector<float> y = {3.0f, 4.0f};
        float param[5] = {-2.0f, 0, 0, 0, 0};
        cublasSrotm(handle, 2, x.data(), 1, y.data(), 1, param);
        CHECK(x[0] == 1.0f && y[0] == 3.0f, "cublasSrotm identity flag=-2 no-op");
    }

    // Double variant with flag=0: H=[1 p12; p21 1]
    {
        std::vector<double> x = {1.0, 2.0};
        std::vector<double> y = {3.0, 4.0};
        // flag=0, h21=2, h12=3  => x'=x+3y, y'=2x+y
        double param[5] = {0.0, 1.0, 2.0, 3.0, 1.0};
        cublasStatus_t st = cublasDrotm(handle, 2, x.data(), 1, y.data(), 1, param);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasDrotm returns success");
        bool ok = (std::fabs(x[0] - (1.0 + 3.0*3.0)) < 1e-9 &&
                   std::fabs(y[0] - (2.0*1.0 + 3.0)) < 1e-9);
        CHECK(ok, "cublasDrotm flag=0 correct");
    }

    cublasDestroy(handle);
}

// ── cublasSrotmg / cublasDrotmg ───────────────────────────────────────────────
static void test_cublas_rotmg() {
    printf("[cublasSrotmg / cublasDrotmg]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // After rotmg, applying the resulting H to [sqrt(d1)*x1; sqrt(d2)*y1]
    // should zero the second component.
    {
        float d1 = 1.0f, d2 = 1.0f, x1 = 3.0f, y1 = 4.0f;
        float param[5];
        cublasStatus_t st = cublasSrotmg(handle, &d1, &d2, &x1, &y1, param);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasSrotmg returns success");
        // param[0] must be a valid flag value
        bool valid_flag = (param[0] == -2.0f || param[0] == -1.0f ||
                           param[0] == 0.0f  || param[0] == 1.0f);
        CHECK(valid_flag, "cublasSrotmg param[0] is valid flag");
    }

    {
        double d1 = 2.0, d2 = 3.0, x1 = 1.0, y1 = 2.0;
        double param[5];
        cublasStatus_t st = cublasDrotmg(handle, &d1, &d2, &x1, &y1, param);
        CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasDrotmg returns success");
        bool valid_flag = (param[0] == -2.0 || param[0] == -1.0 ||
                           param[0] == 0.0  || param[0] == 1.0);
        CHECK(valid_flag, "cublasDrotmg param[0] is valid flag");
    }

    cublasDestroy(handle);
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== extended_api_v4 tests ===\n");

    // Initialize the driver API context (required for cuMemsetD2D*, cuMemGetAddressRange, etc.)
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);

    test_memset2d_async();
    test_memset3d();
    test_memset3d_async();
    test_cu_memset_d2d();
    test_cu_memset_d2d_async();
    test_cu_mem_get_address_range();
    test_cu_pointer_get_attribute();
    test_cublas_rotm();
    test_cublas_rotmg();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
