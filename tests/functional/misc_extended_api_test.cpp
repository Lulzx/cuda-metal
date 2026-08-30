// Functional tests for miscellaneous extended APIs added post-Phase-5:
//   - curandGeneratePoisson, curandGetProperty
//   - cublasGetStatusName, cublasGetStatusString
//   - cufftSetWorkArea, cufftEstimate1d/2d/3d/Many
//   - cudaMalloc3D, cudaMemcpy3D / cudaMemcpy3DAsync
//   - cuMemcpy3D / cuMemcpy3DAsync (driver API)

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cufft.h"
#include "curand.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

static int g_failures = 0;

#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL: %s\n", msg);                \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// ── curand ───────────────────────────────────────────────────────────────────

static void test_curand_get_property() {
    int major = -1, minor = -1, patch = -1;
    CHECK(curandGetProperty(MAJOR_VERSION, &major) == CURAND_STATUS_SUCCESS,
          "curandGetProperty MAJOR_VERSION");
    CHECK(curandGetProperty(MINOR_VERSION, &minor) == CURAND_STATUS_SUCCESS,
          "curandGetProperty MINOR_VERSION");
    CHECK(curandGetProperty(PATCH_LEVEL, &patch) == CURAND_STATUS_SUCCESS,
          "curandGetProperty PATCH_LEVEL");
    CHECK(major >= 0 && minor >= 0 && patch >= 0, "curandGetProperty values >= 0");
}

static void test_curand_generate_poisson() {
    curandGenerator_t gen = nullptr;
    CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
          "curandCreateGenerator for Poisson");
    CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL) == CURAND_STATUS_SUCCESS,
          "seed for Poisson");

    constexpr int kN = 256;
    unsigned int* d_out = nullptr;
    CHECK(cudaMalloc((void**)&d_out, kN * sizeof(unsigned int)) == cudaSuccess,
          "cudaMalloc for Poisson output");

    CHECK(curandGeneratePoisson(gen, d_out, kN, 4.0) == CURAND_STATUS_SUCCESS,
          "curandGeneratePoisson lambda=4");
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "curandGeneratePoisson completes before host inspection");

    // Verify output looks like Poisson (mean ~4, all non-negative)
    double sum = 0.0;
    for (int i = 0; i < kN; ++i) sum += static_cast<double>(d_out[i]);
    double mean = sum / kN;
    CHECK(mean > 0.5 && mean < 20.0, "curandGeneratePoisson mean in [0.5, 20]");

    // Invalid lambda
    CHECK(curandGeneratePoisson(gen, d_out, kN, 0.0) == CURAND_STATUS_OUT_OF_RANGE,
          "curandGeneratePoisson lambda=0 -> OUT_OF_RANGE");

    cudaFree(d_out);
    curandDestroyGenerator(gen);
}

// ── cublas status strings ─────────────────────────────────────────────────────

static void test_cublas_status_strings() {
    const char* name = cublasGetStatusName(CUBLAS_STATUS_SUCCESS);
    CHECK(name != nullptr, "cublasGetStatusName non-null");
    CHECK(std::strstr(name, "SUCCESS") != nullptr, "cublasGetStatusName contains SUCCESS");

    const char* str = cublasGetStatusString(CUBLAS_STATUS_SUCCESS);
    CHECK(str != nullptr, "cublasGetStatusString non-null");
    CHECK(std::strlen(str) > 0, "cublasGetStatusString non-empty");

    // Error status
    const char* err_name = cublasGetStatusName(CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(err_name != nullptr, "cublasGetStatusName NOT_INITIALIZED non-null");
    CHECK(std::strstr(err_name, "NOT_INITIALIZED") != nullptr,
          "cublasGetStatusName NOT_INITIALIZED contains keyword");
}

// ── cufft estimate & set_work_area ───────────────────────────────────────────

static void test_cufft_estimate_and_set_work_area() {
    size_t ws = 0;

    CHECK(cufftEstimate1d(1024, CUFFT_C2C, 1, &ws) == CUFFT_SUCCESS, "cufftEstimate1d");
    CHECK(ws > 0, "cufftEstimate1d workSize > 0");

    ws = 0;
    CHECK(cufftEstimate2d(64, 64, CUFFT_C2C, &ws) == CUFFT_SUCCESS, "cufftEstimate2d");
    CHECK(ws > 0, "cufftEstimate2d workSize > 0");

    ws = 0;
    CHECK(cufftEstimate3d(16, 16, 16, CUFFT_C2C, &ws) == CUFFT_SUCCESS, "cufftEstimate3d");
    CHECK(ws > 0, "cufftEstimate3d workSize > 0");

    ws = 0;
    int n[1] = {512};
    CHECK(cufftEstimateMany(1, n, nullptr, 1, 512, nullptr, 1, 512, CUFFT_C2C, 2, &ws)
          == CUFFT_SUCCESS, "cufftEstimateMany");
    CHECK(ws > 0, "cufftEstimateMany workSize > 0");

    // Bad args
    CHECK(cufftEstimate1d(-1, CUFFT_C2C, 1, &ws) == CUFFT_INVALID_VALUE,
          "cufftEstimate1d invalid nx");
    CHECK(cufftEstimate1d(1024, CUFFT_C2C, 1, nullptr) == CUFFT_INVALID_VALUE,
          "cufftEstimate1d null workSize");
    CHECK(cufftEstimate1d(1024, static_cast<cufftType>(-1), 1, &ws) == CUFFT_INVALID_TYPE,
          "cufftEstimate1d invalid type");
    const int max_int = std::numeric_limits<int>::max();
    CHECK(cufftEstimate3d(max_int, max_int, max_int, CUFFT_Z2Z, &ws) ==
              CUFFT_INVALID_SIZE,
          "cufftEstimate3d overflow");

    // SetWorkArea: create a real plan and call it.
    cufftHandle plan = -1;
    CHECK(cufftPlan1d(&plan, 64, CUFFT_C2C, 1) == CUFFT_SUCCESS,
          "cufftPlan1d for SetWorkArea");
    // Pass any non-null pointer as work area (no-op on this implementation).
    static float dummy_buf[1];
    CHECK(cufftSetWorkArea(plan, static_cast<void*>(dummy_buf)) == CUFFT_SUCCESS,
          "cufftSetWorkArea");
    CHECK(cufftSetWorkArea(-1, static_cast<void*>(dummy_buf)) == CUFFT_INVALID_PLAN,
          "cufftSetWorkArea invalid plan");
    CHECK(cufftGetSize(-1, &ws) == CUFFT_INVALID_PLAN, "cufftGetSize invalid plan");
    cufftDestroy(plan);
}

// ── cudaMalloc3D / cudaMemcpy3D ───────────────────────────────────────────────

static void test_cuda_malloc3d_memcpy3d() {
    // Allocate a 4×4×4 3D region (4 bytes per element → width=16 bytes).
    cudaExtent ext = make_cudaExtent(16, 4, 4);  // 16 bytes wide, 4 rows, 4 slices
    cudaPitchedPtr dp{};
    CHECK(cudaMalloc3D(&dp, ext) == cudaSuccess, "cudaMalloc3D");
    CHECK(dp.ptr != nullptr, "cudaMalloc3D ptr non-null");
    CHECK(dp.pitch >= ext.width, "cudaMalloc3D pitch >= width");

    // Fill with a known pattern.
    unsigned char* base = static_cast<unsigned char*>(dp.ptr);
    size_t total_bytes = dp.pitch * ext.height * ext.depth;
    std::memset(base, 0xAB, total_bytes);

    // Allocate a second region and copy.
    cudaPitchedPtr dp2{};
    CHECK(cudaMalloc3D(&dp2, ext) == cudaSuccess, "cudaMalloc3D dst");
    std::memset(dp2.ptr, 0x00, dp2.pitch * ext.height * ext.depth);

    cudaMemcpy3DParms p{};
    p.srcPtr = dp;
    p.dstPtr = dp2;
    p.extent = ext;
    p.kind   = cudaMemcpyDeviceToDevice;

    CHECK(cudaMemcpy3D(&p) == cudaSuccess, "cudaMemcpy3D");

    // Verify first row of first plane of destination.
    const unsigned char* dst = static_cast<const unsigned char*>(dp2.ptr);
    bool ok = true;
    for (size_t b = 0; b < ext.width; ++b) {
        if (dst[b] != 0xAB) { ok = false; break; }
    }
    CHECK(ok, "cudaMemcpy3D data correct");

    // Async variant (should behave identically on UMA).
    std::memset(dp2.ptr, 0x00, dp2.pitch * ext.height * ext.depth);
    CHECK(cudaMemcpy3DAsync(&p, nullptr) == cudaSuccess, "cudaMemcpy3DAsync");
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "cudaMemcpy3DAsync completes before host inspection");
    ok = true;
    for (size_t b = 0; b < ext.width; ++b) {
        if (dst[b] != 0xAB) { ok = false; break; }
    }
    CHECK(ok, "cudaMemcpy3DAsync data correct");

    cudaFree(dp.ptr);
    cudaFree(dp2.ptr);
}

// ── cuMemcpy3D (driver API) ───────────────────────────────────────────────────

static void test_cu_memcpy3d() {
    CUresult cu_err = cuInit(0);
    CHECK(cu_err == CUDA_SUCCESS, "cuInit for cuMemcpy3D test");
    if (cu_err != CUDA_SUCCESS) return;

    CUcontext ctx = nullptr;
    CHECK(cuCtxCreate(&ctx, 0, 0) == CUDA_SUCCESS, "cuCtxCreate for cuMemcpy3D");

    constexpr size_t kWidth  = 16;  // bytes
    constexpr size_t kHeight = 4;
    constexpr size_t kDepth  = 4;

    // Allocate src/dst as device memory via cuMemAlloc.
    CUdeviceptr d_src = 0, d_dst = 0;
    CHECK(cuMemAlloc(&d_src, kWidth * kHeight * kDepth) == CUDA_SUCCESS, "cuMemAlloc src");
    CHECK(cuMemAlloc(&d_dst, kWidth * kHeight * kDepth) == CUDA_SUCCESS, "cuMemAlloc dst");

    // Fill src via host access (UMA).
    std::memset(reinterpret_cast<void*>(d_src), 0xCD, kWidth * kHeight * kDepth);
    std::memset(reinterpret_cast<void*>(d_dst), 0x00, kWidth * kHeight * kDepth);

    CUDA_MEMCPY3D cp{};
    cp.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.srcDevice     = d_src;
    cp.srcPitch      = kWidth;
    cp.srcHeight     = kHeight;
    cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    cp.dstDevice     = d_dst;
    cp.dstPitch      = kWidth;
    cp.dstHeight     = kHeight;
    cp.WidthInBytes  = kWidth;
    cp.Height        = kHeight;
    cp.Depth         = kDepth;

    CHECK(cuMemcpy3D(&cp) == CUDA_SUCCESS, "cuMemcpy3D");

    const unsigned char* dst = reinterpret_cast<const unsigned char*>(d_dst);
    bool ok = true;
    for (size_t b = 0; b < kWidth * kHeight * kDepth; ++b) {
        if (dst[b] != 0xCD) { ok = false; break; }
    }
    CHECK(ok, "cuMemcpy3D data correct");

    CHECK(cuMemcpy3DAsync(&cp, nullptr) == CUDA_SUCCESS, "cuMemcpy3DAsync");

    cuMemFree(d_src);
    cuMemFree(d_dst);
    cuCtxDestroy(ctx);
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    test_curand_get_property();
    test_curand_generate_poisson();
    test_cublas_status_strings();
    test_cufft_estimate_and_set_work_area();
    test_cuda_malloc3d_memcpy3d();
    test_cu_memcpy3d();

    if (g_failures == 0) {
        std::printf("PASS: misc_extended_api\n");
        return 0;
    }
    std::fprintf(stderr, "FAIL: %d test(s) failed\n", g_failures);
    return 1;
}
