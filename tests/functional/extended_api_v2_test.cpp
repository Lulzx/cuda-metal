// Functional tests for extended APIs — batch 2:
//   curandCreateGeneratorHost
//   cublasGetProperty, cublasSsyr/Dsyr, cublasSsyrk/Dsyrk, cublasSsyr2k/Dsyr2k
//   cuFuncSetAttribute, cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (driver)
//   cudaMemcpyPeer/Async, cudaLaunchHostFunc
//   cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags (runtime)
//   cuCtxPushCurrent/PopCurrent, cuDevicePrimaryCtxRetain/Release
//   cuStreamGetPriority/GetFlags, cuModuleGetGlobal

#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

static int g_failures = 0;

#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL: %s\n", msg);                \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// ── curand ────────────────────────────────────────────────────────────────────

static void test_curand_create_generator_host() {
    curandGenerator_t gen = nullptr;
    CHECK(curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
          "curandCreateGeneratorHost succeeds");
    CHECK(gen != nullptr, "curandCreateGeneratorHost non-null");

    // Set seed and generate a few values
    CHECK(curandSetPseudoRandomGeneratorSeed(gen, 42ULL) == CURAND_STATUS_SUCCESS,
          "curandSetSeed on host generator");

    // On UMA, device memory == host memory so we can pass a malloc'd buffer.
    // Use cudaMalloc which on CuMetal returns a host-coherent pointer.
    float* buf = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&buf), 16 * sizeof(float)) == cudaSuccess,
          "cudaMalloc for host-gen test");
    CHECK(curandGenerateUniform(gen, buf, 16) == CURAND_STATUS_SUCCESS,
          "curandGenerateUniform on host generator");
    for (int i = 0; i < 16; ++i) {
        CHECK(buf[i] >= 0.0f && buf[i] < 1.0f, "host-gen uniform in [0,1)");
    }

    cudaFree(buf);
    curandDestroyGenerator(gen);
}

// ── cublas property ───────────────────────────────────────────────────────────

static void test_cublas_get_property() {
    int major = -1, minor = -1, patch = -1;
    CHECK(cublasGetProperty(MAJOR_VERSION, &major) == CUBLAS_STATUS_SUCCESS,
          "cublasGetProperty MAJOR_VERSION");
    CHECK(cublasGetProperty(MINOR_VERSION, &minor) == CUBLAS_STATUS_SUCCESS,
          "cublasGetProperty MINOR_VERSION");
    CHECK(cublasGetProperty(PATCH_LEVEL, &patch) == CUBLAS_STATUS_SUCCESS,
          "cublasGetProperty PATCH_LEVEL");
    CHECK(major >= 0 && minor >= 0 && patch >= 0, "cublasGetProperty values >= 0");
    // Null pointer returns error
    CHECK(cublasGetProperty(MAJOR_VERSION, nullptr) == CUBLAS_STATUS_INVALID_VALUE,
          "cublasGetProperty null returns error");
}

// ── cublasSsyr / cublasDsyr ───────────────────────────────────────────────────

static void test_cublas_ssyr() {
    // 2x2 identity A, x = [1,0], alpha=1 → upper triangle: A[0,0] += 1
    cublasHandle_t h = nullptr;
    CHECK(cublasCreate(&h) == CUBLAS_STATUS_SUCCESS, "cublasCreate for Ssyr");

    const int n = 2;
    float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};  // column-major identity
    float x[2] = {1.0f, 0.0f};
    const float alpha = 1.0f;
    // Use host pointers on UMA
    float* da = nullptr;
    float* dx = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(float));
    cudaMemcpy(da, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSsyr(h, CUBLAS_FILL_MODE_UPPER, n, &alpha, dx, 1, da, n)
          == CUBLAS_STATUS_SUCCESS, "cublasSsyr succeeds");

    cudaMemcpy(a, da, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    // A[0,0] (column 0, row 0) = 1 + 1*1 = 2
    CHECK(std::fabs(a[0] - 2.0f) < 1e-5f, "cublasSsyr A[0,0]=2");
    // A[1,0] (row 1, col 0 — lower, not updated by UPPER): still 0
    CHECK(std::fabs(a[1] - 0.0f) < 1e-5f, "cublasSsyr A[1,0] unchanged");

    cudaFree(da);
    cudaFree(dx);
    cublasDestroy(h);
}

static void test_cublas_dsyr() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);

    const int n = 2;
    double a[4] = {0.0, 0.0, 0.0, 0.0};
    double x[2] = {2.0, 3.0};
    const double alpha = 1.0;
    double* da = nullptr;
    double* dx = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDsyr(h, CUBLAS_FILL_MODE_LOWER, n, &alpha, dx, 1, da, n)
          == CUBLAS_STATUS_SUCCESS, "cublasDsyr succeeds");

    cudaMemcpy(a, da, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    // lower: A[0,0]=4, A[1,0]=6, A[1,1]=9; A[0,1] unchanged
    CHECK(std::fabs(a[0] - 4.0) < 1e-10, "cublasDsyr A[0,0]=4");
    CHECK(std::fabs(a[1] - 6.0) < 1e-10, "cublasDsyr A[1,0]=6");
    CHECK(std::fabs(a[3] - 9.0) < 1e-10, "cublasDsyr A[1,1]=9");

    cudaFree(da);
    cudaFree(dx);
    cublasDestroy(h);
}

// ── cublasSsyrk / cublasDsyrk ─────────────────────────────────────────────────

static void test_cublas_ssyrk() {
    // C = alpha*A*A^T + beta*C, n=2, k=1, A=[1,2]^T (col-major 2x1), alpha=1, beta=0
    // Result: C[0,0]=1, C[0,1]=C[1,0]=2, C[1,1]=4 (upper only updated)
    cublasHandle_t h = nullptr;
    cublasCreate(&h);

    const int n = 2, k = 1;
    float a[2] = {1.0f, 2.0f};  // 2x1 column-major
    float c[4] = {0.0f};
    const float alpha = 1.0f, beta = 0.0f;
    float* da = nullptr;
    float* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(float));
    cudaMemcpy(da, a, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSsyrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                      &alpha, da, n, &beta, dc, n) == CUBLAS_STATUS_SUCCESS,
          "cublasSsyrk succeeds");

    cudaMemcpy(c, dc, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(c[0] - 1.0f) < 1e-5f, "cublasSsyrk C[0,0]=1");
    // upper: C[0,1] (row 0, col 1) = a[0]*a[1] = 2; stored at c[0+1*2]=c[2]
    CHECK(std::fabs(c[2] - 2.0f) < 1e-5f, "cublasSsyrk C[0,1]=2");
    CHECK(std::fabs(c[3] - 4.0f) < 1e-5f, "cublasSsyrk C[1,1]=4");

    cudaFree(da);
    cudaFree(dc);
    cublasDestroy(h);
}

static void test_cublas_dsyrk() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);

    const int n = 2, k = 2;
    // A is 2x2 identity (column-major): [1,0,0,1]
    double a[4] = {1.0, 0.0, 0.0, 1.0};
    double c[4] = {0.0};
    const double alpha = 1.0, beta = 0.0;
    double* da = nullptr;
    double* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(double), cudaMemcpyDeviceToHost);

    CHECK(cublasDsyrk(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
                      &alpha, da, n, &beta, dc, n) == CUBLAS_STATUS_SUCCESS,
          "cublasDsyrk succeeds");

    cudaMemcpy(c, dc, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    // I*I^T = I → C = I
    CHECK(std::fabs(c[0] - 1.0) < 1e-10, "cublasDsyrk C[0,0]=1");
    CHECK(std::fabs(c[1] - 0.0) < 1e-10, "cublasDsyrk C[1,0]=0");
    CHECK(std::fabs(c[3] - 1.0) < 1e-10, "cublasDsyrk C[1,1]=1");

    cudaFree(da);
    cudaFree(dc);
    cublasDestroy(h);
}

// ── cublasSsyr2k / cublasDsyr2k ───────────────────────────────────────────────

static void test_cublas_ssyr2k() {
    // C = alpha*(A*B^T + B*A^T) + beta*C
    // n=2, k=1, A=[1,0]^T, B=[0,1]^T, alpha=0.5, beta=0
    // A*B^T = [0,1;0,0], B*A^T = [0,0;1,0]
    // sum = [0,1;1,0], *0.5 = [0,0.5;0.5,0]
    // upper: C[0,0]=0, C[0,1]=0.5
    cublasHandle_t h = nullptr;
    cublasCreate(&h);

    const int n = 2, k = 1;
    float a[2] = {1.0f, 0.0f};
    float b[2] = {0.0f, 1.0f};
    float c[4] = {0.0f};
    const float alpha = 0.5f, beta = 0.0f;
    float* da = nullptr;
    float* db = nullptr;
    float* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&db), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(float));
    cudaMemcpy(da, a, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSsyr2k(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k,
                       &alpha, da, n, db, n, &beta, dc, n) == CUBLAS_STATUS_SUCCESS,
          "cublasSsyr2k succeeds");

    cudaMemcpy(c, dc, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(c[0] - 0.0f) < 1e-5f, "cublasSsyr2k C[0,0]=0");
    CHECK(std::fabs(c[2] - 0.5f) < 1e-5f, "cublasSsyr2k C[0,1]=0.5");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cublasDestroy(h);
}

static void test_cublas_dsyr2k() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);

    const int n = 2, k = 1;
    double a[2] = {1.0, 1.0};
    double b[2] = {1.0, 1.0};
    double c[4] = {0.0};
    const double alpha = 1.0, beta = 0.0;
    double* da = nullptr;
    double* db = nullptr;
    double* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 2 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&db), 2 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(double));
    cudaMemcpy(da, a, 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDsyr2k(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k,
                       &alpha, da, n, db, n, &beta, dc, n) == CUBLAS_STATUS_SUCCESS,
          "cublasDsyr2k succeeds");

    cudaMemcpy(c, dc, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    // A=[1,1]^T, B=[1,1]^T → A*B^T+B*A^T = 2*[[1,1],[1,1]]
    // lower: C[0,0]=2, C[1,0]=2, C[1,1]=2
    CHECK(std::fabs(c[0] - 2.0) < 1e-10, "cublasDsyr2k C[0,0]=2");
    CHECK(std::fabs(c[1] - 2.0) < 1e-10, "cublasDsyr2k C[1,0]=2");
    CHECK(std::fabs(c[3] - 2.0) < 1e-10, "cublasDsyr2k C[1,1]=2");

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    cublasDestroy(h);
}

// ── driver: cuFuncSetAttribute / cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ──

static void test_driver_func_set_attribute() {
    // cuFuncSetAttribute is a no-op; should return CUDA_SUCCESS even with null func.
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for func_set_attribute test");
    // Passing nullptr function — spec says CUDA_ERROR_INVALID_VALUE is acceptable,
    // but CuMetal always returns SUCCESS as a no-op.
    const CUresult r = cuFuncSetAttribute(nullptr, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, 256);
    CHECK(r == CUDA_SUCCESS, "cuFuncSetAttribute no-op returns CUDA_SUCCESS");
}

static void test_driver_occupancy_with_flags() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for occupancy_with_flags test");
    int nb1 = 0, nb2 = 0;
    const CUresult r1 = cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb1, nullptr, 256, 0);
    const CUresult r2 = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &nb2, nullptr, 256, 0, 0);
    CHECK(r1 == r2, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags same result");
    CHECK(nb1 == nb2, "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags same count");
}

// ── runtime: cudaMemcpyPeer ───────────────────────────────────────────────────

static void test_cuda_memcpy_peer() {
    float* src = nullptr;
    float* dst = nullptr;
    const float val = 3.14f;
    cudaMalloc(reinterpret_cast<void**>(&src), sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dst), sizeof(float));
    cudaMemcpy(src, &val, sizeof(float), cudaMemcpyHostToDevice);

    // On UMA single GPU: peer copy == regular copy
    CHECK(cudaMemcpyPeer(dst, 0, src, 0, sizeof(float)) == cudaSuccess,
          "cudaMemcpyPeer succeeds");
    float out = 0.0f;
    cudaMemcpy(&out, dst, sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(out - val) < 1e-6f, "cudaMemcpyPeer value correct");

    cudaFree(src);
    cudaFree(dst);
}

static void test_cuda_memcpy_peer_async() {
    float* src = nullptr;
    float* dst = nullptr;
    const float val = 2.71f;
    cudaMalloc(reinterpret_cast<void**>(&src), sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dst), sizeof(float));
    cudaMemcpy(src, &val, sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    CHECK(cudaMemcpyPeerAsync(dst, 0, src, 0, sizeof(float), stream) == cudaSuccess,
          "cudaMemcpyPeerAsync succeeds");
    cudaStreamSynchronize(stream);
    float out = 0.0f;
    cudaMemcpy(&out, dst, sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(out - val) < 1e-6f, "cudaMemcpyPeerAsync value correct");

    cudaStreamDestroy(stream);
    cudaFree(src);
    cudaFree(dst);
}

// ── runtime: cudaLaunchHostFunc ───────────────────────────────────────────────

static void test_cuda_launch_host_func() {
    int flag = 0;
    auto fn = [](void* ud) { *static_cast<int*>(ud) = 42; };
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    CHECK(cudaLaunchHostFunc(stream, fn, &flag) == cudaSuccess,
          "cudaLaunchHostFunc succeeds");
    CHECK(flag == 42, "cudaLaunchHostFunc callback executed");
    cudaStreamDestroy(stream);
}

// ── runtime: cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags ───────────

static void test_runtime_occupancy_with_flags() {
    int nb1 = 0, nb2 = 0;
    const cudaError_t r1 = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &nb1, nullptr, 256, 0);
    const cudaError_t r2 = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        &nb2, nullptr, 256, 0, 0);
    CHECK(r1 == r2, "cudaOccupancyWithFlags same error code as base");
    CHECK(nb1 == nb2, "cudaOccupancyWithFlags same count as base");
}

// ── driver: context push/pop ──────────────────────────────────────────────────

static void test_driver_ctx_push_pop() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for push/pop test");
    CUdevice dev = 0;
    CHECK(cuDeviceGet(&dev, 0) == CUDA_SUCCESS, "cuDeviceGet");

    CUcontext ctx = nullptr;
    CHECK(cuCtxCreate(&ctx, 0, dev) == CUDA_SUCCESS, "cuCtxCreate");
    CHECK(cuCtxPushCurrent(ctx) == CUDA_SUCCESS, "cuCtxPushCurrent");

    CUcontext popped = nullptr;
    CHECK(cuCtxPopCurrent(&popped) == CUDA_SUCCESS, "cuCtxPopCurrent");

    cuCtxDestroy(ctx);
}

// ── driver: primary context ───────────────────────────────────────────────────

static void test_driver_primary_ctx() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for primary ctx test");
    CUdevice dev = 0;
    CHECK(cuDeviceGet(&dev, 0) == CUDA_SUCCESS, "cuDeviceGet for primary");

    CUcontext ctx = nullptr;
    CHECK(cuDevicePrimaryCtxRetain(&ctx, dev) == CUDA_SUCCESS,
          "cuDevicePrimaryCtxRetain");
    CHECK(ctx != nullptr, "cuDevicePrimaryCtxRetain non-null");
    CHECK(cuDevicePrimaryCtxRelease(dev) == CUDA_SUCCESS,
          "cuDevicePrimaryCtxRelease");
}

// ── driver: stream priority / flags ──────────────────────────────────────────

static void test_driver_stream_priority_flags() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for stream priority/flags test");
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    cuDeviceGet(&dev, 0);
    CHECK(cuCtxCreate(&ctx, 0, dev) == CUDA_SUCCESS, "cuCtxCreate for stream flags test");

    CUstream stream = nullptr;
    CHECK(cuStreamCreate(&stream, 0) == CUDA_SUCCESS, "cuStreamCreate");

    int priority = -99;
    CHECK(cuStreamGetPriority(stream, &priority) == CUDA_SUCCESS,
          "cuStreamGetPriority succeeds");
    CHECK(priority == 0, "cuStreamGetPriority returns 0");

    unsigned int flags = 0xDEAD;
    CHECK(cuStreamGetFlags(stream, &flags) == CUDA_SUCCESS,
          "cuStreamGetFlags succeeds");
    CHECK(flags == 0, "cuStreamGetFlags returns 0");

    cuStreamDestroy(stream);
    cuCtxDestroy(ctx);
}

// ── driver: cuModuleGetGlobal ─────────────────────────────────────────────────

static void test_driver_module_get_global() {
    CHECK(cuInit(0) == CUDA_SUCCESS, "cuInit for module_get_global test");
    // CuMetal has no runtime-addressable globals; expect NOT_FOUND
    CUdeviceptr dptr = 0xDEAD;
    size_t bytes = 0xDEAD;
    const CUresult r = cuModuleGetGlobal(&dptr, &bytes, nullptr, "some_global");
    CHECK(r == CUDA_ERROR_NOT_FOUND || r == CUDA_ERROR_INVALID_VALUE,
          "cuModuleGetGlobal returns NOT_FOUND or INVALID_VALUE");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_curand_create_generator_host();
    test_cublas_get_property();
    test_cublas_ssyr();
    test_cublas_dsyr();
    test_cublas_ssyrk();
    test_cublas_dsyrk();
    test_cublas_ssyr2k();
    test_cublas_dsyr2k();
    test_driver_func_set_attribute();
    test_driver_occupancy_with_flags();
    test_cuda_memcpy_peer();
    test_cuda_memcpy_peer_async();
    test_cuda_launch_host_func();
    test_runtime_occupancy_with_flags();
    test_driver_ctx_push_pop();
    test_driver_primary_ctx();
    test_driver_stream_priority_flags();
    test_driver_module_get_global();

    if (g_failures == 0) {
        std::printf("PASS: extended_api_v2 (%d sub-tests)\n", 18);
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d sub-test(s) failed\n", g_failures);
    return 1;
}
