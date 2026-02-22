// extended_api_v5_test.cpp
// Tests: cublasCgemm/cublasZgemm (complex GEMM),
//        cublasCgemv/cublasZgemv (complex GEMV),
//        cudaThreadExit/Synchronize/GetCacheConfig/SetCacheConfig (legacy thread API),
//        cuDevicePrimaryCtxGetState/SetFlags/Reset + cuDeviceGetUuid (driver primary ctx),
//        curandGetGeneratorType/SetGeneratorOrdering/SetQuasiRandomGeneratorDimensions

#include <cstdio>
#include <cmath>
#include <vector>

#include "cuda_runtime.h"
#include "cuda.h"
#include "cublas_v2.h"
#include "curand.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, name)                                            \
    do {                                                             \
        if (cond) { printf("  PASS: %s\n", name); ++g_pass; }       \
        else      { printf("  FAIL: %s\n", name); ++g_fail; }       \
    } while (0)

// ── cublasCgemm ───────────────────────────────────────────────────────────────
// Compute C = alpha * A * B + beta * C  (no transpose) with small matrices.
// A: 2×3, B: 3×2, C: 2×2  — all column-major (cuBLAS default).
static void test_cublas_cgemm() {
    printf("[cublasCgemm]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // A (2×3, col-major):  [1+0i, 0+1i | 1+1i, 0+0i | 0+0i, 1+0i]
    // Column 0: {1,0},{0,1}  column 1: {1,1},{0,0}  column 2: {0,0},{1,0}
    std::vector<cuComplex> A = {{1,0},{0,1},{1,1},{0,0},{0,0},{1,0}};
    // B (3×2, col-major):  identity-like
    // Column 0: {1,0},{0,0},{0,0}  column 1: {0,0},{1,0},{0,0}
    std::vector<cuComplex> B = {{1,0},{0,0},{0,0},{0,0},{1,0},{0,0}};
    // C = 0
    std::vector<cuComplex> C(4, {0,0});

    // Allocate device buffers
    void *dA, *dB, *dC;
    cudaMalloc(&dA, A.size() * sizeof(cuComplex));
    cudaMalloc(&dB, B.size() * sizeof(cuComplex));
    cudaMalloc(&dC, C.size() * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), A.size() * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size() * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1.0f, 0.0f}, beta = {0.0f, 0.0f};
    cublasStatus_t st = cublasCgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     2, 2, 3,
                                     &alpha,
                                     static_cast<cuComplex*>(dA), 2,
                                     static_cast<cuComplex*>(dB), 3,
                                     &beta,
                                     static_cast<cuComplex*>(dC), 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCgemm returns success");

    cudaMemcpy(C.data(), dC, C.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost);

    // C = A[:,0..1] * B[0..1,:]  (3-column contraction)
    // C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    //        = {1,0}*{1,0} + {1,1}*{0,0} + {0,0}*{0,0} = {1,0}
    // C[1,0] = {0,1}*{1,0} + {0,0}*{0,0} + {1,0}*{0,0} = {0,1}
    // C[0,1] = {1,0}*{0,0} + {1,1}*{1,0} + {0,0}*{0,0} = {1,1}
    // C[1,1] = {0,1}*{0,0} + {0,0}*{1,0} + {1,0}*{0,0} = {0,0}
    bool ok = (std::fabsf(C[0].x - 1.0f) < 1e-5f && std::fabsf(C[0].y) < 1e-5f &&
               std::fabsf(C[1].x) < 1e-5f && std::fabsf(C[1].y - 1.0f) < 1e-5f &&
               std::fabsf(C[2].x - 1.0f) < 1e-5f && std::fabsf(C[2].y - 1.0f) < 1e-5f &&
               std::fabsf(C[3].x) < 1e-5f && std::fabsf(C[3].y) < 1e-5f);
    CHECK(ok, "cublasCgemm result correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
}

// ── cublasZgemm ───────────────────────────────────────────────────────────────
// Simple 2×2 * 2×2 = 2×2 double-complex product.
static void test_cublas_zgemm() {
    printf("[cublasZgemm]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // A = [[1+1i, 0], [0, 1-1i]]  (col-major: col0={1+i,0}, col1={0,1-i})
    std::vector<cuDoubleComplex> A = {{1,1},{0,0},{0,0},{1,-1}};
    // B = [[2,0],[0,2]] (col-major: scaled identity)
    std::vector<cuDoubleComplex> B = {{2,0},{0,0},{0,0},{2,0}};
    std::vector<cuDoubleComplex> C(4, {0,0});

    void *dA, *dB, *dC;
    cudaMalloc(&dA, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&dB, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&dC, 4 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZgemm(handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     2, 2, 2,
                                     &alpha,
                                     static_cast<cuDoubleComplex*>(dA), 2,
                                     static_cast<cuDoubleComplex*>(dB), 2,
                                     &beta,
                                     static_cast<cuDoubleComplex*>(dC), 2);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZgemm returns success");

    cudaMemcpy(C.data(), dC, 4 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // C = A * 2I = 2A = [[2+2i,0],[0,2-2i]]
    bool ok = (std::fabs(C[0].x - 2.0) < 1e-10 && std::fabs(C[0].y - 2.0) < 1e-10 &&
               std::fabs(C[1].x) < 1e-10       && std::fabs(C[1].y) < 1e-10       &&
               std::fabs(C[2].x) < 1e-10       && std::fabs(C[2].y) < 1e-10       &&
               std::fabs(C[3].x - 2.0) < 1e-10 && std::fabs(C[3].y - (-2.0)) < 1e-10);
    CHECK(ok, "cublasZgemm result correct");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
}

// ── cublasCgemv ───────────────────────────────────────────────────────────────
// y = alpha * A * x + beta * y, A: 2×2, x: 2, y: 2
static void test_cublas_cgemv() {
    printf("[cublasCgemv]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // A (2×2, col-major): [[1+i, 2], [0, 1-i]]
    // col0 = {1+i, 0}, col1 = {2,0},{1-i,0}  — wait, col-major: A[col*lda+row]
    // A[0,0]={1,1}, A[1,0]={0,0}, A[0,1]={2,0}, A[1,1]={1,-1}
    std::vector<cuComplex> A = {{1,1},{0,0},{2,0},{1,-1}};
    std::vector<cuComplex> x = {{1,0},{1,0}};
    std::vector<cuComplex> y = {{0,0},{0,0}};

    void *dA, *dx, *dy;
    cudaMalloc(&dA, 4 * sizeof(cuComplex));
    cudaMalloc(&dx, 2 * sizeof(cuComplex));
    cudaMalloc(&dy, 2 * sizeof(cuComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), 2 * sizeof(cuComplex), cudaMemcpyHostToDevice);

    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasCgemv(handle, CUBLAS_OP_N, 2, 2,
                                     &alpha,
                                     static_cast<cuComplex*>(dA), 2,
                                     static_cast<cuComplex*>(dx), 1,
                                     &beta,
                                     static_cast<cuComplex*>(dy), 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasCgemv returns success");

    cudaMemcpy(y.data(), dy, 2 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // y[0] = A[0,0]*x[0] + A[0,1]*x[1] = {1,1}*{1,0} + {2,0}*{1,0} = {1,1}+{2,0} = {3,1}
    // y[1] = A[1,0]*x[0] + A[1,1]*x[1] = {0,0}*{1,0} + {1,-1}*{1,0} = {1,-1}
    bool ok = (std::fabsf(y[0].x - 3.0f) < 1e-5f && std::fabsf(y[0].y - 1.0f) < 1e-5f &&
               std::fabsf(y[1].x - 1.0f) < 1e-5f && std::fabsf(y[1].y - (-1.0f)) < 1e-5f);
    CHECK(ok, "cublasCgemv result correct");

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

// ── cublasZgemv (conjugate-transpose) ─────────────────────────────────────────
static void test_cublas_zgemv_conj() {
    printf("[cublasZgemv CUBLAS_OP_C]\n");
    cublasHandle_t handle;
    cublasCreate(&handle);

    // A (2×2): [[1+i, 2-i],[3, 4+i]]  col-major: A={1+i,3,2-i,4+i}
    std::vector<cuDoubleComplex> A = {{1,1},{3,0},{2,-1},{4,1}};
    std::vector<cuDoubleComplex> x = {{1,0},{0,1}};  // x=[1, i]
    std::vector<cuDoubleComplex> y = {{0,0},{0,0}};

    void *dA, *dx, *dy;
    cudaMalloc(&dA, 4 * sizeof(cuDoubleComplex));
    cudaMalloc(&dx, 2 * sizeof(cuDoubleComplex));
    cudaMalloc(&dy, 2 * sizeof(cuDoubleComplex));
    cudaMemcpy(dA, A.data(), 4 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y.data(), 2 * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZgemv(handle, CUBLAS_OP_C, 2, 2,
                                     &alpha,
                                     static_cast<cuDoubleComplex*>(dA), 2,
                                     static_cast<cuDoubleComplex*>(dx), 1,
                                     &beta,
                                     static_cast<cuDoubleComplex*>(dy), 1);
    CHECK(st == CUBLAS_STATUS_SUCCESS, "cublasZgemv CUBLAS_OP_C returns success");

    cudaMemcpy(y.data(), dy, 2 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
    // A^H: row 0 of A^H = conj(col 0 of A) = {1-i, 3}
    // y[0] = conj(A[0,0])*x[0] + conj(A[1,0])*x[1]
    //      = {1,-1}*{1,0} + {3,0}*{0,1} = {1,-1} + {0,3} = {1,2}
    // y[1] = conj(A[0,1])*x[0] + conj(A[1,1])*x[1]
    //      = {2,1}*{1,0} + {4,-1}*{0,1} = {2,1} + {1,4}  = {3,5}
    bool ok = (std::fabs(y[0].x - 1.0) < 1e-9 && std::fabs(y[0].y - 2.0) < 1e-9 &&
               std::fabs(y[1].x - 3.0) < 1e-9 && std::fabs(y[1].y - 5.0) < 1e-9);
    CHECK(ok, "cublasZgemv CUBLAS_OP_C result correct");

    cudaFree(dA); cudaFree(dx); cudaFree(dy);
    cublasDestroy(handle);
}

// ── Legacy thread API ─────────────────────────────────────────────────────────
static void test_legacy_thread_api() {
    printf("[cudaThread* legacy API]\n");

    cudaFuncCache cfg = cudaFuncCachePreferNone;
    cudaError_t r1 = cudaThreadGetCacheConfig(&cfg);
    CHECK(r1 == cudaSuccess, "cudaThreadGetCacheConfig returns success");

    cudaError_t r2 = cudaThreadSetCacheConfig(cudaFuncCachePreferShared);
    CHECK(r2 == cudaSuccess, "cudaThreadSetCacheConfig returns success");

    cudaError_t r3 = cudaThreadSynchronize();
    CHECK(r3 == cudaSuccess, "cudaThreadSynchronize returns success");

    // cudaThreadExit resets the device — re-init afterwards.
    cudaError_t r4 = cudaThreadExit();
    CHECK(r4 == cudaSuccess, "cudaThreadExit returns success");

    // Re-initialize so subsequent tests work.
    cudaSetDevice(0);
}

// ── Driver primary context API ────────────────────────────────────────────────
static void test_driver_primary_ctx() {
    printf("[cuDevicePrimaryCtx* + cuDeviceGetUuid]\n");

    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);

    // GetState before any context.
    unsigned int flags = 0xDEAD;
    int active = -1;
    CUresult r1 = cuDevicePrimaryCtxGetState(dev, &flags, &active);
    CHECK(r1 == CUDA_SUCCESS, "cuDevicePrimaryCtxGetState returns success");
    CHECK(active == 0 || active == 1, "cuDevicePrimaryCtxGetState active is 0 or 1");

    // SetFlags — should succeed whether or not a context exists.
    CUresult r2 = cuDevicePrimaryCtxSetFlags(dev, CU_CTX_SCHED_AUTO);
    CHECK(r2 == CUDA_SUCCESS, "cuDevicePrimaryCtxSetFlags returns success");

    // GetState again — flags should reflect the value we set.
    unsigned int flags2 = 0;
    cuDevicePrimaryCtxGetState(dev, &flags2, nullptr);
    CHECK(flags2 == CU_CTX_SCHED_AUTO, "cuDevicePrimaryCtxGetState flags matches SetFlags");

    // Reset.
    CUresult r3 = cuDevicePrimaryCtxReset(dev);
    CHECK(r3 == CUDA_SUCCESS, "cuDevicePrimaryCtxReset returns success");

    // Invalid device.
    CUresult r4 = cuDevicePrimaryCtxGetState(99, &flags, &active);
    CHECK(r4 == CUDA_ERROR_INVALID_DEVICE, "cuDevicePrimaryCtxGetState invalid device");

    // UUID.
    CUuuid uuid;
    CUresult r5 = cuDeviceGetUuid(&uuid, dev);
    CHECK(r5 == CUDA_SUCCESS, "cuDeviceGetUuid returns success");
    // Verify the "CuMetal1" prefix.
    const unsigned char expected[8] = {0x43,0x75,0x4d,0x65,0x74,0x61,0x6c,0x31};
    bool uuid_ok = true;
    for (int i = 0; i < 8; ++i)
        if (uuid.bytes[i] != expected[i]) { uuid_ok = false; break; }
    CHECK(uuid_ok, "cuDeviceGetUuid returns CuMetal1 prefix");

    // Null pointer.
    CUresult r6 = cuDeviceGetUuid(nullptr, dev);
    CHECK(r6 == CUDA_ERROR_INVALID_VALUE, "cuDeviceGetUuid null ptr error");

    // Re-create context for any subsequent tests.
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);
}

// ── cuRAND generator type / ordering / quasi-dimensions ───────────────────────
static void test_curand_generator_props() {
    printf("[curand generator type / ordering / quasi-dimensions]\n");

    // Create a pseudo generator and query its type.
    curandGenerator_t gen;
    curandStatus_t r1 = curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    CHECK(r1 == CURAND_STATUS_SUCCESS, "curandCreateGenerator PSEUDO_MT19937 success");

    curandRngType_t rtype = CURAND_RNG_PSEUDO_DEFAULT;
    curandStatus_t r2 = curandGetGeneratorType(gen, &rtype);
    CHECK(r2 == CURAND_STATUS_SUCCESS, "curandGetGeneratorType returns success");
    CHECK(rtype == CURAND_RNG_PSEUDO_MT19937, "curandGetGeneratorType returns correct type");

    // Set ordering.
    curandStatus_t r3 = curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_BEST);
    CHECK(r3 == CURAND_STATUS_SUCCESS, "curandSetGeneratorOrdering PSEUDO_BEST success");

    curandStatus_t r4 = curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_LEGACY);
    CHECK(r4 == CURAND_STATUS_SUCCESS, "curandSetGeneratorOrdering PSEUDO_LEGACY success");

    // Invalid ordering.
    curandStatus_t r5 = curandSetGeneratorOrdering(gen, static_cast<curandOrdering_t>(999));
    CHECK(r5 == CURAND_STATUS_OUT_OF_RANGE, "curandSetGeneratorOrdering bad value error");

    curandDestroyGenerator(gen);

    // Create a quasi generator and set dimensions.
    curandGenerator_t qgen;
    curandStatus_t r6 = curandCreateGenerator(&qgen, CURAND_RNG_QUASI_SOBOL32);
    CHECK(r6 == CURAND_STATUS_SUCCESS, "curandCreateGenerator QUASI_SOBOL32 success");

    curandRngType_t qtype = CURAND_RNG_PSEUDO_DEFAULT;
    curandGetGeneratorType(qgen, &qtype);
    CHECK(qtype == CURAND_RNG_QUASI_SOBOL32, "quasi generator type correct");

    curandStatus_t r7 = curandSetQuasiRandomGeneratorDimensions(qgen, 7);
    CHECK(r7 == CURAND_STATUS_SUCCESS, "curandSetQuasiRandomGeneratorDimensions(7) success");

    curandStatus_t r8 = curandSetQuasiRandomGeneratorDimensions(qgen, 0);
    CHECK(r8 == CURAND_STATUS_OUT_OF_RANGE, "curandSetQuasiRandomGeneratorDimensions(0) error");

    curandStatus_t r9 = curandSetQuasiRandomGeneratorDimensions(qgen, 20001);
    CHECK(r9 == CURAND_STATUS_OUT_OF_RANGE, "curandSetQuasiRandomGeneratorDimensions(20001) error");

    curandDestroyGenerator(qgen);

    // Unknown RNG type still rejected.
    curandGenerator_t bad;
    curandStatus_t r10 = curandCreateGenerator(&bad, static_cast<curandRngType_t>(999));
    CHECK(r10 == CURAND_STATUS_TYPE_ERROR, "curandCreateGenerator unknown type error");
}

// ─────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== extended_api_v5 tests ===\n");

    // Driver init.
    cuInit(0);
    CUdevice dev;
    cuDeviceGet(&dev, 0);
    CUcontext ctx;
    cuCtxCreate(&ctx, 0, dev);

    test_cublas_cgemm();
    test_cublas_zgemm();
    test_cublas_cgemv();
    test_cublas_zgemv_conj();
    test_legacy_thread_api();
    test_driver_primary_ctx();
    test_curand_generator_props();

    printf("\nResults: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
