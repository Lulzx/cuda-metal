#include <cusolverSp.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <limits>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// SPD 3x3 matrix (lower triangular stored, symmetric):
// A = [4  2  0]
//     [2  5  1]
//     [0  1  6]
// CSR of full matrix:
// row 0: (0,4) (1,2)
// row 1: (0,2) (1,5) (2,1)
// row 2: (1,1) (2,6)

static void test_cholesky_solve() {
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int rowPtr[] = {0, 2, 5, 7};
    int colInd[] = {0, 1, 0, 1, 2, 1, 2};
    float values[] = {4.0f, 2.0f, 2.0f, 5.0f, 1.0f, 1.0f, 6.0f};
    float b[] = {8.0f, 11.0f, 8.0f};
    float x[3] = {};
    int singularity = 0;

    cusolverStatus_t st = cusolverSpScsrlsvchol(handle, 3, 7, descrA,
                                                 values, rowPtr, colInd, b,
                                                 0.0f, 0, x, &singularity);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cholesky solve success");
    CHECK(singularity == -1, "no singularity");

    // A * x = b → x should be [1, 1, 1] (verify: 4+2=6? No.)
    // Let me recompute: A*[1,1,1] = [4+2, 2+5+1, 1+6] = [6, 8, 7] ≠ b
    // Pick b = A*[1,2,1] = [4+4, 2+10+1, 2+6] = [8, 13, 8]
    // Wait, let me just check that A*x ≈ b
    float Ax0 = 4*x[0] + 2*x[1];
    float Ax1 = 2*x[0] + 5*x[1] + 1*x[2];
    float Ax2 = 1*x[1] + 6*x[2];
    CHECK(std::fabs(Ax0 - b[0]) < 0.1f, "cholesky Ax[0]≈b[0]");
    CHECK(std::fabs(Ax1 - b[1]) < 0.1f, "cholesky Ax[1]≈b[1]");
    CHECK(std::fabs(Ax2 - b[2]) < 0.1f, "cholesky Ax[2]≈b[2]");

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

static void test_qr_solve() {
    cusolverSpHandle_t handle;
    cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    // Simple 2x2: A = [3 1; 1 2], b = [5, 5] → x = [1, 2]
    int rowPtr[] = {0, 2, 4};
    int colInd[] = {0, 1, 0, 1};
    float values[] = {3.0f, 1.0f, 1.0f, 2.0f};
    float b[] = {5.0f, 5.0f};
    float x[2] = {};
    int singularity = 0;

    cusolverStatus_t st = cusolverSpScsrlsvqr(handle, 2, 4, descrA,
                                               values, rowPtr, colInd, b,
                                               0.0f, 0, x, &singularity);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "qr solve success");
    CHECK(singularity == -1, "no singularity (qr)");
    CHECK(std::fabs(x[0] - 1.0f) < 1e-4f, "qr x[0]=1");
    CHECK(std::fabs(x[1] - 2.0f) < 1e-4f, "qr x[1]=2");

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

static void test_handle_lifecycle() {
    cusolverSpHandle_t handle;
    cusolverStatus_t st = cusolverSpCreate(&handle);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cusolverSp create");
    st = cusolverSpDestroy(handle);
    CHECK(st == CUSOLVER_STATUS_SUCCESS, "cusolverSp destroy");
}

static void test_validation_and_singularity() {
    cusolverSpHandle_t handle = nullptr;
    CHECK(cusolverSpCreate(&handle) == CUSOLVER_STATUS_SUCCESS,
          "validation handle create");
    cusparseMatDescr_t descrA = nullptr;
    cusparseCreateMatDescr(&descrA);

    int rowPtr[] = {0, 2, 4};
    int badRowPtr[] = {0, 3, 2};
    int colInd[] = {0, 1, 0, 1};
    int badColInd[] = {0, 2, 0, 1};
    float values[] = {3.0f, 1.0f, 1.0f, 2.0f};
    float b[] = {5.0f, 5.0f};
    float x[] = {91.0f, 92.0f};
    int singularity = 77;

    CHECK(cusolverSpScsrlsvqr(nullptr, 2, 4, descrA, values, rowPtr,
                               colInd, b, 0.0f, 0, x, &singularity) ==
              CUSOLVER_STATUS_NOT_INITIALIZED,
          "sparse null handle rejected");
    CHECK(cusolverSpScsrlsvqr(handle, -1, 4, descrA, values, rowPtr,
                               colInd, b, 0.0f, 0, x, &singularity) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "sparse negative dimension rejected");
    CHECK(cusolverSpScsrlsvqr(handle, 2, 3, descrA, values, rowPtr,
                               colInd, b, 0.0f, 0, x, &singularity) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "sparse nnz mismatch rejected");
    CHECK(cusolverSpScsrlsvqr(handle, 2, 4, descrA, values, badRowPtr,
                               colInd, b, 0.0f, 0, x, &singularity) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "sparse nonmonotonic row offsets rejected");
    CHECK(cusolverSpScsrlsvqr(handle, 2, 4, descrA, values, rowPtr,
                               badColInd, b, 0.0f, 0, x, &singularity) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "sparse out-of-range column rejected");
    CHECK(cusolverSpScsrlsvqr(handle, 2, 4, descrA, values, rowPtr,
                               colInd, b,
                               std::numeric_limits<float>::quiet_NaN(), 0,
                               x, &singularity) == CUSOLVER_STATUS_INVALID_VALUE,
          "sparse nonfinite tolerance rejected");
    CHECK(cusolverSpScsrlsvqr(handle, 2, 4, descrA, values, rowPtr,
                               colInd, b, 0.0f, 1, x, &singularity) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "unsupported sparse reordering rejected");
    CHECK(x[0] == 91.0f && x[1] == 92.0f && singularity == 77,
          "invalid sparse calls leave outputs unchanged");

    float nearSingular[] = {1.0f, 0.0f, 0.0f, 1.0e-8f};
    CHECK(cusolverSpScsrlsvchol(handle, 2, 4, descrA, nearSingular, rowPtr,
                                 colInd, b, 1.0e-3f, 0, x,
                                 &singularity) == CUSOLVER_STATUS_SUCCESS &&
              singularity == 1,
          "sparse Cholesky honors singularity tolerance");

    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

static void test_default_stream_ordering() {
    cusolverSpHandle_t handle = nullptr;
    cusolverSpCreate(&handle);
    cusparseMatDescr_t descrA = nullptr;
    cusparseCreateMatDescr(&descrA);

    int *rowPtr = nullptr, *colInd = nullptr;
    float *values = nullptr, *b = nullptr, *x = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&rowPtr), 3 * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&colInd), 4 * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&values), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&b), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&x), 2 * sizeof(float));
    const int hostRowPtr[] = {0, 2, 4};
    const int hostColInd[] = {0, 1, 0, 1};
    const float hostValues[] = {3.0f, 1.0f, 1.0f, 2.0f};
    const float hostB[] = {5.0f, 5.0f};
    cudaMemcpyAsync(rowPtr, hostRowPtr, sizeof(hostRowPtr),
                    cudaMemcpyHostToDevice, nullptr);
    cudaMemcpyAsync(colInd, hostColInd, sizeof(hostColInd),
                    cudaMemcpyHostToDevice, nullptr);
    cudaMemcpyAsync(values, hostValues, sizeof(hostValues),
                    cudaMemcpyHostToDevice, nullptr);
    cudaMemcpyAsync(b, hostB, sizeof(hostB), cudaMemcpyHostToDevice, nullptr);
    int singularity = 99;
    const cusolverStatus_t status = cusolverSpScsrlsvqr(
        handle, 2, 4, descrA, values, rowPtr, colInd, b, 0.0f, 0, x,
        &singularity);
    float result[2] = {};
    cudaMemcpy(result, x, sizeof(result), cudaMemcpyDeviceToHost);
    CHECK(status == CUSOLVER_STATUS_SUCCESS && singularity == -1 &&
              std::fabs(result[0] - 1.0f) < 1e-4f &&
              std::fabs(result[1] - 2.0f) < 1e-4f,
          "sparse solver orders default-stream inputs");

    cudaFree(x);
    cudaFree(b);
    cudaFree(values);
    cudaFree(colInd);
    cudaFree(rowPtr);
    cusparseDestroyMatDescr(descrA);
    cusolverSpDestroy(handle);
}

int main() {
    test_handle_lifecycle();
    test_validation_and_singularity();
    test_default_stream_ordering();
    test_cholesky_solve();
    test_qr_solve();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
