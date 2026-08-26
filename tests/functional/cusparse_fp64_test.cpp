#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

// 3x3 CSR matrix:
// [1 0 2]
// [0 3 0]
// [4 0 5]
// rowPtr = [0 2 3 5], colInd = [0 2 1 0 2], values = [1 2 3 4 5]

static void test_spmv_fp64() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {0.0, 0.0, 0.0};

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, 3, 3, 5,
                      rowPtr, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_64F);

    double alpha = 1.0, beta = 0.0;
    size_t bufSize = 0;
    const cusparseStatus_t buffer_status =
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matA, vecX, &beta, vecY,
                                CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize);
    CHECK(buffer_status == CUSPARSE_STATUS_SUCCESS && bufSize > 0,
          "SpMV FP64 buffer size is usable");

    cusparseStatus_t st = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY,
                                        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
    CHECK(st == CUSPARSE_STATUS_SUCCESS, "SpMV FP64 returns success");

    // y = A * x = [1*1+2*3, 3*2, 4*1+5*3] = [7, 6, 19]
    CHECK(std::fabs(y[0] - 7.0) < 1e-12, "SpMV FP64 y[0]=7");
    CHECK(std::fabs(y[1] - 6.0) < 1e-12, "SpMV FP64 y[1]=6");
    CHECK(std::fabs(y[2] - 19.0) < 1e-12, "SpMV FP64 y[2]=19");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

static void test_spmv_fp64_alpha_beta() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double x[] = {1.0, 2.0, 3.0};
    double y[] = {10.0, 20.0, 30.0};

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, 3, 3, 5,
                      rowPtr, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnVecDescr_t vecX, vecY;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_64F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_64F);

    double alpha = 2.0, beta = 0.5;
    cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, vecX, &beta, vecY,
                 CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);

    // y = 2*(A*x) + 0.5*y_old = 2*[7,6,19] + 0.5*[10,20,30] = [19, 22, 53]
    CHECK(std::fabs(y[0] - 19.0) < 1e-12, "SpMV FP64 alpha=2 beta=0.5 y[0]=19");
    CHECK(std::fabs(y[1] - 22.0) < 1e-12, "SpMV FP64 alpha=2 beta=0.5 y[1]=22");
    CHECK(std::fabs(y[2] - 53.0) < 1e-12, "SpMV FP64 alpha=2 beta=0.5 y[2]=53");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
}

static void test_spmm_fp64() {
    cusparseHandle_t handle;
    cusparseCreate(&handle);

    // Same 3x3 sparse matrix
    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    double values[] = {1.0, 2.0, 3.0, 4.0, 5.0};

    // Dense 3x2 matrix B (column-major): [[1,4],[2,5],[3,6]]
    double B[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    double C[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA, 3, 3, 5,
                      rowPtr, colInd, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    cusparseDnMatDescr_t matB, matC;
    cusparseCreateDnMat(&matB, 3, 2, 3, B, CUDA_R_64F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, 3, 2, 3, C, CUDA_R_64F, CUSPARSE_ORDER_COL);

    double alpha = 1.0, beta = 0.0;
    size_t bufSize = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC,
                            CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize);

    cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                 &alpha, matA, matB, &beta, matC,
                 CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, nullptr);

    // C = A * B (col-major)
    // Row 0: 1*1+2*3=7, 1*4+2*6=16
    // Row 1: 3*2=6,     3*5=15
    // Row 2: 4*1+5*3=19, 4*4+5*6=46
    // Col-major: C = [7,6,19, 16,15,46]
    CHECK(std::fabs(C[0] - 7.0) < 1e-12, "SpMM FP64 C[0,0]=7");
    CHECK(std::fabs(C[1] - 6.0) < 1e-12, "SpMM FP64 C[1,0]=6");
    CHECK(std::fabs(C[2] - 19.0) < 1e-12, "SpMM FP64 C[2,0]=19");
    CHECK(std::fabs(C[3] - 16.0) < 1e-12, "SpMM FP64 C[0,1]=16");
    CHECK(std::fabs(C[4] - 15.0) < 1e-12, "SpMM FP64 C[1,1]=15");
    CHECK(std::fabs(C[5] - 46.0) < 1e-12, "SpMM FP64 C[2,1]=46");

    cusparseDestroySpMat(matA);
    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroy(handle);
}

int main() {
    test_spmv_fp64();
    test_spmv_fp64_alpha_beta();
    test_spmm_fp64();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
