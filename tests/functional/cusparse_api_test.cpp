#include "cusparse.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstring>

// Small 3×3 CSR matrix:
//   [1 0 2]
//   [0 3 0]
//   [4 0 5]
// CSR: rowPtr = [0,2,3,5], colInd = [0,2,1,0,2], vals = [1,2,3,4,5]

static bool test_handle_lifecycle() {
    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cusparseCreate\n");
        return false;
    }
    int version = 0;
    cusparseGetVersion(handle, &version);
    if (version <= 0) {
        std::fprintf(stderr, "FAIL: cusparseGetVersion returned %d\n", version);
        return false;
    }
    if (cusparseDestroy(handle) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseDestroy\n");
        return false;
    }
    return true;
}

static bool test_mat_descr() {
    cusparseMatDescr_t descr = nullptr;
    if (cusparseCreateMatDescr(&descr) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseCreateMatDescr\n");
        return false;
    }
    if (cusparseGetMatType(descr) != CUSPARSE_MATRIX_TYPE_GENERAL) {
        std::fprintf(stderr, "FAIL: default mat type should be GENERAL\n");
        return false;
    }
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    if (cusparseGetMatType(descr) != CUSPARSE_MATRIX_TYPE_SYMMETRIC) {
        std::fprintf(stderr, "FAIL: cusparseSetMatType\n");
        return false;
    }
    if (cusparseGetMatIndexBase(descr) != CUSPARSE_INDEX_BASE_ZERO) {
        std::fprintf(stderr, "FAIL: default index base should be ZERO\n");
        return false;
    }
    cusparseDestroyMatDescr(descr);
    return true;
}

static bool test_generic_spmv() {
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    // 3×3 CSR matrix
    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // x = [1, 1, 1]
    float x[] = {1.0f, 1.0f, 1.0f};
    // y = [0, 0, 0]
    float y[] = {0.0f, 0.0f, 0.0f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseCreateCsr(&matA, 3, 3, 5,
                      rowPtr, colInd, vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_32F);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    if (cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufSize) !=
            CUSPARSE_STATUS_SUCCESS ||
        bufSize == 0) {
        std::fprintf(stderr, "FAIL: SpMV bufferSize should report usable workspace\n");
        return false;
    }
    if (cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr) !=
        CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: SpMV bufferSize should reject null output\n");
        return false;
    }

    cusparseStatus_t st = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, vecX, &beta, vecY,
                                        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseSpMV returned %d\n", st);
        return false;
    }

    // Expected: y = A*x = [1+2, 3, 4+5] = [3, 3, 9]
    if (std::fabs(y[0] - 3.0f) > 1e-5f || std::fabs(y[1] - 3.0f) > 1e-5f ||
        std::fabs(y[2] - 9.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: SpMV result [%f, %f, %f] != [3, 3, 9]\n", y[0], y[1], y[2]);
        return false;
    }

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    return true;
}

static bool test_legacy_scsrmv() {
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    cusparseMatDescr_t descr = nullptr;
    cusparseCreateMatDescr(&descr);

    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x[] = {2.0f, 3.0f, 4.0f};
    float y[] = {1.0f, 1.0f, 1.0f};

    float alpha = 1.0f, beta = 1.0f;
    cusparseStatus_t st = cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                          3, 3, 5, &alpha, descr,
                                          vals, rowPtr, colInd, x, &beta, y);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseScsrmv returned %d\n", st);
        return false;
    }

    // y = 1*A*x + 1*y_old
    // A*x = [1*2+2*4, 3*3, 4*2+5*4] = [10, 9, 28]
    // y = [10+1, 9+1, 28+1] = [11, 10, 29]
    if (std::fabs(y[0] - 11.0f) > 1e-5f || std::fabs(y[1] - 10.0f) > 1e-5f ||
        std::fabs(y[2] - 29.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: Scsrmv result [%f, %f, %f] != [11, 10, 29]\n", y[0], y[1], y[2]);
        return false;
    }

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    return true;
}

static bool test_generic_spmm() {
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);

    // Same 3×3 CSR
    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    // B = 3×2 dense column-major: identity-like
    // col0: [1,0,0], col1: [0,1,0]
    float B[] = {1.0f, 0.0f, 0.0f,  0.0f, 1.0f, 0.0f};
    float C[] = {0.0f, 0.0f, 0.0f,  0.0f, 0.0f, 0.0f};

    cusparseSpMatDescr_t matA = nullptr;
    cusparseCreateCsr(&matA, 3, 3, 5, rowPtr, colInd, vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t matB = nullptr, matC = nullptr;
    cusparseCreateDnMat(&matB, 3, 2, 3, B, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&matC, 3, 2, 3, C, CUDA_R_32F, CUSPARSE_ORDER_COL);

    float alpha = 1.0f, beta = 0.0f;
    size_t bufSize = 0;
    cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, matA, matB, &beta, matC,
                            CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &bufSize);

    cusparseStatus_t st = cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha, matA, matB, &beta, matC,
                                        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, nullptr);
    if (st != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseSpMM returned %d\n", st);
        return false;
    }

    // C = A * B (col-major, ld=3)
    // col0 of C = A * [1,0,0]^T = col0 of A = [1, 0, 4]
    // col1 of C = A * [0,1,0]^T = col1 of A = [0, 3, 0]
    if (std::fabs(C[0] - 1.0f) > 1e-5f || std::fabs(C[1] - 0.0f) > 1e-5f ||
        std::fabs(C[2] - 4.0f) > 1e-5f ||
        std::fabs(C[3] - 0.0f) > 1e-5f || std::fabs(C[4] - 3.0f) > 1e-5f ||
        std::fabs(C[5] - 0.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: SpMM result incorrect\n");
        return false;
    }

    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    return true;
}

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_mat_descr()) return 1;
    if (!test_generic_spmv()) return 1;
    if (!test_legacy_scsrmv()) return 1;
    if (!test_generic_spmm()) return 1;

    std::printf("PASS: cuSPARSE API tests\n");
    return 0;
}
