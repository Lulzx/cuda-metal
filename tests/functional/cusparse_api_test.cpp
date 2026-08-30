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
    cusparsePointerMode_t pointer_mode = CUSPARSE_POINTER_MODE_DEVICE;
    if (cusparseGetPointerMode(handle, &pointer_mode) != CUSPARSE_STATUS_SUCCESS ||
        pointer_mode != CUSPARSE_POINTER_MODE_HOST ||
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseGetPointerMode(handle, &pointer_mode) != CUSPARSE_STATUS_SUCCESS ||
        pointer_mode != CUSPARSE_POINTER_MODE_HOST ||
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE) !=
            CUSPARSE_STATUS_SUCCESS ||
        cusparseGetPointerMode(handle, &pointer_mode) != CUSPARSE_STATUS_SUCCESS ||
        pointer_mode != CUSPARSE_POINTER_MODE_DEVICE ||
        cusparseSetPointerMode(handle, static_cast<cusparsePointerMode_t>(-1)) !=
            CUSPARSE_STATUS_INVALID_VALUE ||
        cusparseGetPointerMode(handle, nullptr) != CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuSPARSE pointer mode contract\n");
        return false;
    }
    if (cusparseDestroy(handle) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseDestroy\n");
        return false;
    }
    return true;
}

static bool test_device_pointer_mode_spmv() {
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    float* device_alpha = nullptr;
    float* device_beta = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS ||
        cudaMalloc(reinterpret_cast<void**>(&device_alpha), sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_beta), sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: device pointer-mode setup\n");
        return false;
    }

    int rowPtr[] = {0, 2, 3, 5};
    int colInd[] = {0, 2, 1, 0, 2};
    float vals[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float x[] = {1.0f, 1.0f, 1.0f};
    float y[] = {1.0f, 1.0f, 1.0f};
    const float alpha = 2.0f;
    const float beta = 3.0f;
    cudaMemcpy(device_alpha, &alpha, sizeof(alpha), cudaMemcpyHostToDevice);
    cudaMemcpy(device_beta, &beta, sizeof(beta), cudaMemcpyHostToDevice);

    cusparseCreateCsr(&matA, 3, 3, 5, rowPtr, colInd, vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&vecX, 3, x, CUDA_R_32F);
    cusparseCreateDnVec(&vecY, 3, y, CUDA_R_32F);
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);

    size_t buffer_size = 0;
    const bool ok =
        cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                device_alpha, matA, vecX, device_beta, vecY,
                                CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                &buffer_size) == CUSPARSE_STATUS_SUCCESS &&
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, device_alpha,
                     matA, vecX, device_beta, vecY, CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT, nullptr) ==
            CUSPARSE_STATUS_SUCCESS &&
        std::fabs(y[0] - 9.0f) < 1e-5f &&
        std::fabs(y[1] - 9.0f) < 1e-5f &&
        std::fabs(y[2] - 21.0f) < 1e-5f &&
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                     matA, vecX, device_beta, vecY, CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT, nullptr) ==
            CUSPARSE_STATUS_INVALID_VALUE;

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    const bool host_rejects_device =
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, device_alpha,
                     matA, vecX, &beta, vecY, CUDA_R_32F,
                     CUSPARSE_SPMV_ALG_DEFAULT, nullptr) ==
        CUSPARSE_STATUS_INVALID_VALUE;

    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroySpMat(matA);
    cudaFree(device_alpha);
    cudaFree(device_beta);
    cusparseDestroy(handle);
    if (!ok || !host_rejects_device) {
        std::fprintf(stderr, "FAIL: cuSPARSE device scalar pointer-mode contract\n");
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
    if (cusparseSetMatType(descr, static_cast<cusparseMatrixType_t>(-1)) !=
            CUSPARSE_STATUS_INVALID_VALUE ||
        cusparseSetMatIndexBase(descr, static_cast<cusparseIndexBase_t>(-1)) !=
            CUSPARSE_STATUS_INVALID_VALUE ||
        cusparseSetMatFillMode(descr, static_cast<cusparseFillMode_t>(-1)) !=
            CUSPARSE_STATUS_INVALID_VALUE ||
        cusparseSetMatDiagType(descr, static_cast<cusparseDiagType_t>(-1)) !=
            CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: legacy descriptor enum validation\n");
        return false;
    }
    if (cusparseGetMatIndexBase(descr) != CUSPARSE_INDEX_BASE_ZERO) {
        std::fprintf(stderr, "FAIL: default index base should be ZERO\n");
        return false;
    }
    cusparseDestroyMatDescr(descr);
    return true;
}

static bool test_generic_descriptor_validation() {
    int offsets[] = {0, 0};
    float value = 1.0f;
    cusparseSpMatDescr_t sparse = reinterpret_cast<cusparseSpMatDescr_t>(1);
    cusparseDnVecDescr_t vector = reinterpret_cast<cusparseDnVecDescr_t>(1);
    cusparseDnMatDescr_t matrix = reinterpret_cast<cusparseDnMatDescr_t>(1);
    if (cusparseCreateCsr(&sparse, -1, 1, 0, offsets, nullptr, nullptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) !=
            CUSPARSE_STATUS_INVALID_VALUE || sparse != nullptr ||
        cusparseCreateCoo(&sparse, 1, 1, 1, nullptr, nullptr, nullptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_32F) != CUSPARSE_STATUS_INVALID_VALUE ||
        sparse != nullptr ||
        cusparseCreateCsc(&sparse, 1, 1, 0, nullptr, nullptr, nullptr,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) !=
            CUSPARSE_STATUS_INVALID_VALUE || sparse != nullptr ||
        cusparseCreateDnVec(&vector, -1, &value, CUDA_R_32F) !=
            CUSPARSE_STATUS_INVALID_VALUE || vector != nullptr ||
        cusparseCreateDnMat(&matrix, 2, 3, 1, &value, CUDA_R_32F,
                            CUSPARSE_ORDER_COL) != CUSPARSE_STATUS_INVALID_VALUE ||
        matrix != nullptr) {
        std::fprintf(stderr, "FAIL: generic descriptor shape/pointer validation\n");
        return false;
    }

    int col[] = {0};
    float values[] = {1.0f};
    float x_value = 2.0f, y_value = 0.0f;
    cusparseDnVecDescr_t x = nullptr, y = nullptr;
    cusparseHandle_t handle = nullptr;
    cusparseCreate(&handle);
    cusparseCreateCsr(&sparse, 1, 1, 1, offsets, col, values,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
    cusparseCreateDnVec(&x, 1, &x_value, CUDA_R_64F);
    cusparseCreateDnVec(&y, 1, &y_value, CUDA_R_32F);
    const float alpha = 1.0f, beta = 0.0f;
    const bool rejected =
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, sparse,
                     x, &beta, y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                     nullptr) == CUSPARSE_STATUS_NOT_SUPPORTED;
    cusparseDestroyDnVec(x);
    cusparseDestroyDnVec(y);
    cusparseDestroySpMat(sparse);
    cusparseDestroy(handle);
    if (!rejected) {
        std::fprintf(stderr, "FAIL: mixed unsupported SpMV types were accepted\n");
        return false;
    }
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
    if (cusparseSpMV_bufferSize(handle, static_cast<cusparseOperation_t>(-1),
                               &alpha, matA, vecX, &beta, vecY,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                               &bufSize) != CUSPARSE_STATUS_INVALID_VALUE ||
        cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA,
                     vecX, &beta, vecY, CUDA_R_32F,
                     static_cast<cusparseSpMVAlg_t>(-1), nullptr) !=
            CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: SpMV operation/algorithm validation\n");
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

    float *device_alpha = nullptr, *device_beta = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_alpha), sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&device_beta), sizeof(float));
    cudaMemcpy(device_alpha, &alpha, sizeof(alpha), cudaMemcpyHostToDevice);
    cudaMemcpy(device_beta, &beta, sizeof(beta), cudaMemcpyHostToDevice);
    y[0] = y[1] = y[2] = 1.0f;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    if (cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                       3, 3, 5, device_alpha, descr, vals, rowPtr, colInd, x,
                       device_beta, y) != CUSPARSE_STATUS_SUCCESS ||
        std::fabs(y[0] - 11.0f) > 1e-5f ||
        std::fabs(y[1] - 10.0f) > 1e-5f ||
        std::fabs(y[2] - 29.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: legacy CSR device scalar mode\n");
        return false;
    }
    cudaFree(device_alpha);
    cudaFree(device_beta);

    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
    return true;
}

static bool test_legacy_csrmv_transpose_and_validation() {
    cusparseHandle_t handle = nullptr;
    cusparseMatDescr_t descr = nullptr;
    cusparseCreate(&handle);
    cusparseCreateMatDescr(&descr);

    int row_ptr[] = {0, 2, 4};
    int col_ind[] = {0, 2, 1, 2};
    float values_f[] = {1.0f, 2.0f, 3.0f, 4.0f};
    double values_d[] = {1.0, 2.0, 3.0, 4.0};
    float x_f[] = {5.0f, 7.0f};
    double x_d[] = {5.0, 7.0};
    float y_f[] = {1.0f, 2.0f, 3.0f};
    double y_d[] = {1.0, 2.0, 3.0};
    const float alpha_f = 2.0f, beta_f = 1.0f;
    const double alpha_d = 2.0, beta_d = 1.0;

    if (cusparseScsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, 2, 3, 4,
                       &alpha_f, descr, values_f, row_ptr, col_ind, x_f,
                       &beta_f, y_f) != CUSPARSE_STATUS_SUCCESS ||
        std::fabs(y_f[0] - 11.0f) > 1e-5f ||
        std::fabs(y_f[1] - 44.0f) > 1e-5f ||
        std::fabs(y_f[2] - 79.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: legacy float CSR transpose result\n");
        return false;
    }
    if (cusparseDcsrmv(handle, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                       2, 3, 4, &alpha_d, descr, values_d, row_ptr, col_ind,
                       x_d, &beta_d, y_d) != CUSPARSE_STATUS_SUCCESS ||
        std::fabs(y_d[0] - 11.0) > 1e-12 ||
        std::fabs(y_d[1] - 44.0) > 1e-12 ||
        std::fabs(y_d[2] - 79.0) > 1e-12) {
        std::fprintf(stderr, "FAIL: legacy double CSR conjugate-transpose result\n");
        return false;
    }

    const float unchanged[] = {9.0f, 8.0f, 7.0f};
    std::memcpy(y_f, unchanged, sizeof(y_f));
    int bad_col_ind[] = {0, 3, 1, 2};
    if (cusparseScsrmv(handle, CUSPARSE_OPERATION_TRANSPOSE, 2, 3, 4,
                       &alpha_f, descr, values_f, row_ptr, bad_col_ind, x_f,
                       &beta_f, y_f) != CUSPARSE_STATUS_INVALID_VALUE ||
        std::memcmp(y_f, unchanged, sizeof(y_f)) != 0 ||
        cusparseScsrmv(handle, static_cast<cusparseOperation_t>(-1),
                       2, 3, 4, &alpha_f, descr, values_f, row_ptr, col_ind,
                       x_f, &beta_f, y_f) != CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: legacy CSR transpose validation contract\n");
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
    if (cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                               &bufSize) != CUSPARSE_STATUS_SUCCESS ||
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                               &bufSize) != CUSPARSE_STATUS_NOT_SUPPORTED ||
        cusparseSpMM_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F,
                               static_cast<cusparseSpMMAlg_t>(-1),
                               &bufSize) != CUSPARSE_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: SpMM buffer-size validation contract\n");
        return false;
    }

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

    float *device_alpha = nullptr, *device_beta = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device_alpha), sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&device_beta), sizeof(float));
    cudaMemcpy(device_alpha, &alpha, sizeof(alpha), cudaMemcpyHostToDevice);
    cudaMemcpy(device_beta, &beta, sizeof(beta), cudaMemcpyHostToDevice);
    std::memset(C, 0, sizeof(C));
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_DEVICE);
    if (cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     CUSPARSE_OPERATION_NON_TRANSPOSE, device_alpha, matA,
                     matB, device_beta, matC, CUDA_R_32F,
                     CUSPARSE_SPMM_ALG_DEFAULT, nullptr) !=
            CUSPARSE_STATUS_SUCCESS ||
        std::fabs(C[0] - 1.0f) > 1e-5f ||
        std::fabs(C[2] - 4.0f) > 1e-5f ||
        std::fabs(C[4] - 3.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: SpMM device scalar mode\n");
        return false;
    }
    cudaFree(device_alpha);
    cudaFree(device_beta);

    cusparseDestroyDnMat(matB);
    cusparseDestroyDnMat(matC);
    cusparseDestroySpMat(matA);
    cusparseDestroy(handle);
    return true;
}

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_mat_descr()) return 1;
    if (!test_generic_descriptor_validation()) return 1;
    if (!test_generic_spmv()) return 1;
    if (!test_device_pointer_mode_spmv()) return 1;
    if (!test_legacy_scsrmv()) return 1;
    if (!test_legacy_csrmv_transpose_and_validation()) return 1;
    if (!test_generic_spmm()) return 1;

    std::printf("PASS: cuSPARSE API tests\n");
    return 0;
}
