#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum cusparseStatus_t {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
} cusparseStatus_t;

typedef struct cusparseContext* cusparseHandle_t;
typedef struct cusparseMatDescr* cusparseMatDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;

typedef struct CUstream_st* cudaStream_t;

typedef enum cusparseOperation_t {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2,
} cusparseOperation_t;

typedef enum cusparseIndexType_t {
    CUSPARSE_INDEX_16U = 1,
    CUSPARSE_INDEX_32I = 2,
    CUSPARSE_INDEX_64I = 3,
} cusparseIndexType_t;

typedef enum cusparseIndexBase_t {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE = 1,
} cusparseIndexBase_t;

typedef enum cusparseMatrixType_t {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3,
} cusparseMatrixType_t;

typedef enum cusparseDiagType_t {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT = 1,
} cusparseDiagType_t;

typedef enum cusparseFillMode_t {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1,
} cusparseFillMode_t;

typedef enum cusparseOrder_t {
    CUSPARSE_ORDER_COL = 1,
    CUSPARSE_ORDER_ROW = 2,
} cusparseOrder_t;

typedef enum cusparseSpMVAlg_t {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
} cusparseSpMVAlg_t;

typedef enum cusparseSpMMAlg_t {
    CUSPARSE_SPMM_ALG_DEFAULT = 0,
    CUSPARSE_SPMM_CSR_ALG1 = 2,
    CUSPARSE_SPMM_CSR_ALG2 = 3,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
} cusparseSpMMAlg_t;

typedef int cudaDataType;
#ifndef CUDA_R_32F
#define CUDA_R_32F 0
#define CUDA_R_64F 1
#define CUDA_R_16F 2
#define CUDA_R_32I 10
#define CUDA_R_8I  3
#define CUDA_C_32F 4
#define CUDA_C_64F 5
#endif

// Handle management
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
int cusparseGetVersion(cusparseHandle_t handle, int* version);

// Matrix descriptor
cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA);
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA);
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType);

// Generic sparse API
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* csrRowOffsets, void* csrColInd,
                                    void* csrValues,
                                    cusparseIndexType_t csrRowOffsetsType,
                                    cusparseIndexType_t csrColIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType);
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* cooRowInd, void* cooColInd, void* cooValues,
                                    cusparseIndexType_t cooIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType);
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                      int64_t size, void* values, cudaDataType valueType);
cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                      int64_t rows, int64_t cols, int64_t ld,
                                      void* values, cudaDataType valueType,
                                      cusparseOrder_t order);
cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);

// SpMV: y = alpha * op(A) * x + beta * y
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void* beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpMVAlg_t alg,
                                          size_t* bufferSize);
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle,
                               cusparseOperation_t opA,
                               const void* alpha,
                               cusparseSpMatDescr_t matA,
                               cusparseDnVecDescr_t vecX,
                               const void* beta,
                               cusparseDnVecDescr_t vecY,
                               cudaDataType computeType,
                               cusparseSpMVAlg_t alg,
                               void* externalBuffer);

// SpMM: C = alpha * op(A) * op(B) + beta * C
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          cusparseOperation_t opB,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnMatDescr_t matB,
                                          const void* beta,
                                          cusparseDnMatDescr_t matC,
                                          cudaDataType computeType,
                                          cusparseSpMMAlg_t alg,
                                          size_t* bufferSize);
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle,
                               cusparseOperation_t opA,
                               cusparseOperation_t opB,
                               const void* alpha,
                               cusparseSpMatDescr_t matA,
                               cusparseDnMatDescr_t matB,
                               const void* beta,
                               cusparseDnMatDescr_t matC,
                               cudaDataType computeType,
                               cusparseSpMMAlg_t alg,
                               void* externalBuffer);

// SpSV: Sparse triangular solve — op(A) * y = alpha * x
typedef struct cusparseSpSVDescr* cusparseSpSVDescr_t;

typedef enum cusparseSpSVAlg_t {
    CUSPARSE_SPSV_ALG_DEFAULT = 0,
} cusparseSpSVAlg_t;

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr);
cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr);

cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpSVAlg_t alg,
                                          cusparseSpSVDescr_t spsvDescr,
                                          size_t* bufferSize);

cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle,
                                        cusparseOperation_t opA,
                                        const void* alpha,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnVecDescr_t vecX,
                                        cusparseDnVecDescr_t vecY,
                                        cudaDataType computeType,
                                        cusparseSpSVAlg_t alg,
                                        cusparseSpSVDescr_t spsvDescr,
                                        void* externalBuffer);

cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const void* alpha,
                                     cusparseSpMatDescr_t matA,
                                     cusparseDnVecDescr_t vecX,
                                     cusparseDnVecDescr_t vecY,
                                     cudaDataType computeType,
                                     cusparseSpSVAlg_t alg,
                                     cusparseSpSVDescr_t spsvDescr);

// Legacy CSR SpMV
cusparseStatus_t cusparseScsrmv(cusparseHandle_t handle,
                                 cusparseOperation_t transA,
                                 int m, int n, int nnz,
                                 const float* alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float* csrValA,
                                 const int* csrRowPtrA,
                                 const int* csrColIndA,
                                 const float* x,
                                 const float* beta,
                                 float* y);
cusparseStatus_t cusparseDcsrmv(cusparseHandle_t handle,
                                 cusparseOperation_t transA,
                                 int m, int n, int nnz,
                                 const double* alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double* csrValA,
                                 const int* csrRowPtrA,
                                 const int* csrColIndA,
                                 const double* x,
                                 const double* beta,
                                 double* y);

#ifdef __cplusplus
}
#endif
