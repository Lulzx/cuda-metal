#pragma once
#include "library_types.h"
#include "cuComplex.h"
// Real cusparse.h declares every entry point with CUSPARSEAPI (empty off Windows).
// helper_cuda.h keys its cuSPARSE error strings off `#ifdef CUSPARSEAPI`.
#ifndef CUSPARSEAPI
#define CUSPARSEAPI
#endif
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUSPARSE_H_
#define CUSPARSE_H_ 1
#endif

// cuSPARSE 12.0.0 (CUDA 12.0). cusparseGetVersion reports CUSPARSE_VERSION, so
// the header macro and the runtime answer cannot drift apart.
#define CUSPARSE_VER_MAJOR 12
#define CUSPARSE_VER_MINOR 0
#define CUSPARSE_VER_PATCH 0
#define CUSPARSE_VER_BUILD 76
#define CUSPARSE_VERSION \
    (CUSPARSE_VER_MAJOR * 1000 + CUSPARSE_VER_MINOR * 100 + CUSPARSE_VER_PATCH)

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
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10,
    CUSPARSE_STATUS_INSUFFICIENT_RESOURCES = 11,
} cusparseStatus_t;

typedef struct cusparseContext* cusparseHandle_t;
typedef struct cusparseMatDescr* cusparseMatDescr_t;
typedef struct cusparseSpMatDescr* cusparseSpMatDescr_t;
typedef struct cusparseDnVecDescr* cusparseDnVecDescr_t;
typedef struct cusparseDnMatDescr* cusparseDnMatDescr_t;
typedef struct cusparseSpVecDescr* cusparseSpVecDescr_t;
typedef struct csrilu02Info* csrilu02Info_t;
typedef struct csric02Info* csric02Info_t;
typedef struct bsrilu02Info* bsrilu02Info_t;
typedef struct bsric02Info* bsric02Info_t;
typedef struct cusparseSpSMDescr* cusparseSpSMDescr_t;
typedef struct cusparseSpGEMMDescr* cusparseSpGEMMDescr_t;

#ifndef CUMETAL_CUDA_STREAM_T_DEFINED
#define CUMETAL_CUDA_STREAM_T_DEFINED 1
typedef struct CUstream_st* cudaStream_t;
#endif  // CUMETAL_CUDA_STREAM_T_DEFINED

typedef enum cusparseOperation_t {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2,
} cusparseOperation_t;

typedef enum cusparsePointerMode_t {
    CUSPARSE_POINTER_MODE_HOST = 0,
    CUSPARSE_POINTER_MODE_DEVICE = 1,
} cusparsePointerMode_t;

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

typedef enum cusparseSpMatAttribute_t {
    CUSPARSE_SPMAT_FILL_MODE = 0,
    CUSPARSE_SPMAT_DIAG_TYPE = 1,
} cusparseSpMatAttribute_t;

typedef enum cusparseSolvePolicy_t {
    CUSPARSE_SOLVE_POLICY_NO_LEVEL = 0,
    CUSPARSE_SOLVE_POLICY_USE_LEVEL = 1,
} cusparseSolvePolicy_t;

typedef enum cusparseOrder_t {
    CUSPARSE_ORDER_COL = 1,
    CUSPARSE_ORDER_ROW = 2,
} cusparseOrder_t;

typedef enum cusparseAction_t {
    CUSPARSE_ACTION_SYMBOLIC = 0,
    CUSPARSE_ACTION_NUMERIC = 1,
} cusparseAction_t;

typedef enum cusparseDirection_t {
    CUSPARSE_DIRECTION_ROW = 0,
    CUSPARSE_DIRECTION_COLUMN = 1,
} cusparseDirection_t;

typedef enum cusparseFormat_t {
    CUSPARSE_FORMAT_CSR = 1,
    CUSPARSE_FORMAT_CSC = 2,
    CUSPARSE_FORMAT_COO = 3,
    CUSPARSE_FORMAT_BLOCKED_ELL = 5,
    CUSPARSE_FORMAT_BSR = 6,
    CUSPARSE_FORMAT_SLICED_ELLPACK = 7,
} cusparseFormat_t;

typedef enum cusparseCsr2CscAlg_t {
    CUSPARSE_CSR2CSC_ALG_DEFAULT = 1,
    CUSPARSE_CSR2CSC_ALG1 = 1,
    CUSPARSE_CSR2CSC_ALG2 = 2,
} cusparseCsr2CscAlg_t;

typedef enum cusparseSpSMAlg_t {
    CUSPARSE_SPSM_ALG_DEFAULT = 0,
} cusparseSpSMAlg_t;

typedef enum cusparseSpGEMMAlg_t {
    CUSPARSE_SPGEMM_DEFAULT = 0,
    CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC = 1,
    CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC = 2,
    CUSPARSE_SPGEMM_ALG1 = 3,
    CUSPARSE_SPGEMM_ALG2 = 4,
    CUSPARSE_SPGEMM_ALG3 = 5,
} cusparseSpGEMMAlg_t;

typedef enum cusparseSparseToDenseAlg_t {
    CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
} cusparseSparseToDenseAlg_t;

typedef enum cusparseDenseToSparseAlg_t {
    CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0,
} cusparseDenseToSparseAlg_t;

typedef enum cusparseSpMVAlg_t {
    CUSPARSE_SPMV_ALG_DEFAULT = 0,
    CUSPARSE_SPMV_COO_ALG1 = 1,
    CUSPARSE_SPMV_CSR_ALG1 = 2,
    CUSPARSE_SPMV_CSR_ALG2 = 3,
    CUSPARSE_SPMV_COO_ALG2 = 4,
} cusparseSpMVAlg_t;

typedef enum cusparseSpMMAlg_t {
    CUSPARSE_SPMM_ALG_DEFAULT = 0,
    CUSPARSE_SPMM_COO_ALG1 = 1,
    CUSPARSE_SPMM_COO_ALG2 = 2,
    CUSPARSE_SPMM_COO_ALG3 = 3,
    CUSPARSE_SPMM_CSR_ALG1 = 4,
    CUSPARSE_SPMM_COO_ALG4 = 5,
    CUSPARSE_SPMM_CSR_ALG2 = 6,
    CUSPARSE_SPMM_CSR_ALG3 = 12,
} cusparseSpMMAlg_t;


// Handle management
cusparseStatus_t cusparseCreate(cusparseHandle_t* handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);
cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode);
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version);
const char* cusparseGetErrorName(cusparseStatus_t status);
const char* cusparseGetErrorString(cusparseStatus_t status);

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
cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* cscColOffsets, void* cscRowInd,
                                    void* cscValues,
                                    cusparseIndexType_t cscColOffsetsType,
                                    cusparseIndexType_t cscRowIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType);

cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* cooRowInd, void* cooColInd, void* cooValues,
                                    cusparseIndexType_t cooIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType);
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);
cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           const void* data, size_t dataSize);

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t* info);
cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info);
cusparseStatus_t cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                              const cusparseMatDescr_t descrA,
                                              float* csrValA, const int* csrRowPtrA,
                                              const int* csrColIndA, csrilu02Info_t info,
                                              int* bufferSize);
cusparseStatus_t cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                            const cusparseMatDescr_t descrA,
                                            const float* csrValA, const int* csrRowPtrA,
                                            const int* csrColIndA, csrilu02Info_t info,
                                            cusparseSolvePolicy_t policy, void* buffer);
cusparseStatus_t cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseMatDescr_t descrA,
                                   float* csrValA, const int* csrRowPtrA,
                                   const int* csrColIndA, csrilu02Info_t info,
                                   cusparseSolvePolicy_t policy, void* buffer);

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                      int64_t size, void* values, cudaDataType valueType);
cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);
cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values);
cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values);
cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                   int64_t* size, void** values, cudaDataType* valueType);

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
// Analysis hook. cuSPARSE lets a caller pay a one-time cost here so that the
// SpMV calls that follow are cheaper; it is optional, and skipping it changes
// speed, not results.
cusparseStatus_t cusparseSpMV_preprocess(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void* beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpMVAlg_t alg,
                                          void* externalBuffer);
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


// ── Sparse vector descriptors ───────────────────────────────────────────────
cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size,
                                     int64_t nnz, void* indices, void* values,
                                     cusparseIndexType_t idxType, cusparseIndexBase_t idxBase,
                                     cudaDataType valueType);
cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr);
cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz,
                                  void** indices, void** values, cusparseIndexType_t* idxType,
                                  cusparseIndexBase_t* idxBase, cudaDataType* valueType);
cusparseStatus_t cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                                           cusparseIndexBase_t* idxBase);
cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values);
cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values);

// ── Sparse / dense matrix descriptor accessors ──────────────────────────────
cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                int64_t* nnz, void** cooRowInd, void** cooColInd,
                                void** cooValues, cusparseIndexType_t* idxType,
                                cusparseIndexBase_t* idxBase, cudaDataType* valueType);
cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                int64_t* nnz, void** csrRowOffsets, void** csrColInd,
                                void** csrValues, cusparseIndexType_t* csrRowOffsetsType,
                                cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase,
                                cudaDataType* valueType);
cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets,
                                        void* csrColInd, void* csrValues);
cusparseStatus_t cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr, cusparseFormat_t* format);
cusparseStatus_t cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                                           cusparseIndexBase_t* idxBase);
cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values);
cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values);
cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t* rows,
                                      int64_t* cols, int64_t* nnz);
cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr, int* batchCount);

cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols,
                                  int64_t* ld, void** values, cudaDataType* valueType,
                                  cusparseOrder_t* order);
cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values);
cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values);
cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int* batchCount,
                                              int64_t* batchStride);
cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount,
                                              int64_t batchStride);

// ── Generic API: SpVV, Gather, SpSM, SpGEMM, dense<->sparse ─────────────────
cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX,
                                         cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                         const void* result, cudaDataType computeType,
                                         size_t* bufferSize);
cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
                              cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                              void* result, cudaDataType computeType, void* externalBuffer);
cusparseStatus_t cusparseGather(cusparseHandle_t handle, cusparseDnVecDescr_t vecY,
                                cusparseSpVecDescr_t vecX);

cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr);
cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr);
cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         cusparseOperation_t opB, const void* alpha,
                                         cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                         cusparseDnMatDescr_t matC, cudaDataType computeType,
                                         cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                         size_t* bufferSize);
cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                                       cusparseOperation_t opB, const void* alpha,
                                       cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                       cusparseDnMatDescr_t matC, cudaDataType computeType,
                                       cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                       void* externalBuffer);
cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                                    cusparseOperation_t opB, const void* alpha,
                                    cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC, cudaDataType computeType,
                                    cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr);

cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr);
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr);
cusparseStatus_t cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1);
cusparseStatus_t cusparseSpGEMM_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2);
cusparseStatus_t cusparseSpGEMM_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr);

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle,
                                                  cusparseSpMatDescr_t matA,
                                                  cusparseDnMatDescr_t matB,
                                                  cusparseSparseToDenseAlg_t alg,
                                                  size_t* bufferSize);
cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg,
                                       void* externalBuffer);
cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle,
                                                  cusparseDnMatDescr_t matA,
                                                  cusparseSpMatDescr_t matB,
                                                  cusparseDenseToSparseAlg_t alg,
                                                  size_t* bufferSize);
cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle,
                                                cusparseDnMatDescr_t matA,
                                                cusparseSpMatDescr_t matB,
                                                cusparseDenseToSparseAlg_t alg,
                                                void* externalBuffer);
cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle,
                                               cusparseDnMatDescr_t matA,
                                               cusparseSpMatDescr_t matB,
                                               cusparseDenseToSparseAlg_t alg,
                                               void* externalBuffer);

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz,
                                               const void* csrVal, const int* csrRowPtr,
                                               const int* csrColInd, void* cscVal, int* cscColPtr,
                                               int* cscRowInd, cudaDataType valType,
                                               cusparseAction_t copyValues,
                                               cusparseIndexBase_t idxBase,
                                               cusparseCsr2CscAlg_t alg, size_t* bufferSize);
cusparseStatus_t cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz,
                                    const void* csrVal, const int* csrRowPtr,
                                    const int* csrColInd, void* cscVal, int* cscColPtr,
                                    int* cscRowInd, cudaDataType valType,
                                    cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                    cusparseCsr2CscAlg_t alg, void* buffer);

// ── Format conversion, sorting, nnz counting ────────────────────────────────
cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m,
                                  int* csrSortedRowPtr, cusparseIndexBase_t idxBase);
cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz,
                                  int m, int* cooRowInd, cusparseIndexBase_t idxBase);
cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p);
cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int* cooRows, const int* cooCols,
                                                size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                                       int* cooRows, int* cooCols, int* P, void* pBuffer);
cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz,
                                          int* cooRows, int* cooCols, int* P, void* pBuffer);
cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int* csrRowPtr, const int* csrColInd,
                                                size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz,
                                  const cusparseMatDescr_t descrA, const int* csrRowPtr,
                                  int* csrColInd, int* P, void* pBuffer);
cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int* cscColPtr, const int* cscRowInd,
                                                size_t* pBufferSizeInBytes);
cusparseStatus_t cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz,
                                  const cusparseMatDescr_t descrA, const int* cscColPtr,
                                  int* cscRowInd, int* P, void* pBuffer);
cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const int* csrRowPtrA, const int* csrColIndA,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const int* csrRowPtrB, const int* csrColIndB,
                                      const cusparseMatDescr_t descrC, int* csrRowPtrC,
                                      int* nnzTotalDevHostPtr, void* workspace);

// ── Incomplete factorization info objects ───────────────────────────────────
cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t* info);
cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info);
cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t* info);
cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info);
cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t* info);
cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info);
cusparseStatus_t cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info,
                                             int* position);
cusparseStatus_t cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info,
                                            int* position);
// Block (BSR) incomplete factorizations are declared for source compatibility
// but every compute entry point returns CUSPARSE_STATUS_NOT_SUPPORTED; see
// docs/known-gaps.md.
cusparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t info,
                                             int* position);
cusparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t info,
                                            int* position);

// ── Per-precision entry points (S = float, D = double, C = cuComplex,
// Z = cuDoubleComplex). Declared through one macro so the four variants cannot
// drift apart. ──────────────────────────────────────────────────────────────
#define CUMETAL_CUSPARSE_FOR_EACH_TYPE(X) \
    X(S, float) X(D, double) X(C, cuComplex) X(Z, cuDoubleComplex)

#define CUMETAL_CUSPARSE_DECLARE_TYPED(P, T)                                                     \
    cusparseStatus_t cusparse##P##nnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,   \
                                      int n, const cusparseMatDescr_t descrA, const T* A,        \
                                      int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr);   \
    cusparseStatus_t cusparse##P##nnz_compress(                                                  \
        cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const T* csrSortedValA,  \
        const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, T tol);                          \
    cusparseStatus_t cusparse##P##csr2csr_compress(                                              \
        cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,                  \
        const T* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA,        \
        int nnzA, int* nnzPerRow, T* csrSortedValC, int* csrSortedColIndC,                       \
        int* csrSortedRowPtrC, T tol);                                                           \
    cusparseStatus_t cusparse##P##csrgeam2_bufferSizeExt(                                        \
        cusparseHandle_t handle, int m, int n, const T* alpha, const cusparseMatDescr_t descrA,  \
        int nnzA, const T* csrSortedValA, const int* csrSortedRowPtrA,                           \
        const int* csrSortedColIndA, const T* beta, const cusparseMatDescr_t descrB, int nnzB,   \
        const T* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,        \
        const cusparseMatDescr_t descrC, T* csrSortedValC, int* csrSortedRowPtrC,                \
        int* csrSortedColIndC, size_t* pBufferSizeInBytes);                                      \
    cusparseStatus_t cusparse##P##csrgeam2(                                                      \
        cusparseHandle_t handle, int m, int n, const T* alpha, const cusparseMatDescr_t descrA,  \
        int nnzA, const T* csrSortedValA, const int* csrSortedRowPtrA,                           \
        const int* csrSortedColIndA, const T* beta, const cusparseMatDescr_t descrB, int nnzB,   \
        const T* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,        \
        const cusparseMatDescr_t descrC, T* csrSortedValC, int* csrSortedRowPtrC,                \
        int* csrSortedColIndC, void* pBuffer);                                                   \
    cusparseStatus_t cusparse##P##csrilu02_numericBoost(cusparseHandle_t handle,                 \
                                                        csrilu02Info_t info, int enable_boost,   \
                                                        double* tol, T* boost_val);              \
    cusparseStatus_t cusparse##P##csrilu02_bufferSize(                                           \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        T* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,              \
        csrilu02Info_t info, int* pBufferSizeInBytes);                                           \
    cusparseStatus_t cusparse##P##csrilu02_analysis(                                             \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        const T* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,        \
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);                       \
    cusparseStatus_t cusparse##P##csrilu02(                                                      \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        T* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,         \
        csrilu02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);            \
    cusparseStatus_t cusparse##P##csric02_bufferSize(                                            \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        T* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,              \
        csric02Info_t info, int* pBufferSizeInBytes);                                            \
    cusparseStatus_t cusparse##P##csric02_analysis(                                              \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        const T* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,        \
        csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);                        \
    cusparseStatus_t cusparse##P##csric02(                                                       \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        T* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,         \
        csric02Info_t info, cusparseSolvePolicy_t policy, void* pBuffer);                        \
    cusparseStatus_t cusparse##P##bsrilu02_numericBoost(cusparseHandle_t handle,                 \
                                                        bsrilu02Info_t info, int enable_boost,   \
                                                        double* tol, T* boost_val);              \
    cusparseStatus_t cusparse##P##bsrilu02_bufferSize(                                           \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, T* bsrSortedVal, const int* bsrSortedRowPtr,            \
        const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info,                           \
        int* pBufferSizeInBytes);                                                                \
    cusparseStatus_t cusparse##P##bsrilu02_analysis(                                             \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, T* bsrSortedVal, const int* bsrSortedRowPtr,            \
        const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info,                           \
        cusparseSolvePolicy_t policy, void* pBuffer);                                            \
    cusparseStatus_t cusparse##P##bsrilu02(                                                      \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, T* bsrSortedVal, const int* bsrSortedRowPtr,            \
        const int* bsrSortedColInd, int blockDim, bsrilu02Info_t info,                           \
        cusparseSolvePolicy_t policy, void* pBuffer);                                            \
    cusparseStatus_t cusparse##P##bsric02_bufferSize(                                            \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, T* bsrSortedVal, const int* bsrSortedRowPtr,            \
        const int* bsrSortedColInd, int blockDim, bsric02Info_t info,                            \
        int* pBufferSizeInBytes);                                                                \
    cusparseStatus_t cusparse##P##bsric02_analysis(                                              \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, const T* bsrSortedVal, const int* bsrSortedRowPtr,      \
        const int* bsrSortedColInd, int blockDim, bsric02Info_t info,                            \
        cusparseSolvePolicy_t policy, void* pInputBuffer);                                       \
    cusparseStatus_t cusparse##P##bsric02(                                                       \
        cusparseHandle_t handle, cusparseDirection_t dirA, int mb, int nnzb,                     \
        const cusparseMatDescr_t descrA, T* bsrSortedVal, const int* bsrSortedRowPtr,            \
        const int* bsrSortedColInd, int blockDim, bsric02Info_t info,                            \
        cusparseSolvePolicy_t policy, void* pBuffer);                                            \
    cusparseStatus_t cusparse##P##gtsv2_bufferSizeExt(                                           \
        cusparseHandle_t handle, int m, int n, const T* dl, const T* d, const T* du,             \
        const T* B, int ldb, size_t* bufferSizeInBytes);                                         \
    cusparseStatus_t cusparse##P##gtsv2(cusparseHandle_t handle, int m, int n, const T* dl,      \
                                        const T* d, const T* du, T* B, int ldb, void* pBuffer);  \
    cusparseStatus_t cusparse##P##gtsv2_nopivot_bufferSizeExt(                                   \
        cusparseHandle_t handle, int m, int n, const T* dl, const T* d, const T* du,             \
        const T* B, int ldb, size_t* bufferSizeInBytes);                                         \
    cusparseStatus_t cusparse##P##gtsv2_nopivot(cusparseHandle_t handle, int m, int n,           \
                                                const T* dl, const T* d, const T* du, T* B,      \
                                                int ldb, void* pBuffer);                         \
    cusparseStatus_t cusparse##P##gtsv2StridedBatch_bufferSizeExt(                               \
        cusparseHandle_t handle, int m, const T* dl, const T* d, const T* du, const T* x,        \
        int batchCount, int batchStride, size_t* bufferSizeInBytes);                             \
    cusparseStatus_t cusparse##P##gtsv2StridedBatch(                                             \
        cusparseHandle_t handle, int m, const T* dl, const T* d, const T* du, T* x,              \
        int batchCount, int batchStride, void* pBuffer);                                         \
    cusparseStatus_t cusparse##P##gtsvInterleavedBatch_bufferSizeExt(                            \
        cusparseHandle_t handle, int algo, int m, const T* dl, const T* d, const T* du,          \
        const T* x, int batchCount, size_t* pBufferSizeInBytes);                                 \
    cusparseStatus_t cusparse##P##gtsvInterleavedBatch(                                          \
        cusparseHandle_t handle, int algo, int m, T* dl, T* d, T* du, T* x, int batchCount,      \
        void* pBuffer);                                                                          \
    cusparseStatus_t cusparse##P##gpsvInterleavedBatch_bufferSizeExt(                            \
        cusparseHandle_t handle, int algo, int m, const T* ds, const T* dl, const T* d,          \
        const T* du, const T* dw, const T* x, int batchCount, size_t* pBufferSizeInBytes);       \
    cusparseStatus_t cusparse##P##gpsvInterleavedBatch(                                          \
        cusparseHandle_t handle, int algo, int m, T* ds, T* dl, T* d, T* du, T* dw, T* x,        \
        int batchCount, void* pBuffer);

CUMETAL_CUSPARSE_FOR_EACH_TYPE(CUMETAL_CUSPARSE_DECLARE_TYPED)
#undef CUMETAL_CUSPARSE_DECLARE_TYPED

#ifdef __cplusplus
}
#endif
