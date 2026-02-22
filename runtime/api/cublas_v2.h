#pragma once

#include <stddef.h>

#include "cuda_runtime.h"
#include "cuda_fp16.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cublasContext* cublasHandle_t;

typedef enum cublasStatus_t {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
} cublasStatus_t;

typedef enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
} cublasOperation_t;

typedef enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
} cublasFillMode_t;

typedef enum cublasMath_t {
    CUBLAS_DEFAULT_MATH = 0,
    CUBLAS_TENSOR_OP_MATH = 1,
    CUBLAS_PEDANTIC_MATH = 2,
    CUBLAS_TF32_TENSOR_OP_MATH = 3,
} cublasMath_t;

typedef enum cublasComputeType_t {
    CUBLAS_COMPUTE_16F = 64,
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_64F = 70,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 77,
} cublasComputeType_t;

typedef enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT     = 1,
} cublasDiagType_t;

typedef enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT  = 0,
    CUBLAS_SIDE_RIGHT = 1,
} cublasSideMode_t;

// cudaDataType_t — element types used by GemmEx and other extended APIs.
typedef enum cudaDataType_t {
    CUDA_R_16F  =  2,
    CUDA_C_16F  =  6,
    CUDA_R_32F  =  0,
    CUDA_C_32F  =  4,
    CUDA_R_64F  =  1,
    CUDA_C_64F  =  5,
    CUDA_R_8I   =  3,
    CUDA_R_8U   =  8,
    CUDA_R_32I  = 10,
} cudaDataType_t;
typedef cudaDataType_t cudaDataType;

typedef enum cublasGemmAlgo_t {
    CUBLAS_GEMM_DEFAULT            = -1,
    CUBLAS_GEMM_ALGO0              =  0,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP  = 99,
} cublasGemmAlgo_t;

cublasStatus_t cublasCreate(cublasHandle_t* handle);
cublasStatus_t cublasDestroy(cublasHandle_t handle);
cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version);
cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream_id);
cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id);
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode);

cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x,
                           int incx,
                           float* y,
                           int incy);
cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx);
cublasStatus_t cublasScopy(cublasHandle_t handle,
                           int n,
                           const float* x,
                           int incx,
                           float* y,
                           int incy);
cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy);
cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                           int n,
                           const double* alpha,
                           const double* x,
                           int incx,
                           double* y,
                           int incy);
cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx);
cublasStatus_t cublasDcopy(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* y,
                           int incy);
cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy);
cublasStatus_t cublasSdot(cublasHandle_t handle,
                          int n,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* result);
cublasStatus_t cublasDdot(cublasHandle_t handle,
                          int n,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* result);
cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result);
cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result);
cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double* x, int incx, double* result);
cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result);
cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double* x, int incx, int* result);
cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result);
cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double* x, int incx, int* result);

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* b,
                           int ldb,
                           const float* beta,
                           float* c,
                           int ldc);

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const float* alpha,
                                         const float* a,
                                         int lda,
                                         long long int stridea,
                                         const float* b,
                                         int ldb,
                                         long long int strideb,
                                         const float* beta,
                                         float* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count);

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* b,
                           int ldb,
                           const double* beta,
                           double* c,
                           int ldc);

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const double* alpha,
                                         const double* a,
                                         int lda,
                                         long long int stridea,
                                         const double* b,
                                         int ldb,
                                         long long int strideb,
                                         const double* beta,
                                         double* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count);

cublasStatus_t cublasSgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy);

cublasStatus_t cublasDgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy);

cublasStatus_t cublasSger(cublasHandle_t handle,
                          int m,
                          int n,
                          const float* alpha,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* a,
                          int lda);

cublasStatus_t cublasDger(cublasHandle_t handle,
                          int m,
                          int n,
                          const double* alpha,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* a,
                          int lda);

cublasStatus_t cublasSsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy);

cublasStatus_t cublasDsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy);

// GemmEx — extended GEMM with per-matrix data types and compute type.
cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m,
                            int n,
                            int k,
                            const void* alpha,
                            const void* a,
                            cudaDataType_t atype,
                            int lda,
                            const void* b,
                            cudaDataType_t btype,
                            int ldb,
                            const void* beta,
                            void* c,
                            cudaDataType_t ctype,
                            int ldc,
                            cublasComputeType_t compute_type,
                            cublasGemmAlgo_t algo);

// GemmStridedBatchedEx — batched strided GemmEx.
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m,
                                          int n,
                                          int k,
                                          const void* alpha,
                                          const void* a,
                                          cudaDataType_t atype,
                                          int lda,
                                          long long int stridea,
                                          const void* b,
                                          cudaDataType_t btype,
                                          int ldb,
                                          long long int strideb,
                                          const void* beta,
                                          void* c,
                                          cudaDataType_t ctype,
                                          int ldc,
                                          long long int stridec,
                                          int batch_count,
                                          cublasComputeType_t compute_type,
                                          cublasGemmAlgo_t algo);

// Hgemm — half-precision GEMM.
cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const __half* alpha,
                           const __half* a,
                           int lda,
                           const __half* b,
                           int ldb,
                           const __half* beta,
                           __half* c,
                           int ldc);

// SgemmBatched / DgemmBatched — batched GEMM with array-of-pointers interface.
cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const float* alpha,
                                  const float* const a_array[],
                                  int lda,
                                  const float* const b_array[],
                                  int ldb,
                                  const float* beta,
                                  float* const c_array[],
                                  int ldc,
                                  int batch_count);

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const double* alpha,
                                  const double* const a_array[],
                                  int lda,
                                  const double* const b_array[],
                                  int ldb,
                                  const double* beta,
                                  double* const c_array[],
                                  int ldc,
                                  int batch_count);

// Strsm / Dtrsm — triangular solve with multiple right-hand sides.
cublasStatus_t cublasStrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           float* b,
                           int ldb);

cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           double* b,
                           int ldb);

// SetVector / GetVector — transfer a strided vector between host and device.
cublasStatus_t cublasSetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy);

cublasStatus_t cublasGetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy);

// SetMatrix / GetMatrix — transfer a column-major matrix subregion.
cublasStatus_t cublasSetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb);

cublasStatus_t cublasGetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb);

// Async variants (identical to sync on Apple Silicon UMA).
cublasStatus_t cublasSetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t stream);

cublasStatus_t cublasGetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t stream);

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t stream);

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
