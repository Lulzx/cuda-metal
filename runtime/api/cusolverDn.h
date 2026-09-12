#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUSOLVERDN_H_
#define CUSOLVERDN_H_ 1
#endif


#include "cusolver_common.h"

// cublasFillMode_t and cublasSideMode_t are owned by cublas_v2.h; including it
// here keeps one definition in either include order.
#include "cublas_v2.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUMETAL_CUDA_STREAM_T_DEFINED
#define CUMETAL_CUDA_STREAM_T_DEFINED 1
typedef struct CUstream_st* cudaStream_t;
#endif  // CUMETAL_CUDA_STREAM_T_DEFINED
typedef struct cusolverDnContext* cusolverDnHandle_t;
typedef struct cusolverDnParams* cusolverDnParams_t;

// Handle management
cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId);

// LU factorization
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* Workspace,
                                   int* devIpiv, int* devInfo);
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* Workspace,
                                   int* devIpiv, int* devInfo);

// LU solve
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const float* A, int lda,
                                   const int* devIpiv, float* B, int ldb,
                                   int* devInfo);
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const double* A, int lda,
                                   const int* devIpiv, double* B, int ldb,
                                   int* devInfo);

// QR factorization
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* TAU,
                                   float* Workspace, int Lwork, int* devInfo);
cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* TAU,
                                   double* Workspace, int Lwork, int* devInfo);

// Cholesky factorization
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double* A, int lda, int* Lwork);
cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, float* A, int lda, float* Workspace,
                                   int Lwork, int* devInfo);
cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, double* A, int lda, double* Workspace,
                                   int Lwork, int* devInfo);

// Cholesky solve
cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const float* A, int lda,
                                   float* B, int ldb, int* devInfo);
cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const double* A, int lda,
                                   double* B, int ldb, int* devInfo);

// Eigenvalue decomposition (syevd)
cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              const float* A, int lda,
                                              const float* W, int* lwork);
cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              const double* A, int lda,
                                              const double* W, int* lwork);
cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, float* A, int lda,
                                   float* W, float* work, int lwork, int* devInfo);
cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, double* A, int lda,
                                   double* W, double* work, int lwork, int* devInfo);

// SVD
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int* lwork);
cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int* lwork);
cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, float* A, int lda,
                                   float* S, float* U, int ldu, float* VT, int ldvt,
                                   float* work, int lwork, float* rwork, int* devInfo);
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, double* A, int lda,
                                   double* S, double* U, int ldu, double* VT, int ldvt,
                                   double* work, int lwork, double* rwork, int* devInfo);

// Version query. Reports CuMetal's own version, not an NVIDIA library release.
cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value);

// Params handle for the 64-bit generic interfaces.
cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params);
cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params);

// Generic syevd. Bounded subset: dataTypeA, dataTypeW and computeType must be
// the same type, either CUDA_R_32F or CUDA_R_64F; anything else is rejected
// with CUSOLVER_STATUS_INVALID_VALUE. Sizes stay 64-bit on the interface and
// are range-checked against the int-wide LAPACK backend.
cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverDnParams_t params,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int64_t n,
                                             cudaDataType dataTypeA, const void* A,
                                             int64_t lda, cudaDataType dataTypeW,
                                             const void* W, cudaDataType computeType,
                                             size_t* workspaceInBytesOnDevice,
                                             size_t* workspaceInBytesOnHost);
cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle,
                                  cusolverDnParams_t params,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, int64_t n,
                                  cudaDataType dataTypeA, void* A, int64_t lda,
                                  cudaDataType dataTypeW, void* W,
                                  cudaDataType computeType,
                                  void* bufferOnDevice,
                                  size_t workspaceInBytesOnDevice,
                                  void* bufferOnHost,
                                  size_t workspaceInBytesOnHost, int* info);

// Batched generic syevd over strided matrices, homogeneous FP32/FP64 subset.
cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle,
                                                   cusolverDnParams_t params,
                                                   cusolverEigMode_t jobz,
                                                   cublasFillMode_t uplo,
                                                   int64_t n,
                                                   cudaDataType dataTypeA,
                                                   const void* A, int64_t lda,
                                                   int64_t strideA,
                                                   cudaDataType dataTypeW,
                                                   const void* W, int64_t strideW,
                                                   cudaDataType computeType,
                                                   int64_t batchSize,
                                                   size_t* workspaceInBytesOnDevice,
                                                   size_t* workspaceInBytesOnHost);
cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle,
                                        cusolverDnParams_t params,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int64_t n,
                                        cudaDataType dataTypeA, void* A,
                                        int64_t lda, int64_t strideA,
                                        cudaDataType dataTypeW, void* W,
                                        int64_t strideW, cudaDataType computeType,
                                        int64_t batchSize, void* bufferOnDevice,
                                        size_t workspaceInBytesOnDevice,
                                        void* bufferOnHost,
                                        size_t workspaceInBytesOnHost, int* info);

// Jacobi eigensolver configuration. CuMetal runs a real cyclic Jacobi sweep,
// so tolerance, max_sweeps and sort_eig are honoured rather than stored.
cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info);
cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info);
cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps);
cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig);
// Post-run statistics: the worst residual and sweep count across the batch
// processed through this info object.
cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle,
                                             syevjInfo_t info, double* residual);
cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle,
                                          syevjInfo_t info, int* executed_sweeps);

// Batched Jacobi eigensolver. A and W are batchSize contiguous matrices and
// vectors; devInfo[i] reports per-matrix convergence (0 = converged, >0 =
// sweeps exhausted without reaching tolerance).
cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n,
                                                    const float* A, int lda,
                                                    const float* W, int* lwork,
                                                    syevjInfo_t params,
                                                    int batchSize);
cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n,
                                                    const double* A, int lda,
                                                    const double* W, int* lwork,
                                                    syevjInfo_t params,
                                                    int batchSize);
cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, float* A,
                                        int lda, float* W, float* work,
                                        int lwork, int* devInfo,
                                        syevjInfo_t params, int batchSize);
cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, double* A,
                                        int lda, double* W, double* work,
                                        int lwork, int* devInfo,
                                        syevjInfo_t params, int batchSize);

#ifdef __cplusplus
}
#endif
