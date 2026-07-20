#pragma once

#include "cusolver_common.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st* cudaStream_t;
typedef struct cusolverDnContext* cusolverDnHandle_t;

typedef enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2,
} cublasFillMode_t;

typedef enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
} cublasSideMode_t;

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

#ifdef __cplusplus
}
#endif
