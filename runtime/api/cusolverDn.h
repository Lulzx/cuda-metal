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
#include "cuComplex.h"

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

// ── Surface required by CuPy: complex dense, QR helpers, batched Cholesky,
// symmetric-indefinite LDL, bidiagonalisation, Jacobi SVD/EVD, approximate
// SVD and the iterative-refinement (IRS) gesv/gels families. ────────────────
//
// All of these run on Accelerate LAPACK over unified-memory pointers. Workspace
// queries report what the call needs from the caller; routines that allocate
// scratch internally (gesvdj, gesvda, IRS) report a minimal size and accept it.

// gesvdj configuration: tolerance, max_sweeps and sort_svd are stored and
// returned. The decomposition itself is computed by LAPACK's SVD, so the
// executed-sweep count is always 0 and the residual is measured explicitly as
// ||diag(S) - U^H*A*V||_F from the result.
cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info);
cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info);
cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance);
cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps);
cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd);
cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle,
                                              gesvdjInfo_t info, double* residual);
cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle,
                                            gesvdjInfo_t info, int* executed_sweeps);

// Entry points present for all of S, D, C and Z. ORG/ORM/EVJ are the real
// (orgqr, ormqr, syevj) or complex (ungqr, unmqr, heevj) spellings.
#define CUMETAL_CUSOLVER_DN_COMMON(PFX, ELEM, REAL, ORG, ORM, EVJ)                          \
    cusolverStatus_t cusolverDn##PFX##ORG##_bufferSize(                                     \
        cusolverDnHandle_t handle, int m, int n, int k, const ELEM* A, int lda,             \
        const ELEM* tau, int* lwork);                                                       \
    cusolverStatus_t cusolverDn##PFX##ORG(                                                  \
        cusolverDnHandle_t handle, int m, int n, int k, ELEM* A, int lda, const ELEM* tau,  \
        ELEM* work, int lwork, int* info);                                                  \
    cusolverStatus_t cusolverDn##PFX##ORM##_bufferSize(                                     \
        cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,   \
        int n, int k, const ELEM* A, int lda, const ELEM* tau, const ELEM* C, int ldc,      \
        int* lwork);                                                                        \
    cusolverStatus_t cusolverDn##PFX##ORM(                                                  \
        cusolverDnHandle_t handle, cublasSideMode_t side, cublasOperation_t trans, int m,   \
        int n, int k, const ELEM* A, int lda, const ELEM* tau, ELEM* C, int ldc,            \
        ELEM* work, int lwork, int* devInfo);                                               \
    cusolverStatus_t cusolverDn##PFX##potrfBatched(                                         \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, ELEM* Aarray[], int lda,   \
        int* infoArray, int batchSize);                                                     \
    cusolverStatus_t cusolverDn##PFX##potrsBatched(                                         \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, ELEM* A[],       \
        int lda, ELEM* B[], int ldb, int* d_info, int batchSize);                           \
    cusolverStatus_t cusolverDn##PFX##sytrf_bufferSize(                                     \
        cusolverDnHandle_t handle, int n, ELEM* A, int lda, int* lwork);                    \
    cusolverStatus_t cusolverDn##PFX##sytrf(                                                \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, ELEM* A, int lda,          \
        int* ipiv, ELEM* work, int lwork, int* info);                                       \
    cusolverStatus_t cusolverDn##PFX##gebrd_bufferSize(                                     \
        cusolverDnHandle_t handle, int m, int n, int* Lwork);                               \
    cusolverStatus_t cusolverDn##PFX##gebrd(                                                \
        cusolverDnHandle_t handle, int m, int n, ELEM* A, int lda, REAL* D, REAL* E,        \
        ELEM* TAUQ, ELEM* TAUP, ELEM* Work, int Lwork, int* devInfo);                       \
    cusolverStatus_t cusolverDn##PFX##EVJ##_bufferSize(                                     \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        const ELEM* A, int lda, const REAL* W, int* lwork, syevjInfo_t params);             \
    cusolverStatus_t cusolverDn##PFX##EVJ(                                                  \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        ELEM* A, int lda, REAL* W, ELEM* work, int lwork, int* info, syevjInfo_t params);   \
    cusolverStatus_t cusolverDn##PFX##gesvdj_bufferSize(                                    \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n,          \
        const ELEM* A, int lda, const REAL* S, const ELEM* U, int ldu, const ELEM* V,       \
        int ldv, int* lwork, gesvdjInfo_t params);                                          \
    cusolverStatus_t cusolverDn##PFX##gesvdj(                                               \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m, int n, ELEM* A, \
        int lda, REAL* S, ELEM* U, int ldu, ELEM* V, int ldv, ELEM* work, int lwork,        \
        int* info, gesvdjInfo_t params);                                                    \
    cusolverStatus_t cusolverDn##PFX##gesvdjBatched_bufferSize(                             \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, const ELEM* A,     \
        int lda, const REAL* S, const ELEM* U, int ldu, const ELEM* V, int ldv, int* lwork, \
        gesvdjInfo_t params, int batchSize);                                                \
    cusolverStatus_t cusolverDn##PFX##gesvdjBatched(                                        \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m, int n, ELEM* A, int lda,  \
        REAL* S, ELEM* U, int ldu, ELEM* V, int ldv, ELEM* work, int lwork, int* info,      \
        gesvdjInfo_t params, int batchSize);                                                \
    cusolverStatus_t cusolverDn##PFX##gesvdaStridedBatched_bufferSize(                      \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,          \
        const ELEM* d_A, int lda, long long int strideA, const REAL* d_S,                   \
        long long int strideS, const ELEM* d_U, int ldu, long long int strideU,             \
        const ELEM* d_V, int ldv, long long int strideV, int* lwork, int batchSize);        \
    cusolverStatus_t cusolverDn##PFX##gesvdaStridedBatched(                                 \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m, int n,          \
        const ELEM* d_A, int lda, long long int strideA, REAL* d_S, long long int strideS,  \
        ELEM* d_U, int ldu, long long int strideU, ELEM* d_V, int ldv,                      \
        long long int strideV, ELEM* d_work, int lwork, int* d_info, double* h_R_nrmF,      \
        int batchSize);

// Entry points that exist only in complex precision here (the real ones are
// declared above). EVD/EVJ are heevd/heevj.
#define CUMETAL_CUSOLVER_DN_COMPLEX(PFX, ELEM, REAL, EVD, EVJ)                              \
    cusolverStatus_t cusolverDn##PFX##getrf_bufferSize(                                     \
        cusolverDnHandle_t handle, int m, int n, ELEM* A, int lda, int* Lwork);             \
    cusolverStatus_t cusolverDn##PFX##getrf(                                                \
        cusolverDnHandle_t handle, int m, int n, ELEM* A, int lda, ELEM* Workspace,         \
        int* devIpiv, int* devInfo);                                                        \
    cusolverStatus_t cusolverDn##PFX##getrs(                                                \
        cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs, const ELEM* A, \
        int lda, const int* devIpiv, ELEM* B, int ldb, int* devInfo);                       \
    cusolverStatus_t cusolverDn##PFX##geqrf_bufferSize(                                     \
        cusolverDnHandle_t handle, int m, int n, ELEM* A, int lda, int* Lwork);             \
    cusolverStatus_t cusolverDn##PFX##geqrf(                                                \
        cusolverDnHandle_t handle, int m, int n, ELEM* A, int lda, ELEM* TAU,               \
        ELEM* Workspace, int Lwork, int* devInfo);                                          \
    cusolverStatus_t cusolverDn##PFX##potrf_bufferSize(                                     \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, ELEM* A, int lda,          \
        int* Lwork);                                                                        \
    cusolverStatus_t cusolverDn##PFX##potrf(                                                \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, ELEM* A, int lda,          \
        ELEM* Workspace, int Lwork, int* devInfo);                                          \
    cusolverStatus_t cusolverDn##PFX##potrs(                                                \
        cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs, const ELEM* A,   \
        int lda, ELEM* B, int ldb, int* devInfo);                                           \
    cusolverStatus_t cusolverDn##PFX##gesvd_bufferSize(                                     \
        cusolverDnHandle_t handle, int m, int n, int* lwork);                               \
    cusolverStatus_t cusolverDn##PFX##gesvd(                                                \
        cusolverDnHandle_t handle, signed char jobu, signed char jobvt, int m, int n,       \
        ELEM* A, int lda, REAL* S, ELEM* U, int ldu, ELEM* VT, int ldvt, ELEM* work,        \
        int lwork, REAL* rwork, int* devInfo);                                              \
    cusolverStatus_t cusolverDn##PFX##EVD##_bufferSize(                                     \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        const ELEM* A, int lda, const REAL* W, int* lwork);                                 \
    cusolverStatus_t cusolverDn##PFX##EVD(                                                  \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        ELEM* A, int lda, REAL* W, ELEM* work, int lwork, int* devInfo);                    \
    cusolverStatus_t cusolverDn##PFX##EVJ##Batched_bufferSize(                              \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        const ELEM* A, int lda, const REAL* W, int* lwork, syevjInfo_t params,              \
        int batchSize);                                                                     \
    cusolverStatus_t cusolverDn##PFX##EVJ##Batched(                                         \
        cusolverDnHandle_t handle, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,    \
        ELEM* A, int lda, REAL* W, ELEM* work, int lwork, int* info, syevjInfo_t params,    \
        int batchSize);

CUMETAL_CUSOLVER_DN_COMMON(S, float, float, orgqr, ormqr, syevj)
CUMETAL_CUSOLVER_DN_COMMON(D, double, double, orgqr, ormqr, syevj)
CUMETAL_CUSOLVER_DN_COMMON(C, cuComplex, float, ungqr, unmqr, heevj)
CUMETAL_CUSOLVER_DN_COMMON(Z, cuDoubleComplex, double, ungqr, unmqr, heevj)
CUMETAL_CUSOLVER_DN_COMPLEX(C, cuComplex, float, heevd, heevj)
CUMETAL_CUSOLVER_DN_COMPLEX(Z, cuDoubleComplex, double, heevd, heevj)

#undef CUMETAL_CUSOLVER_DN_COMMON
#undef CUMETAL_CUSOLVER_DN_COMPLEX

// Iterative-refinement solvers. The two letters are (main precision, lowest
// precision): the pointers are always in the main precision. CuMetal solves in
// the main precision directly with LAPACK (LU for gesv, QR for gels) and never
// uses the lower one, so *niter is always 0 (no refinement iterations) and the
// answer is at least as accurate as NVIDIA's. dA is preserved. Workspace is a
// fixed minimal size and is not otherwise used. gels needs m >= n.
#define CUMETAL_CUSOLVER_DN_IRS(PP, ELEM)                                                   \
    cusolverStatus_t cusolverDn##PP##gesv_bufferSize(                                       \
        cusolverDnHandle_t handle, int n, int nrhs, ELEM* dA, int ldda, int* dipiv,         \
        ELEM* dB, int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t* lwork_bytes);     \
    cusolverStatus_t cusolverDn##PP##gesv(                                                  \
        cusolverDnHandle_t handle, int n, int nrhs, ELEM* dA, int ldda, int* dipiv,         \
        ELEM* dB, int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t lwork_bytes,       \
        int* iter, int* d_info);                                                            \
    cusolverStatus_t cusolverDn##PP##gels_bufferSize(                                       \
        cusolverDnHandle_t handle, int m, int n, int nrhs, ELEM* dA, int ldda, ELEM* dB,    \
        int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t* lwork_bytes);               \
    cusolverStatus_t cusolverDn##PP##gels(                                                  \
        cusolverDnHandle_t handle, int m, int n, int nrhs, ELEM* dA, int ldda, ELEM* dB,    \
        int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter,      \
        int* d_info);

CUMETAL_CUSOLVER_DN_IRS(DD, double)
CUMETAL_CUSOLVER_DN_IRS(DS, double)
CUMETAL_CUSOLVER_DN_IRS(DH, double)
CUMETAL_CUSOLVER_DN_IRS(DX, double)
CUMETAL_CUSOLVER_DN_IRS(SS, float)
CUMETAL_CUSOLVER_DN_IRS(SH, float)
CUMETAL_CUSOLVER_DN_IRS(SX, float)
CUMETAL_CUSOLVER_DN_IRS(ZZ, cuDoubleComplex)
CUMETAL_CUSOLVER_DN_IRS(ZC, cuDoubleComplex)
CUMETAL_CUSOLVER_DN_IRS(ZK, cuDoubleComplex)
CUMETAL_CUSOLVER_DN_IRS(ZY, cuDoubleComplex)
CUMETAL_CUSOLVER_DN_IRS(CC, cuComplex)
CUMETAL_CUSOLVER_DN_IRS(CK, cuComplex)
CUMETAL_CUSOLVER_DN_IRS(CY, cuComplex)

#undef CUMETAL_CUSOLVER_DN_IRS

#ifdef __cplusplus
}
#endif
