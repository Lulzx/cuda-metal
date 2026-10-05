#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUSOLVERSP_H_
#define CUSOLVERSP_H_ 1
#endif


#include "cusolver_common.h"
#include "cusparse.h"
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
typedef struct cusolverSpContext* cusolverSpHandle_t;

// Handle management
cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* handle);
cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle);
cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId);
cusolverStatus_t cusolverSpGetStream(cusolverSpHandle_t handle, cudaStream_t* streamId);

// Sparse Cholesky (host path) — solve A*x = b where A is SPD
cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const float* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        const float* b,
                                        float tol,
                                        int reorder,
                                        float* x,
                                        int* singularity);

cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const double* csrVal,
                                        const int* csrRowPtr,
                                        const int* csrColInd,
                                        const double* b,
                                        double tol,
                                        int reorder,
                                        double* x,
                                        int* singularity);

// Sparse QR (host path) — solve A*x = b via QR factorization
cusolverStatus_t cusolverSpScsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const float* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const float* b,
                                      float tol,
                                      int reorder,
                                      float* x,
                                      int* singularity);

cusolverStatus_t cusolverSpDcsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const double* csrVal,
                                      const int* csrRowPtr,
                                      const int* csrColInd,
                                      const double* b,
                                      double tol,
                                      int reorder,
                                      double* x,
                                      int* singularity);

// Complex sparse Cholesky / QR solves (host path, dense conversion + LAPACK).
// A is Hermitian positive definite for the Cholesky variants. Unlike the real
// entry points, reorder 0..3 are all accepted: reordering only changes fill-in
// in NVIDIA's sparse factorization and never the solution.
cusolverStatus_t cusolverSpCcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const cuComplex* csrVal, const int* csrRowPtr,
                                        const int* csrColInd, const cuComplex* b,
                                        float tol, int reorder, cuComplex* x,
                                        int* singularity);
cusolverStatus_t cusolverSpZcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const cuDoubleComplex* csrVal,
                                        const int* csrRowPtr, const int* csrColInd,
                                        const cuDoubleComplex* b, double tol,
                                        int reorder, cuDoubleComplex* x,
                                        int* singularity);
cusolverStatus_t cusolverSpCcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const cuComplex* csrVal, const int* csrRowPtr,
                                      const int* csrColInd, const cuComplex* b,
                                      float tol, int reorder, cuComplex* x,
                                      int* singularity);
cusolverStatus_t cusolverSpZcsrlsvqr(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const cuDoubleComplex* csrVal,
                                      const int* csrRowPtr, const int* csrColInd,
                                      const cuDoubleComplex* b, double tol,
                                      int reorder, cuDoubleComplex* x,
                                      int* singularity);

// Shifted inverse iteration: the eigenpair of the general matrix A whose
// eigenvalue is closest to mu0, started from x0. Runs at most maxite
// iterations and stops early once ||A*x - mu*x|| <= eps * max(1, |mu|). The
// last iterate is returned either way (as NVIDIA does); x has unit 2-norm.
cusolverStatus_t cusolverSpScsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const float* csrValA, const int* csrRowPtrA,
                                       const int* csrColIndA, float mu0,
                                       const float* x0, int maxite, float eps,
                                       float* mu, float* x);
cusolverStatus_t cusolverSpDcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const double* csrValA, const int* csrRowPtrA,
                                       const int* csrColIndA, double mu0,
                                       const double* x0, int maxite, double eps,
                                       double* mu, double* x);
cusolverStatus_t cusolverSpCcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const cuComplex* csrValA, const int* csrRowPtrA,
                                       const int* csrColIndA, cuComplex mu0,
                                       const cuComplex* x0, int maxite, float eps,
                                       cuComplex* mu, cuComplex* x);
cusolverStatus_t cusolverSpZcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const cuDoubleComplex* csrValA,
                                       const int* csrRowPtrA, const int* csrColIndA,
                                       cuDoubleComplex mu0, const cuDoubleComplex* x0,
                                       int maxite, double eps, cuDoubleComplex* mu,
                                       cuDoubleComplex* x);

#ifdef __cplusplus
}
#endif
