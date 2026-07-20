#pragma once

#include "cusolver_common.h"
#include "cusparse.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CUstream_st* cudaStream_t;
typedef struct cusolverSpContext* cusolverSpHandle_t;

// Handle management
cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* handle);
cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle);
cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId);

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

#ifdef __cplusplus
}
#endif
