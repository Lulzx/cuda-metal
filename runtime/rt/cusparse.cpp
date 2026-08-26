#include "cusparse.h"
#include "cuda_runtime.h"

#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <mutex>
#include <vector>

// ── cuSPARSE shim ───────────────────────────────────────────────────────────
// CPU-backed sparse matrix operations for Apple Silicon UMA.
// Sparse operations are computed on the CPU using Accelerate-style loops;
// on UMA there is zero copy overhead.

extern "C" {

struct cusparseContext {
    cudaStream_t stream = nullptr;
    cusparsePointerMode_t pointer_mode = CUSPARSE_POINTER_MODE_HOST;
};

struct cusparseMatDescr {
    cusparseMatrixType_t type = CUSPARSE_MATRIX_TYPE_GENERAL;
    cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO;
    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
};

struct cusparseSpMatDescr {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t nnz = 0;
    void* rowOffsets = nullptr;
    void* colInd = nullptr;
    void* values = nullptr;
    cusparseIndexType_t rowType = CUSPARSE_INDEX_32I;
    cusparseIndexType_t colType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType = CUDA_R_32F;
    bool is_csr = true;
};

struct cusparseDnVecDescr {
    int64_t size = 0;
    void* values = nullptr;
    cudaDataType valueType = CUDA_R_32F;
};

struct cusparseDnMatDescr {
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld = 0;
    void* values = nullptr;
    cudaDataType valueType = CUDA_R_32F;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
};

// Handle management

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
    if (handle == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *handle = new cusparseContext();
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    delete handle;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId) {
    if (handle == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId) {
    if (handle == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (streamId) *streamId = handle->stream;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode) {
    if (handle == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mode != CUSPARSE_POINTER_MODE_HOST && mode != CUSPARSE_POINTER_MODE_DEVICE) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    // The current CPU/UMA sparse paths dereference host alpha/beta values when
    // the API call executes. Do not claim device-scalar mode until every
    // implemented sparse operation resolves tracked device scalar storage.
    if (mode == CUSPARSE_POINTER_MODE_DEVICE) return CUSPARSE_STATUS_NOT_SUPPORTED;
    handle->pointer_mode = mode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode) {
    if (handle == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *mode = handle->pointer_mode;
    return CUSPARSE_STATUS_SUCCESS;
}

int cusparseGetVersion(cusparseHandle_t /*handle*/, int* version) {
    if (version) *version = 12000;
    return CUSPARSE_STATUS_SUCCESS;
}

// Matrix descriptor

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t* descrA) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descrA = new cusparseMatDescr();
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    delete descrA;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->type = type;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->type : CUSPARSE_MATRIX_TYPE_GENERAL;
}

cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->base = base;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->base : CUSPARSE_INDEX_BASE_ZERO;
}

cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->fill = fillMode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->diag = diagType;
    return CUSPARSE_STATUS_SUCCESS;
}

// Generic sparse descriptors

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* csrRowOffsets, void* csrColInd,
                                    void* csrValues,
                                    cusparseIndexType_t csrRowOffsetsType,
                                    cusparseIndexType_t csrColIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* sp = new cusparseSpMatDescr();
    sp->rows = rows;
    sp->cols = cols;
    sp->nnz = nnz;
    sp->rowOffsets = csrRowOffsets;
    sp->colInd = csrColInd;
    sp->values = csrValues;
    sp->rowType = csrRowOffsetsType;
    sp->colType = csrColIndType;
    sp->idxBase = idxBase;
    sp->valueType = valueType;
    sp->is_csr = true;
    *spMatDescr = sp;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* cooRowInd, void* cooColInd, void* cooValues,
                                    cusparseIndexType_t cooIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* sp = new cusparseSpMatDescr();
    sp->rows = rows;
    sp->cols = cols;
    sp->nnz = nnz;
    sp->rowOffsets = cooRowInd;
    sp->colInd = cooColInd;
    sp->values = cooValues;
    sp->rowType = cooIdxType;
    sp->colType = cooIdxType;
    sp->idxBase = idxBase;
    sp->valueType = valueType;
    sp->is_csr = false;
    *spMatDescr = sp;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
    delete spMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                      int64_t size, void* values, cudaDataType valueType) {
    if (dnVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* v = new cusparseDnVecDescr();
    v->size = size;
    v->values = values;
    v->valueType = valueType;
    *dnVecDescr = v;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
    delete dnVecDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                      int64_t rows, int64_t cols, int64_t ld,
                                      void* values, cudaDataType valueType,
                                      cusparseOrder_t order) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* m = new cusparseDnMatDescr();
    m->rows = rows;
    m->cols = cols;
    m->ld = ld;
    m->values = values;
    m->valueType = valueType;
    m->order = order;
    *dnMatDescr = m;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
    delete dnMatDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

// SpMV: y = alpha * op(A) * x + beta * y  (CSR, float)
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t /*opA*/,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void* beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpMVAlg_t /*alg*/,
                                          size_t* bufferSize) {
    if (!handle || !alpha || !matA || !vecX || !beta || !vecY || !bufferSize) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    // The CPU/UMA implementation consumes no external workspace, but CUDA
    // callers commonly allocate the reported size unconditionally.
    *bufferSize = 1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle,
                               cusparseOperation_t opA,
                               const void* alpha,
                               cusparseSpMatDescr_t matA,
                               cusparseDnVecDescr_t vecX,
                               const void* beta,
                               cusparseDnVecDescr_t vecY,
                               cudaDataType computeType,
                               cusparseSpMVAlg_t /*alg*/,
                               void* /*externalBuffer*/) {
    if (!handle || !matA || !vecX || !vecY || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    // Synchronize stream before CPU computation on UMA
    if (handle->stream) cudaStreamSynchronize(handle->stream);

    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int64_t m = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? matA->rows : matA->cols;

    if (matA->is_csr) {
        const int* rowPtr = static_cast<const int*>(matA->rowOffsets);
        const int* colIdx = static_cast<const int*>(matA->colInd);
        if (computeType == CUDA_R_64F) {
            const double a = *static_cast<const double*>(alpha);
            const double b = *static_cast<const double*>(beta);
            const double* vals = static_cast<const double*>(matA->values);
            const double* x = static_cast<const double*>(vecX->values);
            double* y = static_cast<double*>(vecY->values);
            for (int64_t i = 0; i < m; ++i) {
                double sum = 0.0;
                const int row_start = rowPtr[i] - base;
                const int row_end = rowPtr[i + 1] - base;
                for (int j = row_start; j < row_end; ++j) {
                    sum += vals[j] * x[colIdx[j] - base];
                }
                y[i] = a * sum + b * y[i];
            }
        } else {
            const float a = *static_cast<const float*>(alpha);
            const float b = *static_cast<const float*>(beta);
            const float* vals = static_cast<const float*>(matA->values);
            const float* x = static_cast<const float*>(vecX->values);
            float* y = static_cast<float*>(vecY->values);
            for (int64_t i = 0; i < m; ++i) {
                float sum = 0.0f;
                const int row_start = rowPtr[i] - base;
                const int row_end = rowPtr[i + 1] - base;
                for (int j = row_start; j < row_end; ++j) {
                    sum += vals[j] * x[colIdx[j] - base];
                }
                y[i] = a * sum + b * y[i];
            }
        }
    } else {
        // COO format
        const int* rowInd = static_cast<const int*>(matA->rowOffsets);
        const int* colIdx = static_cast<const int*>(matA->colInd);
        if (computeType == CUDA_R_64F) {
            const double a = *static_cast<const double*>(alpha);
            const double b = *static_cast<const double*>(beta);
            const double* vals = static_cast<const double*>(matA->values);
            const double* x = static_cast<const double*>(vecX->values);
            double* y = static_cast<double*>(vecY->values);
            for (int64_t i = 0; i < m; ++i) y[i] = b * y[i];
            for (int64_t e = 0; e < matA->nnz; ++e) {
                const int r = rowInd[e] - base;
                const int c = colIdx[e] - base;
                y[r] += a * vals[e] * x[c];
            }
        } else {
            const float a = *static_cast<const float*>(alpha);
            const float b = *static_cast<const float*>(beta);
            const float* vals = static_cast<const float*>(matA->values);
            const float* x = static_cast<const float*>(vecX->values);
            float* y = static_cast<float*>(vecY->values);
            for (int64_t i = 0; i < m; ++i) y[i] = b * y[i];
            for (int64_t e = 0; e < matA->nnz; ++e) {
                const int r = rowInd[e] - base;
                const int c = colIdx[e] - base;
                y[r] += a * vals[e] * x[c];
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// SpMM: C = alpha * op(A) * op(B) + beta * C
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t /*handle*/,
                                          cusparseOperation_t /*opA*/,
                                          cusparseOperation_t /*opB*/,
                                          const void* /*alpha*/,
                                          cusparseSpMatDescr_t /*matA*/,
                                          cusparseDnMatDescr_t /*matB*/,
                                          const void* /*beta*/,
                                          cusparseDnMatDescr_t /*matC*/,
                                          cudaDataType /*computeType*/,
                                          cusparseSpMMAlg_t /*alg*/,
                                          size_t* bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMM(cusparseHandle_t handle,
                               cusparseOperation_t opA,
                               cusparseOperation_t /*opB*/,
                               const void* alpha,
                               cusparseSpMatDescr_t matA,
                               cusparseDnMatDescr_t matB,
                               const void* beta,
                               cusparseDnMatDescr_t matC,
                               cudaDataType computeType,
                               cusparseSpMMAlg_t /*alg*/,
                               void* /*externalBuffer*/) {
    if (!handle || !matA || !matB || !matC || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }

    if (handle->stream) cudaStreamSynchronize(handle->stream);

    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int64_t m = (opA == CUSPARSE_OPERATION_NON_TRANSPOSE) ? matA->rows : matA->cols;
    const int64_t n = matB->cols;
    const int64_t ldb = matB->ld;
    const int64_t ldc = matC->ld;

    if (matA->is_csr) {
        const int* rowPtr = static_cast<const int*>(matA->rowOffsets);
        const int* colIdx = static_cast<const int*>(matA->colInd);
        if (computeType == CUDA_R_64F) {
            const double a = *static_cast<const double*>(alpha);
            const double b = *static_cast<const double*>(beta);
            const double* vals = static_cast<const double*>(matA->values);
            const double* B = static_cast<const double*>(matB->values);
            double* C = static_cast<double*>(matC->values);
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    double sum = 0.0;
                    const int row_start = rowPtr[i] - base;
                    const int row_end = rowPtr[i + 1] - base;
                    for (int k = row_start; k < row_end; ++k) {
                        sum += vals[k] * B[colIdx[k] - base + j * ldb];
                    }
                    C[i + j * ldc] = a * sum + b * C[i + j * ldc];
                }
            }
        } else {
            const float a = *static_cast<const float*>(alpha);
            const float b = *static_cast<const float*>(beta);
            const float* vals = static_cast<const float*>(matA->values);
            const float* B = static_cast<const float*>(matB->values);
            float* C = static_cast<float*>(matC->values);
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    const int row_start = rowPtr[i] - base;
                    const int row_end = rowPtr[i + 1] - base;
                    for (int k = row_start; k < row_end; ++k) {
                        sum += vals[k] * B[colIdx[k] - base + j * ldb];
                    }
                    C[i + j * ldc] = a * sum + b * C[i + j * ldc];
                }
            }
        }
    } else {
        // COO format
        const int* rowInd = static_cast<const int*>(matA->rowOffsets);
        const int* colIdx = static_cast<const int*>(matA->colInd);
        if (computeType == CUDA_R_64F) {
            const double a = *static_cast<const double*>(alpha);
            const double b = *static_cast<const double*>(beta);
            const double* vals = static_cast<const double*>(matA->values);
            const double* B = static_cast<const double*>(matB->values);
            double* C = static_cast<double*>(matC->values);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j)
                    C[i + j * ldc] = b * C[i + j * ldc];
            for (int64_t e = 0; e < matA->nnz; ++e) {
                const int r = rowInd[e] - base;
                const int c = colIdx[e] - base;
                for (int64_t j = 0; j < n; ++j)
                    C[r + j * ldc] += a * vals[e] * B[c + j * ldb];
            }
        } else {
            const float a = *static_cast<const float*>(alpha);
            const float b = *static_cast<const float*>(beta);
            const float* vals = static_cast<const float*>(matA->values);
            const float* B = static_cast<const float*>(matB->values);
            float* C = static_cast<float*>(matC->values);
            for (int64_t i = 0; i < m; ++i)
                for (int64_t j = 0; j < n; ++j)
                    C[i + j * ldc] = b * C[i + j * ldc];
            for (int64_t e = 0; e < matA->nnz; ++e) {
                const int r = rowInd[e] - base;
                const int c = colIdx[e] - base;
                for (int64_t j = 0; j < n; ++j)
                    C[r + j * ldc] += a * vals[e] * B[c + j * ldb];
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// Legacy CSR SpMV (float)
cusparseStatus_t cusparseScsrmv(cusparseHandle_t handle,
                                 cusparseOperation_t /*transA*/,
                                 int m, int /*n*/, int /*nnz*/,
                                 const float* alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float* csrValA,
                                 const int* csrRowPtrA,
                                 const int* csrColIndA,
                                 const float* x,
                                 const float* beta,
                                 float* y) {
    if (!handle || !alpha || !beta || !csrValA || !csrRowPtrA || !csrColIndA || !x || !y) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (handle->stream) cudaStreamSynchronize(handle->stream);

    const int base = descrA ? static_cast<int>(descrA->base) : 0;
    for (int i = 0; i < m; ++i) {
        float sum = 0.0f;
        const int row_start = csrRowPtrA[i] - base;
        const int row_end = csrRowPtrA[i + 1] - base;
        for (int j = row_start; j < row_end; ++j) {
            sum += csrValA[j] * x[csrColIndA[j] - base];
        }
        y[i] = (*alpha) * sum + (*beta) * y[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// Legacy CSR SpMV (double)
cusparseStatus_t cusparseDcsrmv(cusparseHandle_t handle,
                                 cusparseOperation_t /*transA*/,
                                 int m, int /*n*/, int /*nnz*/,
                                 const double* alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double* csrValA,
                                 const int* csrRowPtrA,
                                 const int* csrColIndA,
                                 const double* x,
                                 const double* beta,
                                 double* y) {
    if (!handle || !alpha || !beta || !csrValA || !csrRowPtrA || !csrColIndA || !x || !y) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (handle->stream) cudaStreamSynchronize(handle->stream);

    const int base = descrA ? static_cast<int>(descrA->base) : 0;
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        const int row_start = csrRowPtrA[i] - base;
        const int row_end = csrRowPtrA[i + 1] - base;
        for (int j = row_start; j < row_end; ++j) {
            sum += csrValA[j] * x[csrColIndA[j] - base];
        }
        y[i] = (*alpha) * sum + (*beta) * y[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpSV: Sparse triangular solve ─────────────────────────────────────────────

struct cusparseSpSVDescr {
    // Analysis phase is a no-op on CPU; descriptor exists for API compat
};

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseSpSVDescr();
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t, cusparseOperation_t,
                                          const void*, cusparseSpMatDescr_t,
                                          cusparseDnVecDescr_t, cusparseDnVecDescr_t,
                                          cudaDataType, cusparseSpSVAlg_t,
                                          cusparseSpSVDescr_t, size_t* bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t, cusparseOperation_t,
                                        const void*, cusparseSpMatDescr_t,
                                        cusparseDnVecDescr_t, cusparseDnVecDescr_t,
                                        cudaDataType, cusparseSpSVAlg_t,
                                        cusparseSpSVDescr_t, void*) {
    // CPU triangular solve needs no pre-analysis
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const void* alpha,
                                     cusparseSpMatDescr_t matA,
                                     cusparseDnVecDescr_t vecX,
                                     cusparseDnVecDescr_t vecY,
                                     cudaDataType computeType,
                                     cusparseSpSVAlg_t,
                                     cusparseSpSVDescr_t) {
    if (!handle || !matA || !vecX || !vecY || !alpha) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!matA->is_csr) return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F)
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    if (handle->stream) cudaStreamSynchronize(handle->stream);

    const int* rowPtr = static_cast<const int*>(matA->rowOffsets);
    const int* colIdx = static_cast<const int*>(matA->colInd);
    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int64_t n = matA->rows;

    // Forward substitution: solve L*y = alpha*x row by row
    if (computeType == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double* vals = static_cast<const double*>(matA->values);
        const double* x = static_cast<const double*>(vecX->values);
        double* y = static_cast<double*>(vecY->values);
        for (int64_t i = 0; i < n; ++i) {
            double rhs = a * x[i];
            double diag = 1.0;
            const int rs = rowPtr[i] - base;
            const int re = rowPtr[i + 1] - base;
            for (int j = rs; j < re; ++j) {
                const int c = colIdx[j] - base;
                if (c == i) { diag = vals[j]; }
                else if ((opA == CUSPARSE_OPERATION_NON_TRANSPOSE && c < i) ||
                         (opA != CUSPARSE_OPERATION_NON_TRANSPOSE && c > i)) {
                    rhs -= vals[j] * y[c];
                }
            }
            y[i] = rhs / diag;
        }
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float* vals = static_cast<const float*>(matA->values);
        const float* x = static_cast<const float*>(vecX->values);
        float* y = static_cast<float*>(vecY->values);
        for (int64_t i = 0; i < n; ++i) {
            float rhs = a * x[i];
            float diag = 1.0f;
            const int rs = rowPtr[i] - base;
            const int re = rowPtr[i + 1] - base;
            for (int j = rs; j < re; ++j) {
                const int c = colIdx[j] - base;
                if (c == i) { diag = vals[j]; }
                else if ((opA == CUSPARSE_OPERATION_NON_TRANSPOSE && c < i) ||
                         (opA != CUSPARSE_OPERATION_NON_TRANSPOSE && c > i)) {
                    rhs -= vals[j] * y[c];
                }
            }
            y[i] = rhs / diag;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

}  // extern "C"
