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

// CSC is not stored separately: the arrays a caller hands to
// cusparseCreateCsc describe A-transpose in exactly CSR layout, so the CSR
// kernels serve it once the operation is flipped. Only the tag is needed.
enum SpMatFormat { CUMETAL_SPMAT_CSR, CUMETAL_SPMAT_COO, CUMETAL_SPMAT_CSC };

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
    SpMatFormat format = CUMETAL_SPMAT_CSR;
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

// These entry points compute on the CPU over unified memory, so they must be
// ordered against GPU work the caller already enqueued -- on real CUDA the
// library call joins the handle's stream and that ordering is implicit.
//
// This used to read `if (handle->stream) cudaStreamSynchronize(handle->stream)`,
// which skipped synchronization entirely for a handle left on the default
// stream: the null stream is the default stream, not "no stream". An SpMV would
// then read its input vector while the kernel producing it was still in flight
// and quietly return zeros. cudaStreamSynchronize(nullptr) waits on the default
// stream, which is what CUDA's semantics require here.
static void synchronize_handle_stream(cusparseHandle_t handle) {
    if (handle == nullptr) return;
    cudaStreamSynchronize(handle->stream);
}

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

cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int* version) {
    if (handle == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;
    if (version == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *version = 12000;
    return CUSPARSE_STATUS_SUCCESS;
}

const char* cusparseGetErrorName(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:                   return "CUSPARSE_STATUS_SUCCESS";
        case CUSPARSE_STATUS_NOT_INITIALIZED:           return "CUSPARSE_STATUS_NOT_INITIALIZED";
        case CUSPARSE_STATUS_ALLOC_FAILED:              return "CUSPARSE_STATUS_ALLOC_FAILED";
        case CUSPARSE_STATUS_INVALID_VALUE:             return "CUSPARSE_STATUS_INVALID_VALUE";
        case CUSPARSE_STATUS_ARCH_MISMATCH:             return "CUSPARSE_STATUS_ARCH_MISMATCH";
        case CUSPARSE_STATUS_MAPPING_ERROR:             return "CUSPARSE_STATUS_MAPPING_ERROR";
        case CUSPARSE_STATUS_EXECUTION_FAILED:          return "CUSPARSE_STATUS_EXECUTION_FAILED";
        case CUSPARSE_STATUS_INTERNAL_ERROR:            return "CUSPARSE_STATUS_INTERNAL_ERROR";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
        case CUSPARSE_STATUS_ZERO_PIVOT:                return "CUSPARSE_STATUS_ZERO_PIVOT";
        case CUSPARSE_STATUS_NOT_SUPPORTED:             return "CUSPARSE_STATUS_NOT_SUPPORTED";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:    return "CUSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    }
    return "CUSPARSE_STATUS_UNKNOWN";
}

const char* cusparseGetErrorString(cusparseStatus_t status) {
    switch (status) {
        case CUSPARSE_STATUS_SUCCESS:                   return "success";
        case CUSPARSE_STATUS_NOT_INITIALIZED:           return "library not initialized";
        case CUSPARSE_STATUS_ALLOC_FAILED:              return "resource allocation failed";
        case CUSPARSE_STATUS_INVALID_VALUE:             return "an invalid value was used as an argument";
        case CUSPARSE_STATUS_ARCH_MISMATCH:             return "device architecture mismatch";
        case CUSPARSE_STATUS_MAPPING_ERROR:             return "a texture memory access failed";
        case CUSPARSE_STATUS_EXECUTION_FAILED:          return "the GPU program failed to execute";
        case CUSPARSE_STATUS_INTERNAL_ERROR:            return "an internal operation failed";
        case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "the matrix type is not supported by this function";
        case CUSPARSE_STATUS_ZERO_PIVOT:                return "a zero pivot was encountered";
        case CUSPARSE_STATUS_NOT_SUPPORTED:             return "the operation is not supported";
        case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:    return "insufficient resources";
    }
    return "unknown error";
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
    sp->format = CUMETAL_SPMAT_CSR;
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
    sp->format = CUMETAL_SPMAT_COO;
    *spMatDescr = sp;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t* spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void* cscColOffsets, void* cscRowInd,
                                    void* cscValues,
                                    cusparseIndexType_t cscColOffsetsType,
                                    cusparseIndexType_t cscRowIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType valueType) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    auto* sp = new cusparseSpMatDescr();
    // rows/cols stay the logical shape of A. The arrays are CSR-of-A-transpose:
    // the offset array is indexed by column and the index array holds row ids,
    // which is why the compressed axis below has `cols` entries.
    sp->rows = rows;
    sp->cols = cols;
    sp->nnz = nnz;
    sp->rowOffsets = cscColOffsets;
    sp->colInd = cscRowInd;
    sp->values = cscValues;
    sp->rowType = cscColOffsetsType;
    sp->colType = cscRowIndType;
    sp->idxBase = idxBase;
    sp->valueType = valueType;
    sp->format = CUMETAL_SPMAT_CSC;
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

extern "C++" {

// ── shared sparse kernels ───────────────────────────────────────────────────
//
// A compressed matrix is described by one offset array over a "compressed
// axis" plus one index array. For CSR that axis is the rows of A; for CSC it
// is the columns, which makes the very same arrays a CSR description of
// A-transpose. So there is one kernel here, not three: the caller resolves
// (format, opA) into a single `transpose` flag over the CSR view.
//
// The two directions cannot share a loop. Non-transpose gathers along the
// compressed axis and writes each output once. Transpose scatters into
// arbitrary output positions, so y must be scaled by beta up front and then
// accumulated into.

template <typename T>
static void cumetal_spmv_compressed(int64_t axis, const int* offsets, const int* indices,
                                    const T* vals, int base, bool transpose,
                                    T alpha, T beta,
                                    const T* x, T* y, int64_t ylen) {
    if (!transpose) {
        for (int64_t i = 0; i < axis; ++i) {
            T sum = static_cast<T>(0);
            const int begin = offsets[i] - base;
            const int end = offsets[i + 1] - base;
            for (int k = begin; k < end; ++k) sum += vals[k] * x[indices[k] - base];
            y[i] = alpha * sum + beta * y[i];
        }
        return;
    }
    for (int64_t i = 0; i < ylen; ++i) y[i] = beta * y[i];
    for (int64_t i = 0; i < axis; ++i) {
        const T xi = alpha * x[i];
        const int begin = offsets[i] - base;
        const int end = offsets[i + 1] - base;
        for (int k = begin; k < end; ++k) y[indices[k] - base] += vals[k] * xi;
    }
}

template <typename T>
static void cumetal_spmv_coo(int64_t nnz, const int* rowInd, const int* colInd,
                             const T* vals, int base, bool transpose,
                             T alpha, T beta,
                             const T* x, T* y, int64_t ylen) {
    for (int64_t i = 0; i < ylen; ++i) y[i] = beta * y[i];
    for (int64_t e = 0; e < nnz; ++e) {
        const int r = rowInd[e] - base;
        const int c = colInd[e] - base;
        if (transpose) y[c] += alpha * vals[e] * x[r];
        else           y[r] += alpha * vals[e] * x[c];
    }
}

template <typename T>
static void cumetal_spmm_compressed(int64_t axis, const int* offsets, const int* indices,
                                    const T* vals, int base, bool transpose, int64_t n,
                                    T alpha, T beta,
                                    const T* B, int64_t ldb, T* C, int64_t ldc,
                                    int64_t crows) {
    if (!transpose) {
        for (int64_t i = 0; i < axis; ++i) {
            const int begin = offsets[i] - base;
            const int end = offsets[i + 1] - base;
            for (int64_t j = 0; j < n; ++j) {
                T sum = static_cast<T>(0);
                for (int k = begin; k < end; ++k)
                    sum += vals[k] * B[indices[k] - base + j * ldb];
                C[i + j * ldc] = alpha * sum + beta * C[i + j * ldc];
            }
        }
        return;
    }
    for (int64_t i = 0; i < crows; ++i)
        for (int64_t j = 0; j < n; ++j) C[i + j * ldc] = beta * C[i + j * ldc];
    for (int64_t i = 0; i < axis; ++i) {
        const int begin = offsets[i] - base;
        const int end = offsets[i + 1] - base;
        for (int64_t j = 0; j < n; ++j) {
            const T bij = alpha * B[i + j * ldb];
            for (int k = begin; k < end; ++k)
                C[indices[k] - base + j * ldc] += vals[k] * bij;
        }
    }
}

template <typename T>
static void cumetal_spmm_coo(int64_t nnz, const int* rowInd, const int* colInd,
                             const T* vals, int base, bool transpose, int64_t n,
                             T alpha, T beta,
                             const T* B, int64_t ldb, T* C, int64_t ldc, int64_t crows) {
    for (int64_t i = 0; i < crows; ++i)
        for (int64_t j = 0; j < n; ++j) C[i + j * ldc] = beta * C[i + j * ldc];
    for (int64_t e = 0; e < nnz; ++e) {
        const int r = rowInd[e] - base;
        const int c = colInd[e] - base;
        const int out = transpose ? c : r;
        const int in = transpose ? r : c;
        for (int64_t j = 0; j < n; ++j)
            C[out + j * ldc] += alpha * vals[e] * B[in + j * ldb];
    }
}

// Resolve (storage format, requested operation) into one flag over the CSR
// view, plus the length of that view's compressed axis.
static void cumetal_sparse_view(const cusparseSpMatDescr* mat, cusparseOperation_t op,
                                bool* transpose, int64_t* axis) {
    bool t = (op != CUSPARSE_OPERATION_NON_TRANSPOSE);
    if (mat->format == CUMETAL_SPMAT_CSC) {
        // The arrays are CSR-of-A-transpose, so every operation flips and the
        // compressed axis is A's column count.
        t = !t;
        *axis = mat->cols;
    } else {
        *axis = mat->rows;
    }
    *transpose = t;
}

// The kernels above index with `int`. Reading a 64-bit index array through an
// int* would read half of each entry, so refuse rather than compute garbage.
static bool cumetal_sparse_indices_are_32bit(const cusparseSpMatDescr* mat) {
    return mat->rowType == CUSPARSE_INDEX_32I && mat->colType == CUSPARSE_INDEX_32I;
}

}  // extern "C++"

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
    if (!cumetal_sparse_indices_are_32bit(matA)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }

    // op(A) is m-by-k, so y has m entries and x has k.
    const bool op_t = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t ylen = op_t ? matA->cols : matA->rows;
    const int64_t xlen = op_t ? matA->rows : matA->cols;
    if (vecY->size != ylen || vecX->size != xlen) return CUSPARSE_STATUS_INVALID_VALUE;

    // Order this call after prior work on the handle's stream (see
    // synchronize_handle_stream) before computing on the CPU.
    synchronize_handle_stream(handle);

    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int* offsets = static_cast<const int*>(matA->rowOffsets);
    const int* indices = static_cast<const int*>(matA->colInd);

    bool transpose = false;
    int64_t axis = 0;
    cumetal_sparse_view(matA, opA, &transpose, &axis);

    if (computeType == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double b = *static_cast<const double*>(beta);
        const double* vals = static_cast<const double*>(matA->values);
        const double* x = static_cast<const double*>(vecX->values);
        double* y = static_cast<double*>(vecY->values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmv_coo(matA->nnz, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
        else
            cumetal_spmv_compressed(axis, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        const float* vals = static_cast<const float*>(matA->values);
        const float* x = static_cast<const float*>(vecX->values);
        float* y = static_cast<float*>(vecY->values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmv_coo(matA->nnz, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
        else
            cumetal_spmv_compressed(axis, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
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
                               cusparseOperation_t opB,
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
    if (!cumetal_sparse_indices_are_32bit(matA)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    // The dense loops below index B and C as column-major with a leading
    // dimension and read B in its stored orientation.
    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (matB->order != CUSPARSE_ORDER_COL || matC->order != CUSPARSE_ORDER_COL) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }

    const bool op_t = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t m = op_t ? matA->cols : matA->rows;
    const int64_t k = op_t ? matA->rows : matA->cols;
    const int64_t n = matB->cols;
    if (matB->rows != k || matC->rows != m || matC->cols != n) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    synchronize_handle_stream(handle);

    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int64_t ldb = matB->ld;
    const int64_t ldc = matC->ld;
    const int* offsets = static_cast<const int*>(matA->rowOffsets);
    const int* indices = static_cast<const int*>(matA->colInd);

    bool transpose = false;
    int64_t axis = 0;
    cumetal_sparse_view(matA, opA, &transpose, &axis);

    if (computeType == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double b = *static_cast<const double*>(beta);
        const double* vals = static_cast<const double*>(matA->values);
        const double* B = static_cast<const double*>(matB->values);
        double* C = static_cast<double*>(matC->values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmm_coo(matA->nnz, offsets, indices, vals, base, transpose, n, a, b, B, ldb, C, ldc, m);
        else
            cumetal_spmm_compressed(axis, offsets, indices, vals, base, transpose, n, a, b, B, ldb, C, ldc, m);
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        const float* vals = static_cast<const float*>(matA->values);
        const float* B = static_cast<const float*>(matB->values);
        float* C = static_cast<float*>(matC->values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmm_coo(matA->nnz, offsets, indices, vals, base, transpose, n, a, b, B, ldb, C, ldc, m);
        else
            cumetal_spmm_compressed(axis, offsets, indices, vals, base, transpose, n, a, b, B, ldb, C, ldc, m);
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
    synchronize_handle_stream(handle);

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
    synchronize_handle_stream(handle);

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
    if (matA->format != CUMETAL_SPMAT_CSR) return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F)
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;

    synchronize_handle_stream(handle);

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
