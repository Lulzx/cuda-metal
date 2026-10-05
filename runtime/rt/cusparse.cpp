#include "cusparse.h"
#include "cuda_runtime.h"

#include "metal_backend.h"
#include "runtime_internal.h"
#include "sparse_kernels_msl.h"
#include "library_kernel_source.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <numeric>
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
    cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
    // Longest run in the offset array, computed once on first use. The gather
    // kernel gives one thread per compressed row, so this is the serial depth
    // every other thread waits on. -1 means not yet measured.
    //
    // INVARIANT: a descriptor's sparsity structure is fixed for its lifetime.
    // Values may be rewritten in place, which is what a scaling pass does, but
    // rowOffsets and colInd may not. Any entry point added later that repoints
    // or mutates those arrays must reset this to -1, or the dispatch decision
    // will be made from a stale shape.
    std::int64_t longest_row = -1;
};

// Incomplete-factorization state. Positions are stored zero-based (-1 = none)
// and converted to the matrix's index base only when reported.
struct csrilu02Info {
    int zero_pivot = -1;
    int base = 0;
    bool boost_enabled = false;
    double boost_tol = 0.0;
    std::complex<double> boost_value{0.0, 0.0};
};

struct csric02Info {
    int zero_pivot = -1;
    int base = 0;
};

// Block factorizations are not implemented; the info objects exist so that
// callers can create and destroy them, and every compute call refuses.
struct bsrilu02Info { char reserved = 0; };
struct bsric02Info { char reserved = 0; };

struct cusparseSpVecDescr {
    int64_t size = 0;
    int64_t nnz = 0;
    void* indices = nullptr;
    void* values = nullptr;
    cusparseIndexType_t idxType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType valueType = CUDA_R_32F;
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
    // Set through cusparseDnMatSetStridedBatch. The product kernels address one
    // matrix only, so those that cannot honour a batch refuse when this is > 1.
    int batchCount = 1;
    int64_t batchStride = 0;
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
// and quietly return zeros. On the default stream the call is also ordered
// after every blocking stream (synchronize_for_host_library).
static void synchronize_handle_stream(cusparseHandle_t handle) {
    if (handle == nullptr) return;
    cumetal::rt::synchronize_for_host_library(handle->stream);
}

static const void* scalar_pointer_for_mode_size(cusparsePointerMode_t mode,
                                                const void* pointer,
                                                std::size_t size) {
    if (pointer == nullptr) return nullptr;
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    const bool tracked =
        cumetal::rt::resolve_allocation_for_pointer(pointer, &resolved);
    const bool is_device =
        tracked && resolved.kind == cumetal::rt::AllocationKind::kDevice;
    if (mode == CUSPARSE_POINTER_MODE_HOST) {
        return is_device ? nullptr : pointer;
    }
    if (mode != CUSPARSE_POINTER_MODE_DEVICE || !is_device ||
        resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
        resolved.remaining_size < size) {
        return nullptr;
    }
    return static_cast<const void*>(
        static_cast<const unsigned char*>(resolved.buffer->contents()) +
        resolved.offset);
}

static const void* scalar_pointer_for_mode(cusparsePointerMode_t mode,
                                           const void* pointer,
                                           cudaDataType type) {
    if (type == CUDA_R_64F) {
        return scalar_pointer_for_mode_size(mode, pointer, sizeof(double));
    }
    if (type == CUDA_R_32F) {
        return scalar_pointer_for_mode_size(mode, pointer, sizeof(float));
    }
    return nullptr;
}

static bool valid_operation(cusparseOperation_t operation) {
    return operation == CUSPARSE_OPERATION_NON_TRANSPOSE ||
           operation == CUSPARSE_OPERATION_TRANSPOSE ||
           operation == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
}

static bool valid_spmv_algorithm(cusparseSpMVAlg_t algorithm) {
    return algorithm == CUSPARSE_SPMV_ALG_DEFAULT ||
           algorithm == CUSPARSE_SPMV_COO_ALG1 ||
           algorithm == CUSPARSE_SPMV_CSR_ALG1 ||
           algorithm == CUSPARSE_SPMV_CSR_ALG2 ||
           algorithm == CUSPARSE_SPMV_COO_ALG2;
}

// A COO algorithm names a storage format just as a CSR one does. All of them
// compute the same product here, so the only question is whether the request
// is coherent: CSR_ALG* on a COO matrix (or COO_ALG* on a compressed one) is
// the caller asking for something cuSPARSE itself refuses.
static bool spmv_algorithm_fits_format(cusparseSpMVAlg_t algorithm, SpMatFormat format) {
    const bool coo_alg = algorithm == CUSPARSE_SPMV_COO_ALG1 ||
                         algorithm == CUSPARSE_SPMV_COO_ALG2;
    const bool csr_alg = algorithm == CUSPARSE_SPMV_CSR_ALG1 ||
                         algorithm == CUSPARSE_SPMV_CSR_ALG2;
    return format == CUMETAL_SPMAT_COO ? !csr_alg : !coo_alg;
}

static bool valid_spmm_algorithm(cusparseSpMMAlg_t algorithm) {
    return algorithm == CUSPARSE_SPMM_ALG_DEFAULT ||
           algorithm == CUSPARSE_SPMM_COO_ALG1 ||
           algorithm == CUSPARSE_SPMM_COO_ALG2 ||
           algorithm == CUSPARSE_SPMM_COO_ALG3 ||
           algorithm == CUSPARSE_SPMM_CSR_ALG1 ||
           algorithm == CUSPARSE_SPMM_COO_ALG4 ||
           algorithm == CUSPARSE_SPMM_CSR_ALG2 ||
           algorithm == CUSPARSE_SPMM_CSR_ALG3;
}

static bool valid_index_type(cusparseIndexType_t type) {
    return type == CUSPARSE_INDEX_16U || type == CUSPARSE_INDEX_32I ||
           type == CUSPARSE_INDEX_64I;
}

static bool valid_index_base(cusparseIndexBase_t base) {
    return base == CUSPARSE_INDEX_BASE_ZERO || base == CUSPARSE_INDEX_BASE_ONE;
}

static bool valid_data_type(cudaDataType type) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_C_16F:
        case CUDA_R_16BF:
        case CUDA_C_16BF:
        case CUDA_R_32F:
        case CUDA_C_32F:
        case CUDA_R_64F:
        case CUDA_C_64F:
        case CUDA_R_8I:
        case CUDA_R_8U:
        case CUDA_R_32I:
            return true;
    }
    return false;
}

static bool valid_spmv_types(const cusparseSpMatDescr* matrix,
                             const cusparseDnVecDescr* x,
                             const cusparseDnVecDescr* y,
                             cudaDataType compute_type) {
    return matrix != nullptr && x != nullptr && y != nullptr &&
           (compute_type == CUDA_R_32F || compute_type == CUDA_R_64F) &&
           matrix->valueType == compute_type && x->valueType == compute_type &&
           y->valueType == compute_type;
}

static bool valid_spmm_types(const cusparseSpMatDescr* matrix,
                             const cusparseDnMatDescr* b,
                             const cusparseDnMatDescr* c,
                             cudaDataType compute_type) {
    return matrix != nullptr && b != nullptr && c != nullptr &&
           (compute_type == CUDA_R_32F || compute_type == CUDA_R_64F) &&
           matrix->valueType == compute_type && b->valueType == compute_type &&
           c->valueType == compute_type;
}

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle) {
    if (handle == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *handle = new (std::nothrow) cusparseContext();
    if (*handle == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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
    *version = CUSPARSE_VERSION;
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
    *descrA = new (std::nothrow) cusparseMatDescr();
    if (*descrA == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descrA) {
    delete descrA;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (type != CUSPARSE_MATRIX_TYPE_GENERAL &&
        type != CUSPARSE_MATRIX_TYPE_SYMMETRIC &&
        type != CUSPARSE_MATRIX_TYPE_HERMITIAN &&
        type != CUSPARSE_MATRIX_TYPE_TRIANGULAR) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    descrA->type = type;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseMatrixType_t cusparseGetMatType(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->type : CUSPARSE_MATRIX_TYPE_GENERAL;
}

cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (!valid_index_base(base)) return CUSPARSE_STATUS_INVALID_VALUE;
    descrA->base = base;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseIndexBase_t cusparseGetMatIndexBase(const cusparseMatDescr_t descrA) {
    return descrA ? descrA->base : CUSPARSE_INDEX_BASE_ZERO;
}

cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (fillMode != CUSPARSE_FILL_MODE_LOWER &&
        fillMode != CUSPARSE_FILL_MODE_UPPER) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    descrA->fill = fillMode;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType) {
    if (descrA == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (diagType != CUSPARSE_DIAG_TYPE_NON_UNIT &&
        diagType != CUSPARSE_DIAG_TYPE_UNIT) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
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
    *spMatDescr = nullptr;
    if (rows < 0 || cols < 0 || nnz < 0 || csrRowOffsets == nullptr ||
        (nnz > 0 && (csrColInd == nullptr || csrValues == nullptr)) ||
        !valid_index_type(csrRowOffsetsType) ||
        !valid_index_type(csrColIndType) || !valid_index_base(idxBase) ||
        !valid_data_type(valueType)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* sp = new (std::nothrow) cusparseSpMatDescr();
    if (sp == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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
    *spMatDescr = nullptr;
    if (rows < 0 || cols < 0 || nnz < 0 ||
        (nnz > 0 &&
         (cooRowInd == nullptr || cooColInd == nullptr || cooValues == nullptr)) ||
        !valid_index_type(cooIdxType) || !valid_index_base(idxBase) ||
        !valid_data_type(valueType)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* sp = new (std::nothrow) cusparseSpMatDescr();
    if (sp == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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
    *spMatDescr = nullptr;
    if (rows < 0 || cols < 0 || nnz < 0 || cscColOffsets == nullptr ||
        (nnz > 0 && (cscRowInd == nullptr || cscValues == nullptr)) ||
        !valid_index_type(cscColOffsetsType) ||
        !valid_index_type(cscRowIndType) || !valid_index_base(idxBase) ||
        !valid_data_type(valueType)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* sp = new (std::nothrow) cusparseSpMatDescr();
    if (sp == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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

cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                           cusparseSpMatAttribute_t attribute,
                                           const void* data, size_t dataSize) {
    if (spMatDescr == nullptr || data == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (attribute == CUSPARSE_SPMAT_FILL_MODE) {
        if (dataSize != sizeof(cusparseFillMode_t)) return CUSPARSE_STATUS_INVALID_VALUE;
        const auto value = *static_cast<const cusparseFillMode_t*>(data);
        if (value != CUSPARSE_FILL_MODE_LOWER && value != CUSPARSE_FILL_MODE_UPPER)
            return CUSPARSE_STATUS_INVALID_VALUE;
        spMatDescr->fill = value;
        return CUSPARSE_STATUS_SUCCESS;
    }
    if (attribute == CUSPARSE_SPMAT_DIAG_TYPE) {
        if (dataSize != sizeof(cusparseDiagType_t)) return CUSPARSE_STATUS_INVALID_VALUE;
        const auto value = *static_cast<const cusparseDiagType_t*>(data);
        if (value != CUSPARSE_DIAG_TYPE_NON_UNIT && value != CUSPARSE_DIAG_TYPE_UNIT)
            return CUSPARSE_STATUS_INVALID_VALUE;
        spMatDescr->diag = value;
        return CUSPARSE_STATUS_SUCCESS;
    }
    return CUSPARSE_STATUS_INVALID_VALUE;
}

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t* info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) csrilu02Info();
    if (*info == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

// The csrilu02 / csric02 entry points (all four precisions) are defined with
// the other incomplete-factorization code at the end of this file.

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t* dnVecDescr,
                                      int64_t size, void* values, cudaDataType valueType) {
    if (dnVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *dnVecDescr = nullptr;
    if (size < 0 || (size > 0 && values == nullptr) ||
        !valid_data_type(valueType)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* v = new (std::nothrow) cusparseDnVecDescr();
    if (v == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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

cusparseStatus_t cusparseDnVecGetValues(cusparseDnVecDescr_t dnVecDescr, void** values) {
    if (dnVecDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = dnVecDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

// Repoint a dense vector descriptor at a different buffer, keeping its length
// and element type. Callers that alternate between buffers use this to avoid
// building a descriptor per call.
//
// Nothing derived from `values` is cached on this descriptor, so there is
// nothing to invalidate. That is not true of cusparseSpMatDescr, which caches
// longest_row: see the INVARIANT on it before adding anything that repoints a
// sparse descriptor's arrays.
cusparseStatus_t cusparseDnVecSetValues(cusparseDnVecDescr_t dnVecDescr, void* values) {
    if (dnVecDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    dnVecDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnVecGet(cusparseDnVecDescr_t dnVecDescr,
                                   int64_t* size, void** values, cudaDataType* valueType) {
    if (dnVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (size != nullptr) *size = dnVecDescr->size;
    if (values != nullptr) *values = dnVecDescr->values;
    if (valueType != nullptr) *valueType = dnVecDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t* dnMatDescr,
                                      int64_t rows, int64_t cols, int64_t ld,
                                      void* values, cudaDataType valueType,
                                      cusparseOrder_t order) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *dnMatDescr = nullptr;
    if (rows < 0 || cols < 0 || ld < 0 ||
        (rows > 0 && cols > 0 && values == nullptr) ||
        !valid_data_type(valueType) ||
        (order != CUSPARSE_ORDER_COL && order != CUSPARSE_ORDER_ROW) ||
        (rows > 0 && cols > 0 &&
         ((order == CUSPARSE_ORDER_COL && ld < rows) ||
          (order == CUSPARSE_ORDER_ROW && ld < cols)))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* m = new (std::nothrow) cusparseDnMatDescr();
    if (m == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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
                                          cusparseOperation_t opA,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void* beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpMVAlg_t alg,
                                          size_t* bufferSize) {
    if (!handle || !alpha || !matA || !vecX || !beta || !vecY || !bufferSize) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!valid_spmv_types(matA, vecX, vecY, computeType)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!valid_operation(opA) || !valid_spmv_algorithm(alg)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (!spmv_algorithm_fits_format(alg, matA->format)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType) == nullptr ||
        scalar_pointer_for_mode(handle->pointer_mode, beta, computeType) == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
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

extern "C++" {

// ── type plumbing for the typed (S/D/C/Z) entry points ──────────────────────
//
// The public API spells complex values as cuComplex / cuDoubleComplex. They
// have the layout of std::complex, so one set of templates serves all four
// precisions and the C wrappers only reinterpret pointers.
template <typename C> struct NativeOf { using type = C; };
template <> struct NativeOf<cuComplex> { using type = std::complex<float>; };
template <> struct NativeOf<cuDoubleComplex> { using type = std::complex<double>; };
template <typename C> using native_t = typename NativeOf<C>::type;

static_assert(sizeof(std::complex<float>) == sizeof(cuComplex), "cuComplex layout");
static_assert(sizeof(std::complex<double>) == sizeof(cuDoubleComplex), "cuDoubleComplex layout");

template <typename C> static native_t<C>* nat(C* p) {
    return reinterpret_cast<native_t<C>*>(p);
}
template <typename C> static const native_t<C>* cnat(const C* p) {
    return reinterpret_cast<const native_t<C>*>(p);
}
inline float to_native(float v) { return v; }
inline double to_native(double v) { return v; }
inline std::complex<float> to_native(cuComplex v) { return {v.x, v.y}; }
inline std::complex<double> to_native(cuDoubleComplex v) { return {v.x, v.y}; }

template <typename T> inline double mag(const T& v) { return std::abs(v); }
template <typename T> inline T conj_of(const T& v) { return v; }
template <typename T> inline std::complex<T> conj_of(const std::complex<T>& v) {
    return std::conj(v);
}
template <typename T> inline double real_part(const T& v) { return static_cast<double>(v); }
template <typename T> inline double real_part(const std::complex<T>& v) { return v.real(); }

// ── index arrays of any cusparseIndexType_t ─────────────────────────────────
static int64_t idx_at(const void* p, cusparseIndexType_t t, int64_t i) {
    switch (t) {
        case CUSPARSE_INDEX_16U: return static_cast<const std::uint16_t*>(p)[i];
        case CUSPARSE_INDEX_64I: return static_cast<const std::int64_t*>(p)[i];
        default: return static_cast<const std::int32_t*>(p)[i];
    }
}

static void idx_put(void* p, cusparseIndexType_t t, int64_t i, int64_t v) {
    switch (t) {
        case CUSPARSE_INDEX_16U: static_cast<std::uint16_t*>(p)[i] = static_cast<std::uint16_t>(v); break;
        case CUSPARSE_INDEX_64I: static_cast<std::int64_t*>(p)[i] = v; break;
        default: static_cast<std::int32_t*>(p)[i] = static_cast<std::int32_t>(v); break;
    }
}

// Solve R^t * x = rhs in place, where R is the matrix the CSR arrays describe
// and `fill` names the triangle of R that holds data. Entries in the other
// triangle are ignored, as in cuSPARSE.
//
// Non-transpose gathers along rows. Transpose cannot: R^t's rows are R's
// columns, which CSR cannot walk. It scatters instead -- once x[i] is final,
// subtract its contribution from every equation that still needs it. The
// direction flips with the transpose (R lower => R^t upper => solve backward).
//
// Returns false on a zero or missing diagonal entry.
template <typename T>
static bool cumetal_tri_solve(int64_t n, const int* offsets, const int* indices, const T* vals,
                              int base, cusparseFillMode_t fill, bool unit_diag,
                              bool transpose, bool conjugate, T* x) {
    auto value = [&](int k) { return conjugate ? conj_of(vals[k]) : vals[k]; };
    const bool forward = transpose ? (fill == CUSPARSE_FILL_MODE_UPPER)
                                   : (fill == CUSPARSE_FILL_MODE_LOWER);
    for (int64_t step = 0; step < n; ++step) {
        const int64_t i = forward ? step : n - 1 - step;
        const int begin = offsets[i] - base;
        const int end = offsets[i + 1] - base;
        T diag = static_cast<T>(1);
        bool have_diag = unit_diag;
        T acc = x[i];
        for (int k = begin; k < end; ++k) {
            const int64_t c = indices[k] - base;
            if (c == i) {
                if (!unit_diag) {
                    diag = value(k);
                    have_diag = true;
                }
            } else if (!transpose && (forward ? c < i : c > i)) {
                acc -= value(k) * x[c];
            }
        }
        if (!have_diag || diag == static_cast<T>(0)) return false;
        x[i] = acc / diag;
        if (transpose) {
            for (int k = begin; k < end; ++k) {
                const int64_t c = indices[k] - base;
                if (c != i && (forward ? c > i : c < i)) x[c] -= value(k) * x[i];
            }
        }
    }
    return true;
}

}  // extern "C++"


extern "C++" {

// ── Metal-native SpMV ───────────────────────────────────────────────────────
//
// Only the gather shape runs on the GPU: one output element per compressed row.
// CSR non-transpose and CSC transpose both reduce to that same loop, and those
// are the two products a PDLP iteration is built from (Ax and A'y). The scatter
// shapes would need atomic accumulation into y, and Metal has no FP64 atomic,
// so they stay on the CPU path below. cusparseSpMV still honours any opA; only
// which implementation serves it changes.
//
// Falling back is always allowed: every precondition here is a capability
// question, not a correctness one, and the CPU path computes the same product.

struct SpmvParams {
    std::uint32_t axis;
    std::int32_t base;
    std::uint32_t beta_is_zero;
    std::uint32_t pad;
    std::uint64_t alpha_bits;
    std::uint64_t beta_bits;
};
static_assert(sizeof(SpmvParams) == 32, "SpmvParams must match the MSL layout");

const std::string* sparse_kernels_source_path() {
    return cumetal::rt::stage_library_kernel_source("sparse_kernels",
                                                    cumetal::rt::kSparseKernelsMsl);
}

// CUMETAL_SPARSE_METAL: unset = auto, "1" = always, "0" = never. Matches the
// CUMETAL_MTLHEAP_ALLOC convention.
enum class SparseMetalPolicy { kAuto, kAlways, kNever };

SparseMetalPolicy sparse_metal_policy() {
    static const SparseMetalPolicy policy = [] {
        const char* v = std::getenv("CUMETAL_SPARSE_METAL");
        if (v == nullptr || v[0] == '\0') return SparseMetalPolicy::kAuto;
        if (v[0] == '0') return SparseMetalPolicy::kNever;
        return SparseMetalPolicy::kAlways;
    }();
    return policy;
}

// One thread per compressed row means the longest row is a serial loop no amount
// of parallelism hides, so that length, not the average, is what can defeat the
// kernel. datt256 from the Mittelmann set carries a fully dense row: after
// cuPDLP reformulates it the longest run is 57840 against a mean of 136, and one
// thread grinds through it while the other 11076 idle. In the solver that made
// SpMV 3.3x slower than the CPU loop, even though a synthetic matrix of the same
// dimensions and mean row length ran 6x faster, which is exactly the case a
// uniform-row benchmark cannot see.
//
// Average density is not the discriminator: ex10's rows are also uneven relative
// to their mean (256 against 16.7, so under 7% of the work the scalar kernel
// schedules is useful) and it still runs 4.6x faster, because 69608 rows keep
// the GPU busy and a 256-element tail is short.
//
// So the bound below is on serial depth alone, and it is not a fence around the
// GPU: the cooperative kernel divides that depth by the simdgroup width, which
// makes 4096 the point where one kernel hands off to the other, and only past
// 32x4096 the point where the CPU takes the work back.
constexpr std::int64_t kMaxGatherSerialDepth = 4096;

// Apple GPUs execute 32 threads per simdgroup. The cooperative kernel reads the
// real width from the dispatch and strides over rows, so this is only used to
// estimate what that kernel would cost; being wrong here costs a routing
// decision, never a wrong answer.
constexpr std::int64_t kAssumedSimdWidth = 32;

// Threads the GPU keeps in flight, used only to weigh a kernel's depth against
// its total work. Fitted to the measurements below rather than read from the
// device: any value from a few thousand up routes every measured case the same
// way, so the model is not sensitive to it.
constexpr std::int64_t kEffectiveParallelThreads = 8192;

enum class GatherKernel { kNone, kScalar, kSimd };

std::int64_t longest_row(const cusparseSpMatDescr* mat, std::int64_t axis) {
    if (mat->longest_row < 0) {
        const int* offsets = static_cast<const int*>(mat->rowOffsets);
        if (offsets == nullptr) return -1;
        std::int64_t longest = 0;
        for (std::int64_t i = 0; i < axis; ++i) {
            // A difference of offsets, so the index base cancels.
            const std::int64_t len = static_cast<std::int64_t>(offsets[i + 1]) - offsets[i];
            if (len > longest) longest = len;
        }
        const_cast<cusparseSpMatDescr*>(mat)->longest_row = longest;  // see INVARIANT
    }
    return mat->longest_row;
}

// CUMETAL_SPARSE_METAL_KERNEL: unset = choose from the row distribution,
// "scalar" or "simd" to pin one. Pinning exists so a test can exercise a kernel
// the heuristic would never route to it: without it the cooperative path would
// only ever run on matrices too large for a conformance test to hold.
const char* pinned_gather_kernel() {
    static const char* pinned = [] {
        const char* v = std::getenv("CUMETAL_SPARSE_METAL_KERNEL");
        if (v == nullptr || v[0] == '\0') return static_cast<const char*>(nullptr);
        return v;
    }();
    return pinned;
}

// Each kernel is costed as a makespan: how long the slowest thread runs, which
// is whichever of its depth and its share of the total work is larger.
//
//   scalar   max(L, nnz/P)
//   simd     max(L/W, max(nnz, W*rows)/P)
//
// L is the longest row. The scalar kernel walks it in one thread, so it is that
// kernel's depth outright; the cooperative kernel splits it across a simdgroup,
// so its depth is L/W. The second term is the throughput floor. The cooperative
// kernel's is the larger of the two because a row shorter than W leaves lanes
// idle, and a matrix of uniformly short rows pays for every one of them.
//
// Measured on an M4 Pro at a fixed 1.6M nonzeros, FP64, synchronizing per call,
// uniform rows so the two terms are the throughput ones (us per SpMV):
//
//   row length      4      8     16     32     48     64    128    256    512
//   scalar        523    600    545    488    498    542    441    566    600
//   cooperative  1437    788    630    454    409    375    332    249    423
//
// The crossover sits at the simdgroup width, which is what the model says: below
// it the cooperative kernel is buying a depth reduction on rows that have no
// depth to give, and pays the idle lanes for it.
//
// And with one pathological row against a short remainder, which is the shape
// that motivated all of this (rows/length/longest -> us):
//
//   11078/136/57840   scalar 22911   cooperative 1805   cpu 1995
//   11078/136/16384   scalar  6242   cooperative  636   cpu 2543
//   400000/4/57840    scalar 29903   cooperative 2789   cpu 3147
//   400000/4/256      scalar   500   cooperative 1454   cpu 3018
//
// The last two are the same longest row deciding differently because the rest of
// the matrix differs, which is why the rule is not a bound on L alone.
GatherKernel cheaper_gather_kernel(std::int64_t longest, std::int64_t rows, std::int64_t nnz) {
    const std::int64_t P = kEffectiveParallelThreads;
    const std::int64_t W = kAssumedSimdWidth;
    const std::int64_t scalar_cost = std::max<std::int64_t>(longest, nnz / P);
    const std::int64_t simd_cost =
        std::max<std::int64_t>(longest / W, std::max<std::int64_t>(nnz, W * rows) / P);
    return simd_cost < scalar_cost ? GatherKernel::kSimd : GatherKernel::kScalar;
}

// Which gather kernel to run, or kNone to leave the product on the CPU. `why`
// receives the reason for kNone.
GatherKernel choose_gather_kernel(const cusparseSpMatDescr* mat, std::int64_t axis,
                                  SparseMetalPolicy policy, char* why, std::size_t why_size) {
    const std::int64_t longest = longest_row(mat, axis);
    if (longest < 0) {
        std::snprintf(why, why_size, "the descriptor has no row offsets");
        return GatherKernel::kNone;
    }
    if (const char* pinned = pinned_gather_kernel(); pinned != nullptr) {
        if (pinned[0] == 's' && pinned[1] == 'i') return GatherKernel::kSimd;
        if (pinned[0] == 's') return GatherKernel::kScalar;
        std::snprintf(why, why_size, "CUMETAL_SPARSE_METAL_KERNEL=%s is not a kernel name",
                      pinned);
        return GatherKernel::kNone;
    }
    if (longest == 0) {
        // Every row empty: the answer is a scale of y, and a dispatch to compute
        // it would cost more than the CPU loop that writes it.
        std::snprintf(why, why_size, "the matrix has no nonzeros in any row");
        return GatherKernel::kNone;
    }
    const GatherKernel variant = cheaper_gather_kernel(longest, axis, mat->nnz);
    if (policy == SparseMetalPolicy::kAlways) return variant;

    // Past this depth the CPU's own loop over unified memory wins, whichever
    // kernel would have run. Measured at the bound rather than assumed: a single
    // row of 131072 against a mean of 136 costs the cooperative kernel 4028us
    // against the CPU's 3646, and doubling it to 262144 costs 8266 against 4266.
    const std::int64_t depth = variant == GatherKernel::kSimd
                                   ? (longest + kAssumedSimdWidth - 1) / kAssumedSimdWidth
                                   : longest;
    if (depth <= kMaxGatherSerialDepth) return variant;
    std::snprintf(why, why_size,
                  "longest row %lld leaves the %s kernel a serial depth of %lld, past the "
                  "%lld bound (axis=%lld nnz=%lld)",
                  static_cast<long long>(longest),
                  variant == GatherKernel::kSimd ? "cooperative" : "scalar",
                  static_cast<long long>(depth),
                  static_cast<long long>(kMaxGatherSerialDepth),
                  static_cast<long long>(axis), static_cast<long long>(mat->nnz));
    return GatherKernel::kNone;
}

bool resolve_arg(const void* ptr,
                 std::size_t required_bytes,
                 std::size_t alignment,
                 cumetal::metal_backend::KernelArg* out) {
    return cumetal::rt::resolve_kernel_buffer_arg(ptr, required_bytes, alignment, out);
}

// The GPU path may decline for reasons that are capability questions (an
// unsupported shape) and for reasons that are defects (the kernel failed to
// compile). Both produce the same correct answer through the CPU path, which is
// exactly how a broken kernel stays invisible, so make the reason reportable.
bool spmv_debug() {
    static const bool on = [] {
        const char* v = std::getenv("CUMETAL_DEBUG_SPARSE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

void spmv_note(const char* reason) {
    if (spmv_debug()) std::fprintf(stderr, "CUMETAL_DEBUG_SPARSE: SpMV on CPU (%s)\n", reason);
}

// The GPU path taking a call is as much a routing decision as declining one, and
// a kernel that runs but was the wrong choice looks identical from outside. Say
// which one ran and on what evidence.
void spmv_note_gpu(const char* kernel, const cusparseSpMatDescr* mat, std::int64_t axis) {
    if (!spmv_debug()) return;
    std::fprintf(stderr, "CUMETAL_DEBUG_SPARSE: SpMV on %s (axis=%lld nnz=%lld longest_row=%lld)\n",
                 kernel, static_cast<long long>(axis), static_cast<long long>(mat->nnz),
                 static_cast<long long>(mat->longest_row));
}

// Below this many nonzeros the CPU loop over unified memory wins: a Metal
// dispatch costs on the order of 100 us, which buys a great many scalar
// multiply-adds. A conservative M4 Pro default rather than a property of the
// architecture: it moves with the chip, the element type, the row distribution,
// and any improvement to command submission or to the kernel itself. Measured
// with an SpMV microbenchmark that synchronizes per call, so it times completed
// work rather than enqueue:
//
//   nonzeros    3.2e4    1.3e5    5.1e5    2.0e6    8.2e6    3.2e7
//   Metal/CPU   0.99x    1.00x    3.72x    9.04x    7.48x    5.86x
//
// The crossover itself sits near 1e5 and is largely independent of row density,
// but the band around it is noisy enough that a threshold there wins or loses a
// few percent at random. This sits above that band, where the win is
// unambiguous. The first two columns are at the threshold's CPU side and show
// it costs nothing to route them there.
//
// Synchronizing per call also makes this conservative for a real pipeline, where
// several GPU operations queue behind one another and no single SpMV pays a
// host-visible synchronization. Erring toward the CPU unless the GPU win is
// obvious is the intended bias. The small Netlib instances in the HiGHS demo sit
// below this threshold, so auto mode keeps their sparse products on the CPU;
// large LPs can carry millions of nonzeros, which is where the GPU path earns
// its place.
std::int64_t sparse_metal_threshold_nnz() {
    static const std::int64_t threshold = [] {
        if (const char* v = std::getenv("CUMETAL_SPARSE_METAL_THRESHOLD_NNZ");
            v != nullptr && v[0] != '\0') {
            const long long parsed = std::atoll(v);
            if (parsed > 0) return static_cast<std::int64_t>(parsed);
        }
        return static_cast<std::int64_t>(250000);
    }();
    return threshold;
}

// Returns false when the GPU path did not run, for any reason.
bool try_spmv_gather_metal(cudaStream_t stream,
                           const cusparseSpMatDescr* mat,
                           bool transpose,
                           std::int64_t axis,
                           std::int64_t ylen,
                           const void* alpha,
                           const void* beta,
                           const void* x,
                           void* y,
                           cudaDataType compute_type) {
    const SparseMetalPolicy policy = sparse_metal_policy();
    if (policy == SparseMetalPolicy::kNever) {
        spmv_note("disabled by CUMETAL_SPARSE_METAL=0");
        return false;
    }
    if (transpose) { spmv_note("scatter shape needs atomics"); return false; }
    if (policy == SparseMetalPolicy::kAuto && mat->nnz < sparse_metal_threshold_nnz()) {
        spmv_note("below the measured nonzero threshold");
        return false;
    }
    char why[192] = {0};
    const GatherKernel variant = choose_gather_kernel(mat, axis, policy, why, sizeof(why));
    if (variant == GatherKernel::kNone) {
        spmv_note(why);
        return false;
    }
    // Whatever stream the handle carries is the stream this dispatch belongs on,
    // so that the SpMV is ordered against the caller's other work on it exactly
    // as a real cuSPARSE call would be. A null handle stream resolves to the
    // default stream, which is the same thing the CPU path below would wait on.
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (cumetal::rt::resolve_backend_stream(stream, &backend_stream) != cudaSuccess) {
        spmv_note("the handle's stream does not resolve to a backend stream");
        return false;
    }
    const std::size_t elem = compute_type == CUDA_R_64F ? 8u : 4u;
    const std::size_t nnz = static_cast<std::size_t>(mat->nnz);

    const std::string* source = sparse_kernels_source_path();
    if (source == nullptr) { spmv_note("could not stage the kernel source"); return false; }

    std::vector<cumetal::metal_backend::KernelArg> args(6);
    if (!resolve_arg(mat->rowOffsets, (static_cast<std::size_t>(axis) + 1) * sizeof(int),
                     sizeof(int), &args[0]) ||
        !resolve_arg(mat->colInd, nnz * sizeof(int), sizeof(int), &args[1]) ||
        !resolve_arg(mat->values, nnz * elem, elem, &args[2]) ||
        !resolve_arg(x, elem, elem, &args[3]) ||
        !resolve_arg(y, static_cast<std::size_t>(ylen) * elem, elem, &args[4])) {
        spmv_note("an operand is not a tracked device allocation, or is misaligned");
        return false;
    }

    SpmvParams params{};
    params.axis = static_cast<std::uint32_t>(axis);
    params.base = mat->idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    if (compute_type == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double b = *static_cast<const double*>(beta);
        params.beta_is_zero = b == 0.0 ? 1u : 0u;
        std::memcpy(&params.alpha_bits, &a, sizeof(a));
        std::memcpy(&params.beta_bits, &b, sizeof(b));
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        params.beta_is_zero = b == 0.0f ? 1u : 0u;
        std::uint32_t abits = 0, bbits = 0;
        std::memcpy(&abits, &a, sizeof(a));
        std::memcpy(&bbits, &b, sizeof(b));
        params.alpha_bits = abits;
        params.beta_bits = bbits;
    }
    args[5].kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
    args[5].bytes.resize(sizeof(params));
    std::memcpy(args[5].bytes.data(), &params, sizeof(params));

    constexpr unsigned kBlock = 256;
    cumetal::metal_backend::LaunchConfig config{};
    config.block = dim3(kBlock, 1, 1);
    config.shared_memory_bytes = 0;
    // The FP64 library kernels use the same FP32-pair arithmetic as generic
    // translated FP64.  `library_substitution` alone defaults to exact in the
    // backend because many library kernels really are exact; override it for
    // this typed call so provenance does not claim IEEE binary64 arithmetic.
    config.semantic_quality = compute_type == CUDA_R_64F
                                  ? "reduced_precision_fp64"
                                  : "exact";
    const char* kernel = nullptr;
    if (variant == GatherKernel::kScalar) {
        // One thread per compressed row.
        config.grid = dim3(static_cast<unsigned>((axis + kBlock - 1) / kBlock), 1, 1);
        kernel = compute_type == CUDA_R_64F ? "cumetal_spmv_gather_f64"
                                            : "cumetal_spmv_gather_f32";
    } else {
        // One simdgroup per row. Sized so that a 32-wide simdgroup gives each
        // one a single row; on any other width the kernel's grid stride picks
        // up the remainder rather than dropping it.
        const std::int64_t rows_per_group = kBlock / kAssumedSimdWidth;
        config.grid = dim3(static_cast<unsigned>((axis + rows_per_group - 1) / rows_per_group),
                           1, 1);
        kernel = compute_type == CUDA_R_64F ? "cumetal_spmv_gather_simd_f64"
                                            : "cumetal_spmv_gather_simd_f32";
    }

    std::string error;
    if (cumetal::metal_backend::launch_kernel(*source, kernel, config, args, backend_stream,
                                              &error) != cudaSuccess) {
        spmv_note(error.empty() ? "launch failed" : error.c_str());
        return false;
    }
    spmv_note_gpu(kernel, mat, axis);
    // Profiling aid. Leaving the launch async is correct and is what makes the
    // GPU path worth having, but it also means a caller's own SpMV timers
    // measure enqueue rather than execution, which is easy to misread as a
    // speedup. Setting this attributes the real cost back to the call.
    static const bool sync_for_profiling = [] {
        const char* v = std::getenv("CUMETAL_SPARSE_SYNC");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (sync_for_profiling) {
        cumetal::metal_backend::synchronize(&error);
    }
    return true;
}

}  // extern "C++"

// cuSPARSE lets a caller pay a one-time analysis cost so the SpMV calls that
// follow are cheaper. It is optional there and it is optional here: skipping it
// changes speed, not results.
//
// There is real work to do, though, and making this a bare no-op would waste it.
// Which kernel serves a matrix is decided from its longest row, which costs a
// pass over the offsets, and the first SpMV on a descriptor otherwise pays for
// that pass itself. Doing it here moves the cost to where the caller asked for
// it. Failing to do it is not an error -- the matrix may be a shape the GPU path
// declines anyway -- so this reports success as long as the arguments are
// well-formed.
cusparseStatus_t cusparseSpMV_preprocess(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void* alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void* beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType computeType,
                                          cusparseSpMVAlg_t alg,
                                          void* /*externalBuffer*/) {
    if (!handle || !matA || !vecX || !vecY || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!valid_spmv_types(matA, vecX, vecY, computeType)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!valid_operation(opA) || !valid_spmv_algorithm(alg)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (!spmv_algorithm_fits_format(alg, matA->format)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType) == nullptr ||
        scalar_pointer_for_mode(handle->pointer_mode, beta, computeType) == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (matA->format != CUMETAL_SPMAT_COO && cumetal_sparse_indices_are_32bit(matA)) {
        bool transpose = false;
        int64_t axis = 0;
        cumetal_sparse_view(matA, opA, &transpose, &axis);
        (void)longest_row(matA, axis);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

static cusparseStatus_t spmv_dispatch(cudaStream_t stream,
                                      cusparsePointerMode_t pointer_mode,
                                      cusparseOperation_t opA,
                                      const void* alpha,
                                      cusparseSpMatDescr_t matA,
                                      const void* x_values,
                                      const void* beta,
                                      void* y_values,
                                      cudaDataType computeType);

cusparseStatus_t cusparseSpMV(cusparseHandle_t handle,
                               cusparseOperation_t opA,
                               const void* alpha,
                               cusparseSpMatDescr_t matA,
                               cusparseDnVecDescr_t vecX,
                               const void* beta,
                               cusparseDnVecDescr_t vecY,
                               cudaDataType computeType,
                               cusparseSpMVAlg_t alg,
                               void* /*externalBuffer*/) {
    if (!handle || !matA || !vecX || !vecY || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!valid_spmv_types(matA, vecX, vecY, computeType)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!valid_operation(opA) || !valid_spmv_algorithm(alg)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (!spmv_algorithm_fits_format(alg, matA->format)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!cumetal_sparse_indices_are_32bit(matA)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }

    const std::size_t scalar_size = computeType == CUDA_R_64F ? 8u : 4u;
    const void* resolved_alpha =
        scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType);
    const void* resolved_beta =
        scalar_pointer_for_mode(handle->pointer_mode, beta, computeType);
    if (resolved_alpha == nullptr || resolved_beta == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    // op(A) is m-by-k, so y has m entries and x has k.
    const bool op_t = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t ylen = op_t ? matA->cols : matA->rows;
    const int64_t xlen = op_t ? matA->rows : matA->cols;
    if (vecY->size != ylen || vecX->size != xlen) return CUSPARSE_STATUS_INVALID_VALUE;

    // Under stream capture the call must be recorded, not performed. What gets
    // recorded is the arguments as they stand now: the vector descriptors are
    // read here and their pointers baked into the node, which is what CUDA does
    // and what makes cusparseDnVecSetValues between replays a no-op on an
    // already-captured graph rather than a surprise.
    //
    // The matrix descriptor is held by pointer instead, because its structure is
    // fixed for its lifetime (see the INVARIANT on cusparseSpMatDescr) and its
    // values are meant to be rewritable in place between replays -- a scaling
    // pass does exactly that.
    if (handle->stream != nullptr) {
        struct Scalar { unsigned char bytes[8]; };
        Scalar a{}, b{};
        if (handle->pointer_mode == CUSPARSE_POINTER_MODE_HOST) {
            std::memcpy(a.bytes, resolved_alpha, scalar_size);
            std::memcpy(b.bytes, resolved_beta, scalar_size);
        }
        void* x = vecX->values;
        void* y = vecY->values;
        const cusparsePointerMode_t pointer_mode = handle->pointer_mode;
        if (cumetal::rt::capture_library_call(
                handle->stream, [=](cudaStream_t replay_stream) {
                    const void* captured_alpha =
                        pointer_mode == CUSPARSE_POINTER_MODE_HOST
                            ? static_cast<const void*>(a.bytes)
                            : alpha;
                    const void* captured_beta =
                        pointer_mode == CUSPARSE_POINTER_MODE_HOST
                            ? static_cast<const void*>(b.bytes)
                            : beta;
                    return spmv_dispatch(replay_stream, pointer_mode, opA,
                                         captured_alpha, matA, x, captured_beta,
                                         y, computeType) == CUSPARSE_STATUS_SUCCESS
                               ? cudaSuccess
                               : cudaErrorInvalidValue;
                })) {
            return CUSPARSE_STATUS_SUCCESS;
        }
    }

    return spmv_dispatch(handle->stream, handle->pointer_mode, opA, alpha, matA,
                         vecX->values, beta, vecY->values, computeType);
}

// Runs one SpMV over already-resolved operands. Split out from cusparseSpMV so
// that a graph node can replay it on whatever stream the graph was launched on,
// with the arguments it was captured with.
static cusparseStatus_t spmv_dispatch(cudaStream_t stream,
                                      cusparsePointerMode_t pointer_mode,
                                      cusparseOperation_t opA,
                                      const void* alpha,
                                      cusparseSpMatDescr_t matA,
                                      const void* x_values,
                                      const void* beta,
                                      void* y_values,
                                      cudaDataType computeType) {
    if (pointer_mode == CUSPARSE_POINTER_MODE_DEVICE) {
        // Device scalars may be produced by earlier work in this stream. The
        // current kernels pass them as inline bytes, so make that dependency
        // visible before resolving their shared-memory contents.
        if (cumetal::rt::synchronize_for_host_library(stream) != cudaSuccess) {
            return CUSPARSE_STATUS_EXECUTION_FAILED;
        }
    }
    alpha = scalar_pointer_for_mode(pointer_mode, alpha, computeType);
    beta = scalar_pointer_for_mode(pointer_mode, beta, computeType);
    if (alpha == nullptr || beta == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    const bool op_t = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t ylen = op_t ? matA->cols : matA->rows;
    bool transpose = false;
    int64_t axis = 0;
    cumetal_sparse_view(matA, opA, &transpose, &axis);

    // The GPU path is enqueued on that stream and deliberately does not
    // synchronize: it is ordered against the caller's other stream work the way
    // a real cuSPARSE call is, and a host read of y needs its own
    // synchronization either way. Only the CPU path below has to wait, because
    // it dereferences the operands itself.
    if (matA->format != CUMETAL_SPMAT_COO &&
        try_spmv_gather_metal(stream, matA, transpose, axis, ylen, alpha, beta,
                              x_values, y_values, computeType)) {
        return CUSPARSE_STATUS_SUCCESS;
    }

    // Order this call after prior work on the stream before computing on the CPU.
    cumetal::rt::synchronize_for_host_library(stream);

    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int* offsets = static_cast<const int*>(matA->rowOffsets);
    const int* indices = static_cast<const int*>(matA->colInd);

    if (computeType == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double b = *static_cast<const double*>(beta);
        const double* vals = static_cast<const double*>(matA->values);
        const double* x = static_cast<const double*>(x_values);
        double* y = static_cast<double*>(y_values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmv_coo(matA->nnz, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
        else
            cumetal_spmv_compressed(axis, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        const float* vals = static_cast<const float*>(matA->values);
        const float* x = static_cast<const float*>(x_values);
        float* y = static_cast<float*>(y_values);
        if (matA->format == CUMETAL_SPMAT_COO)
            cumetal_spmv_coo(matA->nnz, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
        else
            cumetal_spmv_compressed(axis, offsets, indices, vals, base, transpose, a, b, x, y, ylen);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

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
                                          size_t* bufferSize) {
    if (!handle || !alpha || !matA || !matB || !beta || !matC || !bufferSize) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!valid_spmm_types(matA, matB, matC, computeType)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!valid_operation(opA) || !valid_operation(opB) ||
        !valid_spmm_algorithm(alg)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (opB != CUSPARSE_OPERATION_NON_TRANSPOSE ||
        matB->order != CUSPARSE_ORDER_COL ||
        matC->order != CUSPARSE_ORDER_COL ||
        matB->batchCount != 1 || matC->batchCount != 1 ||
        !cumetal_sparse_indices_are_32bit(matA)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType) == nullptr ||
        scalar_pointer_for_mode(handle->pointer_mode, beta, computeType) == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    *bufferSize = 0;
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
                               cusparseSpMMAlg_t alg,
                               void* /*externalBuffer*/) {
    if (!handle || !matA || !matB || !matC || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    if (!valid_spmm_types(matA, matB, matC, computeType)) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!valid_operation(opA) || !valid_operation(opB) ||
        !valid_spmm_algorithm(alg)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
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
    // Only one matrix is addressed; a strided batch would silently compute the
    // first member and leave the rest of C untouched.
    if (matB->batchCount != 1 || matC->batchCount != 1) {
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

    alpha = scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType);
    beta = scalar_pointer_for_mode(handle->pointer_mode, beta, computeType);
    if (alpha == nullptr || beta == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

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
                                 cusparseOperation_t transA,
                                 int m, int n, int nnz,
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
    if (m < 0 || n < 0 || nnz < 0 ||
        (transA != CUSPARSE_OPERATION_NON_TRANSPOSE &&
         transA != CUSPARSE_OPERATION_TRANSPOSE &&
         transA != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    alpha = static_cast<const float*>(scalar_pointer_for_mode_size(
        handle->pointer_mode, alpha, sizeof(float)));
    beta = static_cast<const float*>(scalar_pointer_for_mode_size(
        handle->pointer_mode, beta, sizeof(float)));
    if (alpha == nullptr || beta == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    const int base = descrA ? static_cast<int>(descrA->base) : 0;
    if (csrRowPtrA[0] - base != 0 || csrRowPtrA[m] - base != nnz) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    for (int row = 0; row < m; ++row) {
        const int begin = csrRowPtrA[row] - base;
        const int end = csrRowPtrA[row + 1] - base;
        if (begin < 0 || end < begin || end > nnz) {
            return CUSPARSE_STATUS_INVALID_VALUE;
        }
    }
    for (int j = 0; j < nnz; ++j) {
        const int column = csrColIndA[j] - base;
        if (column < 0 || column >= n) return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
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

    for (int column = 0; column < n; ++column) y[column] *= *beta;
    for (int row = 0; row < m; ++row) {
        const int row_start = csrRowPtrA[row] - base;
        const int row_end = csrRowPtrA[row + 1] - base;
        for (int j = row_start; j < row_end; ++j) {
            const int column = csrColIndA[j] - base;
            y[column] += (*alpha) * csrValA[j] * x[row];
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// Legacy CSR SpMV (double)
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
                                 double* y) {
    if (!handle || !alpha || !beta || !csrValA || !csrRowPtrA || !csrColIndA || !x || !y) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || nnz < 0 ||
        (transA != CUSPARSE_OPERATION_NON_TRANSPOSE &&
         transA != CUSPARSE_OPERATION_TRANSPOSE &&
         transA != CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    alpha = static_cast<const double*>(scalar_pointer_for_mode_size(
        handle->pointer_mode, alpha, sizeof(double)));
    beta = static_cast<const double*>(scalar_pointer_for_mode_size(
        handle->pointer_mode, beta, sizeof(double)));
    if (alpha == nullptr || beta == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    const int base = descrA ? static_cast<int>(descrA->base) : 0;
    if (csrRowPtrA[0] - base != 0 || csrRowPtrA[m] - base != nnz) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    for (int row = 0; row < m; ++row) {
        const int begin = csrRowPtrA[row] - base;
        const int end = csrRowPtrA[row + 1] - base;
        if (begin < 0 || end < begin || end > nnz) {
            return CUSPARSE_STATUS_INVALID_VALUE;
        }
    }
    for (int j = 0; j < nnz; ++j) {
        const int column = csrColIndA[j] - base;
        if (column < 0 || column >= n) return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (transA == CUSPARSE_OPERATION_NON_TRANSPOSE) {
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

    for (int column = 0; column < n; ++column) y[column] *= *beta;
    for (int row = 0; row < m; ++row) {
        const int row_start = csrRowPtrA[row] - base;
        const int row_end = csrRowPtrA[row + 1] - base;
        for (int j = row_start; j < row_end; ++j) {
            const int column = csrColIndA[j] - base;
            y[column] += (*alpha) * csrValA[j] * x[row];
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpSV: Sparse triangular solve ─────────────────────────────────────────────

struct cusparseSpSVDescr {
    // Analysis phase is a no-op on CPU; descriptor exists for API compat
};

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t* descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new (std::nothrow) cusparseSpSVDescr();
    if (*descr == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
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
    if (bufferSize) *bufferSize = 1;
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
    if (!valid_spmv_types(matA, vecX, vecY, computeType))
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (!valid_operation(opA)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (matA->rows != matA->cols || vecX->size != matA->rows ||
        vecY->size != matA->rows) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }

    synchronize_handle_stream(handle);

    alpha = scalar_pointer_for_mode(handle->pointer_mode, alpha, computeType);
    if (alpha == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;

    if (!cumetal_sparse_indices_are_32bit(matA)) return CUSPARSE_STATUS_NOT_SUPPORTED;

    const int* rowPtr = static_cast<const int*>(matA->rowOffsets);
    const int* colIdx = static_cast<const int*>(matA->colInd);
    const int base = (matA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    const int64_t n = matA->rows;
    const bool transpose = opA != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const bool unit = matA->diag == CUSPARSE_DIAG_TYPE_UNIT;
    bool ok = true;

    // y = alpha * x, then solve op(A) * y = y in place. The solve walks the
    // CSR arrays through cumetal_tri_solve, which handles the transposed
    // operation by scattering; reading A's rows as if they were op(A)'s rows
    // would be a different (wrong) system.
    if (computeType == CUDA_R_64F) {
        const double a = *static_cast<const double*>(alpha);
        const double* vals = static_cast<const double*>(matA->values);
        const double* x = static_cast<const double*>(vecX->values);
        double* y = static_cast<double*>(vecY->values);
        for (int64_t i = 0; i < n; ++i) y[i] = a * x[i];
        ok = cumetal_tri_solve(n, rowPtr, colIdx, vals, base, matA->fill, unit, transpose,
                               false, y);
    } else {
        const float a = *static_cast<const float*>(alpha);
        const float* vals = static_cast<const float*>(matA->values);
        const float* x = static_cast<const float*>(vecX->values);
        float* y = static_cast<float*>(vecY->values);
        for (int64_t i = 0; i < n; ++i) y[i] = a * x[i];
        ok = cumetal_tri_solve(n, rowPtr, colIdx, vals, base, matA->fill, unit, transpose,
                               false, y);
    }
    if (!ok) return CUSPARSE_STATUS_ZERO_PIVOT;
    return CUSPARSE_STATUS_SUCCESS;
}


// ═════════════════════════════════════════════════════════════════════════════
// Format conversion, sorting, nnz counting, tridiagonal/pentadiagonal solvers,
// incomplete factorizations and the rest of the generic API.
//
// Everything below computes on the CPU over unified memory, like the SpMV/SpMM
// paths above: synchronize the handle's stream, then read and write the
// operands directly. None of it needs scratch from the caller, so every
// *_bufferSize query returns a small non-zero size (callers often allocate it
// unconditionally and some allocators hand back null for zero bytes).
// ═════════════════════════════════════════════════════════════════════════════

extern "C++" {

#define SP_NEED_HANDLE(h)                                               \
    do {                                                                \
        if ((h) == nullptr) return CUSPARSE_STATUS_NOT_INITIALIZED;     \
    } while (0)

constexpr std::size_t kNoWorkspaceBytes = 128;

static int base_of(const cusparseMatDescr* descr) {
    return descr != nullptr && descr->base == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
}

// An offset array is usable when it starts at the index base, never decreases
// and ends at nnz. cuSPARSE does not check this; reading past a malformed one
// would walk off the allocation, so refuse here.
static bool offsets_valid(int64_t rows, int64_t nnz, const int* ptr, int base) {
    if (rows < 0 || nnz < 0 || ptr == nullptr) return false;
    if (ptr[0] - base != 0 || ptr[rows] - base != nnz) return false;
    for (int64_t r = 0; r < rows; ++r) {
        if (ptr[r + 1] < ptr[r]) return false;
    }
    return true;
}

static bool csr_structure_valid(int64_t rows, int64_t cols, int64_t nnz, const int* ptr,
                                const int* ind, int base) {
    if (!offsets_valid(rows, nnz, ptr, base)) return false;
    if (nnz == 0) return true;
    if (ind == nullptr) return false;
    for (int64_t e = 0; e < nnz; ++e) {
        const int c = ind[e] - base;
        if (c < 0 || c >= cols) return false;
    }
    return true;
}

static std::size_t value_size(cudaDataType type) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return 2;
        case CUDA_C_16F:
        case CUDA_C_16BF: return 4;
        case CUDA_R_32F:
        case CUDA_R_32I: return 4;
        case CUDA_C_32F:
        case CUDA_R_64F: return 8;
        case CUDA_C_64F: return 16;
        case CUDA_R_8I:
        case CUDA_R_8U: return 1;
    }
    return 0;
}

// Whether the element at `p` is a stored nonzero. Zero of either sign is zero.
static bool value_nonzero(cudaDataType type, const void* p) {
    switch (type) {
        case CUDA_R_16F:
        case CUDA_R_16BF: return (*static_cast<const std::uint16_t*>(p) & 0x7fffu) != 0;
        case CUDA_C_16F:
        case CUDA_C_16BF: {
            const auto* h = static_cast<const std::uint16_t*>(p);
            return (h[0] & 0x7fffu) != 0 || (h[1] & 0x7fffu) != 0;
        }
        case CUDA_R_32F: return *static_cast<const float*>(p) != 0.0f;
        case CUDA_C_32F: {
            const auto* f = static_cast<const float*>(p);
            return f[0] != 0.0f || f[1] != 0.0f;
        }
        case CUDA_R_64F: return *static_cast<const double*>(p) != 0.0;
        case CUDA_C_64F: {
            const auto* d = static_cast<const double*>(p);
            return d[0] != 0.0 || d[1] != 0.0;
        }
        case CUDA_R_8I:
        case CUDA_R_8U: return *static_cast<const std::uint8_t*>(p) != 0;
        case CUDA_R_32I: return *static_cast<const std::int32_t*>(p) != 0;
    }
    return false;
}

// Dispatch a generic-API call on its compute type. `f` receives a value of the
// native element type (float, double, std::complex<float>, std::complex<double>)
// purely as a tag.
template <typename F>
static cusparseStatus_t dispatch_compute_type(cudaDataType type, F&& f) {
    switch (type) {
        case CUDA_R_32F: return f(float{});
        case CUDA_R_64F: return f(double{});
        case CUDA_C_32F: return f(std::complex<float>{});
        case CUDA_C_64F: return f(std::complex<double>{});
        default: return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
}

// Overwrite `out` with the double-precision complex `v`, narrowing as needed.
inline void assign_from_cd(float& out, const std::complex<double>& v) { out = static_cast<float>(v.real()); }
inline void assign_from_cd(double& out, const std::complex<double>& v) { out = v.real(); }
inline void assign_from_cd(std::complex<float>& out, const std::complex<double>& v) {
    out = std::complex<float>(static_cast<float>(v.real()), static_cast<float>(v.imag()));
}
inline void assign_from_cd(std::complex<double>& out, const std::complex<double>& v) { out = v; }

inline std::complex<double> to_cd(float v) { return {v, 0.0}; }
inline std::complex<double> to_cd(double v) { return {v, 0.0}; }
inline std::complex<double> to_cd(const std::complex<float>& v) { return {v.real(), v.imag()}; }
inline std::complex<double> to_cd(const std::complex<double>& v) { return v; }

// ── coordinate / compressed sorting ─────────────────────────────────────────

template <typename T>
static void apply_permutation(T* data, const std::vector<int>& perm, int64_t offset = 0) {
    std::vector<T> scratch(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) scratch[i] = data[offset + perm[i]];
    for (std::size_t i = 0; i < perm.size(); ++i) data[offset + i] = scratch[i];
}

static cusparseStatus_t coo_sort_impl(cusparseHandle_t handle, int m, int n, int nnz,
                                      int* rows, int* cols, int* P, bool by_row) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || n < 0 || nnz < 0 || (nnz > 0 && (rows == nullptr || cols == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    std::vector<int> perm(static_cast<std::size_t>(nnz));
    std::iota(perm.begin(), perm.end(), 0);
    // Stable, so equal coordinates keep their incoming order and P stays a
    // deterministic permutation.
    std::stable_sort(perm.begin(), perm.end(), [&](int a, int b) {
        const int ka = by_row ? rows[a] : cols[a];
        const int kb = by_row ? rows[b] : cols[b];
        if (ka != kb) return ka < kb;
        return by_row ? cols[a] < cols[b] : rows[a] < rows[b];
    });
    apply_permutation(rows, perm);
    apply_permutation(cols, perm);
    // P is both input and output: sorted_values = values(P), so P composes with
    // whatever permutation the caller already had in it.
    if (P != nullptr) apply_permutation(P, perm);
    return CUSPARSE_STATUS_SUCCESS;
}

// Sort the index array within each segment of a compressed format. `segments`
// is the length of the offset array minus one (m for CSR, n for CSC) and
// `bound` the exclusive limit of the indices it holds.
static cusparseStatus_t sort_compressed_impl(cusparseHandle_t handle, int segments, int bound,
                                             int nnz, const cusparseMatDescr_t descr,
                                             const int* ptr, int* ind, int* P) {
    SP_NEED_HANDLE(handle);
    if (descr == nullptr || segments < 0 || bound < 0 || nnz < 0 ||
        (nnz > 0 && ind == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const int base = base_of(descr);
    synchronize_handle_stream(handle);
    if (!csr_structure_valid(segments, bound, nnz, ptr, ind, base)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    std::vector<int> perm;
    for (int s = 0; s < segments; ++s) {
        const int begin = ptr[s] - base;
        const int end = ptr[s + 1] - base;
        perm.resize(static_cast<std::size_t>(end - begin));
        std::iota(perm.begin(), perm.end(), 0);
        std::stable_sort(perm.begin(), perm.end(),
                         [&](int a, int b) { return ind[begin + a] < ind[begin + b]; });
        apply_permutation(ind, perm, begin);
        if (P != nullptr) apply_permutation(P, perm, begin);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── nnz counting and compression ────────────────────────────────────────────

template <typename C>
static cusparseStatus_t nnz_dense_impl(cusparseHandle_t handle, cusparseDirection_t dirA, int m,
                                       int n, const cusparseMatDescr_t descrA, const C* A,
                                       int lda, int* nnzPerRowColumn, int* nnzTotal) {
    using T = native_t<C>;
    SP_NEED_HANDLE(handle);
    if (descrA == nullptr || m < 0 || n < 0 || lda < std::max(1, m) || nnzTotal == nullptr ||
        (dirA != CUSPARSE_DIRECTION_ROW && dirA != CUSPARSE_DIRECTION_COLUMN)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (descrA->type != CUSPARSE_MATRIX_TYPE_GENERAL) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    const bool by_row = dirA == CUSPARSE_DIRECTION_ROW;
    const int count = by_row ? m : n;
    if ((m > 0 && n > 0 && A == nullptr) || (count > 0 && nnzPerRowColumn == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const T* a = cnat(A);
    std::vector<int> per(static_cast<std::size_t>(count), 0);
    int total = 0;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            if (a[i + static_cast<int64_t>(j) * lda] != T(0)) {
                ++per[static_cast<std::size_t>(by_row ? i : j)];
                ++total;
            }
        }
    }
    for (int k = 0; k < count; ++k) nnzPerRowColumn[k] = per[static_cast<std::size_t>(k)];
    *nnzTotal = total;
    return CUSPARSE_STATUS_SUCCESS;
}

// An entry survives compression when its magnitude exceeds the tolerance's.
template <typename T>
static bool keeps_entry(const T& v, const T& tol) { return mag(v) > mag(tol); }

template <typename C>
static cusparseStatus_t nnz_compress_impl(cusparseHandle_t handle, int m,
                                          const cusparseMatDescr_t descr, const C* val,
                                          const int* rowPtr, int* nnzPerRow, int* nnzC, C tol) {
    using T = native_t<C>;
    SP_NEED_HANDLE(handle);
    if (descr == nullptr || m < 0 || rowPtr == nullptr || nnzC == nullptr ||
        (m > 0 && nnzPerRow == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const int base = base_of(descr);
    synchronize_handle_stream(handle);
    const int64_t nnz = static_cast<int64_t>(rowPtr[m]) - base;
    if (!offsets_valid(m, nnz, rowPtr, base) || (nnz > 0 && val == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const T* v = cnat(val);
    const T t = to_native(tol);
    int total = 0;
    for (int i = 0; i < m; ++i) {
        int row_count = 0;
        for (int k = rowPtr[i] - base; k < rowPtr[i + 1] - base; ++k) {
            if (keeps_entry(v[k], t)) ++row_count;
        }
        nnzPerRow[i] = row_count;
        total += row_count;
    }
    *nnzC = total;
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t csr2csr_compress_impl(cusparseHandle_t handle, int m, int n,
                                              const cusparseMatDescr_t descrA, const C* inVal,
                                              const int* inColInd, const int* inRowPtr, int inNnz,
                                              int* nnzPerRow, C* outVal, int* outColInd,
                                              int* outRowPtr, C tol) {
    using T = native_t<C>;
    SP_NEED_HANDLE(handle);
    if (descrA == nullptr || m < 0 || n < 0 || inNnz < 0 || inRowPtr == nullptr ||
        outRowPtr == nullptr || (m > 0 && nnzPerRow == nullptr) ||
        (inNnz > 0 && (inVal == nullptr || inColInd == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const int base = base_of(descrA);
    synchronize_handle_stream(handle);
    if (!csr_structure_valid(m, n, inNnz, inRowPtr, inColInd, base)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const T* v = cnat(inVal);
    const T t = to_native(tol);
    // nnzPerRow comes from the matching *nnz_compress call. Check it against
    // what this call would keep before writing anything, so a mismatched tol
    // fails cleanly instead of overrunning the output arrays.
    int64_t total = 0;
    for (int i = 0; i < m; ++i) {
        int row_count = 0;
        for (int k = inRowPtr[i] - base; k < inRowPtr[i + 1] - base; ++k) {
            if (keeps_entry(v[k], t)) ++row_count;
        }
        if (row_count != nnzPerRow[i]) return CUSPARSE_STATUS_INVALID_VALUE;
        total += row_count;
    }
    if (total > 0 && (outVal == nullptr || outColInd == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    T* out = nat(outVal);
    int64_t w = 0;
    outRowPtr[0] = base;
    for (int i = 0; i < m; ++i) {
        for (int k = inRowPtr[i] - base; k < inRowPtr[i + 1] - base; ++k) {
            if (!keeps_entry(v[k], t)) continue;
            out[w] = v[k];
            outColInd[w] = inColInd[k];
            ++w;
        }
        outRowPtr[i + 1] = static_cast<int>(w) + base;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── csrgeam2: C = alpha*A + beta*B ──────────────────────────────────────────

static cusparseStatus_t geam2_validate(cusparseHandle_t handle, int m, int n,
                                       const cusparseMatDescr_t descrA, int nnzA,
                                       const int* ptrA, const int* colA,
                                       const cusparseMatDescr_t descrB, int nnzB,
                                       const int* ptrB, const int* colB,
                                       const cusparseMatDescr_t descrC) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || n < 0 || nnzA < 0 || nnzB < 0 || descrA == nullptr || descrB == nullptr ||
        descrC == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (descrA->type != CUSPARSE_MATRIX_TYPE_GENERAL ||
        descrB->type != CUSPARSE_MATRIX_TYPE_GENERAL ||
        descrC->type != CUSPARSE_MATRIX_TYPE_GENERAL) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    }
    synchronize_handle_stream(handle);
    if (!csr_structure_valid(m, n, nnzA, ptrA, colA, base_of(descrA)) ||
        !csr_structure_valid(m, n, nnzB, ptrB, colB, base_of(descrB))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t geam2_buffer_size(cusparseHandle_t handle, int m, int n, const C* alpha,
                                          const cusparseMatDescr_t descrA, int nnzA,
                                          const C* valA, const int* ptrA, const int* colA,
                                          const C* beta, const cusparseMatDescr_t descrB,
                                          int nnzB, const C* valB, const int* ptrB,
                                          const int* colB, const cusparseMatDescr_t descrC,
                                          C* valC, int* ptrC, int* colC, size_t* size) {
    if (alpha == nullptr || beta == nullptr || ptrC == nullptr || size == nullptr ||
        (nnzA > 0 && valA == nullptr) || (nnzB > 0 && valB == nullptr)) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED
                                 : CUSPARSE_STATUS_INVALID_VALUE;
    }
    (void)valC;
    (void)colC;
    const cusparseStatus_t status =
        geam2_validate(handle, m, n, descrA, nnzA, ptrA, colA, descrB, nnzB, ptrB, colB, descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// One row of A and B as sorted, de-duplicated 0-based column lists.
static void geam2_row_columns(const int* ptrA, const int* colA, int baseA, const int* ptrB,
                              const int* colB, int baseB, int row, std::vector<int>* cols) {
    cols->clear();
    for (int k = ptrA[row] - baseA; k < ptrA[row + 1] - baseA; ++k) cols->push_back(colA[k] - baseA);
    for (int k = ptrB[row] - baseB; k < ptrB[row + 1] - baseB; ++k) cols->push_back(colB[k] - baseB);
    std::sort(cols->begin(), cols->end());
    cols->erase(std::unique(cols->begin(), cols->end()), cols->end());
}

template <typename C>
static cusparseStatus_t geam2_compute(cusparseHandle_t handle, int m, int n, const C* alphaC,
                                      const cusparseMatDescr_t descrA, int nnzA, const C* valA,
                                      const int* ptrA, const int* colA, const C* betaC,
                                      const cusparseMatDescr_t descrB, int nnzB, const C* valB,
                                      const int* ptrB, const int* colB,
                                      const cusparseMatDescr_t descrC, C* valC, int* ptrC,
                                      int* colC) {
    using T = native_t<C>;
    if (alphaC == nullptr || betaC == nullptr || ptrC == nullptr ||
        (nnzA > 0 && valA == nullptr) || (nnzB > 0 && valB == nullptr)) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED
                                 : CUSPARSE_STATUS_INVALID_VALUE;
    }
    cusparseStatus_t status =
        geam2_validate(handle, m, n, descrA, nnzA, ptrA, colA, descrB, nnzB, ptrB, colB, descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    const void* alpha_ptr = scalar_pointer_for_mode_size(handle->pointer_mode, alphaC, sizeof(T));
    const void* beta_ptr = scalar_pointer_for_mode_size(handle->pointer_mode, betaC, sizeof(T));
    if (alpha_ptr == nullptr || beta_ptr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    const T alpha = *static_cast<const T*>(alpha_ptr);
    const T beta = *static_cast<const T*>(beta_ptr);
    const int baseA = base_of(descrA), baseB = base_of(descrB), baseC = base_of(descrC);
    const int64_t nnzC = static_cast<int64_t>(ptrC[m]) - baseC;
    if (!offsets_valid(m, nnzC, ptrC, baseC) ||
        (nnzC > 0 && (valC == nullptr || colC == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const T* va = cnat(valA);
    const T* vb = cnat(valB);
    T* vc = nat(valC);
    std::vector<int> cols;
    std::vector<std::pair<int, T>> entries;
    // Check every row against the offsets the Nnz call produced before writing
    // anything, so a stale csrRowPtrC fails cleanly.
    for (int i = 0; i < m; ++i) {
        geam2_row_columns(ptrA, colA, baseA, ptrB, colB, baseB, i, &cols);
        if (static_cast<int64_t>(cols.size()) != static_cast<int64_t>(ptrC[i + 1]) - ptrC[i]) {
            return CUSPARSE_STATUS_INVALID_VALUE;
        }
    }
    for (int i = 0; i < m; ++i) {
        entries.clear();
        for (int k = ptrA[i] - baseA; k < ptrA[i + 1] - baseA; ++k)
            entries.emplace_back(colA[k] - baseA, alpha * va[k]);
        for (int k = ptrB[i] - baseB; k < ptrB[i + 1] - baseB; ++k)
            entries.emplace_back(colB[k] - baseB, beta * vb[k]);
        std::stable_sort(entries.begin(), entries.end(),
                         [](const auto& x, const auto& y) { return x.first < y.first; });
        int64_t w = static_cast<int64_t>(ptrC[i]) - baseC;
        for (std::size_t e = 0; e < entries.size();) {
            const int col = entries[e].first;
            T sum = entries[e].second;
            for (++e; e < entries.size() && entries[e].first == col; ++e) sum += entries[e].second;
            colC[w] = col + baseC;
            vc[w] = sum;
            ++w;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── incomplete factorizations (zero fill-in) ────────────────────────────────

template <typename C>
static cusparseStatus_t incomplete_validate(cusparseHandle_t handle, int m, int nnz,
                                            const cusparseMatDescr_t descrA, const C* val,
                                            const int* ptr, const int* col, const void* info) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || nnz < 0 || info == nullptr || ptr == nullptr || (nnz > 0 && (val == nullptr || col == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    if (!csr_structure_valid(m, m, nnz, ptr, col, base_of(descrA))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// First row whose diagonal entry is absent from the pattern, or -1.
static int first_missing_diagonal(int m, const int* ptr, const int* col, int base,
                                  std::vector<int>* diagpos) {
    diagpos->assign(static_cast<std::size_t>(m), -1);
    for (int i = 0; i < m; ++i) {
        for (int k = ptr[i] - base; k < ptr[i + 1] - base; ++k) {
            if (col[k] - base == i) {
                (*diagpos)[static_cast<std::size_t>(i)] = k;
                break;
            }
        }
    }
    for (int i = 0; i < m; ++i) {
        if ((*diagpos)[static_cast<std::size_t>(i)] < 0) return i;
    }
    return -1;
}

template <typename C>
static cusparseStatus_t csrilu02_analysis_impl(cusparseHandle_t handle, int m, int nnz,
                                               const cusparseMatDescr_t descrA, const C* val,
                                               const int* ptr, const int* col,
                                               csrilu02Info_t info) {
    cusparseStatus_t status = incomplete_validate(handle, m, nnz, descrA, val, ptr, col, info);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    // A structural zero (no diagonal entry) is visible from the pattern alone,
    // so the analysis phase already reports it through zeroPivot.
    std::vector<int> diagpos;
    info->base = base_of(descrA);
    info->zero_pivot = first_missing_diagonal(m, ptr, col, info->base, &diagpos);
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t csrilu02_impl(cusparseHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA, C* valC, const int* ptr,
                                      const int* col, csrilu02Info_t info) {
    using T = native_t<C>;
    cusparseStatus_t status = incomplete_validate(handle, m, nnz, descrA, valC, ptr, col, info);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    T* val = nat(valC);
    const int base = base_of(descrA);
    info->base = base;

    std::vector<int> diagpos;
    int pivot = first_missing_diagonal(m, ptr, col, base, &diagpos);
    auto note = [&](int row) {
        if (pivot < 0 || row < pivot) pivot = row;
    };

    T boost{};
    if (info->boost_enabled) assign_from_cd(boost, info->boost_value);

    // Row-wise IKJ elimination restricted to the existing pattern. `pos` maps a
    // column to its slot in the current row, so updates to positions outside
    // the pattern (fill-in) are dropped.
    std::vector<int> pos(static_cast<std::size_t>(m), -1);
    std::vector<int> lower;
    for (int i = 0; i < m; ++i) {
        const int begin = ptr[i] - base;
        const int end = ptr[i + 1] - base;
        lower.clear();
        for (int k = begin; k < end; ++k) {
            pos[static_cast<std::size_t>(col[k] - base)] = k;
            if (col[k] - base < i) lower.push_back(k);
        }
        std::sort(lower.begin(), lower.end(), [&](int a, int b) { return col[a] < col[b]; });
        for (int k : lower) {
            const int kc = col[k] - base;
            const int dp = diagpos[static_cast<std::size_t>(kc)];
            if (dp < 0 || val[dp] == T(0)) {
                note(kc);  // elimination by a zero pivot is undefined; skip it
                continue;
            }
            val[k] /= val[dp];
            const T multiplier = val[k];
            for (int q = ptr[kc] - base; q < ptr[kc + 1] - base; ++q) {
                const int j = col[q] - base;
                if (j <= kc) continue;
                const int slot = pos[static_cast<std::size_t>(j)];
                if (slot >= 0) val[slot] -= multiplier * val[q];
            }
        }
        const int dp = diagpos[static_cast<std::size_t>(i)];
        if (dp >= 0) {
            if (info->boost_enabled && mag(val[dp]) <= info->boost_tol) {
                val[dp] = boost;
            } else if (val[dp] == T(0)) {
                note(i);
            }
        }
        for (int k = begin; k < end; ++k) pos[static_cast<std::size_t>(col[k] - base)] = -1;
    }
    info->zero_pivot = pivot;
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t csrilu02_boost_impl(cusparseHandle_t handle, csrilu02Info_t info,
                                            int enable_boost, double* tol, C* boost_val) {
    SP_NEED_HANDLE(handle);
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (enable_boost == 0) {
        info->boost_enabled = false;
        return CUSPARSE_STATUS_SUCCESS;
    }
    if (tol == nullptr || boost_val == nullptr || !(*tol >= 0.0)) return CUSPARSE_STATUS_INVALID_VALUE;
    info->boost_enabled = true;
    info->boost_tol = *tol;
    info->boost_value = to_cd(to_native(*boost_val));
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t csric02_analysis_impl(cusparseHandle_t handle, int m, int nnz,
                                              const cusparseMatDescr_t descrA, const C* val,
                                              const int* ptr, const int* col,
                                              csric02Info_t info) {
    cusparseStatus_t status = incomplete_validate(handle, m, nnz, descrA, val, ptr, col, info);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    std::vector<int> diagpos;
    info->base = base_of(descrA);
    info->zero_pivot = first_missing_diagonal(m, ptr, col, info->base, &diagpos);
    return CUSPARSE_STATUS_SUCCESS;
}

// Incomplete Cholesky, A ~= L*L^H (lower) or U^H*U (upper), on A's own pattern.
// Only the triangle named by the descriptor's fill mode is read and written.
// A diagonal that is missing, zero or (for the real part) not positive is
// reported as a zero pivot rather than turned into a NaN.
template <typename C>
static cusparseStatus_t csric02_impl(cusparseHandle_t handle, int m, int nnz,
                                     const cusparseMatDescr_t descrA, C* valC, const int* ptr,
                                     const int* col, csric02Info_t info) {
    using T = native_t<C>;
    cusparseStatus_t status = incomplete_validate(handle, m, nnz, descrA, valC, ptr, col, info);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    T* val = nat(valC);
    const int base = base_of(descrA);
    const bool lower = descrA == nullptr || descrA->fill == CUSPARSE_FILL_MODE_LOWER;
    info->base = base;

    std::vector<int> diagpos;
    int pivot = first_missing_diagonal(m, ptr, col, base, &diagpos);
    auto note = [&](int row) {
        if (pivot < 0 || row < pivot) pivot = row;
    };

    // Per-row (column, position) lists sorted by column, so entries can be
    // found by binary search whether or not the caller sorted its CSR.
    std::vector<std::vector<std::pair<int, int>>> rows(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i) {
        auto& row = rows[static_cast<std::size_t>(i)];
        for (int k = ptr[i] - base; k < ptr[i + 1] - base; ++k) row.emplace_back(col[k] - base, k);
        std::sort(row.begin(), row.end());
    }
    auto find = [&](int r, int c) -> int {
        const auto& row = rows[static_cast<std::size_t>(r)];
        auto it = std::lower_bound(row.begin(), row.end(), std::make_pair(c, -1));
        return it != row.end() && it->first == c ? it->second : -1;
    };

    if (lower) {
        // Left-looking: row i of L from rows 0..i-1.
        for (int i = 0; i < m; ++i) {
            const auto& row_i = rows[static_cast<std::size_t>(i)];
            for (const auto& [j, p] : row_i) {
                if (j > i) break;
                T s = val[p];
                for (const auto& [k, pk] : row_i) {
                    if (k >= j) break;
                    const int pjk = find(j, k);
                    if (pjk >= 0) s -= val[pk] * conj_of(val[pjk]);
                }
                if (j < i) {
                    const int dp = diagpos[static_cast<std::size_t>(j)];
                    if (dp < 0 || val[dp] == T(0)) {
                        note(j);
                        continue;
                    }
                    val[p] = s / val[dp];
                } else if (real_part(s) <= 0.0) {
                    note(i);
                    val[p] = s;
                } else {
                    val[p] = std::sqrt(s);
                }
            }
        }
    } else {
        // Right-looking over the upper triangle: finish row k of U, then
        // subtract its outer product from the trailing rows.
        for (int k = 0; k < m; ++k) {
            const int dp = diagpos[static_cast<std::size_t>(k)];
            if (dp < 0) {
                note(k);
                continue;
            }
            if (real_part(val[dp]) <= 0.0) {
                note(k);
                continue;
            }
            const T root = std::sqrt(val[dp]);
            val[dp] = root;
            const auto& row_k = rows[static_cast<std::size_t>(k)];
            for (const auto& [j, p] : row_k) {
                if (j > k) val[p] /= root;
            }
            for (const auto& [i, pi] : row_k) {
                if (i <= k) continue;
                for (const auto& [j, pj] : row_k) {
                    if (j < i) continue;
                    const int q = find(i, j);
                    if (q >= 0) val[q] -= conj_of(val[pi]) * val[pj];
                }
            }
        }
    }
    info->zero_pivot = pivot;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── tridiagonal / pentadiagonal solvers ─────────────────────────────────────

// Banded Gaussian elimination with partial pivoting for a matrix with `kl`
// sub- and `ku` super-diagonals, `nrhs` right-hand sides. Rows are held in
// windows of 3*kl+ku+1 columns starting at column (row - kl): pivoting can
// widen the upper band by up to kl, and a row swapped into position i has no
// entries left of column i. Returns false on an exactly singular pivot column.
//
// `a` is m rows of that window, `b` is m rows of nrhs values (row-major).
template <typename T>
static bool band_solve(int m, int kl, int ku, int nrhs, std::vector<T>* a_ptr,
                       std::vector<T>* b_ptr) {
    std::vector<T>& a = *a_ptr;
    std::vector<T>& b = *b_ptr;
    const int W = 3 * kl + ku + 1;
    auto at = [&](int r, int c) -> T& {
        return a[static_cast<std::size_t>(r) * W + static_cast<std::size_t>(c - r + kl)];
    };
    for (int i = 0; i < m; ++i) {
        const int last_row = std::min(i + kl, m - 1);
        const int last_col = std::min(i + kl + ku, m - 1);
        int pivot = i;
        double best = mag(at(i, i));
        for (int r = i + 1; r <= last_row; ++r) {
            const double v = mag(at(r, i));
            if (v > best) {
                best = v;
                pivot = r;
            }
        }
        if (best == 0.0) return false;
        if (pivot != i) {
            for (int c = i; c <= last_col; ++c) std::swap(at(i, c), at(pivot, c));
            for (int j = 0; j < nrhs; ++j) {
                std::swap(b[static_cast<std::size_t>(i) * nrhs + j],
                          b[static_cast<std::size_t>(pivot) * nrhs + j]);
            }
        }
        for (int r = i + 1; r <= last_row; ++r) {
            const T factor = at(r, i) / at(i, i);
            if (factor == T(0)) continue;
            for (int c = i + 1; c <= last_col; ++c) at(r, c) -= factor * at(i, c);
            at(r, i) = T(0);
            for (int j = 0; j < nrhs; ++j) {
                b[static_cast<std::size_t>(r) * nrhs + j] -=
                    factor * b[static_cast<std::size_t>(i) * nrhs + j];
            }
        }
    }
    for (int i = m - 1; i >= 0; --i) {
        const int last_col = std::min(i + kl + ku, m - 1);
        for (int j = 0; j < nrhs; ++j) {
            T s = b[static_cast<std::size_t>(i) * nrhs + j];
            for (int c = i + 1; c <= last_col; ++c) {
                s -= at(i, c) * b[static_cast<std::size_t>(c) * nrhs + j];
            }
            b[static_cast<std::size_t>(i) * nrhs + j] = s / at(i, i);
        }
    }
    return true;
}

// Thomas algorithm (no pivoting) for one tridiagonal system with `nrhs`
// right-hand sides. Element i of each diagonal sits at diag[i * dstride]; of
// right-hand side j at x[i * xstride + j * xcolstride]. dl[0] and du[m-1] are
// never read. Returns false on a zero pivot.
template <typename T>
static bool thomas_solve(int m, const T* dl, const T* d, const T* du, std::ptrdiff_t dstride,
                         T* x, std::ptrdiff_t xstride, int nrhs, std::ptrdiff_t xcolstride) {
    if (m == 0) return true;
    std::vector<T> cprime(static_cast<std::size_t>(m));
    T denom = d[0];
    if (denom == T(0)) return false;
    cprime[0] = m > 1 ? du[0] / denom : T(0);
    for (int j = 0; j < nrhs; ++j) x[j * xcolstride] /= denom;
    for (int i = 1; i < m; ++i) {
        denom = d[i * dstride] - dl[i * dstride] * cprime[static_cast<std::size_t>(i - 1)];
        if (denom == T(0)) return false;
        cprime[static_cast<std::size_t>(i)] = i + 1 < m ? du[i * dstride] / denom : T(0);
        for (int j = 0; j < nrhs; ++j) {
            T* xi = x + i * xstride + j * xcolstride;
            *xi = (*xi - dl[i * dstride] * *(xi - xstride)) / denom;
        }
    }
    for (int i = m - 2; i >= 0; --i) {
        for (int j = 0; j < nrhs; ++j) {
            T* xi = x + i * xstride + j * xcolstride;
            *xi -= cprime[static_cast<std::size_t>(i)] * *(xi + xstride);
        }
    }
    return true;
}

template <typename C>
static cusparseStatus_t gtsv2_check(cusparseHandle_t handle, int m, int n, const C* dl,
                                    const C* d, const C* du, const C* B, int ldb) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || n < 0 || ldb < std::max(1, m)) return CUSPARSE_STATUS_INVALID_VALUE;
    if (m > 0 && (dl == nullptr || d == nullptr || du == nullptr || (n > 0 && B == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gtsv2_buffer_size(cusparseHandle_t handle, int m, int n, const C* dl,
                                          const C* d, const C* du, const C* B, int ldb,
                                          size_t* size) {
    if (size == nullptr) return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    const cusparseStatus_t status = gtsv2_check(handle, m, n, dl, d, du, B, ldb);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// A X = B for tridiagonal A and an m-by-n column-major B, overwritten with X.
// dl, d and du are not modified. `pivot` selects partial pivoting (gtsv2) or
// the plain Thomas recurrence (gtsv2_nopivot).
template <typename C>
static cusparseStatus_t gtsv2_impl(cusparseHandle_t handle, int m, int n, const C* dlC,
                                   const C* dC, const C* duC, C* BC, int ldb, bool pivot) {
    using T = native_t<C>;
    cusparseStatus_t status = gtsv2_check(handle, m, n, dlC, dC, duC, BC, ldb);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    if (m == 0 || n == 0) return CUSPARSE_STATUS_SUCCESS;
    const T* dl = cnat(dlC);
    const T* d = cnat(dC);
    const T* du = cnat(duC);
    T* B = nat(BC);
    std::vector<T> x(static_cast<std::size_t>(m) * n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) x[static_cast<std::size_t>(i) * n + j] = B[i + static_cast<int64_t>(j) * ldb];
    bool ok;
    if (pivot) {
        std::vector<T> a(static_cast<std::size_t>(m) * 5, T(0));
        auto at = [&](int r, int c) -> T& { return a[static_cast<std::size_t>(r) * 5 + (c - r + 1)]; };
        for (int r = 0; r < m; ++r) {
            at(r, r) = d[r];
            if (r > 0) at(r, r - 1) = dl[r];
            if (r + 1 < m) at(r, r + 1) = du[r];
        }
        ok = band_solve(m, 1, 1, n, &a, &x);
    } else {
        // Row-major copy: x[i * n + j], so the row stride is n and the column stride 1.
        ok = thomas_solve(m, dl, d, du, 1, x.data(), n, n, 1);
    }
    if (!ok) return CUSPARSE_STATUS_ZERO_PIVOT;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) B[i + static_cast<int64_t>(j) * ldb] = x[static_cast<std::size_t>(i) * n + j];
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gtsv2_strided_check(cusparseHandle_t handle, int m, const C* dl,
                                            const C* d, const C* du, const C* x, int batchCount,
                                            int batchStride) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || batchCount < 1 || batchStride < m) return CUSPARSE_STATUS_INVALID_VALUE;
    if (m > 0 && (dl == nullptr || d == nullptr || du == nullptr || x == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gtsv2_strided_buffer_size(cusparseHandle_t handle, int m, const C* dl,
                                                  const C* d, const C* du, const C* x,
                                                  int batchCount, int batchStride, size_t* size) {
    if (size == nullptr) return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    const cusparseStatus_t status = gtsv2_strided_check(handle, m, dl, d, du, x, batchCount, batchStride);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// batchCount independent systems; system b starts batchStride elements after
// system b-1 in each of dl, d, du and x. x is overwritten with the solutions.
template <typename C>
static cusparseStatus_t gtsv2_strided_impl(cusparseHandle_t handle, int m, const C* dlC,
                                           const C* dC, const C* duC, C* xC, int batchCount,
                                           int batchStride) {
    using T = native_t<C>;
    cusparseStatus_t status = gtsv2_strided_check(handle, m, dlC, dC, duC, xC, batchCount, batchStride);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    const T* dl = cnat(dlC);
    const T* d = cnat(dC);
    const T* du = cnat(duC);
    T* x = nat(xC);
    for (int b = 0; b < batchCount; ++b) {
        const std::ptrdiff_t off = static_cast<std::ptrdiff_t>(b) * batchStride;
        if (!thomas_solve(m, dl + off, d + off, du + off, 1, x + off, 1, 1, 0)) {
            return CUSPARSE_STATUS_ZERO_PIVOT;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gtsv_interleaved_check(cusparseHandle_t handle, int algo, int m,
                                               const C* dl, const C* d, const C* du, const C* x,
                                               int batchCount) {
    SP_NEED_HANDLE(handle);
    // 0 = Thomas, 1 = LU with partial pivoting, 2 = QR.
    if (algo < 0 || algo > 2 || m < 0 || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    if (m > 0 && (dl == nullptr || d == nullptr || du == nullptr || x == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gtsv_interleaved_buffer_size(cusparseHandle_t handle, int algo, int m,
                                                     const C* dl, const C* d, const C* du,
                                                     const C* x, int batchCount, size_t* size) {
    if (size == nullptr) return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    const cusparseStatus_t status = gtsv_interleaved_check(handle, algo, m, dl, d, du, x, batchCount);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// Interleaved layout: element i of system b lives at [i * batchCount + b].
// Algorithm 0 is the Thomas recurrence; 1 and 2 (LU with pivoting, QR) both
// map to pivoted elimination here, which has at least QR's stability.
template <typename C>
static cusparseStatus_t gtsv_interleaved_impl(cusparseHandle_t handle, int algo, int m, C* dlC,
                                              C* dC, C* duC, C* xC, int batchCount) {
    using T = native_t<C>;
    cusparseStatus_t status = gtsv_interleaved_check(handle, algo, m, dlC, dC, duC, xC, batchCount);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    T* dl = nat(dlC);
    T* d = nat(dC);
    T* du = nat(duC);
    T* x = nat(xC);
    for (int b = 0; b < batchCount; ++b) {
        if (algo == 0) {
            if (!thomas_solve(m, dl + b, d + b, du + b, batchCount, x + b, batchCount, 1, 0)) {
                return CUSPARSE_STATUS_ZERO_PIVOT;
            }
            continue;
        }
        std::vector<T> a(static_cast<std::size_t>(m) * 5, T(0));
        std::vector<T> rhs(static_cast<std::size_t>(m));
        auto at = [&](int r, int c) -> T& { return a[static_cast<std::size_t>(r) * 5 + (c - r + 1)]; };
        for (int r = 0; r < m; ++r) {
            const std::size_t k = static_cast<std::size_t>(r) * batchCount + b;
            at(r, r) = d[k];
            if (r > 0) at(r, r - 1) = dl[k];
            if (r + 1 < m) at(r, r + 1) = du[k];
            rhs[static_cast<std::size_t>(r)] = x[k];
        }
        if (!band_solve(m, 1, 1, 1, &a, &rhs)) return CUSPARSE_STATUS_ZERO_PIVOT;
        for (int r = 0; r < m; ++r) x[static_cast<std::size_t>(r) * batchCount + b] = rhs[static_cast<std::size_t>(r)];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gpsv_interleaved_check(cusparseHandle_t handle, int algo, int m,
                                               const C* ds, const C* dl, const C* d, const C* du,
                                               const C* dw, const C* x, int batchCount) {
    SP_NEED_HANDLE(handle);
    // Only algorithm 0 (QR) exists for the pentadiagonal solver.
    if (algo != 0 || m < 0 || batchCount < 1) return CUSPARSE_STATUS_INVALID_VALUE;
    if (m > 0 && (ds == nullptr || dl == nullptr || d == nullptr || du == nullptr ||
                  dw == nullptr || x == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

template <typename C>
static cusparseStatus_t gpsv_interleaved_buffer_size(cusparseHandle_t handle, int algo, int m,
                                                     const C* ds, const C* dl, const C* d,
                                                     const C* du, const C* dw, const C* x,
                                                     int batchCount, size_t* size) {
    if (size == nullptr) return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    const cusparseStatus_t status = gpsv_interleaved_check(handle, algo, m, ds, dl, d, du, dw, x, batchCount);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// Pentadiagonal: ds, dl, d, du, dw hold the diagonals at offsets -2..+2.
template <typename C>
static cusparseStatus_t gpsv_interleaved_impl(cusparseHandle_t handle, int algo, int m, C* dsC,
                                              C* dlC, C* dC, C* duC, C* dwC, C* xC,
                                              int batchCount) {
    using T = native_t<C>;
    cusparseStatus_t status =
        gpsv_interleaved_check(handle, algo, m, dsC, dlC, dC, duC, dwC, xC, batchCount);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    const T* ds = cnat(dsC);
    const T* dl = cnat(dlC);
    const T* d = cnat(dC);
    const T* du = cnat(duC);
    const T* dw = cnat(dwC);
    T* x = nat(xC);
    const int W = 3 * 2 + 2 + 1;
    std::vector<T> a;
    std::vector<T> rhs;
    for (int b = 0; b < batchCount; ++b) {
        a.assign(static_cast<std::size_t>(m) * W, T(0));
        rhs.assign(static_cast<std::size_t>(m), T(0));
        auto at = [&](int r, int c) -> T& { return a[static_cast<std::size_t>(r) * W + (c - r + 2)]; };
        for (int r = 0; r < m; ++r) {
            const std::size_t k = static_cast<std::size_t>(r) * batchCount + b;
            if (r >= 2) at(r, r - 2) = ds[k];
            if (r >= 1) at(r, r - 1) = dl[k];
            at(r, r) = d[k];
            if (r + 1 < m) at(r, r + 1) = du[k];
            if (r + 2 < m) at(r, r + 2) = dw[k];
            rhs[static_cast<std::size_t>(r)] = x[k];
        }
        if (!band_solve(m, 2, 2, 1, &a, &rhs)) return CUSPARSE_STATUS_ZERO_PIVOT;
        for (int r = 0; r < m; ++r) x[static_cast<std::size_t>(r) * batchCount + b] = rhs[static_cast<std::size_t>(r)];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

}  // extern "C++"

extern "C++" {

// ── generic API helpers ─────────────────────────────────────────────────────

template <typename T>
static T& dense_at(const cusparseDnMatDescr* mat, int64_t i, int64_t j) {
    T* v = static_cast<T*>(mat->values);
    return mat->order == CUSPARSE_ORDER_COL ? v[i + j * mat->ld] : v[i * mat->ld + j];
}

static cusparseFormat_t format_of(const cusparseSpMatDescr* mat) {
    switch (mat->format) {
        case CUMETAL_SPMAT_COO: return CUSPARSE_FORMAT_COO;
        case CUMETAL_SPMAT_CSC: return CUSPARSE_FORMAT_CSC;
        default: return CUSPARSE_FORMAT_CSR;
    }
}

static int base_of(const cusparseSpMatDescr* mat) {
    return mat->idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
}

// ── SpSM ────────────────────────────────────────────────────────────────────

static cusparseStatus_t spsm_validate(cusparseHandle_t handle, cusparseOperation_t opA,
                                      cusparseOperation_t opB, const void* alpha,
                                      cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                      cusparseDnMatDescr_t matC, cudaDataType computeType,
                                      cusparseSpSMAlg_t alg) {
    SP_NEED_HANDLE(handle);
    if (alpha == nullptr || matA == nullptr || matB == nullptr || matC == nullptr ||
        !valid_operation(opA) || !valid_operation(opB) || alg != CUSPARSE_SPSM_ALG_DEFAULT) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F && computeType != CUDA_C_32F &&
        computeType != CUDA_C_64F) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->valueType != computeType || matB->valueType != computeType ||
        matC->valueType != computeType || matA->format == CUMETAL_SPMAT_COO ||
        !cumetal_sparse_indices_are_32bit(matA) || matB->batchCount != 1 ||
        matC->batchCount != 1) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    const int64_t n = matA->rows;
    const int64_t nrhs = matC->cols;
    const bool b_transposed = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (matA->rows != matA->cols || matC->rows != n ||
        (b_transposed ? (matB->rows != nrhs || matB->cols != n)
                      : (matB->rows != n || matB->cols != nrhs))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (scalar_pointer_for_mode_size(handle->pointer_mode, alpha, value_size(computeType)) ==
        nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// op(A) * C = alpha * op(B), solved one column at a time.
static cusparseStatus_t spsm_solve_impl(cusparseHandle_t handle, cusparseOperation_t opA,
                                        cusparseOperation_t opB, const void* alpha,
                                        cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                        cusparseDnMatDescr_t matC, cudaDataType computeType) {
    synchronize_handle_stream(handle);
    const void* alpha_ptr =
        scalar_pointer_for_mode_size(handle->pointer_mode, alpha, value_size(computeType));
    bool transpose = false;
    int64_t axis = 0;
    cumetal_sparse_view(matA, opA, &transpose, &axis);
    // A CSC descriptor's arrays are CSR of A-transpose, so the triangle that
    // holds data flips with it.
    cusparseFillMode_t fill = matA->fill;
    if (matA->format == CUMETAL_SPMAT_CSC) {
        fill = fill == CUSPARSE_FILL_MODE_LOWER ? CUSPARSE_FILL_MODE_UPPER
                                                : CUSPARSE_FILL_MODE_LOWER;
    }
    const bool unit = matA->diag == CUSPARSE_DIAG_TYPE_UNIT;
    const bool conjugate = opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    const int base = base_of(matA);
    const int* offsets = static_cast<const int*>(matA->rowOffsets);
    const int* indices = static_cast<const int*>(matA->colInd);
    const bool b_transposed = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const bool b_conjugate = opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    const int64_t n = matA->rows;
    const int64_t nrhs = matC->cols;

    return dispatch_compute_type(computeType, [&](auto tag) -> cusparseStatus_t {
        using T = decltype(tag);
        const T a = *static_cast<const T*>(alpha_ptr);
        const T* vals = static_cast<const T*>(matA->values);
        std::vector<T> x(static_cast<std::size_t>(n));
        for (int64_t j = 0; j < nrhs; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                T b = b_transposed ? dense_at<T>(matB, j, i) : dense_at<T>(matB, i, j);
                if (b_conjugate) b = conj_of(b);
                x[static_cast<std::size_t>(i)] = a * b;
            }
            if (!cumetal_tri_solve(n, offsets, indices, vals, base, fill, unit, transpose,
                                   conjugate, x.data())) {
                return CUSPARSE_STATUS_ZERO_PIVOT;
            }
            for (int64_t i = 0; i < n; ++i) dense_at<T>(matC, i, j) = x[static_cast<std::size_t>(i)];
        }
        return CUSPARSE_STATUS_SUCCESS;
    });
}

// ── SpGEMM ──────────────────────────────────────────────────────────────────

}  // extern "C++"

// The product is computed in cusparseSpGEMM_compute and parked here until
// cusparseSpGEMM_copy writes it into the caller's arrays: cuSPARSE's protocol
// has the caller size C from cusparseSpMatGetSize between those two calls.
struct cusparseSpGEMMDescr {
    bool computed = false;
    int64_t rows = 0;
    int64_t cols = 0;
    cudaDataType valueType = CUDA_R_32F;
    std::vector<int64_t> rowptr;  // 0-based
    std::vector<int64_t> colind;  // 0-based, ascending within a row
    std::vector<unsigned char> values;
};

extern "C++" {

static cusparseStatus_t spgemm_validate(cusparseHandle_t handle, cusparseOperation_t opA,
                                        cusparseOperation_t opB, const void* alpha,
                                        cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB,
                                        const void* beta, cusparseSpMatDescr_t matC,
                                        cudaDataType computeType, cusparseSpGEMMAlg_t alg,
                                        cusparseSpGEMMDescr_t descr) {
    SP_NEED_HANDLE(handle);
    if (alpha == nullptr || beta == nullptr || matA == nullptr || matB == nullptr ||
        matC == nullptr || descr == nullptr || !valid_operation(opA) || !valid_operation(opB) ||
        alg < CUSPARSE_SPGEMM_DEFAULT || alg > CUSPARSE_SPGEMM_ALG3) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (opA != CUSPARSE_OPERATION_NON_TRANSPOSE || opB != CUSPARSE_OPERATION_NON_TRANSPOSE ||
        matA->format != CUMETAL_SPMAT_CSR || matB->format != CUMETAL_SPMAT_CSR ||
        matC->format != CUMETAL_SPMAT_CSR) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F && computeType != CUDA_C_32F &&
        computeType != CUDA_C_64F) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->valueType != computeType || matB->valueType != computeType ||
        matC->valueType != computeType) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (matA->cols != matB->rows || matC->rows != matA->rows || matC->cols != matB->cols) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const std::size_t scalar_size = value_size(computeType);
    if (scalar_pointer_for_mode_size(handle->pointer_mode, alpha, scalar_size) == nullptr ||
        scalar_pointer_for_mode_size(handle->pointer_mode, beta, scalar_size) == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// Gustavson's row-by-row product. Structural entries are kept even when they
// cancel to zero, as cuSPARSE does.
template <typename T>
static cusparseStatus_t spgemm_compute_impl(const cusparseSpMatDescr* A,
                                            const cusparseSpMatDescr* B, T alpha,
                                            cusparseSpGEMMDescr* d) {
    const int64_t m = A->rows;
    const int64_t n = B->cols;
    const int baseA = base_of(A), baseB = base_of(B);
    const T* av = static_cast<const T*>(A->values);
    const T* bv = static_cast<const T*>(B->values);
    std::vector<T> acc(static_cast<std::size_t>(n));
    std::vector<int64_t> mark(static_cast<std::size_t>(n), -1);
    std::vector<int64_t> touched;
    std::vector<T> out;
    d->rowptr.assign(static_cast<std::size_t>(m) + 1, 0);
    d->colind.clear();
    for (int64_t i = 0; i < m; ++i) {
        touched.clear();
        const int64_t a_begin = idx_at(A->rowOffsets, A->rowType, i) - baseA;
        const int64_t a_end = idx_at(A->rowOffsets, A->rowType, i + 1) - baseA;
        for (int64_t ka = a_begin; ka < a_end; ++ka) {
            const int64_t k = idx_at(A->colInd, A->colType, ka) - baseA;
            if (k < 0 || k >= B->rows) return CUSPARSE_STATUS_INVALID_VALUE;
            const int64_t b_begin = idx_at(B->rowOffsets, B->rowType, k) - baseB;
            const int64_t b_end = idx_at(B->rowOffsets, B->rowType, k + 1) - baseB;
            for (int64_t kb = b_begin; kb < b_end; ++kb) {
                const int64_t j = idx_at(B->colInd, B->colType, kb) - baseB;
                if (j < 0 || j >= n) return CUSPARSE_STATUS_INVALID_VALUE;
                if (mark[static_cast<std::size_t>(j)] != i) {
                    mark[static_cast<std::size_t>(j)] = i;
                    acc[static_cast<std::size_t>(j)] = T(0);
                    touched.push_back(j);
                }
                acc[static_cast<std::size_t>(j)] += av[ka] * bv[kb];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (int64_t j : touched) {
            d->colind.push_back(j);
            out.push_back(alpha * acc[static_cast<std::size_t>(j)]);
        }
        d->rowptr[static_cast<std::size_t>(i) + 1] = static_cast<int64_t>(d->colind.size());
    }
    d->values.resize(out.size() * sizeof(T));
    if (!out.empty()) std::memcpy(d->values.data(), out.data(), d->values.size());
    return CUSPARSE_STATUS_SUCCESS;
}

// The offsets of an operand are only trustworthy once checked: a descriptor can
// be built over anything.
static bool spgemm_operand_valid(const cusparseSpMatDescr* M) {
    if (M->rowOffsets == nullptr || (M->nnz > 0 && (M->colInd == nullptr || M->values == nullptr))) {
        return false;
    }
    const int base = base_of(M);
    if (idx_at(M->rowOffsets, M->rowType, 0) - base != 0 ||
        idx_at(M->rowOffsets, M->rowType, M->rows) - base != M->nnz) {
        return false;
    }
    for (int64_t r = 0; r < M->rows; ++r) {
        if (idx_at(M->rowOffsets, M->rowType, r + 1) < idx_at(M->rowOffsets, M->rowType, r)) {
            return false;
        }
    }
    return true;
}

// ── dense <-> sparse ────────────────────────────────────────────────────────

static cusparseStatus_t dense_sparse_validate(cusparseHandle_t handle,
                                              const cusparseDnMatDescr* dense,
                                              const cusparseSpMatDescr* sparse) {
    SP_NEED_HANDLE(handle);
    if (dense == nullptr || sparse == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (dense->valueType != sparse->valueType || value_size(dense->valueType) == 0 ||
        dense->batchCount != 1) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (dense->rows != sparse->rows || dense->cols != sparse->cols) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

static const void* dense_element(const cusparseDnMatDescr* mat, int64_t i, int64_t j) {
    const std::size_t size = value_size(mat->valueType);
    const auto* v = static_cast<const unsigned char*>(mat->values);
    const int64_t at = mat->order == CUSPARSE_ORDER_COL ? i + j * mat->ld : i * mat->ld + j;
    return v + static_cast<std::size_t>(at) * size;
}

// Stored nonzeros of a dense matrix in the order the sparse format lists them:
// row-major for CSR and COO, column-major for CSC. `emit` receives (row, col,
// element pointer).
template <typename F>
static void for_each_dense_nonzero(const cusparseDnMatDescr* dense, bool column_major, F&& emit) {
    if (column_major) {
        for (int64_t j = 0; j < dense->cols; ++j)
            for (int64_t i = 0; i < dense->rows; ++i) {
                const void* e = dense_element(dense, i, j);
                if (value_nonzero(dense->valueType, e)) emit(i, j, e);
            }
    } else {
        for (int64_t i = 0; i < dense->rows; ++i)
            for (int64_t j = 0; j < dense->cols; ++j) {
                const void* e = dense_element(dense, i, j);
                if (value_nonzero(dense->valueType, e)) emit(i, j, e);
            }
    }
}

}  // extern "C++"

// ── sparse vectors ──────────────────────────────────────────────────────────

cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz,
                                     void* indices, void* values, cusparseIndexType_t idxType,
                                     cusparseIndexBase_t idxBase, cudaDataType valueType) {
    if (spVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *spVecDescr = nullptr;
    if (size < 0 || nnz < 0 || nnz > size || (nnz > 0 && (indices == nullptr || values == nullptr)) ||
        !valid_index_type(idxType) || !valid_index_base(idxBase) || !valid_data_type(valueType)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    auto* v = new (std::nothrow) cusparseSpVecDescr();
    if (v == nullptr) return CUSPARSE_STATUS_ALLOC_FAILED;
    v->size = size;
    v->nnz = nnz;
    v->indices = indices;
    v->values = values;
    v->idxType = idxType;
    v->idxBase = idxBase;
    v->valueType = valueType;
    *spVecDescr = v;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t spVecDescr) {
    delete spVecDescr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGet(cusparseSpVecDescr_t spVecDescr, int64_t* size, int64_t* nnz,
                                  void** indices, void** values, cusparseIndexType_t* idxType,
                                  cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    if (spVecDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (size != nullptr) *size = spVecDescr->size;
    if (nnz != nullptr) *nnz = spVecDescr->nnz;
    if (indices != nullptr) *indices = spVecDescr->indices;
    if (values != nullptr) *values = spVecDescr->values;
    if (idxType != nullptr) *idxType = spVecDescr->idxType;
    if (idxBase != nullptr) *idxBase = spVecDescr->idxBase;
    if (valueType != nullptr) *valueType = spVecDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetIndexBase(cusparseSpVecDescr_t spVecDescr,
                                           cusparseIndexBase_t* idxBase) {
    if (spVecDescr == nullptr || idxBase == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *idxBase = spVecDescr->idxBase;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetValues(cusparseSpVecDescr_t spVecDescr, void** values) {
    if (spVecDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = spVecDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecSetValues(cusparseSpVecDescr_t spVecDescr, void* values) {
    if (spVecDescr == nullptr || (values == nullptr && spVecDescr->nnz > 0)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    spVecDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── sparse / dense matrix accessors ─────────────────────────────────────────

cusparseStatus_t cusparseCooGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                int64_t* nnz, void** cooRowInd, void** cooColInd,
                                void** cooValues, cusparseIndexType_t* idxType,
                                cusparseIndexBase_t* idxBase, cudaDataType* valueType) {
    if (spMatDescr == nullptr || spMatDescr->format != CUMETAL_SPMAT_COO) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) *rows = spMatDescr->rows;
    if (cols != nullptr) *cols = spMatDescr->cols;
    if (nnz != nullptr) *nnz = spMatDescr->nnz;
    if (cooRowInd != nullptr) *cooRowInd = spMatDescr->rowOffsets;
    if (cooColInd != nullptr) *cooColInd = spMatDescr->colInd;
    if (cooValues != nullptr) *cooValues = spMatDescr->values;
    if (idxType != nullptr) *idxType = spMatDescr->rowType;
    if (idxBase != nullptr) *idxBase = spMatDescr->idxBase;
    if (valueType != nullptr) *valueType = spMatDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrGet(cusparseSpMatDescr_t spMatDescr, int64_t* rows, int64_t* cols,
                                int64_t* nnz, void** csrRowOffsets, void** csrColInd,
                                void** csrValues, cusparseIndexType_t* csrRowOffsetsType,
                                cusparseIndexType_t* csrColIndType, cusparseIndexBase_t* idxBase,
                                cudaDataType* valueType) {
    if (spMatDescr == nullptr || spMatDescr->format != CUMETAL_SPMAT_CSR) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (rows != nullptr) *rows = spMatDescr->rows;
    if (cols != nullptr) *cols = spMatDescr->cols;
    if (nnz != nullptr) *nnz = spMatDescr->nnz;
    if (csrRowOffsets != nullptr) *csrRowOffsets = spMatDescr->rowOffsets;
    if (csrColInd != nullptr) *csrColInd = spMatDescr->colInd;
    if (csrValues != nullptr) *csrValues = spMatDescr->values;
    if (csrRowOffsetsType != nullptr) *csrRowOffsetsType = spMatDescr->rowType;
    if (csrColIndType != nullptr) *csrColIndType = spMatDescr->colType;
    if (idxBase != nullptr) *idxBase = spMatDescr->idxBase;
    if (valueType != nullptr) *valueType = spMatDescr->valueType;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr, void* csrRowOffsets,
                                        void* csrColInd, void* csrValues) {
    if (spMatDescr == nullptr || spMatDescr->format != CUMETAL_SPMAT_CSR ||
        csrRowOffsets == nullptr ||
        (spMatDescr->nnz > 0 && (csrColInd == nullptr || csrValues == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr->rowOffsets = csrRowOffsets;
    spMatDescr->colInd = csrColInd;
    spMatDescr->values = csrValues;
    // The structure changed: the cached longest row (see the INVARIANT on the
    // descriptor) describes the old offsets.
    spMatDescr->longest_row = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetFormat(cusparseSpMatDescr_t spMatDescr, cusparseFormat_t* format) {
    if (spMatDescr == nullptr || format == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *format = format_of(spMatDescr);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetIndexBase(cusparseSpMatDescr_t spMatDescr,
                                           cusparseIndexBase_t* idxBase) {
    if (spMatDescr == nullptr || idxBase == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *idxBase = spMatDescr->idxBase;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetValues(cusparseSpMatDescr_t spMatDescr, void** values) {
    if (spMatDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = spMatDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

// Values may be rewritten in place; only the sparsity structure is fixed.
cusparseStatus_t cusparseSpMatSetValues(cusparseSpMatDescr_t spMatDescr, void* values) {
    if (spMatDescr == nullptr || (values == nullptr && spMatDescr->nnz > 0)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    spMatDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr, int64_t* rows,
                                      int64_t* cols, int64_t* nnz) {
    if (spMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows != nullptr) *rows = spMatDescr->rows;
    if (cols != nullptr) *cols = spMatDescr->cols;
    if (nnz != nullptr) *nnz = spMatDescr->nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

// Sparse matrices are not batched here, so the count is always one.
cusparseStatus_t cusparseSpMatGetStridedBatch(cusparseSpMatDescr_t spMatDescr, int* batchCount) {
    if (spMatDescr == nullptr || batchCount == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *batchCount = 1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGet(cusparseDnMatDescr_t dnMatDescr, int64_t* rows, int64_t* cols,
                                  int64_t* ld, void** values, cudaDataType* valueType,
                                  cusparseOrder_t* order) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (rows != nullptr) *rows = dnMatDescr->rows;
    if (cols != nullptr) *cols = dnMatDescr->cols;
    if (ld != nullptr) *ld = dnMatDescr->ld;
    if (values != nullptr) *values = dnMatDescr->values;
    if (valueType != nullptr) *valueType = dnMatDescr->valueType;
    if (order != nullptr) *order = dnMatDescr->order;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetValues(cusparseDnMatDescr_t dnMatDescr, void** values) {
    if (dnMatDescr == nullptr || values == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *values = dnMatDescr->values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetValues(cusparseDnMatDescr_t dnMatDescr, void* values) {
    if (dnMatDescr == nullptr || (values == nullptr && dnMatDescr->rows > 0 && dnMatDescr->cols > 0)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    dnMatDescr->values = values;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatGetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int* batchCount,
                                              int64_t* batchStride) {
    if (dnMatDescr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (batchCount != nullptr) *batchCount = dnMatDescr->batchCount;
    if (batchStride != nullptr) *batchStride = dnMatDescr->batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDnMatSetStridedBatch(cusparseDnMatDescr_t dnMatDescr, int batchCount,
                                              int64_t batchStride) {
    if (dnMatDescr == nullptr || batchCount < 1 || batchStride < 0) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    dnMatDescr->batchCount = batchCount;
    dnMatDescr->batchStride = batchStride;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpVV / Gather ───────────────────────────────────────────────────────────

static cusparseStatus_t spvv_validate(cusparseHandle_t handle, cusparseOperation_t opX,
                                      cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                      const void* result, cudaDataType computeType) {
    SP_NEED_HANDLE(handle);
    if (vecX == nullptr || vecY == nullptr || result == nullptr || !valid_operation(opX) ||
        vecX->size != vecY->size) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F && computeType != CUDA_C_32F &&
        computeType != CUDA_C_64F) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (vecX->valueType != computeType || vecY->valueType != computeType) {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVV_bufferSize(cusparseHandle_t handle, cusparseOperation_t opX,
                                         cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
                                         const void* result, cudaDataType computeType,
                                         size_t* bufferSize) {
    if (bufferSize == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status = spvv_validate(handle, opX, vecX, vecY, result, computeType);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *bufferSize = 1;
    return CUSPARSE_STATUS_SUCCESS;
}

// result = sum_i x_i * y[idx_i], with x conjugated for the conjugate-transpose
// operation. `result` is a host or device pointer according to the handle's
// pointer mode; both are directly writable over unified memory.
cusparseStatus_t cusparseSpVV(cusparseHandle_t handle, cusparseOperation_t opX,
                              cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY, void* result,
                              cudaDataType computeType, void* /*externalBuffer*/) {
    cusparseStatus_t status = spvv_validate(handle, opX, vecX, vecY, result, computeType);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    const int base = vecX->idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    const bool conjugate = opX == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    return dispatch_compute_type(computeType, [&](auto tag) -> cusparseStatus_t {
        using T = decltype(tag);
        const T* x = static_cast<const T*>(vecX->values);
        const T* y = static_cast<const T*>(vecY->values);
        T acc{};
        for (int64_t e = 0; e < vecX->nnz; ++e) {
            const int64_t i = idx_at(vecX->indices, vecX->idxType, e) - base;
            if (i < 0 || i >= vecY->size) return CUSPARSE_STATUS_INVALID_VALUE;
            acc += (conjugate ? conj_of(x[e]) : x[e]) * y[i];
        }
        *static_cast<T*>(result) = acc;
        return CUSPARSE_STATUS_SUCCESS;
    });
}

// x.values[e] = y[x.indices[e]]
cusparseStatus_t cusparseGather(cusparseHandle_t handle, cusparseDnVecDescr_t vecY,
                                cusparseSpVecDescr_t vecX) {
    SP_NEED_HANDLE(handle);
    if (vecX == nullptr || vecY == nullptr || vecX->size != vecY->size ||
        (vecX->nnz > 0 && (vecX->indices == nullptr || vecX->values == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const std::size_t size = value_size(vecX->valueType);
    if (vecX->valueType != vecY->valueType || size == 0) return CUSPARSE_STATUS_NOT_SUPPORTED;
    synchronize_handle_stream(handle);
    const int base = vecX->idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    const auto* y = static_cast<const unsigned char*>(vecY->values);
    auto* x = static_cast<unsigned char*>(vecX->values);
    for (int64_t e = 0; e < vecX->nnz; ++e) {
        const int64_t i = idx_at(vecX->indices, vecX->idxType, e) - base;
        if (i < 0 || i >= vecY->size) return CUSPARSE_STATUS_INVALID_VALUE;
        std::memcpy(x + static_cast<std::size_t>(e) * size, y + static_cast<std::size_t>(i) * size, size);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpSM ────────────────────────────────────────────────────────────────────

// Nothing to carry between phases: the solve reads the CSR arrays directly.
struct cusparseSpSMDescr { char reserved = 0; };

cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t* descr) {
    if (descr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new (std::nothrow) cusparseSpSMDescr();
    return *descr == nullptr ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA,
                                         cusparseOperation_t opB, const void* alpha,
                                         cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                         cusparseDnMatDescr_t matC, cudaDataType computeType,
                                         cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                         size_t* bufferSize) {
    if (spsmDescr == nullptr || bufferSize == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status =
        spsm_validate(handle, opA, opB, alpha, matA, matB, matC, computeType, alg);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *bufferSize = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// The solve walks the CSR arrays directly, so there is nothing to analyse.
cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle, cusparseOperation_t opA,
                                       cusparseOperation_t opB, const void* alpha,
                                       cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                       cusparseDnMatDescr_t matC, cudaDataType computeType,
                                       cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr,
                                       void* /*externalBuffer*/) {
    if (spsmDescr == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    return spsm_validate(handle, opA, opB, alpha, matA, matB, matC, computeType, alg);
}

cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle, cusparseOperation_t opA,
                                    cusparseOperation_t opB, const void* alpha,
                                    cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                                    cusparseDnMatDescr_t matC, cudaDataType computeType,
                                    cusparseSpSMAlg_t alg, cusparseSpSMDescr_t spsmDescr) {
    if (spsmDescr == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status =
        spsm_validate(handle, opA, opB, alpha, matA, matB, matC, computeType, alg);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    return spsm_solve_impl(handle, opA, opB, alpha, matA, matB, matC, computeType);
}

// ── SpGEMM ──────────────────────────────────────────────────────────────────

cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t* descr) {
    if (descr == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new (std::nothrow) cusparseSpGEMMDescr();
    return *descr == nullptr ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}

// cuSPARSE's three-phase protocol sizes two scratch buffers. None is needed
// here, but the sizes are reported non-zero and a null buffer means "query
// only" exactly as it does there.
cusparseStatus_t cusparseSpGEMM_workEstimation(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize1, void* externalBuffer1) {
    const cusparseStatus_t status = spgemm_validate(handle, opA, opB, alpha, matA, matB, beta, matC,
                                                    computeType, alg, spgemmDescr);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (bufferSize1 == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (externalBuffer1 == nullptr) *bufferSize1 = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// Computes C = alpha * A * B when given a buffer; with a null buffer it only
// reports the size. After the computing call cusparseSpMatGetSize(C) reports
// the product's nnz so the caller can allocate C's index and value arrays.
cusparseStatus_t cusparseSpGEMM_compute(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr, size_t* bufferSize2, void* externalBuffer2) {
    cusparseStatus_t status = spgemm_validate(handle, opA, opB, alpha, matA, matB, beta, matC,
                                              computeType, alg, spgemmDescr);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (bufferSize2 == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (externalBuffer2 == nullptr) {
        *bufferSize2 = kNoWorkspaceBytes;
        return CUSPARSE_STATUS_SUCCESS;
    }
    synchronize_handle_stream(handle);
    const std::size_t scalar_size = value_size(computeType);
    const void* alpha_ptr = scalar_pointer_for_mode_size(handle->pointer_mode, alpha, scalar_size);
    const void* beta_ptr = scalar_pointer_for_mode_size(handle->pointer_mode, beta, scalar_size);
    // C = alpha*A*B + beta*C is only defined here for beta == 0: the product's
    // pattern replaces C's, so a nonzero beta would need C's old contents merged.
    const bool beta_zero = dispatch_compute_type(computeType, [&](auto tag) -> cusparseStatus_t {
        using T = decltype(tag);
        return *static_cast<const T*>(beta_ptr) == T(0) ? CUSPARSE_STATUS_SUCCESS
                                                         : CUSPARSE_STATUS_INVALID_VALUE;
    }) == CUSPARSE_STATUS_SUCCESS;
    if (!beta_zero) return CUSPARSE_STATUS_NOT_SUPPORTED;
    if (!spgemm_operand_valid(matA) || !spgemm_operand_valid(matB)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    status = dispatch_compute_type(computeType, [&](auto tag) -> cusparseStatus_t {
        using T = decltype(tag);
        return spgemm_compute_impl<T>(matA, matB, *static_cast<const T*>(alpha_ptr), spgemmDescr);
    });
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    spgemmDescr->computed = true;
    spgemmDescr->rows = matA->rows;
    spgemmDescr->cols = matB->cols;
    spgemmDescr->valueType = computeType;
    matC->nnz = static_cast<int64_t>(spgemmDescr->colind.size());
    matC->longest_row = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

// Writes the product computed by cusparseSpGEMM_compute into C's arrays. C's
// row-offset array must already be set, and its index/value arrays must hold
// the nnz that cusparseSpMatGetSize reported.
cusparseStatus_t cusparseSpGEMM_copy(
    cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha,
    cusparseSpMatDescr_t matA, cusparseSpMatDescr_t matB, const void* beta,
    cusparseSpMatDescr_t matC, cudaDataType computeType, cusparseSpGEMMAlg_t alg,
    cusparseSpGEMMDescr_t spgemmDescr) {
    const cusparseStatus_t status = spgemm_validate(handle, opA, opB, alpha, matA, matB, beta, matC,
                                                    computeType, alg, spgemmDescr);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    const int64_t nnz = static_cast<int64_t>(spgemmDescr->colind.size());
    if (!spgemmDescr->computed || spgemmDescr->valueType != computeType ||
        spgemmDescr->rows != matC->rows || spgemmDescr->cols != matC->cols ||
        matC->nnz != nnz || matC->rowOffsets == nullptr ||
        (nnz > 0 && (matC->colInd == nullptr || matC->values == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const int base = base_of(matC);
    for (int64_t i = 0; i <= matC->rows; ++i) {
        idx_put(matC->rowOffsets, matC->rowType, i, spgemmDescr->rowptr[static_cast<std::size_t>(i)] + base);
    }
    for (int64_t e = 0; e < nnz; ++e) {
        idx_put(matC->colInd, matC->colType, e, spgemmDescr->colind[static_cast<std::size_t>(e)] + base);
    }
    if (nnz > 0) std::memcpy(matC->values, spgemmDescr->values.data(), spgemmDescr->values.size());
    return CUSPARSE_STATUS_SUCCESS;
}

// ── dense <-> sparse ────────────────────────────────────────────────────────

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle,
                                                  cusparseSpMatDescr_t matA,
                                                  cusparseDnMatDescr_t matB,
                                                  cusparseSparseToDenseAlg_t alg,
                                                  size_t* bufferSize) {
    if (bufferSize == nullptr || alg != CUSPARSE_SPARSETODENSE_ALG_DEFAULT) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status = dense_sparse_validate(handle, matB, matA);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *bufferSize = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// B = dense(A). Every element of B is written, zeros included.
cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle, cusparseSpMatDescr_t matA,
                                       cusparseDnMatDescr_t matB, cusparseSparseToDenseAlg_t alg,
                                       void* /*externalBuffer*/) {
    cusparseStatus_t status = dense_sparse_validate(handle, matB, matA);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (alg != CUSPARSE_SPARSETODENSE_ALG_DEFAULT) return CUSPARSE_STATUS_INVALID_VALUE;
    if ((matB->rows > 0 && matB->cols > 0 && matB->values == nullptr) ||
        (matA->nnz > 0 && (matA->colInd == nullptr || matA->values == nullptr)) ||
        matA->rowOffsets == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const std::size_t size = value_size(matA->valueType);
    const int base = base_of(matA);
    auto* out = static_cast<unsigned char*>(matB->values);
    auto cell = [&](int64_t i, int64_t j) {
        const int64_t at = matB->order == CUSPARSE_ORDER_COL ? i + j * matB->ld : i * matB->ld + j;
        return out + static_cast<std::size_t>(at) * size;
    };
    // Validate every coordinate before touching B, so a malformed matrix does
    // not leave B half-written.
    auto coordinate_ok = [&](int64_t r, int64_t c) {
        return r >= 0 && r < matA->rows && c >= 0 && c < matA->cols;
    };
    const auto* values = static_cast<const unsigned char*>(matA->values);
    const int64_t axis = matA->format == CUMETAL_SPMAT_CSC ? matA->cols : matA->rows;
    if (matA->format != CUMETAL_SPMAT_COO) {
        if (idx_at(matA->rowOffsets, matA->rowType, 0) - base != 0 ||
            idx_at(matA->rowOffsets, matA->rowType, axis) - base != matA->nnz) {
            return CUSPARSE_STATUS_INVALID_VALUE;
        }
        for (int64_t s = 0; s < axis; ++s) {
            const int64_t begin = idx_at(matA->rowOffsets, matA->rowType, s) - base;
            const int64_t end = idx_at(matA->rowOffsets, matA->rowType, s + 1) - base;
            if (begin > end || end > matA->nnz) return CUSPARSE_STATUS_INVALID_VALUE;
            for (int64_t e = begin; e < end; ++e) {
                const int64_t other = idx_at(matA->colInd, matA->colType, e) - base;
                if (!(matA->format == CUMETAL_SPMAT_CSC ? coordinate_ok(other, s)
                                                       : coordinate_ok(s, other))) {
                    return CUSPARSE_STATUS_INVALID_VALUE;
                }
            }
        }
    } else {
        for (int64_t e = 0; e < matA->nnz; ++e) {
            if (!coordinate_ok(idx_at(matA->rowOffsets, matA->rowType, e) - base,
                               idx_at(matA->colInd, matA->colType, e) - base)) {
                return CUSPARSE_STATUS_INVALID_VALUE;
            }
        }
    }
    for (int64_t j = 0; j < matB->cols; ++j)
        for (int64_t i = 0; i < matB->rows; ++i) std::memset(cell(i, j), 0, size);
    auto store = [&](int64_t r, int64_t c, int64_t e) {
        std::memcpy(cell(r, c), values + static_cast<std::size_t>(e) * size, size);
    };
    if (matA->format == CUMETAL_SPMAT_COO) {
        for (int64_t e = 0; e < matA->nnz; ++e)
            store(idx_at(matA->rowOffsets, matA->rowType, e) - base,
                  idx_at(matA->colInd, matA->colType, e) - base, e);
    } else {
        for (int64_t s = 0; s < axis; ++s) {
            const int64_t begin = idx_at(matA->rowOffsets, matA->rowType, s) - base;
            const int64_t end = idx_at(matA->rowOffsets, matA->rowType, s + 1) - base;
            for (int64_t e = begin; e < end; ++e) {
                const int64_t other = idx_at(matA->colInd, matA->colType, e) - base;
                if (matA->format == CUMETAL_SPMAT_CSC) store(other, s, e);
                else store(s, other, e);
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle,
                                                  cusparseDnMatDescr_t matA,
                                                  cusparseSpMatDescr_t matB,
                                                  cusparseDenseToSparseAlg_t alg,
                                                  size_t* bufferSize) {
    if (bufferSize == nullptr || alg != CUSPARSE_DENSETOSPARSE_ALG_DEFAULT) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status = dense_sparse_validate(handle, matA, matB);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *bufferSize = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// Counts A's nonzeros and fixes B's nnz and (for CSR/CSC) its offset array.
// The caller then allocates B's index and value arrays, points B at them with
// cusparseCsrSetPointers, and calls cusparseDenseToSparse_convert.
cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle,
                                                cusparseDnMatDescr_t matA,
                                                cusparseSpMatDescr_t matB,
                                                cusparseDenseToSparseAlg_t alg,
                                                void* /*externalBuffer*/) {
    cusparseStatus_t status = dense_sparse_validate(handle, matA, matB);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (alg != CUSPARSE_DENSETOSPARSE_ALG_DEFAULT) return CUSPARSE_STATUS_INVALID_VALUE;
    if ((matA->rows > 0 && matA->cols > 0 && matA->values == nullptr) ||
        (matB->format != CUMETAL_SPMAT_COO && matB->rowOffsets == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const bool column_major = matB->format == CUMETAL_SPMAT_CSC;
    const int64_t axis = column_major ? matB->cols : matB->rows;
    std::vector<int64_t> counts(static_cast<std::size_t>(axis), 0);
    int64_t total = 0;
    for_each_dense_nonzero(matA, column_major, [&](int64_t i, int64_t j, const void*) {
        ++counts[static_cast<std::size_t>(column_major ? j : i)];
        ++total;
    });
    if (matB->format != CUMETAL_SPMAT_COO) {
        const int base = base_of(matB);
        int64_t running = 0;
        idx_put(matB->rowOffsets, matB->rowType, 0, base);
        for (int64_t s = 0; s < axis; ++s) {
            running += counts[static_cast<std::size_t>(s)];
            idx_put(matB->rowOffsets, matB->rowType, s + 1, running + base);
        }
    }
    matB->nnz = total;
    matB->longest_row = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_convert(cusparseHandle_t handle, cusparseDnMatDescr_t matA,
                                               cusparseSpMatDescr_t matB,
                                               cusparseDenseToSparseAlg_t alg,
                                               void* /*externalBuffer*/) {
    cusparseStatus_t status = dense_sparse_validate(handle, matA, matB);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    if (alg != CUSPARSE_DENSETOSPARSE_ALG_DEFAULT) return CUSPARSE_STATUS_INVALID_VALUE;
    if ((matA->rows > 0 && matA->cols > 0 && matA->values == nullptr) ||
        (matB->nnz > 0 && (matB->colInd == nullptr || matB->values == nullptr)) ||
        matB->rowOffsets == nullptr) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const bool column_major = matB->format == CUMETAL_SPMAT_CSC;
    const bool coo = matB->format == CUMETAL_SPMAT_COO;
    const int base = base_of(matB);
    const std::size_t size = value_size(matB->valueType);
    // The matrix must still have the nnz the analysis counted; if A changed in
    // between, the arrays the caller sized no longer fit.
    int64_t total = 0;
    for_each_dense_nonzero(matA, column_major, [&](int64_t, int64_t, const void*) { ++total; });
    if (total != matB->nnz) return CUSPARSE_STATUS_INVALID_VALUE;
    int64_t e = 0;
    auto* out = static_cast<unsigned char*>(matB->values);
    for_each_dense_nonzero(matA, column_major, [&](int64_t i, int64_t j, const void* element) {
        if (coo) {
            idx_put(matB->rowOffsets, matB->rowType, e, i + base);
            idx_put(matB->colInd, matB->colType, e, j + base);
        } else {
            idx_put(matB->colInd, matB->colType, e, (column_major ? i : j) + base);
        }
        std::memcpy(out + static_cast<std::size_t>(e) * size, element, size);
        ++e;
    });
    return CUSPARSE_STATUS_SUCCESS;
}

// ── CSR -> CSC ──────────────────────────────────────────────────────────────

static cusparseStatus_t csr2csc_validate(cusparseHandle_t handle, int m, int n, int nnz,
                                         const void* csrVal, const int* csrRowPtr,
                                         const int* csrColInd, void* cscVal, int* cscColPtr,
                                         int* cscRowInd, cudaDataType valType,
                                         cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                         cusparseCsr2CscAlg_t alg) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || n < 0 || nnz < 0 || !valid_index_base(idxBase) ||
        (copyValues != CUSPARSE_ACTION_SYMBOLIC && copyValues != CUSPARSE_ACTION_NUMERIC) ||
        (alg != CUSPARSE_CSR2CSC_ALG1 && alg != CUSPARSE_CSR2CSC_ALG2) ||
        value_size(valType) == 0 || csrRowPtr == nullptr || cscColPtr == nullptr ||
        (nnz > 0 && (csrColInd == nullptr || cscRowInd == nullptr)) ||
        (copyValues == CUSPARSE_ACTION_NUMERIC && nnz > 0 && (csrVal == nullptr || cscVal == nullptr))) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz,
                                               const void* csrVal, const int* csrRowPtr,
                                               const int* csrColInd, void* cscVal, int* cscColPtr,
                                               int* cscRowInd, cudaDataType valType,
                                               cusparseAction_t copyValues,
                                               cusparseIndexBase_t idxBase,
                                               cusparseCsr2CscAlg_t alg, size_t* bufferSize) {
    if (bufferSize == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status = csr2csc_validate(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                                     cscVal, cscColPtr, cscRowInd, valType,
                                                     copyValues, idxBase, alg);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    *bufferSize = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

// Counting-sort transpose. Rows are visited in ascending order, so each column
// lists its row indices ascending. With CUSPARSE_ACTION_SYMBOLIC only the
// structure is produced and the value arrays are never touched.
cusparseStatus_t cusparseCsr2cscEx2(cusparseHandle_t handle, int m, int n, int nnz,
                                    const void* csrVal, const int* csrRowPtr,
                                    const int* csrColInd, void* cscVal, int* cscColPtr,
                                    int* cscRowInd, cudaDataType valType,
                                    cusparseAction_t copyValues, cusparseIndexBase_t idxBase,
                                    cusparseCsr2CscAlg_t alg, void* /*buffer*/) {
    cusparseStatus_t status = csr2csc_validate(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                               cscVal, cscColPtr, cscRowInd, valType, copyValues,
                                               idxBase, alg);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    synchronize_handle_stream(handle);
    const int base = idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    if (!csr_structure_valid(m, n, nnz, csrRowPtr, csrColInd, base)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    std::vector<int> next(static_cast<std::size_t>(n) + 1, 0);
    for (int e = 0; e < nnz; ++e) ++next[static_cast<std::size_t>(csrColInd[e] - base) + 1];
    for (int c = 0; c < n; ++c) next[static_cast<std::size_t>(c) + 1] += next[static_cast<std::size_t>(c)];
    for (int c = 0; c <= n; ++c) cscColPtr[c] = next[static_cast<std::size_t>(c)] + base;
    const std::size_t size = value_size(valType);
    const auto* in = static_cast<const unsigned char*>(csrVal);
    auto* out = static_cast<unsigned char*>(cscVal);
    for (int r = 0; r < m; ++r) {
        for (int e = csrRowPtr[r] - base; e < csrRowPtr[r + 1] - base; ++e) {
            const int dest = next[static_cast<std::size_t>(csrColInd[e] - base)]++;
            cscRowInd[dest] = r + base;
            if (copyValues == CUSPARSE_ACTION_NUMERIC) {
                std::memcpy(out + static_cast<std::size_t>(dest) * size,
                            in + static_cast<std::size_t>(e) * size, size);
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── COO <-> CSR, sorting, permutation ───────────────────────────────────────

cusparseStatus_t cusparseXcoo2csr(cusparseHandle_t handle, const int* cooRowInd, int nnz, int m,
                                  int* csrSortedRowPtr, cusparseIndexBase_t idxBase) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || nnz < 0 || !valid_index_base(idxBase) || csrSortedRowPtr == nullptr ||
        (nnz > 0 && cooRowInd == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    synchronize_handle_stream(handle);
    const int base = idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    std::vector<int> counts(static_cast<std::size_t>(m), 0);
    for (int e = 0; e < nnz; ++e) {
        const int r = cooRowInd[e] - base;
        if (r < 0 || r >= m) return CUSPARSE_STATUS_INVALID_VALUE;
        ++counts[static_cast<std::size_t>(r)];
    }
    csrSortedRowPtr[0] = base;
    for (int r = 0; r < m; ++r) csrSortedRowPtr[r + 1] = csrSortedRowPtr[r] + counts[static_cast<std::size_t>(r)];
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsr2coo(cusparseHandle_t handle, const int* csrSortedRowPtr, int nnz,
                                  int m, int* cooRowInd, cusparseIndexBase_t idxBase) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || nnz < 0 || !valid_index_base(idxBase) || csrSortedRowPtr == nullptr ||
        (nnz > 0 && cooRowInd == nullptr)) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    const int base = idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0;
    synchronize_handle_stream(handle);
    if (!offsets_valid(m, nnz, csrSortedRowPtr, base)) return CUSPARSE_STATUS_INVALID_VALUE;
    for (int r = 0; r < m; ++r) {
        for (int e = csrSortedRowPtr[r] - base; e < csrSortedRowPtr[r + 1] - base; ++e) {
            cooRowInd[e] = r + base;
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateIdentityPermutation(cusparseHandle_t handle, int n, int* p) {
    SP_NEED_HANDLE(handle);
    if (n < 0 || (n > 0 && p == nullptr)) return CUSPARSE_STATUS_INVALID_VALUE;
    synchronize_handle_stream(handle);
    for (int i = 0; i < n; ++i) p[i] = i;
    return CUSPARSE_STATUS_SUCCESS;
}

static cusparseStatus_t sort_buffer_size(cusparseHandle_t handle, int m, int n, int nnz,
                                         size_t* size) {
    SP_NEED_HANDLE(handle);
    if (m < 0 || n < 0 || nnz < 0 || size == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *size = kNoWorkspaceBytes;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcoosort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int*, const int*, size_t* pBufferSizeInBytes) {
    return sort_buffer_size(handle, m, n, nnz, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcoosortByRow(cusparseHandle_t handle, int m, int n, int nnz,
                                       int* cooRows, int* cooCols, int* P, void*) {
    return coo_sort_impl(handle, m, n, nnz, cooRows, cooCols, P, true);
}

cusparseStatus_t cusparseXcoosortByColumn(cusparseHandle_t handle, int m, int n, int nnz,
                                          int* cooRows, int* cooCols, int* P, void*) {
    return coo_sort_impl(handle, m, n, nnz, cooRows, cooCols, P, false);
}

cusparseStatus_t cusparseXcsrsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int*, const int*, size_t* pBufferSizeInBytes) {
    return sort_buffer_size(handle, m, n, nnz, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcsrsort(cusparseHandle_t handle, int m, int n, int nnz,
                                  const cusparseMatDescr_t descrA, const int* csrRowPtr,
                                  int* csrColInd, int* P, void*) {
    return sort_compressed_impl(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, P);
}

cusparseStatus_t cusparseXcscsort_bufferSizeExt(cusparseHandle_t handle, int m, int n, int nnz,
                                                const int*, const int*, size_t* pBufferSizeInBytes) {
    return sort_buffer_size(handle, m, n, nnz, pBufferSizeInBytes);
}

cusparseStatus_t cusparseXcscsort(cusparseHandle_t handle, int m, int n, int nnz,
                                  const cusparseMatDescr_t descrA, const int* cscColPtr,
                                  int* cscRowInd, int* P, void*) {
    return sort_compressed_impl(handle, n, m, nnz, descrA, cscColPtr, cscRowInd, P);
}

// ── csrgeam2 pattern ────────────────────────────────────────────────────────

// Row pointers and nnz of C = A + B: the union of the two patterns, row by row.
cusparseStatus_t cusparseXcsrgeam2Nnz(cusparseHandle_t handle, int m, int n,
                                      const cusparseMatDescr_t descrA, int nnzA,
                                      const int* csrRowPtrA, const int* csrColIndA,
                                      const cusparseMatDescr_t descrB, int nnzB,
                                      const int* csrRowPtrB, const int* csrColIndB,
                                      const cusparseMatDescr_t descrC, int* csrRowPtrC,
                                      int* nnzTotalDevHostPtr, void* /*workspace*/) {
    if (csrRowPtrC == nullptr || nnzTotalDevHostPtr == nullptr) {
        return handle == nullptr ? CUSPARSE_STATUS_NOT_INITIALIZED : CUSPARSE_STATUS_INVALID_VALUE;
    }
    const cusparseStatus_t status = geam2_validate(handle, m, n, descrA, nnzA, csrRowPtrA, csrColIndA,
                                                   descrB, nnzB, csrRowPtrB, csrColIndB, descrC);
    if (status != CUSPARSE_STATUS_SUCCESS) return status;
    const int baseA = base_of(descrA), baseB = base_of(descrB), baseC = base_of(descrC);
    std::vector<int> cols;
    int64_t total = 0;
    csrRowPtrC[0] = baseC;
    for (int i = 0; i < m; ++i) {
        geam2_row_columns(csrRowPtrA, csrColIndA, baseA, csrRowPtrB, csrColIndB, baseB, i, &cols);
        total += static_cast<int64_t>(cols.size());
        csrRowPtrC[i + 1] = static_cast<int>(total) + baseC;
    }
    *nnzTotalDevHostPtr = static_cast<int>(total);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── incomplete factorization info objects ───────────────────────────────────

cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t* info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) csric02Info();
    return *info == nullptr ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateBsrilu02Info(bsrilu02Info_t* info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) bsrilu02Info();
    return *info == nullptr ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsrilu02Info(bsrilu02Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCreateBsric02Info(bsric02Info_t* info) {
    if (info == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) bsric02Info();
    return *info == nullptr ? CUSPARSE_STATUS_ALLOC_FAILED : CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyBsric02Info(bsric02Info_t info) {
    delete info;
    return CUSPARSE_STATUS_SUCCESS;
}

// *position = -1 and success when no zero pivot was seen; otherwise the first
// one, in the matrix's own index base, with CUSPARSE_STATUS_ZERO_PIVOT.
cusparseStatus_t cusparseXcsrilu02_zeroPivot(cusparseHandle_t handle, csrilu02Info_t info,
                                             int* position) {
    SP_NEED_HANDLE(handle);
    if (info == nullptr || position == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (info->zero_pivot >= 0) {
        *position = info->zero_pivot + info->base;
        return CUSPARSE_STATUS_ZERO_PIVOT;
    }
    *position = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXcsric02_zeroPivot(cusparseHandle_t handle, csric02Info_t info,
                                            int* position) {
    SP_NEED_HANDLE(handle);
    if (info == nullptr || position == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;
    if (info->zero_pivot >= 0) {
        *position = info->zero_pivot + info->base;
        return CUSPARSE_STATUS_ZERO_PIVOT;
    }
    *position = -1;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseXbsrilu02_zeroPivot(cusparseHandle_t handle, bsrilu02Info_t,
                                             int*) {
    SP_NEED_HANDLE(handle);
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

cusparseStatus_t cusparseXbsric02_zeroPivot(cusparseHandle_t handle, bsric02Info_t, int*) {
    SP_NEED_HANDLE(handle);
    return CUSPARSE_STATUS_NOT_SUPPORTED;
}

// ── per-precision wrappers ──────────────────────────────────────────────────

#define CUMETAL_SP_DEFINE_TYPED(P, C)                                                            \
    cusparseStatus_t cusparse##P##nnz(cusparseHandle_t handle, cusparseDirection_t dirA, int m,   \
                                      int n, const cusparseMatDescr_t descrA, const C* A,        \
                                      int lda, int* nnzPerRowColumn, int* nnzTotalDevHostPtr) {  \
        return nnz_dense_impl<C>(handle, dirA, m, n, descrA, A, lda, nnzPerRowColumn,            \
                                 nnzTotalDevHostPtr);                                            \
    }                                                                                            \
    cusparseStatus_t cusparse##P##nnz_compress(                                                  \
        cusparseHandle_t handle, int m, const cusparseMatDescr_t descr, const C* csrSortedValA,  \
        const int* csrSortedRowPtrA, int* nnzPerRow, int* nnzC, C tol) {                         \
        return nnz_compress_impl<C>(handle, m, descr, csrSortedValA, csrSortedRowPtrA,           \
                                    nnzPerRow, nnzC, tol);                                       \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csr2csr_compress(                                              \
        cusparseHandle_t handle, int m, int n, const cusparseMatDescr_t descrA,                  \
        const C* csrSortedValA, const int* csrSortedColIndA, const int* csrSortedRowPtrA,        \
        int nnzA, int* nnzPerRow, C* csrSortedValC, int* csrSortedColIndC,                       \
        int* csrSortedRowPtrC, C tol) {                                                          \
        return csr2csr_compress_impl<C>(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,   \
                                        csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,        \
                                        csrSortedColIndC, csrSortedRowPtrC, tol);                \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrgeam2_bufferSizeExt(                                        \
        cusparseHandle_t handle, int m, int n, const C* alpha, const cusparseMatDescr_t descrA,  \
        int nnzA, const C* csrSortedValA, const int* csrSortedRowPtrA,                           \
        const int* csrSortedColIndA, const C* beta, const cusparseMatDescr_t descrB, int nnzB,   \
        const C* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,        \
        const cusparseMatDescr_t descrC, C* csrSortedValC, int* csrSortedRowPtrC,                \
        int* csrSortedColIndC, size_t* pBufferSizeInBytes) {                                     \
        return geam2_buffer_size<C>(handle, m, n, alpha, descrA, nnzA, csrSortedValA,            \
                                    csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,      \
                                    csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,   \
                                    csrSortedValC, csrSortedRowPtrC, csrSortedColIndC,           \
                                    pBufferSizeInBytes);                                         \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrgeam2(                                                      \
        cusparseHandle_t handle, int m, int n, const C* alpha, const cusparseMatDescr_t descrA,  \
        int nnzA, const C* csrSortedValA, const int* csrSortedRowPtrA,                           \
        const int* csrSortedColIndA, const C* beta, const cusparseMatDescr_t descrB, int nnzB,   \
        const C* csrSortedValB, const int* csrSortedRowPtrB, const int* csrSortedColIndB,        \
        const cusparseMatDescr_t descrC, C* csrSortedValC, int* csrSortedRowPtrC,                \
        int* csrSortedColIndC, void*) {                                                          \
        return geam2_compute<C>(handle, m, n, alpha, descrA, nnzA, csrSortedValA,                \
                                csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,          \
                                csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,       \
                                csrSortedValC, csrSortedRowPtrC, csrSortedColIndC);              \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrilu02_numericBoost(cusparseHandle_t handle,                 \
                                                        csrilu02Info_t info, int enable_boost,   \
                                                        double* tol, C* boost_val) {             \
        return csrilu02_boost_impl<C>(handle, info, enable_boost, tol, boost_val);               \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrilu02_bufferSize(                                           \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        C* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,              \
        csrilu02Info_t info, int* pBufferSizeInBytes) {                                          \
        const cusparseStatus_t status = incomplete_validate<C>(                                  \
            handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info);    \
        if (status != CUSPARSE_STATUS_SUCCESS) return status;                                    \
        if (pBufferSizeInBytes == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                 \
        *pBufferSizeInBytes = 1;                                                                 \
        return CUSPARSE_STATUS_SUCCESS;                                                          \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrilu02_analysis(                                             \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        const C* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,        \
        csrilu02Info_t info, cusparseSolvePolicy_t, void*) {                                     \
        return csrilu02_analysis_impl<C>(handle, m, nnz, descrA, csrSortedValA,                  \
                                         csrSortedRowPtrA, csrSortedColIndA, info);              \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csrilu02(                                                      \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        C* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,         \
        csrilu02Info_t info, cusparseSolvePolicy_t, void*) {                                     \
        return csrilu02_impl<C>(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,    \
                                csrSortedColIndA, info);                                         \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csric02_bufferSize(                                            \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        C* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,              \
        csric02Info_t info, int* pBufferSizeInBytes) {                                           \
        const cusparseStatus_t status = incomplete_validate<C>(                                  \
            handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, info);    \
        if (status != CUSPARSE_STATUS_SUCCESS) return status;                                    \
        if (pBufferSizeInBytes == nullptr) return CUSPARSE_STATUS_INVALID_VALUE;                 \
        *pBufferSizeInBytes = 1;                                                                 \
        return CUSPARSE_STATUS_SUCCESS;                                                          \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csric02_analysis(                                              \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        const C* csrSortedValA, const int* csrSortedRowPtrA, const int* csrSortedColIndA,        \
        csric02Info_t info, cusparseSolvePolicy_t, void*) {                                      \
        return csric02_analysis_impl<C>(handle, m, nnz, descrA, csrSortedValA,                   \
                                        csrSortedRowPtrA, csrSortedColIndA, info);               \
    }                                                                                            \
    cusparseStatus_t cusparse##P##csric02(                                                       \
        cusparseHandle_t handle, int m, int nnz, const cusparseMatDescr_t descrA,                \
        C* csrSortedValA_valM, const int* csrSortedRowPtrA, const int* csrSortedColIndA,         \
        csric02Info_t info, cusparseSolvePolicy_t, void*) {                                      \
        return csric02_impl<C>(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,     \
                               csrSortedColIndA, info);                                          \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsrilu02_numericBoost(cusparseHandle_t handle,                 \
                                                        bsrilu02Info_t, int, double*, C*) {      \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsrilu02_bufferSize(                                           \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t, C*,    \
        const int*, const int*, int, bsrilu02Info_t, int*) {                                     \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsrilu02_analysis(                                             \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t, C*,    \
        const int*, const int*, int, bsrilu02Info_t, cusparseSolvePolicy_t, void*) {             \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsrilu02(                                                      \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t, C*,    \
        const int*, const int*, int, bsrilu02Info_t, cusparseSolvePolicy_t, void*) {             \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsric02_bufferSize(                                            \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t, C*,    \
        const int*, const int*, int, bsric02Info_t, int*) {                                      \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsric02_analysis(                                              \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t,        \
        const C*, const int*, const int*, int, bsric02Info_t, cusparseSolvePolicy_t, void*) {    \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##bsric02(                                                       \
        cusparseHandle_t handle, cusparseDirection_t, int, int, const cusparseMatDescr_t, C*,    \
        const int*, const int*, int, bsric02Info_t, cusparseSolvePolicy_t, void*) {              \
        SP_NEED_HANDLE(handle);                                                                  \
        return CUSPARSE_STATUS_NOT_SUPPORTED;                                                    \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2_bufferSizeExt(                                           \
        cusparseHandle_t handle, int m, int n, const C* dl, const C* d, const C* du,             \
        const C* B, int ldb, size_t* bufferSizeInBytes) {                                        \
        return gtsv2_buffer_size<C>(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);         \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2(cusparseHandle_t handle, int m, int n, const C* dl,      \
                                        const C* d, const C* du, C* B, int ldb, void*) {         \
        return gtsv2_impl<C>(handle, m, n, dl, d, du, B, ldb, true);                             \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2_nopivot_bufferSizeExt(                                   \
        cusparseHandle_t handle, int m, int n, const C* dl, const C* d, const C* du,             \
        const C* B, int ldb, size_t* bufferSizeInBytes) {                                        \
        return gtsv2_buffer_size<C>(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes);         \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2_nopivot(cusparseHandle_t handle, int m, int n,           \
                                                const C* dl, const C* d, const C* du, C* B,      \
                                                int ldb, void*) {                                \
        return gtsv2_impl<C>(handle, m, n, dl, d, du, B, ldb, false);                            \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2StridedBatch_bufferSizeExt(                               \
        cusparseHandle_t handle, int m, const C* dl, const C* d, const C* du, const C* x,        \
        int batchCount, int batchStride, size_t* bufferSizeInBytes) {                            \
        return gtsv2_strided_buffer_size<C>(handle, m, dl, d, du, x, batchCount, batchStride,    \
                                            bufferSizeInBytes);                                  \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsv2StridedBatch(                                             \
        cusparseHandle_t handle, int m, const C* dl, const C* d, const C* du, C* x,              \
        int batchCount, int batchStride, void*) {                                                \
        return gtsv2_strided_impl<C>(handle, m, dl, d, du, x, batchCount, batchStride);          \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsvInterleavedBatch_bufferSizeExt(                            \
        cusparseHandle_t handle, int algo, int m, const C* dl, const C* d, const C* du,          \
        const C* x, int batchCount, size_t* pBufferSizeInBytes) {                                \
        return gtsv_interleaved_buffer_size<C>(handle, algo, m, dl, d, du, x, batchCount,        \
                                               pBufferSizeInBytes);                              \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gtsvInterleavedBatch(                                          \
        cusparseHandle_t handle, int algo, int m, C* dl, C* d, C* du, C* x, int batchCount,      \
        void*) {                                                                                 \
        return gtsv_interleaved_impl<C>(handle, algo, m, dl, d, du, x, batchCount);              \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gpsvInterleavedBatch_bufferSizeExt(                            \
        cusparseHandle_t handle, int algo, int m, const C* ds, const C* dl, const C* d,          \
        const C* du, const C* dw, const C* x, int batchCount, size_t* pBufferSizeInBytes) {      \
        return gpsv_interleaved_buffer_size<C>(handle, algo, m, ds, dl, d, du, dw, x,            \
                                               batchCount, pBufferSizeInBytes);                  \
    }                                                                                            \
    cusparseStatus_t cusparse##P##gpsvInterleavedBatch(                                          \
        cusparseHandle_t handle, int algo, int m, C* ds, C* dl, C* d, C* du, C* dw, C* x,        \
        int batchCount, void*) {                                                                 \
        return gpsv_interleaved_impl<C>(handle, algo, m, ds, dl, d, du, dw, x, batchCount);      \
    }

CUMETAL_CUSPARSE_FOR_EACH_TYPE(CUMETAL_SP_DEFINE_TYPED)
#undef CUMETAL_SP_DEFINE_TYPED
#undef SP_NEED_HANDLE

}  // extern "C"
