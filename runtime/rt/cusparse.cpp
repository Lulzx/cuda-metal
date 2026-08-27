#include "cusparse.h"
#include "cuda_runtime.h"

#include "metal_backend.h"
#include "runtime_internal.h"
#include "sparse_kernels_msl.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
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

// The MSL is compiled by the Metal backend's runtime source path, which reads
// from a file, so stage it once per process into the same cache the JIT uses.
const std::string* sparse_kernels_source_path() {
    static const std::string* cached = [] {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::path dir;
        if (const char* d = std::getenv("CUMETAL_CACHE_DIR"); d != nullptr && d[0] != '\0') {
            dir = fs::path(d);
        } else if (const char* home = std::getenv("HOME"); home != nullptr && home[0] != '\0') {
            dir = fs::path(home) / "Library" / "Caches" / "io.cumetal";
        } else {
            dir = fs::temp_directory_path(ec);
            if (ec) return static_cast<std::string*>(nullptr);
        }
        dir /= "library-kernels";
        fs::create_directories(dir, ec);
        if (ec) return static_cast<std::string*>(nullptr);
        const fs::path out = dir / "sparse_kernels.metal";
        // Rewrite unconditionally: the source is compiled into this binary, so a
        // stale file from an older build must not win.
        std::FILE* f = std::fopen(out.c_str(), "wb");
        if (f == nullptr) return static_cast<std::string*>(nullptr);
        const auto& src = cumetal::rt::kSparseKernelsMsl;
        const bool wrote = std::fwrite(src.data(), 1, src.size(), f) == src.size();
        std::fclose(f);
        if (!wrote) return static_cast<std::string*>(nullptr);
        return new std::string(out.string());
    }();
    return cached;
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

// A device pointer resolves to a Metal buffer plus a byte offset. Metal requires
// the offset to satisfy the bound type's alignment, so reject anything that does
// not and let the CPU path take it.
bool resolve_arg(const void* ptr,
                 std::size_t required_bytes,
                 std::size_t alignment,
                 cumetal::metal_backend::KernelArg* out) {
    if (ptr == nullptr) return false;
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(ptr, &resolved)) return false;
    if (resolved.buffer == nullptr || resolved.remaining_size < required_bytes) return false;
    if (resolved.offset % alignment != 0) return false;
    out->kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
    out->buffer = resolved.buffer;
    out->offset = resolved.offset;
    return true;
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
                                          cusparseSpMVAlg_t /*alg*/,
                                          void* /*externalBuffer*/) {
    if (!handle || !matA || !vecX || !vecY || !alpha || !beta) {
        return CUSPARSE_STATUS_INVALID_VALUE;
    }
    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F) {
        return CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
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
        const std::size_t elem = computeType == CUDA_R_64F ? 8u : 4u;
        std::memcpy(a.bytes, alpha, elem);
        std::memcpy(b.bytes, beta, elem);
        void* x = vecX->values;
        void* y = vecY->values;
        if (cumetal::rt::capture_library_call(
                handle->stream, [=](cudaStream_t replay_stream) {
                    return spmv_dispatch(replay_stream, opA, a.bytes, matA, x, b.bytes, y,
                                         computeType) == CUSPARSE_STATUS_SUCCESS
                               ? cudaSuccess
                               : cudaErrorInvalidValue;
                })) {
            return CUSPARSE_STATUS_SUCCESS;
        }
    }

    return spmv_dispatch(handle->stream, opA, alpha, matA, vecX->values, beta, vecY->values,
                         computeType);
}

// Runs one SpMV over already-resolved operands. Split out from cusparseSpMV so
// that a graph node can replay it on whatever stream the graph was launched on,
// with the arguments it was captured with.
static cusparseStatus_t spmv_dispatch(cudaStream_t stream,
                                      cusparseOperation_t opA,
                                      const void* alpha,
                                      cusparseSpMatDescr_t matA,
                                      const void* x_values,
                                      const void* beta,
                                      void* y_values,
                                      cudaDataType computeType) {
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
    cudaStreamSynchronize(stream);

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
