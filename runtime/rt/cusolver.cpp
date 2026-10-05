#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include "cuda_runtime.h"
#include "cumetal_native.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

// ── cuSOLVER shim ───────────────────────────────────────────────────────────
// Dense linear algebra via Apple Accelerate LAPACK.
// On Apple Silicon UMA, device pointers are host-accessible, so LAPACK
// operates directly on the caller's buffers with zero copy.

extern "C" {

struct cusolverDnContext {
    cudaStream_t stream = nullptr;
};

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    *handle = new (std::nothrow) cusolverDnContext();
    if (*handle == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t* streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!streamId) return CUSOLVER_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUSOLVER_STATUS_SUCCESS;
}

static cusolverStatus_t sync_stream(cusolverDnHandle_t handle) {
    if (handle == nullptr) return CUSOLVER_STATUS_NOT_INITIALIZED;
    // A null stream is CUDA's default stream, not an absence of ordering.
    return cudaStreamSynchronize(handle->stream) == cudaSuccess
               ? CUSOLVER_STATUS_SUCCESS
               : CUSOLVER_STATUS_EXECUTION_FAILED;
}

static bool valid_fill(cublasFillMode_t fill) {
    return fill == CUBLAS_FILL_MODE_LOWER || fill == CUBLAS_FILL_MODE_UPPER;
}

static bool valid_eig_mode(cusolverEigMode_t mode) {
    return mode == CUSOLVER_EIG_MODE_NOVECTOR ||
           mode == CUSOLVER_EIG_MODE_VECTOR;
}

static bool valid_svd_job(char job) {
    return job == 'A' || job == 'S' || job == 'O' || job == 'N';
}

static bool checked_workspace_product(int m, int n, int* result) {
    if (result == nullptr || m < 0 || n < 0) return false;
    const long long product = static_cast<long long>(m) * n;
    if (product > std::numeric_limits<int>::max()) return false;
    *result = static_cast<int>(product);
    return true;
}

// ── LU factorization ────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!A || lda < std::max(1, m) || !checked_workspace_product(m, n, Lwork))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!A || lda < std::max(1, m) || !checked_workspace_product(m, n, Lwork))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

// ── LU solve ─────────────────────────────────────────────────────────────────

// ── QR factorization ────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!A || lda < std::max(1, m) || !checked_workspace_product(m, n, Lwork))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!A || lda < std::max(1, m) || !checked_workspace_product(m, n, Lwork))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* TAU,
                                   float* Workspace, int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !A || !TAU || !Workspace || !devInfo ||
        lda < std::max(1, m) || Lwork < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, LW = Lwork, info = 0;
    sgeqrf_(&M, &N, A, &LDA, TAU, Workspace, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* TAU,
                                   double* Workspace, int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !A || !TAU || !Workspace || !devInfo ||
        lda < std::max(1, m) || Lwork < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, LW = Lwork, info = 0;
    dgeqrf_(&M, &N, A, &LDA, TAU, Workspace, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

// ── Cholesky factorization ──────────────────────────────────────────────────

cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              float* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || lda < std::max(1, n) || !Lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle,
                                              cublasFillMode_t uplo, int n,
                                              double* A, int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || lda < std::max(1, n) || !Lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, float* A, int lda, float* Workspace,
                                   int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || !Workspace || !devInfo ||
        lda < std::max(1, n) || Lwork < n)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, info = 0;
    spotrf_(&ul, &N, A, &LDA, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, double* A, int lda, double* Workspace,
                                   int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || !Workspace || !devInfo ||
        lda < std::max(1, n) || Lwork < n)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, info = 0;
    dpotrf_(&ul, &N, A, &LDA, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

// ── Cholesky solve ──────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const float* A, int lda,
                                   float* B, int ldb, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || nrhs < 0 || !A || !B || !devInfo ||
        lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    spotrs_(&ul, &N, &NRHS, const_cast<float*>(A), &LDA, B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrs(cusolverDnHandle_t handle, cublasFillMode_t uplo,
                                   int n, int nrhs, const double* A, int lda,
                                   double* B, int ldb, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || nrhs < 0 || !A || !B || !devInfo ||
        lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    dpotrs_(&ul, &N, &NRHS, const_cast<double*>(A), &LDA, B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

// ── Eigenvalue decomposition (syevd) ────────────────────────────────────────

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              const float* A, int lda,
                                              const float* W, int* lwork) {
    // Query optimal workspace
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || !A || !W ||
        lda < std::max(1, n) || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = 1LL + 6LL * n + 2LL * n * n;
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = std::max(1, static_cast<int>(required));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle,
                                              cusolverEigMode_t jobz,
                                              cublasFillMode_t uplo, int n,
                                              const double* A, int lda,
                                              const double* W, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || !A || !W ||
        lda < std::max(1, n) || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = 1LL + 6LL * n + 2LL * n * n;
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = std::max(1, static_cast<int>(required));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, float* A, int lda,
                                   float* W, float* work, int lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const long long required = 1LL + 6LL * n + 2LL * n * n;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || !A || !W ||
        !work || !devInfo || lda < std::max(1, n) || required < 0 ||
        required > std::numeric_limits<int>::max() ||
        lwork < std::max(1, static_cast<int>(required)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char job = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? 'V' : 'N';
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    // LAPACK's ssyevd also needs integer workspace
    __CLPK_integer liwork = std::max(__CLPK_integer(1), 3 + 5 * N);
    std::unique_ptr<__CLPK_integer[]> iwork(
        new (std::nothrow) __CLPK_integer[static_cast<size_t>(liwork)]);
    if (!iwork) return CUSOLVER_STATUS_ALLOC_FAILED;
    ssyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.get(), &liwork, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                                   cublasFillMode_t uplo, int n, double* A, int lda,
                                   double* W, double* work, int lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const long long required = 1LL + 6LL * n + 2LL * n * n;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || !A || !W ||
        !work || !devInfo || lda < std::max(1, n) || required < 0 ||
        required > std::numeric_limits<int>::max() ||
        lwork < std::max(1, static_cast<int>(required)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char job = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? 'V' : 'N';
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    __CLPK_integer liwork = std::max(__CLPK_integer(1), 3 + 5 * N);
    std::unique_ptr<__CLPK_integer[]> iwork(
        new (std::nothrow) __CLPK_integer[static_cast<size_t>(liwork)]);
    if (!iwork) return CUSOLVER_STATUS_ALLOC_FAILED;
    dsyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.get(), &liwork, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

// ── SVD ─────────────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = 3LL * std::min(m, n) + 2LL * std::max(m, n);
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = std::max(1, static_cast<int>(required));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = 3LL * std::min(m, n) + 2LL * std::max(m, n);
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = std::max(1, static_cast<int>(required));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, float* A, int lda,
                                   float* S, float* U, int ldu, float* VT, int ldvt,
                                   float* work, int lwork, float* /*rwork*/, int* devInfo) {
    char ju = static_cast<char>(jobu);
    char jvt = static_cast<char>(jobvt);
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const int min_dim = std::min(m, n);
    const long long required = 3LL * min_dim + 2LL * std::max(m, n);
    const bool wants_u = ju == 'A' || ju == 'S';
    const bool wants_vt = jvt == 'A' || jvt == 'S';
    const int required_ldvt = jvt == 'A' ? std::max(1, n)
                                         : std::max(1, min_dim);
    if (!valid_svd_job(ju) || !valid_svd_job(jvt) ||
        (ju == 'O' && jvt == 'O') || m < 0 || n < 0 || !A || !S || !work ||
        !devInfo || lda < std::max(1, m) || ldu < 1 || ldvt < 1 ||
        (wants_u && (!U || ldu < std::max(1, m))) ||
        (wants_vt && (!VT || ldvt < required_ldvt)) || required < 0 ||
        required > std::numeric_limits<int>::max() ||
        lwork < std::max(1, static_cast<int>(required)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, LDU = ldu, LDVT = ldvt, LW = lwork, info = 0;
    sgesvd_(&ju, &jvt, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, work, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, signed char jobu,
                                   signed char jobvt, int m, int n, double* A, int lda,
                                   double* S, double* U, int ldu, double* VT, int ldvt,
                                   double* work, int lwork, double* /*rwork*/, int* devInfo) {
    char ju = static_cast<char>(jobu);
    char jvt = static_cast<char>(jobvt);
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const int min_dim = std::min(m, n);
    const long long required = 3LL * min_dim + 2LL * std::max(m, n);
    const bool wants_u = ju == 'A' || ju == 'S';
    const bool wants_vt = jvt == 'A' || jvt == 'S';
    const int required_ldvt = jvt == 'A' ? std::max(1, n)
                                         : std::max(1, min_dim);
    if (!valid_svd_job(ju) || !valid_svd_job(jvt) ||
        (ju == 'O' && jvt == 'O') || m < 0 || n < 0 || !A || !S || !work ||
        !devInfo || lda < std::max(1, m) || ldu < 1 || ldvt < 1 ||
        (wants_u && (!U || ldu < std::max(1, m))) ||
        (wants_vt && (!VT || ldvt < required_ldvt)) || required < 0 ||
        required > std::numeric_limits<int>::max() ||
        lwork < std::max(1, static_cast<int>(required)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, LDU = ldu, LDVT = ldvt, LW = lwork, info = 0;
    dgesvd_(&ju, &jvt, &M, &N, A, &LDA, S, U, &LDU, VT, &LDVT, work, &LW, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

} // extern "C" — temporarily close for C++ templates

// ── cusolverSp: Sparse solver (host path) ─────────────────────────────────────
// Uses dense conversion + LAPACK as a simple-but-correct fallback.
// On UMA this is zero-copy from the caller's perspective.

// Helper: convert CSR to dense column-major matrix (must be outside extern "C")
template <typename T>
static bool validate_csr_sp(int m, int nnz, const int* csrRowPtr,
                            const int* csrColInd, int base) {
    if (m < 0 || nnz < 0 || !csrRowPtr || !csrColInd ||
        csrRowPtr[0] != base || csrRowPtr[m] - base != nnz)
        return false;
    for (int row = 0; row < m; ++row) {
        const int begin = csrRowPtr[row] - base;
        const int end = csrRowPtr[row + 1] - base;
        if (begin < 0 || begin > end || end > nnz) return false;
    }
    for (int entry = 0; entry < nnz; ++entry) {
        const int column = csrColInd[entry] - base;
        if (column < 0 || column >= m) return false;
    }
    return true;
}

template <typename T>
static bool checked_dense_square_size(int m, size_t* elements) {
    if (m < 0 || elements == nullptr) return false;
    const size_t width = static_cast<size_t>(m);
    if (width != 0 && width > std::numeric_limits<size_t>::max() / width)
        return false;
    const size_t count = width * width;
    if (count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    *elements = count;
    return true;
}

template <typename T>
static void csr_to_dense_sp(int m, const T* csrVal, const int* csrRowPtr,
                            const int* csrColInd, int base, T* dense) {
    std::memset(dense, 0, (size_t)m * m * sizeof(T));
    for (int i = 0; i < m; ++i) {
        for (int j = csrRowPtr[i] - base; j < csrRowPtr[i + 1] - base; ++j) {
            const int c = csrColInd[j] - base;
            dense[(size_t)c * m + i] = csrVal[j];  // column-major
        }
    }
}

extern "C" {

struct cusolverSpContext {
    cudaStream_t stream = nullptr;
};

cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t* handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    *handle = new (std::nothrow) cusolverSpContext();
    if (*handle == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle) {
    delete handle;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, cudaStream_t streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    handle->stream = streamId;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverSpGetStream(cusolverSpHandle_t handle, cudaStream_t* streamId) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!streamId) return CUSOLVER_STATUS_INVALID_VALUE;
    *streamId = handle->stream;
    return CUSOLVER_STATUS_SUCCESS;
}

static cusolverStatus_t sync_sp_stream(cusolverSpHandle_t handle) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    return cudaStreamSynchronize(handle->stream) == cudaSuccess
               ? CUSOLVER_STATUS_SUCCESS
               : CUSOLVER_STATUS_EXECUTION_FAILED;
}

// Sparse solve via dense Cholesky (LAPACK spotrf/dpotrf + spotrs/dpotrs)
cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const float* csrVal, const int* csrRowPtr,
                                        const int* csrColInd, const float* b,
                                        float tol, int reorder,
                                        float* x, int* singularity) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnz < 0 || !descrA || !csrVal || !csrRowPtr || !csrColInd ||
        !b || !x || !singularity || !std::isfinite(tol) || tol < 0.0f ||
        reorder != 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_sp_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    const int base = static_cast<int>(cusparseGetMatIndexBase(descrA));
    if (!validate_csr_sp<float>(m, nnz, csrRowPtr, csrColInd, base))
        return CUSOLVER_STATUS_INVALID_VALUE;
    size_t dense_elements = 0;
    if (!checked_dense_square_size<float>(m, &dense_elements))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0) {
        *singularity = -1;
        return CUSOLVER_STATUS_SUCCESS;
    }

    try {
        std::vector<float> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());

        char uplo = 'L';
        __CLPK_integer N = m, nrhs = 1, lda = m, ldb = m, info = 0;
        spotrf_(&uplo, &N, A.data(), &lda, &info);
        if (info != 0) {
            *singularity = static_cast<int>(info - 1);
            return CUSOLVER_STATUS_SUCCESS;
        }
        for (int i = 0; i < m; ++i) {
            if (std::fabs(A[static_cast<size_t>(i) * m + i]) <= tol) {
                *singularity = i;
                return CUSOLVER_STATUS_SUCCESS;
            }
        }
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(float));
        spotrs_(&uplo, &N, &nrhs, A.data(), &lda, x, &ldb, &info);
        *singularity = -1;
        return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
    } catch (const std::bad_alloc&) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
}

cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t handle,
                                        int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const double* csrVal, const int* csrRowPtr,
                                        const int* csrColInd, const double* b,
                                        double tol, int reorder,
                                        double* x, int* singularity) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnz < 0 || !descrA || !csrVal || !csrRowPtr || !csrColInd ||
        !b || !x || !singularity || !std::isfinite(tol) || tol < 0.0 ||
        reorder != 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_sp_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    const int base = static_cast<int>(cusparseGetMatIndexBase(descrA));
    if (!validate_csr_sp<double>(m, nnz, csrRowPtr, csrColInd, base))
        return CUSOLVER_STATUS_INVALID_VALUE;
    size_t dense_elements = 0;
    if (!checked_dense_square_size<double>(m, &dense_elements))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0) {
        *singularity = -1;
        return CUSOLVER_STATUS_SUCCESS;
    }

    try {
        std::vector<double> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());

        char uplo = 'L';
        __CLPK_integer N = m, nrhs = 1, lda = m, ldb = m, info = 0;
        dpotrf_(&uplo, &N, A.data(), &lda, &info);
        if (info != 0) {
            *singularity = static_cast<int>(info - 1);
            return CUSOLVER_STATUS_SUCCESS;
        }
        for (int i = 0; i < m; ++i) {
            if (std::fabs(A[static_cast<size_t>(i) * m + i]) <= tol) {
                *singularity = i;
                return CUSOLVER_STATUS_SUCCESS;
            }
        }
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(double));
        dpotrs_(&uplo, &N, &nrhs, A.data(), &lda, x, &ldb, &info);
        *singularity = -1;
        return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
    } catch (const std::bad_alloc&) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
}

// Sparse QR solve via dense QR (LAPACK sgels/dgels)
cusolverStatus_t cusolverSpScsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const float* csrVal, const int* csrRowPtr,
                                      const int* csrColInd, const float* b,
                                      float tol, int reorder,
                                      float* x, int* singularity) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnz < 0 || !descrA || !csrVal || !csrRowPtr || !csrColInd ||
        !b || !x || !singularity || !std::isfinite(tol) || tol < 0.0f ||
        reorder != 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_sp_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    const int base = static_cast<int>(cusparseGetMatIndexBase(descrA));
    if (!validate_csr_sp<float>(m, nnz, csrRowPtr, csrColInd, base))
        return CUSOLVER_STATUS_INVALID_VALUE;
    size_t dense_elements = 0;
    if (!checked_dense_square_size<float>(m, &dense_elements))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0) {
        *singularity = -1;
        return CUSOLVER_STATUS_SUCCESS;
    }

    try {
        std::vector<float> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(float));

        char trans = 'N';
        __CLPK_integer M = m, N = m, nrhs = 1, lda = m, ldb = m, lwork = -1, info = 0;
        float work_query = 0;
        sgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, &work_query, &lwork, &info);
        if (info != 0 || !std::isfinite(work_query) || work_query < 1.0f ||
            work_query > static_cast<float>(std::numeric_limits<__CLPK_integer>::max()))
            return CUSOLVER_STATUS_INTERNAL_ERROR;
        lwork = static_cast<__CLPK_integer>(work_query);
        std::vector<float> work(static_cast<size_t>(lwork));
        sgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, work.data(), &lwork, &info);
        if (info != 0) return CUSOLVER_STATUS_INTERNAL_ERROR;
        *singularity = -1;
        for (int i = 0; i < m; ++i) {
            if (std::fabs(A[static_cast<size_t>(i) * m + i]) <= tol) {
                *singularity = i;
                break;
            }
        }
        return CUSOLVER_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
}

cusolverStatus_t cusolverSpDcsrlsvqr(cusolverSpHandle_t handle,
                                      int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const double* csrVal, const int* csrRowPtr,
                                      const int* csrColInd, const double* b,
                                      double tol, int reorder,
                                      double* x, int* singularity) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnz < 0 || !descrA || !csrVal || !csrRowPtr || !csrColInd ||
        !b || !x || !singularity || !std::isfinite(tol) || tol < 0.0 ||
        reorder != 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_sp_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    const int base = static_cast<int>(cusparseGetMatIndexBase(descrA));
    if (!validate_csr_sp<double>(m, nnz, csrRowPtr, csrColInd, base))
        return CUSOLVER_STATUS_INVALID_VALUE;
    size_t dense_elements = 0;
    if (!checked_dense_square_size<double>(m, &dense_elements))
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0) {
        *singularity = -1;
        return CUSOLVER_STATUS_SUCCESS;
    }

    try {
        std::vector<double> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(double));

        char trans = 'N';
        __CLPK_integer M = m, N = m, nrhs = 1, lda = m, ldb = m, lwork = -1, info = 0;
        double work_query = 0;
        dgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, &work_query, &lwork, &info);
        if (info != 0 || !std::isfinite(work_query) || work_query < 1.0 ||
            work_query > static_cast<double>(std::numeric_limits<__CLPK_integer>::max()))
            return CUSOLVER_STATUS_INTERNAL_ERROR;
        lwork = static_cast<__CLPK_integer>(work_query);
        std::vector<double> work(static_cast<size_t>(lwork));
        dgels_(&trans, &M, &N, &nrhs, A.data(), &lda, x, &ldb, work.data(), &lwork, &info);
        if (info != 0) return CUSOLVER_STATUS_INTERNAL_ERROR;
        *singularity = -1;
        for (int i = 0; i < m; ++i) {
            if (std::fabs(A[static_cast<size_t>(i) * m + i]) <= tol) {
                *singularity = i;
                break;
            }
        }
        return CUSOLVER_STATUS_SUCCESS;
    } catch (const std::bad_alloc&) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
}

// ── Version query ───────────────────────────────────────────────────────────

cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* value) {
    if (!value) return CUSOLVER_STATUS_INVALID_VALUE;
    // CuMetal's own version, not an NVIDIA library release. Reporting 12.x
    // would let downstream code select paths on a false capability claim.
    switch (type) {
        case MAJOR_VERSION: *value = CUMETAL_VERSION_MAJOR; return CUSOLVER_STATUS_SUCCESS;
        case MINOR_VERSION: *value = CUMETAL_VERSION_MINOR; return CUSOLVER_STATUS_SUCCESS;
        case PATCH_LEVEL:   *value = CUMETAL_VERSION_PATCH; return CUSOLVER_STATUS_SUCCESS;
    }
    return CUSOLVER_STATUS_INVALID_VALUE;
}

// ── Params handle for the generic interfaces ────────────────────────────────

struct cusolverDnParams {
    // Reserved for future generic-API configuration (algorithm selection,
    // deterministic modes). The bounded subset has no knobs yet, but the
    // lifecycle must be real because callers create and destroy it.
};

cusolverStatus_t cusolverDnCreateParams(cusolverDnParams_t* params) {
    if (!params) return CUSOLVER_STATUS_INVALID_VALUE;
    *params = new (std::nothrow) cusolverDnParams();
    if (*params == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyParams(cusolverDnParams_t params) {
    delete params;
    return CUSOLVER_STATUS_SUCCESS;
}

}  // extern "C" — templates must have C++ linkage

// ── Shared syevd driver for the int and 64-bit interfaces ───────────────────

static bool valid_xsyev_types(cudaDataType dataTypeA, cudaDataType dataTypeW,
                              cudaDataType computeType) {
    // Homogeneous FP32/FP64 subset: the three types must agree so A, W and
    // compute share one element type.
    if (dataTypeA != dataTypeW || dataTypeA != computeType) return false;
    return dataTypeA == CUDA_R_32F || dataTypeA == CUDA_R_64F;
}

static bool fits_int(int64_t v) {
    return v >= std::numeric_limits<int>::min() &&
           v <= std::numeric_limits<int>::max();
}

static long long syevd_work_elems(int n) {
    return 1LL + 6LL * n + 2LL * n * n;
}

template <typename T>
static cusolverStatus_t run_syevd(cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                  int n, T* A, int lda, T* W, T* work, int lwork,
                                  int* devInfo) {
    char job = (jobz == CUSOLVER_EIG_MODE_VECTOR) ? 'V' : 'N';
    char ul = (uplo == CUBLAS_FILL_MODE_UPPER) ? 'U' : 'L';
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    __CLPK_integer liwork = std::max(__CLPK_integer(1), 3 + 5 * N);
    std::unique_ptr<__CLPK_integer[]> iwork(
        new (std::nothrow) __CLPK_integer[static_cast<size_t>(liwork)]);
    if (!iwork) return CUSOLVER_STATUS_ALLOC_FAILED;
    if constexpr (std::is_same<T, float>::value) {
        ssyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.get(), &liwork, &info);
    } else {
        dsyevd_(&job, &ul, &N, A, &LDA, W, work, &LW, iwork.get(), &liwork, &info);
    }
    if (devInfo) *devInfo = static_cast<int>(info);
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

extern "C" {

// ── Generic 64-bit syevd ────────────────────────────────────────────────────

cusolverStatus_t cusolverDnXsyevd_bufferSize(cusolverDnHandle_t handle,
                                             cusolverDnParams_t /*params*/,
                                             cusolverEigMode_t jobz,
                                             cublasFillMode_t uplo, int64_t n,
                                             cudaDataType dataTypeA, const void* A,
                                             int64_t lda, cudaDataType dataTypeW,
                                             const void* W, cudaDataType computeType,
                                             size_t* workspaceInBytesOnDevice,
                                             size_t* workspaceInBytesOnHost) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) ||
        !valid_xsyev_types(dataTypeA, dataTypeW, computeType) || n < 0 ||
        !A || !W || !fits_int(n) || !fits_int(lda) ||
        lda < std::max<int64_t>(1, n) || !workspaceInBytesOnDevice ||
        !workspaceInBytesOnHost)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = syevd_work_elems(static_cast<int>(n));
    if (required < 0 || required > std::numeric_limits<int>::max())
        return CUSOLVER_STATUS_INVALID_VALUE;
    const size_t elem_size = dataTypeA == CUDA_R_32F ? sizeof(float) : sizeof(double);
    *workspaceInBytesOnDevice =
        static_cast<size_t>(required) * elem_size;
    // The Accelerate backend allocates its integer workspace internally; no
    // host workspace is required.
    *workspaceInBytesOnHost = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevd(cusolverDnHandle_t handle,
                                  cusolverDnParams_t /*params*/,
                                  cusolverEigMode_t jobz,
                                  cublasFillMode_t uplo, int64_t n,
                                  cudaDataType dataTypeA, void* A, int64_t lda,
                                  cudaDataType dataTypeW, void* W,
                                  cudaDataType computeType,
                                  void* bufferOnDevice,
                                  size_t workspaceInBytesOnDevice,
                                  void* /*bufferOnHost*/,
                                  size_t /*workspaceInBytesOnHost*/, int* info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) ||
        !valid_xsyev_types(dataTypeA, dataTypeW, computeType) || n < 0 ||
        !A || !W || !bufferOnDevice || !info || !fits_int(n) ||
        !fits_int(lda) || lda < std::max<int64_t>(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const int ni = static_cast<int>(n);
    const long long required = syevd_work_elems(ni);
    if (required < 0 || required > std::numeric_limits<int>::max())
        return CUSOLVER_STATUS_INVALID_VALUE;
    const size_t elem_size = dataTypeA == CUDA_R_32F ? sizeof(float) : sizeof(double);
    if (workspaceInBytesOnDevice < static_cast<size_t>(required) * elem_size)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    // LAPACK's lwork is an int; a workspace larger than the requirement just
    // reports the requirement.
    const int lwork = static_cast<int>(std::min<long long>(
        static_cast<long long>(workspaceInBytesOnDevice / elem_size), required));
    if (dataTypeA == CUDA_R_32F) {
        return run_syevd<float>(jobz, uplo, ni, static_cast<float*>(A),
                                static_cast<int>(lda), static_cast<float*>(W),
                                static_cast<float*>(bufferOnDevice), lwork, info);
    }
    return run_syevd<double>(jobz, uplo, ni, static_cast<double*>(A),
                             static_cast<int>(lda), static_cast<double*>(W),
                             static_cast<double*>(bufferOnDevice), lwork, info);
}

// ── Batched generic syevd ───────────────────────────────────────────────────

cusolverStatus_t cusolverDnXsyevBatched_bufferSize(cusolverDnHandle_t handle,
                                                   cusolverDnParams_t /*params*/,
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
                                                   size_t* workspaceInBytesOnHost) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) ||
        !valid_xsyev_types(dataTypeA, dataTypeW, computeType) || n < 0 ||
        batchSize < 0 || !A || !W || !fits_int(n) || !fits_int(lda) ||
        !fits_int(strideA) || !fits_int(strideW) || !fits_int(batchSize) ||
        lda < std::max<int64_t>(1, n) || strideA < static_cast<int64_t>(lda) * n ||
        strideW < n || !workspaceInBytesOnDevice || !workspaceInBytesOnHost)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = syevd_work_elems(static_cast<int>(n));
    if (required < 0 || required > std::numeric_limits<int>::max())
        return CUSOLVER_STATUS_INVALID_VALUE;
    const size_t elem_size = dataTypeA == CUDA_R_32F ? sizeof(float) : sizeof(double);
    // Matrices are processed sequentially; the same workspace serves each one.
    *workspaceInBytesOnDevice = static_cast<size_t>(required) * elem_size;
    *workspaceInBytesOnHost = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevBatched(cusolverDnHandle_t handle,
                                        cusolverDnParams_t /*params*/,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int64_t n,
                                        cudaDataType dataTypeA, void* A,
                                        int64_t lda, int64_t strideA,
                                        cudaDataType dataTypeW, void* W,
                                        int64_t strideW, cudaDataType computeType,
                                        int64_t batchSize, void* bufferOnDevice,
                                        size_t workspaceInBytesOnDevice,
                                        void* /*bufferOnHost*/,
                                        size_t /*workspaceInBytesOnHost*/, int* info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) ||
        !valid_xsyev_types(dataTypeA, dataTypeW, computeType) || n < 0 ||
        batchSize < 0 || !A || !W || !bufferOnDevice || !info || !fits_int(n) ||
        !fits_int(lda) || !fits_int(strideA) || !fits_int(strideW) ||
        !fits_int(batchSize) || lda < std::max<int64_t>(1, n) ||
        strideA < static_cast<int64_t>(lda) * n || strideW < n)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const int ni = static_cast<int>(n);
    const long long required = syevd_work_elems(ni);
    if (required < 0 || required > std::numeric_limits<int>::max())
        return CUSOLVER_STATUS_INVALID_VALUE;
    const size_t elem_size = dataTypeA == CUDA_R_32F ? sizeof(float) : sizeof(double);
    if (workspaceInBytesOnDevice < static_cast<size_t>(required) * elem_size)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    const int lwork = static_cast<int>(std::min<long long>(
        static_cast<long long>(workspaceInBytesOnDevice / elem_size), required));
    const int ldai = static_cast<int>(lda);
    const int64_t sa = strideA, sw = strideW;
    const int64_t batch = batchSize;
    if (dataTypeA == CUDA_R_32F) {
        float* a = static_cast<float*>(A);
        float* w = static_cast<float*>(W);
        for (int64_t i = 0; i < batch; ++i) {
            const cusolverStatus_t st = run_syevd<float>(
                jobz, uplo, ni, a + i * sa, ldai, w + i * sw,
                static_cast<float*>(bufferOnDevice), lwork, &info[i]);
            if (st != CUSOLVER_STATUS_SUCCESS) return st;
        }
    } else {
        double* a = static_cast<double*>(A);
        double* w = static_cast<double*>(W);
        for (int64_t i = 0; i < batch; ++i) {
            const cusolverStatus_t st = run_syevd<double>(
                jobz, uplo, ni, a + i * sa, ldai, w + i * sw,
                static_cast<double*>(bufferOnDevice), lwork, &info[i]);
            if (st != CUSOLVER_STATUS_SUCCESS) return st;
        }
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Jacobi eigensolver (syevjInfo lifecycle + batched) ──────────────────────

struct syevjInfo {
    double tolerance = 0.0;   // 0 selects the implementation default
    int max_sweeps = 100;
    int sort_eig = 1;         // non-zero: ascending sort (cuSOLVER's default)
    // Statistics of the most recent run through this object.
    double residual = 0.0;
    int executed_sweeps = 0;
};

cusolverStatus_t cusolverDnCreateSyevjInfo(syevjInfo_t* info) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) syevjInfo();
    if (*info == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroySyevjInfo(syevjInfo_t info) {
    delete info;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) {
    if (!info || !std::isfinite(tolerance) || tolerance < 0.0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    info->tolerance = tolerance;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) {
    if (!info || max_sweeps < 0) return CUSOLVER_STATUS_INVALID_VALUE;
    info->max_sweeps = max_sweeps;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sort_eig = sort_eig != 0 ? 1 : 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle,
                                             syevjInfo_t info, double* residual) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!info || !residual) return CUSOLVER_STATUS_INVALID_VALUE;
    *residual = info->residual;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle,
                                          syevjInfo_t info, int* executed_sweeps) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!info || !executed_sweeps) return CUSOLVER_STATUS_INVALID_VALUE;
    *executed_sweeps = info->executed_sweeps;
    return CUSOLVER_STATUS_SUCCESS;
}

}  // extern "C" — templates must have C++ linkage

namespace {

template <typename T>
T jacobi_off_norm(const T* a, int n, int lda) {
    T sum = T(0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (i != j) sum += a[i + j * lda] * a[i + j * lda];
    return std::sqrt(sum);
}

// Classical cyclic Jacobi rotation on a full symmetric n×n column-major
// matrix. v, when non-null, accumulates the rotations (n×n, ld = n).
template <typename T>
void jacobi_sweep(T* a, int n, int lda, T* v) {
    for (int p = 0; p < n - 1; ++p) {
        for (int q = p + 1; q < n; ++q) {
            const T apq = a[p + q * lda];
            if (apq == T(0)) continue;
            const T app = a[p + p * lda];
            const T aqq = a[q + q * lda];
            const T theta = (aqq - app) / (T(2) * apq);
            const T sign = theta >= T(0) ? T(1) : T(-1);
            const T t = sign / (std::fabs(theta) + std::sqrt(theta * theta + T(1)));
            const T c = T(1) / std::sqrt(t * t + T(1));
            const T s = t * c;
            for (int k = 0; k < n; ++k) {
                const T akp = a[k + p * lda];
                const T akq = a[k + q * lda];
                a[k + p * lda] = c * akp - s * akq;
                a[k + q * lda] = s * akp + c * akq;
            }
            for (int k = 0; k < n; ++k) {
                const T apk = a[p + k * lda];
                const T aqk = a[q + k * lda];
                a[p + k * lda] = c * apk - s * aqk;
                a[q + k * lda] = s * apk + c * aqk;
            }
            a[p + q * lda] = a[q + p * lda] = T(0);
            if (v) {
                for (int k = 0; k < n; ++k) {
                    const T vkp = v[k + p * n];
                    const T vkq = v[k + q * n];
                    v[k + p * n] = c * vkp - s * vkq;
                    v[k + q * n] = s * vkp + c * vkq;
                }
            }
        }
    }
}

// Runs Jacobi on one symmetric matrix. A is column-major n×n with ld = lda;
// only the uplo triangle is read and it is mirrored to the full matrix first.
// W receives n eigenvalues. work must hold n*n elements when jobz == VECTOR
// (eigenvector accumulation); it is unused otherwise. Returns the sweeps
// executed, reports the final off-diagonal norm through residual_out, and
// sets converged_out when the tolerance was reached inside max_sweeps.
template <typename T>
int jacobi_eigen(T* a, int n, int lda, T* w, T* work,
                 cusolverEigMode_t jobz, cublasFillMode_t uplo,
                 const syevjInfo& params, T* residual_out, bool* converged_out) {
    // Mirror the referenced triangle so the full symmetric matrix is rotated.
    if (uplo == CUBLAS_FILL_MODE_LOWER) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < j; ++i) a[i + j * lda] = a[j + i * lda];
    } else {
        for (int j = 0; j < n; ++j)
            for (int i = j + 1; i < n; ++i) a[i + j * lda] = a[j + i * lda];
    }

    T* v = nullptr;
    if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        v = work;
        std::fill(v, v + static_cast<size_t>(n) * n, T(0));
        for (int i = 0; i < n; ++i) v[i + i * n] = T(1);
    }

    // Default threshold: relative to the diagonal norm at machine epsilon.
    T diag_norm = T(0);
    for (int i = 0; i < n; ++i) diag_norm += a[i + i * lda] * a[i + i * lda];
    diag_norm = std::sqrt(diag_norm);
    const T threshold = params.tolerance > 0.0
                            ? static_cast<T>(params.tolerance)
                            : std::numeric_limits<T>::epsilon() *
                                  std::max(diag_norm, T(1));

    int sweeps = 0;
    T off = jacobi_off_norm(a, n, lda);
    while (off > threshold && sweeps < params.max_sweeps) {
        jacobi_sweep(a, n, lda, v);
        ++sweeps;
        off = jacobi_off_norm(a, n, lda);
    }
    *residual_out = off;
    *converged_out = off <= threshold;

    for (int i = 0; i < n; ++i) w[i] = a[i + i * lda];

    if (params.sort_eig) {
        // Ascending sort of eigenvalues; gather eigenvector columns in the
        // same order.
        std::vector<int> order(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
        std::sort(order.begin(), order.end(),
                  [&](int lhs, int rhs) { return w[lhs] < w[rhs]; });
        std::vector<T> sorted_w(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
            sorted_w[static_cast<size_t>(i)] = w[order[static_cast<size_t>(i)]];
        if (v) {
            std::vector<T> sorted_v(static_cast<size_t>(n) * n);
            for (int j = 0; j < n; ++j) {
                const int src = order[static_cast<size_t>(j)];
                for (int i = 0; i < n; ++i)
                    sorted_v[i + j * n] = v[i + src * n];
            }
            std::copy(sorted_v.begin(), sorted_v.end(), v);
        }
        std::copy(sorted_w.begin(), sorted_w.end(), w);
    }

    if (v) {
        // Eigenvectors overwrite A, matching LAPACK's jobz == 'V' contract.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) a[i + j * lda] = v[i + j * n];
    }
    return sweeps;
}

template <typename T>
cusolverStatus_t syevj_batched_buffer_size(cusolverDnHandle_t handle,
                                           cusolverEigMode_t jobz,
                                           cublasFillMode_t uplo, int n,
                                           const T* A, int lda, const T* W,
                                           int* lwork, syevjInfo_t /*params*/,
                                           int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || batchSize < 0 ||
        !A || !W || lda < std::max(1, n) || !lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    // Eigenvector accumulation needs an n×n scratch per matrix, reused across
    // the batch.
    *lwork = jobz == CUSOLVER_EIG_MODE_VECTOR ? std::max(1, n * n) : 1;
    return CUSOLVER_STATUS_SUCCESS;
}

template <typename T>
cusolverStatus_t syevj_batched(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                             cublasFillMode_t uplo, int n, T* A, int lda, T* W,
                             T* work, int lwork, int* devInfo,
                             syevjInfo_t params, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const int required = jobz == CUSOLVER_EIG_MODE_VECTOR ? std::max(1, n * n) : 1;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || batchSize < 0 ||
        !A || !W || !work || !devInfo || lda < std::max(1, n) || lwork < required)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;

    syevjInfo config = params ? *params : syevjInfo();
    double worst_residual = 0.0;
    int most_sweeps = 0;
    // Matrices are contiguous with stride lda*n; eigenvalue vectors with n.
    for (int b = 0; b < batchSize; ++b) {
        T residual = T(0);
        bool converged = false;
        const int sweeps =
            jacobi_eigen<T>(A + static_cast<size_t>(b) * lda * n, n, lda,
                            W + static_cast<size_t>(b) * n, work, jobz, uplo,
                            config, &residual, &converged);
        // Per-matrix convergence report: 0 = converged, >0 = the tolerance was
        // not reached (max_sweeps = 0 still reports a positive count).
        devInfo[b] = converged ? 0 : std::max(1, sweeps);
        worst_residual = std::max(worst_residual, static_cast<double>(residual));
        most_sweeps = std::max(most_sweeps, sweeps);
    }
    if (params) {
        params->residual = worst_residual;
        params->executed_sweeps = most_sweeps;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

cusolverStatus_t cusolverDnSsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n,
                                                    const float* A, int lda,
                                                    const float* W, int* lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
    return syevj_batched_buffer_size<float>(handle, jobz, uplo, n, A, lda, W,
                                            lwork, params, batchSize);
}

cusolverStatus_t cusolverDnDsyevjBatched_bufferSize(cusolverDnHandle_t handle,
                                                    cusolverEigMode_t jobz,
                                                    cublasFillMode_t uplo, int n,
                                                    const double* A, int lda,
                                                    const double* W, int* lwork,
                                                    syevjInfo_t params,
                                                    int batchSize) {
    return syevj_batched_buffer_size<double>(handle, jobz, uplo, n, A, lda, W,
                                             lwork, params, batchSize);
}

cusolverStatus_t cusolverDnSsyevjBatched(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, float* A,
                                        int lda, float* W, float* work,
                                        int lwork, int* devInfo,
                                        syevjInfo_t params, int batchSize) {
    return syevj_batched<float>(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                devInfo, params, batchSize);
}

cusolverStatus_t cusolverDnDsyevjBatched(cusolverDnHandle_t handle,
                                        cusolverEigMode_t jobz,
                                        cublasFillMode_t uplo, int n, double* A,
                                        int lda, double* W, double* work,
                                        int lwork, int* devInfo,
                                        syevjInfo_t params, int batchSize) {
    return syevj_batched<double>(handle, jobz, uplo, n, A, lda, W, work, lwork,
                                 devInfo, params, batchSize);
}

}  // extern "C"

// ── CuPy surface ────────────────────────────────────────────────────────────
// Complex dense solvers, QR helpers, batched Cholesky, LDL, bidiagonalisation,
// Jacobi SVD/EVD, approximate SVD, the IRS gesv/gels families and the sparse
// complex/eigenvalue extras. Everything below is templated on the element type
// (float, double, cuComplex, cuDoubleComplex) over Accelerate LAPACK.
//
// Status convention follows cuSOLVER: a LAPACK "info > 0" (singular, not
// positive definite, ...) is reported through devInfo with status SUCCESS; a
// LAPACK "info < 0" is an argument error and maps to INVALID_VALUE. The older
// S/D entry points above return INTERNAL_ERROR for info > 0 instead.

namespace {

static_assert(sizeof(__CLPK_integer) == sizeof(int), "LAPACK integer must be 32-bit");

template <class T> struct Traits;
template <> struct Traits<float> {
    using Real = float; using Lp = float; using Std = float;
    static constexpr bool kComplex = false;
};
template <> struct Traits<double> {
    using Real = double; using Lp = double; using Std = double;
    static constexpr bool kComplex = false;
};
template <> struct Traits<cuComplex> {
    using Real = float; using Lp = __CLPK_complex; using Std = std::complex<float>;
    static constexpr bool kComplex = true;
};
template <> struct Traits<cuDoubleComplex> {
    using Real = double; using Lp = __CLPK_doublecomplex; using Std = std::complex<double>;
    static constexpr bool kComplex = true;
};

template <class T> using Real = typename Traits<T>::Real;
template <class T> using Lp = typename Traits<T>::Lp;
template <class T> using Std = typename Traits<T>::Std;

template <class T> Lp<T>* lp(T* p) { return reinterpret_cast<Lp<T>*>(p); }
template <class T> Lp<T>* lpc(const T* p) { return reinterpret_cast<Lp<T>*>(const_cast<T*>(p)); }
template <class T> Std<T>* sp(T* p) { return reinterpret_cast<Std<T>*>(p); }
template <class T> const Std<T>* spc(const T* p) { return reinterpret_cast<const Std<T>*>(p); }

inline float cj(float x) { return x; }
inline double cj(double x) { return x; }
template <class R> std::complex<R> cj(std::complex<R> x) { return std::conj(x); }
inline float ab(float x) { return std::fabs(x); }
inline double ab(double x) { return std::fabs(x); }
template <class R> R ab(std::complex<R> x) { return std::abs(x); }

template <class T> Std<T> to_std(T v) {
    if constexpr (Traits<T>::kComplex) return Std<T>(v.x, v.y);
    else return v;
}
template <class T> T from_std(Std<T> v) {
    if constexpr (Traits<T>::kComplex) {
        T r;
        r.x = v.real();
        r.y = v.imag();
        return r;
    } else {
        return v;
    }
}
template <class T> T cval(double v) { return from_std<T>(Std<T>(static_cast<Real<T>>(v))); }

// Work-size queries come back as a floating value; round up with a margin so
// single precision rounding cannot leave the buffer a unit short.
template <class T> int qint(const T& v) {
    double r;
    if constexpr (Traits<T>::kComplex) r = v.x;
    else r = v;
    return static_cast<int>(std::min(std::ceil(r) + 1.0, 2147483647.0));
}

#define CS_LAPACK(S_, D_, C_, Z_, ...)                                         \
    do {                                                                       \
        if constexpr (std::is_same<T, float>::value) S_(__VA_ARGS__);          \
        else if constexpr (std::is_same<T, double>::value) D_(__VA_ARGS__);    \
        else if constexpr (std::is_same<T, cuComplex>::value) C_(__VA_ARGS__); \
        else Z_(__VA_ARGS__);                                                  \
    } while (0)

// ── LAPACK wrappers; each returns LAPACK's info ─────────────────────────────

template <class T> int op_getrf(int m, int n, T* a, int lda, int* ipiv) {
    __CLPK_integer M = m, N = n, LDA = lda, info = 0;
    CS_LAPACK(sgetrf_, dgetrf_, cgetrf_, zgetrf_, &M, &N, lp(a), &LDA, ipiv, &info);
    return info;
}
template <class T> int op_getrs(char t, int n, int nrhs, const T* a, int lda,
                                const int* ipiv, T* b, int ldb) {
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    CS_LAPACK(sgetrs_, dgetrs_, cgetrs_, zgetrs_, &t, &N, &NRHS, lpc(a), &LDA,
              const_cast<int*>(ipiv), lp(b), &LDB, &info);
    return info;
}
template <class T> int op_geqrf(int m, int n, T* a, int lda, T* tau, T* work, int lwork) {
    __CLPK_integer M = m, N = n, LDA = lda, LW = lwork, info = 0;
    CS_LAPACK(sgeqrf_, dgeqrf_, cgeqrf_, zgeqrf_, &M, &N, lp(a), &LDA, lp(tau), lp(work),
              &LW, &info);
    return info;
}
template <class T> int op_orgqr(int m, int n, int k, T* a, int lda, const T* tau, T* work,
                                int lwork) {
    __CLPK_integer M = m, N = n, K = k, LDA = lda, LW = lwork, info = 0;
    CS_LAPACK(sorgqr_, dorgqr_, cungqr_, zungqr_, &M, &N, &K, lp(a), &LDA, lpc(tau),
              lp(work), &LW, &info);
    return info;
}
template <class T> int op_ormqr(char side, char trans, int m, int n, int k, const T* a,
                                int lda, const T* tau, T* c, int ldc, T* work, int lwork) {
    __CLPK_integer M = m, N = n, K = k, LDA = lda, LDC = ldc, LW = lwork, info = 0;
    CS_LAPACK(sormqr_, dormqr_, cunmqr_, zunmqr_, &side, &trans, &M, &N, &K, lpc(a), &LDA,
              lpc(tau), lp(c), &LDC, lp(work), &LW, &info);
    return info;
}
template <class T> int op_potrf(char uplo, int n, T* a, int lda) {
    __CLPK_integer N = n, LDA = lda, info = 0;
    CS_LAPACK(spotrf_, dpotrf_, cpotrf_, zpotrf_, &uplo, &N, lp(a), &LDA, &info);
    return info;
}
template <class T> int op_potrs(char uplo, int n, int nrhs, const T* a, int lda, T* b,
                                int ldb) {
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    CS_LAPACK(spotrs_, dpotrs_, cpotrs_, zpotrs_, &uplo, &N, &NRHS, lpc(a), &LDA, lp(b),
              &LDB, &info);
    return info;
}
template <class T> int op_sytrf(char uplo, int n, T* a, int lda, int* ipiv, T* work,
                                int lwork) {
    __CLPK_integer N = n, LDA = lda, LW = lwork, info = 0;
    CS_LAPACK(ssytrf_, dsytrf_, csytrf_, zsytrf_, &uplo, &N, lp(a), &LDA, ipiv, lp(work),
              &LW, &info);
    return info;
}
template <class T> int op_gebrd(int m, int n, T* a, int lda, Real<T>* d, Real<T>* e,
                                T* tauq, T* taup, T* work, int lwork) {
    __CLPK_integer M = m, N = n, LDA = lda, LW = lwork, info = 0;
    CS_LAPACK(sgebrd_, dgebrd_, cgebrd_, zgebrd_, &M, &N, lp(a), &LDA, d, e, lp(tauq),
              lp(taup), lp(work), &LW, &info);
    return info;
}
template <class T> int op_gesvd(char ju, char jvt, int m, int n, T* a, int lda,
                                Real<T>* s, T* u, int ldu, T* vt, int ldvt, T* work,
                                int lwork, Real<T>* rwork) {
    __CLPK_integer M = m, N = n, LDA = lda, LDU = ldu, LDVT = ldvt, LW = lwork, info = 0;
    if constexpr (Traits<T>::kComplex) {
        if constexpr (std::is_same<T, cuComplex>::value)
            cgesvd_(&ju, &jvt, &M, &N, lp(a), &LDA, s, lp(u), &LDU, lp(vt), &LDVT, lp(work),
                    &LW, rwork, &info);
        else
            zgesvd_(&ju, &jvt, &M, &N, lp(a), &LDA, s, lp(u), &LDU, lp(vt), &LDVT, lp(work),
                    &LW, rwork, &info);
    } else {
        if constexpr (std::is_same<T, float>::value)
            sgesvd_(&ju, &jvt, &M, &N, lp(a), &LDA, s, lp(u), &LDU, lp(vt), &LDVT, lp(work),
                    &LW, &info);
        else
            dgesvd_(&ju, &jvt, &M, &N, lp(a), &LDA, s, lp(u), &LDU, lp(vt), &LDVT, lp(work),
                    &LW, &info);
    }
    return info;
}
// Hermitian divide-and-conquer eigensolver (complex only here; the real
// syevd entry points are older and call LAPACK directly).
template <class T> int op_heevd(char job, char uplo, int n, T* a, int lda, Real<T>* w,
                                T* work, int lwork, Real<T>* rwork, int lrwork, int* iwork,
                                int liwork) {
    __CLPK_integer N = n, LDA = lda, LW = lwork, LRW = lrwork, LIW = liwork, info = 0;
    if constexpr (std::is_same<T, cuComplex>::value)
        cheevd_(&job, &uplo, &N, lp(a), &LDA, w, lp(work), &LW, rwork, &LRW, iwork, &LIW,
                &info);
    else
        zheevd_(&job, &uplo, &N, lp(a), &LDA, w, lp(work), &LW, rwork, &LRW, iwork, &LIW,
                &info);
    return info;
}
template <class T> int op_gels(char t, int m, int n, int nrhs, T* a, int lda, T* b, int ldb,
                               T* work, int lwork) {
    __CLPK_integer M = m, N = n, NRHS = nrhs, LDA = lda, LDB = ldb, LW = lwork, info = 0;
    CS_LAPACK(sgels_, dgels_, cgels_, zgels_, &t, &M, &N, &NRHS, lp(a), &LDA, lp(b), &LDB,
              lp(work), &LW, &info);
    return info;
}

template <class T>
void gemm(CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, T alpha, const T* A,
          int lda, const T* B, int ldb, T beta, T* C, int ldc) {
    if constexpr (std::is_same<T, float>::value)
        cblas_sgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if constexpr (std::is_same<T, double>::value)
        cblas_dgemm(CblasColMajor, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else if constexpr (std::is_same<T, cuComplex>::value)
        cblas_cgemm(CblasColMajor, ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
    else
        cblas_zgemm(CblasColMajor, ta, tb, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
}

template <class F> cusolverStatus_t guarded(F&& f) {
    try {
        return f();
    } catch (const std::bad_alloc&) {
        return CUSOLVER_STATUS_ALLOC_FAILED;
    } catch (...) {
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    }
}

inline cusolverStatus_t finish(int info, int* devInfo) {
    *devInfo = info;
    return info < 0 ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_SUCCESS;
}

inline bool valid_op(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}
inline char op_char(cublasOperation_t op) {
    return op == CUBLAS_OP_N ? 'N' : (op == CUBLAS_OP_T ? 'T' : 'C');
}

// Unpivoted LU: what cuSOLVER's getrf does when devIpiv is NULL. Returns the
// 1-based index of the first zero pivot, or 0.
template <class T> int lu_nopivot(int m, int n, T* a_, int lda) {
    using S = Std<T>;
    S* a = sp(a_);
    int info = 0;
    for (int k = 0; k < std::min(m, n); ++k) {
        const S piv = a[k + static_cast<size_t>(k) * lda];
        if (piv == S(0)) {
            if (!info) info = k + 1;
            continue;
        }
        for (int i = k + 1; i < m; ++i) a[i + static_cast<size_t>(k) * lda] /= piv;
        for (int j = k + 1; j < n; ++j)
            for (int i = k + 1; i < m; ++i)
                a[i + static_cast<size_t>(j) * lda] -=
                    a[i + static_cast<size_t>(k) * lda] * a[k + static_cast<size_t>(j) * lda];
    }
    return info;
}

// Solve with unpivoted LU factors (getrs with devIpiv == NULL).
template <class T>
int lu_solve_nopivot(cublasOperation_t trans, int n, int nrhs, const T* a_, int lda, T* b_,
                     int ldb) {
    using S = Std<T>;
    const S* a = spc(a_);
    S* b = sp(b_);
    auto A = [&](int i, int j) { return a[i + static_cast<size_t>(j) * lda]; };
    auto op = [&](S x) { return trans == CUBLAS_OP_C ? cj(x) : x; };
    for (int i = 0; i < n; ++i)
        if (A(i, i) == S(0)) return i + 1;
    for (int c = 0; c < nrhs; ++c) {
        S* x = b + static_cast<size_t>(c) * ldb;
        if (trans == CUBLAS_OP_N) {
            for (int i = 0; i < n; ++i)
                for (int k = 0; k < i; ++k) x[i] -= A(i, k) * x[k];
            for (int i = n - 1; i >= 0; --i) {
                for (int k = i + 1; k < n; ++k) x[i] -= A(i, k) * x[k];
                x[i] /= A(i, i);
            }
        } else {
            for (int i = 0; i < n; ++i) {
                for (int k = 0; k < i; ++k) x[i] -= op(A(k, i)) * x[k];
                x[i] /= op(A(i, i));
            }
            for (int i = n - 1; i >= 0; --i)
                for (int k = i + 1; k < n; ++k) x[i] -= op(A(k, i)) * x[k];
        }
    }
    return 0;
}

// ── getrf / getrs / geqrf / potrf / potrs (complex entry points) ────────────

template <class T>
cusolverStatus_t dn_getrf_bs(cusolverDnHandle_t handle, int m, int n, T* A, int lda,
                             int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    (void)A;
    if (lda < std::max(1, m) || !checked_workspace_product(m, n, Lwork))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_getrf(cusolverDnHandle_t handle, int m, int n, T* A, int lda,
                          T* /*Workspace*/, int* devIpiv, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < std::max(1, m) || !A || !devInfo)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (m == 0 || n == 0) return finish(0, devInfo);
    // NULL devIpiv selects factorization without pivoting, as in cuSOLVER.
    const int info = devIpiv ? op_getrf<T>(m, n, A, lda, devIpiv) : lu_nopivot<T>(m, n, A, lda);
    return finish(info, devInfo);
}

template <class T>
cusolverStatus_t dn_getrs(cusolverDnHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                          const T* A, int lda, const int* devIpiv, T* B, int ldb,
                          int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_op(trans) || n < 0 || nrhs < 0 || !A || !B || !devInfo ||
        lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0 || nrhs == 0) return finish(0, devInfo);
    const int info = devIpiv ? op_getrs<T>(op_char(trans), n, nrhs, A, lda, devIpiv, B, ldb)
                             : lu_solve_nopivot<T>(trans, n, nrhs, A, lda, B, ldb);
    return finish(info, devInfo);
}

template <class T>
cusolverStatus_t dn_geqrf_bs(cusolverDnHandle_t handle, int m, int n, T*, int lda,
                             int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < std::max(1, m) || !Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) { *Lwork = 1; return CUSOLVER_STATUS_SUCCESS; }
    T dummy{}, wq{};
    if (op_geqrf<T>(m, n, &dummy, lda, &dummy, &wq, -1) != 0)
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    *Lwork = std::max({1, n, qint(wq)});
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_geqrf(cusolverDnHandle_t handle, int m, int n, T* A, int lda, T* TAU,
                          T* Workspace, int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !A || !TAU || !Workspace || !devInfo || lda < std::max(1, m) ||
        Lwork < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (m == 0 || n == 0) return finish(0, devInfo);
    return finish(op_geqrf<T>(m, n, A, lda, TAU, Workspace, Lwork), devInfo);
}

template <class T>
cusolverStatus_t dn_potrf_bs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T*,
                             int lda, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || lda < std::max(1, n) || !Lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = std::max(1, n);  // LAPACK potrf needs none; kept for parity with the S/D path
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_potrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T* A,
                          int lda, T* /*Workspace*/, int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || !devInfo || lda < std::max(1, n) || Lwork < 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0) return finish(0, devInfo);
    return finish(op_potrf<T>(uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L', n, A, lda), devInfo);
}

template <class T>
cusolverStatus_t dn_potrs(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, int nrhs,
                          const T* A, int lda, T* B, int ldb, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || nrhs < 0 || !A || !B || !devInfo ||
        lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0 || nrhs == 0) return finish(0, devInfo);
    return finish(op_potrs<T>(uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L', n, nrhs, A, lda, B,
                              ldb),
                  devInfo);
}

// ── orgqr / ungqr, ormqr / unmqr ────────────────────────────────────────────

template <class T>
cusolverStatus_t dn_orgqr_bs(cusolverDnHandle_t handle, int m, int n, int k, const T*, int lda,
                             const T*, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0 || n > m || k > n || lda < std::max(1, m) || !lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (n == 0) { *lwork = 1; return CUSOLVER_STATUS_SUCCESS; }
    T dummy{}, wq{};
    if (op_orgqr<T>(m, n, k, &dummy, lda, &dummy, &wq, -1) != 0)
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    *lwork = std::max({1, n, qint(wq)});
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_orgqr(cusolverDnHandle_t handle, int m, int n, int k, T* A, int lda,
                          const T* tau, T* work, int lwork, int* info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || k < 0 || n > m || k > n || !A || !info || lda < std::max(1, m) ||
        (k > 0 && !tau) || !work || lwork < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0) return finish(0, info);
    return finish(op_orgqr<T>(m, n, k, A, lda, tau, work, lwork), info);
}

// Real ormqr takes N or T; complex unmqr takes N or C.
template <class T> bool valid_mq_trans(cublasOperation_t trans) {
    if (Traits<T>::kComplex) return trans == CUBLAS_OP_N || trans == CUBLAS_OP_C;
    return trans == CUBLAS_OP_N || trans == CUBLAS_OP_T;
}

template <class T>
bool valid_ormqr_dims(cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k,
                      int lda, int ldc) {
    if ((side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) || !valid_mq_trans<T>(trans))
        return false;
    if (m < 0 || n < 0 || k < 0) return false;
    const int order = side == CUBLAS_SIDE_LEFT ? m : n;  // A is order x k
    return k <= order && lda >= std::max(1, order) && ldc >= std::max(1, m);
}

template <class T>
cusolverStatus_t dn_ormqr_bs(cusolverDnHandle_t handle, cublasSideMode_t side,
                             cublasOperation_t trans, int m, int n, int k, const T*, int lda,
                             const T*, const T*, int ldc, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_ormqr_dims<T>(side, trans, m, n, k, lda, ldc) || !lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const int minimum = std::max(1, side == CUBLAS_SIDE_LEFT ? n : m);
    if (m == 0 || n == 0 || k == 0) { *lwork = minimum; return CUSOLVER_STATUS_SUCCESS; }
    T dummy{}, wq{};
    if (op_ormqr<T>(side == CUBLAS_SIDE_LEFT ? 'L' : 'R', op_char(trans), m, n, k, &dummy,
                    lda, &dummy, &dummy, ldc, &wq, -1) != 0)
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    *lwork = std::max(minimum, qint(wq));
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_ormqr(cusolverDnHandle_t handle, cublasSideMode_t side,
                          cublasOperation_t trans, int m, int n, int k, const T* A, int lda,
                          const T* tau, T* C, int ldc, T* work, int lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_ormqr_dims<T>(side, trans, m, n, k, lda, ldc) || !C || !devInfo || !work ||
        (k > 0 && (!A || !tau)) ||
        lwork < std::max(1, side == CUBLAS_SIDE_LEFT ? n : m))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (m == 0 || n == 0 || k == 0) return finish(0, devInfo);
    return finish(op_ormqr<T>(side == CUBLAS_SIDE_LEFT ? 'L' : 'R', op_char(trans), m, n, k, A,
                              lda, tau, C, ldc, work, lwork),
                  devInfo);
}

// ── Batched Cholesky (arrays of matrix pointers) ────────────────────────────

template <class T>
cusolverStatus_t dn_potrf_batched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                  T* Aarray[], int lda, int* infoArray, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || batchSize < 0 || lda < std::max(1, n) ||
        (batchSize > 0 && (!Aarray || !infoArray)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    for (int b = 0; b < batchSize; ++b)
        if (!Aarray[b]) return CUSOLVER_STATUS_INVALID_VALUE;
    const char ul = uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    for (int b = 0; b < batchSize; ++b)
        infoArray[b] = n == 0 ? 0 : op_potrf<T>(ul, n, Aarray[b], lda);
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_potrs_batched(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n,
                                  int nrhs, T* A[], int lda, T* B[], int ldb, int* d_info,
                                  int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || nrhs < 0 || batchSize < 0 || lda < std::max(1, n) ||
        ldb < std::max(1, n) || !d_info || (batchSize > 0 && (!A || !B)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    for (int b = 0; b < batchSize; ++b)
        if (!A[b] || !B[b]) return CUSOLVER_STATUS_INVALID_VALUE;
    const char ul = uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L';
    // cuSOLVER reports a single devInfo for the whole batch.
    int worst = 0;
    for (int b = 0; b < batchSize && n > 0 && nrhs > 0; ++b) {
        const int info = op_potrs<T>(ul, n, nrhs, A[b], lda, B[b], ldb);
        if (info != 0 && worst == 0) worst = info;
    }
    return finish(worst, d_info);
}

// ── sytrf (symmetric indefinite LDL^T, complex-symmetric for C/Z) ───────────

template <class T>
cusolverStatus_t dn_sytrf_bs(cusolverDnHandle_t handle, int n, T*, int lda, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || lda < std::max(1, n) || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    if (n == 0) { *lwork = 1; return CUSOLVER_STATUS_SUCCESS; }
    T dummy{}, wq{};
    int dummy_ipiv = 0;
    if (op_sytrf<T>('L', n, &dummy, lda, &dummy_ipiv, &wq, -1) != 0)
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    *lwork = std::max(1, qint(wq));
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_sytrf(cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T* A,
                          int lda, int* ipiv, T* work, int lwork, int* info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_fill(uplo) || n < 0 || !A || !info || lda < std::max(1, n) || !work ||
        lwork < 1 || (n > 0 && !ipiv))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0) return finish(0, info);
    return finish(op_sytrf<T>(uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L', n, A, lda, ipiv,
                              work, lwork),
                  info);
}

// ── gebrd ───────────────────────────────────────────────────────────────────

template <class T>
cusolverStatus_t dn_gebrd_bs(cusolverDnHandle_t handle, int m, int n, int* Lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) { *Lwork = 1; return CUSOLVER_STATUS_SUCCESS; }
    T dummy{}, wq{};
    Real<T> rd{};
    if (op_gebrd<T>(m, n, &dummy, std::max(1, m), &rd, &rd, &dummy, &dummy, &wq, -1) != 0)
        return CUSOLVER_STATUS_INTERNAL_ERROR;
    *Lwork = std::max({1, m, n, qint(wq)});
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_gebrd(cusolverDnHandle_t handle, int m, int n, T* A, int lda, Real<T>* D,
                          Real<T>* E, T* TAUQ, T* TAUP, T* Work, int Lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !A || !D || !E || !TAUQ || !TAUP || !Work || !devInfo ||
        lda < std::max(1, m) || Lwork < std::max({1, m, n}))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (m == 0 || n == 0) return finish(0, devInfo);
    return finish(op_gebrd<T>(m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork), devInfo);
}

// ── Complex gesvd ───────────────────────────────────────────────────────────

template <class T>
cusolverStatus_t dn_gesvd_bs(cusolverDnHandle_t handle, int m, int n, int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || !lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = 3LL * std::min(m, n) + 2LL * std::max(m, n);
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = std::max(1, static_cast<int>(required));
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_gesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                          int m, int n, T* A, int lda, Real<T>* S, T* U, int ldu, T* VT,
                          int ldvt, T* work, int lwork, Real<T>* /*rwork*/, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const char ju = static_cast<char>(jobu), jvt = static_cast<char>(jobvt);
    const int min_dim = std::min(m, n);
    const bool wants_u = ju == 'A' || ju == 'S';
    const bool wants_vt = jvt == 'A' || jvt == 'S';
    const int required_ldvt = jvt == 'A' ? std::max(1, n) : std::max(1, min_dim);
    // LAPACK's complex minimum is 2*min+max; bufferSize reports the larger
    // real-precision figure, which also satisfies it.
    const long long required = 2LL * min_dim + std::max(m, n);
    if (!valid_svd_job(ju) || !valid_svd_job(jvt) || (ju == 'O' && jvt == 'O') || m < 0 ||
        n < 0 || !A || !S || !work || !devInfo || lda < std::max(1, m) || ldu < 1 ||
        ldvt < 1 || (wants_u && (!U || ldu < std::max(1, m))) ||
        (wants_vt && (!VT || ldvt < required_ldvt)) ||
        required > std::numeric_limits<int>::max() ||
        lwork < std::max(1, static_cast<int>(required)))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (m == 0 || n == 0) return finish(0, devInfo);
    return guarded([&] {
        // cuSOLVER lets callers pass rwork = NULL; LAPACK needs 5*min(m,n) reals.
        std::vector<Real<T>> rwork(static_cast<size_t>(std::max(1, 5 * min_dim)));
        return finish(op_gesvd<T>(ju, jvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,
                                  rwork.data()),
                      devInfo);
    });
}

// ── Complex Hermitian heevd ─────────────────────────────────────────────────

inline long long heevd_work_elems(int n, cusolverEigMode_t jobz) {
    const long long nn = n;
    return std::max(1LL, jobz == CUSOLVER_EIG_MODE_VECTOR ? 2 * nn + nn * nn : nn + 1);
}

template <class T>
cusolverStatus_t dn_heevd_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                             cublasFillMode_t uplo, int n, const T*, int lda, const Real<T>*,
                             int* lwork) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || lda < std::max(1, n) || !lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    // The vector size also covers the values-only case.
    const long long required = heevd_work_elems(n, CUSOLVER_EIG_MODE_VECTOR);
    if (required > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = static_cast<int>(required);
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t dn_heevd(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                          cublasFillMode_t uplo, int n, T* A, int lda, Real<T>* W, T* work,
                          int lwork, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || !A || !W || !work ||
        !devInfo || lda < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const long long required = heevd_work_elems(n, jobz);
    if (required > std::numeric_limits<int>::max() || lwork < required)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (n == 0) return finish(0, devInfo);
    return guarded([&] {
        const bool vec = jobz == CUSOLVER_EIG_MODE_VECTOR;
        const long long nn = n;
        const long long lrwork = vec ? 1 + 5 * nn + 2 * nn * nn : nn;
        const long long liwork = vec ? 3 + 5 * nn : 1;
        if (lrwork > std::numeric_limits<int>::max()) return CUSOLVER_STATUS_INVALID_VALUE;
        std::vector<Real<T>> rwork(static_cast<size_t>(lrwork));
        std::vector<int> iwork(static_cast<size_t>(liwork));
        return finish(op_heevd<T>(vec ? 'V' : 'N', uplo == CUBLAS_FILL_MODE_UPPER ? 'U' : 'L',
                                  n, A, lda, W, work, lwork, rwork.data(),
                                  static_cast<int>(lrwork), iwork.data(),
                                  static_cast<int>(liwork)),
                      devInfo);
    });
}

}  // namespace

// ── Jacobi eigensolver: complex Hermitian and the single-matrix entry ───────

namespace {

// One cyclic sweep on a full n x n Hermitian matrix. Each (p,q) rotation first
// removes the phase of a(p,q) with the unitary diag(1, e^{-i phi}) so the 2x2
// block becomes real symmetric, then applies the real Jacobi rotation. v, when
// non-null, accumulates the same transform (n x n, ld = n).
template <class R>
void jacobi_sweep_c(std::complex<R>* a, int n, int lda, std::complex<R>* v) {
    using C = std::complex<R>;
    auto A = [&](int i, int j) -> C& { return a[i + static_cast<size_t>(j) * lda]; };
    for (int p = 0; p < n - 1; ++p) {
        for (int q = p + 1; q < n; ++q) {
            const C apq = A(p, q);
            const R r = std::abs(apq);
            if (r == R(0)) continue;
            const C e = apq / r;
            const C ce = std::conj(e);
            const R app = A(p, p).real(), aqq = A(q, q).real();
            const R theta = (aqq - app) / (R(2) * r);
            const R sign = theta >= R(0) ? R(1) : R(-1);
            const R t = sign / (std::fabs(theta) + std::sqrt(theta * theta + R(1)));
            const R c = R(1) / std::sqrt(t * t + R(1));
            const R s = t * c;
            for (int k = 0; k < n; ++k) A(k, q) *= ce;
            for (int k = 0; k < n; ++k) A(q, k) *= e;
            if (v)
                for (int k = 0; k < n; ++k) v[k + static_cast<size_t>(q) * n] *= ce;
            for (int k = 0; k < n; ++k) {
                const C akp = A(k, p), akq = A(k, q);
                A(k, p) = c * akp - s * akq;
                A(k, q) = s * akp + c * akq;
            }
            for (int k = 0; k < n; ++k) {
                const C apk = A(p, k), aqk = A(q, k);
                A(p, k) = c * apk - s * aqk;
                A(q, k) = s * apk + c * aqk;
            }
            A(p, q) = A(q, p) = C(0);
            A(p, p) = C(A(p, p).real(), 0);
            A(q, q) = C(A(q, q).real(), 0);
            if (v) {
                for (int k = 0; k < n; ++k) {
                    const C vkp = v[k + static_cast<size_t>(p) * n];
                    const C vkq = v[k + static_cast<size_t>(q) * n];
                    v[k + static_cast<size_t>(p) * n] = c * vkp - s * vkq;
                    v[k + static_cast<size_t>(q) * n] = s * vkp + c * vkq;
                }
            }
        }
    }
}

template <class R>
R jacobi_off_norm_c(const std::complex<R>* a, int n, int lda) {
    R sum = R(0);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (i != j) sum += std::norm(a[i + static_cast<size_t>(j) * lda]);
    return std::sqrt(sum);
}

// Counterpart of jacobi_eigen for complex Hermitian input; same contract
// (only the uplo triangle is read, eigenvectors overwrite A, work holds n*n
// elements when vectors are requested).
template <class R>
int jacobi_eigen_c(std::complex<R>* a, int n, int lda, R* w, std::complex<R>* work,
                   cusolverEigMode_t jobz, cublasFillMode_t uplo, const syevjInfo& params,
                   R* residual_out, bool* converged_out) {
    using C = std::complex<R>;
    auto A = [&](int i, int j) -> C& { return a[i + static_cast<size_t>(j) * lda]; };
    for (int j = 0; j < n; ++j) {
        A(j, j) = C(A(j, j).real(), 0);
        for (int i = 0; i < n; ++i) {
            if (uplo == CUBLAS_FILL_MODE_LOWER && i < j) A(i, j) = std::conj(A(j, i));
            if (uplo == CUBLAS_FILL_MODE_UPPER && i > j) A(i, j) = std::conj(A(j, i));
        }
    }
    C* v = nullptr;
    if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
        v = work;
        std::fill(v, v + static_cast<size_t>(n) * n, C(0));
        for (int i = 0; i < n; ++i) v[i + static_cast<size_t>(i) * n] = C(1);
    }
    R diag_norm = R(0);
    for (int i = 0; i < n; ++i) diag_norm += A(i, i).real() * A(i, i).real();
    diag_norm = std::sqrt(diag_norm);
    const R threshold = params.tolerance > 0.0
                            ? static_cast<R>(params.tolerance)
                            : std::numeric_limits<R>::epsilon() * std::max(diag_norm, R(1));
    int sweeps = 0;
    R off = jacobi_off_norm_c(a, n, lda);
    while (off > threshold && sweeps < params.max_sweeps) {
        jacobi_sweep_c(a, n, lda, v);
        ++sweeps;
        off = jacobi_off_norm_c(a, n, lda);
    }
    *residual_out = off;
    *converged_out = off <= threshold;
    for (int i = 0; i < n; ++i) w[i] = A(i, i).real();
    if (params.sort_eig) {
        std::vector<int> order(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) order[static_cast<size_t>(i)] = i;
        std::sort(order.begin(), order.end(), [&](int l, int r2) { return w[l] < w[r2]; });
        std::vector<R> sorted_w(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) sorted_w[static_cast<size_t>(i)] = w[order[static_cast<size_t>(i)]];
        if (v) {
            std::vector<C> sorted_v(static_cast<size_t>(n) * n);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    sorted_v[i + static_cast<size_t>(j) * n] =
                        v[i + static_cast<size_t>(order[static_cast<size_t>(j)]) * n];
            std::copy(sorted_v.begin(), sorted_v.end(), v);
        }
        std::copy(sorted_w.begin(), sorted_w.end(), w);
    }
    if (v)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j) = v[i + static_cast<size_t>(j) * n];
    return sweeps;
}

template <class T>
cusolverStatus_t evj_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                        cublasFillMode_t uplo, int n, const T*, int lda, const Real<T>*,
                        int* lwork, syevjInfo_t, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || batchSize < 0 ||
        lda < std::max(1, n) || !lwork)
        return CUSOLVER_STATUS_INVALID_VALUE;
    if (static_cast<long long>(n) * n > std::numeric_limits<int>::max())
        return CUSOLVER_STATUS_INVALID_VALUE;
    // Eigenvector accumulation scratch, reused across the batch.
    *lwork = jobz == CUSOLVER_EIG_MODE_VECTOR ? std::max(1, n * n) : 1;
    return CUSOLVER_STATUS_SUCCESS;
}

// Single (batchSize == 1, single = true) and batched Jacobi eigensolver for
// any element type. A single call reports n+1 in devInfo when the tolerance
// was not reached (cuSOLVER's contract); the batched path matches the existing
// real batched entry points (sweeps executed, at least 1).
template <class T>
cusolverStatus_t evj_run(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                         cublasFillMode_t uplo, int n, T* A, int lda, Real<T>* W, T* work,
                         int lwork, int* devInfo, syevjInfo_t params, int batchSize,
                         bool single) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const long long nn = static_cast<long long>(n) * n;
    if (!valid_eig_mode(jobz) || !valid_fill(uplo) || n < 0 || batchSize < 0 || !A || !W ||
        !work || !devInfo || lda < std::max(1, n) ||
        nn > std::numeric_limits<int>::max() ||
        lwork < (jobz == CUSOLVER_EIG_MODE_VECTOR ? std::max<long long>(1, nn) : 1))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    return guarded([&] {
        const syevjInfo config = params ? *params : syevjInfo();
        double worst_residual = 0.0;
        int most_sweeps = 0;
        for (int b = 0; b < batchSize; ++b) {
            Real<T> residual = 0;
            bool converged = false;
            int sweeps = 0;
            T* a = A + static_cast<size_t>(b) * lda * n;
            Real<T>* w = W + static_cast<size_t>(b) * n;
            if constexpr (Traits<T>::kComplex) {
                sweeps = jacobi_eigen_c<Real<T>>(sp(a), n, lda, w, sp(work), jobz, uplo, config,
                                                 &residual, &converged);
            } else {
                sweeps = jacobi_eigen<T>(a, n, lda, w, work, jobz, uplo, config, &residual,
                                         &converged);
            }
            devInfo[b] = converged ? 0 : (single ? n + 1 : std::max(1, sweeps));
            worst_residual = std::max(worst_residual, static_cast<double>(residual));
            most_sweeps = std::max(most_sweeps, sweeps);
        }
        if (params) {
            params->residual = worst_residual;
            params->executed_sweeps = most_sweeps;
        }
        return CUSOLVER_STATUS_SUCCESS;
    });
}

template <class T>
cusolverStatus_t evj_single_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                               cublasFillMode_t uplo, int n, const T* A, int lda,
                               const Real<T>* W, int* lwork, syevjInfo_t params) {
    return evj_bs<T>(handle, jobz, uplo, n, A, lda, W, lwork, params, 1);
}

template <class T>
cusolverStatus_t evj_single(cusolverDnHandle_t handle, cusolverEigMode_t jobz,
                            cublasFillMode_t uplo, int n, T* A, int lda, Real<T>* W, T* work,
                            int lwork, int* devInfo, syevjInfo_t params) {
    return evj_run<T>(handle, jobz, uplo, n, A, lda, W, work, lwork, devInfo, params, 1, true);
}

}  // namespace

// ── gesvdj / gesvdjBatched / gesvdaStridedBatched ───────────────────────────
// Computed with LAPACK gesvd (singular values descending, which satisfies both
// sort_svd settings). The decomposition is internal scratch, so these report a
// workspace of one element and ignore the caller's buffer beyond that.

struct gesvdjInfo {
    double tolerance = 0.0;  // 0 = machine epsilon
    int max_sweeps = 100;
    int sort_svd = 1;
    double residual = 0.0;   // ||diag(S) - U^H A V||_F of the last run (batch: worst)
    int executed_sweeps = 0; // always 0: no Jacobi sweeps are run
};

namespace {

template <class T> struct SvdResult {
    std::vector<T> U, VT;       // U: ldu x (full ? m : k), VT: ldvt x n
    std::vector<Real<T>> S;     // k values, descending
    int ldu = 1, ldvt = 1;
};

// SVD of a dense copy of A (m,n >= 1). full selects square U and VT.
template <class T>
int svd_dense(int m, int n, const T* A, int lda, bool full, SvdResult<T>& r) {
    const int k = std::min(m, n);
    std::vector<T> Ac(static_cast<size_t>(m) * n);
    for (int j = 0; j < n; ++j)
        std::memcpy(&Ac[static_cast<size_t>(j) * m], A + static_cast<size_t>(j) * lda,
                    static_cast<size_t>(m) * sizeof(T));
    r.S.assign(static_cast<size_t>(k), Real<T>(0));
    r.ldu = m;
    r.U.assign(static_cast<size_t>(m) * (full ? m : k), T{});
    r.ldvt = full ? n : k;
    r.VT.assign(static_cast<size_t>(r.ldvt) * n, T{});
    const char job = full ? 'A' : 'S';
    std::vector<Real<T>> rwork(static_cast<size_t>(std::max(1, 5 * k)));
    T wq{};
    int info = op_gesvd<T>(job, job, m, n, Ac.data(), m, r.S.data(), r.U.data(), r.ldu,
                           r.VT.data(), r.ldvt, &wq, -1, rwork.data());
    if (info != 0) return info;
    std::vector<T> work(static_cast<size_t>(std::max(1, qint(wq))));
    return op_gesvd<T>(job, job, m, n, Ac.data(), m, r.S.data(), r.U.data(), r.ldu,
                       r.VT.data(), r.ldvt, work.data(), static_cast<int>(work.size()),
                       rwork.data());
}

// ||diag(S) - U_k^H A V_k||_F, the residual cuSOLVER's gesvdj reports.
template <class T>
double svd_offdiag_residual(int m, int n, const T* A, int lda, const SvdResult<T>& r) {
    const int k = std::min(m, n);
    std::vector<T> AV(static_cast<size_t>(m) * k), E(static_cast<size_t>(k) * k);
    gemm<T>(CblasNoTrans, CblasConjTrans, m, k, n, cval<T>(1), A, lda, r.VT.data(), r.ldvt,
            cval<T>(0), AV.data(), m);
    gemm<T>(CblasConjTrans, CblasNoTrans, k, k, m, cval<T>(1), r.U.data(), r.ldu, AV.data(), m,
            cval<T>(0), E.data(), k);
    double sum = 0.0;
    for (int j = 0; j < k; ++j)
        for (int i = 0; i < k; ++i) {
            Std<T> e = to_std<T>(E[static_cast<size_t>(i) + static_cast<size_t>(j) * k]);
            if (i == j) e -= static_cast<Real<T>>(r.S[static_cast<size_t>(i)]);
            sum += static_cast<double>(ab(e)) * static_cast<double>(ab(e));
        }
    return std::sqrt(sum);
}

// ||A - U_r diag(S_r) VT_r||_F for the leading rank triplets.
template <class T>
double svd_rank_residual(int m, int n, int rank, const T* A, int lda, const SvdResult<T>& r) {
    std::vector<T> X(static_cast<size_t>(m) * rank), C(static_cast<size_t>(m) * n);
    for (int j = 0; j < rank; ++j)
        for (int i = 0; i < m; ++i) {
            Std<T> u = to_std<T>(r.U[static_cast<size_t>(i) + static_cast<size_t>(j) * r.ldu]);
            X[static_cast<size_t>(i) + static_cast<size_t>(j) * m] =
                from_std<T>(u * r.S[static_cast<size_t>(j)]);
        }
    for (int j = 0; j < n; ++j)
        std::memcpy(&C[static_cast<size_t>(j) * m], A + static_cast<size_t>(j) * lda,
                    static_cast<size_t>(m) * sizeof(T));
    gemm<T>(CblasNoTrans, CblasNoTrans, m, n, rank, cval<T>(-1), X.data(), m, r.VT.data(),
            r.ldvt, cval<T>(1), C.data(), m);
    double sum = 0.0;
    for (const T& c : C) {
        const double a = ab(to_std<T>(c));
        sum += a * a;
    }
    return std::sqrt(sum);
}

// V (n x cols, ld ldv) from the first cols rows of VT: V = VT^H.
template <class T>
void vt_to_v(const SvdResult<T>& r, int n, int cols, T* V, int ldv) {
    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < n; ++i)
            V[static_cast<size_t>(i) + static_cast<size_t>(j) * ldv] = from_std<T>(
                cj(to_std<T>(r.VT[static_cast<size_t>(j) + static_cast<size_t>(i) * r.ldvt])));
}

template <class T>
void copy_u(const SvdResult<T>& r, int m, int cols, T* U, int ldu) {
    for (int j = 0; j < cols; ++j)
        std::memcpy(U + static_cast<size_t>(j) * ldu, &r.U[static_cast<size_t>(j) * r.ldu],
                    static_cast<size_t>(m) * sizeof(T));
}

template <class T>
cusolverStatus_t gesvdj_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
                           int n, const T*, int lda, const Real<T>*, const T*, int ldu,
                           const T*, int ldv, int* lwork, gesvdjInfo_t) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || (econ != 0 && econ != 1) || m < 0 || n < 0 ||
        lda < std::max(1, m) || !lwork ||
        (jobz == CUSOLVER_EIG_MODE_VECTOR && (ldu < std::max(1, m) || ldv < std::max(1, n))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 1;
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t gesvdj_run(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int econ, int m,
                            int n, T* A, int lda, Real<T>* S, T* U, int ldu, T* V, int ldv,
                            T* /*work*/, int lwork, int* info, gesvdjInfo_t params) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const bool vec = jobz == CUSOLVER_EIG_MODE_VECTOR;
    if (!valid_eig_mode(jobz) || (econ != 0 && econ != 1) || m < 0 || n < 0 || !A || !S ||
        !info || lda < std::max(1, m) || lwork < 1 ||
        (vec && (!U || !V || ldu < std::max(1, m) || ldv < std::max(1, n))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    if (params) { params->residual = 0.0; params->executed_sweeps = 0; }
    if (m == 0 || n == 0) return finish(0, info);
    return guarded([&] {
        const int k = std::min(m, n);
        const bool full = vec && !econ;
        SvdResult<T> r;
        const int rc = svd_dense<T>(m, n, A, lda, full, r);
        if (rc != 0) return finish(rc, info);
        std::copy(r.S.begin(), r.S.end(), S);
        if (vec) {
            copy_u<T>(r, m, full ? m : k, U, ldu);
            vt_to_v<T>(r, n, full ? n : k, V, ldv);
        }
        if (params) params->residual = svd_offdiag_residual<T>(m, n, A, lda, r);
        return finish(0, info);
    });
}

template <class T>
cusolverStatus_t gesvdj_batched_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m,
                                   int n, const T*, int lda, const Real<T>*, const T*, int ldu,
                                   const T*, int ldv, int* lwork, gesvdjInfo_t, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || m < 0 || n < 0 || batchSize < 0 || lda < std::max(1, m) ||
        !lwork ||
        (jobz == CUSOLVER_EIG_MODE_VECTOR && (ldu < std::max(1, m) || ldv < std::max(1, n))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 1;
    return CUSOLVER_STATUS_SUCCESS;
}

// Matrices are strided lda*n, S by min(m,n), U by ldu*m (m x m) and V by
// ldv*n (n x n), as in cuSOLVER's batched Jacobi SVD.
template <class T>
cusolverStatus_t gesvdj_batched(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int m,
                                int n, T* A, int lda, Real<T>* S, T* U, int ldu, T* V, int ldv,
                                T* /*work*/, int lwork, int* info, gesvdjInfo_t params,
                                int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const bool vec = jobz == CUSOLVER_EIG_MODE_VECTOR;
    if (!valid_eig_mode(jobz) || m < 0 || n < 0 || batchSize < 0 || lda < std::max(1, m) ||
        lwork < 1 || (batchSize > 0 && (!A || !S || !info)) ||
        (vec && batchSize > 0 &&
         (!U || !V || ldu < std::max(1, m) || ldv < std::max(1, n))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    return guarded([&] {
        const int k = std::min(m, n);
        double worst = 0.0;
        for (int b = 0; b < batchSize; ++b) {
            if (m == 0 || n == 0) { info[b] = 0; continue; }
            T* a = A + static_cast<size_t>(b) * lda * n;
            SvdResult<T> r;
            const int rc = svd_dense<T>(m, n, a, lda, vec, r);
            info[b] = rc > 0 ? rc : 0;
            if (rc < 0) return CUSOLVER_STATUS_INVALID_VALUE;
            if (rc != 0) continue;
            std::copy(r.S.begin(), r.S.end(), S + static_cast<size_t>(b) * k);
            if (vec) {
                copy_u<T>(r, m, m, U + static_cast<size_t>(b) * ldu * m, ldu);
                vt_to_v<T>(r, n, n, V + static_cast<size_t>(b) * ldv * n, ldv);
            }
            worst = std::max(worst, svd_offdiag_residual<T>(m, n, a, lda, r));
        }
        if (params) { params->residual = worst; params->executed_sweeps = 0; }
        return CUSOLVER_STATUS_SUCCESS;
    });
}

template <class T>
cusolverStatus_t gesvda_bs(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
                           int n, const T*, int lda, long long, const Real<T>*, long long,
                           const T*, int ldu, long long, const T*, int ldv, long long,
                           int* lwork, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!valid_eig_mode(jobz) || m < 0 || n < 0 || rank < 1 || rank > std::min(m, n) ||
        batchSize < 0 || lda < std::max(1, m) || !lwork ||
        (jobz == CUSOLVER_EIG_MODE_VECTOR && (ldu < std::max(1, m) || ldv < std::max(1, n))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork = 1;
    return CUSOLVER_STATUS_SUCCESS;
}

// Leading `rank` singular triplets of each strided matrix. U is m x rank and
// V is n x rank. h_R_nrmF (host, optional) receives ||A - U S V^H||_F per matrix,
// measured from the result, which is cuSOLVER's definition of that output.
template <class T>
cusolverStatus_t gesvda_run(cusolverDnHandle_t handle, cusolverEigMode_t jobz, int rank, int m,
                            int n, const T* A, int lda, long long strideA, Real<T>* S,
                            long long strideS, T* U, int ldu, long long strideU, T* V, int ldv,
                            long long strideV, T* /*work*/, int lwork, int* info,
                            double* h_R_nrmF, int batchSize) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    const bool vec = jobz == CUSOLVER_EIG_MODE_VECTOR;
    if (!valid_eig_mode(jobz) || m < 0 || n < 0 || rank < 1 || rank > std::min(m, n) ||
        batchSize < 0 || lda < std::max(1, m) || lwork < 1 ||
        (batchSize > 0 && (!A || !S || !info)) ||
        (batchSize > 1 && (strideA < static_cast<long long>(lda) * n || strideS < rank)) ||
        (vec && batchSize > 0 &&
         (!U || !V || ldu < std::max(1, m) || ldv < std::max(1, n) ||
          (batchSize > 1 && (strideU < static_cast<long long>(ldu) * rank ||
                             strideV < static_cast<long long>(ldv) * rank)))))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    return guarded([&] {
        for (int b = 0; b < batchSize; ++b) {
            const T* a = A + static_cast<size_t>(b) * static_cast<size_t>(strideA);
            SvdResult<T> r;
            const int rc = svd_dense<T>(m, n, a, lda, false, r);
            info[b] = rc > 0 ? rc : 0;
            if (rc < 0) return CUSOLVER_STATUS_INVALID_VALUE;
            if (rc != 0) continue;
            std::copy(r.S.begin(), r.S.begin() + rank,
                      S + static_cast<size_t>(b) * static_cast<size_t>(strideS));
            if (vec) {
                copy_u<T>(r, m, rank, U + static_cast<size_t>(b) * static_cast<size_t>(strideU),
                          ldu);
                vt_to_v<T>(r, n, rank, V + static_cast<size_t>(b) * static_cast<size_t>(strideV),
                           ldv);
            }
            if (h_R_nrmF) h_R_nrmF[b] = svd_rank_residual<T>(m, n, rank, a, lda, r);
        }
        return CUSOLVER_STATUS_SUCCESS;
    });
}

// ── IRS gesv / gels ─────────────────────────────────────────────────────────
// The second letter of the entry-point name (lowest precision) is accepted and
// ignored: the solve runs in the main precision, so there is no refinement
// loop (*iter = 0) and accuracy is that of a direct LU / QR solve.

constexpr size_t kIrsWorkspaceBytes = 64;

template <class T>
cusolverStatus_t irs_gesv_bs(cusolverDnHandle_t handle, int n, int nrhs, T*, int ldda, int*,
                             T*, int lddb, T*, int lddx, void*, size_t* lwork_bytes) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || nrhs < 0 || ldda < std::max(1, n) || lddb < std::max(1, n) ||
        lddx < std::max(1, n) || !lwork_bytes)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork_bytes = kIrsWorkspaceBytes;
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t irs_gesv(cusolverDnHandle_t handle, int n, int nrhs, T* dA, int ldda,
                          int* dipiv, T* dB, int lddb, T* dX, int lddx, void*,
                          size_t lwork_bytes, int* iter, int* d_info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (n < 0 || nrhs < 0 || ldda < std::max(1, n) || lddb < std::max(1, n) ||
        lddx < std::max(1, n) || !dA || !dB || !dX || !iter || !d_info ||
        lwork_bytes < kIrsWorkspaceBytes)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    *iter = 0;
    if (n == 0) return finish(0, d_info);
    return guarded([&] {
        std::vector<T> LU(static_cast<size_t>(n) * n);
        for (int j = 0; j < n; ++j)
            std::memcpy(&LU[static_cast<size_t>(j) * n], dA + static_cast<size_t>(j) * ldda,
                        static_cast<size_t>(n) * sizeof(T));
        std::vector<int> piv(static_cast<size_t>(n));
        int info = op_getrf<T>(n, n, LU.data(), n, piv.data());
        if (dipiv) std::copy(piv.begin(), piv.end(), dipiv);
        if (info < 0) return finish(info, d_info);
        for (int c = 0; c < nrhs; ++c)
            std::memmove(dX + static_cast<size_t>(c) * lddx, dB + static_cast<size_t>(c) * lddb,
                         static_cast<size_t>(n) * sizeof(T));
        if (info == 0 && nrhs > 0)
            info = op_getrs<T>('N', n, nrhs, LU.data(), n, piv.data(), dX, lddx);
        return finish(info, d_info);
    });
}

template <class T>
cusolverStatus_t irs_gels_bs(cusolverDnHandle_t handle, int m, int n, int nrhs, T*, int ldda,
                             T*, int lddb, T*, int lddx, void*, size_t* lwork_bytes) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || nrhs < 0 || m < n || ldda < std::max(1, m) ||
        lddb < std::max(1, m) || lddx < std::max(1, n) || !lwork_bytes)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *lwork_bytes = kIrsWorkspaceBytes;
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t irs_gels(cusolverDnHandle_t handle, int m, int n, int nrhs, T* dA, int ldda,
                          T* dB, int lddb, T* dX, int lddx, void*, size_t lwork_bytes,
                          int* iter, int* d_info) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || nrhs < 0 || m < n || ldda < std::max(1, m) ||
        lddb < std::max(1, m) || lddx < std::max(1, n) || !dA || !dB || !dX || !iter ||
        !d_info || lwork_bytes < kIrsWorkspaceBytes)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    *iter = 0;
    if (n == 0 || nrhs == 0) return finish(0, d_info);
    return guarded([&] {
        std::vector<T> Ac(static_cast<size_t>(m) * n), Bc(static_cast<size_t>(m) * nrhs);
        for (int j = 0; j < n; ++j)
            std::memcpy(&Ac[static_cast<size_t>(j) * m], dA + static_cast<size_t>(j) * ldda,
                        static_cast<size_t>(m) * sizeof(T));
        for (int j = 0; j < nrhs; ++j)
            std::memcpy(&Bc[static_cast<size_t>(j) * m], dB + static_cast<size_t>(j) * lddb,
                        static_cast<size_t>(m) * sizeof(T));
        T wq{};
        int info = op_gels<T>('N', m, n, nrhs, Ac.data(), m, Bc.data(), m, &wq, -1);
        if (info != 0) return finish(info, d_info);
        std::vector<T> work(static_cast<size_t>(std::max(1, qint(wq))));
        info = op_gels<T>('N', m, n, nrhs, Ac.data(), m, Bc.data(), m, work.data(),
                          static_cast<int>(work.size()));
        if (info == 0)
            for (int j = 0; j < nrhs; ++j)
                std::memcpy(dX + static_cast<size_t>(j) * lddx, &Bc[static_cast<size_t>(j) * m],
                            static_cast<size_t>(n) * sizeof(T));
        return finish(info, d_info);
    });
}

// ── Sparse: complex Cholesky/QR solves and shifted inverse iteration ────────

template <class T>
cusolverStatus_t sp_check_common(cusolverSpHandle_t handle, int m, int nnz,
                                 const cusparseMatDescr_t descrA, const T* csrVal,
                                 const int* csrRowPtr, const int* csrColInd, int* base,
                                 size_t* dense_elements) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || nnz < 0 || !descrA || !csrVal || !csrRowPtr || !csrColInd)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_sp_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    *base = static_cast<int>(cusparseGetMatIndexBase(descrA));
    if (!validate_csr_sp<T>(m, nnz, csrRowPtr, csrColInd, *base) ||
        !checked_dense_square_size<T>(m, dense_elements))
        return CUSOLVER_STATUS_INVALID_VALUE;
    return CUSOLVER_STATUS_SUCCESS;
}

template <class T>
cusolverStatus_t sp_lsvchol(cusolverSpHandle_t handle, int m, int nnz,
                            const cusparseMatDescr_t descrA, const T* csrVal,
                            const int* csrRowPtr, const int* csrColInd, const T* b,
                            Real<T> tol, int reorder, T* x, int* singularity) {
    if (!b || !x || !singularity || !std::isfinite(tol) || tol < 0 || reorder < 0 || reorder > 3)
        return handle ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
    int base = 0;
    size_t dense_elements = 0;
    const cusolverStatus_t st =
        sp_check_common<T>(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, &base,
                           &dense_elements);
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    if (m == 0) { *singularity = -1; return CUSOLVER_STATUS_SUCCESS; }
    return guarded([&] {
        std::vector<T> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
        const int info = op_potrf<T>('L', m, A.data(), m);
        if (info < 0) return CUSOLVER_STATUS_INTERNAL_ERROR;
        if (info > 0) { *singularity = info - 1; return CUSOLVER_STATUS_SUCCESS; }
        for (int i = 0; i < m; ++i)
            if (ab(to_std<T>(A[static_cast<size_t>(i) * m + i])) <= tol) {
                *singularity = i;
                return CUSOLVER_STATUS_SUCCESS;
            }
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(T));
        const int solve = op_potrs<T>('L', m, 1, A.data(), m, x, m);
        *singularity = -1;
        return solve == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
    });
}

template <class T>
cusolverStatus_t sp_lsvqr(cusolverSpHandle_t handle, int m, int nnz,
                          const cusparseMatDescr_t descrA, const T* csrVal,
                          const int* csrRowPtr, const int* csrColInd, const T* b,
                          Real<T> tol, int reorder, T* x, int* singularity) {
    if (!b || !x || !singularity || !std::isfinite(tol) || tol < 0 || reorder < 0 || reorder > 3)
        return handle ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
    int base = 0;
    size_t dense_elements = 0;
    const cusolverStatus_t st =
        sp_check_common<T>(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, &base,
                           &dense_elements);
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    if (m == 0) { *singularity = -1; return CUSOLVER_STATUS_SUCCESS; }
    return guarded([&] {
        std::vector<T> A(dense_elements);
        csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, A.data());
        std::memcpy(x, b, static_cast<size_t>(m) * sizeof(T));
        T wq{};
        if (op_gels<T>('N', m, m, 1, A.data(), m, x, m, &wq, -1) != 0)
            return CUSOLVER_STATUS_INTERNAL_ERROR;
        std::vector<T> work(static_cast<size_t>(std::max(1, qint(wq))));
        if (op_gels<T>('N', m, m, 1, A.data(), m, x, m, work.data(),
                       static_cast<int>(work.size())) != 0)
            return CUSOLVER_STATUS_INTERNAL_ERROR;
        *singularity = -1;
        for (int i = 0; i < m; ++i)
            if (ab(to_std<T>(A[static_cast<size_t>(i) * m + i])) <= tol) {
                *singularity = i;
                break;
            }
        return CUSOLVER_STATUS_SUCCESS;
    });
}

template <class T>
cusolverStatus_t sp_eigvsi(cusolverSpHandle_t handle, int m, int nnz,
                           const cusparseMatDescr_t descrA, const T* csrVal,
                           const int* csrRowPtr, const int* csrColInd, T mu0, const T* x0,
                           int maxite, Real<T> tol, T* mu, T* x) {
    using R = Real<T>;
    using S = Std<T>;
    if (!x0 || !mu || !x || maxite < 1 || !std::isfinite(tol) || tol < 0)
        return handle ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
    int base = 0;
    size_t dense_elements = 0;
    const cusolverStatus_t st =
        sp_check_common<T>(handle, m, nnz, descrA, csrVal, csrRowPtr, csrColInd, &base,
                           &dense_elements);
    if (st != CUSOLVER_STATUS_SUCCESS) return st;
    if (m == 0) { *mu = mu0; return CUSOLVER_STATUS_SUCCESS; }
    return guarded([&] {
        const S* val = spc(csrVal);
        auto matvec = [&](const std::vector<S>& in, std::vector<S>& out) {
            for (int i = 0; i < m; ++i) {
                S acc = S(0);
                for (int j = csrRowPtr[i] - base; j < csrRowPtr[i + 1] - base; ++j)
                    acc += val[j] * in[static_cast<size_t>(csrColInd[j] - base)];
                out[static_cast<size_t>(i)] = acc;
            }
        };
        auto norm2 = [&](const std::vector<S>& v) {
            R sum = 0;
            for (const S& e : v) sum += ab(e) * ab(e);
            return std::sqrt(sum);
        };
        std::vector<S> v(static_cast<size_t>(m)), y(static_cast<size_t>(m)),
            Av(static_cast<size_t>(m));
        for (int i = 0; i < m; ++i) v[static_cast<size_t>(i)] = to_std<T>(x0[i]);
        const R n0 = norm2(v);
        if (!(n0 > 0) || !std::isfinite(n0)) return CUSOLVER_STATUS_INVALID_VALUE;
        for (S& e : v) e /= n0;

        // Factor A - shift*I. If the shift hits an eigenvalue exactly the
        // factor is singular; nudge it, inverse iteration still converges to
        // that eigenvector from the nearby shift.
        std::vector<T> M(dense_elements);
        std::vector<int> piv(static_cast<size_t>(m));
        const S shift0 = to_std<T>(mu0);
        const R nudge = std::sqrt(std::numeric_limits<R>::epsilon()) * (R(1) + ab(shift0));
        S shift = shift0;
        int info = 1;
        for (int attempt = 0; attempt < 4 && info != 0; ++attempt) {
            csr_to_dense_sp(m, csrVal, csrRowPtr, csrColInd, base, M.data());
            S* Ms = sp(M.data());
            for (int i = 0; i < m; ++i) Ms[static_cast<size_t>(i) * m + i] -= shift;
            info = op_getrf<T>(m, m, M.data(), m, piv.data());
            if (info != 0) shift = shift0 + S(nudge * static_cast<R>(1 << (2 * attempt)));
        }
        if (info != 0) return CUSOLVER_STATUS_INTERNAL_ERROR;

        S lambda = shift0;
        for (int it = 0; it < maxite; ++it) {
            y = v;
            if (op_getrs<T>('N', m, 1, M.data(), m, piv.data(), reinterpret_cast<T*>(y.data()), m) != 0)
                return CUSOLVER_STATUS_INTERNAL_ERROR;
            const R yn = norm2(y);
            if (!(yn > 0) || !std::isfinite(yn)) return CUSOLVER_STATUS_INTERNAL_ERROR;
            for (int i = 0; i < m; ++i) v[static_cast<size_t>(i)] = y[static_cast<size_t>(i)] / yn;
            matvec(v, Av);
            S rq = S(0);
            for (int i = 0; i < m; ++i) rq += cj(v[static_cast<size_t>(i)]) * Av[static_cast<size_t>(i)];
            lambda = rq;
            R res = 0;
            for (int i = 0; i < m; ++i) {
                const R d = ab(Av[static_cast<size_t>(i)] - lambda * v[static_cast<size_t>(i)]);
                res += d * d;
            }
            if (std::sqrt(res) <= tol * std::max(R(1), ab(lambda))) break;
        }
        *mu = from_std<T>(lambda);
        for (int i = 0; i < m; ++i) x[i] = from_std<T>(v[static_cast<size_t>(i)]);
        return CUSOLVER_STATUS_SUCCESS;
    });
}

}  // namespace

// ── Exported entry points ───────────────────────────────────────────────────

extern "C" {

cusolverStatus_t cusolverDnCreateGesvdjInfo(gesvdjInfo_t* info) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    *info = new (std::nothrow) gesvdjInfo();
    if (*info == nullptr) return CUSOLVER_STATUS_ALLOC_FAILED;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
    delete info;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) {
    if (!info || !std::isfinite(tolerance) || tolerance < 0.0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    info->tolerance = tolerance;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) {
    if (!info || max_sweeps < 0) return CUSOLVER_STATUS_INVALID_VALUE;
    info->max_sweeps = max_sweeps;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_svd) {
    if (!info) return CUSOLVER_STATUS_INVALID_VALUE;
    info->sort_svd = sort_svd != 0 ? 1 : 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info,
                                              double* residual) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!info || !residual) return CUSOLVER_STATUS_INVALID_VALUE;
    *residual = info->residual;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info,
                                            int* executed_sweeps) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (!info || !executed_sweeps) return CUSOLVER_STATUS_INVALID_VALUE;
    *executed_sweeps = info->executed_sweeps;
    return CUSOLVER_STATUS_SUCCESS;
}

#define CUMETAL_DN_COMMON_IMPL(PFX, ELEM, REAL, ORG, ORM, EVJ)                                  \
    cusolverStatus_t cusolverDn##PFX##ORG##_bufferSize(                                         \
        cusolverDnHandle_t h, int m, int n, int k, const ELEM* A, int lda, const ELEM* tau,     \
        int* lwork) {                                                                           \
        return dn_orgqr_bs<ELEM>(h, m, n, k, A, lda, tau, lwork);                               \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##ORG(cusolverDnHandle_t h, int m, int n, int k, ELEM* A,   \
                                          int lda, const ELEM* tau, ELEM* work, int lwork,      \
                                          int* info) {                                          \
        return dn_orgqr<ELEM>(h, m, n, k, A, lda, tau, work, lwork, info);                      \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##ORM##_bufferSize(                                         \
        cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m, int n,     \
        int k, const ELEM* A, int lda, const ELEM* tau, const ELEM* C, int ldc, int* lwork) {   \
        return dn_ormqr_bs<ELEM>(h, side, trans, m, n, k, A, lda, tau, C, ldc, lwork);          \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##ORM(                                                      \
        cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m, int n,     \
        int k, const ELEM* A, int lda, const ELEM* tau, ELEM* C, int ldc, ELEM* work,           \
        int lwork, int* devInfo) {                                                              \
        return dn_ormqr<ELEM>(h, side, trans, m, n, k, A, lda, tau, C, ldc, work, lwork,        \
                              devInfo);                                                         \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##potrfBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, \
                                                   int n, ELEM* Aarray[], int lda,              \
                                                   int* infoArray, int batchSize) {             \
        return dn_potrf_batched<ELEM>(h, uplo, n, Aarray, lda, infoArray, batchSize);           \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##potrsBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, \
                                                   int n, int nrhs, ELEM* A[], int lda,         \
                                                   ELEM* B[], int ldb, int* d_info,             \
                                                   int batchSize) {                             \
        return dn_potrs_batched<ELEM>(h, uplo, n, nrhs, A, lda, B, ldb, d_info, batchSize);     \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##sytrf_bufferSize(cusolverDnHandle_t h, int n, ELEM* A,    \
                                                       int lda, int* lwork) {                   \
        return dn_sytrf_bs<ELEM>(h, n, A, lda, lwork);                                          \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##sytrf(cusolverDnHandle_t h, cublasFillMode_t uplo,        \
                                            int n, ELEM* A, int lda, int* ipiv, ELEM* work,     \
                                            int lwork, int* info) {                             \
        return dn_sytrf<ELEM>(h, uplo, n, A, lda, ipiv, work, lwork, info);                     \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gebrd_bufferSize(cusolverDnHandle_t h, int m, int n,      \
                                                       int* Lwork) {                            \
        return dn_gebrd_bs<ELEM>(h, m, n, Lwork);                                               \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gebrd(cusolverDnHandle_t h, int m, int n, ELEM* A,        \
                                            int lda, REAL* D, REAL* E, ELEM* TAUQ, ELEM* TAUP,  \
                                            ELEM* Work, int Lwork, int* devInfo) {              \
        return dn_gebrd<ELEM>(h, m, n, A, lda, D, E, TAUQ, TAUP, Work, Lwork, devInfo);         \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVJ##_bufferSize(                                         \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,             \
        const ELEM* A, int lda, const REAL* W, int* lwork, syevjInfo_t params) {                \
        return evj_single_bs<ELEM>(h, jobz, uplo, n, A, lda, W, lwork, params);                 \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVJ(                                                      \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, ELEM* A,    \
        int lda, REAL* W, ELEM* work, int lwork, int* info, syevjInfo_t params) {               \
        return evj_single<ELEM>(h, jobz, uplo, n, A, lda, W, work, lwork, info, params);        \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdj_bufferSize(                                        \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n, const ELEM* A,    \
        int lda, const REAL* S, const ELEM* U, int ldu, const ELEM* V, int ldv, int* lwork,     \
        gesvdjInfo_t params) {                                                                  \
        return gesvdj_bs<ELEM>(h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, lwork, params);  \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdj(                                                   \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ, int m, int n, ELEM* A,          \
        int lda, REAL* S, ELEM* U, int ldu, ELEM* V, int ldv, ELEM* work, int lwork,            \
        int* info, gesvdjInfo_t params) {                                                       \
        return gesvdj_run<ELEM>(h, jobz, econ, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,    \
                                info, params);                                                  \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdjBatched_bufferSize(                                 \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int m, int n, const ELEM* A, int lda,     \
        const REAL* S, const ELEM* U, int ldu, const ELEM* V, int ldv, int* lwork,              \
        gesvdjInfo_t params, int batchSize) {                                                   \
        return gesvdj_batched_bs<ELEM>(h, jobz, m, n, A, lda, S, U, ldu, V, ldv, lwork,         \
                                       params, batchSize);                                      \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdjBatched(                                            \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int m, int n, ELEM* A, int lda, REAL* S,  \
        ELEM* U, int ldu, ELEM* V, int ldv, ELEM* work, int lwork, int* info,                   \
        gesvdjInfo_t params, int batchSize) {                                                   \
        return gesvdj_batched<ELEM>(h, jobz, m, n, A, lda, S, U, ldu, V, ldv, work, lwork,      \
                                    info, params, batchSize);                                   \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdaStridedBatched_bufferSize(                          \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int rank, int m, int n,                   \
        const ELEM* d_A, int lda, long long int strideA, const REAL* d_S,                       \
        long long int strideS, const ELEM* d_U, int ldu, long long int strideU,                 \
        const ELEM* d_V, int ldv, long long int strideV, int* lwork, int batchSize) {           \
        return gesvda_bs<ELEM>(h, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U, ldu,  \
                               strideU, d_V, ldv, strideV, lwork, batchSize);                   \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvdaStridedBatched(                                     \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, int rank, int m, int n,                   \
        const ELEM* d_A, int lda, long long int strideA, REAL* d_S, long long int strideS,      \
        ELEM* d_U, int ldu, long long int strideU, ELEM* d_V, int ldv, long long int strideV,   \
        ELEM* d_work, int lwork, int* d_info, double* h_R_nrmF, int batchSize) {                \
        return gesvda_run<ELEM>(h, jobz, rank, m, n, d_A, lda, strideA, d_S, strideS, d_U,      \
                                ldu, strideU, d_V, ldv, strideV, d_work, lwork, d_info,         \
                                h_R_nrmF, batchSize);                                           \
    }

#define CUMETAL_DN_COMPLEX_IMPL(PFX, ELEM, REAL, EVD, EVJ)                                      \
    cusolverStatus_t cusolverDn##PFX##getrf_bufferSize(cusolverDnHandle_t h, int m, int n,      \
                                                       ELEM* A, int lda, int* Lwork) {          \
        return dn_getrf_bs<ELEM>(h, m, n, A, lda, Lwork);                                       \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##getrf(cusolverDnHandle_t h, int m, int n, ELEM* A,        \
                                            int lda, ELEM* Workspace, int* devIpiv,             \
                                            int* devInfo) {                                     \
        return dn_getrf<ELEM>(h, m, n, A, lda, Workspace, devIpiv, devInfo);                    \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##getrs(cusolverDnHandle_t h, cublasOperation_t trans,      \
                                            int n, int nrhs, const ELEM* A, int lda,            \
                                            const int* devIpiv, ELEM* B, int ldb,               \
                                            int* devInfo) {                                     \
        return dn_getrs<ELEM>(h, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo);             \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##geqrf_bufferSize(cusolverDnHandle_t h, int m, int n,      \
                                                       ELEM* A, int lda, int* Lwork) {          \
        return dn_geqrf_bs<ELEM>(h, m, n, A, lda, Lwork);                                       \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##geqrf(cusolverDnHandle_t h, int m, int n, ELEM* A,        \
                                            int lda, ELEM* TAU, ELEM* Workspace, int Lwork,     \
                                            int* devInfo) {                                     \
        return dn_geqrf<ELEM>(h, m, n, A, lda, TAU, Workspace, Lwork, devInfo);                 \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##potrf_bufferSize(cusolverDnHandle_t h,                    \
                                                       cublasFillMode_t uplo, int n, ELEM* A,   \
                                                       int lda, int* Lwork) {                   \
        return dn_potrf_bs<ELEM>(h, uplo, n, A, lda, Lwork);                                    \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##potrf(cusolverDnHandle_t h, cublasFillMode_t uplo,        \
                                            int n, ELEM* A, int lda, ELEM* Workspace,           \
                                            int Lwork, int* devInfo) {                          \
        return dn_potrf<ELEM>(h, uplo, n, A, lda, Workspace, Lwork, devInfo);                   \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##potrs(cusolverDnHandle_t h, cublasFillMode_t uplo,        \
                                            int n, int nrhs, const ELEM* A, int lda, ELEM* B,   \
                                            int ldb, int* devInfo) {                            \
        return dn_potrs<ELEM>(h, uplo, n, nrhs, A, lda, B, ldb, devInfo);                       \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvd_bufferSize(cusolverDnHandle_t h, int m, int n,      \
                                                       int* lwork) {                            \
        return dn_gesvd_bs<ELEM>(h, m, n, lwork);                                               \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##gesvd(cusolverDnHandle_t h, signed char jobu,             \
                                            signed char jobvt, int m, int n, ELEM* A, int lda,  \
                                            REAL* S, ELEM* U, int ldu, ELEM* VT, int ldvt,      \
                                            ELEM* work, int lwork, REAL* rwork, int* devInfo) { \
        return dn_gesvd<ELEM>(h, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork,   \
                              rwork, devInfo);                                                  \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVD##_bufferSize(                                         \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,             \
        const ELEM* A, int lda, const REAL* W, int* lwork) {                                    \
        return dn_heevd_bs<ELEM>(h, jobz, uplo, n, A, lda, W, lwork);                           \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVD(cusolverDnHandle_t h, cusolverEigMode_t jobz,         \
                                          cublasFillMode_t uplo, int n, ELEM* A, int lda,       \
                                          REAL* W, ELEM* work, int lwork, int* devInfo) {       \
        return dn_heevd<ELEM>(h, jobz, uplo, n, A, lda, W, work, lwork, devInfo);               \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVJ##Batched_bufferSize(                                  \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,             \
        const ELEM* A, int lda, const REAL* W, int* lwork, syevjInfo_t params,                  \
        int batchSize) {                                                                        \
        return evj_bs<ELEM>(h, jobz, uplo, n, A, lda, W, lwork, params, batchSize);             \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PFX##EVJ##Batched(                                             \
        cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, ELEM* A,    \
        int lda, REAL* W, ELEM* work, int lwork, int* info, syevjInfo_t params,                 \
        int batchSize) {                                                                        \
        return evj_run<ELEM>(h, jobz, uplo, n, A, lda, W, work, lwork, info, params,            \
                             batchSize, false);                                                 \
    }

#define CUMETAL_DN_IRS_IMPL(PP, ELEM)                                                           \
    cusolverStatus_t cusolverDn##PP##gesv_bufferSize(                                           \
        cusolverDnHandle_t h, int n, int nrhs, ELEM* dA, int ldda, int* dipiv, ELEM* dB,        \
        int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t* lwork_bytes) {                  \
        return irs_gesv_bs<ELEM>(h, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,   \
                                 lwork_bytes);                                                  \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PP##gesv(                                                      \
        cusolverDnHandle_t h, int n, int nrhs, ELEM* dA, int ldda, int* dipiv, ELEM* dB,        \
        int lddb, ELEM* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter,          \
        int* d_info) {                                                                          \
        return irs_gesv<ELEM>(h, n, nrhs, dA, ldda, dipiv, dB, lddb, dX, lddx, dWorkspace,      \
                              lwork_bytes, iter, d_info);                                       \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PP##gels_bufferSize(                                           \
        cusolverDnHandle_t h, int m, int n, int nrhs, ELEM* dA, int ldda, ELEM* dB, int lddb,   \
        ELEM* dX, int lddx, void* dWorkspace, size_t* lwork_bytes) {                            \
        return irs_gels_bs<ELEM>(h, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,       \
                                 lwork_bytes);                                                  \
    }                                                                                           \
    cusolverStatus_t cusolverDn##PP##gels(                                                      \
        cusolverDnHandle_t h, int m, int n, int nrhs, ELEM* dA, int ldda, ELEM* dB, int lddb,   \
        ELEM* dX, int lddx, void* dWorkspace, size_t lwork_bytes, int* iter, int* d_info) {     \
        return irs_gels<ELEM>(h, m, n, nrhs, dA, ldda, dB, lddb, dX, lddx, dWorkspace,          \
                              lwork_bytes, iter, d_info);                                       \
    }

CUMETAL_DN_COMMON_IMPL(S, float, float, orgqr, ormqr, syevj)
CUMETAL_DN_COMMON_IMPL(D, double, double, orgqr, ormqr, syevj)
CUMETAL_DN_COMMON_IMPL(C, cuComplex, float, ungqr, unmqr, heevj)
CUMETAL_DN_COMMON_IMPL(Z, cuDoubleComplex, double, ungqr, unmqr, heevj)
CUMETAL_DN_COMPLEX_IMPL(C, cuComplex, float, heevd, heevj)
CUMETAL_DN_COMPLEX_IMPL(Z, cuDoubleComplex, double, heevd, heevj)

CUMETAL_DN_IRS_IMPL(DD, double)
CUMETAL_DN_IRS_IMPL(DS, double)
CUMETAL_DN_IRS_IMPL(DH, double)
CUMETAL_DN_IRS_IMPL(DX, double)
CUMETAL_DN_IRS_IMPL(SS, float)
CUMETAL_DN_IRS_IMPL(SH, float)
CUMETAL_DN_IRS_IMPL(SX, float)
CUMETAL_DN_IRS_IMPL(ZZ, cuDoubleComplex)
CUMETAL_DN_IRS_IMPL(ZC, cuDoubleComplex)
CUMETAL_DN_IRS_IMPL(ZK, cuDoubleComplex)
CUMETAL_DN_IRS_IMPL(ZY, cuDoubleComplex)
CUMETAL_DN_IRS_IMPL(CC, cuComplex)
CUMETAL_DN_IRS_IMPL(CK, cuComplex)
CUMETAL_DN_IRS_IMPL(CY, cuComplex)

#undef CUMETAL_DN_COMMON_IMPL
#undef CUMETAL_DN_COMPLEX_IMPL
#undef CUMETAL_DN_IRS_IMPL

#define CUMETAL_SP_IMPL(PFX, ELEM, REAL)                                                        \
    cusolverStatus_t cusolverSp##PFX##csreigvsi(                                                \
        cusolverSpHandle_t h, int m, int nnz, const cusparseMatDescr_t descrA,                  \
        const ELEM* csrValA, const int* csrRowPtrA, const int* csrColIndA, ELEM mu0,            \
        const ELEM* x0, int maxite, REAL eps, ELEM* mu, ELEM* x) {                              \
        return sp_eigvsi<ELEM>(h, m, nnz, descrA, csrValA, csrRowPtrA, csrColIndA, mu0, x0,     \
                               maxite, eps, mu, x);                                             \
    }

CUMETAL_SP_IMPL(S, float, float)
CUMETAL_SP_IMPL(D, double, double)
CUMETAL_SP_IMPL(C, cuComplex, float)
CUMETAL_SP_IMPL(Z, cuDoubleComplex, double)
#undef CUMETAL_SP_IMPL

cusolverStatus_t cusolverSpCcsrlsvchol(cusolverSpHandle_t h, int m, int nnz,
                                        const cusparseMatDescr_t descrA, const cuComplex* val,
                                        const int* rowPtr, const int* colInd, const cuComplex* b,
                                        float tol, int reorder, cuComplex* x, int* singularity) {
    return sp_lsvchol<cuComplex>(h, m, nnz, descrA, val, rowPtr, colInd, b, tol, reorder, x,
                                 singularity);
}
cusolverStatus_t cusolverSpZcsrlsvchol(cusolverSpHandle_t h, int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const cuDoubleComplex* val, const int* rowPtr,
                                        const int* colInd, const cuDoubleComplex* b, double tol,
                                        int reorder, cuDoubleComplex* x, int* singularity) {
    return sp_lsvchol<cuDoubleComplex>(h, m, nnz, descrA, val, rowPtr, colInd, b, tol, reorder,
                                       x, singularity);
}
cusolverStatus_t cusolverSpCcsrlsvqr(cusolverSpHandle_t h, int m, int nnz,
                                      const cusparseMatDescr_t descrA, const cuComplex* val,
                                      const int* rowPtr, const int* colInd, const cuComplex* b,
                                      float tol, int reorder, cuComplex* x, int* singularity) {
    return sp_lsvqr<cuComplex>(h, m, nnz, descrA, val, rowPtr, colInd, b, tol, reorder, x,
                               singularity);
}
cusolverStatus_t cusolverSpZcsrlsvqr(cusolverSpHandle_t h, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const cuDoubleComplex* val, const int* rowPtr,
                                      const int* colInd, const cuDoubleComplex* b, double tol,
                                      int reorder, cuDoubleComplex* x, int* singularity) {
    return sp_lsvqr<cuDoubleComplex>(h, m, nnz, descrA, val, rowPtr, colInd, b, tol, reorder, x,
                                     singularity);
}

// S/D getrf/getrs share the templated implementation: NULL devIpiv means
// unpivoted LU, and info > 0 is reported through devInfo with SUCCESS.
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t h, int m, int n, float* A, int lda,
                                   float* Workspace, int* devIpiv, int* devInfo) {
    return dn_getrf<float>(h, m, n, A, lda, Workspace, devIpiv, devInfo);
}
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t h, int m, int n, double* A, int lda,
                                   double* Workspace, int* devIpiv, int* devInfo) {
    return dn_getrf<double>(h, m, n, A, lda, Workspace, devIpiv, devInfo);
}
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t h, int trans, int n, int nrhs,
                                   const float* A, int lda, const int* devIpiv, float* B,
                                   int ldb, int* devInfo) {
    if (trans < 0 || trans > 2) return h ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
    return dn_getrs<float>(h, static_cast<cublasOperation_t>(trans), n, nrhs, A, lda, devIpiv, B,
                           ldb, devInfo);
}
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t h, int trans, int n, int nrhs,
                                   const double* A, int lda, const int* devIpiv, double* B,
                                   int ldb, int* devInfo) {
    if (trans < 0 || trans > 2) return h ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
    return dn_getrs<double>(h, static_cast<cublasOperation_t>(trans), n, nrhs, A, lda, devIpiv, B,
                            ldb, devInfo);
}

}  // extern "C"
