#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include "cuda_runtime.h"

#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
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

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
                                   float* A, int lda, float* Workspace,
                                   int* devIpiv, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < std::max(1, m) || !A || !Workspace || !devInfo)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, info = 0;
    const size_t pivot_count = static_cast<size_t>(std::max(1, std::min(m, n)));
    std::unique_ptr<__CLPK_integer[]> ipiv(new (std::nothrow) __CLPK_integer[pivot_count]);
    if (!ipiv) return CUSOLVER_STATUS_ALLOC_FAILED;
    sgetrf_(&M, &N, A, &LDA, ipiv.get(), &info);
    if (devIpiv) {
        for (int i = 0; i < std::min(m, n); ++i) devIpiv[i] = static_cast<int>(ipiv[static_cast<size_t>(i)]);
    }
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                   double* A, int lda, double* Workspace,
                                   int* devIpiv, int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || lda < std::max(1, m) || !A || !Workspace || !devInfo)
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    __CLPK_integer M = m, N = n, LDA = lda, info = 0;
    const size_t pivot_count = static_cast<size_t>(std::max(1, std::min(m, n)));
    std::unique_ptr<__CLPK_integer[]> ipiv(new (std::nothrow) __CLPK_integer[pivot_count]);
    if (!ipiv) return CUSOLVER_STATUS_ALLOC_FAILED;
    dgetrf_(&M, &N, A, &LDA, ipiv.get(), &info);
    if (devIpiv) {
        for (int i = 0; i < std::min(m, n); ++i) devIpiv[i] = static_cast<int>(ipiv[static_cast<size_t>(i)]);
    }
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

// ── LU solve ─────────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const float* A, int lda,
                                   const int* devIpiv, float* B, int ldb,
                                   int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (trans < 0 || trans > 2 || n < 0 || nrhs < 0 || !A || !devIpiv || !B ||
        !devInfo || lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char t = trans == 0 ? 'N' : (trans == 1 ? 'T' : 'C');
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    std::unique_ptr<__CLPK_integer[]> ipiv(
        new (std::nothrow) __CLPK_integer[static_cast<size_t>(std::max(1, n))]);
    if (!ipiv) return CUSOLVER_STATUS_ALLOC_FAILED;
    for (int i = 0; i < n; ++i) ipiv[static_cast<size_t>(i)] = devIpiv[i];
    sgetrs_(&t, &N, &NRHS, const_cast<float*>(A), &LDA, ipiv.get(), B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const double* A, int lda,
                                   const int* devIpiv, double* B, int ldb,
                                   int* devInfo) {
    if (!handle) return CUSOLVER_STATUS_NOT_INITIALIZED;
    if (trans < 0 || trans > 2 || n < 0 || nrhs < 0 || !A || !devIpiv || !B ||
        !devInfo || lda < std::max(1, n) || ldb < std::max(1, n))
        return CUSOLVER_STATUS_INVALID_VALUE;
    const cusolverStatus_t sync = sync_stream(handle);
    if (sync != CUSOLVER_STATUS_SUCCESS) return sync;
    char t = trans == 0 ? 'N' : (trans == 1 ? 'T' : 'C');
    __CLPK_integer N = n, NRHS = nrhs, LDA = lda, LDB = ldb, info = 0;
    std::unique_ptr<__CLPK_integer[]> ipiv(
        new (std::nothrow) __CLPK_integer[static_cast<size_t>(std::max(1, n))]);
    if (!ipiv) return CUSOLVER_STATUS_ALLOC_FAILED;
    for (int i = 0; i < n; ++i) ipiv[static_cast<size_t>(i)] = devIpiv[i];
    dgetrs_(&t, &N, &NRHS, const_cast<double*>(A), &LDA, ipiv.get(), B, &LDB, &info);
    if (devInfo) *devInfo = static_cast<int>(info);
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
}

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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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

}  // extern "C"
