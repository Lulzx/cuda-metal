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
    return info == 0 ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_INTERNAL_ERROR;
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
    int sort_eig = 0;         // non-zero: ascending sort after convergence
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
