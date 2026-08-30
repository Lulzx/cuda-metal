#include "cusolverDn.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static bool test_handle_lifecycle() {
    cusolverDnHandle_t handle = nullptr;
    if (cusolverDnCreate(&handle) != CUSOLVER_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cusolverDnCreate\n");
        return false;
    }
    cudaStream_t stream = nullptr;
    if (cusolverDnGetStream(handle, &stream) != CUSOLVER_STATUS_SUCCESS ||
        cusolverDnGetStream(handle, nullptr) != CUSOLVER_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cusolverDnGetStream validation\n");
        return false;
    }
    if (cusolverDnDestroy(handle) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusolverDnDestroy\n");
        return false;
    }
    return true;
}

static bool test_default_stream_ordering() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);
    float *A = nullptr, *b = nullptr, *workspace = nullptr;
    int *ipiv = nullptr, *info = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&A), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&b), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&workspace), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&ipiv), 2 * sizeof(int));
    cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int));

    const float host_A[] = {2.0f, 0.0f, 0.0f, 3.0f};
    const float host_b[] = {4.0f, 9.0f};
    cudaMemset(A, 0, 4 * sizeof(float));
    cudaMemset(b, 0, 2 * sizeof(float));
    cudaMemcpyAsync(A, host_A, sizeof(host_A), cudaMemcpyHostToDevice, nullptr);
    cudaMemcpyAsync(b, host_b, sizeof(host_b), cudaMemcpyHostToDevice, nullptr);

    const bool calls_succeeded =
        cusolverDnSgetrf(handle, 2, 2, A, 2, workspace, ipiv, info) ==
            CUSOLVER_STATUS_SUCCESS &&
        cusolverDnSgetrs(handle, 0, 2, 1, A, 2, ipiv, b, 2, info) ==
            CUSOLVER_STATUS_SUCCESS;
    float result[2] = {};
    cudaMemcpy(result, b, sizeof(result), cudaMemcpyDeviceToHost);

    cudaFree(info);
    cudaFree(ipiv);
    cudaFree(workspace);
    cudaFree(b);
    cudaFree(A);
    cusolverDnDestroy(handle);
    if (!calls_succeeded || std::fabs(result[0] - 2.0f) > 1e-4f ||
        std::fabs(result[1] - 3.0f) > 1e-4f) {
        std::fprintf(stderr,
                     "FAIL: cuSOLVER did not order default-stream inputs\n");
        return false;
    }
    return true;
}

static bool test_execution_validation() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);
    float A[4] = {2.0f, 0.0f, 0.0f, 3.0f};
    float B[2] = {4.0f, 9.0f};
    float W[2] = {};
    float U[4] = {};
    float VT[4] = {};
    float tau[2] = {};
    float work[32] = {};
    int ipiv[2] = {1, 2};
    int info = 17;

    const bool ok =
        cusolverDnSgetrf(nullptr, 2, 2, A, 2, work, ipiv, &info) ==
            CUSOLVER_STATUS_NOT_INITIALIZED &&
        cusolverDnSgetrf(handle, 2, 2, A, 2, nullptr, ipiv, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgetrf(handle, 2, 2, A, 1, work, ipiv, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgetrs(handle, 3, 2, 1, A, 2, ipiv, B, 2, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgetrs(handle, 0, 2, 1, A, 2, ipiv, B, 2, nullptr) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgeqrf(handle, 2, 2, A, 2, tau, work, 1, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgeqrf(handle, 2, 2, A, 2, nullptr, work, 2, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_FULL, 2, A, 2, work, 2,
                         &info) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, 2, A, 2, work, 1,
                         &info) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, 2, 1, A, 2, B, 1,
                         &info) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSsyevd(handle, static_cast<cusolverEigMode_t>(-1),
                         CUBLAS_FILL_MODE_LOWER, 2, A, 2, W, work, 32,
                         &info) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                         CUBLAS_FILL_MODE_LOWER, 2, A, 2, W, work, 1,
                         &info) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgesvd(handle, 'X', 'A', 2, 2, A, 2, W, U, 2, VT, 2,
                         work, 32, nullptr, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgesvd(handle, 'O', 'O', 2, 2, A, 2, W, U, 2, VT, 2,
                         work, 32, nullptr, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgesvd(handle, 'A', 'A', 2, 2, A, 2, W, U, 1, VT, 2,
                         work, 32, nullptr, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgesvd(handle, 'A', 'A', 2, 2, A, 2, W, U, 2, VT, 2,
                         work, 1, nullptr, &info) ==
            CUSOLVER_STATUS_INVALID_VALUE;

    cusolverDnDestroy(handle);
    if (!ok) {
        std::fprintf(stderr, "FAIL: cuSOLVER execution validation contract\n");
        return false;
    }
    return true;
}

static bool test_lu_factorize_solve() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    // Solve Ax = b where A = [[2,0],[0,3]] (diagonal), b = [4,9]
    // Solution: x = [2,3]
    // LAPACK column-major: A_cm = [2,0,0,3]
    float A[] = {2.0f, 0.0f, 0.0f, 3.0f};
    float b[] = {4.0f, 9.0f};
    int ipiv[2] = {};
    int info = -1;

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(handle, 2, 2, A, 2, &lwork);
    std::vector<float> workspace(static_cast<size_t>(lwork));

    cusolverStatus_t st = cusolverDnSgetrf(handle, 2, 2, A, 2, workspace.data(), ipiv, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSgetrf info=%d\n", info);
        return false;
    }

    st = cusolverDnSgetrs(handle, 0, 2, 1, A, 2, ipiv, b, 2, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSgetrs info=%d\n", info);
        return false;
    }

    if (std::fabs(b[0] - 2.0f) > 1e-4f || std::fabs(b[1] - 3.0f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: LU solve result [%f, %f] != [2, 3]\n", b[0], b[1]);
        return false;
    }

    cusolverDnDestroy(handle);
    return true;
}

static bool test_buffer_size_validation() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);
    float A[4] = {};
    float W[2] = {};
    int workspace = -1;
    const bool ok =
        cusolverDnSgetrf_bufferSize(nullptr, 2, 2, A, 2, &workspace) ==
            CUSOLVER_STATUS_NOT_INITIALIZED &&
        cusolverDnSgetrf_bufferSize(handle, -1, 2, A, 2, &workspace) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgetrf_bufferSize(handle, 2, 2, A, 1, &workspace) ==
            CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_FULL, 2, A, 2,
                                    &workspace) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSsyevd_bufferSize(handle,
                                    static_cast<cusolverEigMode_t>(-1),
                                    CUBLAS_FILL_MODE_LOWER, 2, A, 2, W,
                                    &workspace) == CUSOLVER_STATUS_INVALID_VALUE &&
        cusolverDnSgesvd_bufferSize(handle, 2, 2, nullptr) ==
            CUSOLVER_STATUS_INVALID_VALUE;
    cusolverDnDestroy(handle);
    if (!ok) {
        std::fprintf(stderr, "FAIL: cuSOLVER buffer-size validation contract\n");
        return false;
    }
    return true;
}

static bool test_cholesky() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    // Symmetric positive-definite: A = [[4,2],[2,3]] (column-major)
    // Solve Ax = b, b = [8,7] → x = [1,2] (approximately)
    // A_cm = [4,2,2,3]
    float A[] = {4.0f, 2.0f, 2.0f, 3.0f};
    float b[] = {8.0f, 7.0f};
    int info = -1;

    int lwork = 0;
    cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, 2, A, 2, &lwork);
    std::vector<float> workspace(static_cast<size_t>(lwork));

    cusolverStatus_t st = cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, 2, A, 2,
                                            workspace.data(), lwork, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSpotrf info=%d\n", info);
        return false;
    }

    st = cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, 2, 1, A, 2, b, 2, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSpotrs info=%d\n", info);
        return false;
    }

    // x should be approximately [1.25, 1.5]  (A*[1.25,1.5] = [4*1.25+2*1.5, 2*1.25+3*1.5] = [8,7])
    if (std::fabs(b[0] - 1.25f) > 1e-4f || std::fabs(b[1] - 1.5f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: Cholesky solve [%f, %f] != [1.25, 1.5]\n", b[0], b[1]);
        return false;
    }

    cusolverDnDestroy(handle);
    return true;
}

static bool test_eigenvalue() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    // Symmetric: A = [[2,1],[1,2]] → eigenvalues 1 and 3
    float A[] = {2.0f, 1.0f, 1.0f, 2.0f};
    float W[2] = {};
    int info = -1;

    int lwork = 0;
    cusolverDnSsyevd_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                 CUBLAS_FILL_MODE_LOWER, 2, A, 2, W, &lwork);
    std::vector<float> workspace(static_cast<size_t>(lwork));

    cusolverStatus_t st = cusolverDnSsyevd(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                            CUBLAS_FILL_MODE_LOWER, 2, A, 2,
                                            W, workspace.data(), lwork, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSsyevd info=%d\n", info);
        return false;
    }

    // Eigenvalues sorted ascending: 1.0, 3.0
    if (std::fabs(W[0] - 1.0f) > 1e-4f || std::fabs(W[1] - 3.0f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: eigenvalues [%f, %f] != [1, 3]\n", W[0], W[1]);
        return false;
    }

    cusolverDnDestroy(handle);
    return true;
}

static bool test_svd() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);

    // A = [[3,0],[0,4]] → singular values 4, 3 (sorted descending)
    // Column-major: [3, 0, 0, 4]
    float A[] = {3.0f, 0.0f, 0.0f, 4.0f};
    float S[2] = {};
    float U[4] = {};
    float VT[4] = {};
    int info = -1;

    int lwork = 0;
    cusolverDnSgesvd_bufferSize(handle, 2, 2, &lwork);
    std::vector<float> workspace(static_cast<size_t>(lwork));

    cusolverStatus_t st = cusolverDnSgesvd(handle, 'A', 'A', 2, 2, A, 2,
                                            S, U, 2, VT, 2,
                                            workspace.data(), lwork, nullptr, &info);
    if (st != CUSOLVER_STATUS_SUCCESS || info != 0) {
        std::fprintf(stderr, "FAIL: cusolverDnSgesvd info=%d\n", info);
        return false;
    }

    // Singular values: 4 and 3 (descending)
    if (std::fabs(S[0] - 4.0f) > 1e-4f || std::fabs(S[1] - 3.0f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: singular values [%f, %f] != [4, 3]\n", S[0], S[1]);
        return false;
    }

    cusolverDnDestroy(handle);
    return true;
}

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_lu_factorize_solve()) return 1;
    if (!test_buffer_size_validation()) return 1;
    if (!test_execution_validation()) return 1;
    if (!test_default_stream_ordering()) return 1;
    if (!test_cholesky()) return 1;
    if (!test_eigenvalue()) return 1;
    if (!test_svd()) return 1;

    std::printf("PASS: cuSOLVER API tests\n");
    return 0;
}
