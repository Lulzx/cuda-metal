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

// Generic 64-bit syevd over the homogeneous FP32/FP64 subset, plus the
// params-handle lifecycle.
static bool test_generic_eigenvalue() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);
    cusolverDnParams_t params = nullptr;
    if (cusolverDnCreateParams(&params) != CUSOLVER_STATUS_SUCCESS || !params) {
        std::fprintf(stderr, "FAIL: cusolverDnCreateParams\n");
        return false;
    }

    size_t dev_bytes = 0, host_bytes = 1;
    double A64[] = {2.0, 1.0, 1.0, 2.0};  // eigenvalues 1, 3
    double W64[2] = {};
    if (cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                                    CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, A64, 2,
                                    CUDA_R_64F, W64, CUDA_R_64F,
                                    &dev_bytes, &host_bytes) != CUSOLVER_STATUS_SUCCESS ||
        dev_bytes != (1 + 6 * 2 + 2 * 2 * 2) * sizeof(double) ||
        host_bytes != 0) {
        std::fprintf(stderr, "FAIL: Xsyevd_bufferSize bytes=%zu host=%zu\n",
                     dev_bytes, host_bytes);
        return false;
    }
    // Heterogeneous or unsupported type combinations are rejected.
    if (cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                                    CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_32F, A64, 2,
                                    CUDA_R_64F, W64, CUDA_R_64F,
                                    &dev_bytes, &host_bytes) !=
            CUSOLVER_STATUS_INVALID_VALUE ||
        cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                                    CUBLAS_FILL_MODE_LOWER, 2, CUDA_C_64F, A64, 2,
                                    CUDA_C_64F, W64, CUDA_C_64F,
                                    &dev_bytes, &host_bytes) !=
            CUSOLVER_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: Xsyevd_bufferSize should reject mixed/unsupported types\n");
        return false;
    }
    std::vector<char> dev_work(dev_bytes);
    int info = -1;
    if (cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                         CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, A64, 2,
                         CUDA_R_64F, W64, CUDA_R_64F,
                         dev_work.data(), dev_bytes, nullptr, 0, &info) !=
            CUSOLVER_STATUS_SUCCESS ||
        info != 0 || std::fabs(W64[0] - 1.0) > 1e-9 ||
        std::fabs(W64[1] - 3.0) > 1e-9) {
        std::fprintf(stderr, "FAIL: Xsyevd eigenvalues [%g, %g] info=%d\n",
                     W64[0], W64[1], info);
        return false;
    }
    // Insufficient workspace is rejected.
    if (cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                         CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, A64, 2,
                         CUDA_R_64F, W64, CUDA_R_64F,
                         dev_work.data(), dev_bytes - 1, nullptr, 0, &info) !=
        CUSOLVER_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: Xsyevd should reject insufficient workspace\n");
        return false;
    }

    // FP32 through the same interface, padded leading dimension.
    float A32[] = {2.0f, 1.0f, -99.0f, 1.0f, 2.0f, -99.0f};  // lda=3, n=2
    float W32[2] = {};
    if (cusolverDnXsyevd_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                                    CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_32F, A32, 3,
                                    CUDA_R_32F, W32, CUDA_R_32F,
                                    &dev_bytes, &host_bytes) != CUSOLVER_STATUS_SUCCESS ||
        dev_bytes != (1 + 6 * 2 + 2 * 2 * 2) * sizeof(float)) {
        std::fprintf(stderr, "FAIL: Xsyevd FP32 bufferSize\n");
        return false;
    }
    dev_work.assign(dev_bytes, 0);
    if (cusolverDnXsyevd(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                         CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_32F, A32, 3,
                         CUDA_R_32F, W32, CUDA_R_32F,
                         dev_work.data(), dev_bytes, nullptr, 0, &info) !=
            CUSOLVER_STATUS_SUCCESS ||
        std::fabs(W32[0] - 1.0f) > 1e-4f || std::fabs(W32[1] - 3.0f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: Xsyevd FP32 eigenvalues [%g, %g]\n", W32[0], W32[1]);
        return false;
    }

    // Batched: three 2x2 FP64 matrices with a padded stride.
    constexpr int64_t kBatch = 3;
    constexpr int64_t kStrideA = 6;  // lda=3 * n=2
    constexpr int64_t kStrideW = 3;
    double Ab[kBatch * kStrideA] = {};
    double Wb[kBatch * kStrideW] = {};
    const double m0[4] = {2.0, 1.0, 1.0, 2.0};   // eig 1, 3
    const double m1[4] = {3.0, 0.0, 0.0, 1.0};   // eig 1, 3
    const double m2[4] = {5.0, 2.0, 2.0, 5.0};   // eig 3, 7
    const double* ms[kBatch] = {m0, m1, m2};
    // Column-major with lda=3: element (i,j) lands at b*strideA + i + j*3.
    for (int64_t b = 0; b < kBatch; ++b)
        for (int64_t j = 0; j < 2; ++j)
            for (int64_t i = 0; i < 2; ++i)
                Ab[b * kStrideA + i + j * 3] = ms[b][i + j * 2];
    int infob[kBatch] = {-1, -1, -1};
    if (cusolverDnXsyevBatched_bufferSize(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                                          CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, Ab, 3,
                                          kStrideA, CUDA_R_64F, Wb, kStrideW,
                                          CUDA_R_64F, kBatch,
                                          &dev_bytes, &host_bytes) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: XsyevBatched_bufferSize\n");
        return false;
    }
    dev_work.assign(dev_bytes, 0);
    if (cusolverDnXsyevBatched(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                               CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, Ab, 3,
                               kStrideA, CUDA_R_64F, Wb, kStrideW, CUDA_R_64F,
                               kBatch, dev_work.data(), dev_bytes,
                               nullptr, 0, infob) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: XsyevBatched\n");
        return false;
    }
    const double expected[kBatch][2] = {{1.0, 3.0}, {1.0, 3.0}, {3.0, 7.0}};
    for (int64_t b = 0; b < kBatch; ++b) {
        if (infob[b] != 0 || std::fabs(Wb[b * kStrideW] - expected[b][0]) > 1e-9 ||
            std::fabs(Wb[b * kStrideW + 1] - expected[b][1]) > 1e-9) {
            std::fprintf(stderr,
                         "FAIL: batch %lld eigenvalues [%g, %g] info=%d\n",
                         static_cast<long long>(b), Wb[b * kStrideW],
                         Wb[b * kStrideW + 1], infob[b]);
            return false;
        }
    }
    if (cusolverDnXsyevBatched(handle, params, CUSOLVER_EIG_MODE_NOVECTOR,
                               CUBLAS_FILL_MODE_LOWER, 2, CUDA_R_64F, Ab, 3,
                               kStrideA, CUDA_R_64F, Wb, kStrideW, CUDA_R_64F,
                               -1, dev_work.data(), dev_bytes,
                               nullptr, 0, infob) != CUSOLVER_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: XsyevBatched should reject negative batchSize\n");
        return false;
    }

    if (cusolverDnDestroyParams(params) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusolverDnDestroyParams\n");
        return false;
    }
    cusolverDnDestroy(handle);
    return true;
}

// Real Jacobi eigensolver through syevjInfo: tolerance, sweeps, sorting and
// per-matrix nonconvergence are honored.
static bool test_syevj() {
    cusolverDnHandle_t handle = nullptr;
    cusolverDnCreate(&handle);
    syevjInfo_t params = nullptr;
    if (cusolverDnCreateSyevjInfo(&params) != CUSOLVER_STATUS_SUCCESS || !params) {
        std::fprintf(stderr, "FAIL: cusolverDnCreateSyevjInfo\n");
        return false;
    }
    if (cusolverDnXsyevjSetTolerance(params, 1e-10) != CUSOLVER_STATUS_SUCCESS ||
        cusolverDnXsyevjSetMaxSweeps(params, 100) != CUSOLVER_STATUS_SUCCESS ||
        cusolverDnXsyevjSetSortEig(params, 1) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: syevj configuration\n");
        return false;
    }

    // Two 2x2 FP32 matrices packed contiguously, sorted ascending requested.
    float A[] = {5.0f, 2.0f, 2.0f, 5.0f,   // eig 3, 7
                 4.0f, -1.0f, -1.0f, 4.0f}; // eig 3, 5
    float W[4] = {};
    int devInfo[2] = {-1, -1};
    int lwork = 0;
    if (cusolverDnSsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_VECTOR,
                                           CUBLAS_FILL_MODE_LOWER, 2, A, 2, W,
                                           &lwork, params, 2) !=
            CUSOLVER_STATUS_SUCCESS ||
        lwork != 4) {
        std::fprintf(stderr, "FAIL: SsyevjBatched_bufferSize lwork=%d\n", lwork);
        return false;
    }
    std::vector<float> work(static_cast<size_t>(lwork));
    if (cusolverDnSsyevjBatched(handle, CUSOLVER_EIG_MODE_VECTOR,
                                CUBLAS_FILL_MODE_LOWER, 2, A, 2, W, work.data(),
                                lwork, devInfo, params, 2) != CUSOLVER_STATUS_SUCCESS ||
        devInfo[0] != 0 || devInfo[1] != 0) {
        std::fprintf(stderr, "FAIL: SsyevjBatched devInfo=[%d, %d]\n",
                     devInfo[0], devInfo[1]);
        return false;
    }
    if (std::fabs(W[0] - 3.0f) > 1e-4f || std::fabs(W[1] - 7.0f) > 1e-4f ||
        std::fabs(W[2] - 3.0f) > 1e-4f || std::fabs(W[3] - 5.0f) > 1e-4f) {
        std::fprintf(stderr, "FAIL: syevj eigenvalues [%g, %g, %g, %g]\n",
                     W[0], W[1], W[2], W[3]);
        return false;
    }
    // Eigenvectors overwrote A: columns of each 2x2 must be orthonormal.
    // Column-major: column 0 is (V[0], V[1]), column 1 is (V[2], V[3]).
    for (int m = 0; m < 2; ++m) {
        const float* V = A + m * 4;
        const float dot = V[0] * V[2] + V[1] * V[3];
        const float n0 = V[0] * V[0] + V[1] * V[1];
        const float n1 = V[2] * V[2] + V[3] * V[3];
        if (std::fabs(dot) > 1e-4f || std::fabs(n0 - 1.0f) > 1e-4f ||
            std::fabs(n1 - 1.0f) > 1e-4f) {
            std::fprintf(stderr, "FAIL: syevj eigenvectors not orthonormal (matrix %d)\n", m);
            return false;
        }
    }
    double residual = -1.0;
    int sweeps = -1;
    if (cusolverDnXsyevjGetResidual(handle, params, &residual) != CUSOLVER_STATUS_SUCCESS ||
        cusolverDnXsyevjGetSweeps(handle, params, &sweeps) != CUSOLVER_STATUS_SUCCESS ||
        residual < 0.0 || sweeps <= 0) {
        std::fprintf(stderr, "FAIL: syevj stats residual=%g sweeps=%d\n", residual, sweeps);
        return false;
    }

    // Nonconvergence: one sweep cannot diagonalize a general 3x3 symmetric
    // matrix, so a strict tolerance must report a positive devInfo.
    cusolverDnXsyevjSetMaxSweeps(params, 1);
    cusolverDnXsyevjSetTolerance(params, 1e-30);
    float A2[] = {4.0f, 1.0f, 2.0f,
                  1.0f, 5.0f, 3.0f,
                  2.0f, 3.0f, 6.0f};
    float W2[3] = {};
    int info2 = -1;
    if (cusolverDnSsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                CUBLAS_FILL_MODE_LOWER, 3, A2, 3, W2, work.data(),
                                static_cast<int>(work.size()), &info2, params, 1) !=
            CUSOLVER_STATUS_SUCCESS ||
        info2 <= 0) {
        std::fprintf(stderr, "FAIL: syevj nonconvergence should report devInfo > 0, got %d\n",
                     info2);
        return false;
    }

    // FP64 path.
    double A3[] = {2.0, 1.0, 1.0, 2.0};
    double W3[2] = {};
    int info3 = -1;
    int lwork3 = 0;
    cusolverDnXsyevjSetMaxSweeps(params, 100);
    cusolverDnDsyevjBatched_bufferSize(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                       CUBLAS_FILL_MODE_LOWER, 2, A3, 2, W3,
                                       &lwork3, params, 1);
    std::vector<double> work3(static_cast<size_t>(std::max(1, lwork3)));
    if (cusolverDnDsyevjBatched(handle, CUSOLVER_EIG_MODE_NOVECTOR,
                                CUBLAS_FILL_MODE_LOWER, 2, A3, 2, W3,
                                work3.data(), lwork3, &info3, params, 1) !=
            CUSOLVER_STATUS_SUCCESS ||
        info3 != 0 || std::fabs(W3[0] - 1.0) > 1e-9 ||
        std::fabs(W3[1] - 3.0) > 1e-9) {
        std::fprintf(stderr, "FAIL: DsyevjBatched eigenvalues [%g, %g] info=%d\n",
                     W3[0], W3[1], info3);
        return false;
    }

    if (cusolverDnDestroySyevjInfo(params) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusolverDnDestroySyevjInfo\n");
        return false;
    }
    cusolverDnDestroy(handle);
    return true;
}

static bool test_get_property() {
    int major = -1, minor = -1, patch = -1;
    if (cusolverGetProperty(MAJOR_VERSION, &major) != CUSOLVER_STATUS_SUCCESS ||
        cusolverGetProperty(MINOR_VERSION, &minor) != CUSOLVER_STATUS_SUCCESS ||
        cusolverGetProperty(PATCH_LEVEL, &patch) != CUSOLVER_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusolverGetProperty\n");
        return false;
    }
    // CuMetal's own version, not an NVIDIA library release: 12.x here would be
    // a false capability claim downstream code could select paths on.
    if (major >= 10) {
        std::fprintf(stderr, "FAIL: cusolverGetProperty claimed major %d\n", major);
        return false;
    }
    if (cusolverGetProperty(MAJOR_VERSION, nullptr) != CUSOLVER_STATUS_INVALID_VALUE ||
        cusolverGetProperty(static_cast<libraryPropertyType>(-1), &major) !=
            CUSOLVER_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cusolverGetProperty negative path\n");
        return false;
    }
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
    if (!test_generic_eigenvalue()) return 1;
    if (!test_syevj()) return 1;
    if (!test_get_property()) return 1;

    std::printf("PASS: cuSOLVER API tests\n");
    return 0;
}
