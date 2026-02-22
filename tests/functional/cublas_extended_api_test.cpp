// Functional tests for the extended cuBLAS APIs added post-Phase 5:
//   cublasGemmEx, cublasHgemm, cublasSgemmBatched/DgemmBatched,
//   cublasStrsm/Dtrsm, cublasSetVector/GetVector, cublasSetMatrix/GetMatrix.
//
// All tests run on Apple Silicon UMA: host and device pointers are identical.

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cuda_fp16.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

bool near_f(float a, float b, float tol = 1e-4f) {
    return std::fabs(a - b) <= tol * (1.0f + std::fabs(b));
}

bool near_d(double a, double b, double tol = 1e-10) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }

    cublasHandle_t handle = nullptr;
    if (!expect(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate")) return 1;

    // ── cublasSetVector / cublasGetVector ────────────────────────────────────
    {
        // Create host arrays (on UMA these serve as device arrays too).
        void* d_src = nullptr;
        void* d_dst = nullptr;
        if (!expect(cudaMalloc(&d_src, 4 * sizeof(float)) == cudaSuccess, "malloc src")) return 1;
        if (!expect(cudaMalloc(&d_dst, 4 * sizeof(float)) == cudaSuccess, "malloc dst")) return 1;

        float h_src[4] = {1.f, 2.f, 3.f, 4.f};
        // SetVector: copy h_src (inc=1) → d_src (inc=1)
        if (!expect(cublasSetVector(4, sizeof(float), h_src, 1, d_src, 1) ==
                    CUBLAS_STATUS_SUCCESS, "SetVector")) return 1;
        // GetVector: copy d_src (inc=1) → d_dst (inc=1)
        if (!expect(cublasGetVector(4, sizeof(float), d_src, 1, d_dst, 1) ==
                    CUBLAS_STATUS_SUCCESS, "GetVector")) return 1;
        float* out = static_cast<float*>(d_dst);
        for (int i = 0; i < 4; ++i) {
            if (!expect(out[i] == h_src[i], "SetVector/GetVector round-trip")) return 1;
        }
        cudaFree(d_src);
        cudaFree(d_dst);
    }

    // ── cublasSetMatrix / cublasGetMatrix ────────────────────────────────────
    {
        // 3×2 column-major matrix.
        const int rows = 3, cols = 2;
        void* d_a = nullptr;
        void* d_b = nullptr;
        if (!expect(cudaMalloc(&d_a, rows * cols * sizeof(float)) == cudaSuccess, "malloc a")) return 1;
        if (!expect(cudaMalloc(&d_b, rows * cols * sizeof(float)) == cudaSuccess, "malloc b")) return 1;
        float h_a[6] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        if (!expect(cublasSetMatrix(rows, cols, sizeof(float), h_a, rows, d_a, rows) ==
                    CUBLAS_STATUS_SUCCESS, "SetMatrix")) return 1;
        if (!expect(cublasGetMatrix(rows, cols, sizeof(float), d_a, rows, d_b, rows) ==
                    CUBLAS_STATUS_SUCCESS, "GetMatrix")) return 1;
        float* out = static_cast<float*>(d_b);
        for (int i = 0; i < rows * cols; ++i) {
            if (!expect(out[i] == h_a[i], "SetMatrix/GetMatrix round-trip")) return 1;
        }
        cudaFree(d_a);
        cudaFree(d_b);
    }

    // ── cublasGemmEx (CUDA_R_32F) — should give same result as cublasSgemm ──
    // A = I (2×2), B = [[1,2],[3,4]], C = alpha * A * B + beta * C
    {
        const int M = 2, N = 2, K = 2;
        float h_a[4] = {1.f, 0.f, 0.f, 1.f};  // col-major identity
        float h_b[4] = {1.f, 3.f, 2.f, 4.f};  // col-major [[1,2],[3,4]]
        float h_c[4] = {0.f, 0.f, 0.f, 0.f};

        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        if (!expect(cudaMalloc(&d_a, 4 * sizeof(float)) == cudaSuccess, "gemmex malloc a")) return 1;
        if (!expect(cudaMalloc(&d_b, 4 * sizeof(float)) == cudaSuccess, "gemmex malloc b")) return 1;
        if (!expect(cudaMalloc(&d_c, 4 * sizeof(float)) == cudaSuccess, "gemmex malloc c")) return 1;
        std::memcpy(d_a, h_a, 4 * sizeof(float));
        std::memcpy(d_b, h_b, 4 * sizeof(float));
        std::memcpy(d_c, h_c, 4 * sizeof(float));

        const float alpha = 1.f, beta = 0.f;
        const cublasStatus_t st = cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha, d_a, CUDA_R_32F, M,
                    d_b, CUDA_R_32F, K,
            &beta,  d_c, CUDA_R_32F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "GemmEx f32 status")) return 1;

        const float* res = static_cast<const float*>(d_c);
        // I * [[1,2],[3,4]] = [[1,2],[3,4]] → col-major: [1,3,2,4]
        if (!expect(near_f(res[0], 1.f) && near_f(res[1], 3.f) &&
                    near_f(res[2], 2.f) && near_f(res[3], 4.f),
                    "GemmEx f32 result")) return 1;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    // ── cublasGemmEx (CUDA_R_64F) ────────────────────────────────────────────
    {
        const int M = 2, N = 2, K = 2;
        double h_a[4] = {1.0, 0.0, 0.0, 1.0};
        double h_b[4] = {2.0, 6.0, 4.0, 8.0};
        double h_c[4] = {0.0, 0.0, 0.0, 0.0};

        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        cudaMalloc(&d_a, 4 * sizeof(double));
        cudaMalloc(&d_b, 4 * sizeof(double));
        cudaMalloc(&d_c, 4 * sizeof(double));
        std::memcpy(d_a, h_a, 4 * sizeof(double));
        std::memcpy(d_b, h_b, 4 * sizeof(double));
        std::memcpy(d_c, h_c, 4 * sizeof(double));

        const double alpha = 1.0, beta = 0.0;
        const cublasStatus_t st = cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha, d_a, CUDA_R_64F, M,
                    d_b, CUDA_R_64F, K,
            &beta,  d_c, CUDA_R_64F, M,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "GemmEx f64 status")) return 1;

        const double* res = static_cast<const double*>(d_c);
        if (!expect(near_d(res[0], 2.0) && near_d(res[1], 6.0) &&
                    near_d(res[2], 4.0) && near_d(res[3], 8.0),
                    "GemmEx f64 result")) return 1;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    // ── cublasHgemm ──────────────────────────────────────────────────────────
    {
        const int M = 2, N = 2, K = 2;
        // A = I, B = [[1,2],[3,4]] in half precision.
        __half h_a[4], h_b[4], h_c[4];
        float fa[4] = {1.f, 0.f, 0.f, 1.f};
        float fb[4] = {1.f, 3.f, 2.f, 4.f};
        for (int i = 0; i < 4; ++i) {
            h_a[i] = static_cast<__half>(fa[i]);
            h_b[i] = static_cast<__half>(fb[i]);
            h_c[i] = static_cast<__half>(0.f);
        }

        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        cudaMalloc(&d_a, 4 * sizeof(__half));
        cudaMalloc(&d_b, 4 * sizeof(__half));
        cudaMalloc(&d_c, 4 * sizeof(__half));
        std::memcpy(d_a, h_a, 4 * sizeof(__half));
        std::memcpy(d_b, h_b, 4 * sizeof(__half));
        std::memcpy(d_c, h_c, 4 * sizeof(__half));

        const __half alpha = static_cast<__half>(1.f);
        const __half beta  = static_cast<__half>(0.f);
        const cublasStatus_t st = cublasHgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha,
            static_cast<const __half*>(d_a), M,
            static_cast<const __half*>(d_b), K,
            &beta,
            static_cast<__half*>(d_c), M);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "Hgemm status")) return 1;

        const __half* res = static_cast<const __half*>(d_c);
        // Allow a bit more tolerance for fp16.
        if (!expect(near_f(static_cast<float>(res[0]), 1.f, 1e-2f) &&
                    near_f(static_cast<float>(res[1]), 3.f, 1e-2f) &&
                    near_f(static_cast<float>(res[2]), 2.f, 1e-2f) &&
                    near_f(static_cast<float>(res[3]), 4.f, 1e-2f),
                    "Hgemm result")) return 1;

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }

    // ── cublasSgemmBatched ───────────────────────────────────────────────────
    // Two 2×2 identity × [[1,2],[3,4]] batches.
    {
        const int M = 2, N = 2, K = 2, batch = 2;
        float ha[4] = {1.f, 0.f, 0.f, 1.f};
        float hb[4] = {1.f, 3.f, 2.f, 4.f};
        float hc[4] = {0.f, 0.f, 0.f, 0.f};

        float *d_a0, *d_a1, *d_b0, *d_b1, *d_c0, *d_c1;
        cudaMalloc((void**)&d_a0, 4 * sizeof(float)); std::memcpy(d_a0, ha, 4*sizeof(float));
        cudaMalloc((void**)&d_a1, 4 * sizeof(float)); std::memcpy(d_a1, ha, 4*sizeof(float));
        cudaMalloc((void**)&d_b0, 4 * sizeof(float)); std::memcpy(d_b0, hb, 4*sizeof(float));
        cudaMalloc((void**)&d_b1, 4 * sizeof(float)); std::memcpy(d_b1, hb, 4*sizeof(float));
        cudaMalloc((void**)&d_c0, 4 * sizeof(float)); std::memcpy(d_c0, hc, 4*sizeof(float));
        cudaMalloc((void**)&d_c1, 4 * sizeof(float)); std::memcpy(d_c1, hc, 4*sizeof(float));

        const float* a_arr[2] = {d_a0, d_a1};
        const float* b_arr[2] = {d_b0, d_b1};
        float* c_arr[2] = {d_c0, d_c1};
        const float alpha = 1.f, beta = 0.f;
        const cublasStatus_t st = cublasSgemmBatched(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
            &alpha, a_arr, M, b_arr, K, &beta, c_arr, M, batch);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "SgemmBatched status")) return 1;

        for (int bi = 0; bi < batch; ++bi) {
            const float* res = c_arr[bi];
            if (!expect(near_f(res[0],1.f) && near_f(res[1],3.f) &&
                        near_f(res[2],2.f) && near_f(res[3],4.f),
                        "SgemmBatched result")) return 1;
        }
        cudaFree(d_a0); cudaFree(d_a1); cudaFree(d_b0); cudaFree(d_b1);
        cudaFree(d_c0); cudaFree(d_c1);
    }

    // ── cublasStrsm (LEFT, LOWER, N, NON_UNIT) ───────────────────────────────
    // Solve L * X = alpha * B, L = [[2,0],[3,4]], B = [[8],[11]], alpha = 1.
    // x0 = 8/2 = 4, x1 = (11 - 3*4)/4 = (11-12)/4 = -1/4 = -0.25
    {
        const int M = 2, N = 1;
        float h_a[4] = {2.f, 3.f, 0.f, 4.f};  // col-major lower triangular
        float h_b[2] = {8.f, 11.f};

        float *d_a, *d_b;
        cudaMalloc((void**)&d_a, 4 * sizeof(float)); std::memcpy(d_a, h_a, 4*sizeof(float));
        cudaMalloc((void**)&d_b, 2 * sizeof(float)); std::memcpy(d_b, h_b, 2*sizeof(float));

        const float alpha = 1.f;
        const cublasStatus_t st = cublasStrsm(
            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &alpha, d_a, M, d_b, M);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "Strsm status")) return 1;
        if (!expect(near_f(d_b[0], 4.f) && near_f(d_b[1], -0.25f), "Strsm result")) return 1;

        cudaFree(d_a); cudaFree(d_b);
    }

    // ── cublasDtrsm (LEFT, UPPER, N, NON_UNIT) ───────────────────────────────
    // Solve U * X = B, U = [[3,2],[0,5]], B = [[13],[15]], alpha = 1.
    // x1 = 15/5 = 3, x0 = (13 - 2*3)/3 = 7/3
    {
        const int M = 2, N = 1;
        double h_a[4] = {3.0, 0.0, 2.0, 5.0};  // col-major upper triangular
        double h_b[2] = {13.0, 15.0};

        double *d_a, *d_b;
        cudaMalloc((void**)&d_a, 4 * sizeof(double)); std::memcpy(d_a, h_a, 4*sizeof(double));
        cudaMalloc((void**)&d_b, 2 * sizeof(double)); std::memcpy(d_b, h_b, 2*sizeof(double));

        const double alpha = 1.0;
        const cublasStatus_t st = cublasDtrsm(
            handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N, &alpha, d_a, M, d_b, M);
        if (!expect(st == CUBLAS_STATUS_SUCCESS, "Dtrsm status")) return 1;
        if (!expect(near_d(d_b[1], 3.0) && near_d(d_b[0], 7.0/3.0), "Dtrsm result")) return 1;

        cudaFree(d_a); cudaFree(d_b);
    }

    cublasDestroy(handle);
    std::printf("PASS: cuBLAS extended API tests\n");
    return 0;
}
