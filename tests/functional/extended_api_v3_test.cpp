// Functional tests for extended APIs — batch 3:
//   curandGenerateExponential/Double
//   cufftGetProperty
//   cublasSsyr2/Dsyr2, cublasSsymm/Dsymm, cublasStrmv/Dtrmv, cublasStrmm/Dtrmm
//   cublasSrot/Drot, cublasSrotg/Drotg

#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "curand.h"
#include "cufft.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg)                                             \
    do {                                                             \
        if (!(cond)) {                                               \
            std::fprintf(stderr, "FAIL: %s\n", msg);                \
            ++g_failures;                                            \
        }                                                            \
    } while (0)

// ── curand exponential ────────────────────────────────────────────────────────

static void test_curand_exponential() {
    curandGenerator_t gen = nullptr;
    CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS,
          "curandCreateGenerator for exponential test");
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    float* buf = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&buf), 64 * sizeof(float));
    CHECK(curandGenerateExponential(gen, buf, 64) == CURAND_STATUS_SUCCESS,
          "curandGenerateExponential succeeds");
    // All values must be > 0
    for (int i = 0; i < 64; ++i) {
        CHECK(buf[i] > 0.0f, "curandGenerateExponential value > 0");
    }
    cudaFree(buf);

    double* bufd = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&bufd), 32 * sizeof(double));
    CHECK(curandGenerateExponentialDouble(gen, bufd, 32) == CURAND_STATUS_SUCCESS,
          "curandGenerateExponentialDouble succeeds");
    for (int i = 0; i < 32; ++i) {
        CHECK(bufd[i] > 0.0, "curandGenerateExponentialDouble value > 0");
    }
    cudaFree(bufd);
    curandDestroyGenerator(gen);
}

// ── cufftGetProperty ──────────────────────────────────────────────────────────

static void test_cufft_get_property() {
    int major = -1, minor = -1, patch = -1;
    CHECK(cufftGetProperty(MAJOR_VERSION, &major) == CUFFT_SUCCESS,
          "cufftGetProperty MAJOR_VERSION");
    CHECK(cufftGetProperty(MINOR_VERSION, &minor) == CUFFT_SUCCESS,
          "cufftGetProperty MINOR_VERSION");
    CHECK(cufftGetProperty(PATCH_LEVEL, &patch) == CUFFT_SUCCESS,
          "cufftGetProperty PATCH_LEVEL");
    CHECK(major >= 0 && minor >= 0 && patch >= 0, "cufftGetProperty values >= 0");
    CHECK(cufftGetProperty(MAJOR_VERSION, nullptr) == CUFFT_INVALID_VALUE,
          "cufftGetProperty null returns INVALID_VALUE");
}

// ── cublasSsyr2 / Dsyr2 ───────────────────────────────────────────────────────

static void test_cublas_ssyr2() {
    // A (2x2, zero), x=[1,0], y=[0,1], alpha=1 → upper: A[0,1] += x[0]*y[1]+y[0]*x[1] = 0+0=0
    // let x=[1,2], y=[3,4], alpha=1 → upper: A[0,0] += 2*(1*3)=6, A[0,1] += 1*4+3*2=10, A[1,1] += 2*(2*4)=16
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 2;
    float a[4] = {0.0f};
    float x[2] = {1.0f, 2.0f};
    float y[2] = {3.0f, 4.0f};
    const float alpha = 1.0f;
    float* da = nullptr; float* dx = nullptr; float* dy = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dy), 2 * sizeof(float));
    cudaMemcpy(da, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, 2 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSsyr2(h, CUBLAS_FILL_MODE_UPPER, n, &alpha, dx, 1, dy, 1, da, n)
          == CUBLAS_STATUS_SUCCESS, "cublasSsyr2 succeeds");
    cudaMemcpy(a, da, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    // upper: A[0,0]=2*(1*3)=6, A[0,1]=1*4+3*2=10, A[1,1]=2*(2*4)=16
    CHECK(std::fabs(a[0] - 6.0f) < 1e-5f,  "cublasSsyr2 A[0,0]=6");
    CHECK(std::fabs(a[2] - 10.0f) < 1e-5f, "cublasSsyr2 A[0,1]=10");
    CHECK(std::fabs(a[3] - 16.0f) < 1e-5f, "cublasSsyr2 A[1,1]=16");

    cudaFree(da); cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

static void test_cublas_dsyr2() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 2;
    double a[4] = {0.0};
    double x[2] = {1.0, 0.0};
    double y[2] = {0.0, 1.0};
    const double alpha = 2.0;
    double* da = nullptr; double* dx = nullptr; double* dy = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dy), 2 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, 2 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDsyr2(h, CUBLAS_FILL_MODE_LOWER, n, &alpha, dx, 1, dy, 1, da, n)
          == CUBLAS_STATUS_SUCCESS, "cublasDsyr2 succeeds");
    cudaMemcpy(a, da, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    // lower: A[1,0] += 2*(x[1]*y[0]+y[1]*x[0]) = 2*(0+1) = 2
    CHECK(std::fabs(a[0] - 0.0) < 1e-10, "cublasDsyr2 A[0,0]=0");
    CHECK(std::fabs(a[1] - 2.0) < 1e-10, "cublasDsyr2 A[1,0]=2");

    cudaFree(da); cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

// ── cublasSsymm / Dsymm ───────────────────────────────────────────────────────

static void test_cublas_ssymm() {
    // C = alpha*A*B + beta*C, A=2x2 identity (upper), B=[[1,2],[3,4]] (2x2), alpha=1, beta=0
    // A*B = B → C = B
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int m = 2, n = 2;
    float a[4] = {1.0f, 0.0f, 0.0f, 1.0f};  // column-major identity, upper stored
    float b[4] = {1.0f, 3.0f, 2.0f, 4.0f};  // column-major [[1,2],[3,4]]
    float c[4] = {0.0f};
    const float alpha = 1.0f, beta = 0.0f;
    float* da = nullptr; float* db = nullptr; float* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&db), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(float));
    cudaMemcpy(da, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSsymm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, m, n,
                      &alpha, da, m, db, m, &beta, dc, m) == CUBLAS_STATUS_SUCCESS,
          "cublasSsymm succeeds");
    cudaMemcpy(c, dc, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    // I*B = B → c = [1,3,2,4]
    CHECK(std::fabs(c[0] - 1.0f) < 1e-5f, "cublasSsymm C[0,0]=1");
    CHECK(std::fabs(c[1] - 3.0f) < 1e-5f, "cublasSsymm C[1,0]=3");
    CHECK(std::fabs(c[2] - 2.0f) < 1e-5f, "cublasSsymm C[0,1]=2");
    CHECK(std::fabs(c[3] - 4.0f) < 1e-5f, "cublasSsymm C[1,1]=4");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cublasDestroy(h);
}

static void test_cublas_dsymm() {
    // SIDE_RIGHT: C = alpha*B*A + beta*C, A=2x2 diag(2,3) lower, B=identity, C=result
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int m = 2, n = 2;
    // A is 2x2 lower: [2,0;0,3] → a[0]=2, a[1]=0, a[2]=0, a[3]=3 (col-major)
    double a[4] = {2.0, 0.0, 0.0, 3.0};
    double b[4] = {1.0, 0.0, 0.0, 1.0};  // identity
    double c[4] = {0.0};
    const double alpha = 1.0, beta = 0.0;
    double* da = nullptr; double* db = nullptr; double* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&db), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDsymm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, m, n,
                      &alpha, da, m, db, m, &beta, dc, m) == CUBLAS_STATUS_SUCCESS,
          "cublasDsymm succeeds");
    cudaMemcpy(c, dc, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    // I*diag(2,3) = diag(2,3) → c = [2,0;0,3]
    CHECK(std::fabs(c[0] - 2.0) < 1e-10, "cublasDsymm C[0,0]=2");
    CHECK(std::fabs(c[3] - 3.0) < 1e-10, "cublasDsymm C[1,1]=3");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cublasDestroy(h);
}

// ── cublasStrmv / Dtrmv ───────────────────────────────────────────────────────

static void test_cublas_strmv() {
    // Upper triangular A = [[2,3],[0,4]], x = [1,1] → A*x = [5,4]
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 2;
    float a[4] = {2.0f, 0.0f, 3.0f, 4.0f};  // col-major: a[0]=2,a[1]=0,a[2]=3,a[3]=4
    float x[2] = {1.0f, 1.0f};
    float* da = nullptr; float* dx = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(float));
    cudaMemcpy(da, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasStrmv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                      n, da, n, dx, 1) == CUBLAS_STATUS_SUCCESS, "cublasStrmv succeeds");
    cudaMemcpy(x, dx, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(x[0] - 5.0f) < 1e-5f, "cublasStrmv x[0]=5");
    CHECK(std::fabs(x[1] - 4.0f) < 1e-5f, "cublasStrmv x[1]=4");

    cudaFree(da); cudaFree(dx);
    cublasDestroy(h);
}

static void test_cublas_dtrmv_unit() {
    // Unit lower triangular A = [[1,0],[5,1]], x = [2,3] → A*x = [2, 13]
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 2;
    double a[4] = {99.0, 5.0, 99.0, 99.0};  // col-major lower: a[1]=5 is L[1,0]; diagonal ignored
    double x[2] = {2.0, 3.0};
    double* da = nullptr; double* dx = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x, 2 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDtrmv(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                      n, da, n, dx, 1) == CUBLAS_STATUS_SUCCESS, "cublasDtrmv succeeds");
    cudaMemcpy(x, dx, 2 * sizeof(double), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(x[0] - 2.0) < 1e-10, "cublasDtrmv x[0]=2");
    CHECK(std::fabs(x[1] - 13.0) < 1e-10, "cublasDtrmv x[1]=13");

    cudaFree(da); cudaFree(dx);
    cublasDestroy(h);
}

// ── cublasStrmm / Dtrmm ───────────────────────────────────────────────────────

static void test_cublas_strmm() {
    // SIDE_LEFT, UPPER, NO_TRANS, NON_UNIT: C = alpha * A * B
    // A = [[2,3],[0,4]] (upper 2x2), B = [[1,0],[0,1]] (identity 2x2), alpha=1
    // C = A * I = A → C = [[2,3],[0,4]]
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int m = 2, n = 2;
    float a[4] = {2.0f, 0.0f, 3.0f, 4.0f};
    float b[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    float c[4] = {0.0f};
    const float alpha = 1.0f;
    float* da = nullptr; float* db = nullptr; float* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&db), 4 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(float));
    cudaMemcpy(da, a, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasStrmm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                      CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                      m, n, &alpha, da, m, db, m, dc, m) == CUBLAS_STATUS_SUCCESS,
          "cublasStrmm succeeds");
    cudaMemcpy(c, dc, 4 * sizeof(float), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(c[0] - 2.0f) < 1e-5f, "cublasStrmm C[0,0]=2");
    CHECK(std::fabs(c[1] - 0.0f) < 1e-5f, "cublasStrmm C[1,0]=0");
    CHECK(std::fabs(c[2] - 3.0f) < 1e-5f, "cublasStrmm C[0,1]=3");
    CHECK(std::fabs(c[3] - 4.0f) < 1e-5f, "cublasStrmm C[1,1]=4");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cublasDestroy(h);
}

static void test_cublas_dtrmm() {
    // SIDE_RIGHT, LOWER, NO_TRANS, UNIT: C = alpha * B * A, A unit lower
    // A = unit lower 2x2: [[1,0],[5,1]], B = [[1,0],[0,1]] (identity)
    // B * A = A → C = [[1,0],[5,1]] but with unit diag, A's diagonal is 1 not stored
    // a[1]=5 (L[1,0]): C = I * L = L → C[0,0]=1, C[1,0]=5, C[0,1]=0, C[1,1]=1
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int m = 2, n = 2;
    double a[4] = {99.0, 5.0, 99.0, 99.0};
    double b[4] = {1.0, 0.0, 0.0, 1.0};
    double c[4] = {0.0};
    const double alpha = 1.0;
    double* da = nullptr; double* db = nullptr; double* dc = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&da), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&db), 4 * sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dc), 4 * sizeof(double));
    cudaMemcpy(da, a, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, 4 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, c, 4 * sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDtrmm(h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                      CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                      m, n, &alpha, da, m, db, m, dc, m) == CUBLAS_STATUS_SUCCESS,
          "cublasDtrmm succeeds");
    cudaMemcpy(c, dc, 4 * sizeof(double), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(c[0] - 1.0) < 1e-10, "cublasDtrmm C[0,0]=1");
    CHECK(std::fabs(c[1] - 5.0) < 1e-10, "cublasDtrmm C[1,0]=5");
    CHECK(std::fabs(c[2] - 0.0) < 1e-10, "cublasDtrmm C[0,1]=0");
    CHECK(std::fabs(c[3] - 1.0) < 1e-10, "cublasDtrmm C[1,1]=1");

    cudaFree(da); cudaFree(db); cudaFree(dc);
    cublasDestroy(h);
}

// ── cublasSrot / Drot ─────────────────────────────────────────────────────────

static void test_cublas_srot() {
    // 45-degree rotation: c=s=1/sqrt(2), x=[1,0], y=[0,1]
    // After: x[0]=c*1+s*0=c, y[0]=-s*1+c*0=-s
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 2;
    float x[2] = {1.0f, 0.0f};
    float y[2] = {0.0f, 1.0f};
    const float c = 0.5f, s = 0.5f;  // not a true rotation, just testing arithmetic
    float* dx = nullptr; float* dy = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dx), 2 * sizeof(float));
    cudaMalloc(reinterpret_cast<void**>(&dy), 2 * sizeof(float));
    cudaMemcpy(dx, x, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, 2 * sizeof(float), cudaMemcpyHostToDevice);

    CHECK(cublasSrot(h, n, dx, 1, dy, 1, &c, &s) == CUBLAS_STATUS_SUCCESS,
          "cublasSrot succeeds");
    cudaMemcpy(x, dx, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dy, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    // x[0] = 0.5*1 + 0.5*0 = 0.5; y[0] = -0.5*1 + 0.5*0 = -0.5
    CHECK(std::fabs(x[0] - 0.5f) < 1e-5f, "cublasSrot x[0]=0.5");
    CHECK(std::fabs(y[0] + 0.5f) < 1e-5f, "cublasSrot y[0]=-0.5");
    // x[1] = 0.5*0 + 0.5*1 = 0.5; y[1] = -0.5*0 + 0.5*1 = 0.5
    CHECK(std::fabs(x[1] - 0.5f) < 1e-5f, "cublasSrot x[1]=0.5");
    CHECK(std::fabs(y[1] - 0.5f) < 1e-5f, "cublasSrot y[1]=0.5");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

static void test_cublas_drot() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int n = 1;
    double x[1] = {3.0};
    double y[1] = {4.0};
    // c=3/5, s=4/5 (Pythagorean rotation): x' = 9/5+16/5=5, y'=-12/5+12/5=0
    const double c = 3.0/5.0, s = 4.0/5.0;
    double* dx = nullptr; double* dy = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dx), sizeof(double));
    cudaMalloc(reinterpret_cast<void**>(&dy), sizeof(double));
    cudaMemcpy(dx, x, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, sizeof(double), cudaMemcpyHostToDevice);

    CHECK(cublasDrot(h, n, dx, 1, dy, 1, &c, &s) == CUBLAS_STATUS_SUCCESS,
          "cublasDrot succeeds");
    cudaMemcpy(x, dx, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(y, dy, sizeof(double), cudaMemcpyDeviceToHost);
    CHECK(std::fabs(x[0] - 5.0) < 1e-10, "cublasDrot x[0]=5");
    CHECK(std::fabs(y[0] - 0.0) < 1e-10, "cublasDrot y[0]=0");

    cudaFree(dx); cudaFree(dy);
    cublasDestroy(h);
}

// ── cublasSrotg / Drotg ───────────────────────────────────────────────────────

static void test_cublas_srotg() {
    // Given a=3, b=4: r=5, c=3/5, s=4/5
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    float a = 3.0f, b = 4.0f, c = 0.0f, s = 0.0f;
    CHECK(cublasSrotg(h, &a, &b, &c, &s) == CUBLAS_STATUS_SUCCESS,
          "cublasSrotg succeeds");
    CHECK(std::fabs(a - 5.0f) < 1e-4f, "cublasSrotg r=5");
    CHECK(std::fabs(c - 0.6f) < 1e-4f, "cublasSrotg c=0.6");
    CHECK(std::fabs(s - 0.8f) < 1e-4f, "cublasSrotg s=0.8");
    cublasDestroy(h);
}

static void test_cublas_drotg() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    // b=0 case: c=1, s=0, a unchanged
    double a = 7.0, b = 0.0, c = -1.0, s = -1.0;
    CHECK(cublasDrotg(h, &a, &b, &c, &s) == CUBLAS_STATUS_SUCCESS,
          "cublasDrotg succeeds");
    CHECK(std::fabs(c - 1.0) < 1e-10, "cublasDrotg c=1 when b=0");
    CHECK(std::fabs(s - 0.0) < 1e-10, "cublasDrotg s=0 when b=0");
    cublasDestroy(h);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_curand_exponential();
    test_cufft_get_property();
    test_cublas_ssyr2();
    test_cublas_dsyr2();
    test_cublas_ssymm();
    test_cublas_dsymm();
    test_cublas_strmv();
    test_cublas_dtrmv_unit();
    test_cublas_strmm();
    test_cublas_dtrmm();
    test_cublas_srot();
    test_cublas_drot();
    test_cublas_srotg();
    test_cublas_drotg();

    if (g_failures == 0) {
        std::printf("PASS: extended_api_v3 (14 sub-tests)\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d sub-test(s) failed\n", g_failures);
    return 1;
}
