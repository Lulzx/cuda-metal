// Functional test for the cuBLAS entry points CuPy 14.2 needs on top of the base
// set: complex level-1/2/3, geam/dgmm, banded/packed real helpers, batched LU
// helpers (getrf/getrs/getri), complex trsm/gemm batched, and SgemmEx.
//
// Every result is checked against a naive column-major reference written here.
// Apple Silicon UMA: device pointers are host-readable, so inputs are filled and
// outputs inspected directly.

#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", #cond, __LINE__);       \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

// ── scalar traits so one test body covers float/double/complex ──

template <class T>
struct Tr;

template <>
struct Tr<float> {
    using R = float;
    static constexpr bool is_complex = false;
    static float make(double r, double) { return static_cast<float>(r); }
    static float mul(float a, float b) { return a * b; }
    static float add(float a, float b) { return a + b; }
    static float sub(float a, float b) { return a - b; }
    static float conj(float a) { return a; }
    static double re(float a) { return a; }
    static double im(float) { return 0; }
    static double tol() { return 2e-4; }
};
template <>
struct Tr<double> {
    using R = double;
    static constexpr bool is_complex = false;
    static double make(double r, double) { return r; }
    static double mul(double a, double b) { return a * b; }
    static double add(double a, double b) { return a + b; }
    static double sub(double a, double b) { return a - b; }
    static double conj(double a) { return a; }
    static double re(double a) { return a; }
    static double im(double) { return 0; }
    static double tol() { return 1e-10; }
};
template <>
struct Tr<cuComplex> {
    using R = float;
    static constexpr bool is_complex = true;
    static cuComplex make(double r, double i) {
        return cuComplex{static_cast<float>(r), static_cast<float>(i)};
    }
    static cuComplex mul(cuComplex a, cuComplex b) {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
    static cuComplex add(cuComplex a, cuComplex b) { return {a.x + b.x, a.y + b.y}; }
    static cuComplex sub(cuComplex a, cuComplex b) { return {a.x - b.x, a.y - b.y}; }
    static cuComplex conj(cuComplex a) { return {a.x, -a.y}; }
    static double re(cuComplex a) { return a.x; }
    static double im(cuComplex a) { return a.y; }
    static double tol() { return 5e-4; }
};
template <>
struct Tr<cuDoubleComplex> {
    using R = double;
    static constexpr bool is_complex = true;
    static cuDoubleComplex make(double r, double i) { return {r, i}; }
    static cuDoubleComplex mul(cuDoubleComplex a, cuDoubleComplex b) {
        return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
    }
    static cuDoubleComplex add(cuDoubleComplex a, cuDoubleComplex b) { return {a.x + b.x, a.y + b.y}; }
    static cuDoubleComplex sub(cuDoubleComplex a, cuDoubleComplex b) { return {a.x - b.x, a.y - b.y}; }
    static cuDoubleComplex conj(cuDoubleComplex a) { return {a.x, -a.y}; }
    static double re(cuDoubleComplex a) { return a.x; }
    static double im(cuDoubleComplex a) { return a.y; }
    static double tol() { return 1e-10; }
};

template <class T>
bool close(T a, T b, double tol = Tr<T>::tol()) {
    const double dr = Tr<T>::re(a) - Tr<T>::re(b);
    const double di = Tr<T>::im(a) - Tr<T>::im(b);
    const double mag = 1.0 + std::fabs(Tr<T>::re(b)) + std::fabs(Tr<T>::im(b));
    return std::sqrt(dr * dr + di * di) <= tol * mag;
}

bool close_r(double a, double b, double tol) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

unsigned g_seed = 12345u;
double rnd() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return (static_cast<double>((g_seed >> 8) & 0xffff) / 65535.0) * 2.0 - 1.0;
}
template <class T>
T rnd_t() {
    return Tr<T>::make(rnd(), Tr<T>::is_complex ? rnd() : 0.0);
}

template <class T>
T* dalloc(std::size_t n) {
    void* p = nullptr;
    if (cudaMalloc(&p, (n ? n : 1) * sizeof(T)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        std::exit(1);
    }
    return static_cast<T*>(p);
}
template <class T>
T* dfill(const std::vector<T>& v) {
    T* p = dalloc<T>(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return p;
}
template <class T>
std::vector<T> rvec(std::size_t n) {
    std::vector<T> v(n);
    for (auto& e : v) e = rnd_t<T>();
    return v;
}

// op(A)(i,j) of a column-major matrix.
template <class T>
T opel(const std::vector<T>& A, int ld, cublasOperation_t op, int i, int j) {
    if (op == CUBLAS_OP_N) return A[static_cast<std::size_t>(j) * ld + i];
    const T v = A[static_cast<std::size_t>(i) * ld + j];
    return op == CUBLAS_OP_C ? Tr<T>::conj(v) : v;
}

// Effective triangular matrix entry honoring uplo/diag.
template <class T>
T tri(const std::vector<T>& A, int ld, cublasFillMode_t uplo, cublasDiagType_t diag, int i, int j) {
    if (i == j) return diag == CUBLAS_DIAG_UNIT ? Tr<T>::make(1, 0) : A[static_cast<std::size_t>(j) * ld + i];
    const bool stored = (uplo == CUBLAS_FILL_MODE_UPPER) ? (i < j) : (i > j);
    return stored ? A[static_cast<std::size_t>(j) * ld + i] : Tr<T>::make(0, 0);
}

// ── API tables ──

template <class T, class R>
struct ComplexApi {
    cublasStatus_t (*axpy)(cublasHandle_t, int, const T*, const T*, int, T*, int);
    cublasStatus_t (*scal)(cublasHandle_t, int, const T*, T*, int);
    cublasStatus_t (*rscal)(cublasHandle_t, int, const R*, T*, int);
    cublasStatus_t (*dotu)(cublasHandle_t, int, const T*, int, const T*, int, T*);
    cublasStatus_t (*dotc)(cublasHandle_t, int, const T*, int, const T*, int, T*);
    cublasStatus_t (*iamax)(cublasHandle_t, int, const T*, int, int*);
    cublasStatus_t (*iamin)(cublasHandle_t, int, const T*, int, int*);
    cublasStatus_t (*asum)(cublasHandle_t, int, const T*, int, R*);
    cublasStatus_t (*nrm2)(cublasHandle_t, int, const T*, int, R*);
    cublasStatus_t (*geru)(cublasHandle_t, int, int, const T*, const T*, int, const T*, int, T*, int);
    cublasStatus_t (*gerc)(cublasHandle_t, int, int, const T*, const T*, int, const T*, int, T*, int);
    cublasStatus_t (*syrk)(cublasHandle_t, cublasFillMode_t, cublasOperation_t, int, int, const T*,
                           const T*, int, const T*, T*, int);
    cublasStatus_t (*trsm)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                           cublasDiagType_t, int, int, const T*, const T*, int, T*, int);
    cublasStatus_t (*trsmB)(cublasHandle_t, cublasSideMode_t, cublasFillMode_t, cublasOperation_t,
                            cublasDiagType_t, int, int, const T*, const T* const[], int, T* const[],
                            int, int);
    cublasStatus_t (*gemmB)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int,
                            const T*, const T* const[], int, const T* const[], int, const T*,
                            T* const[], int, int);
};

template <class T>
struct CommonApi {
    cublasStatus_t (*geam)(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, const T*,
                           const T*, int, const T*, const T*, int, T*, int);
    cublasStatus_t (*dgmm)(cublasHandle_t, cublasSideMode_t, int, int, const T*, int, const T*, int,
                           T*, int);
    cublasStatus_t (*getrfB)(cublasHandle_t, int, T* const[], int, int*, int*, int);
    cublasStatus_t (*getrsB)(cublasHandle_t, cublasOperation_t, int, int, const T* const[], int,
                             const int*, T* const[], int, int*, int);
    cublasStatus_t (*getriB)(cublasHandle_t, int, const T* const[], int, const int*, T* const[], int,
                             int*, int);
};

// ── complex level-1/2/3 ──

template <class T, class R>
void test_complex(cublasHandle_t h, const ComplexApi<T, R>& api, const char* tag) {
    using P = Tr<T>;
    std::printf("complex %s\n", tag);

    // Negative paths.
    {
        T* x = dalloc<T>(4);
        T al = P::make(1, 0);
        CHECK(api.axpy(nullptr, 4, &al, x, 1, x, 1) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.axpy(h, -1, &al, x, 1, x, 1) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.scal(nullptr, 4, &al, x, 1) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.scal(h, -1, &al, x, 1) == CUBLAS_STATUS_INVALID_VALUE);
        T res;
        CHECK(api.dotu(nullptr, 4, x, 1, x, 1, &res) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.dotc(h, -2, x, 1, x, 1, &res) == CUBLAS_STATUS_INVALID_VALUE);
        int idx;
        R rr;
        CHECK(api.iamax(nullptr, 4, x, 1, &idx) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.iamin(h, -1, x, 1, &idx) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.asum(nullptr, 4, x, 1, &rr) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.nrm2(h, -1, x, 1, &rr) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.rscal(h, -1, &rr, x, 1) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.geru(nullptr, 2, 2, &al, x, 1, x, 1, x, 2) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.gerc(h, -1, 2, &al, x, 1, x, 1, x, 2) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.syrk(nullptr, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 2, 2, &al, x, 2, &al, x, 2) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.syrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_C, 2, 2, &al, x, 2, &al, x, 2) ==
              CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.trsm(nullptr, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                       CUBLAS_DIAG_UNIT, 2, 2, &al, x, 2, x, 2) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.trsm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                       -2, 2, &al, x, 2, x, 2) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.trsmB(nullptr, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                        CUBLAS_DIAG_UNIT, 2, 2, &al, nullptr, 2, nullptr, 2, 1) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.trsmB(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                        2, 2, &al, nullptr, 2, nullptr, 2, -1) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.gemmB(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2, &al, nullptr, 2, nullptr, 2, &al,
                        nullptr, 2, 1) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.gemmB(h, CUBLAS_OP_N, CUBLAS_OP_N, -2, 2, 2, &al, nullptr, 2, nullptr, 2, &al,
                        nullptr, 2, 1) == CUBLAS_STATUS_INVALID_VALUE);
        cudaFree(x);
    }

    const int n = 7;
    // axpy / scal / rscal with strides.
    {
        auto xv = rvec<T>(2 * n), yv = rvec<T>(3 * n);
        T alpha = P::make(0.5, -0.25);
        T* x = dfill(xv);
        T* y = dfill(yv);
        CHECK(api.axpy(h, n, &alpha, x, 2, y, 3) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            ok &= close(y[3 * i], P::add(yv[3 * i], P::mul(alpha, xv[2 * i])));
            ok &= close(y[3 * i + 1], yv[3 * i + 1]);  // untouched gap
        }
        CHECK(ok);
        CHECK(api.scal(h, n, &alpha, x, 2) == CUBLAS_STATUS_SUCCESS);
        ok = true;
        for (int i = 0; i < n; ++i) ok &= close(x[2 * i], P::mul(alpha, xv[2 * i]));
        CHECK(ok);
        // Real scalar variant.
        x[0] = xv[0];
        R rs = static_cast<R>(-3);
        CHECK(api.rscal(h, n, &rs, x, 2) == CUBLAS_STATUS_SUCCESS);
        CHECK(close(x[0], P::make(-3 * P::re(xv[0]), -3 * P::im(xv[0]))));
        cudaFree(x);
        cudaFree(y);
    }
    // dotu / dotc / asum / nrm2 / iamax / iamin, host and device pointer mode.
    {
        auto xv = rvec<T>(n), yv = rvec<T>(n);
        xv[3] = P::make(4.0, -3.0);  // |re|+|im| = 7, the largest
        xv[5] = P::make(0.001, 0.001);  // the smallest
        T* x = dfill(xv);
        T* y = dfill(yv);
        T eu = P::make(0, 0), ec = P::make(0, 0);
        double asum = 0, ssq = 0;
        for (int i = 0; i < n; ++i) {
            eu = P::add(eu, P::mul(xv[i], yv[i]));
            ec = P::add(ec, P::mul(P::conj(xv[i]), yv[i]));
            asum += std::fabs(P::re(xv[i])) + std::fabs(P::im(xv[i]));
            ssq += P::re(xv[i]) * P::re(xv[i]) + P::im(xv[i]) * P::im(xv[i]);
        }
        T r = P::make(0, 0);
        CHECK(api.dotu(h, n, x, 1, y, 1, &r) == CUBLAS_STATUS_SUCCESS);
        CHECK(close(r, eu));
        CHECK(api.dotc(h, n, x, 1, y, 1, &r) == CUBLAS_STATUS_SUCCESS);
        CHECK(close(r, ec));
        R a = 0, nr = 0;
        CHECK(api.asum(h, n, x, 1, &a) == CUBLAS_STATUS_SUCCESS);
        CHECK(close_r(a, asum, P::tol()));
        CHECK(api.nrm2(h, n, x, 1, &nr) == CUBLAS_STATUS_SUCCESS);
        CHECK(close_r(nr, std::sqrt(ssq), P::tol()));
        int imax = 0, imin = 0;
        CHECK(api.iamax(h, n, x, 1, &imax) == CUBLAS_STATUS_SUCCESS);
        CHECK(imax == 4);  // 1-based
        CHECK(api.iamin(h, n, x, 1, &imin) == CUBLAS_STATUS_SUCCESS);
        CHECK(imin == 6);
        // n == 0 results.
        CHECK(api.dotc(h, 0, x, 1, y, 1, &r) == CUBLAS_STATUS_SUCCESS && close(r, P::make(0, 0)));
        CHECK(api.iamax(h, 0, x, 1, &imax) == CUBLAS_STATUS_SUCCESS && imax == 0);

        // Device pointer mode: scalars live in device memory.
        CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE) == CUBLAS_STATUS_SUCCESS);
        T* dres = dalloc<T>(1);
        R* dr = dalloc<R>(1);
        int* di = dalloc<int>(1);
        CHECK(api.dotc(h, n, x, 1, y, 1, dres) == CUBLAS_STATUS_SUCCESS);
        CHECK(close(dres[0], ec));
        CHECK(api.nrm2(h, n, x, 1, dr) == CUBLAS_STATUS_SUCCESS);
        CHECK(close_r(dr[0], std::sqrt(ssq), P::tol()));
        CHECK(api.iamax(h, n, x, 1, di) == CUBLAS_STATUS_SUCCESS && di[0] == 4);
        T* dalpha = dfill(std::vector<T>{P::make(2, 1)});
        auto y0 = std::vector<T>(yv);
        CHECK(api.axpy(h, n, dalpha, x, 1, y, 1) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int i = 0; i < n; ++i) ok &= close(y[i], P::add(y0[i], P::mul(P::make(2, 1), xv[i])));
        CHECK(ok);
        CHECK(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST) == CUBLAS_STATUS_SUCCESS);
        cudaFree(dres); cudaFree(dr); cudaFree(di); cudaFree(dalpha); cudaFree(x); cudaFree(y);
    }
    // geru / gerc.
    {
        const int m = 4, nn = 3, lda = 5;
        auto xv = rvec<T>(2 * m), yv = rvec<T>(nn), Av = rvec<T>(lda * nn);
        T alpha = P::make(1.5, 0.5);
        for (int variant = 0; variant < 2; ++variant) {
            T* x = dfill(xv);
            T* y = dfill(yv);
            T* A = dfill(Av);
            auto call = variant == 0 ? api.geru : api.gerc;
            CHECK(call(h, m, nn, &alpha, x, 2, y, 1, A, lda) == CUBLAS_STATUS_SUCCESS);
            bool ok = true;
            for (int j = 0; j < nn; ++j)
                for (int i = 0; i < m; ++i) {
                    const T yj = variant == 0 ? yv[j] : P::conj(yv[j]);
                    ok &= close(A[j * lda + i],
                                P::add(Av[j * lda + i], P::mul(alpha, P::mul(xv[2 * i], yj))));
                }
            CHECK(ok);
            cudaFree(x); cudaFree(y); cudaFree(A);
        }
    }
    // syrk, both triangles and both ops; the other triangle must stay untouched.
    for (int variant = 0; variant < 2; ++variant) {
        const int nn = 4, k = 3;
        const cublasOperation_t op = variant == 0 ? CUBLAS_OP_N : CUBLAS_OP_T;
        const cublasFillMode_t uplo = variant == 0 ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        const int lda = (op == CUBLAS_OP_N) ? nn + 1 : k + 2;
        const int ac = (op == CUBLAS_OP_N) ? k : nn;
        auto Av = rvec<T>(static_cast<std::size_t>(lda) * ac), Cv = rvec<T>(nn * nn);
        T alpha = P::make(0.5, 0.25), beta = P::make(-1.0, 0.5);
        T* A = dfill(Av);
        T* C = dfill(Cv);
        CHECK(api.syrk(h, uplo, op, nn, k, &alpha, A, lda, &beta, C, nn) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int j = 0; j < nn; ++j)
            for (int i = 0; i < nn; ++i) {
                const bool in_tri = (uplo == CUBLAS_FILL_MODE_UPPER) ? (i <= j) : (i >= j);
                if (!in_tri) {
                    ok &= close(C[j * nn + i], Cv[j * nn + i]);
                    continue;
                }
                T s = P::make(0, 0);
                for (int p = 0; p < k; ++p) {
                    const T a_ip = (op == CUBLAS_OP_N) ? Av[p * lda + i] : Av[i * lda + p];
                    const T a_jp = (op == CUBLAS_OP_N) ? Av[p * lda + j] : Av[j * lda + p];
                    s = P::add(s, P::mul(a_ip, a_jp));  // transpose, not conjugate
                }
                ok &= close(C[j * nn + i], P::add(P::mul(alpha, s), P::mul(beta, Cv[j * nn + i])));
            }
        CHECK(ok);
        cudaFree(A); cudaFree(C);
    }
    // trsm (left/lower/conj, right/upper/unit) and trsmBatched.
    for (int variant = 0; variant < 3; ++variant) {
        const bool left = variant != 1;
        const int m = 4, nn = 3, adim = left ? m : nn;
        const cublasFillMode_t uplo = left ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        const cublasOperation_t op = left ? CUBLAS_OP_C : CUBLAS_OP_N;
        const cublasDiagType_t diag = left ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;
        const int lda = adim + 1, ldb = m + 1;
        auto Av = rvec<T>(static_cast<std::size_t>(lda) * adim), Bv = rvec<T>(static_cast<std::size_t>(ldb) * nn);
        for (int i = 0; i < adim; ++i) Av[i * lda + i] = P::make(2.0 + 0.1 * i, 0.3);  // well conditioned
        T alpha = P::make(0.75, -0.5);
        const int batch = variant == 2 ? 2 : 1;
        std::vector<T*> As, Bs;
        for (int b = 0; b < batch; ++b) {
            As.push_back(dfill(Av));
            Bs.push_back(dfill(Bv));
        }
        cublasStatus_t st;
        if (variant == 2) {
            T** at = reinterpret_cast<T**>(dalloc<void*>(batch));
            T** bt = reinterpret_cast<T**>(dalloc<void*>(batch));
            for (int b = 0; b < batch; ++b) { at[b] = As[b]; bt[b] = Bs[b]; }
            st = api.trsmB(h, CUBLAS_SIDE_LEFT, uplo, op, diag, m, nn, &alpha, at, lda, bt, ldb, batch);
            cudaFree(at); cudaFree(bt);
        } else {
            st = api.trsm(h, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, uplo, op, diag, m, nn,
                          &alpha, As[0], lda, Bs[0], ldb);
        }
        CHECK(st == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        const bool left_eff = variant != 1;
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < nn; ++j)
                for (int i = 0; i < m; ++i) {
                    T s = P::make(0, 0);
                    if (left_eff) {  // op(A) X
                        for (int p = 0; p < m; ++p) {
                            T aip = P::conj(tri(Av, lda, uplo, diag, p, i));  // op=C on (i,p) -> conj(A(p,i))
                            s = P::add(s, P::mul(aip, Bs[b][j * ldb + p]));
                        }
                    } else {  // X op(A)
                        for (int p = 0; p < nn; ++p)
                            s = P::add(s, P::mul(Bs[b][p * ldb + i], tri(Av, lda, uplo, diag, p, j)));
                    }
                    ok &= close(s, P::mul(alpha, Bv[j * ldb + i]), 5e-3);
                }
        }
        CHECK(ok);
        for (auto p : As) cudaFree(p);
        for (auto p : Bs) cudaFree(p);
    }
    // gemmBatched.
    {
        const int m = 2, nn = 3, k = 4, batch = 3;
        const cublasOperation_t ta = CUBLAS_OP_C, tb = CUBLAS_OP_T;
        const int lda = k + 1, ldb = nn + 1, ldc = m + 1;  // A is k x m (op C), B is n x k (op T)
        std::vector<std::vector<T>> Av, Bv, Cv;
        std::vector<T*> A, B, C;
        for (int b = 0; b < batch; ++b) {
            Av.push_back(rvec<T>(static_cast<std::size_t>(lda) * m));
            Bv.push_back(rvec<T>(static_cast<std::size_t>(ldb) * k));
            Cv.push_back(rvec<T>(static_cast<std::size_t>(ldc) * nn));
            A.push_back(dfill(Av.back()));
            B.push_back(dfill(Bv.back()));
            C.push_back(dfill(Cv.back()));
        }
        T** at = reinterpret_cast<T**>(dalloc<void*>(batch));
        T** bt = reinterpret_cast<T**>(dalloc<void*>(batch));
        T** ct = reinterpret_cast<T**>(dalloc<void*>(batch));
        for (int b = 0; b < batch; ++b) { at[b] = A[b]; bt[b] = B[b]; ct[b] = C[b]; }
        T alpha = P::make(1.0, 0.5), beta = P::make(0.5, 0);
        CHECK(api.gemmB(h, ta, tb, m, nn, k, &alpha, at, lda, bt, ldb, &beta, ct, ldc, batch) ==
              CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int b = 0; b < batch; ++b)
            for (int j = 0; j < nn; ++j)
                for (int i = 0; i < m; ++i) {
                    T s = P::make(0, 0);
                    for (int p = 0; p < k; ++p)
                        s = P::add(s, P::mul(opel(Av[b], lda, ta, i, p), opel(Bv[b], ldb, tb, p, j)));
                    ok &= close(C[b][j * ldc + i],
                                P::add(P::mul(alpha, s), P::mul(beta, Cv[b][j * ldc + i])));
                }
        CHECK(ok);
        for (auto p : A) cudaFree(p);
        for (auto p : B) cudaFree(p);
        for (auto p : C) cudaFree(p);
        cudaFree(at); cudaFree(bt); cudaFree(ct);
    }
}

// ── geam / dgmm / batched LU, for all four element types ──

template <class T>
void test_common(cublasHandle_t h, const CommonApi<T>& api, const char* tag) {
    using P = Tr<T>;
    std::printf("common %s\n", tag);
    T one = P::make(1, 0);

    // Negative paths.
    {
        T* x = dalloc<T>(16);
        int* info = dalloc<int>(4);
        CHECK(api.geam(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, &one, x, 2, &one, x, 2, x, 2) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.geam(h, CUBLAS_OP_N, CUBLAS_OP_N, -1, 2, &one, x, 2, &one, x, 2, x, 2) ==
              CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.dgmm(nullptr, CUBLAS_SIDE_LEFT, 2, 2, x, 2, x, 1, x, 2) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.dgmm(h, CUBLAS_SIDE_LEFT, -2, 2, x, 2, x, 1, x, 2) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.getrfB(nullptr, 2, nullptr, 2, nullptr, info, 1) == CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.getrfB(h, -2, nullptr, 2, nullptr, info, 1) == CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.getrsB(nullptr, CUBLAS_OP_N, 2, 1, nullptr, 2, nullptr, nullptr, 2, info, 1) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.getrsB(h, CUBLAS_OP_N, -2, 1, nullptr, 2, nullptr, nullptr, 2, info, 1) ==
              CUBLAS_STATUS_INVALID_VALUE);
        CHECK(api.getriB(nullptr, 2, nullptr, 2, nullptr, nullptr, 2, info, 1) ==
              CUBLAS_STATUS_NOT_INITIALIZED);
        CHECK(api.getriB(h, -2, nullptr, 2, nullptr, nullptr, 2, info, 1) ==
              CUBLAS_STATUS_INVALID_VALUE);
        cudaFree(x); cudaFree(info);
    }

    // geam over transposition combinations.
    {
        const cublasOperation_t ops[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
        const int m = 3, n = 4;
        for (auto ta : ops)
            for (auto tb : ops) {
                const int lda = (ta == CUBLAS_OP_N ? m : n) + 1, ldb = (tb == CUBLAS_OP_N ? m : n) + 2;
                const int ac = ta == CUBLAS_OP_N ? n : m, bc = tb == CUBLAS_OP_N ? n : m;
                auto Av = rvec<T>(static_cast<std::size_t>(lda) * ac);
                auto Bv = rvec<T>(static_cast<std::size_t>(ldb) * bc);
                T alpha = P::make(0.5, 0.25), beta = P::make(-2, 1);
                T* A = dfill(Av);
                T* B = dfill(Bv);
                T* C = dalloc<T>(m * n);
                CHECK(api.geam(h, ta, tb, m, n, &alpha, A, lda, &beta, B, ldb, C, m) ==
                      CUBLAS_STATUS_SUCCESS);
                bool ok = true;
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i)
                        ok &= close(C[j * m + i], P::add(P::mul(alpha, opel(Av, lda, ta, i, j)),
                                                          P::mul(beta, opel(Bv, ldb, tb, i, j))));
                CHECK(ok);
                cudaFree(A); cudaFree(B); cudaFree(C);
            }
        // In place (C == A, untransposed) works; transposed in place is rejected.
        auto Av = rvec<T>(m * n), Bv = rvec<T>(m * n);
        T* A = dfill(Av);
        T* B = dfill(Bv);
        CHECK(api.geam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &one, A, m, &one, B, m, A, m) ==
              CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int i = 0; i < m * n; ++i) ok &= close(A[i], P::add(Av[i], Bv[i]));
        CHECK(ok);
        CHECK(api.geam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, m, &one, A, m, &one, B, n, A, n) ==
              CUBLAS_STATUS_INVALID_VALUE);
        cudaFree(A); cudaFree(B);
    }
    // dgmm: left, right, in-place, negative stride.
    {
        const int m = 3, n = 4, lda = 4, ldc = 5;
        auto Av = rvec<T>(lda * n), xv = rvec<T>(8);
        T* A = dfill(Av);
        T* x = dfill(xv);
        for (int side = 0; side < 2; ++side) {
            T* C = dalloc<T>(ldc * n);
            const auto mode = side == 0 ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;
            const int len = side == 0 ? m : n;
            CHECK(api.dgmm(h, mode, m, n, A, lda, x, 2, C, ldc) == CUBLAS_STATUS_SUCCESS);
            bool ok = true;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i)
                    ok &= close(C[j * ldc + i], P::mul(Av[j * lda + i], xv[2 * (side == 0 ? i : j)]));
            CHECK(ok);
            // Negative increment walks x backwards from the end of the vector.
            CHECK(api.dgmm(h, mode, m, n, A, lda, x, -2, C, ldc) == CUBLAS_STATUS_SUCCESS);
            ok = true;
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < m; ++i) {
                    const int k = side == 0 ? i : j;
                    ok &= close(C[j * ldc + i], P::mul(Av[j * lda + i], xv[2 * (len - 1 - k)]));
                }
            CHECK(ok);
            cudaFree(C);
        }
        CHECK(api.dgmm(h, CUBLAS_SIDE_LEFT, m, n, A, lda, x, 1, A, lda) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) ok &= close(A[j * lda + i], P::mul(Av[j * lda + i], xv[i]));
        CHECK(ok);
        cudaFree(A); cudaFree(x);
    }

    // Batched LU: factor, solve (N/T/C), invert, singular detection.
    {
        const int n = 4, nrhs = 2, lda = 5, ldb = 6, batch = 3;
        std::vector<std::vector<T>> A0, B0;
        std::vector<T*> A, B, Ci;
        for (int b = 0; b < batch; ++b) {
            auto a = rvec<T>(static_cast<std::size_t>(lda) * n);
            for (int i = 0; i < n; ++i) a[i * lda + i] = P::make(i == 0 ? 0.01 : 3.0 + i, 0);  // forces a row swap
            a[0 * lda + 2] = P::make(5.0, 1.0);  // big sub-column entry in column 0, row 2
            A0.push_back(a);
            B0.push_back(rvec<T>(static_cast<std::size_t>(ldb) * nrhs));
            A.push_back(dfill(a));
            B.push_back(dfill(B0.back()));
            Ci.push_back(dalloc<T>(static_cast<std::size_t>(lda) * n));
        }
        auto table = [&](std::vector<T*>& v) {
            T** t = reinterpret_cast<T**>(dalloc<void*>(batch));
            for (int b = 0; b < batch; ++b) t[b] = v[b];
            return t;
        };
        T** At = table(A);
        T** Bt = table(B);
        T** Ct = table(Ci);
        int* piv = dalloc<int>(n * batch);
        int* info = dalloc<int>(batch);
        CHECK(api.getrfB(h, n, At, lda, piv, info, batch) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int b = 0; b < batch; ++b) ok &= (info[b] == 0);
        CHECK(ok);
        CHECK(piv[0] == 3);  // row 2 (1-based 3) wins the first pivot search
        // Reconstruct P*A == L*U to validate the factors directly.
        ok = true;
        for (int b = 0; b < batch; ++b) {
            std::vector<T> pa = A0[b];
            for (int k = 0; k < n; ++k) {
                const int p = piv[b * n + k] - 1;
                if (p != k)
                    for (int j = 0; j < n; ++j) std::swap(pa[j * lda + k], pa[j * lda + p]);
            }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    T s = P::make(0, 0);
                    for (int p = 0; p <= std::min(i, j); ++p) {
                        const T l = (p == i) ? P::make(1, 0) : A[b][p * lda + i];
                        s = P::add(s, P::mul(l, A[b][j * lda + p]));
                    }
                    ok &= close(s, pa[j * lda + i], 1e-3);
                }
        }
        CHECK(ok);

        const cublasOperation_t trs[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
        for (auto tr : trs) {
            if (!P::is_complex && tr == CUBLAS_OP_C) continue;
            for (int b = 0; b < batch; ++b)
                for (int i = 0; i < ldb * nrhs; ++i) B[b][i] = B0[b][i];
            int hinfo = -1;
            CHECK(api.getrsB(h, tr, n, nrhs, At, lda, piv, Bt, ldb, &hinfo, batch) ==
                  CUBLAS_STATUS_SUCCESS);
            CHECK(hinfo == 0);
            ok = true;
            for (int b = 0; b < batch; ++b)
                for (int c = 0; c < nrhs; ++c)
                    for (int i = 0; i < n; ++i) {
                        T s = P::make(0, 0);
                        for (int p = 0; p < n; ++p)
                            s = P::add(s, P::mul(opel(A0[b], lda, tr, i, p), B[b][c * ldb + p]));
                        ok &= close(s, B0[b][c * ldb + i], 2e-3);
                    }
            CHECK(ok);
        }

        CHECK(api.getriB(h, n, At, lda, piv, Ct, lda, info, batch) == CUBLAS_STATUS_SUCCESS);
        ok = true;
        for (int b = 0; b < batch; ++b) {
            ok &= (info[b] == 0);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    T s = P::make(0, 0);
                    for (int p = 0; p < n; ++p) s = P::add(s, P::mul(A0[b][p * lda + i], Ci[b][j * lda + p]));
                    ok &= close(s, i == j ? P::make(1, 0) : P::make(0, 0), 2e-3);
                }
        }
        CHECK(ok);

        // No-pivot LU (PivotArray == NULL) on a diagonally dominant matrix; solve with NULL pivots.
        {
            auto a = rvec<T>(static_cast<std::size_t>(lda) * n);
            for (int i = 0; i < n; ++i) a[i * lda + i] = P::make(10.0 + i, 0);
            auto bv = rvec<T>(static_cast<std::size_t>(ldb) * nrhs);
            T* a1 = dfill(a);
            T* b1 = dfill(bv);
            T** a1t = reinterpret_cast<T**>(dalloc<void*>(1));
            T** b1t = reinterpret_cast<T**>(dalloc<void*>(1));
            a1t[0] = a1;
            b1t[0] = b1;
            int* i1 = dalloc<int>(1);
            CHECK(api.getrfB(h, n, a1t, lda, nullptr, i1, 1) == CUBLAS_STATUS_SUCCESS);
            CHECK(i1[0] == 0);
            int hinfo = -1;
            CHECK(api.getrsB(h, CUBLAS_OP_N, n, nrhs, a1t, lda, nullptr, b1t, ldb, &hinfo, 1) ==
                  CUBLAS_STATUS_SUCCESS);
            ok = true;
            for (int c = 0; c < nrhs; ++c)
                for (int i = 0; i < n; ++i) {
                    T s = P::make(0, 0);
                    for (int p = 0; p < n; ++p) s = P::add(s, P::mul(a[p * lda + i], b1[c * ldb + p]));
                    ok &= close(s, bv[c * ldb + i], 2e-3);
                }
            CHECK(ok);
            cudaFree(a1); cudaFree(b1); cudaFree(a1t); cudaFree(b1t); cudaFree(i1);
        }

        // Singular matrix: info reports the zero pivot, getri leaves C alone.
        {
            std::vector<T> z(static_cast<std::size_t>(lda) * n, P::make(0, 0));
            T* z1 = dfill(z);
            T* zc = dfill(z);
            T** zt = reinterpret_cast<T**>(dalloc<void*>(1));
            T** zct = reinterpret_cast<T**>(dalloc<void*>(1));
            zt[0] = z1;
            zct[0] = zc;
            zc[0] = P::make(7, 0);
            int* zp = dalloc<int>(n);
            int* zi = dalloc<int>(1);
            CHECK(api.getrfB(h, n, zt, lda, zp, zi, 1) == CUBLAS_STATUS_SUCCESS);
            CHECK(zi[0] == 1);
            CHECK(api.getriB(h, n, zt, lda, zp, zct, lda, zi, 1) == CUBLAS_STATUS_SUCCESS);
            CHECK(zi[0] > 0);
            CHECK(close(zc[0], P::make(7, 0)));
            cudaFree(z1); cudaFree(zc); cudaFree(zt); cudaFree(zct); cudaFree(zp); cudaFree(zi);
        }
        for (auto p : A) cudaFree(p);
        for (auto p : B) cudaFree(p);
        for (auto p : Ci) cudaFree(p);
        cudaFree(At); cudaFree(Bt); cudaFree(Ct); cudaFree(piv); cudaFree(info);
    }
}

// ── real band/packed helpers ──

template <class T>
void test_banded_packed(cublasHandle_t h,
                        cublasStatus_t (*sbmv)(cublasHandle_t, cublasFillMode_t, int, int, const T*,
                                               const T*, int, const T*, int, const T*, T*, int),
                        cublasStatus_t (*tpttr)(cublasHandle_t, cublasFillMode_t, int, const T*, T*, int),
                        cublasStatus_t (*trttp)(cublasHandle_t, cublasFillMode_t, int, const T*, int, T*),
                        const char* tag) {
    std::printf("banded/packed %s\n", tag);
    T one = 1;
    T* x = dalloc<T>(32);
    CHECK(sbmv(nullptr, CUBLAS_FILL_MODE_UPPER, 2, 1, &one, x, 2, x, 1, &one, x, 1) ==
          CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(sbmv(h, CUBLAS_FILL_MODE_UPPER, -2, 1, &one, x, 2, x, 1, &one, x, 1) ==
          CUBLAS_STATUS_INVALID_VALUE);
    CHECK(sbmv(h, CUBLAS_FILL_MODE_UPPER, 4, 2, &one, x, 2, x, 1, &one, x, 1) ==
          CUBLAS_STATUS_INVALID_VALUE);  // lda < k + 1
    CHECK(tpttr(nullptr, CUBLAS_FILL_MODE_UPPER, 2, x, x, 2) == CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(tpttr(h, CUBLAS_FILL_MODE_UPPER, -2, x, x, 2) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(trttp(nullptr, CUBLAS_FILL_MODE_UPPER, 2, x, 2, x) == CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(trttp(h, CUBLAS_FILL_MODE_UPPER, -2, x, 2, x) == CUBLAS_STATUS_INVALID_VALUE);
    cudaFree(x);

    // sbmv against a dense symmetric reference built from the band storage.
    for (int variant = 0; variant < 2; ++variant) {
        const cublasFillMode_t uplo = variant == 0 ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        const int n = 6, k = 2, lda = k + 2;
        auto band = rvec<T>(static_cast<std::size_t>(lda) * n), xv = rvec<T>(2 * n), yv = rvec<T>(n);
        std::vector<double> dense(n * n, 0.0);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                if (uplo == CUBLAS_FILL_MODE_UPPER && i <= j && j - i <= k)
                    dense[j * n + i] = dense[i * n + j] = band[j * lda + (k + i - j)];
                if (uplo == CUBLAS_FILL_MODE_LOWER && i >= j && i - j <= k)
                    dense[j * n + i] = dense[i * n + j] = band[j * lda + (i - j)];
            }
        T alpha = 0.75, beta = -0.5;
        T* A = dfill(band);
        T* xd = dfill(xv);
        T* yd = dfill(yv);
        CHECK(sbmv(h, uplo, n, k, &alpha, A, lda, xd, 2, &beta, yd, 1) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int j = 0; j < n; ++j) s += dense[j * n + i] * xv[2 * j];
            ok &= close_r(yd[i], alpha * s + beta * yv[i], Tr<T>::tol());
        }
        CHECK(ok);
        cudaFree(A); cudaFree(xd); cudaFree(yd);
    }

    // trttp / tpttr: column-major packed, upper (0,0)(0,1)(1,1)... and lower (0,0)(1,0)(2,0)...
    for (int variant = 0; variant < 2; ++variant) {
        const cublasFillMode_t uplo = variant == 0 ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
        const int n = 4, lda = 5, np = n * (n + 1) / 2;
        std::vector<T> a(static_cast<std::size_t>(lda) * n);
        for (std::size_t i = 0; i < a.size(); ++i) a[i] = static_cast<T>(i + 1);
        std::vector<T> expect;
        for (int j = 0; j < n; ++j)
            for (int i = (uplo == CUBLAS_FILL_MODE_UPPER ? 0 : j); i <= (uplo == CUBLAS_FILL_MODE_UPPER ? j : n - 1); ++i)
                expect.push_back(a[j * lda + i]);
        T* A = dfill(a);
        T* AP = dalloc<T>(np);
        CHECK(trttp(h, uplo, n, A, lda, AP) == CUBLAS_STATUS_SUCCESS);
        bool ok = true;
        for (int i = 0; i < np; ++i) ok &= (AP[i] == expect[i]);
        CHECK(ok);
        T* back = dalloc<T>(static_cast<std::size_t>(lda) * n);
        for (int i = 0; i < lda * n; ++i) back[i] = -1;
        CHECK(tpttr(h, uplo, n, AP, back, lda) == CUBLAS_STATUS_SUCCESS);
        ok = true;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < lda; ++i) {
                const bool in_tri = i < n && ((uplo == CUBLAS_FILL_MODE_UPPER) ? i <= j : i >= j);
                ok &= in_tri ? (back[j * lda + i] == a[j * lda + i]) : (back[j * lda + i] == -1);
            }
        CHECK(ok);
        cudaFree(A); cudaFree(AP); cudaFree(back);
    }
}

void test_sgemmex(cublasHandle_t h) {
    std::printf("sgemmex\n");
    const int m = 3, n = 2, k = 4;
    float alpha = 2.0f, beta = 0.5f;
    std::vector<float> av(m * k), bv(k * n), cv(m * n);
    for (auto& v : av) v = static_cast<float>(rnd());
    for (auto& v : bv) v = static_cast<float>(rnd());
    for (auto& v : cv) v = static_cast<float>(rnd());
    float* A = dfill(av);
    float* B = dfill(bv);
    float* C = dfill(cv);
    CHECK(cublasSgemmEx(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_32F, m, B,
                        CUDA_R_32F, k, &beta, C, CUDA_R_32F, m) == CUBLAS_STATUS_NOT_INITIALIZED);
    CHECK(cublasSgemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, -1, n, k, &alpha, A, CUDA_R_32F, m, B,
                        CUDA_R_32F, k, &beta, C, CUDA_R_32F, m) == CUBLAS_STATUS_INVALID_VALUE);
    CHECK(cublasSgemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_8I, m, B, CUDA_R_8I,
                        k, &beta, C, CUDA_R_32F, m) == CUBLAS_STATUS_NOT_SUPPORTED);
    CHECK(cublasSgemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_32F, m, B,
                        CUDA_R_32F, k, &beta, C, CUDA_R_32F, m) == CUBLAS_STATUS_SUCCESS);
    bool ok = true;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = 0;
            for (int p = 0; p < k; ++p) s += av[p * m + i] * bv[j * k + p];
            ok &= close_r(C[j * m + i], alpha * s + beta * cv[j * m + i], 1e-4);
        }
    CHECK(ok);

    // F16 operands, F32 output. Values are exactly representable in half.
    std::vector<__half> ah(m * k), bh(k * n);
    std::vector<float> af(m * k), bf(k * n);
    for (int i = 0; i < m * k; ++i) { af[i] = static_cast<float>((i % 5) - 2) * 0.25f; ah[i] = __float2half(af[i]); }
    for (int i = 0; i < k * n; ++i) { bf[i] = static_cast<float>((i % 3) - 1) * 0.5f; bh[i] = __float2half(bf[i]); }
    __half* Ah = dfill(ah);
    __half* Bh = dfill(bh);
    float* C2 = dfill(cv);
    CHECK(cublasSgemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, Ah, CUDA_R_16F, m, Bh,
                        CUDA_R_16F, k, &beta, C2, CUDA_R_32F, m) == CUBLAS_STATUS_SUCCESS);
    ok = true;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = 0;
            for (int p = 0; p < k; ++p) s += af[p * m + i] * bf[j * k + p];
            ok &= close_r(C2[j * m + i], alpha * s + beta * cv[j * m + i], 1e-3);
        }
    CHECK(ok);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(Ah); cudaFree(Bh); cudaFree(C2);
}

}  // namespace

// The cuBLAS 10 spelling of GemmEx takes the compute type as a cudaDataType.
static void test_legacy_gemmex(cublasHandle_t h) {
    const int n = 3;
    float *a = nullptr, *b = nullptr, *c = nullptr;
    CHECK(cudaMallocManaged(&a, n * n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMallocManaged(&b, n * n * sizeof(float)) == cudaSuccess);
    CHECK(cudaMallocManaged(&c, 2 * n * n * sizeof(float)) == cudaSuccess);
    for (int i = 0; i < n * n; ++i) {
        a[i] = static_cast<float>(i + 1);
        b[i] = static_cast<float>(2 * i - 3);
    }
    for (int i = 0; i < 2 * n * n; ++i) c[i] = -1.0f;
    const float alpha = 1.0f, beta = 0.0f;
    CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, CUDA_R_32F, n, b,
                       CUDA_R_32F, n, &beta, c, CUDA_R_32F, n, CUDA_R_32F,
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    // Stride 0 on A and B repeats the same product into both batch slots.
    CHECK(cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a,
                                     CUDA_R_32F, n, 0, b, CUDA_R_32F, n, 0, &beta, c + 0,
                                     CUDA_R_32F, n, n * n, 2, CUDA_R_32F,
                                     CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS);
    CHECK(cudaDeviceSynchronize() == cudaSuccess);
    for (int batch = 0; batch < 2; ++batch) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double ref = 0;
                for (int l = 0; l < n; ++l) ref += double(a[i + l * n]) * b[l + j * n];
                CHECK(close_r(c[batch * n * n + i + j * n], ref, 1e-5));
            }
        }
    }
    CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, CUDA_R_32F, n, b,
                       CUDA_R_32F, n, &beta, c, CUDA_R_32F, n, CUDA_R_8I,
                       CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_NOT_SUPPORTED);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

// BLAS never reads C (or geam's B) when beta is zero. Outputs are often
// recycled pool memory holding NaN bit patterns; CuPy's float64 matmul came
// back half NaN when the reference loops computed 0 * NaN. Every output below
// starts as NaN and every call passes beta = 0.
template <class T>
T* nan_buffer(size_t count) {
    T* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != cudaSuccess) return nullptr;
    std::vector<unsigned char> bytes(count * sizeof(T), 0xff);  // all-ones is a NaN
    std::copy(bytes.begin(), bytes.end(), reinterpret_cast<unsigned char*>(p));
    return p;
}

template <class T>
bool all_finite(const T* p, size_t count) {
    const auto* r = reinterpret_cast<const typename Tr<T>::R*>(p);
    const size_t n = count * (Tr<T>::is_complex ? 2 : 1);
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(r[i])) return false;
    return true;
}

template <class T>
T* filled(size_t count, double seed) {
    T* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)) != cudaSuccess) return nullptr;
    for (size_t i = 0; i < count; ++i)
        p[i] = Tr<T>::make(std::sin(seed + 0.37 * static_cast<double>(i)),
                           std::cos(seed + 0.11 * static_cast<double>(i)));
    return p;
}

template <class T, class Gemm>
void check_gemm_beta_zero(cublasHandle_t h, Gemm gemm, const char* name) {
    const int m = 7, n = 5, k = 3;
    T* a = filled<T>(m * k, 1.0);
    T* b = filled<T>(k * n, 2.0);
    T* c = nan_buffer<T>(m * n);
    const T one = Tr<T>::make(1, 0), zero = Tr<T>::make(0, 0);
    const bool ok = gemm(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &one, a, m, b, k, &zero, c, m) ==
                    CUBLAS_STATUS_SUCCESS;
    cudaDeviceSynchronize();
    bool right = ok;
    for (int j = 0; j < n && right; ++j)
        for (int i = 0; i < m && right; ++i) {
            T want = zero;
            for (int l = 0; l < k; ++l) want = Tr<T>::add(want, Tr<T>::mul(a[i + l * m], b[l + j * k]));
            right = close(c[i + j * m], want);
        }
    if (!right) std::fprintf(stderr, "FAIL: %s with beta = 0 read the NaN output\n", name);
    CHECK(right);
    cudaFree(a); cudaFree(b); cudaFree(c);
}

void test_beta_zero_ignores_output(cublasHandle_t h) {
    check_gemm_beta_zero<float>(h, cublasSgemm, "Sgemm");
    check_gemm_beta_zero<double>(h, cublasDgemm, "Dgemm");
    check_gemm_beta_zero<cuComplex>(h, cublasCgemm, "Cgemm");
    check_gemm_beta_zero<cuDoubleComplex>(h, cublasZgemm, "Zgemm");

    const int n = 6, k = 4;
    const float fone = 1, fzero = 0;
    const double done = 1, dzero = 0;
    const cuComplex cone{1, 0}, czero{0, 0};
    const cuDoubleComplex zone{1, 0}, zzero{0, 0};
    float* fa = filled<float>(n * n, 3.0);
    double* da = filled<double>(n * n, 3.0);
    double* db = filled<double>(n * n, 4.0);
    cuComplex* ca = filled<cuComplex>(n * n, 3.0);
    cuDoubleComplex* za = filled<cuDoubleComplex>(n * n, 3.0);
    cuDoubleComplex* zb = filled<cuDoubleComplex>(n * n, 4.0);
    float* fx = filled<float>(n, 5.0);
    double* dx = filled<double>(n, 5.0);
    cuComplex* cx = filled<cuComplex>(n, 5.0);

    struct Case { const char* name; bool ok; bool finite; };
    std::vector<Case> cases;
    {
        float* y = nan_buffer<float>(n);
        bool ok = cublasSgemv(h, CUBLAS_OP_T, n, n, &fone, fa, n, fx, 1, &fzero, y, 1) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Sgemv", ok, all_finite(y, n)}); cudaFree(y);
    }
    {
        double* y = nan_buffer<double>(n);
        bool ok = cublasDgemv(h, CUBLAS_OP_N, n, n, &done, da, n, dx, 1, &dzero, y, 1) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Dgemv", ok, all_finite(y, n)}); cudaFree(y);
    }
    {
        float* y = nan_buffer<float>(n);
        bool ok = cublasSsymv(h, CUBLAS_FILL_MODE_LOWER, n, &fone, fa, n, fx, 1, &fzero, y, 1) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Ssymv", ok, all_finite(y, n)}); cudaFree(y);
    }
    {
        double* c = nan_buffer<double>(n * n);
        bool ok = cublasDsyrk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &done, da, n, &dzero, c, n) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize();
        bool finite = true;  // only the referenced triangle is written
        for (int j = 0; j < n; ++j) finite &= all_finite(c + j * n, static_cast<size_t>(j + 1));
        cases.push_back({"Dsyrk", ok, finite}); cudaFree(c);
    }
    {
        double* c = nan_buffer<double>(n * n);
        bool ok = cublasDsymm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n, n, &done, da, n, db, n, &dzero, c, n) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Dsymm", ok, all_finite(c, n * n)}); cudaFree(c);
    }
    {
        cuComplex* y = nan_buffer<cuComplex>(n);
        bool ok = cublasChemv(h, CUBLAS_FILL_MODE_LOWER, n, &cone, ca, n, cx, 1, &czero, y, 1) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Chemv", ok, all_finite(y, n)}); cudaFree(y);
    }
    {
        cuComplex* c = nan_buffer<cuComplex>(n * n);
        bool ok = cublasCherk(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, n, k, &fone, ca, n, &fzero, c, n) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize();
        bool finite = true;
        for (int j = 0; j < n; ++j) finite &= all_finite(c + j * n, static_cast<size_t>(j + 1));
        cases.push_back({"Cherk", ok, finite}); cudaFree(c);
    }
    {
        cuDoubleComplex* c = nan_buffer<cuDoubleComplex>(n * n);
        bool ok = cublasZhemm(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, n, n, &zone, za, n, zb, n, &zzero, c, n) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Zhemm", ok, all_finite(c, n * n)}); cudaFree(c);
    }
    {
        double* b = nan_buffer<double>(n * n);  // geam: B is not read either
        double* c = nan_buffer<double>(n * n);
        bool ok = cublasDgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &done, da, n, &dzero, b, n, c, n) == CUBLAS_STATUS_SUCCESS;
        cudaDeviceSynchronize(); cases.push_back({"Dgeam", ok, all_finite(c, n * n)}); cudaFree(b); cudaFree(c);
    }
    for (const Case& item : cases) {
        if (!item.ok || !item.finite)
            std::fprintf(stderr, "FAIL: %s with beta = 0: status_ok=%d finite=%d\n", item.name,
                         item.ok, item.finite);
        CHECK(item.ok && item.finite);
    }
    cudaFree(fa); cudaFree(da); cudaFree(db); cudaFree(ca); cudaFree(za); cudaFree(zb);
    cudaFree(fx); cudaFree(dx); cudaFree(cx);
}

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit\n");
        return 1;
    }
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasCreate\n");
        return 1;
    }

    ComplexApi<cuComplex, float> c = {cublasCaxpy,  cublasCscal,  cublasCsscal, cublasCdotu,
                                      cublasCdotc,  cublasIcamax, cublasIcamin, cublasScasum,
                                      cublasScnrm2, cublasCgeru,  cublasCgerc,  cublasCsyrk,
                                      cublasCtrsm,  cublasCtrsmBatched, cublasCgemmBatched};
    test_complex(h, c, "C");
    ComplexApi<cuDoubleComplex, double> z = {cublasZaxpy,  cublasZscal,  cublasZdscal, cublasZdotu,
                                             cublasZdotc,  cublasIzamax, cublasIzamin, cublasDzasum,
                                             cublasDznrm2, cublasZgeru,  cublasZgerc,  cublasZsyrk,
                                             cublasZtrsm,  cublasZtrsmBatched, cublasZgemmBatched};
    test_complex(h, z, "Z");

    test_common<float>(h, {cublasSgeam, cublasSdgmm, cublasSgetrfBatched, cublasSgetrsBatched,
                           cublasSgetriBatched}, "S");
    test_common<double>(h, {cublasDgeam, cublasDdgmm, cublasDgetrfBatched, cublasDgetrsBatched,
                            cublasDgetriBatched}, "D");
    test_common<cuComplex>(h, {cublasCgeam, cublasCdgmm, cublasCgetrfBatched, cublasCgetrsBatched,
                               cublasCgetriBatched}, "C");
    test_common<cuDoubleComplex>(h, {cublasZgeam, cublasZdgmm, cublasZgetrfBatched,
                                     cublasZgetrsBatched, cublasZgetriBatched}, "Z");

    test_banded_packed<float>(h, cublasSsbmv, cublasStpttr, cublasStrttp, "S");
    test_banded_packed<double>(h, cublasDsbmv, cublasDtpttr, cublasDtrttp, "D");
    test_sgemmex(h);
    test_legacy_gemmex(h);
    test_beta_zero_ignores_output(h);

    cublasDestroy(h);
    if (g_failures != 0) {
        std::fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("cublas_cupy_surface_test: PASS\n");
    return 0;
}
