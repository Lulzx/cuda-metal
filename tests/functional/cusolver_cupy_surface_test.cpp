// Covers the cuSOLVER entry points CuPy links against: complex dense solvers,
// QR helpers (orgqr/ungqr, ormqr/unmqr), batched Cholesky, sytrf, gebrd,
// Jacobi EVD/SVD, approximate SVD, the IRS gesv/gels families and the sparse
// complex/eigenvalue extras. Each case solves a small well-conditioned problem
// and verifies the mathematical property of the result, not just the status.
#include "cusolverDn.h"
#include "cusolverSp.h"
#include "cusparse.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__);          \
            return false;                                                         \
        }                                                                         \
    } while (0)

// ── Host-side typing ────────────────────────────────────────────────────────

template <class T> struct TT;
template <> struct TT<float> {
    using H = float; using R = float; static constexpr bool cx = false;
    static constexpr double tol = 3e-4;
};
template <> struct TT<double> {
    using H = double; using R = double; static constexpr bool cx = false;
    static constexpr double tol = 1e-10;
};
template <> struct TT<cuComplex> {
    using H = std::complex<float>; using R = float; static constexpr bool cx = true;
    static constexpr double tol = 3e-4;
};
template <> struct TT<cuDoubleComplex> {
    using H = std::complex<double>; using R = double; static constexpr bool cx = true;
    static constexpr double tol = 1e-10;
};

template <class T> using H = typename TT<T>::H;
template <class T> using Rl = typename TT<T>::R;

static double cabs2(float x) { return std::fabs(x); }
static double cabs2(double x) { return std::fabs(x); }
template <class R> static double cabs2(std::complex<R> x) { return std::abs(x); }
static float cjg(float x) { return x; }
static double cjg(double x) { return x; }
template <class R> static std::complex<R> cjg(std::complex<R> x) { return std::conj(x); }

template <class Hs> struct Mat {
    int r = 0, c = 0;
    std::vector<Hs> d;
    Mat(int r_, int c_) : r(r_), c(c_), d(static_cast<size_t>(r_) * c_) {}
    Hs& operator()(int i, int j) { return d[static_cast<size_t>(i) + static_cast<size_t>(j) * r]; }
    const Hs& operator()(int i, int j) const {
        return d[static_cast<size_t>(i) + static_cast<size_t>(j) * r];
    }
};

template <class Hs> static Mat<Hs> mul(const Mat<Hs>& A, const Mat<Hs>& B) {
    Mat<Hs> C(A.r, B.c);
    for (int j = 0; j < B.c; ++j)
        for (int k = 0; k < A.c; ++k)
            for (int i = 0; i < A.r; ++i) C(i, j) += A(i, k) * B(k, j);
    return C;
}
template <class Hs> static Mat<Hs> adj(const Mat<Hs>& A) {
    Mat<Hs> B(A.c, A.r);
    for (int j = 0; j < A.c; ++j)
        for (int i = 0; i < A.r; ++i) B(j, i) = cjg(A(i, j));
    return B;
}
template <class Hs> static double fdiff(const Mat<Hs>& A, const Mat<Hs>& B) {
    double s = 0;
    for (size_t i = 0; i < A.d.size(); ++i) {
        const double d = cabs2(A.d[i] - B.d[i]);
        s += d * d;
    }
    return std::sqrt(s);
}
template <class Hs> static double fnorm(const Mat<Hs>& A) {
    double s = 0;
    for (const Hs& v : A.d) s += cabs2(v) * cabs2(v);
    return std::sqrt(s);
}
template <class Hs> static Mat<Hs> eye(int n) {
    Mat<Hs> I(n, n);
    for (int i = 0; i < n; ++i) I(i, i) = Hs(1);
    return I;
}

// Deterministic, non-degenerate pseudo-random entries in [-1, 1].
template <class T> static H<T> entry(int i, int j, int seed) {
    const double re = std::sin(1.7 * i + 2.3 * j + 0.9 * seed + 0.31 * i * j);
    if constexpr (TT<T>::cx) {
        const double im = std::cos(0.7 * i - 1.1 * j + 1.3 * seed + 0.17 * i * j);
        return H<T>(static_cast<Rl<T>>(re), static_cast<Rl<T>>(im));
    } else {
        return static_cast<H<T>>(re);
    }
}
template <class T> static Mat<H<T>> gen(int m, int n, int seed) {
    Mat<H<T>> A(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) A(i, j) = entry<T>(i, j, seed);
    return A;
}
// Strictly diagonally dominant general matrix (well conditioned, no pivoting
// needed).
template <class T> static Mat<H<T>> gen_dd(int n, int seed) {
    Mat<H<T>> A = gen<T>(n, n, seed);
    for (int i = 0; i < n; ++i) A(i, i) += H<T>(static_cast<Rl<T>>(2.0 * n));
    return A;
}
// Hermitian (symmetric when real) matrix from a random one.
template <class T> static Mat<H<T>> gen_herm(int n, int seed) {
    Mat<H<T>> G = gen<T>(n, n, seed), A(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) A(i, j) = (G(i, j) + cjg(G(j, i))) * Rl<T>(0.5);
    return A;
}
// Hermitian positive definite: B B^H + n I.
template <class T> static Mat<H<T>> gen_spd(int n, int seed) {
    Mat<H<T>> B = gen<T>(n, n, seed), A = mul(B, adj(B));
    for (int i = 0; i < n; ++i) A(i, i) += H<T>(static_cast<Rl<T>>(n));
    return A;
}

template <class T> static T* P(Mat<H<T>>& m) { return reinterpret_cast<T*>(m.d.data()); }
template <class T> static T* P(std::vector<H<T>>& v) { return reinterpret_cast<T*>(v.data()); }
template <class T> static const T* PC(const Mat<H<T>>& m) {
    return reinterpret_cast<const T*>(m.d.data());
}

// ── Per-precision entry-point table ─────────────────────────────────────────

template <class T> struct Api;
#define DEFINE_API(T, PFX, ORG, ORM, EVJ)                                                  \
    template <> struct Api<T> {                                                            \
        template <class... A> static auto getrf_bs(A... a) { return cusolverDn##PFX##getrf_bufferSize(a...); } \
        template <class... A> static auto getrf(A... a) { return cusolverDn##PFX##getrf(a...); }               \
        template <class... A> static auto getrs(A... a) { return cusolverDn##PFX##getrs(a...); }               \
        template <class... A> static auto geqrf_bs(A... a) { return cusolverDn##PFX##geqrf_bufferSize(a...); } \
        template <class... A> static auto geqrf(A... a) { return cusolverDn##PFX##geqrf(a...); }               \
        template <class... A> static auto orgqr_bs(A... a) { return cusolverDn##PFX##ORG##_bufferSize(a...); } \
        template <class... A> static auto orgqr(A... a) { return cusolverDn##PFX##ORG(a...); }                 \
        template <class... A> static auto ormqr_bs(A... a) { return cusolverDn##PFX##ORM##_bufferSize(a...); } \
        template <class... A> static auto ormqr(A... a) { return cusolverDn##PFX##ORM(a...); }                 \
        template <class... A> static auto potrf_bs(A... a) { return cusolverDn##PFX##potrf_bufferSize(a...); } \
        template <class... A> static auto potrf(A... a) { return cusolverDn##PFX##potrf(a...); }               \
        template <class... A> static auto potrs(A... a) { return cusolverDn##PFX##potrs(a...); }               \
        template <class... A> static auto potrfB(A... a) { return cusolverDn##PFX##potrfBatched(a...); }       \
        template <class... A> static auto potrsB(A... a) { return cusolverDn##PFX##potrsBatched(a...); }       \
        template <class... A> static auto sytrf_bs(A... a) { return cusolverDn##PFX##sytrf_bufferSize(a...); } \
        template <class... A> static auto sytrf(A... a) { return cusolverDn##PFX##sytrf(a...); }               \
        template <class... A> static auto gebrd_bs(A... a) { return cusolverDn##PFX##gebrd_bufferSize(a...); } \
        template <class... A> static auto gebrd(A... a) { return cusolverDn##PFX##gebrd(a...); }               \
        template <class... A> static auto evj_bs(A... a) { return cusolverDn##PFX##EVJ##_bufferSize(a...); }   \
        template <class... A> static auto evj(A... a) { return cusolverDn##PFX##EVJ(a...); }                   \
        template <class... A> static auto gesvdj_bs(A... a) { return cusolverDn##PFX##gesvdj_bufferSize(a...); } \
        template <class... A> static auto gesvdj(A... a) { return cusolverDn##PFX##gesvdj(a...); }             \
        template <class... A> static auto gesvdjB_bs(A... a) { return cusolverDn##PFX##gesvdjBatched_bufferSize(a...); } \
        template <class... A> static auto gesvdjB(A... a) { return cusolverDn##PFX##gesvdjBatched(a...); }     \
        template <class... A> static auto gesvda_bs(A... a) { return cusolverDn##PFX##gesvdaStridedBatched_bufferSize(a...); } \
        template <class... A> static auto gesvda(A... a) { return cusolverDn##PFX##gesvdaStridedBatched(a...); } \
    };
DEFINE_API(float, S, orgqr, ormqr, syevj)
DEFINE_API(double, D, orgqr, ormqr, syevj)
DEFINE_API(cuComplex, C, ungqr, unmqr, heevj)
DEFINE_API(cuDoubleComplex, Z, ungqr, unmqr, heevj)

static cusolverDnHandle_t g_handle = nullptr;

#define OK CUSOLVER_STATUS_SUCCESS

// ── Dense LU ────────────────────────────────────────────────────────────────

template <class T> static bool test_lu(const char* name) {
    using Hs = H<T>;
    const int n = 5, nrhs = 2;
    Mat<Hs> A0 = gen_dd<T>(n, 1), A = A0, B0 = gen<T>(n, nrhs, 2);
    int lwork = 0;
    CHECK(Api<T>::getrf_bs(g_handle, n, n, P<T>(A), n, &lwork) == OK && lwork >= 0, name);
    std::vector<Hs> work(static_cast<size_t>(std::max(1, lwork)));
    std::vector<int> ipiv(n);
    int info = -1;
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(A), n, P<T>(work), ipiv.data(), &info) == OK &&
              info == 0,
          "getrf");
    const cublasOperation_t ops[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    for (cublasOperation_t op : ops) {
        Mat<Hs> X = B0;
        info = -1;
        CHECK(Api<T>::getrs(g_handle, op, n, nrhs, PC<T>(A), n, ipiv.data(), P<T>(X), n, &info) ==
                      OK && info == 0,
              "getrs");
        Mat<Hs> Aop = op == CUBLAS_OP_N ? A0 : (op == CUBLAS_OP_T ? adj(A0) : adj(A0));
        if (op == CUBLAS_OP_T && TT<T>::cx) {  // plain transpose, no conjugation
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) Aop(i, j) = A0(j, i);
        }
        CHECK(fdiff(mul(Aop, X), B0) < TT<T>::tol * 50, "getrs: op(A)*X != B");
    }
    CHECK(Api<T>::getrs(g_handle, CUBLAS_OP_N, n, nrhs, PC<T>(A), n, ipiv.data(), P<T>(B0), n - 1,
                        &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "getrs ldb");
    CHECK(Api<T>::getrf(nullptr, n, n, P<T>(A), n, P<T>(work), ipiv.data(), &info) ==
              CUSOLVER_STATUS_NOT_INITIALIZED,
          "getrf null handle");
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(A), n - 1, P<T>(work), ipiv.data(), &info) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "getrf lda");
    return true;
}

// Complex-only behaviour: unpivoted LU (devIpiv == NULL) and info > 0 with
// SUCCESS status for a singular matrix.
template <class T> static bool test_lu_complex_extras(const char* name) {
    using Hs = H<T>;
    const int n = 4;
    Mat<Hs> A0 = gen_dd<T>(n, 3), A = A0, B0 = gen<T>(n, 1, 4), X = B0;
    std::vector<Hs> work(static_cast<size_t>(n * n));
    int info = -1;
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(A), n, P<T>(work), nullptr, &info) == OK && info == 0,
          "getrf without pivoting");
    const cublasOperation_t ops[3] = {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C};
    for (cublasOperation_t op : ops) {
        X = B0;
        CHECK(Api<T>::getrs(g_handle, op, n, 1, PC<T>(A), n, nullptr, P<T>(X), n, &info) == OK &&
                  info == 0,
              "getrs without pivoting");
        Mat<Hs> Aop(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                Aop(i, j) = op == CUBLAS_OP_N ? A0(i, j)
                                              : (op == CUBLAS_OP_T ? A0(j, i) : cjg(A0(j, i)));
        CHECK(fdiff(mul(Aop, X), B0) < TT<T>::tol * 50, "unpivoted getrs: op(A)*X != B");
    }
    Mat<Hs> S(n, n);  // zero matrix is singular
    std::vector<int> ipiv(n);
    info = -1;
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(S), n, P<T>(work), ipiv.data(), &info) == OK &&
              info > 0,
          "singular getrf reports info > 0 with success");
    (void)name;
    return true;
}

// ── QR family ───────────────────────────────────────────────────────────────

template <class T> static bool test_qr(const char* name) {
    using Hs = H<T>;
    const int m = 6, n = 4;
    Mat<Hs> A0 = gen<T>(m, n, 5), A = A0;
    int lw = 0;
    CHECK(Api<T>::geqrf_bs(g_handle, m, n, P<T>(A), m, &lw) == OK, name);
    std::vector<Hs> tau(n), work(static_cast<size_t>(std::max(lw, 1)));
    int info = -1;
    CHECK(Api<T>::geqrf(g_handle, m, n, P<T>(A), m, P<T>(tau), P<T>(work), lw, &info) == OK &&
              info == 0,
          "geqrf");
    Mat<Hs> R(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) R(i, j) = A(i, j);

    // Explicit Q (m x n) via orgqr/ungqr.
    int lwq = 0;
    CHECK(Api<T>::orgqr_bs(g_handle, m, n, n, PC<T>(A), m, reinterpret_cast<const T*>(tau.data()),
                           &lwq) == OK && lwq >= 1,
          "orgqr bufferSize");
    std::vector<Hs> workq(static_cast<size_t>(lwq));
    Mat<Hs> Q = A;
    info = -1;
    CHECK(Api<T>::orgqr(g_handle, m, n, n, P<T>(Q), m, reinterpret_cast<const T*>(tau.data()),
                        P<T>(workq), lwq, &info) == OK && info == 0,
          "orgqr");
    CHECK(fdiff(mul(adj(Q), Q), eye<Hs>(n)) < TT<T>::tol * 50, "Q^H Q != I");
    CHECK(fdiff(mul(Q, R), A0) < TT<T>::tol * 50, "Q R != A");

    // ormqr/unmqr: Q^H applied to A must give R (zero below the diagonal).
    const cublasOperation_t trans = TT<T>::cx ? CUBLAS_OP_C : CUBLAS_OP_T;
    int lwm = 0;
    Mat<Hs> C = A0;
    CHECK(Api<T>::ormqr_bs(g_handle, CUBLAS_SIDE_LEFT, trans, m, n, n, PC<T>(A), m,
                           reinterpret_cast<const T*>(tau.data()), PC<T>(C), m, &lwm) == OK &&
              lwm >= 1,
          "ormqr bufferSize");
    std::vector<Hs> workm(static_cast<size_t>(lwm));
    info = -1;
    CHECK(Api<T>::ormqr(g_handle, CUBLAS_SIDE_LEFT, trans, m, n, n, PC<T>(A), m,
                        reinterpret_cast<const T*>(tau.data()), P<T>(C), m, P<T>(workm), lwm,
                        &info) == OK && info == 0,
          "ormqr left");
    Mat<Hs> Rfull(m, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i <= j; ++i) Rfull(i, j) = R(i, j);
    CHECK(fdiff(C, Rfull) < TT<T>::tol * 50, "Q^H A != R");

    // Right side, no transpose: I_m * Q_full leaves Q in its first n columns.
    Mat<Hs> Ifull = eye<Hs>(m);
    int lwr = 0;
    CHECK(Api<T>::ormqr_bs(g_handle, CUBLAS_SIDE_RIGHT, CUBLAS_OP_N, m, m, n, PC<T>(A), m,
                           reinterpret_cast<const T*>(tau.data()), PC<T>(Ifull), m, &lwr) == OK,
          "ormqr right bufferSize");
    std::vector<Hs> workr(static_cast<size_t>(std::max(1, lwr)));
    CHECK(Api<T>::ormqr(g_handle, CUBLAS_SIDE_RIGHT, CUBLAS_OP_N, m, m, n, PC<T>(A), m,
                        reinterpret_cast<const T*>(tau.data()), P<T>(Ifull), m, P<T>(workr), lwr,
                        &info) == OK && info == 0,
          "ormqr right");
    double d = 0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) d += std::pow(cabs2(Ifull(i, j) - Q(i, j)), 2);
    CHECK(std::sqrt(d) < TT<T>::tol * 50, "I*Q_full leading columns != orgqr Q");

    // Negative paths.
    CHECK(Api<T>::orgqr(g_handle, n, m, n, P<T>(Q), m, reinterpret_cast<const T*>(tau.data()),
                        P<T>(workq), lwq, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "orgqr n > m");
    CHECK(Api<T>::orgqr(nullptr, m, n, n, P<T>(Q), m, reinterpret_cast<const T*>(tau.data()),
                        P<T>(workq), lwq, &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "orgqr null handle");
    const cublasOperation_t bad = TT<T>::cx ? CUBLAS_OP_T : CUBLAS_OP_C;
    CHECK(Api<T>::ormqr(g_handle, CUBLAS_SIDE_LEFT, bad, m, n, n, PC<T>(A), m,
                        reinterpret_cast<const T*>(tau.data()), P<T>(C), m, P<T>(workm), lwm,
                        &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "ormqr unsupported trans");
    return true;
}

// ── Cholesky: batched (all precisions) and single complex ───────────────────

template <class T> static bool test_potrf_batched(const char* name) {
    using Hs = H<T>;
    const int n = 4, batch = 3;
    std::vector<Mat<Hs>> A0, A, B0, B;
    for (int b = 0; b < batch; ++b) {
        A0.push_back(gen_spd<T>(n, 10 + b));
        A.push_back(A0.back());
        B0.push_back(gen<T>(n, 1, 20 + b));
        B.push_back(B0.back());
    }
    std::vector<T*> Ap(batch), Bp(batch);
    for (int b = 0; b < batch; ++b) { Ap[b] = P<T>(A[b]); Bp[b] = P<T>(B[b]); }
    std::vector<int> infos(batch, -1);
    CHECK(Api<T>::potrfB(g_handle, CUBLAS_FILL_MODE_LOWER, n, Ap.data(), n, infos.data(), batch) ==
              OK, name);
    for (int b = 0; b < batch; ++b) CHECK(infos[b] == 0, "potrfBatched info");
    int dinfo = -1;
    CHECK(Api<T>::potrsB(g_handle, CUBLAS_FILL_MODE_LOWER, n, 1, Ap.data(), n, Bp.data(), n,
                         &dinfo, batch) == OK && dinfo == 0,
          "potrsBatched");
    for (int b = 0; b < batch; ++b)
        CHECK(fdiff(mul(A0[b], B[b]), B0[b]) < TT<T>::tol * 100, "potrsBatched: A x != b");

    // A matrix that is not positive definite reports info > 0 but succeeds.
    Mat<Hs> bad = eye<Hs>(n);
    bad(2, 2) = Hs(-1);
    T* badp[1] = {P<T>(bad)};
    int binfo = -1;
    CHECK(Api<T>::potrfB(g_handle, CUBLAS_FILL_MODE_LOWER, n, badp, n, &binfo, 1) == OK &&
              binfo == 3,
          "potrfBatched non-SPD info");
    CHECK(Api<T>::potrfB(g_handle, CUBLAS_FILL_MODE_LOWER, n, Ap.data(), n - 1, infos.data(),
                         batch) == CUSOLVER_STATUS_INVALID_VALUE,
          "potrfBatched lda");
    CHECK(Api<T>::potrfB(nullptr, CUBLAS_FILL_MODE_LOWER, n, Ap.data(), n, infos.data(), batch) ==
              CUSOLVER_STATUS_NOT_INITIALIZED,
          "potrfBatched null handle");
    CHECK(Api<T>::potrfB(g_handle, CUBLAS_FILL_MODE_LOWER, n, Ap.data(), n, infos.data(), -1) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "potrfBatched negative batch");
    return true;
}

template <class T> static bool test_potrf_complex(const char* name) {
    using Hs = H<T>;
    const int n = 5;
    for (cublasFillMode_t uplo : {CUBLAS_FILL_MODE_LOWER, CUBLAS_FILL_MODE_UPPER}) {
        Mat<Hs> A0 = gen_spd<T>(n, 7), A = A0, B0 = gen<T>(n, 2, 8), B = B0;
        int lw = 0;
        CHECK(Api<T>::potrf_bs(g_handle, uplo, n, P<T>(A), n, &lw) == OK, name);
        std::vector<Hs> work(static_cast<size_t>(std::max(1, lw)));
        int info = -1;
        CHECK(Api<T>::potrf(g_handle, uplo, n, P<T>(A), n, P<T>(work), lw, &info) == OK &&
                  info == 0,
              "potrf");
        CHECK(Api<T>::potrs(g_handle, uplo, n, 2, PC<T>(A), n, P<T>(B), n, &info) == OK &&
                  info == 0,
              "potrs");
        CHECK(fdiff(mul(A0, B), B0) < TT<T>::tol * 100, "potrs: A x != b");
    }
    return true;
}

// ── sytrf ───────────────────────────────────────────────────────────────────

template <class T> static bool test_sytrf(const char* name) {
    using Hs = H<T>;
    const int n = 4;
    // Symmetric (not Hermitian) and strongly diagonal, so Bunch-Kaufman takes
    // only 1x1 pivots and the factorization is A = L D L^T.
    Mat<Hs> G = gen<T>(n, n, 9), A0(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) A0(i, j) = (G(i, j) + G(j, i)) * Rl<T>(0.5);
    for (int i = 0; i < n; ++i) A0(i, i) += Hs(static_cast<Rl<T>>(10 + i));
    Mat<Hs> A = A0;
    int lw = 0;
    CHECK(Api<T>::sytrf_bs(g_handle, n, P<T>(A), n, &lw) == OK && lw >= 1, name);
    std::vector<Hs> work(static_cast<size_t>(lw));
    std::vector<int> ipiv(n);
    int info = -1;
    CHECK(Api<T>::sytrf(g_handle, CUBLAS_FILL_MODE_LOWER, n, P<T>(A), n, ipiv.data(), P<T>(work),
                        lw, &info) == OK && info == 0,
          "sytrf");
    for (int i = 0; i < n; ++i) CHECK(ipiv[i] == i + 1, "sytrf took a non-1x1 pivot");
    Mat<Hs> L = eye<Hs>(n), Dm(n, n);
    for (int j = 0; j < n; ++j) {
        Dm(j, j) = A(j, j);
        for (int i = j + 1; i < n; ++i) L(i, j) = A(i, j);
    }
    Mat<Hs> Lt(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) Lt(i, j) = L(j, i);  // transpose, no conjugation
    CHECK(fdiff(mul(mul(L, Dm), Lt), A0) < TT<T>::tol * 50, "L D L^T != A");
    CHECK(Api<T>::sytrf(g_handle, CUBLAS_FILL_MODE_LOWER, n, P<T>(A), n - 1, ipiv.data(),
                        P<T>(work), lw, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "sytrf lda");
    CHECK(Api<T>::sytrf(nullptr, CUBLAS_FILL_MODE_LOWER, n, P<T>(A), n, ipiv.data(), P<T>(work),
                        lw, &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "sytrf null handle");
    return true;
}

// ── gebrd: orthogonal invariance of the Frobenius norm ──────────────────────

template <class T> static bool test_gebrd(const char* name) {
    using Hs = H<T>;
    const int m = 5, n = 3;
    Mat<Hs> A0 = gen<T>(m, n, 11), A = A0;
    int lw = 0;
    CHECK(Api<T>::gebrd_bs(g_handle, m, n, &lw) == OK && lw >= std::max(m, n), name);
    std::vector<Hs> work(static_cast<size_t>(lw)), tauq(n), taup(n);
    std::vector<Rl<T>> d(n), e(n);
    int info = -1;
    CHECK(Api<T>::gebrd(g_handle, m, n, P<T>(A), m, d.data(), e.data(), P<T>(tauq), P<T>(taup),
                        P<T>(work), lw, &info) == OK && info == 0,
          "gebrd");
    double s = 0;
    for (int i = 0; i < n; ++i) s += double(d[i]) * d[i];
    for (int i = 0; i < n - 1; ++i) s += double(e[i]) * e[i];
    CHECK(std::fabs(std::sqrt(s) - fnorm(A0)) < TT<T>::tol * 50, "||B||_F != ||A||_F");
    CHECK(Api<T>::gebrd(g_handle, m, n, P<T>(A), m, d.data(), e.data(), P<T>(tauq), P<T>(taup),
                        P<T>(work), 1, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "gebrd lwork too small");
    CHECK(Api<T>::gebrd(nullptr, m, n, P<T>(A), m, d.data(), e.data(), P<T>(tauq), P<T>(taup),
                        P<T>(work), lw, &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "gebrd null handle");
    return true;
}

// ── Complex gesvd and heevd ─────────────────────────────────────────────────

template <class T, class BS, class SVD>
static bool test_gesvd_complex(const char* name, BS bs, SVD svd) {
    using Hs = H<T>;
    const int m = 5, n = 3;
    Mat<Hs> A0 = gen<T>(m, n, 12), A = A0, U(m, n), VT(n, n);
    std::vector<Rl<T>> S(n);
    int lw = 0;
    CHECK(bs(g_handle, m, n, &lw) == OK && lw >= 1, name);
    std::vector<Hs> work(static_cast<size_t>(lw));
    int info = -1;
    // rwork == NULL is legal for cuSOLVER.
    CHECK(svd(g_handle, static_cast<signed char>('S'), static_cast<signed char>('S'), m, n,
              P<T>(A), m, S.data(), P<T>(U), m, P<T>(VT), n, P<T>(work), lw,
              static_cast<Rl<T>*>(nullptr), &info) == OK && info == 0,
          "gesvd");
    Mat<Hs> Sm(n, n);
    for (int i = 0; i < n; ++i) Sm(i, i) = Hs(S[i]);
    CHECK(fdiff(mul(mul(U, Sm), VT), A0) < TT<T>::tol * 50, "U S V^H != A");
    CHECK(fdiff(mul(adj(U), U), eye<Hs>(n)) < TT<T>::tol * 50, "U not orthonormal");
    for (int i = 0; i + 1 < n; ++i) CHECK(S[i] >= S[i + 1], "singular values not descending");
    CHECK(svd(nullptr, static_cast<signed char>('S'), static_cast<signed char>('S'), m, n, P<T>(A),
              m, S.data(), P<T>(U), m, P<T>(VT), n, P<T>(work), lw, static_cast<Rl<T>*>(nullptr),
              &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "gesvd null handle");
    CHECK(svd(g_handle, static_cast<signed char>('S'), static_cast<signed char>('S'), m, n,
              P<T>(A), m - 1, S.data(), P<T>(U), m, P<T>(VT), n, P<T>(work), lw,
              static_cast<Rl<T>*>(nullptr), &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "gesvd lda");
    return true;
}

template <class T, class BS, class EVD>
static bool test_heevd(const char* name, BS bs, EVD evd) {
    using Hs = H<T>;
    const int n = 5;
    Mat<Hs> A0 = gen_herm<T>(n, 13), A = A0;
    std::vector<Rl<T>> W(n), W2(n);
    int lw = 0;
    CHECK(bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, PC<T>(A), n, W.data(),
             &lw) == OK && lw >= 1,
          name);
    std::vector<Hs> work(static_cast<size_t>(lw));
    int info = -1;
    CHECK(evd(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(A), n, W.data(),
              P<T>(work), lw, &info) == OK && info == 0,
          "heevd");
    for (int i = 0; i + 1 < n; ++i) CHECK(W[i] <= W[i + 1], "eigenvalues not ascending");
    CHECK(fdiff(mul(adj(A), A), eye<Hs>(n)) < TT<T>::tol * 50, "eigenvectors not unitary");
    Mat<Hs> AV = mul(A0, A), VL(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) VL(i, j) = A(i, j) * Rl<T>(W[j]);
    CHECK(fdiff(AV, VL) < TT<T>::tol * 50, "A v != lambda v");
    // Values-only agrees with the vector run.
    Mat<Hs> B = A0;
    CHECK(evd(g_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, P<T>(B), n,
              W2.data(), P<T>(work), lw, &info) == OK && info == 0,
          "heevd values only");
    for (int i = 0; i < n; ++i) CHECK(std::fabs(W[i] - W2[i]) < TT<T>::tol * 50, "values differ");
    CHECK(evd(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(B), n, W2.data(),
              P<T>(work), 1, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "heevd lwork too small");
    return true;
}

// ── Jacobi eigensolver (single and batched) ─────────────────────────────────

template <class T> static bool check_eig(const Mat<H<T>>& A0, const Mat<H<T>>& V,
                                         const std::vector<Rl<T>>& W, double tol) {
    const int n = A0.r;
    Mat<H<T>> AV = mul(A0, V), VL(n, n);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) VL(i, j) = V(i, j) * Rl<T>(W[j]);
    return fdiff(AV, VL) < tol && fdiff(mul(adj(V), V), eye<H<T>>(n)) < tol;
}

template <class T> static bool test_evj(const char* name) {
    using Hs = H<T>;
    const int n = 6;
    Mat<Hs> A0 = gen_herm<T>(n, 14), A = A0;
    syevjInfo_t params = nullptr;
    CHECK(cusolverDnCreateSyevjInfo(&params) == OK, name);
    std::vector<Rl<T>> W(n);
    int lw = 0;
    CHECK(Api<T>::evj_bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, PC<T>(A), n,
                         W.data(), &lw, params) == OK && lw >= 1,
          "evj bufferSize");
    std::vector<Hs> work(static_cast<size_t>(lw));
    int info = -1;
    CHECK(Api<T>::evj(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(A), n,
                      W.data(), P<T>(work), lw, &info, params) == OK && info == 0,
          "evj");
    for (int i = 0; i + 1 < n; ++i) CHECK(W[i] <= W[i + 1], "evj default must sort ascending");
    CHECK(check_eig<T>(A0, A, W, TT<T>::tol * 100), "evj: A v != lambda v / not orthonormal");
    double residual = -1;
    int sweeps = -1;
    CHECK(cusolverDnXsyevjGetResidual(g_handle, params, &residual) == OK &&
              cusolverDnXsyevjGetSweeps(g_handle, params, &sweeps) == OK,
          "evj get stats");
    CHECK(residual >= 0 && residual < 1e-2 && sweeps > 0 && sweeps <= 100, "evj stats implausible");

    // Upper triangle, values only: same spectrum.
    Mat<Hs> B = A0;
    std::vector<Rl<T>> W2(n);
    CHECK(Api<T>::evj(g_handle, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_UPPER, n, P<T>(B), n,
                      W2.data(), P<T>(work), lw, &info, params) == OK && info == 0,
          "evj values only");
    for (int i = 0; i < n; ++i) CHECK(std::fabs(W[i] - W2[i]) < TT<T>::tol * 100, "evj values differ");

    // Starved sweep budget: reports non-convergence as n+1 and still succeeds.
    CHECK(cusolverDnXsyevjSetMaxSweeps(params, 1) == OK &&
              cusolverDnXsyevjSetTolerance(params, 1e-30) == OK,
          "evj set params");
    B = A0;
    CHECK(Api<T>::evj(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(B), n,
                      W2.data(), P<T>(work), lw, &info, params) == OK && info == n + 1,
          "evj non-convergence must report n+1");
    CHECK(cusolverDnXsyevjGetSweeps(g_handle, params, &sweeps) == OK && sweeps == 1,
          "evj executed sweeps must honour max_sweeps");
    CHECK(Api<T>::evj(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(B), n - 1,
                      W2.data(), P<T>(work), lw, &info, params) == CUSOLVER_STATUS_INVALID_VALUE,
          "evj lda");
    CHECK(Api<T>::evj(nullptr, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, P<T>(B), n,
                      W2.data(), P<T>(work), lw, &info, params) ==
              CUSOLVER_STATUS_NOT_INITIALIZED,
          "evj null handle");
    cusolverDnDestroySyevjInfo(params);
    return true;
}

template <class T, class BS, class RUN>
static bool test_heevj_batched(const char* name, BS bs, RUN run) {
    using Hs = H<T>;
    const int n = 4, batch = 3;
    std::vector<Mat<Hs>> A0;
    Mat<Hs> A(n, n * batch);
    for (int b = 0; b < batch; ++b) {
        A0.push_back(gen_herm<T>(n, 30 + b));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) A(i, j + b * n) = A0[b](i, j);
    }
    syevjInfo_t params = nullptr;
    CHECK(cusolverDnCreateSyevjInfo(&params) == OK, name);
    std::vector<Rl<T>> W(static_cast<size_t>(n) * batch);
    int lw = 0;
    CHECK(bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, PC<T>(A), n, W.data(),
             &lw, params, batch) == OK && lw >= 1,
          "bufferSize");
    std::vector<Hs> work(static_cast<size_t>(lw));
    std::vector<int> info(batch, -1);
    CHECK(run(g_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, P<T>(A), n, W.data(),
              P<T>(work), lw, info.data(), params, batch) == OK,
          "batched run");
    for (int b = 0; b < batch; ++b) {
        CHECK(info[b] == 0, "batched info");
        Mat<Hs> V(n, n);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) V(i, j) = A(i, j + b * n);
        std::vector<Rl<T>> w(W.begin() + b * n, W.begin() + (b + 1) * n);
        CHECK(check_eig<T>(A0[b], V, w, TT<T>::tol * 100), "batched: A v != lambda v");
    }
    cusolverDnDestroySyevjInfo(params);
    return true;
}

// ── Jacobi SVD and approximate SVD ──────────────────────────────────────────

template <class T> static bool test_gesvdj(const char* name) {
    using Hs = H<T>;
    const int m = 5, n = 3, k = 3;
    gesvdjInfo_t params = nullptr;
    CHECK(cusolverDnCreateGesvdjInfo(&params) == OK, name);
    CHECK(cusolverDnXgesvdjSetTolerance(params, 1e-12) == OK &&
              cusolverDnXgesvdjSetMaxSweeps(params, 50) == OK &&
              cusolverDnXgesvdjSetSortEig(params, 1) == OK,
          "gesvdj setters");
    CHECK(cusolverDnXgesvdjSetTolerance(nullptr, 1e-12) == CUSOLVER_STATUS_INVALID_VALUE &&
              cusolverDnXgesvdjSetTolerance(params, -1.0) == CUSOLVER_STATUS_INVALID_VALUE &&
              cusolverDnXgesvdjSetMaxSweeps(params, -1) == CUSOLVER_STATUS_INVALID_VALUE,
          "gesvdj setter negative paths");
    for (int econ = 0; econ <= 1; ++econ) {
        const int ucols = econ ? k : m, vcols = econ ? k : n;
        Mat<Hs> A0 = gen<T>(m, n, 15), A = A0, U(m, ucols), V(n, vcols);
        std::vector<Rl<T>> S(k);
        int lw = 0;
        CHECK(Api<T>::gesvdj_bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, PC<T>(A), m,
                                S.data(), PC<T>(U), m, PC<T>(V), n, &lw, params) == OK && lw >= 1,
              "gesvdj bufferSize");
        std::vector<Hs> work(static_cast<size_t>(lw));
        int info = -1;
        CHECK(Api<T>::gesvdj(g_handle, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, P<T>(A), m, S.data(),
                             P<T>(U), m, P<T>(V), n, P<T>(work), lw, &info, params) == OK &&
                  info == 0,
              "gesvdj");
        Mat<Hs> Uk(m, k), Vk(n, k), Sm(k, k);
        for (int j = 0; j < k; ++j) {
            Sm(j, j) = Hs(S[j]);
            for (int i = 0; i < m; ++i) Uk(i, j) = U(i, j);
            for (int i = 0; i < n; ++i) Vk(i, j) = V(i, j);
        }
        CHECK(fdiff(mul(mul(Uk, Sm), adj(Vk)), A0) < TT<T>::tol * 50, "U S V^H != A");
        CHECK(fdiff(mul(adj(U), U), eye<Hs>(ucols)) < TT<T>::tol * 50, "U not orthonormal");
        CHECK(fdiff(mul(adj(V), V), eye<Hs>(vcols)) < TT<T>::tol * 50, "V not orthonormal");
        for (int i = 0; i + 1 < k; ++i) CHECK(S[i] >= S[i + 1], "gesvdj values not descending");
        double residual = -1;
        int sweeps = -1;
        CHECK(cusolverDnXgesvdjGetResidual(g_handle, params, &residual) == OK &&
                  cusolverDnXgesvdjGetSweeps(g_handle, params, &sweeps) == OK,
              "gesvdj stats");
        CHECK(residual >= 0 && residual < TT<T>::tol * 50 && sweeps == 0,
              "gesvdj residual must be measured, sweeps must be 0");
    }
    // Values only.
    {
        Mat<Hs> A = gen<T>(m, n, 15), U(m, m), V(n, n);
        std::vector<Rl<T>> S(k), S2(k);
        Mat<Hs> Av = A;
        std::vector<Hs> work(1);
        int info = -1;
        CHECK(Api<T>::gesvdj(g_handle, CUSOLVER_EIG_MODE_NOVECTOR, 0, m, n, P<T>(A), m, S.data(),
                             nullptr, m, nullptr, n, P<T>(work), 1, &info, params) == OK &&
                  info == 0,
              "gesvdj values only");
        CHECK(Api<T>::gesvdj(g_handle, CUSOLVER_EIG_MODE_VECTOR, 0, m, n, P<T>(Av), m, S2.data(),
                             P<T>(U), m, P<T>(V), n, P<T>(work), 1, &info, params) == OK,
              "gesvdj reference");
        for (int i = 0; i < k; ++i) CHECK(std::fabs(S[i] - S2[i]) < TT<T>::tol * 50, "values differ");
        CHECK(Api<T>::gesvdj(g_handle, CUSOLVER_EIG_MODE_VECTOR, 0, m, n, P<T>(Av), m, S2.data(),
                             P<T>(U), m - 1, P<T>(V), n, P<T>(work), 1, &info, params) ==
                  CUSOLVER_STATUS_INVALID_VALUE,
              "gesvdj ldu too small");
        CHECK(Api<T>::gesvdj(nullptr, CUSOLVER_EIG_MODE_VECTOR, 0, m, n, P<T>(Av), m, S2.data(),
                             P<T>(U), m, P<T>(V), n, P<T>(work), 1, &info, params) ==
                  CUSOLVER_STATUS_NOT_INITIALIZED,
              "gesvdj null handle");
    }
    cusolverDnDestroyGesvdjInfo(params);
    return true;
}

template <class T> static bool test_gesvdj_batched(const char* name) {
    using Hs = H<T>;
    const int m = 4, n = 3, batch = 3, k = 3;
    gesvdjInfo_t params = nullptr;
    CHECK(cusolverDnCreateGesvdjInfo(&params) == OK, name);
    std::vector<Mat<Hs>> A0;
    Mat<Hs> A(m, n * batch), U(m, m * batch), V(n, n * batch);
    for (int b = 0; b < batch; ++b) {
        A0.push_back(gen<T>(m, n, 40 + b));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) A(i, j + b * n) = A0[b](i, j);
    }
    std::vector<Rl<T>> S(static_cast<size_t>(k) * batch);
    int lw = 0;
    CHECK(Api<T>::gesvdjB_bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, m, n, PC<T>(A), m, S.data(),
                             PC<T>(U), m, PC<T>(V), n, &lw, params, batch) == OK && lw >= 1,
          "gesvdjBatched bufferSize");
    std::vector<Hs> work(static_cast<size_t>(lw));
    std::vector<int> info(batch, -1);
    CHECK(Api<T>::gesvdjB(g_handle, CUSOLVER_EIG_MODE_VECTOR, m, n, P<T>(A), m, S.data(), P<T>(U),
                          m, P<T>(V), n, P<T>(work), lw, info.data(), params, batch) == OK,
          "gesvdjBatched");
    for (int b = 0; b < batch; ++b) {
        CHECK(info[b] == 0, "gesvdjBatched info");
        Mat<Hs> Uk(m, k), Vk(n, k), Sm(k, k);
        for (int j = 0; j < k; ++j) {
            Sm(j, j) = Hs(S[b * k + j]);
            for (int i = 0; i < m; ++i) Uk(i, j) = U(i, j + b * m);
            for (int i = 0; i < n; ++i) Vk(i, j) = V(i, j + b * n);
        }
        CHECK(fdiff(mul(mul(Uk, Sm), adj(Vk)), A0[b]) < TT<T>::tol * 50, "batched U S V^H != A");
    }
    CHECK(Api<T>::gesvdjB(g_handle, CUSOLVER_EIG_MODE_VECTOR, m, n, P<T>(A), m - 1, S.data(),
                          P<T>(U), m, P<T>(V), n, P<T>(work), lw, info.data(), params, batch) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "gesvdjBatched lda");
    cusolverDnDestroyGesvdjInfo(params);
    return true;
}

template <class T> static bool test_gesvda(const char* name) {
    using Hs = H<T>;
    const int m = 6, n = 4, batch = 2;
    std::vector<Mat<Hs>> A0;
    Mat<Hs> A(m, n * batch);
    for (int b = 0; b < batch; ++b) {
        A0.push_back(gen<T>(m, n, 50 + b));
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) A(i, j + b * n) = A0[b](i, j);
    }
    const long long strideA = static_cast<long long>(m) * n;
    for (int rank : {n, 2}) {
        Mat<Hs> U(m, rank * batch), V(n, rank * batch);
        std::vector<Rl<T>> S(static_cast<size_t>(rank) * batch);
        int lw = 0;
        CHECK(Api<T>::gesvda_bs(g_handle, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, PC<T>(A), m,
                                strideA, S.data(), static_cast<long long>(rank), PC<T>(U), m,
                                static_cast<long long>(m) * rank, PC<T>(V), n,
                                static_cast<long long>(n) * rank, &lw, batch) == OK && lw >= 1,
              name);
        std::vector<Hs> work(static_cast<size_t>(lw));
        std::vector<int> info(batch, -1);
        std::vector<double> nrm(batch, -1.0);
        CHECK(Api<T>::gesvda(g_handle, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, PC<T>(A), m, strideA,
                             S.data(), static_cast<long long>(rank), P<T>(U), m,
                             static_cast<long long>(m) * rank, P<T>(V), n,
                             static_cast<long long>(n) * rank, P<T>(work), lw, info.data(),
                             nrm.data(), batch) == OK,
              "gesvdaStridedBatched");
        for (int b = 0; b < batch; ++b) {
            CHECK(info[b] == 0, "gesvda info");
            Mat<Hs> Ur(m, rank), Vr(n, rank), Sm(rank, rank);
            for (int j = 0; j < rank; ++j) {
                Sm(j, j) = Hs(S[b * rank + j]);
                for (int i = 0; i < m; ++i) Ur(i, j) = U(i, j + b * rank);
                for (int i = 0; i < n; ++i) Vr(i, j) = V(i, j + b * rank);
            }
            const double err = fdiff(mul(mul(Ur, Sm), adj(Vr)), A0[b]);
            CHECK(std::fabs(err - nrm[b]) < TT<T>::tol * 100, "gesvda h_R_nrmF != ||A - U S V^H||");
            if (rank == n) CHECK(nrm[b] < TT<T>::tol * 50, "full-rank gesvda must be exact");
            else CHECK(nrm[b] > 1e-3, "truncated gesvda must have a nonzero residual");
            CHECK(fdiff(mul(adj(Ur), Ur), eye<Hs>(rank)) < TT<T>::tol * 50, "gesvda U not orthonormal");
        }
    }
    Mat<Hs> U(m, n), V(n, n);
    std::vector<Rl<T>> S(n);
    std::vector<Hs> work(1);
    int info = 0;
    CHECK(Api<T>::gesvda(g_handle, CUSOLVER_EIG_MODE_VECTOR, n + 1, m, n, PC<T>(A), m, strideA,
                         S.data(), static_cast<long long>(n), P<T>(U), m,
                         static_cast<long long>(m) * n, P<T>(V), n, static_cast<long long>(n) * n,
                         P<T>(work), 1, &info, nullptr, 1) == CUSOLVER_STATUS_INVALID_VALUE,
          "gesvda rank > min(m,n)");
    CHECK(Api<T>::gesvda(nullptr, CUSOLVER_EIG_MODE_VECTOR, n, m, n, PC<T>(A), m, strideA,
                         S.data(), static_cast<long long>(n), P<T>(U), m,
                         static_cast<long long>(m) * n, P<T>(V), n, static_cast<long long>(n) * n,
                         P<T>(work), 1, &info, nullptr, 1) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "gesvda null handle");
    return true;
}

// ── IRS gesv / gels ─────────────────────────────────────────────────────────

template <class T, class GESV_BS, class GESV, class GELS_BS, class GELS>
static bool test_irs(const char* name, GESV_BS gesv_bs, GESV gesv, GELS_BS gels_bs, GELS gels) {
    using Hs = H<T>;
    const int n = 4, nrhs = 2;
    Mat<Hs> A0 = gen_dd<T>(n, 60), A = A0, B0 = gen<T>(n, nrhs, 61), B = B0, X(n, nrhs);
    std::vector<int> ipiv(n, 0);
    size_t lwb = 0;
    CHECK(gesv_bs(g_handle, n, nrhs, P<T>(A), n, ipiv.data(), P<T>(B), n, P<T>(X), n, nullptr,
                  &lwb) == OK, name);
    std::vector<unsigned char> ws(lwb + 1);
    int niter = -99, info = -1;
    CHECK(gesv(g_handle, n, nrhs, P<T>(A), n, ipiv.data(), P<T>(B), n, P<T>(X), n, ws.data(), lwb,
               &niter, &info) == OK && info == 0,
          "IRS gesv");
    CHECK(niter == 0, "IRS gesv must report 0 refinement iterations");
    CHECK(fdiff(mul(A0, X), B0) < TT<T>::tol * 50, "IRS gesv: A X != B");
    CHECK(fdiff(A, A0) == 0 && fdiff(B, B0) == 0, "IRS gesv must not modify dA or dB");
    CHECK(std::all_of(ipiv.begin(), ipiv.end(), [&](int p) { return p >= 1 && p <= n; }),
          "IRS gesv pivots");
    // Singular matrix: info > 0, still SUCCESS.
    Mat<Hs> Z(n, n);
    CHECK(gesv(g_handle, n, nrhs, P<T>(Z), n, nullptr, P<T>(B), n, P<T>(X), n, ws.data(), lwb,
               &niter, &info) == OK && info > 0,
          "IRS gesv singular");
    CHECK(gesv(g_handle, n, nrhs, P<T>(A), n, ipiv.data(), P<T>(B), n, P<T>(X), n, ws.data(),
               lwb - 1, &niter, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "IRS gesv short workspace");
    CHECK(gesv(nullptr, n, nrhs, P<T>(A), n, ipiv.data(), P<T>(B), n, P<T>(X), n, ws.data(), lwb,
               &niter, &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "IRS gesv null handle");
    CHECK(gesv(g_handle, n, nrhs, P<T>(A), n - 1, ipiv.data(), P<T>(B), n, P<T>(X), n, ws.data(),
               lwb, &niter, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "IRS gesv ldda");

    // Least squares: the residual must be orthogonal to range(A).
    const int m = 7, nn = 3;
    Mat<Hs> LA0 = gen<T>(m, nn, 62), LA = LA0, LB0 = gen<T>(m, nrhs, 63), LB = LB0, LX(nn, nrhs);
    CHECK(gels_bs(g_handle, m, nn, nrhs, P<T>(LA), m, P<T>(LB), m, P<T>(LX), nn, nullptr, &lwb) ==
              OK,
          "IRS gels bufferSize");
    ws.assign(lwb + 1, 0);
    CHECK(gels(g_handle, m, nn, nrhs, P<T>(LA), m, P<T>(LB), m, P<T>(LX), nn, ws.data(), lwb,
               &niter, &info) == OK && info == 0 && niter == 0,
          "IRS gels");
    Mat<Hs> Rres = mul(LA0, LX);
    for (size_t i = 0; i < Rres.d.size(); ++i) Rres.d[i] -= LB0.d[i];
    CHECK(fnorm(mul(adj(LA0), Rres)) < TT<T>::tol * 100, "IRS gels: residual not orthogonal to A");
    CHECK(fdiff(LA, LA0) == 0 && fdiff(LB, LB0) == 0, "IRS gels must not modify dA or dB");
    CHECK(gels(g_handle, nn, m, nrhs, P<T>(LA), nn, P<T>(LB), nn, P<T>(LX), m, ws.data(), lwb,
               &niter, &info) == CUSOLVER_STATUS_INVALID_VALUE,
          "IRS gels m < n");
    CHECK(gels(nullptr, m, nn, nrhs, P<T>(LA), m, P<T>(LB), m, P<T>(LX), nn, ws.data(), lwb,
               &niter, &info) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "IRS gels null handle");
    return true;
}

// All four precisions: unpivoted LU round trip, and a singular matrix reports
// SUCCESS with devInfo == first zero pivot (S/D used to return INTERNAL_ERROR).
template <class T> static bool test_lu_nopivot_and_singular(const char* name) {
    using Hs = H<T>;
    const int n = 4;
    Mat<Hs> A0 = gen_dd<T>(n, 3), A = A0, B0 = gen<T>(n, 2, 4), X = B0;
    std::vector<Hs> work(static_cast<size_t>(n * n));
    int info = -1;
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(A), n, P<T>(work), nullptr, &info) == OK && info == 0,
          name);
    CHECK(Api<T>::getrs(g_handle, CUBLAS_OP_N, n, 2, PC<T>(A), n, nullptr, P<T>(X), n, &info) == OK &&
              info == 0,
          "unpivoted getrs");
    CHECK(fdiff(mul(A0, X), B0) < TT<T>::tol * 50, "unpivoted LU round trip");
    Mat<Hs> S = eye<Hs>(n);
    S(2, 2) = Hs(0);  // singular: first zero pivot is 3 (1-based)
    std::vector<int> ipiv(n);
    info = -1;
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(S), n, P<T>(work), ipiv.data(), &info) == OK && info == 3,
          "singular getrf must be SUCCESS with devInfo == 3");
    info = -1;
    Mat<Hs> Sn = eye<Hs>(n);
    Sn(2, 2) = Hs(0);
    CHECK(Api<T>::getrf(g_handle, n, n, P<T>(Sn), n, P<T>(work), nullptr, &info) == OK && info == 3,
          "singular unpivoted getrf devInfo == 3");
    Mat<Hs> N = eye<Hs>(n);
    N(1, 1) = Hs(-1);  // not positive definite: potrf info == 2
    CHECK(Api<T>::potrf(g_handle, CUBLAS_FILL_MODE_LOWER, n, P<T>(N), n, P<T>(work), n, &info) == OK &&
              info == 2,
          "non-SPD potrf must be SUCCESS with devInfo == 2");
    return true;
}

#define RUN_IRS(PP, T)                                                                         \
    if (!test_irs<T>(#PP, cusolverDn##PP##gesv_bufferSize, cusolverDn##PP##gesv,               \
                     cusolverDn##PP##gels_bufferSize, cusolverDn##PP##gels))                   \
        return 1;

// ── Sparse ──────────────────────────────────────────────────────────────────

struct Csr {
    std::vector<int> rowPtr, colInd;
};

// Tridiagonal pattern, n x n.
static Csr tridiag_pattern(int n) {
    Csr c;
    c.rowPtr.push_back(0);
    for (int i = 0; i < n; ++i) {
        if (i > 0) c.colInd.push_back(i - 1);
        c.colInd.push_back(i);
        if (i + 1 < n) c.colInd.push_back(i + 1);
        c.rowPtr.push_back(static_cast<int>(c.colInd.size()));
    }
    return c;
}

// Hermitian tridiagonal: diag d, super-diagonal u, sub-diagonal conj(u).
template <class T> static std::vector<H<T>> tridiag_values(int n, H<T> d, H<T> u) {
    std::vector<H<T>> v;
    for (int i = 0; i < n; ++i) {
        if (i > 0) v.push_back(cjg(u));
        v.push_back(d);
        if (i + 1 < n) v.push_back(u);
    }
    return v;
}

template <class T>
static std::vector<H<T>> csr_matvec(const Csr& c, const std::vector<H<T>>& v,
                                    const std::vector<H<T>>& x) {
    const int n = static_cast<int>(c.rowPtr.size()) - 1;
    std::vector<H<T>> y(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i)
        for (int j = c.rowPtr[i]; j < c.rowPtr[i + 1]; ++j)
            y[static_cast<size_t>(i)] += v[static_cast<size_t>(j)] * x[static_cast<size_t>(c.colInd[j])];
    return y;
}

template <class Fn, class T>
static bool test_sp_solve(const char* name, Fn fn) {
    using Hs = H<T>;
    const int n = 6;
    cusolverSpHandle_t sp = nullptr;
    CHECK(cusolverSpCreate(&sp) == OK, name);
    cusparseMatDescr_t descr = nullptr;
    CHECK(cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS, "cusparseCreateMatDescr");
    Csr c = tridiag_pattern(n);
    // Hermitian tridiagonal, eigenvalues >= 4 - 2*sqrt(1.25) > 0: positive definite.
    std::vector<Hs> v = tridiag_values<T>(n, Hs(4, 0), Hs(-1, Rl<T>(0.5)));
    std::vector<Hs> b(n), x(n);
    for (int i = 0; i < n; ++i) b[i] = entry<T>(i, 0, 70);
    int sing = -2;
    CHECK(fn(sp, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), reinterpret_cast<const T*>(b.data()),
             static_cast<Rl<T>>(1e-9), 0, reinterpret_cast<T*>(x.data()), &sing) == OK &&
              sing == -1,
          "sparse solve");
    std::vector<Hs> ax = csr_matvec<T>(c, v, x);
    double err = 0;
    for (int i = 0; i < n; ++i) err += std::pow(cabs2(ax[i] - b[i]), 2);
    CHECK(std::sqrt(err) < TT<T>::tol * 100, "sparse solve: A x != b");
    CHECK(fn(nullptr, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), reinterpret_cast<const T*>(b.data()),
             static_cast<Rl<T>>(1e-9), 0, reinterpret_cast<T*>(x.data()), &sing) ==
              CUSOLVER_STATUS_NOT_INITIALIZED,
          "sparse solve null handle");
    CHECK(fn(sp, n, static_cast<int>(v.size()) - 1, descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), reinterpret_cast<const T*>(b.data()),
             static_cast<Rl<T>>(1e-9), 0, reinterpret_cast<T*>(x.data()), &sing) ==
              CUSOLVER_STATUS_INVALID_VALUE,
          "sparse solve inconsistent nnz");
    cusparseDestroyMatDescr(descr);
    cusolverSpDestroy(sp);
    return true;
}

template <class T, class Fn> static bool test_eigvsi(const char* name, Fn fn) {
    using Hs = H<T>;
    const int n = 6;
    cusolverSpHandle_t sp = nullptr;
    CHECK(cusolverSpCreate(&sp) == OK, name);
    cusparseMatDescr_t descr = nullptr;
    CHECK(cusparseCreateMatDescr(&descr) == CUSPARSE_STATUS_SUCCESS, "cusparseCreateMatDescr");
    Csr c = tridiag_pattern(n);
    // Real: 1-D Laplacian, eigenvalues 2 - 2 cos(k pi / 7). Complex: same with
    // |off-diagonal| = sqrt(1.25), eigenvalues 2 - 2 sqrt(1.25) cos(k pi / 7).
    const double pi = std::acos(-1.0);
    const double off = TT<T>::cx ? std::sqrt(1.25) : 1.0;
    std::vector<Hs> v;
    if constexpr (TT<T>::cx) v = tridiag_values<T>(n, Hs(2, 0), Hs(-1, Rl<T>(0.5)));
    else v = tridiag_values<T>(n, Hs(2), Hs(-1));
    const double expect = 2.0 - 2.0 * off * std::cos(pi / 7.0);  // smallest eigenvalue
    std::vector<Hs> x0(n, Hs(1)), x(n);
    for (int i = 0; i < n; ++i) x0[i] = entry<T>(i, 0, 80);
    H<T> mu{};
    const Hs mu0 = Hs(static_cast<Rl<T>>(expect + 0.04));
    const Rl<T> eps = static_cast<Rl<T>>(TT<T>::tol * 1e-2);
    CHECK(fn(sp, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), *reinterpret_cast<const T*>(&mu0),
             reinterpret_cast<const T*>(x0.data()), 200, eps, reinterpret_cast<T*>(&mu),
             reinterpret_cast<T*>(x.data())) == OK,
          "csreigvsi");
    CHECK(std::fabs(cabs2(mu) - std::fabs(expect)) < TT<T>::tol * 100 &&
              std::fabs(std::real(mu) - expect) < TT<T>::tol * 100,
          "csreigvsi: eigenvalue is not the one nearest the shift");
    std::vector<Hs> ax = csr_matvec<T>(c, v, x);
    double res = 0, nx = 0;
    for (int i = 0; i < n; ++i) {
        res += std::pow(cabs2(ax[i] - mu * x[i]), 2);
        nx += std::pow(cabs2(x[i]), 2);
    }
    CHECK(std::sqrt(res) < TT<T>::tol * 100, "csreigvsi: A x != mu x");
    CHECK(std::fabs(std::sqrt(nx) - 1.0) < TT<T>::tol * 100, "csreigvsi: x not unit norm");
    // Negative paths.
    CHECK(fn(sp, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), *reinterpret_cast<const T*>(&mu0),
             reinterpret_cast<const T*>(x0.data()), 0, eps, reinterpret_cast<T*>(&mu),
             reinterpret_cast<T*>(x.data())) == CUSOLVER_STATUS_INVALID_VALUE,
          "csreigvsi maxite = 0");
    std::vector<Hs> zeros(n);
    CHECK(fn(sp, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), *reinterpret_cast<const T*>(&mu0),
             reinterpret_cast<const T*>(zeros.data()), 50, eps, reinterpret_cast<T*>(&mu),
             reinterpret_cast<T*>(x.data())) == CUSOLVER_STATUS_INVALID_VALUE,
          "csreigvsi zero initial vector");
    CHECK(fn(nullptr, n, static_cast<int>(v.size()), descr, reinterpret_cast<const T*>(v.data()),
             c.rowPtr.data(), c.colInd.data(), *reinterpret_cast<const T*>(&mu0),
             reinterpret_cast<const T*>(x0.data()), 50, eps, reinterpret_cast<T*>(&mu),
             reinterpret_cast<T*>(x.data())) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "csreigvsi null handle");
    cusparseDestroyMatDescr(descr);
    cusolverSpDestroy(sp);
    return true;
}

static bool test_sp_stream() {
    cusolverSpHandle_t sp = nullptr;
    CHECK(cusolverSpCreate(&sp) == OK, "cusolverSpCreate");
    cudaStream_t s = nullptr;
    CHECK(cudaStreamCreate(&s) == cudaSuccess, "cudaStreamCreate");
    cudaStream_t got = reinterpret_cast<cudaStream_t>(0x1);
    CHECK(cusolverSpGetStream(sp, &got) == OK && got == nullptr, "SpGetStream default");
    CHECK(cusolverSpSetStream(sp, s) == OK && cusolverSpGetStream(sp, &got) == OK && got == s,
          "SpGetStream round trip");
    CHECK(cusolverSpGetStream(sp, nullptr) == CUSOLVER_STATUS_INVALID_VALUE, "SpGetStream null out");
    CHECK(cusolverSpGetStream(nullptr, &got) == CUSOLVER_STATUS_NOT_INITIALIZED,
          "SpGetStream null handle");
    cusolverSpDestroy(sp);
    cudaStreamDestroy(s);
    return true;
}

#define RUN(expr)                  \
    do {                           \
        if (!(expr)) return 1;     \
    } while (0)

int main() {
    if (cusolverDnCreate(&g_handle) != OK) {
        std::fprintf(stderr, "FAIL: cusolverDnCreate\n");
        return 1;
    }

    RUN(test_lu<float>("S getrf"));
    RUN(test_lu<double>("D getrf"));
    RUN(test_lu<cuComplex>("C getrf"));
    RUN(test_lu<cuDoubleComplex>("Z getrf"));
    RUN(test_lu_nopivot_and_singular<float>("S unpivoted"));
    RUN(test_lu_nopivot_and_singular<double>("D unpivoted"));
    RUN(test_lu_nopivot_and_singular<cuComplex>("C unpivoted"));
    RUN(test_lu_nopivot_and_singular<cuDoubleComplex>("Z unpivoted"));
    RUN(test_lu_complex_extras<cuComplex>("C getrf extras"));
    RUN(test_lu_complex_extras<cuDoubleComplex>("Z getrf extras"));

    RUN(test_qr<float>("S geqrf/orgqr/ormqr"));
    RUN(test_qr<double>("D geqrf/orgqr/ormqr"));
    RUN(test_qr<cuComplex>("C geqrf/ungqr/unmqr"));
    RUN(test_qr<cuDoubleComplex>("Z geqrf/ungqr/unmqr"));

    RUN(test_potrf_batched<float>("S potrfBatched"));
    RUN(test_potrf_batched<double>("D potrfBatched"));
    RUN(test_potrf_batched<cuComplex>("C potrfBatched"));
    RUN(test_potrf_batched<cuDoubleComplex>("Z potrfBatched"));
    RUN(test_potrf_complex<cuComplex>("C potrf/potrs"));
    RUN(test_potrf_complex<cuDoubleComplex>("Z potrf/potrs"));

    RUN(test_sytrf<float>("S sytrf"));
    RUN(test_sytrf<double>("D sytrf"));
    RUN(test_sytrf<cuComplex>("C sytrf"));
    RUN(test_sytrf<cuDoubleComplex>("Z sytrf"));

    RUN(test_gebrd<float>("S gebrd"));
    RUN(test_gebrd<double>("D gebrd"));
    RUN(test_gebrd<cuComplex>("C gebrd"));
    RUN(test_gebrd<cuDoubleComplex>("Z gebrd"));

    RUN(test_gesvd_complex<cuComplex>("C gesvd", cusolverDnCgesvd_bufferSize, cusolverDnCgesvd));
    RUN(test_gesvd_complex<cuDoubleComplex>("Z gesvd", cusolverDnZgesvd_bufferSize, cusolverDnZgesvd));
    RUN(test_heevd<cuComplex>("C heevd", cusolverDnCheevd_bufferSize, cusolverDnCheevd));
    RUN(test_heevd<cuDoubleComplex>("Z heevd", cusolverDnZheevd_bufferSize, cusolverDnZheevd));

    RUN(test_evj<float>("S syevj"));
    RUN(test_evj<double>("D syevj"));
    RUN(test_evj<cuComplex>("C heevj"));
    RUN(test_evj<cuDoubleComplex>("Z heevj"));
    RUN(test_heevj_batched<cuComplex>("C heevjBatched", cusolverDnCheevjBatched_bufferSize,
                                      cusolverDnCheevjBatched));
    RUN(test_heevj_batched<cuDoubleComplex>("Z heevjBatched", cusolverDnZheevjBatched_bufferSize,
                                            cusolverDnZheevjBatched));

    RUN(test_gesvdj<float>("S gesvdj"));
    RUN(test_gesvdj<double>("D gesvdj"));
    RUN(test_gesvdj<cuComplex>("C gesvdj"));
    RUN(test_gesvdj<cuDoubleComplex>("Z gesvdj"));
    RUN(test_gesvdj_batched<float>("S gesvdjBatched"));
    RUN(test_gesvdj_batched<double>("D gesvdjBatched"));
    RUN(test_gesvdj_batched<cuComplex>("C gesvdjBatched"));
    RUN(test_gesvdj_batched<cuDoubleComplex>("Z gesvdjBatched"));
    RUN(test_gesvda<float>("S gesvda"));
    RUN(test_gesvda<double>("D gesvda"));
    RUN(test_gesvda<cuComplex>("C gesvda"));
    RUN(test_gesvda<cuDoubleComplex>("Z gesvda"));

    RUN_IRS(DD, double)
    RUN_IRS(DS, double)
    RUN_IRS(DH, double)
    RUN_IRS(DX, double)
    RUN_IRS(SS, float)
    RUN_IRS(SH, float)
    RUN_IRS(SX, float)
    RUN_IRS(ZZ, cuDoubleComplex)
    RUN_IRS(ZC, cuDoubleComplex)
    RUN_IRS(ZK, cuDoubleComplex)
    RUN_IRS(ZY, cuDoubleComplex)
    RUN_IRS(CC, cuComplex)
    RUN_IRS(CK, cuComplex)
    RUN_IRS(CY, cuComplex)

    RUN(test_sp_stream());
    RUN((test_sp_solve<decltype(&cusolverSpCcsrlsvchol), cuComplex>("C csrlsvchol",
                                                                    cusolverSpCcsrlsvchol)));
    RUN((test_sp_solve<decltype(&cusolverSpZcsrlsvchol), cuDoubleComplex>(
        "Z csrlsvchol", cusolverSpZcsrlsvchol)));
    RUN((test_sp_solve<decltype(&cusolverSpCcsrlsvqr), cuComplex>("C csrlsvqr",
                                                                  cusolverSpCcsrlsvqr)));
    RUN((test_sp_solve<decltype(&cusolverSpZcsrlsvqr), cuDoubleComplex>(
        "Z csrlsvqr", cusolverSpZcsrlsvqr)));
    RUN(test_eigvsi<float>("S csreigvsi", cusolverSpScsreigvsi));
    RUN(test_eigvsi<double>("D csreigvsi", cusolverSpDcsreigvsi));
    RUN(test_eigvsi<cuComplex>("C csreigvsi", cusolverSpCcsreigvsi));
    RUN(test_eigvsi<cuDoubleComplex>("Z csreigvsi", cusolverSpZcsreigvsi));

    cusolverDnDestroy(g_handle);
    std::printf("PASS: cuSOLVER CuPy surface tests\n");
    return 0;
}
