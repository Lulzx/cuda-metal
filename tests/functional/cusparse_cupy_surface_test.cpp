// Covers the cuSPARSE surface CuPy 14 links against: format conversion and
// sorting, nnz counting and compression, csrgeam2, csrilu02/csric02 with zero
// pivot reporting, the gtsv/gpsv solvers, sparse vectors, descriptor accessors,
// SpVV/Gather/SpSM/SpGEMM and dense<->sparse conversion.
//
// Every result is checked against a dense reference computed here, or against a
// defining property of the factorization (for ILU(0): (L*U)(i,j) == A(i,j) on
// the sparsity pattern). Each feature also has a negative path.

#include "cusparse.h"
#include "cuda_runtime.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond, msg)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL: %s (line %d)\n", (msg), __LINE__);    \
            return false;                                                     \
        }                                                                     \
    } while (0)

// ── NVIDIA's public enum values and version macros ──────────────────────────
static_assert(CUSPARSE_ACTION_SYMBOLIC == 0 && CUSPARSE_ACTION_NUMERIC == 1, "Action");
static_assert(CUSPARSE_DIRECTION_ROW == 0 && CUSPARSE_DIRECTION_COLUMN == 1, "Direction");
static_assert(CUSPARSE_FORMAT_CSR == 1 && CUSPARSE_FORMAT_CSC == 2 && CUSPARSE_FORMAT_COO == 3,
              "Format");
static_assert(CUSPARSE_CSR2CSC_ALG1 == 1, "Csr2CscAlg");
static_assert(CUSPARSE_SPSM_ALG_DEFAULT == 0, "SpSMAlg");
static_assert(CUSPARSE_SPGEMM_DEFAULT == 0 && CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC == 1 &&
                  CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC == 2 && CUSPARSE_SPGEMM_ALG1 == 3 &&
                  CUSPARSE_SPGEMM_ALG2 == 4 && CUSPARSE_SPGEMM_ALG3 == 5,
              "SpGEMMAlg");
static_assert(CUSPARSE_SPMV_COO_ALG1 == 1 && CUSPARSE_SPMV_CSR_ALG1 == 2 &&
                  CUSPARSE_SPMV_CSR_ALG2 == 3 && CUSPARSE_SPMV_COO_ALG2 == 4,
              "SpMVAlg");
static_assert(CUSPARSE_SPMM_COO_ALG1 == 1 && CUSPARSE_SPMM_CSR_ALG1 == 4 &&
                  CUSPARSE_SPMM_CSR_ALG2 == 6 && CUSPARSE_SPMM_CSR_ALG3 == 12,
              "SpMMAlg");
static_assert(CUSPARSE_VERSION == 12000, "CUDA 12.0 era cuSPARSE");

typedef std::complex<double> cplx;

static bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }
static bool near(const cplx& a, const cplx& b, double tol = 1e-9) { return std::abs(a - b) <= tol; }

// Dense (row-major) image of a CSR matrix.
template <typename T>
static std::vector<T> csr_dense(int m, int n, const int* ptr, const int* col, const T* val,
                                int base) {
    std::vector<T> d(static_cast<size_t>(m) * n, T(0));
    for (int i = 0; i < m; ++i)
        for (int k = ptr[i] - base; k < ptr[i + 1] - base; ++k)
            d[static_cast<size_t>(i) * n + (col[k] - base)] += val[k];
    return d;
}

static bool sorted_unique(const int* ptr, const int* col, int m, int base) {
    for (int i = 0; i < m; ++i)
        for (int k = ptr[i] - base + 1; k < ptr[i + 1] - base; ++k)
            if (col[k] <= col[k - 1]) return false;
    return true;
}

template <typename T>
static bool dense_near(const std::vector<T>& a, const std::vector<T>& b, double tol = 1e-9) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (!near(a[i], b[i], tol)) return false;
    return true;
}

// ── conversion and sorting ──────────────────────────────────────────────────

static bool test_coo_csr_conversion() {
    cusparseHandle_t h = nullptr;
    CHECK(cusparseCreate(&h) == CUSPARSE_STATUS_SUCCESS, "create");
    const int coo[] = {0, 0, 1, 3, 3, 3};
    int ptr[5] = {};
    CHECK(cusparseXcoo2csr(h, coo, 6, 4, ptr, CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS,
          "coo2csr");
    const int want[] = {0, 2, 3, 3, 6};
    for (int i = 0; i < 5; ++i) CHECK(ptr[i] == want[i], "coo2csr row pointer, base 0");

    int coo1[6];
    for (int i = 0; i < 6; ++i) coo1[i] = coo[i] + 1;
    int ptr1[5] = {};
    CHECK(cusparseXcoo2csr(h, coo1, 6, 4, ptr1, CUSPARSE_INDEX_BASE_ONE) == CUSPARSE_STATUS_SUCCESS,
          "coo2csr base 1");
    for (int i = 0; i < 5; ++i) CHECK(ptr1[i] == want[i] + 1, "coo2csr row pointer, base 1");

    int back[6] = {};
    CHECK(cusparseXcsr2coo(h, ptr, 6, 4, back, CUSPARSE_INDEX_BASE_ZERO) == CUSPARSE_STATUS_SUCCESS,
          "csr2coo");
    for (int i = 0; i < 6; ++i) CHECK(back[i] == coo[i], "csr2coo round trip, base 0");
    CHECK(cusparseXcsr2coo(h, ptr1, 6, 4, back, CUSPARSE_INDEX_BASE_ONE) == CUSPARSE_STATUS_SUCCESS,
          "csr2coo base 1");
    for (int i = 0; i < 6; ++i) CHECK(back[i] == coo1[i], "csr2coo round trip, base 1");

    // Negative paths.
    const int bad_row[] = {0, 4};
    CHECK(cusparseXcoo2csr(h, bad_row, 2, 4, ptr, CUSPARSE_INDEX_BASE_ZERO) ==
              CUSPARSE_STATUS_INVALID_VALUE, "row index >= m rejected");
    CHECK(cusparseXcoo2csr(h, coo, 6, -1, ptr, CUSPARSE_INDEX_BASE_ZERO) ==
              CUSPARSE_STATUS_INVALID_VALUE, "negative m rejected");
    CHECK(cusparseXcoo2csr(h, coo, 6, 4, nullptr, CUSPARSE_INDEX_BASE_ZERO) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null output rejected");
    CHECK(cusparseXcoo2csr(h, coo, 6, 4, ptr, static_cast<cusparseIndexBase_t>(2)) ==
              CUSPARSE_STATUS_INVALID_VALUE, "bad index base rejected");
    CHECK(cusparseXcoo2csr(nullptr, coo, 6, 4, ptr, CUSPARSE_INDEX_BASE_ZERO) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle rejected");
    const int decreasing[] = {0, 3, 2, 3, 6};
    CHECK(cusparseXcsr2coo(h, decreasing, 6, 4, back, CUSPARSE_INDEX_BASE_ZERO) ==
              CUSPARSE_STATUS_INVALID_VALUE, "malformed row pointer rejected");
    cusparseDestroy(h);
    return true;
}

static bool test_sort_and_permutation() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    int p[4] = {};
    CHECK(cusparseCreateIdentityPermutation(h, 4, p) == CUSPARSE_STATUS_SUCCESS, "identity");
    for (int i = 0; i < 4; ++i) CHECK(p[i] == i, "identity values");
    CHECK(cusparseCreateIdentityPermutation(h, -1, p) == CUSPARSE_STATUS_INVALID_VALUE, "n < 0");
    CHECK(cusparseCreateIdentityPermutation(h, 4, nullptr) == CUSPARSE_STATUS_INVALID_VALUE, "null p");

    size_t bytes = 0;
    int rows[] = {2, 0, 1, 0};
    int cols[] = {1, 3, 0, 2};
    CHECK(cusparseXcoosort_bufferSizeExt(h, 3, 4, 4, rows, cols, &bytes) == CUSPARSE_STATUS_SUCCESS &&
              bytes > 0, "coosort buffer size");
    std::vector<unsigned char> buf(bytes);

    int r1[4], c1[4], P1[4];
    std::memcpy(r1, rows, sizeof(rows));
    std::memcpy(c1, cols, sizeof(cols));
    cusparseCreateIdentityPermutation(h, 4, P1);
    CHECK(cusparseXcoosortByRow(h, 3, 4, 4, r1, c1, P1, buf.data()) == CUSPARSE_STATUS_SUCCESS,
          "coosortByRow");
    const int wr[] = {0, 0, 1, 2}, wc[] = {2, 3, 0, 1}, wp[] = {3, 1, 2, 0};
    for (int i = 0; i < 4; ++i)
        CHECK(r1[i] == wr[i] && c1[i] == wc[i] && P1[i] == wp[i], "coosortByRow result");

    std::memcpy(r1, rows, sizeof(rows));
    std::memcpy(c1, cols, sizeof(cols));
    cusparseCreateIdentityPermutation(h, 4, P1);
    CHECK(cusparseXcoosortByColumn(h, 3, 4, 4, r1, c1, P1, buf.data()) == CUSPARSE_STATUS_SUCCESS,
          "coosortByColumn");
    const int cr[] = {1, 2, 0, 0}, cc[] = {0, 1, 2, 3}, cp[] = {2, 0, 3, 1};
    for (int i = 0; i < 4; ++i)
        CHECK(r1[i] == cr[i] && c1[i] == cc[i] && P1[i] == cp[i], "coosortByColumn result");

    // P is composed, not overwritten: sorting twice keeps values(P) consistent.
    double vals[] = {10, 20, 30, 40};
    std::memcpy(r1, rows, sizeof(rows));
    std::memcpy(c1, cols, sizeof(cols));
    cusparseCreateIdentityPermutation(h, 4, P1);
    cusparseXcoosortByRow(h, 3, 4, 4, r1, c1, P1, buf.data());
    CHECK(vals[P1[0]] == 40 && vals[P1[1]] == 20 && vals[P1[2]] == 30 && vals[P1[3]] == 10,
          "sorted values = values(P)");
    CHECK(cusparseXcoosortByRow(h, 3, 4, 4, r1, c1, nullptr, nullptr) == CUSPARSE_STATUS_SUCCESS,
          "P may be null");
    CHECK(cusparseXcoosortByRow(h, 3, 4, 4, nullptr, c1, nullptr, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null rows rejected");
    CHECK(cusparseXcoosortByRow(nullptr, 3, 4, 4, r1, c1, nullptr, nullptr) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle rejected");

    // csrsort / cscsort, base 0 and 1.
    cusparseMatDescr_t d0 = nullptr, d1 = nullptr;
    cusparseCreateMatDescr(&d0);
    cusparseCreateMatDescr(&d1);
    cusparseSetMatIndexBase(d1, CUSPARSE_INDEX_BASE_ONE);
    for (int base = 0; base <= 1; ++base) {
        const int ptr[] = {base, base + 3, base + 5};
        int ind[] = {2 + base, 0 + base, 1 + base, 1 + base, 0 + base};
        int P[5];
        cusparseCreateIdentityPermutation(h, 5, P);
        CHECK(cusparseXcsrsort_bufferSizeExt(h, 2, 3, 5, ptr, ind, &bytes) == CUSPARSE_STATUS_SUCCESS,
              "csrsort buffer size");
        CHECK(cusparseXcsrsort(h, 2, 3, 5, base ? d1 : d0, ptr, ind, P, buf.data()) ==
                  CUSPARSE_STATUS_SUCCESS, "csrsort");
        const int wi[] = {0, 1, 2, 0, 1}, wperm[] = {1, 2, 0, 4, 3};
        for (int i = 0; i < 5; ++i)
            CHECK(ind[i] == wi[i] + base && P[i] == wperm[i], "csrsort result");

        int ind2[] = {2 + base, 0 + base, 1 + base, 1 + base, 0 + base};
        CHECK(cusparseXcscsort_bufferSizeExt(h, 3, 2, 5, ptr, ind2, &bytes) == CUSPARSE_STATUS_SUCCESS,
              "cscsort buffer size");
        CHECK(cusparseXcscsort(h, 3, 2, 5, base ? d1 : d0, ptr, ind2, nullptr, buf.data()) ==
                  CUSPARSE_STATUS_SUCCESS, "cscsort");
        for (int i = 0; i < 5; ++i) CHECK(ind2[i] == wi[i] + base, "cscsort result");
    }
    const int ptr[] = {0, 3, 5};
    int too_big[] = {2, 0, 7, 1, 0};
    CHECK(cusparseXcsrsort(h, 2, 3, 5, d0, ptr, too_big, nullptr, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "column index >= n rejected");
    CHECK(cusparseXcsrsort(h, 2, 3, 5, nullptr, ptr, too_big, nullptr, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null descriptor rejected");
    cusparseDestroyMatDescr(d0);
    cusparseDestroyMatDescr(d1);
    cusparseDestroy(h);
    return true;
}

static bool test_nnz_and_compress() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    cusparseMatDescr_t d = nullptr;
    cusparseCreateMatDescr(&d);

    // 3x3 dense, column-major, lda = 4:  [1 0 2; 0 0 3; 0 0 0]
    double A[12] = {};
    A[0 + 0 * 4] = 1;
    A[0 + 2 * 4] = 2;
    A[1 + 2 * 4] = 3;
    int per_row[3] = {}, per_col[3] = {}, total = -1;
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_ROW, 3, 3, d, A, 4, per_row, &total) ==
              CUSPARSE_STATUS_SUCCESS && total == 3 && per_row[0] == 2 && per_row[1] == 1 &&
              per_row[2] == 0, "Dnnz by row");
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_COLUMN, 3, 3, d, A, 4, per_col, &total) ==
              CUSPARSE_STATUS_SUCCESS && total == 3 && per_col[0] == 1 && per_col[1] == 0 &&
              per_col[2] == 2, "Dnnz by column");
    float Af[12] = {};
    Af[1] = 5.0f;
    CHECK(cusparseSnnz(h, CUSPARSE_DIRECTION_ROW, 3, 3, d, Af, 4, per_row, &total) ==
              CUSPARSE_STATUS_SUCCESS && total == 1 && per_row[1] == 1, "Snnz");
    cuDoubleComplex Z[4] = {};
    Z[3] = make_cuDoubleComplex(0.0, 2.0);  // imaginary-only entries are nonzero
    CHECK(cusparseZnnz(h, CUSPARSE_DIRECTION_ROW, 2, 2, d, Z, 2, per_row, &total) ==
              CUSPARSE_STATUS_SUCCESS && total == 1 && per_row[1] == 1, "Znnz imaginary-only");
    cuComplex Cc[4] = {};
    Cc[0] = make_cuComplex(0.0f, -1.0f);
    CHECK(cusparseCnnz(h, CUSPARSE_DIRECTION_ROW, 2, 2, d, Cc, 2, per_row, &total) ==
              CUSPARSE_STATUS_SUCCESS && total == 1 && per_row[0] == 1, "Cnnz");
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_ROW, 3, 3, d, A, 2, per_row, &total) ==
              CUSPARSE_STATUS_INVALID_VALUE, "lda < m rejected");
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_ROW, -1, 3, d, A, 4, per_row, &total) ==
              CUSPARSE_STATUS_INVALID_VALUE, "negative m rejected");
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_ROW, 3, 3, d, A, 4, per_row, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null total rejected");
    CHECK(cusparseDnnz(h, static_cast<cusparseDirection_t>(5), 3, 3, d, A, 4, per_row, &total) ==
              CUSPARSE_STATUS_INVALID_VALUE, "bad direction rejected");
    CHECK(cusparseDnnz(nullptr, CUSPARSE_DIRECTION_ROW, 3, 3, d, A, 4, per_row, &total) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle rejected");
    cusparseSetMatType(d, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    CHECK(cusparseDnnz(h, CUSPARSE_DIRECTION_ROW, 3, 3, d, A, 4, per_row, &total) ==
              CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "non-general type rejected");
    cusparseSetMatType(d, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Compression, base 0 then base 1.
    for (int base = 0; base <= 1; ++base) {
        cusparseSetMatIndexBase(d, base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO);
        const int ptr[] = {base, base + 3, base + 5, base + 6};
        const int col[] = {base + 0, base + 1, base + 2, base + 0, base + 1, base + 2};
        const double val[] = {1.0, 0.001, -2.0, 0.0, 0.5, 0.02};
        int nnz_row[3] = {}, nnz_c = -1;
        CHECK(cusparseDnnz_compress(h, 3, d, val, ptr, nnz_row, &nnz_c, 0.01) ==
                  CUSPARSE_STATUS_SUCCESS && nnz_c == 4 && nnz_row[0] == 2 && nnz_row[1] == 1 &&
                  nnz_row[2] == 1, "Dnnz_compress");
        double out_val[4] = {};
        int out_col[4] = {}, out_ptr[4] = {};
        CHECK(cusparseDcsr2csr_compress(h, 3, 3, d, val, col, ptr, 6, nnz_row, out_val, out_col,
                                        out_ptr, 0.01) == CUSPARSE_STATUS_SUCCESS,
              "Dcsr2csr_compress");
        const double wv[] = {1.0, -2.0, 0.5, 0.02};
        const int wc[] = {0, 2, 1, 2}, wptr[] = {0, 2, 3, 4};
        for (int i = 0; i < 4; ++i)
            CHECK(out_val[i] == wv[i] && out_col[i] == wc[i] + base, "compress values and columns");
        for (int i = 0; i < 4; ++i) CHECK(out_ptr[i] == wptr[i] + base, "compress row pointer");
        int wrong[3] = {2, 2, 0};
        CHECK(cusparseDcsr2csr_compress(h, 3, 3, d, val, col, ptr, 6, wrong, out_val, out_col,
                                        out_ptr, 0.01) == CUSPARSE_STATUS_INVALID_VALUE,
              "nnzPerRow that disagrees with tol is rejected");
    }
    // Complex tolerance compares magnitudes.
    const cuDoubleComplex zv[] = {make_cuDoubleComplex(0.0, 0.5), make_cuDoubleComplex(0.001, 0.001)};
    const int zptr[] = {0, 2}, zcol[] = {0, 1};
    int znnz_row[1] = {}, znnz = -1;
    cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
    CHECK(cusparseZnnz_compress(h, 1, d, zv, zptr, znnz_row, &znnz, make_cuDoubleComplex(0.01, 0.0)) ==
              CUSPARSE_STATUS_SUCCESS && znnz == 1, "Znnz_compress");
    cuDoubleComplex zout[1];
    int zocol[1], zoptr[2];
    CHECK(cusparseZcsr2csr_compress(h, 1, 2, d, zv, zcol, zptr, 2, znnz_row, zout, zocol, zoptr,
                                    make_cuDoubleComplex(0.01, 0.0)) == CUSPARSE_STATUS_SUCCESS &&
              zout[0].y == 0.5 && zocol[0] == 0, "Zcsr2csr_compress");
    cusparseDestroyMatDescr(d);
    cusparseDestroy(h);
    return true;
}

static bool test_csr2csc() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    for (int base = 0; base <= 1; ++base) {
        const int ptr[] = {base, base + 2, base + 4, base + 6};
        const int col[] = {base + 0, base + 2, base + 1, base + 3, base + 0, base + 2};
        const double val[] = {1, 2, 3, 4, 5, 6};
        int cptr[5] = {}, crow[6] = {};
        double cval[6] = {};
        const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
        size_t bytes = 0;
        CHECK(cusparseCsr2cscEx2_bufferSize(h, 3, 4, 6, val, ptr, col, cval, cptr, crow, CUDA_R_64F,
                                            CUSPARSE_ACTION_NUMERIC, ib, CUSPARSE_CSR2CSC_ALG1,
                                            &bytes) == CUSPARSE_STATUS_SUCCESS && bytes > 0,
              "csr2cscEx2 buffer size");
        std::vector<unsigned char> buf(bytes);
        CHECK(cusparseCsr2cscEx2(h, 3, 4, 6, val, ptr, col, cval, cptr, crow, CUDA_R_64F,
                                 CUSPARSE_ACTION_NUMERIC, ib, CUSPARSE_CSR2CSC_ALG1, buf.data()) ==
                  CUSPARSE_STATUS_SUCCESS, "csr2cscEx2");
        const int wptr[] = {0, 2, 3, 5, 6}, wrow[] = {0, 2, 1, 0, 2, 1};
        const double wval[] = {1, 5, 3, 2, 6, 4};
        for (int i = 0; i < 5; ++i) CHECK(cptr[i] == wptr[i] + base, "csc column pointer");
        for (int i = 0; i < 6; ++i)
            CHECK(crow[i] == wrow[i] + base && cval[i] == wval[i], "csc rows and values");

        // Symbolic: structure only, values untouched.
        double sentinel[6];
        std::fill(sentinel, sentinel + 6, -7.0);
        int sptr[5] = {}, srow[6] = {};
        CHECK(cusparseCsr2cscEx2(h, 3, 4, 6, val, ptr, col, sentinel, sptr, srow, CUDA_R_64F,
                                 CUSPARSE_ACTION_SYMBOLIC, ib, CUSPARSE_CSR2CSC_ALG1, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS, "symbolic csr2cscEx2");
        for (int i = 0; i < 6; ++i)
            CHECK(srow[i] == wrow[i] + base && sentinel[i] == -7.0, "symbolic leaves values alone");
    }
    // Complex values travel as bytes.
    const int ptr[] = {0, 1, 2}, col[] = {1, 0};
    const cuDoubleComplex zv[] = {make_cuDoubleComplex(1, 2), make_cuDoubleComplex(3, 4)};
    cuDoubleComplex zo[2];
    int zp[3], zr[2];
    CHECK(cusparseCsr2cscEx2(h, 2, 2, 2, zv, ptr, col, zo, zp, zr, CUDA_C_64F,
                             CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, nullptr) == CUSPARSE_STATUS_SUCCESS &&
              zo[0].x == 3 && zo[0].y == 4 && zo[1].x == 1 && zo[1].y == 2, "complex csr2csc");
    CHECK(cusparseCsr2cscEx2(h, 2, 2, 2, zv, ptr, col, zo, zp, zr, CUDA_C_64F,
                             CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                             static_cast<cusparseCsr2CscAlg_t>(0), nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "bad algorithm rejected");
    const int badcol[] = {5, 0};
    CHECK(cusparseCsr2cscEx2(h, 2, 2, 2, zv, ptr, badcol, zo, zp, zr, CUDA_C_64F,
                             CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, nullptr) == CUSPARSE_STATUS_INVALID_VALUE,
          "column index out of range rejected");
    CHECK(cusparseCsr2cscEx2(nullptr, 2, 2, 2, zv, ptr, col, zo, zp, zr, CUDA_C_64F,
                             CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO,
                             CUSPARSE_CSR2CSC_ALG1, nullptr) == CUSPARSE_STATUS_NOT_INITIALIZED,
          "null handle rejected");
    cusparseDestroy(h);
    return true;
}

// ── csrgeam2 ────────────────────────────────────────────────────────────────

static bool test_csrgeam2() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    cusparseMatDescr_t dA = nullptr, dB = nullptr, dC = nullptr;
    cusparseCreateMatDescr(&dA);
    cusparseCreateMatDescr(&dB);
    cusparseCreateMatDescr(&dC);
    cusparseSetMatIndexBase(dB, CUSPARSE_INDEX_BASE_ONE);  // mixed bases
    cusparseSetMatIndexBase(dC, CUSPARSE_INDEX_BASE_ONE);

    // A = [1 0 2; 0 3 0; 4 0 5], B (base 1) = [0 6 0; 7 0 8; 0 0 9]
    const int pA[] = {0, 2, 3, 5}, cA[] = {0, 2, 1, 0, 2};
    const double vA[] = {1, 2, 3, 4, 5};
    const int pB[] = {1, 2, 4, 5}, cB[] = {2, 1, 3, 3};
    const double vB[] = {6, 7, 8, 9};
    const double alpha = 2.0, beta = -1.0;
    size_t bytes = 0;
    int pC[4] = {}, nnzC = -1;
    CHECK(cusparseDcsrgeam2_bufferSizeExt(h, 3, 3, &alpha, dA, 5, vA, pA, cA, &beta, dB, 4, vB, pB,
                                          cB, dC, nullptr, pC, nullptr, &bytes) ==
              CUSPARSE_STATUS_SUCCESS && bytes > 0, "buffer size");
    std::vector<unsigned char> buf(bytes);
    CHECK(cusparseXcsrgeam2Nnz(h, 3, 3, dA, 5, pA, cA, dB, 4, pB, cB, dC, pC, &nnzC, buf.data()) ==
              CUSPARSE_STATUS_SUCCESS, "Xcsrgeam2Nnz");
    CHECK(nnzC == 8 && pC[0] == 1 && pC[1] == 4 && pC[2] == 7 && pC[3] == 9, "union pattern, base 1");
    std::vector<double> vC(nnzC);
    std::vector<int> cC(nnzC);
    CHECK(cusparseDcsrgeam2(h, 3, 3, &alpha, dA, 5, vA, pA, cA, &beta, dB, 4, vB, pB, cB, dC,
                            vC.data(), pC, cC.data(), buf.data()) == CUSPARSE_STATUS_SUCCESS,
          "Dcsrgeam2");
    CHECK(sorted_unique(pC, cC.data(), 3, 1), "output columns sorted");
    const auto got = csr_dense<double>(3, 3, pC, cC.data(), vC.data(), 1);
    const std::vector<double> want = {2, -6, 4, -7, 6, -8, 8, 0, 1};
    CHECK(dense_near(got, want), "C = 2A - B");

    // Stale row pointers (from different operands) must be refused.
    int stale[4] = {1, 2, 3, 4};
    CHECK(cusparseDcsrgeam2(h, 3, 3, &alpha, dA, 5, vA, pA, cA, &beta, dB, 4, vB, pB, cB, dC,
                            vC.data(), stale, cC.data(), buf.data()) == CUSPARSE_STATUS_INVALID_VALUE,
          "mismatched csrRowPtrC rejected");
    CHECK(cusparseDcsrgeam2(h, 3, 3, nullptr, dA, 5, vA, pA, cA, &beta, dB, 4, vB, pB, cB, dC,
                            vC.data(), pC, cC.data(), buf.data()) == CUSPARSE_STATUS_INVALID_VALUE,
          "null alpha rejected");
    CHECK(cusparseDcsrgeam2(nullptr, 3, 3, &alpha, dA, 5, vA, pA, cA, &beta, dB, 4, vB, pB, cB, dC,
                            vC.data(), pC, cC.data(), buf.data()) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle rejected");
    CHECK(cusparseXcsrgeam2Nnz(h, -1, 3, dA, 5, pA, cA, dB, 4, pB, cB, dC, pC, &nnzC, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "negative m rejected");
    cusparseSetMatType(dA, CUSPARSE_MATRIX_TYPE_TRIANGULAR);
    CHECK(cusparseXcsrgeam2Nnz(h, 3, 3, dA, 5, pA, cA, dB, 4, pB, cB, dC, pC, &nnzC, nullptr) ==
              CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "non-general type rejected");
    cusparseSetMatType(dA, CUSPARSE_MATRIX_TYPE_GENERAL);

    // Float and complex variants, 2x2 identity-ish.
    {
        cusparseMatDescr_t z = nullptr;
        cusparseCreateMatDescr(&z);
        const int p[] = {0, 1, 2}, c[] = {0, 1};
        const cuDoubleComplex a[] = {make_cuDoubleComplex(1, 1), make_cuDoubleComplex(2, 0)};
        const cuDoubleComplex b[] = {make_cuDoubleComplex(0, 1), make_cuDoubleComplex(0, -2)};
        const cuDoubleComplex al = make_cuDoubleComplex(0, 1), be = make_cuDoubleComplex(2, 0);
        int pz[3], nz = 0;
        CHECK(cusparseXcsrgeam2Nnz(h, 2, 2, z, 2, p, c, z, 2, p, c, z, pz, &nz, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS && nz == 2, "complex pattern");
        cuDoubleComplex vz[2];
        int cz[2];
        CHECK(cusparseZcsrgeam2(h, 2, 2, &al, z, 2, a, p, c, &be, z, 2, b, p, c, z, vz, pz, cz,
                                nullptr) == CUSPARSE_STATUS_SUCCESS, "Zcsrgeam2");
        // i*(1+i) + 2*i = -1 + 3i ; i*2 + 2*(-2i) = -2i
        CHECK(near(vz[0].x, -1.0) && near(vz[0].y, 3.0) && near(vz[1].x, 0.0) && near(vz[1].y, -2.0),
              "complex alpha*A + beta*B");
        const float fa = 1.0f, fb = 1.0f;
        const float av[] = {1.0f, 2.0f}, bv[] = {3.0f, 4.0f};
        float fv[2];
        CHECK(cusparseScsrgeam2(h, 2, 2, &fa, z, 2, av, p, c, &fb, z, 2, bv, p, c, z, fv, pz, cz,
                                nullptr) == CUSPARSE_STATUS_SUCCESS && fv[0] == 4.0f && fv[1] == 6.0f,
              "Scsrgeam2");
        cusparseDestroyMatDescr(z);
    }
    cusparseDestroyMatDescr(dA);
    cusparseDestroyMatDescr(dB);
    cusparseDestroyMatDescr(dC);
    cusparseDestroy(h);
    return true;
}

// ── incomplete factorizations ───────────────────────────────────────────────

// (L*U)(i,j) with unit-lower L and upper U packed in `lu` (dense, row-major).
static double lu_entry(const std::vector<double>& lu, int n, int i, int j) {
    double s = 0;
    for (int k = 0; k <= std::min(i, j); ++k) {
        const double l = (k == i) ? 1.0 : lu[static_cast<size_t>(i) * n + k];
        s += l * lu[static_cast<size_t>(k) * n + j];
    }
    return s;
}

static bool test_csrilu02() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    cusparseMatDescr_t d = nullptr;
    cusparseCreateMatDescr(&d);
    csrilu02Info_t info = nullptr;
    cusparseCreateCsrilu02Info(&info);

    // 4x4 with a sparse pattern: (LU)(i,j) must equal A(i,j) on the pattern.
    const int ptr0[] = {0, 3, 6, 9, 12};
    const int col0[] = {0, 1, 3, 0, 1, 2, 1, 2, 3, 0, 2, 3};
    const double a0[] = {4, 1, 1, 1, 4, 1, 1, 4, 1, 1, 1, 4};
    for (int base = 0; base <= 1; ++base) {
        cusparseSetMatIndexBase(d, base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO);
        int ptr[5], col[12];
        for (int i = 0; i < 5; ++i) ptr[i] = ptr0[i] + base;
        for (int i = 0; i < 12; ++i) col[i] = col0[i] + base;
        double val[12];
        std::memcpy(val, a0, sizeof(val));
        int bytes = 0, pos = 99;
        CHECK(cusparseDcsrilu02_bufferSize(h, 4, 12, d, val, ptr, col, info, &bytes) ==
                  CUSPARSE_STATUS_SUCCESS && bytes > 0, "bufferSize");
        std::vector<unsigned char> buf(bytes);
        CHECK(cusparseDcsrilu02_analysis(h, 4, 12, d, val, ptr, col, info,
                                         CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf.data()) ==
                  CUSPARSE_STATUS_SUCCESS, "analysis");
        CHECK(cusparseXcsrilu02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_SUCCESS && pos == -1,
              "no zero pivot after analysis");
        CHECK(cusparseDcsrilu02(h, 4, 12, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                buf.data()) == CUSPARSE_STATUS_SUCCESS, "factorization");
        pos = 99;
        CHECK(cusparseXcsrilu02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_SUCCESS && pos == -1,
              "zeroPivot reports -1 on success");
        const auto lu = csr_dense<double>(4, 4, ptr, col, val, base);
        const auto orig = csr_dense<double>(4, 4, ptr, col, a0, base);
        for (int i = 0; i < 4; ++i)
            for (int k = ptr[i] - base; k < ptr[i + 1] - base; ++k) {
                const int j = col[k] - base;
                CHECK(near(lu_entry(lu, 4, i, j), orig[static_cast<size_t>(i) * 4 + j], 1e-12),
                      "(L*U)(i,j) == A(i,j) on the pattern");
            }
        // The fill-in position (1,3)/(3,1) must stay absent.
        CHECK(lu[1 * 4 + 3] == 0.0 && lu[3 * 4 + 1] == 0.0, "no fill-in outside the pattern");
    }

    // Numerical zero pivot at row 1: [[1,1],[1,1]].
    cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
    for (int base = 0; base <= 1; ++base) {
        cusparseSetMatIndexBase(d, base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO);
        const int ptr[] = {base, base + 2, base + 4}, col[] = {base, base + 1, base, base + 1};
        double val[] = {1, 1, 1, 1};
        int pos = 99;
        cusparseDcsrilu02_analysis(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                   nullptr);
        CHECK(cusparseDcsrilu02(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                nullptr) == CUSPARSE_STATUS_SUCCESS,
              "factorization itself succeeds; the pivot is queried separately");
        CHECK(cusparseXcsrilu02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_ZERO_PIVOT &&
                  pos == 1 + base, "numerical zero pivot position in the matrix's base");
    }

    // Numeric boost replaces a tiny pivot instead of reporting it.
    {
        cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
        const int ptr[] = {0, 2, 4}, col[] = {0, 1, 0, 1};
        double val[] = {1, 1, 1, 1};
        double tol = 1e-6, boost = 0.25;
        int pos = 99;
        CHECK(cusparseDcsrilu02_numericBoost(h, info, 1, &tol, &boost) == CUSPARSE_STATUS_SUCCESS,
              "numericBoost enable");
        cusparseDcsrilu02_analysis(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                   nullptr);
        cusparseDcsrilu02(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, nullptr);
        CHECK(cusparseXcsrilu02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_SUCCESS && pos == -1 &&
                  val[3] == 0.25, "boosted pivot is neither reported nor left at zero");
        CHECK(cusparseDcsrilu02_numericBoost(h, info, 1, nullptr, &boost) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "null tol rejected");
        CHECK(cusparseDcsrilu02_numericBoost(h, info, 0, nullptr, nullptr) == CUSPARSE_STATUS_SUCCESS,
              "boost disabled needs no pointers");
    }

    // Structural zero (no diagonal in row 1) is visible straight after analysis.
    {
        cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
        const int ptr[] = {0, 2, 3}, col[] = {0, 1, 0};
        double val[] = {2, 1, 1};
        int pos = 99;
        cusparseDcsrilu02_analysis(h, 2, 3, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                   nullptr);
        CHECK(cusparseXcsrilu02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_ZERO_PIVOT && pos == 1,
              "structural zero reported by analysis");
    }

    // Complex.
    {
        const int ptr[] = {0, 2, 4}, col[] = {0, 1, 0, 1};
        const cplx a[] = {{2, 1}, {1, -1}, {0, 1}, {3, 0}};
        cuDoubleComplex val[4];
        std::memcpy(val, a, sizeof(val));
        cusparseZcsrilu02_analysis(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                   nullptr);
        CHECK(cusparseZcsrilu02(h, 2, 4, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                nullptr) == CUSPARSE_STATUS_SUCCESS, "Zcsrilu02");
        const cplx l10 = cplx(0, 1) / cplx(2, 1);
        const cplx u11 = cplx(3, 0) - l10 * cplx(1, -1);
        const cplx* r = reinterpret_cast<const cplx*>(val);
        CHECK(near(r[2], l10) && near(r[3], u11), "complex ILU(0) multipliers");
    }

    // Negative paths.
    {
        const int ptr[] = {0, 1}, col[] = {0};
        double val[] = {1};
        int bytes = 0, pos = 0;
        CHECK(cusparseDcsrilu02_bufferSize(nullptr, 1, 1, d, val, ptr, col, info, &bytes) ==
                  CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
        CHECK(cusparseDcsrilu02_bufferSize(h, -1, 1, d, val, ptr, col, info, &bytes) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "negative m");
        CHECK(cusparseDcsrilu02_bufferSize(h, 1, 1, d, val, ptr, col, nullptr, &bytes) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "null info");
        const int bad_ptr[] = {0, 2};
        CHECK(cusparseDcsrilu02(h, 1, 1, d, val, bad_ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                nullptr) == CUSPARSE_STATUS_INVALID_VALUE, "row pointer disagrees with nnz");
        CHECK(cusparseXcsrilu02_zeroPivot(h, nullptr, &pos) == CUSPARSE_STATUS_INVALID_VALUE,
              "zeroPivot null info");
        CHECK(cusparseXcsrilu02_zeroPivot(nullptr, info, &pos) == CUSPARSE_STATUS_NOT_INITIALIZED,
              "zeroPivot null handle");
    }
    cusparseDestroyCsrilu02Info(info);
    cusparseDestroyMatDescr(d);
    cusparseDestroy(h);
    return true;
}

static bool test_csric02() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    cusparseMatDescr_t d = nullptr;
    cusparseCreateMatDescr(&d);
    csric02Info_t info = nullptr;
    CHECK(cusparseCreateCsric02Info(&info) == CUSPARSE_STATUS_SUCCESS && info != nullptr, "create info");

    // Dense SPD 3x3: with a full pattern IC(0) is the exact Cholesky factor.
    const double A[3][3] = {{4, 2, 1}, {2, 5, 3}, {1, 3, 6}};
    for (int upper = 0; upper <= 1; ++upper) {
        cusparseSetMatFillMode(d, upper ? CUSPARSE_FILL_MODE_UPPER : CUSPARSE_FILL_MODE_LOWER);
        std::vector<int> ptr(1, 0), col;
        std::vector<double> val;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (upper ? j >= i : j <= i) {
                    col.push_back(j);
                    val.push_back(A[i][j]);
                }
            }
            ptr.push_back(static_cast<int>(col.size()));
        }
        const int nnz = static_cast<int>(col.size());
        int bytes = 0, pos = 99;
        CHECK(cusparseDcsric02_bufferSize(h, 3, nnz, d, val.data(), ptr.data(), col.data(), info,
                                          &bytes) == CUSPARSE_STATUS_SUCCESS && bytes > 0,
              "bufferSize");
        CHECK(cusparseDcsric02_analysis(h, 3, nnz, d, val.data(), ptr.data(), col.data(), info,
                                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS, "analysis");
        CHECK(cusparseDcsric02(h, 3, nnz, d, val.data(), ptr.data(), col.data(), info,
                               CUSPARSE_SOLVE_POLICY_NO_LEVEL, nullptr) == CUSPARSE_STATUS_SUCCESS,
              "factorization");
        CHECK(cusparseXcsric02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_SUCCESS && pos == -1,
              "no zero pivot");
        // Rebuild F (the factor as a full matrix) and verify F*F^T or F^T*F == A.
        double F[3][3] = {};
        for (int i = 0; i < 3; ++i)
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) F[i][col[k]] = val[k];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double s = 0;
                for (int k = 0; k < 3; ++k) s += upper ? F[k][i] * F[k][j] : F[i][k] * F[j][k];
                CHECK(near(s, A[i][j], 1e-12), upper ? "U^T U == A" : "L L^T == A");
            }
    }

    // Sparse lower pattern (no entry at (3,1)): (L L^T)(i,j) == A(i,j) on the pattern.
    {
        cusparseSetMatFillMode(d, CUSPARSE_FILL_MODE_LOWER);
        // Row 3 stores columns {0,3}: the pattern lacks (3,1) and (3,2).
        const int ptr2[] = {0, 1, 3, 5, 7};
        const int col2[] = {0, 0, 1, 1, 2, 0, 3};
        const double a[] = {4, 1, 4, 1, 4, 1, 4};
        double val[7];
        std::memcpy(val, a, sizeof(val));
        CHECK(cusparseDcsric02_analysis(h, 4, 7, d, val, ptr2, col2, info,
                                        CUSPARSE_SOLVE_POLICY_NO_LEVEL, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS, "sparse analysis");
        CHECK(cusparseDcsric02(h, 4, 7, d, val, ptr2, col2, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                               nullptr) == CUSPARSE_STATUS_SUCCESS, "sparse factorization");
        double L[4][4] = {};
        for (int i = 0; i < 4; ++i)
            for (int k = ptr2[i]; k < ptr2[i + 1]; ++k) L[i][col2[k]] = val[k];
        for (int i = 0; i < 4; ++i)
            for (int k = ptr2[i]; k < ptr2[i + 1]; ++k) {
                const int j = col2[k];
                double s = 0;
                for (int t = 0; t < 4; ++t) s += L[i][t] * L[j][t];
                CHECK(near(s, a[k], 1e-12), "(L L^T)(i,j) == A(i,j) on the pattern");
            }
    }

    // Not positive definite: [[1,2],[2,1]] -> pivot at row 1, base-aware.
    for (int base = 0; base <= 1; ++base) {
        cusparseSetMatIndexBase(d, base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO);
        cusparseSetMatFillMode(d, CUSPARSE_FILL_MODE_LOWER);
        const int ptr[] = {base, base + 1, base + 3}, col[] = {base, base, base + 1};
        double val[] = {1, 2, 1};
        int pos = 99;
        cusparseDcsric02_analysis(h, 2, 3, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                  nullptr);
        CHECK(cusparseDcsric02(h, 2, 3, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                               nullptr) == CUSPARSE_STATUS_SUCCESS, "factorization call succeeds");
        CHECK(cusparseXcsric02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_ZERO_PIVOT &&
                  pos == 1 + base, "non-positive pivot reported");
    }
    // Structural zero.
    {
        cusparseSetMatIndexBase(d, CUSPARSE_INDEX_BASE_ZERO);
        const int ptr[] = {0, 1, 2}, col[] = {0, 0};
        double val[] = {1, 2};
        int pos = 99;
        cusparseDcsric02_analysis(h, 2, 2, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                  nullptr);
        CHECK(cusparseXcsric02_zeroPivot(h, info, &pos) == CUSPARSE_STATUS_ZERO_PIVOT && pos == 1,
              "missing diagonal reported by analysis");
    }
    // Hermitian complex: [[4, 1+i],[1-i, 3]], lower stored.
    {
        cusparseSetMatFillMode(d, CUSPARSE_FILL_MODE_LOWER);
        const int ptr[] = {0, 1, 3}, col[] = {0, 0, 1};
        const cplx a[] = {{4, 0}, {1, -1}, {3, 0}};
        cuDoubleComplex val[3];
        std::memcpy(val, a, sizeof(val));
        cusparseZcsric02_analysis(h, 2, 3, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                                  nullptr);
        CHECK(cusparseZcsric02(h, 2, 3, d, val, ptr, col, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL,
                               nullptr) == CUSPARSE_STATUS_SUCCESS, "Zcsric02");
        const cplx* r = reinterpret_cast<const cplx*>(val);
        CHECK(near(r[0], cplx(2, 0)) && near(r[1], cplx(1, -1) / 2.0) &&
                  near(r[2], cplx(std::sqrt(2.5), 0)), "complex IC(0)");
    }
    CHECK(cusparseDcsric02_bufferSize(nullptr, 1, 1, d, nullptr, nullptr, nullptr, info, nullptr) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
    CHECK(cusparseDestroyCsric02Info(info) == CUSPARSE_STATUS_SUCCESS, "destroy info");
    cusparseDestroyMatDescr(d);
    cusparseDestroy(h);
    return true;
}

static bool test_bsr_factorizations_refuse() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    cusparseMatDescr_t d = nullptr;
    cusparseCreateMatDescr(&d);
    bsrilu02Info_t bi = nullptr;
    bsric02Info_t bc = nullptr;
    CHECK(cusparseCreateBsrilu02Info(&bi) == CUSPARSE_STATUS_SUCCESS && bi != nullptr, "bsrilu02 info");
    CHECK(cusparseCreateBsric02Info(&bc) == CUSPARSE_STATUS_SUCCESS && bc != nullptr, "bsric02 info");
    const int ptr[] = {0, 1}, col[] = {0};
    double val[] = {1, 0, 0, 1};
    int bytes = 0, pos = 0;
    // A stub that returned success would be a wrong answer; these must refuse.
    CHECK(cusparseDbsrilu02_bufferSize(h, CUSPARSE_DIRECTION_ROW, 1, 1, d, val, ptr, col, 2, bi,
                                       &bytes) == CUSPARSE_STATUS_NOT_SUPPORTED, "bsrilu02 refuses");
    CHECK(cusparseDbsric02(h, CUSPARSE_DIRECTION_ROW, 1, 1, d, val, ptr, col, 2, bc,
                           CUSPARSE_SOLVE_POLICY_NO_LEVEL, nullptr) == CUSPARSE_STATUS_NOT_SUPPORTED,
          "bsric02 refuses");
    CHECK(cusparseXbsrilu02_zeroPivot(h, bi, &pos) == CUSPARSE_STATUS_NOT_SUPPORTED, "bsrilu02 zeroPivot");
    CHECK(cusparseDbsrilu02_bufferSize(nullptr, CUSPARSE_DIRECTION_ROW, 1, 1, d, val, ptr, col, 2, bi,
                                       &bytes) == CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
    CHECK(cusparseDestroyBsrilu02Info(bi) == CUSPARSE_STATUS_SUCCESS &&
              cusparseDestroyBsric02Info(bc) == CUSPARSE_STATUS_SUCCESS, "destroy infos");
    cusparseDestroyMatDescr(d);
    cusparseDestroy(h);
    return true;
}

// ── gtsv / gpsv ─────────────────────────────────────────────────────────────

// y = T*x for a tridiagonal T (diagonals read at stride `ds`, x at stride `xs`).
template <typename T>
static T tri_apply(int m, const T* dl, const T* d, const T* du, const T* x, int i, int ds = 1,
                   int xs = 1) {
    T s = d[i * ds] * x[i * xs];
    if (i > 0) s += dl[i * ds] * x[(i - 1) * xs];
    if (i + 1 < m) s += du[i * ds] * x[(i + 1) * xs];
    return s;
}

static bool test_gtsv2() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    const int m = 6, n = 3, ldb = 8;
    // Diagonally dominant but unsymmetric; dl[0] and du[m-1] are junk that must be ignored.
    std::vector<double> dl = {99, 1, 2, 1, 0.5, 1}, d = {4, 5, 6, 5, 4, 7}, du = {1, 2, 1, 1, 3, 99};
    std::vector<double> B(ldb * n, -1.0), B0;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) B[i + j * ldb] = 1.0 + i + 10.0 * j;
    B0 = B;
    size_t bytes = 0;
    CHECK(cusparseDgtsv2_bufferSizeExt(h, m, n, dl.data(), d.data(), du.data(), B.data(), ldb,
                                       &bytes) == CUSPARSE_STATUS_SUCCESS && bytes > 0, "buffer size");
    std::vector<unsigned char> buf(bytes);
    const std::vector<double> dl0 = dl, d0 = d, du0 = du;
    CHECK(cusparseDgtsv2(h, m, n, dl.data(), d.data(), du.data(), B.data(), ldb, buf.data()) ==
              CUSPARSE_STATUS_SUCCESS, "Dgtsv2");
    CHECK(dl == dl0 && d == d0 && du == du0, "gtsv2 leaves the diagonals untouched");
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            CHECK(near(tri_apply(m, dl.data(), d.data(), du.data(), &B[j * ldb], i), B0[i + j * ldb], 1e-10),
                  "A*X == B");
    CHECK(B[m] == -1.0 && B[m + 1] == -1.0, "padding rows of B untouched");

    std::vector<double> Bn = B0;
    CHECK(cusparseDgtsv2_nopivot(h, m, n, dl.data(), d.data(), du.data(), Bn.data(), ldb, nullptr) ==
              CUSPARSE_STATUS_SUCCESS, "Dgtsv2_nopivot");
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            CHECK(near(Bn[i + j * ldb], B[i + j * ldb], 1e-10), "nopivot matches pivoted on a dominant system");
    CHECK(cusparseDgtsv2_nopivot_bufferSizeExt(h, m, n, dl.data(), d.data(), du.data(), Bn.data(), ldb,
                                               &bytes) == CUSPARSE_STATUS_SUCCESS && bytes > 0,
          "nopivot buffer size");

    // Needs pivoting: d[0] == 0.  [0 1 0; 1 2 1; 0 1 2] x = b
    {
        const double pdl[] = {0, 1, 1}, pd[] = {0, 2, 2}, pdu[] = {1, 1, 0};
        double b[] = {1, 4, 3}, b2[] = {1, 4, 3};
        CHECK(cusparseDgtsv2(h, 3, 1, pdl, pd, pdu, b, 3, nullptr) == CUSPARSE_STATUS_SUCCESS,
              "pivoting handles a zero leading diagonal");
        for (int i = 0; i < 3; ++i) CHECK(near(tri_apply(3, pdl, pd, pdu, b, i), b2[i]), "pivoted residual");
        CHECK(cusparseDgtsv2_nopivot(h, 3, 1, pdl, pd, pdu, b2, 3, nullptr) == CUSPARSE_STATUS_ZERO_PIVOT,
              "nopivot reports the zero pivot");
        const double z[] = {0, 0, 0};
        double bz[] = {1, 1, 1};
        CHECK(cusparseDgtsv2(h, 3, 1, z, z, z, bz, 3, nullptr) == CUSPARSE_STATUS_ZERO_PIVOT,
              "singular system reported");
    }
    // Single precision and complex.
    {
        const float fdl[] = {0, 1, 1}, fd[] = {4, 4, 4}, fdu[] = {1, 1, 0};
        float fb[] = {5, 6, 5};
        CHECK(cusparseSgtsv2(h, 3, 1, fdl, fd, fdu, fb, 3, nullptr) == CUSPARSE_STATUS_SUCCESS, "Sgtsv2");
        for (int i = 0; i < 3; ++i)
            CHECK(std::fabs(tri_apply(3, fdl, fd, fdu, fb, i) - (i == 1 ? 6.0f : 5.0f)) < 1e-5f, "float residual");
        const cplx zdl[] = {0, {1, 1}, {0, 1}}, zd[] = {{2, 1}, {3, 0}, {4, -1}}, zdu[] = {{1, 0}, {0, 2}, 0};
        const cplx zb0[] = {{1, 2}, {3, 4}, {5, 6}};
        cplx zb[3] = {zb0[0], zb0[1], zb0[2]};
        CHECK(cusparseZgtsv2(h, 3, 1, reinterpret_cast<const cuDoubleComplex*>(zdl),
                             reinterpret_cast<const cuDoubleComplex*>(zd),
                             reinterpret_cast<const cuDoubleComplex*>(zdu),
                             reinterpret_cast<cuDoubleComplex*>(zb), 3, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS, "Zgtsv2");
        for (int i = 0; i < 3; ++i) CHECK(near(tri_apply(3, zdl, zd, zdu, zb, i), zb0[i], 1e-10), "complex residual");
    }
    // Negative paths.
    CHECK(cusparseDgtsv2(h, m, n, dl.data(), d.data(), du.data(), B.data(), m - 1, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "ldb < m rejected");
    CHECK(cusparseDgtsv2(h, -1, n, dl.data(), d.data(), du.data(), B.data(), ldb, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "negative m rejected");
    CHECK(cusparseDgtsv2(h, m, n, nullptr, d.data(), du.data(), B.data(), ldb, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null dl rejected");
    CHECK(cusparseDgtsv2(nullptr, m, n, dl.data(), d.data(), du.data(), B.data(), ldb, nullptr) ==
              CUSPARSE_STATUS_NOT_INITIALIZED, "null handle rejected");
    CHECK(cusparseDgtsv2_bufferSizeExt(h, m, n, dl.data(), d.data(), du.data(), B.data(), ldb, nullptr) ==
              CUSPARSE_STATUS_INVALID_VALUE, "null size output rejected");
    cusparseDestroy(h);
    return true;
}

static bool test_gtsv2_strided_and_interleaved() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);

    // Strided batch: 3 systems of size 5, stride 8.
    {
        const int m = 5, batch = 3, stride = 8;
        std::vector<double> dl(batch * stride, 77), d(batch * stride, 77), du(batch * stride, 77),
            x(batch * stride, 77);
        std::vector<double> rhs(batch * stride, 77);
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < m; ++i) {
                dl[b * stride + i] = 0.5 + b + (i == 0 ? 50 : 0);
                d[b * stride + i] = 6.0 + i + b;
                du[b * stride + i] = 1.0 + 0.25 * i;
                x[b * stride + i] = rhs[b * stride + i] = 1.0 + i * (b + 1);
            }
        size_t bytes = 0;
        CHECK(cusparseDgtsv2StridedBatch_bufferSizeExt(h, m, dl.data(), d.data(), du.data(), x.data(),
                                                       batch, stride, &bytes) ==
                  CUSPARSE_STATUS_SUCCESS && bytes > 0, "strided buffer size");
        CHECK(cusparseDgtsv2StridedBatch(h, m, dl.data(), d.data(), du.data(), x.data(), batch, stride,
                                         nullptr) == CUSPARSE_STATUS_SUCCESS, "Dgtsv2StridedBatch");
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < m; ++i)
                CHECK(near(tri_apply(m, &dl[b * stride], &d[b * stride], &du[b * stride], &x[b * stride], i),
                           rhs[b * stride + i], 1e-10), "strided A*x == b");
        CHECK(x[m] == 77 && x[stride - 1] == 77, "strided padding untouched");
        CHECK(cusparseDgtsv2StridedBatch(h, m, dl.data(), d.data(), du.data(), x.data(), batch, m - 1,
                                         nullptr) == CUSPARSE_STATUS_INVALID_VALUE, "stride < m rejected");
        CHECK(cusparseDgtsv2StridedBatch(h, m, dl.data(), d.data(), du.data(), x.data(), 0, stride,
                                         nullptr) == CUSPARSE_STATUS_INVALID_VALUE, "batchCount < 1 rejected");
    }
    // Interleaved: element i of system b at [i*batch + b]; all three algorithms.
    for (int algo = 0; algo <= 2; ++algo) {
        const int m = 5, batch = 4;
        std::vector<float> dl(m * batch), d(m * batch), du(m * batch), x(m * batch);
        for (int i = 0; i < m; ++i)
            for (int b = 0; b < batch; ++b) {
                dl[i * batch + b] = 1.0f + 0.1f * b;
                d[i * batch + b] = 5.0f + i + b;
                du[i * batch + b] = 2.0f - 0.2f * i;
                x[i * batch + b] = 1.0f + i + 3.0f * b;
            }
        const auto dl0 = dl, d0 = d, du0 = du, x0 = x;
        size_t bytes = 0;
        CHECK(cusparseSgtsvInterleavedBatch_bufferSizeExt(h, algo, m, dl.data(), d.data(), du.data(),
                                                          x.data(), batch, &bytes) ==
                  CUSPARSE_STATUS_SUCCESS && bytes > 0, "interleaved buffer size");
        CHECK(cusparseSgtsvInterleavedBatch(h, algo, m, dl.data(), d.data(), du.data(), x.data(), batch,
                                            nullptr) == CUSPARSE_STATUS_SUCCESS, "SgtsvInterleavedBatch");
        for (int b = 0; b < batch; ++b)
            for (int i = 0; i < m; ++i) {
                const float got = tri_apply(m, &dl0[b], &d0[b], &du0[b], &x[b], i, batch, batch);
                CHECK(std::fabs(got - x0[i * batch + b]) < 1e-4f, "interleaved A*x == b");
            }
    }
    {
        std::vector<double> a(8, 1.0);
        CHECK(cusparseDgtsvInterleavedBatch(h, 3, 4, a.data(), a.data(), a.data(), a.data(), 2, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "unknown algorithm rejected");
        CHECK(cusparseDgtsvInterleavedBatch(nullptr, 0, 4, a.data(), a.data(), a.data(), a.data(), 2,
                                            nullptr) == CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
    }
    cusparseDestroy(h);
    return true;
}

static bool test_gpsv_interleaved() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    const int m = 7, batch = 3;
    const int N = m * batch;
    std::vector<double> ds(N), dl(N), d(N), du(N), dw(N), x(N);
    for (int i = 0; i < m; ++i)
        for (int b = 0; b < batch; ++b) {
            const int k = i * batch + b;
            ds[k] = 0.3 + 0.1 * b;
            dl[k] = 1.0 + 0.2 * i;
            d[k] = 2.5 + 0.5 * i + b;
            du[k] = 2.0 - 0.1 * i;
            dw[k] = 0.7;
            x[k] = 1.0 + i + 2.0 * b;
        }
    const auto ds0 = ds, dl0 = dl, d0 = d, du0 = du, dw0 = dw, x0 = x;
    size_t bytes = 0;
    CHECK(cusparseDgpsvInterleavedBatch_bufferSizeExt(h, 0, m, ds.data(), dl.data(), d.data(), du.data(),
                                                      dw.data(), x.data(), batch, &bytes) ==
              CUSPARSE_STATUS_SUCCESS && bytes > 0, "buffer size");
    CHECK(cusparseDgpsvInterleavedBatch(h, 0, m, ds.data(), dl.data(), d.data(), du.data(), dw.data(),
                                        x.data(), batch, nullptr) == CUSPARSE_STATUS_SUCCESS,
          "DgpsvInterleavedBatch");
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < m; ++i) {
            auto at = [&](const std::vector<double>& v, int r) { return v[r * batch + b]; };
            double s = at(d0, i) * at(x, i);
            if (i >= 2) s += at(ds0, i) * at(x, i - 2);
            if (i >= 1) s += at(dl0, i) * at(x, i - 1);
            if (i + 1 < m) s += at(du0, i) * at(x, i + 1);
            if (i + 2 < m) s += at(dw0, i) * at(x, i + 2);
            CHECK(near(s, at(x0, i), 1e-9), "pentadiagonal A*x == b");
        }
    CHECK(cusparseDgpsvInterleavedBatch(h, 1, m, ds.data(), dl.data(), d.data(), du.data(), dw.data(),
                                        x.data(), batch, nullptr) == CUSPARSE_STATUS_INVALID_VALUE,
          "only algorithm 0 exists");
    CHECK(cusparseDgpsvInterleavedBatch(h, 0, m, ds.data(), dl.data(), d.data(), du.data(), nullptr,
                                        x.data(), batch, nullptr) == CUSPARSE_STATUS_INVALID_VALUE,
          "null diagonal rejected");
    std::vector<double> zeros(N, 0.0), rhs(N, 1.0);
    CHECK(cusparseDgpsvInterleavedBatch(h, 0, m, zeros.data(), zeros.data(), zeros.data(), zeros.data(),
                                        zeros.data(), rhs.data(), batch, nullptr) ==
              CUSPARSE_STATUS_ZERO_PIVOT, "singular system reported");
    cusparseDestroy(h);
    return true;
}

// ── generic API ─────────────────────────────────────────────────────────────

static bool test_spvec_spvv_gather() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    for (int base = 0; base <= 1; ++base) {
        int idx[] = {1 + base, 3 + base, 4 + base};
        double xv[] = {1, 2, 3};
        double yv[] = {1, 2, 3, 4, 5, 6};
        cusparseSpVecDescr_t x = nullptr;
        cusparseDnVecDescr_t y = nullptr;
        const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
        CHECK(cusparseCreateSpVec(&x, 6, 3, idx, xv, CUSPARSE_INDEX_32I, ib, CUDA_R_64F) ==
                  CUSPARSE_STATUS_SUCCESS, "create sparse vector");
        cusparseCreateDnVec(&y, 6, yv, CUDA_R_64F);

        int64_t size = 0, nnz = 0;
        void *gi = nullptr, *gv = nullptr;
        cusparseIndexType_t it;
        cusparseIndexBase_t gb;
        cudaDataType vt;
        CHECK(cusparseSpVecGet(x, &size, &nnz, &gi, &gv, &it, &gb, &vt) == CUSPARSE_STATUS_SUCCESS &&
                  size == 6 && nnz == 3 && gi == idx && gv == xv && it == CUSPARSE_INDEX_32I &&
                  gb == ib && vt == CUDA_R_64F, "SpVecGet");
        cusparseIndexBase_t b2;
        void* vv = nullptr;
        CHECK(cusparseSpVecGetIndexBase(x, &b2) == CUSPARSE_STATUS_SUCCESS && b2 == ib, "GetIndexBase");
        CHECK(cusparseSpVecGetValues(x, &vv) == CUSPARSE_STATUS_SUCCESS && vv == xv, "GetValues");
        double other[] = {10, 20, 30};
        CHECK(cusparseSpVecSetValues(x, other) == CUSPARSE_STATUS_SUCCESS, "SetValues");
        cusparseSpVecGetValues(x, &vv);
        CHECK(vv == other, "SetValues took effect");
        cusparseSpVecSetValues(x, xv);

        double result = 0;
        size_t bytes = 0;
        CHECK(cusparseSpVV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &result, CUDA_R_64F,
                                      &bytes) == CUSPARSE_STATUS_SUCCESS, "SpVV buffer size");
        CHECK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &result, CUDA_R_64F, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS && near(result, 1 * 2 + 2 * 4 + 3 * 5), "SpVV dot product");

        double gathered[3] = {};
        cusparseSpVecDescr_t gx = nullptr;
        cusparseCreateSpVec(&gx, 6, 3, idx, gathered, CUSPARSE_INDEX_32I, ib, CUDA_R_64F);
        CHECK(cusparseGather(h, y, gx) == CUSPARSE_STATUS_SUCCESS && gathered[0] == 2 &&
                  gathered[1] == 4 && gathered[2] == 5, "Gather");
        cusparseDestroySpVec(gx);

        idx[2] = 6 + base;  // out of range
        CHECK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &result, CUDA_R_64F, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "out-of-range index rejected by SpVV");
        cusparseDestroySpVec(x);
        cusparseDestroyDnVec(y);
    }
    // Complex with conjugation.
    {
        int idx[] = {0, 2};
        const cplx xv[] = {{1, 1}, {2, 0}};
        const cplx yv[] = {{1, 2}, {9, 9}, {0, 1}};
        cplx r;
        cusparseSpVecDescr_t x = nullptr;
        cusparseDnVecDescr_t y = nullptr;
        cusparseCreateSpVec(&x, 3, 2, idx, const_cast<cplx*>(xv), CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
        cusparseCreateDnVec(&y, 3, const_cast<cplx*>(yv), CUDA_C_64F);
        CHECK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &r, CUDA_C_64F, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS && near(r, xv[0] * yv[0] + xv[1] * yv[2]), "complex dot");
        CHECK(cusparseSpVV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, x, y, &r, CUDA_C_64F, nullptr) ==
                  CUSPARSE_STATUS_SUCCESS && near(r, std::conj(xv[0]) * yv[0] + std::conj(xv[1]) * yv[2]),
              "conjugated dot");
        CHECK(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, x, y, &r, CUDA_R_64F, nullptr) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "compute type must match the vectors");
        cusparseDestroySpVec(x);
        cusparseDestroyDnVec(y);
    }
    // Negative paths.
    {
        double v[2] = {};
        int i[2] = {0, 1};
        cusparseSpVecDescr_t x = nullptr;
        CHECK(cusparseCreateSpVec(nullptr, 4, 2, i, v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                  CUDA_R_64F) == CUSPARSE_STATUS_INVALID_VALUE, "null output");
        CHECK(cusparseCreateSpVec(&x, 4, 5, i, v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                  CUDA_R_64F) == CUSPARSE_STATUS_INVALID_VALUE && x == nullptr,
              "nnz > size rejected");
        CHECK(cusparseCreateSpVec(&x, -1, 0, i, v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                  CUDA_R_64F) == CUSPARSE_STATUS_INVALID_VALUE, "negative size rejected");
        CHECK(cusparseCreateSpVec(&x, 4, 2, nullptr, v, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                  CUDA_R_64F) == CUSPARSE_STATUS_INVALID_VALUE, "null indices rejected");
        CHECK(cusparseSpVecGet(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "Get on null descriptor");
        double result;
        CHECK(cusparseSpVV(nullptr, CUSPARSE_OPERATION_NON_TRANSPOSE, nullptr, nullptr, &result,
                           CUDA_R_64F, nullptr) == CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
    }
    cusparseDestroy(h);
    return true;
}

static bool test_descriptor_accessors() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    int ptrA[] = {0, 1, 3}, colA[] = {0, 0, 1};
    double valA[] = {1, 2, 3};
    int ptrB[] = {0, 2, 3}, colB[] = {0, 1, 1};
    double valB[] = {1, 1, 4};
    cusparseSpMatDescr_t csr = nullptr;
    CHECK(cusparseCreateCsr(&csr, 2, 2, 3, ptrA, colA, valA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS, "create csr");
    int64_t r = 0, c = 0, nnz = 0;
    void *gp = nullptr, *gc = nullptr, *gv = nullptr;
    cusparseIndexType_t t1, t2;
    cusparseIndexBase_t ib;
    cudaDataType vt;
    CHECK(cusparseCsrGet(csr, &r, &c, &nnz, &gp, &gc, &gv, &t1, &t2, &ib, &vt) == CUSPARSE_STATUS_SUCCESS &&
              r == 2 && c == 2 && nnz == 3 && gp == ptrA && gc == colA && gv == valA &&
              t1 == CUSPARSE_INDEX_32I && t2 == CUSPARSE_INDEX_32I && ib == CUSPARSE_INDEX_BASE_ZERO &&
              vt == CUDA_R_64F, "CsrGet");
    cusparseFormat_t fmt;
    CHECK(cusparseSpMatGetFormat(csr, &fmt) == CUSPARSE_STATUS_SUCCESS && fmt == CUSPARSE_FORMAT_CSR,
          "GetFormat csr");
    CHECK(cusparseSpMatGetIndexBase(csr, &ib) == CUSPARSE_STATUS_SUCCESS && ib == CUSPARSE_INDEX_BASE_ZERO,
          "GetIndexBase");
    CHECK(cusparseSpMatGetSize(csr, &r, &c, &nnz) == CUSPARSE_STATUS_SUCCESS && r == 2 && c == 2 && nnz == 3,
          "GetSize");
    int batch = 0;
    CHECK(cusparseSpMatGetStridedBatch(csr, &batch) == CUSPARSE_STATUS_SUCCESS && batch == 1,
          "GetStridedBatch");
    CHECK(cusparseCooGet(csr, &r, &c, &nnz, &gp, &gc, &gv, &t1, &ib, &vt) == CUSPARSE_STATUS_INVALID_VALUE,
          "CooGet on a CSR descriptor rejected");

    // Repointing the structure must be seen by the next product (the cached
    // longest-row hint is reset). y = A x with x = [1, 1].
    double x[] = {1, 1}, y[] = {0, 0};
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    cusparseCreateDnVec(&vx, 2, x, CUDA_R_64F);
    cusparseCreateDnVec(&vy, 2, y, CUDA_R_64F);
    const double one = 1.0, zero = 0.0;
    CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, csr, vx, &zero, vy, CUDA_R_64F,
                       CUSPARSE_SPMV_ALG_DEFAULT, nullptr) == CUSPARSE_STATUS_SUCCESS &&
              y[0] == 1 && y[1] == 5, "SpMV before repointing");
    CHECK(cusparseCsrSetPointers(csr, ptrB, colB, valB) == CUSPARSE_STATUS_SUCCESS, "CsrSetPointers");
    CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, csr, vx, &zero, vy, CUDA_R_64F,
                       CUSPARSE_SPMV_ALG_DEFAULT, nullptr) == CUSPARSE_STATUS_SUCCESS &&
              y[0] == 2 && y[1] == 4, "SpMV after repointing sees the new structure");
    CHECK(cusparseCsrSetPointers(csr, nullptr, colB, valB) == CUSPARSE_STATUS_INVALID_VALUE,
          "null row offsets rejected");
    CHECK(cusparseCsrSetPointers(csr, ptrB, nullptr, valB) == CUSPARSE_STATUS_INVALID_VALUE,
          "null column indices rejected when nnz > 0");
    double valC[] = {2, 2, 8};
    CHECK(cusparseSpMatSetValues(csr, valC) == CUSPARSE_STATUS_SUCCESS &&
              cusparseSpMatGetValues(csr, &gv) == CUSPARSE_STATUS_SUCCESS && gv == valC,
          "Set/GetValues");
    CHECK(cusparseSpMatSetValues(nullptr, valC) == CUSPARSE_STATUS_INVALID_VALUE, "SetValues null");

    int cr[] = {1, 2}, cc[] = {1, 2};  // base 1
    double cv[] = {1, 1};
    cusparseSpMatDescr_t coo = nullptr;
    cusparseCreateCoo(&coo, 2, 2, 2, cr, cc, cv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_64F);
    CHECK(cusparseCooGet(coo, &r, &c, &nnz, &gp, &gc, &gv, &t1, &ib, &vt) == CUSPARSE_STATUS_SUCCESS &&
              gp == cr && gc == cc && gv == cv && ib == CUSPARSE_INDEX_BASE_ONE, "CooGet");
    CHECK(cusparseSpMatGetFormat(coo, &fmt) == CUSPARSE_STATUS_SUCCESS && fmt == CUSPARSE_FORMAT_COO,
          "GetFormat coo");
    CHECK(cusparseCsrSetPointers(coo, ptrB, colB, valB) == CUSPARSE_STATUS_INVALID_VALUE,
          "CsrSetPointers on COO rejected");

    // Dense matrix accessors and strided batch.
    double dm[6] = {};
    cusparseDnMatDescr_t dn = nullptr;
    cusparseCreateDnMat(&dn, 2, 3, 2, dm, CUDA_R_64F, CUSPARSE_ORDER_COL);
    int64_t ld = 0, bs = -1;
    cusparseOrder_t order;
    CHECK(cusparseDnMatGet(dn, &r, &c, &ld, &gv, &vt, &order) == CUSPARSE_STATUS_SUCCESS && r == 2 &&
              c == 3 && ld == 2 && gv == dm && vt == CUDA_R_64F && order == CUSPARSE_ORDER_COL, "DnMatGet");
    CHECK(cusparseDnMatGetStridedBatch(dn, &batch, &bs) == CUSPARSE_STATUS_SUCCESS && batch == 1 && bs == 0,
          "default batch");
    CHECK(cusparseDnMatSetStridedBatch(dn, 2, 6) == CUSPARSE_STATUS_SUCCESS &&
              cusparseDnMatGetStridedBatch(dn, &batch, &bs) == CUSPARSE_STATUS_SUCCESS && batch == 2 && bs == 6,
          "SetStridedBatch round trip");
    CHECK(cusparseDnMatSetStridedBatch(dn, 0, 6) == CUSPARSE_STATUS_INVALID_VALUE, "batchCount 0 rejected");
    CHECK(cusparseDnMatSetStridedBatch(nullptr, 1, 0) == CUSPARSE_STATUS_INVALID_VALUE, "null descriptor");
    double dm2[6] = {};
    CHECK(cusparseDnMatSetValues(dn, dm2) == CUSPARSE_STATUS_SUCCESS &&
              cusparseDnMatGetValues(dn, &gv) == CUSPARSE_STATUS_SUCCESS && gv == dm2, "Dn Set/GetValues");

    // SpMM addresses one matrix; a configured batch must be refused, not silently ignored.
    double b6[4] = {1, 0, 0, 1}, c6[4] = {};
    cusparseDnMatDescr_t mb = nullptr, mc = nullptr;
    cusparseCreateDnMat(&mb, 2, 2, 2, b6, CUDA_R_64F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&mc, 2, 2, 2, c6, CUDA_R_64F, CUSPARSE_ORDER_COL);
    cusparseDnMatSetStridedBatch(mb, 2, 4);
    size_t bytes = 0;
    CHECK(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &one, csr, mb, &zero, mc, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
                                  &bytes) == CUSPARSE_STATUS_NOT_SUPPORTED, "batched SpMM bufferSize refused");
    CHECK(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, csr,
                       mb, &zero, mc, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, nullptr) ==
              CUSPARSE_STATUS_NOT_SUPPORTED, "batched SpMM refused");
    cusparseDnMatSetStridedBatch(mb, 1, 0);
    // The NVIDIA numeric values for the algorithms CuPy passes are accepted.
    CHECK(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, csr,
                       mb, &zero, mc, CUDA_R_64F, static_cast<cusparseSpMMAlg_t>(4), nullptr) ==
              CUSPARSE_STATUS_SUCCESS, "CSRMM_ALG1 (4) accepted");
    CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, coo, vx, &zero, vy, CUDA_R_64F,
                       CUSPARSE_SPMV_COO_ALG1, nullptr) == CUSPARSE_STATUS_SUCCESS,
          "COOMV_ALG (1) on a COO matrix accepted");
    CHECK(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, coo, vx, &zero, vy, CUDA_R_64F,
                       CUSPARSE_SPMV_CSR_ALG1, nullptr) == CUSPARSE_STATUS_NOT_SUPPORTED,
          "a CSR algorithm on a COO matrix refused");

    int version = 0;
    CHECK(cusparseGetVersion(h, &version) == CUSPARSE_STATUS_SUCCESS && version == CUSPARSE_VERSION,
          "cusparseGetVersion agrees with CUSPARSE_VERSION");
    cusparseDestroyDnMat(mb);
    cusparseDestroyDnMat(mc);
    cusparseDestroyDnMat(dn);
    cusparseDestroySpMat(coo);
    cusparseDestroySpMat(csr);
    cusparseDestroyDnVec(vx);
    cusparseDestroyDnVec(vy);
    cusparseDestroy(h);
    return true;
}

// Dense 3x4 with zeros, used by the dense<->sparse tests:
//   [1 0 2 0; 0 0 3 0; 4 5 0 6]
static const double kDense[3][4] = {{1, 0, 2, 0}, {0, 0, 3, 0}, {4, 5, 0, 6}};

static bool test_dense_sparse_conversion() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    for (int order_i = 0; order_i < 2; ++order_i) {
        const cusparseOrder_t order = order_i ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
        const int ld = order_i ? 5 : 4;  // padded leading dimensions
        std::vector<double> dense(static_cast<size_t>(order_i ? 3 * 5 : 4 * 4), -9.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 4; ++j)
                dense[order_i ? i * ld + j : i + j * ld] = kDense[i][j];
        cusparseDnMatDescr_t dn = nullptr;
        CHECK(cusparseCreateDnMat(&dn, 3, 4, ld, dense.data(), CUDA_R_64F, order) == CUSPARSE_STATUS_SUCCESS,
              "create dense");

        for (int base = 0; base <= 1; ++base) {
            const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
            // CSR
            std::vector<int> ptr(4, 0), col(6, 0);
            std::vector<double> val(6, 0.0);
            cusparseSpMatDescr_t sp = nullptr;
            CHECK(cusparseCreateCsr(&sp, 3, 4, 0, ptr.data(), nullptr, nullptr, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_32I, ib, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS,
                  "create empty csr");
            size_t bytes = 0;
            CHECK(cusparseDenseToSparse_bufferSize(h, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bytes) ==
                      CUSPARSE_STATUS_SUCCESS, "DenseToSparse buffer size");
            std::vector<unsigned char> buf(bytes + 1);
            CHECK(cusparseDenseToSparse_analysis(h, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buf.data()) ==
                      CUSPARSE_STATUS_SUCCESS, "analysis");
            int64_t rr, cc, nnz;
            cusparseSpMatGetSize(sp, &rr, &cc, &nnz);
            CHECK(nnz == 6, "analysis reports nnz");
            CHECK(ptr[0] == base && ptr[1] == base + 2 && ptr[2] == base + 3 && ptr[3] == base + 6,
                  "analysis writes the row offsets");
            CHECK(cusparseCsrSetPointers(sp, ptr.data(), col.data(), val.data()) == CUSPARSE_STATUS_SUCCESS,
                  "CsrSetPointers");
            CHECK(cusparseDenseToSparse_convert(h, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buf.data()) ==
                      CUSPARSE_STATUS_SUCCESS, "convert");
            const int wc[] = {0, 2, 2, 0, 1, 3};
            const double wv[] = {1, 2, 3, 4, 5, 6};
            for (int k = 0; k < 6; ++k) CHECK(col[k] == wc[k] + base && val[k] == wv[k], "csr contents");

            // And back to a dense matrix.
            std::vector<double> back(dense.size(), 123.0);
            cusparseDnMatDescr_t dn2 = nullptr;
            cusparseCreateDnMat(&dn2, 3, 4, ld, back.data(), CUDA_R_64F, order);
            CHECK(cusparseSparseToDense_bufferSize(h, sp, dn2, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes) ==
                      CUSPARSE_STATUS_SUCCESS, "SparseToDense buffer size");
            CHECK(cusparseSparseToDense(h, sp, dn2, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf.data()) ==
                      CUSPARSE_STATUS_SUCCESS, "SparseToDense");
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j)
                    CHECK(back[order_i ? i * ld + j : i + j * ld] == kDense[i][j], "dense round trip");
            // Padding (outside the 3x4 window) is not part of B and must be left alone.
            CHECK(back[order_i ? 4 : 3] == 123.0, "leading-dimension padding untouched");
            cusparseDestroyDnMat(dn2);

            // A changed between analysis and convert no longer fits the arrays.
            dense[order_i ? 1 * ld + 0 : 1 + 0 * ld] = 8.0;
            CHECK(cusparseDenseToSparse_convert(h, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, buf.data()) ==
                      CUSPARSE_STATUS_INVALID_VALUE, "stale analysis rejected");
            dense[order_i ? 1 * ld + 0 : 1 + 0 * ld] = 0.0;
            cusparseDestroySpMat(sp);

            // CSC
            std::vector<int> cptr(5, 0), crow(6, 0);
            std::vector<double> cval(6, 0.0);
            cusparseSpMatDescr_t csc = nullptr;
            cusparseCreateCsc(&csc, 3, 4, 0, cptr.data(), nullptr, nullptr, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, ib, CUDA_R_64F);
            CHECK(cusparseDenseToSparse_analysis(h, dn, csc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                      CUSPARSE_STATUS_SUCCESS, "csc analysis");
            const int wcp[] = {0, 2, 3, 5, 6};
            for (int k = 0; k < 5; ++k) CHECK(cptr[k] == wcp[k] + base, "csc column offsets");
            // CSC has no CsrSetPointers; the arrays are bound through a fresh descriptor.
            cusparseDestroySpMat(csc);
            cusparseCreateCsc(&csc, 3, 4, 6, cptr.data(), crow.data(), cval.data(), CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, ib, CUDA_R_64F);
            CHECK(cusparseDenseToSparse_convert(h, dn, csc, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                      CUSPARSE_STATUS_SUCCESS, "csc convert");
            const int wrow[] = {0, 2, 2, 0, 1, 2};
            const double wcv[] = {1, 4, 5, 2, 3, 6};
            for (int k = 0; k < 6; ++k) CHECK(crow[k] == wrow[k] + base && cval[k] == wcv[k], "csc contents");
            std::vector<double> back2(dense.size(), 0.0);
            cusparseDnMatDescr_t dn3 = nullptr;
            cusparseCreateDnMat(&dn3, 3, 4, ld, back2.data(), CUDA_R_64F, order);
            CHECK(cusparseSparseToDense(h, csc, dn3, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr) ==
                      CUSPARSE_STATUS_SUCCESS, "csc to dense");
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 4; ++j)
                    CHECK(back2[order_i ? i * ld + j : i + j * ld] == kDense[i][j], "csc round trip");
            cusparseDestroyDnMat(dn3);
            cusparseDestroySpMat(csc);

            // COO: nnz is fixed at creation, so analysis only has to confirm it.
            std::vector<int> crr(6, 0), ccc(6, 0);
            std::vector<double> cov(6, 0.0);
            cusparseSpMatDescr_t coo = nullptr;
            cusparseCreateCoo(&coo, 3, 4, 0, crr.data(), ccc.data(), cov.data(), CUSPARSE_INDEX_32I, ib,
                              CUDA_R_64F);
            CHECK(cusparseDenseToSparse_analysis(h, dn, coo, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                      CUSPARSE_STATUS_SUCCESS, "coo analysis");
            cusparseSpMatGetSize(coo, &rr, &cc, &nnz);
            CHECK(nnz == 6, "coo analysis sets nnz");
            CHECK(cusparseDenseToSparse_convert(h, dn, coo, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                      CUSPARSE_STATUS_SUCCESS, "coo convert");
            const int wrr[] = {0, 0, 1, 2, 2, 2};
            for (int k = 0; k < 6; ++k)
                CHECK(crr[k] == wrr[k] + base && ccc[k] == wc[k] + base && cov[k] == wv[k], "coo contents");
            cusparseDestroySpMat(coo);
        }
        cusparseDestroyDnMat(dn);
    }
    // Negative paths.
    {
        double dv[4] = {1, 0, 0, 1};
        float fv[4] = {};
        int ptr[3] = {};
        cusparseDnMatDescr_t dn = nullptr, dnf = nullptr;
        cusparseSpMatDescr_t sp = nullptr;
        cusparseCreateDnMat(&dn, 2, 2, 2, dv, CUDA_R_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dnf, 2, 2, 2, fv, CUDA_R_32F, CUSPARSE_ORDER_COL);
        cusparseCreateCsr(&sp, 2, 2, 0, ptr, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        CHECK(cusparseDenseToSparse_analysis(h, dnf, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "value type mismatch");
        CHECK(cusparseDenseToSparse_analysis(h, dn, sp, static_cast<cusparseDenseToSparseAlg_t>(7), nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "unknown algorithm");
        CHECK(cusparseDenseToSparse_analysis(nullptr, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                  CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
        CHECK(cusparseSparseToDense(h, sp, nullptr, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "null dense");
        cusparseDnMatDescr_t big = nullptr;
        double bv[9] = {};
        cusparseCreateDnMat(&big, 3, 3, 3, bv, CUDA_R_64F, CUSPARSE_ORDER_COL);
        CHECK(cusparseSparseToDense(h, sp, big, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "shape mismatch");
        cusparseDnMatSetStridedBatch(dn, 2, 4);
        CHECK(cusparseDenseToSparse_analysis(h, dn, sp, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, nullptr) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "batched dense refused");
        cusparseDestroyDnMat(big);
        cusparseDestroyDnMat(dn);
        cusparseDestroyDnMat(dnf);
        cusparseDestroySpMat(sp);
    }
    cusparseDestroy(h);
    return true;
}

static bool test_spgemm() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    for (int base = 0; base <= 1; ++base) {
        const cusparseIndexBase_t ib = base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO;
        // A (3x4) = [1 0 2 0; 0 3 0 0; 4 0 0 5], B (4x2) = [1 2; 0 0; 3 0; 0 4]
        std::vector<int> pA = {0, 2, 3, 5}, cA = {0, 2, 1, 0, 3};
        std::vector<double> vA = {1, 2, 3, 4, 5};
        std::vector<int> pB = {0, 2, 2, 3, 4}, cB = {0, 1, 0, 1};
        std::vector<double> vB = {1, 2, 3, 4};
        for (auto& x : pA) x += base;
        for (auto& x : cA) x += base;
        for (auto& x : pB) x += base;
        for (auto& x : cB) x += base;
        std::vector<int> pC(4, base);
        cusparseSpMatDescr_t A = nullptr, B = nullptr, C = nullptr;
        cusparseCreateCsr(&A, 3, 4, 5, pA.data(), cA.data(), vA.data(), CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, ib, CUDA_R_64F);
        cusparseCreateCsr(&B, 4, 2, 4, pB.data(), cB.data(), vB.data(), CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, ib, CUDA_R_64F);
        CHECK(cusparseCreateCsr(&C, 3, 2, 0, pC.data(), nullptr, nullptr, CUSPARSE_INDEX_32I,
                                CUSPARSE_INDEX_32I, ib, CUDA_R_64F) == CUSPARSE_STATUS_SUCCESS, "create C");
        cusparseSpGEMMDescr_t g = nullptr;
        CHECK(cusparseSpGEMM_createDescr(&g) == CUSPARSE_STATUS_SUCCESS, "create descr");
        const double alpha = 2.0, beta = 0.0;
        const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
        size_t b1 = 0, b2 = 0;
        CHECK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                            g, &b1, nullptr) == CUSPARSE_STATUS_SUCCESS && b1 > 0,
              "workEstimation size query");
        std::vector<unsigned char> buf1(b1);
        CHECK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT,
                                            g, &b1, buf1.data()) == CUSPARSE_STATUS_SUCCESS,
              "workEstimation");
        CHECK(cusparseSpGEMM_compute(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g,
                                     &b2, nullptr) == CUSPARSE_STATUS_SUCCESS && b2 > 0,
              "compute size query");
        int64_t cr = 0, cc = 0, cnnz = -1;
        cusparseSpMatGetSize(C, &cr, &cc, &cnnz);
        CHECK(cnnz == 0, "a size query does not compute");
        std::vector<unsigned char> buf2(b2);
        CHECK(cusparseSpGEMM_compute(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g,
                                     &b2, buf2.data()) == CUSPARSE_STATUS_SUCCESS, "compute");
        cusparseSpMatGetSize(C, &cr, &cc, &cnnz);
        CHECK(cr == 3 && cc == 2, "C shape");
        // A*B = [7 2; 0 0; 4 28]. Row 1 is A(1,1)*B(1,:) and B's row 1 is empty, so row 1
        // has no structural entries; the other rows have two each.
        CHECK(cnnz == 4, "C nnz is the structural count");
        std::vector<int> cCol(cnnz);
        std::vector<double> cVal(cnnz);
        CHECK(cusparseCsrSetPointers(C, pC.data(), cCol.data(), cVal.data()) == CUSPARSE_STATUS_SUCCESS,
              "CsrSetPointers on C");
        CHECK(cusparseSpGEMM_copy(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g) ==
                  CUSPARSE_STATUS_SUCCESS, "copy");
        CHECK(sorted_unique(pC.data(), cCol.data(), 3, base), "C columns sorted");
        const auto got = csr_dense<double>(3, 2, pC.data(), cCol.data(), cVal.data(), base);
        const std::vector<double> want = {14, 4, 0, 0, 8, 56};
        CHECK(dense_near(got, want), "C == 2 * A * B");
        cusparseSpGEMM_destroyDescr(g);
        cusparseDestroySpMat(A);
        cusparseDestroySpMat(B);
        cusparseDestroySpMat(C);
    }
    // Negative paths.
    {
        std::vector<int> p = {0, 1, 2}, c = {0, 1}, pc = {0, 0, 0};
        std::vector<double> v = {1, 1};
        cusparseSpMatDescr_t A = nullptr, B = nullptr, C = nullptr, Wrong = nullptr;
        cusparseCreateCsr(&A, 2, 2, 2, p.data(), c.data(), v.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        B = A;
        cusparseCreateCsr(&C, 2, 2, 0, pc.data(), nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseCreateCsr(&Wrong, 3, 2, 0, pc.data(), nullptr, nullptr, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseSpGEMMDescr_t g = nullptr;
        cusparseSpGEMM_createDescr(&g);
        const double one = 1.0, zero = 0.0, nonzero_beta = 0.5;
        size_t sz = 0;
        const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
        CHECK(cusparseSpGEMM_workEstimation(h, CUSPARSE_OPERATION_TRANSPOSE, N, &one, A, B, &zero, C,
                                            CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g, &sz, nullptr) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "transposed operand refused");
        CHECK(cusparseSpGEMM_workEstimation(h, N, N, &one, A, B, &zero, Wrong, CUDA_R_64F,
                                            CUSPARSE_SPGEMM_DEFAULT, g, &sz, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "C shape mismatch");
        std::vector<unsigned char> buf(256);
        sz = buf.size();
        CHECK(cusparseSpGEMM_compute(h, N, N, &one, A, B, &nonzero_beta, C, CUDA_R_64F,
                                     CUSPARSE_SPGEMM_DEFAULT, g, &sz, buf.data()) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "nonzero beta refused");
        CHECK(cusparseSpGEMM_copy(h, N, N, &one, A, B, &zero, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "copy before compute rejected");
        CHECK(cusparseSpGEMM_workEstimation(h, N, N, &one, A, B, &zero, C, CUDA_R_64F,
                                            static_cast<cusparseSpGEMMAlg_t>(9), g, &sz, nullptr) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "unknown algorithm");
        CHECK(cusparseSpGEMM_workEstimation(nullptr, N, N, &one, A, B, &zero, C, CUDA_R_64F,
                                            CUSPARSE_SPGEMM_DEFAULT, g, &sz, nullptr) ==
                  CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
        cusparseSpGEMM_destroyDescr(g);
        cusparseDestroySpMat(A);
        cusparseDestroySpMat(C);
        cusparseDestroySpMat(Wrong);
    }
    cusparseDestroy(h);
    return true;
}

// ── SpSM / SpSV ─────────────────────────────────────────────────────────────

// Largest |op(A)*C - alpha*op(B)| for dense row-major op(A) (n x n), C and B n x k.
static double spsm_residual(int n, int k, const std::vector<double>& opA, const std::vector<double>& C,
                            const std::vector<double>& B, double alpha) {
    double worst = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < k; ++j) {
            double s = 0;
            for (int t = 0; t < n; ++t) s += opA[i * n + t] * C[t * k + j];
            worst = std::max(worst, std::fabs(s - alpha * B[i * k + j]));
        }
    return worst;
}

static bool test_spsm_and_spsv_transpose() {
    cusparseHandle_t h = nullptr;
    cusparseCreate(&h);
    // Lower A = [2 0 0; 1 3 0; 4 2 5] in CSR, base 0 and 1.
    const int n = 3, k = 2;
    for (int base = 0; base <= 1; ++base) {
        std::vector<int> ptr = {0, 1, 3, 6}, col = {0, 0, 1, 0, 1, 2};
        std::vector<double> val = {2, 1, 3, 4, 2, 5};
        for (auto& x : ptr) x += base;
        for (auto& x : col) x += base;
        const std::vector<double> denseL = {2, 0, 0, 1, 3, 0, 4, 2, 5};
        std::vector<double> denseLT(9);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) denseLT[i * 3 + j] = denseL[j * 3 + i];

        for (int variant = 0; variant < 4; ++variant) {
            // variant bit0: op(A) transposed; bit1: B/C row-major and op(B) transposed.
            const bool transA = variant & 1, rowmajor = variant & 2;
            const std::vector<double> B = {1, 2, 3, 4, 5, 6};  // n x k logical, row-major
            std::vector<double> storedB, storedC(6, 0.0);
            // opB transposed means B is stored k x n.
            if (rowmajor) {
                storedB.assign(6, 0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < k; ++j) storedB[j * n + i] = B[i * k + j];  // k x n row-major
            } else {
                storedB.assign(6, 0);
                for (int i = 0; i < n; ++i)
                    for (int j = 0; j < k; ++j) storedB[i + j * n] = B[i * k + j];  // n x k col-major
            }
            cusparseSpMatDescr_t A = nullptr;
            cusparseCreateCsr(&A, n, n, 6, ptr.data(), col.data(), val.data(), CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_32I, base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F);
            cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
            cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
            cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
            cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
            cusparseDnMatDescr_t dB = nullptr, dC = nullptr;
            if (rowmajor)
                cusparseCreateDnMat(&dB, k, n, n, storedB.data(), CUDA_R_64F, CUSPARSE_ORDER_ROW);
            else
                cusparseCreateDnMat(&dB, n, k, n, storedB.data(), CUDA_R_64F, CUSPARSE_ORDER_COL);
            cusparseCreateDnMat(&dC, n, k, k, storedC.data(), CUDA_R_64F, CUSPARSE_ORDER_ROW);
            cusparseSpSMDescr_t sd = nullptr;
            CHECK(cusparseSpSM_createDescr(&sd) == CUSPARSE_STATUS_SUCCESS, "create SpSM descr");
            const double alpha = 2.0;
            const auto opA = transA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            const auto opB = rowmajor ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            size_t bytes = 0;
            CHECK(cusparseSpSM_bufferSize(h, opA, opB, &alpha, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT,
                                          sd, &bytes) == CUSPARSE_STATUS_SUCCESS && bytes > 0,
                  "SpSM buffer size");
            std::vector<unsigned char> buf(bytes);
            CHECK(cusparseSpSM_analysis(h, opA, opB, &alpha, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd,
                                        buf.data()) == CUSPARSE_STATUS_SUCCESS, "SpSM analysis");
            CHECK(cusparseSpSM_solve(h, opA, opB, &alpha, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                      CUSPARSE_STATUS_SUCCESS, "SpSM solve");
            CHECK(spsm_residual(n, k, transA ? denseLT : denseL, storedC, B, alpha) < 1e-12,
                  "op(A) * C == alpha * op(B)");
            cusparseSpSM_destroyDescr(sd);
            cusparseDestroyDnMat(dB);
            cusparseDestroyDnMat(dC);
            cusparseDestroySpMat(A);
        }
    }
    // Upper triangular and unit diagonal.
    {
        // U = [2 1 4; 0 3 2; 0 0 5], diagonal declared unit => treated as [1 1 4; 0 1 2; 0 0 1].
        std::vector<int> ptr = {0, 3, 5, 6}, col = {0, 1, 2, 1, 2, 2};
        std::vector<double> val = {2, 1, 4, 3, 2, 5};
        std::vector<double> storedB = {1, 2, 3, 0, 0, 0}, storedC(3, 0.0);
        cusparseSpMatDescr_t A = nullptr;
        cusparseCreateCsr(&A, 3, 3, 6, ptr.data(), col.data(), val.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_UNIT;
        cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
        cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
        cusparseDnMatDescr_t dB = nullptr, dC = nullptr;
        cusparseCreateDnMat(&dB, 3, 1, 3, storedB.data(), CUDA_R_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dC, 3, 1, 3, storedC.data(), CUDA_R_64F, CUSPARSE_ORDER_COL);
        cusparseSpSMDescr_t sd = nullptr;
        cusparseSpSM_createDescr(&sd);
        const double one = 1.0;
        const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
        CHECK(cusparseSpSM_solve(h, N, N, &one, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_SUCCESS, "unit upper solve");
        // x2 = 3; x1 = 2 - 2*3 = -4; x0 = 1 - 1*(-4) - 4*3 = -7
        CHECK(near(storedC[0], -7) && near(storedC[1], -4) && near(storedC[2], 3), "unit upper values");
        // Transposed: U^T y = b, U^T unit lower: y0 = 1; y1 = 2 - 1*1 = 1; y2 = 3 - 4*1 - 2*1 = -3
        CHECK(cusparseSpSM_solve(h, CUSPARSE_OPERATION_TRANSPOSE, N, &one, A, dB, dC, CUDA_R_64F,
                                 CUSPARSE_SPSM_ALG_DEFAULT, sd) == CUSPARSE_STATUS_SUCCESS &&
                  near(storedC[0], 1) && near(storedC[1], 1) && near(storedC[2], -3),
              "unit upper transposed");
        // A CSC descriptor of the same arrays is the matrix U^T.
        cusparseSpMatDescr_t csc = nullptr;
        cusparseCreateCsc(&csc, 3, 3, 6, ptr.data(), col.data(), val.data(), CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        cusparseFillMode_t lower = CUSPARSE_FILL_MODE_LOWER;  // the CSC matrix (U^T) is lower
        cusparseSpMatSetAttribute(csc, CUSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower));
        cusparseSpMatSetAttribute(csc, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
        std::fill(storedC.begin(), storedC.end(), 0.0);
        CHECK(cusparseSpSM_solve(h, N, N, &one, csc, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_SUCCESS && near(storedC[0], 1) && near(storedC[1], 1) &&
                  near(storedC[2], -3), "CSC descriptor solves with its transpose");
        cusparseDestroySpMat(csc);

        // Zero / missing diagonal.
        cusparseDiagType_t nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
        cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit, sizeof(nonunit));
        val[3] = 0;
        CHECK(cusparseSpSM_solve(h, N, N, &one, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_ZERO_PIVOT, "zero diagonal reported");

        // Negative paths.
        CHECK(cusparseSpSM_solve(nullptr, N, N, &one, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_NOT_INITIALIZED, "null handle");
        CHECK(cusparseSpSM_solve(h, N, N, &one, A, dB, dC, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "compute type mismatch");
        CHECK(cusparseSpSM_solve(h, N, N, nullptr, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "null alpha");
        CHECK(cusparseSpSM_solve(h, N, N, &one, A, dB, dC, CUDA_R_64F, static_cast<cusparseSpSMAlg_t>(3), sd) ==
                  CUSPARSE_STATUS_INVALID_VALUE, "unknown algorithm");
        cusparseDnMatSetStridedBatch(dB, 2, 3);
        CHECK(cusparseSpSM_solve(h, N, N, &one, A, dB, dC, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_NOT_SUPPORTED, "batched right-hand side refused");
        cusparseSpSM_destroyDescr(sd);
        cusparseDestroyDnMat(dB);
        cusparseDestroyDnMat(dC);
        cusparseDestroySpMat(A);
    }
    // Complex, conjugate-transpose: L^H y = b with L = [2 0; 1+i 3].
    {
        std::vector<int> ptr = {0, 1, 3}, col = {0, 0, 1};
        std::vector<cplx> val = {{2, 0}, {1, 1}, {3, 0}};
        std::vector<cplx> b = {{1, 1}, {2, -1}}, c(2);
        cusparseSpMatDescr_t A = nullptr;
        cusparseCreateCsr(&A, 2, 2, 3, ptr.data(), col.data(), val.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);
        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
        cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
        cusparseDnMatDescr_t dB = nullptr, dC = nullptr;
        cusparseCreateDnMat(&dB, 2, 1, 2, b.data(), CUDA_C_64F, CUSPARSE_ORDER_COL);
        cusparseCreateDnMat(&dC, 2, 1, 2, c.data(), CUDA_C_64F, CUSPARSE_ORDER_COL);
        cusparseSpSMDescr_t sd = nullptr;
        cusparseSpSM_createDescr(&sd);
        const cplx alpha(1, 0);
        CHECK(cusparseSpSM_solve(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, A, dB, dC, CUDA_C_64F, CUSPARSE_SPSM_ALG_DEFAULT, sd) ==
                  CUSPARSE_STATUS_SUCCESS, "complex SpSM");
        // L^H = [2 1-i; 0 3]: c1 = b1/3; c0 = (b0 - (1-i) c1)/2.
        const cplx c1 = b[1] / 3.0, c0 = (b[0] - cplx(1, -1) * c1) / 2.0;
        CHECK(near(c[0], c0) && near(c[1], c1), "conjugate-transpose solve");
        cusparseSpSM_destroyDescr(sd);
        cusparseDestroyDnMat(dB);
        cusparseDestroyDnMat(dC);
        cusparseDestroySpMat(A);
    }
    // Regression: SpSV with a transposed operation used to read A's rows as if they were
    // op(A)'s rows. L^T y = x with L = [2 0; 1 3]: y1 = x1/3, y0 = (x0 - 1*y1)/2.
    {
        std::vector<int> ptr = {0, 1, 3}, col = {0, 0, 1};
        std::vector<float> val = {2, 1, 3}, x = {4, 6}, y(2, 0.0f);
        cusparseSpMatDescr_t A = nullptr;
        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        cusparseCreateCsr(&A, 2, 2, 3, ptr.data(), col.data(), val.data(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vx, 2, x.data(), CUDA_R_32F);
        cusparseCreateDnVec(&vy, 2, y.data(), CUDA_R_32F);
        cusparseSpSVDescr_t sv = nullptr;
        cusparseSpSV_createDescr(&sv);
        const float one = 1.0f;
        CHECK(cusparseSpSV_solve(h, CUSPARSE_OPERATION_TRANSPOSE, &one, A, vx, vy, CUDA_R_32F,
                                 CUSPARSE_SPSV_ALG_DEFAULT, sv) == CUSPARSE_STATUS_SUCCESS &&
                  std::fabs(y[1] - 2.0f) < 1e-6f && std::fabs(y[0] - 1.0f) < 1e-6f,
              "SpSV transposed lower solve");
        cusparseSpSV_destroyDescr(sv);
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        cusparseDestroySpMat(A);
    }
    cusparseDestroy(h);
    return true;
}

int main() {
    if (!test_coo_csr_conversion()) return 1;
    if (!test_sort_and_permutation()) return 1;
    if (!test_nnz_and_compress()) return 1;
    if (!test_csr2csc()) return 1;
    if (!test_csrgeam2()) return 1;
    if (!test_csrilu02()) return 1;
    if (!test_csric02()) return 1;
    if (!test_bsr_factorizations_refuse()) return 1;
    if (!test_gtsv2()) return 1;
    if (!test_gtsv2_strided_and_interleaved()) return 1;
    if (!test_gpsv_interleaved()) return 1;
    if (!test_spvec_spvv_gather()) return 1;
    if (!test_descriptor_accessors()) return 1;
    if (!test_dense_sparse_conversion()) return 1;
    if (!test_spgemm()) return 1;
    if (!test_spsm_and_spsv_transpose()) return 1;
    std::printf("PASS: cuSPARSE CuPy surface tests\n");
    return 0;
}
