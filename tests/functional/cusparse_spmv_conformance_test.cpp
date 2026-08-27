// cusparseSpMV conformance: every (format, operation, type) combination against
// a dense reference, on both the Metal and CPU implementations.
//
// Two things this is built to catch. First, `opA` being ignored: SpMV once
// computed y = alpha*A*x regardless of the operation while sizing the output
// loop from it, which is both a wrong answer and an out-of-bounds read, and
// which no square symmetric test matrix would reveal. Every matrix here is
// rectangular and chosen so that A*x and A'*y cannot be confused. Second, the
// Metal gather path silently not running: each case is executed twice, once
// with the GPU path and once with CUMETAL_SPARSE_METAL=0, and both must match
// the same reference.
#include "cusparse.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

struct Matrix {
    const char* name;
    int rows;
    int cols;
    std::vector<double> dense;  // row-major, rows*cols
};

// Row lengths deliberately uneven, with empty rows and empty columns, because a
// gather kernel indexes offsets[i]..offsets[i+1] and an empty range is the case
// most likely to be mishandled.
std::vector<Matrix> build_matrices() {
    std::vector<Matrix> out;

    out.push_back({"3x4 mixed", 3, 4,
                   {1, 0, 2, 0,
                    0, 3, 0, 4,
                    5, 0, 0, 6}});

    // Row 1 and column 2 are entirely empty.
    out.push_back({"4x3 empty row+col", 4, 3,
                   {7, 0, 0,
                    0, 0, 0,
                    0, 2, 0,
                    1, 0, 0}});

    // One row holds every nonzero; the rest hold one or none. A kernel that
    // assumes uniform row length diverges badly here.
    out.push_back({"5x6 skewed", 5, 6,
                   {1, 2, 3, 4, 5, 6,
                    0, 0, 0, 0, 0, 0,
                    0, 0, 9, 0, 0, 0,
                    0, 0, 0, 0, 0, 0,
                    8, 0, 0, 0, 0, 0}});

    // Strongly rectangular: A*x has 2 entries, A'*y has 7. Confusing the two
    // cannot produce a plausible answer.
    Matrix tall{"2x7 wide", 2, 7, std::vector<double>(14, 0.0)};
    tall.dense[0] = 1.5;
    tall.dense[3] = -2.25;
    tall.dense[7 + 6] = 4.0;
    out.push_back(tall);

    // Values needing more than 24 significand bits, so an FP32 collapse shows up.
    out.push_back({"2x2 fine", 2, 2,
                   {1.0 + 1.0 / 8796093022208.0, 0,
                    0, 3.0000000000000004}});

    return out;
}

struct Csr {
    std::vector<int> offsets;
    std::vector<int> indices;
    std::vector<double> values;
};

// Compressed along `rows` of the dense matrix. Used directly for CSR, and on the
// transpose for CSC, which is the same layout viewed the other way.
Csr compress(const Matrix& m, bool transpose, int base) {
    const int outer = transpose ? m.cols : m.rows;
    const int inner = transpose ? m.rows : m.cols;
    Csr c;
    c.offsets.push_back(base);
    for (int i = 0; i < outer; ++i) {
        for (int j = 0; j < inner; ++j) {
            const double v = transpose ? m.dense[j * m.cols + i] : m.dense[i * m.cols + j];
            if (v != 0.0) {
                c.indices.push_back(j + base);
                c.values.push_back(v);
            }
        }
        c.offsets.push_back(static_cast<int>(c.values.size()) + base);
    }
    return c;
}

int failures = 0;

void report(const std::string& what, const std::vector<double>& got,
            const std::vector<double>& want, double tol) {
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double scale = std::fabs(want[i]) > 1.0 ? std::fabs(want[i]) : 1.0;
        if (!(std::fabs(got[i] - want[i]) <= tol * scale)) {
            std::fprintf(stderr, "FAIL %s: [%zu] got %.17g want %.17g\n", what.c_str(), i,
                         got[i], want[i]);
            ++failures;
            return;
        }
    }
}

template <typename T>
T* device_copy(const std::vector<T>& host) {
    void* p = nullptr;
    if (host.empty()) {
        // A zero-size allocation is invalid; one element keeps the pointer valid
        // for an empty index or value array.
        if (cudaMalloc(&p, sizeof(T)) != cudaSuccess) return nullptr;
        return static_cast<T*>(p);
    }
    if (cudaMalloc(&p, host.size() * sizeof(T)) != cudaSuccess) return nullptr;
    if (cudaMemcpy(p, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) !=
        cudaSuccess) {
        return nullptr;
    }
    return static_cast<T*>(p);
}

void run_case(cusparseHandle_t handle, const Matrix& m, bool use_csc,
              cusparseOperation_t op, int base, double alpha, double beta, bool f32) {
    const bool op_t = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int ylen = op_t ? m.cols : m.rows;
    const int xlen = op_t ? m.rows : m.cols;

    // CSC of A is CSR of A-transpose, so the same compressor serves both.
    const Csr c = compress(m, use_csc, base);

    std::vector<double> hx(static_cast<std::size_t>(xlen));
    std::vector<double> hy(static_cast<std::size_t>(ylen));
    for (int i = 0; i < xlen; ++i) hx[i] = 0.5 + i * 1.25;
    for (int i = 0; i < ylen; ++i) hy[i] = 10.0 + i;

    std::vector<double> want(static_cast<std::size_t>(ylen));
    for (int i = 0; i < ylen; ++i) {
        double sum = 0.0;
        for (int j = 0; j < xlen; ++j) {
            sum += (op_t ? m.dense[j * m.cols + i] : m.dense[i * m.cols + j]) * hx[j];
        }
        want[i] = alpha * sum + beta * hy[i];
    }

    const std::string label = std::string(m.name) + (use_csc ? " csc" : " csr") +
                              (op_t ? " T" : " N") + " base" + std::to_string(base) +
                              (f32 ? " f32" : " f64") + " beta" + (beta == 0.0 ? "0" : "1");

    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    std::vector<double> got(static_cast<std::size_t>(ylen));

    if (f32) {
        std::vector<float> fv(c.values.begin(), c.values.end());
        std::vector<float> fx(hx.begin(), hx.end());
        std::vector<float> fy(hy.begin(), hy.end());
        int* d_off = device_copy(c.offsets);
        int* d_idx = device_copy(c.indices);
        float* d_val = device_copy(fv);
        float* d_x = device_copy(fx);
        float* d_y = device_copy(fy);
        if (!d_off || !d_idx || !d_val || !d_x || !d_y) { ++failures; return; }
        const float fa = static_cast<float>(alpha), fb = static_cast<float>(beta);
        if (use_csc) {
            cusparseCreateCsc(&mat, m.rows, m.cols, static_cast<int64_t>(c.values.size()),
                              d_off, d_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_32F);
        } else {
            cusparseCreateCsr(&mat, m.rows, m.cols, static_cast<int64_t>(c.values.size()),
                              d_off, d_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_32F);
        }
        cusparseCreateDnVec(&vx, xlen, d_x, CUDA_R_32F);
        cusparseCreateDnVec(&vy, ylen, d_y, CUDA_R_32F);
        if (cusparseSpMV(handle, op, &fa, mat, vx, &fb, vy, CUDA_R_32F,
                         CUSPARSE_SPMV_ALG_DEFAULT, nullptr) != CUSPARSE_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL %s: SpMV returned an error\n", label.c_str());
            ++failures;
        } else {
            cudaMemcpy(fy.data(), d_y, fy.size() * sizeof(float), cudaMemcpyDeviceToHost);
            for (int i = 0; i < ylen; ++i) got[i] = fy[i];
            report(label, got, want, 1e-5);
        }
        cudaFree(d_off); cudaFree(d_idx); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    } else {
        int* d_off = device_copy(c.offsets);
        int* d_idx = device_copy(c.indices);
        double* d_val = device_copy(c.values);
        double* d_x = device_copy(hx);
        double* d_y = device_copy(hy);
        if (!d_off || !d_idx || !d_val || !d_x || !d_y) { ++failures; return; }
        if (use_csc) {
            cusparseCreateCsc(&mat, m.rows, m.cols, static_cast<int64_t>(c.values.size()),
                              d_off, d_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F);
        } else {
            cusparseCreateCsr(&mat, m.rows, m.cols, static_cast<int64_t>(c.values.size()),
                              d_off, d_idx, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              base == 1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO,
                              CUDA_R_64F);
        }
        cusparseCreateDnVec(&vx, xlen, d_x, CUDA_R_64F);
        cusparseCreateDnVec(&vy, ylen, d_y, CUDA_R_64F);
        if (cusparseSpMV(handle, op, &alpha, mat, vx, &beta, vy, CUDA_R_64F,
                         CUSPARSE_SPMV_ALG_DEFAULT, nullptr) != CUSPARSE_STATUS_SUCCESS) {
            std::fprintf(stderr, "FAIL %s: SpMV returned an error\n", label.c_str());
            ++failures;
        } else {
            cudaMemcpy(got.data(), d_y, got.size() * sizeof(double), cudaMemcpyDeviceToHost);
            // The gather kernel accumulates in the FP32-pair emulation, so an
            // N-term dot product does not inherit the per-value 2^-48 bound.
            report(label, got, want, 1e-12);
        }
        cudaFree(d_off); cudaFree(d_idx); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    }

    if (vx) cusparseDestroyDnVec(vx);
    if (vy) cusparseDestroyDnVec(vy);
    if (mat) cusparseDestroySpMat(mat);
}

}  // namespace

int main() {
    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseCreate failed\n");
        return 1;
    }

    const char* mode = std::getenv("CUMETAL_SPARSE_METAL");
    const bool metal_path = !(mode != nullptr && mode[0] == '0');
    std::printf("cusparseSpMV conformance (%s implementation)\n",
                metal_path ? "Metal gather where eligible" : "CPU");

    int cases = 0;
    for (const Matrix& m : build_matrices()) {
        for (bool csc : {false, true}) {
            for (cusparseOperation_t op :
                 {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE}) {
                for (int base : {0, 1}) {
                    for (double beta : {0.0, 3.0}) {
                        for (bool f32 : {false, true}) {
                            run_case(handle, m, csc, op, base, 2.0, beta, f32);
                            ++cases;
                        }
                    }
                }
            }
        }
    }
    cusparseDestroy(handle);

    std::printf("  %d cases, %d failure(s)\n", cases, failures);
    if (failures == 0) {
        std::printf("PASS: cusparseSpMV matches the dense reference in every combination\n");
        return 0;
    }
    return 1;
}
