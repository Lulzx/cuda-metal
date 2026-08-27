// Capturing cusparseSpMV into a CUDA graph and replaying it.
//
// This is the shape HiGHS's own PDLP GPU path uses: it captures a fixed run of
// iterations once, then replays that graph every outer step, so a cuSPARSE call
// that executes eagerly during capture would both compute at the wrong time and
// be missing from every replay.
//
// Three things are checked, and the first two are the ones a stub would pass by
// accident:
//   1. capture does not execute -- y is untouched between BeginCapture and
//      EndCapture;
//   2. replay does execute, and with the operands the node was captured with;
//   3. replay reads device memory as it stands at launch, not as it stood at
//      capture, which is the whole reason a graph is worth building.
#include "cusparse.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int failures = 0;

void check(const char* what, double got, double want) {
    const double scale = std::fabs(want) > 1.0 ? std::fabs(want) : 1.0;
    if (!(std::fabs(got - want) <= 1e-12 * scale)) {
        std::fprintf(stderr, "FAIL %s: got %.17g want %.17g\n", what, got, want);
        ++failures;
    }
}

}  // namespace

int main() {
    // A 4x5 matrix with uneven rows, one of them empty.
    const int rows = 4, cols = 5;
    const std::vector<int> offsets{0, 2, 2, 5, 7};
    const std::vector<int> indices{0, 3, 1, 2, 4, 0, 4};
    const std::vector<double> values{1.5, -2.0, 3.0, 0.5, 4.0, -1.0, 2.5};
    const int nnz = static_cast<int>(values.size());

    auto reference = [&](const std::vector<double>& x, double alpha, double beta,
                         const std::vector<double>& y_in) {
        std::vector<double> y(rows);
        for (int i = 0; i < rows; ++i) {
            double sum = 0.0;
            for (int k = offsets[i]; k < offsets[i + 1]; ++k) sum += values[k] * x[indices[k]];
            y[i] = alpha * sum + beta * y_in[i];
        }
        return y;
    };

    cusparseHandle_t handle = nullptr;
    if (cusparseCreate(&handle) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseCreate\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate\n");
        return 1;
    }
    cusparseSetStream(handle, stream);

    void *d_off = nullptr, *d_idx = nullptr, *d_val = nullptr, *d_x = nullptr, *d_y = nullptr;
    cudaMalloc(&d_off, offsets.size() * sizeof(int));
    cudaMalloc(&d_idx, indices.size() * sizeof(int));
    cudaMalloc(&d_val, values.size() * sizeof(double));
    cudaMalloc(&d_x, cols * sizeof(double));
    cudaMalloc(&d_y, rows * sizeof(double));
    cudaMemcpy(d_off, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idx, indices.data(), indices.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, values.data(), values.size() * sizeof(double), cudaMemcpyHostToDevice);

    cusparseSpMatDescr_t mat = nullptr;
    cusparseCreateCsr(&mat, rows, cols, nnz, d_off, d_idx, d_val, CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
    cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
    cusparseCreateDnVec(&vx, cols, d_x, CUDA_R_64F);
    cusparseCreateDnVec(&vy, rows, d_y, CUDA_R_64F);

    // The analysis hook is optional; calling it must not change the answer.
    const double alpha = 2.0, beta = 0.0;
    if (cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta,
                                vy, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                                nullptr) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseSpMV_preprocess returned an error\n");
        ++failures;
    }

    // Repointing a dense descriptor is how the caller alternates buffers.
    void* d_x_alias = d_x;
    if (cusparseDnVecSetValues(vx, d_x_alias) != CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseDnVecSetValues returned an error\n");
        ++failures;
    }
    void* readback = nullptr;
    cusparseDnVecGetValues(vx, &readback);
    if (readback != d_x_alias) {
        std::fprintf(stderr, "FAIL: DnVecGetValues did not return what SetValues stored\n");
        ++failures;
    }

    std::vector<double> x_capture{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y_zero(rows, 0.0);
    cudaMemcpy(d_x, x_capture.data(), cols * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y_zero.data(), rows * sizeof(double), cudaMemcpyHostToDevice);

    // ── 1. capture must not execute ─────────────────────────────────────────
    if (cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamBeginCapture\n");
        return 1;
    }
    if (cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta, vy,
                     CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr) !=
        CUSPARSE_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cusparseSpMV during capture returned an error\n");
        ++failures;
    }
    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(stream, &graph) != cudaSuccess || graph == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamEndCapture\n");
        return 1;
    }
    cudaDeviceSynchronize();
    std::vector<double> got(rows, -1.0);
    cudaMemcpy(got.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
    for (int i = 0; i < rows; ++i) check("capture executed the SpMV", got[i], 0.0);

    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess || exec == nullptr) {
        std::fprintf(stderr, "FAIL: cudaGraphInstantiate\n");
        return 1;
    }
    cudaGraphDestroy(graph);

    // ── 2. replay executes with the captured operands ───────────────────────
    if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGraphLaunch\n");
        return 1;
    }
    cudaStreamSynchronize(stream);
    cudaMemcpy(got.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
    std::vector<double> want = reference(x_capture, alpha, beta, y_zero);
    for (int i = 0; i < rows; ++i) check("first replay", got[i], want[i]);

    // ── 3. replay reads device memory as it stands at launch ────────────────
    std::vector<double> x_later{-1.0, 0.5, 7.0, 2.0, -3.0};
    cudaMemcpy(d_x, x_later.data(), cols * sizeof(double), cudaMemcpyHostToDevice);
    for (int replay = 0; replay < 3; ++replay) {
        if (cudaGraphLaunch(exec, stream) != cudaSuccess) {
            std::fprintf(stderr, "FAIL: cudaGraphLaunch (replay %d)\n", replay);
            return 1;
        }
        cudaStreamSynchronize(stream);
        cudaMemcpy(got.data(), d_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
        want = reference(x_later, alpha, beta, y_zero);
        for (int i = 0; i < rows; ++i) check("replay with new x", got[i], want[i]);
    }

    cudaGraphExecDestroy(exec);
    cusparseDestroyDnVec(vx);
    cusparseDestroyDnVec(vy);
    cusparseDestroySpMat(mat);
    cudaFree(d_off); cudaFree(d_idx); cudaFree(d_val); cudaFree(d_x); cudaFree(d_y);
    cudaStreamDestroy(stream);
    cusparseDestroy(handle);

    if (failures == 0) {
        std::printf("PASS: cusparseSpMV captures into a graph and replays correctly\n");
        return 0;
    }
    std::printf("  %d failure(s)\n", failures);
    return 1;
}
