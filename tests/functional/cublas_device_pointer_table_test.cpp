#include "cublas_v2.h"
#include "cuda_fp16.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstdio>

namespace {

bool check(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
    return condition;
}

bool near(float actual, float expected) {
    return std::fabs(actual - expected) <= 1.0e-3f * (1.0f + std::fabs(expected));
}

template <typename T>
bool allocate_and_copy(T** output, const T* input, std::size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(output), count * sizeof(T)) == cudaSuccess &&
           cudaMemcpy(*output, input, count * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}

}  // namespace

int main() {
    if (!check(cudaInit(0) == cudaSuccess, "cudaInit")) return 1;

    cublasHandle_t handle = nullptr;
    if (!check(cublasCreate(&handle) == CUBLAS_STATUS_SUCCESS, "cublasCreate")) return 1;

    constexpr int batch_count = 2;
    constexpr int dim = 2;
    const float identity[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    const float matrix[4] = {1.0f, 3.0f, 2.0f, 4.0f};
    const float zeros[4] = {};
    float *a[batch_count]{}, *b[batch_count]{}, *c[batch_count]{};
    for (int i = 0; i < batch_count; ++i) {
        if (!check(allocate_and_copy(&a[i], identity, 4), "allocate A") ||
            !check(allocate_and_copy(&b[i], matrix, 4), "allocate B") ||
            !check(allocate_and_copy(&c[i], zeros, 4), "allocate C")) {
            return 1;
        }
    }

    // Batched LU consumes the same device-resident pointer-table ABI. Check a
    // pivoted factorization, a singular matrix's per-batch info, and the
    // no-pivot CUDA spelling rather than accepting a host-only pointer table.
    {
        const double nonsingular[4] = {4.0, 2.0, 1.0, 3.0};
        const double singular[4] = {1.0, 2.0, 2.0, 4.0};
        double* lu[batch_count]{};
        if (!check(allocate_and_copy(&lu[0], nonsingular, 4), "allocate LU matrix 0") ||
            !check(allocate_and_copy(&lu[1], singular, 4), "allocate LU matrix 1")) {
            return 1;
        }
        double** lu_table = nullptr;
        int* pivots = nullptr;
        int* info = nullptr;
        if (!check(cudaMalloc(reinterpret_cast<void**>(&lu_table), sizeof(lu)) == cudaSuccess,
                   "allocate LU pointer table") ||
            !check(cudaMemcpy(lu_table, lu, sizeof(lu), cudaMemcpyHostToDevice) == cudaSuccess,
                   "copy LU pointer table") ||
            !check(cudaMalloc(reinterpret_cast<void**>(&pivots),
                              batch_count * dim * sizeof(int)) == cudaSuccess,
                   "allocate LU pivots") ||
            !check(cudaMalloc(reinterpret_cast<void**>(&info),
                              batch_count * sizeof(int)) == cudaSuccess,
                   "allocate LU info")) {
            return 1;
        }
        if (!check(cublasDgetrfBatched(handle, dim, lu_table, dim, pivots, info,
                                       batch_count) == CUBLAS_STATUS_SUCCESS,
                   "device-table DgetrfBatched")) {
            return 1;
        }
        int host_info[batch_count]{};
        int host_pivots[batch_count * dim]{};
        double host_lu[4]{};
        if (!check(cudaMemcpy(host_info, info, sizeof(host_info),
                              cudaMemcpyDeviceToHost) == cudaSuccess,
                   "copy batched LU info") ||
            !check(cudaMemcpy(host_pivots, pivots, sizeof(host_pivots),
                              cudaMemcpyDeviceToHost) == cudaSuccess,
                   "copy batched LU pivots") ||
            !check(cudaMemcpy(host_lu, lu[0], sizeof(host_lu),
                              cudaMemcpyDeviceToHost) == cudaSuccess,
                   "copy batched LU factors") ||
            !check(host_info[0] == 0 && host_info[1] == 2,
                   "batched LU per-matrix info") ||
            !check(std::fabs(host_lu[0] - 4.0) < 1.0e-12 &&
                       std::fabs(host_lu[1] - 0.5) < 1.0e-12 &&
                       std::fabs(host_lu[2] - 1.0) < 1.0e-12 &&
                       std::fabs(host_lu[3] - 2.5) < 1.0e-12 &&
                       host_pivots[0] == 1 && host_pivots[1] == 2,
                   "pivoted batched LU factors")) {
            return 1;
        }
        if (!check(cudaMemcpy(lu[0], nonsingular, sizeof(nonsingular),
                              cudaMemcpyHostToDevice) == cudaSuccess,
                   "reset no-pivot LU matrix") ||
            !check(cublasDgetrfBatched(handle, dim, lu_table, dim, nullptr, info, 1) ==
                       CUBLAS_STATUS_SUCCESS,
                   "no-pivot DgetrfBatched")) {
            return 1;
        }
        int no_pivot_info = -1;
        double no_pivot_lu[4]{};
        if (!check(cudaMemcpy(&no_pivot_info, info, sizeof(no_pivot_info),
                              cudaMemcpyDeviceToHost) == cudaSuccess &&
                       cudaMemcpy(no_pivot_lu, lu[0], sizeof(no_pivot_lu),
                                  cudaMemcpyDeviceToHost) == cudaSuccess &&
                       no_pivot_info == 0 &&
                       std::fabs(no_pivot_lu[1] - 0.5) < 1.0e-12 &&
                       std::fabs(no_pivot_lu[3] - 2.5) < 1.0e-12,
                   "copy and verify no-pivot DgetrfBatched")) {
            return 1;
        }
        if (!check(cublasDgetrfBatched(handle, dim, lu_table, dim, pivots, info,
                                       batch_count + 1) == CUBLAS_STATUS_INVALID_VALUE,
                   "reject truncated LU pointer table")) {
            return 1;
        }
        cudaFree(info);
        cudaFree(pivots);
        cudaFree(lu_table);
        for (double* matrix_ptr : lu) cudaFree(matrix_ptr);

        float* slu = nullptr;
        float** slu_table = nullptr;
        int* sinfo = nullptr;
        const float float_matrix[4] = {4.0f, 2.0f, 1.0f, 3.0f};
        if (!check(allocate_and_copy(&slu, float_matrix, 4), "allocate float LU matrix") ||
            !check(cudaMalloc(reinterpret_cast<void**>(&slu_table), sizeof(slu)) ==
                       cudaSuccess,
                   "allocate float LU table") ||
            !check(cudaMemcpy(slu_table, &slu, sizeof(slu), cudaMemcpyHostToDevice) ==
                       cudaSuccess,
                   "copy float LU table") ||
            !check(cudaMalloc(reinterpret_cast<void**>(&sinfo), sizeof(int)) == cudaSuccess,
                   "allocate float LU info") ||
            !check(cublasSgetrfBatched(handle, dim, slu_table, dim, nullptr, sinfo, 1) ==
                       CUBLAS_STATUS_SUCCESS,
                   "no-pivot SgetrfBatched")) {
            return 1;
        }
        int host_sinfo = -1;
        float host_slu[4]{};
        if (!check(cudaMemcpy(&host_sinfo, sinfo, sizeof(host_sinfo),
                              cudaMemcpyDeviceToHost) == cudaSuccess &&
                       cudaMemcpy(host_slu, slu, sizeof(host_slu),
                                  cudaMemcpyDeviceToHost) == cudaSuccess &&
                       host_sinfo == 0 && near(host_slu[1], 0.5f) &&
                       near(host_slu[3], 2.5f),
                   "copy and verify no-pivot SgetrfBatched")) {
            return 1;
        }
        cudaFree(sinfo);
        cudaFree(slu_table);
        cudaFree(slu);
    }

    const float** a_table = nullptr;
    const float** b_table = nullptr;
    float** c_table = nullptr;
    if (!check(cudaMalloc(reinterpret_cast<void**>(&a_table), sizeof(a)) == cudaSuccess,
               "allocate A pointer table") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&b_table), sizeof(b)) == cudaSuccess,
               "allocate B pointer table") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&c_table), sizeof(c)) == cudaSuccess,
               "allocate C pointer table") ||
        !check(cudaMemcpy(a_table, a, sizeof(a), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy A pointer table") ||
        !check(cudaMemcpy(b_table, b, sizeof(b), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy B pointer table") ||
        !check(cudaMemcpy(c_table, c, sizeof(c), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy C pointer table")) {
        return 1;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    if (!check(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  dim, dim, dim, &alpha,
                                  a_table, dim, b_table, dim, &beta,
                                  c_table, dim, batch_count) == CUBLAS_STATUS_SUCCESS,
               "device-resident SgemmBatched pointer tables")) {
        return 1;
    }
    for (int i = 0; i < batch_count; ++i) {
        float result[4]{};
        if (!check(cudaMemcpy(result, c[i], sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess,
                   "copy batched result") ||
            !check(near(result[0], 1.0f) && near(result[1], 3.0f) &&
                       near(result[2], 2.0f) && near(result[3], 4.0f),
                   "batched result values")) {
            return 1;
        }
    }

    __half half_identity[4];
    __half half_matrix[4];
    for (int i = 0; i < 4; ++i) {
        half_identity[i] = static_cast<__half>(identity[i]);
        half_matrix[i] = static_cast<__half>(matrix[i]);
    }
    __half *a16[batch_count]{}, *b16[batch_count]{};
    float* c32[batch_count]{};
    for (int i = 0; i < batch_count; ++i) {
        if (!check(allocate_and_copy(&a16[i], half_identity, 4), "allocate FP16 A") ||
            !check(allocate_and_copy(&b16[i], half_matrix, 4), "allocate FP16 B") ||
            !check(allocate_and_copy(&c32[i], zeros, 4), "allocate FP32 C")) {
            return 1;
        }
    }

    const void** a16_table = nullptr;
    const void** b16_table = nullptr;
    void** c32_table = nullptr;
    if (!check(cudaMalloc(reinterpret_cast<void**>(&a16_table), sizeof(a16)) == cudaSuccess,
               "allocate FP16 A pointer table") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&b16_table), sizeof(b16)) == cudaSuccess,
               "allocate FP16 B pointer table") ||
        !check(cudaMalloc(reinterpret_cast<void**>(&c32_table), sizeof(c32)) == cudaSuccess,
               "allocate FP32 C pointer table") ||
        !check(cudaMemcpy(a16_table, a16, sizeof(a16), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy FP16 A pointer table") ||
        !check(cudaMemcpy(b16_table, b16, sizeof(b16), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy FP16 B pointer table") ||
        !check(cudaMemcpy(c32_table, c32, sizeof(c32), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy FP32 C pointer table") ||
        !check(cublasGemmBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                   dim, dim, dim, &alpha,
                                   a16_table, CUDA_R_16F, dim,
                                   b16_table, CUDA_R_16F, dim,
                                   &beta, c32_table, CUDA_R_32F, dim,
                                   batch_count, CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT) == CUBLAS_STATUS_SUCCESS,
               "device-table mixed GemmBatchedEx")) {
        return 1;
    }
    for (int i = 0; i < batch_count; ++i) {
        float result[4]{};
        if (!check(cudaMemcpy(result, c32[i], sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess,
                   "copy mixed batched result") ||
            !check(near(result[0], 1.0f) && near(result[1], 3.0f) &&
                       near(result[2], 2.0f) && near(result[3], 4.0f),
                   "mixed batched result values")) {
            return 1;
        }
    }

    // A device table that is too short for batch_count must fail cleanly.
    const float** short_table = nullptr;
    if (!check(cudaMalloc(reinterpret_cast<void**>(&short_table), sizeof(void*)) == cudaSuccess,
               "allocate short pointer table") ||
        !check(cudaMemcpy(short_table, a, sizeof(void*), cudaMemcpyHostToDevice) == cudaSuccess,
               "copy short pointer table") ||
        !check(cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                  dim, dim, dim, &alpha,
                                  short_table, dim, b_table, dim, &beta,
                                  c_table, dim, batch_count) == CUBLAS_STATUS_INVALID_VALUE,
               "reject truncated device pointer table")) {
        return 1;
    }

    for (int i = 0; i < batch_count; ++i) {
        cudaFree(a[i]);
        cudaFree(b[i]);
        cudaFree(c[i]);
        cudaFree(a16[i]);
        cudaFree(b16[i]);
        cudaFree(c32[i]);
    }
    cudaFree(a_table);
    cudaFree(b_table);
    cudaFree(c_table);
    cudaFree(short_table);
    cudaFree(a16_table);
    cudaFree(b16_table);
    cudaFree(c32_table);
    cublasDestroy(handle);
    std::puts("PASS: GPU-address pointer-table identity");
    return 0;
}
