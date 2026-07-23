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
