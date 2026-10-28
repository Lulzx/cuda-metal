#include "cublasLt.h"
#include "cuda_runtime.h"
#include "cuda_fp16.h"
#include "cuda_bf16.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>

static bool test_handle_lifecycle() {
    cublasLtHandle_t handle = nullptr;
    cublasStatus_t st = cublasLtCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cublasLtCreate returned %d\n", st);
        return false;
    }
    st = cublasLtDestroy(handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasLtDestroy returned %d\n", st);
        return false;
    }
    return true;
}

static bool test_matmul_desc() {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS || desc == nullptr) {
        std::fprintf(stderr, "FAIL: cublasLtMatmulDescCreate returned %d\n", st);
        return false;
    }

    cublasOperation_t transa = CUBLAS_OP_T;
    st = cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                        &transa, sizeof(transa));
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: SetAttribute TRANSA returned %d\n", st);
        return false;
    }

    cublasOperation_t got = CUBLAS_OP_N;
    size_t written = 0;
    st = cublasLtMatmulDescGetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                        &got, sizeof(got), &written);
    if (st != CUBLAS_STATUS_SUCCESS || got != CUBLAS_OP_T) {
        std::fprintf(stderr, "FAIL: GetAttribute TRANSA got %d (expected %d)\n", got, CUBLAS_OP_T);
        return false;
    }

    st = cublasLtMatmulDescDestroy(desc);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasLtMatmulDescDestroy returned %d\n", st);
        return false;
    }
    return true;
}

static bool test_matrix_layout() {
    cublasLtMatrixLayout_t layout = nullptr;
    cublasStatus_t st = cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, 4, 4, 4);
    if (st != CUBLAS_STATUS_SUCCESS || layout == nullptr) {
        std::fprintf(stderr, "FAIL: cublasLtMatrixLayoutCreate returned %d\n", st);
        return false;
    }

    int32_t batch = 3;
    st = cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                          &batch, sizeof(batch));
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: SetAttribute BATCH_COUNT returned %d\n", st);
        return false;
    }

    st = cublasLtMatrixLayoutDestroy(layout);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasLtMatrixLayoutDestroy returned %d\n", st);
        return false;
    }
    return true;
}

static bool test_preference_and_heuristic() {
    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 4, 4, 4);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 4, 4, 4);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 4, 4, 4);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 4, 4, 4);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    size_t ws = 1024;
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t results[4];
    int count = 0;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(handle, desc,
                                                        Adesc, Bdesc, Cdesc, Ddesc,
                                                        pref, 4, results, &count);
    if (st != CUBLAS_STATUS_SUCCESS || count != 1) {
        std::fprintf(stderr, "FAIL: AlgoGetHeuristic returned %d, count=%d\n", st, count);
        return false;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_sgemm_via_lt() {
    // D = alpha * A * B + beta * C, with 2x2 matrices
    // A = [1 3; 2 4] (col-major), B = [5 7; 6 8], C = [0 0; 0 0]
    // A*B = [1*5+3*6, 1*7+3*8; 2*5+4*6, 2*7+4*8] = [23 31; 34 46]
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[] = {0, 0, 0, 0};
    float D[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 2, 2, 2);

    cublasStatus_t st = cublasLtMatmul(handle, desc,
                                        &alpha, A, Adesc, B, Bdesc,
                                        &beta, C, Cdesc, D, Ddesc,
                                        nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasLtMatmul returned %d\n", st);
        return false;
    }

    float expected[] = {23, 34, 31, 46};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(D[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: D[%d] = %f, expected %f\n", i, D[i], expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_epilogue_bias() {
    // D = A * B + bias, 2x2
    float A[] = {1, 0, 0, 1}; // identity
    float B[] = {2, 3, 4, 5};
    float C[] = {0, 0, 0, 0};
    float D[4] = {};
    float bias[] = {10, 20}; // added to each column
    float alpha = 1.0f, beta = 0.0f;

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_BIAS;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    const void* bp = bias;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bp, sizeof(bp));

    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 2, 2, 2);

    cublasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc,
                   &beta, C, Cdesc, D, Ddesc,
                   nullptr, nullptr, 0, nullptr);

    // I*B = B = [2 4; 3 5], then + bias [10,20] per row => [12 14; 23 25]
    float expected[] = {12, 23, 14, 25};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(D[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: bias D[%d] = %f, expected %f\n", i, D[i], expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_hgemm_via_lt() {
    // Same 2x2 product as the FP32 test, in FP16 on host buffers
    // (float-domain fallback) and on device buffers (MPS fast path).
    const float a_vals[] = {1, 2, 3, 4};
    const float b_vals[] = {5, 6, 7, 8};
    const float expected[] = {23, 34, 31, 46};
    __half A[4], B[4], C[4] = {}, D[4] = {};
    for (int i = 0; i < 4; ++i) {
        A[i] = static_cast<__half>(a_vals[i]);
        B[i] = static_cast<__half>(b_vals[i]);
    }
    float alpha = 1.0f, beta = 0.0f;

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t desc = nullptr;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_16F, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: FP16 Lt desc create\n");
        return false;
    }
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                           Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, 2, 2, 2);

    cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc,
                                       &beta, C, Cdesc, D, Ddesc, nullptr,
                                       nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: host FP16 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(static_cast<float>(D[i]) - expected[i]) > 0.05f) {
            std::fprintf(stderr, "FAIL: host FP16 D[%d] = %f, expected %f\n", i,
                         static_cast<float>(D[i]), expected[i]);
            return false;
        }
    }

    // Device buffers exercise the MPS gemm_f16 path.
    __half *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(A)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dB), sizeof(B)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dC), sizeof(C)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dD), sizeof(D)) != cudaSuccess ||
        cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dC, C, sizeof(C), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: FP16 Lt device setup\n");
        return false;
    }
    st = cublasLtMatmul(handle, desc, &alpha, dA, Adesc, dB, Bdesc, &beta, dC,
                        Cdesc, dD, Ddesc, nullptr, nullptr, 0, nullptr);
    __half out[4] = {};
    cudaMemcpy(out, dD, sizeof(out), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: device FP16 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(static_cast<float>(out[i]) - expected[i]) > 0.05f) {
            std::fprintf(stderr, "FAIL: device FP16 D[%d] = %f, expected %f\n",
                         i, static_cast<float>(out[i]), expected[i]);
            return false;
        }
    }

    // Mixed precision: FP16 inputs, FP32 output, 32F compute.
    cublasLtMatmulDesc_t desc32 = nullptr;
    cublasLtMatrixLayout_t C32 = nullptr, D32 = nullptr;
    cublasLtMatmulDescCreate(&desc32, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&C32, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&D32, CUDA_R_32F, 2, 2, 2);
    float hostC[4] = {}, hostD[4] = {};
    st = cublasLtMatmul(handle, desc32, &alpha, A, Adesc, B, Bdesc, &beta,
                        hostC, C32, hostD, D32, nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: mixed FP16->FP32 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(hostD[i] - expected[i]) > 1e-4f) {
            std::fprintf(stderr, "FAIL: mixed D[%d] = %f, expected %f\n", i,
                         hostD[i], expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayoutDestroy(D32);
    cublasLtMatrixLayoutDestroy(C32);
    cublasLtMatmulDescDestroy(desc32);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_epilogue_device() {
    // BIAS and GELU epilogues on device buffers exercise the MPSGraph
    // backend path (host buffers keep the CPU epilogue path).
    float A[] = {1, 0, 0, 1};
    float B[] = {2, -3, 4, -5};
    float C[] = {0, 0, 0, 0};
    float bias[] = {10, 20};
    float alpha = 1.0f, beta = 0.0f;

    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr,
          *dBias = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dA), sizeof(A)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dB), sizeof(B)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dC), sizeof(C)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dD), 4 * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dBias), sizeof(bias)) != cudaSuccess ||
        cudaMemcpy(dA, A, sizeof(A), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dB, B, sizeof(B), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dC, C, sizeof(C), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dBias, bias, sizeof(bias), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: epilogue device setup\n");
        return false;
    }

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                           Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 2, 2, 2);

    auto run_epilogue = [&](cublasLtEpilogue_t epi, float* out) {
        cublasLtMatmulDesc_t desc = nullptr;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epi, sizeof(epi));
        if (epi == CUBLASLT_EPILOGUE_BIAS || epi == CUBLASLT_EPILOGUE_GELU_BIAS) {
            const void* bp = dBias;
            cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &bp, sizeof(bp));
        }
        cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha, dA, Adesc, dB,
                                           Bdesc, &beta, dC, Cdesc, dD, Ddesc,
                                           nullptr, nullptr, 0, nullptr);
        cublasLtMatmulDescDestroy(desc);
        if (st != CUBLAS_STATUS_SUCCESS) return false;
        return cudaMemcpy(out, dD, 4 * sizeof(float),
                          cudaMemcpyDeviceToHost) == cudaSuccess;
    };

    float out[4] = {};
    // I*B + bias = [2+10, -3+20, 4+10, -5+20] (col-major)
    if (!run_epilogue(CUBLASLT_EPILOGUE_BIAS, out)) {
        std::fprintf(stderr, "FAIL: device BIAS epilogue run\n");
        return false;
    }
    const float expected_bias[] = {12, 17, 14, 15};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(out[i] - expected_bias[i]) > 1e-4f) {
            std::fprintf(stderr, "FAIL: device BIAS D[%d] = %f, expected %f\n",
                         i, out[i], expected_bias[i]);
            return false;
        }
    }
    // GELU over I*B = [2, -3, 4, -5]
    if (!run_epilogue(CUBLASLT_EPILOGUE_GELU, out)) {
        std::fprintf(stderr, "FAIL: device GELU epilogue run\n");
        return false;
    }
    auto gelu = [](float x) {
        return x * 0.5f * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    };
    const float gelu_in[] = {2, -3, 4, -5};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(out[i] - gelu(gelu_in[i])) > 2e-3f) {
            std::fprintf(stderr, "FAIL: device GELU D[%d] = %f, expected %f\n",
                         i, out[i], gelu(gelu_in[i]));
            return false;
        }
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD); cudaFree(dBias);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_bf16_via_lt() {
    // BF16 inputs under 32F compute, BF16 and FP32 outputs (host path).
    const float a_vals[] = {1, 2, 3, 4};
    const float b_vals[] = {5, 6, 7, 8};
    const float expected[] = {23, 34, 31, 46};
    __nv_bfloat16 A[4], B[4], C[4] = {}, D[4] = {};
    for (int i = 0; i < 4; ++i) {
        A[i] = static_cast<__nv_bfloat16>(a_vals[i]);
        B[i] = static_cast<__nv_bfloat16>(b_vals[i]);
    }
    float alpha = 1.0f, beta = 0.0f;

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t desc = nullptr;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: BF16 Lt desc create\n");
        return false;
    }
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                           Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_16BF, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_16BF, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, 2, 2, 2);

    cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc,
                                       &beta, C, Cdesc, D, Ddesc, nullptr,
                                       nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: BF16 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(static_cast<float>(D[i]) - expected[i]) > 0.5f) {
            std::fprintf(stderr, "FAIL: BF16 D[%d] = %f, expected %f\n", i,
                         static_cast<float>(D[i]), expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayout_t C32 = nullptr, D32 = nullptr;
    cublasLtMatrixLayoutCreate(&C32, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&D32, CUDA_R_32F, 2, 2, 2);
    float hostC[4] = {}, hostD[4] = {};
    st = cublasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc, &beta, hostC,
                        C32, hostD, D32, nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: BF16->FP32 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(hostD[i] - expected[i]) > 0.5f) {
            std::fprintf(stderr, "FAIL: BF16->FP32 D[%d] = %f, expected %f\n", i,
                         hostD[i], expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayoutDestroy(D32);
    cublasLtMatrixLayoutDestroy(C32);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_int8_via_lt() {
    // INT8 inputs, INT32 output: A (2x3) * B (3x2) with alpha=2, beta=3.
    const int8_t A[] = {1, -2, 3, 4, 5, -6};
    const int8_t B[] = {7, 8, -9, 10, 11, 12};
    const int32_t C[] = {1, 2, 3, 4};
    int32_t D[] = {0, 0, 0, 0};
    // col-major: A = [1 3 5; -2 4 -6] rows... computed in col-major below.
    auto at = [](const int8_t* m, int ld, int r, int c) {
        return (int)m[c * ld + r];
    };
    int32_t expected[4];
    for (int c = 0; c < 2; ++c)
        for (int r = 0; r < 2; ++r) {
            int acc = 0;
            for (int k = 0; k < 3; ++k) acc += at(A, 2, r, k) * at(B, 3, k, c);
            expected[c * 2 + r] = 2 * acc + 3 * C[c * 2 + r];
        }
    int32_t alpha = 2, beta = 3;

    cublasLtHandle_t handle = nullptr;
    cublasLtCreate(&handle);
    cublasLtMatmulDesc_t desc = nullptr;
    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: INT8 Lt desc create\n");
        return false;
    }
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr,
                           Ddesc = nullptr;
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8I, 2, 3, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8I, 3, 2, 3);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32I, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32I, 2, 2, 2);

    cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha, A, Adesc, B, Bdesc,
                                       &beta, C, Cdesc, D, Ddesc, nullptr,
                                       nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: INT8 LtMatmul returned %d\n", st);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (D[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: INT8 D[%d] = %d, expected %d\n", i,
                         D[i], expected[i]);
            return false;
        }
    }

    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

static bool test_null_args() {
    if (cublasLtCreate(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtDestroy(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatmulDescCreate(nullptr, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatmulDescDestroy(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatrixLayoutCreate(nullptr, CUDA_R_32F, 1, 1, 1) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatrixLayoutDestroy(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatmulPreferenceCreate(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    if (cublasLtMatmulPreferenceDestroy(nullptr) != CUBLAS_STATUS_INVALID_VALUE) return false;
    return true;
}

static bool test_validation_and_stream_ordering() {
    cublasLtMatrixLayout_t invalid = nullptr;
    cublasLtMatmulDesc_t unsupportedDesc = nullptr;
    if (cublasLtMatrixLayoutCreate(&invalid, CUDA_R_32F, 0, 2, 2) !=
            CUBLAS_STATUS_INVALID_VALUE ||
        cublasLtMatrixLayoutCreate(&invalid, CUDA_C_32F, 2, 2, 2) !=
            CUBLAS_STATUS_NOT_SUPPORTED ||
        cublasLtMatrixLayoutCreate(&invalid, CUDA_R_32F, 2, 2, 1) !=
            CUBLAS_STATUS_INVALID_VALUE ||
        cublasLtMatmulDescCreate(&unsupportedDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F) !=
            CUBLAS_STATUS_NOT_SUPPORTED || unsupportedDesc != nullptr) {
        std::fprintf(stderr, "FAIL: cuBLASLt creation validation\n");
        return false;
    }

    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtCreate(&handle);
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, 2, 2, 2);
    cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, 2, 2, 2);

    int32_t zeroBatch = 0;
    if (cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &zeroBatch, sizeof(zeroBatch)) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: zero cuBLASLt batch count was accepted\n");
        return false;
    }
    int32_t batch = 0;
    size_t written = 0;
    cublasLtMatrixLayoutGetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                     &batch, sizeof(batch), &written);
    if (batch != 1) {
        std::fprintf(stderr, "FAIL: rejected batch count mutated layout\n");
        return false;
    }
    int32_t rowMajor = 1;
    if (cublasLtMatrixLayoutSetAttribute(Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER,
                                         &rowMajor, sizeof(rowMajor)) !=
        CUBLAS_STATUS_NOT_SUPPORTED) {
        std::fprintf(stderr, "FAIL: unsupported row-major layout was accepted\n");
        return false;
    }
    cublasOperation_t invalidOp = static_cast<cublasOperation_t>(99);
    if (cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &invalidOp, sizeof(invalidOp)) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: invalid cuBLASLt transpose was accepted\n");
        return false;
    }
    cublasOperation_t currentOp = CUBLAS_OP_T;
    cublasLtMatmulDescGetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &currentOp, sizeof(currentOp), &written);
    if (currentOp != CUBLAS_OP_N) {
        std::fprintf(stderr, "FAIL: rejected transpose mutated descriptor\n");
        return false;
    }

    float hostA[] = {1, 2, 3, 4};
    float hostB[] = {5, 6, 7, 8};
    float C[] = {0, 0, 0, 0};
    float D[] = {-1, -1, -1, -1};
    float alpha = 1.0f, beta = 0.0f;
    float *deviceA = nullptr, *deviceB = nullptr;
    cudaStream_t stream = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&deviceA), sizeof(hostA)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&deviceB), sizeof(hostB)) != cudaSuccess ||
        cudaStreamCreate(&stream) != cudaSuccess ||
        cudaMemcpyAsync(deviceA, hostA, sizeof(hostA), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemcpyAsync(deviceB, hostB, sizeof(hostB), cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cuBLASLt stream test setup\n");
        return false;
    }

    cublasStatus_t st = cublasLtMatmul(handle, desc, &alpha,
                                        deviceA, Adesc, deviceB, Bdesc,
                                        &beta, C, Cdesc, D, Ddesc,
                                        nullptr, nullptr, 0, stream);
    const float expected[] = {23, 34, 31, 46};
    for (int i = 0; i < 4; ++i) {
        if (st != CUBLAS_STATUS_SUCCESS || std::fabs(D[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: stream-ordered cuBLASLt D[%d]=%f status=%d\n",
                         i, D[i], st);
            return false;
        }
    }

    float* deviceShort = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&deviceShort), 3 * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cuBLASLt short allocation setup\n");
        return false;
    }
    D[0] = -13.0f;
    st = cublasLtMatmul(handle, desc, &alpha, deviceA + 1, Adesc, deviceB, Bdesc,
                        &beta, C, Cdesc, D, Ddesc,
                        nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_INVALID_VALUE || D[0] != -13.0f) {
        std::fprintf(stderr, "FAIL: interior-short cuBLASLt input was accepted\n");
        return false;
    }
    st = cublasLtMatmul(handle, desc, &alpha, deviceA, Adesc, deviceB, Bdesc,
                        &beta, C, Cdesc, deviceShort, Ddesc,
                        nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: undersized cuBLASLt output was accepted\n");
        return false;
    }
    cudaFree(deviceShort);

    int32_t twoBatches = 2;
    for (cublasLtMatrixLayout_t layout : {Adesc, Bdesc, Cdesc, Ddesc}) {
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &twoBatches, sizeof(twoBatches));
    }
    D[0] = -11.0f;
    st = cublasLtMatmul(handle, desc, &alpha, hostA, Adesc, hostB, Bdesc,
                        &beta, C, Cdesc, D, Ddesc,
                        nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_NOT_SUPPORTED || D[0] != -11.0f) {
        std::fprintf(stderr, "FAIL: overlapping batched cuBLASLt layout was accepted\n");
        return false;
    }
    int32_t oneBatch = 1;
    for (cublasLtMatrixLayout_t layout : {Adesc, Bdesc, Cdesc, Ddesc}) {
        cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                         &oneBatch, sizeof(oneBatch));
    }

    cublasLtMatrixLayout_t badD = nullptr;
    cublasLtMatrixLayoutCreate(&badD, CUDA_R_32F, 3, 2, 3);
    float sentinel[] = {-7, -7, -7, -7, -7, -7};
    st = cublasLtMatmul(handle, desc, &alpha, hostA, Adesc, hostB, Bdesc,
                        &beta, sentinel, badD, sentinel, badD,
                        nullptr, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_INVALID_VALUE || sentinel[0] != -7.0f) {
        std::fprintf(stderr, "FAIL: invalid cuBLASLt geometry wrote output or returned %d\n", st);
        return false;
    }
    int fakeAlgo = 0;
    st = cublasLtMatmul(handle, desc, &alpha, hostA, Adesc, hostB, Bdesc,
                        &beta, C, Cdesc, D, Ddesc,
                        &fakeAlgo, nullptr, 0, nullptr);
    if (st != CUBLAS_STATUS_NOT_SUPPORTED) {
        std::fprintf(stderr, "FAIL: unsupported cuBLASLt algorithm was accepted\n");
        return false;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(deviceB);
    cudaFree(deviceA);
    cublasLtMatrixLayoutDestroy(badD);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(handle);
    return true;
}

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_matmul_desc()) return 1;
    if (!test_matrix_layout()) return 1;
    if (!test_preference_and_heuristic()) return 1;
    if (!test_sgemm_via_lt()) return 1;
    if (!test_hgemm_via_lt()) return 1;
    if (!test_bf16_via_lt()) return 1;
    if (!test_int8_via_lt()) return 1;
    if (!test_epilogue_device()) return 1;
    if (!test_epilogue_bias()) return 1;
    if (!test_null_args()) return 1;
    if (!test_validation_and_stream_ordering()) return 1;

    std::printf("PASS: cublasLt API tests\n");
    return 0;
}
