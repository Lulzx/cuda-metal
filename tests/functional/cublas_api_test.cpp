#include "cublas_v2.h"
#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main() {
    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    if (cublasCreate(nullptr) != CUBLAS_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_NOT_INITIALIZED for null handle out ptr\n");
        return 1;
    }

    cublasHandle_t handle = nullptr;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cublasCreate failed\n");
        return 1;
    }
    int cublas_version = 0;
    if (cublasGetVersion(handle, &cublas_version) != CUBLAS_STATUS_SUCCESS || cublas_version <= 0) {
        std::fprintf(stderr, "FAIL: cublasGetVersion failed\n");
        return 1;
    }
    if (cublasGetVersion(handle, nullptr) != CUBLAS_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_NOT_INITIALIZED for null version ptr\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }
    if (cublasSetStream(handle, stream) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSetStream failed\n");
        return 1;
    }
    cudaStream_t queried_stream = nullptr;
    if (cublasGetStream(handle, &queried_stream) != CUBLAS_STATUS_SUCCESS || queried_stream != stream) {
        std::fprintf(stderr, "FAIL: cublasGetStream mismatch\n");
        return 1;
    }
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    if (cublasGetMathMode(handle, &math_mode) != CUBLAS_STATUS_SUCCESS ||
        math_mode != CUBLAS_DEFAULT_MATH) {
        std::fprintf(stderr, "FAIL: cublasGetMathMode default mismatch\n");
        return 1;
    }
    if (cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSetMathMode failed\n");
        return 1;
    }
    if (cublasGetMathMode(handle, &math_mode) != CUBLAS_STATUS_SUCCESS ||
        math_mode != CUBLAS_TF32_TENSOR_OP_MATH) {
        std::fprintf(stderr, "FAIL: cublasGetMathMode updated mismatch\n");
        return 1;
    }
    if (cublasGetMathMode(handle, nullptr) != CUBLAS_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_NOT_INITIALIZED for null math-mode ptr\n");
        return 1;
    }
    if (cublasSetMathMode(handle, static_cast<cublasMath_t>(-1)) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid math mode\n");
        return 1;
    }

    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_DEVICE;
    if (cublasGetPointerMode(handle, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
        pointer_mode != CUBLAS_POINTER_MODE_HOST) {
        std::fprintf(stderr, "FAIL: cublasGetPointerMode default mismatch\n");
        return 1;
    }
    if (cublasGetPointerMode(handle, nullptr) != CUBLAS_STATUS_NOT_INITIALIZED ||
        cublasSetPointerMode(handle, static_cast<cublasPointerMode_t>(-1)) !=
            CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuBLAS pointer-mode negative path mismatch\n");
        return 1;
    }

    constexpr int kPointerModeCount = 3;
    const float pointer_mode_x[kPointerModeCount] = {1.0f, 2.0f, 3.0f};
    const float pointer_mode_y[kPointerModeCount] = {4.0f, 5.0f, 6.0f};
    const float pointer_mode_alpha = 2.0f;
    float* dev_pointer_mode_x = nullptr;
    float* dev_pointer_mode_y = nullptr;
    float* dev_pointer_mode_scalar = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_pointer_mode_x), sizeof(pointer_mode_x)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_pointer_mode_y), sizeof(pointer_mode_y)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_pointer_mode_scalar), sizeof(float)) !=
            cudaSuccess ||
        cudaMemcpy(dev_pointer_mode_x, pointer_mode_x, sizeof(pointer_mode_x),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_pointer_mode_y, pointer_mode_y, sizeof(pointer_mode_y),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_pointer_mode_scalar, &pointer_mode_alpha, sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: pointer-mode setup\n");
        return 1;
    }
    if (cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE) != CUBLAS_STATUS_SUCCESS ||
        cublasGetPointerMode(handle, &pointer_mode) != CUBLAS_STATUS_SUCCESS ||
        pointer_mode != CUBLAS_POINTER_MODE_DEVICE ||
        cublasSaxpy(handle, kPointerModeCount, dev_pointer_mode_scalar,
                    dev_pointer_mode_x, 1, dev_pointer_mode_y, 1) != CUBLAS_STATUS_SUCCESS ||
        cublasSscal(handle, kPointerModeCount, dev_pointer_mode_scalar,
                    dev_pointer_mode_x, 1) != CUBLAS_STATUS_SUCCESS ||
        cublasSdot(handle, kPointerModeCount, dev_pointer_mode_x, 1,
                   dev_pointer_mode_y, 1, dev_pointer_mode_scalar) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cuBLAS device pointer mode operations\n");
        return 1;
    }
    float pointer_mode_result = 0.0f;
    float pointer_mode_x_result[kPointerModeCount] = {};
    float pointer_mode_y_result[kPointerModeCount] = {};
    if (cudaMemcpy(&pointer_mode_result, dev_pointer_mode_scalar, sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(pointer_mode_x_result, dev_pointer_mode_x, sizeof(pointer_mode_x_result),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(pointer_mode_y_result, dev_pointer_mode_y, sizeof(pointer_mode_y_result),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        !nearly_equal(pointer_mode_result, 120.0f) ||
        !nearly_equal(pointer_mode_x_result[0], 2.0f) ||
        !nearly_equal(pointer_mode_x_result[1], 4.0f) ||
        !nearly_equal(pointer_mode_x_result[2], 6.0f) ||
        !nearly_equal(pointer_mode_y_result[0], 6.0f) ||
        !nearly_equal(pointer_mode_y_result[1], 9.0f) ||
        !nearly_equal(pointer_mode_y_result[2], 12.0f)) {
        std::fprintf(stderr, "FAIL: cuBLAS device pointer mode result mismatch\n");
        return 1;
    }
    if (cublasSaxpy(handle, kPointerModeCount, &pointer_mode_alpha,
                    dev_pointer_mode_x, 1, dev_pointer_mode_y, 1) !=
            CUBLAS_STATUS_INVALID_VALUE ||
        cublasSdot(handle, kPointerModeCount, dev_pointer_mode_x, 1,
                   dev_pointer_mode_y, 1, &pointer_mode_result) !=
            CUBLAS_STATUS_INVALID_VALUE ||
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST) != CUBLAS_STATUS_SUCCESS ||
        cublasSscal(handle, kPointerModeCount, dev_pointer_mode_scalar,
                    dev_pointer_mode_x, 1) != CUBLAS_STATUS_INVALID_VALUE ||
        cublasSdot(handle, kPointerModeCount, dev_pointer_mode_x, 1,
                   dev_pointer_mode_y, 1, dev_pointer_mode_scalar) !=
            CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: cuBLAS pointer mode did not reject wrong scalar location\n");
        return 1;
    }
    if (cudaFree(dev_pointer_mode_x) != cudaSuccess ||
        cudaFree(dev_pointer_mode_y) != cudaSuccess ||
        cudaFree(dev_pointer_mode_scalar) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: pointer-mode cleanup\n");
        return 1;
    }

    constexpr int kVecCount = 2048;
    std::vector<float> host_x(kVecCount);
    std::vector<float> host_y(kVecCount);
    std::vector<float> expected_y(kVecCount);
    for (int i = 0; i < kVecCount; ++i) {
        host_x[i] = static_cast<float>((i * 7) % 29) * 0.25f;
        host_y[i] = static_cast<float>((i * 3) % 17) * 0.5f;
        expected_y[i] = host_y[i];
    }
    const float alpha_axpy = 1.75f;
    for (int i = 0; i < kVecCount; ++i) {
        expected_y[i] = alpha_axpy * host_x[i] + expected_y[i];
    }

    float* dev_x = nullptr;
    float* dev_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_x), host_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_y), host_y.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SAXPY failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_x, host_x.data(), host_x.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_y, host_y.data(), host_y.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SAXPY failed\n");
        return 1;
    }

    if (cublasSaxpy(handle, kVecCount, &alpha_axpy, dev_x, 1, dev_y, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSaxpy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_y.data(), dev_y, host_y.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SAXPY failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_y[i], expected_y[i])) {
            std::fprintf(stderr,
                         "FAIL: SAXPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_y[i]),
                         static_cast<double>(expected_y[i]));
            return 1;
        }
    }

    if (cublasSaxpy(handle, kVecCount, &alpha_axpy, host_x.data(), 1, dev_y, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host x pointer\n");
        return 1;
    }

    const float alpha_scal = -0.5f;
    std::vector<float> expected_scal = host_x;
    for (float& value : expected_scal) {
        value *= alpha_scal;
    }
    if (cublasSscal(handle, kVecCount, &alpha_scal, dev_x, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSscal failed\n");
        return 1;
    }
    if (cudaMemcpy(host_x.data(), dev_x, host_x.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SSCAL failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_x[i], expected_scal[i])) {
            std::fprintf(stderr,
                         "FAIL: SSCAL mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_x[i]),
                         static_cast<double>(expected_scal[i]));
            return 1;
        }
    }
    if (cublasSscal(handle, kVecCount, &alpha_scal, expected_scal.data(), 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host x in SSCAL\n");
        return 1;
    }
    if (cublasSscal(handle, kVecCount, nullptr, dev_x, 1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for null alpha in SSCAL\n");
        return 1;
    }
    std::vector<float> expected_swap_x = host_y;
    std::vector<float> expected_swap_y = host_x;
    if (cublasSswap(handle, kVecCount, dev_x, 1, dev_y, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSswap failed\n");
        return 1;
    }
    if (cudaMemcpy(host_x.data(), dev_x, host_x.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        cudaMemcpy(host_y.data(), dev_y, host_y.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SSWAP failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_x[i], expected_swap_x[i]) || !nearly_equal(host_y[i], expected_swap_y[i])) {
            std::fprintf(stderr, "FAIL: SSWAP mismatch at %d\n", i);
            return 1;
        }
    }
    if (cublasSswap(handle, kVecCount, expected_swap_x.data(), 1, dev_y, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SSWAP\n");
        return 1;
    }

    std::vector<float> host_copy_src(kVecCount);
    std::vector<float> host_copy_dst(kVecCount, 0.0f);
    for (int i = 0; i < kVecCount; ++i) {
        host_copy_src[i] = static_cast<float>((i * 13) % 43) * 0.125f;
    }
    if (cudaMemcpy(dev_x, host_copy_src.data(), host_copy_src.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_y, host_copy_dst.data(), host_copy_dst.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SCOPY failed\n");
        return 1;
    }
    if (cublasScopy(handle, kVecCount, dev_x, 1, dev_y, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasScopy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_copy_dst.data(),
                   dev_y,
                   host_copy_dst.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SCOPY failed\n");
        return 1;
    }
    for (int i = 0; i < kVecCount; ++i) {
        if (!nearly_equal(host_copy_dst[i], host_copy_src[i])) {
            std::fprintf(stderr,
                         "FAIL: SCOPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_copy_dst[i]),
                         static_cast<double>(host_copy_src[i]));
            return 1;
        }
    }
    if (cublasScopy(handle, kVecCount, host_copy_src.data(), 1, dev_y, 1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SCOPY\n");
        return 1;
    }

    constexpr int m = 2;
    constexpr int n = 3;
    constexpr int k = 4;
    constexpr int lda = m;
    constexpr int ldb = k;
    constexpr int ldc = m;
    std::vector<float> host_a(lda * k);
    std::vector<float> host_b(ldb * n);
    std::vector<float> host_c(ldc * n);
    std::vector<float> expected_c(ldc * n);

    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            host_a[row + col * lda] = 1.0f + static_cast<float>(row) * 0.5f +
                                      static_cast<float>(col) * 0.25f;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            host_b[row + col * ldb] =
                0.5f + static_cast<float>(row + 1) * static_cast<float>(col + 2) * 0.125f;
        }
    }
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            host_c[row + col * ldc] = static_cast<float>((row + 1) * (col + 2)) * 0.1f;
            expected_c[row + col * ldc] = host_c[row + col * ldc];
        }
    }

    const float alpha_gemm = 1.25f;
    const float beta_gemm = 0.75f;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                sum += host_a[row + p * lda] * host_b[p + col * ldb];
            }
            expected_c[row + col * ldc] = alpha_gemm * sum + beta_gemm * expected_c[row + col * ldc];
        }
    }

    float* dev_a = nullptr;
    float* dev_b = nullptr;
    float* dev_c = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_a), host_a.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_b), host_b.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_c), host_c.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_a, host_a.data(), host_a.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), host_b.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_c, host_c.data(), host_c.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha_gemm,
                    dev_a,
                    lda,
                    dev_b,
                    ldb,
                    &beta_gemm,
                    dev_c,
                    ldc) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm failed\n");
        return 1;
    }

    if (cudaMemcpy(host_c.data(), dev_c, host_c.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_c.size(); ++i) {
        if (!nearly_equal(host_c[i], expected_c[i])) {
            std::fprintf(stderr,
                         "FAIL: SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_c[i]),
                         static_cast<double>(expected_c[i]));
            return 1;
        }
    }

    constexpr int mt = 3;
    constexpr int nt = 2;
    constexpr int kt = 4;
    constexpr int lda_t = kt;  // A is stored as (kt x mt), then transposed in op(A)
    constexpr int ldb_t = kt;  // B is (kt x nt)
    constexpr int ldc_t = mt;  // C is (mt x nt)

    std::vector<float> host_at(lda_t * mt);
    std::vector<float> host_bt(ldb_t * nt);
    std::vector<float> host_ct(ldc_t * nt);
    std::vector<float> expected_ct(ldc_t * nt);

    for (int col = 0; col < mt; ++col) {
        for (int row = 0; row < kt; ++row) {
            host_at[row + col * lda_t] =
                0.2f + static_cast<float>(row + 1) * static_cast<float>(col + 1) * 0.1f;
        }
    }
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < kt; ++row) {
            host_bt[row + col * ldb_t] =
                0.3f + static_cast<float>(row + 2) * static_cast<float>(col + 1) * 0.07f;
        }
    }
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < mt; ++row) {
            host_ct[row + col * ldc_t] = static_cast<float>((row + col + 1) * 0.15f);
            expected_ct[row + col * ldc_t] = host_ct[row + col * ldc_t];
        }
    }

    const float alpha_t = 0.9f;
    const float beta_t = 0.4f;
    for (int col = 0; col < nt; ++col) {
        for (int row = 0; row < mt; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < kt; ++p) {
                const float a_value = host_at[p + row * lda_t];  // transposed access
                const float b_value = host_bt[p + col * ldb_t];
                sum += a_value * b_value;
            }
            expected_ct[row + col * ldc_t] = alpha_t * sum + beta_t * expected_ct[row + col * ldc_t];
        }
    }

    float* dev_at = nullptr;
    float* dev_bt = nullptr;
    float* dev_ct = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_at), host_at.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bt), host_bt.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ct), host_ct.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for transposed SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_at, host_at.data(), host_at.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bt, host_bt.data(), host_bt.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_ct, host_ct.data(), host_ct.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for transposed SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    mt,
                    nt,
                    kt,
                    &alpha_t,
                    dev_at,
                    lda_t,
                    dev_bt,
                    ldb_t,
                    &beta_t,
                    dev_ct,
                    ldc_t) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm transposed case failed\n");
        return 1;
    }
    if (cudaMemcpy(host_ct.data(), dev_ct, host_ct.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for transposed SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_ct.size(); ++i) {
        if (!nearly_equal(host_ct[i], expected_ct[i])) {
            std::fprintf(stderr,
                         "FAIL: transposed SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_ct[i]),
                         static_cast<double>(expected_ct[i]));
            return 1;
        }
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_T,
                    CUBLAS_OP_N,
                    mt,
                    nt,
                    kt,
                    &alpha_t,
                    dev_at,
                    mt - 1,
                    dev_bt,
                    ldb_t,
                    &beta_t,
                    dev_ct,
                    ldc_t) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid lda\n");
        return 1;
    }

    constexpr int mn = 2;
    constexpr int nn = 3;
    constexpr int kn = 4;
    constexpr int lda_n = mn;  // A is (mn x kn)
    constexpr int ldb_n = nn;  // B is stored as (nn x kn), then transposed in op(B)
    constexpr int ldc_n = mn;  // C is (mn x nn)

    std::vector<float> host_an(lda_n * kn);
    std::vector<float> host_bn(ldb_n * kn);
    std::vector<float> host_cn(ldc_n * nn);
    std::vector<float> expected_cn(ldc_n * nn);

    for (int col = 0; col < kn; ++col) {
        for (int row = 0; row < mn; ++row) {
            host_an[row + col * lda_n] =
                0.4f + static_cast<float>(row + 1) * static_cast<float>(col + 1) * 0.09f;
        }
    }
    for (int col = 0; col < kn; ++col) {
        for (int row = 0; row < nn; ++row) {
            host_bn[row + col * ldb_n] =
                0.1f + static_cast<float>(row + 2) * static_cast<float>(col + 1) * 0.06f;
        }
    }
    for (int col = 0; col < nn; ++col) {
        for (int row = 0; row < mn; ++row) {
            host_cn[row + col * ldc_n] = static_cast<float>((row + 1) * (col + 1)) * 0.2f;
            expected_cn[row + col * ldc_n] = host_cn[row + col * ldc_n];
        }
    }

    const float alpha_nt = 1.1f;
    const float beta_nt = 0.3f;
    for (int col = 0; col < nn; ++col) {
        for (int row = 0; row < mn; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < kn; ++p) {
                const float a_value = host_an[row + p * lda_n];
                const float b_value = host_bn[col + p * ldb_n];  // transposed access
                sum += a_value * b_value;
            }
            expected_cn[row + col * ldc_n] = alpha_nt * sum + beta_nt * expected_cn[row + col * ldc_n];
        }
    }

    float* dev_an = nullptr;
    float* dev_bn = nullptr;
    float* dev_cn = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_an), host_an.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bn), host_bn.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_cn), host_cn.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for N,T SGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_an, host_an.data(), host_an.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bn, host_bn.data(), host_bn.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_cn, host_cn.data(), host_cn.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for N,T SGEMM failed\n");
        return 1;
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    mn,
                    nn,
                    kn,
                    &alpha_nt,
                    dev_an,
                    lda_n,
                    dev_bn,
                    ldb_n,
                    &beta_nt,
                    dev_cn,
                    ldc_n) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemm N,T case failed\n");
        return 1;
    }
    if (cudaMemcpy(host_cn.data(), dev_cn, host_cn.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for N,T SGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_cn.size(); ++i) {
        if (!nearly_equal(host_cn[i], expected_cn[i])) {
            std::fprintf(stderr,
                         "FAIL: N,T SGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_cn[i]),
                         static_cast<double>(expected_cn[i]));
            return 1;
        }
    }

    if (cublasSgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    mn,
                    nn,
                    kn,
                    &alpha_nt,
                    dev_an,
                    lda_n,
                    dev_bn,
                    nn - 1,
                    &beta_nt,
                    dev_cn,
                    ldc_n) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid ldb\n");
        return 1;
    }

    constexpr int sb_m = 2;
    constexpr int sb_n = 2;
    constexpr int sb_k = 3;
    constexpr int sb_batch_count = 3;
    constexpr int sb_lda = sb_m;
    constexpr int sb_ldb = sb_k;
    constexpr int sb_ldc = sb_m;
    constexpr long long sb_stridea = static_cast<long long>(sb_lda) * sb_k;
    constexpr long long sb_strideb = static_cast<long long>(sb_ldb) * sb_n;
    constexpr long long sb_stridec = static_cast<long long>(sb_ldc) * sb_n;
    std::vector<float> host_sa(static_cast<std::size_t>(sb_stridea * sb_batch_count));
    std::vector<float> host_sb(static_cast<std::size_t>(sb_strideb * sb_batch_count));
    std::vector<float> host_sc(static_cast<std::size_t>(sb_stridec * sb_batch_count));
    std::vector<float> expected_sc = host_sc;

    for (int batch = 0; batch < sb_batch_count; ++batch) {
        for (int col = 0; col < sb_k; ++col) {
            for (int row = 0; row < sb_m; ++row) {
                host_sa[static_cast<std::size_t>(batch * sb_stridea) + row + col * sb_lda] =
                    0.2f + static_cast<float>(batch + 1) * 0.05f +
                    static_cast<float>((row + 1) * (col + 1)) * 0.1f;
            }
        }
        for (int col = 0; col < sb_n; ++col) {
            for (int row = 0; row < sb_k; ++row) {
                host_sb[static_cast<std::size_t>(batch * sb_strideb) + row + col * sb_ldb] =
                    0.1f + static_cast<float>(batch + 1) * 0.03f +
                    static_cast<float>((row + 2) * (col + 1)) * 0.08f;
            }
        }
        for (int col = 0; col < sb_n; ++col) {
            for (int row = 0; row < sb_m; ++row) {
                const std::size_t index =
                    static_cast<std::size_t>(batch * sb_stridec) + row + col * sb_ldc;
                host_sc[index] = static_cast<float>(batch + row + col + 1) * 0.2f;
                expected_sc[index] = host_sc[index];
            }
        }
    }

    const float sb_alpha = 0.9f;
    const float sb_beta = 0.4f;
    for (int batch = 0; batch < sb_batch_count; ++batch) {
        const std::size_t a_base = static_cast<std::size_t>(batch * sb_stridea);
        const std::size_t b_base = static_cast<std::size_t>(batch * sb_strideb);
        const std::size_t c_base = static_cast<std::size_t>(batch * sb_stridec);
        for (int col = 0; col < sb_n; ++col) {
            for (int row = 0; row < sb_m; ++row) {
                float sum = 0.0f;
                for (int p = 0; p < sb_k; ++p) {
                    sum += host_sa[a_base + row + p * sb_lda] * host_sb[b_base + p + col * sb_ldb];
                }
                expected_sc[c_base + row + col * sb_ldc] =
                    sb_alpha * sum + sb_beta * expected_sc[c_base + row + col * sb_ldc];
            }
        }
    }

    float* dev_sa = nullptr;
    float* dev_sb = nullptr;
    float* dev_sc = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_sa), host_sa.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_sb), host_sb.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_sc), host_sc.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGEMM strided-batched failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_sa, host_sa.data(), host_sa.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_sb, host_sb.data(), host_sb.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_sc, host_sc.data(), host_sc.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGEMM strided-batched failed\n");
        return 1;
    }
    if (cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  sb_m,
                                  sb_n,
                                  sb_k,
                                  &sb_alpha,
                                  dev_sa,
                                  sb_lda,
                                  sb_stridea,
                                  dev_sb,
                                  sb_ldb,
                                  sb_strideb,
                                  &sb_beta,
                                  dev_sc,
                                  sb_ldc,
                                  sb_stridec,
                                  sb_batch_count) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemmStridedBatched failed\n");
        return 1;
    }
    if (cudaMemcpy(host_sc.data(), dev_sc, host_sc.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr,
                     "FAIL: cudaMemcpy device->host for SGEMM strided-batched failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_sc.size(); ++i) {
        if (!nearly_equal(host_sc[i], expected_sc[i])) {
            std::fprintf(stderr,
                         "FAIL: SGEMM strided-batched mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_sc[i]),
                         static_cast<double>(expected_sc[i]));
            return 1;
        }
    }
    if (cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  sb_m,
                                  sb_n,
                                  sb_k,
                                  &sb_alpha,
                                  host_sa.data(),
                                  sb_lda,
                                  sb_stridea,
                                  dev_sb,
                                  sb_ldb,
                                  sb_strideb,
                                  &sb_beta,
                                  dev_sc,
                                  sb_ldc,
                                  sb_stridec,
                                  sb_batch_count) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host A in strided-batched SGEMM\n");
        return 1;
    }
    if (cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  sb_m,
                                  sb_n,
                                  sb_k,
                                  &sb_alpha,
                                  dev_sa,
                                  sb_lda,
                                  -1,
                                  dev_sb,
                                  sb_ldb,
                                  sb_strideb,
                                  &sb_beta,
                                  dev_sc,
                                  sb_ldc,
                                  sb_stridec,
                                  sb_batch_count) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for negative stride\n");
        return 1;
    }
    if (cublasSgemmStridedBatched(handle,
                                  CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  sb_m,
                                  sb_n,
                                  sb_k,
                                  &sb_alpha,
                                  dev_sa,
                                  sb_lda,
                                  sb_stridea,
                                  dev_sb,
                                  sb_ldb,
                                  sb_strideb,
                                  &sb_beta,
                                  dev_sc,
                                  sb_ldc,
                                  sb_stridec,
                                  0) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemmStridedBatched batch_count==0 should succeed\n");
        return 1;
    }
    if (cudaFree(dev_sa) != cudaSuccess || cudaFree(dev_sb) != cudaSuccess ||
        cudaFree(dev_sc) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for SGEMM strided-batched buffers failed\n");
        return 1;
    }

    constexpr int md = 2;
    constexpr int nd = 2;
    constexpr int kd = 3;
    constexpr int lda_d = md;
    constexpr int ldb_d = kd;
    constexpr int ldc_d = md;
    std::vector<double> host_ad(lda_d * kd);
    std::vector<double> host_bd(ldb_d * nd);
    std::vector<double> host_cd(ldc_d * nd);
    std::vector<double> expected_cd(ldc_d * nd);

    for (int col = 0; col < kd; ++col) {
        for (int row = 0; row < md; ++row) {
            host_ad[row + col * lda_d] =
                0.25 + static_cast<double>(row + 1) * static_cast<double>(col + 1) * 0.05;
        }
    }
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < kd; ++row) {
            host_bd[row + col * ldb_d] =
                0.5 + static_cast<double>(row + 1) * static_cast<double>(col + 2) * 0.03;
        }
    }
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < md; ++row) {
            host_cd[row + col * ldc_d] = static_cast<double>((row + 1) * (col + 1)) * 0.2;
            expected_cd[row + col * ldc_d] = host_cd[row + col * ldc_d];
        }
    }

    const double alpha_d = 1.2;
    const double beta_d = 0.4;
    for (int col = 0; col < nd; ++col) {
        for (int row = 0; row < md; ++row) {
            double sum = 0.0;
            for (int p = 0; p < kd; ++p) {
                sum += host_ad[row + p * lda_d] * host_bd[p + col * ldb_d];
            }
            expected_cd[row + col * ldc_d] = alpha_d * sum + beta_d * expected_cd[row + col * ldc_d];
        }
    }

    double* dev_ad = nullptr;
    double* dev_bd = nullptr;
    double* dev_cd = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ad), host_ad.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_bd), host_bd.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_cd), host_cd.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DGEMM failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ad, host_ad.data(), host_ad.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_bd, host_bd.data(), host_bd.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_cd, host_cd.data(), host_cd.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DGEMM failed\n");
        return 1;
    }

    if (cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    md,
                    nd,
                    kd,
                    &alpha_d,
                    dev_ad,
                    lda_d,
                    dev_bd,
                    ldb_d,
                    &beta_d,
                    dev_cd,
                    ldc_d) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDgemm failed\n");
        return 1;
    }
    if (cudaMemcpy(host_cd.data(), dev_cd, host_cd.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DGEMM failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_cd.size(); ++i) {
        if (std::fabs(host_cd[i] - expected_cd[i]) > 1e-9) {
            std::fprintf(stderr,
                         "FAIL: DGEMM mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         host_cd[i],
                         expected_cd[i]);
            return 1;
        }
    }
    if (cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    md,
                    nd,
                    kd,
                    &alpha_d,
                    host_ad.data(),
                    lda_d,
                    dev_bd,
                    ldb_d,
                    &beta_d,
                    dev_cd,
                    ldc_d) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host A in DGEMM\n");
        return 1;
    }

    constexpr int gemv_m = 3;
    constexpr int gemv_n = 4;
    constexpr int gemv_lda = gemv_m;
    std::vector<float> host_gemv_a(gemv_lda * gemv_n);
    std::vector<float> host_gemv_x_n(gemv_n);
    std::vector<float> host_gemv_y_n(gemv_m);
    std::vector<float> expected_gemv_y_n(gemv_m);
    for (int col = 0; col < gemv_n; ++col) {
        for (int row = 0; row < gemv_m; ++row) {
            host_gemv_a[row + col * gemv_lda] =
                0.3f + static_cast<float>(row + 1) * static_cast<float>(col + 2) * 0.08f;
        }
    }
    for (int i = 0; i < gemv_n; ++i) {
        host_gemv_x_n[i] = 0.25f + static_cast<float>(i + 1) * 0.2f;
    }
    for (int i = 0; i < gemv_m; ++i) {
        host_gemv_y_n[i] = -0.1f + static_cast<float>(i + 1) * 0.05f;
        expected_gemv_y_n[i] = host_gemv_y_n[i];
    }
    const float gemv_alpha = 1.15f;
    const float gemv_beta = 0.4f;
    for (int row = 0; row < gemv_m; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < gemv_n; ++col) {
            sum += host_gemv_a[row + col * gemv_lda] * host_gemv_x_n[col];
        }
        expected_gemv_y_n[row] = gemv_alpha * sum + gemv_beta * expected_gemv_y_n[row];
    }

    float* dev_gemv_a = nullptr;
    float* dev_gemv_x = nullptr;
    float* dev_gemv_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_gemv_a), host_gemv_a.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_gemv_x), host_gemv_x_n.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_gemv_y), host_gemv_y_n.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGEMV failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_gemv_a,
                   host_gemv_a.data(),
                   host_gemv_a.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_gemv_x,
                   host_gemv_x_n.data(),
                   host_gemv_x_n.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_gemv_y,
                   host_gemv_y_n.data(),
                   host_gemv_y_n.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGEMV failed\n");
        return 1;
    }
    if (cublasSgemv(handle,
                    CUBLAS_OP_N,
                    gemv_m,
                    gemv_n,
                    &gemv_alpha,
                    dev_gemv_a,
                    gemv_lda,
                    dev_gemv_x,
                    1,
                    &gemv_beta,
                    dev_gemv_y,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemv (N) failed\n");
        return 1;
    }
    if (cudaMemcpy(host_gemv_y_n.data(),
                   dev_gemv_y,
                   host_gemv_y_n.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SGEMV (N) failed\n");
        return 1;
    }
    for (int i = 0; i < gemv_m; ++i) {
        if (!nearly_equal(host_gemv_y_n[i], expected_gemv_y_n[i])) {
            std::fprintf(stderr,
                         "FAIL: SGEMV (N) mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_gemv_y_n[i]),
                         static_cast<double>(expected_gemv_y_n[i]));
            return 1;
        }
    }

    std::vector<float> host_gemv_x_t(gemv_m);
    std::vector<float> host_gemv_y_t(gemv_n);
    std::vector<float> expected_gemv_y_t(gemv_n);
    for (int i = 0; i < gemv_m; ++i) {
        host_gemv_x_t[i] = 0.15f + static_cast<float>(i + 1) * 0.17f;
    }
    for (int i = 0; i < gemv_n; ++i) {
        host_gemv_y_t[i] = 0.05f * static_cast<float>(i + 1);
        expected_gemv_y_t[i] = host_gemv_y_t[i];
    }
    for (int col = 0; col < gemv_n; ++col) {
        float sum = 0.0f;
        for (int row = 0; row < gemv_m; ++row) {
            sum += host_gemv_a[row + col * gemv_lda] * host_gemv_x_t[row];
        }
        expected_gemv_y_t[col] = gemv_alpha * sum + gemv_beta * expected_gemv_y_t[col];
    }
    float* dev_gemv_x_t = nullptr;
    float* dev_gemv_y_t = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_gemv_x_t), host_gemv_x_t.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_gemv_y_t), host_gemv_y_t.size() * sizeof(float)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGEMV transpose failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_gemv_x_t,
                   host_gemv_x_t.data(),
                   host_gemv_x_t.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_gemv_y_t,
                   host_gemv_y_t.data(),
                   host_gemv_y_t.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGEMV transpose failed\n");
        return 1;
    }
    if (cublasSgemv(handle,
                    CUBLAS_OP_T,
                    gemv_m,
                    gemv_n,
                    &gemv_alpha,
                    dev_gemv_a,
                    gemv_lda,
                    dev_gemv_x_t,
                    1,
                    &gemv_beta,
                    dev_gemv_y_t,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSgemv (T) failed\n");
        return 1;
    }
    if (cudaMemcpy(host_gemv_y_t.data(),
                   dev_gemv_y_t,
                   host_gemv_y_t.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SGEMV (T) failed\n");
        return 1;
    }
    for (int i = 0; i < gemv_n; ++i) {
        if (!nearly_equal(host_gemv_y_t[i], expected_gemv_y_t[i])) {
            std::fprintf(stderr,
                         "FAIL: SGEMV (T) mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_gemv_y_t[i]),
                         static_cast<double>(expected_gemv_y_t[i]));
            return 1;
        }
    }
    if (cublasSgemv(handle,
                    CUBLAS_OP_N,
                    gemv_m,
                    gemv_n,
                    &gemv_alpha,
                    host_gemv_a.data(),
                    gemv_lda,
                    dev_gemv_x,
                    1,
                    &gemv_beta,
                    dev_gemv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host A in SGEMV\n");
        return 1;
    }
    if (cublasSgemv(handle,
                    CUBLAS_OP_N,
                    gemv_m,
                    gemv_n,
                    &gemv_alpha,
                    dev_gemv_a,
                    gemv_m - 1,
                    dev_gemv_x,
                    1,
                    &gemv_beta,
                    dev_gemv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid lda in SGEMV\n");
        return 1;
    }

    if (cudaFree(dev_gemv_a) != cudaSuccess || cudaFree(dev_gemv_x) != cudaSuccess ||
        cudaFree(dev_gemv_y) != cudaSuccess || cudaFree(dev_gemv_x_t) != cudaSuccess ||
        cudaFree(dev_gemv_y_t) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for SGEMV buffers failed\n");
        return 1;
    }

    constexpr int dgemv_m = 2;
    constexpr int dgemv_n = 3;
    constexpr int dgemv_lda = dgemv_m;
    std::vector<double> host_dgemv_a(dgemv_lda * dgemv_n);
    std::vector<double> host_dgemv_x_n(dgemv_n);
    std::vector<double> host_dgemv_y_n(dgemv_m);
    std::vector<double> expected_dgemv_y_n(dgemv_m);
    for (int col = 0; col < dgemv_n; ++col) {
        for (int row = 0; row < dgemv_m; ++row) {
            host_dgemv_a[row + col * dgemv_lda] =
                0.2 + static_cast<double>(row + 1) * static_cast<double>(col + 1) * 0.07;
        }
    }
    for (int i = 0; i < dgemv_n; ++i) {
        host_dgemv_x_n[i] = 0.1 + static_cast<double>(i + 1) * 0.25;
    }
    for (int i = 0; i < dgemv_m; ++i) {
        host_dgemv_y_n[i] = -0.2 + static_cast<double>(i + 1) * 0.11;
        expected_dgemv_y_n[i] = host_dgemv_y_n[i];
    }
    const double dgemv_alpha = 0.95;
    const double dgemv_beta = -0.3;
    for (int row = 0; row < dgemv_m; ++row) {
        double sum = 0.0;
        for (int col = 0; col < dgemv_n; ++col) {
            sum += host_dgemv_a[row + col * dgemv_lda] * host_dgemv_x_n[col];
        }
        expected_dgemv_y_n[row] = dgemv_alpha * sum + dgemv_beta * expected_dgemv_y_n[row];
    }
    double* dev_dgemv_a = nullptr;
    double* dev_dgemv_x = nullptr;
    double* dev_dgemv_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dgemv_a), host_dgemv_a.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dgemv_x), host_dgemv_x_n.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dgemv_y), host_dgemv_y_n.size() * sizeof(double)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DGEMV failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dgemv_a,
                   host_dgemv_a.data(),
                   host_dgemv_a.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dgemv_x,
                   host_dgemv_x_n.data(),
                   host_dgemv_x_n.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dgemv_y,
                   host_dgemv_y_n.data(),
                   host_dgemv_y_n.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DGEMV failed\n");
        return 1;
    }
    if (cublasDgemv(handle,
                    CUBLAS_OP_N,
                    dgemv_m,
                    dgemv_n,
                    &dgemv_alpha,
                    dev_dgemv_a,
                    dgemv_lda,
                    dev_dgemv_x,
                    1,
                    &dgemv_beta,
                    dev_dgemv_y,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDgemv (N) failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dgemv_y_n.data(),
                   dev_dgemv_y,
                   host_dgemv_y_n.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DGEMV (N) failed\n");
        return 1;
    }
    for (int i = 0; i < dgemv_m; ++i) {
        if (std::fabs(host_dgemv_y_n[i] - expected_dgemv_y_n[i]) > 1e-10) {
            std::fprintf(stderr,
                         "FAIL: DGEMV (N) mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dgemv_y_n[i],
                         expected_dgemv_y_n[i]);
            return 1;
        }
    }
    if (cublasDgemv(handle,
                    CUBLAS_OP_N,
                    dgemv_m,
                    dgemv_n,
                    &dgemv_alpha,
                    dev_dgemv_a,
                    dgemv_lda,
                    host_dgemv_x_n.data(),
                    1,
                    &dgemv_beta,
                    dev_dgemv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DGEMV\n");
        return 1;
    }
    if (cudaFree(dev_dgemv_a) != cudaSuccess || cudaFree(dev_dgemv_x) != cudaSuccess ||
        cudaFree(dev_dgemv_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for DGEMV buffers failed\n");
        return 1;
    }

    constexpr int sger_m = 3;
    constexpr int sger_n = 2;
    constexpr int sger_lda = sger_m;
    std::vector<float> host_sger_a(sger_lda * sger_n);
    std::vector<float> host_sger_x(sger_m);
    std::vector<float> host_sger_y(sger_n);
    std::vector<float> expected_sger_a(sger_lda * sger_n);
    for (int row = 0; row < sger_m; ++row) {
        host_sger_x[row] = 0.2f + static_cast<float>(row + 1) * 0.35f;
    }
    for (int col = 0; col < sger_n; ++col) {
        host_sger_y[col] = -0.1f + static_cast<float>(col + 1) * 0.45f;
        for (int row = 0; row < sger_m; ++row) {
            host_sger_a[row + col * sger_lda] =
                0.05f + static_cast<float>(row + 1) * static_cast<float>(col + 2) * 0.12f;
            expected_sger_a[row + col * sger_lda] = host_sger_a[row + col * sger_lda];
        }
    }
    const float sger_alpha = 0.75f;
    for (int col = 0; col < sger_n; ++col) {
        for (int row = 0; row < sger_m; ++row) {
            expected_sger_a[row + col * sger_lda] += sger_alpha * host_sger_x[row] * host_sger_y[col];
        }
    }

    float* dev_sger_a = nullptr;
    float* dev_sger_x = nullptr;
    float* dev_sger_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_sger_a), host_sger_a.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_sger_x), host_sger_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_sger_y), host_sger_y.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SGER failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_sger_a,
                   host_sger_a.data(),
                   host_sger_a.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_sger_x,
                   host_sger_x.data(),
                   host_sger_x.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_sger_y,
                   host_sger_y.data(),
                   host_sger_y.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SGER failed\n");
        return 1;
    }
    if (cublasSger(handle,
                   sger_m,
                   sger_n,
                   &sger_alpha,
                   dev_sger_x,
                   1,
                   dev_sger_y,
                   1,
                   dev_sger_a,
                   sger_lda) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSger failed\n");
        return 1;
    }
    if (cudaMemcpy(host_sger_a.data(),
                   dev_sger_a,
                   host_sger_a.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SGER failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_sger_a.size(); ++i) {
        if (!nearly_equal(host_sger_a[i], expected_sger_a[i])) {
            std::fprintf(stderr,
                         "FAIL: SGER mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_sger_a[i]),
                         static_cast<double>(expected_sger_a[i]));
            return 1;
        }
    }
    if (cublasSger(handle,
                   sger_m,
                   sger_n,
                   &sger_alpha,
                   host_sger_x.data(),
                   1,
                   dev_sger_y,
                   1,
                   dev_sger_a,
                   sger_lda) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SGER\n");
        return 1;
    }
    if (cublasSger(handle,
                   sger_m,
                   sger_n,
                   &sger_alpha,
                   dev_sger_x,
                   1,
                   dev_sger_y,
                   1,
                   dev_sger_a,
                   sger_m - 1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid lda in SGER\n");
        return 1;
    }
    if (cudaFree(dev_sger_a) != cudaSuccess || cudaFree(dev_sger_x) != cudaSuccess ||
        cudaFree(dev_sger_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for SGER buffers failed\n");
        return 1;
    }

    constexpr int dger_m = 2;
    constexpr int dger_n = 3;
    constexpr int dger_lda = dger_m;
    std::vector<double> host_dger_a(dger_lda * dger_n);
    std::vector<double> host_dger_x(dger_m);
    std::vector<double> host_dger_y(dger_n);
    std::vector<double> expected_dger_a(dger_lda * dger_n);
    for (int row = 0; row < dger_m; ++row) {
        host_dger_x[row] = 0.1 + static_cast<double>(row + 1) * 0.4;
    }
    for (int col = 0; col < dger_n; ++col) {
        host_dger_y[col] = 0.2 + static_cast<double>(col + 1) * 0.25;
        for (int row = 0; row < dger_m; ++row) {
            host_dger_a[row + col * dger_lda] =
                -0.05 + static_cast<double>(row + 1) * static_cast<double>(col + 1) * 0.09;
            expected_dger_a[row + col * dger_lda] = host_dger_a[row + col * dger_lda];
        }
    }
    const double dger_alpha = -1.1;
    for (int col = 0; col < dger_n; ++col) {
        for (int row = 0; row < dger_m; ++row) {
            expected_dger_a[row + col * dger_lda] += dger_alpha * host_dger_x[row] * host_dger_y[col];
        }
    }

    double* dev_dger_a = nullptr;
    double* dev_dger_x = nullptr;
    double* dev_dger_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dger_a), host_dger_a.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dger_x), host_dger_x.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dger_y), host_dger_y.size() * sizeof(double)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DGER failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dger_a,
                   host_dger_a.data(),
                   host_dger_a.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dger_x,
                   host_dger_x.data(),
                   host_dger_x.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dger_y,
                   host_dger_y.data(),
                   host_dger_y.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DGER failed\n");
        return 1;
    }
    if (cublasDger(handle,
                   dger_m,
                   dger_n,
                   &dger_alpha,
                   dev_dger_x,
                   1,
                   dev_dger_y,
                   1,
                   dev_dger_a,
                   dger_lda) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDger failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dger_a.data(),
                   dev_dger_a,
                   host_dger_a.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DGER failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < host_dger_a.size(); ++i) {
        if (std::fabs(host_dger_a[i] - expected_dger_a[i]) > 1e-12) {
            std::fprintf(stderr,
                         "FAIL: DGER mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         host_dger_a[i],
                         expected_dger_a[i]);
            return 1;
        }
    }
    if (cublasDger(handle,
                   dger_m,
                   dger_n,
                   &dger_alpha,
                   dev_dger_x,
                   1,
                   host_dger_y.data(),
                   1,
                   dev_dger_a,
                   dger_lda) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host Y in DGER\n");
        return 1;
    }
    if (cudaFree(dev_dger_a) != cudaSuccess || cudaFree(dev_dger_x) != cudaSuccess ||
        cudaFree(dev_dger_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for DGER buffers failed\n");
        return 1;
    }

    constexpr int ssymv_n = 3;
    constexpr int ssymv_lda = ssymv_n;
    const float ssymv_sym[ssymv_n][ssymv_n] = {
        {2.0f, -1.0f, 0.5f},
        {-1.0f, 3.0f, 1.25f},
        {0.5f, 1.25f, 4.0f},
    };
    std::vector<float> host_ssymv_a_upper(ssymv_lda * ssymv_n, 0.0f);
    std::vector<float> host_ssymv_a_lower(ssymv_lda * ssymv_n, 0.0f);
    for (int col = 0; col < ssymv_n; ++col) {
        for (int row = 0; row < ssymv_n; ++row) {
            host_ssymv_a_upper[row + col * ssymv_lda] =
                (row <= col) ? ssymv_sym[row][col] : -77.0f;
            host_ssymv_a_lower[row + col * ssymv_lda] =
                (row >= col) ? ssymv_sym[row][col] : -88.0f;
        }
    }
    std::vector<float> host_ssymv_x = {1.0f, -2.0f, 0.5f};
    std::vector<float> host_ssymv_y_init = {0.2f, -0.3f, 0.4f};
    std::vector<float> expected_ssymv_y = host_ssymv_y_init;
    const float ssymv_alpha = 1.1f;
    const float ssymv_beta = 0.6f;
    for (int row = 0; row < ssymv_n; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < ssymv_n; ++col) {
            sum += ssymv_sym[row][col] * host_ssymv_x[col];
        }
        expected_ssymv_y[row] = ssymv_alpha * sum + ssymv_beta * expected_ssymv_y[row];
    }

    float* dev_ssymv_a = nullptr;
    float* dev_ssymv_x = nullptr;
    float* dev_ssymv_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ssymv_a), host_ssymv_a_upper.size() * sizeof(float)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ssymv_x), host_ssymv_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ssymv_y), host_ssymv_y_init.size() * sizeof(float)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SSYMV failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ssymv_x,
                   host_ssymv_x.data(),
                   host_ssymv_x.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SSYMV x failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ssymv_a,
                   host_ssymv_a_upper.data(),
                   host_ssymv_a_upper.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_ssymv_y,
                   host_ssymv_y_init.data(),
                   host_ssymv_y_init.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SSYMV upper failed\n");
        return 1;
    }
    if (cublasSsymv(handle,
                    CUBLAS_FILL_MODE_UPPER,
                    ssymv_n,
                    &ssymv_alpha,
                    dev_ssymv_a,
                    ssymv_lda,
                    dev_ssymv_x,
                    1,
                    &ssymv_beta,
                    dev_ssymv_y,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSsymv (upper) failed\n");
        return 1;
    }
    std::vector<float> host_ssymv_y = host_ssymv_y_init;
    if (cudaMemcpy(host_ssymv_y.data(),
                   dev_ssymv_y,
                   host_ssymv_y.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SSYMV upper failed\n");
        return 1;
    }
    for (int i = 0; i < ssymv_n; ++i) {
        if (!nearly_equal(host_ssymv_y[i], expected_ssymv_y[i])) {
            std::fprintf(stderr,
                         "FAIL: SSYMV upper mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_ssymv_y[i]),
                         static_cast<double>(expected_ssymv_y[i]));
            return 1;
        }
    }
    if (cudaMemcpy(dev_ssymv_a,
                   host_ssymv_a_lower.data(),
                   host_ssymv_a_lower.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_ssymv_y,
                   host_ssymv_y_init.data(),
                   host_ssymv_y_init.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SSYMV lower failed\n");
        return 1;
    }
    if (cublasSsymv(handle,
                    CUBLAS_FILL_MODE_LOWER,
                    ssymv_n,
                    &ssymv_alpha,
                    dev_ssymv_a,
                    ssymv_lda,
                    dev_ssymv_x,
                    1,
                    &ssymv_beta,
                    dev_ssymv_y,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSsymv (lower) failed\n");
        return 1;
    }
    if (cudaMemcpy(host_ssymv_y.data(),
                   dev_ssymv_y,
                   host_ssymv_y.size() * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for SSYMV lower failed\n");
        return 1;
    }
    for (int i = 0; i < ssymv_n; ++i) {
        if (!nearly_equal(host_ssymv_y[i], expected_ssymv_y[i])) {
            std::fprintf(stderr,
                         "FAIL: SSYMV lower mismatch at %d (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_ssymv_y[i]),
                         static_cast<double>(expected_ssymv_y[i]));
            return 1;
        }
    }
    if (cublasSsymv(handle,
                    static_cast<cublasFillMode_t>(99),
                    ssymv_n,
                    &ssymv_alpha,
                    dev_ssymv_a,
                    ssymv_lda,
                    dev_ssymv_x,
                    1,
                    &ssymv_beta,
                    dev_ssymv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid uplo in SSYMV\n");
        return 1;
    }
    if (cublasSsymv(handle,
                    CUBLAS_FILL_MODE_UPPER,
                    ssymv_n,
                    &ssymv_alpha,
                    dev_ssymv_a,
                    ssymv_n - 1,
                    dev_ssymv_x,
                    1,
                    &ssymv_beta,
                    dev_ssymv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for invalid lda in SSYMV\n");
        return 1;
    }
    if (cublasSsymv(handle,
                    CUBLAS_FILL_MODE_UPPER,
                    ssymv_n,
                    &ssymv_alpha,
                    dev_ssymv_a,
                    ssymv_lda,
                    host_ssymv_x.data(),
                    1,
                    &ssymv_beta,
                    dev_ssymv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SSYMV\n");
        return 1;
    }
    if (cudaFree(dev_ssymv_a) != cudaSuccess || cudaFree(dev_ssymv_x) != cudaSuccess ||
        cudaFree(dev_ssymv_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for SSYMV buffers failed\n");
        return 1;
    }

    constexpr int dsymv_n = 2;
    constexpr int dsymv_lda = dsymv_n;
    const double dsymv_sym[dsymv_n][dsymv_n] = {
        {1.5, -0.75},
        {-0.75, 2.25},
    };
    std::vector<double> host_dsymv_a_upper(dsymv_lda * dsymv_n, 0.0);
    for (int col = 0; col < dsymv_n; ++col) {
        for (int row = 0; row < dsymv_n; ++row) {
            host_dsymv_a_upper[row + col * dsymv_lda] =
                (row <= col) ? dsymv_sym[row][col] : -42.0;
        }
    }
    std::vector<double> host_dsymv_x = {0.4, -1.2};
    std::vector<double> host_dsymv_y = {0.1, 0.2};
    std::vector<double> expected_dsymv_y = host_dsymv_y;
    const double dsymv_alpha = -0.8;
    const double dsymv_beta = 1.3;
    for (int row = 0; row < dsymv_n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < dsymv_n; ++col) {
            sum += dsymv_sym[row][col] * host_dsymv_x[col];
        }
        expected_dsymv_y[row] = dsymv_alpha * sum + dsymv_beta * expected_dsymv_y[row];
    }

    double* dev_dsymv_a = nullptr;
    double* dev_dsymv_x = nullptr;
    double* dev_dsymv_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dsymv_a), host_dsymv_a_upper.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dsymv_x), host_dsymv_x.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dsymv_y), host_dsymv_y.size() * sizeof(double)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DSYMV failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dsymv_a,
                   host_dsymv_a_upper.data(),
                   host_dsymv_a_upper.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dsymv_x,
                   host_dsymv_x.data(),
                   host_dsymv_x.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_dsymv_y,
                   host_dsymv_y.data(),
                   host_dsymv_y.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DSYMV failed\n");
        return 1;
    }
    if (cublasDsymv(handle,
                    CUBLAS_FILL_MODE_UPPER,
                    dsymv_n,
                    &dsymv_alpha,
                    dev_dsymv_a,
                    dsymv_lda,
                    dev_dsymv_x,
                    1,
                    &dsymv_beta,
                    dev_dsymv_y,
                    1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDsymv failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dsymv_y.data(),
                   dev_dsymv_y,
                   host_dsymv_y.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DSYMV failed\n");
        return 1;
    }
    for (int i = 0; i < dsymv_n; ++i) {
        if (std::fabs(host_dsymv_y[i] - expected_dsymv_y[i]) > 1e-12) {
            std::fprintf(stderr,
                         "FAIL: DSYMV mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dsymv_y[i],
                         expected_dsymv_y[i]);
            return 1;
        }
    }
    if (cublasDsymv(handle,
                    CUBLAS_FILL_MODE_UPPER,
                    dsymv_n,
                    &dsymv_alpha,
                    host_dsymv_a_upper.data(),
                    dsymv_lda,
                    dev_dsymv_x,
                    1,
                    &dsymv_beta,
                    dev_dsymv_y,
                    1) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host A in DSYMV\n");
        return 1;
    }
    if (cudaFree(dev_dsymv_a) != cudaSuccess || cudaFree(dev_dsymv_x) != cudaSuccess ||
        cudaFree(dev_dsymv_y) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for DSYMV buffers failed\n");
        return 1;
    }

    constexpr int kDoubleCount = 1536;
    std::vector<double> host_dx(kDoubleCount);
    std::vector<double> host_dy(kDoubleCount);
    for (int i = 0; i < kDoubleCount; ++i) {
        host_dx[i] = 0.125 + static_cast<double>((i * 5) % 23) * 0.25;
        host_dy[i] = 0.5 + static_cast<double>((i * 3) % 17) * 0.1;
    }
    std::vector<double> expected_daxpy = host_dy;
    const double alpha_daxpy = -1.3;
    for (int i = 0; i < kDoubleCount; ++i) {
        expected_daxpy[i] = alpha_daxpy * host_dx[i] + expected_daxpy[i];
    }

    double* dev_dx = nullptr;
    double* dev_dy = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dx), host_dx.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dy), host_dy.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DAXPY failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dx, host_dx.data(), host_dx.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_dy, host_dy.data(), host_dy.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DAXPY failed\n");
        return 1;
    }
    if (cublasDaxpy(handle, kDoubleCount, &alpha_daxpy, dev_dx, 1, dev_dy, 1) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDaxpy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dy.data(), dev_dy, host_dy.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DAXPY failed\n");
        return 1;
    }
    for (int i = 0; i < kDoubleCount; ++i) {
        if (std::fabs(host_dy[i] - expected_daxpy[i]) > 1e-10) {
            std::fprintf(stderr,
                         "FAIL: DAXPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dy[i],
                         expected_daxpy[i]);
            return 1;
        }
    }
    if (cublasDaxpy(handle, kDoubleCount, &alpha_daxpy, host_dx.data(), 1, dev_dy, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DAXPY\n");
        return 1;
    }

    const double alpha_dscal = 0.25;
    std::vector<double> expected_dscal = host_dx;
    for (double& value : expected_dscal) {
        value *= alpha_dscal;
    }
    if (cublasDscal(handle, kDoubleCount, &alpha_dscal, dev_dx, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDscal failed\n");
        return 1;
    }
    if (cudaMemcpy(host_dx.data(), dev_dx, host_dx.size() * sizeof(double), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DSCAL failed\n");
        return 1;
    }
    for (int i = 0; i < kDoubleCount; ++i) {
        if (std::fabs(host_dx[i] - expected_dscal[i]) > 1e-10) {
            std::fprintf(stderr,
                         "FAIL: DSCAL mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_dx[i],
                         expected_dscal[i]);
            return 1;
        }
    }
    if (cublasDscal(handle, kDoubleCount, &alpha_dscal, expected_dscal.data(), 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DSCAL\n");
        return 1;
    }

    std::vector<double> host_copy_dsrc(kDoubleCount);
    std::vector<double> host_copy_ddst(kDoubleCount, 0.0);
    for (int i = 0; i < kDoubleCount; ++i) {
        host_copy_dsrc[i] = 0.25 + static_cast<double>((i * 7) % 59) * 0.05;
    }
    if (cudaMemcpy(dev_dx, host_copy_dsrc.data(), host_copy_dsrc.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_dy, host_copy_ddst.data(), host_copy_ddst.size() * sizeof(double), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DCOPY failed\n");
        return 1;
    }
    if (cublasDcopy(handle, kDoubleCount, dev_dx, 1, dev_dy, 1) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDcopy failed\n");
        return 1;
    }
    if (cudaMemcpy(host_copy_ddst.data(),
                   dev_dy,
                   host_copy_ddst.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DCOPY failed\n");
        return 1;
    }
    for (int i = 0; i < kDoubleCount; ++i) {
        if (std::fabs(host_copy_ddst[i] - host_copy_dsrc[i]) > 1e-12) {
            std::fprintf(stderr,
                         "FAIL: DCOPY mismatch at %d (got=%f expected=%f)\n",
                         i,
                         host_copy_ddst[i],
                         host_copy_dsrc[i]);
            return 1;
        }
    }
    if (cublasDcopy(handle, kDoubleCount, host_copy_dsrc.data(), 1, dev_dy, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DCOPY\n");
        return 1;
    }

    constexpr int kSwapDoubleCount = 512;
    std::vector<double> host_swap_dx(kSwapDoubleCount);
    std::vector<double> host_swap_dy(kSwapDoubleCount);
    for (int i = 0; i < kSwapDoubleCount; ++i) {
        host_swap_dx[i] = 0.75 + static_cast<double>((i * 3) % 41) * 0.05;
        host_swap_dy[i] = -0.25 + static_cast<double>((i * 5) % 37) * 0.04;
    }
    std::vector<double> expected_swap_dx = host_swap_dy;
    std::vector<double> expected_swap_dy = host_swap_dx;
    double* dev_swap_dx = nullptr;
    double* dev_swap_dy = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_swap_dx), host_swap_dx.size() * sizeof(double)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_swap_dy), host_swap_dy.size() * sizeof(double)) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DSWAP failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_swap_dx,
                   host_swap_dx.data(),
                   host_swap_dx.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_swap_dy,
                   host_swap_dy.data(),
                   host_swap_dy.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DSWAP failed\n");
        return 1;
    }
    if (cublasDswap(handle, kSwapDoubleCount, dev_swap_dx, 1, dev_swap_dy, 1) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDswap failed\n");
        return 1;
    }
    if (cudaMemcpy(host_swap_dx.data(),
                   dev_swap_dx,
                   host_swap_dx.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_swap_dy.data(),
                   dev_swap_dy,
                   host_swap_dy.size() * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for DSWAP failed\n");
        return 1;
    }
    for (int i = 0; i < kSwapDoubleCount; ++i) {
        if (std::fabs(host_swap_dx[i] - expected_swap_dx[i]) > 1e-12 ||
            std::fabs(host_swap_dy[i] - expected_swap_dy[i]) > 1e-12) {
            std::fprintf(stderr, "FAIL: DSWAP mismatch at %d\n", i);
            return 1;
        }
    }
    if (cublasDswap(handle, kSwapDoubleCount, host_swap_dx.data(), 1, dev_swap_dy, 1) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DSWAP\n");
        return 1;
    }

    constexpr int kDotCount = 1024;
    std::vector<float> host_dot_x(kDotCount);
    std::vector<float> host_dot_y(kDotCount);
    for (int i = 0; i < kDotCount; ++i) {
        host_dot_x[i] = static_cast<float>((i * 2) % 31) * 0.2f;
        host_dot_y[i] = static_cast<float>((i * 5) % 19) * 0.3f;
    }
    float* dev_dot_x = nullptr;
    float* dev_dot_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_dot_x), host_dot_x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_dot_y), host_dot_y.size() * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for SDOT failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_dot_x, host_dot_x.data(), host_dot_x.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess ||
        cudaMemcpy(dev_dot_y, host_dot_y.data(), host_dot_y.size() * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for SDOT failed\n");
        return 1;
    }

    float expected_sdot = 0.0f;
    for (int i = 0; i < kDotCount; ++i) {
        expected_sdot += host_dot_x[i] * host_dot_y[i];
    }
    float dot_result = 0.0f;
    if (cublasSdot(handle, kDotCount, dev_dot_x, 1, dev_dot_y, 1, &dot_result) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSdot failed\n");
        return 1;
    }
    if (std::fabs(dot_result - expected_sdot) > 5e-3f) {
        std::fprintf(stderr,
                     "FAIL: SDOT mismatch (got=%f expected=%f)\n",
                     static_cast<double>(dot_result),
                     static_cast<double>(expected_sdot));
        return 1;
    }
    if (cublasSdot(handle, kDotCount, host_dot_x.data(), 1, dev_dot_y, 1, &dot_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SDOT\n");
        return 1;
    }

    std::vector<double> host_ddot_x(kDotCount);
    std::vector<double> host_ddot_y(kDotCount);
    for (int i = 0; i < kDotCount; ++i) {
        host_ddot_x[i] = 0.1 + static_cast<double>((i * 7) % 29) * 0.2;
        host_ddot_y[i] = 0.2 + static_cast<double>((i * 11) % 17) * 0.15;
    }
    double* dev_ddot_x = nullptr;
    double* dev_ddot_y = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_ddot_x), host_ddot_x.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&dev_ddot_y), host_ddot_y.size() * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for DDOT failed\n");
        return 1;
    }
    if (cudaMemcpy(dev_ddot_x,
                   host_ddot_x.data(),
                   host_ddot_x.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_ddot_y,
                   host_ddot_y.data(),
                   host_ddot_y.size() * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device for DDOT failed\n");
        return 1;
    }

    double expected_ddot = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_ddot += host_ddot_x[i] * host_ddot_y[i];
    }
    double ddot_result = 0.0;
    if (cublasDdot(handle, kDotCount, dev_ddot_x, 1, dev_ddot_y, 1, &ddot_result) !=
        CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDdot failed\n");
        return 1;
    }
    if (std::fabs(ddot_result - expected_ddot) > 1e-9) {
        std::fprintf(stderr, "FAIL: DDOT mismatch (got=%f expected=%f)\n", ddot_result, expected_ddot);
        return 1;
    }
    if (cublasDdot(handle, kDotCount, dev_ddot_x, 1, host_ddot_y.data(), 1, &ddot_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host Y in DDOT\n");
        return 1;
    }

    double expected_snrm2_acc = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_snrm2_acc += static_cast<double>(host_dot_x[i]) * static_cast<double>(host_dot_x[i]);
    }
    const float expected_snrm2 = static_cast<float>(std::sqrt(expected_snrm2_acc));
    float snrm2_result = 0.0f;
    if (cublasSnrm2(handle, kDotCount, dev_dot_x, 1, &snrm2_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSnrm2 failed\n");
        return 1;
    }
    if (std::fabs(snrm2_result - expected_snrm2) > 1e-4f) {
        std::fprintf(stderr,
                     "FAIL: SNRM2 mismatch (got=%f expected=%f)\n",
                     static_cast<double>(snrm2_result),
                     static_cast<double>(expected_snrm2));
        return 1;
    }
    if (cublasSnrm2(handle, kDotCount, host_dot_x.data(), 1, &snrm2_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SNRM2\n");
        return 1;
    }

    double expected_dnrm2_acc = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_dnrm2_acc += host_ddot_x[i] * host_ddot_x[i];
    }
    const double expected_dnrm2 = std::sqrt(expected_dnrm2_acc);
    double dnrm2_result = 0.0;
    if (cublasDnrm2(handle, kDotCount, dev_ddot_x, 1, &dnrm2_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDnrm2 failed\n");
        return 1;
    }
    if (std::fabs(dnrm2_result - expected_dnrm2) > 1e-10) {
        std::fprintf(stderr, "FAIL: DNRM2 mismatch (got=%f expected=%f)\n", dnrm2_result, expected_dnrm2);
        return 1;
    }
    if (cublasDnrm2(handle, kDotCount, host_ddot_x.data(), 1, &dnrm2_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DNRM2\n");
        return 1;
    }

    double expected_sasum_acc = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_sasum_acc += std::fabs(static_cast<double>(host_dot_x[i]));
    }
    const float expected_sasum = static_cast<float>(expected_sasum_acc);
    float sasum_result = 0.0f;
    if (cublasSasum(handle, kDotCount, dev_dot_x, 1, &sasum_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasSasum failed\n");
        return 1;
    }
    if (std::fabs(sasum_result - expected_sasum) > 1e-4f) {
        std::fprintf(stderr,
                     "FAIL: SASUM mismatch (got=%f expected=%f)\n",
                     static_cast<double>(sasum_result),
                     static_cast<double>(expected_sasum));
        return 1;
    }
    if (cublasSasum(handle, kDotCount, host_dot_x.data(), 1, &sasum_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in SASUM\n");
        return 1;
    }

    double expected_dasum = 0.0;
    for (int i = 0; i < kDotCount; ++i) {
        expected_dasum += std::fabs(host_ddot_x[i]);
    }
    double dasum_result = 0.0;
    if (cublasDasum(handle, kDotCount, dev_ddot_x, 1, &dasum_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDasum failed\n");
        return 1;
    }
    if (std::fabs(dasum_result - expected_dasum) > 1e-10) {
        std::fprintf(stderr, "FAIL: DASUM mismatch (got=%f expected=%f)\n", dasum_result, expected_dasum);
        return 1;
    }
    if (cublasDasum(handle, kDotCount, host_ddot_x.data(), 1, &dasum_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in DASUM\n");
        return 1;
    }

    int expected_isamax = 1;
    float best_isamax = std::fabs(host_dot_x[0]);
    for (int i = 1; i < kDotCount; ++i) {
        const float value = std::fabs(host_dot_x[i]);
        if (value > best_isamax) {
            best_isamax = value;
            expected_isamax = i + 1;
        }
    }
    int isamax_result = 0;
    if (cublasIsamax(handle, kDotCount, dev_dot_x, 1, &isamax_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasIsamax failed\n");
        return 1;
    }
    if (isamax_result != expected_isamax) {
        std::fprintf(stderr,
                     "FAIL: ISAMAX mismatch (got=%d expected=%d)\n",
                     isamax_result,
                     expected_isamax);
        return 1;
    }
    if (cublasIsamax(handle, kDotCount, host_dot_x.data(), 1, &isamax_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in ISAMAX\n");
        return 1;
    }

    int expected_idamax = 1;
    double best_idamax = std::fabs(host_ddot_x[0]);
    for (int i = 1; i < kDotCount; ++i) {
        const double value = std::fabs(host_ddot_x[i]);
        if (value > best_idamax) {
            best_idamax = value;
            expected_idamax = i + 1;
        }
    }
    int idamax_result = 0;
    if (cublasIdamax(handle, kDotCount, dev_ddot_x, 1, &idamax_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasIdamax failed\n");
        return 1;
    }
    if (idamax_result != expected_idamax) {
        std::fprintf(stderr,
                     "FAIL: IDAMAX mismatch (got=%d expected=%d)\n",
                     idamax_result,
                     expected_idamax);
        return 1;
    }
    if (cublasIdamax(handle, kDotCount, host_ddot_x.data(), 1, &idamax_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in IDAMAX\n");
        return 1;
    }

    int expected_isamin = 1;
    float best_isamin = std::fabs(host_dot_x[0]);
    for (int i = 1; i < kDotCount; ++i) {
        const float value = std::fabs(host_dot_x[i]);
        if (value < best_isamin) {
            best_isamin = value;
            expected_isamin = i + 1;
        }
    }
    int isamin_result = 0;
    if (cublasIsamin(handle, kDotCount, dev_dot_x, 1, &isamin_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasIsamin failed\n");
        return 1;
    }
    if (isamin_result != expected_isamin) {
        std::fprintf(stderr,
                     "FAIL: ISAMIN mismatch (got=%d expected=%d)\n",
                     isamin_result,
                     expected_isamin);
        return 1;
    }
    if (cublasIsamin(handle, kDotCount, host_dot_x.data(), 1, &isamin_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in ISAMIN\n");
        return 1;
    }

    int expected_idamin = 1;
    double best_idamin = std::fabs(host_ddot_x[0]);
    for (int i = 1; i < kDotCount; ++i) {
        const double value = std::fabs(host_ddot_x[i]);
        if (value < best_idamin) {
            best_idamin = value;
            expected_idamin = i + 1;
        }
    }
    int idamin_result = 0;
    if (cublasIdamin(handle, kDotCount, dev_ddot_x, 1, &idamin_result) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasIdamin failed\n");
        return 1;
    }
    if (idamin_result != expected_idamin) {
        std::fprintf(stderr,
                     "FAIL: IDAMIN mismatch (got=%d expected=%d)\n",
                     idamin_result,
                     expected_idamin);
        return 1;
    }
    if (cublasIdamin(handle, kDotCount, host_ddot_x.data(), 1, &idamin_result) !=
        CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for host X in IDAMIN\n");
        return 1;
    }

    int* dev_index = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&dev_index), sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for index result failed\n");
        return 1;
    }
    if (cublasIsamax(handle, kDotCount, dev_dot_x, 1, dev_index) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for device result in ISAMAX\n");
        return 1;
    }
    if (cublasIdamax(handle, kDotCount, dev_ddot_x, 1, dev_index) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for device result in IDAMAX\n");
        return 1;
    }
    if (cublasIsamin(handle, kDotCount, dev_dot_x, 1, dev_index) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for device result in ISAMIN\n");
        return 1;
    }
    if (cublasIdamin(handle, kDotCount, dev_ddot_x, 1, dev_index) != CUBLAS_STATUS_INVALID_VALUE) {
        std::fprintf(stderr, "FAIL: expected CUBLAS_STATUS_INVALID_VALUE for device result in IDAMIN\n");
        return 1;
    }

    int zero_isamax = -1;
    if (cublasIsamax(handle, 0, dev_dot_x, 1, &zero_isamax) != CUBLAS_STATUS_SUCCESS || zero_isamax != 0) {
        std::fprintf(stderr, "FAIL: cublasIsamax n==0 behavior mismatch\n");
        return 1;
    }
    int zero_idamax = -1;
    if (cublasIdamax(handle, 0, dev_ddot_x, 1, &zero_idamax) != CUBLAS_STATUS_SUCCESS ||
        zero_idamax != 0) {
        std::fprintf(stderr, "FAIL: cublasIdamax n==0 behavior mismatch\n");
        return 1;
    }
    int zero_isamin = -1;
    if (cublasIsamin(handle, 0, dev_dot_x, 1, &zero_isamin) != CUBLAS_STATUS_SUCCESS || zero_isamin != 0) {
        std::fprintf(stderr, "FAIL: cublasIsamin n==0 behavior mismatch\n");
        return 1;
    }
    int zero_idamin = -1;
    if (cublasIdamin(handle, 0, dev_ddot_x, 1, &zero_idamin) != CUBLAS_STATUS_SUCCESS ||
        zero_idamin != 0) {
        std::fprintf(stderr, "FAIL: cublasIdamin n==0 behavior mismatch\n");
        return 1;
    }

    if (cudaFree(dev_x) != cudaSuccess || cudaFree(dev_y) != cudaSuccess ||
        cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_c) != cudaSuccess || cudaFree(dev_at) != cudaSuccess ||
        cudaFree(dev_bt) != cudaSuccess || cudaFree(dev_ct) != cudaSuccess ||
        cudaFree(dev_an) != cudaSuccess || cudaFree(dev_bn) != cudaSuccess ||
        cudaFree(dev_cn) != cudaSuccess || cudaFree(dev_ad) != cudaSuccess ||
        cudaFree(dev_bd) != cudaSuccess || cudaFree(dev_cd) != cudaSuccess ||
        cudaFree(dev_dx) != cudaSuccess || cudaFree(dev_dy) != cudaSuccess ||
        cudaFree(dev_swap_dx) != cudaSuccess || cudaFree(dev_swap_dy) != cudaSuccess ||
        cudaFree(dev_dot_x) != cudaSuccess || cudaFree(dev_dot_y) != cudaSuccess ||
        cudaFree(dev_ddot_x) != cudaSuccess || cudaFree(dev_ddot_y) != cudaSuccess ||
        cudaFree(dev_index) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    if (cublasDestroy(handle) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cublasDestroy failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuBLAS shim operations validated\n");
    return 0;
}
