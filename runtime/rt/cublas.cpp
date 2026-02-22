#include "cublas_v2.h"
#include "metal_backend.h"
#include "runtime_internal.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

struct cublasContext {
    cudaStream_t stream = nullptr;
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

namespace {

constexpr int kCublasCompatVersion = 12000;

bool is_valid_operation(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

bool is_valid_fill_mode(cublasFillMode_t mode) {
    return mode == CUBLAS_FILL_MODE_LOWER || mode == CUBLAS_FILL_MODE_UPPER;
}

bool is_valid_math_mode(cublasMath_t mode) {
    return mode == CUBLAS_DEFAULT_MATH || mode == CUBLAS_TENSOR_OP_MATH ||
           mode == CUBLAS_PEDANTIC_MATH || mode == CUBLAS_TF32_TENSOR_OP_MATH;
}

cublasStatus_t map_cuda_status_to_cublas(cudaError_t status) {
    if (status == cudaSuccess) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (status == cudaErrorInvalidValue || status == cudaErrorInvalidDevicePointer) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (status == cudaErrorMemoryAllocation) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <typename T>
T sym_element(const T* a, int lda, int row, int col, cublasFillMode_t uplo) {
    if (uplo == CUBLAS_FILL_MODE_UPPER) {
        return (row <= col) ? a[row + col * lda] : a[col + row * lda];
    }
    return (row >= col) ? a[row + col * lda] : a[col + row * lda];
}

cublasStatus_t synchronize_handle_stream(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const cudaError_t status = cudaStreamSynchronize(handle->stream);
    if (status != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Helper: read element of symmetric n×n matrix (column-major, upper or lower stored).
template<typename T>
static inline T symm_elem(const T* a, int lda, int i, int j, bool upper) {
    return upper ? (i <= j ? a[i + j * lda] : a[j + i * lda])
                 : (i >= j ? a[i + j * lda] : a[j + i * lda]);
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasHandle_t created = new (std::nothrow) cublasContext();
    if (created == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = created;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) {
    if (handle == nullptr || version == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *version = kCublasCompatVersion;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream_id) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->stream = stream_id;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id) {
    if (handle == nullptr || stream_id == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *stream_id = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_math_mode(mode)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->math_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
    if (handle == nullptr || mode == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *mode = handle->math_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x,
                           int incx,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy(cublasHandle_t handle,
                           int n,
                           const float* x,
                           int incx,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        const int xi = i * incx;
        const int yi = i * incy;
        const float tmp = x[xi];
        x[xi] = y[yi];
        y[yi] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                           int n,
                           const double* alpha,
                           const double* x,
                           int incx,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || alpha == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        const int xi = i * incx;
        const int yi = i * incy;
        const double tmp = x[xi];
        x[xi] = y[yi];
        y[yi] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(cublasHandle_t handle,
                          int n,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
    }
    *result = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(cublasHandle_t handle,
                          int n,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i * incx] * y[i * incy];
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(x[i * incx]));
    }
    *result = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(x[i * incx]);
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i * incx]);
        sum_sq += v * v;
    }
    *result = static_cast<float>(std::sqrt(sum_sq));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = x[i * incx];
        sum_sq += v * v;
    }
    *result = std::sqrt(sum_sq);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    float best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const float value = std::fabs(x[i * incx]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax(cublasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    double best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const double value = std::fabs(x[i * incx]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    float best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const float value = std::fabs(x[i * incx]);
        if (value < best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin(cublasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    double best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const double value = std::fabs(x[i * incx]);
        if (value < best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* b,
                           int ldb,
                           const float* beta,
                           float* c,
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::string error;
    const cudaError_t gemm_status = cumetal::metal_backend::gemm_f32(
        transa != CUBLAS_OP_N,
        transb != CUBLAS_OP_N,
        m,
        n,
        k,
        *alpha,
        a_resolved.buffer,
        a_resolved.offset,
        lda,
        b_resolved.buffer,
        b_resolved.offset,
        ldb,
        *beta,
        c_resolved.buffer,
        c_resolved.offset,
        ldc,
        nullptr,
        &error);
    return map_cuda_status_to_cublas(gemm_status);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const float* alpha,
                                         const float* a,
                                         int lda,
                                         long long int stridea,
                                         const float* b,
                                         int ldb,
                                         long long int strideb,
                                         const float* beta,
                                         float* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || batch_count < 0 || alpha == nullptr || beta == nullptr ||
        stridea < 0 || strideb < 0 || stridec < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch_count == 0 || m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::string error;
    const cudaError_t gemm_status = cumetal::metal_backend::gemm_strided_batched_f32(
        transa != CUBLAS_OP_N,
        transb != CUBLAS_OP_N,
        m,
        n,
        k,
        *alpha,
        a_resolved.buffer,
        a_resolved.offset,
        lda,
        static_cast<std::size_t>(stridea) * sizeof(float),
        b_resolved.buffer,
        b_resolved.offset,
        ldb,
        static_cast<std::size_t>(strideb) * sizeof(float),
        *beta,
        c_resolved.buffer,
        c_resolved.offset,
        ldc,
        static_cast<std::size_t>(stridec) * sizeof(float),
        batch_count,
        nullptr,
        &error);

    return map_cuda_status_to_cublas(gemm_status);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* b,
                           int ldb,
                           const double* beta,
                           double* c,
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                const double a_value = (transa == CUBLAS_OP_N) ? a[row + p * lda] : a[p + row * lda];
                const double b_value = (transb == CUBLAS_OP_N) ? b[p + col * ldb] : b[col + p * ldb];
                sum += a_value * b_value;
            }
            c[row + col * ldc] = alpha_value * sum + beta_value * c[row + col * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const double* alpha,
                                         const double* a,
                                         int lda,
                                         long long int stridea,
                                         const double* b,
                                         int ldb,
                                         long long int strideb,
                                         const double* beta,
                                         double* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || batch_count < 0 || alpha == nullptr || beta == nullptr ||
        stridea < 0 || strideb < 0 || stridec < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch_count == 0 || m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int batch = 0; batch < batch_count; ++batch) {
        const double* a_batch = a + batch * stridea;
        const double* b_batch = b + batch * strideb;
        double* c_batch = c + batch * stridec;
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                double sum = 0.0;
                for (int p = 0; p < k; ++p) {
                    const double a_val = (transa == CUBLAS_OP_N) ? a_batch[row + p * lda]
                                                                  : a_batch[p + row * lda];
                    const double b_val = (transb == CUBLAS_OP_N) ? b_batch[p + col * ldb]
                                                                  : b_batch[col + p * ldb];
                    sum += a_val * b_val;
                }
                c_batch[row + col * ldc] = alpha_value * sum + beta_value * c_batch[row + col * ldc];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int output_len = (trans == CUBLAS_OP_N) ? m : n;
    if (output_len == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    const float beta_value = *beta;
    if (trans == CUBLAS_OP_N) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < n; ++col) {
                sum += a[row + col * lda] * x[col * incx];
            }
            y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
        }
    } else {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int row = 0; row < m; ++row) {
                sum += a[row + col * lda] * x[row * incx];
            }
            y[col * incy] = alpha_value * sum + beta_value * y[col * incy];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int output_len = (trans == CUBLAS_OP_N) ? m : n;
    if (output_len == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    if (trans == CUBLAS_OP_N) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int col = 0; col < n; ++col) {
                sum += a[row + col * lda] * x[col * incx];
            }
            y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
        }
    } else {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int row = 0; row < m; ++row) {
                sum += a[row + col * lda] * x[row * incx];
            }
            y[col * incy] = alpha_value * sum + beta_value * y[col * incy];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(cublasHandle_t handle,
                          int m,
                          int n,
                          const float* alpha,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* a,
                          int lda) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (m < 0 || n < 0 || alpha == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0 ||
        cumetalRuntimeIsDevicePointer(a) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int col = 0; col < n; ++col) {
        const float y_value = y[col * incy];
        for (int row = 0; row < m; ++row) {
            a[row + col * lda] += alpha_value * x[row * incx] * y_value;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(cublasHandle_t handle,
                          int m,
                          int n,
                          const double* alpha,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* a,
                          int lda) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (m < 0 || n < 0 || alpha == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0 ||
        cumetalRuntimeIsDevicePointer(a) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int col = 0; col < n; ++col) {
        const double y_value = y[col * incy];
        for (int row = 0; row < m; ++row) {
            a[row + col * lda] += alpha_value * x[row * incx] * y_value;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (n > 1 ? n : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    const float beta_value = *beta;
    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += sym_element(a, lda, row, col, uplo) * x[col * incx];
        }
        y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (n > 1 ? n : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += sym_element(a, lda, row, col, uplo) * x[col * incx];
        }
        y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// GemmEx / GemmStridedBatchedEx
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* a, cudaDataType_t atype, int lda,
                            const void* b, cudaDataType_t btype, int ldb,
                            const void* beta,
                            void* c, cudaDataType_t ctype, int ldc,
                            cublasComputeType_t compute_type,
                            cublasGemmAlgo_t /* algo */) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    // Route to the appropriate typed GEMM based on compute/data types.
    if ((compute_type == CUBLAS_COMPUTE_32F ||
         compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) &&
        atype == CUDA_R_32F && btype == CUDA_R_32F && ctype == CUDA_R_32F) {
        return cublasSgemm(handle, transa, transb, m, n, k,
                           static_cast<const float*>(alpha),
                           static_cast<const float*>(a), lda,
                           static_cast<const float*>(b), ldb,
                           static_cast<const float*>(beta),
                           static_cast<float*>(c), ldc);
    }

    if (compute_type == CUBLAS_COMPUTE_64F &&
        atype == CUDA_R_64F && btype == CUDA_R_64F && ctype == CUDA_R_64F) {
        return cublasDgemm(handle, transa, transb, m, n, k,
                           static_cast<const double*>(alpha),
                           static_cast<const double*>(a), lda,
                           static_cast<const double*>(b), ldb,
                           static_cast<const double*>(beta),
                           static_cast<double*>(c), ldc);
    }

    // FP16 compute or mixed types: upconvert to float, compute, downconvert.
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (a == nullptr || b == nullptr || c == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    // Helper: read one scalar element as float from a typed buffer.
    auto read_f32 = [](const void* ptr, int idx, cudaDataType_t t) -> float {
        switch (t) {
            case CUDA_R_32F: return static_cast<const float*>(ptr)[idx];
            case CUDA_R_64F: return static_cast<float>(static_cast<const double*>(ptr)[idx]);
            case CUDA_R_16F: return static_cast<float>(static_cast<const __half*>(ptr)[idx]);
            default:         return 0.0f;
        }
    };
    auto write_f32 = [](void* ptr, int idx, float val, cudaDataType_t t) {
        switch (t) {
            case CUDA_R_32F: static_cast<float*>(ptr)[idx] = val; break;
            case CUDA_R_64F: static_cast<double*>(ptr)[idx] = static_cast<double>(val); break;
            case CUDA_R_16F: static_cast<__half*>(ptr)[idx] = static_cast<__half>(val); break;
            default: break;
        }
    };

    const float alpha_f = read_f32(alpha, 0, (atype == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F);
    const float beta_f  = read_f32(beta,  0, (ctype == CUDA_R_64F) ? CUDA_R_64F : CUDA_R_32F);

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) return sync_status;

    // Naive reference implementation for mixed/fp16 types.
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int p = 0; p < k; ++p) {
                float a_val = 0.0f, b_val = 0.0f;
                if (transa == CUBLAS_OP_N) {
                    a_val = read_f32(a, row + p * lda, atype);
                } else {
                    a_val = read_f32(a, p + row * lda, atype);
                }
                if (transb == CUBLAS_OP_N) {
                    b_val = read_f32(b, p + col * ldb, btype);
                } else {
                    b_val = read_f32(b, col + p * ldb, btype);
                }
                sum += a_val * b_val;
            }
            const float c_old = read_f32(c, row + col * ldc, ctype);
            write_f32(c, row + col * ldc, alpha_f * sum + beta_f * c_old, ctype);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          const void* alpha,
                                          const void* a, cudaDataType_t atype, int lda,
                                          long long int stridea,
                                          const void* b, cudaDataType_t btype, int ldb,
                                          long long int strideb,
                                          const void* beta,
                                          void* c, cudaDataType_t ctype, int ldc,
                                          long long int stridec,
                                          int batch_count,
                                          cublasComputeType_t compute_type,
                                          cublasGemmAlgo_t algo) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch_count < 0) return CUBLAS_STATUS_INVALID_VALUE;

    // Route fp32 strided batched via cublasSgemmStridedBatched.
    if ((compute_type == CUBLAS_COMPUTE_32F ||
         compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) &&
        atype == CUDA_R_32F && btype == CUDA_R_32F && ctype == CUDA_R_32F) {
        return cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                         static_cast<const float*>(alpha),
                                         static_cast<const float*>(a), lda, stridea,
                                         static_cast<const float*>(b), ldb, strideb,
                                         static_cast<const float*>(beta),
                                         static_cast<float*>(c), ldc, stridec,
                                         batch_count);
    }
    if (compute_type == CUBLAS_COMPUTE_64F &&
        atype == CUDA_R_64F && btype == CUDA_R_64F && ctype == CUDA_R_64F) {
        return cublasDgemmStridedBatched(handle, transa, transb, m, n, k,
                                         static_cast<const double*>(alpha),
                                         static_cast<const double*>(a), lda, stridea,
                                         static_cast<const double*>(b), ldb, strideb,
                                         static_cast<const double*>(beta),
                                         static_cast<double*>(c), ldc, stridec,
                                         batch_count);
    }

    // Delegate each batch slice to GemmEx.
    auto byte_offset = [](const void* base, long long int elems, cudaDataType_t t) -> const void* {
        std::size_t sz = 4;
        if (t == CUDA_R_64F) sz = 8;
        else if (t == CUDA_R_16F) sz = 2;
        return static_cast<const char*>(base) + elems * sz;
    };
    auto byte_offset_mutable = [](void* base, long long int elems, cudaDataType_t t) -> void* {
        std::size_t sz = 4;
        if (t == CUDA_R_64F) sz = 8;
        else if (t == CUDA_R_16F) sz = 2;
        return static_cast<char*>(base) + elems * sz;
    };

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                         byte_offset(a, stridea * bi, atype), atype, lda,
                         byte_offset(b, strideb * bi, btype), btype, ldb,
                         beta,
                         byte_offset_mutable(c, stridec * bi, ctype), ctype, ldc,
                         compute_type, algo);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hgemm — half-precision GEMM (implemented via fp32 upconvert)
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m, int n, int k,
                           const __half* alpha,
                           const __half* a, int lda,
                           const __half* b, int ldb,
                           const __half* beta,
                           __half* c, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const float falpha = static_cast<float>(*alpha);
    const float fbeta  = static_cast<float>(*beta);
    return cublasGemmEx(handle, transa, transb, m, n, k,
                        &falpha, a, CUDA_R_16F, lda,
                                 b, CUDA_R_16F, ldb,
                        &fbeta,  c, CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ─────────────────────────────────────────────────────────────────────────────
// SgemmBatched / DgemmBatched — array-of-pointers batched GEMM
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float* alpha,
                                  const float* const a_array[], int lda,
                                  const float* const b_array[], int ldb,
                                  const float* beta,
                                  float* const c_array[], int ldc,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batch_count < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasSgemm(handle, transa, transb, m, n, k,
                        alpha, a_array[bi], lda, b_array[bi], ldb,
                        beta, c_array[bi], ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double* alpha,
                                  const double* const a_array[], int lda,
                                  const double* const b_array[], int ldb,
                                  const double* beta,
                                  double* const c_array[], int ldc,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batch_count < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasDgemm(handle, transa, transb, m, n, k,
                        alpha, a_array[bi], lda, b_array[bi], ldb,
                        beta, c_array[bi], ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Strsm / Dtrsm — triangular solve with multiple RHS
// Solves: op(A) * X = alpha * B  (CUBLAS_SIDE_LEFT)
//      or X * op(A) = alpha * B  (CUBLAS_SIDE_RIGHT)
// Result is written back into B.
// ─────────────────────────────────────────────────────────────────────────────

// Macro to generate trsm body for float/double (avoids template inside extern "C").
// TRSM: solve op(A) * X = alpha * B  (side=LEFT) or  X * op(A) = alpha * B  (side=RIGHT).
// B is m×n, A is m×m (LEFT) or n×n (RIGHT). Result written into B.
#define CUMETAL_TRSM_BODY(T, zero_val, one_val)                                     \
    do {                                                                             \
        if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;                \
        if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;                   \
        if (side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT)                  \
            return CUBLAS_STATUS_INVALID_VALUE;                                     \
        if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;          \
        if (!is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;         \
        if (diag != CUBLAS_DIAG_NON_UNIT && diag != CUBLAS_DIAG_UNIT)              \
            return CUBLAS_STATUS_INVALID_VALUE;                                     \
        if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;                    \
        if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;                         \
        if (a == nullptr || b == nullptr) return CUBLAS_STATUS_INVALID_VALUE;       \
        const cublasStatus_t _sync = synchronize_handle_stream(handle);             \
        if (_sync != CUBLAS_STATUS_SUCCESS) return _sync;                           \
        const T alpha_val = *alpha;                                                  \
        const bool _unit = (diag == CUBLAS_DIAG_UNIT);                              \
        const bool _lower = (uplo == CUBLAS_FILL_MODE_LOWER);                       \
        const bool _notrans = (trans == CUBLAS_OP_N);                               \
        if (alpha_val != one_val) {                                                  \
            for (int _c = 0; _c < n; ++_c)                                          \
                for (int _r = 0; _r < m; ++_r)                                      \
                    b[_r + _c * ldb] *= alpha_val;                                  \
        }                                                                            \
        auto _ae = [&](int r, int c) -> T {                                         \
            return _notrans ? a[r + c * lda] : a[c + r * lda]; };                  \
        auto _ad = [&](int i) -> T {                                                \
            return _unit ? one_val : a[i + i * lda]; };                             \
        if (side == CUBLAS_SIDE_LEFT) {                                              \
            for (int _c = 0; _c < n; ++_c) {                                        \
                if ((_lower && _notrans) || (!_lower && !_notrans)) {               \
                    for (int _r = 0; _r < m; ++_r) {                               \
                        T _s = b[_r + _c * ldb];                                    \
                        for (int _p = 0; _p < _r; ++_p) _s -= _ae(_r,_p)*b[_p+_c*ldb];\
                        b[_r + _c * ldb] = _s / _ad(_r);                           \
                    }                                                                \
                } else {                                                             \
                    for (int _r = m-1; _r >= 0; --_r) {                            \
                        T _s = b[_r + _c * ldb];                                    \
                        for (int _p = _r+1; _p < m; ++_p) _s -= _ae(_r,_p)*b[_p+_c*ldb];\
                        b[_r + _c * ldb] = _s / _ad(_r);                           \
                    }                                                                \
                }                                                                    \
            }                                                                        \
        } else {                                                                     \
            if ((!_lower && _notrans) || (_lower && !_notrans)) {                   \
                for (int _c = 0; _c < n; ++_c) {                                   \
                    T _dv = _ad(_c);                                                 \
                    for (int _r = 0; _r < m; ++_r) b[_r+_c*ldb] /= _dv;           \
                    for (int _p = _c+1; _p < n; ++_p) {                            \
                        T _f = _ae(_c,_p);                                           \
                        for (int _r = 0; _r < m; ++_r) b[_r+_p*ldb] -= _f*b[_r+_c*ldb];\
                    }                                                                \
                }                                                                    \
            } else {                                                                 \
                for (int _c = n-1; _c >= 0; --_c) {                               \
                    T _dv = _ad(_c);                                                 \
                    for (int _r = 0; _r < m; ++_r) b[_r+_c*ldb] /= _dv;           \
                    for (int _p = 0; _p < _c; ++_p) {                              \
                        T _f = _ae(_c,_p);                                           \
                        for (int _r = 0; _r < m; ++_r) b[_r+_p*ldb] -= _f*b[_r+_c*ldb];\
                    }                                                                \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        return CUBLAS_STATUS_SUCCESS;                                                \
    } while (0)

cublasStatus_t cublasStrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m, int n,
                           const float* alpha,
                           const float* a, int lda,
                           float* b, int ldb) {
    CUMETAL_TRSM_BODY(float, 0.0f, 1.0f);
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m, int n,
                           const double* alpha,
                           const double* a, int lda,
                           double* b, int ldb) {
    CUMETAL_TRSM_BODY(double, 0.0, 1.0);
}

#undef CUMETAL_TRSM_BODY

// ─────────────────────────────────────────────────────────────────────────────
// SetVector / GetVector / SetMatrix / GetMatrix
// On Apple Silicon UMA all memory is coherent; these are strided memcpy helpers.
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasSetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy) {
    if (n < 0 || elem_size <= 0 || incx <= 0 || incy <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const std::size_t es = static_cast<std::size_t>(elem_size);
    for (int i = 0; i < n; ++i) {
        std::memcpy(static_cast<char*>(y) + static_cast<std::size_t>(i * incy) * es,
                    static_cast<const char*>(x) + static_cast<std::size_t>(i * incx) * es,
                    es);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy) {
    return cublasSetVector(n, elem_size, x, incx, y, incy);
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb) {
    if (rows < 0 || cols < 0 || elem_size <= 0 || lda < rows || ldb < rows)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return CUBLAS_STATUS_SUCCESS;
    if (a == nullptr || b == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const std::size_t es  = static_cast<std::size_t>(elem_size);
    const std::size_t row_bytes = static_cast<std::size_t>(rows) * es;
    for (int col = 0; col < cols; ++col) {
        std::memcpy(static_cast<char*>(b) + static_cast<std::size_t>(col * ldb) * es,
                    static_cast<const char*>(a) + static_cast<std::size_t>(col * lda) * es,
                    row_bytes);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb) {
    return cublasSetMatrix(rows, cols, elem_size, a, lda, b, ldb);
}

cublasStatus_t cublasSetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t /* stream */) {
    return cublasSetVector(n, elem_size, x, incx, y, incy);
}

cublasStatus_t cublasGetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t /* stream */) {
    return cublasSetVector(n, elem_size, x, incx, y, incy);
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t /* stream */) {
    return cublasSetMatrix(rows, cols, elem_size, a, lda, b, ldb);
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t /* stream */) {
    return cublasSetMatrix(rows, cols, elem_size, a, lda, b, ldb);
}

const char* cublasGetStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "cuBLAS operation completed successfully";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cuBLAS library not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "cuBLAS resource allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "Invalid value passed to cuBLAS function";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "Feature not supported on this architecture";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "Memory mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "cuBLAS kernel execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "cuBLAS internal error";
        default:
            return "Unknown cuBLAS status";
    }
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
    if (value == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (type) {
        case MAJOR_VERSION: *value = 12; break;
        case MINOR_VERSION: *value = 0;  break;
        case PATCH_LEVEL:   *value = 0;  break;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-1 update: A = alpha * x * x^T + A  (column-major, only upper/lower triangle updated)
cublasStatus_t cublasSsyr(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const float* alpha,
                           const float* x, int incx,
                           float* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float a_val = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const float xj = a_val * x[j * incx];
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += xj * x[i * incx];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const double* alpha,
                           const double* x, int incx,
                           double* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double a_val = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const double xj = a_val * x[j * incx];
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += xj * x[i * incx];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-k update: C = alpha * op(A) * op(A)^T + beta * C  (only upper/lower triangle)
cublasStatus_t cublasSsyrk(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const float* alpha, const float* a, int lda,
                            const float* beta,  float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    // lda: if no_trans, A is n×k; else k×n
    if (no_trans && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && lda < k) return CUBLAS_STATUS_INVALID_VALUE;
    const float av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                const float ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const float aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrk(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const double* alpha, const double* a, int lda,
                            const double* beta,  double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && lda < k) return CUBLAS_STATUS_INVALID_VALUE;
    const double av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                const double ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const double aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-2k update: C = alpha * (op(A)*op(B)^T + op(B)*op(A)^T) + beta * C
cublasStatus_t cublasSsyr2k(cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const float* alpha,
                             const float* a, int lda,
                             const float* b, int ldb,
                             const float* beta,
                             float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && (lda < n || ldb < n)) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && (lda < k || ldb < k)) return CUBLAS_STATUS_INVALID_VALUE;
    const float av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                const float ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const float bj = no_trans ? b[j + l * ldb] : b[l + j * ldb];
                const float bi = no_trans ? b[i + l * ldb] : b[l + i * ldb];
                const float aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * bj + bi * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2k(cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const double* alpha,
                             const double* a, int lda,
                             const double* b, int ldb,
                             const double* beta,
                             double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && (lda < n || ldb < n)) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && (lda < k || ldb < k)) return CUBLAS_STATUS_INVALID_VALUE;
    const double av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                const double ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const double bj = no_trans ? b[j + l * ldb] : b[l + j * ldb];
                const double bi = no_trans ? b[i + l * ldb] : b[l + i * ldb];
                const double aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * bj + bi * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS2: Ssyr2 / Dsyr2 — symmetric rank-2 update: A := α·x·yᵀ + α·y·xᵀ + A ────────

cublasStatus_t cublasSsyr2(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            int n,
                            const float* alpha,
                            const float* x, int incx,
                            const float* y, int incy,
                            float* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += av * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            int n,
                            const double* alpha,
                            const double* x, int incx,
                            const double* y, int incy,
                            double* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += av * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS3: Ssymm / Dsymm — symmetric matrix-matrix multiply ──────────────────────────

cublasStatus_t cublasSsymm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            int m, int n,
                            const float* alpha,
                            const float* a, int lda,
                            const float* b, int ldb,
                            const float* beta,
                            float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
        beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            if (left) {
                for (int k = 0; k < ka; ++k) {
                    sum += symm_elem(a, lda, i, k, upper) * b[k + j * ldb];
                }
            } else {
                for (int k = 0; k < ka; ++k) {
                    sum += b[i + k * ldb] * symm_elem(a, lda, k, j, upper);
                }
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            int m, int n,
                            const double* alpha,
                            const double* a, int lda,
                            const double* b, int ldb,
                            const double* beta,
                            double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
        beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            if (left) {
                for (int k = 0; k < ka; ++k) {
                    sum += symm_elem(a, lda, i, k, upper) * b[k + j * ldb];
                }
            } else {
                for (int k = 0; k < ka; ++k) {
                    sum += b[i + k * ldb] * symm_elem(a, lda, k, j, upper);
                }
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS2: Strmv / Dtrmv — triangular matrix-vector multiply: x := op(A)·x ──────────

cublasStatus_t cublasStrmv(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int n,
                            const float* a, int lda,
                            float* x, int incx) {
    if (handle == nullptr || a == nullptr || x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper    = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit     = (diag  == CUBLAS_DIAG_UNIT);
    std::vector<float> tmp(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            bool in_tri;
            float aik;
            if (no_trans) {
                in_tri = upper ? (k >= i) : (k <= i);
                aik    = (k == i && unit) ? 1.0f : a[i + k * lda];
            } else {
                in_tri = upper ? (i >= k) : (i <= k);
                aik    = (k == i && unit) ? 1.0f : a[k + i * lda];
            }
            if (in_tri) sum += aik * x[k * incx];
        }
        tmp[i] = sum;
    }
    for (int i = 0; i < n; ++i) x[i * incx] = tmp[i];
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmv(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int n,
                            const double* a, int lda,
                            double* x, int incx) {
    if (handle == nullptr || a == nullptr || x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper    = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit     = (diag  == CUBLAS_DIAG_UNIT);
    std::vector<double> tmp(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            bool in_tri;
            double aik;
            if (no_trans) {
                in_tri = upper ? (k >= i) : (k <= i);
                aik    = (k == i && unit) ? 1.0 : a[i + k * lda];
            } else {
                in_tri = upper ? (i >= k) : (i <= k);
                aik    = (k == i && unit) ? 1.0 : a[k + i * lda];
            }
            if (in_tri) sum += aik * x[k * incx];
        }
        tmp[i] = sum;
    }
    for (int i = 0; i < n; ++i) x[i * incx] = tmp[i];
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS3: Strmm / Dtrmm — triangular matrix-matrix multiply ─────────────────────────
// B := alpha · op(A) · B  (SIDE_LEFT)  or  B := alpha · B · op(A)  (SIDE_RIGHT)
// cuBLAS v2 API: result written to C (B is input, C is output).

cublasStatus_t cublasStrmm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m, int n,
                            const float* alpha,
                            const float* a, int lda,
                            const float* b, int ldb,
                            float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av     = *alpha;
    const bool upper   = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit    = (diag  == CUBLAS_DIAG_UNIT);
    const bool left    = (side  == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < ka; ++k) {
                float aelem;
                bool in_tri;
                if (left) {
                    int ar = no_trans ? i : k;
                    int ac = no_trans ? k : i;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0f : a[ar + ac * lda];
                } else {
                    int ar = no_trans ? k : j;
                    int ac = no_trans ? j : k;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0f : a[ar + ac * lda];
                }
                if (in_tri) sum += aelem * b[left ? (k + j * ldb) : (i + k * ldb)];
            }
            c[i + j * ldc] = av * sum;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m, int n,
                            const double* alpha,
                            const double* a, int lda,
                            const double* b, int ldb,
                            double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av    = *alpha;
    const bool upper   = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit    = (diag  == CUBLAS_DIAG_UNIT);
    const bool left    = (side  == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int k = 0; k < ka; ++k) {
                double aelem;
                bool in_tri;
                if (left) {
                    int ar = no_trans ? i : k;
                    int ac = no_trans ? k : i;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0 : a[ar + ac * lda];
                } else {
                    int ar = no_trans ? k : j;
                    int ac = no_trans ? j : k;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0 : a[ar + ac * lda];
                }
                if (in_tri) sum += aelem * b[left ? (k + j * ldb) : (i + k * ldb)];
            }
            c[i + j * ldc] = av * sum;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS1: Srot / Drot — apply Givens rotation ───────────────────────────────────────

cublasStatus_t cublasSrot(cublasHandle_t handle,
                           int n,
                           float* x, int incx,
                           float* y, int incy,
                           const float* c,
                           const float* s) {
    if (handle == nullptr || x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float cv = *c, sv = *s;
    for (int i = 0; i < n; ++i) {
        const float xi = x[i * incx];
        const float yi = y[i * incy];
        x[i * incx] =  cv * xi + sv * yi;
        y[i * incy] = -sv * xi + cv * yi;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrot(cublasHandle_t handle,
                           int n,
                           double* x, int incx,
                           double* y, int incy,
                           const double* c,
                           const double* s) {
    if (handle == nullptr || x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double cv = *c, sv = *s;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i * incx];
        const double yi = y[i * incy];
        x[i * incx] =  cv * xi + sv * yi;
        y[i * incy] = -sv * xi + cv * yi;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS1: Srotg / Drotg — construct Givens rotation ─────────────────────────────────
// Given (a,b): compute (c,s,r,z) such that [c s; -s c]*[a;b] = [r;0].
// a is overwritten with r; b with z.

cublasStatus_t cublasSrotg(cublasHandle_t handle,
                            float* a, float* b,
                            float* c, float* s) {
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float fa = std::fabs(*a), fb = std::fabs(*b);
    if (fb == 0.0f) {
        *c = 1.0f; *s = 0.0f; *b = 0.0f;
    } else if (fa == 0.0f) {
        *c = 0.0f; *s = (*b > 0.0f) ? 1.0f : -1.0f;
        *a = fb; *b = 1.0f;
    } else if (fb > fa) {
        const float t  = *a / *b;
        const float sg = (*b > 0.0f) ? 1.0f : -1.0f;
        *s = sg / std::sqrt(1.0f + t * t);
        *c = *s * t;
        *a = *b / *s;
        *b = (std::fabs(*c) > 0.0f && std::fabs(*c) < 1.0f) ? 1.0f / *c : 1.0f;
    } else {
        const float t  = *b / *a;
        const float sg = (*a > 0.0f) ? 1.0f : -1.0f;
        *c = sg / std::sqrt(1.0f + t * t);
        *s = *c * t;
        *a = *a / *c;
        *b = (std::fabs(*s) < 1.0f) ? *s : (std::fabs(*c) > 0.0f ? 1.0f / *c : 1.0f);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotg(cublasHandle_t handle,
                            double* a, double* b,
                            double* c, double* s) {
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double fa = std::fabs(*a), fb = std::fabs(*b);
    if (fb == 0.0) {
        *c = 1.0; *s = 0.0; *b = 0.0;
    } else if (fa == 0.0) {
        *c = 0.0; *s = (*b > 0.0) ? 1.0 : -1.0;
        *a = fb; *b = 1.0;
    } else if (fb > fa) {
        const double t  = *a / *b;
        const double sg = (*b > 0.0) ? 1.0 : -1.0;
        *s = sg / std::sqrt(1.0 + t * t);
        *c = *s * t;
        *a = *b / *s;
        *b = (std::fabs(*c) > 0.0 && std::fabs(*c) < 1.0) ? 1.0 / *c : 1.0;
    } else {
        const double t  = *b / *a;
        const double sg = (*a > 0.0) ? 1.0 : -1.0;
        *c = sg / std::sqrt(1.0 + t * t);
        *s = *c * t;
        *a = *a / *c;
        *b = (std::fabs(*s) < 1.0) ? *s : (std::fabs(*c) > 0.0 ? 1.0 / *c : 1.0);
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"
