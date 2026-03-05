#include "cudnn.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

// cuDNN shim: CPU-backed convolution and tensor ops via im2col + GEMM (Accelerate).
// Sufficient for inference workloads on Apple Silicon UMA.

struct cudnnContext {
    cudaStream_t stream = nullptr;
};

struct cudnnTensorStruct {
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 0, c = 0, h = 0, w = 0;
    int nStride = 0, cStride = 0, hStride = 0, wStride = 0;
};

struct cudnnFilterStruct {
    cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int k = 0, c = 0, h = 0, w = 0;
};

struct cudnnConvolutionStruct {
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    int dilation_h = 1, dilation_w = 1;
    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
    cudnnMathType_t mathType = CUDNN_DEFAULT_MATH;
    int groupCount = 1;
};

struct cudnnActivationStruct {
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    double coef = 0.0;
};

namespace {

bool debug_cudnn() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("CUMETAL_DEBUG_CUDNN");
        v = (e && e[0] && e[0] != '0') ? 1 : 0;
    }
    return v != 0;
}

#define CUDNN_DEBUG(fmt, ...)                                                 \
    do {                                                                       \
        if (debug_cudnn())                                                     \
            std::fprintf(stderr, "[cuDNN] " fmt "\n", ##__VA_ARGS__);          \
    } while (0)

size_t dtype_size(cudnnDataType_t dt) {
    switch (dt) {
        case CUDNN_DATA_FLOAT: return 4;
        case CUDNN_DATA_DOUBLE: return 8;
        case CUDNN_DATA_HALF:
        case CUDNN_DATA_BFLOAT16: return 2;
        case CUDNN_DATA_INT8:
        case CUDNN_DATA_UINT8: return 1;
        case CUDNN_DATA_INT32: return 4;
        case CUDNN_DATA_INT64: return 8;
        default: return 4;
    }
}

void compute_strides(cudnnTensorStruct* t) {
    if (t->format == CUDNN_TENSOR_NCHW) {
        t->wStride = 1;
        t->hStride = t->w;
        t->cStride = t->h * t->w;
        t->nStride = t->c * t->h * t->w;
    } else { // NHWC
        t->cStride = 1;
        t->wStride = t->c;
        t->hStride = t->w * t->c;
        t->nStride = t->h * t->w * t->c;
    }
}

// im2col: expand input for GEMM-based convolution
void im2col_f32(const float* data_im, int C, int H, int W,
                int kH, int kW, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_col) {
    int outH = (H + 2 * pad_h - (dilation_h * (kH - 1) + 1)) / stride_h + 1;
    int outW = (W + 2 * pad_w - (dilation_w * (kW - 1) + 1)) / stride_w + 1;

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        int ih = oh * stride_h - pad_h + kh * dilation_h;
                        int iw = ow * stride_w - pad_w + kw * dilation_w;
                        int col_idx = ((c * kH + kh) * kW + kw) * outH * outW + oh * outW + ow;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            data_col[col_idx] = data_im[(c * H + ih) * W + iw];
                        } else {
                            data_col[col_idx] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// col2im: inverse of im2col — accumulate columns back into image
void col2im_f32(const float* data_col, int C, int H, int W,
                int kH, int kW, int pad_h, int pad_w,
                int stride_h, int stride_w, int dilation_h, int dilation_w,
                float* data_im) {
    int outH = (H + 2 * pad_h - (dilation_h * (kH - 1) + 1)) / stride_h + 1;
    int outW = (W + 2 * pad_w - (dilation_w * (kW - 1) + 1)) / stride_w + 1;

    std::memset(data_im, 0, C * H * W * sizeof(float));

    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                for (int oh = 0; oh < outH; ++oh) {
                    for (int ow = 0; ow < outW; ++ow) {
                        int ih = oh * stride_h - pad_h + kh * dilation_h;
                        int iw = ow * stride_w - pad_w + kw * dilation_w;
                        int col_idx = ((c * kH + kh) * kW + kw) * outH * outW + oh * outW + ow;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                            data_im[(c * H + ih) * W + iw] += data_col[col_idx];
                        }
                    }
                }
            }
        }
    }
}

} // namespace

extern "C" {

size_t cudnnGetVersion(void) { return 8907; } // 8.9.7

const char* cudnnGetErrorString(cudnnStatus_t status) {
    switch (status) {
        case CUDNN_STATUS_SUCCESS: return "CUDNN_STATUS_SUCCESS";
        case CUDNN_STATUS_NOT_INITIALIZED: return "CUDNN_STATUS_NOT_INITIALIZED";
        case CUDNN_STATUS_ALLOC_FAILED: return "CUDNN_STATUS_ALLOC_FAILED";
        case CUDNN_STATUS_BAD_PARAM: return "CUDNN_STATUS_BAD_PARAM";
        case CUDNN_STATUS_INTERNAL_ERROR: return "CUDNN_STATUS_INTERNAL_ERROR";
        case CUDNN_STATUS_INVALID_VALUE: return "CUDNN_STATUS_INVALID_VALUE";
        case CUDNN_STATUS_ARCH_MISMATCH: return "CUDNN_STATUS_ARCH_MISMATCH";
        case CUDNN_STATUS_MAPPING_ERROR: return "CUDNN_STATUS_MAPPING_ERROR";
        case CUDNN_STATUS_EXECUTION_FAILED: return "CUDNN_STATUS_EXECUTION_FAILED";
        case CUDNN_STATUS_NOT_SUPPORTED: return "CUDNN_STATUS_NOT_SUPPORTED";
        default: return "CUDNN_STATUS_UNKNOWN";
    }
}

// ── Handle ──

cudnnStatus_t cudnnCreate(cudnnHandle_t* handle) {
    if (!handle) return CUDNN_STATUS_BAD_PARAM;
    auto* ctx = new (std::nothrow) cudnnContext;
    if (!ctx) return CUDNN_STATUS_ALLOC_FAILED;
    *handle = ctx;
    CUDNN_DEBUG("cudnnCreate handle=%p", (void*)ctx);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDestroy(cudnnHandle_t handle) {
    if (!handle) return CUDNN_STATUS_BAD_PARAM;
    CUDNN_DEBUG("cudnnDestroy handle=%p", (void*)handle);
    delete handle;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream) {
    if (!handle) return CUDNN_STATUS_BAD_PARAM;
    handle->stream = stream;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* stream) {
    if (!handle || !stream) return CUDNN_STATUS_BAD_PARAM;
    *stream = handle->stream;
    return CUDNN_STATUS_SUCCESS;
}

// ── Tensor descriptor ──

cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc) {
    if (!tensorDesc) return CUDNN_STATUS_BAD_PARAM;
    *tensorDesc = new (std::nothrow) cudnnTensorStruct;
    return *tensorDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc) {
    delete tensorDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnTensorFormat_t format,
                                          cudnnDataType_t dataType,
                                          int n, int c, int h, int w) {
    if (!tensorDesc) return CUDNN_STATUS_BAD_PARAM;
    tensorDesc->format = format;
    tensorDesc->dataType = dataType;
    tensorDesc->n = n; tensorDesc->c = c; tensorDesc->h = h; tensorDesc->w = w;
    compute_strides(tensorDesc);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnDataType_t* dataType,
                                          int* n, int* c, int* h, int* w,
                                          int* nStride, int* cStride,
                                          int* hStride, int* wStride) {
    if (!tensorDesc) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = tensorDesc->dataType;
    if (n) *n = tensorDesc->n;
    if (c) *c = tensorDesc->c;
    if (h) *h = tensorDesc->h;
    if (w) *w = tensorDesc->w;
    if (nStride) *nStride = tensorDesc->nStride;
    if (cStride) *cStride = tensorDesc->cStride;
    if (hStride) *hStride = tensorDesc->hStride;
    if (wStride) *wStride = tensorDesc->wStride;
    return CUDNN_STATUS_SUCCESS;
}

// ── Filter descriptor ──

cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc) {
    if (!filterDesc) return CUDNN_STATUS_BAD_PARAM;
    *filterDesc = new (std::nothrow) cudnnFilterStruct;
    return *filterDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc) {
    delete filterDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                          cudnnDataType_t dataType,
                                          cudnnTensorFormat_t format,
                                          int k, int c, int h, int w) {
    if (!filterDesc) return CUDNN_STATUS_BAD_PARAM;
    filterDesc->dataType = dataType;
    filterDesc->format = format;
    filterDesc->k = k; filterDesc->c = c; filterDesc->h = h; filterDesc->w = w;
    return CUDNN_STATUS_SUCCESS;
}

// ── Convolution descriptor ──

cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc) {
    if (!convDesc) return CUDNN_STATUS_BAD_PARAM;
    *convDesc = new (std::nothrow) cudnnConvolutionStruct;
    return *convDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc) {
    delete convDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                               int pad_h, int pad_w,
                                               int u, int v,
                                               int dilation_h, int dilation_w,
                                               cudnnConvolutionMode_t mode,
                                               cudnnDataType_t computeType) {
    if (!convDesc) return CUDNN_STATUS_BAD_PARAM;
    convDesc->pad_h = pad_h; convDesc->pad_w = pad_w;
    convDesc->stride_h = u; convDesc->stride_w = v;
    convDesc->dilation_h = dilation_h; convDesc->dilation_w = dilation_w;
    convDesc->mode = mode;
    convDesc->computeType = computeType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                           cudnnMathType_t mathType) {
    if (!convDesc) return CUDNN_STATUS_BAD_PARAM;
    convDesc->mathType = mathType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                                             int groupCount) {
    if (!convDesc) return CUDNN_STATUS_BAD_PARAM;
    convDesc->groupCount = groupCount;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnTensorDescriptor_t inputTensorDesc,
                                                     cudnnFilterDescriptor_t filterDesc,
                                                     int* n, int* c, int* h, int* w) {
    if (!convDesc || !inputTensorDesc || !filterDesc) return CUDNN_STATUS_BAD_PARAM;
    int kH = filterDesc->h, kW = filterDesc->w;
    int pH = convDesc->pad_h, pW = convDesc->pad_w;
    int sH = convDesc->stride_h, sW = convDesc->stride_w;
    int dH = convDesc->dilation_h, dW = convDesc->dilation_w;

    if (n) *n = inputTensorDesc->n;
    if (c) *c = filterDesc->k;
    if (h) *h = (inputTensorDesc->h + 2 * pH - (dH * (kH - 1) + 1)) / sH + 1;
    if (w) *w = (inputTensorDesc->w + 2 * pW - (dW * (kW - 1) + 1)) / sW + 1;
    return CUDNN_STATUS_SUCCESS;
}

// ── Forward convolution ──

cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t /*handle*/,
                                                       cudnnTensorDescriptor_t xDesc,
                                                       cudnnFilterDescriptor_t wDesc,
                                                       cudnnConvolutionDescriptor_t convDesc,
                                                       cudnnTensorDescriptor_t /*yDesc*/,
                                                       cudnnConvolutionFwdAlgo_t /*algo*/,
                                                       size_t* sizeInBytes) {
    if (!sizeInBytes || !xDesc || !wDesc || !convDesc) return CUDNN_STATUS_BAD_PARAM;
    // im2col workspace: C_in * kH * kW * outH * outW * sizeof(float)
    int C = xDesc->c / convDesc->groupCount;
    int kH = wDesc->h, kW = wDesc->w;
    int pH = convDesc->pad_h, pW = convDesc->pad_w;
    int sH = convDesc->stride_h, sW = convDesc->stride_w;
    int dH = convDesc->dilation_h, dW = convDesc->dilation_w;
    int outH = (xDesc->h + 2 * pH - (dH * (kH - 1) + 1)) / sH + 1;
    int outW = (xDesc->w + 2 * pW - (dW * (kW - 1) + 1)) / sW + 1;
    *sizeInBytes = (size_t)C * kH * kW * outH * outW * dtype_size(xDesc->dataType);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t /*handle*/,
                                                    cudnnTensorDescriptor_t /*xDesc*/,
                                                    cudnnFilterDescriptor_t /*wDesc*/,
                                                    cudnnConvolutionDescriptor_t /*convDesc*/,
                                                    cudnnTensorDescriptor_t /*yDesc*/,
                                                    int requestedAlgoCount,
                                                    int* returnedAlgoCount,
                                                    cudnnConvolutionFwdAlgoPerf_t* perfResults) {
    if (!returnedAlgoCount) return CUDNN_STATUS_BAD_PARAM;
    int count = std::min(requestedAlgoCount, 1);
    if (count > 0 && perfResults) {
        std::memset(&perfResults[0], 0, sizeof(cudnnConvolutionFwdAlgoPerf_t));
        perfResults[0].algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
        perfResults[0].time = 0.0f;
        perfResults[0].memory = 0;
        perfResults[0].mathType = CUDNN_DEFAULT_MATH;
    }
    *returnedAlgoCount = count;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t /*handle*/,
                                       const void* alpha,
                                       cudnnTensorDescriptor_t xDesc, const void* x,
                                       cudnnFilterDescriptor_t wDesc, const void* w,
                                       cudnnConvolutionDescriptor_t convDesc,
                                       cudnnConvolutionFwdAlgo_t /*algo*/,
                                       void* workSpace, size_t /*workSpaceSizeInBytes*/,
                                       const void* beta,
                                       cudnnTensorDescriptor_t yDesc, void* y) {
    if (!alpha || !xDesc || !x || !wDesc || !w || !convDesc || !beta || !yDesc || !y)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    const float* wf = static_cast<const float*>(w);
    float* yf = static_cast<float*>(y);

    int N = xDesc->n;
    int C_in = xDesc->c;
    int H = xDesc->h, W = xDesc->w;
    int K = wDesc->k;
    int kH = wDesc->h, kW = wDesc->w;
    int groups = convDesc->groupCount;
    int C_per_group = C_in / groups;
    int K_per_group = K / groups;

    int outH = (H + 2 * convDesc->pad_h - (convDesc->dilation_h * (kH - 1) + 1)) / convDesc->stride_h + 1;
    int outW = (W + 2 * convDesc->pad_w - (convDesc->dilation_w * (kW - 1) + 1)) / convDesc->stride_w + 1;

    int col_size = C_per_group * kH * kW;
    int spatial = outH * outW;

    CUDNN_DEBUG("convForward N=%d C=%d H=%d W=%d K=%d kH=%d kW=%d outH=%d outW=%d groups=%d",
                N, C_in, H, W, K, kH, kW, outH, outW, groups);

    // Use workspace for im2col if provided, else allocate
    float* col_buf = static_cast<float*>(workSpace);
    bool own_col = false;
    if (!col_buf) {
        col_buf = static_cast<float*>(std::malloc(col_size * spatial * sizeof(float)));
        if (!col_buf) return CUDNN_STATUS_ALLOC_FAILED;
        own_col = true;
    }

    // Scale existing output by beta
    int y_size = N * K * outH * outW;
    if (b != 0.0f && b != 1.0f) {
        cblas_sscal(y_size, b, yf, 1);
    } else if (b == 0.0f) {
        std::memset(yf, 0, y_size * sizeof(float));
    }

    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* x_g = xf + n * C_in * H * W + g * C_per_group * H * W;
            const float* w_g = wf + g * K_per_group * C_per_group * kH * kW;
            float* y_g = yf + n * K * outH * outW + g * K_per_group * outH * outW;

            im2col_f32(x_g, C_per_group, H, W, kH, kW,
                       convDesc->pad_h, convDesc->pad_w,
                       convDesc->stride_h, convDesc->stride_w,
                       convDesc->dilation_h, convDesc->dilation_w,
                       col_buf);

            // y_g = alpha * w_g * col_buf + y_g (already scaled by beta)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        K_per_group, spatial, col_size,
                        a, w_g, col_size, col_buf, spatial,
                        1.0f, y_g, spatial);
        }
    }

    if (own_col) std::free(col_buf);
    return CUDNN_STATUS_SUCCESS;
}

// ── v7 algorithm finder (same as FindAlgorithm, just the modern name) ──

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                                      cudnnTensorDescriptor_t xDesc,
                                                      cudnnFilterDescriptor_t wDesc,
                                                      cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnTensorDescriptor_t yDesc,
                                                      int requestedAlgoCount,
                                                      int* returnedAlgoCount,
                                                      cudnnConvolutionFwdAlgoPerf_t* perfResults) {
    return cudnnFindConvolutionForwardAlgorithm(handle, xDesc, wDesc, convDesc, yDesc,
                                                 requestedAlgoCount, returnedAlgoCount, perfResults);
}

// ── Backward convolution (data) ──

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t /*handle*/,
                                                            cudnnFilterDescriptor_t wDesc,
                                                            cudnnTensorDescriptor_t dyDesc,
                                                            cudnnConvolutionDescriptor_t convDesc,
                                                            cudnnTensorDescriptor_t /*dxDesc*/,
                                                            cudnnConvolutionBwdDataAlgo_t /*algo*/,
                                                            size_t* sizeInBytes) {
    if (!sizeInBytes || !wDesc || !dyDesc || !convDesc) return CUDNN_STATUS_BAD_PARAM;
    int C = wDesc->c;
    int kH = wDesc->h, kW = wDesc->w;
    int outH = dyDesc->h, outW = dyDesc->w;
    *sizeInBytes = (size_t)C * kH * kW * outH * outW * sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t /*handle*/,
                                                         cudnnFilterDescriptor_t /*wDesc*/,
                                                         cudnnTensorDescriptor_t /*dyDesc*/,
                                                         cudnnConvolutionDescriptor_t /*convDesc*/,
                                                         cudnnTensorDescriptor_t /*dxDesc*/,
                                                         int requestedAlgoCount,
                                                         int* returnedAlgoCount,
                                                         cudnnConvolutionBwdDataAlgoPerf_t* perfResults) {
    if (!returnedAlgoCount) return CUDNN_STATUS_BAD_PARAM;
    int count = std::min(requestedAlgoCount, 1);
    if (count > 0 && perfResults) {
        std::memset(&perfResults[0], 0, sizeof(cudnnConvolutionBwdDataAlgoPerf_t));
        perfResults[0].algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
    }
    *returnedAlgoCount = count;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t /*handle*/,
                                            const void* alpha,
                                            cudnnFilterDescriptor_t wDesc, const void* w,
                                            cudnnTensorDescriptor_t dyDesc, const void* dy,
                                            cudnnConvolutionDescriptor_t convDesc,
                                            cudnnConvolutionBwdDataAlgo_t /*algo*/,
                                            void* workSpace, size_t /*workSpaceSizeInBytes*/,
                                            const void* beta,
                                            cudnnTensorDescriptor_t dxDesc, void* dx) {
    if (!alpha || !wDesc || !w || !dyDesc || !dy || !convDesc || !beta || !dxDesc || !dx)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* wf = static_cast<const float*>(w);
    const float* dyf = static_cast<const float*>(dy);
    float* dxf = static_cast<float*>(dx);

    int N = dyDesc->n;
    int K = wDesc->k;
    int C_in = wDesc->c;
    int H = dxDesc->h, W = dxDesc->w;
    int kH = wDesc->h, kW = wDesc->w;
    int groups = convDesc->groupCount;
    int C_per_group = C_in / groups;
    int K_per_group = K / groups;
    int outH = dyDesc->h, outW = dyDesc->w;
    int col_size = C_per_group * kH * kW;
    int spatial = outH * outW;

    float* col_buf = static_cast<float*>(workSpace);
    bool own_col = false;
    if (!col_buf) {
        col_buf = static_cast<float*>(std::malloc(col_size * spatial * sizeof(float)));
        if (!col_buf) return CUDNN_STATUS_ALLOC_FAILED;
        own_col = true;
    }

    // Scale dx by beta
    int dx_size = N * C_in * H * W;
    if (b != 1.0f) {
        if (b == 0.0f) std::memset(dxf, 0, dx_size * sizeof(float));
        else cblas_sscal(dx_size, b, dxf, 1);
    }

    // dx = alpha * wT * dy, then col2im
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* w_g = wf + g * K_per_group * C_per_group * kH * kW;
            const float* dy_g = dyf + n * K * outH * outW + g * K_per_group * outH * outW;
            float* dx_g = dxf + n * C_in * H * W + g * C_per_group * H * W;

            // col = wT * dy: (col_size x K_per_group) * (K_per_group x spatial) = (col_size x spatial)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                        col_size, spatial, K_per_group,
                        a, w_g, col_size, dy_g, spatial,
                        0.0f, col_buf, spatial);

            // Accumulate col2im into dx
            col2im_f32(col_buf, C_per_group, H, W, kH, kW,
                       convDesc->pad_h, convDesc->pad_w,
                       convDesc->stride_h, convDesc->stride_w,
                       convDesc->dilation_h, convDesc->dilation_w,
                       dx_g);
        }
    }

    if (own_col) std::free(col_buf);
    return CUDNN_STATUS_SUCCESS;
}

// ── Backward convolution (filter) ──

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t /*handle*/,
                                                              cudnnTensorDescriptor_t xDesc,
                                                              cudnnTensorDescriptor_t /*dyDesc*/,
                                                              cudnnConvolutionDescriptor_t convDesc,
                                                              cudnnFilterDescriptor_t dwDesc,
                                                              cudnnConvolutionBwdFilterAlgo_t /*algo*/,
                                                              size_t* sizeInBytes) {
    if (!sizeInBytes || !xDesc || !convDesc || !dwDesc) return CUDNN_STATUS_BAD_PARAM;
    int C = xDesc->c / convDesc->groupCount;
    int kH = dwDesc->h, kW = dwDesc->w;
    int pH = convDesc->pad_h, pW = convDesc->pad_w;
    int sH = convDesc->stride_h, sW = convDesc->stride_w;
    int dH = convDesc->dilation_h, dW = convDesc->dilation_w;
    int outH = (xDesc->h + 2 * pH - (dH * (kH - 1) + 1)) / sH + 1;
    int outW = (xDesc->w + 2 * pW - (dW * (kW - 1) + 1)) / sW + 1;
    *sizeInBytes = (size_t)C * kH * kW * outH * outW * sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t /*handle*/,
                                                           cudnnTensorDescriptor_t /*xDesc*/,
                                                           cudnnTensorDescriptor_t /*dyDesc*/,
                                                           cudnnConvolutionDescriptor_t /*convDesc*/,
                                                           cudnnFilterDescriptor_t /*dwDesc*/,
                                                           int requestedAlgoCount,
                                                           int* returnedAlgoCount,
                                                           cudnnConvolutionBwdFilterAlgoPerf_t* perfResults) {
    if (!returnedAlgoCount) return CUDNN_STATUS_BAD_PARAM;
    int count = std::min(requestedAlgoCount, 1);
    if (count > 0 && perfResults) {
        std::memset(&perfResults[0], 0, sizeof(cudnnConvolutionBwdFilterAlgoPerf_t));
        perfResults[0].algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        perfResults[0].status = CUDNN_STATUS_SUCCESS;
    }
    *returnedAlgoCount = count;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t /*handle*/,
                                              const void* alpha,
                                              cudnnTensorDescriptor_t xDesc, const void* x,
                                              cudnnTensorDescriptor_t dyDesc, const void* dy,
                                              cudnnConvolutionDescriptor_t convDesc,
                                              cudnnConvolutionBwdFilterAlgo_t /*algo*/,
                                              void* workSpace, size_t /*workSpaceSizeInBytes*/,
                                              const void* beta,
                                              cudnnFilterDescriptor_t dwDesc, void* dw) {
    if (!alpha || !xDesc || !x || !dyDesc || !dy || !convDesc || !beta || !dwDesc || !dw)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    const float* dyf = static_cast<const float*>(dy);
    float* dwf = static_cast<float*>(dw);

    int N = xDesc->n;
    int C_in = xDesc->c;
    int H = xDesc->h, W = xDesc->w;
    int K = dwDesc->k;
    int kH = dwDesc->h, kW = dwDesc->w;
    int groups = convDesc->groupCount;
    int C_per_group = C_in / groups;
    int K_per_group = K / groups;

    int outH = dyDesc->h, outW = dyDesc->w;
    int col_size = C_per_group * kH * kW;
    int spatial = outH * outW;

    float* col_buf = static_cast<float*>(workSpace);
    bool own_col = false;
    if (!col_buf) {
        col_buf = static_cast<float*>(std::malloc(col_size * spatial * sizeof(float)));
        if (!col_buf) return CUDNN_STATUS_ALLOC_FAILED;
        own_col = true;
    }

    // Scale dw by beta
    int dw_size = K * C_per_group * kH * kW;
    if (b != 1.0f) {
        if (b == 0.0f) std::memset(dwf, 0, dw_size * sizeof(float));
        else cblas_sscal(dw_size, b, dwf, 1);
    }

    // dw += alpha * dy * col^T  (accumulated over batch)
    for (int n = 0; n < N; ++n) {
        for (int g = 0; g < groups; ++g) {
            const float* x_g = xf + n * C_in * H * W + g * C_per_group * H * W;
            const float* dy_g = dyf + n * K * outH * outW + g * K_per_group * outH * outW;
            float* dw_g = dwf + g * K_per_group * C_per_group * kH * kW;

            im2col_f32(x_g, C_per_group, H, W, kH, kW,
                       convDesc->pad_h, convDesc->pad_w,
                       convDesc->stride_h, convDesc->stride_w,
                       convDesc->dilation_h, convDesc->dilation_w,
                       col_buf);

            // dw_g += alpha * dy_g * col_buf^T: (K_per_group x spatial) * (spatial x col_size)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        K_per_group, col_size, spatial,
                        a, dy_g, spatial, col_buf, spatial,
                        1.0f, dw_g, col_size);
        }
    }

    if (own_col) std::free(col_buf);
    return CUDNN_STATUS_SUCCESS;
}

// ── Backward bias ──

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t /*handle*/,
                                            const void* alpha,
                                            cudnnTensorDescriptor_t dyDesc, const void* dy,
                                            const void* beta,
                                            cudnnTensorDescriptor_t dbDesc, void* db) {
    if (!alpha || !dyDesc || !dy || !beta || !dbDesc || !db) return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* dyf = static_cast<const float*>(dy);
    float* dbf = static_cast<float*>(db);

    int N = dyDesc->n, C = dyDesc->c, H = dyDesc->h, W = dyDesc->w;

    // db[c] = beta * db[c] + alpha * sum_over(n,h,w) dy[n,c,h,w]
    for (int c = 0; c < C; ++c) {
        float sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    sum += dyf[((n * C + c) * H + h) * W + w];
                }
            }
        }
        dbf[c] = b * dbf[c] + a * sum;
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Activation ──

cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc) {
    if (!activationDesc) return CUDNN_STATUS_BAD_PARAM;
    *activationDesc = new (std::nothrow) cudnnActivationStruct;
    return *activationDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc) {
    delete activationDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                            cudnnActivationMode_t mode,
                                            cudnnNanPropagation_t reluNanOpt,
                                            double coef) {
    if (!activationDesc) return CUDNN_STATUS_BAD_PARAM;
    activationDesc->mode = mode;
    activationDesc->nanOpt = reluNanOpt;
    activationDesc->coef = coef;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnActivationForward(cudnnHandle_t /*handle*/,
                                      cudnnActivationDescriptor_t activationDesc,
                                      const void* alpha,
                                      cudnnTensorDescriptor_t xDesc, const void* x,
                                      const void* beta,
                                      cudnnTensorDescriptor_t yDesc, void* y) {
    if (!activationDesc || !alpha || !xDesc || !x || !beta || !yDesc || !y)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);
    int count = xDesc->n * xDesc->c * xDesc->h * xDesc->w;

    for (int i = 0; i < count; ++i) {
        float val = xf[i];
        switch (activationDesc->mode) {
            case CUDNN_ACTIVATION_SIGMOID:
                val = 1.0f / (1.0f + std::exp(-val)); break;
            case CUDNN_ACTIVATION_RELU:
                val = val > 0.0f ? val : 0.0f; break;
            case CUDNN_ACTIVATION_TANH:
                val = std::tanh(val); break;
            case CUDNN_ACTIVATION_CLIPPED_RELU:
                val = std::min(std::max(val, 0.0f), (float)activationDesc->coef); break;
            case CUDNN_ACTIVATION_ELU:
                val = val > 0.0f ? val : (float)activationDesc->coef * (std::exp(val) - 1.0f); break;
            case CUDNN_ACTIVATION_SWISH:
                val = val / (1.0f + std::exp(-val)); break;
            case CUDNN_ACTIVATION_IDENTITY:
            default:
                break;
        }
        yf[i] = a * val + b * yf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Tensor operations ──

cudnnStatus_t cudnnAddTensor(cudnnHandle_t /*handle*/,
                              const void* alpha,
                              cudnnTensorDescriptor_t aDesc, const void* A,
                              const void* beta,
                              cudnnTensorDescriptor_t cDesc, void* C) {
    if (!alpha || !aDesc || !A || !beta || !cDesc || !C) return CUDNN_STATUS_BAD_PARAM;
    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    float* cf = static_cast<float*>(C);
    const float* af = static_cast<const float*>(A);
    int count = cDesc->n * cDesc->c * cDesc->h * cDesc->w;

    // C = beta * C + alpha * A (broadcast A over C dimensions)
    int a_count = aDesc->n * aDesc->c * aDesc->h * aDesc->w;
    if (a_count == count) {
        cblas_sscal(count, b, cf, 1);
        cblas_saxpy(count, a, af, 1, cf, 1);
    } else {
        // Bias-add: A is 1×C×1×1, broadcast over N×C×H×W
        for (int i = 0; i < count; ++i) {
            int c_idx = (i / (cDesc->h * cDesc->w)) % cDesc->c;
            cf[i] = b * cf[i] + a * af[c_idx];
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t /*handle*/,
                                    const void* alpha,
                                    cudnnTensorDescriptor_t xDesc, const void* x,
                                    const void* beta,
                                    cudnnTensorDescriptor_t yDesc, void* y) {
    if (!alpha || !xDesc || !x || !beta || !yDesc || !y) return CUDNN_STATUS_BAD_PARAM;
    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    float* yf = static_cast<float*>(y);
    const float* xf = static_cast<const float*>(x);
    int count = yDesc->n * yDesc->c * yDesc->h * yDesc->w;
    for (int i = 0; i < count; ++i)
        yf[i] = a * xf[i] + b * yf[i];
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetTensor(cudnnHandle_t /*handle*/,
                              cudnnTensorDescriptor_t yDesc,
                              void* y,
                              const void* valuePtr) {
    if (!yDesc || !y || !valuePtr) return CUDNN_STATUS_BAD_PARAM;
    float val = *static_cast<const float*>(valuePtr);
    float* yf = static_cast<float*>(y);
    int count = yDesc->n * yDesc->c * yDesc->h * yDesc->w;
    for (int i = 0; i < count; ++i) yf[i] = val;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t /*handle*/,
                                cudnnTensorDescriptor_t yDesc,
                                void* y,
                                const void* alpha) {
    if (!yDesc || !y || !alpha) return CUDNN_STATUS_BAD_PARAM;
    float a = *static_cast<const float*>(alpha);
    float* yf = static_cast<float*>(y);
    int count = yDesc->n * yDesc->c * yDesc->h * yDesc->w;
    cblas_sscal(count, a, yf, 1);
    return CUDNN_STATUS_SUCCESS;
}

// ── Softmax ──

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t /*handle*/,
                                   cudnnSoftmaxAlgorithm_t algo,
                                   cudnnSoftmaxMode_t mode,
                                   const void* alpha,
                                   cudnnTensorDescriptor_t xDesc, const void* x,
                                   const void* beta,
                                   cudnnTensorDescriptor_t yDesc, void* y) {
    if (!alpha || !xDesc || !x || !beta || !yDesc || !y) return CUDNN_STATUS_BAD_PARAM;
    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;

    if (mode == CUDNN_SOFTMAX_MODE_INSTANCE) {
        int spatial = C * H * W;
        for (int n = 0; n < N; ++n) {
            const float* src = xf + n * spatial;
            float* dst = yf + n * spatial;
            float maxval = *std::max_element(src, src + spatial);
            float sum = 0.0f;
            for (int i = 0; i < spatial; ++i) {
                dst[i] = std::exp(src[i] - maxval);
                sum += dst[i];
            }
            if (algo == CUDNN_SOFTMAX_LOG) {
                float logsum = std::log(sum);
                for (int i = 0; i < spatial; ++i)
                    dst[i] = a * (src[i] - maxval - logsum) + b * dst[i];
            } else {
                for (int i = 0; i < spatial; ++i)
                    dst[i] = a * (dst[i] / sum) + b * dst[i];
            }
        }
    } else { // MODE_CHANNEL
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    float maxval = -1e30f;
                    for (int c = 0; c < C; ++c) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        maxval = std::max(maxval, xf[idx]);
                    }
                    float sum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        sum += std::exp(xf[idx] - maxval);
                    }
                    for (int c = 0; c < C; ++c) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        if (algo == CUDNN_SOFTMAX_LOG)
                            yf[idx] = a * (xf[idx] - maxval - std::log(sum)) + b * yf[idx];
                        else
                            yf[idx] = a * (std::exp(xf[idx] - maxval) / sum) + b * yf[idx];
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Batch normalization ──

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnScaleBiasMeanVarDesc*/,
    const void* bnScale, const void* bnBias,
    const void* estimatedMean, const void* estimatedVariance,
    double epsilon) {
    if (!alpha || !beta || !xDesc || !x || !yDesc || !y ||
        !bnScale || !bnBias || !estimatedMean || !estimatedVariance)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);
    const float* scale = static_cast<const float*>(bnScale);
    const float* bias = static_cast<const float*>(bnBias);
    const float* mean = static_cast<const float*>(estimatedMean);
    const float* var = static_cast<const float*>(estimatedVariance);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;

    if (mode == CUDNN_BATCHNORM_PER_ACTIVATION) {
        int spatial = C * H * W;
        for (int n = 0; n < N; ++n) {
            for (int i = 0; i < spatial; ++i) {
                float norm = (xf[n * spatial + i] - mean[i]) / std::sqrt(var[i] + (float)epsilon);
                yf[n * spatial + i] = a * (scale[i] * norm + bias[i]) + b * yf[n * spatial + i];
            }
        }
    } else { // SPATIAL or SPATIAL_PERSISTENT
        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < C; ++c) {
                float inv_std = 1.0f / std::sqrt(var[c] + (float)epsilon);
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float norm = (xf[idx] - mean[c]) * inv_std;
                        yf[idx] = a * (scale[c] * norm + bias[c]) + b * yf[idx];
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

} // extern "C"
