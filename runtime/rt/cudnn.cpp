#include "cudnn.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

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

struct cudnnPoolingStruct {
    cudnnPoolingMode_t mode = CUDNN_POOLING_MAX;
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    int windowH = 1, windowW = 1;
    int padH = 0, padW = 0;
    int strideH = 1, strideW = 1;
};

struct cudnnDropoutStruct {
    float dropout = 0.0f;
    void* states = nullptr;
    size_t stateSize = 0;
    unsigned long long seed = 0;
};

struct cudnnOpTensorStruct {
    cudnnOpTensorOp_t op = CUDNN_OP_TENSOR_ADD;
    cudnnDataType_t compType = CUDNN_DATA_FLOAT;
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
};

struct cudnnReduceTensorStruct {
    cudnnReduceTensorOp_t op = CUDNN_REDUCE_TENSOR_ADD;
    cudnnDataType_t compType = CUDNN_DATA_FLOAT;
    cudnnNanPropagation_t nanOpt = CUDNN_NOT_PROPAGATE_NAN;
    cudnnReduceTensorIndices_t indices = CUDNN_REDUCE_TENSOR_NO_INDICES;
    cudnnIndicesType_t indicesType = CUDNN_32BIT_INDICES;
};

struct cudnnRNNStruct {
    int hiddenSize = 0;
    int numLayers = 1;
    cudnnDropoutDescriptor_t dropoutDesc = nullptr;
    cudnnRNNInputMode_t inputMode = CUDNN_LINEAR_INPUT;
    cudnnDirectionMode_t direction = CUDNN_UNIDIRECTIONAL;
    cudnnRNNMode_t cellMode = CUDNN_LSTM;
    cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
    cudnnDataType_t mathPrec = CUDNN_DATA_FLOAT;
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

// ── Fused conv + bias + activation ──

cudnnStatus_t cudnnConvolutionBiasActivationForward(
    cudnnHandle_t handle,
    const void* alpha1,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void* workSpace, size_t workSpaceSizeInBytes,
    const void* alpha2,
    cudnnTensorDescriptor_t zDesc, const void* z,
    cudnnTensorDescriptor_t biasDesc, const void* bias,
    cudnnActivationDescriptor_t activationDesc,
    cudnnTensorDescriptor_t yDesc, void* y) {
    if (!handle || !alpha1 || !xDesc || !x || !wDesc || !w || !convDesc ||
        !alpha2 || !yDesc || !y)
        return CUDNN_STATUS_BAD_PARAM;

    // Step 1: y = conv(x, w) with alpha1
    float zero = 0.0f;
    cudnnStatus_t st = cudnnConvolutionForward(handle, alpha1, xDesc, x, wDesc, w,
                                                convDesc, algo, workSpace, workSpaceSizeInBytes,
                                                &zero, yDesc, y);
    if (st != CUDNN_STATUS_SUCCESS) return st;

    // Step 2: y = y + alpha2 * z (residual add)
    float a2 = *static_cast<const float*>(alpha2);
    if (zDesc && z && a2 != 0.0f) {
        float* yf = static_cast<float*>(y);
        const float* zf = static_cast<const float*>(z);
        int count = yDesc->n * yDesc->c * yDesc->h * yDesc->w;
        cblas_saxpy(count, a2, zf, 1, yf, 1);
    }

    // Step 3: y = y + bias (broadcast over N,H,W)
    if (biasDesc && bias) {
        float one = 1.0f;
        float one2 = 1.0f;
        st = cudnnAddTensor(handle, &one, biasDesc, bias, &one2, yDesc, y);
        if (st != CUDNN_STATUS_SUCCESS) return st;
    }

    // Step 4: activation in-place
    if (activationDesc && activationDesc->mode != CUDNN_ACTIVATION_IDENTITY) {
        float one = 1.0f;
        float zero2 = 0.0f;
        st = cudnnActivationForward(handle, activationDesc, &one, yDesc, y, &zero2, yDesc, y);
        if (st != CUDNN_STATUS_SUCCESS) return st;
    }

    return CUDNN_STATUS_SUCCESS;
}

// ── Activation backward ──

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t /*handle*/,
                                       cudnnActivationDescriptor_t activationDesc,
                                       const void* alpha,
                                       cudnnTensorDescriptor_t yDesc, const void* y,
                                       cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       cudnnTensorDescriptor_t xDesc, const void* x,
                                       const void* beta,
                                       cudnnTensorDescriptor_t dxDesc, void* dx) {
    if (!activationDesc || !alpha || !yDesc || !y || !dyDesc || !dy ||
        !xDesc || !x || !beta || !dxDesc || !dx)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* yf = static_cast<const float*>(y);
    const float* dyf = static_cast<const float*>(dy);
    const float* xf = static_cast<const float*>(x);
    float* dxf = static_cast<float*>(dx);
    int count = xDesc->n * xDesc->c * xDesc->h * xDesc->w;

    for (int i = 0; i < count; ++i) {
        float grad = 0.0f;
        switch (activationDesc->mode) {
            case CUDNN_ACTIVATION_SIGMOID: {
                float s = yf[i];
                grad = dyf[i] * s * (1.0f - s);
                break;
            }
            case CUDNN_ACTIVATION_RELU:
                grad = xf[i] > 0.0f ? dyf[i] : 0.0f;
                break;
            case CUDNN_ACTIVATION_TANH: {
                float t = yf[i];
                grad = dyf[i] * (1.0f - t * t);
                break;
            }
            case CUDNN_ACTIVATION_CLIPPED_RELU:
                grad = (xf[i] > 0.0f && xf[i] < (float)activationDesc->coef) ? dyf[i] : 0.0f;
                break;
            case CUDNN_ACTIVATION_ELU:
                grad = xf[i] > 0.0f ? dyf[i] : dyf[i] * (yf[i] + (float)activationDesc->coef);
                break;
            case CUDNN_ACTIVATION_SWISH: {
                float sig = 1.0f / (1.0f + std::exp(-xf[i]));
                grad = dyf[i] * (sig + xf[i] * sig * (1.0f - sig));
                break;
            }
            case CUDNN_ACTIVATION_IDENTITY:
            default:
                grad = dyf[i];
                break;
        }
        dxf[i] = a * grad + b * dxf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Pooling ──

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc) {
    if (!poolingDesc) return CUDNN_STATUS_BAD_PARAM;
    *poolingDesc = new (std::nothrow) cudnnPoolingStruct;
    return *poolingDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc) {
    delete poolingDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                           cudnnPoolingMode_t mode,
                                           cudnnNanPropagation_t maxpoolingNanOpt,
                                           int windowHeight, int windowWidth,
                                           int verticalPadding, int horizontalPadding,
                                           int verticalStride, int horizontalStride) {
    if (!poolingDesc) return CUDNN_STATUS_BAD_PARAM;
    poolingDesc->mode = mode;
    poolingDesc->nanOpt = maxpoolingNanOpt;
    poolingDesc->windowH = windowHeight;
    poolingDesc->windowW = windowWidth;
    poolingDesc->padH = verticalPadding;
    poolingDesc->padW = horizontalPadding;
    poolingDesc->strideH = verticalStride;
    poolingDesc->strideW = horizontalStride;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetPooling2dForwardOutputDim(cudnnPoolingDescriptor_t poolingDesc,
                                                 cudnnTensorDescriptor_t inputTensorDesc,
                                                 int* n, int* c, int* h, int* w) {
    if (!poolingDesc || !inputTensorDesc) return CUDNN_STATUS_BAD_PARAM;
    if (n) *n = inputTensorDesc->n;
    if (c) *c = inputTensorDesc->c;
    if (h) *h = (inputTensorDesc->h + 2 * poolingDesc->padH - poolingDesc->windowH) / poolingDesc->strideH + 1;
    if (w) *w = (inputTensorDesc->w + 2 * poolingDesc->padW - poolingDesc->windowW) / poolingDesc->strideW + 1;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingForward(cudnnHandle_t /*handle*/,
                                   cudnnPoolingDescriptor_t poolingDesc,
                                   const void* alpha,
                                   cudnnTensorDescriptor_t xDesc, const void* x,
                                   const void* beta,
                                   cudnnTensorDescriptor_t yDesc, void* y) {
    if (!poolingDesc || !alpha || !xDesc || !x || !beta || !yDesc || !y)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;
    int kH = poolingDesc->windowH, kW = poolingDesc->windowW;
    int pH = poolingDesc->padH, pW = poolingDesc->padW;
    int sH = poolingDesc->strideH, sW = poolingDesc->strideW;
    int outH = (H + 2 * pH - kH) / sH + 1;
    int outW = (W + 2 * pW - kW) / sW + 1;
    bool is_max = (poolingDesc->mode == CUDNN_POOLING_MAX ||
                   poolingDesc->mode == CUDNN_POOLING_MAX_DETERMINISTIC);
    bool exclude_pad = (poolingDesc->mode == CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    float val = is_max ? -1e30f : 0.0f;
                    int pool_count = 0;
                    for (int wh = 0; wh < kH; ++wh) {
                        for (int ww = 0; ww < kW; ++ww) {
                            int ih = oh * sH - pH + wh;
                            int iw = ow * sW - pW + ww;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                float v = xf[((n * C + c) * H + ih) * W + iw];
                                if (is_max) val = std::max(val, v);
                                else val += v;
                                ++pool_count;
                            }
                        }
                    }
                    if (!is_max) {
                        int divisor = exclude_pad ? pool_count : (kH * kW);
                        if (divisor > 0) val /= divisor;
                    }
                    int oidx = ((n * C + c) * outH + oh) * outW + ow;
                    yf[oidx] = a * val + b * yf[oidx];
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t /*handle*/,
                                    cudnnPoolingDescriptor_t poolingDesc,
                                    const void* alpha,
                                    cudnnTensorDescriptor_t yDesc, const void* y,
                                    cudnnTensorDescriptor_t dyDesc, const void* dy,
                                    cudnnTensorDescriptor_t xDesc, const void* x,
                                    const void* beta,
                                    cudnnTensorDescriptor_t dxDesc, void* dx) {
    if (!poolingDesc || !alpha || !yDesc || !y || !dyDesc || !dy ||
        !xDesc || !x || !beta || !dxDesc || !dx)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* yf = static_cast<const float*>(y);
    const float* dyf = static_cast<const float*>(dy);
    const float* xf = static_cast<const float*>(x);
    float* dxf = static_cast<float*>(dx);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;
    int kH = poolingDesc->windowH, kW = poolingDesc->windowW;
    int pH = poolingDesc->padH, pW = poolingDesc->padW;
    int sH = poolingDesc->strideH, sW = poolingDesc->strideW;
    int outH = (H + 2 * pH - kH) / sH + 1;
    int outW = (W + 2 * pW - kW) / sW + 1;
    bool is_max = (poolingDesc->mode == CUDNN_POOLING_MAX ||
                   poolingDesc->mode == CUDNN_POOLING_MAX_DETERMINISTIC);
    bool exclude_pad = (poolingDesc->mode == CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING);

    // Scale dx by beta
    int dx_size = N * C * H * W;
    if (b == 0.0f) std::memset(dxf, 0, dx_size * sizeof(float));
    else if (b != 1.0f) cblas_sscal(dx_size, b, dxf, 1);

    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    int oidx = ((n * C + c) * outH + oh) * outW + ow;
                    float dy_val = dyf[oidx];
                    if (is_max) {
                        float y_val = yf[oidx];
                        for (int wh = 0; wh < kH; ++wh) {
                            for (int ww = 0; ww < kW; ++ww) {
                                int ih = oh * sH - pH + wh;
                                int iw = ow * sW - pW + ww;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int iidx = ((n * C + c) * H + ih) * W + iw;
                                    if (xf[iidx] == y_val)
                                        dxf[iidx] += a * dy_val;
                                }
                            }
                        }
                    } else {
                        int pool_count = 0;
                        for (int wh = 0; wh < kH; ++wh) {
                            for (int ww = 0; ww < kW; ++ww) {
                                int ih = oh * sH - pH + wh;
                                int iw = ow * sW - pW + ww;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    ++pool_count;
                            }
                        }
                        int divisor = exclude_pad ? pool_count : (kH * kW);
                        float grad = (divisor > 0) ? (a * dy_val / divisor) : 0.0f;
                        for (int wh = 0; wh < kH; ++wh) {
                            for (int ww = 0; ww < kW; ++ww) {
                                int ih = oh * sH - pH + wh;
                                int iw = ow * sW - pW + ww;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                    dxf[((n * C + c) * H + ih) * W + iw] += grad;
                            }
                        }
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Dropout ──

cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc) {
    if (!dropoutDesc) return CUDNN_STATUS_BAD_PARAM;
    *dropoutDesc = new (std::nothrow) cudnnDropoutStruct;
    return *dropoutDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc) {
    delete dropoutDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                         cudnnHandle_t /*handle*/,
                                         float dropout,
                                         void* states, size_t stateSizeInBytes,
                                         unsigned long long seed) {
    if (!dropoutDesc) return CUDNN_STATUS_BAD_PARAM;
    dropoutDesc->dropout = dropout;
    dropoutDesc->states = states;
    dropoutDesc->stateSize = stateSizeInBytes;
    dropoutDesc->seed = seed;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t /*handle*/, size_t* sizeInBytes) {
    if (!sizeInBytes) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = 64; // minimal state for our RNG seed
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutForward(cudnnHandle_t /*handle*/,
                                   cudnnDropoutDescriptor_t dropoutDesc,
                                   cudnnTensorDescriptor_t xdesc, const void* x,
                                   cudnnTensorDescriptor_t ydesc, void* y,
                                   void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    if (!dropoutDesc || !xdesc || !x || !ydesc || !y)
        return CUDNN_STATUS_BAD_PARAM;

    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);
    int count = xdesc->n * xdesc->c * xdesc->h * xdesc->w;

    if (dropoutDesc->dropout <= 0.0f) {
        std::memcpy(yf, xf, count * sizeof(float));
        if (reserveSpace && reserveSpaceSizeInBytes >= (size_t)count)
            std::memset(reserveSpace, 1, count); // all kept
        return CUDNN_STATUS_SUCCESS;
    }

    // Simple xorshift64 RNG from seed
    unsigned long long state = dropoutDesc->seed ? dropoutDesc->seed : 42;
    unsigned char* mask = static_cast<unsigned char*>(reserveSpace);
    float scale = 1.0f / (1.0f - dropoutDesc->dropout);
    unsigned int threshold = (unsigned int)(dropoutDesc->dropout * 4294967295.0f);

    for (int i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bool drop = ((unsigned int)(state & 0xFFFFFFFF)) < threshold;
        if (mask && (size_t)i < reserveSpaceSizeInBytes)
            mask[i] = drop ? 0 : 1;
        yf[i] = drop ? 0.0f : xf[i] * scale;
    }
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t /*handle*/,
                                    cudnnDropoutDescriptor_t dropoutDesc,
                                    cudnnTensorDescriptor_t dydesc, const void* dy,
                                    cudnnTensorDescriptor_t dxdesc, void* dx,
                                    void* reserveSpace, size_t reserveSpaceSizeInBytes) {
    if (!dropoutDesc || !dydesc || !dy || !dxdesc || !dx)
        return CUDNN_STATUS_BAD_PARAM;

    const float* dyf = static_cast<const float*>(dy);
    float* dxf = static_cast<float*>(dx);
    int count = dydesc->n * dydesc->c * dydesc->h * dydesc->w;

    if (dropoutDesc->dropout <= 0.0f) {
        std::memcpy(dxf, dyf, count * sizeof(float));
        return CUDNN_STATUS_SUCCESS;
    }

    const unsigned char* mask = static_cast<const unsigned char*>(reserveSpace);
    float scale = 1.0f / (1.0f - dropoutDesc->dropout);

    for (int i = 0; i < count; ++i) {
        bool kept = mask && (size_t)i < reserveSpaceSizeInBytes && mask[i];
        dxf[i] = kept ? dyf[i] * scale : 0.0f;
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Nd tensor descriptor ──

cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnDataType_t dataType,
                                          int nbDims,
                                          const int dimA[],
                                          const int strideA[]) {
    if (!tensorDesc || !dimA || !strideA || nbDims < 1)
        return CUDNN_STATUS_BAD_PARAM;

    tensorDesc->dataType = dataType;
    // Map Nd dims to our 4d representation (pad leading dims with 1)
    tensorDesc->n = nbDims >= 1 ? dimA[0] : 1;
    tensorDesc->c = nbDims >= 2 ? dimA[1] : 1;
    tensorDesc->h = nbDims >= 3 ? dimA[2] : 1;
    tensorDesc->w = nbDims >= 4 ? dimA[3] : 1;
    tensorDesc->nStride = nbDims >= 1 ? strideA[0] : 1;
    tensorDesc->cStride = nbDims >= 2 ? strideA[1] : 1;
    tensorDesc->hStride = nbDims >= 3 ? strideA[2] : 1;
    tensorDesc->wStride = nbDims >= 4 ? strideA[3] : 1;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          int nbDimsRequested,
                                          cudnnDataType_t* dataType,
                                          int* nbDims,
                                          int dimA[],
                                          int strideA[]) {
    if (!tensorDesc) return CUDNN_STATUS_BAD_PARAM;
    if (dataType) *dataType = tensorDesc->dataType;
    if (nbDims) *nbDims = 4;
    if (dimA && nbDimsRequested >= 1) dimA[0] = tensorDesc->n;
    if (dimA && nbDimsRequested >= 2) dimA[1] = tensorDesc->c;
    if (dimA && nbDimsRequested >= 3) dimA[2] = tensorDesc->h;
    if (dimA && nbDimsRequested >= 4) dimA[3] = tensorDesc->w;
    if (strideA && nbDimsRequested >= 1) strideA[0] = tensorDesc->nStride;
    if (strideA && nbDimsRequested >= 2) strideA[1] = tensorDesc->cStride;
    if (strideA && nbDimsRequested >= 3) strideA[2] = tensorDesc->hStride;
    if (strideA && nbDimsRequested >= 4) strideA[3] = tensorDesc->wStride;
    return CUDNN_STATUS_SUCCESS;
}

// ── Batch normalization forward training ──

cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t /*bnScaleBiasMeanVarDesc*/,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon,
    void* resultSaveMean, void* resultSaveInvVariance) {
    if (!alpha || !beta || !xDesc || !x || !yDesc || !y || !bnScale || !bnBias)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* xf = static_cast<const float*>(x);
    float* yf = static_cast<float*>(y);
    const float* scale = static_cast<const float*>(bnScale);
    const float* bias = static_cast<const float*>(bnBias);
    float* runMean = static_cast<float*>(resultRunningMean);
    float* runVar = static_cast<float*>(resultRunningVariance);
    float* saveMean = static_cast<float*>(resultSaveMean);
    float* saveInvVar = static_cast<float*>(resultSaveInvVariance);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;
    double ema = exponentialAverageFactor;

    if (mode == CUDNN_BATCHNORM_PER_ACTIVATION) {
        int spatial = C * H * W;
        for (int i = 0; i < spatial; ++i) {
            // Compute mean and variance over N
            float mean = 0.0f;
            for (int n = 0; n < N; ++n) mean += xf[n * spatial + i];
            mean /= N;
            float var = 0.0f;
            for (int n = 0; n < N; ++n) {
                float d = xf[n * spatial + i] - mean;
                var += d * d;
            }
            var /= N;
            float inv_std = 1.0f / std::sqrt(var + (float)epsilon);
            if (saveMean) saveMean[i] = mean;
            if (saveInvVar) saveInvVar[i] = inv_std;
            if (runMean) runMean[i] = (float)((1.0 - ema) * runMean[i] + ema * mean);
            if (runVar) {
                float uvar = (N > 1) ? var * N / (N - 1) : var;
                runVar[i] = (float)((1.0 - ema) * runVar[i] + ema * uvar);
            }
            for (int n = 0; n < N; ++n) {
                float norm = (xf[n * spatial + i] - mean) * inv_std;
                yf[n * spatial + i] = a * (scale[i] * norm + bias[i]) + b * yf[n * spatial + i];
            }
        }
    } else { // SPATIAL or SPATIAL_PERSISTENT
        int HW = H * W;
        for (int c = 0; c < C; ++c) {
            float mean = 0.0f;
            for (int n = 0; n < N; ++n)
                for (int hw = 0; hw < HW; ++hw)
                    mean += xf[((n * C + c) * H) * W + hw]; // simplified for contiguous NCHW
            // Proper indexing
            mean = 0.0f;
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w)
                        mean += xf[((n * C + c) * H + h) * W + w];
            mean /= (N * HW);

            float var = 0.0f;
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        float d = xf[((n * C + c) * H + h) * W + w] - mean;
                        var += d * d;
                    }
            var /= (N * HW);

            float inv_std = 1.0f / std::sqrt(var + (float)epsilon);
            if (saveMean) saveMean[c] = mean;
            if (saveInvVar) saveInvVar[c] = inv_std;
            if (runMean) runMean[c] = (float)((1.0 - ema) * runMean[c] + ema * mean);
            if (runVar) {
                float uvar = (N * HW > 1) ? var * (N * HW) / (N * HW - 1) : var;
                runVar[c] = (float)((1.0 - ema) * runVar[c] + ema * uvar);
            }

            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float norm = (xf[idx] - mean) * inv_std;
                        yf[idx] = a * (scale[c] * norm + bias[c]) + b * yf[idx];
                    }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Batch normalization backward ──

cudnnStatus_t cudnnBatchNormalizationBackward(
    cudnnHandle_t /*handle*/,
    cudnnBatchNormMode_t mode,
    const void* alphaDataDiff, const void* betaDataDiff,
    const void* alphaParamDiff, const void* betaParamDiff,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnTensorDescriptor_t dxDesc, void* dx,
    cudnnTensorDescriptor_t /*dBnScaleBiasDesc*/,
    const void* bnScale,
    void* dBnScaleResult, void* dBnBiasResult,
    double epsilon,
    const void* savedMean, const void* savedInvVariance) {
    if (!alphaDataDiff || !betaDataDiff || !alphaParamDiff || !betaParamDiff ||
        !xDesc || !x || !dyDesc || !dy || !dxDesc || !dx || !bnScale)
        return CUDNN_STATUS_BAD_PARAM;

    float aData = *static_cast<const float*>(alphaDataDiff);
    float bData = *static_cast<const float*>(betaDataDiff);
    float aParam = *static_cast<const float*>(alphaParamDiff);
    float bParam = *static_cast<const float*>(betaParamDiff);
    const float* xf = static_cast<const float*>(x);
    const float* dyf = static_cast<const float*>(dy);
    float* dxf = static_cast<float*>(dx);
    const float* scale = static_cast<const float*>(bnScale);
    float* dscale = static_cast<float*>(dBnScaleResult);
    float* dbias = static_cast<float*>(dBnBiasResult);
    const float* sMean = static_cast<const float*>(savedMean);
    const float* sInvVar = static_cast<const float*>(savedInvVariance);

    int N = xDesc->n, C = xDesc->c, H = xDesc->h, W = xDesc->w;

    // Scale dx by betaDataDiff
    int total = N * C * H * W;
    if (bData == 0.0f) std::memset(dxf, 0, total * sizeof(float));
    else if (bData != 1.0f) cblas_sscal(total, bData, dxf, 1);

    if (mode == CUDNN_BATCHNORM_PER_ACTIVATION) {
        // Per-activation: each element of C*H*W is independently normalized over N
        int spatial = C * H * W;
        for (int i = 0; i < spatial; ++i) {
            float mean = sMean ? sMean[i] : 0.0f;
            float inv_var = sInvVar ? sInvVar[i] : 1.0f;

            float ds = 0.0f, db_val = 0.0f;
            for (int n = 0; n < N; ++n) {
                float xhat = (xf[n * spatial + i] - mean) * inv_var;
                ds += dyf[n * spatial + i] * xhat;
                db_val += dyf[n * spatial + i];
            }
            if (dscale) dscale[i] = aParam * ds + bParam * dscale[i];
            if (dbias) dbias[i] = aParam * db_val + bParam * dbias[i];

            for (int n = 0; n < N; ++n) {
                float xhat = (xf[n * spatial + i] - mean) * inv_var;
                float grad = scale[i] * inv_var * (dyf[n * spatial + i] - (db_val + xhat * ds) / N);
                dxf[n * spatial + i] += aData * grad;
            }
        }
    } else { // SPATIAL
        int HW = H * W;
        int M = N * HW;
        for (int c = 0; c < C; ++c) {
            float mean = sMean ? sMean[c] : 0.0f;
            float inv_var = sInvVar ? sInvVar[c] : 1.0f;

            float ds = 0.0f, db_val = 0.0f;
            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float xhat = (xf[idx] - mean) * inv_var;
                        ds += dyf[idx] * xhat;
                        db_val += dyf[idx];
                    }
            if (dscale) dscale[c] = aParam * ds + bParam * dscale[c];
            if (dbias) dbias[c] = aParam * db_val + bParam * dbias[c];

            for (int n = 0; n < N; ++n)
                for (int h = 0; h < H; ++h)
                    for (int w = 0; w < W; ++w) {
                        int idx = ((n * C + c) * H + h) * W + w;
                        float xhat = (xf[idx] - mean) * inv_var;
                        float grad = scale[c] * inv_var * (dyf[idx] - (db_val + xhat * ds) / M);
                        dxf[idx] += aData * grad;
                    }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── Softmax backward ──

cudnnStatus_t cudnnSoftmaxBackward(cudnnHandle_t /*handle*/,
                                    cudnnSoftmaxAlgorithm_t algo,
                                    cudnnSoftmaxMode_t mode,
                                    const void* alpha,
                                    cudnnTensorDescriptor_t yDesc, const void* y,
                                    cudnnTensorDescriptor_t dyDesc, const void* dy,
                                    const void* beta,
                                    cudnnTensorDescriptor_t dxDesc, void* dx) {
    if (!alpha || !yDesc || !y || !dyDesc || !dy || !beta || !dxDesc || !dx)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* yf = static_cast<const float*>(y);
    const float* dyf = static_cast<const float*>(dy);
    float* dxf = static_cast<float*>(dx);

    int N = yDesc->n, C = yDesc->c, H = yDesc->h, W = yDesc->w;

    if (mode == CUDNN_SOFTMAX_MODE_INSTANCE) {
        int spatial = C * H * W;
        for (int n = 0; n < N; ++n) {
            const float* y_n = yf + n * spatial;
            const float* dy_n = dyf + n * spatial;
            float* dx_n = dxf + n * spatial;
            if (algo == CUDNN_SOFTMAX_LOG) {
                float sum_dy = 0.0f;
                for (int i = 0; i < spatial; ++i) sum_dy += dy_n[i];
                for (int i = 0; i < spatial; ++i)
                    dx_n[i] = a * (dy_n[i] - std::exp(y_n[i]) * sum_dy) + b * dx_n[i];
            } else {
                float dot = 0.0f;
                for (int i = 0; i < spatial; ++i) dot += y_n[i] * dy_n[i];
                for (int i = 0; i < spatial; ++i)
                    dx_n[i] = a * y_n[i] * (dy_n[i] - dot) + b * dx_n[i];
            }
        }
    } else { // MODE_CHANNEL
        for (int n = 0; n < N; ++n) {
            for (int h = 0; h < H; ++h) {
                for (int w = 0; w < W; ++w) {
                    if (algo == CUDNN_SOFTMAX_LOG) {
                        float sum_dy = 0.0f;
                        for (int c = 0; c < C; ++c)
                            sum_dy += dyf[((n * C + c) * H + h) * W + w];
                        for (int c = 0; c < C; ++c) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            dxf[idx] = a * (dyf[idx] - std::exp(yf[idx]) * sum_dy) + b * dxf[idx];
                        }
                    } else {
                        float dot = 0.0f;
                        for (int c = 0; c < C; ++c) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            dot += yf[idx] * dyf[idx];
                        }
                        for (int c = 0; c < C; ++c) {
                            int idx = ((n * C + c) * H + h) * W + w;
                            dxf[idx] = a * yf[idx] * (dyf[idx] - dot) + b * dxf[idx];
                        }
                    }
                }
            }
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── OpTensor ──

cudnnStatus_t cudnnCreateOpTensorDescriptor(cudnnOpTensorDescriptor_t* opTensorDesc) {
    if (!opTensorDesc) return CUDNN_STATUS_BAD_PARAM;
    *opTensorDesc = new (std::nothrow) cudnnOpTensorStruct;
    return *opTensorDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc) {
    delete opTensorDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetOpTensorDescriptor(cudnnOpTensorDescriptor_t opTensorDesc,
                                          cudnnOpTensorOp_t opTensorOp,
                                          cudnnDataType_t opTensorCompType,
                                          cudnnNanPropagation_t opTensorNanOpt) {
    if (!opTensorDesc) return CUDNN_STATUS_BAD_PARAM;
    opTensorDesc->op = opTensorOp;
    opTensorDesc->compType = opTensorCompType;
    opTensorDesc->nanOpt = opTensorNanOpt;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnOpTensor(cudnnHandle_t /*handle*/,
                             cudnnOpTensorDescriptor_t opTensorDesc,
                             const void* alpha1,
                             cudnnTensorDescriptor_t aDesc, const void* A,
                             const void* alpha2,
                             cudnnTensorDescriptor_t bDesc, const void* B,
                             const void* beta,
                             cudnnTensorDescriptor_t cDesc, void* C) {
    if (!opTensorDesc || !alpha1 || !aDesc || !A || !alpha2 || !bDesc || !B ||
        !beta || !cDesc || !C)
        return CUDNN_STATUS_BAD_PARAM;

    float a1 = *static_cast<const float*>(alpha1);
    float a2 = *static_cast<const float*>(alpha2);
    float b = *static_cast<const float*>(beta);
    const float* af = static_cast<const float*>(A);
    const float* bf = static_cast<const float*>(B);
    float* cf = static_cast<float*>(C);

    int c_count = cDesc->n * cDesc->c * cDesc->h * cDesc->w;
    int a_count = aDesc->n * aDesc->c * aDesc->h * aDesc->w;
    int b_count = bDesc->n * bDesc->c * bDesc->h * bDesc->w;

    for (int i = 0; i < c_count; ++i) {
        float va = af[a_count == c_count ? i : (i % a_count)];
        float vb = bf[b_count == c_count ? i : (i % b_count)];
        float result = 0.0f;
        switch (opTensorDesc->op) {
            case CUDNN_OP_TENSOR_ADD: result = a1 * va + a2 * vb; break;
            case CUDNN_OP_TENSOR_MUL: result = a1 * va * a2 * vb; break;
            case CUDNN_OP_TENSOR_MIN: result = std::min(a1 * va, a2 * vb); break;
            case CUDNN_OP_TENSOR_MAX: result = std::max(a1 * va, a2 * vb); break;
            case CUDNN_OP_TENSOR_SQRT: result = a1 * std::sqrt(va); break;
            case CUDNN_OP_TENSOR_NOT: result = (va == 0.0f) ? 1.0f : 0.0f; break;
            default: result = a1 * va + a2 * vb; break;
        }
        cf[i] = result + b * cf[i];
    }
    return CUDNN_STATUS_SUCCESS;
}

// ── ReduceTensor ──

cudnnStatus_t cudnnCreateReduceTensorDescriptor(cudnnReduceTensorDescriptor_t* reduceTensorDesc) {
    if (!reduceTensorDesc) return CUDNN_STATUS_BAD_PARAM;
    *reduceTensorDesc = new (std::nothrow) cudnnReduceTensorStruct;
    return *reduceTensorDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc) {
    delete reduceTensorDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetReduceTensorDescriptor(cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                              cudnnReduceTensorOp_t reduceTensorOp,
                                              cudnnDataType_t reduceTensorCompType,
                                              cudnnNanPropagation_t reduceTensorNanOpt,
                                              cudnnReduceTensorIndices_t reduceTensorIndices,
                                              cudnnIndicesType_t reduceTensorIndicesType) {
    if (!reduceTensorDesc) return CUDNN_STATUS_BAD_PARAM;
    reduceTensorDesc->op = reduceTensorOp;
    reduceTensorDesc->compType = reduceTensorCompType;
    reduceTensorDesc->nanOpt = reduceTensorNanOpt;
    reduceTensorDesc->indices = reduceTensorIndices;
    reduceTensorDesc->indicesType = reduceTensorIndicesType;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetReductionWorkspaceSize(cudnnHandle_t /*handle*/,
                                              cudnnReduceTensorDescriptor_t /*reduceTensorDesc*/,
                                              cudnnTensorDescriptor_t aDesc,
                                              cudnnTensorDescriptor_t /*cDesc*/,
                                              size_t* sizeInBytes) {
    if (!sizeInBytes || !aDesc) return CUDNN_STATUS_BAD_PARAM;
    *sizeInBytes = 0; // CPU reduction needs no extra workspace
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnReduceTensor(cudnnHandle_t /*handle*/,
                                 cudnnReduceTensorDescriptor_t reduceTensorDesc,
                                 void* /*indices*/, size_t /*indicesSizeInBytes*/,
                                 void* /*workspace*/, size_t /*workspaceSizeInBytes*/,
                                 const void* alpha,
                                 cudnnTensorDescriptor_t aDesc, const void* A,
                                 const void* beta,
                                 cudnnTensorDescriptor_t cDesc, void* C) {
    if (!reduceTensorDesc || !alpha || !aDesc || !A || !beta || !cDesc || !C)
        return CUDNN_STATUS_BAD_PARAM;

    float a = *static_cast<const float*>(alpha);
    float b = *static_cast<const float*>(beta);
    const float* af = static_cast<const float*>(A);
    float* cf = static_cast<float*>(C);

    int a_total = aDesc->n * aDesc->c * aDesc->h * aDesc->w;
    int c_total = cDesc->n * cDesc->c * cDesc->h * cDesc->w;

    // Simple full reduction: reduce all of A into each element of C
    // For channel-wise reduction: reduce over N,H,W per channel
    if (c_total == 1) {
        // Full reduction to scalar
        float result = 0.0f;
        switch (reduceTensorDesc->op) {
            case CUDNN_REDUCE_TENSOR_ADD:
            case CUDNN_REDUCE_TENSOR_AVG:
                for (int i = 0; i < a_total; ++i) result += af[i];
                if (reduceTensorDesc->op == CUDNN_REDUCE_TENSOR_AVG) result /= a_total;
                break;
            case CUDNN_REDUCE_TENSOR_MUL:
            case CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS:
                result = 1.0f;
                for (int i = 0; i < a_total; ++i) {
                    if (reduceTensorDesc->op == CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS && af[i] == 0.0f) continue;
                    result *= af[i];
                }
                break;
            case CUDNN_REDUCE_TENSOR_MIN:
                result = af[0];
                for (int i = 1; i < a_total; ++i) result = std::min(result, af[i]);
                break;
            case CUDNN_REDUCE_TENSOR_MAX:
                result = af[0];
                for (int i = 1; i < a_total; ++i) result = std::max(result, af[i]);
                break;
            case CUDNN_REDUCE_TENSOR_AMAX:
                result = std::fabs(af[0]);
                for (int i = 1; i < a_total; ++i) result = std::max(result, std::fabs(af[i]));
                break;
            case CUDNN_REDUCE_TENSOR_NORM1:
                for (int i = 0; i < a_total; ++i) result += std::fabs(af[i]);
                break;
            case CUDNN_REDUCE_TENSOR_NORM2:
                for (int i = 0; i < a_total; ++i) result += af[i] * af[i];
                result = std::sqrt(result);
                break;
            default: break;
        }
        cf[0] = a * result + b * cf[0];
    } else {
        // Per-channel reduction: c_total == C, reduce over N,H,W
        int N = aDesc->n, aC = aDesc->c, H = aDesc->h, W = aDesc->w;
        int HW = H * W;
        for (int c = 0; c < aC && c < c_total; ++c) {
            float result = 0.0f;
            bool first = true;
            for (int n = 0; n < N; ++n) {
                for (int hw = 0; hw < HW; ++hw) {
                    float v = af[((n * aC + c) * H) * W + hw];
                    // Fix indexing for proper NCHW layout
                    int h = hw / W, w = hw % W;
                    v = af[((n * aC + c) * H + h) * W + w];
                    switch (reduceTensorDesc->op) {
                        case CUDNN_REDUCE_TENSOR_ADD:
                        case CUDNN_REDUCE_TENSOR_AVG:
                            result += v; break;
                        case CUDNN_REDUCE_TENSOR_MIN:
                            result = first ? v : std::min(result, v); break;
                        case CUDNN_REDUCE_TENSOR_MAX:
                            result = first ? v : std::max(result, v); break;
                        case CUDNN_REDUCE_TENSOR_AMAX:
                            result = first ? std::fabs(v) : std::max(result, std::fabs(v)); break;
                        case CUDNN_REDUCE_TENSOR_NORM1:
                            result += std::fabs(v); break;
                        case CUDNN_REDUCE_TENSOR_NORM2:
                            result += v * v; break;
                        default: result += v; break;
                    }
                    first = false;
                }
            }
            if (reduceTensorDesc->op == CUDNN_REDUCE_TENSOR_AVG) result /= (N * HW);
            if (reduceTensorDesc->op == CUDNN_REDUCE_TENSOR_NORM2) result = std::sqrt(result);
            cf[c] = a * result + b * cf[c];
        }
    }
    return CUDNN_STATUS_SUCCESS;
}

// --- RNN ---

cudnnStatus_t cudnnCreateRNNDescriptor(cudnnRNNDescriptor_t* rnnDesc) {
    if (!rnnDesc) return CUDNN_STATUS_BAD_PARAM;
    *rnnDesc = new (std::nothrow) cudnnRNNStruct;
    return *rnnDesc ? CUDNN_STATUS_SUCCESS : CUDNN_STATUS_ALLOC_FAILED;
}

cudnnStatus_t cudnnDestroyRNNDescriptor(cudnnRNNDescriptor_t rnnDesc) {
    delete rnnDesc;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnSetRNNDescriptor_v6(cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
                                        int hiddenSize, int numLayers,
                                        cudnnDropoutDescriptor_t dropoutDesc,
                                        cudnnRNNInputMode_t inputMode,
                                        cudnnDirectionMode_t direction,
                                        cudnnRNNMode_t cellMode,
                                        cudnnRNNAlgo_t algo,
                                        cudnnDataType_t mathPrec) {
    if (!rnnDesc) return CUDNN_STATUS_BAD_PARAM;
    rnnDesc->hiddenSize = hiddenSize;
    rnnDesc->numLayers = numLayers;
    rnnDesc->dropoutDesc = dropoutDesc;
    rnnDesc->inputMode = inputMode;
    rnnDesc->direction = direction;
    rnnDesc->cellMode = cellMode;
    rnnDesc->algo = algo;
    rnnDesc->mathPrec = mathPrec;
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNParamsSize(cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
                                     cudnnTensorDescriptor_t xDesc,
                                     size_t* sizeInBytes, cudnnDataType_t) {
    if (!rnnDesc || !xDesc || !sizeInBytes) return CUDNN_STATUS_BAD_PARAM;
    int H = rnnDesc->hiddenSize;
    int inputSize = (xDesc->h == 1 && xDesc->w == 1) ? xDesc->c :
                    (xDesc->w > 0 ? xDesc->w : xDesc->c);
    int numDirs = (rnnDesc->direction == CUDNN_BIDIRECTIONAL) ? 2 : 1;
    int numLayers = rnnDesc->numLayers;
    int gateCount = 1;
    switch (rnnDesc->cellMode) {
        case CUDNN_LSTM: gateCount = 4; break;
        case CUDNN_GRU:  gateCount = 3; break;
        default:         gateCount = 1; break;
    }
    // Layer 0: W_ih(gateCount*H, inputSize) + W_hh(gateCount*H, H) + biases
    // Layer L>0: W_ih(gateCount*H, H*numDirs) + W_hh(gateCount*H, H) + biases
    size_t params = 0;
    for (int l = 0; l < numLayers; l++) {
        for (int d = 0; d < numDirs; d++) {
            int inSz = (l == 0) ? inputSize : H * numDirs;
            params += (size_t)gateCount * H * inSz;  // W_ih
            params += (size_t)gateCount * H * H;      // W_hh
            params += (size_t)gateCount * H * 2;       // bias_ih + bias_hh
        }
    }
    *sizeInBytes = params * sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNWorkspaceSize(cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
                                        int seqLength, const cudnnTensorDescriptor_t*,
                                        size_t* sizeInBytes) {
    if (!rnnDesc || !sizeInBytes) return CUDNN_STATUS_BAD_PARAM;
    int H = rnnDesc->hiddenSize;
    int numDirs = (rnnDesc->direction == CUDNN_BIDIRECTIONAL) ? 2 : 1;
    int gateCount = (rnnDesc->cellMode == CUDNN_LSTM) ? 4 :
                    (rnnDesc->cellMode == CUDNN_GRU) ? 3 : 1;
    // Workspace for intermediate gate activations
    *sizeInBytes = (size_t)seqLength * numDirs * gateCount * H * sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnGetRNNTrainingReserveSize(cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
                                              int seqLength, const cudnnTensorDescriptor_t*,
                                              size_t* sizeInBytes) {
    if (!rnnDesc || !sizeInBytes) return CUDNN_STATUS_BAD_PARAM;
    int H = rnnDesc->hiddenSize;
    int numDirs = (rnnDesc->direction == CUDNN_BIDIRECTIONAL) ? 2 : 1;
    int gateCount = (rnnDesc->cellMode == CUDNN_LSTM) ? 4 :
                    (rnnDesc->cellMode == CUDNN_GRU) ? 3 : 1;
    // Reserve space stores pre-activation gate values + hidden states for backward
    *sizeInBytes = (size_t)seqLength * rnnDesc->numLayers * numDirs * (gateCount + 1) * H * sizeof(float);
    return CUDNN_STATUS_SUCCESS;
}

namespace {

// Simple GEMM: C = alpha*A*B + beta*C  (A: MxK, B: KxN, C: MxN, row-major)
static void rnn_gemm(int M, int N, int K, float alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     float beta, float* C, int ldc) {
    // Use Accelerate cblas in col-major form: interpret row-major A*B as col-major B^T * A^T
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

static inline float rnn_sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
static inline float rnn_tanh(float x) { return std::tanh(x); }

// Run one LSTM step: given input x[inputSize], prev h[H], prev c[H],
// compute new h[H] and c[H].
// W_ih: (4H, inputSize), W_hh: (4H, H), b_ih: (4H), b_hh: (4H)
static void lstm_step(int inputSize, int H,
                      const float* x, const float* h_prev, const float* c_prev,
                      const float* W_ih, const float* W_hh,
                      const float* b_ih, const float* b_hh,
                      float* h_out, float* c_out, float* gates_buf) {
    // gates = W_ih * x + b_ih + W_hh * h_prev + b_hh
    int gateSize = 4 * H;
    for (int i = 0; i < gateSize; i++) gates_buf[i] = b_ih[i] + b_hh[i];
    rnn_gemm(gateSize, 1, inputSize, 1.0f, W_ih, inputSize, x, 1, 1.0f, gates_buf, 1);
    rnn_gemm(gateSize, 1, H, 1.0f, W_hh, H, h_prev, 1, 1.0f, gates_buf, 1);

    // Split into i, f, g, o gates
    const float* gi = gates_buf;
    const float* gf = gates_buf + H;
    const float* gg = gates_buf + 2 * H;
    const float* go = gates_buf + 3 * H;

    for (int j = 0; j < H; j++) {
        float i_val = rnn_sigmoid(gi[j]);
        float f_val = rnn_sigmoid(gf[j]);
        float g_val = rnn_tanh(gg[j]);
        float o_val = rnn_sigmoid(go[j]);
        c_out[j] = f_val * c_prev[j] + i_val * g_val;
        h_out[j] = o_val * rnn_tanh(c_out[j]);
    }
}

// Run one vanilla RNN step (RELU or TANH)
static void rnn_step(int inputSize, int H, cudnnRNNMode_t mode,
                     const float* x, const float* h_prev,
                     const float* W_ih, const float* W_hh,
                     const float* b_ih, const float* b_hh,
                     float* h_out, float* gates_buf) {
    for (int i = 0; i < H; i++) gates_buf[i] = b_ih[i] + b_hh[i];
    rnn_gemm(H, 1, inputSize, 1.0f, W_ih, inputSize, x, 1, 1.0f, gates_buf, 1);
    rnn_gemm(H, 1, H, 1.0f, W_hh, H, h_prev, 1, 1.0f, gates_buf, 1);
    for (int j = 0; j < H; j++) {
        h_out[j] = (mode == CUDNN_RNN_RELU) ? std::max(0.0f, gates_buf[j]) : rnn_tanh(gates_buf[j]);
    }
}

// Run one GRU step
static void gru_step(int inputSize, int H,
                     const float* x, const float* h_prev,
                     const float* W_ih, const float* W_hh,
                     const float* b_ih, const float* b_hh,
                     float* h_out, float* gates_buf) {
    // gates_ih = W_ih * x + b_ih  (3H)
    // gates_hh = W_hh * h + b_hh  (3H)
    int gateSize = 3 * H;
    float* g_ih = gates_buf;
    float* g_hh = gates_buf + gateSize;
    for (int i = 0; i < gateSize; i++) { g_ih[i] = b_ih[i]; g_hh[i] = b_hh[i]; }
    rnn_gemm(gateSize, 1, inputSize, 1.0f, W_ih, inputSize, x, 1, 1.0f, g_ih, 1);
    rnn_gemm(gateSize, 1, H, 1.0f, W_hh, H, h_prev, 1, 1.0f, g_hh, 1);

    // r = sigmoid(g_ih[0:H] + g_hh[0:H])
    // z = sigmoid(g_ih[H:2H] + g_hh[H:2H])
    // n = tanh(g_ih[2H:3H] + r * g_hh[2H:3H])
    // h = (1 - z) * n + z * h_prev
    for (int j = 0; j < H; j++) {
        float r = rnn_sigmoid(g_ih[j] + g_hh[j]);
        float z = rnn_sigmoid(g_ih[H + j] + g_hh[H + j]);
        float n = rnn_tanh(g_ih[2 * H + j] + r * g_hh[2 * H + j]);
        h_out[j] = (1.0f - z) * n + z * h_prev[j];
    }
}

} // anonymous namespace

// Core RNN forward implementation (shared by inference and training)
static cudnnStatus_t rnn_forward_impl(cudnnHandle_t, cudnnRNNDescriptor_t rnnDesc,
                                       int seqLength, const cudnnTensorDescriptor_t* xDesc,
                                       const float* x, const float* hx, const float* cx,
                                       const float* w, float* y, float* hy, float* cy) {
    if (!rnnDesc || !xDesc || !x || !w || !y) return CUDNN_STATUS_BAD_PARAM;

    int H = rnnDesc->hiddenSize;
    int numLayers = rnnDesc->numLayers;
    int numDirs = (rnnDesc->direction == CUDNN_BIDIRECTIONAL) ? 2 : 1;
    int batchSize = xDesc[0] ? xDesc[0]->n : 0;
    int inputSize = xDesc[0] ? (xDesc[0]->w > 0 ? xDesc[0]->w : xDesc[0]->c) : 0;
    if (batchSize <= 0 || inputSize <= 0) return CUDNN_STATUS_BAD_PARAM;

    int gateCount = 1;
    switch (rnnDesc->cellMode) {
        case CUDNN_LSTM: gateCount = 4; break;
        case CUDNN_GRU:  gateCount = 3; break;
        default:         gateCount = 1; break;
    }

    // Temp buffers
    int maxGateBuf = gateCount * H * 2; // enough for GRU's double-buffer
    std::vector<float> gates_buf(maxGateBuf);
    std::vector<float> h_cur(numLayers * numDirs * batchSize * H, 0.0f);
    std::vector<float> h_next(numLayers * numDirs * batchSize * H, 0.0f);
    std::vector<float> c_cur, c_next;
    if (rnnDesc->cellMode == CUDNN_LSTM) {
        c_cur.resize(numLayers * numDirs * batchSize * H, 0.0f);
        c_next.resize(numLayers * numDirs * batchSize * H, 0.0f);
    }

    // Initialize hidden state from hx
    if (hx) {
        std::memcpy(h_cur.data(), hx, numLayers * numDirs * batchSize * H * sizeof(float));
    }
    if (cx && rnnDesc->cellMode == CUDNN_LSTM) {
        std::memcpy(c_cur.data(), cx, numLayers * numDirs * batchSize * H * sizeof(float));
    }

    // Input buffer: starts as x, then becomes y from previous layer
    // y layout: (seqLength, batchSize, H * numDirs)
    int yFeatureSize = H * numDirs;
    std::vector<float> layer_input(seqLength * batchSize * std::max(inputSize, yFeatureSize));
    std::memcpy(layer_input.data(), x, seqLength * batchSize * inputSize * sizeof(float));

    const float* wPtr = w;

    for (int layer = 0; layer < numLayers; layer++) {
        for (int dir = 0; dir < numDirs; dir++) {
            int dirIdx = layer * numDirs + dir;
            int inSz = (layer == 0) ? inputSize : yFeatureSize;
            size_t wih_sz = (size_t)gateCount * H * inSz;
            size_t whh_sz = (size_t)gateCount * H * H;
            size_t bih_sz = (size_t)gateCount * H;
            size_t bhh_sz = (size_t)gateCount * H;

            const float* W_ih = wPtr; wPtr += wih_sz;
            const float* W_hh = wPtr; wPtr += whh_sz;
            const float* b_ih = wPtr; wPtr += bih_sz;
            const float* b_hh = wPtr; wPtr += bhh_sz;

            // Process sequence
            for (int t_raw = 0; t_raw < seqLength; t_raw++) {
                int t = (dir == 0) ? t_raw : (seqLength - 1 - t_raw);
                for (int b = 0; b < batchSize; b++) {
                    const float* inp = layer_input.data() + t * batchSize * inSz + b * inSz;
                    float* h_prev = h_cur.data() + dirIdx * batchSize * H + b * H;
                    float* h_out = h_next.data() + dirIdx * batchSize * H + b * H;

                    switch (rnnDesc->cellMode) {
                        case CUDNN_LSTM: {
                            float* c_p = c_cur.data() + dirIdx * batchSize * H + b * H;
                            float* c_o = c_next.data() + dirIdx * batchSize * H + b * H;
                            lstm_step(inSz, H, inp, h_prev, c_p, W_ih, W_hh, b_ih, b_hh, h_out, c_o, gates_buf.data());
                            std::memcpy(c_p, c_o, H * sizeof(float));
                            break;
                        }
                        case CUDNN_GRU:
                            gru_step(inSz, H, inp, h_prev, W_ih, W_hh, b_ih, b_hh, h_out, gates_buf.data());
                            break;
                        default:
                            rnn_step(inSz, H, rnnDesc->cellMode, inp, h_prev, W_ih, W_hh, b_ih, b_hh, h_out, gates_buf.data());
                            break;
                    }

                    // Write to output y at position (t, b, dir*H)
                    float* yOut = y + t * batchSize * yFeatureSize + b * yFeatureSize + dir * H;
                    std::memcpy(yOut, h_out, H * sizeof(float));

                    // Update h_cur for next timestep
                    std::memcpy(h_prev, h_out, H * sizeof(float));
                }
            }
        }

        // Prepare input for next layer: copy from y
        if (layer < numLayers - 1) {
            std::memcpy(layer_input.data(), y, seqLength * batchSize * yFeatureSize * sizeof(float));
        }
    }

    // Copy final hidden states
    if (hy) {
        std::memcpy(hy, h_cur.data(), numLayers * numDirs * batchSize * H * sizeof(float));
    }
    if (cy && rnnDesc->cellMode == CUDNN_LSTM) {
        std::memcpy(cy, c_cur.data(), numLayers * numDirs * batchSize * H * sizeof(float));
    }

    return CUDNN_STATUS_SUCCESS;
}

cudnnStatus_t cudnnRNNForwardInference(cudnnHandle_t handle,
                                        cudnnRNNDescriptor_t rnnDesc,
                                        int seqLength,
                                        const cudnnTensorDescriptor_t* xDesc,
                                        const void* x,
                                        cudnnTensorDescriptor_t, const void* hx,
                                        cudnnTensorDescriptor_t, const void* cx,
                                        cudnnFilterDescriptor_t, const void* w,
                                        const cudnnTensorDescriptor_t* yDesc, void* y,
                                        cudnnTensorDescriptor_t, void* hy,
                                        cudnnTensorDescriptor_t, void* cy,
                                        void*, size_t) {
    (void)yDesc;
    return rnn_forward_impl(handle, rnnDesc, seqLength, xDesc,
                            (const float*)x, (const float*)hx, (const float*)cx,
                            (const float*)w, (float*)y, (float*)hy, (float*)cy);
}

cudnnStatus_t cudnnRNNForwardTraining(cudnnHandle_t handle,
                                       cudnnRNNDescriptor_t rnnDesc,
                                       int seqLength,
                                       const cudnnTensorDescriptor_t* xDesc,
                                       const void* x,
                                       cudnnTensorDescriptor_t, const void* hx,
                                       cudnnTensorDescriptor_t, const void* cx,
                                       cudnnFilterDescriptor_t, const void* w,
                                       const cudnnTensorDescriptor_t* yDesc, void* y,
                                       cudnnTensorDescriptor_t, void* hy,
                                       cudnnTensorDescriptor_t, void* cy,
                                       void*, size_t,
                                       void*, size_t) {
    (void)yDesc;
    // Training forward is identical to inference for outputs; reserveSpace is for backward
    return rnn_forward_impl(handle, rnnDesc, seqLength, xDesc,
                            (const float*)x, (const float*)hx, (const float*)cx,
                            (const float*)w, (float*)y, (float*)hy, (float*)cy);
}

} // extern "C"
