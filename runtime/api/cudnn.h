#pragma once

#include "cuda_runtime.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cudnnContext* cudnnHandle_t;
typedef struct cudnnTensorStruct* cudnnTensorDescriptor_t;
typedef struct cudnnFilterStruct* cudnnFilterDescriptor_t;
typedef struct cudnnConvolutionStruct* cudnnConvolutionDescriptor_t;
typedef struct cudnnActivationStruct* cudnnActivationDescriptor_t;
typedef struct cudnnPoolingStruct* cudnnPoolingDescriptor_t;
typedef struct cudnnDropoutStruct* cudnnDropoutDescriptor_t;

typedef enum cudnnStatus_t {
    CUDNN_STATUS_SUCCESS = 0,
    CUDNN_STATUS_NOT_INITIALIZED = 1,
    CUDNN_STATUS_ALLOC_FAILED = 2,
    CUDNN_STATUS_BAD_PARAM = 3,
    CUDNN_STATUS_INTERNAL_ERROR = 4,
    CUDNN_STATUS_INVALID_VALUE = 5,
    CUDNN_STATUS_ARCH_MISMATCH = 6,
    CUDNN_STATUS_MAPPING_ERROR = 7,
    CUDNN_STATUS_EXECUTION_FAILED = 8,
    CUDNN_STATUS_NOT_SUPPORTED = 9,
} cudnnStatus_t;

typedef enum cudnnDataType_t {
    CUDNN_DATA_FLOAT = 0,
    CUDNN_DATA_DOUBLE = 1,
    CUDNN_DATA_HALF = 2,
    CUDNN_DATA_INT8 = 3,
    CUDNN_DATA_INT32 = 4,
    CUDNN_DATA_INT8x4 = 5,
    CUDNN_DATA_UINT8 = 6,
    CUDNN_DATA_UINT8x4 = 7,
    CUDNN_DATA_INT8x32 = 8,
    CUDNN_DATA_BFLOAT16 = 9,
    CUDNN_DATA_INT64 = 10,
} cudnnDataType_t;

typedef enum cudnnTensorFormat_t {
    CUDNN_TENSOR_NCHW = 0,
    CUDNN_TENSOR_NHWC = 1,
    CUDNN_TENSOR_NCHW_VECT_C = 2,
} cudnnTensorFormat_t;

typedef enum cudnnConvolutionMode_t {
    CUDNN_CONVOLUTION = 0,
    CUDNN_CROSS_CORRELATION = 1,
} cudnnConvolutionMode_t;

typedef enum cudnnConvolutionFwdAlgo_t {
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM = 0,
    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM = 1,
    CUDNN_CONVOLUTION_FWD_ALGO_GEMM = 2,
    CUDNN_CONVOLUTION_FWD_ALGO_DIRECT = 3,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT = 4,
    CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING = 5,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD = 6,
    CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED = 7,
} cudnnConvolutionFwdAlgo_t;

typedef enum cudnnConvolutionFwdPreference_t {
    CUDNN_CONVOLUTION_FWD_NO_WORKSPACE = 0,
    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST = 1,
    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT = 2,
} cudnnConvolutionFwdPreference_t;

typedef enum cudnnActivationMode_t {
    CUDNN_ACTIVATION_SIGMOID = 0,
    CUDNN_ACTIVATION_RELU = 1,
    CUDNN_ACTIVATION_TANH = 2,
    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
    CUDNN_ACTIVATION_ELU = 4,
    CUDNN_ACTIVATION_IDENTITY = 5,
    CUDNN_ACTIVATION_SWISH = 6,
} cudnnActivationMode_t;

typedef enum cudnnNanPropagation_t {
    CUDNN_NOT_PROPAGATE_NAN = 0,
    CUDNN_PROPAGATE_NAN = 1,
} cudnnNanPropagation_t;

typedef enum cudnnMathType_t {
    CUDNN_DEFAULT_MATH = 0,
    CUDNN_TENSOR_OP_MATH = 1,
    CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION = 2,
    CUDNN_FMA_MATH = 3,
} cudnnMathType_t;

typedef struct cudnnConvolutionFwdAlgoPerf_t {
    cudnnConvolutionFwdAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionFwdAlgoPerf_t;

// Library version
size_t cudnnGetVersion(void);
const char* cudnnGetErrorString(cudnnStatus_t status);

// Handle
cudnnStatus_t cudnnCreate(cudnnHandle_t* handle);
cudnnStatus_t cudnnDestroy(cudnnHandle_t handle);
cudnnStatus_t cudnnSetStream(cudnnHandle_t handle, cudaStream_t stream);
cudnnStatus_t cudnnGetStream(cudnnHandle_t handle, cudaStream_t* stream);

// Tensor descriptor
cudnnStatus_t cudnnCreateTensorDescriptor(cudnnTensorDescriptor_t* tensorDesc);
cudnnStatus_t cudnnDestroyTensorDescriptor(cudnnTensorDescriptor_t tensorDesc);
cudnnStatus_t cudnnSetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnTensorFormat_t format,
                                          cudnnDataType_t dataType,
                                          int n, int c, int h, int w);
cudnnStatus_t cudnnGetTensor4dDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnDataType_t* dataType,
                                          int* n, int* c, int* h, int* w,
                                          int* nStride, int* cStride,
                                          int* hStride, int* wStride);

// Filter descriptor
cudnnStatus_t cudnnCreateFilterDescriptor(cudnnFilterDescriptor_t* filterDesc);
cudnnStatus_t cudnnDestroyFilterDescriptor(cudnnFilterDescriptor_t filterDesc);
cudnnStatus_t cudnnSetFilter4dDescriptor(cudnnFilterDescriptor_t filterDesc,
                                          cudnnDataType_t dataType,
                                          cudnnTensorFormat_t format,
                                          int k, int c, int h, int w);

// Convolution descriptor
cudnnStatus_t cudnnCreateConvolutionDescriptor(cudnnConvolutionDescriptor_t* convDesc);
cudnnStatus_t cudnnDestroyConvolutionDescriptor(cudnnConvolutionDescriptor_t convDesc);
cudnnStatus_t cudnnSetConvolution2dDescriptor(cudnnConvolutionDescriptor_t convDesc,
                                               int pad_h, int pad_w,
                                               int u, int v,
                                               int dilation_h, int dilation_w,
                                               cudnnConvolutionMode_t mode,
                                               cudnnDataType_t computeType);
cudnnStatus_t cudnnSetConvolutionMathType(cudnnConvolutionDescriptor_t convDesc,
                                           cudnnMathType_t mathType);
cudnnStatus_t cudnnSetConvolutionGroupCount(cudnnConvolutionDescriptor_t convDesc,
                                             int groupCount);
cudnnStatus_t cudnnGetConvolution2dForwardOutputDim(cudnnConvolutionDescriptor_t convDesc,
                                                     cudnnTensorDescriptor_t inputTensorDesc,
                                                     cudnnFilterDescriptor_t filterDesc,
                                                     int* n, int* c, int* h, int* w);

// Forward convolution
cudnnStatus_t cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle_t handle,
                                                       cudnnTensorDescriptor_t xDesc,
                                                       cudnnFilterDescriptor_t wDesc,
                                                       cudnnConvolutionDescriptor_t convDesc,
                                                       cudnnTensorDescriptor_t yDesc,
                                                       cudnnConvolutionFwdAlgo_t algo,
                                                       size_t* sizeInBytes);

cudnnStatus_t cudnnFindConvolutionForwardAlgorithm(cudnnHandle_t handle,
                                                    cudnnTensorDescriptor_t xDesc,
                                                    cudnnFilterDescriptor_t wDesc,
                                                    cudnnConvolutionDescriptor_t convDesc,
                                                    cudnnTensorDescriptor_t yDesc,
                                                    int requestedAlgoCount,
                                                    int* returnedAlgoCount,
                                                    cudnnConvolutionFwdAlgoPerf_t* perfResults);

cudnnStatus_t cudnnConvolutionForward(cudnnHandle_t handle,
                                       const void* alpha,
                                       cudnnTensorDescriptor_t xDesc, const void* x,
                                       cudnnFilterDescriptor_t wDesc, const void* w,
                                       cudnnConvolutionDescriptor_t convDesc,
                                       cudnnConvolutionFwdAlgo_t algo,
                                       void* workSpace, size_t workSpaceSizeInBytes,
                                       const void* beta,
                                       cudnnTensorDescriptor_t yDesc, void* y);

// Backward convolution
typedef enum cudnnConvolutionBwdDataAlgo_t {
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING = 3,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED = 5,
} cudnnConvolutionBwdDataAlgo_t;

typedef enum cudnnConvolutionBwdFilterAlgo_t {
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0 = 0,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1 = 1,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT = 2,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3 = 3,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD = 4,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED = 5,
    CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING = 6,
} cudnnConvolutionBwdFilterAlgo_t;

typedef struct cudnnConvolutionBwdDataAlgoPerf_t {
    cudnnConvolutionBwdDataAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdDataAlgoPerf_t;

typedef struct cudnnConvolutionBwdFilterAlgoPerf_t {
    cudnnConvolutionBwdFilterAlgo_t algo;
    cudnnStatus_t status;
    float time;
    size_t memory;
    cudnnMathType_t mathType;
    int reserved[3];
} cudnnConvolutionBwdFilterAlgoPerf_t;

cudnnStatus_t cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle_t handle,
                                                      cudnnTensorDescriptor_t xDesc,
                                                      cudnnFilterDescriptor_t wDesc,
                                                      cudnnConvolutionDescriptor_t convDesc,
                                                      cudnnTensorDescriptor_t yDesc,
                                                      int requestedAlgoCount,
                                                      int* returnedAlgoCount,
                                                      cudnnConvolutionFwdAlgoPerf_t* perfResults);

cudnnStatus_t cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle_t handle,
                                                            cudnnFilterDescriptor_t wDesc,
                                                            cudnnTensorDescriptor_t dyDesc,
                                                            cudnnConvolutionDescriptor_t convDesc,
                                                            cudnnTensorDescriptor_t dxDesc,
                                                            cudnnConvolutionBwdDataAlgo_t algo,
                                                            size_t* sizeInBytes);

cudnnStatus_t cudnnFindConvolutionBackwardDataAlgorithm(cudnnHandle_t handle,
                                                         cudnnFilterDescriptor_t wDesc,
                                                         cudnnTensorDescriptor_t dyDesc,
                                                         cudnnConvolutionDescriptor_t convDesc,
                                                         cudnnTensorDescriptor_t dxDesc,
                                                         int requestedAlgoCount,
                                                         int* returnedAlgoCount,
                                                         cudnnConvolutionBwdDataAlgoPerf_t* perfResults);

cudnnStatus_t cudnnConvolutionBackwardData(cudnnHandle_t handle,
                                            const void* alpha,
                                            cudnnFilterDescriptor_t wDesc, const void* w,
                                            cudnnTensorDescriptor_t dyDesc, const void* dy,
                                            cudnnConvolutionDescriptor_t convDesc,
                                            cudnnConvolutionBwdDataAlgo_t algo,
                                            void* workSpace, size_t workSpaceSizeInBytes,
                                            const void* beta,
                                            cudnnTensorDescriptor_t dxDesc, void* dx);

cudnnStatus_t cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle_t handle,
                                                              cudnnTensorDescriptor_t xDesc,
                                                              cudnnTensorDescriptor_t dyDesc,
                                                              cudnnConvolutionDescriptor_t convDesc,
                                                              cudnnFilterDescriptor_t dwDesc,
                                                              cudnnConvolutionBwdFilterAlgo_t algo,
                                                              size_t* sizeInBytes);

cudnnStatus_t cudnnFindConvolutionBackwardFilterAlgorithm(cudnnHandle_t handle,
                                                           cudnnTensorDescriptor_t xDesc,
                                                           cudnnTensorDescriptor_t dyDesc,
                                                           cudnnConvolutionDescriptor_t convDesc,
                                                           cudnnFilterDescriptor_t dwDesc,
                                                           int requestedAlgoCount,
                                                           int* returnedAlgoCount,
                                                           cudnnConvolutionBwdFilterAlgoPerf_t* perfResults);

cudnnStatus_t cudnnConvolutionBackwardFilter(cudnnHandle_t handle,
                                              const void* alpha,
                                              cudnnTensorDescriptor_t xDesc, const void* x,
                                              cudnnTensorDescriptor_t dyDesc, const void* dy,
                                              cudnnConvolutionDescriptor_t convDesc,
                                              cudnnConvolutionBwdFilterAlgo_t algo,
                                              void* workSpace, size_t workSpaceSizeInBytes,
                                              const void* beta,
                                              cudnnFilterDescriptor_t dwDesc, void* dw);

cudnnStatus_t cudnnConvolutionBackwardBias(cudnnHandle_t handle,
                                            const void* alpha,
                                            cudnnTensorDescriptor_t dyDesc, const void* dy,
                                            const void* beta,
                                            cudnnTensorDescriptor_t dbDesc, void* db);

// Activation
cudnnStatus_t cudnnCreateActivationDescriptor(cudnnActivationDescriptor_t* activationDesc);
cudnnStatus_t cudnnDestroyActivationDescriptor(cudnnActivationDescriptor_t activationDesc);
cudnnStatus_t cudnnSetActivationDescriptor(cudnnActivationDescriptor_t activationDesc,
                                            cudnnActivationMode_t mode,
                                            cudnnNanPropagation_t reluNanOpt,
                                            double coef);
cudnnStatus_t cudnnActivationForward(cudnnHandle_t handle,
                                      cudnnActivationDescriptor_t activationDesc,
                                      const void* alpha,
                                      cudnnTensorDescriptor_t xDesc, const void* x,
                                      const void* beta,
                                      cudnnTensorDescriptor_t yDesc, void* y);

cudnnStatus_t cudnnActivationBackward(cudnnHandle_t handle,
                                       cudnnActivationDescriptor_t activationDesc,
                                       const void* alpha,
                                       cudnnTensorDescriptor_t yDesc, const void* y,
                                       cudnnTensorDescriptor_t dyDesc, const void* dy,
                                       cudnnTensorDescriptor_t xDesc, const void* x,
                                       const void* beta,
                                       cudnnTensorDescriptor_t dxDesc, void* dx);

// Pooling
typedef enum cudnnPoolingMode_t {
    CUDNN_POOLING_MAX = 0,
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING = 1,
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING = 2,
    CUDNN_POOLING_MAX_DETERMINISTIC = 3,
} cudnnPoolingMode_t;

cudnnStatus_t cudnnCreatePoolingDescriptor(cudnnPoolingDescriptor_t* poolingDesc);
cudnnStatus_t cudnnDestroyPoolingDescriptor(cudnnPoolingDescriptor_t poolingDesc);
cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t poolingDesc,
                                           cudnnPoolingMode_t mode,
                                           cudnnNanPropagation_t maxpoolingNanOpt,
                                           int windowHeight, int windowWidth,
                                           int verticalPadding, int horizontalPadding,
                                           int verticalStride, int horizontalStride);
cudnnStatus_t cudnnGetPooling2dForwardOutputDim(cudnnPoolingDescriptor_t poolingDesc,
                                                 cudnnTensorDescriptor_t inputTensorDesc,
                                                 int* n, int* c, int* h, int* w);
cudnnStatus_t cudnnPoolingForward(cudnnHandle_t handle,
                                   cudnnPoolingDescriptor_t poolingDesc,
                                   const void* alpha,
                                   cudnnTensorDescriptor_t xDesc, const void* x,
                                   const void* beta,
                                   cudnnTensorDescriptor_t yDesc, void* y);
cudnnStatus_t cudnnPoolingBackward(cudnnHandle_t handle,
                                    cudnnPoolingDescriptor_t poolingDesc,
                                    const void* alpha,
                                    cudnnTensorDescriptor_t yDesc, const void* y,
                                    cudnnTensorDescriptor_t dyDesc, const void* dy,
                                    cudnnTensorDescriptor_t xDesc, const void* x,
                                    const void* beta,
                                    cudnnTensorDescriptor_t dxDesc, void* dx);

// Dropout
cudnnStatus_t cudnnCreateDropoutDescriptor(cudnnDropoutDescriptor_t* dropoutDesc);
cudnnStatus_t cudnnDestroyDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc);
cudnnStatus_t cudnnSetDropoutDescriptor(cudnnDropoutDescriptor_t dropoutDesc,
                                         cudnnHandle_t handle,
                                         float dropout,
                                         void* states, size_t stateSizeInBytes,
                                         unsigned long long seed);
cudnnStatus_t cudnnDropoutGetStatesSize(cudnnHandle_t handle, size_t* sizeInBytes);
cudnnStatus_t cudnnDropoutForward(cudnnHandle_t handle,
                                   cudnnDropoutDescriptor_t dropoutDesc,
                                   cudnnTensorDescriptor_t xdesc, const void* x,
                                   cudnnTensorDescriptor_t ydesc, void* y,
                                   void* reserveSpace, size_t reserveSpaceSizeInBytes);
cudnnStatus_t cudnnDropoutBackward(cudnnHandle_t handle,
                                    cudnnDropoutDescriptor_t dropoutDesc,
                                    cudnnTensorDescriptor_t dydesc, const void* dy,
                                    cudnnTensorDescriptor_t dxdesc, void* dx,
                                    void* reserveSpace, size_t reserveSpaceSizeInBytes);

// Nd tensor descriptor (used by frameworks instead of 4d)
cudnnStatus_t cudnnSetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          cudnnDataType_t dataType,
                                          int nbDims,
                                          const int dimA[],
                                          const int strideA[]);
cudnnStatus_t cudnnGetTensorNdDescriptor(cudnnTensorDescriptor_t tensorDesc,
                                          int nbDimsRequested,
                                          cudnnDataType_t* dataType,
                                          int* nbDims,
                                          int dimA[],
                                          int strideA[]);

// Tensor operations
cudnnStatus_t cudnnAddTensor(cudnnHandle_t handle,
                              const void* alpha,
                              cudnnTensorDescriptor_t aDesc, const void* A,
                              const void* beta,
                              cudnnTensorDescriptor_t cDesc, void* C);

cudnnStatus_t cudnnTransformTensor(cudnnHandle_t handle,
                                    const void* alpha,
                                    cudnnTensorDescriptor_t xDesc, const void* x,
                                    const void* beta,
                                    cudnnTensorDescriptor_t yDesc, void* y);

cudnnStatus_t cudnnSetTensor(cudnnHandle_t handle,
                              cudnnTensorDescriptor_t yDesc,
                              void* y,
                              const void* valuePtr);

cudnnStatus_t cudnnScaleTensor(cudnnHandle_t handle,
                                cudnnTensorDescriptor_t yDesc,
                                void* y,
                                const void* alpha);

// Softmax
typedef enum cudnnSoftmaxAlgorithm_t {
    CUDNN_SOFTMAX_FAST = 0,
    CUDNN_SOFTMAX_ACCURATE = 1,
    CUDNN_SOFTMAX_LOG = 2,
} cudnnSoftmaxAlgorithm_t;

typedef enum cudnnSoftmaxMode_t {
    CUDNN_SOFTMAX_MODE_INSTANCE = 0,
    CUDNN_SOFTMAX_MODE_CHANNEL = 1,
} cudnnSoftmaxMode_t;

cudnnStatus_t cudnnSoftmaxForward(cudnnHandle_t handle,
                                   cudnnSoftmaxAlgorithm_t algo,
                                   cudnnSoftmaxMode_t mode,
                                   const void* alpha,
                                   cudnnTensorDescriptor_t xDesc, const void* x,
                                   const void* beta,
                                   cudnnTensorDescriptor_t yDesc, void* y);

// Batch normalization
typedef enum cudnnBatchNormMode_t {
    CUDNN_BATCHNORM_PER_ACTIVATION = 0,
    CUDNN_BATCHNORM_SPATIAL = 1,
    CUDNN_BATCHNORM_SPATIAL_PERSISTENT = 2,
} cudnnBatchNormMode_t;

cudnnStatus_t cudnnBatchNormalizationForwardInference(
    cudnnHandle_t handle,
    cudnnBatchNormMode_t mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    const void* estimatedMean, const void* estimatedVariance,
    double epsilon);

#ifdef __cplusplus
}
#endif
