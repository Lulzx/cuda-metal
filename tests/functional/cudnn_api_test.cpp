#include "cudnn.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static bool test_handle_lifecycle() {
    cudnnHandle_t handle = nullptr;
    cudnnStatus_t st = cudnnCreate(&handle);
    if (st != CUDNN_STATUS_SUCCESS || handle == nullptr) {
        std::fprintf(stderr, "FAIL: cudnnCreate returned %d\n", st);
        return false;
    }
    st = cudnnDestroy(handle);
    if (st != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: cudnnDestroy returned %d\n", st);
        return false;
    }
    return true;
}

static bool test_tensor_descriptor() {
    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 4, 4);

    cudnnDataType_t dt;
    int n, c, h, w, ns, cs, hs, ws;
    cudnnGetTensor4dDescriptor(desc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws);
    if (dt != CUDNN_DATA_FLOAT || n != 1 || c != 3 || h != 4 || w != 4) {
        std::fprintf(stderr, "FAIL: tensor descriptor values wrong\n");
        return false;
    }
    if (ws != 1 || hs != 4 || cs != 16 || ns != 48) {
        std::fprintf(stderr, "FAIL: strides wrong: ns=%d cs=%d hs=%d ws=%d\n", ns, cs, hs, ws);
        return false;
    }
    cudnnDestroyTensorDescriptor(desc);
    return true;
}

static bool test_conv_output_dim() {
    cudnnTensorDescriptor_t xDesc = nullptr;
    cudnnFilterDescriptor_t wDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 5, 5);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 3);
    cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);

    int n, c, h, w;
    cudnnGetConvolution2dForwardOutputDim(convDesc, xDesc, wDesc, &n, &c, &h, &w);
    // 5x5 input, 3x3 kernel, pad=1, stride=1 => 5x5 output
    if (n != 1 || c != 1 || h != 5 || w != 5) {
        std::fprintf(stderr, "FAIL: conv output dim %dx%dx%dx%d (expected 1x1x5x5)\n", n, c, h, w);
        return false;
    }

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    return true;
}

static bool test_conv_forward_identity() {
    // 1x1x3x3 input convolved with 1x1x1x1 identity kernel => same output
    float input[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float kernel[1] = {1.0f};
    float output[9] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
    cudnnFilterDescriptor_t wDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;

    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 1, 1);
    cudnnSetConvolution2dDescriptor(convDesc, 0, 0, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 3, 3);

    cudnnStatus_t st = cudnnConvolutionForward(handle, &alpha,
                                                xDesc, input, wDesc, kernel,
                                                convDesc,
                                                CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                                                nullptr, 0, &beta, yDesc, output);
    if (st != CUDNN_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: convForward returned %d\n", st);
        return false;
    }

    for (int i = 0; i < 9; ++i) {
        if (std::fabs(output[i] - input[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: conv output[%d]=%f expected %f\n", i, output[i], input[i]);
            return false;
        }
    }

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_activation_relu() {
    float input[] = {-2, -1, 0, 1, 2};
    float output[5] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnActivationDescriptor_t act = nullptr;
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 5);

    cudnnActivationForward(handle, act, &alpha, desc, input, &beta, desc, output);

    float expected[] = {0, 0, 0, 1, 2};
    for (int i = 0; i < 5; ++i) {
        if (std::fabs(output[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: relu[%d]=%f expected %f\n", i, output[i], expected[i]);
            return false;
        }
    }

    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(handle);
    return true;
}

static bool test_softmax() {
    float input[] = {1.0f, 2.0f, 3.0f};
    float output[3] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 3, 1, 1);

    cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL,
                        &alpha, desc, input, &beta, desc, output);

    float sum = 0;
    for (int i = 0; i < 3; ++i) sum += output[i];
    if (std::fabs(sum - 1.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: softmax sum=%f (expected 1.0)\n", sum);
        return false;
    }
    // output should be monotonically increasing
    if (output[0] >= output[1] || output[1] >= output[2]) {
        std::fprintf(stderr, "FAIL: softmax not monotonic\n");
        return false;
    }

    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(handle);
    return true;
}

static bool test_backward_bias() {
    // dy: 1x2x2x2, db should sum over N,H,W per channel
    float dy[] = {1, 2, 3, 4, 10, 20, 30, 40};
    float db[2] = {99, 99}; // should be overwritten
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t dyDesc = nullptr, dbDesc = nullptr;
    cudnnCreateTensorDescriptor(&dyDesc);
    cudnnCreateTensorDescriptor(&dbDesc);
    cudnnSetTensor4dDescriptor(dyDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 2, 2);
    cudnnSetTensor4dDescriptor(dbDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);

    cudnnConvolutionBackwardBias(handle, &alpha, dyDesc, dy, &beta, dbDesc, db);

    // channel 0: 1+2+3+4 = 10, channel 1: 10+20+30+40 = 100
    if (std::fabs(db[0] - 10.0f) > 1e-5f || std::fabs(db[1] - 100.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: backward bias db=[%f,%f] expected [10,100]\n", db[0], db[1]);
        return false;
    }

    cudnnDestroyTensorDescriptor(dbDesc);
    cudnnDestroyTensorDescriptor(dyDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_find_algo_v7() {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
    cudnnFilterDescriptor_t wDesc = nullptr;
    cudnnConvolutionDescriptor_t convDesc = nullptr;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);

    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    cudnnSetFilter4dDescriptor(wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, 3, 3);
    cudnnSetConvolution2dDescriptor(convDesc, 1, 1, 1, 1, 1, 1,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);

    cudnnConvolutionFwdAlgoPerf_t perf[4];
    int count = 0;
    cudnnStatus_t st = cudnnGetConvolutionForwardAlgorithm_v7(handle, xDesc, wDesc, convDesc, yDesc,
                                                               4, &count, perf);
    if (st != CUDNN_STATUS_SUCCESS || count < 1) {
        std::fprintf(stderr, "FAIL: v7 algo finder returned %d, count=%d\n", st, count);
        return false;
    }

    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroyFilterDescriptor(wDesc);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_batch_norm_inference() {
    // 1x2x1x1 tensor, scale=[1,1], bias=[0,0], mean=[0,0], var=[1,1], eps=0
    // => y = (x - mean) / sqrt(var + eps) * scale + bias = x
    float x[] = {3.0f, -2.0f};
    float y[2] = {};
    float scale[] = {1.0f, 1.0f};
    float bias[] = {0.0f, 0.0f};
    float mean[] = {0.0f, 0.0f};
    float var[] = {1.0f, 1.0f};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc = nullptr, bnDesc = nullptr;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&bnDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);
    cudnnSetTensor4dDescriptor(bnDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 2, 1, 1);

    cudnnBatchNormalizationForwardInference(handle, CUDNN_BATCHNORM_SPATIAL,
                                             &alpha, &beta, xDesc, x, xDesc, y,
                                             bnDesc, scale, bias, mean, var, 0.0);

    if (std::fabs(y[0] - 3.0f) > 1e-5f || std::fabs(y[1] - (-2.0f)) > 1e-5f) {
        std::fprintf(stderr, "FAIL: batchnorm y=[%f,%f] expected [3,-2]\n", y[0], y[1]);
        return false;
    }

    cudnnDestroyTensorDescriptor(bnDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_version_and_error() {
    size_t ver = cudnnGetVersion();
    if (ver == 0) {
        std::fprintf(stderr, "FAIL: cudnnGetVersion returned 0\n");
        return false;
    }
    const char* err = cudnnGetErrorString(CUDNN_STATUS_SUCCESS);
    if (!err || std::strlen(err) == 0) {
        std::fprintf(stderr, "FAIL: cudnnGetErrorString returned null/empty\n");
        return false;
    }
    return true;
}

static bool test_activation_backward() {
    float x[] = {-1.0f, 0.5f, 2.0f};
    float y[] = {0.0f, 0.5f, 2.0f}; // relu(x)
    float dy[] = {1.0f, 1.0f, 1.0f};
    float dx[3] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnActivationDescriptor_t act = nullptr;
    cudnnCreateActivationDescriptor(&act);
    cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);

    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 3);

    cudnnActivationBackward(handle, act, &alpha, desc, y, desc, dy, desc, x, &beta, desc, dx);

    // relu backward: dx = dy * (x > 0)
    float expected[] = {0.0f, 1.0f, 1.0f};
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(dx[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: relu backward[%d]=%f expected %f\n", i, dx[i], expected[i]);
            return false;
        }
    }

    cudnnDestroyActivationDescriptor(act);
    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(handle);
    return true;
}

static bool test_pooling_max() {
    // 1x1x4x4 input, 2x2 max pool, stride 2 => 1x1x2x2
    float x[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
    float y[4] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 4, 4);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2);

    cudnnPoolingDescriptor_t pool = nullptr;
    cudnnCreatePoolingDescriptor(&pool);
    cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN,
                                 2, 2, 0, 0, 2, 2);

    // Verify output dims
    int n, c, h, w;
    cudnnGetPooling2dForwardOutputDim(pool, xDesc, &n, &c, &h, &w);
    if (h != 2 || w != 2) {
        std::fprintf(stderr, "FAIL: pool output dim %dx%d expected 2x2\n", h, w);
        return false;
    }

    cudnnPoolingForward(handle, pool, &alpha, xDesc, x, &beta, yDesc, y);

    // Max of each 2x2 block: {6, 8, 14, 16}
    float expected[] = {6.0f, 8.0f, 14.0f, 16.0f};
    for (int i = 0; i < 4; ++i) {
        if (std::fabs(y[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: maxpool[%d]=%f expected %f\n", i, y[i], expected[i]);
            return false;
        }
    }

    cudnnDestroyPoolingDescriptor(pool);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_pooling_avg() {
    // 1x1x2x2 input, 2x2 avg pool => 1x1x1x1
    float x[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float y[1] = {};
    float alpha = 1.0f, beta = 0.0f;

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t xDesc = nullptr, yDesc = nullptr;
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnCreateTensorDescriptor(&yDesc);
    cudnnSetTensor4dDescriptor(xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 2, 2);
    cudnnSetTensor4dDescriptor(yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 1);

    cudnnPoolingDescriptor_t pool = nullptr;
    cudnnCreatePoolingDescriptor(&pool);
    cudnnSetPooling2dDescriptor(pool, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                 CUDNN_NOT_PROPAGATE_NAN, 2, 2, 0, 0, 2, 2);

    cudnnPoolingForward(handle, pool, &alpha, xDesc, x, &beta, yDesc, y);

    // avg = (2+4+6+8)/4 = 5.0
    if (std::fabs(y[0] - 5.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: avgpool=%f expected 5.0\n", y[0]);
        return false;
    }

    cudnnDestroyPoolingDescriptor(pool);
    cudnnDestroyTensorDescriptor(yDesc);
    cudnnDestroyTensorDescriptor(xDesc);
    cudnnDestroy(handle);
    return true;
}

static bool test_dropout_passthrough() {
    // dropout=0 should be identity
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float y[4] = {};

    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    cudnnDropoutDescriptor_t drop = nullptr;
    cudnnCreateDropoutDescriptor(&drop);
    cudnnSetDropoutDescriptor(drop, handle, 0.0f, nullptr, 0, 42);

    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);
    cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 4);

    cudnnDropoutForward(handle, drop, desc, x, desc, y, nullptr, 0);

    for (int i = 0; i < 4; ++i) {
        if (std::fabs(y[i] - x[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: dropout passthrough[%d]=%f expected %f\n", i, y[i], x[i]);
            return false;
        }
    }

    cudnnDestroyDropoutDescriptor(drop);
    cudnnDestroyTensorDescriptor(desc);
    cudnnDestroy(handle);
    return true;
}

static bool test_dropout_states_size() {
    cudnnHandle_t handle = nullptr;
    cudnnCreate(&handle);

    size_t size = 0;
    cudnnDropoutGetStatesSize(handle, &size);
    if (size == 0) {
        std::fprintf(stderr, "FAIL: dropout states size is 0\n");
        return false;
    }

    cudnnDestroy(handle);
    return true;
}

static bool test_tensor_nd_descriptor() {
    cudnnTensorDescriptor_t desc = nullptr;
    cudnnCreateTensorDescriptor(&desc);

    int dims[] = {2, 3, 4, 5};
    int strides[] = {60, 20, 5, 1};
    cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 4, dims, strides);

    cudnnDataType_t dt;
    int nbDims = 0;
    int outDims[4] = {}, outStrides[4] = {};
    cudnnGetTensorNdDescriptor(desc, 4, &dt, &nbDims, outDims, outStrides);

    if (dt != CUDNN_DATA_FLOAT || nbDims != 4) {
        std::fprintf(stderr, "FAIL: Nd descriptor dt=%d nbDims=%d\n", dt, nbDims);
        return false;
    }
    for (int i = 0; i < 4; ++i) {
        if (outDims[i] != dims[i] || outStrides[i] != strides[i]) {
            std::fprintf(stderr, "FAIL: Nd dim[%d]=%d/%d stride=%d/%d\n",
                         i, outDims[i], dims[i], outStrides[i], strides[i]);
            return false;
        }
    }

    cudnnDestroyTensorDescriptor(desc);
    return true;
}

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_tensor_descriptor()) return 1;
    if (!test_conv_output_dim()) return 1;
    if (!test_conv_forward_identity()) return 1;
    if (!test_activation_relu()) return 1;
    if (!test_softmax()) return 1;
    if (!test_backward_bias()) return 1;
    if (!test_find_algo_v7()) return 1;
    if (!test_batch_norm_inference()) return 1;
    if (!test_version_and_error()) return 1;
    if (!test_activation_backward()) return 1;
    if (!test_pooling_max()) return 1;
    if (!test_pooling_avg()) return 1;
    if (!test_dropout_passthrough()) return 1;
    if (!test_dropout_states_size()) return 1;
    if (!test_tensor_nd_descriptor()) return 1;

    std::printf("PASS: cuDNN API tests\n");
    return 0;
}
