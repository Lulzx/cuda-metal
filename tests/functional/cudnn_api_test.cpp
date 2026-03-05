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

int main() {
    if (!test_handle_lifecycle()) return 1;
    if (!test_tensor_descriptor()) return 1;
    if (!test_conv_output_dim()) return 1;
    if (!test_conv_forward_identity()) return 1;
    if (!test_activation_relu()) return 1;
    if (!test_softmax()) return 1;
    if (!test_version_and_error()) return 1;

    std::printf("PASS: cuDNN API tests\n");
    return 0;
}
