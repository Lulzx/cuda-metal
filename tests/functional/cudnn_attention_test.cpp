#include <cudnn.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_attn_descriptor_lifecycle() {
    cudnnAttnDescriptor_t attnDesc;
    cudnnStatus_t st = cudnnCreateAttnDescriptor(&attnDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "create attn descriptor");

    st = cudnnSetAttnDescriptor(attnDesc,
                                 0,    // attnMode
                                 8,    // nHeads
                                 1.0,  // smScaler
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH,
                                 nullptr, nullptr, // dropout descs
                                 64, 64, 64,       // qSize, kSize, vSize
                                 0, 0, 0, 0,       // proj sizes (0 = no projection)
                                 128, 128,         // max seq lengths
                                 32, 1);           // max batch, beam
    CHECK(st == CUDNN_STATUS_SUCCESS, "set attn descriptor");
    CHECK(cudnnSetAttnDescriptor(attnDesc, 0, 8,
                                 std::numeric_limits<double>::quiet_NaN(),
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                 64, 64, 64, 0, 0, 0, 0, 128, 128, 32, 1) ==
              CUDNN_STATUS_BAD_PARAM,
          "reject non-finite attention scaler");
    CHECK(cudnnSetAttnDescriptor(attnDesc, 0, 8, 1.0,
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                 64, 64, 64, -1, 0, 0, 0, 128, 128, 32, 1) ==
              CUDNN_STATUS_BAD_PARAM,
          "reject negative attention projection size");
    CHECK(cudnnSetAttnDescriptor(attnDesc, 0, 8, 1.0,
                                 CUDNN_DATA_HALF, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                 64, 64, 64, 0, 0, 0, 0, 128, 128, 32, 1) ==
              CUDNN_STATUS_NOT_SUPPORTED,
          "reject unsupported attention datatype");

    st = cudnnDestroyAttnDescriptor(attnDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "destroy attn descriptor");
}

static void test_attn_buffer_sizes() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnAttnDescriptor_t attnDesc;
    cudnnCreateAttnDescriptor(&attnDesc);
    cudnnSetAttnDescriptor(attnDesc, 0, 4, 1.0,
                            CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                            CUDNN_DEFAULT_MATH,
                            nullptr, nullptr,
                            32, 32, 32,   // q/k/v size
                            0, 0, 0, 0,   // no projections
                            64, 64, 16, 1);

    size_t weightSize = 0, workSize = 0, reserveSize = 0;
    cudnnStatus_t st = cudnnGetMultiHeadAttnBuffers(handle, attnDesc,
                                                     &weightSize, &workSize, &reserveSize);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get attn buffers");
    CHECK(weightSize == 0, "projection-free attention requires no weights");

    cudnnAttnDescriptor_t overflowDesc;
    cudnnCreateAttnDescriptor(&overflowDesc);
    CHECK(cudnnSetAttnDescriptor(overflowDesc, 0, std::numeric_limits<int>::max(), 1.0,
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                 std::numeric_limits<int>::max(), 1, 1,
                                 std::numeric_limits<int>::max(), 0, 0, 0,
                                 1, 1, 1, 1) == CUDNN_STATUS_SUCCESS,
          "configure overflowing attention weights");
    CHECK(cudnnGetMultiHeadAttnBuffers(handle, overflowDesc, &weightSize,
                                       nullptr, nullptr) == CUDNN_STATUS_BAD_PARAM,
          "reject overflowing attention weight geometry");
    cudnnDestroyAttnDescriptor(overflowDesc);

    cudnnDestroyAttnDescriptor(attnDesc);
    cudnnDestroy(handle);
}

static void test_seq_data_descriptor() {
    cudnnSeqDataDescriptor_t seqDesc;
    cudnnStatus_t st = cudnnCreateSeqDataDescriptor(&seqDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "create seq data descriptor");

    int dims[] = {64, 32, 1, 128}; // time, batch, beam, vect
    cudnnSeqDataAxis_t axes[] = {CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
                                  CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
    int seqLengths[] = {64};
    st = cudnnSetSeqDataDescriptor(seqDesc, CUDNN_DATA_FLOAT, 4, dims, axes, 1, seqLengths, nullptr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "set seq data descriptor");
    int invalidLength[] = {65};
    CHECK(cudnnSetSeqDataDescriptor(seqDesc, CUDNN_DATA_FLOAT, 4, dims, axes, 1,
                                    invalidLength, nullptr) == CUDNN_STATUS_BAD_PARAM,
          "reject sequence length beyond time extent");
    float padding = 0.0f;
    CHECK(cudnnSetSeqDataDescriptor(seqDesc, CUDNN_DATA_FLOAT, 4, dims, axes, 1,
                                    seqLengths, &padding) == CUDNN_STATUS_NOT_SUPPORTED,
          "reject unsupported sequence padding fill");

    st = cudnnDestroySeqDataDescriptor(seqDesc);
    CHECK(st == CUDNN_STATUS_SUCCESS, "destroy seq data descriptor");
}

static void test_attn_weight_pointers() {
    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnAttnDescriptor_t attnDesc;
    cudnnCreateAttnDescriptor(&attnDesc);
    cudnnSetAttnDescriptor(attnDesc, 0, 2, 1.0,
                            CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                            CUDNN_DEFAULT_MATH,
                            nullptr, nullptr,
                            16, 16, 16,
                            8, 8, 8, 16,
                            32, 32, 8, 1);

    size_t weightSize = 0;
    cudnnGetMultiHeadAttnBuffers(handle, attnDesc, &weightSize, nullptr, nullptr);

    // Allocate dummy weights
    float* weights = new float[weightSize / sizeof(float)]();
    void* qAddr = nullptr;
    void* kAddr = nullptr;

    cudnnStatus_t st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                                     CUDNN_MH_ATTN_Q_WEIGHTS,
                                                     weightSize, weights,
                                                     nullptr, &qAddr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get Q weights");
    CHECK(qAddr == weights, "Q weights at start");

    st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                       CUDNN_MH_ATTN_K_WEIGHTS,
                                       weightSize, weights,
                                       nullptr, &kAddr);
    CHECK(st == CUDNN_STATUS_SUCCESS, "get K weights");
    CHECK(kAddr > qAddr, "K weights after Q weights");
    void* invalidAddr = reinterpret_cast<void*>(1);
    st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                       CUDNN_MH_ATTN_O_WEIGHTS,
                                       weightSize - sizeof(float), weights,
                                       nullptr, &invalidAddr);
    CHECK(st == CUDNN_STATUS_BAD_PARAM && invalidAddr == nullptr,
          "reject undersized attention weight buffer");
    st = cudnnGetMultiHeadAttnWeights(handle, attnDesc,
                                       CUDNN_MH_ATTN_Q_BIASES,
                                       weightSize, weights, nullptr, &invalidAddr);
    CHECK(st == CUDNN_STATUS_NOT_SUPPORTED && invalidAddr == nullptr,
          "reject unsupported attention bias query");

    delete[] weights;
    cudnnDestroyAttnDescriptor(attnDesc);
    cudnnDestroy(handle);
}

static void test_attn_forward_numerical_and_rejection() {
    cudnnHandle_t handle = nullptr;
    cudnnAttnDescriptor_t attnDesc = nullptr;
    cudnnSeqDataDescriptor_t qDesc = nullptr, kDesc = nullptr;
    cudnnSeqDataDescriptor_t vDesc = nullptr, oDesc = nullptr;
    CHECK(cudnnCreate(&handle) == CUDNN_STATUS_SUCCESS, "create attention handle");
    CHECK(cudnnCreateAttnDescriptor(&attnDesc) == CUDNN_STATUS_SUCCESS,
          "create numerical attention descriptor");
    CHECK(cudnnSetAttnDescriptor(attnDesc, 0, 2, 1.0,
                                 CUDNN_DATA_FLOAT, CUDNN_DATA_FLOAT,
                                 CUDNN_DEFAULT_MATH, nullptr, nullptr,
                                 4, 4, 4, 0, 0, 0, 0,
                                 2, 2, 1, 1) == CUDNN_STATUS_SUCCESS,
          "configure bounded FP32 attention");

    int dims[] = {2, 1, 1, 4};
    cudnnSeqDataAxis_t axes[] = {
        CUDNN_SEQDATA_TIME_DIM, CUDNN_SEQDATA_BATCH_DIM,
        CUDNN_SEQDATA_BEAM_DIM, CUDNN_SEQDATA_VECT_DIM};
    int lengths[] = {2};
    for (auto** desc : {&qDesc, &kDesc, &vDesc, &oDesc}) {
        CHECK(cudnnCreateSeqDataDescriptor(desc) == CUDNN_STATUS_SUCCESS,
              "create numerical sequence descriptor");
        CHECK(cudnnSetSeqDataDescriptor(*desc, CUDNN_DATA_FLOAT, 4, dims, axes,
                                        1, lengths, nullptr) == CUDNN_STATUS_SUCCESS,
              "configure canonical sequence descriptor");
    }

    const float q[] = {1, 0, 1, 0, 0, 1, 0, 1};
    const float k[] = {1, 0, 0, 1, 0, 1, 1, 0};
    const float v[] = {2, 4, 10, 20, 6, 8, 30, 40};
    float output[8];
    for (float& x : output) x = std::numeric_limits<float>::quiet_NaN();

    const cudnnStatus_t status = cudnnMultiHeadAttnForward(
        handle, attnDesc, -1, nullptr, nullptr, nullptr, nullptr,
        qDesc, q, nullptr, kDesc, k, vDesc, v, oDesc, output,
        0, nullptr, 0, nullptr, 0, nullptr);
    CHECK(status == CUDNN_STATUS_SUCCESS, "bounded attention forward succeeds");
    CHECK(cudaDeviceSynchronize() == cudaSuccess,
          "bounded attention completes in its CUDA stream timeline");

    std::vector<float> expected(8);
    for (int t = 0; t < 2; ++t) {
        for (int h = 0; h < 2; ++h) {
            float score[2] = {};
            for (int s = 0; s < 2; ++s) {
                for (int c = 0; c < 2; ++c)
                    score[s] += q[t * 4 + h * 2 + c] * k[s * 4 + h * 2 + c];
            }
            const float m = std::fmax(score[0], score[1]);
            const float e0 = std::exp(score[0] - m);
            const float e1 = std::exp(score[1] - m);
            for (int c = 0; c < 2; ++c) {
                expected[t * 4 + h * 2 + c] =
                    (e0 * v[h * 2 + c] + e1 * v[4 + h * 2 + c]) / (e0 + e1);
            }
        }
    }
    bool numerical = true;
    for (int i = 0; i < 8; ++i)
        numerical &= std::isfinite(output[i]) && std::fabs(output[i] - expected[i]) < 1e-5f;
    CHECK(numerical, "attention output matches independent softmax oracle");

    CHECK(cudnnMultiHeadAttnForward(
              handle, attnDesc, 0, nullptr, nullptr, nullptr, nullptr,
              qDesc, q, nullptr, kDesc, k, vDesc, v, oDesc, output,
              0, nullptr, 0, nullptr, 0, nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
          "incremental attention is explicitly rejected");
    float fakeWeight = 1.0f;
    CHECK(cudnnMultiHeadAttnForward(
              handle, attnDesc, -1, nullptr, nullptr, nullptr, nullptr,
              qDesc, q, nullptr, kDesc, k, vDesc, v, oDesc, output,
              sizeof(fakeWeight), &fakeWeight, 0, nullptr, 0, nullptr) ==
              CUDNN_STATUS_NOT_SUPPORTED,
          "projection weights are explicitly rejected");
    CHECK(cudnnMultiHeadAttnForward(
              handle, attnDesc, -1, nullptr, nullptr, nullptr, nullptr,
              qDesc, q, q, kDesc, k, vDesc, v, oDesc, output,
              0, nullptr, 0, nullptr, 0, nullptr) == CUDNN_STATUS_NOT_SUPPORTED,
          "attention residual input is explicitly rejected");
    CHECK(cudnnMultiHeadAttnForward(
              handle, attnDesc, -1, nullptr, nullptr, nullptr, nullptr,
              qDesc, q, nullptr, kDesc, k, vDesc, v, oDesc,
              const_cast<float*>(q), 0, nullptr, 0, nullptr, 0, nullptr) ==
              CUDNN_STATUS_BAD_PARAM,
          "reject overlapping attention output");

    float* shortDevice = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&shortDevice), sizeof(float)) == cudaSuccess,
          "allocate short tracked attention input");
    for (float& x : output) x = -7.0f;
    CHECK(cudnnMultiHeadAttnForward(
              handle, attnDesc, -1, nullptr, nullptr, nullptr, nullptr,
              qDesc, shortDevice, nullptr, kDesc, k, vDesc, v, oDesc, output,
              0, nullptr, 0, nullptr, 0, nullptr) == CUDNN_STATUS_BAD_PARAM,
          "reject undersized tracked attention input");
    bool untouched = true;
    for (float x : output) untouched &= x == -7.0f;
    CHECK(untouched, "attention bounds failure leaves output untouched");
    cudaFree(shortDevice);

    for (auto desc : {qDesc, kDesc, vDesc, oDesc}) cudnnDestroySeqDataDescriptor(desc);
    cudnnDestroyAttnDescriptor(attnDesc);
    cudnnDestroy(handle);
}

int main() {
    test_attn_descriptor_lifecycle();
    test_attn_buffer_sizes();
    test_seq_data_descriptor();
    test_attn_weight_pointers();
    test_attn_forward_numerical_and_rejection();
    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
