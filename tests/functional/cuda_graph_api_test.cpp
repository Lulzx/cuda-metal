#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>

static bool test_graph_create_destroy() {
    cudaGraph_t graph = nullptr;
    cudaError_t err = cudaGraphCreate(&graph, 0);
    if (err != cudaSuccess || graph == nullptr) {
        std::fprintf(stderr, "FAIL: cudaGraphCreate returned %d\n", err);
        return false;
    }

    size_t numNodes = 999;
    err = cudaGraphGetNodes(graph, nullptr, &numNodes);
    if (err != cudaSuccess || numNodes != 0) {
        std::fprintf(stderr, "FAIL: empty graph should have 0 nodes, got %zu\n", numNodes);
        return false;
    }

    err = cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGraphDestroy returned %d\n", err);
        return false;
    }
    return true;
}

static bool test_graph_instantiate_launch() {
    cudaGraph_t graph = nullptr;
    cudaGraphCreate(&graph, 0);

    cudaGraphExec_t exec = nullptr;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess || exec == nullptr) {
        std::fprintf(stderr, "FAIL: cudaGraphInstantiate returned %d\n", err);
        return false;
    }

    // Launch an empty graph — should succeed as a no-op
    err = cudaGraphLaunch(exec, nullptr);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGraphLaunch returned %d\n", err);
        return false;
    }

    err = cudaGraphExecDestroy(exec);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaGraphExecDestroy returned %d\n", err);
        return false;
    }

    cudaGraphDestroy(graph);
    return true;
}

static bool test_stream_capture_status() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusActive;
    cudaError_t err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess || status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "FAIL: uncaptured stream should report None, got %d\n", status);
        return false;
    }

    // Begin capture
    err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamBeginCapture returned %d\n", err);
        return false;
    }

    err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess || status != cudaStreamCaptureStatusActive) {
        std::fprintf(stderr, "FAIL: capturing stream should report Active, got %d\n", status);
        return false;
    }

    // End capture
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess || graph == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamEndCapture returned %d\n", err);
        return false;
    }

    // After end, should be None again
    err = cudaStreamIsCapturing(stream, &status);
    if (err != cudaSuccess || status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "FAIL: post-capture stream should report None\n");
        return false;
    }

    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

static bool test_graph_null_args() {
    // Null graph should fail
    if (cudaGraphDestroy(nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaGraphDestroy(null) should return InvalidValue\n");
        return false;
    }
    if (cudaGraphExecDestroy(nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaGraphExecDestroy(null) should return InvalidValue\n");
        return false;
    }
    if (cudaGraphCreate(nullptr, 0) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cudaGraphCreate(null) should return InvalidValue\n");
        return false;
    }
    return true;
}

static bool test_capture_memcpy_replay() {
    // Capture a memcpyAsync during stream capture, then replay via graph launch
    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float* dev = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev), sizeof(src));
    std::memset(dev, 0, sizeof(src));

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // Begin capture
    cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "FAIL: BeginCapture returned %d\n", err);
        return false;
    }

    // This should be recorded, not executed
    cudaMemcpyAsync(dev, src, sizeof(src), cudaMemcpyHostToDevice, stream);

    // Verify data was NOT copied yet (capture should defer execution)
    float check[4] = {};
    std::memcpy(check, dev, sizeof(check));
    if (check[0] != 0.0f) {
        std::fprintf(stderr, "FAIL: memcpy should be deferred during capture\n");
        return false;
    }

    // End capture
    cudaGraph_t graph = nullptr;
    err = cudaStreamEndCapture(stream, &graph);
    if (err != cudaSuccess || graph == nullptr) {
        std::fprintf(stderr, "FAIL: EndCapture returned %d\n", err);
        return false;
    }

    // Graph should have 1 node
    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    if (numNodes != 1) {
        std::fprintf(stderr, "FAIL: expected 1 captured node, got %zu\n", numNodes);
        return false;
    }

    // Instantiate and launch
    cudaGraphExec_t exec = nullptr;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    // Now data should be copied
    float result[4] = {};
    std::memcpy(result, dev, sizeof(result));
    for (int i = 0; i < 4; ++i) {
        if (result[i] != src[i]) {
            std::fprintf(stderr, "FAIL: replay memcpy mismatch at [%d]: %f != %f\n",
                         i, result[i], src[i]);
            return false;
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(dev);
    return true;
}

static bool test_capture_memset_replay() {
    float* dev = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev), 64);
    std::memset(dev, 0xFF, 64);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    cudaMemsetAsync(dev, 0, 64, stream);

    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(stream, &graph);

    size_t numNodes = 0;
    cudaGraphGetNodes(graph, nullptr, &numNodes);
    if (numNodes != 1) {
        std::fprintf(stderr, "FAIL: expected 1 memset node, got %zu\n", numNodes);
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    // Verify memset took effect after replay
    unsigned char check[64];
    std::memcpy(check, dev, 64);
    for (int i = 0; i < 64; ++i) {
        if (check[i] != 0) {
            std::fprintf(stderr, "FAIL: memset replay didn't zero byte %d\n", i);
            return false;
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(dev);
    return true;
}

int main() {
    if (!test_graph_create_destroy()) return 1;
    if (!test_graph_instantiate_launch()) return 1;
    if (!test_stream_capture_status()) return 1;
    if (!test_graph_null_args()) return 1;
    if (!test_capture_memcpy_replay()) return 1;
    if (!test_capture_memset_replay()) return 1;

    std::printf("PASS: CUDA Graph API tests\n");
    return 0;
}
