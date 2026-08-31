#include "cuda_runtime.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static void graph_host_increment(void* data) {
    ++*static_cast<int*>(data);
}

static bool test_graph_create_destroy() {
    cudaGraph_t invalid_graph = reinterpret_cast<cudaGraph_t>(1);
    if (cudaGraphCreate(&invalid_graph, 1) != cudaErrorInvalidValue ||
        invalid_graph != nullptr) {
        std::fprintf(stderr, "FAIL: cudaGraphCreate accepted non-zero flags\n");
        return false;
    }
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

static bool test_graph_pitched_copy_and_typed_memset() {
    constexpr std::size_t source_pitch = 8;
    constexpr std::size_t destination_pitch = 10;
    constexpr std::size_t height = 3;
    constexpr std::size_t depth = 2;
    unsigned char source[source_pitch * height * depth]{};
    unsigned char copied[destination_pitch * height * depth]{};
    for (std::size_t z = 0; z < depth; ++z) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < 4; ++x) {
                source[z * source_pitch * height + y * source_pitch + x] =
                    static_cast<unsigned char>(1 + z * 32 + y * 8 + x);
            }
        }
    }

    void* copy_target = nullptr;
    void* memset_target = nullptr;
    if (cudaMalloc(&copy_target, sizeof(copied)) != cudaSuccess ||
        cudaMalloc(&memset_target, 3 * 16) != cudaSuccess ||
        cudaMemset(copy_target, 0, sizeof(copied)) != cudaSuccess ||
        cudaMemset(memset_target, 0, 3 * 16) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph pitched-node allocation failed\n");
        return false;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphNode_t copy_in = nullptr;
    cudaGraphNode_t typed_set = nullptr;
    cudaGraphNode_t copy_out = nullptr;
    cudaMemcpy3DParms copy_params{};
    copy_params.srcPtr = make_cudaPitchedPtr(source, source_pitch, 4, height);
    copy_params.dstPtr = make_cudaPitchedPtr(copy_target, destination_pitch, 4, height);
    copy_params.extent = make_cudaExtent(4, height, depth);
    copy_params.kind = cudaMemcpyHostToDevice;
    cudaMemsetParams memset_params{};
    memset_params.dst = memset_target;
    memset_params.pitch = 16;
    memset_params.value = 0x12345678u;
    memset_params.elementSize = 4;
    memset_params.width = 3;
    memset_params.height = 3;
    cudaMemcpy3DParms invalid_copy = copy_params;
    invalid_copy.srcPtr.pitch = 3;
    cudaMemcpy3DParms overflowing_copy = copy_params;
    overflowing_copy.srcPos.z = 1;
    overflowing_copy.extent.depth = static_cast<std::size_t>(-1);
    cudaMemsetParams invalid_memset = memset_params;
    invalid_memset.elementSize = 3;
    cudaMemsetParams overflowing_memset = memset_params;
    overflowing_memset.pitch = static_cast<std::size_t>(-1);
    overflowing_memset.height = 2;

    if (cudaGraphCreate(&graph, 0) != cudaSuccess ||
        cudaGraphAddMemcpyNode(&copy_in, graph, nullptr, 0, &invalid_copy) !=
            cudaErrorInvalidValue ||
        cudaGraphAddMemcpyNode(&copy_in, graph, nullptr, 0, &overflowing_copy) !=
            cudaErrorInvalidValue ||
        cudaGraphAddMemsetNode(&typed_set, graph, nullptr, 0, &invalid_memset) !=
            cudaErrorInvalidValue ||
        cudaGraphAddMemsetNode(&typed_set, graph, nullptr, 0, &overflowing_memset) !=
            cudaErrorInvalidValue ||
        cudaGraphAddMemcpyNode(&copy_in, graph, nullptr, 0, &copy_params) != cudaSuccess ||
        cudaGraphAddMemsetNode(&typed_set, graph, &copy_in, 1, &memset_params) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph pitched-node construction failed\n");
        return false;
    }
    copy_params.srcPtr = make_cudaPitchedPtr(copy_target, destination_pitch, 4, height);
    copy_params.dstPtr = make_cudaPitchedPtr(copied, destination_pitch, 4, height);
    copy_params.kind = cudaMemcpyDeviceToHost;
    if (cudaGraphAddMemcpyNode(&copy_out, graph, &typed_set, 1, &copy_params) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph pitched copy-out node construction failed\n");
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph pitched-node replay failed\n");
        return false;
    }
    for (std::size_t z = 0; z < depth; ++z) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < destination_pitch; ++x) {
                const unsigned char expected = x < 4
                    ? source[z * source_pitch * height + y * source_pitch + x]
                    : 0;
                if (copied[z * destination_pitch * height + y * destination_pitch + x] !=
                    expected) {
                    std::fprintf(stderr, "FAIL: graph pitched 3D copy mismatch\n");
                    return false;
                }
            }
        }
    }
    unsigned char memset_result[3 * 16]{};
    if (cudaMemcpy(memset_result, memset_target, sizeof(memset_result),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph typed memset readback failed\n");
        return false;
    }
    for (std::size_t row = 0; row < 3; ++row) {
        for (std::size_t column = 0; column < 4; ++column) {
            std::uint32_t value = 0;
            std::memcpy(&value, memset_result + row * 16 + column * 4, sizeof(value));
            const std::uint32_t expected = column < 3 ? 0x12345678u : 0u;
            if (value != expected) {
                std::fprintf(stderr, "FAIL: graph typed pitched memset mismatch\n");
                return false;
            }
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(memset_target);
    cudaFree(copy_target);
    return true;
}

static bool test_graph_array_copy_geometry() {
    constexpr std::size_t width = 4;
    constexpr std::size_t height = 3;
    constexpr std::size_t depth = 2;
    std::uint16_t source[width * height * depth]{};
    std::uint16_t zero[width * height * depth]{};
    std::uint16_t result[width * height * depth]{};
    for (std::size_t index = 0; index < width * height * depth; ++index) {
        source[index] = static_cast<std::uint16_t>(100 + index);
    }

    const cudaChannelFormatDesc desc =
        cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray_t source_array = nullptr;
    cudaArray_t destination_array = nullptr;
    if (cudaMalloc3DArray(&source_array, &desc,
                          make_cudaExtent(width, height, depth), 0) != cudaSuccess ||
        cudaMalloc3DArray(&destination_array, &desc,
                          make_cudaExtent(width, height, depth), 0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph array allocation failed\n");
        return false;
    }

    cudaMemcpy3DParms initialize{};
    initialize.srcPtr = make_cudaPitchedPtr(
        zero, width * sizeof(std::uint16_t), width, height);
    initialize.dstArray = destination_array;
    initialize.extent = make_cudaExtent(width, height, depth);
    initialize.kind = cudaMemcpyHostToDevice;
    if (cudaMemcpy3D(&initialize) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph destination-array initialization failed\n");
        return false;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphNode_t upload = nullptr;
    cudaGraphNode_t array_copy = nullptr;
    cudaGraphNode_t download = nullptr;
    cudaMemcpy3DParms upload_params{};
    upload_params.srcPtr = make_cudaPitchedPtr(
        source, width * sizeof(std::uint16_t), width, height);
    upload_params.dstArray = source_array;
    upload_params.extent = make_cudaExtent(width, height, depth);
    upload_params.kind = cudaMemcpyHostToDevice;

    cudaMemcpy3DParms array_params{};
    array_params.srcArray = source_array;
    array_params.srcPos = make_cudaPos(1, 1, 0);
    array_params.dstArray = destination_array;
    array_params.dstPos = make_cudaPos(0, 0, 0);
    array_params.extent = make_cudaExtent(3, 2, 2);
    array_params.kind = cudaMemcpyDeviceToDevice;
    cudaMemcpy3DParms invalid_array = array_params;
    invalid_array.srcPos.x = width - 1;

    cudaMemcpy3DParms download_params{};
    download_params.srcArray = destination_array;
    download_params.dstPtr = make_cudaPitchedPtr(
        result, width * sizeof(std::uint16_t), width, height);
    download_params.extent = make_cudaExtent(width, height, depth);
    download_params.kind = cudaMemcpyDeviceToHost;

    if (cudaGraphCreate(&graph, 0) != cudaSuccess ||
        cudaGraphAddMemcpyNode(&array_copy, graph, nullptr, 0,
                               &invalid_array) != cudaErrorInvalidValue ||
        cudaMemcpy3D(&invalid_array) != cudaErrorInvalidValue ||
        cudaGraphAddMemcpyNode(&upload, graph, nullptr, 0, &upload_params) !=
            cudaSuccess ||
        cudaGraphAddMemcpyNode(&array_copy, graph, &upload, 1, &array_params) !=
            cudaSuccess ||
        cudaGraphAddMemcpyNode(&download, graph, &array_copy, 1,
                               &download_params) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph array-copy node construction failed\n");
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph array-copy replay failed\n");
        return false;
    }
    for (std::size_t z = 0; z < depth; ++z) {
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                const std::size_t destination_index = z * width * height + y * width + x;
                const std::uint16_t expected = x < 3 && y < 2
                    ? source[z * width * height + (y + 1) * width + (x + 1)]
                    : 0;
                if (result[destination_index] != expected) {
                    std::fprintf(stderr,
                                 "FAIL: graph array-copy mismatch at (%zu,%zu,%zu)\n",
                                 x, y, z);
                    return false;
                }
            }
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFreeArray(destination_array);
    cudaFreeArray(source_array);

    cudaArray_t invalid_allocation = reinterpret_cast<cudaArray_t>(1);
    const cudaChannelFormatDesc empty_desc =
        cudaCreateChannelDesc(0, 0, 0, 0, cudaChannelFormatKindUnsigned);
    if (cudaMallocArray(&invalid_allocation, &empty_desc, 1, 1, 0) !=
            cudaErrorInvalidValue ||
        invalid_allocation != nullptr ||
        cudaMalloc3DArray(&invalid_allocation, &desc,
                          make_cudaExtent(static_cast<std::size_t>(-1), 2, 2),
                          0) != cudaErrorInvalidValue ||
        invalid_allocation != nullptr) {
        std::fprintf(stderr, "FAIL: invalid or overflowing array allocation accepted\n");
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

static bool test_event_linked_capture_lifetime() {
    cudaStream_t origin = nullptr;
    cudaStream_t joined = nullptr;
    cudaEvent_t event = nullptr;
    cudaStreamCreate(&origin);
    cudaStreamCreate(&joined);
    cudaEventCreate(&event);

    std::uint32_t* device_value = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_value), sizeof(*device_value)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: event-linked capture allocation failed\n");
        return false;
    }

    if (cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal) != cudaSuccess ||
        cudaMemsetAsync(device_value, 0x11, sizeof(*device_value), origin) != cudaSuccess ||
        cudaEventRecord(event, origin) != cudaSuccess ||
        cudaStreamWaitEvent(joined, event, 0) != cudaSuccess ||
        cudaMemsetAsync(device_value, 0x22, sizeof(*device_value), joined) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: event-linked stream capture setup failed\n");
        return false;
    }

    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(joined, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusActive) {
        std::fprintf(stderr, "FAIL: event wait did not join the active capture\n");
        return false;
    }

    cudaGraph_t graph = nullptr;
    if (cudaStreamEndCapture(origin, &graph) != cudaSuccess || graph == nullptr ||
        cudaStreamIsCapturing(joined, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "FAIL: ending capture did not release joined stream\n");
        return false;
    }

    size_t node_count = 0;
    cudaGraphExec_t exec = nullptr;
    std::uint32_t value = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess ||
        node_count != 2 ||
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphLaunch(exec, origin) != cudaSuccess ||
        cudaStreamSynchronize(origin) != cudaSuccess ||
        cudaMemcpy(&value, device_value, sizeof(value), cudaMemcpyDeviceToHost) !=
            cudaSuccess ||
        value != 0x22222222u) {
        std::fprintf(stderr,
                     "FAIL: event-linked cross-stream replay nodes=%zu value=0x%08x\n",
                     node_count, value);
        return false;
    }

    if (cudaStreamWaitEvent(joined, event, 0) != cudaSuccess ||
        cudaStreamIsCapturing(joined, &status) != cudaSuccess ||
        status != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "FAIL: stale captured event resurrected its graph\n");
        return false;
    }

    cudaGraph_t origin_graph = nullptr;
    cudaGraph_t joined_graph = nullptr;
    if (cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal) != cudaSuccess ||
        cudaStreamBeginCapture(joined, cudaStreamCaptureModeGlobal) != cudaSuccess ||
        cudaEventRecord(event, origin) != cudaSuccess ||
        cudaStreamWaitEvent(joined, event, 0) != cudaErrorInvalidValue ||
        cudaStreamEndCapture(origin, &origin_graph) != cudaSuccess ||
        cudaStreamEndCapture(joined, &joined_graph) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: conflicting event-linked captures were not rejected\n");
        return false;
    }

    cudaGraphDestroy(joined_graph);
    cudaGraphDestroy(origin_graph);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(device_value);
    cudaEventDestroy(event);
    cudaStreamDestroy(joined);
    cudaStreamDestroy(origin);
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

static bool test_graph_dependencies_and_updates() {
    cudaGraph_t graph = nullptr;
    cudaGraph_t other_graph = nullptr;
    cudaGraphCreate(&graph, 0);
    cudaGraphCreate(&other_graph, 0);

    int callback_count = 0;
    cudaHostNodeParams host_params{graph_host_increment, &callback_count};
    cudaGraphNode_t root = nullptr;
    cudaGraphNode_t child = nullptr;
    cudaGraphNode_t foreign = nullptr;
    if (cudaGraphAddHostNode(&root, graph, nullptr, 0, &host_params) != cudaSuccess ||
        cudaGraphAddHostNode(&child, graph, &root, 1, &host_params) != cudaSuccess ||
        cudaGraphAddHostNode(&foreign, other_graph, nullptr, 0, &host_params) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: canonical host-node construction failed\n");
        return false;
    }

    size_t root_count = 0;
    if (cudaGraphGetRootNodes(graph, nullptr, &root_count) != cudaSuccess || root_count != 1) {
        std::fprintf(stderr, "FAIL: dependency graph should expose exactly one root\n");
        return false;
    }
    cudaGraphNode_t invalid = nullptr;
    if (cudaGraphAddHostNode(&invalid, graph, &foreign, 1, &host_params) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: dependency from a foreign graph was accepted\n");
        return false;
    }

    cudaKernelNodeParams kernel_params{};
    kernel_params.func = reinterpret_cast<const void*>(&graph_host_increment);
    kernel_params.gridDim = dim3(1, 1, 1);
    kernel_params.blockDim = dim3(1, 1, 1);
    cudaGraphNode_t kernel_node = nullptr;
    if (cudaGraphAddKernelNode(&kernel_node, graph, &child, 1, &kernel_params) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: kernel-node construction failed\n");
        return false;
    }
    if (cudaGraphKernelNodeSetParams(kernel_node, &kernel_params) != cudaSuccess ||
        cudaGraphKernelNodeSetParams(child, &kernel_params) != cudaErrorInvalidValue ||
        cudaGraphKernelNodeSetParams(kernel_node, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph kernel-node update contract mismatch\n");
        return false;
    }
    cudaGraph_t cloned = nullptr;
    size_t cloned_root_count = 0;
    if (cudaGraphClone(&cloned, graph) != cudaSuccess ||
        cudaGraphGetRootNodes(cloned, nullptr, &cloned_root_count) != cudaSuccess ||
        cloned_root_count != 1) {
        std::fprintf(stderr, "FAIL: graph clone did not preserve dependency topology\n");
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphExecKernelNodeSetParams(exec, kernel_node, &kernel_params) != cudaSuccess ||
        cudaGraphExecKernelNodeSetParams(exec, child, &kernel_params) != cudaErrorInvalidValue ||
        cudaGraphExecKernelNodeSetParams(exec, kernel_node, nullptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: executable kernel-node update contract mismatch\n");
        return false;
    }
    cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
    if (cudaGraphExecUpdate(exec, cloned, nullptr, &update_result) != cudaSuccess ||
        update_result != cudaGraphExecUpdateSuccess) {
        std::fprintf(stderr, "FAIL: same-topology executable graph update failed\n");
        return false;
    }
    cudaGraphNode_t cloned_extra = nullptr;
    if (cudaGraphAddHostNode(&cloned_extra, cloned, nullptr, 0, &host_params) != cudaSuccess ||
        cudaGraphExecUpdate(exec, cloned, nullptr, &update_result) != cudaSuccess ||
        update_result != cudaGraphExecUpdateErrorTopologyChanged) {
        std::fprintf(stderr, "FAIL: topology-changing graph update was not rejected\n");
        return false;
    }
    cudaGraphExecUpdateResultInfo update_info{};
    if (cudaGraphExecUpdate(exec, cloned, &update_info) !=
            cudaErrorGraphExecUpdateFailure ||
        update_info.result != cudaGraphExecUpdateErrorTopologyChanged) {
        std::fprintf(stderr,
                     "FAIL: CUDA 12 graph update did not return the update-failure error\n");
        return false;
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(cloned);
    cudaGraphDestroy(other_graph);
    cudaGraphDestroy(graph);
    return true;
}

static bool test_graph_parameter_setters() {
    unsigned char graph_source[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    unsigned char exec_source[8] = {9, 10, 11, 12, 13, 14, 15, 16};
    unsigned char copied[8] = {};
    void* copy_target = nullptr;
    void* memset_target = nullptr;
    if (cudaMalloc(&copy_target, sizeof(copied)) != cudaSuccess ||
        cudaMalloc(&memset_target, sizeof(copied)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph setter allocation failed\n");
        return false;
    }

    cudaGraph_t graph = nullptr;
    cudaGraphNode_t copy_node = nullptr;
    cudaGraphNode_t memset_node = nullptr;
    cudaGraphNode_t host_node = nullptr;
    int graph_callback_count = 0;
    int exec_callback_count = 0;
    cudaMemsetParams memset_params{};
    memset_params.dst = memset_target;
    memset_params.value = 0x22;
    memset_params.elementSize = 1;
    memset_params.width = sizeof(copied);
    memset_params.height = 1;
    cudaHostNodeParams host_params{graph_host_increment, &graph_callback_count};
    if (cudaGraphCreate(&graph, 0) != cudaSuccess ||
        cudaGraphAddMemcpyNode1D(&copy_node, graph, nullptr, 0, copy_target,
                                 graph_source, sizeof(graph_source),
                                 cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaGraphAddMemsetNode(&memset_node, graph, &copy_node, 1,
                               &memset_params) != cudaSuccess ||
        cudaGraphAddHostNode(&host_node, graph, &memset_node, 1,
                             &host_params) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph setter node construction failed\n");
        return false;
    }

    cudaMemcpy3DParms pitched{};
    pitched.srcPtr = make_cudaPitchedPtr(graph_source, sizeof(graph_source),
                                         sizeof(graph_source), 1);
    pitched.dstPtr = make_cudaPitchedPtr(copy_target, sizeof(copied),
                                         sizeof(copied), 1);
    pitched.extent = make_cudaExtent(sizeof(graph_source), 1, 1);
    pitched.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3DParms invalid_pitched = pitched;
    invalid_pitched.extent.width = 0;
    cudaMemsetParams invalid_memset = memset_params;
    invalid_memset.elementSize = 3;
    cudaHostNodeParams invalid_host{nullptr, nullptr};
    if (cudaGraphMemcpyNodeSetParams(copy_node, &pitched) != cudaSuccess ||
        cudaGraphMemcpyNodeSetParams(copy_node, &invalid_pitched) !=
            cudaErrorInvalidValue ||
        cudaGraphMemcpyNodeSetParams(host_node, &pitched) != cudaErrorInvalidValue ||
        cudaGraphMemsetNodeSetParams(memset_node, &memset_params) != cudaSuccess ||
        cudaGraphMemsetNodeSetParams(memset_node, &invalid_memset) !=
            cudaErrorInvalidValue ||
        cudaGraphHostNodeSetParams(host_node, &host_params) != cudaSuccess ||
        cudaGraphHostNodeSetParams(host_node, &invalid_host) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: mutable graph-node validation mismatch\n");
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    cudaHostNodeParams exec_host_params{graph_host_increment, &exec_callback_count};
    cudaMemsetParams exec_memset_params = memset_params;
    exec_memset_params.value = 0x7b;
    if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphExecMemcpyNodeSetParams1D(
            exec, copy_node, copy_target, exec_source, sizeof(exec_source),
            cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaGraphExecMemcpyNodeSetParams(exec, copy_node, &invalid_pitched) !=
            cudaErrorInvalidValue ||
        cudaGraphExecMemcpyNodeSetParams1D(
            exec, host_node, copy_target, exec_source, sizeof(exec_source),
            cudaMemcpyHostToDevice) != cudaErrorInvalidValue ||
        cudaGraphExecMemsetNodeSetParams(exec, memset_node, &exec_memset_params) !=
            cudaSuccess ||
        cudaGraphExecMemsetNodeSetParams(exec, copy_node, &exec_memset_params) !=
            cudaErrorInvalidValue ||
        cudaGraphExecHostNodeSetParams(exec, host_node, &exec_host_params) !=
            cudaSuccess ||
        cudaGraphExecHostNodeSetParams(exec, memset_node, &exec_host_params) !=
            cudaErrorInvalidValue ||
        cudaGraphLaunch(exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(copied, copy_target, sizeof(copied), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
        std::fprintf(stderr, "FAIL: executable graph-node parameter update failed\n");
        return false;
    }

    unsigned char memset_bytes[8] = {};
    if (cudaMemcpy(memset_bytes, memset_target, sizeof(memset_bytes),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        std::memcmp(copied, exec_source, sizeof(copied)) != 0 ||
        graph_callback_count != 0 || exec_callback_count != 1) {
        std::fprintf(stderr, "FAIL: executable graph-node updates did not replay\n");
        return false;
    }
    for (unsigned char value : memset_bytes) {
        if (value != 0x7b) {
            std::fprintf(stderr, "FAIL: executable memset update wrote 0x%02x\n", value);
            return false;
        }
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaFree(memset_target);
    cudaFree(copy_target);
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

static bool test_capture_host_replay() {
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    int callback_count = 0;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    if (cudaLaunchHostFunc(stream, graph_host_increment, &callback_count) != cudaSuccess ||
        callback_count != 0) {
        std::fprintf(stderr, "FAIL: captured host function executed eagerly\n");
        return false;
    }
    cudaGraph_t graph = nullptr;
    cudaStreamEndCapture(stream, &graph);
    cudaGraphExec_t exec = nullptr;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);
    if (callback_count != 1) {
        std::fprintf(stderr, "FAIL: captured host function did not replay exactly once\n");
        return false;
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    return true;
}

static bool test_graph_memory_nodes() {
    std::uint64_t reserved_before = 0;
    if (cudaDeviceGetGraphMemAttribute(
            0, cudaGraphMemAttrReservedMemCurrent, &reserved_before) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: initial graph-memory attribute query failed\n");
        return false;
    }

    cudaGraph_t graph = nullptr;
    if (cudaGraphCreate(&graph, 0) != cudaSuccess) return false;

    cudaMemAccessDesc access{};
    access.location.type = cudaMemLocationTypeDevice;
    access.location.id = 0;
    access.flags = cudaMemAccessFlagsProtReadWrite;
    cudaMemAllocNodeParams alloc{};
    alloc.poolProps.allocType = cudaMemAllocationTypePinned;
    alloc.poolProps.handleTypes = cudaMemHandleTypeNone;
    alloc.poolProps.location.type = cudaMemLocationTypeDevice;
    alloc.poolProps.location.id = 0;
    alloc.accessDescs = &access;
    alloc.accessDescCount = 1;
    alloc.bytesize = 64;

    cudaGraphNode_t invalid_alloc_node = nullptr;
    cudaMemAllocNodeParams invalid_alloc = alloc;
    invalid_alloc.poolProps.location.id = 1;
    if (cudaGraphAddMemAllocNode(&invalid_alloc_node, graph, nullptr, 0,
                                 &invalid_alloc) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph allocation accepted an invalid device\n");
        return false;
    }
    invalid_alloc = alloc;
    invalid_alloc.accessDescs->flags = cudaMemAccessFlagsProtNone;
    if (cudaGraphAddMemAllocNode(&invalid_alloc_node, graph, nullptr, 0,
                                 &invalid_alloc) != cudaErrorNotSupported) {
        std::fprintf(stderr, "FAIL: graph allocation accepted unsupported access\n");
        return false;
    }
    access.flags = cudaMemAccessFlagsProtReadWrite;

    cudaGraphNode_t alloc_node = nullptr;
    if (cudaGraphAddMemAllocNode(&alloc_node, graph, nullptr, 0, &alloc) !=
            cudaSuccess ||
        alloc.dptr == nullptr) {
        std::fprintf(stderr, "FAIL: graph allocation node creation failed\n");
        return false;
    }
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    cudaMemAllocNodeParams queried{};
    if (cudaGraphNodeGetType(alloc_node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeMemAlloc ||
        cudaGraphMemAllocNodeGetParams(alloc_node, &queried) != cudaSuccess ||
        queried.dptr != alloc.dptr || queried.bytesize != alloc.bytesize ||
        queried.accessDescCount != 1) {
        std::fprintf(stderr, "FAIL: graph allocation metadata mismatch\n");
        return false;
    }
    cudaGraphNode_t invalid_free = nullptr;
    if (cudaGraphAddMemFreeNode(&invalid_free, graph, nullptr, 0, alloc.dptr) !=
        cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: unordered graph free node was accepted\n");
        return false;
    }

    cudaMemsetParams memset{};
    memset.dst = alloc.dptr;
    memset.value = 0x5a;
    memset.elementSize = 1;
    memset.width = alloc.bytesize;
    memset.height = 1;
    cudaGraphNode_t memset_node = nullptr;
    if (cudaGraphAddMemsetNode(&memset_node, graph, &alloc_node, 1, &memset) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph memset after allocation failed\n");
        return false;
    }

    unsigned char copied[64] = {};
    cudaGraphNode_t copy_node = nullptr;
    if (cudaGraphAddMemcpyNode1D(&copy_node, graph, &memset_node, 1, copied,
                                 alloc.dptr, sizeof(copied),
                                 cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaGraphAddMemcpyNode1D(&invalid_alloc_node, graph, &memset_node, 1,
                                 nullptr, alloc.dptr, sizeof(copied),
                                 cudaMemcpyDeviceToHost) != cudaErrorInvalidValue ||
        cudaGraphAddMemcpyNode1D(
            &invalid_alloc_node, graph, &memset_node, 1, copied, alloc.dptr,
            sizeof(copied), static_cast<cudaMemcpyKind>(99)) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph 1D memcpy-node validation failed\n");
        return false;
    }
    cudaGraphNode_t free_node = nullptr;
    if (cudaGraphAddMemFreeNode(&free_node, graph, &copy_node, 1, alloc.dptr) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: ordered graph free node creation failed\n");
        return false;
    }
    void* queried_free = nullptr;
    if (cudaGraphNodeGetType(free_node, &type) != cudaSuccess ||
        type != cudaGraphNodeTypeMemFree ||
        cudaGraphMemFreeNodeGetParams(free_node, &queried_free) != cudaSuccess ||
        queried_free != alloc.dptr ||
        cudaGraphAddMemFreeNode(&invalid_free, graph, &copy_node, 1, alloc.dptr) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph free-node validation mismatch\n");
        return false;
    }

    cudaGraph_t clone = nullptr;
    if (cudaGraphClone(&clone, graph) != cudaErrorNotSupported) {
        std::fprintf(stderr, "FAIL: graph containing memory nodes was cloned\n");
        return false;
    }
    cudaGraphExec_t exec = nullptr;
    cudaGraphExec_t duplicate_exec = nullptr;
    cudaGraphExec_t invalid_flags_exec = nullptr;
    if (cudaGraphInstantiateWithFlags(&invalid_flags_exec, graph, 2ULL) !=
            cudaErrorInvalidValue ||
        invalid_flags_exec != nullptr ||
        cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess ||
        cudaGraphInstantiate(&duplicate_exec, graph, nullptr, nullptr, 0) !=
            cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph-memory instantiation contract mismatch\n");
        return false;
    }

    std::uint64_t reserved_during = 0;
    std::uint64_t used = 0;
    if (cudaDeviceGetGraphMemAttribute(
            0, cudaGraphMemAttrReservedMemCurrent, &reserved_during) != cudaSuccess ||
        reserved_during < reserved_before + alloc.bytesize ||
        cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used) !=
            cudaSuccess ||
        used != 0) {
        std::fprintf(stderr, "FAIL: graph-memory pre-launch counters mismatch\n");
        return false;
    }
    if (cudaGraphLaunch(exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph allocation/free replay failed\n");
        return false;
    }
    for (unsigned char value : copied) {
        if (value != 0x5a) {
            std::fprintf(stderr, "FAIL: graph allocation contents were not preserved\n");
            return false;
        }
    }
    if (cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used) !=
            cudaSuccess ||
        used != 0 || cudaFree(alloc.dptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph free did not end allocation lifetime\n");
        return false;
    }

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    std::uint64_t reserved_after = 0;
    if (cudaDeviceGraphMemTrim(0) != cudaSuccess ||
        cudaDeviceGetGraphMemAttribute(
            0, cudaGraphMemAttrReservedMemCurrent, &reserved_after) != cudaSuccess ||
        reserved_after != reserved_before) {
        std::fprintf(stderr, "FAIL: graph reserved memory was not released\n");
        return false;
    }

    // An allocation without a free node remains live after graph destruction
    // until cudaFree, matching CUDA's externally freed graph-allocation path.
    cudaGraph_t external_graph = nullptr;
    cudaGraphCreate(&external_graph, 0);
    cudaMemAllocNodeParams external_alloc = alloc;
    external_alloc.dptr = nullptr;
    cudaGraphNode_t external_node = nullptr;
    cudaGraphExec_t external_exec = nullptr;
    if (cudaGraphAddMemAllocNode(&external_node, external_graph, nullptr, 0,
                                 &external_alloc) != cudaSuccess ||
        cudaGraphInstantiate(&external_exec, external_graph, nullptr, nullptr, 0) !=
            cudaSuccess ||
        cudaGraphLaunch(external_exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: externally freed graph allocation did not launch\n");
        return false;
    }
    cudaGraphExecDestroy(external_exec);
    cudaGraphDestroy(external_graph);
    if (cudaMemset(external_alloc.dptr, 0x33, external_alloc.bytesize) != cudaSuccess ||
        cudaFree(external_alloc.dptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: graph allocation did not outlive its graph\n");
        return false;
    }

    // A free-only graph may release an allocation node owned by a different
    // graph. The free graph can be instantiated before the allocation becomes
    // live, as in NVIDIA's graphMemoryNodes sample.
    cudaGraph_t owner_graph = nullptr;
    cudaGraph_t free_graph = nullptr;
    cudaGraphExec_t owner_exec = nullptr;
    cudaGraphExec_t free_exec = nullptr;
    cudaMemAllocNodeParams cross_alloc = alloc;
    cross_alloc.dptr = nullptr;
    cudaGraphNode_t owner_node = nullptr;
    cudaGraphNode_t cross_free_node = nullptr;
    cudaGraphNode_t conflicting_free_node = nullptr;
    if (cudaGraphCreate(&owner_graph, 0) != cudaSuccess ||
        cudaGraphAddMemAllocNode(&owner_node, owner_graph, nullptr, 0,
                                 &cross_alloc) != cudaSuccess ||
        cudaGraphInstantiate(&owner_exec, owner_graph, nullptr, nullptr, 0) !=
            cudaSuccess ||
        cudaGraphCreate(&free_graph, 0) != cudaSuccess ||
        cudaGraphAddMemFreeNode(&cross_free_node, free_graph, nullptr, 0,
                                cross_alloc.dptr) != cudaSuccess ||
        cudaGraphAddMemFreeNode(&conflicting_free_node, owner_graph, &owner_node, 1,
                                cross_alloc.dptr) != cudaErrorInvalidValue ||
        cudaGraphInstantiate(&free_exec, free_graph, nullptr, nullptr, 0) !=
            cudaSuccess ||
        cudaGraphLaunch(owner_exec, nullptr) != cudaSuccess ||
        cudaGraphLaunch(owner_exec, nullptr) != cudaErrorInvalidValue ||
        cudaFreeAsync(cross_alloc.dptr, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaGraphLaunch(owner_exec, nullptr) != cudaSuccess ||
        cudaGraphLaunch(free_exec, nullptr) != cudaSuccess ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaFree(cross_alloc.dptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: cross-graph allocation free failed\n");
        return false;
    }
    cudaGraphExecDestroy(free_exec);
    cudaGraphDestroy(free_graph);
    cudaGraphExecDestroy(owner_exec);
    cudaGraphDestroy(owner_graph);

    // Auto-free instantiation ends otherwise-unfreed allocations after every
    // launch and permits the same executable to launch again at the fixed address.
    cudaGraph_t auto_graph = nullptr;
    cudaGraphCreate(&auto_graph, 0);
    cudaMemAllocNodeParams auto_alloc = alloc;
    auto_alloc.dptr = nullptr;
    cudaGraphNode_t auto_node = nullptr;
    cudaGraphExec_t auto_exec = nullptr;
    if (cudaGraphAddMemAllocNode(&auto_node, auto_graph, nullptr, 0, &auto_alloc) !=
            cudaSuccess ||
        cudaGraphInstantiateWithFlags(
            &auto_exec, auto_graph, cudaGraphInstantiateFlagAutoFreeOnLaunch) !=
            cudaSuccess ||
        cudaGraphLaunch(auto_exec, nullptr) != cudaSuccess ||
        cudaGraphLaunch(auto_exec, nullptr) != cudaSuccess ||
        cudaDeviceGetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &used) !=
            cudaSuccess ||
        used != 0 || cudaFree(auto_alloc.dptr) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph auto-free-on-launch contract mismatch\n");
        return false;
    }
    cudaGraphExecDestroy(auto_exec);
    cudaGraphDestroy(auto_graph);

    if (cudaDeviceGetGraphMemAttribute(
            0, cudaGraphMemAttrReservedMemCurrent, &reserved_after) != cudaSuccess ||
        reserved_after != reserved_before) {
        std::fprintf(stderr, "FAIL: graph allocation lifetime leaked reserved memory\n");
        return false;
    }

    std::uint64_t zero = 0;
    std::uint64_t one = 1;
    if (cudaDeviceSetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &zero) !=
            cudaSuccess ||
        cudaDeviceSetGraphMemAttribute(
            0, cudaGraphMemAttrReservedMemHigh, &zero) != cudaSuccess ||
        cudaDeviceSetGraphMemAttribute(0, cudaGraphMemAttrUsedMemCurrent, &zero) !=
            cudaErrorInvalidValue ||
        cudaDeviceSetGraphMemAttribute(0, cudaGraphMemAttrUsedMemHigh, &one) !=
            cudaErrorInvalidValue ||
        cudaDeviceGetGraphMemAttribute(-1, cudaGraphMemAttrUsedMemCurrent, &zero) !=
            cudaErrorInvalidValue ||
        cudaDeviceGraphMemTrim(1) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "FAIL: graph-memory high-water reset failed\n");
        return false;
    }
    return true;
}

int main() {
    if (!test_graph_create_destroy()) return 1;
    if (!test_graph_instantiate_launch()) return 1;
    if (!test_stream_capture_status()) return 1;
    if (!test_event_linked_capture_lifetime()) return 1;
    if (!test_graph_null_args()) return 1;
    if (!test_graph_dependencies_and_updates()) return 1;
    if (!test_graph_parameter_setters()) return 1;
    if (!test_graph_pitched_copy_and_typed_memset()) return 1;
    if (!test_graph_array_copy_geometry()) return 1;
    if (!test_capture_memcpy_replay()) return 1;
    if (!test_capture_memset_replay()) return 1;
    if (!test_capture_host_replay()) return 1;
    if (!test_graph_memory_nodes()) return 1;

    std::printf("PASS: CUDA Graph API tests\n");
    return 0;
}
