// Graph-owned user objects, graph upload, DOT export and memory-pool access.
//
// These are the driver and runtime entry points a host reaches for once it
// drives CUDA graphs itself rather than through the runtime's convenience
// paths: NVIDIA Warp attaches its per-graph allocation bookkeeping to a user
// object so the graph's destruction frees it. The reference counting is the
// part worth testing -- a destructor that runs early frees memory the graph
// still replays through, and one that never runs leaks per-graph state for the
// life of the process.
#include "cuda_runtime.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

int g_destroy_calls = 0;
void* g_destroy_payload = nullptr;

void count_destroy(void* payload) {
    ++g_destroy_calls;
    g_destroy_payload = payload;
}

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

void host_noop(void*) {}

// A graph with two host nodes, the second depending on the first, so the DOT
// export has both a node list and an edge to render.
cudaError_t build_two_node_graph(cudaGraph_t* graph) {
    const cudaError_t created = cudaGraphCreate(graph, 0);
    if (created != cudaSuccess) return created;

    cudaHostNodeParams params{};
    params.fn = host_noop;
    params.userData = nullptr;

    cudaGraphNode_t first = nullptr;
    const cudaError_t first_added = cudaGraphAddHostNode(&first, *graph, nullptr, 0, &params);
    if (first_added != cudaSuccess) return first_added;

    cudaGraphNode_t second = nullptr;
    return cudaGraphAddHostNode(&second, *graph, &first, 1, &params);
}

}  // namespace

int main() {
    int payload = 7;

    // The move flag transfers the caller's reference: the graph ends up holding
    // the only one, and destroying the graph runs the destructor exactly once.
    {
        g_destroy_calls = 0;
        g_destroy_payload = nullptr;

        cudaGraph_t graph = nullptr;
        if (!expect(cudaGraphCreate(&graph, 0) == cudaSuccess, "cudaGraphCreate")) return 1;

        cudaUserObject_t object = nullptr;
        if (!expect(cudaUserObjectCreate(&object, &payload, count_destroy, 1,
                                         cudaUserObjectNoDestructorSync) == cudaSuccess &&
                        object != nullptr,
                    "cudaUserObjectCreate")) {
            return 1;
        }
        if (!expect(cudaGraphRetainUserObject(graph, object, 1, cudaGraphUserObjectMove) ==
                        cudaSuccess,
                    "cudaGraphRetainUserObject with move")) {
            return 1;
        }
        if (!expect(g_destroy_calls == 0, "retaining does not destroy")) return 1;

        if (!expect(cudaGraphDestroy(graph) == cudaSuccess, "cudaGraphDestroy")) return 1;
        if (!expect(g_destroy_calls == 1 && g_destroy_payload == &payload,
                    "destroying the graph runs the destructor once, on the payload")) {
            return 1;
        }
    }

    // Without the move flag the graph takes a reference of its own, so the
    // caller's outlives the graph and the destructor waits for it.
    {
        g_destroy_calls = 0;

        cudaGraph_t graph = nullptr;
        if (!expect(cudaGraphCreate(&graph, 0) == cudaSuccess, "cudaGraphCreate")) return 1;

        cudaUserObject_t object = nullptr;
        if (!expect(cudaUserObjectCreate(&object, &payload, count_destroy, 1, 0) == cudaSuccess,
                    "cudaUserObjectCreate without flags")) {
            return 1;
        }
        if (!expect(cudaGraphRetainUserObject(graph, object, 1, 0) == cudaSuccess,
                    "cudaGraphRetainUserObject without move")) {
            return 1;
        }
        if (!expect(cudaGraphDestroy(graph) == cudaSuccess, "cudaGraphDestroy")) return 1;
        if (!expect(g_destroy_calls == 0, "the caller's own reference keeps the object alive")) {
            return 1;
        }
        if (!expect(cudaUserObjectRelease(object, 1) == cudaSuccess, "cudaUserObjectRelease")) {
            return 1;
        }
        if (!expect(g_destroy_calls == 1, "releasing the last reference destroys")) return 1;
    }

    // An explicit release drops exactly the references the graph holds, and
    // over-releasing is a caller error rather than a silent double free.
    {
        g_destroy_calls = 0;

        cudaGraph_t graph = nullptr;
        if (!expect(cudaGraphCreate(&graph, 0) == cudaSuccess, "cudaGraphCreate")) return 1;

        cudaUserObject_t object = nullptr;
        if (!expect(cudaUserObjectCreate(&object, &payload, count_destroy, 2, 0) == cudaSuccess,
                    "cudaUserObjectCreate with two references")) {
            return 1;
        }
        if (!expect(cudaGraphRetainUserObject(graph, object, 2, cudaGraphUserObjectMove) ==
                        cudaSuccess,
                    "cudaGraphRetainUserObject with two moved references")) {
            return 1;
        }
        if (!expect(cudaGraphReleaseUserObject(graph, object, 3) == cudaErrorInvalidValue,
                    "releasing more references than the graph holds is rejected")) {
            return 1;
        }
        if (!expect(g_destroy_calls == 0, "a rejected release destroys nothing")) return 1;
        if (!expect(cudaGraphReleaseUserObject(graph, object, 2) == cudaSuccess,
                    "cudaGraphReleaseUserObject")) {
            return 1;
        }
        if (!expect(g_destroy_calls == 1, "the release ran the destructor")) return 1;
        // The graph no longer holds the object, so destroying it must not
        // release again.
        if (!expect(cudaGraphDestroy(graph) == cudaSuccess, "cudaGraphDestroy after release")) {
            return 1;
        }
        if (!expect(g_destroy_calls == 1, "destroying the graph does not double-release")) {
            return 1;
        }
    }

    // Negative paths.
    {
        cudaUserObject_t object = nullptr;
        if (!expect(cudaUserObjectCreate(&object, &payload, nullptr, 1, 0) == cudaErrorInvalidValue,
                    "a user object requires a destructor")) {
            return 1;
        }
        if (!expect(cudaUserObjectCreate(&object, &payload, count_destroy, 0, 0) ==
                        cudaErrorInvalidValue,
                    "a user object requires a non-zero initial refcount")) {
            return 1;
        }
        if (!expect(cudaUserObjectRetain(nullptr, 1) == cudaErrorInvalidValue,
                    "retaining a null object is rejected")) {
            return 1;
        }
    }

    // cudaGraphUpload has nothing to stage on CuMetal, but it must accept a
    // real executable graph and reject a null one.
    {
        cudaGraph_t graph = nullptr;
        if (!expect(build_two_node_graph(&graph) == cudaSuccess, "build graph for upload")) {
            return 1;
        }
        cudaGraphExec_t exec = nullptr;
        if (!expect(cudaGraphInstantiateWithFlags(&exec, graph, 0) == cudaSuccess,
                    "cudaGraphInstantiateWithFlags")) {
            return 1;
        }
        if (!expect(cudaGraphUpload(exec, nullptr) == cudaSuccess, "cudaGraphUpload")) return 1;
        if (!expect(cudaGraphUpload(nullptr, nullptr) == cudaErrorInvalidValue,
                    "uploading a null executable graph is rejected")) {
            return 1;
        }
        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(graph);
    }

    // The DOT export names every node and the edge between them.
    {
        cudaGraph_t graph = nullptr;
        if (!expect(build_two_node_graph(&graph) == cudaSuccess, "build graph for DOT export")) {
            return 1;
        }
        const std::filesystem::path dot_path =
            std::filesystem::temp_directory_path() / "cumetal-graph-user-object-test.dot";
        std::error_code ec;
        std::filesystem::remove(dot_path, ec);

        if (!expect(cudaGraphDebugDotPrint(graph, dot_path.c_str(), 0) == cudaSuccess,
                    "cudaGraphDebugDotPrint")) {
            return 1;
        }
        std::ifstream dot(dot_path);
        const std::string text((std::istreambuf_iterator<char>(dot)),
                               std::istreambuf_iterator<char>());
        if (!expect(text.find("digraph") != std::string::npos &&
                        text.find("node0") != std::string::npos &&
                        text.find("node1") != std::string::npos &&
                        text.find("node0 -> node1") != std::string::npos &&
                        text.find("host") != std::string::npos,
                    "the DOT file describes both nodes and their edge")) {
            std::fprintf(stderr, "  dot:\n%s\n", text.c_str());
            return 1;
        }
        std::filesystem::remove(dot_path, ec);

        if (!expect(cudaGraphDebugDotPrint(graph, "/nonexistent-directory/graph.dot", 0) ==
                        cudaErrorOperatingSystem,
                    "an unwritable path is reported, not ignored")) {
            return 1;
        }
        if (!expect(cudaGraphDebugDotPrint(nullptr, dot_path.c_str(), 0) == cudaErrorInvalidValue,
                    "a null graph is rejected")) {
            return 1;
        }
        cudaGraphDestroy(graph);
    }

    // Memory-pool access. CuMetal has one device, which always has read-write
    // access to its own pool and cannot have it revoked; no other device exists
    // to grant access to.
    {
        cudaMemPool_t pool = nullptr;
        if (!expect(cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess,
                    "cudaDeviceGetDefaultMemPool")) {
            return 1;
        }

        cudaMemAccessFlags flags = cudaMemAccessFlagsProtNone;
        cudaMemLocation self{};
        self.type = cudaMemLocationTypeDevice;
        self.id = 0;
        if (!expect(cudaMemPoolGetAccess(&flags, pool, &self) == cudaSuccess &&
                        flags == cudaMemAccessFlagsProtReadWrite,
                    "the owning device has read-write access to its pool")) {
            return 1;
        }

        cudaMemLocation peer{};
        peer.type = cudaMemLocationTypeDevice;
        peer.id = 1;
        flags = cudaMemAccessFlagsProtReadWrite;
        if (!expect(cudaMemPoolGetAccess(&flags, pool, &peer) == cudaSuccess &&
                        flags == cudaMemAccessFlagsProtNone,
                    "a device that does not exist has no access")) {
            return 1;
        }

        cudaMemAccessDesc desc{};
        desc.location = self;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        if (!expect(cudaMemPoolSetAccess(pool, &desc, 1) == cudaSuccess,
                    "granting the owning device access it already has succeeds")) {
            return 1;
        }

        desc.flags = cudaMemAccessFlagsProtNone;
        if (!expect(cudaMemPoolSetAccess(pool, &desc, 1) == cudaErrorInvalidValue,
                    "the owning device's access cannot be revoked")) {
            return 1;
        }

        desc.location = peer;
        desc.flags = cudaMemAccessFlagsProtReadWrite;
        if (!expect(cudaMemPoolSetAccess(pool, &desc, 1) == cudaErrorInvalidDevice,
                    "granting access to a device that does not exist is rejected")) {
            return 1;
        }
    }

    std::printf("PASS: graph user objects, upload, DOT export and mempool access\n");
    return 0;
}
