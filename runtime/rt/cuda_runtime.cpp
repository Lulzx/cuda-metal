#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "allocation_table.h"
#include "cumetal_diag.h"
#include "library_conflict.h"
#include "metal_backend.h"
#include "registration.h"
#include "native_registration.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

struct CUstream_st {};

// ── Diagnostic trace (CUMETAL_TRACE=1) ───────────────────────────────────────
// One-shot, line-buffered log of every CUDA op cumetal actually executes, used
// to root-cause which offloaded op silently corrupts output (e.g. llama.cpp at
// NGL=0). Disabled by default; no cost when off.
namespace {

constexpr int kCudaVisibleMultiprocessorCount = 1;
constexpr std::uint64_t kMaxResidentCooperativeBlocks = 4;

extern "C" int cumetalDriverRuntimeActivatePrimaryContext(int device);
bool trace_enabled() {
    static int v = -1;
    if (v < 0) v = cumetal::diag_env_truthy("CUMETAL_TRACE") ? 1 : 0;
    return v == 1;
}
void trace_op(const char* tag, const char* detail) {
    if (!trace_enabled()) return;
    std::fprintf(stderr, "CUMETAL_TRACE %s %s\n", tag, detail ? detail : "");
    std::fflush(stderr);
}

// Uncached parse; callers should go through load_inline_static_shared_bytes below.
bool load_inline_static_shared_bytes_uncached(const char* metallib_path,
                                              const char* expected_kernel,
                                              std::size_t* out_bytes) {
    *out_bytes = 0;
    std::ifstream abi(std::string(metallib_path) + ".cumetal-abi");
    if (!abi) {
        // Prebuilt/legacy metallibs legitimately have no sidecar.
        return true;
    }

    std::string line;
    if (!std::getline(abi, line) || line != "CUMETAL_ABI_V1" ||
        !std::getline(abi, line)) {
        return false;
    }
    {
        std::istringstream kernel_line(line);
        std::string keyword;
        std::string kernel_name;
        std::string extra;
        if (!(kernel_line >> keyword >> kernel_name) || keyword != "kernel" ||
            kernel_name != expected_kernel || (kernel_line >> extra)) {
            return false;
        }
    }

    bool saw_shared = false;
    while (std::getline(abi, line)) {
        if (line.empty()) {
            continue;
        }
        std::istringstream record(line);
        std::string keyword;
        if (!(record >> keyword)) {
            return false;
        }
        if (keyword == "shared") {
            unsigned long long bytes = 0;
            std::string extra;
            if (saw_shared || !(record >> bytes) || (record >> extra) ||
                bytes > 16ull * 1024ull * 1024ull) {
                return false;
            }
            *out_bytes = static_cast<std::size_t>(bytes);
            saw_shared = true;
            continue;
        }
        if (keyword == "arg") {
            std::string kind;
            unsigned long size = 0;
            std::string extra;
            if (!(record >> kind >> size) || (record >> extra) ||
                (kind != "buffer" && kind != "bytes") || size == 0 || size > 4096) {
                return false;
            }
            continue;
        }
        return false;
    }
    return true;
}

struct InlineAbiCacheEntry {
    bool valid = false;
    std::size_t shared_bytes = 0;
    bool file_present = false;
    struct timespec mtime {};
    off_t size = 0;
};

inline struct timespec sidecar_mtime(const struct stat& st) {
#if defined(__APPLE__)
    return st.st_mtimespec;
#else
    return st.st_mtim;
#endif
}

// Caches the sidecar parse per (metallib, kernel), revalidated via stat() so an on-disk change is
// still picked up -- an existing test rewrites a sidecar mid-run and must see the new contents.
//
// The revalidation key is (presence, mtime, size). That is exact on APFS, whose timestamps are
// nanosecond-resolution: two back-to-back rewrites of the same file always land on distinct mtimes,
// so a same-size in-place edit still invalidates. It would NOT be exact on a volume with coarse
// (1s/2s) timestamps -- an SMB/NFS mount or a FAT/exFAT disk -- where a same-size rewrite inside one
// tick would serve a stale shared-byte count, i.e. silently allocate the wrong amount of static
// threadgroup memory. CuMetal is Apple-Silicon-only and metallibs live in APFS build trees, so this
// is accepted rather than hashing the contents (which would defeat the point of the cache).
bool load_inline_static_shared_bytes(const char* metallib_path,
                                     const char* expected_kernel,
                                     std::size_t* out_bytes) {
    if (metallib_path == nullptr || expected_kernel == nullptr || out_bytes == nullptr) {
        return false;
    }
    const std::string sidecar_path = std::string(metallib_path) + ".cumetal-abi";
    struct stat st {};
    const bool file_present = ::stat(sidecar_path.c_str(), &st) == 0;
    const struct timespec mtime = sidecar_mtime(st);

    static std::mutex cache_mutex;
    // Keyed on the pair rather than a concatenation: no separator can collide with a path or a
    // kernel name that happens to contain it.
    static std::map<std::pair<std::string, std::string>, InlineAbiCacheEntry> cache;
    const std::pair<std::string, std::string> cache_key(metallib_path, expected_kernel);

    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const auto found = cache.find(cache_key);
        if (found != cache.end() && found->second.file_present == file_present &&
            (!file_present ||
             (found->second.mtime.tv_sec == mtime.tv_sec &&
              found->second.mtime.tv_nsec == mtime.tv_nsec && found->second.size == st.st_size))) {
            *out_bytes = found->second.shared_bytes;
            return found->second.valid;
        }
    }

    // Parsed outside the lock: a cold miss on one kernel must not serialize launches of every other
    // kernel. Two threads racing the same miss both parse and store the same result, which is fine.
    InlineAbiCacheEntry entry;
    entry.file_present = file_present;
    entry.mtime = mtime;
    entry.size = st.st_size;
    entry.valid =
        load_inline_static_shared_bytes_uncached(metallib_path, expected_kernel, &entry.shared_bytes);
    *out_bytes = entry.shared_bytes;
    {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache[cache_key] = entry;
    }
    return entry.valid;
}
}  // namespace

// ── CUDA Graphs (spec §2.2 — previously deferred, now implemented) ──────────
// Graph nodes record operations; instantiation creates an executable sequence.
// On Apple Silicon (single GPU, UMA), graphs are replayed sequentially.

struct GraphAllocationState;

enum class GraphFreePlacement {
    kNone,
    kOwningGraph,
    kExternalGraph,
};

struct GraphLifetimeState {
    std::mutex mutex;
    bool contains_memory_nodes = false;
    std::size_t executable_instances = 0;
};

struct GraphAllocationState {
    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    void* base = nullptr;
    void* alias = nullptr;
    std::size_t size = 0;
    bool active = false;
    bool reserved_accounted = false;
    GraphFreePlacement free_placement = GraphFreePlacement::kNone;
};

struct cudaGraphNode_st {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    std::vector<cudaGraphNode_st*> dependencies;

    // Kernel node data
    const void* func = nullptr;
    dim3 grid_dim{};
    dim3 block_dim{};
    void** kernel_args = nullptr;
    std::vector<std::vector<std::uint8_t>> kernel_arg_values;
    int num_args = 0;
    size_t shared_mem = 0;

    // Memcpy node data
    void* dst = nullptr;
    const void* src = nullptr;
    size_t count = 0;
    cudaMemcpyKind memcpy_kind = cudaMemcpyDefault;

    // Memset node data
    int memset_value = 0;

    // Host node data
    cudaHostFn_t host_fn = nullptr;
    void* host_user_data = nullptr;

    // Captured library call. A cuSPARSE or cuBLAS entry point launches its work
    // through the backend rather than through cudaLaunchKernel, so there is no
    // cudaKernelNodeParams to record; what gets recorded instead is a closure
    // over the arguments as they stood at capture time. It replays as a kernel
    // node because that is what it is -- CUDA records the kernels a captured
    // library call launches -- and cudaGraphNodeGetType agrees.
    std::function<cudaError_t(cudaStream_t)> library_op;

    // Graph allocation/free node data. The backing buffer is reserved when the
    // allocation node is created so its CUDA address remains fixed, but it is
    // inserted into the live allocation table only when replay reaches the
    // allocation node.
    std::shared_ptr<GraphAllocationState> graph_allocation;
    cudaMemPoolProps graph_pool_props{};
    std::vector<cudaMemAccessDesc> graph_access_descs;
    void* graph_free_ptr = nullptr;
};

struct cudaGraph_st {
    std::vector<cudaGraphNode_st*> nodes;
    std::shared_ptr<GraphLifetimeState> lifetime =
        std::make_shared<GraphLifetimeState>();
    ~cudaGraph_st() {
        for (auto* n : nodes) { delete n; }
    }
};

struct cudaGraphExec_st {
    // Owned topological copy of the graph for replay. source_node_index keeps
    // the CUDA graph-node identity used by executable update APIs.
    std::vector<cudaGraphNode_st> nodes;
    std::unordered_map<const cudaGraphNode_st*, std::size_t> source_node_index;
    std::vector<std::vector<std::size_t>> dependency_indices;
    std::shared_ptr<GraphLifetimeState> lifetime;
    bool auto_free_on_launch = false;
};

struct CUevent_st {
    bool disable_timing = false;
    bool recorded_once = false;
    bool complete = true;
    bool timing_valid = false;
    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    std::chrono::steady_clock::time_point timestamp{};
    // During stream capture an event carries the graph frontier to a stream
    // that waits on it. CUDA uses this to join event-linked streams into the
    // same capture. The graph owns the event-independent node lifetime.
    cudaGraph_t capture_graph = nullptr;
    std::mutex mutex;
};

struct CuMetalArray {
    void* data = nullptr;
    size_t width = 0;
    size_t height = 0;
    size_t depth = 1;
    unsigned int flags = cudaArrayDefault;
    cudaChannelFormatDesc desc{};
};

namespace {

constexpr int kCudaCompatVersion = 12000;
constexpr std::size_t kDefaultPrintfFifoSize = 1024u * 1024u;

struct RuntimeState {
    std::once_flag init_once;
    cudaError_t init_status = cudaSuccess;
    std::string init_error;
    int current_device = 0;
    unsigned int device_flags = cudaDeviceScheduleAuto;
    cumetal::rt::AllocationTable allocations;
    std::mutex pending_free_mutex;
    std::unordered_set<void*> pending_async_frees;
    std::mutex graph_memory_mutex;
    std::unordered_map<void*, std::weak_ptr<GraphAllocationState>> graph_allocations;
    std::unordered_map<void*, std::shared_ptr<GraphAllocationState>>
        live_graph_allocations;
    std::uint64_t graph_used_current = 0;
    std::uint64_t graph_used_high = 0;
    std::uint64_t graph_reserved_current = 0;
    std::uint64_t graph_reserved_high = 0;
    std::mutex stream_mutex;
    struct StreamRecord {
        std::shared_ptr<cumetal::metal_backend::Stream> backend;
        unsigned int flags = cudaStreamDefault;
        cudaStreamAttrValue access_policy{};
    };
    std::unordered_map<cudaStream_t, StreamRecord> streams;
    cudaStreamAttrValue default_stream_access_policy{};
    std::mutex device_heap_mutex;
    std::size_t device_heap_size = 8u * 1024u * 1024u;
    std::shared_ptr<cumetal::metal_backend::Buffer> device_heap;
    std::size_t persisting_l2_limit = 0;
    std::size_t printf_fifo_size = kDefaultPrintfFifoSize;
};

RuntimeState& runtime_state() {
    // Immortal on purpose. This is process-lifetime state guarded by a mutex, and a
    // function-local static gets an atexit destructor: anything that touches it during
    // teardown -- another static's destructor, a detached worker, a Metal completion
    // handler -- then locks a destroyed mutex. That surfaced as an intermittent
    // "mutex lock failed: Invalid argument" abort *after* a test had already printed
    // PASS. Leaking one object at exit is the fix; the OS reclaims it.
    static RuntimeState* state = new RuntimeState();
    return *state;
}

thread_local cudaError_t tls_last_error = cudaSuccess;
// A generated CUDA kernel launch has no return value at the call site: Clang's
// host stub calls cudaLaunchKernel and leaves callers to observe failures at a
// later error or synchronization API.  Pipeline creation can fail before Metal
// has a command buffer to enqueue, so retain that failure separately from the
// ordinary last-error slot until cudaDeviceSynchronize consumes it.
thread_local cudaError_t tls_pending_launch_error = cudaSuccess;
thread_local std::shared_ptr<cumetal::metal_backend::Stream> tls_per_thread_stream;

// Per-stream graph capture state.
struct CaptureState {
    bool capturing = false;
    cudaGraph_t graph = nullptr;
};
std::mutex g_capture_mutex;
std::unordered_map<cudaStream_t, CaptureState> g_captures;

// Check if a stream is being captured; if so, return its graph for recording.
// Caller must hold no lock — this function acquires g_capture_mutex.
cudaGraph_t get_capture_graph(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_captures.find(stream);
    if (it != g_captures.end() && it->second.capturing) {
        return it->second.graph;
    }
    return nullptr;
}

bool graph_contains_node(cudaGraph_t graph, cudaGraphNode_t node) {
    return graph != nullptr && node != nullptr &&
           std::find(graph->nodes.begin(), graph->nodes.end(), node) != graph->nodes.end();
}

bool assign_graph_dependencies(cudaGraph_t graph,
                               cudaGraphNode_t node,
                               const cudaGraphNode_t* dependencies,
                               std::size_t dependency_count) {
    if (graph == nullptr || node == nullptr ||
        (dependency_count != 0 && dependencies == nullptr)) {
        return false;
    }
    node->dependencies.clear();
    node->dependencies.reserve(dependency_count);
    for (std::size_t i = 0; i < dependency_count; ++i) {
        if (!graph_contains_node(graph, dependencies[i]) || dependencies[i] == node ||
            std::find(node->dependencies.begin(), node->dependencies.end(), dependencies[i]) !=
                node->dependencies.end()) {
            node->dependencies.clear();
            return false;
        }
        node->dependencies.push_back(dependencies[i]);
    }
    return true;
}

void append_captured_graph_node(cudaGraph_t graph, cudaGraphNode_t node) {
    // Operations captured from one CUDA stream are ordered. Preserve that edge
    // explicitly so graph introspection agrees with replay semantics.
    if (!graph->nodes.empty()) {
        node->dependencies.push_back(graph->nodes.back());
    }
    graph->nodes.push_back(node);
}

bool topologically_order_graph(cudaGraph_t graph,
                               std::vector<cudaGraphNode_t>* ordered) {
    if (graph == nullptr || ordered == nullptr) return false;
    ordered->clear();
    ordered->reserve(graph->nodes.size());

    std::unordered_map<cudaGraphNode_t, std::size_t> node_index;
    node_index.reserve(graph->nodes.size());
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
        if (graph->nodes[i] == nullptr || !node_index.emplace(graph->nodes[i], i).second) {
            return false;
        }
    }

    std::vector<std::size_t> indegree(graph->nodes.size(), 0);
    std::vector<std::vector<std::size_t>> dependents(graph->nodes.size());
    for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
        for (cudaGraphNode_t dependency : graph->nodes[i]->dependencies) {
            const auto it = node_index.find(dependency);
            if (it == node_index.end()) return false;
            ++indegree[i];
            dependents[it->second].push_back(i);
        }
    }

    // Choose ready nodes in graph insertion order for deterministic replay.
    std::vector<bool> emitted(graph->nodes.size(), false);
    while (ordered->size() != graph->nodes.size()) {
        std::size_t ready = graph->nodes.size();
        for (std::size_t i = 0; i < graph->nodes.size(); ++i) {
            if (!emitted[i] && indegree[i] == 0) {
                ready = i;
                break;
            }
        }
        if (ready == graph->nodes.size()) return false;
        emitted[ready] = true;
        ordered->push_back(graph->nodes[ready]);
        for (std::size_t dependent : dependents[ready]) {
            --indegree[dependent];
        }
    }
    return true;
}

bool snapshot_graph_kernel_arguments(cudaGraphNode_t node,
                                     const void* function,
                                     void** arguments) {
    if (node == nullptr || function == nullptr) return false;

    cumetal::registration::RegisteredKernel kernel;
    const bool registered =
        cumetal::native_registration::lookup_kernel(function, &kernel) ||
        cumetal::registration::lookup_registered_kernel(function, &kernel);
    if (!registered || kernel.arg_info.empty()) {
        // Zero-argument and synthetic nodes need no snapshot. Refuse to retain
        // an untyped caller-owned argv because its lifetime is unknowable.
        node->kernel_args = nullptr;
        node->kernel_arg_values.clear();
        return arguments == nullptr;
    }
    if (arguments == nullptr) return false;

    node->kernel_arg_values.clear();
    node->kernel_arg_values.reserve(kernel.arg_info.size());
    for (std::size_t i = 0; i < kernel.arg_info.size(); ++i) {
        const std::size_t size = kernel.arg_info[i].size_bytes;
        if (arguments[i] == nullptr || size == 0 || size > 64u * 1024u) {
            node->kernel_arg_values.clear();
            return false;
        }
        std::vector<std::uint8_t> value(size);
        std::memcpy(value.data(), arguments[i], size);
        node->kernel_arg_values.push_back(std::move(value));
    }
    node->kernel_args = nullptr;
    return true;
}

struct PendingLaunchArgument {
    size_t offset = 0;
    size_t size = 0;
};

struct PendingLaunchState {
    bool configured = false;
    dim3 grid_dim{};
    dim3 block_dim{};
    size_t shared_mem = 0;
    cudaStream_t stream = nullptr;
    std::vector<std::uint8_t> storage;
    std::vector<PendingLaunchArgument> arguments;
};

thread_local PendingLaunchState tls_pending_launch;

void clear_pending_launch_state() {
    tls_pending_launch.configured = false;
    tls_pending_launch.grid_dim = dim3{};
    tls_pending_launch.block_dim = dim3{};
    tls_pending_launch.shared_mem = 0;
    tls_pending_launch.stream = nullptr;
    tls_pending_launch.storage.clear();
    tls_pending_launch.arguments.clear();
}

void set_last_error(cudaError_t error) {
    tls_last_error = error;
}

void record_pending_launch_error(cudaError_t error) {
    if (error != cudaSuccess && tls_pending_launch_error == cudaSuccess) {
        tls_pending_launch_error = error;
    }
}

cudaError_t take_pending_launch_error() {
    const cudaError_t error = tls_pending_launch_error;
    tls_pending_launch_error = cudaSuccess;
    return error;
}

// ── Device printf drain (spec §5.3) ─────────────────────────────────────────
// Ring-buffer layout (all words are uint32):
//   buf[0]          = atomic write-word-count (total words written after index 0)
//   buf[1..]        = packed records: [fmt_id, payload_words, arg words...]
// Float args are stored as as_type<uint>(f); 64-bit ABI slots occupy low/high
// words so a following 32-bit argument keeps its proper position.

std::uint32_t minimum_printf_words(std::string_view fmt) {
    std::uint32_t words = 0;
    for (std::size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') continue;
        if (++i >= fmt.size()) break;
        if (fmt[i] == '%') continue;
        while (i < fmt.size() && std::strchr("-+ #0", fmt[i]) != nullptr) ++i;
        if (i < fmt.size() && fmt[i] == '*') {
            ++words;
            ++i;
        } else {
            while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) ++i;
        }
        if (i < fmt.size() && fmt[i] == '.') {
            ++i;
            if (i < fmt.size() && fmt[i] == '*') {
                ++words;
                ++i;
            } else {
                while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) ++i;
            }
        }
        bool wide = false;
        while (i < fmt.size() && std::strchr("lhzjt", fmt[i]) != nullptr) {
            wide |= fmt[i] == 'l' || fmt[i] == 'z' || fmt[i] == 'j' || fmt[i] == 't';
            ++i;
        }
        if (i >= fmt.size()) break;
        words += wide || fmt[i] == 'p' || fmt[i] == 's' ? 2u : 1u;
    }
    return words;
}

template <typename T>
void emit_printf_value(const std::string& spec,
                       bool dynamic_width,
                       int width,
                       bool dynamic_precision,
                       int precision,
                       T value) {
    if (dynamic_width && dynamic_precision) {
        std::fprintf(stderr, spec.c_str(), width, precision, value);
    } else if (dynamic_width) {
        std::fprintf(stderr, spec.c_str(), width, value);
    } else if (dynamic_precision) {
        std::fprintf(stderr, spec.c_str(), precision, value);
    } else {
        std::fprintf(stderr, spec.c_str(), value);
    }
}

bool resolve_printf_string_pointer(
    std::uint64_t raw,
    const std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>>&
        registered_buffers,
    const char** begin,
    std::size_t* remaining) {
    if (raw == 0 || begin == nullptr || remaining == nullptr) return false;

    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    RuntimeState& state = runtime_state();
    const void* pointer = reinterpret_cast<const void*>(
        static_cast<std::uintptr_t>(raw));
    if (state.allocations.resolve(pointer, &resolved) &&
        resolved.buffer != nullptr && resolved.buffer->contents() != nullptr) {
        *begin = static_cast<const char*>(resolved.buffer->contents()) +
                 resolved.offset;
        *remaining = resolved.remaining_size;
        return true;
    }

    // Module constant/global buffers are safe runtime-owned storage, but they
    // are not ordinary cudaMalloc allocations and therefore intentionally do
    // not live in AllocationTable. Match both their Metal GPU address and UMA
    // host mapping without ever dereferencing an arbitrary device value.
    for (const auto& buffer : registered_buffers) {
        if (buffer == nullptr || buffer->contents() == nullptr ||
            buffer->length() == 0) {
            continue;
        }
        const std::uintptr_t raw_address = static_cast<std::uintptr_t>(raw);
        const std::uintptr_t bases[] = {
            buffer->device_address(),
            reinterpret_cast<std::uintptr_t>(buffer->contents()),
        };
        for (const std::uintptr_t base : bases) {
            if (base == 0 || raw_address < base) continue;
            const std::uintptr_t offset = raw_address - base;
            if (offset >= buffer->length()) continue;
            *begin = static_cast<const char*>(buffer->contents()) + offset;
            *remaining = buffer->length() - offset;
            return true;
        }
    }
    return false;
}

void drain_one_printf_record(const std::string& fmt,
                             const std::uint32_t* args,
                             std::uint32_t n_args,
                             const std::vector<std::shared_ptr<
                                 cumetal::metal_backend::Buffer>>&
                                 registered_string_buffers) {
    std::uint32_t arg_idx = 0;
    for (std::size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') {
            std::fputc(fmt[i], stderr);
            continue;
        }
        ++i;
        if (i >= fmt.size()) { break; }
        if (fmt[i] == '%') {
            std::fputc('%', stderr);
            continue;
        }
        // Reconstruct specifier string
        std::string spec = "%";
        // Flags
        while (i < fmt.size() &&
               (fmt[i] == '-' || fmt[i] == '+' || fmt[i] == ' ' ||
                fmt[i] == '#' || fmt[i] == '0')) {
            spec += fmt[i++];
        }
        // Width. Dynamic width and precision are promoted int slots preceding
        // the conversion value in CUDA's packed varargs tuple.
        bool dynamic_width = false;
        int width = 0;
        if (i < fmt.size() && fmt[i] == '*') {
            dynamic_width = true;
            spec += fmt[i++];
            if (arg_idx < n_args) {
                width = static_cast<int>(args[arg_idx++]);
            }
        } else {
            while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) {
                spec += fmt[i++];
            }
        }
        // Precision
        bool dynamic_precision = false;
        int precision = 0;
        if (i < fmt.size() && fmt[i] == '.') {
            spec += fmt[i++];
            if (i < fmt.size() && fmt[i] == '*') {
                dynamic_precision = true;
                spec += fmt[i++];
                if (arg_idx < n_args) {
                    precision = static_cast<int>(args[arg_idx++]);
                }
            } else {
                while (i < fmt.size() && std::isdigit(static_cast<unsigned char>(fmt[i]))) {
                    spec += fmt[i++];
                }
            }
        }
        // Track length modifiers so packed 64-bit scalar slots do not shift
        // every following argument. String pointers are also 64-bit in Clang's
        // CUDA ABI, although string materialization remains a separate gap.
        bool wide_modifier = false;
        while (i < fmt.size() &&
               (fmt[i] == 'l' || fmt[i] == 'h' || fmt[i] == 'z' ||
                fmt[i] == 'j' || fmt[i] == 't')) {
            if (fmt[i] == 'l' || fmt[i] == 'z' || fmt[i] == 'j' || fmt[i] == 't') {
                wide_modifier = true;
            }
            ++i;
        }
        if (i >= fmt.size()) { break; }
        const char conv = fmt[i];

        if (arg_idx >= n_args) {
            spec += conv;
            std::fputs(spec.c_str(), stderr);
            continue;
        }
        if (conv == 's') {
            if (arg_idx + 1u >= n_args) {
                arg_idx = n_args;
                std::fputs("[string]", stderr);
                continue;
            }
            const std::uint64_t raw = static_cast<std::uint64_t>(args[arg_idx]) |
                (static_cast<std::uint64_t>(args[arg_idx + 1u]) << 32u);
            arg_idx += 2u;
            spec += 's';

            // Only tracked allocations and the registered module buffers bound
            // for this launch are safe to materialize on the host.
            constexpr std::size_t kMaxDeviceStringBytes = 256u;
            const char* begin = nullptr;
            std::size_t remaining = 0;
            if (!resolve_printf_string_pointer(
                    raw, registered_string_buffers, &begin, &remaining)) {
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision, "[string]");
                continue;
            }
            const std::size_t bounded_size =
                std::min(remaining, kMaxDeviceStringBytes);
            const void* terminator = std::memchr(begin, '\0', bounded_size);
            if (terminator == nullptr) {
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision,
                                  "[unterminated-string]");
                continue;
            }
            const std::size_t length =
                static_cast<const char*>(terminator) - begin;
            const std::string materialized(begin, length);
            emit_printf_value(spec, dynamic_width, width,
                              dynamic_precision, precision,
                              materialized.c_str());
            continue;
        }
        if (conv == 'p' && arg_idx + 1u < n_args) {
            const std::uint64_t raw = static_cast<std::uint64_t>(args[arg_idx]) |
                (static_cast<std::uint64_t>(args[arg_idx + 1u]) << 32u);
            arg_idx += 2u;
            spec += 'p';
            emit_printf_value(
                spec, dynamic_width, width, dynamic_precision, precision,
                reinterpret_cast<void*>(static_cast<std::uintptr_t>(raw)));
            continue;
        }
        if (wide_modifier && arg_idx + 1u < n_args) {
            const std::uint64_t raw = static_cast<std::uint64_t>(args[arg_idx]) |
                (static_cast<std::uint64_t>(args[arg_idx + 1u]) << 32u);
            arg_idx += 2u;
            spec += "ll";
            spec += conv;
            if (conv == 'd' || conv == 'i') {
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision,
                                  static_cast<long long>(raw));
            } else if (conv == 'u' || conv == 'o' || conv == 'x' || conv == 'X') {
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision,
                                  static_cast<unsigned long long>(raw));
            } else {
                std::fputs(spec.c_str(), stderr);
            }
            continue;
        }
        const std::uint32_t raw = args[arg_idx++];
        if (conv == 'f' || conv == 'e' || conv == 'g' ||
            conv == 'F' || conv == 'E' || conv == 'G' ||
            conv == 'a' || conv == 'A') {
            spec += conv;
            // CUDA C varargs promote floating arguments to binary64. Retain a
            // one-word fallback for hand-written PTX fixtures that predate the
            // Clang ABI decoder.
            const std::uint32_t remaining_minimum =
                minimum_printf_words(std::string_view(fmt).substr(i + 1u));
            if (arg_idx < n_args &&
                n_args - (arg_idx + 1u) >= remaining_minimum) {
                const std::uint64_t bits = static_cast<std::uint64_t>(raw) |
                    (static_cast<std::uint64_t>(args[arg_idx++]) << 32u);
                double value = 0.0;
                std::memcpy(&value, &bits, sizeof(value));
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision, value);
            } else {
                float value = 0.0f;
                std::memcpy(&value, &raw, sizeof(value));
                emit_printf_value(spec, dynamic_width, width,
                                  dynamic_precision, precision,
                                  static_cast<double>(value));
            }
        } else if (conv == 'd' || conv == 'i') {
            spec += conv;
            emit_printf_value(spec, dynamic_width, width,
                              dynamic_precision, precision,
                              static_cast<int>(raw));
        } else if (conv == 'u' || conv == 'o' || conv == 'x' || conv == 'X') {
            spec += conv;
            emit_printf_value(spec, dynamic_width, width,
                              dynamic_precision, precision, raw);
        } else if (conv == 'c') {
            spec += conv;
            emit_printf_value(spec, dynamic_width, width,
                              dynamic_precision, precision,
                              static_cast<int>(raw));
        } else {
            spec += conv;
            std::fputs(spec.c_str(), stderr);
        }
    }
}

void drain_printf_buffer(const void* buf_bytes,
                         std::uint32_t cap_words,
                         const std::vector<std::string>& formats,
                         const std::vector<std::shared_ptr<
                             cumetal::metal_backend::Buffer>>&
                             registered_string_buffers) {
    if (buf_bytes == nullptr || cap_words == 0 || formats.empty()) {
        return;
    }
    const std::uint32_t* buf = static_cast<const std::uint32_t*>(buf_bytes);
    const std::uint32_t total_words = buf[0];
    if (total_words == 0) return;
    // Reservations are monotonic. If the cursor exceeds capacity, every
    // complete record before the first rejected reservation is still a valid
    // prefix and must be drained; later reservations cannot create holes in
    // that prefix because their positions only increase.
    const std::uint32_t available_words =
        std::min(total_words, cap_words - 1u);
    // Walk records starting at index 1
    std::uint32_t i = 1u;
    while (i + 1u <= available_words) {
        const std::uint32_t fmt_id = buf[i];
        const std::uint32_t n_args = buf[i + 1u];
        if (n_args > available_words ||
            i + 2u + n_args > available_words + 1u) {
            break;
        }
        if (fmt_id < static_cast<std::uint32_t>(formats.size())) {
            drain_one_printf_record(formats[fmt_id], buf + i + 2u, n_args,
                                    registered_string_buffers);
        }
        i += 2u + n_args;
    }
}

cudaError_t ensure_initialized() {
    RuntimeState& state = runtime_state();
    std::call_once(state.init_once, [&state]() {
        const std::string conflict_warning =
            cumetal::error::detect_loaded_libcuda_conflict(reinterpret_cast<const void*>(&cudaInit));
        if (!conflict_warning.empty()) {
            std::fprintf(stderr, "%s\n", conflict_warning.c_str());
        }

        std::string error;
        state.init_status = cumetal::metal_backend::initialize(&error);
        state.init_error = error;
    });
    return state.init_status;
}

cudaError_t fail(cudaError_t error) {
    if (error != cudaSuccess) {
        static int debug_errors = -1;
        if (debug_errors < 0) {
            const char* v = std::getenv("CUMETAL_DEBUG_ERRORS");
            debug_errors = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        }
        if (debug_errors) {
            Dl_info info{};
            const void* caller = __builtin_return_address(0);
            const bool have_symbol = caller != nullptr && dladdr(caller, &info) != 0 && info.dli_sname != nullptr;
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_ERRORS: %s -> %d\n",
                         have_symbol ? info.dli_sname : "<unknown>",
                         static_cast<int>(error));
        }
    }
    set_last_error(error);
    return error;
}

bool resolve_stream_handle(cudaStream_t stream,
                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = found->second.backend;
    return true;
}

bool is_legacy_stream_handle(cudaStream_t stream) {
    return stream == nullptr || stream == cudaStreamLegacy;
}

bool is_per_thread_stream_handle(cudaStream_t stream) {
    return stream == cudaStreamPerThread;
}

bool resolve_stream_flags(cudaStream_t stream, unsigned int* out_flags) {
    if (out_flags == nullptr) {
        return false;
    }
    if (is_legacy_stream_handle(stream) || is_per_thread_stream_handle(stream)) {
        *out_flags = cudaStreamDefault;
        return true;
    }
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }
    *out_flags = found->second.flags;
    return true;
}

cudaError_t ensure_per_thread_stream(std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    if (tls_per_thread_stream != nullptr) {
        *out_stream = tls_per_thread_stream;
        return cudaSuccess;
    }

    std::string error;
    std::shared_ptr<cumetal::metal_backend::Stream> created;
    const cudaError_t status =
        cumetal::metal_backend::create_stream(&created, &error, false);
    if (status != cudaSuccess || created == nullptr) {
        return status == cudaSuccess ? cudaErrorUnknown : status;
    }
    tls_per_thread_stream = created;
    *out_stream = std::move(created);
    return cudaSuccess;
}

cudaError_t resolve_runtime_stream(cudaStream_t stream,
                                   std::shared_ptr<cumetal::metal_backend::Stream>* out_stream,
                                   bool* is_legacy_stream) {
    if (out_stream == nullptr) {
        return cudaErrorInvalidValue;
    }
    out_stream->reset();
    if (is_legacy_stream != nullptr) {
        *is_legacy_stream = false;
    }

    if (is_legacy_stream_handle(stream)) {
        *out_stream = cumetal::metal_backend::legacy_default_stream();
        if (*out_stream == nullptr) {
            return cudaErrorInitializationError;
        }
        if (is_legacy_stream != nullptr) {
            *is_legacy_stream = true;
        }
        return cudaSuccess;
    }

    if (is_per_thread_stream_handle(stream)) {
        return ensure_per_thread_stream(out_stream);
    }

    if (!resolve_stream_handle(stream, out_stream)) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

bool erase_stream_handle(cudaStream_t stream,
                         std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    if (stream == nullptr || out_stream == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) {
        return false;
    }

    *out_stream = std::move(found->second.backend);
    state.streams.erase(found);
    return true;
}

cudaError_t validate_memcpy_kind(cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToHost:
        case cudaMemcpyHostToDevice:
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyDeviceToDevice:
        case cudaMemcpyDefault:
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t validate_host_alloc_flags(unsigned int flags) {
    constexpr unsigned int kSupportedHostAllocFlags =
        cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined;
    if ((flags & ~kSupportedHostAllocFlags) != 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t validate_host_register_flags(unsigned int flags) {
    constexpr unsigned int kSupportedHostRegisterFlags =
        cudaHostRegisterPortable | cudaHostRegisterMapped |
        cudaHostRegisterIoMemory | cudaHostRegisterReadOnly;
    if ((flags & ~kSupportedHostRegisterFlags) != 0) {
        return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

cudaError_t validate_device_flags(unsigned int flags) {
    constexpr unsigned int kSupportedDeviceFlags =
        cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync |
        cudaDeviceMapHost | cudaDeviceLmemResizeToMax;

    if ((flags & ~kSupportedDeviceFlags) != 0) {
        return cudaErrorInvalidValue;
    }

    const unsigned int schedule_bits =
        flags & (cudaDeviceScheduleSpin | cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync);
    if (schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleYield | cudaDeviceScheduleBlockingSync) ||
        schedule_bits == (cudaDeviceScheduleSpin | cudaDeviceScheduleYield |
                          cudaDeviceScheduleBlockingSync)) {
        return cudaErrorInvalidValue;
    }

    return cudaSuccess;
}

bool is_device_pointer(const void* ptr) {
    if (ptr == nullptr) {
        return false;
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved);
}

void relocate_embedded_device_pointers(
    std::vector<std::uint8_t>* bytes,
    std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>>* resident_buffers = nullptr) {
    if (bytes == nullptr || bytes->size() < sizeof(std::uintptr_t)) return;
    RuntimeState& state = runtime_state();
    // CUDA permits aggregates passed by value to contain device pointers. The
    // public CuMetal pointer is normally the CPU mapping of a shared MTLBuffer,
    // while a pointer dereferenced inside Metal must be its GPU virtual address.
    // Pointer fields follow the platform ABI's natural pointer alignment.
    for (std::size_t offset = 0; offset + sizeof(std::uintptr_t) <= bytes->size();
         offset += alignof(std::uintptr_t)) {
        std::uintptr_t candidate = 0;
        std::memcpy(&candidate, bytes->data() + offset, sizeof(candidate));
        if (candidate == 0) continue;
        cumetal::rt::AllocationTable::ResolvedAllocation resolved;
        if (!state.allocations.resolve(reinterpret_cast<void*>(candidate), &resolved) ||
            resolved.buffer == nullptr) {
            continue;
        }
        const std::uintptr_t gpu_base = resolved.buffer->device_address();
        if (gpu_base == 0 ||
            gpu_base > std::numeric_limits<std::uintptr_t>::max() - resolved.offset) {
            continue;
        }
        const std::uintptr_t relocated = gpu_base + resolved.offset;
        std::memcpy(bytes->data() + offset, &relocated, sizeof(relocated));
        if (resident_buffers != nullptr) {
            resident_buffers->push_back(resolved.buffer);
        }
    }
}

void restore_embedded_host_pointers(std::vector<std::uint8_t>* bytes) {
    if (bytes == nullptr || bytes->size() < sizeof(std::uintptr_t)) return;
    RuntimeState& state = runtime_state();
    for (std::size_t offset = 0; offset + sizeof(std::uintptr_t) <= bytes->size();
         offset += alignof(std::uintptr_t)) {
        std::uintptr_t candidate = 0;
        std::memcpy(&candidate, bytes->data() + offset, sizeof(candidate));
        if (candidate == 0) continue;
        cumetal::rt::AllocationTable::ResolvedAllocation resolved;
        if (!state.allocations.resolve(reinterpret_cast<void*>(candidate), &resolved) ||
            resolved.buffer == nullptr || resolved.buffer->contents() == nullptr) {
            continue;
        }
        const std::uintptr_t host_base =
            reinterpret_cast<std::uintptr_t>(resolved.buffer->contents());
        if (host_base > std::numeric_limits<std::uintptr_t>::max() - resolved.offset) {
            continue;
        }
        const std::uintptr_t restored = host_base + resolved.offset;
        std::memcpy(bytes->data() + offset, &restored, sizeof(restored));
    }
}

bool use_metal_device_addresses() {
    const char* value = std::getenv("CUMETAL_USE_METAL_DEVICE_ADDRESSES");
    return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 && std::strcmp(value, "FALSE") != 0;
}

void collect_texture_resource_residency(
    std::uintptr_t handle,
    std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>>* buffers);

cudaError_t activate_graph_allocation(
    const std::shared_ptr<GraphAllocationState>& allocation) {
    if (allocation == nullptr || allocation->buffer == nullptr ||
        allocation->base == nullptr || allocation->size == 0) {
        return cudaErrorInvalidValue;
    }
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
    if (allocation->active) return cudaErrorInvalidValue;

    std::string error;
    if (!state.allocations.insert(allocation->base, allocation->size,
                                  cumetal::rt::AllocationKind::kDevice, 0,
                                  allocation->buffer, &error)) {
        return cudaErrorMemoryAllocation;
    }
    if (allocation->alias != nullptr && allocation->alias != allocation->base &&
        !state.allocations.insert(allocation->alias, allocation->size,
                                  cumetal::rt::AllocationKind::kDevice, 0,
                                  allocation->buffer, &error, true)) {
        state.allocations.erase(allocation->base);
        return cudaErrorMemoryAllocation;
    }
    allocation->active = true;
    // Keep every launched allocation alive independently of graph handles.
    // A later free node (in this graph or another graph) or cudaFree removes
    // this ownership. This also preserves an allocation if execution stops
    // before a same-graph free node is reached.
    state.live_graph_allocations[allocation->base] = allocation;
    state.graph_used_current += allocation->size;
    state.graph_used_high =
        std::max(state.graph_used_high, state.graph_used_current);
    return cudaSuccess;
}

cudaError_t deactivate_graph_allocation(
    const std::shared_ptr<GraphAllocationState>& allocation) {
    if (allocation == nullptr) return cudaErrorInvalidValue;
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
    if (!allocation->active) return cudaErrorInvalidValue;
    if (!state.allocations.erase(allocation->base)) {
        return cudaErrorInvalidDevicePointer;
    }
    allocation->active = false;
    state.live_graph_allocations.erase(allocation->base);
    state.graph_used_current =
        allocation->size > state.graph_used_current
            ? 0
            : state.graph_used_current - allocation->size;
    return cudaSuccess;
}

void release_graph_allocation(GraphAllocationState* allocation) {
    if (allocation == nullptr) return;
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
    if (allocation->active) {
        (void)state.allocations.erase(allocation->base);
        state.graph_used_current =
            allocation->size > state.graph_used_current
                ? 0
                : state.graph_used_current - allocation->size;
    }
    if (allocation->reserved_accounted) {
        state.graph_reserved_current =
            allocation->size > state.graph_reserved_current
                ? 0
                : state.graph_reserved_current - allocation->size;
    }
    state.graph_allocations.erase(allocation->base);
    state.live_graph_allocations.erase(allocation->base);
}

void reset_graph_allocation_state() {
    RuntimeState& state = runtime_state();
    std::vector<std::shared_ptr<GraphAllocationState>> release_after_unlock;
    {
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        release_after_unlock.reserve(state.live_graph_allocations.size());
        for (auto& [base, allocation] : state.live_graph_allocations) {
            (void)base;
            if (allocation != nullptr) {
                allocation->active = false;
                release_after_unlock.push_back(allocation);
            }
        }
        for (auto& [base, weak] : state.graph_allocations) {
            (void)base;
            if (const auto allocation = weak.lock(); allocation != nullptr) {
                allocation->active = false;
            }
        }
        state.live_graph_allocations.clear();
        state.graph_used_current = 0;
    }
}

const void* host_accessible_pointer(const void* ptr, std::size_t count) {
    if (ptr == nullptr) {
        return nullptr;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(ptr, &resolved)) {
        return ptr;
    }
    if (count > resolved.remaining_size || resolved.buffer == nullptr ||
        resolved.buffer->contents() == nullptr) {
        return nullptr;
    }
    return static_cast<const unsigned char*>(resolved.buffer->contents()) + resolved.offset;
}

void* host_accessible_pointer(void* ptr, std::size_t count) {
    return const_cast<void*>(
        host_accessible_pointer(static_cast<const void*>(ptr), count));
}

cudaError_t query_runtime_kernel_properties(
    const void* function,
    cumetal::metal_backend::KernelProperties* properties) {
    if (function == nullptr || properties == nullptr)
        return cudaErrorInvalidValue;
    cumetal::registration::RegisteredKernel kernel;
    if (!cumetal::native_registration::lookup_kernel(function, &kernel) &&
        !cumetal::registration::lookup_registered_kernel(function, &kernel)) {
        return cudaErrorInvalidValue;
    }
    if (kernel.metallib_path.empty() || kernel.kernel_name.empty())
        return cudaErrorInvalidValue;
    std::string error;
    return cumetal::metal_backend::query_kernel_properties(
        kernel.metallib_path, kernel.kernel_name, properties, &error);
}

cudaError_t resolve_memcpy_kind(void* dst, const void* src, cudaMemcpyKind kind, cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const cudaError_t kind_status = validate_memcpy_kind(kind);
    if (kind_status != cudaSuccess) {
        return kind_status;
    }

    const bool dst_is_device = is_device_pointer(dst);
    const bool src_is_device = is_device_pointer(src);

    if (kind == cudaMemcpyDefault) {
        if (dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToDevice;
        } else if (dst_is_device && !src_is_device) {
            *resolved_kind = cudaMemcpyHostToDevice;
        } else if (!dst_is_device && src_is_device) {
            *resolved_kind = cudaMemcpyDeviceToHost;
        } else {
            *resolved_kind = cudaMemcpyHostToHost;
        }
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToHost:
            break;
        case cudaMemcpyHostToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToHost:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device || !src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            break;
        case cudaMemcpyDefault:
            break;
    }

    *resolved_kind = kind;
    return cudaSuccess;
}

cudaError_t resolve_memcpy_to_symbol_kind(const void* src,
                                          cudaMemcpyKind kind,
                                          cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool src_is_device = is_device_pointer(src);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = src_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyHostToDevice:
            *resolved_kind = cudaMemcpyHostToDevice;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!src_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t resolve_memcpy_from_symbol_kind(void* dst,
                                            cudaMemcpyKind kind,
                                            cudaMemcpyKind* resolved_kind) {
    if (resolved_kind == nullptr) {
        return cudaErrorInvalidValue;
    }

    const bool dst_is_device = is_device_pointer(dst);
    if (kind == cudaMemcpyDefault) {
        *resolved_kind = dst_is_device ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        return cudaSuccess;
    }

    switch (kind) {
        case cudaMemcpyDeviceToHost:
            *resolved_kind = cudaMemcpyDeviceToHost;
            return cudaSuccess;
        case cudaMemcpyDeviceToDevice:
            if (!dst_is_device) {
                return cudaErrorInvalidDevicePointer;
            }
            *resolved_kind = cudaMemcpyDeviceToDevice;
            return cudaSuccess;
        default:
            return cudaErrorInvalidValue;
    }
}

cudaError_t checked_symbol_ptr(const void* symbol,
                               size_t count,
                               size_t offset,
                               const unsigned char** out_ptr) {
    if (symbol == nullptr || out_ptr == nullptr) {
        return cudaErrorInvalidValue;
    }

    const void* resolved_symbol = symbol;
    std::size_t resolved_size = 0;
    if (cumetal::native_registration::lookup_symbol(
            symbol, &resolved_symbol, &resolved_size) ||
        cumetal::registration::lookup_registered_symbol(
            symbol, &resolved_symbol, &resolved_size)) {
        if (resolved_size > 0 && (offset > resolved_size || count > (resolved_size - offset))) {
            return cudaErrorInvalidValue;
        }
    }

    if (offset > (std::numeric_limits<size_t>::max() - count)) {
        return cudaErrorInvalidValue;
    }

    *out_ptr = static_cast<const unsigned char*>(resolved_symbol) + offset;
    return cudaSuccess;
}

cudaError_t synchronize_stream_for_host_op(cudaStream_t stream,
                                           std::shared_ptr<cumetal::metal_backend::Stream>* out_stream) {
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return resolve_status;
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    if (status != cudaSuccess) {
        return status;
    }

    if (out_stream != nullptr) {
        *out_stream = std::move(backend_stream);
    }
    return cudaSuccess;
}

cudaError_t enqueue_stream_host_op(cudaStream_t stream, std::function<void()> operation) {
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess) {
        return resolve_status;
    }
    std::string error;
    return cumetal::metal_backend::enqueue_host_function(
        backend_stream, std::move(operation), &error);
}

cudaError_t update_event_completion(cudaEvent_t event, bool wait_for_completion) {
    if (event == nullptr) {
        return cudaErrorInvalidValue;
    }

    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    std::uint64_t ticket = 0;
    {
        std::lock_guard<std::mutex> lock(event->mutex);
        if (event->complete) {
            return cudaSuccess;
        }
        stream = event->stream;
        ticket = event->ticket;
    }

    if (stream == nullptr || ticket == 0) {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->complete = true;
        if (!event->disable_timing && !event->timing_valid) {
            event->timestamp = std::chrono::steady_clock::now();
            event->timing_valid = true;
        }
        return cudaSuccess;
    }

    std::string error;
    bool is_complete = false;
    if (wait_for_completion) {
        const cudaError_t status = cumetal::metal_backend::stream_wait_ticket(stream, ticket, &error);
        if (status != cudaSuccess) {
            return status;
        }
        is_complete = true;
    } else {
        const cudaError_t status = cumetal::metal_backend::stream_query_ticket(stream, ticket,
                                                                                &is_complete, &error);
        if (status != cudaSuccess) {
            return status;
        }
        if (!is_complete) {
            return cudaErrorNotReady;
        }
    }

    std::lock_guard<std::mutex> lock(event->mutex);
    event->complete = true;
    if (!event->disable_timing && !event->timing_valid) {
        event->timestamp = std::chrono::steady_clock::now();
        event->timing_valid = true;
    }
    return cudaSuccess;
}

template <typename T>
T read_scalar_launch_arg(void** args, std::uint32_t index) {
    T value{};
    std::memcpy(&value, args[index], sizeof(T));
    return value;
}

template <typename T>
T* read_pointer_launch_arg(void** args, std::uint32_t index) {
    T* value = nullptr;
    std::memcpy(&value, args[index], sizeof(value));
    return value;
}

bool kernel_name_contains(const std::string& kernel_name, std::string_view needle) {
    return kernel_name.find(needle) != std::string::npos;
}

bool kernel_name_matches_env_list(const std::string& kernel_name, const char* env_var) {
    if (env_var == nullptr || env_var[0] == '\0') {
        return false;
    }

    std::string token;
    for (const char* p = env_var;; ++p) {
        const char c = *p;
        if (c == ',' || c == '\0') {
            std::size_t begin = 0;
            while (begin < token.size() &&
                   std::isspace(static_cast<unsigned char>(token[begin])) != 0) {
                ++begin;
            }
            std::size_t end = token.size();
            while (end > begin &&
                   std::isspace(static_cast<unsigned char>(token[end - 1])) != 0) {
                --end;
            }
            if (end > begin) {
                const std::string_view needle(token.data() + begin, end - begin);
                if (kernel_name.find(needle) != std::string::npos) {
                    return true;
                }
            }
            token.clear();
            if (c == '\0') {
                break;
            }
            continue;
        }
        token.push_back(c);
    }

    return false;
}

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') {
        return false;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return normalized == "1" || normalized == "true" || normalized == "yes" ||
           normalized == "on";
}

bool llmc_emulation_enabled() {
    // CUDA kernels must execute through Metal by default.  The old behavior
    // silently intercepted known llm.c kernels and ran host loops on UMA,
    // which made a successful CUDA launch indistinguishable from CPU
    // emulation.  Keep that implementation only as an explicit diagnostic
    // escape hatch for comparing results while bringing up a new lowering.
    return env_truthy(std::getenv("CUMETAL_ENABLE_LLMC_CPU_EMULATION")) &&
           !env_truthy(std::getenv("CUMETAL_DISABLE_LLMC_EMULATION"));
}

bool llmc_emulation_skips_kernel(const std::string& kernel_name) {
    return kernel_name_matches_env_list(kernel_name, std::getenv("CUMETAL_LLMC_EMULATION_SKIP"));
}

bool llmc_emulation_trace_enabled() {
    return env_truthy(std::getenv("CUMETAL_TRACE_LLMC_EMULATION"));
}

std::atomic<std::uint64_t>& llmc_emulation_count() {
    static std::atomic<std::uint64_t> count{0};
    return count;
}

void note_llmc_emulation_hit(const std::string& kernel_name, std::uint32_t arg_count) {
    const std::uint64_t hit = llmc_emulation_count().fetch_add(1, std::memory_order_relaxed) + 1;
    cumetal::warn_once(
        "llmc-cpu-emulation",
        "CUMETAL_ENABLE_LLMC_CPU_EMULATION is running CUDA kernels on the CPU; "
        "disable it to require Apple GPU execution");
    if (cumetal::diag_env_truthy("CUMETAL_TRACE_GPU")) {
        std::fprintf(stderr,
                     "CUMETAL_PROVENANCE event=kernel_launch kernel=\"%s\" "
                     "source=cpu_fallback provenance=cpu_fallback "
                     "semantic_quality=cpu_fallback device=cpu compile_cache_hit=false "
                     "launch_success=true duration_ns=-1 grid=unknown block=unknown "
                     "unsupported_reason=\"explicit llm.c CPU emulation\"\n",
                     kernel_name.c_str());
    }
    if (llmc_emulation_trace_enabled()) {
        std::fprintf(stderr,
                     "INFO: CUMETAL_LLMC_EMULATION kernel=%s arg_count=%u hit=%llu\n",
                     kernel_name.c_str(),
                     static_cast<unsigned int>(arg_count),
                     static_cast<unsigned long long>(hit));
    }
}


cudaError_t synchronize_for_emulated_kernel(
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream) {
    (void)legacy_stream;
    if (backend_stream == nullptr) {
        return cudaSuccess;
    }

    std::string error;
    return cumetal::metal_backend::stream_synchronize(backend_stream, &error);
}

cudaError_t emulate_matmul_forward_kernel4(
    dim3 grid_dim,
    dim3 block_dim,
    void** args,
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream) {
    float* out = read_pointer_launch_arg<float>(args, 0);
    const float* inp = read_pointer_launch_arg<const float>(args, 1);
    const float* weight = read_pointer_launch_arg<const float>(args, 2);
    const float* bias = read_pointer_launch_arg<const float>(args, 3);
    const int c = read_scalar_launch_arg<int>(args, 4);
    const int oc = read_scalar_launch_arg<int>(args, 5);

    if (out == nullptr || inp == nullptr || weight == nullptr || c <= 0 || oc <= 0) {
        return cudaErrorInvalidValue;
    }

    const int tile_rows = static_cast<int>(block_dim.x) * 8;
    const int tile_cols = static_cast<int>(block_dim.y) * 8;
    if (tile_rows <= 0 || tile_cols <= 0) {
        return cudaErrorInvalidValue;
    }

    const int m = static_cast<int>(grid_dim.x) * tile_rows;
    if (m <= 0) {
        return cudaSuccess;
    }

    const cudaError_t sync_status = synchronize_for_emulated_kernel(legacy_stream, backend_stream);
    if (sync_status != cudaSuccess) {
        return sync_status;
    }

    if (bias != nullptr) {
        for (int row = 0; row < m; ++row) {
            std::memcpy(out + static_cast<std::size_t>(row) * static_cast<std::size_t>(oc),
                        bias,
                        static_cast<std::size_t>(oc) * sizeof(float));
        }
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation weight_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation inp_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation out_resolved;
    if (!state.allocations.resolve(weight, &weight_resolved) ||
        !state.allocations.resolve(inp, &inp_resolved) ||
        !state.allocations.resolve(out, &out_resolved)) {
        return cudaErrorInvalidDevicePointer;
    }

    std::string error;
    return cumetal::metal_backend::gemm_f32(
        /*transpose_left=*/true,
        /*transpose_right=*/false,
        oc,
        m,
        c,
        1.0f,
        weight_resolved.buffer,
        weight_resolved.offset,
        c,
        inp_resolved.buffer,
        inp_resolved.offset,
        c,
        bias != nullptr ? 1.0f : 0.0f,
        out_resolved.buffer,
        out_resolved.offset,
        oc,
        backend_stream,
        &error);
}

cudaError_t try_emulate_llmc_registered_kernel(
    const std::string& kernel_name,
    std::uint32_t arg_count,
    dim3 grid_dim,
    dim3 block_dim,
    void** args,
    bool legacy_stream,
    const std::shared_ptr<cumetal::metal_backend::Stream>& backend_stream,
    bool* handled) {
    if (handled == nullptr) {
        return cudaErrorInvalidValue;
    }
    *handled = false;

    if (args == nullptr) {
        return cudaErrorInvalidValue;
    }

    if (kernel_name_contains(kernel_name, "matmul_forward_kernel4")) {
        *handled = true;
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        return emulate_matmul_forward_kernel4(grid_dim, block_dim, args, legacy_stream, backend_stream);
    }

    const bool known_llmc_kernel =
        kernel_name_contains(kernel_name, "encoder_forward_kernel3") ||
        kernel_name_contains(kernel_name, "encoder_backward_kernel") ||
        kernel_name_contains(kernel_name, "layernorm_forward_kernel3") ||
        kernel_name_contains(kernel_name, "permute_kernel") ||
        kernel_name_contains(kernel_name, "unpermute_kernel") ||
        kernel_name_contains(kernel_name, "softmax_forward_kernel5") ||
        kernel_name_contains(kernel_name, "residual_forward_kernel") ||
        kernel_name_contains(kernel_name, "gelu_forward_kernel") ||
        kernel_name_contains(kernel_name, "gelu_backward_kernel") ||
        kernel_name_contains(kernel_name, "matmul_backward_bias_kernel4") ||
        kernel_name_contains(kernel_name, "layernorm_backward_kernel2") ||
        kernel_name_contains(kernel_name, "softmax_autoregressive_backward_kernel") ||
        kernel_name_contains(kernel_name, "adamw_kernel2") ||
        kernel_name_contains(kernel_name, "fused_classifier_kernel3");
    if (!known_llmc_kernel) {
        return cudaSuccess;
    }

    const cudaError_t sync_status = synchronize_for_emulated_kernel(legacy_stream, backend_stream);
    if (sync_status != cudaSuccess) {
        return sync_status;
    }

    if (kernel_name_contains(kernel_name, "encoder_forward_kernel3")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const int* inp = read_pointer_launch_arg<const int>(args, 1);
        const float* wte = read_pointer_launch_arg<const float>(args, 2);
        const float* wpe = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int t = read_scalar_launch_arg<int>(args, 5);
        const int c = read_scalar_launch_arg<int>(args, 6);
        if (out == nullptr || inp == nullptr || wte == nullptr || wpe == nullptr || b <= 0 || t <= 0 ||
            c <= 0) {
            return cudaErrorInvalidValue;
        }
        for (int bi = 0; bi < b; ++bi) {
            for (int ti = 0; ti < t; ++ti) {
                const int token = inp[bi * t + ti];
                const std::size_t out_base =
                    (static_cast<std::size_t>(bi) * static_cast<std::size_t>(t) +
                     static_cast<std::size_t>(ti)) *
                    static_cast<std::size_t>(c);
                const std::size_t wte_base = static_cast<std::size_t>(token) * static_cast<std::size_t>(c);
                const std::size_t wpe_base = static_cast<std::size_t>(ti) * static_cast<std::size_t>(c);
                for (int ci = 0; ci < c; ++ci) {
                    out[out_base + static_cast<std::size_t>(ci)] =
                        wte[wte_base + static_cast<std::size_t>(ci)] +
                        wpe[wpe_base + static_cast<std::size_t>(ci)];
                }
            }
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "encoder_backward_kernel")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* dwte = read_pointer_launch_arg<float>(args, 0);
        float* dwpe = read_pointer_launch_arg<float>(args, 1);
        const float* dout = read_pointer_launch_arg<const float>(args, 2);
        const int* inp = read_pointer_launch_arg<const int>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int t = read_scalar_launch_arg<int>(args, 5);
        const int c = read_scalar_launch_arg<int>(args, 6);
        if (dwte == nullptr || dwpe == nullptr || dout == nullptr || inp == nullptr || b <= 0 || t <= 0 ||
            c <= 0) {
            return cudaErrorInvalidValue;
        }
        for (int bi = 0; bi < b; ++bi) {
            for (int ti = 0; ti < t; ++ti) {
                const int token = inp[bi * t + ti];
                const std::size_t dout_base =
                    (static_cast<std::size_t>(bi) * static_cast<std::size_t>(t) +
                     static_cast<std::size_t>(ti)) *
                    static_cast<std::size_t>(c);
                const std::size_t dwte_base = static_cast<std::size_t>(token) * static_cast<std::size_t>(c);
                const std::size_t dwpe_base = static_cast<std::size_t>(ti) * static_cast<std::size_t>(c);
                for (int ci = 0; ci < c; ++ci) {
                    const float grad = dout[dout_base + static_cast<std::size_t>(ci)];
                    dwte[dwte_base + static_cast<std::size_t>(ci)] += grad;
                    dwpe[dwpe_base + static_cast<std::size_t>(ci)] += grad;
                }
            }
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "layernorm_forward_kernel3")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        float* mean = read_pointer_launch_arg<float>(args, 1);
        float* rstd = read_pointer_launch_arg<float>(args, 2);
        const float* inp = read_pointer_launch_arg<const float>(args, 3);
        const float* weight = read_pointer_launch_arg<const float>(args, 4);
        const float* bias = read_pointer_launch_arg<const float>(args, 5);
        const int n = read_scalar_launch_arg<int>(args, 6);
        const int c = read_scalar_launch_arg<int>(args, 7);
        if (out == nullptr || inp == nullptr || weight == nullptr || bias == nullptr || n <= 0 || c <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int row = 0; row < n; ++row) {
            const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
            const float* x = inp + base;
            float sum = 0.0f;
            for (int ci = 0; ci < c; ++ci) {
                sum += x[ci];
            }
            const float m = sum / static_cast<float>(c);
            if (mean != nullptr) {
                mean[row] = m;
            }
            float var_sum = 0.0f;
            for (int ci = 0; ci < c; ++ci) {
                const float diff = x[ci] - m;
                var_sum += diff * diff;
            }
            const float s = 1.0f / std::sqrt(var_sum / static_cast<float>(c) + 1.0e-5f);
            if (rstd != nullptr) {
                rstd[row] = s;
            }
            float* o = out + base;
            for (int ci = 0; ci < c; ++ci) {
                const float norm = (x[ci] - m) * s;
                o[ci] = norm * weight[ci] + bias[ci];
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "permute_kernel_backward") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* dq = read_pointer_launch_arg<const float>(args, 1);
        const float* dk = read_pointer_launch_arg<const float>(args, 2);
        const float* dv = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int n = read_scalar_launch_arg<int>(args, 5);
        const int nh = read_scalar_launch_arg<int>(args, 6);
        const int d = read_scalar_launch_arg<int>(args, 7);
        if (dinp == nullptr || dq == nullptr || dk == nullptr || dv == nullptr || b <= 0 || n <= 0 ||
            nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t inp_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(3 * nh * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(3 * nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        dinp[inp_idx] = dq[idx];
                        dinp[inp_idx + static_cast<std::size_t>(nh * d)] = dk[idx];
                        dinp[inp_idx + static_cast<std::size_t>(2 * nh * d)] = dv[idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "permute_kernel") &&
        !kernel_name_contains(kernel_name, "unpermute_kernel")) {
        if (arg_count < 8) {
            return cudaErrorInvalidValue;
        }
        float* q = read_pointer_launch_arg<float>(args, 0);
        float* k = read_pointer_launch_arg<float>(args, 1);
        float* v = read_pointer_launch_arg<float>(args, 2);
        const float* inp = read_pointer_launch_arg<const float>(args, 3);
        const int b = read_scalar_launch_arg<int>(args, 4);
        const int n = read_scalar_launch_arg<int>(args, 5);
        const int nh = read_scalar_launch_arg<int>(args, 6);
        const int d = read_scalar_launch_arg<int>(args, 7);
        if (q == nullptr || k == nullptr || v == nullptr || inp == nullptr || b <= 0 || n <= 0 ||
            nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t inp_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(n) *
                                 static_cast<std::size_t>(3 * nh * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(3 * nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        q[idx] = inp[inp_idx];
                        k[idx] = inp[inp_idx + static_cast<std::size_t>(nh * d)];
                        v[idx] = inp[inp_idx + static_cast<std::size_t>(2 * nh * d)];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "unpermute_kernel_backward")) {
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* dout = read_pointer_launch_arg<const float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int nh = read_scalar_launch_arg<int>(args, 4);
        const int d = read_scalar_launch_arg<int>(args, 5);
        if (dinp == nullptr || dout == nullptr || b <= 0 || n <= 0 || nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t other_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) *
                                 static_cast<std::size_t>(n * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        dinp[idx] = dout[other_idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "unpermute_kernel")) {
        if (arg_count < 6) {
            return cudaErrorInvalidValue;
        }
        float* inp = read_pointer_launch_arg<float>(args, 0);
        float* out = read_pointer_launch_arg<float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int nh = read_scalar_launch_arg<int>(args, 4);
        const int d = read_scalar_launch_arg<int>(args, 5);
        if (inp == nullptr || out == nullptr || b <= 0 || n <= 0 || nh <= 0 || d <= 0) {
            return cudaErrorInvalidValue;
        }

        for (int bi = 0; bi < b; ++bi) {
            for (int nhi = 0; nhi < nh; ++nhi) {
                for (int ni = 0; ni < n; ++ni) {
                    for (int di = 0; di < d; ++di) {
                        const std::size_t idx =
                            (((static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) +
                               static_cast<std::size_t>(nhi)) *
                                  static_cast<std::size_t>(n) +
                              static_cast<std::size_t>(ni)) *
                                 static_cast<std::size_t>(d)) +
                            static_cast<std::size_t>(di);
                        const std::size_t other_idx =
                            (static_cast<std::size_t>(bi) * static_cast<std::size_t>(nh) *
                                 static_cast<std::size_t>(n * d)) +
                            (static_cast<std::size_t>(ni) * static_cast<std::size_t>(nh * d)) +
                            static_cast<std::size_t>(nhi * d + di);
                        out[other_idx] = inp[idx];
                    }
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "softmax_forward_kernel5")) {
        if (arg_count < 5) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float inv_temperature = read_scalar_launch_arg<float>(args, 1);
        const float* inp = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        const int t = read_scalar_launch_arg<int>(args, 4);
        if (out == nullptr || inp == nullptr || n <= 0 || t <= 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t rows = static_cast<std::size_t>(n) * static_cast<std::size_t>(t);
        for (std::size_t row = 0; row < rows; ++row) {
            const int own_pos = static_cast<int>(row % static_cast<std::size_t>(t));
            const float* x = inp + row * static_cast<std::size_t>(t);
            float* y = out + row * static_cast<std::size_t>(t);

            float max_val = -FLT_MAX;
            for (int i = 0; i <= own_pos; ++i) {
                max_val = std::max(max_val, x[i]);
            }

            float sum = 0.0f;
            for (int i = 0; i <= own_pos; ++i) {
                sum += std::exp(inv_temperature * (x[i] - max_val));
            }
            const float norm = sum > 0.0f ? (1.0f / sum) : 0.0f;

            for (int i = 0; i <= own_pos; ++i) {
                y[i] = std::exp(inv_temperature * (x[i] - max_val)) * norm;
            }
            for (int i = own_pos + 1; i < t; ++i) {
                y[i] = 0.0f;
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "residual_forward_kernel")) {
        if (arg_count < 4) {
            return cudaErrorInvalidValue;
        }
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float* inp1 = read_pointer_launch_arg<const float>(args, 1);
        const float* inp2 = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        if (out == nullptr || inp1 == nullptr || inp2 == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            out[i] = inp1[i] + inp2[i];
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "gelu_forward_kernel")) {
        if (arg_count < 3) {
            return cudaErrorInvalidValue;
        }
        constexpr float kGeluScaling = 0.7978845608028654f;
        float* out = read_pointer_launch_arg<float>(args, 0);
        const float* inp = read_pointer_launch_arg<const float>(args, 1);
        const int n = read_scalar_launch_arg<int>(args, 2);
        if (out == nullptr || inp == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            const float x = inp[i];
            const float cube = 0.044715f * x * x * x;
            out[i] = 0.5f * x * (1.0f + std::tanh(kGeluScaling * (x + cube)));
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "gelu_backward_kernel")) {
        if (arg_count < 4) {
            return cudaErrorInvalidValue;
        }
        constexpr float kGeluScaling = 0.7978845608028654f;
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        const float* inp = read_pointer_launch_arg<const float>(args, 1);
        const float* dout = read_pointer_launch_arg<const float>(args, 2);
        const int n = read_scalar_launch_arg<int>(args, 3);
        if (dinp == nullptr || inp == nullptr || dout == nullptr || n < 0) {
            return cudaErrorInvalidValue;
        }
        for (int i = 0; i < n; ++i) {
            const float x = inp[i];
            const float cube = 0.044715f * x * x * x;
            const float tanh_arg = kGeluScaling * (x + cube);
            const float tanh_out = std::tanh(tanh_arg);
            const float cosh_out = std::cosh(tanh_arg);
            const float sech2 = 1.0f / (cosh_out * cosh_out);
            const float local_grad = 0.5f * (1.0f + tanh_out) +
                                     x * 0.5f * sech2 * kGeluScaling *
                                         (1.0f + 3.0f * 0.044715f * x * x);
            dinp[i] = local_grad * dout[i];
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "matmul_backward_bias_kernel4")) {
        if (arg_count < 5) {
            return cudaErrorInvalidValue;
        }
        float* dbias = read_pointer_launch_arg<float>(args, 0);
        const float* dout = read_pointer_launch_arg<const float>(args, 1);
        const int b = read_scalar_launch_arg<int>(args, 2);
        const int t = read_scalar_launch_arg<int>(args, 3);
        const int oc = read_scalar_launch_arg<int>(args, 4);
        if (dbias == nullptr || dout == nullptr || b <= 0 || t <= 0 || oc <= 0) {
            return cudaErrorInvalidValue;
        }
        const int rows = b * t;
        for (int col = 0; col < oc; ++col) {
            float sum = 0.0f;
            for (int row = 0; row < rows; ++row) {
                sum += dout[static_cast<std::size_t>(row) * static_cast<std::size_t>(oc) +
                            static_cast<std::size_t>(col)];
            }
            dbias[col] += sum;
        }
        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "layernorm_backward_kernel2")) {
        if (arg_count < 11) {
            return cudaErrorInvalidValue;
        }
        float* dinp = read_pointer_launch_arg<float>(args, 0);
        float* dweight = read_pointer_launch_arg<float>(args, 1);
        float* dbias = read_pointer_launch_arg<float>(args, 2);
        const float* dout = read_pointer_launch_arg<const float>(args, 3);
        const float* inp = read_pointer_launch_arg<const float>(args, 4);
        const float* weight = read_pointer_launch_arg<const float>(args, 5);
        const float* mean = read_pointer_launch_arg<const float>(args, 6);
        const float* rstd = read_pointer_launch_arg<const float>(args, 7);
        const int b = read_scalar_launch_arg<int>(args, 8);
        const int t = read_scalar_launch_arg<int>(args, 9);
        const int c = read_scalar_launch_arg<int>(args, 10);
        if (dinp == nullptr || dweight == nullptr || dbias == nullptr || dout == nullptr || inp == nullptr ||
            weight == nullptr || mean == nullptr || rstd == nullptr || b <= 0 || t <= 0 || c <= 0) {
            return cudaErrorInvalidValue;
        }

        const int n = b * t;
        const float inv_c = 1.0f / static_cast<float>(c);
        const int warps_per_block = std::max(1u, block_dim.x / 32u);
        const int block_count = static_cast<int>(grid_dim.x);

        std::vector<float> block_dbias(static_cast<std::size_t>(c));
        std::vector<float> block_dweight(static_cast<std::size_t>(c));

        for (int block = 0; block < block_count; ++block) {
            std::fill(block_dbias.begin(), block_dbias.end(), 0.0f);
            std::fill(block_dweight.begin(), block_dweight.end(), 0.0f);

            for (int warp_rank = 0; warp_rank < warps_per_block; ++warp_rank) {
                const int row = block * warps_per_block + warp_rank;
                if (row >= n) {
                    continue;
                }

                const std::size_t base =
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(c);
                const float* dout_row = dout + base;
                const float* inp_row = inp + base;
                float* dinp_row = dinp + base;
                const float mean_row = mean[row];
                const float rstd_row = rstd[row];

                float dnorm_mean = 0.0f;
                float dnorm_norm_mean = 0.0f;
                for (int ci = 0; ci < c; ++ci) {
                    const float norm = (inp_row[ci] - mean_row) * rstd_row;
                    const float dnorm = weight[ci] * dout_row[ci];
                    dnorm_mean += dnorm;
                    dnorm_norm_mean += dnorm * norm;
                }
                dnorm_mean *= inv_c;
                dnorm_norm_mean *= inv_c;

                for (int ci = 0; ci < c; ++ci) {
                    const float norm = (inp_row[ci] - mean_row) * rstd_row;
                    const float dnorm = weight[ci] * dout_row[ci];
                    block_dbias[static_cast<std::size_t>(ci)] += dout_row[ci];
                    block_dweight[static_cast<std::size_t>(ci)] += norm * dout_row[ci];
                    const float dval = (dnorm - dnorm_mean - norm * dnorm_norm_mean) * rstd_row;
                    dinp_row[ci] += dval;
                }
            }

            for (int ci = 0; ci < c; ++ci) {
                dbias[ci] += block_dbias[static_cast<std::size_t>(ci)];
                dweight[ci] += block_dweight[static_cast<std::size_t>(ci)];
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "softmax_autoregressive_backward_kernel")) {
        if (arg_count < 7) {
            return cudaErrorInvalidValue;
        }
        float* dpreatt = read_pointer_launch_arg<float>(args, 0);
        const float* datt = read_pointer_launch_arg<const float>(args, 1);
        const float* att = read_pointer_launch_arg<const float>(args, 2);
        const int t = read_scalar_launch_arg<int>(args, 4);
        const float scale = read_scalar_launch_arg<float>(args, 6);
        if (dpreatt == nullptr || datt == nullptr || att == nullptr || t <= 0) {
            return cudaErrorInvalidValue;
        }

        const int heads = static_cast<int>(grid_dim.y);
        if (heads <= 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t head_stride =
            static_cast<std::size_t>(t) * static_cast<std::size_t>(t);
        for (int head = 0; head < heads; ++head) {
            const std::size_t head_base = static_cast<std::size_t>(head) * head_stride;
            for (int row = 0; row < t; ++row) {
                const std::size_t row_base =
                    head_base + static_cast<std::size_t>(row) * static_cast<std::size_t>(t);
                float local_sum = 0.0f;
                for (int col = 0; col <= row; ++col) {
                    local_sum += att[row_base + static_cast<std::size_t>(col)] *
                                 datt[row_base + static_cast<std::size_t>(col)];
                }
                for (int col = 0; col <= row; ++col) {
                    const float a = att[row_base + static_cast<std::size_t>(col)];
                    const float da = datt[row_base + static_cast<std::size_t>(col)];
                    dpreatt[row_base + static_cast<std::size_t>(col)] = scale * a * (da - local_sum);
                }
                for (int col = row + 1; col < t; ++col) {
                    dpreatt[row_base + static_cast<std::size_t>(col)] = 0.0f;
                }
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "adamw_kernel2")) {
        if (arg_count < 12) {
            return cudaErrorInvalidValue;
        }
        float* params = read_pointer_launch_arg<float>(args, 0);
        float* grads = read_pointer_launch_arg<float>(args, 1);
        float* m = read_pointer_launch_arg<float>(args, 2);
        float* v = read_pointer_launch_arg<float>(args, 3);
        const std::int64_t num_parameters = read_scalar_launch_arg<std::int64_t>(args, 4);
        const float learning_rate = read_scalar_launch_arg<float>(args, 5);
        const float beta1 = read_scalar_launch_arg<float>(args, 6);
        const float beta2 = read_scalar_launch_arg<float>(args, 7);
        const float beta1_correction = read_scalar_launch_arg<float>(args, 8);
        const float beta2_correction = read_scalar_launch_arg<float>(args, 9);
        const float eps = read_scalar_launch_arg<float>(args, 10);
        const float weight_decay = read_scalar_launch_arg<float>(args, 11);
        if (params == nullptr || grads == nullptr || m == nullptr || v == nullptr || num_parameters < 0) {
            return cudaErrorInvalidValue;
        }

        const std::size_t count = static_cast<std::size_t>(num_parameters);
        for (std::size_t i = 0; i < count; ++i) {
            const float grad = grads[i];
            float m_val = m[i];
            float v_val = v[i];
            m_val = beta1 * m_val + (1.0f - beta1) * grad;
            v_val = beta2 * v_val + (1.0f - beta2) * (grad * grad);
            m[i] = m_val;
            v[i] = v_val;
            const float m_hat = m_val / beta1_correction;
            const float v_hat = v_val / beta2_correction;
            params[i] -= learning_rate *
                         (m_hat / (std::sqrt(v_hat) + eps) + weight_decay * params[i]);
        }

        *handled = true;
        return cudaSuccess;
    }

    if (kernel_name_contains(kernel_name, "fused_classifier_kernel3")) {
        if (arg_count < 9) {
            return cudaErrorInvalidValue;
        }
        float* logits = read_pointer_launch_arg<float>(args, 0);
        float* losses = read_pointer_launch_arg<float>(args, 1);
        float* probs = read_pointer_launch_arg<float>(args, 2);
        const float* dlosses = read_pointer_launch_arg<const float>(args, 3);
        const int* targets = read_pointer_launch_arg<const int>(args, 4);
        const int b = read_scalar_launch_arg<int>(args, 5);
        const int t = read_scalar_launch_arg<int>(args, 6);
        const int v = read_scalar_launch_arg<int>(args, 7);
        const int p = read_scalar_launch_arg<int>(args, 8);
        if (logits == nullptr || losses == nullptr || targets == nullptr || b <= 0 || t <= 0 || v <= 0 ||
            p <= 0 || v > p) {
            return cudaErrorInvalidValue;
        }

        const int n = b * t;
        const float default_dloss = 1.0f / static_cast<float>(n);
        for (int row = 0; row < n; ++row) {
            const int target = targets[row];
            if (target < 0 || target >= v) {
                return cudaErrorInvalidValue;
            }

            float* row_logits = logits + static_cast<std::size_t>(row) * static_cast<std::size_t>(p);
            float max_val = -FLT_MAX;
            for (int col = 0; col < v; ++col) {
                max_val = std::max(max_val, row_logits[col]);
            }

            float sum = 0.0f;
            for (int col = 0; col < v; ++col) {
                sum += std::exp(row_logits[col] - max_val);
            }
            const float inv_sum = sum > 0.0f ? (1.0f / sum) : 0.0f;
            const float target_prob = std::exp(row_logits[target] - max_val) * inv_sum;
            losses[row] = -std::log(std::max(target_prob, 1.0e-30f));

            const float dloss = dlosses != nullptr ? dlosses[row] : default_dloss;
            for (int col = 0; col < v; ++col) {
                const float prob = std::exp(row_logits[col] - max_val) * inv_sum;
                if (probs != nullptr) {
                    probs[static_cast<std::size_t>(row) * static_cast<std::size_t>(p) +
                          static_cast<std::size_t>(col)] = prob;
                }
                const float indicator = (col == target) ? 1.0f : 0.0f;
                row_logits[col] = (prob - indicator) * dloss;
            }
        }

        *handled = true;
        return cudaSuccess;
    }

    return cudaSuccess;
}

}  // namespace

namespace cumetal::rt {

bool resolve_allocation_for_pointer(const void* ptr, AllocationTable::ResolvedAllocation* out) {
    if (ptr == nullptr || out == nullptr) {
        return false;
    }
    RuntimeState& state = runtime_state();
    return state.allocations.resolve(ptr, out);
}

cudaError_t enqueue_host_operation(cudaStream_t stream, std::function<void()> operation) {
    return enqueue_stream_host_op(stream, std::move(operation));
}

cudaError_t resolve_backend_stream(cudaStream_t stream,
                                   std::shared_ptr<cumetal::metal_backend::Stream>* out) {
    return resolve_runtime_stream(stream, out, nullptr);
}

bool capture_library_call(cudaStream_t stream, std::function<cudaError_t(cudaStream_t)> op) {
    cudaGraph_t graph = get_capture_graph(stream);
    if (graph == nullptr || !op) {
        return false;
    }
    auto* node = new (std::nothrow) cudaGraphNode_st();
    if (node == nullptr) {
        return false;
    }
    node->type = cudaGraphNodeTypeKernel;
    node->library_op = std::move(op);
    append_captured_graph_node(graph, node);
    return true;
}

}  // namespace cumetal::rt

extern "C" {

int cumetalRuntimeIsDevicePointer(const void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    return state.allocations.resolve(ptr, &resolved) ? 1 : 0;
}

// Returns 1 if ptr is in a known allocation; sets *base_out and *size_out.
// base_out = start of the containing allocation, size_out = total allocation size.
int cumetalRuntimeGetAllocationInfo(const void* ptr, void** base_out, size_t* size_out) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(ptr, &resolved)) {
        return 0;
    }
    const auto raw = reinterpret_cast<std::uintptr_t>(ptr);
    // base = ptr - offset into allocation
    const std::uintptr_t base_addr = raw - resolved.offset;
    if (base_out) *base_out = reinterpret_cast<void*>(base_addr);
    if (size_out) *size_out = resolved.offset + resolved.remaining_size;
    return 1;
}

void* cumetalRuntimeGetHostPointer(const void* ptr, size_t count) {
    return const_cast<void*>(host_accessible_pointer(ptr, count));
}

// Returns 1 if ptr is a managed (unified) allocation.
int cumetalRuntimeIsManaged(const void* ptr) {
    if (ptr == nullptr) {
        return 0;
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(ptr, &resolved)) {
        return 0;
    }
    // cuMemAllocManaged allocations have the same coherent UMA properties.
    // We can distinguish them by host_alloc_flags (managed = no host flags set).
    return (resolved.kind == cumetal::rt::AllocationKind::kDevice) ? 1 : 0;
}

cudaError_t cudaInit(unsigned int flags) {
    if (flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t status = ensure_initialized();
    return fail(status);
}

cudaError_t cudaDriverGetVersion(int* driver_version) {
    if (driver_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *driver_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaRuntimeGetVersion(int* runtime_version) {
    if (runtime_version == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *runtime_version = kCudaCompatVersion;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceCount(int* count) {
    if (count == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    *count = 1;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDevice(int* device) {
    if (device == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *device = state.current_device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDevice(int device) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }
    if (cumetalDriverRuntimeActivatePrimaryContext(device) == 0) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    state.current_device = device;
    return fail(cudaSuccess);
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    const cudaError_t flags_status = validate_device_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    RuntimeState& state = runtime_state();
    state.device_flags = flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceFlags(unsigned int* flags) {
    if (flags == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    *flags = state.device_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int device) {
    if (prop == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (device != 0) {
        return fail(cudaErrorInvalidValue);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    std::memset(prop, 0, sizeof(*prop));
    std::strncpy(prop->name, backend_props.name.c_str(), sizeof(prop->name) - 1);
    prop->name[sizeof(prop->name) - 1] = '\0';
    prop->totalGlobalMem = backend_props.total_global_mem;
    prop->warpSize = 32;
    // Metal exposes GPU-core count but no public guarantee that an arbitrary
    // kernel has one threadgroup simultaneously resident on every core. CUDA's
    // standard cooperative-launch sizing formula multiplies this value by
    // occupancy, so expose one guaranteed-progress partition. Explicit
    // cooperative launches retain the separately tested four-block ceiling.
    prop->multiProcessorCount = kCudaVisibleMultiprocessorCount;
    prop->maxThreadsPerBlock = backend_props.max_threads_per_block > 0
                                   ? std::min(backend_props.max_threads_per_block, 1024)
                                   : 1024;
    prop->maxThreadsDim[0] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[1] = prop->maxThreadsPerBlock;
    prop->maxThreadsDim[2] = prop->maxThreadsPerBlock;
    prop->maxGridSize[0] = 2147483647;
    prop->maxGridSize[1] = 65535;
    prop->maxGridSize[2] = 65535;
    prop->sharedMemPerBlock =
        backend_props.shared_mem_per_block > 0 ? backend_props.shared_mem_per_block : (32 * 1024);
    prop->sharedMemPerBlockOptin = static_cast<size_t>(prop->sharedMemPerBlock);
    prop->regsPerBlock = 65536;
    prop->major = 8;   // Synthetic Ampere-equivalent per spec §6.8
    prop->minor = 0;
    prop->unifiedAddressing = 1;        // UMA: CPU and GPU share physical DRAM
    prop->managedMemory = 1;            // cudaMallocManaged == cudaMalloc on UMA
    // Reported 0, not 1. CUDA's promise here is *coherent* concurrent access:
    // the host may read and write managed memory while a kernel is running and
    // both see each other's stores. Metal does not offer that. A shared-storage
    // MTLBuffer is visible to both, but the memory model only guarantees the
    // host observes a kernel's writes once its command buffer has completed,
    // and there is no system-scope atomic between CPU and GPU at all -- Metal's
    // atomics are device scope.
    //
    // Claiming 1 is not harmless, because callers branch on it. NVIDIA's
    // systemWideAtomics sample skips its own cudaDeviceSynchronize when this is
    // 1 and then has the CPU and a kernel increment the same counter through
    // atomicAdd_system; on CuMetal the device's increments were simply lost and
    // it failed, deterministically. With 0 it takes the serialized path the
    // attribute is there to select, and passes for the right reason.
    prop->concurrentManagedAccess = 0;
    prop->maxBufferArguments = 31;      // Metal buffer argument limit per kernel
    // Additional fields (spec §6.8)
    prop->clockRate = 1296000;          // ~1.3 GHz in kHz (conservative estimate)
    prop->memoryClockRate = 1296000;    // Same as GPU clock on UMA (shared memory controller)
    prop->memoryBusWidth = 128;         // 128-bit memory bus (conservative; M-series varies)
    prop->totalConstMem = 64 * 1024;    // 64 KB constant memory per module (spec §5.4.1)
    prop->sharedMemPerMultiprocessor = prop->sharedMemPerBlock; // Same as per-block on Metal
    prop->maxThreadsPerMultiProcessor = 2048; // Conservative estimate for M-series
    prop->l2CacheSize = 4 * 1024 * 1024; // 4 MB L2 (varies by chip; conservative)
    prop->canMapHostMemory = 1;          // UMA: all host memory is device-accessible
    prop->integrated = 1;               // Apple Silicon is an integrated GPU
    prop->concurrentKernels = 1;        // Metal supports concurrent command buffer execution
    prop->asyncEngineCount = 0;         // UMA makes async memcpy a memcpy, no DMA engine
    prop->computeMode = 0;              // 0 = cudaComputeModeDefault
    prop->pciBusID = 0;                 // No PCI bus on Apple Silicon
    prop->pciDeviceID = 0;
    prop->pciDomainID = 0;
    prop->tccDriver = 0;                // Not a Tesla compute cluster driver
    prop->kernelExecTimeoutEnabled = 0; // Metal does not enforce kernel timeout by default
    // Apple Silicon is UMA, but CuMetal can only bind pointers backed by a
    // tracked MTLBuffer. Arbitrary malloc pointers are therefore not currently
    // valid kernel arguments; advertising pageable access made applications
    // select a path the launch code rejects.
    prop->pageableMemoryAccess = 0;
    prop->pageableMemoryAccessUsesHostPageTables = 0;
    // Cooperative grids are bounded to one resident threadgroup per reported
    // processor and use the compiler-injected device-wide barrier state.
    prop->cooperativeLaunch = 1;
    prop->cooperativeMultiDeviceLaunch = 0;
    // Metal owns physical cache residency, but CUDA access-policy windows are
    // correctness-neutral performance hints. Accept and round-trip a
    // conservative quarter-L2 window so portable callers need no alternate
    // control path.
    prop->persistingL2CacheMaxSize = prop->l2CacheSize / 4;
    prop->accessPolicyMaxWindowSize = prop->l2CacheSize / 4;
    // Apple Silicon unified memory is not ECC-protected. GROMACS prints this
    // field in its hardware summary.
    prop->ECCEnabled = 0;
    for (size_t i = 0; i < sizeof(prop->cumetalReserved) / sizeof(prop->cumetalReserved[0]); ++i) {
        prop->cumetalReserved[i] = 0;
    }

    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetAttribute(int* value, int attr, int device) {
    if (value == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    cudaDeviceProp prop{};
    const cudaError_t status = cudaGetDeviceProperties(&prop, device);
    if (status != cudaSuccess) {
        return fail(status);
    }

    switch (attr) {
        case cudaDevAttrMaxThreadsPerBlock:
            *value = prop.maxThreadsPerBlock;
            break;
        case cudaDevAttrMaxBlockDimX:
            *value = prop.maxThreadsDim[0];
            break;
        case cudaDevAttrMaxBlockDimY:
            *value = prop.maxThreadsDim[1];
            break;
        case cudaDevAttrMaxBlockDimZ:
            *value = prop.maxThreadsDim[2];
            break;
        case cudaDevAttrMaxGridDimX:
            *value = prop.maxGridSize[0];
            break;
        case cudaDevAttrMaxGridDimY:
            *value = prop.maxGridSize[1];
            break;
        case cudaDevAttrMaxGridDimZ:
            *value = prop.maxGridSize[2];
            break;
        case cudaDevAttrMaxSharedMemoryPerBlock:
            *value = prop.sharedMemPerBlock;
            break;
        case cudaDevAttrWarpSize:
            *value = prop.warpSize;
            break;
        case cudaDevAttrMultiProcessorCount:
            *value = prop.multiProcessorCount;
            break;
        case cudaDevAttrMaxRegistersPerBlock:
            *value = 65536;  // Metal has no per-block register limit; return generous value
            break;
        case cudaDevAttrClockRate:
            *value = 1296000;  // kHz — conservative estimate for M-series GPU
            break;
        case cudaDevAttrTextureAlignment:
            *value = 512;
            break;
        case cudaDevAttrGpuOverlap:
            *value = 1;  // Metal supports async compute + copy overlap
            break;
        case cudaDevAttrComputeCapabilityMajor:
            *value = 8;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case cudaDevAttrComputeCapabilityMinor:
            *value = 0;  // Ampere-equivalent feature set (spec §6.8)
            break;
        case cudaDevAttrUnifiedAddressing:
        case cudaDevAttrManagedMemory:
        case cudaDevAttrCanMapHostMemory:
        case cudaDevAttrIntegrated:
        case cudaDevAttrConcurrentKernels:
        case cudaDevAttrMemoryPoolsSupported:
            *value = 1;
            break;
        case cudaDevAttrConcurrentManagedAccess:
        case cudaDevAttrPageableMemoryAccess:
        case cudaDevAttrPageableMemoryAccessUsesHostPageTables:
            // See cudaDeviceProp::concurrentManagedAccess: sharing the address
            // space is not the same promise as coherent concurrent access, and
            // Metal only makes the first one.
            *value = 0;
            break;
        case cudaDevAttrCooperativeLaunch:
            *value = 1;
            break;
        case cudaDevAttrMemoryBusWidth:
            *value = 128;
            break;
        case cudaDevAttrL2CacheSize:
            *value = 4 * 1024 * 1024;
            break;
        case cudaDevAttrMaxThreadsPerMultiProcessor:
            *value = 2048;
            break;
        case cudaDevAttrMemoryClockRate:
            *value = 1296000;
            break;
        case cudaDevAttrComputeMode:
        case cudaDevAttrPciBusId:
        case cudaDevAttrPciDeviceId:
        case cudaDevAttrPciDomainId:
        case cudaDevAttrTccDriver:
        case cudaDevAttrKernelExecTimeout:
        case cudaDevAttrAsyncEngineCount:
            *value = 0;
            break;
        case cudaDevAttrSharedMemPerBlockOptin: {
            cudaDeviceProp prop{};
            cudaGetDeviceProperties(&prop, device);
            *value = static_cast<int>(prop.sharedMemPerBlockOptin);
            break;
        }
        default:
            return fail(cudaErrorInvalidValue);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemGetInfo(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cumetal::metal_backend::DeviceProperties backend_props;
    std::string error;
    const cudaError_t query_status =
        cumetal::metal_backend::query_device_properties(&backend_props, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }

    RuntimeState& state = runtime_state();
    const std::size_t allocated_bytes = state.allocations.total_allocated_size();
    const std::size_t total_mem = backend_props.total_global_mem;

    *total_bytes = total_mem;
    *free_bytes = allocated_bytes >= total_mem ? 0 : (total_mem - allocated_bytes);
    return fail(cudaSuccess);
}

cudaError_t cudaMalloc(void** dev_ptr, size_t size) {
    if (dev_ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(size, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr) {
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation : alloc_status);
    }

    void* host_base = buffer->contents();
    const std::uintptr_t device_address = buffer->device_address();
    if (host_base == nullptr || (use_metal_device_addresses() && device_address == 0)) {
        return fail(cudaErrorMemoryAllocation);
    }
    void* device_base = reinterpret_cast<void*>(device_address);
    void* base = use_metal_device_addresses() ? device_base : host_base;

    RuntimeState& state = runtime_state();
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kDevice,
                                  /*host_alloc_flags=*/0,
                                  buffer, &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    // Metal kernels store MTLBuffer GPU virtual addresses when they write
    // pointers into device-resident tables.  CUDA's host-facing pointer can be
    // either that address or the shared-memory CPU mapping, depending on the
    // compatibility mode.  Track both identities so a pointer produced on
    // either side resolves to the same allocation and byte offset.
    void* alias = (base == host_base) ? device_base : host_base;
    if (alias != nullptr && alias != base &&
        !state.allocations.insert(alias, size, cumetal::rt::AllocationKind::kDevice,
                                  /*host_alloc_flags=*/0,
                                  buffer, &error, /*alias=*/true)) {
        state.allocations.erase(base);
        return fail(cudaErrorMemoryAllocation);
    }

    *dev_ptr = base;
    if (trace_enabled()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "malloc size=%zu ptr=%p", size, base);
        trace_op("MALLOC", buf);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMallocManaged(void** dev_ptr, size_t size, unsigned int flags) {
    // CUDA's C++ overload defaults to cudaMemAttachGlobal. Zero was accepted
    // by older CuMetal headers, so retain it as a compatibility spelling.
    // cudaMemAttachHost makes an allocation initially host-accessible. Shared
    // storage on Apple Silicon already has that property, and later stream
    // attachment is an ordering hint rather than a migration. SINGLE still
    // needs per-stream accessibility state and remains unsupported.
    if (flags != 0 && flags != cudaMemAttachGlobal && flags != cudaMemAttachHost) {
        return fail(cudaErrorInvalidValue);
    }
    return cudaMalloc(dev_ptr, size);
}

// Pitched 2D allocation — align pitch to 512 bytes (matching cudaDevAttrTextureAlignment).
cudaError_t cudaMallocPitch(void** dev_ptr, size_t* pitch, size_t width, size_t height) {
    if (dev_ptr == nullptr || pitch == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    constexpr size_t kAlign = 512;
    *pitch = (width + kAlign - 1) & ~(kAlign - 1);
    return cudaMalloc(dev_ptr, *pitch * height);
}

cudaError_t cudaMalloc3D(cudaPitchedPtr* pitchedDevPtr, cudaExtent extent) {
    if (pitchedDevPtr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (extent.width == 0 || extent.height == 0 || extent.depth == 0) {
        pitchedDevPtr->ptr   = nullptr;
        pitchedDevPtr->pitch = 0;
        pitchedDevPtr->xsize = extent.width;
        pitchedDevPtr->ysize = extent.height;
        return fail(cudaSuccess);
    }
    constexpr size_t kAlign = 512;
    size_t pitch = (extent.width + kAlign - 1) & ~(kAlign - 1);
    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, pitch * extent.height * extent.depth);
    if (err != cudaSuccess) return err;
    pitchedDevPtr->ptr   = ptr;
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
    return fail(cudaSuccess);
}

cudaError_t cudaHostAlloc(void** ptr, size_t size, unsigned int flags) {
    if (ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t flags_status = validate_host_alloc_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(size, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr) {
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation : alloc_status);
    }

    void* base = buffer->contents();
    if (base == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    if (!state.allocations.insert(base, size, cumetal::rt::AllocationKind::kHost, flags,
                                  buffer, &error)) {
        return fail(cudaErrorMemoryAllocation);
    }

    void* device_alias = reinterpret_cast<void*>(buffer->device_address());
    if (device_alias != nullptr && device_alias != base) {
        if (!state.allocations.insert(device_alias, size,
                                      cumetal::rt::AllocationKind::kHost, flags,
                                      buffer, &error, /*alias=*/true)) {
            state.allocations.erase(base);
            return fail(cudaErrorMemoryAllocation);
        }
    }

    *ptr = base;
    return fail(cudaSuccess);
}

cudaError_t cudaMallocHost(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocDefault);
}

cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags) {
    if (ptr == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t flags_status = validate_host_register_flags(flags);
    if (flags_status != cudaSuccess) {
        return fail(flags_status);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    // CuMetal uses unified memory on Apple Silicon; host registration is a compatibility no-op.
    return fail(cudaSuccess);
}

cudaError_t cudaHostUnregister(void* ptr) {
    if (ptr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaHostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned int flags) {
    if (dev_ptr == nullptr || host_ptr == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *dev_ptr = use_metal_device_addresses()
                   ? reinterpret_cast<void*>(resolved.buffer->device_address() + resolved.offset)
                   : host_ptr;
    return fail(cudaSuccess);
}

cudaError_t cudaHostGetFlags(unsigned int* flags, void* host_ptr) {
    if (flags == nullptr || host_ptr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(host_ptr, &resolved) || resolved.offset != 0 ||
        resolved.kind != cumetal::rt::AllocationKind::kHost) {
        return fail(cudaErrorInvalidValue);
    }

    *flags = resolved.host_alloc_flags;
    return fail(cudaSuccess);
}

cudaError_t cudaFreeHost(void* ptr) {
    return cudaFree(ptr);
}

cudaError_t cudaFree(void* dev_ptr) {
    if (dev_ptr == nullptr) {
        return fail(cudaSuccess);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    RuntimeState& state = runtime_state();
    std::shared_ptr<GraphAllocationState> graph_allocation;
    {
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        const auto found = state.graph_allocations.find(dev_ptr);
        if (found != state.graph_allocations.end()) {
            graph_allocation = found->second.lock();
            if (graph_allocation == nullptr) state.graph_allocations.erase(found);
        }
    }
    if (graph_allocation != nullptr) {
        const cudaError_t status = deactivate_graph_allocation(graph_allocation);
        return fail(status == cudaErrorInvalidDevicePointer
                        ? cudaErrorInvalidDevicePointer
                        : status);
    }
    if (!state.allocations.erase(dev_ptr)) {
        return fail(cudaErrorInvalidDevicePointer);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        void* host_dst = host_accessible_pointer(dst, count);
        const void* host_src = host_accessible_pointer(src, count);
        if (host_dst == nullptr || host_src == nullptr) {
            return fail(cudaErrorInvalidValue);
        }
        if (resolved_kind == cudaMemcpyHostToDevice) {
            std::vector<std::uint8_t> staged(count);
            std::memcpy(staged.data(), host_src, count);
            relocate_embedded_device_pointers(&staged);
            std::memcpy(host_dst, staged.data(), count);
        } else if (resolved_kind == cudaMemcpyDeviceToHost) {
            std::vector<std::uint8_t> staged(count);
            std::memcpy(staged.data(), host_src, count);
            restore_embedded_host_pointers(&staged);
            std::memcpy(host_dst, staged.data(), count);
        } else {
            std::memcpy(host_dst, host_src, count);
        }
    }

    if (trace_enabled()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "memcpy kind=%d count=%zu dst=%p src=%p",
                      static_cast<int>(kind), count, dst, src);
        trace_op("CPY", buf);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyAsync(void* dst,
                            const void* src,
                            size_t count,
                            cudaMemcpyKind kind,
                            cudaStream_t stream) {
    if ((dst == nullptr || src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    // Graph capture: record memcpy node instead of executing
    if (cudaGraph_t g = get_capture_graph(stream)) {
        auto* node = new cudaGraphNode_st();
        node->type = cudaGraphNodeTypeMemcpy;
        node->dst = dst;
        node->src = src;
        node->count = count;
        node->memcpy_kind = kind;
        append_captured_graph_node(g, node);
        return fail(cudaSuccess);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    // When both ends are tracked allocations this is a Metal blit in the
    // stream's own command buffer -- a real stream-ordered GPU copy.
    //
    // The fallback below is a host function, and a host function is expensive:
    // two command buffers, a dispatch_async and a shared-event round trip, plus
    // a cudaDeviceSynchronize that then has to wait for it. That is a great deal
    // of machinery for what cuPDLP asks of it, which is to move eight bytes from
    // one device buffer to another between two kernels; it was 77% of a PDLP
    // iteration on datt256. It stays for the cases the blit cannot serve, which
    // is pageable host memory on either end.
    cumetal::rt::AllocationTable::ResolvedAllocation dst_alloc;
    cumetal::rt::AllocationTable::ResolvedAllocation src_alloc;
    if (count > 0 && resolve_allocation_for_pointer(dst, &dst_alloc) &&
        resolve_allocation_for_pointer(src, &src_alloc) &&
        dst_alloc.buffer != nullptr && src_alloc.buffer != nullptr &&
        dst_alloc.remaining_size >= count && src_alloc.remaining_size >= count) {
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        if (resolve_runtime_stream(stream, &backend_stream, nullptr) == cudaSuccess) {
            std::string error;
            const cudaError_t blit_status = cumetal::metal_backend::blit_copy(
                dst_alloc.buffer, dst_alloc.offset, src_alloc.buffer, src_alloc.offset, count,
                backend_stream, &error);
            if (blit_status == cudaSuccess) {
                if (trace_enabled()) {
                    char buf[128];
                    std::snprintf(buf, sizeof(buf),
                                  "memcpyAsync(blit) kind=%d count=%zu dst=%p src=%p",
                                  static_cast<int>(kind), count, dst, src);
                    trace_op("CPYA", buf);
                }
                return fail(cudaSuccess);
            }
        }
    }

    void* host_dst = host_accessible_pointer(dst, count);
    const void* host_src = host_accessible_pointer(src, count);
    if ((host_dst == nullptr || host_src == nullptr) && count > 0) {
        return fail(cudaErrorInvalidValue);
    }
    std::shared_ptr<std::vector<std::uint8_t>> staged_h2d;
    if (count > 0 && resolved_kind == cudaMemcpyHostToDevice) {
        staged_h2d = std::make_shared<std::vector<std::uint8_t>>(count);
        std::memcpy(staged_h2d->data(), host_src, count);
        relocate_embedded_device_pointers(staged_h2d.get());
    }
    const cudaError_t enqueue_status = enqueue_stream_host_op(
        stream, [host_dst, host_src, count, resolved_kind, staged_h2d]() {
            if (count == 0) return;
            if (staged_h2d != nullptr) {
                std::memcpy(host_dst, staged_h2d->data(), count);
            } else if (resolved_kind == cudaMemcpyDeviceToHost) {
                std::vector<std::uint8_t> staged(count);
                std::memcpy(staged.data(), host_src, count);
                restore_embedded_host_pointers(&staged);
                std::memcpy(host_dst, staged.data(), count);
            } else {
                std::memcpy(host_dst, host_src, count);
            }
        });
    if (enqueue_status != cudaSuccess) return fail(enqueue_status);

    if (trace_enabled()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "memcpyAsync kind=%d count=%zu dst=%p src=%p",
                      static_cast<int>(kind), count, dst, src);
        trace_op("CPYA", buf);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbol(const void* symbol,
                               const void* src,
                               size_t count,
                               size_t offset,
                               cudaMemcpyKind kind) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyFromSymbol(void* dst,
                                 const void* symbol,
                                 size_t count,
                                 size_t offset,
                                 cudaMemcpyKind kind) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        std::memcpy(dst, symbol_ptr, count);
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpyToSymbolAsync(const void* symbol,
                                    const void* src,
                                    size_t count,
                                    size_t offset,
                                    cudaMemcpyKind kind,
                                    cudaStream_t stream) {
    if (symbol == nullptr || (src == nullptr && count > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_to_symbol_kind(src, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t enqueue_status = enqueue_stream_host_op(
        stream, [symbol_ptr, src, count]() {
            if (count > 0)
                std::memcpy(const_cast<unsigned char*>(symbol_ptr), src, count);
        });
    return fail(enqueue_status);
}

cudaError_t cudaMemcpyFromSymbolAsync(void* dst,
                                      const void* symbol,
                                      size_t count,
                                      size_t offset,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
    if ((dst == nullptr && count > 0) || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
    const cudaError_t kind_status = resolve_memcpy_from_symbol_kind(dst, kind, &resolved_kind);
    if (kind_status != cudaSuccess) {
        return fail(kind_status);
    }
    (void)resolved_kind;

    const unsigned char* symbol_ptr = nullptr;
    const cudaError_t symbol_status = checked_symbol_ptr(symbol, count, offset, &symbol_ptr);
    if (symbol_status != cudaSuccess) {
        return fail(symbol_status);
    }

    const cudaError_t enqueue_status = enqueue_stream_host_op(
        stream, [dst, symbol_ptr, count]() {
            if (count > 0) std::memcpy(dst, symbol_ptr, count);
        });
    return fail(enqueue_status);
}

cudaError_t cudaMemset(void* dev_ptr, int value, size_t count) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    if (count > 0) {
        void* host_ptr = host_accessible_pointer(dev_ptr, count);
        if (host_ptr == nullptr) {
            return fail(cudaErrorInvalidValue);
        }
        std::memset(host_ptr, value, count);
    }

    if (trace_enabled()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "memset val=%d count=%zu ptr=%p", value, count, dev_ptr);
        trace_op("SET", buf);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemsetAsync(void* dev_ptr, int value, size_t count, cudaStream_t stream) {
    if (dev_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    // Graph capture: record memset node instead of executing
    if (cudaGraph_t g = get_capture_graph(stream)) {
        auto* node = new cudaGraphNode_st();
        node->type = cudaGraphNodeTypeMemset;
        node->dst = dev_ptr;
        node->memset_value = value;
        node->count = count;
        append_captured_graph_node(g, node);
        return fail(cudaSuccess);
    }

    void* host_ptr = host_accessible_pointer(dev_ptr, count);
    if (host_ptr == nullptr && count > 0) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t enqueue_status = enqueue_stream_host_op(
        stream, [host_ptr, value, count]() {
            if (count > 0) std::memset(host_ptr, value, count);
        });
    if (enqueue_status != cudaSuccess) return fail(enqueue_status);

    if (trace_enabled()) {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "memsetAsync val=%d count=%zu ptr=%p", value, count, dev_ptr);
        trace_op("SETA", buf);
    }
    return fail(cudaSuccess);
}

// 2D pitched memcpy — on UMA, copy width bytes per row for height rows.
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch,
                          const void* src, size_t spitch,
                          size_t width, size_t height,
                          cudaMemcpyKind kind) {
    if ((dst == nullptr || src == nullptr) && (width > 0 && height > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (width > 0 && height > 0) {
        cudaMemcpyKind resolved_kind = cudaMemcpyDefault;
        const cudaError_t kind_status = resolve_memcpy_kind(dst, src, kind, &resolved_kind);
        if (kind_status != cudaSuccess) {
            return fail(kind_status);
        }
    }

    // This is a synchronous API. The row copy is performed by the host over
    // unified memory, so all preceding Metal work must be visible first just
    // as it is for cudaMemcpy. In particular, device-to-device copies into a
    // cudaArray are commonly used between a producer kernel and texture fetch.
    std::string sync_error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&sync_error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    auto* d = static_cast<uint8_t*>(host_accessible_pointer(
        dst, height == 0 ? 0 : (height - 1) * dpitch + width));
    const auto* s = static_cast<const uint8_t*>(host_accessible_pointer(
        src, height == 0 ? 0 : (height - 1) * spitch + width));
    if ((d == nullptr || s == nullptr) && width > 0 && height > 0) {
        return fail(cudaErrorInvalidValue);
    }
    for (size_t row = 0; row < height; ++row) {
        if (width > 0) {
            std::memcpy(d + row * dpitch, s + row * spitch, width);
        }
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch,
                               const void* src, size_t spitch,
                               size_t width, size_t height,
                               cudaMemcpyKind kind, cudaStream_t stream) {
    (void)kind;
    auto* d = static_cast<uint8_t*>(host_accessible_pointer(
        dst, height == 0 ? 0 : (height - 1) * dpitch + width));
    const auto* s = static_cast<const uint8_t*>(host_accessible_pointer(
        src, height == 0 ? 0 : (height - 1) * spitch + width));
    if ((d == nullptr || s == nullptr) && width > 0 && height > 0)
        return fail(cudaErrorInvalidValue);
    return fail(enqueue_stream_host_op(stream, [=]() {
        for (size_t row = 0; row < height; ++row)
            if (width > 0) std::memcpy(d + row * dpitch, s + row * spitch, width);
    }));
}

cudaError_t cudaMemset2D(void* dev_ptr, size_t pitch,
                          int value, size_t width, size_t height) {
    if (dev_ptr == nullptr && (width > 0 && height > 0)) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    auto* d = static_cast<uint8_t*>(host_accessible_pointer(
        dev_ptr, height == 0 ? 0 : (height - 1) * pitch + width));
    if (d == nullptr && width > 0 && height > 0) {
        return fail(cudaErrorInvalidValue);
    }
    for (size_t row = 0; row < height; ++row) {
        if (width > 0) {
            std::memset(d + row * pitch, value, width);
        }
    }

    return fail(cudaSuccess);
}

cudaError_t cudaMemset2DAsync(void* dev_ptr, size_t pitch,
                               int value, size_t width, size_t height,
                               cudaStream_t stream) {
    auto* d = static_cast<uint8_t*>(host_accessible_pointer(
        dev_ptr, height == 0 ? 0 : (height - 1) * pitch + width));
    if (d == nullptr && width > 0 && height > 0)
        return fail(cudaErrorInvalidValue);
    return fail(enqueue_stream_host_op(stream, [=]() {
        for (size_t row = 0; row < height; ++row)
            if (width > 0) std::memset(d + row * pitch, value, width);
    }));
}

// cudaMemset3D — fills a 3D pitched allocation plane-by-row.
cudaError_t cudaMemset3D(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent) {
    if (pitchedDevPtr.ptr == nullptr && (extent.width > 0 && extent.height > 0 && extent.depth > 0)) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    const size_t pitch      = pitchedDevPtr.pitch;
    const size_t plane_size = pitch * pitchedDevPtr.ysize;
    const size_t span = extent.depth == 0 || extent.height == 0
                            ? 0
                            : (extent.depth - 1) * plane_size +
                                  (extent.height - 1) * pitch + extent.width;
    auto* base = static_cast<uint8_t*>(
        host_accessible_pointer(pitchedDevPtr.ptr, span));
    if (base == nullptr && extent.width > 0 && extent.height > 0 && extent.depth > 0) {
        return fail(cudaErrorInvalidValue);
    }
    for (size_t z = 0; z < extent.depth; ++z) {
        for (size_t y = 0; y < extent.height; ++y) {
            if (extent.width > 0) {
                std::memset(base + z * plane_size + y * pitch, value, extent.width);
            }
        }
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemset3DAsync(cudaPitchedPtr pitchedDevPtr, int value, cudaExtent extent,
                               cudaStream_t stream) {
    const size_t plane_size = pitchedDevPtr.pitch * pitchedDevPtr.ysize;
    const size_t span = extent.depth == 0 || extent.height == 0
                            ? 0
                            : (extent.depth - 1) * plane_size +
                                  (extent.height - 1) * pitchedDevPtr.pitch + extent.width;
    auto* base = static_cast<uint8_t*>(
        host_accessible_pointer(pitchedDevPtr.ptr, span));
    if (base == nullptr && extent.width > 0 && extent.height > 0 && extent.depth > 0)
        return fail(cudaErrorInvalidValue);
    return fail(enqueue_stream_host_op(stream, [=]() {
        for (size_t z = 0; z < extent.depth; ++z)
            for (size_t y = 0; y < extent.height; ++y)
                if (extent.width > 0)
                    std::memset(base + z * plane_size + y * pitchedDevPtr.pitch,
                                value, extent.width);
    }));
}

cudaError_t cudaMemcpy3D(const cudaMemcpy3DParms* p) {
    if (p == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (p->srcArray != nullptr || p->dstArray != nullptr) {
        const auto* source_array = reinterpret_cast<const CuMetalArray*>(p->srcArray);
        auto* destination_array = reinterpret_cast<CuMetalArray*>(p->dstArray);
        const CuMetalArray* shape_array = source_array != nullptr ? source_array : destination_array;
        if (shape_array == nullptr || (source_array == nullptr && p->srcPtr.ptr == nullptr) ||
            (destination_array == nullptr && p->dstPtr.ptr == nullptr)) {
            return fail(cudaErrorInvalidValue);
        }
        const size_t element_size = static_cast<size_t>(
            (shape_array->desc.x + shape_array->desc.y + shape_array->desc.z +
             shape_array->desc.w + 7) / 8);
        if (element_size == 0) return fail(cudaErrorInvalidValue);
        if (source_array != nullptr && destination_array != nullptr) {
            const size_t destination_element_size = static_cast<size_t>(
                (destination_array->desc.x + destination_array->desc.y +
                 destination_array->desc.z + destination_array->desc.w + 7) / 8);
            if (destination_element_size != element_size) return fail(cudaErrorInvalidValue);
        }
        const size_t width_bytes = p->extent.width * element_size;
        const size_t source_pitch = source_array != nullptr
                                        ? source_array->width * element_size
                                        : (p->srcPtr.pitch ? p->srcPtr.pitch : width_bytes);
        const size_t destination_pitch = destination_array != nullptr
                                             ? destination_array->width * element_size
                                             : (p->dstPtr.pitch ? p->dstPtr.pitch : width_bytes);
        const size_t source_height = source_array != nullptr
                                         ? source_array->height
                                         : (p->srcPtr.ysize ? p->srcPtr.ysize : p->extent.height);
        const size_t destination_height = destination_array != nullptr
                                              ? destination_array->height
                                              : (p->dstPtr.ysize ? p->dstPtr.ysize : p->extent.height);
        const char* source = source_array != nullptr
                                 ? static_cast<const char*>(source_array->data)
                                 : static_cast<const char*>(host_accessible_pointer(
                                       p->srcPtr.ptr, source_pitch * source_height * p->extent.depth));
        char* destination = destination_array != nullptr
                                ? static_cast<char*>(destination_array->data)
                                : static_cast<char*>(host_accessible_pointer(
                                      p->dstPtr.ptr,
                                      destination_pitch * destination_height * p->extent.depth));
        if (source == nullptr || destination == nullptr) return fail(cudaErrorInvalidValue);
        const size_t source_x = p->srcPos.x * (source_array != nullptr ? element_size : 1);
        const size_t destination_x =
            p->dstPos.x * (destination_array != nullptr ? element_size : 1);
        for (size_t z = 0; z < p->extent.depth; ++z) {
            for (size_t y = 0; y < p->extent.height; ++y) {
                const char* source_row = source +
                    (p->srcPos.z + z) * source_pitch * source_height +
                    (p->srcPos.y + y) * source_pitch + source_x;
                char* destination_row = destination +
                    (p->dstPos.z + z) * destination_pitch * destination_height +
                    (p->dstPos.y + y) * destination_pitch + destination_x;
                std::memcpy(destination_row, source_row, width_bytes);
            }
        }
        return fail(cudaSuccess);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    size_t      src_pitch  = p->srcPtr.pitch  ? p->srcPtr.pitch  : p->extent.width;
    size_t      dst_pitch  = p->dstPtr.pitch  ? p->dstPtr.pitch  : p->extent.width;
    size_t      src_height = p->srcPtr.ysize  ? p->srcPtr.ysize  : p->extent.height;
    size_t      dst_height = p->dstPtr.ysize  ? p->dstPtr.ysize  : p->extent.height;
    const size_t src_span = p->extent.depth == 0 || p->extent.height == 0
                                ? 0
                                : (p->srcPos.z + p->extent.depth - 1) * src_pitch * src_height +
                                      (p->srcPos.y + p->extent.height - 1) * src_pitch +
                                      p->srcPos.x + p->extent.width;
    const size_t dst_span = p->extent.depth == 0 || p->extent.height == 0
                                ? 0
                                : (p->dstPos.z + p->extent.depth - 1) * dst_pitch * dst_height +
                                      (p->dstPos.y + p->extent.height - 1) * dst_pitch +
                                      p->dstPos.x + p->extent.width;
    const char* src_base = static_cast<const char*>(
        host_accessible_pointer(p->srcPtr.ptr, src_span));
    char* dst_base = static_cast<char*>(
        host_accessible_pointer(p->dstPtr.ptr, dst_span));
    if ((src_base == nullptr || dst_base == nullptr) &&
        p->extent.width > 0 && p->extent.height > 0 && p->extent.depth > 0) {
        return fail(cudaErrorInvalidValue);
    }

    for (size_t z = 0; z < p->extent.depth; ++z) {
        const size_t src_z_off = (p->srcPos.z + z) * src_pitch * src_height;
        const size_t dst_z_off = (p->dstPos.z + z) * dst_pitch * dst_height;
        for (size_t y = 0; y < p->extent.height; ++y) {
            const char* src_row = src_base + src_z_off
                                + (p->srcPos.y + y) * src_pitch + p->srcPos.x;
            char*       dst_row = dst_base + dst_z_off
                                + (p->dstPos.y + y) * dst_pitch + p->dstPos.x;
            std::memcpy(dst_row, src_row, p->extent.width);
        }
    }
    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy3DAsync(const cudaMemcpy3DParms* p, cudaStream_t stream) {
    if (p == nullptr)
        return fail(cudaErrorInvalidValue);
    if (p->srcArray != nullptr || p->dstArray != nullptr) {
        // The array storage is UMA and the synchronous copy already preserves
        // all data semantics. Completing eagerly is valid for an async API.
        return cudaMemcpy3D(p);
    }
    const cudaMemcpy3DParms params = *p;
    const size_t src_pitch = params.srcPtr.pitch ? params.srcPtr.pitch : params.extent.width;
    const size_t dst_pitch = params.dstPtr.pitch ? params.dstPtr.pitch : params.extent.width;
    const size_t src_height = params.srcPtr.ysize ? params.srcPtr.ysize : params.extent.height;
    const size_t dst_height = params.dstPtr.ysize ? params.dstPtr.ysize : params.extent.height;
    const size_t src_span = params.extent.depth == 0 || params.extent.height == 0
                                ? 0
                                : (params.srcPos.z + params.extent.depth - 1) *
                                          src_pitch * src_height +
                                      (params.srcPos.y + params.extent.height - 1) * src_pitch +
                                      params.srcPos.x + params.extent.width;
    const size_t dst_span = params.extent.depth == 0 || params.extent.height == 0
                                ? 0
                                : (params.dstPos.z + params.extent.depth - 1) *
                                          dst_pitch * dst_height +
                                      (params.dstPos.y + params.extent.height - 1) * dst_pitch +
                                      params.dstPos.x + params.extent.width;
    const char* src_base = static_cast<const char*>(
        host_accessible_pointer(params.srcPtr.ptr, src_span));
    char* dst_base = static_cast<char*>(
        host_accessible_pointer(params.dstPtr.ptr, dst_span));
    if ((src_base == nullptr || dst_base == nullptr) &&
        params.extent.width > 0 && params.extent.height > 0 && params.extent.depth > 0)
        return fail(cudaErrorInvalidValue);
    return fail(enqueue_stream_host_op(stream, [=]() {
        for (size_t z = 0; z < params.extent.depth; ++z) {
            const size_t src_z = (params.srcPos.z + z) * src_pitch * src_height;
            const size_t dst_z = (params.dstPos.z + z) * dst_pitch * dst_height;
            for (size_t y = 0; y < params.extent.height; ++y) {
                std::memcpy(dst_base + dst_z + (params.dstPos.y + y) * dst_pitch +
                                params.dstPos.x,
                            src_base + src_z + (params.srcPos.y + y) * src_pitch +
                                params.srcPos.x,
                            params.extent.width);
            }
        }
    }));
}

cudaError_t cudaMemcpy3DPeerAsync(const cudaMemcpy3DPeerParms* p, cudaStream_t stream) {
    if (p == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    cudaMemcpy3DParms local{};
    local.srcPos = p->srcPos;
    local.srcPtr = p->srcPtr;
    local.dstPos = p->dstPos;
    local.dstPtr = p->dstPtr;
    local.extent = p->extent;
    local.kind = cudaMemcpyDefault;
    (void)p->srcDevice;
    (void)p->dstDevice;
    return cudaMemcpy3DAsync(&local, stream);
}

cudaError_t cudaDeviceReset(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t sync_status = cumetal::metal_backend::synchronize(&error);
    if (sync_status != cudaSuccess) {
        return fail(sync_status);
    }

    RuntimeState& state = runtime_state();
    std::vector<std::pair<cudaStream_t, std::shared_ptr<cumetal::metal_backend::Stream>>> streams;
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        streams.reserve(state.streams.size());
        for (auto& [handle, record] : state.streams) {
            streams.emplace_back(handle, std::move(record.backend));
        }
        state.streams.clear();
    }

    for (auto& [handle, backend_stream] : streams) {
        if (backend_stream != nullptr) {
            std::string destroy_error;
            (void)cumetal::metal_backend::destroy_stream(backend_stream, &destroy_error);
        }
        delete handle;
    }

    if (tls_per_thread_stream != nullptr) {
        std::string destroy_error;
        (void)cumetal::metal_backend::destroy_stream(tls_per_thread_stream, &destroy_error);
        tls_per_thread_stream.reset();
    }

    reset_graph_allocation_state();
    state.allocations.clear();
    {
        std::lock_guard<std::mutex> lock(state.device_heap_mutex);
        state.device_heap.reset();
        state.device_heap_size = 8u * 1024u * 1024u;
        state.persisting_l2_limit = 0;
        state.printf_fifo_size = kDefaultPrintfFifoSize;
    }
    // Not clear(): cudaDeviceReset destroys the primary context, not the module
    // registry that __cudaRegisterFatBinary built when the image loaded. GROMACS
    // calls cudaDeviceReset at the end of device detection and then launches
    // kernels for the rest of the run.
    cumetal::native_registration::reset_device_state();
    cumetal::registration::reset_device_state();
    clear_pending_launch_state();
    tls_pending_launch_error = cudaSuccess;
    state.current_device = 0;
    state.device_flags = cudaDeviceScheduleAuto;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSynchronize(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::synchronize(&error);
    const cudaError_t pending_launch_error = take_pending_launch_error();
    if (status != cudaSuccess) {
        return fail(status);
    }
    return fail(pending_launch_error);
}

cudaError_t cudaStreamCreate(cudaStream_t* stream) {
    return cudaStreamCreateWithFlags(stream, cudaStreamDefault);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned int flags) {
    if (stream == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (flags != cudaStreamDefault && flags != cudaStreamNonBlocking) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    std::string error;
    const cudaError_t status = cumetal::metal_backend::create_stream(
        &backend_stream, &error, flags == cudaStreamDefault);
    if (status != cudaSuccess || backend_stream == nullptr) {
        return fail(status == cudaSuccess ? cudaErrorUnknown : status);
    }

    auto* handle = new (std::nothrow) CUstream_st{};
    if (handle == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    RuntimeState& state = runtime_state();
    {
        std::lock_guard<std::mutex> lock(state.stream_mutex);
        state.streams.emplace(
            handle, RuntimeState::StreamRecord{std::move(backend_stream), flags, {}});
    }

    *stream = handle;
    return fail(cudaSuccess);
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags) {
    if (flags == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    if (!resolve_stream_flags(stream, flags)) {
        return fail(cudaErrorInvalidValue);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaStreamSetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   const cudaStreamAttrValue* value) {
    if (value == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (attr != cudaStreamAttributeAccessPolicyWindow) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }
    const cudaAccessPolicyWindow& window = value->accessPolicyWindow;
    if (window.hitRatio < 0.0f || window.hitRatio > 1.0f ||
        (window.hitProp != cudaAccessPropertyNormal &&
         window.hitProp != cudaAccessPropertyStreaming &&
         window.hitProp != cudaAccessPropertyPersisting) ||
        (window.missProp != cudaAccessPropertyNormal &&
         window.missProp != cudaAccessPropertyStreaming &&
         window.missProp != cudaAccessPropertyPersisting)) {
        return fail(cudaErrorInvalidValue);
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess ||
        window.num_bytes > static_cast<std::size_t>(prop.accessPolicyMaxWindowSize)) {
        return fail(cudaErrorInvalidValue);
    }
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    if (stream == nullptr || stream == cudaStreamLegacy ||
        stream == cudaStreamPerThread) {
        state.default_stream_access_policy = *value;
        return fail(cudaSuccess);
    }
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) return fail(cudaErrorInvalidValue);
    found->second.access_policy = *value;
    return fail(cudaSuccess);
}

cudaError_t cudaStreamGetAttribute(cudaStream_t stream, cudaStreamAttrID attr,
                                   cudaStreamAttrValue* value) {
    if (value == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (attr != cudaStreamAttributeAccessPolicyWindow) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.stream_mutex);
    if (stream == nullptr || stream == cudaStreamLegacy ||
        stream == cudaStreamPerThread) {
        *value = state.default_stream_access_policy;
        return fail(cudaSuccess);
    }
    const auto found = state.streams.find(stream);
    if (found == state.streams.end()) return fail(cudaErrorInvalidValue);
    *value = found->second.access_policy;
    return fail(cudaSuccess);
}

cudaError_t cudaCtxResetPersistingL2Cache(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    // Accepted performance hint; Metal selects actual cache residency.
    return fail(cudaSuccess);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    if (!erase_stream_handle(stream, &backend_stream)) {
        return fail(cudaErrorInvalidValue);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::destroy_stream(backend_stream, &error);
    delete stream;
    return fail(status);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }

    std::string error;
    const cudaError_t status = cumetal::metal_backend::stream_synchronize(backend_stream, &error);
    return fail(status);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess || backend_stream == nullptr) {
        return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    bool complete = true;
    std::string error;
    const cudaError_t tail_status =
        cumetal::metal_backend::stream_tail_ticket(backend_stream, &tail_ticket, &error);
    if (tail_status != cudaSuccess) {
        return fail(tail_status);
    }
    const cudaError_t query_status =
        cumetal::metal_backend::stream_query_ticket(backend_stream, tail_ticket, &complete, &error);
    if (query_status != cudaSuccess) {
        return fail(query_status);
    }
    return fail(complete ? cudaSuccess : cudaErrorNotReady);
}

cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode) {
    switch (mode) {
        case cudaStreamCaptureModeGlobal:
        case cudaStreamCaptureModeThreadLocal:
        case cudaStreamCaptureModeRelaxed:
            break;
        default:
            return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto& cap = g_captures[stream];
    if (cap.capturing) {
        return fail(cudaErrorInvalidValue);
    }
    cap.capturing = true;
    cap.graph = new cudaGraph_st();
    return fail(cudaSuccess);
}

cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph) {
    if (pGraph == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_captures.find(stream);
    if (it == g_captures.end() || !it->second.capturing) {
        *pGraph = nullptr;
        return fail(cudaErrorInvalidValue);
    }
    *pGraph = it->second.graph;
    cudaGraph_t completed_graph = it->second.graph;
    for (auto capture = g_captures.begin(); capture != g_captures.end();) {
        if (capture->second.capturing && capture->second.graph == completed_graph) {
            capture = g_captures.erase(capture);
        } else {
            ++capture;
        }
    }
    return fail(cudaSuccess);
}

cudaError_t cudaStreamIsCapturing(cudaStream_t stream,
                                   cudaStreamCaptureStatus* pCaptureStatus) {
    if (pCaptureStatus == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_captures.find(stream);
    if (it != g_captures.end() && it->second.capturing) {
        *pCaptureStatus = cudaStreamCaptureStatusActive;
    } else {
        *pCaptureStatus = cudaStreamCaptureStatusNone;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int /*flags*/) {
    if (pGraph == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *pGraph = new cudaGraph_st();
    return fail(cudaSuccess);
}

cudaError_t cudaGraphClone(cudaGraph_t* pGraphClone, cudaGraph_t originalGraph) {
    if (pGraphClone == nullptr || originalGraph == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    {
        std::lock_guard<std::mutex> lock(originalGraph->lifetime->mutex);
        if (originalGraph->lifetime->contains_memory_nodes) {
            return fail(cudaErrorNotSupported);
        }
    }
    auto* clone = new (std::nothrow) cudaGraph_st();
    if (clone == nullptr) return fail(cudaErrorMemoryAllocation);

    std::unordered_map<const cudaGraphNode_st*, cudaGraphNode_st*> cloned_nodes;
    cloned_nodes.reserve(originalGraph->nodes.size());
    clone->nodes.reserve(originalGraph->nodes.size());
    for (const cudaGraphNode_st* source : originalGraph->nodes) {
        if (source == nullptr) {
            delete clone;
            return fail(cudaErrorInvalidValue);
        }
        auto* copied = new (std::nothrow) cudaGraphNode_st(*source);
        if (copied == nullptr) {
            delete clone;
            return fail(cudaErrorMemoryAllocation);
        }
        copied->dependencies.clear();
        cloned_nodes.emplace(source, copied);
        clone->nodes.push_back(copied);
    }
    for (std::size_t i = 0; i < originalGraph->nodes.size(); ++i) {
        for (const cudaGraphNode_st* dependency : originalGraph->nodes[i]->dependencies) {
            const auto it = cloned_nodes.find(dependency);
            if (it == cloned_nodes.end()) {
                delete clone;
                return fail(cudaErrorInvalidValue);
            }
            clone->nodes[i]->dependencies.push_back(it->second);
        }
    }
    *pGraphClone = clone;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph) {
    if (graph == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    delete graph;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph,
                                  cudaGraphNode_t* pErrorNode, char* /*pLogBuffer*/,
                                  size_t /*bufferSize*/) {
    if (pGraphExec == nullptr || graph == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (pErrorNode) { *pErrorNode = nullptr; }
    // Keep the availability check and instance-count reservation atomic. A
    // split check/increment lets two concurrent instantiations both observe
    // zero and violates CUDA's one-executable rule for memory-node graphs.
    std::unique_lock<std::mutex> lifetime_lock(graph->lifetime->mutex);
    if (graph->lifetime->contains_memory_nodes &&
        graph->lifetime->executable_instances != 0) {
        return fail(cudaErrorInvalidValue);
    }

    std::vector<cudaGraphNode_t> ordered;
    if (!topologically_order_graph(graph, &ordered)) {
        return fail(cudaErrorInvalidValue);
    }

    auto* exec = new (std::nothrow) cudaGraphExec_st();
    if (exec == nullptr) return fail(cudaErrorMemoryAllocation);
    exec->nodes.reserve(ordered.size());
    exec->source_node_index.reserve(ordered.size());
    exec->dependency_indices.reserve(ordered.size());
    std::unordered_map<const cudaGraphNode_st*, std::size_t> ordered_index;
    ordered_index.reserve(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) ordered_index.emplace(ordered[i], i);
    for (const auto* node : ordered) {
        std::vector<std::size_t> dependencies;
        dependencies.reserve(node->dependencies.size());
        for (const cudaGraphNode_st* dependency : node->dependencies) {
            const auto dependency_it = ordered_index.find(dependency);
            if (dependency_it == ordered_index.end()) {
                delete exec;
                return fail(cudaErrorInvalidValue);
            }
            dependencies.push_back(dependency_it->second);
        }
        std::sort(dependencies.begin(), dependencies.end());
        cudaGraphNode_st copy = *node;
        copy.dependencies.clear();
        exec->source_node_index.emplace(node, exec->nodes.size());
        exec->nodes.push_back(std::move(copy));
        exec->dependency_indices.push_back(std::move(dependencies));
    }
    exec->lifetime = graph->lifetime;
    if (exec->lifetime->contains_memory_nodes) {
        ++exec->lifetime->executable_instances;
    }
    lifetime_lock.unlock();
    *pGraphExec = exec;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec,
                                           cudaGraph_t graph,
                                           unsigned long long flags) {
    if ((flags & ~static_cast<unsigned long long>(
                     cudaGraphInstantiateFlagAutoFreeOnLaunch)) != 0) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t status =
        cudaGraphInstantiate(pGraphExec, graph, nullptr, nullptr, 0);
    if (status == cudaSuccess && pGraphExec != nullptr && *pGraphExec != nullptr) {
        (*pGraphExec)->auto_free_on_launch =
            (flags & cudaGraphInstantiateFlagAutoFreeOnLaunch) != 0;
    }
    return fail(status);
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
    if (graphExec == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    // Replay all recorded operations sequentially on the given stream.
    for (const auto& node : graphExec->nodes) {
        cudaError_t err = cudaSuccess;
        switch (node.type) {
            case cudaGraphNodeTypeKernel:
            {
                if (node.library_op) {
                    err = node.library_op(stream);
                    break;
                }
                std::vector<void*> argument_ptrs;
                argument_ptrs.reserve(node.kernel_arg_values.size());
                for (const auto& value : node.kernel_arg_values) {
                    argument_ptrs.push_back(
                        const_cast<std::uint8_t*>(value.data()));
                }
                err = cudaLaunchKernel(node.func, node.grid_dim, node.block_dim,
                                        argument_ptrs.empty() ? node.kernel_args : argument_ptrs.data(),
                                        node.shared_mem, stream);
                break;
            }
            case cudaGraphNodeTypeMemcpy:
                err = cudaMemcpyAsync(node.dst, node.src, node.count, node.memcpy_kind, stream);
                break;
            case cudaGraphNodeTypeMemset:
                err = cudaMemsetAsync(node.dst, node.memset_value, node.count, stream);
                break;
            case cudaGraphNodeTypeHost:
                if (node.host_fn) {
                    err = cudaStreamSynchronize(stream);
                    if (err == cudaSuccess) {
                        node.host_fn(node.host_user_data);
                    }
                }
                break;
            case cudaGraphNodeTypeMemAlloc:
                err = activate_graph_allocation(node.graph_allocation);
                break;
            case cudaGraphNodeTypeMemFree:
                err = deactivate_graph_allocation(node.graph_allocation);
                break;
            default:
                break;
        }
        if (err != cudaSuccess) {
            return fail(err);
        }
    }
    if (graphExec->auto_free_on_launch) {
        std::unordered_set<GraphAllocationState*> seen;
        for (const auto& node : graphExec->nodes) {
            if (node.type != cudaGraphNodeTypeMemAlloc ||
                node.graph_allocation == nullptr ||
                !seen.insert(node.graph_allocation.get()).second) {
                continue;
            }
            RuntimeState& state = runtime_state();
            bool active = false;
            {
                std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
                active = node.graph_allocation->active;
            }
            if (active) {
                const cudaError_t err =
                    deactivate_graph_allocation(node.graph_allocation);
                if (err != cudaSuccess) return fail(err);
            }
        }
    }
    return fail(cudaSuccess);
}

cudaError_t cudaGraphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                cudaGraphNode_t* hErrorNode_out,
                                cudaGraphExecUpdateResult* updateResult_out) {
    if (hGraphExec == nullptr || hGraph == nullptr || updateResult_out == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (hErrorNode_out != nullptr) *hErrorNode_out = nullptr;
    *updateResult_out = cudaGraphExecUpdateError;
    {
        std::lock_guard<std::mutex> lock(hGraph->lifetime->mutex);
        if (hGraph->lifetime->contains_memory_nodes) {
            *updateResult_out = cudaGraphExecUpdateErrorNotSupported;
            return fail(cudaSuccess);
        }
    }

    std::vector<cudaGraphNode_t> ordered;
    if (!topologically_order_graph(hGraph, &ordered)) {
        return fail(cudaErrorInvalidValue);
    }
    if (ordered.size() != hGraphExec->nodes.size()) {
        *updateResult_out = cudaGraphExecUpdateErrorTopologyChanged;
        return fail(cudaSuccess);
    }

    std::unordered_map<const cudaGraphNode_st*, std::size_t> ordered_index;
    ordered_index.reserve(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) ordered_index.emplace(ordered[i], i);
    std::vector<std::vector<std::size_t>> dependencies;
    dependencies.reserve(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        if (ordered[i]->type != hGraphExec->nodes[i].type) {
            if (hErrorNode_out != nullptr) *hErrorNode_out = ordered[i];
            *updateResult_out = cudaGraphExecUpdateErrorNodeTypeChanged;
            return fail(cudaSuccess);
        }
        std::vector<std::size_t> node_dependencies;
        node_dependencies.reserve(ordered[i]->dependencies.size());
        for (const cudaGraphNode_st* dependency : ordered[i]->dependencies) {
            const auto it = ordered_index.find(dependency);
            if (it == ordered_index.end()) return fail(cudaErrorInvalidValue);
            node_dependencies.push_back(it->second);
        }
        std::sort(node_dependencies.begin(), node_dependencies.end());
        dependencies.push_back(std::move(node_dependencies));
    }
    if (dependencies != hGraphExec->dependency_indices) {
        *updateResult_out = cudaGraphExecUpdateErrorTopologyChanged;
        return fail(cudaSuccess);
    }

    std::vector<cudaGraphNode_st> updated_nodes;
    std::unordered_map<const cudaGraphNode_st*, std::size_t> updated_index;
    updated_nodes.reserve(ordered.size());
    updated_index.reserve(ordered.size());
    for (const cudaGraphNode_st* source : ordered) {
        cudaGraphNode_st copy = *source;
        copy.dependencies.clear();
        updated_index.emplace(source, updated_nodes.size());
        updated_nodes.push_back(std::move(copy));
    }
    hGraphExec->nodes = std::move(updated_nodes);
    hGraphExec->source_node_index = std::move(updated_index);
    hGraphExec->dependency_indices = std::move(dependencies);
    *updateResult_out = cudaGraphExecUpdateSuccess;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec) {
    if (graphExec == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (graphExec->lifetime != nullptr) {
        std::lock_guard<std::mutex> lock(graphExec->lifetime->mutex);
        if (graphExec->lifetime->contains_memory_nodes &&
            graphExec->lifetime->executable_instances != 0) {
            --graphExec->lifetime->executable_instances;
        }
    }
    delete graphExec;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes) {
    if (graph == nullptr || numNodes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (nodes == nullptr) {
        *numNodes = graph->nodes.size();
    } else {
        const size_t n = std::min(*numNodes, graph->nodes.size());
        for (size_t i = 0; i < n; ++i) {
            nodes[i] = graph->nodes[i];
        }
        *numNodes = n;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaGraphGetRootNodes(cudaGraph_t graph, cudaGraphNode_t* pRootNodes,
                                   size_t* pNumRootNodes) {
    if (graph == nullptr || pNumRootNodes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    std::vector<cudaGraphNode_t> roots;
    for (cudaGraphNode_t node : graph->nodes) {
        if (node != nullptr && node->dependencies.empty()) roots.push_back(node);
    }
    if (pRootNodes == nullptr) {
        *pNumRootNodes = roots.size();
    } else {
        const std::size_t count = std::min(*pNumRootNodes, roots.size());
        for (std::size_t i = 0; i < count; ++i) pRootNodes[i] = roots[i];
        *pNumRootNodes = count;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaKernelNodeParams* pNodeParams) {
    if (!pGraphNode || !graph || !pNodeParams) return fail(cudaErrorInvalidValue);
    auto* node = new cudaGraphNode_st();
    node->type = cudaGraphNodeTypeKernel;
    node->func = pNodeParams->func;
    node->grid_dim = pNodeParams->gridDim;
    node->block_dim = pNodeParams->blockDim;
    node->shared_mem = pNodeParams->sharedMemBytes;
    if (!snapshot_graph_kernel_arguments(node, node->func, pNodeParams->kernelParams)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddMemcpyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaMemcpy3DParms* pCopyParams) {
    if (!pGraphNode || !graph || !pCopyParams) return fail(cudaErrorInvalidValue);
    auto* node = new cudaGraphNode_st();
    node->type = cudaGraphNodeTypeMemcpy;
    // Simplified: store src/dst/count from the 3D params for linear replay
    node->src = pCopyParams->srcPtr.ptr;
    node->dst = pCopyParams->dstPtr.ptr;
    node->count = pCopyParams->extent.width * pCopyParams->extent.height * pCopyParams->extent.depth;
    node->memcpy_kind = pCopyParams->kind;
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode,
                                      cudaGraph_t graph,
                                      const cudaGraphNode_t* pDependencies,
                                      size_t numDependencies,
                                      void* dst,
                                      const void* src,
                                      size_t count,
                                      cudaMemcpyKind kind) {
    if (pGraphNode == nullptr || graph == nullptr || dst == nullptr || src == nullptr ||
        count == 0 || validate_memcpy_kind(kind) != cudaSuccess) {
        return fail(cudaErrorInvalidValue);
    }
    auto* node = new (std::nothrow) cudaGraphNode_st();
    if (node == nullptr) return fail(cudaErrorMemoryAllocation);
    node->type = cudaGraphNodeTypeMemcpy;
    node->dst = dst;
    node->src = src;
    node->count = count;
    node->memcpy_kind = kind;
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                    const cudaMemsetParams* pMemsetParams) {
    if (!pGraphNode || !graph || !pMemsetParams) return fail(cudaErrorInvalidValue);
    auto* node = new cudaGraphNode_st();
    node->type = cudaGraphNodeTypeMemset;
    node->dst = pMemsetParams->dst;
    node->memset_value = static_cast<int>(pMemsetParams->value);
    node->count = pMemsetParams->width * pMemsetParams->height * pMemsetParams->elementSize;
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddHostNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph,
                                  const cudaGraphNode_t* pDependencies, size_t numDependencies,
                                  const cudaHostNodeParams* pNodeParams) {
    if (!pGraphNode || !graph || !pNodeParams || !pNodeParams->fn) {
        return fail(cudaErrorInvalidValue);
    }
    auto* node = new cudaGraphNode_st();
    node->type = cudaGraphNodeTypeHost;
    node->host_fn = pNodeParams->fn;
    node->host_user_data = pNodeParams->userData;
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddMemAllocNode(cudaGraphNode_t* pGraphNode,
                                     cudaGraph_t graph,
                                     const cudaGraphNode_t* pDependencies,
                                     size_t numDependencies,
                                     cudaMemAllocNodeParams* nodeParams) {
    if (pGraphNode == nullptr || graph == nullptr || nodeParams == nullptr ||
        nodeParams->bytesize == 0 ||
        (nodeParams->accessDescCount != 0 && nodeParams->accessDescs == nullptr) ||
        nodeParams->poolProps.allocType != cudaMemAllocationTypePinned ||
        nodeParams->poolProps.handleTypes != cudaMemHandleTypeNone ||
        nodeParams->poolProps.location.type != cudaMemLocationTypeDevice ||
        nodeParams->poolProps.location.id != 0) {
        return fail(cudaErrorInvalidValue);
    }
    for (size_t i = 0; i < nodeParams->accessDescCount; ++i) {
        const cudaMemAccessDesc& access = nodeParams->accessDescs[i];
        if (access.location.type != cudaMemLocationTypeDevice ||
            access.location.id != 0 ||
            (access.flags != cudaMemAccessFlagsProtRead &&
             access.flags != cudaMemAccessFlagsProtReadWrite)) {
            return fail(cudaErrorNotSupported);
        }
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) return fail(init_status);

    auto* node = new (std::nothrow) cudaGraphNode_st();
    if (node == nullptr) return fail(cudaErrorMemoryAllocation);
    node->type = cudaGraphNodeTypeMemAlloc;
    node->graph_pool_props = nodeParams->poolProps;
    if (nodeParams->accessDescCount != 0) {
        node->graph_access_descs.assign(nodeParams->accessDescs,
                                        nodeParams->accessDescs +
                                            nodeParams->accessDescCount);
    }
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> buffer;
    std::string error;
    const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(
        nodeParams->bytesize, &buffer, &error);
    if (alloc_status != cudaSuccess || buffer == nullptr ||
        buffer->contents() == nullptr) {
        delete node;
        return fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation
                                                : alloc_status);
    }
    const std::uintptr_t device_address = buffer->device_address();
    if (use_metal_device_addresses() && device_address == 0) {
        delete node;
        return fail(cudaErrorMemoryAllocation);
    }
    void* host_base = buffer->contents();
    void* device_base = reinterpret_cast<void*>(device_address);
    void* base = use_metal_device_addresses() ? device_base : host_base;
    void* alias = base == host_base ? device_base : host_base;

    std::shared_ptr<GraphAllocationState> allocation(
        new (std::nothrow) GraphAllocationState(),
        [](GraphAllocationState* value) {
            release_graph_allocation(value);
            delete value;
        });
    if (allocation == nullptr) {
        delete node;
        return fail(cudaErrorMemoryAllocation);
    }
    allocation->buffer = std::move(buffer);
    allocation->base = base;
    allocation->alias = alias;
    allocation->size = nodeParams->bytesize;
    node->graph_allocation = allocation;

    RuntimeState& state = runtime_state();
    bool duplicate_address = false;
    {
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        if (state.graph_allocations.contains(base)) {
            duplicate_address = true;
        } else {
            state.graph_allocations.emplace(base, allocation);
            state.graph_reserved_current += allocation->size;
            allocation->reserved_accounted = true;
            state.graph_reserved_high =
                std::max(state.graph_reserved_high, state.graph_reserved_current);
        }
    }
    if (duplicate_address) {
        delete node;
        return fail(cudaErrorMemoryAllocation);
    }
    {
        std::lock_guard<std::mutex> lock(graph->lifetime->mutex);
        graph->lifetime->contains_memory_nodes = true;
    }
    graph->nodes.push_back(node);
    nodeParams->dptr = base;
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphAddMemFreeNode(cudaGraphNode_t* pGraphNode,
                                    cudaGraph_t graph,
                                    const cudaGraphNode_t* pDependencies,
                                    size_t numDependencies,
                                    void* dptr) {
    if (pGraphNode == nullptr || graph == nullptr || dptr == nullptr ||
        (numDependencies != 0 && pDependencies == nullptr)) {
        return fail(cudaErrorInvalidValue);
    }
    std::shared_ptr<GraphAllocationState> allocation;
    cudaGraphNode_t allocation_node = nullptr;
    for (cudaGraphNode_t candidate : graph->nodes) {
        if (candidate != nullptr && candidate->type == cudaGraphNodeTypeMemAlloc &&
            candidate->graph_allocation != nullptr &&
            candidate->graph_allocation->base == dptr) {
            allocation = candidate->graph_allocation;
            allocation_node = candidate;
        }
        if (candidate != nullptr && candidate->type == cudaGraphNodeTypeMemFree &&
            candidate->graph_free_ptr == dptr) {
            return fail(cudaErrorInvalidValue);
        }
    }
    if (allocation == nullptr) {
        RuntimeState& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        const auto found = state.graph_allocations.find(dptr);
        if (found != state.graph_allocations.end()) {
            allocation = found->second.lock();
        }
    }
    if (allocation == nullptr) return fail(cudaErrorInvalidValue);

    std::unordered_set<cudaGraphNode_t> visited;
    std::function<bool(cudaGraphNode_t)> reaches_allocation =
        [&](cudaGraphNode_t node) -> bool {
        if (node == allocation_node) return true;
        if (node == nullptr || !visited.insert(node).second) return false;
        for (cudaGraphNode_t dependency : node->dependencies) {
            if (reaches_allocation(dependency)) return true;
        }
        return false;
    };
    bool ordered_after_allocation = allocation_node == nullptr;
    for (size_t i = 0; i < numDependencies; ++i) {
        if (!graph_contains_node(graph, pDependencies[i])) {
            return fail(cudaErrorInvalidValue);
        }
        visited.clear();
        if (reaches_allocation(pDependencies[i])) ordered_after_allocation = true;
    }
    if (!ordered_after_allocation) return fail(cudaErrorInvalidValue);

    auto* node = new (std::nothrow) cudaGraphNode_st();
    if (node == nullptr) return fail(cudaErrorMemoryAllocation);
    node->type = cudaGraphNodeTypeMemFree;
    node->graph_allocation = std::move(allocation);
    node->graph_free_ptr = dptr;
    if (!assign_graph_dependencies(graph, node, pDependencies, numDependencies)) {
        delete node;
        return fail(cudaErrorInvalidValue);
    }
    {
        RuntimeState& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        if (node->graph_allocation->free_placement != GraphFreePlacement::kNone) {
            delete node;
            return fail(cudaErrorInvalidValue);
        }
        node->graph_allocation->free_placement =
            allocation_node == nullptr ? GraphFreePlacement::kExternalGraph
                                       : GraphFreePlacement::kOwningGraph;
    }
    {
        std::lock_guard<std::mutex> lock(graph->lifetime->mutex);
        graph->lifetime->contains_memory_nodes = true;
    }
    graph->nodes.push_back(node);
    *pGraphNode = node;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphMemAllocNodeGetParams(cudaGraphNode_t node,
                                           cudaMemAllocNodeParams* params_out) {
    if (node == nullptr || params_out == nullptr ||
        node->type != cudaGraphNodeTypeMemAlloc ||
        node->graph_allocation == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *params_out = {};
    params_out->poolProps = node->graph_pool_props;
    params_out->accessDescs = node->graph_access_descs.empty()
                                  ? nullptr
                                  : node->graph_access_descs.data();
    params_out->accessDescCount = node->graph_access_descs.size();
    params_out->bytesize = node->graph_allocation->size;
    params_out->dptr = node->graph_allocation->base;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphMemFreeNodeGetParams(cudaGraphNode_t node,
                                          void** dptr_out) {
    if (node == nullptr || dptr_out == nullptr ||
        node->type != cudaGraphNodeTypeMemFree) {
        return fail(cudaErrorInvalidValue);
    }
    *dptr_out = node->graph_free_ptr;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetGraphMemAttribute(int device,
                                            cudaGraphMemAttributeType attr,
                                            void* value) {
    if (device != 0 || value == nullptr) return fail(cudaErrorInvalidValue);
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
    auto* out = static_cast<std::uint64_t*>(value);
    switch (attr) {
        case cudaGraphMemAttrUsedMemCurrent: *out = state.graph_used_current; break;
        case cudaGraphMemAttrUsedMemHigh: *out = state.graph_used_high; break;
        case cudaGraphMemAttrReservedMemCurrent:
            *out = state.graph_reserved_current;
            break;
        case cudaGraphMemAttrReservedMemHigh: *out = state.graph_reserved_high; break;
        default: return fail(cudaErrorInvalidValue);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSetGraphMemAttribute(int device,
                                            cudaGraphMemAttributeType attr,
                                            void* value) {
    if (device != 0 || value == nullptr) return fail(cudaErrorInvalidValue);
    const std::uint64_t requested = *static_cast<const std::uint64_t*>(value);
    if (requested != 0) return fail(cudaErrorInvalidValue);
    RuntimeState& state = runtime_state();
    std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
    switch (attr) {
        case cudaGraphMemAttrUsedMemHigh:
            state.graph_used_high = state.graph_used_current;
            break;
        case cudaGraphMemAttrReservedMemHigh:
            state.graph_reserved_high = state.graph_reserved_current;
            break;
        default:
            return fail(cudaErrorInvalidValue);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGraphMemTrim(int device) {
    if (device != 0) return fail(cudaErrorInvalidValue);
    // Graph-owned buffers retain fixed virtual addresses for the graph's
    // lifetime. CuMetal currently keeps no additional unused graph allocator
    // cache, so there is nothing eligible to return here.
    return fail(cudaSuccess);
}

cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec,
                                              cudaGraphNode_t hNode,
                                              const cudaKernelNodeParams* nodeParams) {
    if (hGraphExec == nullptr || hNode == nullptr || nodeParams == nullptr ||
        nodeParams->func == nullptr || nodeParams->gridDim.x == 0 ||
        nodeParams->gridDim.y == 0 || nodeParams->gridDim.z == 0 ||
        nodeParams->blockDim.x == 0 || nodeParams->blockDim.y == 0 ||
        nodeParams->blockDim.z == 0) {
        return fail(cudaErrorInvalidValue);
    }
    const auto it = hGraphExec->source_node_index.find(hNode);
    if (it == hGraphExec->source_node_index.end()) return fail(cudaErrorInvalidValue);
    cudaGraphNode_st& node = hGraphExec->nodes[it->second];
    if (node.type != cudaGraphNodeTypeKernel) return fail(cudaErrorInvalidValue);
    cudaGraphNode_st updated = node;
    updated.func = nodeParams->func;
    updated.grid_dim = nodeParams->gridDim;
    updated.block_dim = nodeParams->blockDim;
    updated.shared_mem = nodeParams->sharedMemBytes;
    if (!snapshot_graph_kernel_arguments(&updated, updated.func, nodeParams->kernelParams)) {
        return fail(cudaErrorInvalidValue);
    }
    node = std::move(updated);
    return fail(cudaSuccess);
}

cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, cudaGraphNodeType* pType) {
    if (!node || !pType) return fail(cudaErrorInvalidValue);
    *pType = node->type;
    return fail(cudaSuccess);
}

cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* pCaptureStatus,
                                      unsigned long long* pId) {
    if (!pCaptureStatus) return fail(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(g_capture_mutex);
    auto it = g_captures.find(stream);
    if (it != g_captures.end() && it->second.capturing) {
        *pCaptureStatus = cudaStreamCaptureStatusActive;
        if (pId) *pId = reinterpret_cast<unsigned long long>(it->second.graph);
    } else {
        *pCaptureStatus = cudaStreamCaptureStatusNone;
        if (pId) *pId = 0;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  cudaStreamCallback_t callback,
                                  void* user_data,
                                  unsigned int flags) {
    if (callback == nullptr || flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t tail_ticket = 0;
    std::string ticket_error;
    const cudaError_t ticket_status =
        cumetal::metal_backend::stream_tail_ticket(
            backend_stream, &tail_ticket, &ticket_error);
    if (ticket_status != cudaSuccess) {
        return fail(ticket_status);
    }

    std::thread([stream, callback, user_data, backend_stream, tail_ticket, legacy_stream]() mutable {
        std::string error;
        (void)legacy_stream;
        const cudaError_t callback_status =
            cumetal::metal_backend::stream_wait_ticket(backend_stream, tail_ticket, &error);
        callback(stream, callback_status, user_data);
    }).detach();

    return fail(cudaSuccess);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags) {
    // cudaEventWaitExternal only marks the wait as crossing a captured graph's
    // boundary; the wait itself is unchanged. GROMACS passes it on every
    // cross-stream dependency, so accepting it is required, not cosmetic.
    if (event == nullptr || (flags & ~static_cast<unsigned int>(cudaEventWaitExternal)) != 0) {
        return fail(cudaErrorInvalidValue);
    }

    cudaGraph_t event_capture_graph = nullptr;
    {
        std::lock_guard<std::mutex> lock(event->mutex);
        event_capture_graph = event->capture_graph;
    }
    if (event_capture_graph != nullptr) {
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        const bool capture_still_active = std::any_of(
            g_captures.begin(), g_captures.end(),
            [event_capture_graph](const auto& entry) {
                return entry.second.capturing &&
                       entry.second.graph == event_capture_graph;
            });
        if (!capture_still_active) {
            // A captured event only links streams while its originating graph
            // is actively being captured.  Once EndCapture has removed every
            // participant, waiting on the event must not resurrect that graph.
            event_capture_graph = nullptr;
        }
    }
    if (event_capture_graph != nullptr) {
        std::lock_guard<std::mutex> lock(g_capture_mutex);
        auto& capture = g_captures[stream];
        if (capture.capturing && capture.graph != event_capture_graph) {
            return fail(cudaErrorInvalidValue);
        }
        capture.capturing = true;
        capture.graph = event_capture_graph;
        return fail(cudaSuccess);
    }

    if (!is_legacy_stream_handle(stream)) {
        std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
        const cudaError_t resolve_status = resolve_runtime_stream(stream, &backend_stream, nullptr);
        if (resolve_status != cudaSuccess || backend_stream == nullptr) {
            return fail(resolve_status == cudaSuccess ? cudaErrorInvalidValue : resolve_status);
        }
    }

    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventCreate(cudaEvent_t* event) {
    return cudaEventCreateWithFlags(event, cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const unsigned int unsupported_flags = flags & ~(cudaEventDefault | cudaEventBlockingSync |
                                                      cudaEventDisableTiming);
    if (unsupported_flags != 0) {
        return fail(cudaErrorInvalidValue);
    }

    auto* created = new (std::nothrow) CUevent_st{};
    if (created == nullptr) {
        return fail(cudaErrorMemoryAllocation);
    }

    created->disable_timing = (flags & cudaEventDisableTiming) != 0;
    created->complete = true;
    created->recorded_once = false;
    created->timing_valid = false;
    created->ticket = 0;
    *event = created;
    return fail(cudaSuccess);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    delete event;
    return fail(cudaSuccess);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
    if (event == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    if (cudaGraph_t capture_graph = get_capture_graph(stream)) {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->capture_graph = capture_graph;
        event->recorded_once = true;
        event->complete = true;
        event->timing_valid = false;
        return fail(cudaSuccess);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }

    std::uint64_t marker_ticket = 0;
    (void)legacy_stream;
    std::string error;
    const cudaError_t marker_status =
        cumetal::metal_backend::stream_record_marker(backend_stream, &marker_ticket, &error);
    if (marker_status != cudaSuccess) {
        return fail(marker_status);
    }

    {
        std::lock_guard<std::mutex> lock(event->mutex);
        event->stream = std::move(backend_stream);
        event->ticket = marker_ticket;
        event->recorded_once = true;
        event->complete = false;
        event->timing_valid = !event->disable_timing;
        event->capture_graph = nullptr;
        if (!event->disable_timing) {
            event->timestamp = std::chrono::steady_clock::now();
        }
    }

    return fail(cudaSuccess);
}

// cudaEventRecordExternal / cudaEventRecordDefault only distinguish how an
// event participates in a stream capture. CuMetal records the marker the same
// way for both, so the flag is validated and dropped.
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream, unsigned int flags) {
    if ((flags & ~static_cast<unsigned int>(cudaEventRecordExternal)) != 0u) {
        return fail(cudaErrorInvalidValue);
    }
    return cudaEventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/true);
    return fail(status);
}

cudaError_t cudaEventQuery(cudaEvent_t event) {
    const cudaError_t status = update_event_completion(event, /*wait_for_completion=*/false);
    return fail(status);
}

cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
    if (ms == nullptr || start == nullptr || end == nullptr) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t start_status = update_event_completion(start, /*wait_for_completion=*/false);
    if (start_status != cudaSuccess) {
        return fail(start_status);
    }
    const cudaError_t end_status = update_event_completion(end, /*wait_for_completion=*/false);
    if (end_status != cudaSuccess) {
        return fail(end_status);
    }

    std::chrono::steady_clock::time_point start_timestamp;
    std::chrono::steady_clock::time_point end_timestamp;
    {
        std::lock_guard<std::mutex> lock(start->mutex);
        if (start->disable_timing || !start->recorded_once || !start->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        start_timestamp = start->timestamp;
    }
    {
        std::lock_guard<std::mutex> lock(end->mutex);
        if (end->disable_timing || !end->recorded_once || !end->timing_valid) {
            return fail(cudaErrorInvalidValue);
        }
        end_timestamp = end->timestamp;
    }

    *ms = std::chrono::duration<float, std::milli>(end_timestamp - start_timestamp).count();
    return fail(cudaSuccess);
}

cudaError_t cudaLaunchKernel(const void* func,
                             dim3 grid_dim,
                             dim3 block_dim,
                             void** args,
                             size_t shared_mem,
                             cudaStream_t stream) {
    auto launch_fail = [&](cudaError_t err, const char* why) -> cudaError_t {
        static int debug_launch = -1;
        if (debug_launch < 0) {
            const char* v = std::getenv("CUMETAL_DEBUG_LAUNCH");
            debug_launch = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        }
        if (debug_launch && err != cudaSuccess) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_LAUNCH: cudaLaunchKernel fail err=%d why=%s func=%p grid=(%u,%u,%u) block=(%u,%u,%u)\n",
                         static_cast<int>(err),
                         why != nullptr ? why : "?",
                         func,
                         grid_dim.x, grid_dim.y, grid_dim.z,
                         block_dim.x, block_dim.y, block_dim.z);
        }
        // Generated <<<...>>> host stubs discard cudaLaunchKernel's return
        // value. Preserve every launch failure, including validation and lazy
        // registration/JIT failures that return before backend submission, so
        // the next device synchronization cannot report a false success.
        record_pending_launch_error(err);
        return fail(err);
    };

    if (func == nullptr || grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
        return launch_fail(cudaErrorInvalidValue, "invalid launch dims/func");
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return launch_fail(init_status, "ensure_initialized");
    }

    // Graph capture: record kernel node instead of launching
    if (cudaGraph_t g = get_capture_graph(stream)) {
        auto* node = new cudaGraphNode_st();
        node->type = cudaGraphNodeTypeKernel;
        node->func = func;
        node->grid_dim = grid_dim;
        node->block_dim = block_dim;
        node->shared_mem = shared_mem;
        if (!snapshot_graph_kernel_arguments(node, func, args)) {
            delete node;
            return launch_fail(cudaErrorInvalidValue, "graph capture could not snapshot kernel arguments");
        }
        append_captured_graph_node(g, node);
        return fail(cudaSuccess);
    }

    cumetal::registration::RegisteredKernel registered_kernel;
    const bool use_registered_kernel =
        cumetal::native_registration::lookup_kernel(func, &registered_kernel) ||
        cumetal::registration::lookup_registered_kernel(func, &registered_kernel);

    if (trace_enabled()) {
        char buf[256];
        std::snprintf(buf, sizeof(buf), "launch name='%s' grid=(%u,%u,%u) block=(%u,%u,%u)",
                      use_registered_kernel ? registered_kernel.kernel_name.c_str() : "?",
                      grid_dim.x, grid_dim.y, grid_dim.z,
                      block_dim.x, block_dim.y, block_dim.z);
        trace_op("LAUNCH", buf);
    }

    cumetalKernel_t kernel_copy{};
    const cumetalKernel_t* kernel = nullptr;
    std::size_t inline_static_shared_bytes = 0;
    std::uint32_t arg_count = 0;
    const cumetalKernelArgInfo_t* arg_info = nullptr;

    if (use_registered_kernel) {
        if (registered_kernel.metallib_path.empty() || registered_kernel.kernel_name.empty()) {
            std::fprintf(stderr,
                         "CUMETAL: registered kernel missing metallib/name: func=%p kernel='%s' metallib='%s'\n",
                         func,
                         registered_kernel.kernel_name.c_str(),
                         registered_kernel.metallib_path.c_str());
            return launch_fail(cudaErrorInvalidValue, "registered kernel missing metallib/name");
        }

        // A kernel that takes no parameters may be launched with a null argv --
        // CUDA permits it, and GROMACS's device sanity check does exactly that
        // with `static __global__ void dummy_kernel() {}`. Only a kernel whose
        // PTX ABI says it wants arguments makes a null argv an error.
        if (args == nullptr) {
            if (!registered_kernel.arg_info.empty()) {
                std::fprintf(stderr,
                             "CUMETAL: kernel '%s' expects %zu argument(s) but was launched with a null argv\n",
                             registered_kernel.kernel_name.c_str(),
                             registered_kernel.arg_info.size());
                return launch_fail(cudaErrorInvalidValue, "registered kernel args null");
            }
            arg_count = 0;
        } else if (!registered_kernel.arg_info.empty()) {
            const std::uint32_t ptx_arg_count =
                static_cast<std::uint32_t>(registered_kernel.arg_info.size());
            // Clip to null-terminator: some callers pass nullptr as sentinel after real args
            std::uint32_t clipped = 0;
            for (; clipped < ptx_arg_count; ++clipped) {
                if (args[clipped] == nullptr) {
                    break;
                }
            }
            arg_count = clipped;
            arg_info = registered_kernel.arg_info.data();
        } else {
            // No PTX ABI info for this kernel, so the argument count has to be inferred from the
            // caller's null-terminated argv.
            //
            // This used to consult a hardcoded table of llm.c kernel names first and, on a match,
            // force that kernel's argument count. That is name-driven behavior on the real GPU
            // launch path: any kernel whose name merely *contained* e.g.
            // "layernorm_forward_kernel3" would have had 8 arguments forced regardless of its
            // actual signature, binding the wrong number of buffers. Same defect class as the
            // lowering templates in docs/known-gaps.md, and unlike the llm.c CPU emulation below
            // it was not gated behind an opt-in flag.
            //
            // Instrumenting this branch showed it is never reached anywhere in the test suite --
            // including the llm.c and llama.cpp conformance gates -- because ABI resolution now
            // populates arg_info. So the table protected nothing and only carried collision risk.
            // Inference from the actual arguments is at least driven by the call, not the name.
            cumetal::warn_once(
                "launch-arg-count-inferred",
                "a registered kernel had no PTX argument ABI; inferring the argument count from "
                "the caller's null-terminated argv. If this kernel misbehaves, the inference is "
                "the first thing to suspect");
            std::size_t inferred_count = 0;
            for (; inferred_count < 31; ++inferred_count) {
                if (args[inferred_count] == nullptr) {
                    break;
                }
            }
            if (inferred_count == 31) {
                return launch_fail(cudaErrorInvalidValue, "arg-count inference hit sentinel limit");
            }
            arg_count = static_cast<std::uint32_t>(inferred_count);
        }
    } else {
        std::memcpy(&kernel_copy, func, sizeof(kernel_copy));
        kernel = &kernel_copy;
        if (kernel->metallib_path == nullptr || kernel->kernel_name == nullptr || kernel->arg_count > 31) {
            return launch_fail(cudaErrorInvalidValue, "inline kernel descriptor invalid");
        }
        if (kernel->arg_count > 0 && args == nullptr) {
            return launch_fail(cudaErrorInvalidValue, "inline kernel args null");
        }
        arg_count = kernel->arg_count;
        arg_info = kernel->arg_info;
        if (!load_inline_static_shared_bytes(kernel->metallib_path,
                                             kernel->kernel_name,
                                             &inline_static_shared_bytes)) {
            return launch_fail(cudaErrorInvalidValue, "inline kernel ABI sidecar invalid");
        }
    }

    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    bool legacy_stream = false;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, &legacy_stream);
    if (resolve_status != cudaSuccess) {
        return launch_fail(resolve_status, "resolve_runtime_stream");
    }
    if (use_registered_kernel && llmc_emulation_enabled() &&
        !llmc_emulation_skips_kernel(registered_kernel.kernel_name)) {
        bool emulated = false;
        const cudaError_t emulation_status =
            try_emulate_llmc_registered_kernel(registered_kernel.kernel_name,
                                               arg_count,
                                               grid_dim,
                                               block_dim,
                                               args,
                                               legacy_stream,
                                               backend_stream,
                                               &emulated);
        if (emulation_status != cudaSuccess) {
            return launch_fail(emulation_status, "llmc emulation");
        }
        if (emulated) {
            note_llmc_emulation_hit(registered_kernel.kernel_name, arg_count);
            return launch_fail(cudaSuccess, "llmc emulated");
        }
    }

    RuntimeState& state = runtime_state();

    // ggml CUDA uses this ABI helper only to populate batched cuBLAS pointer arrays.
    // Generic PTX lowering cannot yet reliably materialize its pointer stores, so construct
    // the exact table here.  Table entries must use Metal GPU virtual addresses: cuBLAS later
    // resolves those identities back to their shared MTLBuffer allocations.
    if (use_registered_kernel &&
        kernel_name_contains(registered_kernel.kernel_name, "k_compute_batched_ptrs")) {
        if (cumetal::diag_env_truthy("CUMETAL_TRACE_GPU")) {
            std::fprintf(stderr,
                         "CUMETAL_PROVENANCE event=kernel_launch "
                         "kernel=\"%s\" source=runtime_helper provenance=runtime_helper "
                         "semantic_quality=exact device=cpu "
                         "compile_cache_hit=false launch_success=true duration_ns=-1 "
                         "grid=(%u,%u,%u) block=(%u,%u,%u) "
                         "unsupported_reason=\"exact GPU pointer-table construction\"\n",
                         registered_kernel.kernel_name.c_str(),
                         grid_dim.x,
                         grid_dim.y,
                         grid_dim.z,
                         block_dim.x,
                         block_dim.y,
                         block_dim.z);
        }
        if (args == nullptr || arg_count < 16) {
            return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs arg count");
        }

        const cudaError_t host_sync = synchronize_stream_for_host_op(stream, nullptr);
        if (host_sync != cudaSuccess) {
            return launch_fail(host_sync, "k_compute_batched_ptrs pre-sync");
        }

        auto read_ptr_arg = [&](std::uint32_t idx, void** out_ptr) -> bool {
            if (idx >= arg_count || args[idx] == nullptr || out_ptr == nullptr) {
                return false;
            }
            std::memcpy(out_ptr, args[idx], sizeof(void*));
            return true;
        };
        auto read_i64_arg = [&](std::uint32_t idx, std::int64_t* out) -> bool {
            if (idx >= arg_count || args[idx] == nullptr || out == nullptr) {
                return false;
            }
            std::memcpy(out, args[idx], sizeof(std::int64_t));
            return true;
        };
        auto read_size_arg = [&](std::uint32_t idx, std::size_t* out) -> bool {
            if (idx >= arg_count || args[idx] == nullptr || out == nullptr) {
                return false;
            }
            std::memcpy(out, args[idx], sizeof(std::size_t));
            return true;
        };

        void* src0_ptr = nullptr;
        void* src1_ptr = nullptr;
        void* dst_ptr = nullptr;
        void* ptrs_src_ptr = nullptr;
        void* ptrs_dst_ptr = nullptr;
        std::int64_t ne12 = 0, ne13 = 0, ne23 = 0;
        std::int64_t r2 = 0, r3 = 0;
        std::size_t nb02 = 0, nb03 = 0, nb12 = 0, nb13 = 0, nbd2 = 0, nbd3 = 0;

        const bool parsed =
            read_ptr_arg(0, &src0_ptr) &&
            read_ptr_arg(1, &src1_ptr) &&
            read_ptr_arg(2, &dst_ptr) &&
            read_ptr_arg(3, &ptrs_src_ptr) &&
            read_ptr_arg(4, &ptrs_dst_ptr) &&
            read_i64_arg(5, &ne12) &&
            read_i64_arg(6, &ne13) &&
            read_i64_arg(7, &ne23) &&
            read_size_arg(8, &nb02) &&
            read_size_arg(9, &nb03) &&
            read_size_arg(10, &nb12) &&
            read_size_arg(11, &nb13) &&
            read_size_arg(12, &nbd2) &&
            read_size_arg(13, &nbd3) &&
            read_i64_arg(14, &r2) &&
            read_i64_arg(15, &r3);
        if (!parsed || ne12 < 0 || ne13 < 0 || ne23 < 0 || r2 <= 0 || r3 <= 0) {
            return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs parse");
        }

        cumetal::rt::AllocationTable::ResolvedAllocation src0_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation src1_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation dst_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation ptrs_src_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation ptrs_dst_resolved;
        if (!state.allocations.resolve(src0_ptr, &src0_resolved) ||
            !state.allocations.resolve(src1_ptr, &src1_resolved) ||
            !state.allocations.resolve(dst_ptr, &dst_resolved) ||
            !state.allocations.resolve(ptrs_src_ptr, &ptrs_src_resolved) ||
            !state.allocations.resolve(ptrs_dst_ptr, &ptrs_dst_resolved)) {
            return launch_fail(cudaErrorInvalidDevicePointer, "k_compute_batched_ptrs resolve");
        }

        if (ne12 != 0 && ne13 > std::numeric_limits<std::int64_t>::max() / ne12) {
            return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs dimension overflow");
        }
        if (ne23 != ne12 * ne13) {
            return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs ne23 mismatch");
        }
        const auto table_count = static_cast<std::size_t>(ne23);
        if (table_count > std::numeric_limits<std::size_t>::max() / (2 * sizeof(void*)) ||
            ptrs_src_resolved.remaining_size < table_count * 2 * sizeof(void*) ||
            ptrs_dst_resolved.remaining_size < table_count * sizeof(void*)) {
            return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs table bounds");
        }

        auto device_base = [](const cumetal::rt::AllocationTable::ResolvedAllocation& allocation)
            -> std::uintptr_t {
            if (!allocation.buffer) {
                return 0;
            }
            std::uintptr_t base = allocation.buffer->device_address();
            if (base == 0 && allocation.buffer->contents() != nullptr) {
                base = reinterpret_cast<std::uintptr_t>(allocation.buffer->contents());
            }
            if (base == 0 || base > std::numeric_limits<std::uintptr_t>::max() - allocation.offset) {
                return 0;
            }
            return base + allocation.offset;
        };
        const std::uintptr_t src0_base = device_base(src0_resolved);
        const std::uintptr_t src1_base = device_base(src1_resolved);
        const std::uintptr_t dst_base = device_base(dst_resolved);
        auto* ptrs_src_base = static_cast<char*>(ptrs_src_resolved.buffer->contents());
        auto* ptrs_dst_base = static_cast<char*>(ptrs_dst_resolved.buffer->contents());
        if (src0_base == 0 || src1_base == 0 || dst_base == 0 || ptrs_src_base == nullptr ||
            ptrs_dst_base == nullptr) {
            return launch_fail(cudaErrorInvalidDevicePointer, "k_compute_batched_ptrs address");
        }
        ptrs_src_base += ptrs_src_resolved.offset;
        ptrs_dst_base += ptrs_dst_resolved.offset;

        auto checked_offset = [](std::int64_t i2,
                                 std::size_t stride2,
                                 std::int64_t i3,
                                 std::size_t stride3,
                                 std::size_t remaining,
                                 std::size_t* out) -> bool {
            const auto u2 = static_cast<std::size_t>(i2);
            const auto u3 = static_cast<std::size_t>(i3);
            if ((stride2 != 0 && u2 > std::numeric_limits<std::size_t>::max() / stride2) ||
                (stride3 != 0 && u3 > std::numeric_limits<std::size_t>::max() / stride3)) {
                return false;
            }
            const std::size_t offset2 = u2 * stride2;
            const std::size_t offset3 = u3 * stride3;
            if (offset2 > std::numeric_limits<std::size_t>::max() - offset3) {
                return false;
            }
            *out = offset2 + offset3;
            return *out < remaining;
        };
        auto store_pointer = [](char* table, std::size_t index, std::uintptr_t value) {
            std::memcpy(table + index * sizeof(void*), &value, sizeof(void*));
        };

        for (std::int64_t i13 = 0; i13 < ne13; ++i13) {
            for (std::int64_t i12 = 0; i12 < ne12; ++i12) {
                const std::int64_t i03 = i13 / r3;
                const std::int64_t i02 = i12 / r2;
                const std::size_t index = static_cast<std::size_t>(i12 + i13 * ne12);
                std::size_t src0_offset = 0, src1_offset = 0, dst_offset = 0;
                if (!checked_offset(i02, nb02, i03, nb03, src0_resolved.remaining_size,
                                    &src0_offset) ||
                    !checked_offset(i12, nb12, i13, nb13, src1_resolved.remaining_size,
                                    &src1_offset) ||
                    !checked_offset(i12, nbd2, i13, nbd3, dst_resolved.remaining_size,
                                    &dst_offset) ||
                    src0_base > std::numeric_limits<std::uintptr_t>::max() - src0_offset ||
                    src1_base > std::numeric_limits<std::uintptr_t>::max() - src1_offset ||
                    dst_base > std::numeric_limits<std::uintptr_t>::max() - dst_offset) {
                    return launch_fail(cudaErrorInvalidValue, "k_compute_batched_ptrs tensor bounds");
                }
                store_pointer(ptrs_src_base, index, src0_base + src0_offset);
                store_pointer(ptrs_src_base, table_count + index, src1_base + src1_offset);
                store_pointer(ptrs_dst_base, index, dst_base + dst_offset);
            }
        }

        return launch_fail(cudaSuccess, "k_compute_batched_ptrs runtime helper");
    }

    // Printf ring buffer size: cudaDeviceSetLimit controls the process value;
    // CUMETAL_PRINTF_BUFFER_SIZE remains an explicit per-process override.
    const std::uint32_t kPrintfCapWords = [&]() -> std::uint32_t {
        RuntimeState& state = runtime_state();
        std::size_t configured_bytes = 0;
        {
            std::lock_guard<std::mutex> lock(state.device_heap_mutex);
            configured_bytes = state.printf_fifo_size;
        }
        const char* env = std::getenv("CUMETAL_PRINTF_BUFFER_SIZE");
        if (env != nullptr && env[0] != '\0') {
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(env, &end, 10);
            if (errno == 0 && end != env && *end == '\0' && value > 0 &&
                value <= std::numeric_limits<std::size_t>::max()) {
                configured_bytes = static_cast<std::size_t>(value);
            }
        }
        const std::size_t words = configured_bytes / sizeof(std::uint32_t);
        return static_cast<std::uint32_t>(std::clamp<std::size_t>(
            words, 1u, std::numeric_limits<std::uint32_t>::max()));
    }();
    const bool needs_printf = use_registered_kernel && !registered_kernel.printf_formats.empty();
    const bool needs_device_heap =
        use_registered_kernel && registered_kernel.uses_device_heap;
    const bool needs_device_launch_queue =
        use_registered_kernel && registered_kernel.uses_device_launch_queue;
    constexpr std::uint32_t kDeviceLaunchQueueBytes = 1024u * 1024u;
    constexpr std::uint32_t kDeviceLaunchRecordAreaBytes = 64u * 1024u;
    constexpr std::uint32_t kDeviceLaunchRecordWords = 16u;
    constexpr std::uint32_t kDeviceLaunchMaxRecords = 1023u;

    std::vector<cumetal::metal_backend::KernelArg> launch_args;
    std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>> resident_buffers;
    std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>>
        registered_printf_string_buffers;
    launch_args.reserve(static_cast<std::size_t>(arg_count) +
                        registered_kernel.global_symbols.size() +
                        (registered_kernel.constant_symbols.empty() ? 0u : 1u) +
                        (needs_device_heap ? 1u : 0u) +
                        (needs_printf ? 2u : 0u));

    for (std::uint32_t i = 0; i < arg_count; ++i) {
        if (args == nullptr || args[i] == nullptr) {
            return launch_fail(cudaErrorInvalidValue, "arg slot pointer null");
        }

        cumetalKernelArgInfo_t info{
            .kind = CUMETAL_ARG_BUFFER,
            .size_bytes = static_cast<std::uint32_t>(sizeof(void*)),
        };
        if (arg_info != nullptr) {
            info = arg_info[i];
        } else if (use_registered_kernel) {
            std::uintptr_t value = 0;
            std::memcpy(&value, args[i], sizeof(value));
            cumetal::rt::AllocationTable::ResolvedAllocation resolved_ptr;
            if (!state.allocations.resolve(reinterpret_cast<void*>(value), &resolved_ptr)) {
                info.kind = CUMETAL_ARG_BYTES;
                info.size_bytes = value <= 0xFFFFFFFFull ? 4u : 8u;
            }
        }

        if (info.kind == CUMETAL_ARG_BUFFER) {
            void* device_ptr = *reinterpret_cast<void**>(args[i]);
            // A null pointer is a valid CUDA kernel argument value. Whether
            // dereferencing it is legal is kernel-dependent; binding it must
            // not be rejected by the launch API.
            if (device_ptr == nullptr) {
                cumetal::metal_backend::KernelArg arg;
                arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
                arg.buffer = nullptr;
                arg.offset = 0;
                launch_args.push_back(std::move(arg));
                continue;
            }

            cumetal::rt::AllocationTable::ResolvedAllocation resolved;
            if (!state.allocations.resolve(device_ptr, &resolved)) {
                if (use_registered_kernel) {
                    std::uintptr_t raw_value = 0;
                    std::memcpy(&raw_value, args[i], sizeof(raw_value));
                    // PTX param inference can over-classify some unannotated .u64 scalars
                    // as pointers. If the "pointer" is a tiny integer, preserve forward
                    // progress by treating it as an immediate 64-bit kernel argument.
                    if (raw_value != 0 && raw_value <= 0xFFFFFFFFull) {
                        cumetal::metal_backend::KernelArg scalar_arg;
                        scalar_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
                        scalar_arg.bytes.resize(sizeof(std::uint64_t));
                        std::memcpy(scalar_arg.bytes.data(), &raw_value, sizeof(raw_value));
                        launch_args.push_back(std::move(scalar_arg));
                        continue;
                    }
                }
                static int debug_launch = -1;
                if (debug_launch < 0) {
                    const char* v = std::getenv("CUMETAL_DEBUG_LAUNCH");
                    debug_launch = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
                }
                if (debug_launch) {
                    const char* which_name = use_registered_kernel ? registered_kernel.kernel_name.c_str() :
                                                              (kernel != nullptr ? kernel->kernel_name : "<null>");
                    std::fprintf(stderr,
                                 "CUMETAL_DEBUG_LAUNCH: buffer arg resolve failed kernel=%s arg=%u raw=%p arg_count=%u\n",
                                 which_name != nullptr ? which_name : "<null>",
                                 i,
                                 device_ptr,
                                 arg_count);
                }
                return launch_fail(cudaErrorInvalidDevicePointer, "buffer arg resolve");
            }

            collect_texture_resource_residency(
                reinterpret_cast<std::uintptr_t>(device_ptr), &resident_buffers);

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            arg.buffer = std::move(resolved.buffer);
            arg.offset = resolved.offset;
            launch_args.push_back(std::move(arg));
        } else {
            // CUDA's kernel parameter limit: 4 KB before 12.1, 32764 bytes
            // after. The Metal backend binds anything over setBytes' 4 KB cap
            // through a staged buffer instead.
            if (info.size_bytes == 0 || info.size_bytes > 32764) {
                return launch_fail(cudaErrorInvalidValue, "byte arg size invalid");
            }

            cumetal::metal_backend::KernelArg arg;
            arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            arg.bytes.resize(info.size_bytes);
            std::memcpy(arg.bytes.data(), args[i], info.size_bytes);
            relocate_embedded_device_pointers(&arg.bytes, &resident_buffers);
            launch_args.push_back(std::move(arg));
        }
    }

    // Writable module-scope __device__ variables are persistent shared Metal
    // buffers owned by the registration record. Symbol copies and every kernel
    // launch bind the same UMA storage, so GPU writes survive subsequent
    // launches and become visible after the existing symbol-copy synchronization.
    for (const auto& symbol : registered_kernel.global_symbols) {
        if (symbol.buffer == nullptr || symbol.size == 0) {
            return launch_fail(cudaErrorInvalidValue,
                               "referenced device global was not registered");
        }
        cumetal::metal_backend::KernelArg global_arg;
        global_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
        global_arg.buffer = symbol.buffer;
        global_arg.offset = 0;
        launch_args.push_back(std::move(global_arg));
        registered_printf_string_buffers.push_back(symbol.buffer);
    }

    // Clang emits module-scope __constant__ storage as external PTX symbols,
    // not declared kernel parameters. Per spec section 5.4.1, lay their CPU
    // shadows into one module constant buffer and bind it at reserved Metal
    // index 30. Writable __device__ globals intentionally do not use this path:
    // they require persistent GPU storage and copy-back semantics.
    if (!registered_kernel.constant_symbols.empty()) {
        if (registered_kernel.constant_buffer_size == 0 ||
            registered_kernel.constant_buffer_size > 64u * 1024u) {
            return launch_fail(cudaErrorInvalidValue,
                               "registered constant buffer size is invalid");
        }
        std::shared_ptr<cumetal::metal_backend::Buffer> constant_buffer;
        std::string alloc_error;
        const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(
            registered_kernel.constant_buffer_size, &constant_buffer, &alloc_error);
        if (alloc_status != cudaSuccess || constant_buffer == nullptr ||
            constant_buffer->contents() == nullptr) {
            return launch_fail(alloc_status == cudaSuccess ? cudaErrorMemoryAllocation
                                                           : alloc_status,
                               "constant symbol buffer allocation failed");
        }
        std::memset(constant_buffer->contents(), 0,
                    registered_kernel.constant_buffer_size);
        for (const auto& symbol : registered_kernel.constant_symbols) {
            if (symbol.address == nullptr || symbol.size == 0 ||
                symbol.offset > registered_kernel.constant_buffer_size ||
                symbol.size > registered_kernel.constant_buffer_size - symbol.offset) {
                return launch_fail(cudaErrorInvalidValue,
                                   "referenced constant symbol was not registered");
            }
            std::memcpy(static_cast<unsigned char*>(constant_buffer->contents()) +
                            symbol.offset,
                        symbol.address, symbol.size);
        }

        cumetal::metal_backend::KernelArg constant_arg;
        constant_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
        constant_arg.buffer = constant_buffer;
        constant_arg.offset = 0;
        constant_arg.binding_index = 30;
        launch_args.push_back(std::move(constant_arg));
        registered_printf_string_buffers.push_back(std::move(constant_buffer));
    }

    // Device malloc/free share one context-lifetime heap across every kernel.
    // The first 16 bytes hold bump, free-list head, capacity, and reserved state;
    // allocations start at offset 16 and remain valid across launches.
    if (needs_device_heap) {
        std::shared_ptr<cumetal::metal_backend::Buffer> heap;
        {
            std::lock_guard<std::mutex> lock(state.device_heap_mutex);
            if (state.device_heap == nullptr) {
                if (state.device_heap_size < 32u ||
                    state.device_heap_size > std::numeric_limits<std::uint32_t>::max()) {
                    return launch_fail(cudaErrorInvalidValue,
                                       "device heap size is outside 32-bit allocator range");
                }
                std::string alloc_error;
                const cudaError_t alloc_status =
                    cumetal::metal_backend::allocate_buffer(
                        state.device_heap_size, &state.device_heap, &alloc_error);
                if (alloc_status != cudaSuccess || state.device_heap == nullptr ||
                    state.device_heap->contents() == nullptr) {
                    state.device_heap.reset();
                    return launch_fail(alloc_status == cudaSuccess
                                           ? cudaErrorMemoryAllocation
                                           : alloc_status,
                                       "device heap allocation failed");
                }
                std::memset(state.device_heap->contents(), 0,
                            state.device_heap_size);
                auto* header = static_cast<std::uint32_t*>(
                    state.device_heap->contents());
                header[0] = 16u;
                header[1] = 0u;
                header[2] = static_cast<std::uint32_t>(state.device_heap_size);
            }
            heap = state.device_heap;
        }
        cumetal::metal_backend::KernelArg heap_arg;
        heap_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
        heap_arg.buffer = std::move(heap);
        heap_arg.offset = 0;
        launch_args.push_back(std::move(heap_arg));
    }

    // Append hidden printf ring-buffer args if the kernel uses device printf (spec §5.3).
    std::shared_ptr<cumetal::metal_backend::Buffer> printf_buffer;
    if (needs_printf) {
        const std::size_t kBufBytes =
            static_cast<std::size_t>(kPrintfCapWords) * sizeof(std::uint32_t);
        std::string alloc_error;
        const cudaError_t alloc_status =
            cumetal::metal_backend::allocate_buffer(kBufBytes, &printf_buffer, &alloc_error);
        if (alloc_status != cudaSuccess) {
            return launch_fail(alloc_status, "printf buffer alloc");
        }
        std::memset(printf_buffer->contents(), 0, kBufBytes);
        {
            cumetal::metal_backend::KernelArg buf_arg;
            buf_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            buf_arg.buffer = printf_buffer;
            buf_arg.offset = 0;
            launch_args.push_back(std::move(buf_arg));
        }
        {
            cumetal::metal_backend::KernelArg cap_arg;
            cap_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            cap_arg.bytes.resize(sizeof(std::uint32_t));
            const std::uint32_t cap = kPrintfCapWords;
            std::memcpy(cap_arg.bytes.data(), &cap, sizeof(cap));
            launch_args.push_back(std::move(cap_arg));
        }
    }

    std::shared_ptr<cumetal::metal_backend::Buffer> device_launch_queue;
    if (needs_device_launch_queue) {
        std::string alloc_error;
        const cudaError_t alloc_status = cumetal::metal_backend::allocate_buffer(
            kDeviceLaunchQueueBytes, &device_launch_queue, &alloc_error);
        if (alloc_status != cudaSuccess || device_launch_queue == nullptr ||
            device_launch_queue->contents() == nullptr) {
            return launch_fail(alloc_status != cudaSuccess ? alloc_status
                                                            : cudaErrorMemoryAllocation,
                               "device launch queue alloc");
        }
        std::memset(device_launch_queue->contents(), 0, kDeviceLaunchQueueBytes);
        auto* queue_words =
            static_cast<std::uint32_t*>(device_launch_queue->contents());
        queue_words[0] = kDeviceLaunchRecordAreaBytes;
        queue_words[1] = 0;
        {
            cumetal::metal_backend::KernelArg queue_arg;
            queue_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBuffer;
            queue_arg.buffer = device_launch_queue;
            queue_arg.offset = 0;
            launch_args.push_back(std::move(queue_arg));
        }
        {
            cumetal::metal_backend::KernelArg capacity_arg;
            capacity_arg.kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
            capacity_arg.bytes.resize(sizeof(std::uint32_t));
            std::memcpy(capacity_arg.bytes.data(), &kDeviceLaunchQueueBytes,
                        sizeof(kDeviceLaunchQueueBytes));
            launch_args.push_back(std::move(capacity_arg));
        }
    }

    // Use the user-specified dynamic shared memory size; fall back to the static
    // shared memory size computed from the PTX .shared declarations (for kernels
    // that use static __shared__ arrays without any dynamic shared memory).
    const std::size_t effective_shared_mem =
        (shared_mem > 0) ? shared_mem
        : (use_registered_kernel ? registered_kernel.static_shared_bytes
                                 : inline_static_shared_bytes);

    cumetal::metal_backend::LaunchConfig config{
        .grid = grid_dim,
        .block = block_dim,
        .shared_memory_bytes = effective_shared_mem,
        .provenance =
            use_registered_kernel ? registered_kernel.provenance : "precompiled_metallib",
        .semantic_quality =
            use_registered_kernel ? registered_kernel.semantic_quality : "exact",
        .resident_buffers = std::move(resident_buffers),
    };

    const char* metallib_path =
        use_registered_kernel ? registered_kernel.metallib_path.c_str() : kernel->metallib_path;
    const char* kernel_name =
        use_registered_kernel ? registered_kernel.kernel_name.c_str() : kernel->kernel_name;

    std::string error;
    const cudaError_t status =
        cumetal::metal_backend::launch_kernel(metallib_path, kernel_name, config, launch_args,
                                              backend_stream, &error);

    // Opt-in model-level oracle for GGML RMS norm. It validates the exact
    // buffers and packed broadcast metadata bound by llama.cpp, after the Metal
    // command completes, so synthetic probes cannot hide an ABI mismatch.
    if (status == cudaSuccess && use_registered_kernel &&
        cumetal::diag_env_truthy("CUMETAL_VALIDATE_GGML_RMS") &&
        kernel_name_contains(registered_kernel.kernel_name, "rms_norm_f32")) {
        std::string sync_error;
        const cudaError_t sync_status =
            cumetal::metal_backend::stream_synchronize(backend_stream, &sync_error);
        if (sync_status != cudaSuccess) {
            return launch_fail(sync_status, "GGML RMS validation sync");
        }

        auto scalar_u64 = [&](std::size_t index) -> std::uint64_t {
            std::uint64_t value = 0;
            if (index < launch_args.size() &&
                launch_args[index].kind ==
                    cumetal::metal_backend::KernelArg::Kind::kBytes) {
                const auto& bytes = launch_args[index].bytes;
                std::memcpy(&value, bytes.data(),
                            std::min(bytes.size(), sizeof(value)));
            }
            return value;
        };
        auto packed_divisor = [&](std::size_t index) -> std::uint32_t {
            std::uint32_t value = 0;
            if (index < launch_args.size() &&
                launch_args[index].kind ==
                    cumetal::metal_backend::KernelArg::Kind::kBytes &&
                launch_args[index].bytes.size() >= 12) {
                std::memcpy(&value, launch_args[index].bytes.data() + 8,
                            sizeof(value));
            }
            return value;
        };
        auto buffer_f32 = [&](std::size_t index) -> const float* {
            if (index >= launch_args.size()) return nullptr;
            const auto& arg = launch_args[index];
            if (arg.kind !=
                    cumetal::metal_backend::KernelArg::Kind::kBuffer ||
                arg.buffer == nullptr) {
                return nullptr;
            }
            return reinterpret_cast<const float*>(
                static_cast<const char*>(arg.buffer->contents()) + arg.offset);
        };

        const int ncols = static_cast<int>(scalar_u64(2));
        const std::int64_t stride_row =
            static_cast<std::int64_t>(scalar_u64(3));
        const std::int64_t stride_channel =
            static_cast<std::int64_t>(scalar_u64(4));
        const std::int64_t stride_sample =
            static_cast<std::int64_t>(scalar_u64(5));
        std::uint32_t eps_bits = static_cast<std::uint32_t>(scalar_u64(6));
        float eps = 0.0f;
        std::memcpy(&eps, &eps_bits, sizeof(eps));
        const float* x = buffer_f32(0);
        const float* dst = buffer_f32(1);
        const float* mul = buffer_f32(7);
        const float* add = buffer_f32(15);
        const auto& x_arg = launch_args[0];
        const auto& dst_arg = launch_args[1];
        const bool same_backing_buffer =
            x_arg.kind == cumetal::metal_backend::KernelArg::Kind::kBuffer &&
            dst_arg.kind == cumetal::metal_backend::KernelArg::Kind::kBuffer &&
            x_arg.buffer != nullptr && x_arg.buffer == dst_arg.buffer;
        const std::int64_t mul_sr =
            static_cast<std::int64_t>(scalar_u64(8));
        const std::int64_t mul_sc =
            static_cast<std::int64_t>(scalar_u64(9));
        const std::int64_t mul_ss =
            static_cast<std::int64_t>(scalar_u64(10));
        const std::int64_t add_sr =
            static_cast<std::int64_t>(scalar_u64(16));
        const std::int64_t add_sc =
            static_cast<std::int64_t>(scalar_u64(17));
        const std::int64_t add_ss =
            static_cast<std::int64_t>(scalar_u64(18));
        const std::uint32_t mul_nc = packed_divisor(11);
        const std::uint32_t mul_nr = packed_divisor(12);
        const std::uint32_t mul_nch = packed_divisor(13);
        const std::uint32_t mul_ns = packed_divisor(14);
        const std::uint32_t add_nc = packed_divisor(19);
        const std::uint32_t add_nr = packed_divisor(20);
        const std::uint32_t add_nch = packed_divisor(21);
        const std::uint32_t add_ns = packed_divisor(22);

        float max_abs = 0.0f;
        std::size_t max_index = 0;
        float max_got = 0.0f;
        float max_expected = 0.0f;
        if (x != nullptr && dst != nullptr && ncols > 0 && x != dst) {
            for (std::uint32_t sample = 0; sample < grid_dim.z; ++sample) {
                for (std::uint32_t channel = 0; channel < grid_dim.y;
                     ++channel) {
                    for (std::uint32_t row = 0; row < grid_dim.x; ++row) {
                        const float* row_x =
                            x + static_cast<std::int64_t>(sample) *
                                    stride_sample +
                            static_cast<std::int64_t>(channel) *
                                    stride_channel +
                            static_cast<std::int64_t>(row) * stride_row;
                        double sum = 0.0;
                        for (int col = 0; col < ncols; ++col) {
                            sum += static_cast<double>(row_x[col]) * row_x[col];
                        }
                        const float scale =
                            1.0f /
                            std::sqrt(static_cast<float>(sum / ncols) + eps);
                        const std::size_t dense_row =
                            ((static_cast<std::size_t>(sample) * grid_dim.y +
                              channel) *
                                 grid_dim.x +
                             row) *
                            static_cast<std::size_t>(ncols);
                        for (int col = 0; col < ncols; ++col) {
                            float expected = row_x[col] * scale;
                            if (mul != nullptr) {
                                const std::size_t mi =
                                    (mul_ns ? sample % mul_ns : 0) * mul_ss +
                                    (mul_nch ? channel % mul_nch : 0) * mul_sc +
                                    (mul_nr ? row % mul_nr : 0) * mul_sr +
                                    (mul_nc ? static_cast<std::uint32_t>(col) %
                                                  mul_nc
                                            : 0);
                                expected *= mul[mi];
                            }
                            if (add != nullptr) {
                                const std::size_t ai =
                                    (add_ns ? sample % add_ns : 0) * add_ss +
                                    (add_nch ? channel % add_nch : 0) * add_sc +
                                    (add_nr ? row % add_nr : 0) * add_sr +
                                    (add_nc ? static_cast<std::uint32_t>(col) %
                                                  add_nc
                                            : 0);
                                expected += add[ai];
                            }
                            const std::size_t index =
                                dense_row + static_cast<std::size_t>(col);
                            const float abs_error =
                                std::fabs(dst[index] - expected);
                            if (abs_error > max_abs) {
                                max_abs = abs_error;
                                max_index = index;
                                max_got = dst[index];
                                max_expected = expected;
                            }
                        }
                    }
                }
            }
        }
        std::fprintf(stderr,
                     "CUMETAL_VALIDATE_GGML_RMS kernel=\"%s\" shape=(%d,%u,%u,%u) "
                     "mul=%s add=%s backing=(%p@%zu,%p@%zu same=%s delta=%lld) "
                     "strides=(%lld,%lld,%lld) max_abs=%g index=%zu got=%g expected=%g\n",
                     registered_kernel.kernel_name.c_str(), ncols, grid_dim.x,
                     grid_dim.y, grid_dim.z, mul ? "yes" : "no",
                     add ? "yes" : "no", static_cast<void*>(x_arg.buffer.get()),
                     x_arg.offset, static_cast<void*>(dst_arg.buffer.get()),
                     dst_arg.offset, same_backing_buffer ? "yes" : "no",
                     static_cast<long long>(dst_arg.offset) -
                         static_cast<long long>(x_arg.offset),
                     static_cast<long long>(stride_row),
                     static_cast<long long>(stride_channel),
                     static_cast<long long>(stride_sample), max_abs, max_index,
                     max_got, max_expected);
    }

    // Targeted rms_norm probe (CUMETAL_TRACE): dump the bound scalar args and a
    // few input/output floats to verify arg binding + numerical correctness.
    static std::atomic<std::uint32_t> rms_probe_count{0};
    if (trace_enabled() && use_registered_kernel &&
        kernel_name_contains(registered_kernel.kernel_name, "rms_norm_f32") &&
        rms_probe_count.fetch_add(1) < 2) {
        std::fprintf(stderr, "CUMETAL_TRACE RMS argc=%u\n", (unsigned)launch_args.size());
        for (std::uint32_t i = 0; i < launch_args.size() && i < 23; ++i) {
            const auto& a = launch_args[i];
            const char* k = (a.kind == cumetal::metal_backend::KernelArg::Kind::kBuffer) ? "buf" : "byt";
            if (a.kind == cumetal::metal_backend::KernelArg::Kind::kBytes) {
                std::uint64_t v = 0;
                std::memcpy(&v, a.bytes.data(), std::min<std::size_t>(8u, a.bytes.size()));
                std::fprintf(stderr, "CUMETAL_TRACE RMSARG i=%u %s size=%u off=%zu val=%llu\n",
                             i, k, (unsigned)a.bytes.size(), (size_t)0, (unsigned long long)v);
            } else {
                std::fprintf(stderr, "CUMETAL_TRACE RMSARG i=%u %s off=%zu buf=%p\n",
                             i, k, (size_t)a.offset, (void*)a.buffer.get());
            }
        }
        auto dump_buf_rows = [&](const char* tag, std::uint32_t idx, long ncols, long stride) {
            if (idx >= launch_args.size()) return;
            const auto& a = launch_args[idx];
            if (a.kind != cumetal::metal_backend::KernelArg::Kind::kBuffer || a.buffer == nullptr) {
                std::fprintf(stderr, "CUMETAL_TRACE RMSBUF %s idx=%u (null)\n", tag, idx); return;
            }
            const float* p = reinterpret_cast<const float*>(
                static_cast<char*>(a.buffer->contents()) + a.offset);
            std::fprintf(stderr, "CUMETAL_TRACE RMSBUF %s idx=%u off=%zu r0=%g,%g,%g r1=%g,%g,%g\n",
                         tag, idx, (size_t)a.offset, p[0], p[1], p[2],
                         ncols > 0 && stride > 0 ? p[stride] : 0.f, 0.f, 0.f);
            (void)ncols;
        };
        std::uint64_t ncols_raw = 0, stride_raw = 0;
        if (launch_args.size() > 6 && launch_args[2].kind == cumetal::metal_backend::KernelArg::Kind::kBytes)
            std::memcpy(&ncols_raw, launch_args[2].bytes.data(), launch_args[2].bytes.size());
        if (launch_args.size() > 6 && launch_args[3].kind == cumetal::metal_backend::KernelArg::Kind::kBytes)
            std::memcpy(&stride_raw, launch_args[3].bytes.data(), launch_args[3].bytes.size());
        dump_buf_rows("x", 0, (long)ncols_raw, (long)stride_raw);
        dump_buf_rows("dst", 1, (long)ncols_raw, (long)stride_raw);
        dump_buf_rows("mul", 7, (long)ncols_raw, (long)stride_raw);
    }
    if (status != cudaSuccess) {
        static int debug_launch = -1;
        if (debug_launch < 0) {
            const char* v = std::getenv("CUMETAL_DEBUG_LAUNCH");
            debug_launch = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        }
        if (debug_launch) {
            const char* which_name = use_registered_kernel ? registered_kernel.kernel_name.c_str() :
                                                          (kernel != nullptr ? kernel->kernel_name : "<null>");
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_LAUNCH: launch_kernel failed err=%d kernel=%s args=%u shared=%zu msg=%s\n",
                         static_cast<int>(status),
                         which_name != nullptr ? which_name : "<null>",
                         arg_count,
                         shared_mem,
                         error.c_str());
        }
    }

    // Drain device printf output after kernel completes.
    if (needs_printf && printf_buffer != nullptr && status == cudaSuccess) {
        if (backend_stream != nullptr) {
            // Async stream: synchronize to ensure kernel output is visible.
            std::string sync_error;
            cumetal::metal_backend::stream_synchronize(backend_stream, &sync_error);
        }
        drain_printf_buffer(printf_buffer->contents(), kPrintfCapWords,
                            registered_kernel.printf_formats,
                            registered_printf_string_buffers);
    }

    // CUDA dynamic parallelism is represented by a device-written launch queue.
    // Synchronize the parent, then dispatch every recorded child through the
    // ordinary registered-kernel path. Child kernels receive fresh queues, so
    // recursive launches preserve parent-before-child completion semantics.
    if (needs_device_launch_queue && device_launch_queue != nullptr &&
        status == cudaSuccess) {
        if (backend_stream != nullptr) {
            std::string sync_error;
            const cudaError_t sync_status =
                cumetal::metal_backend::stream_synchronize(backend_stream, &sync_error);
            if (sync_status != cudaSuccess) {
                return launch_fail(sync_status, "device launch parent sync");
            }
        }

        const auto* queue_words = static_cast<const std::uint32_t*>(
            device_launch_queue->contents());
        const auto* queue_bytes = static_cast<const std::uint8_t*>(
            device_launch_queue->contents());
        const std::uint32_t recorded = queue_words[1];
        if (recorded > kDeviceLaunchMaxRecords) {
            return launch_fail(cudaErrorLaunchOutOfResources,
                               "device launch queue overflow");
        }
        for (std::uint32_t record_index = 0; record_index < recorded;
             ++record_index) {
            const std::uint32_t* record =
                queue_words + 4u + record_index * kDeviceLaunchRecordWords;
            const std::uint32_t record_kind = record[15];
            if (record_kind == 1u) {
                const std::uint64_t destination_bits =
                    static_cast<std::uint64_t>(record[0]) |
                    (static_cast<std::uint64_t>(record[1]) << 32u);
                const std::uint64_t source_bits =
                    static_cast<std::uint64_t>(record[2]) |
                    (static_cast<std::uint64_t>(record[3]) << 32u);
                const std::uint64_t count_bits =
                    static_cast<std::uint64_t>(record[4]) |
                    (static_cast<std::uint64_t>(record[5]) << 32u);
                if (destination_bits == 0 || source_bits == 0 ||
                    count_bits > std::numeric_limits<std::size_t>::max() ||
                    record[6] > static_cast<std::uint32_t>(cudaMemcpyDefault)) {
                    return launch_fail(cudaErrorInvalidValue,
                                       "invalid device memcpy record");
                }
                if (const char* debug = std::getenv("CUMETAL_DEBUG_LAUNCH");
                    debug != nullptr && debug[0] != '\0' && debug[0] != '0') {
                    cumetal::rt::AllocationTable::ResolvedAllocation debug_dst;
                    cumetal::rt::AllocationTable::ResolvedAllocation debug_src;
                    const bool dst_found = resolve_allocation_for_pointer(
                        reinterpret_cast<void*>(static_cast<std::uintptr_t>(destination_bits)),
                        &debug_dst);
                    const bool src_found = resolve_allocation_for_pointer(
                        reinterpret_cast<const void*>(static_cast<std::uintptr_t>(source_bits)),
                        &debug_src);
                    std::fprintf(stderr,
                                 "CUMETAL_DEBUG_LAUNCH: device memcpy record dst=0x%llx(%d off=%zu rem=%zu) src=0x%llx(%d off=%zu rem=%zu) count=%llu kind=%u\n",
                                 static_cast<unsigned long long>(destination_bits),
                                 dst_found ? 1 : 0, debug_dst.offset, debug_dst.remaining_size,
                                 static_cast<unsigned long long>(source_bits),
                                 src_found ? 1 : 0, debug_src.offset, debug_src.remaining_size,
                                 static_cast<unsigned long long>(count_bits), record[6]);
                }
                const cudaError_t copy_status = cudaMemcpyAsync(
                    reinterpret_cast<void*>(static_cast<std::uintptr_t>(destination_bits)),
                    reinterpret_cast<const void*>(static_cast<std::uintptr_t>(source_bits)),
                    static_cast<std::size_t>(count_bits),
                    static_cast<cudaMemcpyKind>(record[6]), stream);
                if (copy_status != cudaSuccess) {
                    return launch_fail(copy_status, "device memcpy enqueue");
                }
                continue;
            }
            if (record_kind != 0u) {
                return launch_fail(cudaErrorInvalidValue,
                                   "unknown device queue record kind");
            }
            const std::uint64_t token =
                static_cast<std::uint64_t>(record[0]) |
                (static_cast<std::uint64_t>(record[1]) << 32u);
            const std::uint32_t parameter_offset = record[2];
            const std::uint32_t parameter_size = record[3];
            const dim3 child_grid(record[4], record[5], record[6]);
            const dim3 child_block(record[7], record[8], record[9]);
            const std::size_t child_shared = record[10];

            if (parameter_offset < kDeviceLaunchRecordAreaBytes ||
                parameter_offset > kDeviceLaunchQueueBytes ||
                parameter_size > kDeviceLaunchQueueBytes - parameter_offset ||
                child_grid.x == 0 || child_grid.y == 0 || child_grid.z == 0 ||
                child_block.x == 0 || child_block.y == 0 || child_block.z == 0) {
                return launch_fail(cudaErrorInvalidConfiguration,
                                   "invalid device launch record");
            }

            const void* child_host_function = nullptr;
            cumetal::registration::RegisteredKernel child_kernel;
            if (!cumetal::registration::lookup_device_kernel_alias(
                    registered_kernel.module_handle, token,
                    &child_host_function, &child_kernel) ||
                child_host_function == nullptr) {
                return launch_fail(cudaErrorInvalidDeviceFunction,
                                   "unknown device launch kernel token");
            }

            std::vector<void*> child_args;
            child_args.reserve(child_kernel.arg_info.size());
            std::size_t cursor = 0;
            for (const auto& info : child_kernel.arg_info) {
                const std::size_t alignment =
                    std::max<std::size_t>(1, std::min<std::size_t>(8, info.size_bytes));
                cursor = (cursor + alignment - 1u) & ~(alignment - 1u);
                if (cursor > parameter_size ||
                    info.size_bytes > parameter_size - cursor) {
                    return launch_fail(cudaErrorInvalidValue,
                                       "device launch parameter buffer is too small");
                }
                child_args.push_back(const_cast<std::uint8_t*>(
                    queue_bytes + parameter_offset + cursor));
                cursor += info.size_bytes;
            }
            child_args.push_back(nullptr);

            if (const char* debug = std::getenv("CUMETAL_DEBUG_LAUNCH");
                debug != nullptr && debug[0] != '\0' && debug[0] != '0') {
                std::fprintf(stderr,
                             "CUMETAL_DEBUG_LAUNCH: device child record kernel=%s grid=(%u,%u,%u) block=(%u,%u,%u) args=",
                             child_kernel.kernel_name.c_str(), child_grid.x, child_grid.y,
                             child_grid.z, child_block.x, child_block.y, child_block.z);
                for (std::size_t i = 0; i < child_kernel.arg_info.size(); ++i) {
                    std::uint64_t value = 0;
                    const std::size_t bytes = std::min<std::size_t>(
                        sizeof(value), child_kernel.arg_info[i].size_bytes);
                    std::memcpy(&value, child_args[i], bytes);
                    std::fprintf(stderr, "%s%zu:0x%llx/%u/k%d", i == 0 ? "" : ",",
                                 i, static_cast<unsigned long long>(value),
                                 child_kernel.arg_info[i].size_bytes,
                                 static_cast<int>(child_kernel.arg_info[i].kind));
                    if (child_kernel.arg_info[i].size_bytes > sizeof(value) &&
                        child_kernel.arg_info[i].size_bytes % sizeof(std::uint32_t) == 0) {
                        std::fprintf(stderr, "[");
                        for (std::size_t word = 0;
                             word < child_kernel.arg_info[i].size_bytes /
                                        sizeof(std::uint32_t);
                             ++word) {
                            std::uint32_t word_value = 0;
                            std::memcpy(&word_value,
                                        static_cast<const std::uint8_t*>(child_args[i]) +
                                            word * sizeof(word_value),
                                        sizeof(word_value));
                            std::fprintf(stderr, "%s0x%x", word == 0 ? "" : ",",
                                         word_value);
                        }
                        std::fprintf(stderr, "]");
                    }
                }
                std::fprintf(stderr, "\n");
            }

            const cudaError_t child_status = cudaLaunchKernel(
                child_host_function, child_grid, child_block,
                child_args.data(), child_shared, stream);
            if (child_status != cudaSuccess) {
                return launch_fail(child_status, "device child launch");
            }
        }
    }

    // Metal-backend commands now encode per-buffer MTLSharedEvent dependencies,
    // including when several CUDA pointers alias suballocations of one buffer.
    // Registered launches can therefore remain asynchronous without losing
    // ordering against MPS/cuBLAS work on another command queue. Keep explicit
    // synchronization switches for diagnosis and performance comparisons.
    if (status == cudaSuccess) {
        static int sync_each_launch = -1;
        static int sync_registered_launch = -1;
        if (sync_each_launch < 0) {
            const char* v = std::getenv("CUMETAL_SYNC_EACH_LAUNCH");
            sync_each_launch = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        }
        if (sync_registered_launch < 0) {
            const char* v =
                std::getenv("CUMETAL_SYNC_REGISTERED_LAUNCH");
            sync_registered_launch =
                (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
        }
        const bool synchronize_launch =
            sync_each_launch ||
            (use_registered_kernel && sync_registered_launch);
        if (synchronize_launch) {
            std::string sync_error;
            const cudaError_t sync_status =
                (backend_stream != nullptr)
                    ? cumetal::metal_backend::stream_synchronize(backend_stream, &sync_error)
                    : cumetal::metal_backend::synchronize(&sync_error);
            if (sync_status != cudaSuccess) {
                return launch_fail(sync_status, "post-launch debug sync");
            }
        }
    }

    return launch_fail(status, "metal_backend::launch_kernel");
}

cudaError_t cudaConfigureCall(dim3 grid_dim,
                              dim3 block_dim,
                              size_t shared_mem,
                              cudaStream_t stream) {
    if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 ||
        block_dim.y == 0 || block_dim.z == 0) {
        return fail(cudaErrorInvalidValue);
    }

    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }

    std::shared_ptr<cumetal::metal_backend::Stream> resolved_stream;
    bool legacy_stream = false;
    const cudaError_t stream_status =
        resolve_runtime_stream(stream, &resolved_stream, &legacy_stream);
    if (stream_status != cudaSuccess) {
        return fail(stream_status);
    }

    clear_pending_launch_state();
    tls_pending_launch.configured = true;
    tls_pending_launch.grid_dim = grid_dim;
    tls_pending_launch.block_dim = block_dim;
    tls_pending_launch.shared_mem = shared_mem;
    tls_pending_launch.stream = stream;
    return fail(cudaSuccess);
}

cudaError_t cudaSetupArgument(const void* arg, size_t size, size_t offset) {
    if (!tls_pending_launch.configured || arg == nullptr || size == 0) {
        return fail(cudaErrorInvalidValue);
    }

    if (offset > (std::numeric_limits<size_t>::max() - size)) {
        return fail(cudaErrorInvalidValue);
    }

    const size_t end = offset + size;
    if (end > tls_pending_launch.storage.size()) {
        tls_pending_launch.storage.resize(end, 0);
    }
    std::memcpy(tls_pending_launch.storage.data() + offset, arg, size);

    bool found = false;
    for (PendingLaunchArgument& pending_arg : tls_pending_launch.arguments) {
        if (pending_arg.offset != offset) {
            continue;
        }
        pending_arg.size = size;
        found = true;
        break;
    }
    if (!found) {
        tls_pending_launch.arguments.push_back(PendingLaunchArgument{
            .offset = offset,
            .size = size,
        });
    }

    return fail(cudaSuccess);
}

cudaError_t cudaLaunch(const void* func) {
    if (func == nullptr || !tls_pending_launch.configured) {
        return fail(cudaErrorInvalidValue);
    }

    std::vector<PendingLaunchArgument> ordered_args = tls_pending_launch.arguments;
    std::sort(ordered_args.begin(),
              ordered_args.end(),
              [](const PendingLaunchArgument& lhs, const PendingLaunchArgument& rhs) {
                  return lhs.offset < rhs.offset;
              });

    std::vector<void*> launch_args;
    launch_args.reserve(ordered_args.size() + 1);

    size_t previous_end = 0;
    for (const PendingLaunchArgument& pending_arg : ordered_args) {
        if (pending_arg.size == 0 ||
            pending_arg.offset > (std::numeric_limits<size_t>::max() - pending_arg.size)) {
            clear_pending_launch_state();
            return fail(cudaErrorInvalidValue);
        }

        const size_t end = pending_arg.offset + pending_arg.size;
        if (end > tls_pending_launch.storage.size() || pending_arg.offset < previous_end) {
            clear_pending_launch_state();
            return fail(cudaErrorInvalidValue);
        }
        previous_end = end;

        launch_args.push_back(reinterpret_cast<void*>(tls_pending_launch.storage.data() +
                                                      pending_arg.offset));
    }
    launch_args.push_back(nullptr);

    const dim3 grid_dim = tls_pending_launch.grid_dim;
    const dim3 block_dim = tls_pending_launch.block_dim;
    const size_t shared_mem = tls_pending_launch.shared_mem;
    const cudaStream_t stream = tls_pending_launch.stream;

    const cudaError_t status =
        cudaLaunchKernel(func, grid_dim, block_dim, launch_args.data(), shared_mem, stream);
    clear_pending_launch_state();
    return status;
}

cudaError_t cudaGetLastError(void) {
    const cudaError_t value = tls_last_error;
    tls_last_error = cudaSuccess;
    return value;
}

cudaError_t cudaPeekAtLastError(void) {
    return tls_last_error;
}

const char* cudaGetErrorName(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";
        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorNotSupported:
            return "cudaErrorNotSupported";
        case cudaErrorCudartUnloading:
            return "cudaErrorCudartUnloading";
        case cudaErrorInvalidDeviceFunction:
            return "cudaErrorInvalidDeviceFunction";
        case cudaErrorInvalidConfiguration:
            return "cudaErrorInvalidConfiguration";
        case cudaErrorInvalidDevice:
            return "cudaErrorInvalidDevice";
        case cudaErrorInvalidMemcpyDirection:
            return "cudaErrorInvalidMemcpyDirection";
        case cudaErrorInsufficientDriver:
            return "cudaErrorInsufficientDriver";
        case cudaErrorNoDevice:
            return "cudaErrorNoDevice";
        case cudaErrorInvalidResourceHandle:
            return "cudaErrorInvalidResourceHandle";
        case cudaErrorLaunchOutOfResources:
            return "cudaErrorLaunchOutOfResources";
        case cudaErrorAssert:
            return "cudaErrorAssert";
        case cudaErrorLaunchFailure:
            return "cudaErrorLaunchFailure";
        case cudaErrorCooperativeLaunchTooLarge:
            return "cudaErrorCooperativeLaunchTooLarge";
        case cudaErrorNotPermitted:
            return "cudaErrorNotPermitted";
        case cudaErrorGraphExecUpdateFailure:
            return "cudaErrorGraphExecUpdateFailure";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

const char* cudaGetErrorString(cudaError_t error) {
    switch (error) {
        case cudaSuccess:
            return "cudaSuccess";
        case cudaErrorInvalidValue:
            return "cudaErrorInvalidValue";
        case cudaErrorMemoryAllocation:
            return "cudaErrorMemoryAllocation";
        case cudaErrorInitializationError:
            return "cudaErrorInitializationError";
        case cudaErrorLaunchTimeout:
            return "cudaErrorLaunchTimeout";
        case cudaErrorInvalidDevicePointer:
            return "cudaErrorInvalidDevicePointer";
        case cudaErrorNotReady:
            return "cudaErrorNotReady";
        case cudaErrorDevicesUnavailable:
            return "cudaErrorDevicesUnavailable";
        case cudaErrorPeerAccessAlreadyEnabled:
            return "cudaErrorPeerAccessAlreadyEnabled";
        case cudaErrorPeerAccessNotEnabled:
            return "cudaErrorPeerAccessNotEnabled";
        case cudaErrorIllegalAddress:
            return "cudaErrorIllegalAddress";
        case cudaErrorNotSupported:
            return "operation not supported";
        case cudaErrorCudartUnloading:
            return "driver shutting down";
        case cudaErrorInvalidDeviceFunction:
            return "invalid device function";
        case cudaErrorInvalidConfiguration:
            return "invalid configuration argument";
        case cudaErrorInvalidDevice:
            return "invalid device ordinal";
        case cudaErrorInvalidMemcpyDirection:
            return "invalid copy direction for memcpy";
        case cudaErrorInsufficientDriver:
            return "CUDA driver version is insufficient for CUDA runtime version";
        case cudaErrorNoDevice:
            return "no CUDA-capable device is detected";
        case cudaErrorInvalidResourceHandle:
            return "invalid resource handle";
        case cudaErrorLaunchOutOfResources:
            return "too many resources requested for launch";
        case cudaErrorCooperativeLaunchTooLarge:
            return "too many blocks in cooperative launch";
        case cudaErrorAssert:
            return "device-side assert triggered";
        case cudaErrorLaunchFailure:
            return "unspecified launch failure";
        case cudaErrorNotPermitted:
            return "operation not permitted";
        case cudaErrorGraphExecUpdateFailure:
            return "graph executable update failure";
        case cudaErrorUnknown:
            return "cudaErrorUnknown";
    }
    return "cudaErrorUnknown";
}

cudaError_t cudaProfilerStart(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

cudaError_t cudaProfilerStop(void) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    return fail(cudaSuccess);
}

// Occupancy API — kernel-specific Metal-backed estimate.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                          const void* func,
                                                          int blockSize,
                                                          size_t dynamicSMemSize) {
    if (numBlocks == nullptr || blockSize <= 0) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const cudaError_t query = query_runtime_kernel_properties(func, &kernel);
    if (query != cudaSuccess) return fail(query);
    if (blockSize > kernel.max_threads_per_threadgroup)
        return fail(cudaErrorInvalidValue);
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
        return fail(cudaErrorInvalidValue);
    const size_t total_shared =
        kernel.static_threadgroup_memory_bytes + dynamicSMemSize;
    if (total_shared > static_cast<size_t>(device.sharedMemPerBlock))
        return fail(cudaErrorInvalidValue);
    const int thread_bound =
        std::max(1, kernel.max_threads_per_threadgroup / blockSize);
    const int memory_bound =
        total_shared == 0
            ? thread_bound
            : std::max(1, static_cast<int>(device.sharedMemPerBlock / total_shared));
    // A conservative one-block residency guarantee makes the grid-wide
    // software barrier safe: every block selected by CUDA's standard
    // multiprocessorCount * occupancy launch formula can make progress.
    *numBlocks = std::min(1, std::min(thread_bound, memory_bound));
    return fail(cudaSuccess);
}

cudaError_t cudaOccupancyMaxPotentialBlockSize(int* minGridSize,
                                               int* blockSize,
                                               const void* func,
                                               size_t dynamicSMemSize,
                                               int blockSizeLimit) {
    if (minGridSize == nullptr || blockSize == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const cudaError_t query = query_runtime_kernel_properties(func, &kernel);
    if (query != cudaSuccess) return fail(query);
    int chosen_block = kernel.max_threads_per_threadgroup;
    if (blockSizeLimit > 0)
        chosen_block = std::min(chosen_block, blockSizeLimit);
    const int width = std::max(1, kernel.thread_execution_width);
    chosen_block = std::max(width, (chosen_block / width) * width);
    int active_blocks = 0;
    const cudaError_t occupancy = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, func, chosen_block, dynamicSMemSize);
    if (occupancy != cudaSuccess) return fail(occupancy);
    *blockSize = chosen_block;
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess && prop.multiProcessorCount > 0) {
        *minGridSize = prop.multiProcessorCount * active_blocks;
    } else {
        *minGridSize = 16;  // safe fallback
    }
    return fail(cudaSuccess);
}

cudaError_t cudaFuncGetAttributes(cudaFuncAttributes* attr, const void* func) {
    if (attr == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    cumetal::metal_backend::KernelProperties kernel{};
    const cudaError_t query = query_runtime_kernel_properties(func, &kernel);
    if (query != cudaSuccess) return fail(query);
    cudaDeviceProp device{};
    if (cudaGetDeviceProperties(&device, 0) != cudaSuccess)
        return fail(cudaErrorInvalidValue);
    *attr = {};
    attr->maxThreadsPerBlock = kernel.max_threads_per_threadgroup;
    attr->sharedSizeBytes = kernel.static_threadgroup_memory_bytes;
    attr->maxDynamicSharedSizeBytes =
        kernel.static_threadgroup_memory_bytes >=
                static_cast<size_t>(device.sharedMemPerBlock)
            ? 0
            : static_cast<int>(
                  static_cast<size_t>(device.sharedMemPerBlock) -
                  kernel.static_threadgroup_memory_bytes);
    // Metal pipelines have no NVIDIA PTX or SASS target version.
    attr->ptxVersion = 0;
    attr->binaryVersion = 0;
    return fail(cudaSuccess);
}

// No-ops — Metal has no L1/shared-memory configuration knobs.
cudaError_t cudaFuncSetCacheConfig(const void* /*func*/, cudaFuncCache /*cacheConfig*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaFuncSetSharedMemConfig(const void* /*func*/, cudaSharedMemConfig /*config*/) {
    return fail(cudaSuccess);
}

// No-op: Metal has no per-function attribute knobs corresponding to CUDA's.
// cudaFuncAttributeMaxDynamicSharedMemorySize is validated at launch time instead.
cudaError_t cudaFuncSetAttribute(const void* /*func*/, cudaFuncAttribute attr, int /*value*/) {
    if (attr != cudaFuncAttributeMaxDynamicSharedMemorySize &&
        attr != cudaFuncAttributePreferredSharedMemoryCarveout) {
        return fail(cudaErrorInvalidValue);
    }
    return fail(cudaSuccess);
}

// Device-level L1/shared-memory config — no-ops on Metal.
cudaError_t cudaDeviceSetCacheConfig(cudaFuncCache /*cacheConfig*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetCacheConfig(cudaFuncCache* pCacheConfig) {
    if (pCacheConfig == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *pCacheConfig = cudaFuncCachePreferNone;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSetSharedMemConfig(cudaSharedMemConfig /*config*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetSharedMemConfig(cudaSharedMemConfig* pConfig) {
    if (pConfig == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *pConfig = cudaSharedMemBankSizeFourByte;  // default on CUDA
    return fail(cudaSuccess);
}

// Symbol address query: CuMetal registers __device__ variables as host-accessible
// pointers (UMA). The symbol pointer is the device address directly.
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol) {
    if (devPtr == nullptr || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    // On UMA the symbol's host address IS the device address, but a registered
    // symbol may still carry a remapping. This used to return the raw pointer
    // unconditionally while cudaMemcpyToSymbol resolved through the registration
    // table -- so the two disagreed about where a symbol lives, and the address
    // handed to the caller was not the memory the runtime read and wrote.
    const unsigned char* resolved = nullptr;
    const cudaError_t status = checked_symbol_ptr(symbol, 0, 0, &resolved);
    if (status != cudaSuccess) {
        return fail(status);
    }
    *devPtr = const_cast<void*>(static_cast<const void*>(resolved));
    return fail(cudaSuccess);
}

// Symbol sizes come from __cudaRegisterVar and are stable for the life of the
// registration module.
cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol) {
    if (size == nullptr || symbol == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    const void* resolved = nullptr;
    std::size_t resolved_size = 0;
    if ((!cumetal::native_registration::lookup_symbol(
             symbol, &resolved, &resolved_size) &&
         !cumetal::registration::lookup_registered_symbol(
             symbol, &resolved, &resolved_size)) || resolved == nullptr ||
        resolved_size == 0) {
        return fail(cudaErrorInvalidValue);
    }
    *size = resolved_size;
    return fail(cudaSuccess);
}

// Unified Memory advisory APIs — no-ops on Apple Silicon UMA.
// Prefetch hints, locality advice, and access pattern hints have no effect when
// CPU and GPU share the same physical memory (UMA).
cudaError_t cudaMemPrefetchAsync(const void* /*devPtr*/,
                                  size_t /*count*/,
                                  int /*dstDevice*/,
                                  cudaStream_t /*stream*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMemAdvise(const void* /*devPtr*/,
                           size_t /*count*/,
                           cudaMemoryAdvise /*advice*/,
                           int /*device*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMemRangeGetAttribute(void* data,
                                      size_t dataSize,
                                      cudaMemRangeAttribute attribute,
                                      const void* /*devPtr*/,
                                      size_t /*count*/) {
    if (data == nullptr || dataSize == 0) {
        return fail(cudaErrorInvalidValue);
    }
    // On UMA: read-mostly is effectively always on; preferred location is device 0.
    if (attribute == cudaMemRangeAttributeReadMostly && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 1;
    } else if (attribute == cudaMemRangeAttributePreferredLocation && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;  // device 0
    } else if (attribute == cudaMemRangeAttributeLastPrefetchLocation && dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;
    } else if (dataSize >= sizeof(int)) {
        *reinterpret_cast<int*>(data) = 0;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaStreamAttachMemAsync(cudaStream_t stream, void* dev_ptr,
                                     size_t /*length*/, unsigned int flags) {
    if (dev_ptr == nullptr ||
        (flags != cudaMemAttachGlobal && flags != cudaMemAttachHost &&
         flags != cudaMemAttachSingle)) {
        return fail(cudaErrorInvalidValue);
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(dev_ptr, &resolved)) {
        return fail(cudaErrorInvalidValue);
    }
    return fail(enqueue_stream_host_op(stream, []() {}));
}

// ── Async memory pool API ────────────────────────────────────────────────────
// Allocation itself is host-side on UMA, but its lifetime still follows the
// selected stream: allocation returns immediately and free is deferred until
// prior work in that stream completes.

struct cudaMemPool_st {
    int device = 0;
};

static cudaMemPool_st g_default_mempool;

cudaError_t cudaMallocAsync(void** dev_ptr, size_t size, cudaStream_t stream) {
    if (dev_ptr == nullptr || size == 0) return fail(cudaErrorInvalidValue);
    if (cudaGraph_t graph = get_capture_graph(stream)) {
        cudaMemAllocNodeParams params{};
        params.bytesize = size;
        params.poolProps.allocType = cudaMemAllocationTypePinned;
        params.poolProps.handleTypes = cudaMemHandleTypeNone;
        params.poolProps.location.type = cudaMemLocationTypeDevice;
        params.poolProps.location.id = 0;
        cudaGraphNode_t dependency = graph->nodes.empty() ? nullptr : graph->nodes.back();
        cudaGraphNode_t node = nullptr;
        const cudaError_t status = cudaGraphAddMemAllocNode(
            &node, graph, dependency == nullptr ? nullptr : &dependency,
            dependency == nullptr ? 0 : 1, &params);
        if (status == cudaSuccess) *dev_ptr = params.dptr;
        return fail(status);
    }
    const cudaError_t stream_status = enqueue_stream_host_op(stream, []() {});
    if (stream_status != cudaSuccess) return fail(stream_status);
    return cudaMalloc(dev_ptr, size);
}

cudaError_t cudaFreeAsync(void* dev_ptr, cudaStream_t stream) {
    if (dev_ptr == nullptr) return fail(cudaSuccess);
    if (cudaGraph_t graph = get_capture_graph(stream)) {
        cudaGraphNode_t dependency = graph->nodes.empty() ? nullptr : graph->nodes.back();
        cudaGraphNode_t node = nullptr;
        return fail(cudaGraphAddMemFreeNode(
            &node, graph, dependency == nullptr ? nullptr : &dependency,
            dependency == nullptr ? 0 : 1, dev_ptr));
    }
    RuntimeState& state = runtime_state();
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!state.allocations.resolve(dev_ptr, &resolved) || resolved.offset != 0) {
        return fail(cudaErrorInvalidDevicePointer);
    }
    {
        std::lock_guard<std::mutex> lock(state.pending_free_mutex);
        if (!state.pending_async_frees.insert(dev_ptr).second) {
            return fail(cudaErrorInvalidDevicePointer);
        }
    }
    std::shared_ptr<GraphAllocationState> graph_allocation;
    {
        std::lock_guard<std::mutex> lock(state.graph_memory_mutex);
        const auto found = state.graph_allocations.find(dev_ptr);
        if (found != state.graph_allocations.end()) {
            graph_allocation = found->second.lock();
        }
    }
    const cudaError_t status = enqueue_stream_host_op(
        stream, [dev_ptr, graph_allocation = std::move(graph_allocation)]() {
        RuntimeState& callback_state = runtime_state();
        if (graph_allocation != nullptr) {
            (void)deactivate_graph_allocation(graph_allocation);
        } else {
            (void)callback_state.allocations.erase(dev_ptr);
        }
        std::lock_guard<std::mutex> lock(callback_state.pending_free_mutex);
        callback_state.pending_async_frees.erase(dev_ptr);
    });
    if (status != cudaSuccess) {
        std::lock_guard<std::mutex> lock(state.pending_free_mutex);
        state.pending_async_frees.erase(dev_ptr);
    }
    return fail(status);
}

cudaError_t cudaMemPoolCreate(cudaMemPool_t* pool, const cudaMemPoolProps* /*poolProps*/) {
    if (!pool) return fail(cudaErrorInvalidValue);
    *pool = new cudaMemPool_st();
    return fail(cudaSuccess);
}

cudaError_t cudaMemPoolDestroy(cudaMemPool_t pool) {
    if (pool && pool != &g_default_mempool) delete pool;
    return fail(cudaSuccess);
}

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr /*attr*/, void* /*value*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t /*pool*/, cudaMemPoolAttr attr, void* value) {
    if (!value) return fail(cudaErrorInvalidValue);
    // Return zero for all counters
    switch (attr) {
        case cudaMemPoolAttrReleaseThreshold:
            *static_cast<size_t*>(value) = 0;
            break;
        case cudaMemPoolAttrReservedMemCurrent:
        case cudaMemPoolAttrReservedMemHigh:
        case cudaMemPoolAttrUsedMemCurrent:
        case cudaMemPoolAttrUsedMemHigh:
            *static_cast<size_t*>(value) = 0;
            break;
        default:
            *static_cast<int*>(value) = 1;
            break;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* pool, int /*device*/) {
    if (!pool) return fail(cudaErrorInvalidValue);
    *pool = &g_default_mempool;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceSetMemPool(int /*device*/, cudaMemPool_t /*pool*/) {
    return fail(cudaSuccess);
}

cudaError_t cudaMallocFromPoolAsync(void** dev_ptr, size_t size, cudaMemPool_t /*pool*/, cudaStream_t stream) {
    return cudaMallocAsync(dev_ptr, size, stream);
}

// Pointer attribute query — classifies a pointer as host, device, or managed.
cudaError_t cudaPointerGetAttributes(cudaPointerAttributes* attributes, const void* ptr) {
    if (attributes == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    *attributes = {};
    attributes->device = 0;
    // On UMA, every CuMetal allocation is simultaneously host- and device-accessible.
    // Report the pointer as managed so callers handle it correctly.
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    const bool is_device_ptr =
        (ptr != nullptr) && runtime_state().allocations.resolve(ptr, &resolved);
    if (is_device_ptr) {
        attributes->type = cudaMemoryTypeManaged;
        attributes->devicePointer = const_cast<void*>(ptr);
        attributes->hostPointer = const_cast<void*>(ptr);
    } else {
        attributes->type = cudaMemoryTypeHost;
        attributes->hostPointer = const_cast<void*>(ptr);
    }
    return fail(cudaSuccess);
}

// Device selection by property — always returns device 0 (single GPU on Apple Silicon).
cudaError_t cudaChooseDevice(int* device, const cudaDeviceProp* /*prop*/) {
    if (device == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *device = 0;
    return fail(cudaSuccess);
}

// Peer access — Apple Silicon has a single GPU; no peer-to-peer access (spec §2.2).
cudaError_t cudaDeviceCanAccessPeer(int* can_access_peer, int /*device*/, int /*peer_device*/) {
    if (can_access_peer == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    *can_access_peer = 0;
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceEnablePeerAccess(int /*peer_device*/, unsigned int /*flags*/) {
    return fail(cudaErrorInvalidValue);
}

cudaError_t cudaDeviceDisablePeerAccess(int /*peer_device*/) {
    return fail(cudaErrorInvalidValue);
}

// Stream with priority — priority is ignored; Metal has no priority queue.
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* stream, unsigned int flags,
                                          int /*priority*/) {
    return cudaStreamCreateWithFlags(stream, flags);
}

// Priority range — Metal has no stream priority; both bounds are 0.
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;
    return fail(cudaSuccess);
}

// Device limits — cache-policy values are retained as correctness-neutral
// performance hints even though Metal controls physical cache residency.
cudaError_t cudaDeviceSetLimit(cudaLimit limit, size_t value) {
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    if (limit == cudaLimitPersistingL2CacheSize) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess ||
            value > static_cast<std::size_t>(prop.persistingL2CacheMaxSize)) {
            return fail(cudaErrorInvalidValue);
        }
        RuntimeState& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.device_heap_mutex);
        state.persisting_l2_limit = value;
    }
    if (limit == cudaLimitMallocHeapSize) {
        if (value < 32u || value > std::numeric_limits<std::uint32_t>::max()) {
            return fail(cudaErrorInvalidValue);
        }
        RuntimeState& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.device_heap_mutex);
        if (state.device_heap != nullptr && value != state.device_heap_size) {
            return fail(cudaErrorInvalidValue);
        }
        state.device_heap_size = value;
    }
    if (limit == cudaLimitPrintfFifoSize) {
        if (value == 0 ||
            value > static_cast<std::size_t>(
                        std::numeric_limits<std::uint32_t>::max()) *
                        sizeof(std::uint32_t)) {
            return fail(cudaErrorInvalidValue);
        }
        const std::size_t rounded =
            (value + sizeof(std::uint32_t) - 1u) &
            ~(sizeof(std::uint32_t) - 1u);
        RuntimeState& state = runtime_state();
        std::lock_guard<std::mutex> lock(state.device_heap_mutex);
        state.printf_fifo_size = rounded;
    }
    return fail(cudaSuccess);
}

cudaError_t cudaDeviceGetLimit(size_t* pValue, cudaLimit limit) {
    if (pValue == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    switch (limit) {
        case cudaLimitStackSize:
            *pValue = 1024;
            break;
        case cudaLimitPrintfFifoSize:
            {
                RuntimeState& state = runtime_state();
                std::lock_guard<std::mutex> lock(state.device_heap_mutex);
                *pValue = state.printf_fifo_size;
            }
            break;
        case cudaLimitMallocHeapSize:
            {
                RuntimeState& state = runtime_state();
                std::lock_guard<std::mutex> lock(state.device_heap_mutex);
                *pValue = state.device_heap_size;
            }
            break;
        case cudaLimitPersistingL2CacheSize:
            {
                RuntimeState& state = runtime_state();
                std::lock_guard<std::mutex> lock(state.device_heap_mutex);
                *pValue = state.persisting_l2_limit;
            }
            break;
        default:
            *pValue = 0;
            break;
    }
    return fail(cudaSuccess);
}

// Peer memcpy — single GPU; ignore device IDs and forward to standard memcpy.
cudaError_t cudaMemcpyPeer(void* dst, int /*dstDevice*/,
                            const void* src, int /*srcDevice*/,
                            size_t count) {
    return cudaMemcpy(dst, src, count, cudaMemcpyDefault);
}

cudaError_t cudaMemcpyPeerAsync(void* dst, int /*dstDevice*/,
                                 const void* src, int /*srcDevice*/,
                                 size_t count, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, cudaMemcpyDefault, stream);
}

// cudaLaunchHostFunc — enqueue a CPU function in the stream timeline. Later
// stream work and stream synchronization wait for the function to return.
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData) {
    if (fn == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    if (cudaGraph_t graph = get_capture_graph(stream)) {
        auto* node = new (std::nothrow) cudaGraphNode_st();
        if (node == nullptr) return fail(cudaErrorMemoryAllocation);
        node->type = cudaGraphNodeTypeHost;
        node->host_fn = fn;
        node->host_user_data = userData;
        append_captured_graph_node(graph, node);
        return fail(cudaSuccess);
    }
    const cudaError_t init_status = ensure_initialized();
    if (init_status != cudaSuccess) {
        return fail(init_status);
    }
    std::shared_ptr<cumetal::metal_backend::Stream> backend_stream;
    const cudaError_t resolve_status =
        resolve_runtime_stream(stream, &backend_stream, nullptr);
    if (resolve_status != cudaSuccess) {
        return fail(resolve_status);
    }
    std::string error;
    return fail(cumetal::metal_backend::enqueue_host_function(
        backend_stream, [fn, userData]() { fn(userData); }, &error));
}

// cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags — flags ignored on Metal.
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks,
                                                                    const void* func,
                                                                    int blockSize,
                                                                    size_t dynamicSMemSize,
                                                                    unsigned int /*flags*/) {
    return cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, func, blockSize,
                                                         dynamicSMemSize);
}

// The compiler lowers cooperative_groups::grid_group::sync to a device-wide
// sense-reversing barrier. Bound the grid to the conservative resident set used
// by the occupancy queries so a waiting block cannot starve an undispatched one.
cudaError_t cudaLaunchCooperativeKernel(const void* func,
                                         dim3 gridDim,
                                         dim3 blockDim,
                                         void** args,
                                         size_t sharedMem,
                                         cudaStream_t stream) {
    const std::uint64_t block_count =
        static_cast<std::uint64_t>(gridDim.x) * gridDim.y * gridDim.z;
    if (block_count == 0 || block_count > kMaxResidentCooperativeBlocks) {
        return fail(cudaErrorCooperativeLaunchTooLarge);
    }
    return cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

// ── Texture / Surface API ────────────────────────────────────────────────────
// Metal exposes native texture and sampler objects, but CuMetal does not yet
// carry CUDA resource/sampler descriptors through the compiler's kernel ABI to
// Metal bindings. These host objects therefore retain lifecycle/copy state only;
// there is deliberately no fake linear-load sampling fallback.

cudaError_t cudaMallocArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                             size_t width, size_t height, unsigned int flags) {
    if (array == nullptr || desc == nullptr || width == 0) {
        return fail(cudaErrorInvalidValue);
    }
    const size_t elem_size = static_cast<size_t>(
        (desc->x + desc->y + desc->z + desc->w + 7) / 8);
    if (height == 0) { height = 1; }
    auto* a = new CuMetalArray();
    a->width = width;
    a->height = height;
    a->depth = 1;
    a->flags = flags;
    a->desc = *desc;
    void* ptr = nullptr;
    const cudaError_t err = cudaMalloc(&ptr, width * height * elem_size);
    if (err != cudaSuccess) {
        delete a;
        return fail(err);
    }
    a->data = ptr;
    *array = reinterpret_cast<cudaArray_t>(a);
    return fail(cudaSuccess);
}

cudaError_t cudaMalloc3DArray(cudaArray_t* array, const cudaChannelFormatDesc* desc,
                              cudaExtent extent, unsigned int flags) {
    if (array == nullptr || desc == nullptr || extent.width == 0 ||
        extent.height == 0 || extent.depth == 0) {
        return fail(cudaErrorInvalidValue);
    }
    const size_t elem_size = static_cast<size_t>(
        (desc->x + desc->y + desc->z + desc->w + 7) / 8);
    if (elem_size == 0) return fail(cudaErrorInvalidValue);
    auto* created = new (std::nothrow) CuMetalArray();
    if (created == nullptr) return fail(cudaErrorMemoryAllocation);
    created->width = extent.width;
    created->height = extent.height;
    created->depth = extent.depth;
    created->flags = flags;
    created->desc = *desc;
    void* data = nullptr;
    const cudaError_t status =
        cudaMalloc(&data, extent.width * extent.height * extent.depth * elem_size);
    if (status != cudaSuccess) {
        delete created;
        return fail(status);
    }
    created->data = data;
    *array = reinterpret_cast<cudaArray_t>(created);
    return fail(cudaSuccess);
}

cudaError_t cudaFreeArray(cudaArray_t array) {
    if (array == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    auto* a = reinterpret_cast<CuMetalArray*>(array);
    cudaFree(a->data);
    delete a;
    return fail(cudaSuccess);
}

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                 const void* src, size_t spitch, size_t width,
                                 size_t height, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    auto* a = reinterpret_cast<CuMetalArray*>(dst);
    const size_t elem_size = static_cast<size_t>(
        (a->desc.x + a->desc.y + a->desc.z + a->desc.w + 7) / 8);
    const size_t dpitch = a->width * elem_size;
    auto* dst_base = static_cast<char*>(a->data) + hOffset * dpitch + wOffset;
    return cudaMemcpy2D(dst_base, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DFromArray(void* dst, size_t dpitch, cudaArray_const_t src,
                                   size_t wOffset, size_t hOffset, size_t width,
                                   size_t height, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    auto* a = reinterpret_cast<const CuMetalArray*>(src);
    const size_t elem_size = static_cast<size_t>(
        (a->desc.x + a->desc.y + a->desc.z + a->desc.w + 7) / 8);
    const size_t spitch = a->width * elem_size;
    const auto* src_base = static_cast<const char*>(a->data) + hOffset * spitch + wOffset;
    return cudaMemcpy2D(dst, dpitch, src_base, spitch, width, height, kind);
}

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                               const void* src, size_t count, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    auto* a = reinterpret_cast<CuMetalArray*>(dst);
    const size_t elem_size = static_cast<size_t>(
        (a->desc.x + a->desc.y + a->desc.z + a->desc.w + 7) / 8);
    auto* dst_ptr = static_cast<char*>(a->data) + hOffset * a->width * elem_size + wOffset;
    return cudaMemcpy(dst_ptr, src, count, kind);
}

cudaError_t cudaMemcpyFromArray(void* dst, cudaArray_const_t src, size_t wOffset,
                                 size_t hOffset, size_t count, cudaMemcpyKind kind) {
    if (dst == nullptr || src == nullptr) {
        return fail(cudaErrorInvalidValue);
    }
    auto* a = reinterpret_cast<const CuMetalArray*>(src);
    const size_t elem_size = static_cast<size_t>(
        (a->desc.x + a->desc.y + a->desc.z + a->desc.w + 7) / 8);
    const auto* src_ptr = static_cast<const char*>(a->data) + hOffset * a->width * elem_size + wOffset;
    return cudaMemcpy(dst, src_ptr, count, kind);
}

namespace {
std::mutex g_tex_mutex;
struct TextureObjectRecord {
    cudaResourceDesc resource{};
    cudaTextureDesc texture{};
    cudaResourceViewDesc view{};
    void* device_descriptor = nullptr;
};
std::unordered_map<cudaTextureObject_t, TextureObjectRecord> g_texture_objects;
struct SurfaceObjectRecord {
    cudaResourceDesc resource{};
    void* device_descriptor = nullptr;
};
std::unordered_map<cudaSurfaceObject_t, SurfaceObjectRecord> g_surface_objects;

void collect_texture_resource_residency(
    std::uintptr_t handle,
    std::vector<std::shared_ptr<cumetal::metal_backend::Buffer>>* buffers) {
    if (handle == 0 || buffers == nullptr) return;

    const cudaResourceDesc* resource = nullptr;
    std::lock_guard<std::mutex> lock(g_tex_mutex);
    if (const auto texture = g_texture_objects.find(
            static_cast<cudaTextureObject_t>(handle));
        texture != g_texture_objects.end()) {
        resource = &texture->second.resource;
    } else if (const auto surface = g_surface_objects.find(
                   static_cast<cudaSurfaceObject_t>(handle));
               surface != g_surface_objects.end()) {
        resource = &surface->second.resource;
    }
    if (resource == nullptr) return;

    const void* data = nullptr;
    if (resource->resType == cudaResourceTypeArray) {
        const auto* array =
            reinterpret_cast<const CuMetalArray*>(resource->res.array.array);
        if (array != nullptr) data = array->data;
    } else if (resource->resType == cudaResourceTypePitch2D) {
        data = resource->res.pitch2D.devPtr;
    } else if (resource->resType == cudaResourceTypeLinear) {
        data = resource->res.linear.devPtr;
    }
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (data != nullptr && runtime_state().allocations.resolve(data, &resolved) &&
        resolved.buffer != nullptr) {
        buffers->push_back(std::move(resolved.buffer));
    }
}

bool valid_resource_desc(const cudaResourceDesc& desc) {
    switch (desc.resType) {
        case cudaResourceDesc::cudaResourceTypeArray:
            return desc.res.array.array != nullptr;
        case cudaResourceDesc::cudaResourceTypeLinear:
            return desc.res.linear.devPtr != nullptr && desc.res.linear.sizeInBytes > 0;
        case cudaResourceDesc::cudaResourceTypePitch2D:
            return desc.res.pitch2D.devPtr != nullptr && desc.res.pitch2D.width > 0 &&
                   desc.res.pitch2D.height > 0 && desc.res.pitch2D.pitchInBytes > 0;
        default:
            return false;
    }
}

bool build_texture_descriptor(const cudaResourceDesc& resource,
                              const cudaTextureDesc* texture,
                              __cumetal_texture_descriptor* descriptor) {
    if (descriptor == nullptr) return false;
    *descriptor = {};
    const void* data = nullptr;
    cudaChannelFormatDesc channel{};
    if (resource.resType == cudaResourceTypeArray) {
        const auto* array = reinterpret_cast<const CuMetalArray*>(resource.res.array.array);
        if (array == nullptr) return false;
        data = array->data;
        channel = array->desc;
        descriptor->width = array->width;
        descriptor->height = array->height;
        descriptor->depth = array->depth;
        descriptor->element_bytes = static_cast<unsigned int>(
            (channel.x + channel.y + channel.z + channel.w + 7) / 8);
        descriptor->pitch_bytes = array->width * descriptor->element_bytes;
    } else if (resource.resType == cudaResourceTypePitch2D) {
        data = resource.res.pitch2D.devPtr;
        channel = resource.res.pitch2D.desc;
        descriptor->width = resource.res.pitch2D.width;
        descriptor->height = resource.res.pitch2D.height;
        descriptor->depth = 1;
        descriptor->element_bytes = static_cast<unsigned int>(
            (channel.x + channel.y + channel.z + channel.w + 7) / 8);
        descriptor->pitch_bytes = resource.res.pitch2D.pitchInBytes;
    } else if (resource.resType == cudaResourceTypeLinear) {
        data = resource.res.linear.devPtr;
        channel = resource.res.linear.desc;
        descriptor->element_bytes = static_cast<unsigned int>(
            (channel.x + channel.y + channel.z + channel.w + 7) / 8);
        if (descriptor->element_bytes == 0) return false;
        descriptor->width = resource.res.linear.sizeInBytes / descriptor->element_bytes;
        descriptor->height = 1;
        descriptor->depth = 1;
        descriptor->pitch_bytes = resource.res.linear.sizeInBytes;
    } else {
        return false;
    }
    if (descriptor->element_bytes == 0) return false;
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!runtime_state().allocations.resolve(data, &resolved) || resolved.buffer == nullptr ||
        resolved.buffer->device_address() == 0) {
        return false;
    }
    descriptor->data = resolved.buffer->device_address() + resolved.offset;
    descriptor->channel_kind = static_cast<unsigned int>(channel.f);
    if (texture != nullptr) {
        descriptor->read_mode = static_cast<unsigned int>(texture->readMode);
        descriptor->filter_mode = static_cast<unsigned int>(texture->filterMode);
        descriptor->normalized_coords = static_cast<unsigned int>(texture->normalizedCoords != 0);
        for (int axis = 0; axis < 3; ++axis)
            descriptor->address_mode[axis] = static_cast<unsigned int>(texture->addressMode[axis]);
    }
    return true;
}
}  // namespace

cudaError_t cudaCreateTextureObject(cudaTextureObject_t* pTexObject,
                                     const cudaResourceDesc* pResDesc,
                                     const cudaTextureDesc* pTexDesc,
                                     const cudaResourceViewDesc* pResViewDesc) {
    if (pTexObject == nullptr || pResDesc == nullptr || pTexDesc == nullptr ||
        !valid_resource_desc(*pResDesc)) {
        return fail(cudaErrorInvalidValue);
    }
    void* descriptor_allocation = nullptr;
    if (cudaMalloc(&descriptor_allocation, sizeof(__cumetal_texture_descriptor)) != cudaSuccess)
        return fail(cudaErrorMemoryAllocation);
    auto* descriptor = static_cast<__cumetal_texture_descriptor*>(descriptor_allocation);
    if (!build_texture_descriptor(*pResDesc, pTexDesc, descriptor)) {
        cudaFree(descriptor_allocation);
        return fail(cudaErrorInvalidValue);
    }
    const cudaTextureObject_t handle =
        static_cast<cudaTextureObject_t>(reinterpret_cast<std::uintptr_t>(descriptor_allocation));
    TextureObjectRecord record;
    record.resource = *pResDesc;
    record.texture = *pTexDesc;
    if (pResViewDesc != nullptr) record.view = *pResViewDesc;
    record.device_descriptor = descriptor_allocation;
    {
        std::lock_guard<std::mutex> lock(g_tex_mutex);
        g_texture_objects[handle] = record;
    }
    *pTexObject = handle;
    return fail(cudaSuccess);
}

cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    // A zero handle is the null texture object and destroying it is a no-op, the
    // same way freeing a null pointer is. GROMACS relies on this: a parameter
    // lookup table with no entries never creates a texture, and the teardown
    // path still calls destroy on the zero-initialized member.
    if (texObject == 0) {
        return fail(cudaSuccess);
    }
    void* descriptor = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tex_mutex);
        const auto found = g_texture_objects.find(texObject);
        if (found == g_texture_objects.end()) return fail(cudaErrorInvalidResourceHandle);
        descriptor = found->second.device_descriptor;
        g_texture_objects.erase(found);
    }
    return cudaFree(descriptor);
}

cudaError_t cudaGetTextureObjectResourceDesc(cudaResourceDesc* pResDesc,
                                              cudaTextureObject_t texObject) {
    if (pResDesc == nullptr) return fail(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(g_tex_mutex);
    const auto found = g_texture_objects.find(texObject);
    if (found == g_texture_objects.end()) return fail(cudaErrorInvalidResourceHandle);
    *pResDesc = found->second.resource;
    return fail(cudaSuccess);
}

cudaError_t cudaGetTextureObjectTextureDesc(cudaTextureDesc* pTexDesc,
                                             cudaTextureObject_t texObject) {
    if (pTexDesc == nullptr) return fail(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(g_tex_mutex);
    const auto found = g_texture_objects.find(texObject);
    if (found == g_texture_objects.end()) return fail(cudaErrorInvalidResourceHandle);
    *pTexDesc = found->second.texture;
    return fail(cudaSuccess);
}

cudaError_t cudaGetTextureObjectResourceViewDesc(cudaResourceViewDesc* pResViewDesc,
                                                  cudaTextureObject_t texObject) {
    if (pResViewDesc == nullptr) return fail(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(g_tex_mutex);
    const auto found = g_texture_objects.find(texObject);
    if (found == g_texture_objects.end()) return fail(cudaErrorInvalidResourceHandle);
    *pResViewDesc = found->second.view;
    return fail(cudaSuccess);
}

cudaError_t cudaCreateSurfaceObject(cudaSurfaceObject_t* pSurfObject,
                                     const cudaResourceDesc* pResDesc) {
    if (pSurfObject == nullptr || pResDesc == nullptr || !valid_resource_desc(*pResDesc)) {
        return fail(cudaErrorInvalidValue);
    }
    cudaTextureDesc texture{};
    void* descriptor_allocation = nullptr;
    if (cudaMalloc(&descriptor_allocation, sizeof(__cumetal_texture_descriptor)) != cudaSuccess)
        return fail(cudaErrorMemoryAllocation);
    if (!build_texture_descriptor(*pResDesc, &texture,
                                  static_cast<__cumetal_texture_descriptor*>(descriptor_allocation))) {
        cudaFree(descriptor_allocation);
        return fail(cudaErrorInvalidValue);
    }
    const cudaSurfaceObject_t handle =
        static_cast<cudaSurfaceObject_t>(reinterpret_cast<std::uintptr_t>(descriptor_allocation));
    SurfaceObjectRecord record{.resource = *pResDesc,
                               .device_descriptor = descriptor_allocation};
    {
        std::lock_guard<std::mutex> lock(g_tex_mutex);
        g_surface_objects[handle] = record;
    }
    *pSurfObject = handle;
    return fail(cudaSuccess);
}

cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    // Null handle, no-op -- see cudaDestroyTextureObject.
    if (surfObject == 0) {
        return fail(cudaSuccess);
    }
    void* descriptor = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_tex_mutex);
        const auto found = g_surface_objects.find(surfObject);
        if (found == g_surface_objects.end()) return fail(cudaErrorInvalidResourceHandle);
        descriptor = found->second.device_descriptor;
        g_surface_objects.erase(found);
    }
    return cudaFree(descriptor);
}

cudaError_t cudaGetSurfaceObjectResourceDesc(cudaResourceDesc* pResDesc,
                                              cudaSurfaceObject_t surfObject) {
    if (pResDesc == nullptr) return fail(cudaErrorInvalidValue);
    std::lock_guard<std::mutex> lock(g_tex_mutex);
    const auto found = g_surface_objects.find(surfObject);
    if (found == g_surface_objects.end()) return fail(cudaErrorInvalidResourceHandle);
    *pResDesc = found->second.resource;
    return fail(cudaSuccess);
}

cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource**,
                                         unsigned int, unsigned int) {
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource**, cudaStream_t) {
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource**, cudaStream_t) {
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaGraphicsResourceGetMappedPointer(void**, size_t*,
                                                 cudaGraphicsResource*) {
    return fail(cudaErrorNotSupported);
}

cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource*) {
    return fail(cudaErrorNotSupported);
}

cudaChannelFormatDesc cudaCreateChannelDesc(int x, int y, int z, int w,
                                             cudaChannelFormatKind f) {
    cudaChannelFormatDesc desc{};
    desc.x = x;
    desc.y = y;
    desc.z = z;
    desc.w = w;
    desc.f = f;
    return desc;
}

// ── Legacy thread API (batch 5) ───────────────────────────────────────────────
// These functions were deprecated in CUDA 5.0 but remain common in legacy code.

cudaError_t cudaThreadExit(void) {
    return cudaDeviceReset();
}

cudaError_t cudaThreadSynchronize(void) {
    return cudaDeviceSynchronize();
}

cudaError_t cudaThreadGetCacheConfig(cudaFuncCache* pCacheConfig) {
    return cudaDeviceGetCacheConfig(pCacheConfig);
}

cudaError_t cudaThreadSetCacheConfig(cudaFuncCache cacheConfig) {
    return cudaDeviceSetCacheConfig(cacheConfig);
}

}  // extern "C"

// Hash memory shim moved to runtime/rt/hash_shim.cpp to avoid redefinition
// conflicts with libc++ headers from the active SDK (see that file for details).
