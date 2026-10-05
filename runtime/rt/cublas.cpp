#include "cublas_v2.h"
#include "cuda_bf16.h"
#include "metal_backend.h"
#include "runtime_internal.h"
#include "blas1_kernels_msl.h"
#include "library_kernel_source.h"

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <type_traits>
#include <vector>
#include <Accelerate/Accelerate.h>

struct cublasContext {
    cudaStream_t stream = nullptr;
    cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
    cublasPointerMode_t pointer_mode = CUBLAS_POINTER_MODE_HOST;
    // Caller-provided workspace. nullptr selects the default pool; a non-null
    // pointer with size 0 is a real (empty) user workspace, not the default.
    // CuMetal's Accelerate/Metal paths manage their own scratch and do not
    // sub-allocate from this span; it is validated and tracked so callers get
    // CUDA's configuration semantics.
    void* workspace = nullptr;
    std::size_t workspace_size = 0;
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

namespace {

constexpr int kCublasCompatVersion = 12000;

bool debug_cublas_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("CUMETAL_DEBUG_CUBLAS");
        enabled = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

bool cublas_cpu_reference_enabled() {
    static int enabled = -1;
    if (enabled < 0) {
        const char* v = std::getenv("CUMETAL_CUBLAS_CPU_REFERENCE");
        enabled = (v != nullptr && v[0] != '\0' && v[0] != '0') ? 1 : 0;
    }
    return enabled != 0;
}

cudaDataType_t scale_type_for_compute(cublasComputeType_t compute_type, cudaDataType_t atype) {
    switch (compute_type) {
        case CUBLAS_COMPUTE_64F:
            return CUDA_R_64F;
        case CUBLAS_COMPUTE_16F:
            return CUDA_R_16F;
        case CUBLAS_COMPUTE_32F:
        case CUBLAS_COMPUTE_32F_FAST_TF32:
            return CUDA_R_32F;
        default:
            break;
    }
    // Fallback to operand type for older/less common modes.
    return atype;
}

bool is_valid_operation(cublasOperation_t op) {
    return op == CUBLAS_OP_N || op == CUBLAS_OP_T || op == CUBLAS_OP_C;
}

bool is_valid_fill_mode(cublasFillMode_t mode) {
    return mode == CUBLAS_FILL_MODE_LOWER || mode == CUBLAS_FILL_MODE_UPPER;
}

bool is_valid_math_mode(cublasMath_t mode) {
    return mode == CUBLAS_DEFAULT_MATH || mode == CUBLAS_TENSOR_OP_MATH ||
           mode == CUBLAS_PEDANTIC_MATH || mode == CUBLAS_TF32_TENSOR_OP_MATH;
}

bool is_valid_pointer_mode(cublasPointerMode_t mode) {
    return mode == CUBLAS_POINTER_MODE_HOST || mode == CUBLAS_POINTER_MODE_DEVICE;
}

template <typename T>
T* scalar_pointer_for_mode(cublasHandle_t handle, T* pointer) {
    if (handle == nullptr || pointer == nullptr) return nullptr;
    cublasPointerMode_t mode;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        mode = handle->pointer_mode;
    }
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    const bool tracked = cumetal::rt::resolve_allocation_for_pointer(pointer, &resolved);
    const bool is_device = tracked && resolved.kind == cumetal::rt::AllocationKind::kDevice;
    if (mode == CUBLAS_POINTER_MODE_HOST) {
        return is_device ? nullptr : pointer;
    }
    if (!is_device || resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
        resolved.remaining_size < sizeof(T)) {
        return nullptr;
    }
    return reinterpret_cast<T*>(
        static_cast<unsigned char*>(resolved.buffer->contents()) + resolved.offset);
}

template <typename T>
const T* scalar_pointer_for_mode(cublasHandle_t handle, const T* pointer) {
    return scalar_pointer_for_mode(handle, const_cast<T*>(pointer));
}

// ── FP64 level-1 on the GPU ─────────────────────────────────────────────────
//
// cuPDLP-C's GPU path calls cublasDaxpy, cublasDdot, cublasDnrm2 and
// cublasDscal on vectors the length of the LP. Serving those from the scalar
// CPU loops below made them 26% of a profiled PDLP solve on datt256 while the
// sparse products were already on the GPU, so the level-1 calls had become the
// thing worth moving.
//
// Every entry point keeps its CPU loop. The GPU path is allowed to decline --
// for a short vector, an untracked pointer, a misaligned offset, a kernel that
// will not compile -- and the answer is the same either way, only slower. That
// also means a broken kernel is invisible from the results alone, which is what
// CUMETAL_DEBUG_CUBLAS_BLAS1 is for.

// Mirrors Blas1Params in blas1_kernels_msl.h.
struct Blas1Params {
    std::uint32_t n;
    std::uint32_t incx;
    std::uint32_t incy;
    std::uint32_t op;
    std::uint64_t alpha_bits;
};
static_assert(sizeof(Blas1Params) == 24, "Blas1Params must match the MSL layout");

constexpr std::uint32_t kReduceOpDot = 0;
constexpr std::uint32_t kReduceOpSumSq = 1;
constexpr std::uint32_t kReduceOpAbsSum = 2;

constexpr unsigned kBlas1Block = 256;

// Partials the reduction leaves for the host to fold. Capping the grid keeps
// that fold trivially short; the kernel walks the vector with a grid stride, so
// a cap costs each thread more elements rather than leaving any uncomputed.
constexpr unsigned kMaxReducePartials = 1024;

// CUMETAL_BLAS_METAL: unset = auto, "1" = always, "0" = never. Same convention
// as CUMETAL_SPARSE_METAL.
enum class Blas1MetalPolicy { kAuto, kAlways, kNever };

Blas1MetalPolicy blas1_metal_policy() {
    static const Blas1MetalPolicy policy = [] {
        const char* v = std::getenv("CUMETAL_BLAS_METAL");
        if (v == nullptr || v[0] == '\0') return Blas1MetalPolicy::kAuto;
        if (v[0] == '0') return Blas1MetalPolicy::kNever;
        return Blas1MetalPolicy::kAlways;
    }();
    return policy;
}

// The elementwise kernels and the reductions cross over at very different
// lengths, so one threshold cannot serve both. Completed-call latency on an
// M4 Pro (cumetal_cublas_blas1_metal_bench, microseconds):
//
//        n        axpy cpu/gpu       dot cpu/gpu      nrm2 cpu/gpu
//     1024          2.8 / 11.5        1.8 / 110.8       2.5 / 107.0
//     4096          7.4 /  5.1        5.0 / 107.3       8.3 / 106.8
//    16384         26.9 /  5.8       18.4 / 106.2      30.1 / 106.7
//    65536        100.5 / 10.6       73.0 / 108.5     126.2 / 106.0
//   262144        430.6 / 35.6      350.4 / 135.0     516.1 / 129.2
//  1048576       1628.5 / 172.6    1361.5 / 157.8    2000.9 / 138.8
//
// axpy pays only for the enqueue, which the command-buffer batching amortizes,
// so it is ahead from a few thousand elements. The reductions have to
// synchronize to return a scalar to the host, and that wait is the flat ~106 us
// floor in their columns -- it does not shrink with n, so nothing below about
// 100k can win no matter how fast the kernel is.
//
// Both defaults sit past their measured crossover rather than on it, for the
// same reason the sparse threshold does: right at the crossing the two paths
// are within noise of each other and the choice wins or loses a few percent at
// random.
constexpr long long kDefaultBlas1ElementwiseThresholdN = 4096;
constexpr long long kDefaultBlas1ReduceThresholdN = 131072;

long long threshold_from_env(const char* name, long long fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return fallback;
    char* end = nullptr;
    const long long parsed = std::strtoll(v, &end, 10);
    return (end != nullptr && *end == '\0' && parsed >= 0) ? parsed : fallback;
}

long long blas1_elementwise_threshold_n() {
    static const long long threshold = threshold_from_env(
        "CUMETAL_BLAS_METAL_THRESHOLD_N", kDefaultBlas1ElementwiseThresholdN);
    return threshold;
}

long long blas1_reduce_threshold_n() {
    static const long long threshold = threshold_from_env(
        "CUMETAL_BLAS_METAL_REDUCE_THRESHOLD_N", kDefaultBlas1ReduceThresholdN);
    return threshold;
}

bool blas1_debug() {
    static const bool on = [] {
        const char* v = std::getenv("CUMETAL_DEBUG_CUBLAS_BLAS1");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

void blas1_note(const char* op, const char* why) {
    if (blas1_debug()) {
        fprintf(stderr, "CUMETAL_DEBUG_CUBLAS_BLAS1: %s on the CPU: %s\n", op, why);
    }
}

void blas1_note_gpu(const char* kernel, long long n) {
    if (blas1_debug()) {
        fprintf(stderr, "CUMETAL_DEBUG_CUBLAS_BLAS1: %s on the Apple GPU n=%lld\n", kernel, n);
    }
}

const std::string* blas1_source_path() {
    return cumetal::rt::stage_library_kernel_source("blas1_kernels",
                                                    cumetal::rt::kBlas1KernelsMsl);
}

// Shared by every level-1 kernel: the size gate, the stream, and the staged
// source. Returns false with a reason whenever the CPU loop should run instead.
bool blas1_metal_prologue(const char* op,
                          cublasHandle_t handle,
                          int n,
                          long long threshold,
                          std::shared_ptr<cumetal::metal_backend::Stream>* stream,
                          const std::string** source) {
    const Blas1MetalPolicy policy = blas1_metal_policy();
    if (policy == Blas1MetalPolicy::kNever) {
        blas1_note(op, "CUMETAL_BLAS_METAL=0");
        return false;
    }
    if (policy != Blas1MetalPolicy::kAlways && n < threshold) {
        blas1_note(op, "below the dispatch-cost threshold");
        return false;
    }
    // The handle's stream is the stream this work belongs on, so it is ordered
    // against the caller's other work exactly as real cuBLAS would order it. A
    // null stream resolves to the default stream, which is the same one the CPU
    // path would synchronize against.
    if (cumetal::rt::resolve_backend_stream(handle->stream, stream) != cudaSuccess) {
        blas1_note(op, "the handle's stream does not resolve to a backend stream");
        return false;
    }
    *source = blas1_source_path();
    if (*source == nullptr) {
        blas1_note(op, "could not stage the kernel source");
        return false;
    }
    return true;
}

void blas1_pack_params(Blas1Params params, cumetal::metal_backend::KernelArg* out) {
    out->kind = cumetal::metal_backend::KernelArg::Kind::kBytes;
    out->bytes.resize(sizeof(params));
    std::memcpy(out->bytes.data(), &params, sizeof(params));
}

std::size_t blas1_span_bytes(int n, int inc) {
    // The last element touched is (n-1)*inc, so the allocation has to reach
    // through it -- not merely hold n elements.
    return (static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(inc) + 1) *
           sizeof(double);
}

// axpy / scal / copy / swap: nothing is read back, so the launch stays async and
// the caller's own stream ordering decides when it lands. Synchronizing here
// would be the single change most likely to make the GPU path slower than the
// loop it replaced.
bool blas1_launch_elementwise(const char* op,
                              const char* kernel,
                              cublasHandle_t handle,
                              int n,
                              std::vector<cumetal::metal_backend::KernelArg> args,
                              Blas1Params params) {
    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    const std::string* source = nullptr;
    if (!blas1_metal_prologue(op, handle, n, blas1_elementwise_threshold_n(),
                              &stream, &source)) {
        return false;
    }

    blas1_pack_params(params, &args.back());

    cumetal::metal_backend::LaunchConfig config{};
    config.block = dim3(kBlas1Block, 1, 1);
    config.grid = dim3(static_cast<unsigned>((static_cast<long long>(n) + kBlas1Block - 1) /
                                             kBlas1Block), 1, 1);
    config.shared_memory_bytes = 0;
    config.semantic_quality = "reduced_precision_fp64";

    std::string error;
    if (cumetal::metal_backend::launch_kernel(*source, kernel, config, args, stream,
                                              &error) != cudaSuccess) {
        blas1_note(op, error.empty() ? "launch failed" : error.c_str());
        return false;
    }
    blas1_note_gpu(kernel, n);
    return true;
}

// A device scratch buffer for the reduction partials, grown as needed and kept
// for the process. Allocating one per call would put a cudaMalloc on the path of
// every dot product.
double* reduce_partials_buffer(unsigned count) {
    static std::mutex mutex;
    static double* buffer = nullptr;
    static unsigned capacity = 0;

    const std::lock_guard<std::mutex> lock(mutex);
    if (count > capacity) {
        double* grown = nullptr;
        if (cudaMalloc(reinterpret_cast<void**>(&grown), count * sizeof(double)) != cudaSuccess) {
            return nullptr;
        }
        if (buffer != nullptr) cudaFree(buffer);
        buffer = grown;
        capacity = count;
    }
    return buffer;
}

// dot / nrm2 / asum. The result has to reach the host, so unlike the
// elementwise kernels this one synchronizes -- which is what real cuBLAS does
// too in CUBLAS_POINTER_MODE_HOST.
//
// The host folds the per-threadgroup partials in true binary64. That is both
// cheaper than another dispatch and more accurate than folding them in the
// emulated pair.
bool blas1_reduce(const char* op,
                  cublasHandle_t handle,
                  int n,
                  const double* x, int incx,
                  const double* y, int incy,
                  std::uint32_t reduce_op,
                  double* out) {
    std::shared_ptr<cumetal::metal_backend::Stream> stream;
    const std::string* source = nullptr;
    if (!blas1_metal_prologue(op, handle, n, blas1_reduce_threshold_n(),
                              &stream, &source)) {
        return false;
    }

    const unsigned groups = static_cast<unsigned>(std::min<long long>(
        (static_cast<long long>(n) + kBlas1Block - 1) / kBlas1Block, kMaxReducePartials));
    double* partials = reduce_partials_buffer(groups);
    if (partials == nullptr) {
        blas1_note(op, "could not allocate the partials buffer");
        return false;
    }

    std::vector<cumetal::metal_backend::KernelArg> args(4);
    // y is unused for the single-vector reductions, but Metal still binds the
    // argument, so point it at x rather than leaving it null.
    const double* y_arg = reduce_op == kReduceOpDot ? y : x;
    const int y_inc = reduce_op == kReduceOpDot ? incy : incx;
    if (!cumetal::rt::resolve_kernel_buffer_arg(partials, groups * sizeof(double),
                                                sizeof(double), &args[0]) ||
        !cumetal::rt::resolve_kernel_buffer_arg(x, blas1_span_bytes(n, incx),
                                                sizeof(double), &args[1]) ||
        !cumetal::rt::resolve_kernel_buffer_arg(y_arg, blas1_span_bytes(n, y_inc),
                                                sizeof(double), &args[2])) {
        blas1_note(op, "an operand is not a tracked device allocation, or is misaligned");
        return false;
    }

    Blas1Params params{};
    params.n = static_cast<std::uint32_t>(n);
    params.incx = static_cast<std::uint32_t>(incx);
    params.incy = static_cast<std::uint32_t>(y_inc);
    params.op = reduce_op;
    blas1_pack_params(params, &args[3]);

    cumetal::metal_backend::LaunchConfig config{};
    config.block = dim3(kBlas1Block, 1, 1);
    config.grid = dim3(groups, 1, 1);
    // One float2 per simdgroup. The execution width is a device property the
    // host does not know here, so size for the narrowest plausible one: being
    // generous wastes a few hundred bytes, being short would corrupt the
    // reduction.
    config.shared_memory_bytes = (kBlas1Block / 8) * sizeof(float) * 2;
    config.semantic_quality = "reduced_precision_fp64";

    std::string error;
    if (cumetal::metal_backend::launch_kernel(*source, "cumetal_dreduce_f64", config, args,
                                              stream, &error) != cudaSuccess) {
        blas1_note(op, error.empty() ? "launch failed" : error.c_str());
        return false;
    }
    if (cumetal::metal_backend::synchronize(&error) != cudaSuccess) {
        blas1_note(op, error.empty() ? "synchronize failed" : error.c_str());
        return false;
    }

    double total = 0.0;
    for (unsigned i = 0; i < groups; ++i) total += partials[i];
    *out = total;
    blas1_note_gpu("cumetal_dreduce_f64", n);
    return true;
}

cublasStatus_t map_cuda_status_to_cublas(cudaError_t status) {
    if (status == cudaSuccess) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (status == cudaErrorInvalidValue || status == cudaErrorInvalidDevicePointer) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (status == cudaErrorMemoryAllocation) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <typename T>
T sym_element(const T* a, int lda, int row, int col, cublasFillMode_t uplo) {
    if (uplo == CUBLAS_FILL_MODE_UPPER) {
        return (row <= col) ? a[row + col * lda] : a[col + row * lda];
    }
    return (row >= col) ? a[row + col * lda] : a[col + row * lda];
}

cublasStatus_t synchronize_handle_stream(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const cudaError_t status = cudaStreamSynchronize(handle->stream);
    if (status != cudaSuccess) {
        if (debug_cublas_enabled()) {
            fprintf(stderr, "CUMETAL_DEBUG_CUBLAS: synchronize_handle_stream failed err=%d stream=%p\n",
                    static_cast<int>(status), static_cast<void*>(handle->stream));
        }
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    return CUBLAS_STATUS_SUCCESS;
}

bool read_pointer_table(const void* table,
                        int count,
                        std::vector<void*>* pointers) {
    if (count < 0 || pointers == nullptr || (count > 0 && table == nullptr)) {
        return false;
    }
    pointers->clear();
    pointers->reserve(static_cast<std::size_t>(count));
    if (count == 0) {
        return true;
    }

    const std::size_t table_bytes = static_cast<std::size_t>(count) * sizeof(void*);
    const unsigned char* table_contents = static_cast<const unsigned char*>(table);
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (cumetal::rt::resolve_allocation_for_pointer(table, &resolved)) {
        if (resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
            resolved.remaining_size < table_bytes) {
            return false;
        }
        table_contents = static_cast<const unsigned char*>(resolved.buffer->contents()) +
                         resolved.offset;
    }

    for (int i = 0; i < count; ++i) {
        void* pointer = nullptr;
        std::memcpy(&pointer,
                    table_contents + static_cast<std::size_t>(i) * sizeof(void*),
                    sizeof(pointer));
        pointers->push_back(pointer);
    }
    return true;
}

template <typename T>
bool resolve_writable_span(void* pointer, std::size_t count, T** output) {
    if (pointer == nullptr || output == nullptr) return false;
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(pointer, &resolved) ||
        resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
        count > resolved.remaining_size / sizeof(T)) {
        return false;
    }
    *output = reinterpret_cast<T*>(
        static_cast<unsigned char*>(resolved.buffer->contents()) + resolved.offset);
    return true;
}

void lapack_getrf(int n, float* matrix, int lda, int* pivots, int* info) {
    sgetrf_(&n, &n, matrix, &lda, pivots, info);
}

void lapack_getrf(int n, double* matrix, int lda, int* pivots, int* info) {
    dgetrf_(&n, &n, matrix, &lda, pivots, info);
}

template <typename T>
int getrf_without_pivoting(int n, T* matrix, int lda) {
    for (int k = 0; k < n; ++k) {
        const T diagonal = matrix[k + k * lda];
        if (diagonal == static_cast<T>(0)) return k + 1;
        for (int row = k + 1; row < n; ++row) {
            matrix[row + k * lda] /= diagonal;
        }
        for (int col = k + 1; col < n; ++col) {
            const T upper = matrix[k + col * lda];
            for (int row = k + 1; row < n; ++row) {
                matrix[row + col * lda] -= matrix[row + k * lda] * upper;
            }
        }
    }
    return 0;
}

template <typename T>
cublasStatus_t getrf_batched(cublasHandle_t handle,
                             int n,
                             T* const a_array[],
                             int lda,
                             int* pivot_array,
                             int* info_array,
                             int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || batch_count < 0 || lda < std::max(1, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || batch_count == 0) return CUBLAS_STATUS_SUCCESS;
    if (a_array == nullptr || info_array == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) return sync_status;

    std::vector<void*> matrices;
    if (!read_pointer_table(a_array, batch_count, &matrices)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    int* info = nullptr;
    if (!resolve_writable_span(info_array, static_cast<std::size_t>(batch_count), &info)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    int* pivots = nullptr;
    if (pivot_array != nullptr &&
        !resolve_writable_span(pivot_array,
                               static_cast<std::size_t>(n) * batch_count,
                               &pivots)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const std::size_t matrix_elements = static_cast<std::size_t>(lda) * n;
    for (int batch = 0; batch < batch_count; ++batch) {
        T* matrix = nullptr;
        if (!resolve_writable_span(matrices[batch], matrix_elements, &matrix)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        int factor_info = 0;
        if (pivots != nullptr) {
            lapack_getrf(n, matrix, lda,
                         pivots + static_cast<std::size_t>(batch) * n,
                         &factor_info);
        } else {
            factor_info = getrf_without_pivoting(n, matrix, lda);
        }
        info[batch] = factor_info;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Helper: read element of symmetric n×n matrix (column-major, upper or lower stored).
template<typename T>
static inline T symm_elem(const T* a, int lda, int i, int j, bool upper) {
    return upper ? (i <= j ? a[i + j * lda] : a[j + i * lda])
                 : (i >= j ? a[i + j * lda] : a[j + i * lda]);
}

// Helper: read element of complex Hermitian n×n matrix (col-major, upper or lower stored).
// Off-diagonal elements in the non-stored triangle are conj of the stored triangle.
static inline cuComplex herm_elem_f(const cuComplex* a, int lda, int i, int j, bool upper) {
    if (upper) return (i <= j) ? a[i + j * lda] : cuComplex{a[j + i * lda].x, -a[j + i * lda].y};
    return         (i >= j) ? a[i + j * lda] : cuComplex{a[j + i * lda].x, -a[j + i * lda].y};
}
static inline cuDoubleComplex herm_elem_d(const cuDoubleComplex* a, int lda, int i, int j, bool upper) {
    if (upper) return (i <= j) ? a[i + j * lda] : cuDoubleComplex{a[j + i * lda].x, -a[j + i * lda].y};
    return         (i >= j) ? a[i + j * lda] : cuDoubleComplex{a[j + i * lda].x, -a[j + i * lda].y};
}

}  // namespace

extern "C" {

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    cublasHandle_t created = new (std::nothrow) cublasContext();
    if (created == nullptr) {
        return CUBLAS_STATUS_ALLOC_FAILED;
    }
    *handle = created;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int* version) {
    if (handle == nullptr || version == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    *version = kCublasCompatVersion;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream_id) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->stream = stream_id;
    // CUDA unconditionally resets the handle's workspace to the default pool
    // when the stream changes.
    handle->workspace = nullptr;
    handle->workspace_size = 0;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t* stream_id) {
    if (handle == nullptr || stream_id == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *stream_id = handle->stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void* workspace,
                                  std::size_t workspace_size) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (workspace == nullptr) {
        // NULL selects the default pool.
        std::lock_guard<std::mutex> lock(handle->mutex);
        handle->workspace = nullptr;
        handle->workspace_size = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    // CUDA requires the user workspace to be 256-byte aligned.
    if (reinterpret_cast<std::uintptr_t>(workspace) % 256 != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // The whole span must fit inside one tracked allocation. An interior
    // pointer is fine as long as [workspace, workspace + size) stays within
    // its allocation; an untracked pointer is not a valid workspace.
    cumetal::rt::AllocationTable::ResolvedAllocation resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(workspace, &resolved) ||
        workspace_size > resolved.remaining_size) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->workspace = workspace;
    handle->workspace_size = workspace_size;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_math_mode(mode)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->math_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t* mode) {
    if (handle == nullptr || mode == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *mode = handle->math_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle, cublasPointerMode_t mode) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_pointer_mode(mode)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    handle->pointer_mode = mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle, cublasPointerMode_t* mode) {
    if (handle == nullptr || mode == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(handle->mutex);
    *mode = handle->pointer_mode;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle,
                           int n,
                           const float* alpha,
                           const float* x,
                           int incx,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const float* alpha_value_ptr = scalar_pointer_for_mode(handle, alpha);
    if (n < 0 || incx <= 0 || incy <= 0 || alpha_value_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) return sync_status;

    const float alpha_value = *alpha_value_ptr;
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const float* alpha_value_ptr = scalar_pointer_for_mode(handle, alpha);
    if (n < 0 || incx <= 0 || alpha_value_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) return sync_status;

    const float alpha_value = *alpha_value_ptr;
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasScopy(cublasHandle_t handle,
                           int n,
                           const float* x,
                           int incx,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) return sync_status;

    for (int i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        const int xi = i * incx;
        const int yi = i * incy;
        const float tmp = x[xi];
        x[xi] = y[yi];
        y[yi] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle,
                           int n,
                           const double* alpha,
                           const double* x,
                           int incx,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const double* alpha_value_ptr = scalar_pointer_for_mode(handle, alpha);
    if (n < 0 || incx <= 0 || incy <= 0 || alpha_value_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // No synchronize here on purpose. The GPU path below enqueues on the
    // handle's own stream, so it is already ordered against the caller's other
    // work; waiting first would serialize every axpy against the GPU and give
    // back exactly the latency this path exists to remove. The CPU fallback
    // does its own synchronize before touching the data.
    const double alpha_value = *alpha_value_ptr;
    {
        std::vector<cumetal::metal_backend::KernelArg> args(3);
        Blas1Params params{};
        params.n = static_cast<std::uint32_t>(n);
        params.incx = static_cast<std::uint32_t>(incx);
        params.incy = static_cast<std::uint32_t>(incy);
        std::memcpy(&params.alpha_bits, &alpha_value, sizeof(alpha_value));
        if (cumetal::rt::resolve_kernel_buffer_arg(y, blas1_span_bytes(n, incy),
                                                   sizeof(double), &args[0]) &&
            cumetal::rt::resolve_kernel_buffer_arg(x, blas1_span_bytes(n, incx),
                                                   sizeof(double), &args[1]) &&
            blas1_launch_elementwise("Daxpy", "cumetal_daxpy_f64", handle, n,
                                     std::move(args), params)) {
            return CUBLAS_STATUS_SUCCESS;
        }
    }
    const cublasStatus_t cpu_sync = synchronize_handle_stream(handle);
    if (cpu_sync != CUBLAS_STATUS_SUCCESS) {
        return cpu_sync;
    }
    for (int i = 0; i < n; ++i) {
        y[i * incy] = alpha_value * x[i * incx] + y[i * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    const double* alpha_value_ptr = scalar_pointer_for_mode(handle, alpha);
    if (n < 0 || incx <= 0 || alpha_value_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    // No synchronize here on purpose. The GPU path below enqueues on the
    // handle's own stream, so it is already ordered against the caller's other
    // work; waiting first would serialize every axpy against the GPU and give
    // back exactly the latency this path exists to remove. The CPU fallback
    // does its own synchronize before touching the data.
    const double alpha_value = *alpha_value_ptr;
    {
        std::vector<cumetal::metal_backend::KernelArg> args(2);
        Blas1Params params{};
        params.n = static_cast<std::uint32_t>(n);
        params.incx = static_cast<std::uint32_t>(incx);
        params.incy = static_cast<std::uint32_t>(incx);
        std::memcpy(&params.alpha_bits, &alpha_value, sizeof(alpha_value));
        if (cumetal::rt::resolve_kernel_buffer_arg(x, blas1_span_bytes(n, incx),
                                                   sizeof(double), &args[0]) &&
            blas1_launch_elementwise("Dscal", "cumetal_dscal_f64", handle, n,
                                     std::move(args), params)) {
            return CUBLAS_STATUS_SUCCESS;
        }
    }
    const cublasStatus_t cpu_sync = synchronize_handle_stream(handle);
    if (cpu_sync != CUBLAS_STATUS_SUCCESS) {
        return cpu_sync;
    }
    for (int i = 0; i < n; ++i) {
        x[i * incx] *= alpha_value;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDcopy(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        y[i * incy] = x[i * incx];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    for (int i = 0; i < n; ++i) {
        const int xi = i * incx;
        const int yi = i * incy;
        const double tmp = x[xi];
        x[xi] = y[yi];
        y[yi] = tmp;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSdot(cublasHandle_t handle,
                          int n,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    float* result_ptr = scalar_pointer_for_mode(handle, result);
    if (n < 0 || incx <= 0 || incy <= 0 || result_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result_ptr = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(x[i * incx]) * static_cast<double>(y[i * incy]);
    }
    *result_ptr = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot(cublasHandle_t handle,
                          int n,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    double* result_ptr = scalar_pointer_for_mode(handle, result);
    if (n < 0 || incx <= 0 || incy <= 0 || result_ptr == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result_ptr = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double gpu_result = 0.0;
    if (blas1_reduce("Ddot", handle, n, x, incx, y, incy, kReduceOpDot, &gpu_result)) {
        *result_ptr = gpu_result;
        return CUBLAS_STATUS_SUCCESS;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += x[i * incx] * y[i * incy];
    }
    *result_ptr = sum;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(x[i * incx]));
    }
    *result = static_cast<float>(sum);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDasum(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += std::fabs(x[i * incx]);
    }
    *result = sum;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float* x, int incx, float* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0f;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(x[i * incx]);
        sum_sq += v * v;
    }
    *result = static_cast<float>(std::sqrt(sum_sq));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle,
                           int n,
                           const double* x,
                           int incx,
                           double* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0.0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    double sum_sq = 0.0;
    if (blas1_reduce("Dnrm2", handle, n, x, incx, nullptr, incx, kReduceOpSumSq, &sum_sq)) {
        *result = std::sqrt(sum_sq);
        return CUBLAS_STATUS_SUCCESS;
    }

    sum_sq = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = x[i * incx];
        sum_sq += v * v;
    }
    *result = std::sqrt(sum_sq);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    float best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const float value = std::fabs(x[i * incx]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamax(cublasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    double best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const double value = std::fabs(x[i * incx]);
        if (value > best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float* x, int incx, int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    float best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const float value = std::fabs(x[i * incx]);
        if (value < best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasIdamin(cublasHandle_t handle,
                            int n,
                            const double* x,
                            int incx,
                            int* result) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (n < 0 || incx <= 0 || result == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (x == nullptr && n > 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        *result = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(result) != 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    int best_index = 0;
    double best_value = std::fabs(x[0]);
    for (int i = 1; i < n; ++i) {
        const double value = std::fabs(x[i * incx]);
        if (value < best_value) {
            best_value = value;
            best_index = i;
        }
    }
    *result = best_index + 1;  // cuBLAS uses 1-based indexing.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* b,
                           int ldb,
                           const float* beta,
                           float* c,
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::string error;
    const cudaError_t gemm_status = cumetal::metal_backend::gemm_f32(
        transa != CUBLAS_OP_N,
        transb != CUBLAS_OP_N,
        m,
        n,
        k,
        *alpha,
        a_resolved.buffer,
        a_resolved.offset,
        lda,
        b_resolved.buffer,
        b_resolved.offset,
        ldb,
        *beta,
        c_resolved.buffer,
        c_resolved.offset,
        ldc,
        nullptr,
        &error);
    if (gemm_status != cudaSuccess && debug_cublas_enabled()) {
        fprintf(stderr,
                "CUMETAL_DEBUG_CUBLAS: cublasSgemm failed err=%d transa=%d transb=%d m=%d n=%d k=%d "
                "lda=%d ldb=%d ldc=%d a_off=%zu b_off=%zu c_off=%zu msg=%s\n",
                static_cast<int>(gemm_status),
                static_cast<int>(transa != CUBLAS_OP_N),
                static_cast<int>(transb != CUBLAS_OP_N),
                m, n, k, lda, ldb, ldc,
                a_resolved.offset, b_resolved.offset, c_resolved.offset,
                error.c_str());
    }
    return map_cuda_status_to_cublas(gemm_status);
}

cublasStatus_t cublasSgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const float* alpha,
                                         const float* a,
                                         int lda,
                                         long long int stridea,
                                         const float* b,
                                         int ldb,
                                         long long int strideb,
                                         const float* beta,
                                         float* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || batch_count < 0 || alpha == nullptr || beta == nullptr ||
        stridea < 0 || strideb < 0 || stridec < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch_count == 0 || m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
    cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
    if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
        !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    std::string error;
    const cudaError_t gemm_status = cumetal::metal_backend::gemm_strided_batched_f32(
        transa != CUBLAS_OP_N,
        transb != CUBLAS_OP_N,
        m,
        n,
        k,
        *alpha,
        a_resolved.buffer,
        a_resolved.offset,
        lda,
        static_cast<std::size_t>(stridea) * sizeof(float),
        b_resolved.buffer,
        b_resolved.offset,
        ldb,
        static_cast<std::size_t>(strideb) * sizeof(float),
        *beta,
        c_resolved.buffer,
        c_resolved.offset,
        ldc,
        static_cast<std::size_t>(stridec) * sizeof(float),
        batch_count,
        nullptr,
        &error);

    return map_cuda_status_to_cublas(gemm_status);
}

cublasStatus_t cublasDgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* b,
                           int ldb,
                           const double* beta,
                           double* c,
                           int ldc) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(b) == 0 ||
        cumetalRuntimeIsDevicePointer(c) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
    const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
    if (lda < (a_rows > 1 ? a_rows : 1) || ldb < (b_rows > 1 ? b_rows : 1) || ldc < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int p = 0; p < k; ++p) {
                const double a_value = (transa == CUBLAS_OP_N) ? a[row + p * lda] : a[p + row * lda];
                const double b_value = (transb == CUBLAS_OP_N) ? b[p + col * ldb] : b[col + p * ldb];
                sum += a_value * b_value;
            }
            c[row + col * ldc] = alpha_value * sum + beta_value * c[row + col * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const double* alpha,
                                         const double* a,
                                         int lda,
                                         long long int stridea,
                                         const double* b,
                                         int ldb,
                                         long long int strideb,
                                         const double* beta,
                                         double* c,
                                         int ldc,
                                         long long int stridec,
                                         int batch_count) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || batch_count < 0 || alpha == nullptr || beta == nullptr ||
        stridea < 0 || strideb < 0 || stridec < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch_count == 0 || m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int batch = 0; batch < batch_count; ++batch) {
        const double* a_batch = a + batch * stridea;
        const double* b_batch = b + batch * strideb;
        double* c_batch = c + batch * stridec;
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < m; ++row) {
                double sum = 0.0;
                for (int p = 0; p < k; ++p) {
                    const double a_val = (transa == CUBLAS_OP_N) ? a_batch[row + p * lda]
                                                                  : a_batch[p + row * lda];
                    const double b_val = (transb == CUBLAS_OP_N) ? b_batch[p + col * ldb]
                                                                  : b_batch[col + p * ldb];
                    sum += a_val * b_val;
                }
                c_batch[row + col * ldc] = alpha_value * sum + beta_value * c_batch[row + col * ldc];
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int output_len = (trans == CUBLAS_OP_N) ? m : n;
    if (output_len == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    const float beta_value = *beta;
    if (trans == CUBLAS_OP_N) {
        for (int row = 0; row < m; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < n; ++col) {
                sum += a[row + col * lda] * x[col * incx];
            }
            y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
        }
    } else {
        for (int col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (int row = 0; row < m; ++row) {
                sum += a[row + col * lda] * x[row * incx];
            }
            y[col * incy] = alpha_value * sum + beta_value * y[col * incy];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemv(cublasHandle_t handle,
                           cublasOperation_t trans,
                           int m,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const int output_len = (trans == CUBLAS_OP_N) ? m : n;
    if (output_len == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    if (trans == CUBLAS_OP_N) {
        for (int row = 0; row < m; ++row) {
            double sum = 0.0;
            for (int col = 0; col < n; ++col) {
                sum += a[row + col * lda] * x[col * incx];
            }
            y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
        }
    } else {
        for (int col = 0; col < n; ++col) {
            double sum = 0.0;
            for (int row = 0; row < m; ++row) {
                sum += a[row + col * lda] * x[row * incx];
            }
            y[col * incy] = alpha_value * sum + beta_value * y[col * incy];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSger(cublasHandle_t handle,
                          int m,
                          int n,
                          const float* alpha,
                          const float* x,
                          int incx,
                          const float* y,
                          int incy,
                          float* a,
                          int lda) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (m < 0 || n < 0 || alpha == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0 ||
        cumetalRuntimeIsDevicePointer(a) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    for (int col = 0; col < n; ++col) {
        const float y_value = y[col * incy];
        for (int row = 0; row < m; ++row) {
            a[row + col * lda] += alpha_value * x[row * incx] * y_value;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger(cublasHandle_t handle,
                          int m,
                          int n,
                          const double* alpha,
                          const double* x,
                          int incx,
                          const double* y,
                          int incy,
                          double* a,
                          int lda) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (m < 0 || n < 0 || alpha == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (m > 1 ? m : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(x) == 0 || cumetalRuntimeIsDevicePointer(y) == 0 ||
        cumetalRuntimeIsDevicePointer(a) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    for (int col = 0; col < n; ++col) {
        const double y_value = y[col * incy];
        for (int row = 0; row < m; ++row) {
            a[row + col * lda] += alpha_value * x[row * incx] * y_value;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const float* alpha,
                           const float* a,
                           int lda,
                           const float* x,
                           int incx,
                           const float* beta,
                           float* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (n > 1 ? n : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const float alpha_value = *alpha;
    const float beta_value = *beta;
    for (int row = 0; row < n; ++row) {
        float sum = 0.0f;
        for (int col = 0; col < n; ++col) {
            sum += sym_element(a, lda, row, col, uplo) * x[col * incx];
        }
        y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymv(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const double* alpha,
                           const double* a,
                           int lda,
                           const double* x,
                           int incx,
                           const double* beta,
                           double* y,
                           int incy) {
    if (handle == nullptr) {
        return CUBLAS_STATUS_NOT_INITIALIZED;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n < 0 || alpha == nullptr || beta == nullptr || incx <= 0 || incy <= 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < (n > 1 ? n : 1)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if (a == nullptr || x == nullptr || y == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (cumetalRuntimeIsDevicePointer(a) == 0 || cumetalRuntimeIsDevicePointer(x) == 0 ||
        cumetalRuntimeIsDevicePointer(y) == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    const double alpha_value = *alpha;
    const double beta_value = *beta;
    for (int row = 0; row < n; ++row) {
        double sum = 0.0;
        for (int col = 0; col < n; ++col) {
            sum += sym_element(a, lda, row, col, uplo) * x[col * incx];
        }
        y[row * incy] = alpha_value * sum + beta_value * y[row * incy];
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// GemmEx / GemmStridedBatchedEx
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int m, int n, int k,
                            const void* alpha,
                            const void* a, cudaDataType_t atype, int lda,
                            const void* b, cudaDataType_t btype, int ldb,
                            const void* beta,
                            void* c, cudaDataType_t ctype, int ldc,
                            cublasComputeType_t compute_type,
                            cublasGemmAlgo_t /* algo */) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    // Route to the appropriate typed GEMM based on compute/data types.
    if ((compute_type == CUBLAS_COMPUTE_32F ||
         compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) &&
        atype == CUDA_R_32F && btype == CUDA_R_32F && ctype == CUDA_R_32F) {
        return cublasSgemm(handle, transa, transb, m, n, k,
                           static_cast<const float*>(alpha),
                           static_cast<const float*>(a), lda,
                           static_cast<const float*>(b), ldb,
                           static_cast<const float*>(beta),
                           static_cast<float*>(c), ldc);
    }

    if (compute_type == CUBLAS_COMPUTE_64F &&
        atype == CUDA_R_64F && btype == CUDA_R_64F && ctype == CUDA_R_64F) {
        return cublasDgemm(handle, transa, transb, m, n, k,
                           static_cast<const double*>(alpha),
                           static_cast<const double*>(a), lda,
                           static_cast<const double*>(b), ldb,
                           static_cast<const double*>(beta),
                           static_cast<double*>(c), ldc);
    }

    // FP16 compute or mixed types: upconvert to float, compute, downconvert.
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (a == nullptr || b == nullptr || c == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    // The conversion below dereferences device allocations from the CPU. Apple
    // Silicon unified memory makes the address accessible, but it does not make
    // preceding asynchronous Metal work complete. In particular, llama.cpp
    // dequantizes weights and converts activations on the handle stream
    // immediately before GemmEx. Synchronize before reading A/B/C so GEMM cannot
    // consume stale contents from those producer kernels.
    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    // Helper: read one scalar element as float from a typed buffer.
    auto read_f32 = [](const void* ptr, int idx, cudaDataType_t t) -> float {
        switch (t) {
            case CUDA_R_32F: return static_cast<const float*>(ptr)[idx];
            case CUDA_R_64F: return static_cast<float>(static_cast<const double*>(ptr)[idx]);
            case CUDA_R_16F: return static_cast<float>(static_cast<const __half*>(ptr)[idx]);
            case CUDA_R_16BF: return static_cast<float>(static_cast<const __nv_bfloat16*>(ptr)[idx]);
            default:         return 0.0f;
        }
    };
    auto write_f32 = [](void* ptr, int idx, float val, cudaDataType_t t) {
        switch (t) {
            case CUDA_R_32F: static_cast<float*>(ptr)[idx] = val; break;
            case CUDA_R_64F: static_cast<double*>(ptr)[idx] = static_cast<double>(val); break;
            case CUDA_R_16F: static_cast<__half*>(ptr)[idx] = static_cast<__half>(val); break;
            case CUDA_R_16BF: static_cast<__nv_bfloat16*>(ptr)[idx] = __nv_bfloat16(val); break;
            default: break;
        }
    };

    const cudaDataType_t alpha_type = scale_type_for_compute(compute_type, atype);
    const cudaDataType_t beta_type  = scale_type_for_compute(compute_type, ctype);
    const float alpha_f = read_f32(alpha, 0, alpha_type);
    const float beta_f  = read_f32(beta,  0, beta_type);

    // llama.cpp commonly multiplies FP16 operands with FP32 accumulation and
    // output.  MPS accepts independently typed input/result matrices, so bind
    // the tracked allocations directly instead of allocating and filling two
    // temporary FP32 matrices on the CPU.
    if (atype == CUDA_R_16F && btype == CUDA_R_16F && ctype == CUDA_R_32F &&
        (compute_type == CUBLAS_COMPUTE_32F ||
         compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) &&
        !cublas_cpu_reference_enabled()) {
        if (debug_cublas_enabled()) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_CUBLAS: native FP16/FP32 GemmEx "
                         "transa=%d transb=%d m=%d n=%d k=%d "
                         "lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
                         static_cast<int>(transa), static_cast<int>(transb),
                         m, n, k, lda, ldb, ldc,
                         static_cast<double>(alpha_f),
                         static_cast<double>(beta_f));
        }
        const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
        const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
        if (lda < std::max(a_rows, 1) || ldb < std::max(b_rows, 1) ||
            ldc < std::max(m, 1)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }

        cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
        if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
            !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
            !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }

        std::string error;
        const cudaError_t gemm_status = cumetal::metal_backend::gemm_f16_f32(
            transa != CUBLAS_OP_N, transb != CUBLAS_OP_N, m, n, k,
            alpha_f, a_resolved.buffer, a_resolved.offset, lda,
            b_resolved.buffer, b_resolved.offset, ldb,
            beta_f, c_resolved.buffer, c_resolved.offset, ldc,
            nullptr, &error);
        if (gemm_status != cudaSuccess && debug_cublas_enabled()) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_CUBLAS: native FP16/FP32 GemmEx failed "
                         "m=%d n=%d k=%d msg=%s\n",
                         m, n, k, error.c_str());
        }
        return map_cuda_status_to_cublas(gemm_status);
    }

    // Native FP16 library lowering. GGML's llama output projection uses
    // half A^T x half B -> half C; expanding its 49k x 576 weights to FP32 on
    // the CPU for every token dominates the one-layer workload. MPS accepts
    // FP16 matrices directly and accumulates the multiplication on Apple GPU.
    if (atype == CUDA_R_16F && btype == CUDA_R_16F && ctype == CUDA_R_16F &&
        compute_type == CUBLAS_COMPUTE_16F &&
        !cublas_cpu_reference_enabled()) {
        if (debug_cublas_enabled()) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_CUBLAS: native FP16 GemmEx "
                         "transa=%d transb=%d m=%d n=%d k=%d "
                         "lda=%d ldb=%d ldc=%d alpha=%g beta=%g\n",
                         static_cast<int>(transa), static_cast<int>(transb),
                         m, n, k, lda, ldb, ldc,
                         static_cast<double>(alpha_f),
                         static_cast<double>(beta_f));
        }
        const int a_rows = (transa == CUBLAS_OP_N) ? m : k;
        const int b_rows = (transb == CUBLAS_OP_N) ? k : n;
        if (lda < std::max(a_rows, 1) || ldb < std::max(b_rows, 1) ||
            ldc < std::max(m, 1)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }

        cumetal::rt::AllocationTable::ResolvedAllocation a_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation b_resolved;
        cumetal::rt::AllocationTable::ResolvedAllocation c_resolved;
        if (!cumetal::rt::resolve_allocation_for_pointer(a, &a_resolved) ||
            !cumetal::rt::resolve_allocation_for_pointer(b, &b_resolved) ||
            !cumetal::rt::resolve_allocation_for_pointer(c, &c_resolved)) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }

        std::string error;
        const cudaError_t gemm_status = cumetal::metal_backend::gemm_f16(
            transa != CUBLAS_OP_N, transb != CUBLAS_OP_N, m, n, k,
            alpha_f, a_resolved.buffer, a_resolved.offset, lda,
            b_resolved.buffer, b_resolved.offset, ldb,
            beta_f, c_resolved.buffer, c_resolved.offset, ldc,
            nullptr, &error);
        if (gemm_status != cudaSuccess && debug_cublas_enabled()) {
            std::fprintf(stderr,
                         "CUMETAL_DEBUG_CUBLAS: native FP16 GemmEx failed "
                         "m=%d n=%d k=%d msg=%s\n",
                         m, n, k, error.c_str());
        }
        return map_cuda_status_to_cublas(gemm_status);
    }

    // GPU-accelerated path: upconvert FP16/BF16 → FP32 on CPU (fast on Apple
    // Silicon UMA shared memory), run Metal GPU GEMM (cublasSgemm), then
    // downconvert FP32 → FP16/BF16 if the output type requires it.
    // This is substantially faster than the naive O(M·N·K) CPU loop.
    {
        // Pointer tables produced on the GPU contain Metal virtual addresses,
        // not necessarily the CPU mapping returned by cudaMalloc. Resolve both
        // identities through the allocation table before CPU conversion.
        const auto cpu_mapping = [](const void* ptr) -> const void* {
            cumetal::rt::AllocationTable::ResolvedAllocation resolved;
            if (!cumetal::rt::resolve_allocation_for_pointer(ptr, &resolved) ||
                resolved.buffer == nullptr || resolved.buffer->contents() == nullptr) {
                return ptr;
            }
            return static_cast<const unsigned char*>(resolved.buffer->contents()) +
                   resolved.offset;
        };
        const void* a_cpu = cpu_mapping(a);
        const void* b_cpu = cpu_mapping(b);
        void* c_cpu = const_cast<void*>(cpu_mapping(c));

        // A memory footprint: lda × (transa==N ? k : m) elements
        // B memory footprint: ldb × (transb==N ? n : k) elements
        // C memory footprint: ldc × n elements
        const int a_cols = (transa == CUBLAS_OP_N) ? k : m;
        const int b_cols = (transb == CUBLAS_OP_N) ? n : k;
        const std::size_t a_elems = static_cast<std::size_t>(lda) * a_cols;
        const std::size_t b_elems = static_cast<std::size_t>(ldb) * b_cols;
        const std::size_t c_elems = static_cast<std::size_t>(ldc) * n;

        float* a_f32 = nullptr;
        float* b_f32 = nullptr;
        float* c_f32 = nullptr;

        if (cudaMalloc(reinterpret_cast<void**>(&a_f32), a_elems * sizeof(float)) != cudaSuccess ||
            cudaMalloc(reinterpret_cast<void**>(&b_f32), b_elems * sizeof(float)) != cudaSuccess) {
            cudaFree(a_f32);
            cudaFree(b_f32);
            return CUBLAS_STATUS_ALLOC_FAILED;
        }

        const bool c_already_f32 = (ctype == CUDA_R_32F);
        if (!c_already_f32) {
            if (cudaMalloc(reinterpret_cast<void**>(&c_f32), c_elems * sizeof(float)) != cudaSuccess) {
                cudaFree(a_f32);
                cudaFree(b_f32);
                return CUBLAS_STATUS_ALLOC_FAILED;
            }
        }

        // Upconvert A → F32 (CPU-side, Apple Silicon UMA shared memory)
        for (std::size_t i = 0; i < a_elems; ++i) {
            a_f32[i] = read_f32(a_cpu, static_cast<int>(i), atype);
        }
        // Upconvert B → F32
        for (std::size_t i = 0; i < b_elems; ++i) {
            b_f32[i] = read_f32(b_cpu, static_cast<int>(i), btype);
        }

        float* c_out = c_already_f32 ? static_cast<float*>(c_cpu) : c_f32;

        // If C is not F32 and beta != 0, seed c_out with beta*C (converted)
        if (!c_already_f32 && beta_f != 0.0f) {
            for (std::size_t i = 0; i < c_elems; ++i) {
                c_out[i] = beta_f * read_f32(c_cpu, static_cast<int>(i), ctype);
            }
        }

        const float effective_beta = (!c_already_f32 && beta_f != 0.0f) ? 1.0f : beta_f;

        // Diagnostic escape hatch: run the same column-major GEMM through
        // Accelerate so model-level failures can distinguish malformed inputs
        // from the default Metal/MPS implementation. This is opt-in only and
        // never used by the normal GPU path.
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        if (cublas_cpu_reference_enabled()) {
            cblas_sgemm(CblasColMajor,
                        transa == CUBLAS_OP_N ? CblasNoTrans : CblasTrans,
                        transb == CUBLAS_OP_N ? CblasNoTrans : CblasTrans,
                        m, n, k, alpha_f, a_f32, lda, b_f32, ldb,
                        effective_beta, c_out, ldc);
            if (debug_cublas_enabled()) {
                std::fprintf(stderr,
                             "CUMETAL_DEBUG_CUBLAS: CPU reference GEMM "
                             "m=%d n=%d k=%d\n",
                             m, n, k);
            }
        } else {
            // Run Metal GPU GEMM (F32 × F32 → F32)
            st = cublasSgemm(handle, transa, transb, m, n, k,
                             &alpha_f, a_f32, lda,
                             b_f32, ldb,
                             &effective_beta, c_out, ldc);
        }

        // Downconvert C F32 → target type if necessary
        if (!c_already_f32 && st == CUBLAS_STATUS_SUCCESS) {
            for (std::size_t i = 0; i < c_elems; ++i) {
                write_f32(c_cpu, static_cast<int>(i), c_out[i], ctype);
            }
        }

        cudaFree(a_f32);
        cudaFree(b_f32);
        if (!c_already_f32) cudaFree(c_f32);
        return st;
    }
}

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
                                          cublasOperation_t transa,
                                          cublasOperation_t transb,
                                          int m, int n, int k,
                                          const void* alpha,
                                          const void* a, cudaDataType_t atype, int lda,
                                          long long int stridea,
                                          const void* b, cudaDataType_t btype, int ldb,
                                          long long int strideb,
                                          const void* beta,
                                          void* c, cudaDataType_t ctype, int ldc,
                                          long long int stridec,
                                          int batch_count,
                                          cublasComputeType_t compute_type,
                                          cublasGemmAlgo_t algo) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch_count < 0) return CUBLAS_STATUS_INVALID_VALUE;

    // Route fp32 strided batched via cublasSgemmStridedBatched.
    if ((compute_type == CUBLAS_COMPUTE_32F ||
         compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) &&
        atype == CUDA_R_32F && btype == CUDA_R_32F && ctype == CUDA_R_32F) {
        return cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                         static_cast<const float*>(alpha),
                                         static_cast<const float*>(a), lda, stridea,
                                         static_cast<const float*>(b), ldb, strideb,
                                         static_cast<const float*>(beta),
                                         static_cast<float*>(c), ldc, stridec,
                                         batch_count);
    }
    if (compute_type == CUBLAS_COMPUTE_64F &&
        atype == CUDA_R_64F && btype == CUDA_R_64F && ctype == CUDA_R_64F) {
        return cublasDgemmStridedBatched(handle, transa, transb, m, n, k,
                                         static_cast<const double*>(alpha),
                                         static_cast<const double*>(a), lda, stridea,
                                         static_cast<const double*>(b), ldb, strideb,
                                         static_cast<const double*>(beta),
                                         static_cast<double*>(c), ldc, stridec,
                                         batch_count);
    }

    // Delegate each batch slice to GemmEx.
    auto byte_offset = [](const void* base, long long int elems, cudaDataType_t t) -> const void* {
        std::size_t sz = 4;
        if (t == CUDA_R_64F) sz = 8;
        else if (t == CUDA_R_16F || t == CUDA_R_16BF) sz = 2;
        return static_cast<const char*>(base) + elems * sz;
    };
    auto byte_offset_mutable = [](void* base, long long int elems, cudaDataType_t t) -> void* {
        std::size_t sz = 4;
        if (t == CUDA_R_64F) sz = 8;
        else if (t == CUDA_R_16F || t == CUDA_R_16BF) sz = 2;
        return static_cast<char*>(base) + elems * sz;
    };

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasGemmEx(handle, transa, transb, m, n, k, alpha,
                         byte_offset(a, stridea * bi, atype), atype, lda,
                         byte_offset(b, strideb * bi, btype), btype, ldb,
                         beta,
                         byte_offset_mutable(c, stridec * bi, ctype), ctype, ldc,
                         compute_type, algo);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// Hgemm — half-precision GEMM (implemented via fp32 upconvert)
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m, int n, int k,
                           const __half* alpha,
                           const __half* a, int lda,
                           const __half* b, int ldb,
                           const __half* beta,
                           __half* c, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const float falpha = static_cast<float>(*alpha);
    const float fbeta  = static_cast<float>(*beta);
    return cublasGemmEx(handle, transa, transb, m, n, k,
                        &falpha, a, CUDA_R_16F, lda,
                                 b, CUDA_R_16F, ldb,
                        &fbeta,  c, CUDA_R_16F, ldc,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// ─────────────────────────────────────────────────────────────────────────────
// SgemmBatched / DgemmBatched — array-of-pointers batched GEMM
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasSgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const float* alpha,
                                  const float* const a_array[], int lda,
                                  const float* const b_array[], int ldb,
                                  const float* beta,
                                  float* const c_array[], int ldc,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batch_count < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (batch_count == 0 || m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    std::vector<void*> a_pointers;
    std::vector<void*> b_pointers;
    std::vector<void*> c_pointers;
    if (!read_pointer_table(a_array, batch_count, &a_pointers) ||
        !read_pointer_table(b_array, batch_count, &b_pointers) ||
        !read_pointer_table(c_array, batch_count, &c_pointers)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasSgemm(handle, transa, transb, m, n, k,
                        alpha, static_cast<const float*>(a_pointers[bi]), lda,
                        static_cast<const float*>(b_pointers[bi]), ldb,
                        beta, static_cast<float*>(c_pointers[bi]), ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmBatched(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m, int n, int k,
                                  const double* alpha,
                                  const double* const a_array[], int lda,
                                  const double* const b_array[], int ldb,
                                  const double* beta,
                                  double* const c_array[], int ldc,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batch_count < 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (batch_count == 0 || m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (a_array == nullptr || b_array == nullptr || c_array == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    std::vector<void*> a_pointers;
    std::vector<void*> b_pointers;
    std::vector<void*> c_pointers;
    if (!read_pointer_table(a_array, batch_count, &a_pointers) ||
        !read_pointer_table(b_array, batch_count, &b_pointers) ||
        !read_pointer_table(c_array, batch_count, &c_pointers)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasDgemm(handle, transa, transb, m, n, k,
                        alpha, static_cast<const double*>(a_pointers[bi]), lda,
                        static_cast<const double*>(b_pointers[bi]), ldb,
                        beta, static_cast<double*>(c_pointers[bi]), ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb,
                                   int m, int n, int k,
                                   const void* alpha,
                                   const void* const a_array[], cudaDataType_t atype, int lda,
                                   const void* const b_array[], cudaDataType_t btype, int ldb,
                                   const void* beta,
                                   void* const c_array[], cudaDataType_t ctype, int ldc,
                                   int batch_count,
                                   cublasComputeType_t compute_type,
                                   cublasGemmAlgo_t algo) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m < 0 || n < 0 || k < 0 || batch_count < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (alpha == nullptr || beta == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch_count == 0 || m == 0 || n == 0) {
        return CUBLAS_STATUS_SUCCESS;
    }
    if ((batch_count > 0) && (a_array == nullptr || b_array == nullptr || c_array == nullptr)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }

    const cublasStatus_t sync_status = synchronize_handle_stream(handle);
    if (sync_status != CUBLAS_STATUS_SUCCESS) {
        return sync_status;
    }

    std::vector<void*> a_pointers;
    std::vector<void*> b_pointers;
    std::vector<void*> c_pointers;
    if (!read_pointer_table(a_array, batch_count, &a_pointers) ||
        !read_pointer_table(b_array, batch_count, &b_pointers) ||
        !read_pointer_table(c_array, batch_count, &c_pointers)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s =
            cublasGemmEx(handle, transa, transb, m, n, k,
                         alpha,
                         a_pointers[bi], atype, lda,
                         b_pointers[bi], btype, ldb,
                         beta,
                         c_pointers[bi], ctype, ldc,
                         compute_type, algo);
        if (s != CUBLAS_STATUS_SUCCESS) {
            if (debug_cublas_enabled()) {
                const void* a_ptr = a_pointers[bi];
                const void* b_ptr = b_pointers[bi];
                const void* c_ptr = c_pointers[bi];
                std::fprintf(stderr,
                             "CUMETAL_DEBUG_CUBLAS: cublasGemmBatchedEx failed batch=%d status=%d m=%d n=%d k=%d lda=%d ldb=%d ldc=%d atype=%d btype=%d ctype=%d compute=%d a=%p b=%p c=%p dev(a,b,c)=(%d,%d,%d)\n",
                             bi, static_cast<int>(s), m, n, k, lda, ldb, ldc,
                             static_cast<int>(atype), static_cast<int>(btype), static_cast<int>(ctype),
                             static_cast<int>(compute_type),
                             a_ptr, b_ptr, c_ptr,
                             cumetalRuntimeIsDevicePointer(a_ptr),
                             cumetalRuntimeIsDevicePointer(b_ptr),
                             cumetalRuntimeIsDevicePointer(c_ptr));
            }
            return s;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    float* const a_array[],
                                    int lda,
                                    int* pivot_array,
                                    int* info_array,
                                    int batch_count) {
    return getrf_batched(handle, n, a_array, lda, pivot_array, info_array,
                         batch_count);
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle,
                                    int n,
                                    double* const a_array[],
                                    int lda,
                                    int* pivot_array,
                                    int* info_array,
                                    int batch_count) {
    return getrf_batched(handle, n, a_array, lda, pivot_array, info_array,
                         batch_count);
}

// ─────────────────────────────────────────────────────────────────────────────
// Strsm / Dtrsm — triangular solve with multiple RHS
// Solves: op(A) * X = alpha * B  (CUBLAS_SIDE_LEFT)
//      or X * op(A) = alpha * B  (CUBLAS_SIDE_RIGHT)
// Result is written back into B.
// ─────────────────────────────────────────────────────────────────────────────

// Macro to generate trsm body for float/double (avoids template inside extern "C").
// TRSM: solve op(A) * X = alpha * B  (side=LEFT) or  X * op(A) = alpha * B  (side=RIGHT).
// B is m×n, A is m×m (LEFT) or n×n (RIGHT). Result written into B.
#define CUMETAL_TRSM_BODY(T, zero_val, one_val)                                     \
    do {                                                                             \
        if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;                \
        if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;                   \
        if (side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT)                  \
            return CUBLAS_STATUS_INVALID_VALUE;                                     \
        if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;          \
        if (!is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;         \
        if (diag != CUBLAS_DIAG_NON_UNIT && diag != CUBLAS_DIAG_UNIT)              \
            return CUBLAS_STATUS_INVALID_VALUE;                                     \
        if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;                    \
        if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;                         \
        if (a == nullptr || b == nullptr) return CUBLAS_STATUS_INVALID_VALUE;       \
        const cublasStatus_t _sync = synchronize_handle_stream(handle);             \
        if (_sync != CUBLAS_STATUS_SUCCESS) return _sync;                           \
        const T alpha_val = *alpha;                                                  \
        const bool _unit = (diag == CUBLAS_DIAG_UNIT);                              \
        const bool _lower = (uplo == CUBLAS_FILL_MODE_LOWER);                       \
        const bool _notrans = (trans == CUBLAS_OP_N);                               \
        if (alpha_val != one_val) {                                                  \
            for (int _c = 0; _c < n; ++_c)                                          \
                for (int _r = 0; _r < m; ++_r)                                      \
                    b[_r + _c * ldb] *= alpha_val;                                  \
        }                                                                            \
        auto _ae = [&](int r, int c) -> T {                                         \
            return _notrans ? a[r + c * lda] : a[c + r * lda]; };                  \
        auto _ad = [&](int i) -> T {                                                \
            return _unit ? one_val : a[i + i * lda]; };                             \
        if (side == CUBLAS_SIDE_LEFT) {                                              \
            for (int _c = 0; _c < n; ++_c) {                                        \
                if ((_lower && _notrans) || (!_lower && !_notrans)) {               \
                    for (int _r = 0; _r < m; ++_r) {                               \
                        T _s = b[_r + _c * ldb];                                    \
                        for (int _p = 0; _p < _r; ++_p) _s -= _ae(_r,_p)*b[_p+_c*ldb];\
                        b[_r + _c * ldb] = _s / _ad(_r);                           \
                    }                                                                \
                } else {                                                             \
                    for (int _r = m-1; _r >= 0; --_r) {                            \
                        T _s = b[_r + _c * ldb];                                    \
                        for (int _p = _r+1; _p < m; ++_p) _s -= _ae(_r,_p)*b[_p+_c*ldb];\
                        b[_r + _c * ldb] = _s / _ad(_r);                           \
                    }                                                                \
                }                                                                    \
            }                                                                        \
        } else {                                                                     \
            if ((!_lower && _notrans) || (_lower && !_notrans)) {                   \
                for (int _c = 0; _c < n; ++_c) {                                   \
                    T _dv = _ad(_c);                                                 \
                    for (int _r = 0; _r < m; ++_r) b[_r+_c*ldb] /= _dv;           \
                    for (int _p = _c+1; _p < n; ++_p) {                            \
                        T _f = _ae(_c,_p);                                           \
                        for (int _r = 0; _r < m; ++_r) b[_r+_p*ldb] -= _f*b[_r+_c*ldb];\
                    }                                                                \
                }                                                                    \
            } else {                                                                 \
                for (int _c = n-1; _c >= 0; --_c) {                               \
                    T _dv = _ad(_c);                                                 \
                    for (int _r = 0; _r < m; ++_r) b[_r+_c*ldb] /= _dv;           \
                    for (int _p = 0; _p < _c; ++_p) {                              \
                        T _f = _ae(_c,_p);                                           \
                        for (int _r = 0; _r < m; ++_r) b[_r+_p*ldb] -= _f*b[_r+_c*ldb];\
                    }                                                                \
                }                                                                    \
            }                                                                        \
        }                                                                            \
        return CUBLAS_STATUS_SUCCESS;                                                \
    } while (0)

cublasStatus_t cublasStrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m, int n,
                           const float* alpha,
                           const float* a, int lda,
                           float* b, int ldb) {
    CUMETAL_TRSM_BODY(float, 0.0f, 1.0f);
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle,
                           cublasSideMode_t side,
                           cublasFillMode_t uplo,
                           cublasOperation_t trans,
                           cublasDiagType_t diag,
                           int m, int n,
                           const double* alpha,
                           const double* a, int lda,
                           double* b, int ldb) {
    CUMETAL_TRSM_BODY(double, 0.0, 1.0);
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag,
                                  int m, int n,
                                  const float* alpha,
                                  const float* const a_array[], int lda,
                                  float* const b_array[], int ldb,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch_count < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (batch_count > 0 && (a_array == nullptr || b_array == nullptr)) return CUBLAS_STATUS_INVALID_VALUE;

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s = cublasStrsm(handle, side, uplo, trans, diag,
                                             m, n, alpha,
                                             a_array[bi], lda,
                                             b_array[bi], ldb);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
                                  cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag,
                                  int m, int n,
                                  const double* alpha,
                                  const double* const a_array[], int lda,
                                  double* const b_array[], int ldb,
                                  int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (batch_count < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (batch_count > 0 && (a_array == nullptr || b_array == nullptr)) return CUBLAS_STATUS_INVALID_VALUE;

    for (int bi = 0; bi < batch_count; ++bi) {
        const cublasStatus_t s = cublasDtrsm(handle, side, uplo, trans, diag,
                                             m, n, alpha,
                                             a_array[bi], lda,
                                             b_array[bi], ldb);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

#undef CUMETAL_TRSM_BODY

// ─────────────────────────────────────────────────────────────────────────────
// SetVector / GetVector / SetMatrix / GetMatrix
// On Apple Silicon UMA all memory is coherent; these are strided memcpy helpers.
// ─────────────────────────────────────────────────────────────────────────────

cublasStatus_t cublasSetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy) {
    if (n < 0 || elem_size <= 0 || incx <= 0 || incy <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const std::size_t es = static_cast<std::size_t>(elem_size);
    for (int i = 0; i < n; ++i) {
        std::memcpy(static_cast<char*>(y) + static_cast<std::size_t>(i * incy) * es,
                    static_cast<const char*>(x) + static_cast<std::size_t>(i * incx) * es,
                    es);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVector(int n, int elem_size,
                               const void* x, int incx,
                               void* y, int incy) {
    return cublasSetVector(n, elem_size, x, incx, y, incy);
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb) {
    if (rows < 0 || cols < 0 || elem_size <= 0 || lda < rows || ldb < rows)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return CUBLAS_STATUS_SUCCESS;
    if (a == nullptr || b == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const std::size_t es  = static_cast<std::size_t>(elem_size);
    const std::size_t row_bytes = static_cast<std::size_t>(rows) * es;
    for (int col = 0; col < cols; ++col) {
        std::memcpy(static_cast<char*>(b) + static_cast<std::size_t>(col * ldb) * es,
                    static_cast<const char*>(a) + static_cast<std::size_t>(col * lda) * es,
                    row_bytes);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elem_size,
                               const void* a, int lda,
                               void* b, int ldb) {
    return cublasSetMatrix(rows, cols, elem_size, a, lda, b, ldb);
}

cublasStatus_t cublasSetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t stream) {
    if (n < 0 || elem_size <= 0 || incx <= 0 || incy <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (!x || !y) return CUBLAS_STATUS_INVALID_VALUE;
    const size_t es = static_cast<size_t>(elem_size);
    return cudaMemcpy2DAsync(y, static_cast<size_t>(incy) * es,
                             x, static_cast<size_t>(incx) * es,
                             es, static_cast<size_t>(n),
                             cudaMemcpyDefault, stream) == cudaSuccess
               ? CUBLAS_STATUS_SUCCESS
               : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasGetVectorAsync(int n, int elem_size,
                                    const void* x, int incx,
                                    void* y, int incy,
                                    cudaStream_t stream) {
    return cublasSetVectorAsync(n, elem_size, x, incx, y, incy, stream);
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t stream) {
    if (rows < 0 || cols < 0 || elem_size <= 0 || lda < rows || ldb < rows)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (rows == 0 || cols == 0) return CUBLAS_STATUS_SUCCESS;
    if (!a || !b) return CUBLAS_STATUS_INVALID_VALUE;
    const size_t es = static_cast<size_t>(elem_size);
    return cudaMemcpy2DAsync(b, static_cast<size_t>(ldb) * es,
                             a, static_cast<size_t>(lda) * es,
                             static_cast<size_t>(rows) * es,
                             static_cast<size_t>(cols),
                             cudaMemcpyDefault, stream) == cudaSuccess
               ? CUBLAS_STATUS_SUCCESS
               : CUBLAS_STATUS_EXECUTION_FAILED;
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elem_size,
                                    const void* a, int lda,
                                    void* b, int ldb,
                                    cudaStream_t stream) {
    return cublasSetMatrixAsync(rows, cols, elem_size, a, lda, b, ldb, stream);
}

const char* cublasGetStatusName(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
        default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

const char* cublasGetStatusString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS:
            return "cuBLAS operation completed successfully";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "cuBLAS library not initialized";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "cuBLAS resource allocation failed";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "Invalid value passed to cuBLAS function";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "Feature not supported on this architecture";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "Memory mapping error";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "cuBLAS kernel execution failed";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "cuBLAS internal error";
        default:
            return "Unknown cuBLAS status";
    }
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int* value) {
    if (value == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    switch (type) {
        case MAJOR_VERSION: *value = 12; break;
        case MINOR_VERSION: *value = 0;  break;
        case PATCH_LEVEL:   *value = 0;  break;
        default:
            return CUBLAS_STATUS_INVALID_VALUE;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-1 update: A = alpha * x * x^T + A  (column-major, only upper/lower triangle updated)
cublasStatus_t cublasSsyr(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const float* alpha,
                           const float* x, int incx,
                           float* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float a_val = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const float xj = a_val * x[j * incx];
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += xj * x[i * incx];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr(cublasHandle_t handle,
                           cublasFillMode_t uplo,
                           int n,
                           const double* alpha,
                           const double* x, int incx,
                           double* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double a_val = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const double xj = a_val * x[j * incx];
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += xj * x[i * incx];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-k update: C = alpha * op(A) * op(A)^T + beta * C  (only upper/lower triangle)
cublasStatus_t cublasSsyrk(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const float* alpha, const float* a, int lda,
                            const float* beta,  float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    // lda: if no_trans, A is n×k; else k×n
    if (no_trans && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && lda < k) return CUBLAS_STATUS_INVALID_VALUE;
    const float av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                const float ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const float aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrk(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const double* alpha, const double* a, int lda,
                            const double* beta,  double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && lda < k) return CUBLAS_STATUS_INVALID_VALUE;
    const double av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                const double ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const double aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Symmetric rank-2k update: C = alpha * (op(A)*op(B)^T + op(B)*op(A)^T) + beta * C
cublasStatus_t cublasSsyr2k(cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const float* alpha,
                             const float* a, int lda,
                             const float* b, int ldb,
                             const float* beta,
                             float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && (lda < n || ldb < n)) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && (lda < k || ldb < k)) return CUBLAS_STATUS_INVALID_VALUE;
    const float av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                const float ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const float bj = no_trans ? b[j + l * ldb] : b[l + j * ldb];
                const float bi = no_trans ? b[i + l * ldb] : b[l + i * ldb];
                const float aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * bj + bi * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2k(cublasHandle_t handle,
                             cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const double* alpha,
                             const double* a, int lda,
                             const double* b, int ldb,
                             const double* beta,
                             double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || k <= 0 || ldc < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    if (no_trans && (lda < n || ldb < n)) return CUBLAS_STATUS_INVALID_VALUE;
    if (!no_trans && (lda < k || ldb < k)) return CUBLAS_STATUS_INVALID_VALUE;
    const double av = *alpha, bv = *beta;
    for (int j = 0; j < n; ++j) {
        const int i_end = upper ? j + 1 : n;
        for (int i = upper ? 0 : j; i < i_end; ++i) {
            double sum = 0.0;
            for (int l = 0; l < k; ++l) {
                const double ai = no_trans ? a[i + l * lda] : a[l + i * lda];
                const double bj = no_trans ? b[j + l * ldb] : b[l + j * ldb];
                const double bi = no_trans ? b[i + l * ldb] : b[l + i * ldb];
                const double aj = no_trans ? a[j + l * lda] : a[l + j * lda];
                sum += ai * bj + bi * aj;
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS2: Ssyr2 / Dsyr2 — symmetric rank-2 update: A := α·x·yᵀ + α·y·xᵀ + A ────────

cublasStatus_t cublasSsyr2(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            int n,
                            const float* alpha,
                            const float* x, int incx,
                            const float* y, int incy,
                            float* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += av * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            int n,
                            const double* alpha,
                            const double* x, int incx,
                            const double* y, int incy,
                            double* a, int lda) {
    if (handle == nullptr || alpha == nullptr || x == nullptr || y == nullptr || a == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_start = upper ? 0 : j;
        const int i_end   = upper ? j + 1 : n;
        for (int i = i_start; i < i_end; ++i) {
            a[i + j * lda] += av * (x[i * incx] * y[j * incy] + y[i * incy] * x[j * incx]);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS3: Ssymm / Dsymm — symmetric matrix-matrix multiply ──────────────────────────

cublasStatus_t cublasSsymm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            int m, int n,
                            const float* alpha,
                            const float* a, int lda,
                            const float* b, int ldb,
                            const float* beta,
                            float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
        beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            if (left) {
                for (int k = 0; k < ka; ++k) {
                    sum += symm_elem(a, lda, i, k, upper) * b[k + j * ldb];
                }
            } else {
                for (int k = 0; k < ka; ++k) {
                    sum += b[i + k * ldb] * symm_elem(a, lda, k, j, upper);
                }
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            int m, int n,
                            const double* alpha,
                            const double* a, int lda,
                            const double* b, int ldb,
                            const double* beta,
                            double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr ||
        beta == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            if (left) {
                for (int k = 0; k < ka; ++k) {
                    sum += symm_elem(a, lda, i, k, upper) * b[k + j * ldb];
                }
            } else {
                for (int k = 0; k < ka; ++k) {
                    sum += b[i + k * ldb] * symm_elem(a, lda, k, j, upper);
                }
            }
            c[i + j * ldc] = av * sum + bv * c[i + j * ldc];
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS2: Strmv / Dtrmv — triangular matrix-vector multiply: x := op(A)·x ──────────

cublasStatus_t cublasStrmv(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int n,
                            const float* a, int lda,
                            float* x, int incx) {
    if (handle == nullptr || a == nullptr || x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper    = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit     = (diag  == CUBLAS_DIAG_UNIT);
    std::vector<float> tmp(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            bool in_tri;
            float aik;
            if (no_trans) {
                in_tri = upper ? (k >= i) : (k <= i);
                aik    = (k == i && unit) ? 1.0f : a[i + k * lda];
            } else {
                in_tri = upper ? (i >= k) : (i <= k);
                aik    = (k == i && unit) ? 1.0f : a[k + i * lda];
            }
            if (in_tri) sum += aik * x[k * incx];
        }
        tmp[i] = sum;
    }
    for (int i = 0; i < n; ++i) x[i * incx] = tmp[i];
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmv(cublasHandle_t handle,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int n,
                            const double* a, int lda,
                            double* x, int incx) {
    if (handle == nullptr || a == nullptr || x == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || lda < n) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const bool upper    = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit     = (diag  == CUBLAS_DIAG_UNIT);
    std::vector<double> tmp(n);
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            bool in_tri;
            double aik;
            if (no_trans) {
                in_tri = upper ? (k >= i) : (k <= i);
                aik    = (k == i && unit) ? 1.0 : a[i + k * lda];
            } else {
                in_tri = upper ? (i >= k) : (i <= k);
                aik    = (k == i && unit) ? 1.0 : a[k + i * lda];
            }
            if (in_tri) sum += aik * x[k * incx];
        }
        tmp[i] = sum;
    }
    for (int i = 0; i < n; ++i) x[i * incx] = tmp[i];
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS3: Strmm / Dtrmm — triangular matrix-matrix multiply ─────────────────────────
// B := alpha · op(A) · B  (SIDE_LEFT)  or  B := alpha · B · op(A)  (SIDE_RIGHT)
// cuBLAS v2 API: result written to C (B is input, C is output).

cublasStatus_t cublasStrmm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m, int n,
                            const float* alpha,
                            const float* a, int lda,
                            const float* b, int ldb,
                            float* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float av     = *alpha;
    const bool upper   = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit    = (diag  == CUBLAS_DIAG_UNIT);
    const bool left    = (side  == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float sum = 0.0f;
            for (int k = 0; k < ka; ++k) {
                float aelem;
                bool in_tri;
                if (left) {
                    int ar = no_trans ? i : k;
                    int ac = no_trans ? k : i;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0f : a[ar + ac * lda];
                } else {
                    int ar = no_trans ? k : j;
                    int ac = no_trans ? j : k;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0f : a[ar + ac * lda];
                }
                if (in_tri) sum += aelem * b[left ? (k + j * ldb) : (i + k * ldb)];
            }
            c[i + j * ldc] = av * sum;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmm(cublasHandle_t handle,
                            cublasSideMode_t side,
                            cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            cublasDiagType_t diag,
                            int m, int n,
                            const double* alpha,
                            const double* a, int lda,
                            const double* b, int ldb,
                            double* c, int ldc) {
    if (handle == nullptr || alpha == nullptr || a == nullptr || b == nullptr || c == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m <= 0 || n <= 0 || ldc < m || ldb < m) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double av    = *alpha;
    const bool upper   = (uplo  == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    const bool unit    = (diag  == CUBLAS_DIAG_UNIT);
    const bool left    = (side  == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    if ((left && lda < m) || (!left && lda < n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int k = 0; k < ka; ++k) {
                double aelem;
                bool in_tri;
                if (left) {
                    int ar = no_trans ? i : k;
                    int ac = no_trans ? k : i;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0 : a[ar + ac * lda];
                } else {
                    int ar = no_trans ? k : j;
                    int ac = no_trans ? j : k;
                    in_tri = upper ? (ar <= ac) : (ar >= ac);
                    aelem  = (ar == ac && unit) ? 1.0 : a[ar + ac * lda];
                }
                if (in_tri) sum += aelem * b[left ? (k + j * ldb) : (i + k * ldb)];
            }
            c[i + j * ldc] = av * sum;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // end extern "C" — templates must have C++ linkage

// ── BLAS1: Srotm / Drotm — apply modified Givens rotation ───────────────────────────
// param[0] = flag: -2 identity, -1 general H, 0 simplified (diag=1), 1 simplified (off-diag=±1)
// H = [h11 h12; h21 h22] where encoding per flag:
//   flag=-2: H=I (no-op)
//   flag=-1: H=[p1 p2; p3 p4]
//   flag= 0: H=[1  p2; p3  1]
//   flag= 1: H=[p1  1; -1 p4]

template<typename T>
static void rotm_impl(int n, T* x, int incx, T* y, int incy, const T* param) {
    const T flag = param[0];
    if (flag == T(-2)) return; // identity
    T h11, h12, h21, h22;
    if (flag == T(-1)) {
        h11 = param[1]; h12 = param[3];
        h21 = param[2]; h22 = param[4];
    } else if (flag == T(0)) {
        h11 = T(1);     h12 = param[3];
        h21 = param[2]; h22 = T(1);
    } else { // flag == 1
        h11 = param[1]; h12 = T(1);
        h21 = T(-1);    h22 = param[4];
    }
    for (int i = 0; i < n; ++i) {
        T xi = x[i * incx];
        T yi = y[i * incy];
        x[i * incx] = h11 * xi + h12 * yi;
        y[i * incy] = h21 * xi + h22 * yi;
    }
}

extern "C" {

cublasStatus_t cublasSrotm(cublasHandle_t handle, int n,
                            float* x, int incx, float* y, int incy,
                            const float* param) {
    if (handle == nullptr || x == nullptr || y == nullptr || param == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx == 0 || incy == 0) return CUBLAS_STATUS_SUCCESS;
    rotm_impl(n, x, incx, y, incy, param);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotm(cublasHandle_t handle, int n,
                            double* x, int incx, double* y, int incy,
                            const double* param) {
    if (handle == nullptr || x == nullptr || y == nullptr || param == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (n <= 0 || incx == 0 || incy == 0) return CUBLAS_STATUS_SUCCESS;
    rotm_impl(n, x, incx, y, incy, param);
    return CUBLAS_STATUS_SUCCESS;
}

} // end extern "C" for rotm — rotmg template must have C++ linkage

// ── BLAS1: Srotmg / Drotmg — construct modified Givens rotation ──────────────────────
// Computes H such that H * [sqrt(d1)*x1; sqrt(d2)*y1] = [r; 0]
// Updates d1, d2, x1 in-place; param[0..4] encodes flag and H elements.

template<typename T>
static void rotmg_impl(T* d1, T* d2, T* x1, T y1, T* param) {
    // Based on Lawson et al. algorithm
    constexpr T GAMMA = T(4096);
    constexpr T GAMMA_SQ = GAMMA * GAMMA;
    constexpr T INV_GAMMA_SQ = T(1) / GAMMA_SQ;

    T flag, h11, h12, h21, h22;

    if (*d1 < T(0)) {
        // Force d1 >= 0 by making H identity → zeroes x1
        flag = T(-1); h11 = T(0); h12 = T(0); h21 = T(0); h22 = T(0);
        *d1 = T(0); *d2 = T(0); *x1 = T(0);
        param[0] = flag; param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
        return;
    }

    T p2 = (*d2) * y1;
    if (p2 == T(0)) {
        // H = identity: no rotation needed
        param[0] = T(-2);
        return;
    }

    T p1 = (*d1) * (*x1);
    T q2 = p2 * y1;
    T q1 = p1 * (*x1);

    if (std::abs(q1) > std::abs(q2)) {
        // flag = 0: h12, h21 stored; h11=h22=1
        h21 = -y1 / (*x1);
        h12 = p2 / p1;
        T u = T(1) - h12 * h21;
        if (u > T(0)) {
            flag = T(0);
            *d1 /= u;
            *d2 /= u;
            *x1 *= u;
        } else {
            flag = T(-1);
            h11 = T(0); h22 = T(0);
            *d1 = T(0); *d2 = T(0); *x1 = T(0);
        }
    } else {
        if (q2 < T(0)) {
            // Cannot make positive d1
            flag = T(-1); h11 = T(0); h12 = T(0); h21 = T(0); h22 = T(0);
            *d1 = T(0); *d2 = T(0); *x1 = T(0);
            param[0] = flag; param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
            return;
        }
        // flag = 1: h11, h22 stored; h12=1, h21=-1
        flag = T(1);
        h11 = p1 / p2;
        h22 = (*x1) / y1;
        T u = T(1) + h11 * h22;
        T temp_d1 = *d2 / u;
        *d2 = *d1 / u;
        *d1 = temp_d1;
        *x1 = y1 * u;
        h12 = T(1); h21 = T(-1);
    }

    // Rescale to avoid overflow/underflow
    if (flag != T(-1)) {
        while (*d1 <= INV_GAMMA_SQ || *d1 >= GAMMA_SQ) {
            if (*d1 <= INV_GAMMA_SQ) {
                flag = T(-1); *d1 *= GAMMA_SQ; *d2 *= GAMMA_SQ;
                if (flag == T(0)) { h11 /= GAMMA; h12 /= GAMMA; }
                else               { h11 /= GAMMA; h22 /= GAMMA; }
                *x1 /= GAMMA;
            } else {
                flag = T(-1); *d1 /= GAMMA_SQ; *d2 /= GAMMA_SQ;
                if (flag == T(0)) { h11 *= GAMMA; h12 *= GAMMA; }
                else               { h11 *= GAMMA; h22 *= GAMMA; }
                *x1 *= GAMMA;
            }
        }
    }

    if (flag == T(0)) {
        param[1] = T(1); param[2] = h21; param[3] = h12; param[4] = T(1);
    } else if (flag == T(1)) {
        param[1] = h11; param[2] = T(-1); param[3] = T(1); param[4] = h22;
    } else {
        param[1] = h11; param[2] = h21; param[3] = h12; param[4] = h22;
    }
    param[0] = flag;
}

extern "C" {

cublasStatus_t cublasSrotmg(cublasHandle_t handle,
                             float* d1, float* d2, float* x1, const float* y1,
                             float* param) {
    if (handle == nullptr || d1 == nullptr || d2 == nullptr || x1 == nullptr
        || y1 == nullptr || param == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    rotmg_impl(d1, d2, x1, *y1, param);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotmg(cublasHandle_t handle,
                             double* d1, double* d2, double* x1, const double* y1,
                             double* param) {
    if (handle == nullptr || d1 == nullptr || d2 == nullptr || x1 == nullptr
        || y1 == nullptr || param == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    rotmg_impl(d1, d2, x1, *y1, param);
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS1: Srot / Drot — apply Givens rotation ───────────────────────────────────────

cublasStatus_t cublasSrot(cublasHandle_t handle,
                           int n,
                           float* x, int incx,
                           float* y, int incy,
                           const float* c,
                           const float* s) {
    if (handle == nullptr || x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float cv = *c, sv = *s;
    for (int i = 0; i < n; ++i) {
        const float xi = x[i * incx];
        const float yi = y[i * incy];
        x[i * incx] =  cv * xi + sv * yi;
        y[i * incy] = -sv * xi + cv * yi;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrot(cublasHandle_t handle,
                           int n,
                           double* x, int incx,
                           double* y, int incy,
                           const double* c,
                           const double* s) {
    if (handle == nullptr || x == nullptr || y == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n <= 0 || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double cv = *c, sv = *s;
    for (int i = 0; i < n; ++i) {
        const double xi = x[i * incx];
        const double yi = y[i * incy];
        x[i * incx] =  cv * xi + sv * yi;
        y[i * incy] = -sv * xi + cv * yi;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── BLAS1: Srotg / Drotg — construct Givens rotation ─────────────────────────────────
// Given (a,b): compute (c,s,r,z) such that [c s; -s c]*[a;b] = [r;0].
// a is overwritten with r; b with z.

cublasStatus_t cublasSrotg(cublasHandle_t handle,
                            float* a, float* b,
                            float* c, float* s) {
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const float fa = std::fabs(*a), fb = std::fabs(*b);
    if (fb == 0.0f) {
        *c = 1.0f; *s = 0.0f; *b = 0.0f;
    } else if (fa == 0.0f) {
        *c = 0.0f; *s = (*b > 0.0f) ? 1.0f : -1.0f;
        *a = fb; *b = 1.0f;
    } else if (fb > fa) {
        const float t  = *a / *b;
        const float sg = (*b > 0.0f) ? 1.0f : -1.0f;
        *s = sg / std::sqrt(1.0f + t * t);
        *c = *s * t;
        *a = *b / *s;
        *b = (std::fabs(*c) > 0.0f && std::fabs(*c) < 1.0f) ? 1.0f / *c : 1.0f;
    } else {
        const float t  = *b / *a;
        const float sg = (*a > 0.0f) ? 1.0f : -1.0f;
        *c = sg / std::sqrt(1.0f + t * t);
        *s = *c * t;
        *a = *a / *c;
        *b = (std::fabs(*s) < 1.0f) ? *s : (std::fabs(*c) > 0.0f ? 1.0f / *c : 1.0f);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDrotg(cublasHandle_t handle,
                            double* a, double* b,
                            double* c, double* s) {
    if (handle == nullptr || a == nullptr || b == nullptr || c == nullptr || s == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const double fa = std::fabs(*a), fb = std::fabs(*b);
    if (fb == 0.0) {
        *c = 1.0; *s = 0.0; *b = 0.0;
    } else if (fa == 0.0) {
        *c = 0.0; *s = (*b > 0.0) ? 1.0 : -1.0;
        *a = fb; *b = 1.0;
    } else if (fb > fa) {
        const double t  = *a / *b;
        const double sg = (*b > 0.0) ? 1.0 : -1.0;
        *s = sg / std::sqrt(1.0 + t * t);
        *c = *s * t;
        *a = *b / *s;
        *b = (std::fabs(*c) > 0.0 && std::fabs(*c) < 1.0) ? 1.0 / *c : 1.0;
    } else {
        const double t  = *b / *a;
        const double sg = (*a > 0.0) ? 1.0 : -1.0;
        *c = sg / std::sqrt(1.0 + t * t);
        *s = *c * t;
        *a = *a / *c;
        *b = (std::fabs(*s) < 1.0) ? *s : (std::fabs(*c) > 0.0 ? 1.0 / *c : 1.0);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Complex GEMM / GEMV (batch 5) ────────────────────────────────────────────
// Apple UMA: all MTLBuffers are StorageModeShared; CPU can access them directly.
// MPS has no complex MPSMatrixMultiplication, so we use a reference CPU loop.

static inline cuComplex cmul_f(cuComplex a, cuComplex b) {
    return { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
}
static inline cuComplex cadd_f(cuComplex a, cuComplex b) { return { a.x + b.x, a.y + b.y }; }
static inline cuComplex cconj_f(cuComplex a) { return { a.x, -a.y }; }

static inline cuDoubleComplex cmul_d(cuDoubleComplex a, cuDoubleComplex b) {
    return { a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x };
}
static inline cuDoubleComplex cadd_d(cuDoubleComplex a, cuDoubleComplex b) {
    return { a.x + b.x, a.y + b.y };
}
static inline cuDoubleComplex cconj_d(cuDoubleComplex a) { return { a.x, -a.y }; }

// Column-major element accessor for op(A).
static inline cuComplex celem_f(const cuComplex* A, int ld,
                                 cublasOperation_t op, int row, int col) {
    if (op == CUBLAS_OP_N) return A[(size_t)col * ld + row];
    if (op == CUBLAS_OP_T) return A[(size_t)row * ld + col];
    return cconj_f(A[(size_t)row * ld + col]);
}
static inline cuDoubleComplex celem_d(const cuDoubleComplex* A, int ld,
                                       cublasOperation_t op, int row, int col) {
    if (op == CUBLAS_OP_N) return A[(size_t)col * ld + row];
    if (op == CUBLAS_OP_T) return A[(size_t)row * ld + col];
    return cconj_d(A[(size_t)row * ld + col]);
}

static inline enum CBLAS_TRANSPOSE cublas_to_cblas_trans(cublasOperation_t op) {
    switch (op) {
        case CUBLAS_OP_N: return CblasNoTrans;
        case CUBLAS_OP_T: return CblasTrans;
        default:          return CblasConjTrans;
    }
}

cublasStatus_t cublasCgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const cuComplex* alpha,
                            const cuComplex* A, int lda,
                            const cuComplex* B, int ldb,
                            const cuComplex* beta,
                            cuComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_st = synchronize_handle_stream(handle);
    if (sync_st != CUBLAS_STATUS_SUCCESS) return sync_st;

    cblas_cgemm(CblasColMajor,
                cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm(cublasHandle_t handle,
                            cublasOperation_t transa, cublasOperation_t transb,
                            int m, int n, int k,
                            const cuDoubleComplex* alpha,
                            const cuDoubleComplex* A, int lda,
                            const cuDoubleComplex* B, int ldb,
                            const cuDoubleComplex* beta,
                            cuDoubleComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb))
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || alpha == nullptr || beta == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_st = synchronize_handle_stream(handle);
    if (sync_st != CUBLAS_STATUS_SUCCESS) return sync_st;

    cblas_zgemm(CblasColMajor,
                cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCgemv(cublasHandle_t handle,
                            cublasOperation_t trans,
                            int m, int n,
                            const cuComplex* alpha,
                            const cuComplex* A, int lda,
                            const cuComplex* x, int incx,
                            const cuComplex* beta,
                            cuComplex* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_st = synchronize_handle_stream(handle);
    if (sync_st != CUBLAS_STATUS_SUCCESS) return sync_st;

    cblas_cgemv(CblasColMajor, cublas_to_cblas_trans(trans),
                m, n, alpha, A, lda, x, incx, beta, y, incy);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv(cublasHandle_t handle,
                            cublasOperation_t trans,
                            int m, int n,
                            const cuDoubleComplex* alpha,
                            const cuDoubleComplex* A, int lda,
                            const cuDoubleComplex* x, int incx,
                            const cuDoubleComplex* beta,
                            cuDoubleComplex* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;

    const cublasStatus_t sync_st = synchronize_handle_stream(handle);
    if (sync_st != CUBLAS_STATUS_SUCCESS) return sync_st;

    cblas_zgemv(CblasColMajor, cublas_to_cblas_trans(trans),
                m, n, alpha, A, lda, x, incx, beta, y, incy);
    return CUBLAS_STATUS_SUCCESS;
}

// ── Complex Hermitian operations (batch 6) ────────────────────────────────────

// Chemv / Zhemv — y = alpha * A * x + beta * y, A Hermitian n×n.
cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo,
                            int n, const cuComplex* alpha,
                            const cuComplex* A, int lda,
                            const cuComplex* x, int incx,
                            const cuComplex* beta,
                            cuComplex* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const cuComplex al = *alpha, be = *beta;
    for (int i = 0; i < n; ++i) {
        cuComplex dot = {0.0f, 0.0f};
        for (int j = 0; j < n; ++j)
            dot = cadd_f(dot, cmul_f(herm_elem_f(A, lda, i, j, upper), x[(size_t)j * incx]));
        y[(size_t)i * incy] = cadd_f(cmul_f(al, dot), cmul_f(be, y[(size_t)i * incy]));
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo,
                            int n, const cuDoubleComplex* alpha,
                            const cuDoubleComplex* A, int lda,
                            const cuDoubleComplex* x, int incx,
                            const cuDoubleComplex* beta,
                            cuDoubleComplex* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const cuDoubleComplex al = *alpha, be = *beta;
    for (int i = 0; i < n; ++i) {
        cuDoubleComplex dot = {0.0, 0.0};
        for (int j = 0; j < n; ++j)
            dot = cadd_d(dot, cmul_d(herm_elem_d(A, lda, i, j, upper), x[(size_t)j * incx]));
        y[(size_t)i * incy] = cadd_d(cmul_d(al, dot), cmul_d(be, y[(size_t)i * incy]));
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Cher / Zher — A = alpha * x * x^H + A.  alpha is real.
cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo,
                           int n, const float* alpha,
                           const cuComplex* x, int incx,
                           cuComplex* A, int lda) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const float av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            // A[i,j] += av * x[i] * conj(x[j])
            cuComplex xj_c = {x[(size_t)j * incx].x, -x[(size_t)j * incx].y};
            cuComplex prod = cmul_f(x[(size_t)i * incx], xj_c);
            A[i + (size_t)j * lda].x += av * prod.x;
            A[i + (size_t)j * lda].y += av * prod.y;
        }
        // Force diagonal imaginary to zero (Hermitian invariant).
        A[j + (size_t)j * lda].y = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo,
                           int n, const double* alpha,
                           const cuDoubleComplex* x, int incx,
                           cuDoubleComplex* A, int lda) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const double av = *alpha;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuDoubleComplex xj_c = {x[(size_t)j * incx].x, -x[(size_t)j * incx].y};
            cuDoubleComplex prod = cmul_d(x[(size_t)i * incx], xj_c);
            A[i + (size_t)j * lda].x += av * prod.x;
            A[i + (size_t)j * lda].y += av * prod.y;
        }
        A[j + (size_t)j * lda].y = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Cher2 / Zher2 — A = alpha * x * y^H + conj(alpha) * y * x^H + A.
cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo,
                            int n, const cuComplex* alpha,
                            const cuComplex* x, int incx,
                            const cuComplex* y, int incy,
                            cuComplex* A, int lda) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuComplex al = *alpha, al_c = {al.x, -al.y};
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            // al * x[i] * conj(y[j]) + conj(al) * y[i] * conj(x[j])
            cuComplex yj_c = {y[(size_t)j * incy].x, -y[(size_t)j * incy].y};
            cuComplex xj_c = {x[(size_t)j * incx].x, -x[(size_t)j * incx].y};
            cuComplex t1 = cmul_f(al,   cmul_f(x[(size_t)i * incx], yj_c));
            cuComplex t2 = cmul_f(al_c, cmul_f(y[(size_t)i * incy], xj_c));
            A[i + (size_t)j * lda] = cadd_f(A[i + (size_t)j * lda], cadd_f(t1, t2));
        }
        A[j + (size_t)j * lda].y = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo,
                            int n, const cuDoubleComplex* alpha,
                            const cuDoubleComplex* x, int incx,
                            const cuDoubleComplex* y, int incy,
                            cuDoubleComplex* A, int lda) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || alpha == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuDoubleComplex al = *alpha, al_c = {al.x, -al.y};
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuDoubleComplex yj_c = {y[(size_t)j * incy].x, -y[(size_t)j * incy].y};
            cuDoubleComplex xj_c = {x[(size_t)j * incx].x, -x[(size_t)j * incx].y};
            cuDoubleComplex t1 = cmul_d(al,   cmul_d(x[(size_t)i * incx], yj_c));
            cuDoubleComplex t2 = cmul_d(al_c, cmul_d(y[(size_t)i * incy], xj_c));
            A[i + (size_t)j * lda] = cadd_d(A[i + (size_t)j * lda], cadd_d(t1, t2));
        }
        A[j + (size_t)j * lda].y = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Cherk / Zherk — C = alpha * op(A) * op(A)^H + beta * C.  alpha, beta real.
// trans=N: op(A)=A (n×k); trans=C: op(A)=A^H (k×n → result n×n).
cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const float* alpha, const cuComplex* A, int lda,
                            const float* beta,  cuComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || k < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const float av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuComplex sum = {0.0f, 0.0f};
            for (int l = 0; l < k; ++l) {
                // no_trans: A is n×k → ai = A[i,l], aj = A[j,l]
                const cuComplex ai = no_trans ? A[i + (size_t)l * lda]
                                              : cuComplex{A[l + (size_t)i * lda].x, -A[l + (size_t)i * lda].y};
                const cuComplex aj_c = no_trans ? cuComplex{A[j + (size_t)l * lda].x, -A[j + (size_t)l * lda].y}
                                                : A[l + (size_t)j * lda];
                sum = cadd_f(sum, cmul_f(ai, aj_c));
            }
            cuComplex& cij = C[i + (size_t)j * ldc];
            cij.x = av * sum.x + bv * cij.x;
            cij.y = av * sum.y + bv * cij.y;
        }
        // Force diagonal imaginary to zero.
        C[j + (size_t)j * ldc].y = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans,
                            int n, int k,
                            const double* alpha, const cuDoubleComplex* A, int lda,
                            const double* beta,  cuDoubleComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || k < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const double av = *alpha, bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuDoubleComplex sum = {0.0, 0.0};
            for (int l = 0; l < k; ++l) {
                const cuDoubleComplex ai = no_trans ? A[i + (size_t)l * lda]
                                                    : cuDoubleComplex{A[l + (size_t)i * lda].x, -A[l + (size_t)i * lda].y};
                const cuDoubleComplex aj_c = no_trans ? cuDoubleComplex{A[j + (size_t)l * lda].x, -A[j + (size_t)l * lda].y}
                                                      : A[l + (size_t)j * lda];
                sum = cadd_d(sum, cmul_d(ai, aj_c));
            }
            cuDoubleComplex& cij = C[i + (size_t)j * ldc];
            cij.x = av * sum.x + bv * cij.x;
            cij.y = av * sum.y + bv * cij.y;
        }
        C[j + (size_t)j * ldc].y = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Cher2k / Zher2k — C = alpha * op(A) * op(B)^H + conj(alpha) * op(B) * op(A)^H + beta * C.
// beta is real.
cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const cuComplex* alpha,
                             const cuComplex* A, int lda,
                             const cuComplex* B, int ldb,
                             const float* beta, cuComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || k < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuComplex al = *alpha, al_c = {al.x, -al.y};
    const float bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuComplex s1 = {0, 0}, s2 = {0, 0};
            for (int l = 0; l < k; ++l) {
                const cuComplex ai = no_trans ? A[i + (size_t)l * lda]
                                              : cuComplex{A[l + (size_t)i * lda].x, -A[l + (size_t)i * lda].y};
                const cuComplex bj_c = no_trans ? cuComplex{B[j + (size_t)l * ldb].x, -B[j + (size_t)l * ldb].y}
                                                : B[l + (size_t)j * ldb];
                const cuComplex bi = no_trans ? B[i + (size_t)l * ldb]
                                              : cuComplex{B[l + (size_t)i * ldb].x, -B[l + (size_t)i * ldb].y};
                const cuComplex aj_c = no_trans ? cuComplex{A[j + (size_t)l * lda].x, -A[j + (size_t)l * lda].y}
                                                : A[l + (size_t)j * lda];
                s1 = cadd_f(s1, cmul_f(ai, bj_c));
                s2 = cadd_f(s2, cmul_f(bi, aj_c));
            }
            cuComplex update = cadd_f(cmul_f(al, s1), cmul_f(al_c, s2));
            cuComplex& cij = C[i + (size_t)j * ldc];
            cij.x = update.x + bv * cij.x;
            cij.y = update.y + bv * cij.y;
        }
        C[j + (size_t)j * ldc].y = 0.0f;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans,
                             int n, int k,
                             const cuDoubleComplex* alpha,
                             const cuDoubleComplex* A, int lda,
                             const cuDoubleComplex* B, int ldb,
                             const double* beta, cuDoubleComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || !is_valid_operation(trans)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n < 0 || k < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || k == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuDoubleComplex al = *alpha, al_c = {al.x, -al.y};
    const double bv = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool no_trans = (trans == CUBLAS_OP_N);
    for (int j = 0; j < n; ++j) {
        const int i_lo = upper ? 0 : j;
        const int i_hi = upper ? j : n - 1;
        for (int i = i_lo; i <= i_hi; ++i) {
            cuDoubleComplex s1 = {0, 0}, s2 = {0, 0};
            for (int l = 0; l < k; ++l) {
                const cuDoubleComplex ai = no_trans ? A[i + (size_t)l * lda]
                                                    : cuDoubleComplex{A[l + (size_t)i * lda].x, -A[l + (size_t)i * lda].y};
                const cuDoubleComplex bj_c = no_trans ? cuDoubleComplex{B[j + (size_t)l * ldb].x, -B[j + (size_t)l * ldb].y}
                                                      : B[l + (size_t)j * ldb];
                const cuDoubleComplex bi = no_trans ? B[i + (size_t)l * ldb]
                                                    : cuDoubleComplex{B[l + (size_t)i * ldb].x, -B[l + (size_t)i * ldb].y};
                const cuDoubleComplex aj_c = no_trans ? cuDoubleComplex{A[j + (size_t)l * lda].x, -A[j + (size_t)l * lda].y}
                                                      : A[l + (size_t)j * lda];
                s1 = cadd_d(s1, cmul_d(ai, bj_c));
                s2 = cadd_d(s2, cmul_d(bi, aj_c));
            }
            cuDoubleComplex update = cadd_d(cmul_d(al, s1), cmul_d(al_c, s2));
            cuDoubleComplex& cij = C[i + (size_t)j * ldc];
            cij.x = update.x + bv * cij.x;
            cij.y = update.y + bv * cij.y;
        }
        C[j + (size_t)j * ldc].y = 0.0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Chemm / Zhemm — C = alpha * A * B + beta * C (or B * A), A Hermitian.
cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                            int m, int n,
                            const cuComplex* alpha,
                            const cuComplex* A, int lda,
                            const cuComplex* B, int ldb,
                            const cuComplex* beta,
                            cuComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuComplex al = *alpha, be = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuComplex sum = {0.0f, 0.0f};
            for (int l = 0; l < ka; ++l) {
                const cuComplex h = left ? herm_elem_f(A, lda, i, l, upper)
                                         : herm_elem_f(A, lda, l, j, upper);
                const cuComplex b = left ? B[l + (size_t)j * ldb]
                                         : B[i + (size_t)l * ldb];
                sum = cadd_f(sum, cmul_f(h, b));
            }
            C[i + (size_t)j * ldc] = cadd_f(cmul_f(al, sum), cmul_f(be, C[i + (size_t)j * ldc]));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                            int m, int n,
                            const cuDoubleComplex* alpha,
                            const cuDoubleComplex* A, int lda,
                            const cuDoubleComplex* B, int ldb,
                            const cuDoubleComplex* beta,
                            cuDoubleComplex* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || alpha == nullptr || beta == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    const cuDoubleComplex al = *alpha, be = *beta;
    const bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    const bool left  = (side == CUBLAS_SIDE_LEFT);
    const int ka = left ? m : n;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuDoubleComplex sum = {0.0, 0.0};
            for (int l = 0; l < ka; ++l) {
                const cuDoubleComplex h = left ? herm_elem_d(A, lda, i, l, upper)
                                               : herm_elem_d(A, lda, l, j, upper);
                const cuDoubleComplex b = left ? B[l + (size_t)j * ldb]
                                               : B[i + (size_t)l * ldb];
                sum = cadd_d(sum, cmul_d(h, b));
            }
            C[i + (size_t)j * ldc] = cadd_d(cmul_d(al, sum), cmul_d(be, C[i + (size_t)j * ldc]));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// CgemmStridedBatched / ZgemmStridedBatched — batched complex GEMM with stride offsets.
// Iterates over batchCount instances, each offset by strideA/B/C elements.
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const cuComplex* alpha,
                                          const cuComplex* A, int lda, long long int strideA,
                                          const cuComplex* B, int ldb, long long int strideB,
                                          const cuComplex* beta,
                                          cuComplex* C, int ldc, long long int strideC,
                                          int batchCount) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (batchCount == 0 || m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    for (int b = 0; b < batchCount; ++b) {
        cblas_cgemm(CblasColMajor,
                    cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                    m, n, k,
                    alpha, A + (size_t)b * strideA, lda,
                    B + (size_t)b * strideB, ldb,
                    beta, C + (size_t)b * strideC, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t handle,
                                          cublasOperation_t transa, cublasOperation_t transb,
                                          int m, int n, int k,
                                          const cuDoubleComplex* alpha,
                                          const cuDoubleComplex* A, int lda, long long int strideA,
                                          const cuDoubleComplex* B, int ldb, long long int strideB,
                                          const cuDoubleComplex* beta,
                                          cuDoubleComplex* C, int ldc, long long int strideC,
                                          int batchCount) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0 || batchCount < 0) return CUBLAS_STATUS_INVALID_VALUE;
    if (batchCount == 0 || m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (alpha == nullptr || beta == nullptr || A == nullptr || B == nullptr || C == nullptr)
        return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t ss = synchronize_handle_stream(handle);
    if (ss != CUBLAS_STATUS_SUCCESS) return ss;
    for (int b = 0; b < batchCount; ++b) {
        cblas_zgemm(CblasColMajor,
                    cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                    m, n, k,
                    alpha, A + (size_t)b * strideA, lda,
                    B + (size_t)b * strideB, ldb,
                    beta, C + (size_t)b * strideC, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // extern "C"

// ── CuPy surface: complex level-1/2/3, geam/dgmm, packed/banded, LU helpers ───
//
// Everything below runs on the CPU against unified-memory pointers after the
// handle's stream has drained, like the complex GEMM above. The arithmetic is
// written once over float/double/cuComplex/cuDoubleComplex through Ops<T>.

namespace {

template <class T>
struct Ops;

#define CUMETAL_REAL_OPS(T)                                                       \
    template <>                                                                    \
    struct Ops<T> {                                                                \
        using Real = T;                                                            \
        static T zero() { return 0; }                                              \
        static T one() { return 1; }                                               \
        static T make(double r, double) { return static_cast<T>(r); }              \
        static T mul(T a, T b) { return a * b; }                                   \
        static T add(T a, T b) { return a + b; }                                   \
        static T sub(T a, T b) { return a - b; }                                   \
        static T div(T a, T b) { return a / b; }                                   \
        static T conj(T a) { return a; }                                           \
        static double re(T a) { return a; }                                        \
        static double im(T) { return 0.0; }                                        \
        static double abs1(T a) { return std::fabs(a); }                           \
        static bool is_zero(T a) { return a == 0; }                                \
    }

#define CUMETAL_COMPLEX_OPS(T, R)                                                  \
    template <>                                                                    \
    struct Ops<T> {                                                                \
        using Real = R;                                                            \
        static T zero() { return T{0, 0}; }                                        \
        static T one() { return T{1, 0}; }                                         \
        static T make(double r, double i) {                                        \
            return T{static_cast<R>(r), static_cast<R>(i)};                        \
        }                                                                          \
        static T mul(T a, T b) {                                                   \
            return T{a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};                \
        }                                                                          \
        static T add(T a, T b) { return T{a.x + b.x, a.y + b.y}; }                 \
        static T sub(T a, T b) { return T{a.x - b.x, a.y - b.y}; }                 \
        static T div(T a, T b) {                                                   \
            const R d = b.x * b.x + b.y * b.y;                                     \
            return T{(a.x * b.x + a.y * b.y) / d, (a.y * b.x - a.x * b.y) / d};    \
        }                                                                          \
        static T conj(T a) { return T{a.x, -a.y}; }                                \
        static double re(T a) { return a.x; }                                      \
        static double im(T a) { return a.y; }                                      \
        static double abs1(T a) { return std::fabs(a.x) + std::fabs(a.y); }        \
        static bool is_zero(T a) { return a.x == 0 && a.y == 0; }                  \
    }

CUMETAL_REAL_OPS(float);
CUMETAL_REAL_OPS(double);
CUMETAL_COMPLEX_OPS(cuComplex, float);
CUMETAL_COMPLEX_OPS(cuDoubleComplex, double);
#undef CUMETAL_REAL_OPS
#undef CUMETAL_COMPLEX_OPS

// Scalar inputs honor the handle's pointer mode; in device mode the value is
// fetched through the tracked allocation, in host mode it is read in place.
template <class S>
bool load_scalar(cublasHandle_t handle, const S* p, S* out) {
    if (p == nullptr) return false;
    cublasPointerMode_t mode;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        mode = handle->pointer_mode;
    }
    if (mode == CUBLAS_POINTER_MODE_DEVICE) {
        cumetal::rt::AllocationTable::ResolvedAllocation resolved;
        if (cumetal::rt::resolve_allocation_for_pointer(p, &resolved)) {
            if (resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
                resolved.remaining_size < sizeof(S)) {
                return false;
            }
            std::memcpy(out,
                        static_cast<const unsigned char*>(resolved.buffer->contents()) +
                            resolved.offset,
                        sizeof(S));
            return true;
        }
    }
    *out = *p;
    return true;
}

// Scalar outputs: same rule, returning where to store the result.
template <class S>
S* result_slot(cublasHandle_t handle, S* p) {
    if (p == nullptr) return nullptr;
    cublasPointerMode_t mode;
    {
        std::lock_guard<std::mutex> lock(handle->mutex);
        mode = handle->pointer_mode;
    }
    if (mode == CUBLAS_POINTER_MODE_DEVICE) {
        cumetal::rt::AllocationTable::ResolvedAllocation resolved;
        if (cumetal::rt::resolve_allocation_for_pointer(p, &resolved)) {
            if (resolved.buffer == nullptr || resolved.buffer->contents() == nullptr ||
                resolved.remaining_size < sizeof(S)) {
                return nullptr;
            }
            return reinterpret_cast<S*>(
                static_cast<unsigned char*>(resolved.buffer->contents()) + resolved.offset);
        }
    }
    return p;
}

// BLAS element index for a vector of length n with stride inc (inc < 0 walks
// backwards from the end, as in reference BLAS).
inline std::size_t vec_index(int i, int n, int inc) {
    return inc > 0 ? static_cast<std::size_t>(i) * inc
                   : static_cast<std::size_t>(n - 1 - i) * (-static_cast<long long>(inc));
}

// ── level 1 ──

template <class T>
cublasStatus_t axpy_impl(cublasHandle_t handle, int n, const T* alpha, const T* x, int incx,
                         T* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || incx == 0 || incy == 0) return CUBLAS_STATUS_INVALID_VALUE;
    T a;
    if (!load_scalar(handle, alpha, &a)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if (Ops<T>::is_zero(a)) return CUBLAS_STATUS_SUCCESS;
    for (int i = 0; i < n; ++i) {
        T& yi = y[vec_index(i, n, incy)];
        yi = Ops<T>::add(yi, Ops<T>::mul(a, x[vec_index(i, n, incx)]));
    }
    return CUBLAS_STATUS_SUCCESS;
}

// x *= alpha, where alpha is either the element type or its real type.
template <class T, class A>
cublasStatus_t scal_impl(cublasHandle_t handle, int n, const A* alpha, T* x, int incx) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || incx <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    A a;
    if (!load_scalar(handle, alpha, &a)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    for (int i = 0; i < n; ++i) {
        T& xi = x[static_cast<std::size_t>(i) * incx];
        if constexpr (std::is_same<A, T>::value) {
            xi = Ops<T>::mul(a, xi);
        } else {
            xi = Ops<T>::make(static_cast<double>(a) * Ops<T>::re(xi),
                              static_cast<double>(a) * Ops<T>::im(xi));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t dot_impl(cublasHandle_t handle, int n, const T* x, int incx, const T* y, int incy,
                        T* result, bool conj_x) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    T* out = result_slot(handle, result);
    if (n < 0 || incx == 0 || incy == 0 || out == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) {
        *out = Ops<T>::zero();
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    double sr = 0.0, si = 0.0;
    for (int i = 0; i < n; ++i) {
        T xv = x[vec_index(i, n, incx)];
        if (conj_x) xv = Ops<T>::conj(xv);
        const T yv = y[vec_index(i, n, incy)];
        sr += Ops<T>::re(xv) * Ops<T>::re(yv) - Ops<T>::im(xv) * Ops<T>::im(yv);
        si += Ops<T>::re(xv) * Ops<T>::im(yv) + Ops<T>::im(xv) * Ops<T>::re(yv);
    }
    *out = Ops<T>::make(sr, si);
    return CUBLAS_STATUS_SUCCESS;
}

// kind: 0 = asum (|re|+|im|), 1 = nrm2.
template <class T>
cublasStatus_t norm_impl(cublasHandle_t handle, int n, const T* x, int incx,
                         typename Ops<T>::Real* result, int kind) {
    using R = typename Ops<T>::Real;
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    R* out = result_slot(handle, result);
    if (n < 0 || incx <= 0 || out == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) {
        *out = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if (kind == 0) {
        double sum = 0.0;
        for (int i = 0; i < n; ++i) sum += Ops<T>::abs1(x[static_cast<std::size_t>(i) * incx]);
        *out = static_cast<R>(sum);
        return CUBLAS_STATUS_SUCCESS;
    }
    // Scaled sum of squares so large FP64 entries do not overflow.
    double scale = 0.0;
    for (int i = 0; i < n; ++i) {
        const T v = x[static_cast<std::size_t>(i) * incx];
        scale = std::max(scale, std::max(std::fabs(Ops<T>::re(v)), std::fabs(Ops<T>::im(v))));
    }
    if (scale == 0.0 || !std::isfinite(scale)) {
        *out = static_cast<R>(scale);
        return CUBLAS_STATUS_SUCCESS;
    }
    double ssq = 0.0;
    for (int i = 0; i < n; ++i) {
        const T v = x[static_cast<std::size_t>(i) * incx];
        const double r = Ops<T>::re(v) / scale, m = Ops<T>::im(v) / scale;
        ssq += r * r + m * m;
    }
    *out = static_cast<R>(scale * std::sqrt(ssq));
    return CUBLAS_STATUS_SUCCESS;
}

// First index (1-based) of the largest (find_max) or smallest |re|+|im|.
template <class T>
cublasStatus_t iamaxmin_impl(cublasHandle_t handle, int n, const T* x, int incx, int* result,
                             bool find_max) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    int* out = result_slot(handle, result);
    if (n < 0 || incx <= 0 || out == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0) {
        *out = 0;
        return CUBLAS_STATUS_SUCCESS;
    }
    if (x == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    int best = 0;
    double best_value = Ops<T>::abs1(x[0]);
    for (int i = 1; i < n; ++i) {
        const double v = Ops<T>::abs1(x[static_cast<std::size_t>(i) * incx]);
        if (find_max ? (v > best_value) : (v < best_value)) {
            best_value = v;
            best = i;
        }
    }
    *out = best + 1;
    return CUBLAS_STATUS_SUCCESS;
}

// ── level 2 ──

template <class T>
cublasStatus_t ger_impl(cublasHandle_t handle, int m, int n, const T* alpha, const T* x, int incx,
                        const T* y, int incy, T* A, int lda, bool conj_y) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || incx == 0 || incy == 0 || lda < std::max(1, m)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T a;
    if (!load_scalar(handle, alpha, &a)) return CUBLAS_STATUS_INVALID_VALUE;
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (x == nullptr || y == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if (Ops<T>::is_zero(a)) return CUBLAS_STATUS_SUCCESS;
    for (int j = 0; j < n; ++j) {
        T yj = y[vec_index(j, n, incy)];
        if (conj_y) yj = Ops<T>::conj(yj);
        const T t = Ops<T>::mul(a, yj);
        for (int i = 0; i < m; ++i) {
            T& aij = A[static_cast<std::size_t>(j) * lda + i];
            aij = Ops<T>::add(aij, Ops<T>::mul(x[vec_index(i, m, incx)], t));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── extensions ──

template <class T>
T op_element(const T* A, int ld, cublasOperation_t op, int row, int col) {
    if (op == CUBLAS_OP_N) return A[static_cast<std::size_t>(col) * ld + row];
    const T v = A[static_cast<std::size_t>(row) * ld + col];
    return op == CUBLAS_OP_C ? Ops<T>::conj(v) : v;
}

template <class T>
cublasStatus_t geam_impl(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                         int m, int n, const T* alpha, const T* A, int lda, const T* beta,
                         const T* B, int ldb, T* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb) || m < 0 || n < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T a, b;
    if (!load_scalar(handle, alpha, &a) || !load_scalar(handle, beta, &b)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, transa == CUBLAS_OP_N ? m : n) ||
        ldb < std::max(1, transb == CUBLAS_OP_N ? m : n) || ldc < std::max(1, m)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    // In-place use is only well defined for the untransposed, same-layout case.
    if ((C == A && (transa != CUBLAS_OP_N || lda != ldc)) ||
        (C == B && (transb != CUBLAS_OP_N || ldb != ldc))) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            C[static_cast<std::size_t>(j) * ldc + i] =
                Ops<T>::add(Ops<T>::mul(a, op_element(A, lda, transa, i, j)),
                            Ops<T>::mul(b, op_element(B, ldb, transb, i, j)));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t dgmm_impl(cublasHandle_t handle, cublasSideMode_t mode, int m, int n, const T* A,
                         int lda, const T* x, int incx, T* C, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if ((mode != CUBLAS_SIDE_LEFT && mode != CUBLAS_SIDE_RIGHT) || m < 0 || n < 0 || incx == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m) || ldc < std::max(1, m)) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    const int xlen = (mode == CUBLAS_SIDE_LEFT) ? m : n;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            const T xv = x[vec_index(mode == CUBLAS_SIDE_LEFT ? i : j, xlen, incx)];
            C[static_cast<std::size_t>(j) * ldc + i] =
                Ops<T>::mul(A[static_cast<std::size_t>(j) * lda + i], xv);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t tpttr_impl(cublasHandle_t handle, cublasFillMode_t uplo, int n, const T* AP, T* A,
                          int lda) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || lda < std::max(1, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (AP == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    std::size_t k = 0;
    for (int j = 0; j < n; ++j) {
        const int lo = (uplo == CUBLAS_FILL_MODE_UPPER) ? 0 : j;
        const int hi = (uplo == CUBLAS_FILL_MODE_UPPER) ? j : n - 1;
        for (int i = lo; i <= hi; ++i) A[static_cast<std::size_t>(j) * lda + i] = AP[k++];
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t trttp_impl(cublasHandle_t handle, cublasFillMode_t uplo, int n, const T* A, int lda,
                          T* AP) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || lda < std::max(1, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (AP == nullptr || A == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    std::size_t k = 0;
    for (int j = 0; j < n; ++j) {
        const int lo = (uplo == CUBLAS_FILL_MODE_UPPER) ? 0 : j;
        const int hi = (uplo == CUBLAS_FILL_MODE_UPPER) ? j : n - 1;
        for (int i = lo; i <= hi; ++i) AP[k++] = A[static_cast<std::size_t>(j) * lda + i];
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t sbmv_impl(cublasHandle_t handle, cublasFillMode_t uplo, int n, int k, const T* alpha,
                         const T* A, int lda, const T* x, int incx, const T* beta, T* y, int incy) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_fill_mode(uplo) || n < 0 || k < 0 || lda < k + 1 || incx == 0 || incy == 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T a, b;
    if (!load_scalar(handle, alpha, &a) || !load_scalar(handle, beta, &b)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || x == nullptr || y == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    // Band storage: upper keeps A(i,j) at A[(k + i - j) + j*lda] for j-k <= i <= j;
    // lower keeps it at A[(i - j) + j*lda] for j <= i <= j+k.
    std::vector<double> acc(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
        const double xj = x[vec_index(j, n, incx)];
        const int lo = (uplo == CUBLAS_FILL_MODE_UPPER) ? std::max(0, j - k) : j;
        const int hi = (uplo == CUBLAS_FILL_MODE_UPPER) ? j : std::min(n - 1, j + k);
        for (int i = lo; i <= hi; ++i) {
            const std::size_t off = (uplo == CUBLAS_FILL_MODE_UPPER)
                                        ? static_cast<std::size_t>(k + i - j)
                                        : static_cast<std::size_t>(i - j);
            const double v = A[off + static_cast<std::size_t>(j) * lda];
            acc[i] += v * xj;
            if (i != j) acc[j] += v * x[vec_index(i, n, incx)];
        }
    }
    for (int i = 0; i < n; ++i) {
        T& yi = y[vec_index(i, n, incy)];
        const T scaled = (b == static_cast<T>(0)) ? static_cast<T>(0) : b * yi;
        yi = scaled + a * static_cast<T>(acc[i]);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── complex level 3 via Accelerate ──

inline CBLAS_UPLO to_cblas_uplo(cublasFillMode_t u) {
    return u == CUBLAS_FILL_MODE_UPPER ? CblasUpper : CblasLower;
}

inline cublasStatus_t check_trsm_args(cublasSideMode_t side, cublasFillMode_t uplo,
                                      cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                                      int lda, int ldb) {
    if ((side != CUBLAS_SIDE_LEFT && side != CUBLAS_SIDE_RIGHT) || !is_valid_fill_mode(uplo) ||
        !is_valid_operation(trans) ||
        (diag != CUBLAS_DIAG_NON_UNIT && diag != CUBLAS_DIAG_UNIT) || m < 0 || n < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const int adim = (side == CUBLAS_SIDE_LEFT) ? m : n;
    if (lda < std::max(1, adim) || ldb < std::max(1, m)) return CUBLAS_STATUS_INVALID_VALUE;
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t ctrsm_core(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag, int m, int n, T a,
                          const T* A, int lda, T* B, int ldb) {
    if (m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    const CBLAS_SIDE cside = (side == CUBLAS_SIDE_LEFT) ? CblasLeft : CblasRight;
    const CBLAS_DIAG cdiag = (diag == CUBLAS_DIAG_UNIT) ? CblasUnit : CblasNonUnit;
    if constexpr (std::is_same<T, cuComplex>::value) {
        cblas_ctrsm(CblasColMajor, cside, to_cblas_uplo(uplo), cublas_to_cblas_trans(trans), cdiag,
                    m, n, &a, A, lda, B, ldb);
    } else {
        cblas_ztrsm(CblasColMajor, cside, to_cblas_uplo(uplo), cublas_to_cblas_trans(trans), cdiag,
                    m, n, &a, A, lda, B, ldb);
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t ctrsm_impl(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
                          cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                          const T* alpha, const T* A, int lda, T* B, int ldb) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    const cublasStatus_t chk = check_trsm_args(side, uplo, trans, diag, m, n, lda, ldb);
    if (chk != CUBLAS_STATUS_SUCCESS) return chk;
    T a;
    if (!load_scalar(handle, alpha, &a)) return CUBLAS_STATUS_INVALID_VALUE;
    return ctrsm_core(handle, side, uplo, trans, diag, m, n, a, A, lda, B, ldb);
}

template <class T>
cublasStatus_t csyrk_impl(cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
                          int n, int k, const T* alpha, const T* A, int lda, const T* beta, T* C,
                          int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    // Complex SYRK is defined for N and T only; C belongs to HERK.
    if (!is_valid_fill_mode(uplo) || (trans != CUBLAS_OP_N && trans != CUBLAS_OP_T) || n < 0 ||
        k < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (lda < std::max(1, trans == CUBLAS_OP_N ? n : k) || ldc < std::max(1, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T a, b;
    if (!load_scalar(handle, alpha, &a) || !load_scalar(handle, beta, &b)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || C == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    if constexpr (std::is_same<T, cuComplex>::value) {
        cblas_csyrk(CblasColMajor, to_cblas_uplo(uplo), cublas_to_cblas_trans(trans), n, k, &a, A,
                    lda, &b, C, ldc);
    } else {
        cblas_zsyrk(CblasColMajor, to_cblas_uplo(uplo), cublas_to_cblas_trans(trans), n, k, &a, A,
                    lda, &b, C, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t trsm_batched_impl(cublasHandle_t handle, cublasSideMode_t side,
                                 cublasFillMode_t uplo, cublasOperation_t trans,
                                 cublasDiagType_t diag, int m, int n, const T* alpha,
                                 const T* const A[], int lda, T* const B[], int ldb,
                                 int batch_count) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    const cublasStatus_t chk = check_trsm_args(side, uplo, trans, diag, m, n, lda, ldb);
    if (chk != CUBLAS_STATUS_SUCCESS || batch_count < 0) return CUBLAS_STATUS_INVALID_VALUE;
    T a;
    if (!load_scalar(handle, alpha, &a)) return CUBLAS_STATUS_INVALID_VALUE;
    if (batch_count == 0) return CUBLAS_STATUS_SUCCESS;
    std::vector<void*> as, bs;
    if (!read_pointer_table(A, batch_count, &as) || !read_pointer_table(B, batch_count, &bs)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int i = 0; i < batch_count; ++i) {
        const cublasStatus_t s =
            ctrsm_core(handle, side, uplo, trans, diag, m, n, a, static_cast<const T*>(as[i]), lda,
                       static_cast<T*>(bs[i]), ldb);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── batched LU ──

// LU with partial pivoting (LAPACK getrf semantics: 1-based pivots, info = first
// zero U(k,k), elimination continues). ipiv == nullptr factors without pivoting.
template <class T>
int lu_factor(int n, T* a, int lda, int* ipiv) {
    int info = 0;
    auto at = [&](int i, int j) -> T& { return a[static_cast<std::size_t>(j) * lda + i]; };
    for (int k = 0; k < n; ++k) {
        int p = k;
        if (ipiv != nullptr) {
            double best = Ops<T>::abs1(at(k, k));
            for (int i = k + 1; i < n; ++i) {
                const double v = Ops<T>::abs1(at(i, k));
                if (v > best) {
                    best = v;
                    p = i;
                }
            }
            ipiv[k] = p + 1;
        }
        if (Ops<T>::is_zero(at(p, k))) {
            if (info == 0) info = k + 1;
            continue;
        }
        if (p != k) {
            for (int j = 0; j < n; ++j) std::swap(at(k, j), at(p, j));
        }
        const T pivot = at(k, k);
        for (int i = k + 1; i < n; ++i) at(i, k) = Ops<T>::div(at(i, k), pivot);
        for (int j = k + 1; j < n; ++j) {
            const T u = at(k, j);
            for (int i = k + 1; i < n; ++i) {
                at(i, j) = Ops<T>::sub(at(i, j), Ops<T>::mul(at(i, k), u));
            }
        }
    }
    return info;
}

// Solve op(A) X = B in place given the LU factors and pivots from lu_factor.
template <class T>
void lu_solve(cublasOperation_t trans, int n, int nrhs, const T* lu, int lda, const int* ipiv,
              T* b, int ldb) {
    auto L = [&](int i, int j) { return lu[static_cast<std::size_t>(j) * lda + i]; };
    auto swap_rows = [&](int col, int from, int to, int step) {
        for (int i = from; i != to; i += step) {
            const int p = (ipiv != nullptr ? ipiv[i] : i + 1) - 1;
            if (p != i) std::swap(b[static_cast<std::size_t>(col) * ldb + i],
                                  b[static_cast<std::size_t>(col) * ldb + p]);
        }
    };
    for (int c = 0; c < nrhs; ++c) {
        T* x = b + static_cast<std::size_t>(c) * ldb;
        if (trans == CUBLAS_OP_N) {
            swap_rows(c, 0, n, 1);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < i; ++j) x[i] = Ops<T>::sub(x[i], Ops<T>::mul(L(i, j), x[j]));
            for (int i = n - 1; i >= 0; --i) {
                for (int j = i + 1; j < n; ++j) x[i] = Ops<T>::sub(x[i], Ops<T>::mul(L(i, j), x[j]));
                x[i] = Ops<T>::div(x[i], L(i, i));
            }
        } else {
            const bool cj = (trans == CUBLAS_OP_C);
            auto E = [&](int i, int j) { return cj ? Ops<T>::conj(L(i, j)) : L(i, j); };
            // (P^T L U)^T = U^T L^T P : solve U^T, then L^T, then undo the swaps.
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < i; ++j) x[i] = Ops<T>::sub(x[i], Ops<T>::mul(E(j, i), x[j]));
                x[i] = Ops<T>::div(x[i], E(i, i));
            }
            for (int i = n - 1; i >= 0; --i)
                for (int j = i + 1; j < n; ++j) x[i] = Ops<T>::sub(x[i], Ops<T>::mul(E(j, i), x[j]));
            swap_rows(c, n - 1, -1, -1);
        }
    }
}

template <class T>
cublasStatus_t getrf_batched_any(cublasHandle_t handle, int n, T* const A[], int lda, int* P,
                                 int* info, int batch) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || batch < 0 || lda < std::max(1, n)) return CUBLAS_STATUS_INVALID_VALUE;
    if (n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || info == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    std::vector<void*> mats;
    if (!read_pointer_table(A, batch, &mats)) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batch; ++b) {
        if (mats[b] == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
        info[b] = lu_factor(n, static_cast<T*>(mats[b]), lda,
                            P != nullptr ? P + static_cast<std::size_t>(b) * n : nullptr);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// info is a host array here, as in cuBLAS getrsBatched.
template <class T>
cublasStatus_t getrs_batched_any(cublasHandle_t handle, cublasOperation_t trans, int n, int nrhs,
                                 const T* const A[], int lda, const int* ipiv, T* const B[],
                                 int ldb, int* info, int batch) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(trans) || n < 0 || nrhs < 0 || batch < 0 || lda < std::max(1, n) ||
        ldb < std::max(1, n) || info == nullptr) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    *info = 0;
    if (n == 0 || nrhs == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || B == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    std::vector<void*> as, bs;
    if (!read_pointer_table(A, batch, &as) || !read_pointer_table(B, batch, &bs)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int b = 0; b < batch; ++b) {
        if (as[b] == nullptr || bs[b] == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
        lu_solve(trans, n, nrhs, static_cast<const T*>(as[b]), lda,
                 ipiv != nullptr ? ipiv + static_cast<std::size_t>(b) * n : nullptr,
                 static_cast<T*>(bs[b]), ldb);
    }
    return CUBLAS_STATUS_SUCCESS;
}

// C = inv(A) from getrf factors by solving LU X = I; info[b] = k > 0 marks a
// singular matrix (U(k,k) == 0), whose C is left untouched.
template <class T>
cublasStatus_t getri_batched_any(cublasHandle_t handle, int n, const T* const A[], int lda,
                                 const int* P, T* const C[], int ldc, int* info, int batch) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (n < 0 || batch < 0 || lda < std::max(1, n) || ldc < std::max(1, n)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (n == 0 || batch == 0) return CUBLAS_STATUS_SUCCESS;
    if (A == nullptr || C == nullptr || info == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    std::vector<void*> as, cs;
    if (!read_pointer_table(A, batch, &as) || !read_pointer_table(C, batch, &cs)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    for (int b = 0; b < batch; ++b) {
        if (as[b] == nullptr || cs[b] == nullptr) return CUBLAS_STATUS_INVALID_VALUE;
        const T* lu = static_cast<const T*>(as[b]);
        T* out = static_cast<T*>(cs[b]);
        int singular = 0;
        for (int k = 0; k < n && singular == 0; ++k) {
            if (Ops<T>::is_zero(lu[static_cast<std::size_t>(k) * lda + k])) singular = k + 1;
        }
        info[b] = singular;
        if (singular != 0) continue;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                out[static_cast<std::size_t>(j) * ldc + i] = (i == j) ? Ops<T>::one() : Ops<T>::zero();
        lu_solve(CUBLAS_OP_N, n, n, lu, lda,
                 P != nullptr ? P + static_cast<std::size_t>(b) * n : nullptr, out, ldc);
    }
    return CUBLAS_STATUS_SUCCESS;
}

template <class T>
cublasStatus_t gemm_batched_any(cublasHandle_t handle, cublasOperation_t transa,
                                cublasOperation_t transb, int m, int n, int k, const T* alpha,
                                const T* const A[], int lda, const T* const B[], int ldb,
                                const T* beta, T* const C[], int ldc, int batch) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    if (!is_valid_operation(transa) || !is_valid_operation(transb) || m < 0 || n < 0 || k < 0 ||
        batch < 0) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    T a, b;
    if (!load_scalar(handle, alpha, &a) || !load_scalar(handle, beta, &b)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (batch == 0 || m == 0 || n == 0) return CUBLAS_STATUS_SUCCESS;
    if (lda < std::max(1, transa == CUBLAS_OP_N ? m : k) ||
        ldb < std::max(1, transb == CUBLAS_OP_N ? k : n) || ldc < std::max(1, m)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    std::vector<void*> as, bs, cs;
    if (!read_pointer_table(A, batch, &as) || !read_pointer_table(B, batch, &bs) ||
        !read_pointer_table(C, batch, &cs)) {
        return CUBLAS_STATUS_INVALID_VALUE;
    }
    const cublasStatus_t st = synchronize_handle_stream(handle);
    if (st != CUBLAS_STATUS_SUCCESS) return st;
    for (int i = 0; i < batch; ++i) {
        if (as[i] == nullptr || bs[i] == nullptr || cs[i] == nullptr) {
            return CUBLAS_STATUS_INVALID_VALUE;
        }
        if constexpr (std::is_same<T, cuComplex>::value) {
            cblas_cgemm(CblasColMajor, cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                        m, n, k, &a, as[i], lda, bs[i], ldb, &b, cs[i], ldc);
        } else {
            cblas_zgemm(CblasColMajor, cublas_to_cblas_trans(transa), cublas_to_cblas_trans(transb),
                        m, n, k, &a, as[i], lda, bs[i], ldb, &b, cs[i], ldc);
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

#define CUMETAL_CX_LEVEL1(P, C, R)                                                                  \
    cublasStatus_t cublas##P##axpy(cublasHandle_t h, int n, const C* alpha, const C* x, int incx,  \
                                   C* y, int incy) {                                                \
        return axpy_impl(h, n, alpha, x, incx, y, incy);                                            \
    }                                                                                               \
    cublasStatus_t cublas##P##scal(cublasHandle_t h, int n, const C* alpha, C* x, int incx) {       \
        return scal_impl(h, n, alpha, x, incx);                                                     \
    }                                                                                               \
    cublasStatus_t cublas##P##dotu(cublasHandle_t h, int n, const C* x, int incx, const C* y,       \
                                   int incy, C* result) {                                           \
        return dot_impl(h, n, x, incx, y, incy, result, false);                                     \
    }                                                                                               \
    cublasStatus_t cublas##P##dotc(cublasHandle_t h, int n, const C* x, int incx, const C* y,       \
                                   int incy, C* result) {                                           \
        return dot_impl(h, n, x, incx, y, incy, result, true);                                      \
    }                                                                                               \
    cublasStatus_t cublas##P##geru(cublasHandle_t h, int m, int n, const C* alpha, const C* x,      \
                                   int incx, const C* y, int incy, C* A, int lda) {                 \
        return ger_impl(h, m, n, alpha, x, incx, y, incy, A, lda, false);                           \
    }                                                                                               \
    cublasStatus_t cublas##P##gerc(cublasHandle_t h, int m, int n, const C* alpha, const C* x,      \
                                   int incx, const C* y, int incy, C* A, int lda) {                 \
        return ger_impl(h, m, n, alpha, x, incx, y, incy, A, lda, true);                            \
    }                                                                                               \
    cublasStatus_t cublas##P##syrk(cublasHandle_t h, cublasFillMode_t uplo,                         \
                                   cublasOperation_t trans, int n, int k, const C* alpha,           \
                                   const C* A, int lda, const C* beta, C* Cm, int ldc) {            \
        return csyrk_impl(h, uplo, trans, n, k, alpha, A, lda, beta, Cm, ldc);                      \
    }                                                                                               \
    cublasStatus_t cublas##P##trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,  \
                                   cublasOperation_t trans, cublasDiagType_t diag, int m, int n,    \
                                   const C* alpha, const C* A, int lda, C* B, int ldb) {            \
        return ctrsm_impl(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);                 \
    }                                                                                               \
    cublasStatus_t cublas##P##trsmBatched(cublasHandle_t h, cublasSideMode_t side,                  \
                                          cublasFillMode_t uplo, cublasOperation_t trans,           \
                                          cublasDiagType_t diag, int m, int n, const C* alpha,      \
                                          const C* const A[], int lda, C* const B[], int ldb,       \
                                          int batchCount) {                                         \
        return trsm_batched_impl(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb,           \
                                 batchCount);                                                       \
    }                                                                                               \
    cublasStatus_t cublas##P##gemmBatched(cublasHandle_t h, cublasOperation_t ta,                   \
                                          cublasOperation_t tb, int m, int n, int k,                \
                                          const C* alpha, const C* const A[], int lda,              \
                                          const C* const B[], int ldb, const C* beta,               \
                                          C* const Cm[], int ldc, int batchCount) {                 \
        return gemm_batched_any(h, ta, tb, m, n, k, alpha, A, lda, B, ldb, beta, Cm, ldc,           \
                                batchCount);                                                        \
    }                                                                                               \
    cublasStatus_t cublas##P##getrfBatched(cublasHandle_t h, int n, C* const A[], int lda,          \
                                           int* piv, int* info, int batchSize) {                    \
        return getrf_batched_any(h, n, A, lda, piv, info, batchSize);                               \
    }

#define CUMETAL_ANY_LU_AND_EXT(P, T)                                                                \
    cublasStatus_t cublas##P##geam(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,    \
                                   int m, int n, const T* alpha, const T* A, int lda,               \
                                   const T* beta, const T* B, int ldb, T* Cm, int ldc) {            \
        return geam_impl(h, ta, tb, m, n, alpha, A, lda, beta, B, ldb, Cm, ldc);                    \
    }                                                                                               \
    cublasStatus_t cublas##P##dgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n,           \
                                   const T* A, int lda, const T* x, int incx, T* Cm, int ldc) {     \
        return dgmm_impl(h, mode, m, n, A, lda, x, incx, Cm, ldc);                                  \
    }                                                                                               \
    cublasStatus_t cublas##P##getrsBatched(cublasHandle_t h, cublasOperation_t trans, int n,        \
                                           int nrhs, const T* const A[], int lda, const int* ipiv,  \
                                           T* const B[], int ldb, int* info, int batchSize) {       \
        return getrs_batched_any(h, trans, n, nrhs, A, lda, ipiv, B, ldb, info, batchSize);         \
    }                                                                                               \
    cublasStatus_t cublas##P##getriBatched(cublasHandle_t h, int n, const T* const A[], int lda,    \
                                           const int* piv, T* const Cm[], int ldc, int* info,       \
                                           int batchSize) {                                         \
        return getri_batched_any(h, n, A, lda, piv, Cm, ldc, info, batchSize);                      \
    }

CUMETAL_CX_LEVEL1(C, cuComplex, float)
CUMETAL_CX_LEVEL1(Z, cuDoubleComplex, double)
CUMETAL_ANY_LU_AND_EXT(S, float)
CUMETAL_ANY_LU_AND_EXT(D, double)
CUMETAL_ANY_LU_AND_EXT(C, cuComplex)
CUMETAL_ANY_LU_AND_EXT(Z, cuDoubleComplex)

#undef CUMETAL_CX_LEVEL1
#undef CUMETAL_ANY_LU_AND_EXT

cublasStatus_t cublasIcamax(cublasHandle_t h, int n, const cuComplex* x, int incx, int* result) {
    return iamaxmin_impl(h, n, x, incx, result, true);
}
cublasStatus_t cublasIcamin(cublasHandle_t h, int n, const cuComplex* x, int incx, int* result) {
    return iamaxmin_impl(h, n, x, incx, result, false);
}
cublasStatus_t cublasIzamax(cublasHandle_t h, int n, const cuDoubleComplex* x, int incx,
                            int* result) {
    return iamaxmin_impl(h, n, x, incx, result, true);
}
cublasStatus_t cublasIzamin(cublasHandle_t h, int n, const cuDoubleComplex* x, int incx,
                            int* result) {
    return iamaxmin_impl(h, n, x, incx, result, false);
}

cublasStatus_t cublasScasum(cublasHandle_t h, int n, const cuComplex* x, int incx, float* r) {
    return norm_impl(h, n, x, incx, r, 0);
}
cublasStatus_t cublasScnrm2(cublasHandle_t h, int n, const cuComplex* x, int incx, float* r) {
    return norm_impl(h, n, x, incx, r, 1);
}
cublasStatus_t cublasDzasum(cublasHandle_t h, int n, const cuDoubleComplex* x, int incx,
                            double* r) {
    return norm_impl(h, n, x, incx, r, 0);
}
cublasStatus_t cublasDznrm2(cublasHandle_t h, int n, const cuDoubleComplex* x, int incx,
                            double* r) {
    return norm_impl(h, n, x, incx, r, 1);
}

cublasStatus_t cublasCsscal(cublasHandle_t h, int n, const float* alpha, cuComplex* x, int incx) {
    return scal_impl(h, n, alpha, x, incx);
}
cublasStatus_t cublasZdscal(cublasHandle_t h, int n, const double* alpha, cuDoubleComplex* x,
                            int incx) {
    return scal_impl(h, n, alpha, x, incx);
}

cublasStatus_t cublasSsbmv(cublasHandle_t h, cublasFillMode_t uplo, int n, int k, const float* alpha,
                           const float* A, int lda, const float* x, int incx, const float* beta,
                           float* y, int incy) {
    return sbmv_impl(h, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}
cublasStatus_t cublasDsbmv(cublasHandle_t h, cublasFillMode_t uplo, int n, int k,
                           const double* alpha, const double* A, int lda, const double* x, int incx,
                           const double* beta, double* y, int incy) {
    return sbmv_impl(h, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t cublasStpttr(cublasHandle_t h, cublasFillMode_t uplo, int n, const float* AP,
                            float* A, int lda) {
    return tpttr_impl(h, uplo, n, AP, A, lda);
}
cublasStatus_t cublasDtpttr(cublasHandle_t h, cublasFillMode_t uplo, int n, const double* AP,
                            double* A, int lda) {
    return tpttr_impl(h, uplo, n, AP, A, lda);
}
cublasStatus_t cublasStrttp(cublasHandle_t h, cublasFillMode_t uplo, int n, const float* A, int lda,
                            float* AP) {
    return trttp_impl(h, uplo, n, A, lda, AP);
}
cublasStatus_t cublasDtrttp(cublasHandle_t h, cublasFillMode_t uplo, int n, const double* A,
                            int lda, double* AP) {
    return trttp_impl(h, uplo, n, A, lda, AP);
}

cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k, const float* alpha,
                             const void* A, cudaDataType_t Atype, int lda, const void* B,
                             cudaDataType_t Btype, int ldb, const float* beta, void* C,
                             cudaDataType_t Ctype, int ldc) {
    if (handle == nullptr) return CUBLAS_STATUS_NOT_INITIALIZED;
    // GemmEx falls through to a zero-filling reader for types it does not know,
    // so refuse anything outside the float formats it converts.
    auto supported = [](cudaDataType_t t) {
        return t == CUDA_R_32F || t == CUDA_R_16F || t == CUDA_R_16BF;
    };
    if (!supported(Atype) || !supported(Btype) || !supported(Ctype)) {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta,
                        C, Ctype, ldc, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

}  // extern "C"
