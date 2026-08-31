// Single-precision cuFFT on the Apple GPU.
//
// The transform is separable, so an N-D transform is a sequence of 1-D
// transforms along each axis. Each of those is a batch of independent lines, and
// each line is Stockham autosort radix-2: log2(L) dispatches, one per pass,
// ping-ponging between two scratch buffers. Autosort is the reason there is no
// bit-reversal kernel -- the permutation is folded into the movement each pass
// already does.
//
// Small-factor non-power-of-two lengths use one mixed-radix staged dispatch.
// Bluestein remains the general fallback for lengths with larger prime factors.
//
// Every step stays on the stream the caller's plan is bound to, so this is
// ordered against the rest of their work exactly as the CPU path's synchronize
// would have been, without the round trip to the host that made the FFT a
// synchronous excursion in the middle of an otherwise GPU-resident PME step.
#include "cufft_metal.h"

#include "fft_kernels_msl.h"
#include "library_kernel_source.h"
#include "metal_backend.h"
#include "runtime_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <numbers>
#include <string>
#include <vector>

namespace cumetal::rt::fft_metal {
namespace {

constexpr unsigned kBlock = 256;

// ── policy and diagnostics ───────────────────────────────────────────────────

// CUMETAL_FFT_METAL: unset = auto, "1" = always, "0" = never. Same convention as
// CUMETAL_SPARSE_METAL and CUMETAL_BLAS_METAL.
enum class Policy { kAuto, kAlways, kNever };

Policy policy() {
    static const Policy value = [] {
        const char* v = std::getenv("CUMETAL_FFT_METAL");
        if (v == nullptr || v[0] == '\0') return Policy::kAuto;
        if (v[0] == '0') return Policy::kNever;
        return Policy::kAlways;
    }();
    return value;
}

bool vkfft_enabled() {
    const char* value = std::getenv("CUMETAL_FFT_VKFFT");
    return value == nullptr || value[0] == '\0' || value[0] != '0';
}

bool debug() {
    static const bool on = [] {
        const char* v = std::getenv("CUMETAL_DEBUG_FFT");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

void note(const char* reason) {
    if (debug()) std::fprintf(stderr, "CUMETAL_DEBUG_FFT: on CPU (%s)\n", reason);
}

void note_gpu(int rank, const int* n, int passes) {
    if (!debug()) return;
    std::fprintf(stderr, "CUMETAL_DEBUG_FFT: on the Apple GPU rank=%d dims=%dx%dx%d dispatches=%d\n",
                 rank, n[0], rank > 1 ? n[1] : 1, rank > 2 ? n[2] : 1, passes);
}

// Below this many elements the dispatch cost dominates and the host transform is
// simply faster. A PME mesh is tens of thousands of points, so it is never near
// this; a 16-point test transform always is.
constexpr std::size_t kMinElementsForGpu = 4096;

// ── parameter blocks, matching the MSL layouts ───────────────────────────────

struct PassParams {
    std::uint32_t outer;
    std::uint32_t length;
    std::uint32_t inner;
    std::uint32_t half_span;
    std::uint32_t block_span;
    float sign;
    std::uint32_t pad0;
    std::uint32_t pad1;
};
static_assert(sizeof(PassParams) == 32, "PassParams must match the MSL layout");

struct LayoutParams {
    std::uint32_t dim0;
    std::uint32_t dim1;
    std::uint32_t dim2;
    std::uint32_t embed1;
    std::uint32_t embed2;
    std::uint32_t stride;
    std::uint32_t base;
    std::uint32_t fast_full;
};
static_assert(sizeof(LayoutParams) == 32, "LayoutParams must match the MSL layout");

struct BluesteinParams {
    std::uint32_t lines;
    std::uint32_t length;
    std::uint32_t padded;
    std::uint32_t outer;
    std::uint32_t inner;
    float scale;
    std::uint32_t pad0;
    std::uint32_t pad1;
};
static_assert(sizeof(BluesteinParams) == 32, "BluesteinParams must match the MSL layout");

template <typename T>
metal_backend::KernelArg bytes_arg(const T& value) {
    metal_backend::KernelArg arg;
    arg.kind = metal_backend::KernelArg::Kind::kBytes;
    arg.bytes.resize(sizeof(T));
    std::memcpy(arg.bytes.data(), &value, sizeof(T));
    return arg;
}

const std::string* source_path() {
    return stage_library_kernel_source("fft_kernels", kFftKernelsMsl);
}

// ── device scratch ───────────────────────────────────────────────────────────
//
// One process-lifetime allocation, bump-carved per call. Every transform needs
// several same-shaped buffers and a per-call cudaMalloc would land on the path
// of every PME step.
class Arena {
  public:
    // The dispatches this scratch feeds are asynchronous, so the memory is still
    // live after execute() returns. Reusing it is only safe when the new work is
    // ordered behind the old, which the stream guarantees -- but only for the
    // same stream. A different stream, or freeing the buffer in order to grow
    // it, has to wait for the previous owner first, or an earlier transform ends
    // up reading scratch a later one has already overwritten.
    bool reserve(std::size_t complex_elements, cudaStream_t stream) {
        if (used_ && last_stream_ != stream) {
            cudaStreamSynchronize(last_stream_);
        }
        last_stream_ = stream;
        used_ = true;
        if (complex_elements <= capacity_) {
            cursor_ = 0;
            return true;
        }
        void* grown = nullptr;
        if (cudaMalloc(&grown, complex_elements * 2 * sizeof(float)) != cudaSuccess) {
            return false;
        }
        if (base_ != nullptr) {
            cudaStreamSynchronize(stream);
            cudaFree(base_);
        }
        base_ = static_cast<float*>(grown);
        capacity_ = complex_elements;
        cursor_ = 0;
        return true;
    }
    // Returns a device pointer to `elements` complex values, or nullptr.
    void* take(std::size_t elements) {
        if (cursor_ + elements > capacity_) return nullptr;
        void* p = base_ + cursor_ * 2;
        cursor_ += elements;
        return p;
    }

    // Bluestein's work buffers live only for the duration of one axis, so each
    // axis hands its allocation back rather than stacking another on top. Without
    // this the reservation would have to cover every axis at once, and a grid
    // whose axes all need Bluestein -- 56x56x56, which is rnase's PME mesh --
    // exhausts the arena and silently takes the CPU path instead.
    std::size_t mark() const { return cursor_; }
    void release(std::size_t marked) { cursor_ = marked; }

  private:
    float* base_ = nullptr;
    std::size_t capacity_ = 0;
    std::size_t cursor_ = 0;
    cudaStream_t last_stream_ = nullptr;
    bool used_ = false;
};

std::mutex& arena_mutex() {
    static std::mutex m;
    return m;
}
Arena& arena() {
    static Arena a;
    return a;
}

// ── Bluestein filter cache ───────────────────────────────────────────────────
//
// chirp[j] and FFT(b) depend only on (length, direction), so they are computed
// once on the host and kept on the device. Recomputing them per call would put
// an O(m log m) host transform back on the path this exists to remove.
struct Filter {
    void* chirp = nullptr;   // device float2[n]
    void* spectrum = nullptr;  // device float2[m]
    std::size_t padded = 0;
};

std::size_t next_pow2_at_least(std::size_t v) {
    std::size_t m = 1;
    while (m < v) m <<= 1;
    return m;
}

const Filter* bluestein_filter(std::size_t n, bool inverse) {
    static std::map<std::pair<std::size_t, bool>, Filter> cache;
    const auto key = std::make_pair(n, inverse);
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second.chirp != nullptr ? &it->second : nullptr;

    Filter filter;
    filter.padded = next_pow2_at_least(2 * n - 1);
    const std::size_t m = filter.padded;

    // chirp[j] = exp(sign * i * pi * j^2 / n), with j^2 reduced modulo 2n first:
    // the exponent is periodic there, and j*j would otherwise lose precision
    // long before the cosine sees it.
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<float> chirp_host(2 * n);
    std::vector<double> br(m, 0.0), bi(m, 0.0);
    std::vector<double> chirp_re(n), chirp_im(n);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t reduced = (j % (2 * n)) * (j % (2 * n)) % (2 * n);
        const double angle =
            sign * std::numbers::pi * static_cast<double>(reduced) / static_cast<double>(n);
        chirp_re[j] = std::cos(angle);
        chirp_im[j] = std::sin(angle);
        chirp_host[2 * j] = static_cast<float>(chirp_re[j]);
        chirp_host[2 * j + 1] = static_cast<float>(chirp_im[j]);
    }
    br[0] = chirp_re[0];
    bi[0] = -chirp_im[0];
    for (std::size_t j = 1; j < n; ++j) {
        br[j] = chirp_re[j];
        bi[j] = -chirp_im[j];
        br[m - j] = chirp_re[j];
        bi[m - j] = -chirp_im[j];
    }
    fft_pow2(br, bi, false);

    std::vector<float> spectrum_host(2 * m);
    for (std::size_t k = 0; k < m; ++k) {
        spectrum_host[2 * k] = static_cast<float>(br[k]);
        spectrum_host[2 * k + 1] = static_cast<float>(bi[k]);
    }

    if (cudaMalloc(&filter.chirp, chirp_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&filter.spectrum, spectrum_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMemcpy(filter.chirp, chirp_host.data(), chirp_host.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(filter.spectrum, spectrum_host.data(),
                   spectrum_host.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        if (filter.chirp != nullptr) cudaFree(filter.chirp);
        if (filter.spectrum != nullptr) cudaFree(filter.spectrum);
        cache.emplace(key, Filter{});
        return nullptr;
    }
    const auto inserted = cache.emplace(key, filter);
    return &inserted.first->second;
}

// ── dispatch helper ──────────────────────────────────────────────────────────

struct Context {
    const std::string* source = nullptr;
    std::shared_ptr<metal_backend::Stream> stream;
    int dispatches = 0;
};

bool buffer_arg(const void* ptr, std::size_t bytes, metal_backend::KernelArg* out) {
    return resolve_kernel_buffer_arg(ptr, bytes, alignof(float) * 2, out);
}

bool dispatch(Context& ctx, const char* kernel, std::size_t threads,
              std::vector<metal_backend::KernelArg> args) {
    if (threads == 0) return true;
    metal_backend::LaunchConfig config{};
    config.block = dim3(kBlock, 1, 1);
    config.grid = dim3(static_cast<unsigned>((threads + kBlock - 1) / kBlock), 1, 1);
    config.semantic_quality = "exact";
    std::string error;
    if (metal_backend::launch_kernel(*ctx.source, kernel, config, args, ctx.stream, &error) !=
        cudaSuccess) {
        note(error.empty() ? "launch failed" : error.c_str());
        return false;
    }
    ++ctx.dispatches;
    return true;
}

// A field is a contiguous interleaved-complex grid plus a same-sized spare, so a
// pass can always write somewhere other than where it reads.
struct Field {
    void* cur = nullptr;
    void* alt = nullptr;
    int dims[3] = {1, 1, 1};
    int rank = 1;

    std::size_t count() const {
        std::size_t total = 1;
        for (int a = 0; a < rank; ++a) total *= static_cast<std::size_t>(dims[a]);
        return total;
    }
    std::size_t bytes() const { return count() * 2 * sizeof(float); }
    void swap() { std::swap(cur, alt); }
};

bool is_pow2(std::size_t v) { return v != 0 && (v & (v - 1)) == 0; }

// Threadgroup memory the device will give one threadgroup, read once. Anything
// larger than this has to go through the multi-dispatch path.
std::size_t threadgroup_memory_budget() {
    static const std::size_t budget = [] {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return std::size_t{0};
        return static_cast<std::size_t>(prop.sharedMemPerBlock);
    }();
    return budget;
}

// One threadgroup runs every pass of one line against threadgroup memory,
// replacing log2(L) device round trips with one. Returns false when the line
// does not fit, leaving the caller to dispatch the passes individually.
bool transform_axis_staged(Context& ctx, Field& field, bool inverse, std::size_t length,
                           std::size_t outer, std::size_t inner) {
    const std::size_t staged_bytes = 2 * length * 2 * sizeof(float);
    if (staged_bytes > threadgroup_memory_budget()) return false;
    const std::size_t lines = outer * inner;
    if (lines == 0) return true;

    // One thread per two elements, capped so a long axis strides instead of
    // asking for a threadgroup the device will not schedule.
    unsigned threads = static_cast<unsigned>(std::min<std::size_t>(length / 2, 256));
    if (threads < 32) threads = 32;

    PassParams params{};
    params.outer = static_cast<std::uint32_t>(outer);
    params.length = static_cast<std::uint32_t>(length);
    params.inner = static_cast<std::uint32_t>(inner);
    params.half_span = static_cast<std::uint32_t>(length / 2);
    params.block_span = 1;
    params.sign = inverse ? 1.0f : -1.0f;

    std::vector<metal_backend::KernelArg> args(3);
    if (!buffer_arg(field.cur, field.bytes(), &args[0]) ||
        !buffer_arg(field.alt, field.bytes(), &args[1])) {
        note("field scratch does not resolve to a Metal buffer");
        return false;
    }
    args[2] = bytes_arg(params);

    metal_backend::LaunchConfig config{};
    config.block = dim3(threads, 1, 1);
    config.grid = dim3(static_cast<unsigned>(lines), 1, 1);
    config.shared_memory_bytes = staged_bytes;
    config.semantic_quality = "exact";
    std::string error;
    if (metal_backend::launch_kernel(*ctx.source, "cumetal_fft_stockham_line_f32", config,
                                     args, ctx.stream, &error) != cudaSuccess) {
        note(error.empty() ? "staged launch failed" : error.c_str());
        return false;
    }
    ++ctx.dispatches;
    field.swap();
    return true;
}

// log2(L) Stockham passes over one axis.
bool transform_axis_pow2(Context& ctx, Field& field, int axis, bool inverse,
                         std::size_t outer, std::size_t inner) {
    const std::size_t length = static_cast<std::size_t>(field.dims[axis]);
    if (transform_axis_staged(ctx, field, inverse, length, outer, inner)) return true;
    const std::size_t threads = outer * inner * (length / 2);
    PassParams params{};
    params.outer = static_cast<std::uint32_t>(outer);
    params.length = static_cast<std::uint32_t>(length);
    params.inner = static_cast<std::uint32_t>(inner);
    params.sign = inverse ? 1.0f : -1.0f;

    for (std::size_t half_span = length / 2, block_span = 1; half_span >= 1;
         half_span /= 2, block_span *= 2) {
        params.half_span = static_cast<std::uint32_t>(half_span);
        params.block_span = static_cast<std::uint32_t>(block_span);
        std::vector<metal_backend::KernelArg> args(3);
        if (!buffer_arg(field.cur, field.bytes(), &args[0]) ||
            !buffer_arg(field.alt, field.bytes(), &args[1])) {
            note("field scratch does not resolve to a Metal buffer");
            return false;
        }
        args[2] = bytes_arg(params);
        if (!dispatch(ctx, "cumetal_fft_stockham_f32", threads, std::move(args))) return false;
        field.swap();
        if (half_span == 1) break;
    }
    return true;
}

bool mixed_radix_supported(std::size_t length) {
    while (length > 1) {
        if (length % 5 == 0) length /= 5;
        else if (length % 4 == 0) length /= 4;
        else if (length % 3 == 0) length /= 3;
        else if (length % 2 == 0) length /= 2;
        else if (length <= 13) length = 1;
        else return false;
    }
    return true;
}

bool transform_axis_mixed(Context& ctx, Field& field, bool inverse, std::size_t length,
                          std::size_t outer, std::size_t inner) {
    constexpr std::size_t kLineTile = 16;
    const bool tiled = inner > 1 &&
        2 * length * kLineTile * 2 * sizeof(float) <= threadgroup_memory_budget();
    const std::size_t staged_bytes =
        2 * length * (tiled ? kLineTile : 1) * 2 * sizeof(float);
    if (staged_bytes > threadgroup_memory_budget()) return false;
    const std::size_t lines = outer * inner;
    if (lines == 0) return true;

    unsigned threads = tiled ? 256u
                             : static_cast<unsigned>(std::min<std::size_t>(length, 256));
    if (threads < 32) threads = 32;
    PassParams params{};
    params.outer = static_cast<std::uint32_t>(outer);
    params.length = static_cast<std::uint32_t>(length);
    params.inner = static_cast<std::uint32_t>(inner);
    params.sign = inverse ? 1.0f : -1.0f;

    std::vector<metal_backend::KernelArg> args(3);
    if (!buffer_arg(field.cur, field.bytes(), &args[0]) ||
        !buffer_arg(field.alt, field.bytes(), &args[1])) {
        note("mixed-radix field buffers do not resolve");
        return false;
    }
    args[2] = bytes_arg(params);
    metal_backend::LaunchConfig config{};
    config.block = dim3(threads, 1, 1);
    config.grid = dim3(static_cast<unsigned>(tiled ? (lines + kLineTile - 1) / kLineTile
                                                   : lines), 1, 1);
    config.shared_memory_bytes = staged_bytes;
    config.semantic_quality = "exact";
    std::string error;
    const char* kernel = tiled ? "cumetal_fft_stockham_mixed_tile16_f32"
                               : "cumetal_fft_stockham_mixed_line_f32";
    if (metal_backend::launch_kernel(*ctx.source, kernel,
                                     config, args, ctx.stream, &error) != cudaSuccess) {
        note(error.empty() ? "mixed-radix launch failed" : error.c_str());
        return false;
    }
    ++ctx.dispatches;
    field.swap();
    return true;
}

// Bluestein over one axis, reading and writing the field in place.
bool transform_axis_bluestein(Context& ctx, Field& field, int axis, bool inverse,
                              std::size_t outer, std::size_t inner) {
    const std::size_t length = static_cast<std::size_t>(field.dims[axis]);
    const Filter* filter = bluestein_filter(length, inverse);
    if (filter == nullptr) {
        note("could not build the Bluestein filter");
        return false;
    }
    const std::size_t m = filter->padded;
    const std::size_t lines = outer * inner;
    const std::size_t padded_elements = lines * m;

    const std::size_t marked = arena().mark();
    struct ReleaseOnReturn {
        std::size_t marked;
        ~ReleaseOnReturn() { arena().release(marked); }
    } release_on_return{marked};

    void* work = arena().take(padded_elements);
    void* work_alt = arena().take(padded_elements);
    if (work == nullptr || work_alt == nullptr) {
        note("scratch arena exhausted for the Bluestein work buffers");
        return false;
    }
    const std::size_t work_bytes = padded_elements * 2 * sizeof(float);

    BluesteinParams params{};
    params.lines = static_cast<std::uint32_t>(lines);
    params.length = static_cast<std::uint32_t>(length);
    params.padded = static_cast<std::uint32_t>(m);
    params.outer = static_cast<std::uint32_t>(outer);
    params.inner = static_cast<std::uint32_t>(inner);
    params.scale = 1.0f / static_cast<float>(m);

    {
        std::vector<metal_backend::KernelArg> args(4);
        if (!buffer_arg(field.cur, field.bytes(), &args[0]) ||
            !buffer_arg(filter->chirp, length * 2 * sizeof(float), &args[1]) ||
            !buffer_arg(work, work_bytes, &args[2])) {
            note("Bluestein pre-pass buffers do not resolve");
            return false;
        }
        args[3] = bytes_arg(params);
        if (!dispatch(ctx, "cumetal_fft_bluestein_pre_f32", padded_elements, std::move(args)))
            return false;
    }

    // The convolution runs on the padded contiguous work buffer: one line per
    // row, so the pass kernels see outer = lines and inner = 1. dims[0] is the
    // whole buffer rather than one line so bytes() bounds the binding correctly.
    Field convolution;
    convolution.cur = work;
    convolution.alt = work_alt;
    convolution.rank = 1;
    convolution.dims[0] = static_cast<int>(padded_elements);

    if (!transform_axis_staged(ctx, convolution, false, m, lines, 1)) {
        PassParams pass{};
        pass.outer = static_cast<std::uint32_t>(lines);
        pass.length = static_cast<std::uint32_t>(m);
        pass.inner = 1;
        pass.sign = -1.0f;  // the convolution's forward transform, both operands
        for (std::size_t half_span = m / 2, block_span = 1; half_span >= 1;
             half_span /= 2, block_span *= 2) {
            pass.half_span = static_cast<std::uint32_t>(half_span);
            pass.block_span = static_cast<std::uint32_t>(block_span);
            std::vector<metal_backend::KernelArg> args(3);
            if (!buffer_arg(convolution.cur, work_bytes, &args[0]) ||
                !buffer_arg(convolution.alt, work_bytes, &args[1])) {
                note("Bluestein work buffers do not resolve");
                return false;
            }
            args[2] = bytes_arg(pass);
            if (!dispatch(ctx, "cumetal_fft_stockham_f32", lines * (m / 2), std::move(args)))
                return false;
            convolution.swap();
            if (half_span == 1) break;
        }
    }

    {
        std::vector<metal_backend::KernelArg> args(3);
        if (!buffer_arg(convolution.cur, work_bytes, &args[0]) ||
            !buffer_arg(filter->spectrum, m * 2 * sizeof(float), &args[1])) {
            note("Bluestein filter multiply buffers do not resolve");
            return false;
        }
        args[2] = bytes_arg(params);
        if (!dispatch(ctx, "cumetal_fft_bluestein_mul_f32", padded_elements, std::move(args)))
            return false;
    }

    if (!transform_axis_staged(ctx, convolution, true, m, lines, 1)) {
        PassParams pass{};
        pass.outer = static_cast<std::uint32_t>(lines);
        pass.length = static_cast<std::uint32_t>(m);
        pass.inner = 1;
        pass.sign = 1.0f;  // inverse, unnormalized; the 1/m is folded into post
        for (std::size_t half_span = m / 2, block_span = 1; half_span >= 1;
             half_span /= 2, block_span *= 2) {
            pass.half_span = static_cast<std::uint32_t>(half_span);
            pass.block_span = static_cast<std::uint32_t>(block_span);
            std::vector<metal_backend::KernelArg> args(3);
            if (!buffer_arg(convolution.cur, work_bytes, &args[0]) ||
                !buffer_arg(convolution.alt, work_bytes, &args[1])) {
                note("Bluestein work buffers do not resolve");
                return false;
            }
            args[2] = bytes_arg(pass);
            if (!dispatch(ctx, "cumetal_fft_stockham_f32", lines * (m / 2), std::move(args)))
                return false;
            convolution.swap();
            if (half_span == 1) break;
        }
    }

    {
        std::vector<metal_backend::KernelArg> args(4);
        if (!buffer_arg(convolution.cur, work_bytes, &args[0]) ||
            !buffer_arg(filter->chirp, length * 2 * sizeof(float), &args[1]) ||
            !buffer_arg(field.cur, field.bytes(), &args[2])) {
            note("Bluestein post-pass buffers do not resolve");
            return false;
        }
        args[3] = bytes_arg(params);
        if (!dispatch(ctx, "cumetal_fft_bluestein_post_f32", lines * length, std::move(args)))
            return false;
    }
    return true;
}

bool transform_axis(Context& ctx, Field& field, int axis, bool inverse) {
    const std::size_t length = static_cast<std::size_t>(field.dims[axis]);
    if (length <= 1) return true;
    std::size_t inner = 1;
    for (int a = axis + 1; a < field.rank; ++a) inner *= static_cast<std::size_t>(field.dims[a]);
    std::size_t outer = 1;
    for (int a = 0; a < axis; ++a) outer *= static_cast<std::size_t>(field.dims[a]);
    if (is_pow2(length)) {
        return transform_axis_pow2(ctx, field, axis, inverse, outer, inner);
    }
    if (mixed_radix_supported(length) &&
        transform_axis_mixed(ctx, field, inverse, length, outer, inner)) {
        return true;
    }
    return transform_axis_bluestein(ctx, field, axis, inverse, outer, inner);
}

LayoutParams make_layout(const Layout& layout, int rank, const int* dims, long long base,
                         int fast_full) {
    LayoutParams p{};
    // A rank below 3 is expressed as leading extents of 1, so the MSL addressing
    // is one expression rather than three.
    const int pad = 3 - rank;
    for (int a = 0; a < 3; ++a) {
        const int source = a - pad;
        const int extent = source < 0 ? 1 : dims[source];
        const int embed = source < 0 ? 1 : layout.embed[source];
        if (a == 0) p.dim0 = static_cast<std::uint32_t>(extent);
        if (a == 1) {
            p.dim1 = static_cast<std::uint32_t>(extent);
            p.embed1 = static_cast<std::uint32_t>(embed);
        }
        if (a == 2) {
            p.dim2 = static_cast<std::uint32_t>(extent);
            p.embed2 = static_cast<std::uint32_t>(embed);
        }
    }
    p.stride = static_cast<std::uint32_t>(layout.stride);
    p.base = static_cast<std::uint32_t>(base);
    p.fast_full = static_cast<std::uint32_t>(fast_full);
    return p;
}

// Extent of the buffer a layout addresses, in elements, so the binding can be
// checked against the allocation rather than trusted.
std::size_t layout_span(const Layout& layout, int rank, const int* dims, long long base) {
    long long last = dims[0] - 1;
    for (int a = 1; a < rank; ++a) last = last * layout.embed[a] + (dims[a] - 1);
    return static_cast<std::size_t>(base + last * layout.stride + 1);
}

}  // namespace

void fft_pow2(std::vector<double>& re, std::vector<double>& im, bool inverse) {
    const std::size_t m = re.size();
    for (std::size_t i = 1, j = 0; i < m; ++i) {
        std::size_t bit = m >> 1;
        for (; (j & bit) != 0; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (std::size_t len = 2; len <= m; len <<= 1) {
        const double step =
            2.0 * std::numbers::pi / static_cast<double>(len) * (inverse ? 1.0 : -1.0);
        const std::size_t half = len / 2;
        for (std::size_t base = 0; base < m; base += len) {
            for (std::size_t k = 0; k < half; ++k) {
                const double angle = step * static_cast<double>(k);
                const double wr = std::cos(angle);
                const double wi = std::sin(angle);
                const double lo_re = re[base + k];
                const double lo_im = im[base + k];
                const double hi_re = re[base + k + half] * wr - im[base + k + half] * wi;
                const double hi_im = re[base + k + half] * wi + im[base + k + half] * wr;
                re[base + k] = lo_re + hi_re;
                im[base + k] = lo_im + hi_im;
                re[base + k + half] = lo_re - hi_re;
                im[base + k + half] = lo_im - hi_im;
            }
        }
    }
}

bool prepare(const Request& request) {
    if (policy() == Policy::kNever || !vkfft_enabled() || request.rank != 3 ||
        request.batch != 1 || request.kind == Kind::kC2C ||
        request.input.stride != 1 || request.output.stride != 1) {
        return false;
    }
    std::size_t elements = 1;
    for (int axis = 0; axis < 3; ++axis) {
        if (request.n[axis] <= 0) return false;
        elements *= static_cast<std::size_t>(request.n[axis]);
    }
    if (policy() != Policy::kAlways && elements < kMinElementsForGpu) return false;

    const Layout& real_layout =
        request.kind == Kind::kR2C ? request.input : request.output;
    const Layout& complex_layout =
        request.kind == Kind::kR2C ? request.output : request.input;
    metal_backend::Fft3dR2CConfig config{};
    for (int axis = 0; axis < 3; ++axis) {
        config.n[axis] = request.n[axis];
        config.real_embed[axis] = real_layout.embed[axis];
        config.complex_embed[axis] = complex_layout.embed[axis];
    }
    std::shared_ptr<metal_backend::Stream> stream;
    if (resolve_backend_stream(request.stream, &stream) != cudaSuccess) return false;
    std::string error;
    const bool ready =
        metal_backend::prepare_fft_r2c_3d_f32(config, stream, &error) == cudaSuccess;
    if (!ready) note(error.empty() ? "VkFFT plan preparation declined" : error.c_str());
    return ready;
}

bool execute(const Request& request) {
    if (policy() == Policy::kNever) {
        note("CUMETAL_FFT_METAL=0");
        return false;
    }
    if (request.idata == nullptr || request.odata == nullptr) return false;

    const int rank = request.rank;
    if (debug()) {
        std::fprintf(stderr,
                     "CUMETAL_DEBUG_FFT: layout kind=%d rank=%d batch=%d "
                     "in(stride=%lld dist=%lld embed=%lldx%lldx%lld) "
                     "out(stride=%lld dist=%lld embed=%lldx%lldx%lld)\n",
                     static_cast<int>(request.kind), rank, request.batch,
                     static_cast<long long>(request.input.stride), request.input.dist,
                     static_cast<long long>(request.input.embed[0]),
                     static_cast<long long>(request.input.embed[1]),
                     static_cast<long long>(request.input.embed[2]),
                     static_cast<long long>(request.output.stride), request.output.dist,
                     static_cast<long long>(request.output.embed[0]),
                     static_cast<long long>(request.output.embed[1]),
                     static_cast<long long>(request.output.embed[2]));
    }
    int real_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) real_dims[a] = request.n[a];
    int complex_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) complex_dims[a] = real_dims[a];
    const bool real_transform = request.kind != Kind::kC2C;
    if (real_transform) complex_dims[rank - 1] = real_dims[rank - 1] / 2 + 1;

    std::size_t real_count = 1;
    std::size_t complex_count = 1;
    for (int a = 0; a < rank; ++a) {
        real_count *= static_cast<std::size_t>(real_dims[a]);
        complex_count *= static_cast<std::size_t>(complex_dims[a]);
    }

    if (policy() != Policy::kAlways && real_count < kMinElementsForGpu) {
        note("below the dispatch-cost threshold");
        return false;
    }

    Context ctx;
    if (resolve_backend_stream(request.stream, &ctx.stream) != cudaSuccess) {
        note("the plan's stream does not resolve to a backend stream");
        return false;
    }

    // VkFFT's generated Metal kernels are the production path for the dense,
    // out-of-place 3-D real transform used by PME. Keep the project-owned
    // Stockham engine below as the explicit general-layout fallback.
    if (vkfft_enabled() && rank == 3 && request.batch == 1 && real_transform &&
        request.input.stride == 1 && request.output.stride == 1 &&
        request.idata != request.odata) {
        const Layout& real_layout =
            request.kind == Kind::kR2C ? request.input : request.output;
        const Layout& complex_layout =
            request.kind == Kind::kR2C ? request.output : request.input;
        const void* real_ptr =
            request.kind == Kind::kR2C ? request.idata : request.odata;
        const void* complex_ptr =
            request.kind == Kind::kR2C ? request.odata : request.idata;
        const std::size_t real_bytes =
            layout_span(real_layout, rank, real_dims, 0) * sizeof(float);
        const std::size_t complex_bytes =
            layout_span(complex_layout, rank, complex_dims, 0) * 2 * sizeof(float);
        metal_backend::KernelArg real_arg;
        metal_backend::KernelArg complex_arg;
        if (buffer_arg(real_ptr, real_bytes, &real_arg) &&
            buffer_arg(complex_ptr, complex_bytes, &complex_arg)) {
            metal_backend::Fft3dR2CConfig config{};
            for (int axis = 0; axis < 3; ++axis) {
                config.n[axis] = real_dims[axis];
                config.real_embed[axis] = real_layout.embed[axis];
                config.complex_embed[axis] = complex_layout.embed[axis];
            }
            config.inverse = request.kind == Kind::kC2R;
            std::string error;
            if (metal_backend::fft_r2c_3d_f32(
                    config, real_arg.buffer, real_arg.offset,
                    complex_arg.buffer, complex_arg.offset, ctx.stream, &error) == cudaSuccess) {
                if (debug()) {
                    std::fprintf(stderr,
                                 "CUMETAL_DEBUG_FFT: VkFFT Apple GPU rank=3 dims=%dx%dx%d\n",
                                 real_dims[0], real_dims[1], real_dims[2]);
                }
                return true;
            }
            note(error.empty() ? "VkFFT declined" : error.c_str());
        }
    }
    ctx.source = source_path();
    if (ctx.source == nullptr) {
        note("could not stage the kernel source");
        return false;
    }

    // Worst case: a full field and its spare, a half field and its spare, and
    // two Bluestein work buffers for the longest axis. Reserving once means a
    // later take() cannot fail part-way through a transform.
    // Only one axis holds Bluestein scratch at a time, so the reservation needs
    // the largest single axis rather than their sum. The fastest axis is
    // transformed on the full field and the others on the half field, which have
    // different line counts; taking the larger of the two keeps this an upper
    // bound without tracking which field each axis lands on.
    std::size_t longest_padded = 0;
    for (int a = 0; a < rank; ++a) {
        const std::size_t length = static_cast<std::size_t>(real_dims[a]);
        if (is_pow2(length)) continue;
        const std::size_t lines = real_count / length;
        longest_padded =
            std::max(longest_padded, lines * next_pow2_at_least(2 * length - 1));
    }
    const std::lock_guard<std::mutex> lock(arena_mutex());
    const std::size_t needed =
        2 * real_count + (real_transform ? 2 * complex_count : 0) + 2 * longest_padded;
    if (!arena().reserve(needed, request.stream)) {
        note("could not allocate device scratch");
        return false;
    }

    Field full;
    full.rank = rank;
    for (int a = 0; a < rank; ++a) full.dims[a] = real_dims[a];
    full.cur = arena().take(real_count);
    full.alt = arena().take(real_count);
    Field half;
    half.rank = rank;
    for (int a = 0; a < rank; ++a) half.dims[a] = complex_dims[a];
    if (real_transform) {
        half.cur = arena().take(complex_count);
        half.alt = arena().take(complex_count);
    }
    if (full.cur == nullptr || full.alt == nullptr ||
        (real_transform && (half.cur == nullptr || half.alt == nullptr))) {
        note("scratch arena exhausted");
        return false;
    }

    const std::size_t element_size = sizeof(float);
    for (int b = 0; b < request.batch; ++b) {
        const long long in_base = static_cast<long long>(b) * request.input.dist;
        const long long out_base = static_cast<long long>(b) * request.output.dist;

        if (request.kind == Kind::kR2C) {
            const LayoutParams in_params =
                make_layout(request.input, rank, real_dims, in_base, 0);
            std::vector<metal_backend::KernelArg> args(3);
            const std::size_t in_bytes =
                layout_span(request.input, rank, real_dims, in_base) * element_size;
            if (!buffer_arg(request.idata, in_bytes, &args[0]) ||
                !buffer_arg(full.cur, full.bytes(), &args[1])) {
                note("input grid does not resolve to a Metal buffer");
                return false;
            }
            args[2] = bytes_arg(in_params);
            if (!dispatch(ctx, "cumetal_fft_load_real_f32", real_count, std::move(args)))
                return false;

            if (!transform_axis(ctx, full, rank - 1, false)) return false;

            LayoutParams truncate = make_layout(Layout{}, rank, complex_dims, 0, 0);
            truncate.fast_full = static_cast<std::uint32_t>(real_dims[rank - 1]);
            std::vector<metal_backend::KernelArg> targs(3);
            if (!buffer_arg(full.cur, full.bytes(), &targs[0]) ||
                !buffer_arg(half.cur, half.bytes(), &targs[1])) {
                note("truncate buffers do not resolve");
                return false;
            }
            targs[2] = bytes_arg(truncate);
            if (!dispatch(ctx, "cumetal_fft_truncate_f32", complex_count, std::move(targs)))
                return false;

            for (int a = 0; a < rank - 1; ++a) {
                if (!transform_axis(ctx, half, a, false)) return false;
            }

            const LayoutParams out_params =
                make_layout(request.output, rank, complex_dims, out_base, 0);
            const std::size_t out_bytes =
                layout_span(request.output, rank, complex_dims, out_base) * 2 * element_size;
            std::vector<metal_backend::KernelArg> sargs(3);
            if (!buffer_arg(half.cur, half.bytes(), &sargs[0]) ||
                !buffer_arg(request.odata, out_bytes, &sargs[1])) {
                note("output grid does not resolve to a Metal buffer");
                return false;
            }
            sargs[2] = bytes_arg(out_params);
            if (!dispatch(ctx, "cumetal_fft_store_complex_f32", complex_count,
                          std::move(sargs)))
                return false;
        } else if (request.kind == Kind::kC2R) {
            const LayoutParams in_params =
                make_layout(request.input, rank, complex_dims, in_base, 0);
            const std::size_t in_bytes =
                layout_span(request.input, rank, complex_dims, in_base) * 2 * element_size;
            std::vector<metal_backend::KernelArg> args(3);
            if (!buffer_arg(request.idata, in_bytes, &args[0]) ||
                !buffer_arg(half.cur, half.bytes(), &args[1])) {
                note("input grid does not resolve to a Metal buffer");
                return false;
            }
            args[2] = bytes_arg(in_params);
            if (!dispatch(ctx, "cumetal_fft_load_complex_f32", complex_count, std::move(args)))
                return false;

            for (int a = 0; a < rank - 1; ++a) {
                if (!transform_axis(ctx, half, a, true)) return false;
            }

            LayoutParams expand = make_layout(Layout{}, rank, real_dims, 0, 0);
            expand.fast_full = static_cast<std::uint32_t>(complex_dims[rank - 1]);
            std::vector<metal_backend::KernelArg> eargs(3);
            if (!buffer_arg(half.cur, half.bytes(), &eargs[0]) ||
                !buffer_arg(full.cur, full.bytes(), &eargs[1])) {
                note("Hermitian expand buffers do not resolve");
                return false;
            }
            eargs[2] = bytes_arg(expand);
            if (!dispatch(ctx, "cumetal_fft_hermitian_expand_f32", real_count,
                          std::move(eargs)))
                return false;

            if (!transform_axis(ctx, full, rank - 1, true)) return false;

            const LayoutParams out_params =
                make_layout(request.output, rank, real_dims, out_base, 0);
            const std::size_t out_bytes =
                layout_span(request.output, rank, real_dims, out_base) * element_size;
            std::vector<metal_backend::KernelArg> sargs(3);
            if (!buffer_arg(full.cur, full.bytes(), &sargs[0]) ||
                !buffer_arg(request.odata, out_bytes, &sargs[1])) {
                note("output grid does not resolve to a Metal buffer");
                return false;
            }
            sargs[2] = bytes_arg(out_params);
            if (!dispatch(ctx, "cumetal_fft_store_real_f32", real_count, std::move(sargs)))
                return false;
        } else {
            const LayoutParams in_params =
                make_layout(request.input, rank, real_dims, in_base, 0);
            const std::size_t in_bytes =
                layout_span(request.input, rank, real_dims, in_base) * 2 * element_size;
            std::vector<metal_backend::KernelArg> args(3);
            if (!buffer_arg(request.idata, in_bytes, &args[0]) ||
                !buffer_arg(full.cur, full.bytes(), &args[1])) {
                note("input grid does not resolve to a Metal buffer");
                return false;
            }
            args[2] = bytes_arg(in_params);
            if (!dispatch(ctx, "cumetal_fft_load_complex_f32", real_count, std::move(args)))
                return false;

            for (int a = 0; a < rank; ++a) {
                if (!transform_axis(ctx, full, a, request.inverse)) return false;
            }

            const LayoutParams out_params =
                make_layout(request.output, rank, real_dims, out_base, 0);
            const std::size_t out_bytes =
                layout_span(request.output, rank, real_dims, out_base) * 2 * element_size;
            std::vector<metal_backend::KernelArg> sargs(3);
            if (!buffer_arg(full.cur, full.bytes(), &sargs[0]) ||
                !buffer_arg(request.odata, out_bytes, &sargs[1])) {
                note("output grid does not resolve to a Metal buffer");
                return false;
            }
            sargs[2] = bytes_arg(out_params);
            if (!dispatch(ctx, "cumetal_fft_store_complex_f32", real_count, std::move(sargs)))
                return false;
        }
    }

    note_gpu(rank, real_dims, ctx.dispatches);
    return true;
}

}  // namespace cumetal::rt::fft_metal
