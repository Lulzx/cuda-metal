// cuFFT shim backed by Apple Accelerate vDSP (spec §11 Phase 4.5).
//
// The DFT convention matches cuFFT and is unnormalized in both directions:
//   FORWARD:  X[k] = Σ x[n] · e^{-2πi k n/N}
//   INVERSE:  x[n] = Σ X[k] · e^{+2πi k n/N}   (caller divides by N)
//
// Ranks 1 to 3 are supported for every transform type, together with cuFFT's
// advanced data layout (inembed/onembed/stride/dist). Multidimensional
// transforms are separable, so each is executed as a sequence of 1-D transforms,
// one axis at a time, over a contiguous split-complex working copy of the grid.
// Real transforms carry the half-spectrum on the fastest axis, n/2 + 1 as cuFFT
// defines it.
//
// This runs on the CPU over unified memory; it is not a Metal kernel. A caller
// that offloads an FFT here gets the right answer, not GPU execution, and the
// stream is synchronized first so it reads what earlier kernels wrote.

#include "cufftXt.h"

#include "cufft_metal.h"
#include "runtime_internal.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <type_traits>
#include <utility>
#include <mutex>
#include <new>
#include <numbers>
#include <vector>

namespace {

// ── Plan storage ─────────────────────────────────────────────────────────────

// cuFFT's "advanced data layout". Element (x0 … x_{rank-1}) of batch b sits at
//   b*dist + stride * (((x0 * embed[1]) + x1) * embed[2] + x2)
// embed[0] never participates in addressing, matching cuFFT. The layout is what
// makes a padded grid describable: GROMACS's PME real grid is padded on the
// fastest axis, so embed[rank-1] exceeds n[rank-1] and the rows are strided.
struct FftLayout {
    int embed[3] = {1, 1, 1};
    int stride = 1;
    long long dist = 0;
};

long long layout_offset(const FftLayout& layout, int rank, const int* index) {
    long long offset = index[0];
    for (int axis = 1; axis < rank; ++axis) {
        offset = offset * layout.embed[axis] + index[axis];
    }
    return offset * layout.stride;
}

struct CufftPlanEntry {
    cufftType type = CUFFT_C2C;
    int rank = 1;
    // Logical dimensions, slowest axis first. For R2C/C2R these are the *real*
    // dimensions; the complex side is n[rank-1]/2 + 1 on the fastest axis.
    std::vector<int> n;
    int batch = 1;
    FftLayout input;
    FftLayout output;
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    cudaStream_t stream = nullptr;
};

struct CufftState {
    std::mutex mutex;
    std::map<cufftHandle, CufftPlanEntry> plans;
    int next_handle = 1;

    CufftPlanEntry* get(cufftHandle h) {
        auto it = plans.find(h);
        return it == plans.end() ? nullptr : &it->second;
    }
};

CufftState& state() {
    // Immortal on purpose. This is process-lifetime state guarded by a mutex, and a
    // function-local static gets an atexit destructor: anything that touches it during
    // teardown -- another static's destructor, a detached worker, a Metal completion
    // handler -- then locks a destroyed mutex. That surfaced as an intermittent
    // "mutex lock failed: Invalid argument" abort *after* a test had already printed
    // PASS. Leaking one object at exit is the fix; the OS reclaims it.
    static CufftState* s = new CufftState();
    return *s;
}

// ── 1-D transform engine ─────────────────────────────────────────────────────
//
// Every transform here is built from unnormalized 1-D complex DFTs, which is
// cuFFT's convention in both directions: an inverse transform of a forward one
// returns N times the input, and the caller divides.
//
// vDSP accepts only lengths of the form f * 2^k with f in {1, 3, 5, 15}. PME
// grids are 2-3-5-7 smooth, so a factor of 7 on any axis falls outside that set.
// Those lengths go through Bluestein's chirp-z algorithm, which expresses a DFT
// of arbitrary length as a power-of-two convolution. That is O(n log n), so
// unlike the bounded direct sum it replaces there is no size ceiling and no
// length that has to be rejected.

// Setup cache keyed by length and direction. Every caller already holds
// CufftState::mutex, so no additional lock is needed. Setups are process-lifetime
// and deliberately never destroyed: they are shared across plans, so a plan's
// destruction cannot know whether another plan still needs one.
vDSP_DFT_Setup vdsp_setup(vDSP_Length n, vDSP_DFT_Direction dir, float) {
    static std::map<std::pair<vDSP_Length, int>, vDSP_DFT_Setup> cache;
    const auto key = std::make_pair(n, static_cast<int>(dir));
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    vDSP_DFT_Setup setup = vDSP_DFT_zop_CreateSetup(nullptr, n, dir);
    cache.emplace(key, setup);
    return setup;
}

vDSP_DFT_SetupD vdsp_setup(vDSP_Length n, vDSP_DFT_Direction dir, double) {
    static std::map<std::pair<vDSP_Length, int>, vDSP_DFT_SetupD> cache;
    const auto key = std::make_pair(n, static_cast<int>(dir));
    const auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    vDSP_DFT_SetupD setup = vDSP_DFT_zop_CreateSetupD(nullptr, n, dir);
    cache.emplace(key, setup);
    return setup;
}

// Bluestein's inner power-of-two transforms come from the Metal path's shared
// implementation so the CPU and GPU chirp convolutions fold identically.
namespace fft_metal = cumetal::rt::fft_metal;
using fft_metal::fft_pow2;

template <typename Scalar>
void bluestein(Scalar* re, Scalar* im, std::size_t n, bool inverse) {
    std::size_t m = 1;
    while (m < 2 * n - 1) m <<= 1;

    // chirp[j] = exp(sign * i * pi * j^2 / n). j^2 is reduced modulo 2n first:
    // the exponent is periodic there, and for a long axis j*j would otherwise
    // lose precision long before the cosine is evaluated.
    const double sign = inverse ? 1.0 : -1.0;
    std::vector<double> chirp_re(n), chirp_im(n);
    for (std::size_t j = 0; j < n; ++j) {
        const std::size_t reduced = (j % (2 * n)) * (j % (2 * n)) % (2 * n);
        const double angle =
            sign * std::numbers::pi * static_cast<double>(reduced) / static_cast<double>(n);
        chirp_re[j] = std::cos(angle);
        chirp_im[j] = std::sin(angle);
    }

    std::vector<double> ar(m, 0.0), ai(m, 0.0), br(m, 0.0), bi(m, 0.0);
    for (std::size_t j = 0; j < n; ++j) {
        const double xr = static_cast<double>(re[j]);
        const double xi = static_cast<double>(im[j]);
        ar[j] = xr * chirp_re[j] - xi * chirp_im[j];
        ai[j] = xr * chirp_im[j] + xi * chirp_re[j];
    }
    br[0] = chirp_re[0];
    bi[0] = -chirp_im[0];
    for (std::size_t j = 1; j < n; ++j) {
        br[j] = chirp_re[j];
        bi[j] = -chirp_im[j];
        br[m - j] = chirp_re[j];
        bi[m - j] = -chirp_im[j];
    }

    fft_pow2(ar, ai, false);
    fft_pow2(br, bi, false);
    for (std::size_t k = 0; k < m; ++k) {
        const double pr = ar[k] * br[k] - ai[k] * bi[k];
        const double pi_ = ar[k] * bi[k] + ai[k] * br[k];
        ar[k] = pr;
        ai[k] = pi_;
    }
    fft_pow2(ar, ai, true);

    const double scale = 1.0 / static_cast<double>(m);
    for (std::size_t k = 0; k < n; ++k) {
        const double vr = ar[k] * scale;
        const double vi = ai[k] * scale;
        re[k] = static_cast<Scalar>(vr * chirp_re[k] - vi * chirp_im[k]);
        im[k] = static_cast<Scalar>(vr * chirp_im[k] + vi * chirp_re[k]);
    }
}

// In-place unnormalized DFT of a contiguous split-complex line.
template <typename Scalar>
void fft_1d(Scalar* re, Scalar* im, std::size_t n, bool inverse) {
    if (n <= 1) return;
    const vDSP_DFT_Direction dir = inverse ? vDSP_DFT_INVERSE : vDSP_DFT_FORWARD;
    auto setup = vdsp_setup(static_cast<vDSP_Length>(n), dir, Scalar{});
    if (setup == nullptr) {
        bluestein(re, im, n, inverse);
        return;
    }
    std::vector<Scalar> out_re(n), out_im(n);
    if constexpr (std::is_same_v<Scalar, float>) {
        vDSP_DFT_Execute(setup, re, im, out_re.data(), out_im.data());
    } else {
        vDSP_DFT_ExecuteD(setup, re, im, out_re.data(), out_im.data());
    }
    std::copy(out_re.begin(), out_re.end(), re);
    std::copy(out_im.begin(), out_im.end(), im);
}

// ── N-D field ────────────────────────────────────────────────────────────────
//
// A contiguous split-complex array, slowest axis first. Multidimensional
// transforms are separable, so each axis is a batch of independent 1-D
// transforms; doing them one axis at a time is what makes any rank correct
// rather than the single flattened transform this replaces, which computed a
// different function entirely.
template <typename Scalar>
struct ComplexField {
    int dims[3] = {1, 1, 1};
    int rank = 1;
    std::vector<Scalar> re;
    std::vector<Scalar> im;

    std::size_t count() const {
        std::size_t total = 1;
        for (int axis = 0; axis < rank; ++axis) total *= static_cast<std::size_t>(dims[axis]);
        return total;
    }
    void resize(int r, const int* d) {
        rank = r;
        for (int axis = 0; axis < r; ++axis) dims[axis] = d[axis];
        re.assign(count(), Scalar{});
        im.assign(count(), Scalar{});
    }
};

template <typename Scalar>
void transform_axis(ComplexField<Scalar>& field, int axis, bool inverse) {
    const std::size_t length = static_cast<std::size_t>(field.dims[axis]);
    if (length <= 1) return;
    std::size_t inner = 1;
    for (int a = axis + 1; a < field.rank; ++a) inner *= static_cast<std::size_t>(field.dims[a]);
    std::size_t outer = 1;
    for (int a = 0; a < axis; ++a) outer *= static_cast<std::size_t>(field.dims[a]);

    std::vector<Scalar> line_re(length), line_im(length);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t i = 0; i < inner; ++i) {
            const std::size_t base = o * length * inner + i;
            for (std::size_t j = 0; j < length; ++j) {
                line_re[j] = field.re[base + j * inner];
                line_im[j] = field.im[base + j * inner];
            }
            fft_1d(line_re.data(), line_im.data(), length, inverse);
            for (std::size_t j = 0; j < length; ++j) {
                field.re[base + j * inner] = line_re[j];
                field.im[base + j * inner] = line_im[j];
            }
        }
    }
}

// Walks every logical index of `dims` in row-major order, calling
// visit(contiguous_index, index[]).
template <typename Visit>
void for_each_index(int rank, const int* dims, Visit visit) {
    int index[3] = {0, 0, 0};
    std::size_t linear = 0;
    const int d0 = dims[0];
    const int d1 = rank > 1 ? dims[1] : 1;
    const int d2 = rank > 2 ? dims[2] : 1;
    for (index[0] = 0; index[0] < d0; ++index[0]) {
        for (index[1] = 0; index[1] < d1; ++index[1]) {
            for (index[2] = 0; index[2] < d2; ++index[2]) {
                visit(linear++, index);
            }
        }
    }
}

// ── Execution ────────────────────────────────────────────────────────────────

template <typename Scalar, typename Complex>
cufftResult exec_c2c_nd(const CufftPlanEntry& p, Complex* idata, Complex* odata,
                        int direction) {
    if (idata == nullptr || odata == nullptr) return CUFFT_INVALID_VALUE;
    const bool inverse = (direction == CUFFT_INVERSE);
    const int rank = p.rank;
    int dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) dims[a] = p.n[a];

    ComplexField<Scalar> field;
    field.resize(rank, dims);
    for (int b = 0; b < p.batch; ++b) {
        const long long in_base = static_cast<long long>(b) * p.input.dist;
        const long long out_base = static_cast<long long>(b) * p.output.dist;
        for_each_index(rank, dims, [&](std::size_t linear, const int* index) {
            const Complex& v = idata[in_base + layout_offset(p.input, rank, index)];
            field.re[linear] = static_cast<Scalar>(v.x);
            field.im[linear] = static_cast<Scalar>(v.y);
        });
        for (int a = 0; a < rank; ++a) transform_axis(field, a, inverse);
        for_each_index(rank, dims, [&](std::size_t linear, const int* index) {
            Complex& v = odata[out_base + layout_offset(p.output, rank, index)];
            v.x = field.re[linear];
            v.y = field.im[linear];
        });
    }
    return CUFFT_SUCCESS;
}

template <typename Scalar, typename Real, typename Complex>
cufftResult exec_r2c_nd(const CufftPlanEntry& p, Real* idata, Complex* odata) {
    if (idata == nullptr || odata == nullptr) return CUFFT_INVALID_VALUE;
    const int rank = p.rank;
    int real_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) real_dims[a] = p.n[a];
    int complex_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) complex_dims[a] = real_dims[a];
    complex_dims[rank - 1] = real_dims[rank - 1] / 2 + 1;

    ComplexField<Scalar> full;
    full.resize(rank, real_dims);
    ComplexField<Scalar> half;
    half.resize(rank, complex_dims);

    for (int b = 0; b < p.batch; ++b) {
        const long long in_base = static_cast<long long>(b) * p.input.dist;
        const long long out_base = static_cast<long long>(b) * p.output.dist;

        // Load the real grid as a complex field with zero imaginary part and
        // transform the fastest axis. Running a full complex DFT there rather
        // than a packed real one costs about 2x on that axis alone and keeps the
        // half-spectrum unpacking out of the picture entirely.
        std::fill(full.im.begin(), full.im.end(), Scalar{});
        for_each_index(rank, real_dims, [&](std::size_t linear, const int* index) {
            full.re[linear] =
                static_cast<Scalar>(idata[in_base + layout_offset(p.input, rank, index)]);
        });
        transform_axis(full, rank - 1, false);

        // Keep the non-redundant half, then transform the remaining axes on it.
        for_each_index(rank, complex_dims, [&](std::size_t linear, const int* index) {
            std::size_t source = index[0];
            for (int a = 1; a < rank; ++a) source = source * real_dims[a] + index[a];
            half.re[linear] = full.re[source];
            half.im[linear] = full.im[source];
        });
        for (int a = 0; a < rank - 1; ++a) transform_axis(half, a, false);

        for_each_index(rank, complex_dims, [&](std::size_t linear, const int* index) {
            Complex& v = odata[out_base + layout_offset(p.output, rank, index)];
            v.x = half.re[linear];
            v.y = half.im[linear];
        });
    }
    return CUFFT_SUCCESS;
}

template <typename Scalar, typename Real, typename Complex>
cufftResult exec_c2r_nd(const CufftPlanEntry& p, Complex* idata, Real* odata) {
    if (idata == nullptr || odata == nullptr) return CUFFT_INVALID_VALUE;
    const int rank = p.rank;
    int real_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) real_dims[a] = p.n[a];
    int complex_dims[3] = {1, 1, 1};
    for (int a = 0; a < rank; ++a) complex_dims[a] = real_dims[a];
    const int fast_real = real_dims[rank - 1];
    const int fast_complex = fast_real / 2 + 1;
    complex_dims[rank - 1] = fast_complex;

    ComplexField<Scalar> half;
    half.resize(rank, complex_dims);
    ComplexField<Scalar> full;
    full.resize(rank, real_dims);

    for (int b = 0; b < p.batch; ++b) {
        const long long in_base = static_cast<long long>(b) * p.input.dist;
        const long long out_base = static_cast<long long>(b) * p.output.dist;

        for_each_index(rank, complex_dims, [&](std::size_t linear, const int* index) {
            const Complex& v = idata[in_base + layout_offset(p.input, rank, index)];
            half.re[linear] = static_cast<Scalar>(v.x);
            half.im[linear] = static_cast<Scalar>(v.y);
        });

        // Inverting the slower axes first is what makes the Hermitian
        // reconstruction below a per-line mirror. The 3-D symmetry of a real
        // grid's spectrum couples (k0, k1, k2) to (-k0, -k1, -k2); once axes 0
        // and 1 are back in real space, each remaining line is the spectrum of a
        // real sequence on its own and mirrors within itself.
        for (int a = 0; a < rank - 1; ++a) transform_axis(half, a, true);

        for_each_index(rank, real_dims, [&](std::size_t linear, const int* index) {
            const int k = index[rank - 1];
            const bool mirrored = k >= fast_complex;
            const int source_k = mirrored ? fast_real - k : k;
            std::size_t source = index[0];
            for (int a = 1; a < rank - 1; ++a) source = source * complex_dims[a] + index[a];
            source = source * static_cast<std::size_t>(fast_complex) +
                     static_cast<std::size_t>(source_k);
            full.re[linear] = half.re[source];
            full.im[linear] = mirrored ? -half.im[source] : half.im[source];
        });
        transform_axis(full, rank - 1, true);

        for_each_index(rank, real_dims, [&](std::size_t linear, const int* index) {
            odata[out_base + layout_offset(p.output, rank, index)] =
                static_cast<Real>(full.re[linear]);
        });
    }
    return CUFFT_SUCCESS;
}

// ── Plan factory helper ───────────────────────────────────────────────────────

// Complex-side dimensions for a real transform; identical to `n` otherwise.
void complex_dims_for(cufftType type, int rank, const int* n, int* out) {
    for (int a = 0; a < rank; ++a) out[a] = n[a];
    if (type == CUFFT_R2C || type == CUFFT_C2R || type == CUFFT_D2Z || type == CUFFT_Z2D) {
        out[rank - 1] = n[rank - 1] / 2 + 1;
    }
}

bool layout_from_embed(FftLayout& layout, int rank, const int* embed, int stride, int dist,
                       const int* default_dims) {
    if (stride < 1) return false;
    for (int a = 0; a < rank; ++a) {
        layout.embed[a] = (embed != nullptr) ? embed[a] : default_dims[a];
        // embed[0] is not used for addressing, so cuFFT does not constrain it.
        if (a > 0 && layout.embed[a] < default_dims[a]) return false;
    }
    layout.stride = stride;
    if (dist > 0) {
        layout.dist = dist;
    } else {
        std::size_t span = 1;
        for (int a = 0; a < rank; ++a) {
            if (static_cast<std::size_t>(layout.embed[a]) >
                std::numeric_limits<std::size_t>::max() / span) {
                return false;
            }
            span *= static_cast<std::size_t>(layout.embed[a]);
        }
        if (static_cast<std::size_t>(stride) >
            std::numeric_limits<std::size_t>::max() / span) {
            return false;
        }
        span *= static_cast<std::size_t>(stride);
        if (span > static_cast<std::size_t>(std::numeric_limits<long long>::max())) {
            return false;
        }
        layout.dist = static_cast<long long>(span);
    }
    return true;
}

bool layout_element_count(const FftLayout& layout, int rank, const int* dims, int batch,
                          std::size_t* count) {
    std::size_t offset = 0;
    for (int axis = 0; axis < rank; ++axis) {
        if (axis > 0) {
            if (offset != 0 && static_cast<std::size_t>(layout.embed[axis]) >
                                   std::numeric_limits<std::size_t>::max() / offset) {
                return false;
            }
            offset *= static_cast<std::size_t>(layout.embed[axis]);
        }
        const std::size_t coordinate = static_cast<std::size_t>(dims[axis] - 1);
        if (coordinate > std::numeric_limits<std::size_t>::max() - offset) return false;
        offset += coordinate;
    }
    if (offset != 0 && static_cast<std::size_t>(layout.stride) >
                           std::numeric_limits<std::size_t>::max() / offset) {
        return false;
    }
    offset *= static_cast<std::size_t>(layout.stride);

    const std::size_t batch_offset = static_cast<std::size_t>(batch - 1);
    if (static_cast<std::size_t>(layout.dist) > 0 &&
        batch_offset > std::numeric_limits<std::size_t>::max() /
                           static_cast<std::size_t>(layout.dist)) {
        return false;
    }
    const std::size_t batch_base = batch_offset * static_cast<std::size_t>(layout.dist);
    if (offset >= std::numeric_limits<std::size_t>::max() - batch_base) return false;
    *count = batch_base + offset + 1;
    return true;
}

void prepare_metal_plan(const CufftPlanEntry& plan);

static cufftResult make_plan(cufftHandle h, int rank, const int* n, cufftType type, int batch,
                             const int* inembed, int istride, int idist,
                             const int* onembed, int ostride, int odist,
                             size_t* workSize) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(h);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (n == nullptr || rank < 1 || rank > 3 || batch < 1) {
        return CUFFT_INVALID_VALUE;
    }
    switch (type) {
        case CUFFT_R2C:
        case CUFFT_C2R:
        case CUFFT_C2C:
        case CUFFT_D2Z:
        case CUFFT_Z2D:
        case CUFFT_Z2Z: break;
        default: return CUFFT_INVALID_TYPE;
    }
    std::size_t total_elements = 1;
    for (int i = 0; i < rank; ++i) {
        if (n[i] < 1 || static_cast<std::size_t>(n[i]) >
                            std::numeric_limits<std::size_t>::max() / total_elements) {
            return CUFFT_INVALID_SIZE;
        }
        total_elements *= static_cast<std::size_t>(n[i]);
    }

    // The two sides of a real transform have different fastest-axis extents, so
    // each layout is validated against the domain it actually describes.
    int complex_dims[3] = {1, 1, 1};
    complex_dims_for(type, rank, n, complex_dims);
    const bool input_is_real = (type == CUFFT_R2C || type == CUFFT_D2Z);
    const bool output_is_real = (type == CUFFT_C2R || type == CUFFT_Z2D);
    const int* input_dims = input_is_real ? n : complex_dims;
    const int* output_dims = output_is_real ? n : complex_dims;

    FftLayout input;
    FftLayout output;
    if (!layout_from_embed(input, rank, inembed, istride, idist, input_dims) ||
        !layout_from_embed(output, rank, onembed, ostride, odist, output_dims)) {
        return CUFFT_INVALID_VALUE;
    }
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    if (!layout_element_count(input, rank, input_dims, batch, &input_elements) ||
        !layout_element_count(output, rank, output_dims, batch, &output_elements)) {
        return CUFFT_INVALID_SIZE;
    }

    p->type = type;
    p->rank = rank;
    p->n.assign(n, n + rank);
    p->batch = batch;
    p->input = input;
    p->output = output;
    p->input_elements = input_elements;
    p->output_elements = output_elements;
    if (workSize != nullptr) {
        *workSize = 0;
    }
    prepare_metal_plan(*p);
    return CUFFT_SUCCESS;
}

cufftResult synchronize_plan_stream(const CufftPlanEntry& plan) {
    return cudaStreamSynchronize(plan.stream) == cudaSuccess ? CUFFT_SUCCESS
                                                              : CUFFT_EXEC_FAILED;
}

cufftResult validate_execution_buffers(const CufftPlanEntry& plan, const void* input,
                                       std::size_t input_element_size, const void* output,
                                       std::size_t output_element_size) {
    cumetal::rt::AllocationTable::ResolvedAllocation input_allocation;
    cumetal::rt::AllocationTable::ResolvedAllocation output_allocation;
    if (!cumetal::rt::resolve_allocation_for_pointer(input, &input_allocation) ||
        input_allocation.kind != cumetal::rt::AllocationKind::kDevice ||
        !cumetal::rt::resolve_allocation_for_pointer(output, &output_allocation) ||
        output_allocation.kind != cumetal::rt::AllocationKind::kDevice) {
        return CUFFT_INVALID_VALUE;
    }
    if (plan.input_elements > input_allocation.remaining_size / input_element_size ||
        plan.output_elements > output_allocation.remaining_size / output_element_size) {
        return CUFFT_INVALID_VALUE;
    }
    return CUFFT_SUCCESS;
}

// ── Metal dispatch ───────────────────────────────────────────────────────────
//
// Single precision only. Metal has no FP64, and running the double entry points
// through the emulated pair would cost more than the host transform they would
// replace, so those keep the CPU implementation below.
//
// Declining here is always safe: the CPU path computes the same transform, and
// nothing has been written to the output when this returns false.
fft_metal::Layout to_metal_layout(const FftLayout& layout) {
    fft_metal::Layout out;
    for (int a = 0; a < 3; ++a) out.embed[a] = layout.embed[a];
    out.stride = layout.stride;
    out.dist = layout.dist;
    return out;
}

bool try_exec_metal(const CufftPlanEntry& p, fft_metal::Kind kind, const void* idata,
                    void* odata, bool inverse) {
    fft_metal::Request request;
    request.kind = kind;
    request.rank = p.rank;
    for (int a = 0; a < p.rank; ++a) request.n[a] = p.n[a];
    request.batch = p.batch;
    request.input = to_metal_layout(p.input);
    request.output = to_metal_layout(p.output);
    request.inverse = inverse;
    request.stream = p.stream;
    request.idata = idata;
    request.odata = odata;
    return fft_metal::execute(request);
}

void prepare_metal_plan(const CufftPlanEntry& p) {
    if (p.type != CUFFT_R2C && p.type != CUFFT_C2R) return;
    fft_metal::Request request;
    request.kind = p.type == CUFFT_R2C ? fft_metal::Kind::kR2C : fft_metal::Kind::kC2R;
    request.rank = p.rank;
    for (int axis = 0; axis < p.rank; ++axis) request.n[axis] = p.n[axis];
    request.batch = p.batch;
    request.input = to_metal_layout(p.input);
    request.output = to_metal_layout(p.output);
    request.stream = p.stream;
    (void)fft_metal::prepare(request);
}

}  // namespace

// ── Public API ────────────────────────────────────────────────────────────────

extern "C" {

cufftResult cufftGetVersion(int* version) {
    if (version == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    *version = 10500;  // report cuFFT 10.5 (CUDA 11.x series)
    return CUFFT_SUCCESS;
}

cufftResult cufftCreate(cufftHandle* plan) {
    if (plan == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    const cufftHandle h = s.next_handle++;
    s.plans[h] = CufftPlanEntry{};
    *plan = h;
    return CUFFT_SUCCESS;
}

cufftResult cufftDestroy(cufftHandle plan) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    auto it = s.plans.find(plan);
    if (it == s.plans.end()) {
        return CUFFT_INVALID_PLAN;
    }
    // vDSP setups are cached by length across plans, not owned by one, so there
    // is nothing here to destroy.
    s.plans.erase(it);
    return CUFFT_SUCCESS;
}

cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    p->stream = stream;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetSize(cufftHandle plan, size_t* workSize) {
    if (workSize == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.get(plan) == nullptr) return CUFFT_INVALID_PLAN;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

// ── Plan creation helpers ─────────────────────────────────────────────────────

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    int n[1] = {nx};
    r = make_plan(*plan, 1, n, type, batch, nullptr, 1, 0, nullptr, 1, 0, nullptr);
    if (r != CUFFT_SUCCESS) {
        cufftDestroy(*plan);
        *plan = 0;
    }
    return r;
}

cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    int n[2] = {nx, ny};
    r = make_plan(*plan, 2, n, type, 1, nullptr, 1, 0, nullptr, 1, 0, nullptr);
    if (r != CUFFT_SUCCESS) {
        cufftDestroy(*plan);
        *plan = 0;
    }
    return r;
}

cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    int n[3] = {nx, ny, nz};
    r = make_plan(*plan, 3, n, type, 1, nullptr, 1, 0, nullptr, 1, 0, nullptr);
    if (r != CUFFT_SUCCESS) {
        cufftDestroy(*plan);
        *plan = 0;
    }
    return r;
}

cufftResult cufftPlanMany(cufftHandle* plan,
                           int rank,
                           int* n,
                           int* inembed,
                           int istride,
                           int idist,
                           int* onembed,
                           int ostride,
                           int odist,
                           cufftType type,
                           int batch) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    r = make_plan(*plan, rank, n, type, batch, inembed, istride, idist, onembed,
                  ostride, odist, nullptr);
    if (r != CUFFT_SUCCESS) {
        cufftDestroy(*plan);
        *plan = 0;
    }
    return r;
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                              size_t* workSize) {
    int n[1] = {nx};
    return make_plan(plan, 1, n, type, batch, nullptr, 1, 0, nullptr, 1, 0, workSize);
}

cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                              size_t* workSize) {
    int n[2] = {nx, ny};
    return make_plan(plan, 2, n, type, 1, nullptr, 1, 0, nullptr, 1, 0, workSize);
}

cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                              size_t* workSize) {
    int n[3] = {nx, ny, nz};
    return make_plan(plan, 3, n, type, 1, nullptr, 1, 0, nullptr, 1, 0, workSize);
}

cufftResult cufftMakePlanMany(cufftHandle plan,
                               int rank,
                               int* n,
                               int* inembed,
                               int istride,
                               int idist,
                               int* onembed,
                               int ostride,
                               int odist,
                               cufftType type,
                               int batch,
                               size_t* workSize) {
    return make_plan(plan, rank, n, type, batch, inembed, istride, idist, onembed,
                     ostride, odist, workSize);
}

cufftResult cufftXtMakePlanMany(cufftHandle plan,
                                 int rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int istride,
                                 long long int idist,
                                 cudaDataType inputtype,
                                 long long int* onembed,
                                 long long int ostride,
                                 long long int odist,
                                 cudaDataType outputtype,
                                 long long int batch,
                                 size_t* workSize,
                                 cudaDataType executiontype) {
    if (n == nullptr || workSize == nullptr || rank < 1 || rank > 3 || batch < 1) {
        return CUFFT_INVALID_VALUE;
    }

    cufftType type = CUFFT_C2C;
    if (inputtype == CUDA_C_32F && outputtype == CUDA_C_32F &&
        executiontype == CUDA_C_32F) {
        type = CUFFT_C2C;
    } else if (inputtype == CUDA_R_32F && outputtype == CUDA_C_32F &&
               executiontype == CUDA_C_32F) {
        type = CUFFT_R2C;
    } else if (inputtype == CUDA_C_32F && outputtype == CUDA_R_32F &&
               executiontype == CUDA_C_32F) {
        type = CUFFT_C2R;
    } else if (inputtype == CUDA_C_64F && outputtype == CUDA_C_64F &&
               executiontype == CUDA_C_64F) {
        type = CUFFT_Z2Z;
    } else if (inputtype == CUDA_R_64F && outputtype == CUDA_C_64F &&
               executiontype == CUDA_C_64F) {
        type = CUFFT_D2Z;
    } else if (inputtype == CUDA_C_64F && outputtype == CUDA_R_64F &&
               executiontype == CUDA_C_64F) {
        type = CUFFT_Z2D;
    } else {
        return CUFFT_NOT_SUPPORTED;
    }

    if (batch > std::numeric_limits<int>::max()) return CUFFT_INVALID_SIZE;
    std::vector<int> dimensions;
    dimensions.reserve(static_cast<std::size_t>(rank));
    for (int i = 0; i < rank; ++i) {
        if (n[i] < 1 || n[i] > std::numeric_limits<int>::max()) {
            return CUFFT_INVALID_SIZE;
        }
        dimensions.push_back(static_cast<int>(n[i]));
    }

    // The Xt entry point takes 64-bit extents; the plan stores 32-bit ones, so a
    // layout that does not fit is rejected rather than silently truncated.
    int in_embed[3] = {0, 0, 0};
    int out_embed[3] = {0, 0, 0};
    auto narrow_embed = [&](const long long* wide, int* narrow) -> bool {
        if (wide == nullptr) return true;
        for (int i = 0; i < rank; ++i) {
            if (wide[i] < 1 || wide[i] > std::numeric_limits<int>::max()) return false;
            narrow[i] = static_cast<int>(wide[i]);
        }
        return true;
    };
    if (!narrow_embed(inembed, in_embed) || !narrow_embed(onembed, out_embed)) {
        return CUFFT_INVALID_SIZE;
    }
    if (istride < 1 || istride > std::numeric_limits<int>::max() || idist < 0 ||
        idist > std::numeric_limits<int>::max() || ostride < 1 ||
        ostride > std::numeric_limits<int>::max() || odist < 0 ||
        odist > std::numeric_limits<int>::max()) {
        return CUFFT_INVALID_SIZE;
    }

    return make_plan(plan, rank, dimensions.data(), type, static_cast<int>(batch),
                     inembed != nullptr ? in_embed : nullptr, static_cast<int>(istride),
                     static_cast<int>(idist),
                     onembed != nullptr ? out_embed : nullptr, static_cast<int>(ostride),
                     static_cast<int>(odist), workSize);
}

// ── Execute ───────────────────────────────────────────────────────────────────

cufftResult cufftExecC2C(cufftHandle plan, cufftComplex* idata, cufftComplex* odata,
                          int direction) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_C2C) {
        return CUFFT_INVALID_TYPE;
    }
    if (direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) {
        return CUFFT_INVALID_VALUE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftComplex), odata, sizeof(cufftComplex));
    if (validation != CUFFT_SUCCESS) return validation;
    if (try_exec_metal(*p, fft_metal::Kind::kC2C, idata, odata,
                       direction == CUFFT_INVERSE)) {
        return CUFFT_SUCCESS;
    }
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2c_nd<float, cufftComplex>(*p, idata, odata, direction);
}

cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_R2C) {
        return CUFFT_INVALID_TYPE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftReal), odata, sizeof(cufftComplex));
    if (validation != CUFFT_SUCCESS) return validation;
    if (try_exec_metal(*p, fft_metal::Kind::kR2C, idata, odata, false)) {
        return CUFFT_SUCCESS;
    }
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_r2c_nd<float, cufftReal, cufftComplex>(*p, idata, odata);
}

cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_C2R) {
        return CUFFT_INVALID_TYPE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftComplex), odata, sizeof(cufftReal));
    if (validation != CUFFT_SUCCESS) return validation;
    if (try_exec_metal(*p, fft_metal::Kind::kC2R, idata, odata, true)) {
        return CUFFT_SUCCESS;
    }
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2r_nd<float, cufftReal, cufftComplex>(*p, idata, odata);
}

cufftResult cufftExecZ2Z(cufftHandle plan, cufftDoubleComplex* idata,
                          cufftDoubleComplex* odata, int direction) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_Z2Z) {
        return CUFFT_INVALID_TYPE;
    }
    if (direction != CUFFT_FORWARD && direction != CUFFT_INVERSE) {
        return CUFFT_INVALID_VALUE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftDoubleComplex), odata, sizeof(cufftDoubleComplex));
    if (validation != CUFFT_SUCCESS) return validation;
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2c_nd<double, cufftDoubleComplex>(*p, idata, odata, direction);
}

cufftResult cufftExecD2Z(cufftHandle plan, cufftDoubleReal* idata,
                          cufftDoubleComplex* odata) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_D2Z) {
        return CUFFT_INVALID_TYPE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftDoubleReal), odata, sizeof(cufftDoubleComplex));
    if (validation != CUFFT_SUCCESS) return validation;
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_r2c_nd<double, cufftDoubleReal, cufftDoubleComplex>(*p, idata, odata);
}

cufftResult cufftExecZ2D(cufftHandle plan, cufftDoubleComplex* idata,
                          cufftDoubleReal* odata) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    CufftPlanEntry* p = s.get(plan);
    if (p == nullptr) {
        return CUFFT_INVALID_PLAN;
    }
    if (p->type != CUFFT_Z2D) {
        return CUFFT_INVALID_TYPE;
    }
    cufftResult validation = validate_execution_buffers(
        *p, idata, sizeof(cufftDoubleComplex), odata, sizeof(cufftDoubleReal));
    if (validation != CUFFT_SUCCESS) return validation;
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2r_nd<double, cufftDoubleReal, cufftDoubleComplex>(*p, idata, odata);
}

// SetWorkArea — on UMA, vDSP manages its own scratch; no external workspace is used.
cufftResult cufftSetWorkArea(cufftHandle plan, void* /*workArea*/) {
    CufftState& s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.get(plan) == nullptr) return CUFFT_INVALID_PLAN;
    return CUFFT_SUCCESS;
}

// ── Estimate* ────────────────────────────────────────────────────────────────
// Return a conservative upper-bound on scratch memory without building a full plan.
// On this implementation vDSP manages scratch internally, so 0 is a valid answer,
// but we return a small non-zero estimate to satisfy callers that test workSize > 0.

static bool valid_transform_type(cufftType type) {
    return type == CUFFT_R2C || type == CUFFT_C2R || type == CUFFT_C2C ||
           type == CUFFT_D2Z || type == CUFFT_Z2D || type == CUFFT_Z2Z;
}

static bool estimate_work_bytes(size_t total_complex_elements, cufftType type,
                                size_t* work_size) {
    // Two split-real buffers of float/double per element, plus vDSP setup overhead.
    size_t elem_bytes = (type == CUFFT_D2Z || type == CUFFT_Z2D || type == CUFFT_Z2Z)
                            ? sizeof(double)
                            : sizeof(float);
    const size_t bytes_per_element = elem_bytes * 2;
    if (total_complex_elements >
        (std::numeric_limits<size_t>::max() - 4096) / bytes_per_element) {
        return false;
    }
    *work_size = total_complex_elements * bytes_per_element + 4096;
    return true;
}

cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || batch <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    if (!valid_transform_type(type)) return CUFFT_INVALID_TYPE;
    if (static_cast<size_t>(batch) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(nx)) {
        return CUFFT_INVALID_SIZE;
    }
    const size_t total = static_cast<size_t>(nx) * static_cast<size_t>(batch);
    if (!estimate_work_bytes(total, type, workSize)) return CUFFT_INVALID_SIZE;
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || ny <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    if (!valid_transform_type(type)) return CUFFT_INVALID_TYPE;
    const size_t x = static_cast<size_t>(nx);
    if (static_cast<size_t>(ny) > std::numeric_limits<size_t>::max() / x) {
        return CUFFT_INVALID_SIZE;
    }
    if (!estimate_work_bytes(x * static_cast<size_t>(ny), type, workSize)) {
        return CUFFT_INVALID_SIZE;
    }
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || ny <= 0 || nz <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    if (!valid_transform_type(type)) return CUFFT_INVALID_TYPE;
    size_t total = static_cast<size_t>(nx);
    for (int dimension : {ny, nz}) {
        if (static_cast<size_t>(dimension) >
            std::numeric_limits<size_t>::max() / total) {
            return CUFFT_INVALID_SIZE;
        }
        total *= static_cast<size_t>(dimension);
    }
    if (!estimate_work_bytes(total, type, workSize)) return CUFFT_INVALID_SIZE;
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimateMany(int rank, int* n,
                               int* /*inembed*/, int /*istride*/, int /*idist*/,
                               int* /*onembed*/, int /*ostride*/, int /*odist*/,
                               cufftType type, int batch, size_t* workSize) {
    if (workSize == nullptr || n == nullptr || rank < 1 || rank > 3 || batch <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    if (!valid_transform_type(type)) return CUFFT_INVALID_TYPE;
    size_t total = static_cast<size_t>(batch);
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) return CUFFT_INVALID_VALUE;
        if (static_cast<size_t>(n[i]) > std::numeric_limits<size_t>::max() / total) {
            return CUFFT_INVALID_SIZE;
        }
        total *= static_cast<size_t>(n[i]);
    }
    if (!estimate_work_bytes(total, type, workSize)) return CUFFT_INVALID_SIZE;
    return CUFFT_SUCCESS;
}

cufftResult cufftGetProperty(libraryPropertyType type, int* value) {
    if (value == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    switch (type) {
        case MAJOR_VERSION: *value = 10; break;
        case MINOR_VERSION: *value = 5;  break;
        case PATCH_LEVEL:   *value = 0;  break;
        default:
            return CUFFT_INVALID_VALUE;
    }
    return CUFFT_SUCCESS;
}

}  // extern "C"
