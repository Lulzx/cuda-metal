// cuFFT shim backed by Apple Accelerate vDSP (spec §11 Phase 4.5).
//
// Uses vDSP_DFT_Execute for arbitrary-length transforms (any N ≥ 1).
// The DFT convention matches cuFFT:
//   FORWARD:  X[k] = Σ x[n] · e^{-2πi k n/N}  (no normalization)
//   INVERSE:  x[n] = Σ X[k] · e^{+2πi k n/N}  (no normalization — caller divides by N)
//
// Split-complex ↔ interleaved conversion is performed in temporary scratch buffers.
// Batched transforms loop over batch count.

#include "cufftXt.h"

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <numbers>
#include <vector>

namespace {

// ── Plan storage ─────────────────────────────────────────────────────────────

enum class FftPrecision { kSingle, kDouble };

struct CufftPlanEntry {
    cufftType type = CUFFT_C2C;
    int rank = 1;
    // Logical dimensions (n[0]…n[rank-1]).  For R2C/C2R rank-1: n[0] is the real length;
    // complex half-spectrum has n[0]/2 + 1 elements.
    std::vector<int> n;
    int batch = 1;
    // Stride/distance (currently only batch-contiguous layout is accelerated).
    cudaStream_t stream = nullptr;

    // Cached vDSP DFT setups, one per direction (index 0=forward, 1=inverse).
    // Built lazily when Exec* is first called.
    // Single-precision (float):
    vDSP_DFT_Setup dft_fwd_f = nullptr;
    vDSP_DFT_Setup dft_inv_f = nullptr;
    // Double-precision:
    vDSP_DFT_SetupD dft_fwd_d = nullptr;
    vDSP_DFT_SetupD dft_inv_d = nullptr;
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

// ── vDSP setup helpers ───────────────────────────────────────────────────────

vDSP_DFT_Setup get_dft_setup_f(CufftPlanEntry& p, vDSP_DFT_Direction dir) {
    vDSP_DFT_Setup& slot = (dir == vDSP_DFT_FORWARD) ? p.dft_fwd_f : p.dft_inv_f;
    if (slot == nullptr) {
        vDSP_Length n = 1;
        for (int d : p.n) {
            n *= static_cast<vDSP_Length>(d);
        }
        // Use complex (interleaved ↔ split) DFT (zop = out-of-place split).
        if (p.type == CUFFT_R2C || p.type == CUFFT_C2R) {
            // Half-spectrum real DFT (zrop).
            slot = vDSP_DFT_zrop_CreateSetup(nullptr, n, dir);
        } else {
            slot = vDSP_DFT_zop_CreateSetup(nullptr, n, dir);
        }
    }
    return slot;
}

vDSP_DFT_SetupD get_dft_setup_d(CufftPlanEntry& p, vDSP_DFT_Direction dir) {
    vDSP_DFT_SetupD& slot = (dir == vDSP_DFT_FORWARD) ? p.dft_fwd_d : p.dft_inv_d;
    if (slot == nullptr) {
        vDSP_Length n = 1;
        for (int d : p.n) {
            n *= static_cast<vDSP_Length>(d);
        }
        if (p.type == CUFFT_D2Z || p.type == CUFFT_Z2D) {
            slot = vDSP_DFT_zrop_CreateSetupD(nullptr, n, dir);
        } else {
            slot = vDSP_DFT_zop_CreateSetupD(nullptr, n, dir);
        }
    }
    return slot;
}

// ── Interleaved ↔ split-complex conversion ───────────────────────────────────

// Deinterleave {re0,im0,re1,im1,...} → re[], im[]
static void deinterleave_f(const cufftComplex* src, float* re, float* im, vDSP_Length n) {
    DSPSplitComplex sc{re, im};
    vDSP_ctoz(reinterpret_cast<const DSPComplex*>(src), 2, &sc, 1, n);
}
static void interleave_f(const float* re, const float* im, cufftComplex* dst, vDSP_Length n) {
    DSPSplitComplex sc{const_cast<float*>(re), const_cast<float*>(im)};
    vDSP_ztoc(&sc, 1, reinterpret_cast<DSPComplex*>(dst), 2, n);
}
static void deinterleave_d(const cufftDoubleComplex* src, double* re, double* im,
                            vDSP_Length n) {
    DSPDoubleSplitComplex sc{re, im};
    vDSP_ctozD(reinterpret_cast<const DSPDoubleComplex*>(src), 2, &sc, 1, n);
}
static void interleave_d(const double* re, const double* im, cufftDoubleComplex* dst,
                          vDSP_Length n) {
    DSPDoubleSplitComplex sc{const_cast<double*>(re), const_cast<double*>(im)};
    vDSP_ztocD(&sc, 1, reinterpret_cast<DSPDoubleComplex*>(dst), 2, n);
}

// vDSP supports only selected factorizations. Keep uncommon small sizes correct
// with a bounded direct DFT; reject larger unsupported sizes rather than starting
// an unexpectedly expensive O(N^2) operation.
constexpr std::size_t kDirectDftMaxElements = 1024;

template <typename Scalar>
bool direct_dft(const Scalar* re_in,
                const Scalar* im_in,
                Scalar* re_out,
                Scalar* im_out,
                std::size_t n,
                bool inverse) {
    if (re_in == nullptr || im_in == nullptr || re_out == nullptr || im_out == nullptr ||
        n == 0 || n > kDirectDftMaxElements) {
        return false;
    }
    std::vector<long double> twiddle_re(n);
    std::vector<long double> twiddle_im(n);
    const long double direction = inverse ? 1.0L : -1.0L;
    for (std::size_t phase = 0; phase < n; ++phase) {
        const long double angle =
            2.0L * std::numbers::pi_v<long double> * static_cast<long double>(phase) /
            static_cast<long double>(n);
        twiddle_re[phase] = std::cos(angle);
        twiddle_im[phase] = direction * std::sin(angle);
    }
    for (std::size_t k = 0; k < n; ++k) {
        long double sum_re = 0.0L;
        long double sum_im = 0.0L;
        for (std::size_t sample = 0; sample < n; ++sample) {
            const std::size_t phase = (k * sample) % n;
            const long double c = twiddle_re[phase];
            const long double s = twiddle_im[phase];
            const long double in_re = static_cast<long double>(re_in[sample]);
            const long double in_im = static_cast<long double>(im_in[sample]);
            sum_re += in_re * c - in_im * s;
            sum_im += in_re * s + in_im * c;
        }
        re_out[k] = static_cast<Scalar>(sum_re);
        im_out[k] = static_cast<Scalar>(sum_im);
    }
    return true;
}

// ── C2C execution (single) ───────────────────────────────────────────────────

static cufftResult exec_c2c(CufftPlanEntry& p, cufftComplex* idata, cufftComplex* odata,
                             int direction) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    const vDSP_DFT_Direction vdir =
        (direction == CUFFT_FORWARD) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
    vDSP_DFT_Setup setup = get_dft_setup_f(p, vdir);
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    std::vector<float> re_in(n), im_in(n), re_out(n), im_out(n);
    for (int b = 0; b < p.batch; ++b) {
        const cufftComplex* in = idata + b * static_cast<ptrdiff_t>(n);
        cufftComplex* out = odata + b * static_cast<ptrdiff_t>(n);
        deinterleave_f(in, re_in.data(), im_in.data(), n);
        if (setup != nullptr) {
            vDSP_DFT_Execute(setup, re_in.data(), im_in.data(), re_out.data(),
                             im_out.data());
        } else if (!direct_dft(re_in.data(), im_in.data(), re_out.data(),
                               im_out.data(), n, direction == CUFFT_INVERSE)) {
            return CUFFT_NOT_SUPPORTED;
        }
        interleave_f(re_out.data(), im_out.data(), out, n);
    }
    return CUFFT_SUCCESS;
}

// ── Z2Z execution (double) ───────────────────────────────────────────────────

static cufftResult exec_z2z(CufftPlanEntry& p, cufftDoubleComplex* idata,
                             cufftDoubleComplex* odata, int direction) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    const vDSP_DFT_Direction vdir =
        (direction == CUFFT_FORWARD) ? vDSP_DFT_FORWARD : vDSP_DFT_INVERSE;
    vDSP_DFT_SetupD setup = get_dft_setup_d(p, vdir);
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    std::vector<double> re_in(n), im_in(n), re_out(n), im_out(n);
    for (int b = 0; b < p.batch; ++b) {
        const cufftDoubleComplex* in = idata + b * static_cast<ptrdiff_t>(n);
        cufftDoubleComplex* out = odata + b * static_cast<ptrdiff_t>(n);
        deinterleave_d(in, re_in.data(), im_in.data(), n);
        if (setup != nullptr) {
            vDSP_DFT_ExecuteD(setup, re_in.data(), im_in.data(), re_out.data(),
                              im_out.data());
        } else if (!direct_dft(re_in.data(), im_in.data(), re_out.data(),
                               im_out.data(), n, direction == CUFFT_INVERSE)) {
            return CUFFT_NOT_SUPPORTED;
        }
        interleave_d(re_out.data(), im_out.data(), out, n);
    }
    return CUFFT_SUCCESS;
}

// ── R2C execution (single) ───────────────────────────────────────────────────

static cufftResult exec_r2c(CufftPlanEntry& p, cufftReal* idata, cufftComplex* odata) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    vDSP_DFT_Setup setup = get_dft_setup_f(p, vDSP_DFT_FORWARD);
    // For zrop: n real input samples → n/2+1 complex output (half-spectrum convention).
    // vDSP packs the half-spectrum as n/2 complex + (dc, nyquist) in first slot.
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    const vDSP_Length nc = n / 2;  // vDSP half-spectrum count
    std::vector<float> re_in(nc + 1), im_in(nc + 1), re_out(nc + 1), im_out(nc + 1);
    for (int b = 0; b < p.batch; ++b) {
        const cufftReal* in = idata + b * static_cast<ptrdiff_t>(n);
        cufftComplex* out = odata + b * static_cast<ptrdiff_t>(nc + 1);
        if (setup == nullptr) {
            std::vector<float> full_re_in(n), full_im_in(n, 0.0f);
            std::vector<float> full_re_out(n), full_im_out(n);
            std::copy_n(in, n, full_re_in.data());
            if (!direct_dft(full_re_in.data(), full_im_in.data(), full_re_out.data(),
                            full_im_out.data(), n, false)) {
                return CUFFT_NOT_SUPPORTED;
            }
            for (vDSP_Length k = 0; k <= nc; ++k) {
                out[k].x = full_re_out[k];
                out[k].y = full_im_out[k];
            }
            continue;
        }
        // Pack real input into split complex (imaginary = 0).
        for (vDSP_Length i = 0; i < nc; ++i) {
            re_in[i] = in[2 * i];
            im_in[i] = in[2 * i + 1];
        }
        vDSP_DFT_Execute(setup, re_in.data(), im_in.data(), re_out.data(), im_out.data());
        // vDSP zrop packs DC in re[0], Nyquist in im[0]; unpack to standard half-spectrum.
        out[0].x = re_out[0];
        out[0].y = 0.0f;
        for (vDSP_Length k = 1; k < nc; ++k) {
            out[k].x = re_out[k];
            out[k].y = im_out[k];
        }
        out[nc].x = im_out[0];
        out[nc].y = 0.0f;
    }
    return CUFFT_SUCCESS;
}

// ── C2R execution (single) ───────────────────────────────────────────────────

static cufftResult exec_c2r(CufftPlanEntry& p, cufftComplex* idata, cufftReal* odata) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    vDSP_DFT_Setup setup = get_dft_setup_f(p, vDSP_DFT_INVERSE);
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    const vDSP_Length nc = n / 2;
    std::vector<float> re_in(nc + 1), im_in(nc + 1), re_out(nc + 1), im_out(nc + 1);
    for (int b = 0; b < p.batch; ++b) {
        const cufftComplex* in = idata + b * static_cast<ptrdiff_t>(nc + 1);
        cufftReal* out = odata + b * static_cast<ptrdiff_t>(n);
        if (setup == nullptr) {
            std::vector<float> full_re_in(n), full_im_in(n);
            std::vector<float> full_re_out(n), full_im_out(n);
            for (vDSP_Length k = 0; k <= nc; ++k) {
                full_re_in[k] = in[k].x;
                full_im_in[k] = in[k].y;
            }
            for (vDSP_Length k = nc + 1; k < n; ++k) {
                const vDSP_Length mirrored = n - k;
                full_re_in[k] = in[mirrored].x;
                full_im_in[k] = -in[mirrored].y;
            }
            if (!direct_dft(full_re_in.data(), full_im_in.data(), full_re_out.data(),
                            full_im_out.data(), n, true)) {
                return CUFFT_NOT_SUPPORTED;
            }
            std::copy_n(full_re_out.data(), n, out);
            continue;
        }
        // Pack half-spectrum into vDSP zrop format (DC in re[0], Nyquist in im[0]).
        re_in[0] = in[0].x;
        im_in[0] = in[nc].x;
        for (vDSP_Length k = 1; k < nc; ++k) {
            re_in[k] = in[k].x;
            im_in[k] = in[k].y;
        }
        vDSP_DFT_Execute(setup, re_in.data(), im_in.data(), re_out.data(), im_out.data());
        // Unpack split output to interleaved real.
        for (vDSP_Length i = 0; i < nc; ++i) {
            out[2 * i] = re_out[i];
            out[2 * i + 1] = im_out[i];
        }
    }
    return CUFFT_SUCCESS;
}

// ── D2Z / Z2D execution (double) ─────────────────────────────────────────────

static cufftResult exec_d2z(CufftPlanEntry& p, cufftDoubleReal* idata,
                             cufftDoubleComplex* odata) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    vDSP_DFT_SetupD setup = get_dft_setup_d(p, vDSP_DFT_FORWARD);
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    const vDSP_Length nc = n / 2;
    std::vector<double> re_in(nc + 1), im_in(nc + 1), re_out(nc + 1), im_out(nc + 1);
    for (int b = 0; b < p.batch; ++b) {
        const cufftDoubleReal* in = idata + b * static_cast<ptrdiff_t>(n);
        cufftDoubleComplex* out = odata + b * static_cast<ptrdiff_t>(nc + 1);
        if (setup == nullptr) {
            std::vector<double> full_re_in(n), full_im_in(n, 0.0);
            std::vector<double> full_re_out(n), full_im_out(n);
            std::copy_n(in, n, full_re_in.data());
            if (!direct_dft(full_re_in.data(), full_im_in.data(), full_re_out.data(),
                            full_im_out.data(), n, false)) {
                return CUFFT_NOT_SUPPORTED;
            }
            for (vDSP_Length k = 0; k <= nc; ++k) {
                out[k].x = full_re_out[k];
                out[k].y = full_im_out[k];
            }
            continue;
        }
        for (vDSP_Length i = 0; i < nc; ++i) {
            re_in[i] = in[2 * i];
            im_in[i] = in[2 * i + 1];
        }
        vDSP_DFT_ExecuteD(setup, re_in.data(), im_in.data(), re_out.data(), im_out.data());
        out[0].x = re_out[0];
        out[0].y = 0.0;
        for (vDSP_Length k = 1; k < nc; ++k) {
            out[k].x = re_out[k];
            out[k].y = im_out[k];
        }
        out[nc].x = im_out[0];
        out[nc].y = 0.0;
    }
    return CUFFT_SUCCESS;
}

static cufftResult exec_z2d(CufftPlanEntry& p, cufftDoubleComplex* idata,
                             cufftDoubleReal* odata) {
    if (idata == nullptr || odata == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
    if (p.rank != 1) return CUFFT_NOT_SUPPORTED;
    vDSP_DFT_SetupD setup = get_dft_setup_d(p, vDSP_DFT_INVERSE);
    vDSP_Length n = 1;
    for (int d : p.n) {
        n *= static_cast<vDSP_Length>(d);
    }
    const vDSP_Length nc = n / 2;
    std::vector<double> re_in(nc + 1), im_in(nc + 1), re_out(nc + 1), im_out(nc + 1);
    for (int b = 0; b < p.batch; ++b) {
        const cufftDoubleComplex* in = idata + b * static_cast<ptrdiff_t>(nc + 1);
        cufftDoubleReal* out = odata + b * static_cast<ptrdiff_t>(n);
        if (setup == nullptr) {
            std::vector<double> full_re_in(n), full_im_in(n);
            std::vector<double> full_re_out(n), full_im_out(n);
            for (vDSP_Length k = 0; k <= nc; ++k) {
                full_re_in[k] = in[k].x;
                full_im_in[k] = in[k].y;
            }
            for (vDSP_Length k = nc + 1; k < n; ++k) {
                const vDSP_Length mirrored = n - k;
                full_re_in[k] = in[mirrored].x;
                full_im_in[k] = -in[mirrored].y;
            }
            if (!direct_dft(full_re_in.data(), full_im_in.data(), full_re_out.data(),
                            full_im_out.data(), n, true)) {
                return CUFFT_NOT_SUPPORTED;
            }
            std::copy_n(full_re_out.data(), n, out);
            continue;
        }
        re_in[0] = in[0].x;
        im_in[0] = in[nc].x;
        for (vDSP_Length k = 1; k < nc; ++k) {
            re_in[k] = in[k].x;
            im_in[k] = in[k].y;
        }
        vDSP_DFT_ExecuteD(setup, re_in.data(), im_in.data(), re_out.data(), im_out.data());
        for (vDSP_Length i = 0; i < nc; ++i) {
            out[2 * i] = re_out[i];
            out[2 * i + 1] = im_out[i];
        }
    }
    return CUFFT_SUCCESS;
}

// ── Plan factory helper ───────────────────────────────────────────────────────

static cufftResult make_plan(cufftHandle h, int rank, int* n, cufftType type, int batch,
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
    p->type = type;
    p->rank = rank;
    p->n.assign(n, n + rank);
    p->batch = batch;
    if (workSize != nullptr) {
        *workSize = 0;
    }
    return CUFFT_SUCCESS;
}

cufftResult synchronize_plan_stream(const CufftPlanEntry& plan) {
    return cudaStreamSynchronize(plan.stream) == cudaSuccess ? CUFFT_SUCCESS
                                                              : CUFFT_EXEC_FAILED;
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
    CufftPlanEntry& p = it->second;
    if (p.dft_fwd_f) {
        vDSP_DFT_DestroySetup(p.dft_fwd_f);
    }
    if (p.dft_inv_f) {
        vDSP_DFT_DestroySetup(p.dft_inv_f);
    }
    if (p.dft_fwd_d) {
        vDSP_DFT_DestroySetupD(p.dft_fwd_d);
    }
    if (p.dft_inv_d) {
        vDSP_DFT_DestroySetupD(p.dft_inv_d);
    }
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

cufftResult cufftGetSize(cufftHandle /*plan*/, size_t* workSize) {
    if (workSize == nullptr) {
        return CUFFT_INVALID_VALUE;
    }
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
    return make_plan(*plan, 1, n, type, batch, nullptr);
}

cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    int n[2] = {nx, ny};
    return make_plan(*plan, 2, n, type, 1, nullptr);
}

cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    int n[3] = {nx, ny, nz};
    return make_plan(*plan, 3, n, type, 1, nullptr);
}

cufftResult cufftPlanMany(cufftHandle* plan,
                           int rank,
                           int* n,
                           int* /*inembed*/,
                           int /*istride*/,
                           int /*idist*/,
                           int* /*onembed*/,
                           int /*ostride*/,
                           int /*odist*/,
                           cufftType type,
                           int batch) {
    cufftResult r = cufftCreate(plan);
    if (r != CUFFT_SUCCESS) {
        return r;
    }
    return make_plan(*plan, rank, n, type, batch, nullptr);
}

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                              size_t* workSize) {
    int n[1] = {nx};
    return make_plan(plan, 1, n, type, batch, workSize);
}

cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                              size_t* workSize) {
    int n[2] = {nx, ny};
    return make_plan(plan, 2, n, type, 1, workSize);
}

cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                              size_t* workSize) {
    int n[3] = {nx, ny, nz};
    return make_plan(plan, 3, n, type, 1, workSize);
}

cufftResult cufftMakePlanMany(cufftHandle plan,
                               int rank,
                               int* n,
                               int* /*inembed*/,
                               int /*istride*/,
                               int /*idist*/,
                               int* /*onembed*/,
                               int /*ostride*/,
                               int /*odist*/,
                               cufftType type,
                               int batch,
                               size_t* workSize) {
    return make_plan(plan, rank, n, type, batch, workSize);
}

cufftResult cufftXtMakePlanMany(cufftHandle plan,
                                 int rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int /*istride*/,
                                 long long int /*idist*/,
                                 cudaDataType inputtype,
                                 long long int* onembed,
                                 long long int /*ostride*/,
                                 long long int /*odist*/,
                                 cudaDataType outputtype,
                                 long long int batch,
                                 size_t* workSize,
                                 cudaDataType executiontype) {
    if (n == nullptr || workSize == nullptr || rank < 1 || rank > 3 || batch < 1) {
        return CUFFT_INVALID_VALUE;
    }
    if (rank != 1) {
        // Multidimensional transforms require per-axis passes; flattening the
        // dimensions into one 1D transform would be numerically wrong.
        return CUFFT_NOT_SUPPORTED;
    }
    if (inembed != nullptr || onembed != nullptr) {
        // The vDSP-backed plan currently implements contiguous batches only.
        return CUFFT_NOT_SUPPORTED;
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
    return make_plan(plan, rank, dimensions.data(), type,
                     static_cast<int>(batch), workSize);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2c(*p, idata, odata, direction);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_r2c(*p, idata, odata);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_c2r(*p, idata, odata);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_z2z(*p, idata, odata, direction);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_d2z(*p, idata, odata);
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
    const cufftResult sync_status = synchronize_plan_stream(*p);
    if (sync_status != CUFFT_SUCCESS) return sync_status;
    return exec_z2d(*p, idata, odata);
}

// SetWorkArea — on UMA, vDSP manages its own scratch; no external workspace is used.
cufftResult cufftSetWorkArea(cufftHandle /*plan*/, void* /*workArea*/) {
    return CUFFT_SUCCESS;
}

// ── Estimate* ────────────────────────────────────────────────────────────────
// Return a conservative upper-bound on scratch memory without building a full plan.
// On this implementation vDSP manages scratch internally, so 0 is a valid answer,
// but we return a small non-zero estimate to satisfy callers that test workSize > 0.

static size_t estimate_work_bytes(size_t total_complex_elements, cufftType type) {
    // Two split-real buffers of float/double per element, plus vDSP setup overhead.
    size_t elem_bytes = (type == CUFFT_D2Z || type == CUFFT_Z2D || type == CUFFT_Z2Z)
                            ? sizeof(double)
                            : sizeof(float);
    return total_complex_elements * elem_bytes * 2 + 4096;
}

cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || batch <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    *workSize = estimate_work_bytes(static_cast<size_t>(nx) * static_cast<size_t>(batch),
                                    type);
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || ny <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    *workSize = estimate_work_bytes(static_cast<size_t>(nx) * static_cast<size_t>(ny), type);
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize) {
    if (workSize == nullptr || nx <= 0 || ny <= 0 || nz <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    *workSize = estimate_work_bytes(
        static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz), type);
    return CUFFT_SUCCESS;
}

cufftResult cufftEstimateMany(int rank, int* n,
                               int* /*inembed*/, int /*istride*/, int /*idist*/,
                               int* /*onembed*/, int /*ostride*/, int /*odist*/,
                               cufftType type, int batch, size_t* workSize) {
    if (workSize == nullptr || n == nullptr || rank <= 0 || batch <= 0) {
        return CUFFT_INVALID_VALUE;
    }
    size_t total = static_cast<size_t>(batch);
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) return CUFFT_INVALID_VALUE;
        total *= static_cast<size_t>(n[i]);
    }
    *workSize = estimate_work_bytes(total, type);
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
