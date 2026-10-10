#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CURAND_KERNEL_H_
#define CURAND_KERNEL_H_ 1
#endif

// CuMetal curand_kernel.h — device-side random number generation.
// Provides curandState types and device functions for in-kernel RNG.
// On Apple Silicon UMA, these run as host-callable functions since
// device memory is host-accessible.

#include <cstdint>
#include <cmath>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __forceinline__
#if defined(__clang__) || defined(__GNUC__)
#define __forceinline__ __inline__ __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

// Xorshift128+ state — fast, high-quality PRNG
struct curandStateXORWOW {
    uint32_t d;
    uint32_t v[5];
    int boxmuller_flag;
    float boxmuller_extra;
    // Box-Muller produces a pair; the binary64 generators cache their spare in
    // its own slot so a mixed float/double call sequence cannot hand back a
    // narrowed value from the other precision's cache.
    int boxmuller_flag_double;
    double boxmuller_extra_double;
};

typedef curandStateXORWOW curandState_t;
typedef curandStateXORWOW curandState;

// Philox state
struct curandStatePhilox4_32_10 {
    uint32_t ctr[4];
    uint32_t key[2];
    uint32_t output[4];
    int STATE;
    int boxmuller_flag;
    float boxmuller_extra;
    int boxmuller_flag_double;
    double boxmuller_extra_double;
};

typedef curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

// MRG32k3a state: cuRAND's layout. Components are kept below the moduli
// (m1 = 2^32 - 209, m2 = 2^32 - 22853), never all zero.
struct curandStateMRG32k3a {
    unsigned int s1[3];
    unsigned int s2[3];
    int boxmuller_flag;
    int boxmuller_flag_double;
    float boxmuller_extra;
    double boxmuller_extra_double;
};

typedef curandStateMRG32k3a curandStateMRG32k3a_t;

// --- Init ---

static __host__ __device__ __forceinline__
void curand_init(unsigned long long seed, unsigned long long sequence,
                 unsigned long long offset, curandState_t* state) {
    // Initialize XORWOW state from seed
    uint64_t s = seed + sequence;
    state->d = 6615241;
    for (int i = 0; i < 5; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        state->v[i] = (uint32_t)(s >> 32);
    }
    // Skip ahead by offset
    for (unsigned long long i = 0; i < offset; i++) {
        uint32_t t = state->v[0] ^ (state->v[0] >> 2);
        state->v[0] = state->v[1];
        state->v[1] = state->v[2];
        state->v[2] = state->v[3];
        state->v[3] = state->v[4];
        state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
        state->d += 362437;
    }
    state->boxmuller_flag = 0;
    state->boxmuller_extra = 0.0f;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra_double = 0.0;
}

static __host__ __device__ __forceinline__
unsigned int curand(curandStatePhilox4_32_10_t* state);

static __host__ __device__ __forceinline__
void curand_init(unsigned long long seed, unsigned long long sequence,
                 unsigned long long offset, curandStatePhilox4_32_10_t* state) {
    state->key[0] = (uint32_t)(seed);
    state->key[1] = (uint32_t)(seed >> 32);
    // CUDA specifies 2^66 scalar outputs per subsequence; Philox produces
    // four outputs per counter. Position whole blocks, then consume the
    // partial block through the same cache used by scalar generation.
    // https://docs.nvidia.com/cuda/curand/group__DEVICE.html
    state->ctr[0] = (uint32_t)(offset >> 2);
    state->ctr[1] = (uint32_t)(offset >> 34);
    state->ctr[2] = (uint32_t)(sequence);
    state->ctr[3] = (uint32_t)(sequence >> 32);
    state->STATE = 0;
    state->boxmuller_flag = 0;
    state->boxmuller_extra = 0.0f;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra_double = 0.0;
    for (unsigned int i = 0; i < (offset & 3ULL); ++i) (void)curand(state);
}

// --- Core generation ---

static __host__ __device__ __forceinline__
unsigned int curand(curandState_t* state) {
    uint32_t t = state->v[0] ^ (state->v[0] >> 2);
    state->v[0] = state->v[1];
    state->v[1] = state->v[2];
    state->v[2] = state->v[3];
    state->v[3] = state->v[4];
    state->v[4] = (state->v[4] ^ (state->v[4] << 4)) ^ (t ^ (t << 1));
    state->d += 362437;
    return state->v[4] + state->d;
}

// Philox round function
namespace curand_detail {
static __host__ __device__ __forceinline__
uint32_t mulhilo32(uint32_t a, uint32_t b, uint32_t* hi) {
    uint64_t product = (uint64_t)a * (uint64_t)b;
    *hi = (uint32_t)(product >> 32);
    return (uint32_t)product;
}

static __host__ __device__ __forceinline__
void philox_round(uint32_t ctr[4], const uint32_t key[2]) {
    uint32_t hi0, hi2;
    uint32_t lo0 = mulhilo32(0xD2511F53u, ctr[0], &hi0);
    uint32_t lo2 = mulhilo32(0xCD9E8D57u, ctr[2], &hi2);
    ctr[0] = hi2 ^ ctr[1] ^ key[0];
    ctr[1] = lo2;
    ctr[2] = hi0 ^ ctr[3] ^ key[1];
    ctr[3] = lo0;
}
} // namespace curand_detail

static __host__ __device__ __forceinline__
unsigned int curand(curandStatePhilox4_32_10_t* state) {
    if (state->STATE == 0) {
        // Run 10 rounds of Philox
        uint32_t ctr[4] = {state->ctr[0], state->ctr[1], state->ctr[2], state->ctr[3]};
        uint32_t key[2] = {state->key[0], state->key[1]};
        for (int i = 0; i < 10; i++) {
            curand_detail::philox_round(ctr, key);
            key[0] += 0x9E3779B9u;
            key[1] += 0xBB67AE85u;
        }
        state->output[0] = ctr[0];
        state->output[1] = ctr[1];
        state->output[2] = ctr[2];
        state->output[3] = ctr[3];
        // Increment counter
        state->ctr[0]++;
        if (state->ctr[0] == 0) {
            state->ctr[1]++;
            if (state->ctr[1] == 0) {
                state->ctr[2]++;
                if (state->ctr[2] == 0) state->ctr[3]++;
            }
        }
    }
    uint32_t result = state->output[state->STATE];
    state->STATE = (state->STATE + 1) & 3;
    return result;
}

// --- MRG32k3a (L'Ecuyer 1999) ---
//
// Exact 64-bit integer arithmetic: every product is below 2^53, so nothing
// depends on CuMetal's emulated binary64.
namespace curand_detail {
static const int64_t kMrgM1 = 4294967087LL;
static const int64_t kMrgM2 = 4294944443LL;

static __host__ __device__ __forceinline__
uint64_t splitmix64(uint64_t* x) {
    uint64_t z = (*x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// One step; returns z in [1, m1].
static __host__ __device__ __forceinline__
uint32_t mrg32k3a_next(curandStateMRG32k3a* state) {
    int64_t p1 = 1403580LL * state->s1[1] - 810728LL * state->s1[0];
    p1 %= kMrgM1;
    if (p1 < 0) p1 += kMrgM1;
    state->s1[0] = state->s1[1];
    state->s1[1] = state->s1[2];
    state->s1[2] = (uint32_t)p1;
    int64_t p2 = 527612LL * state->s2[2] - 1370589LL * state->s2[0];
    p2 %= kMrgM2;
    if (p2 < 0) p2 += kMrgM2;
    state->s2[0] = state->s2[1];
    state->s2[1] = state->s2[2];
    state->s2[2] = (uint32_t)p2;
    return (uint32_t)(p1 > p2 ? p1 - p2 : p1 - p2 + kMrgM1);
}
} // namespace curand_detail

// Streams start at hashed points of the 2^191 period rather than at cuRAND's
// 2^76-spaced subsequences, so values differ from NVIDIA's for the same seed.
static __host__ __device__ __forceinline__
void curand_init(unsigned long long seed, unsigned long long sequence,
                 unsigned long long offset, curandStateMRG32k3a_t* state) {
    uint64_t mix = seed;
    uint64_t seq_mix = sequence;
    mix ^= curand_detail::splitmix64(&seq_mix);
    for (int i = 0; i < 3; i++) {
        uint64_t r = curand_detail::splitmix64(&mix);
        state->s1[i] = (uint32_t)((r & 0xFFFFFFFFULL) % (uint64_t)curand_detail::kMrgM1);
        state->s2[i] = (uint32_t)((r >> 32) % (uint64_t)curand_detail::kMrgM2);
    }
    if ((state->s1[0] | state->s1[1] | state->s1[2]) == 0) state->s1[0] = 12345u;
    if ((state->s2[0] | state->s2[1] | state->s2[2]) == 0) state->s2[0] = 12345u;
    for (unsigned long long i = 0; i < offset; i++) curand_detail::mrg32k3a_next(state);
    state->boxmuller_flag = 0;
    state->boxmuller_flag_double = 0;
    state->boxmuller_extra = 0.0f;
    state->boxmuller_extra_double = 0.0;
}

static __host__ __device__ __forceinline__
unsigned int curand(curandStateMRG32k3a_t* state) {
    return curand_detail::mrg32k3a_next(state);
}

// --- Uniform distribution (0, 1] ---
//
// cuRAND documents (0, 1]: callers may take log(u) without a zero guard, and
// CuPy maps an exact 1 back to 0 because it relies on that interval.

namespace curand_detail {
template <typename State>
static __host__ __device__ __forceinline__ float uniform_float(State* state) {
    return (float)((curand(state) >> 8) + 1u) * (1.0f / 16777216.0f);
}
} // namespace curand_detail

static __host__ __device__ __forceinline__
float curand_uniform(curandState_t* state) {
    return curand_detail::uniform_float(state);
}

static __host__ __device__ __forceinline__
float curand_uniform(curandStatePhilox4_32_10_t* state) {
    return curand_detail::uniform_float(state);
}

// MRG32k3a draws lie in [1, m1], not over all 32 bits, so scale by 1/(m1 + 1).
static __host__ __device__ __forceinline__
float curand_uniform(curandStateMRG32k3a_t* state) {
    return (float)curand(state) * 2.3283064365386963e-10f;
}

// Built from the bit pattern rather than by converting the 53-bit integer:
// `(double)(u64)` lowers to PTX `cvt.rn.f64.u64`, which CuMetal's FP32-pair FP64
// emulation has no primitive for, so the integer-divide spelling made the whole
// kernel unlowerable. Setting the exponent to 1.0 and filling the 52-bit
// mantissa gives [1,2) directly; subtracting from two lands in (0,1] with
// uniform 2^-52 spacing and no integer-to-float conversion at all.
static __host__ __device__ __forceinline__
double __cumetal_curand_bits_to_unit_double(uint64_t bits) {
    const uint64_t pattern = 0x3FF0000000000000ULL | (bits >> 12);
    double value;
    __builtin_memcpy(&value, &pattern, sizeof(value));
    return 2.0 - value;
}

namespace curand_detail {
template <typename State>
static __host__ __device__ __forceinline__ double uniform_double(State* state) {
    uint32_t a = curand(state);
    uint32_t b = curand(state);
    return __cumetal_curand_bits_to_unit_double(((uint64_t)a << 32) | b);
}

// Box-Muller. u1 is in (0, 1], so log(u1) is finite.
template <typename State>
static __host__ __device__ __forceinline__ float normal_float(State* state) {
    if (state->boxmuller_flag) {
        state->boxmuller_flag = 0;
        return state->boxmuller_extra;
    }
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    state->boxmuller_flag = 1;
    state->boxmuller_extra = r * sinf(theta);
    return r * cosf(theta);
}

// Drawn from curand_uniform_double, not from the binary32 stream: taking the
// float path and widening would leave the low 29 bits of every sample zero,
// which shows up as a visible lattice in a Gaussian tail and defeats the point
// of asking for a double. The spare has its own slot so a mixed float/double
// call sequence never hands back a value cached at the other precision.
template <typename State>
static __host__ __device__ __forceinline__ double normal_double(State* state) {
    if (state->boxmuller_flag_double) {
        state->boxmuller_flag_double = 0;
        return state->boxmuller_extra_double;
    }
    double u1 = uniform_double(state);
    double u2 = uniform_double(state);
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    state->boxmuller_flag_double = 1;
    state->boxmuller_extra_double = r * sin(theta);
    return r * cos(theta);
}

// Inverse transform for small lambda, normal approximation for large.
template <typename State>
static __host__ __device__ __forceinline__ unsigned int poisson(State* state, double lambda) {
    if (lambda < 30.0) {
        double L = exp(-lambda);
        unsigned int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= curand_uniform(state);
        } while (p > L);
        return k - 1;
    }
    float n = normal_float(state);
    int result = (int)(lambda + sqrt(lambda) * n + 0.5);
    return result < 0 ? 0 : (unsigned int)result;
}
} // namespace curand_detail

// One overload set per generator, all routed through the templates above.
#define CUMETAL_CURAND_DISTRIBUTIONS(STATE)                                         \
    static __host__ __device__ __forceinline__                                     \
    double curand_uniform_double(STATE* state) {                                   \
        return curand_detail::uniform_double(state);                               \
    }                                                                              \
    static __host__ __device__ __forceinline__                                     \
    float curand_normal(STATE* state) { return curand_detail::normal_float(state); } \
    static __host__ __device__ __forceinline__                                     \
    double curand_normal_double(STATE* state) {                                    \
        return curand_detail::normal_double(state);                                \
    }                                                                              \
    static __host__ __device__ __forceinline__                                     \
    float curand_log_normal(STATE* state, float mean, float stddev) {              \
        return expf(mean + stddev * curand_normal(state));                         \
    }                                                                              \
    static __host__ __device__ __forceinline__                                     \
    double curand_log_normal_double(STATE* state, double mean, double stddev) {    \
        return exp(mean + stddev * curand_normal_double(state));                   \
    }                                                                              \
    static __host__ __device__ __forceinline__                                     \
    unsigned int curand_poisson(STATE* state, double lambda) {                     \
        return curand_detail::poisson(state, lambda);                              \
    }

CUMETAL_CURAND_DISTRIBUTIONS(curandState_t)
CUMETAL_CURAND_DISTRIBUTIONS(curandStatePhilox4_32_10_t)
CUMETAL_CURAND_DISTRIBUTIONS(curandStateMRG32k3a_t)
#undef CUMETAL_CURAND_DISTRIBUTIONS

// --- Convenience: skip ahead ---

static __host__ __device__ __forceinline__
void skipahead(unsigned long long n, curandState_t* state) {
    for (unsigned long long i = 0; i < n; i++) {
        curand(state);
    }
}

static __host__ __device__ __forceinline__
void skipahead_sequence(unsigned long long n, curandState_t* state) {
    // Re-initialize with advanced sequence
    curand_init(0ULL, n, 0ULL, state);
}
