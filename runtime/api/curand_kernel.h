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
};

typedef curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

// MRG32k3a state
struct curandStateMRG32k3a {
    double s1[3];
    double s2[3];
    int boxmuller_flag;
    double boxmuller_extra;
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
}

static __host__ __device__ __forceinline__
void curand_init(unsigned long long seed, unsigned long long sequence,
                 unsigned long long offset, curandStatePhilox4_32_10_t* state) {
    state->key[0] = (uint32_t)(seed);
    state->key[1] = (uint32_t)(seed >> 32);
    state->ctr[0] = (uint32_t)(sequence);
    state->ctr[1] = (uint32_t)(sequence >> 32);
    state->ctr[2] = (uint32_t)(offset);
    state->ctr[3] = (uint32_t)(offset >> 32);
    state->STATE = 0;
    state->boxmuller_flag = 0;
    state->boxmuller_extra = 0.0f;
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

// --- Uniform distribution [0, 1) ---

static __host__ __device__ __forceinline__
float curand_uniform(curandState_t* state) {
    return (float)(curand(state) & 0x7FFFFFFFu) / (float)2147483648.0f;
}

static __host__ __device__ __forceinline__
float curand_uniform(curandStatePhilox4_32_10_t* state) {
    return (float)(curand(state) & 0x7FFFFFFFu) / (float)2147483648.0f;
}

static __host__ __device__ __forceinline__
double curand_uniform_double(curandState_t* state) {
    uint32_t a = curand(state);
    uint32_t b = curand(state);
    uint64_t combined = ((uint64_t)a << 32) | b;
    return (double)(combined >> 11) / (double)(1ULL << 53);
}

// --- Normal distribution (Box-Muller) ---

static __host__ __device__ __forceinline__
float curand_normal(curandState_t* state) {
    if (state->boxmuller_flag) {
        state->boxmuller_flag = 0;
        return state->boxmuller_extra;
    }
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    if (u1 < 1e-12f) u1 = 1e-12f;
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    state->boxmuller_flag = 1;
    state->boxmuller_extra = r * sinf(theta);
    return r * cosf(theta);
}

static __host__ __device__ __forceinline__
float curand_normal(curandStatePhilox4_32_10_t* state) {
    if (state->boxmuller_flag) {
        state->boxmuller_flag = 0;
        return state->boxmuller_extra;
    }
    float u1 = curand_uniform(state);
    float u2 = curand_uniform(state);
    if (u1 < 1e-12f) u1 = 1e-12f;
    float r = sqrtf(-2.0f * logf(u1));
    float theta = 2.0f * 3.14159265358979323846f * u2;
    state->boxmuller_flag = 1;
    state->boxmuller_extra = r * sinf(theta);
    return r * cosf(theta);
}

// --- Log-normal ---

static __host__ __device__ __forceinline__
float curand_log_normal(curandState_t* state, float mean, float stddev) {
    return expf(mean + stddev * curand_normal(state));
}

// --- Poisson (inverse transform for small lambda, normal approx for large) ---

static __host__ __device__ __forceinline__
unsigned int curand_poisson(curandState_t* state, double lambda) {
    if (lambda < 30.0) {
        double L = exp(-lambda);
        unsigned int k = 0;
        double p = 1.0;
        do {
            k++;
            p *= curand_uniform(state);
        } while (p > L);
        return k - 1;
    } else {
        // Normal approximation for large lambda
        float n = curand_normal(state);
        int result = (int)(lambda + sqrt(lambda) * n + 0.5);
        return result < 0 ? 0 : (unsigned int)result;
    }
}

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
