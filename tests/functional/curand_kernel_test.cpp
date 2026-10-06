#include <curand_kernel.h>
#include <cstdio>
#include <cmath>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_xorwow_init_and_generate() {
    curandState_t state;
    curand_init(12345ULL, 0ULL, 0ULL, &state);
    unsigned int val = curand(&state);
    CHECK(val != 0, "curand XORWOW generates non-zero");

    // Generate 1000 values, check they're not all the same
    unsigned int first = curand(&state);
    bool all_same = true;
    for (int i = 0; i < 999; i++) {
        if (curand(&state) != first) { all_same = false; break; }
    }
    CHECK(!all_same, "curand XORWOW produces varied output");
}

static void test_uniform_range() {
    curandState_t state;
    curand_init(42ULL, 0ULL, 0ULL, &state);

    bool in_range = true;
    for (int i = 0; i < 10000; i++) {
        float u = curand_uniform(&state);
        if (!(u > 0.0f && u <= 1.0f)) { in_range = false; break; }
    }
    CHECK(in_range, "curand_uniform in (0, 1]");
}

static void test_uniform_mean() {
    curandState_t state;
    curand_init(123ULL, 0ULL, 0ULL, &state);

    double sum = 0;
    int N = 100000;
    for (int i = 0; i < N; i++) sum += curand_uniform(&state);
    double mean = sum / N;
    CHECK(std::fabs(mean - 0.5) < 0.01, "curand_uniform mean ~0.5");
}

static void test_normal_distribution() {
    curandState_t state;
    curand_init(456ULL, 0ULL, 0ULL, &state);

    double sum = 0, sum_sq = 0;
    int N = 100000;
    for (int i = 0; i < N; i++) {
        float n = curand_normal(&state);
        sum += n;
        sum_sq += n * n;
    }
    double mean = sum / N;
    double var = sum_sq / N - mean * mean;
    CHECK(std::fabs(mean) < 0.02, "curand_normal mean ~0");
    CHECK(std::fabs(var - 1.0) < 0.05, "curand_normal variance ~1");
}

static void test_philox_init_and_generate() {
    curandStatePhilox4_32_10_t state;
    curand_init(789ULL, 0ULL, 0ULL, &state);

    unsigned int val = curand(&state);
    CHECK(val != 0, "curand Philox generates non-zero");

    float u = curand_uniform(&state);
    CHECK(u > 0.0f && u <= 1.0f, "curand_uniform Philox in (0, 1]");
}

static void test_different_seeds_different_output() {
    curandState_t s1, s2;
    curand_init(100ULL, 0ULL, 0ULL, &s1);
    curand_init(200ULL, 0ULL, 0ULL, &s2);

    unsigned int v1 = curand(&s1);
    unsigned int v2 = curand(&s2);
    CHECK(v1 != v2, "different seeds produce different output");
}

static void test_different_sequences_different_output() {
    curandState_t s1, s2;
    curand_init(100ULL, 0ULL, 0ULL, &s1);
    curand_init(100ULL, 1ULL, 0ULL, &s2);

    unsigned int v1 = curand(&s1);
    unsigned int v2 = curand(&s2);
    CHECK(v1 != v2, "different sequences produce different output");
}

static void test_log_normal() {
    curandState_t state;
    curand_init(321ULL, 0ULL, 0ULL, &state);

    bool all_positive = true;
    for (int i = 0; i < 1000; i++) {
        float val = curand_log_normal(&state, 0.0f, 1.0f);
        if (val <= 0.0f) { all_positive = false; break; }
    }
    CHECK(all_positive, "curand_log_normal all positive");
}

// cuRAND's interval is (0, 1]: code written against it takes log(u) unguarded.
// The old [0, 1) draws returned exactly 0 about once per 2^24 floats.
template <typename State>
static void check_unit_interval(const char* name) {
    State state;
    curand_init(7ULL, 3ULL, 0ULL, &state);
    bool f_ok = true, d_ok = true, finite = true;
    for (int i = 0; i < 200000; i++) {
        float u = curand_uniform(&state);
        double d = curand_uniform_double(&state);
        if (!(u > 0.0f && u <= 1.0f)) f_ok = false;
        if (!(d > 0.0 && d <= 1.0)) d_ok = false;
        if (!std::isfinite(std::log(u)) || !std::isfinite(std::log(d))) finite = false;
    }
    char msg[128];
    snprintf(msg, sizeof msg, "%s curand_uniform in (0, 1]", name);
    CHECK(f_ok, msg);
    snprintf(msg, sizeof msg, "%s curand_uniform_double in (0, 1]", name);
    CHECK(d_ok, msg);
    snprintf(msg, sizeof msg, "%s log(uniform) always finite", name);
    CHECK(finite, msg);
}

template <typename State>
static void check_moments(const char* name) {
    State state;
    curand_init(2024ULL, 11ULL, 0ULL, &state);
    const int n = 200000;
    double us = 0, uss = 0, ns = 0, nss = 0, ds = 0, dss = 0;
    for (int i = 0; i < n; i++) {
        double u = curand_uniform(&state);
        double z = curand_normal(&state);
        double d = curand_normal_double(&state);
        us += u; uss += u * u; ns += z; nss += z * z; ds += d; dss += d * d;
    }
    double um = us / n, uv = uss / n - um * um;
    double nm = ns / n, nv = nss / n - nm * nm;
    double dm = ds / n, dv = dss / n - dm * dm;
    char msg[160];
    snprintf(msg, sizeof msg, "%s uniform mean %.4f var %.4f (1/12 = 0.0833)", name, um, uv);
    CHECK(std::fabs(um - 0.5) < 0.005 && std::fabs(uv - 1.0 / 12) < 0.002, msg);
    snprintf(msg, sizeof msg, "%s normal mean %.4f var %.4f", name, nm, nv);
    CHECK(std::fabs(nm) < 0.01 && std::fabs(nv - 1.0) < 0.02, msg);
    snprintf(msg, sizeof msg, "%s normal_double mean %.4f var %.4f", name, dm, dv);
    CHECK(std::fabs(dm) < 0.01 && std::fabs(dv - 1.0) < 0.02, msg);
}

// The recurrence itself: from L'Ecuyer's reference seed (all components 12345)
// MRG32k3a's first output is 0.12701112204657714 (as published with mrg-random, Rosetta Code).
static void test_mrg32k3a_reference_recurrence() {
    curandStateMRG32k3a_t state;
    curand_init(1ULL, 0ULL, 0ULL, &state);
    for (int i = 0; i < 3; i++) state.s1[i] = state.s2[i] = 12345u;
    double u = curand(&state) / 4294967088.0;
    char msg[96];
    snprintf(msg, sizeof msg, "MRG32k3a reference first draw %.10f", u);
    CHECK(std::fabs(u - 0.12701112204657714) < 1e-12, msg);
}

// Adjacent sequences (one per thread in CuPy) must not be correlated.
static void test_mrg32k3a_sequences_independent() {
    curandStateMRG32k3a_t a, b;
    curand_init(99ULL, 0ULL, 0ULL, &a);
    curand_init(99ULL, 1ULL, 0ULL, &b);
    const int n = 100000;
    double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
    for (int i = 0; i < n; i++) {
        double x = curand_uniform(&a), y = curand_uniform(&b);
        sa += x; sb += y; sab += x * y; saa += x * x; sbb += y * y;
    }
    double cov = sab / n - (sa / n) * (sb / n);
    double r = cov / std::sqrt((saa / n - (sa / n) * (sa / n)) * (sbb / n - (sb / n) * (sb / n)));
    char msg[96];
    snprintf(msg, sizeof msg, "MRG32k3a adjacent sequences uncorrelated (r=%.4f)", r);
    CHECK(std::fabs(r) < 0.02, msg);
}

int main() {
    check_unit_interval<curandState_t>("XORWOW");
    check_unit_interval<curandStatePhilox4_32_10_t>("Philox");
    check_unit_interval<curandStateMRG32k3a_t>("MRG32k3a");
    check_moments<curandState_t>("XORWOW");
    check_moments<curandStatePhilox4_32_10_t>("Philox");
    check_moments<curandStateMRG32k3a_t>("MRG32k3a");
    test_mrg32k3a_reference_recurrence();
    test_mrg32k3a_sequences_independent();
    test_xorwow_init_and_generate();
    test_uniform_range();
    test_uniform_mean();
    test_normal_distribution();
    test_philox_init_and_generate();
    test_different_seeds_different_output();
    test_different_sequences_different_output();
    test_log_normal();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
