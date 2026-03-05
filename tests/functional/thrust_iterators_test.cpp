#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <numeric>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_transform_iterator() {
    std::vector<int> data = {1, 2, 3, 4, 5};
    auto square = [](int x) { return x * x; };
    auto begin = thrust::make_transform_iterator(data.begin(), square);
    auto end = thrust::make_transform_iterator(data.end(), square);

    // Sum of squares: 1+4+9+16+25 = 55
    int sum = 0;
    for (auto it = begin; it != end; ++it) sum += *it;
    CHECK(sum == 55, "transform_iterator sum of squares");

    // Random access
    CHECK(begin[2] == 9, "transform_iterator random access");
}

static void test_constant_iterator() {
    auto it = thrust::make_constant_iterator(42);
    CHECK(*it == 42, "constant_iterator dereference");
    CHECK(it[100] == 42, "constant_iterator random access");

    auto it2 = it + 10;
    CHECK(it2 - it == 10, "constant_iterator arithmetic");
    CHECK(*it2 == 42, "constant_iterator still 42 after advance");
}

static void test_discard_iterator() {
    auto it = thrust::make_discard_iterator();
    // Should accept writes without crashing
    *it = 42;
    auto it2 = it + 5;
    CHECK(it2 - it == 5, "discard_iterator arithmetic");
    it2[3] = 99; // should not crash
    CHECK(true, "discard_iterator writes succeed");
}

static void test_permutation_iterator() {
    std::vector<float> data = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f};
    std::vector<int> indices = {4, 2, 0, 3, 1};
    auto begin = thrust::make_permutation_iterator(data.begin(), indices.begin());

    CHECK(*begin == 50.0f, "permutation_iterator first=data[4]");
    CHECK(begin[1] == 30.0f, "permutation_iterator second=data[2]");
    CHECK(begin[2] == 10.0f, "permutation_iterator third=data[0]");

    // Write through permutation
    begin[0] = 99.0f;
    CHECK(data[4] == 99.0f, "permutation_iterator write-through");
}

static void test_transform_iterator_with_reduce() {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto negate = [](float x) { return -x; };
    auto begin = thrust::make_transform_iterator(data.begin(), negate);
    auto end = thrust::make_transform_iterator(data.end(), negate);

    float sum = thrust::reduce(begin, end, 0.0f);
    CHECK(std::fabs(sum - (-10.0f)) < 1e-5f, "transform_iterator + reduce");
}

static void test_constant_with_counting() {
    // Combine counting and constant iterators to generate index-weighted values
    thrust::counting_iterator<int> idx(0);
    auto c = thrust::make_constant_iterator(3);
    // c[i] * idx[i] = 3 * i for i=0..4
    int sum = 0;
    for (int i = 0; i < 5; i++) sum += c[i] * idx[i];
    CHECK(sum == 3 * (0 + 1 + 2 + 3 + 4), "constant * counting = 30");
}

int main() {
    test_transform_iterator();
    test_constant_iterator();
    test_discard_iterator();
    test_permutation_iterator();
    test_transform_iterator_with_reduce();
    test_constant_with_counting();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
