#include "thrust/thrust.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static bool test_device_vector() {
    thrust::device_vector<float> dv(4, 1.0f);
    if (dv.size() != 4) { std::fprintf(stderr, "FAIL: dv size\n"); return false; }
    if (std::fabs(dv[0] - 1.0f) > 1e-5f) { std::fprintf(stderr, "FAIL: dv[0]\n"); return false; }
    return true;
}

static bool test_host_device_copy() {
    std::vector<int> hv = {10, 20, 30, 40};
    thrust::device_vector<int> dv(hv);
    thrust::host_vector<int> result(dv);
    for (int i = 0; i < 4; ++i) {
        if (result[i] != hv[i]) {
            std::fprintf(stderr, "FAIL: host_device_copy[%d]=%d expected %d\n", i, result[i], hv[i]);
            return false;
        }
    }
    return true;
}

static bool test_sort() {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    thrust::sort(data, data + 8);
    for (int i = 1; i < 8; ++i) {
        if (data[i] < data[i-1]) {
            std::fprintf(stderr, "FAIL: sort not ordered at %d\n", i);
            return false;
        }
    }
    return true;
}

static bool test_sort_by_key() {
    int keys[] = {3, 1, 2};
    float vals[] = {30.0f, 10.0f, 20.0f};
    thrust::sort_by_key(keys, keys + 3, vals);
    // keys should be {1, 2, 3}, vals should be {10, 20, 30}
    if (keys[0] != 1 || keys[1] != 2 || keys[2] != 3) {
        std::fprintf(stderr, "FAIL: sort_by_key keys\n"); return false;
    }
    if (vals[0] != 10.0f || vals[1] != 20.0f || vals[2] != 30.0f) {
        std::fprintf(stderr, "FAIL: sort_by_key vals\n"); return false;
    }
    return true;
}

static bool test_reduce() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float sum = thrust::reduce(data, data + 4);
    if (std::fabs(sum - 10.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: reduce sum=%f expected 10\n", sum);
        return false;
    }
    float prod = thrust::reduce(data, data + 4, 1.0f, thrust::multiplies<float>());
    if (std::fabs(prod - 24.0f) > 1e-5f) {
        std::fprintf(stderr, "FAIL: reduce prod=%f expected 24\n", prod);
        return false;
    }
    return true;
}

static bool test_inclusive_scan() {
    int data[] = {1, 2, 3, 4};
    int result[4] = {};
    thrust::inclusive_scan(data, data + 4, result);
    int expected[] = {1, 3, 6, 10};
    for (int i = 0; i < 4; ++i) {
        if (result[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: inclusive_scan[%d]=%d expected %d\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool test_exclusive_scan() {
    int data[] = {1, 2, 3, 4};
    int result[4] = {};
    thrust::exclusive_scan(data, data + 4, result, 0);
    int expected[] = {0, 1, 3, 6};
    for (int i = 0; i < 4; ++i) {
        if (result[i] != expected[i]) {
            std::fprintf(stderr, "FAIL: exclusive_scan[%d]=%d expected %d\n", i, result[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool test_transform() {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {10.0f, 20.0f, 30.0f};
    float c[3] = {};
    thrust::transform(a, a + 3, b, c, thrust::plus<float>());
    float expected[] = {11.0f, 22.0f, 33.0f};
    for (int i = 0; i < 3; ++i) {
        if (std::fabs(c[i] - expected[i]) > 1e-5f) {
            std::fprintf(stderr, "FAIL: transform[%d]=%f expected %f\n", i, c[i], expected[i]);
            return false;
        }
    }
    return true;
}

static bool test_fill_and_sequence() {
    int data[4] = {};
    thrust::fill(data, data + 4, 42);
    for (int i = 0; i < 4; ++i) {
        if (data[i] != 42) { std::fprintf(stderr, "FAIL: fill[%d]=%d\n", i, data[i]); return false; }
    }
    thrust::sequence(data, data + 4);
    for (int i = 0; i < 4; ++i) {
        if (data[i] != i) { std::fprintf(stderr, "FAIL: sequence[%d]=%d\n", i, data[i]); return false; }
    }
    return true;
}

static bool test_counting_iterator() {
    auto it = thrust::make_counting_iterator(0);
    int sum = thrust::reduce(it, it + 5, 0); // 0+1+2+3+4 = 10
    if (sum != 10) {
        std::fprintf(stderr, "FAIL: counting_iterator sum=%d expected 10\n", sum);
        return false;
    }
    return true;
}

static bool test_device_ptr() {
    float data[] = {5.0f, 3.0f, 1.0f, 4.0f, 2.0f};
    auto dp = thrust::device_pointer_cast(data);
    thrust::sort(dp, dp + 5);
    for (int i = 1; i < 5; ++i) {
        if (data[i] < data[i-1]) {
            std::fprintf(stderr, "FAIL: device_ptr sort not ordered at %d\n", i);
            return false;
        }
    }
    float* raw = thrust::raw_pointer_cast(dp);
    if (raw != data) {
        std::fprintf(stderr, "FAIL: raw_pointer_cast mismatch\n");
        return false;
    }
    return true;
}

static bool test_min_max_element() {
    float data[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f};
    auto mn = thrust::min_element(data, data + 5);
    auto mx = thrust::max_element(data, data + 5);
    if (*mn != 1.0f) { std::fprintf(stderr, "FAIL: min_element=%f\n", *mn); return false; }
    if (*mx != 5.0f) { std::fprintf(stderr, "FAIL: max_element=%f\n", *mx); return false; }
    return true;
}

static bool test_copy_generate_random_and_is_sorted() {
    thrust::default_random_engine engine(7);
    thrust::uniform_int_distribution<int> distribution(10, 20);
    int generated[8] = {};
    thrust::generate(generated, generated + 8, [&] { return distribution(engine); });
    for (int value : generated) {
        if (value < 10 || value > 20) {
            std::fprintf(stderr, "FAIL: generated random value out of range\n");
            return false;
        }
    }
    int copied[8] = {};
    thrust::copy(generated, generated + 8, copied);
    if (!std::equal(generated, generated + 8, copied)) {
        std::fprintf(stderr, "FAIL: thrust::copy mismatch\n");
        return false;
    }
    thrust::sort(copied, copied + 8);
    if (!thrust::is_sorted(copied, copied + 8)) {
        std::fprintf(stderr, "FAIL: thrust::is_sorted rejected sorted data\n");
        return false;
    }
    return true;
}

static bool test_segmented_algorithms_and_stream_ordering() {
    int keys[] = {1, 0, 0, 1, 0};
    int values[] = {5, 3, 4, 8, 2};
    int scanned[5] = {};
    thrust::inclusive_scan_by_key(keys, keys + 5, values, scanned,
                                  thrust::greater_equal<int>(),
                                  thrust::minimum<int>());
    const int expected_scan[] = {5, 3, 3, 8, 2};
    if (!std::equal(scanned, scanned + 5, expected_scan)) {
        std::fprintf(stderr, "FAIL: inclusive_scan_by_key segmentation\n");
        return false;
    }

    int adjacent[5] = {};
    thrust::adjacent_difference(keys, keys + 5, adjacent,
                                thrust::not_equal_to<int>());
    const int expected_adjacent[] = {1, 1, 0, 1, 1};
    if (!std::equal(adjacent, adjacent + 5, expected_adjacent)) {
        std::fprintf(stderr, "FAIL: adjacent_difference predicate\n");
        return false;
    }

    auto allocation = thrust::device_malloc<unsigned char>(256);
    if (cudaMemsetAsync(allocation.get(), 7, 256, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: queued memset for Thrust ordering test\n");
        return false;
    }
    const unsigned int sum = thrust::reduce(allocation, allocation + 256, 0u);
    thrust::device_free(allocation);
    if (sum != 256u * 7u) {
        std::fprintf(stderr, "FAIL: host Thrust algorithm raced preceding GPU work\n");
        return false;
    }
    return true;
}

int main() {
    if (!test_device_vector()) return 1;
    if (!test_host_device_copy()) return 1;
    if (!test_sort()) return 1;
    if (!test_sort_by_key()) return 1;
    if (!test_reduce()) return 1;
    if (!test_inclusive_scan()) return 1;
    if (!test_exclusive_scan()) return 1;
    if (!test_transform()) return 1;
    if (!test_fill_and_sequence()) return 1;
    if (!test_counting_iterator()) return 1;
    if (!test_device_ptr()) return 1;
    if (!test_min_max_element()) return 1;
    if (!test_copy_generate_random_and_is_sorted()) return 1;
    if (!test_segmented_algorithms_and_stream_ordering()) return 1;

    std::printf("PASS: thrust API tests\n");
    return 0;
}
