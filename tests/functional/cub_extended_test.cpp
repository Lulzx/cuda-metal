#include <cub/cub.h>
#include <cub/device/device_find.cuh>
#include <cub/device/device_segmented_scan.cuh>
#include <cub/device/device_transform.cuh>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_block_load() {
    constexpr int BLOCK = 2;
    constexpr int ITEMS = 3;
    typename cub::BlockLoad<float, BLOCK, ITEMS>::TempStorage temp;
    cub::BlockLoad<float, BLOCK, ITEMS> loader(temp, 0);

    float data[] = {1, 2, 3, 4, 5, 6};
    float items[ITEMS];
    loader.Load(data, items);
    // Thread 0, blocked: items[0..2] = data[0..2]
    CHECK(items[0] == 1 && items[1] == 2 && items[2] == 3, "BlockLoad full tile");
}

static void test_block_load_partial() {
    constexpr int BLOCK = 2;
    constexpr int ITEMS = 3;
    typename cub::BlockLoad<int, BLOCK, ITEMS>::TempStorage temp;
    cub::BlockLoad<int, BLOCK, ITEMS> loader(temp, 0);

    int data[] = {10, 20, 30};
    int items[ITEMS];
    loader.Load(data, items, 2, -1); // only 2 valid items, rest = -1
    CHECK(items[0] == 10 && items[1] == 20 && items[2] == -1, "BlockLoad partial tile");
}

static void test_block_store() {
    constexpr int BLOCK = 2;
    constexpr int ITEMS = 2;
    typename cub::BlockStore<int, BLOCK, ITEMS>::TempStorage temp;
    cub::BlockStore<int, BLOCK, ITEMS> storer(temp, 0);

    int items[ITEMS] = {100, 200};
    int output[4] = {0};
    storer.Store(output, items);
    CHECK(output[0] == 100 && output[1] == 200, "BlockStore full tile");
}

static void test_device_select_if() {
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int output[8] = {0};
    int num_selected = 0;
    size_t temp_bytes = 0;

    auto is_even = [](int x) { return x % 2 == 0; };
    cub::DeviceSelect::If(nullptr, temp_bytes, data, output, &num_selected, 8, is_even);
    char temp_buf[1];
    cub::DeviceSelect::If(temp_buf, temp_bytes, data, output, &num_selected, 8, is_even);
    CHECK(num_selected == 4, "DeviceSelect::If count=4");
    CHECK(output[0] == 2 && output[1] == 4 && output[2] == 6 && output[3] == 8,
          "DeviceSelect::If values");
}

static void test_device_select_flagged() {
    int data[] = {10, 20, 30, 40, 50};
    int flags[] = {1, 0, 1, 0, 1};
    int output[5] = {0};
    int num_selected = 0;
    size_t temp_bytes = 0;
    char temp_buf[1];

    cub::DeviceSelect::Flagged(nullptr, temp_bytes, data, flags, output, &num_selected, 5);
    cub::DeviceSelect::Flagged(temp_buf, temp_bytes, data, flags, output, &num_selected, 5);
    CHECK(num_selected == 3, "DeviceSelect::Flagged count=3");
    CHECK(output[0] == 10 && output[1] == 30 && output[2] == 50, "DeviceSelect::Flagged values");
}

static void test_device_select_unique() {
    int data[] = {1, 1, 2, 2, 2, 3, 3, 4};
    int output[8] = {0};
    int num_selected = 0;
    size_t temp_bytes = 0;
    char temp_buf[1];

    cub::DeviceSelect::Unique(nullptr, temp_bytes, data, output, &num_selected, 8);
    cub::DeviceSelect::Unique(temp_buf, temp_bytes, data, output, &num_selected, 8);
    CHECK(num_selected == 4, "DeviceSelect::Unique count=4");
    CHECK(output[0] == 1 && output[1] == 2 && output[2] == 3 && output[3] == 4,
          "DeviceSelect::Unique values");
}

static void test_device_histogram_even() {
    float samples[] = {0.5f, 1.5f, 2.5f, 1.5f, 3.5f};
    int histogram[4] = {0};
    size_t temp_bytes = 0;
    char temp_buf[1];

    // 5 bins: [0,1), [1,2), [2,3), [3,4)
    cub::DeviceHistogram::HistogramEven(nullptr, temp_bytes, samples, histogram, 5, 0.0f, 4.0f, 5);
    cub::DeviceHistogram::HistogramEven(temp_buf, temp_bytes, samples, histogram, 5, 0.0f, 4.0f, 5);
    CHECK(histogram[0] == 1 && histogram[1] == 2 && histogram[2] == 1 && histogram[3] == 1,
          "DeviceHistogram::HistogramEven");
}

static void test_device_rle_encode() {
    int data[] = {1, 1, 1, 2, 2, 3, 3, 3, 3};
    int unique_out[3], counts_out[3], num_runs = 0;
    size_t temp_bytes = 0;
    char temp_buf[1];

    cub::DeviceRunLengthEncode::Encode(nullptr, temp_bytes, data, unique_out, counts_out, &num_runs, 9);
    cub::DeviceRunLengthEncode::Encode(temp_buf, temp_bytes, data, unique_out, counts_out, &num_runs, 9);
    CHECK(num_runs == 3, "DeviceRunLengthEncode::Encode num_runs=3");
    CHECK(unique_out[0] == 1 && unique_out[1] == 2 && unique_out[2] == 3,
          "DeviceRunLengthEncode::Encode values");
    CHECK(counts_out[0] == 3 && counts_out[1] == 2 && counts_out[2] == 4,
          "DeviceRunLengthEncode::Encode counts");
}

static void test_device_find() {
    int data[] = {1, 3, 5, 8, 8, 12};
    int values[] = {8, 9};
    int output[2] = {-1, -1};
    size_t temp_bytes = 0;
    char storage[1];
    auto even = [](int value) { return value % 2 == 0; };

    CHECK(cub::DeviceFind::FindIf(nullptr, temp_bytes, data, output, even, 6) == cudaSuccess &&
              temp_bytes > 0,
          "DeviceFind::FindIf storage query");
    CHECK(cub::DeviceFind::FindIf(storage, temp_bytes, data, output, even, 6) == cudaSuccess &&
              output[0] == 3,
          "DeviceFind::FindIf result");
    CHECK(cub::DeviceFind::LowerBound(storage, temp_bytes, data, 6, values, 2,
                                      output, std::less<int>{}) == cudaSuccess &&
              output[0] == 3 && output[1] == 5,
          "DeviceFind::LowerBound results");
    CHECK(cub::DeviceFind::UpperBound(storage, temp_bytes, data, 6, values, 2,
                                      output, std::less<int>{}) == cudaSuccess &&
              output[0] == 5 && output[1] == 5,
          "DeviceFind::UpperBound results");
    CHECK(cub::DeviceFind::FindIf(storage, temp_bytes, data, output, even, -1) ==
              cudaErrorInvalidValue,
          "DeviceFind rejects a negative item count");
}

static void test_device_segmented_scan() {
    int input[] = {1, 2, 3, 9, 4};
    int offsets[] = {0, 3, 5};
    int output[5] = {};
    size_t temp_bytes = 1;
    char storage[1];

    CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
              storage, temp_bytes, input, output, offsets, offsets + 1, 2) == cudaSuccess &&
              output[0] == 0 && output[1] == 1 && output[2] == 3 &&
              output[3] == 0 && output[4] == 9,
          "DeviceSegmentedScan exclusive sums");
    CHECK(cub::DeviceSegmentedScan::InclusiveSegmentedScan(
              storage, temp_bytes, input, output, offsets, offsets + 1, 2,
              [](int lhs, int rhs) { return lhs > rhs ? lhs : rhs; }) == cudaSuccess &&
              output[0] == 1 && output[1] == 2 && output[2] == 3 &&
              output[3] == 9 && output[4] == 9,
          "DeviceSegmentedScan inclusive custom operation");
    CHECK(cub::DeviceSegmentedScan::ExclusiveSegmentedSum(
              storage, temp_bytes, input, output, offsets, offsets + 1, -1) ==
              cudaErrorInvalidValue,
          "DeviceSegmentedScan rejects a negative segment count");
}

static void test_device_transform() {
    int lhs[] = {1, 2, 3};
    int rhs[] = {4, 5, 6};
    int sum[3] = {};
    int difference[3] = {};

    CHECK(cub::DeviceTransform::Transform(
              std::tuple{lhs, rhs}, std::tuple{sum, difference}, 3,
              [](int a, int b) { return std::tuple{a + b, a - b}; }) == cudaSuccess &&
              sum[0] == 5 && sum[1] == 7 && sum[2] == 9 &&
              difference[0] == -3 && difference[1] == -3 && difference[2] == -3,
          "DeviceTransform multi-input multi-output tuple transform");
}

static void test_double_buffer_radix_sort() {
    // The DoubleBuffer form is what NVIDIA Warp's sort.cu and sparse.cu call.
    // The caller reads the result through Current(), so a shim that sorts in
    // place must leave the selector alone.
    int keys[8] = {5, 3, 9, 1, 7, 2, 8, 4};
    int alternate_keys[8] = {};
    int values[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int alternate_values[8] = {};

    cub::DoubleBuffer<int> d_keys(keys, alternate_keys);
    cub::DoubleBuffer<int> d_values(values, alternate_values);
    CHECK(d_keys.Current() == keys && d_keys.Alternate() == alternate_keys,
          "DoubleBuffer selects its first buffer");

    size_t temp_bytes = 0;
    CHECK(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, d_keys, d_values, 8) == cudaSuccess &&
              temp_bytes > 0,
          "DeviceRadixSort DoubleBuffer temp-storage query");

    std::vector<char> storage(temp_bytes);
    CHECK(cub::DeviceRadixSort::SortPairs(storage.data(), temp_bytes, d_keys, d_values, 8) ==
              cudaSuccess,
          "DeviceRadixSort DoubleBuffer sort");

    const int* sorted_keys = d_keys.Current();
    const int* sorted_values = d_values.Current();
    CHECK(sorted_keys[0] == 1 && sorted_keys[3] == 4 && sorted_keys[7] == 9,
          "DeviceRadixSort DoubleBuffer keys ascending");
    // Values follow their keys: key 1 was at index 3, key 9 at index 2.
    CHECK(sorted_values[0] == 3 && sorted_values[7] == 2,
          "DeviceRadixSort DoubleBuffer values follow keys");
}

static void test_radix_sort_is_stable() {
    // CUB's radix sort is stable and callers depend on it: Warp's sparse path
    // run-length encodes the sorted keys, so equal keys must keep their input
    // order for the block indices to pair up.
    const int keys_in[6] = {2, 1, 2, 1, 2, 1};
    const int values_in[6] = {0, 1, 2, 3, 4, 5};
    int keys_out[6] = {};
    int values_out[6] = {};

    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_in, keys_out, values_in, values_out, 6);
    std::vector<char> storage(temp_bytes);
    CHECK(cub::DeviceRadixSort::SortPairs(storage.data(), temp_bytes, keys_in, keys_out, values_in,
                                          values_out, 6) == cudaSuccess &&
              values_out[0] == 1 && values_out[1] == 3 && values_out[2] == 5 &&
              values_out[3] == 0 && values_out[4] == 2 && values_out[5] == 4,
          "DeviceRadixSort keeps equal keys in input order");
}

static void test_segmented_radix_sort() {
    // Three segments, the middle one empty; each is sorted independently and
    // nothing crosses a segment boundary.
    int keys[6] = {9, 4, 7, 3, 8, 1};
    int values[6] = {0, 1, 2, 3, 4, 5};
    const int begin_offsets[3] = {0, 3, 3};
    const int end_offsets[3] = {3, 3, 6};

    cub::DoubleBuffer<int> d_keys(keys, nullptr);
    cub::DoubleBuffer<int> d_values(values, nullptr);

    size_t temp_bytes = 0;
    CHECK(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_bytes, d_keys, d_values, 6, 3,
                                                   begin_offsets, end_offsets, 0, 32) == cudaSuccess,
          "DeviceSegmentedRadixSort temp-storage query");

    std::vector<char> storage(temp_bytes);
    CHECK(cub::DeviceSegmentedRadixSort::SortPairs(storage.data(), temp_bytes, d_keys, d_values, 6, 3,
                                                   begin_offsets, end_offsets, 0, 32) == cudaSuccess &&
              keys[0] == 4 && keys[1] == 7 && keys[2] == 9 &&
              keys[3] == 1 && keys[4] == 3 && keys[5] == 8,
          "DeviceSegmentedRadixSort sorts each segment in place");
    CHECK(values[0] == 1 && values[1] == 2 && values[2] == 0 &&
              values[3] == 5 && values[4] == 3 && values[5] == 4,
          "DeviceSegmentedRadixSort permutes values with their segment");
}

int main() {
    test_block_load();
    test_block_load_partial();
    test_block_store();
    test_device_select_if();
    test_device_select_flagged();
    test_device_select_unique();
    test_device_histogram_even();
    test_device_rle_encode();
    test_device_find();
    test_device_segmented_scan();
    test_device_transform();
    test_double_buffer_radix_sort();
    test_radix_sort_is_stable();
    test_segmented_radix_sort();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
