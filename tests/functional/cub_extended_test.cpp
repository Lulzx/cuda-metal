#include <cub/cub.h>
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

int main() {
    test_block_load();
    test_block_load_partial();
    test_block_store();
    test_device_select_if();
    test_device_select_flagged();
    test_device_select_unique();
    test_device_histogram_even();
    test_device_rle_encode();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
