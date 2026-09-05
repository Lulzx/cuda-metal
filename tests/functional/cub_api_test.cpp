#include <cub/cub.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { fprintf(stderr, "FAIL: %s\n", msg); g_fail++; } \
    else { printf("PASS: %s\n", msg); } \
} while(0)

static void test_block_reduce() {
    constexpr int BLOCK = 4;
    typename cub::BlockReduce<float, BLOCK>::TempStorage temp;
    cub::BlockReduce<float, BLOCK> reducer(temp, 0);

    // Simulate thread 0 calling Reduce with data from all threads
    // In this sequential fallback, thread 0 writes all values then reduces.
    // TempStorage is uninitialized bytes so that it is legal in __shared__, so
    // the slots are reached through data() rather than as an array member.
    temp.data()[0] = 1.0f;
    temp.data()[1] = 2.0f;
    temp.data()[2] = 3.0f;
    temp.data()[3] = 4.0f;
    float result = 0;
    // Thread 0 performs the full reduction
    {
        cub::BlockReduce<float, BLOCK> r(temp, 0);
        result = r.Sum(1.0f); // thread 0's input=1, but all slots pre-filled
    }
    // Since thread 0 sees data[0]=input(1.0), data[1..3] from pre-fill:
    // Sum writes data[0]=1.0, then sums data[0..3] = 1+2+3+4=10
    CHECK(result == 10.0f, "BlockReduce::Sum");
}

static void test_block_scan() {
    constexpr int BLOCK = 4;
    typename cub::BlockScan<int, BLOCK>::TempStorage temp;

    // Pre-fill values for all threads
    temp.data[0] = 1;
    temp.data[1] = 2;
    temp.data[2] = 3;
    temp.data[3] = 4;

    cub::BlockScan<int, BLOCK> scanner(temp, 0);
    int out;
    scanner.ExclusiveSum(1, out);
    CHECK(out == 0, "BlockScan::ExclusiveSum thread0=0");

    // Check that thread 2 got prefix sum of 1+2=3
    CHECK(temp.data[2] == 3, "BlockScan::ExclusiveSum thread2=3");
}

static void test_warp_reduce() {
    constexpr int WARP = 4;
    typename cub::WarpReduce<float, WARP>::TempStorage temp;

    // Pre-fill warp data
    temp.data[0] = 10.0f;
    temp.data[1] = 20.0f;
    temp.data[2] = 30.0f;
    temp.data[3] = 40.0f;

    cub::WarpReduce<float, WARP> reducer(temp, 0);
    float result = reducer.Sum(10.0f); // lane 0
    CHECK(result == 100.0f, "WarpReduce::Sum");
}

static void test_warp_scan() {
    constexpr int WARP = 4;
    typename cub::WarpScan<int, WARP>::TempStorage temp;

    temp.data[0] = 1;
    temp.data[1] = 2;
    temp.data[2] = 3;
    temp.data[3] = 4;

    cub::WarpScan<int, WARP> scanner(temp, 0);
    int out;
    scanner.InclusiveSum(1, out);
    CHECK(out == 1, "WarpScan::InclusiveSum lane0=1");
    CHECK(temp.data[3] == 10, "WarpScan::InclusiveSum lane3=10");
}

static void test_device_reduce_sum() {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float result = 0;

    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(nullptr, temp_bytes, data, &result, 5);
    CHECK(temp_bytes == 1, "DeviceReduce::Sum temp_bytes query");

    char temp_buf[1];
    cub::DeviceReduce::Sum(temp_buf, temp_bytes, data, &result, 5);
    CHECK(result == 15.0f, "DeviceReduce::Sum result=15");
}

static void test_device_reduce_min_max() {
    int data[] = {5, 3, 8, 1, 7};
    int result;
    size_t temp_bytes = 0;
    char temp_buf[1];

    cub::DeviceReduce::Min(nullptr, temp_bytes, data, &result, 5);
    cub::DeviceReduce::Min(temp_buf, temp_bytes, data, &result, 5);
    CHECK(result == 1, "DeviceReduce::Min result=1");

    cub::DeviceReduce::Max(temp_buf, temp_bytes, data, &result, 5);
    CHECK(result == 8, "DeviceReduce::Max result=8");
}

static void test_device_scan() {
    int in[] = {1, 2, 3, 4, 5};
    int out[5];
    size_t temp_bytes = 0;
    char temp_buf[1];

    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, in, out, 5);
    cub::DeviceScan::ExclusiveSum(temp_buf, temp_bytes, in, out, 5);
    CHECK(out[0] == 0 && out[1] == 1 && out[2] == 3 && out[3] == 6 && out[4] == 10,
          "DeviceScan::ExclusiveSum");

    cub::DeviceScan::InclusiveSum(temp_buf, temp_bytes, in, out, 5);
    CHECK(out[0] == 1 && out[1] == 3 && out[2] == 6 && out[3] == 10 && out[4] == 15,
          "DeviceScan::InclusiveSum");
}

static void test_device_radix_sort() {
    int keys_in[] = {5, 3, 8, 1, 7};
    int keys_out[5];
    size_t temp_bytes = 0;

    cub::DeviceRadixSort::SortKeys(nullptr, temp_bytes, keys_in, keys_out, 5);
    CHECK(temp_bytes > 0, "DeviceRadixSort::SortKeys temp_bytes query");

    std::vector<char> temp_buf(temp_bytes);
    cub::DeviceRadixSort::SortKeys(temp_buf.data(), temp_bytes, keys_in, keys_out, 5);
    CHECK(keys_out[0] == 1 && keys_out[1] == 3 && keys_out[2] == 5 &&
          keys_out[3] == 7 && keys_out[4] == 8,
          "DeviceRadixSort::SortKeys ascending");

    cub::DeviceRadixSort::SortKeysDescending(temp_buf.data(), temp_bytes, keys_in, keys_out, 5);
    CHECK(keys_out[0] == 8 && keys_out[1] == 7 && keys_out[2] == 5 &&
          keys_out[3] == 3 && keys_out[4] == 1,
          "DeviceRadixSort::SortKeysDescending");
}

static void test_device_radix_sort_pairs() {
    float keys_in[] = {3.0f, 1.0f, 2.0f};
    float keys_out[3];
    int vals_in[] = {30, 10, 20};
    int vals_out[3];
    size_t temp_bytes = 0;

    cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys_in, keys_out, vals_in, vals_out, 3);
    std::vector<char> temp_buf(temp_bytes);
    cub::DeviceRadixSort::SortPairs(temp_buf.data(), temp_bytes, keys_in, keys_out, vals_in, vals_out, 3);
    CHECK(keys_out[0] == 1.0f && keys_out[1] == 2.0f && keys_out[2] == 3.0f,
          "DeviceRadixSort::SortPairs keys");
    CHECK(vals_out[0] == 10 && vals_out[1] == 20 && vals_out[2] == 30,
          "DeviceRadixSort::SortPairs values");
}

static void test_block_exchange() {
    constexpr int BLOCK = 2;
    constexpr int ITEMS = 3;
    typename cub::BlockExchange<int, BLOCK, ITEMS>::TempStorage temp;

    cub::BlockExchange<int, BLOCK, ITEMS> exchange(temp, 0);
    int items[ITEMS] = {1, 2, 3};
    int out[ITEMS];
    exchange.BlockedToStriped(items, out);
    // Thread 0 blocked items: [1,2,3] at positions 0,1,2
    // Striped output for thread 0: positions 0, 2, 4 → data[0], data[2], data[4]
    CHECK(out[0] == 1 && out[1] == 3, "BlockExchange::BlockedToStriped");
}

int main() {
    test_block_reduce();
    test_block_scan();
    test_warp_reduce();
    test_warp_scan();
    test_device_reduce_sum();
    test_device_reduce_min_max();
    test_device_scan();
    test_device_radix_sort();
    test_device_radix_sort_pairs();
    test_block_exchange();

    printf("\n%s (%d failures)\n", g_fail ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_fail);
    return g_fail ? 1 : 0;
}
