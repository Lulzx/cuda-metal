// Decoupled look-back scan, the protocol CUB's DeviceScan, DeviceSelect and
// onesweep radix sort use to pass prefixes between thread blocks in one pass.
//
// Each tile publishes a {status, value} descriptor with one scoped PTX store
// (`st.relaxed.gpu.v2.u32`, or `.v2.u64` for 64-bit values) and later tiles
// poll it with one scoped load. The pair must be single-copy atomic: a reader
// that sees the new status with the old value adds a stale prefix. CuMetal
// used to split these into plain 32-bit device accesses, and CUB's scans
// returned wrong prefixes in whole tiles while reporting success.
//
// Tile ids come from an atomic counter, so a tile only ever waits on tiles a
// running block has already claimed. A 32-bit `.acquire`/`.release` flag
// covers the native-width path.
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

constexpr int kTile = 256;
constexpr unsigned kInvalid = 0, kPartial = 1, kInclusive = 2;

__device__ void publish32(unsigned long long* slot, unsigned status, unsigned value) {
    asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};" ::"l"(slot), "r"(status), "r"(value)
                 : "memory");
}
__device__ void poll32(const unsigned long long* slot, unsigned& status, unsigned& value) {
    asm volatile("ld.relaxed.gpu.v2.u32 {%0, %1}, [%2];"
                 : "=r"(status), "=r"(value)
                 : "l"(slot)
                 : "memory");
}
__device__ void publish64(ulonglong2* slot, unsigned long long status, unsigned long long value) {
    asm volatile("st.relaxed.gpu.v2.u64 [%0], {%1, %2};" ::"l"(slot), "l"(status), "l"(value)
                 : "memory");
}
__device__ void poll64(const ulonglong2* slot, unsigned long long& status,
                       unsigned long long& value) {
    asm volatile("ld.relaxed.gpu.v2.u64 {%0, %1}, [%2];"
                 : "=l"(status), "=l"(value)
                 : "l"(slot)
                 : "memory");
}

template <typename T>
__device__ T block_sum(T value, T* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kTile / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const T total = scratch[0];
    __syncthreads();
    return total;
}

// out[t] = inclusive prefix of the per-tile sums, t = tile id.
__global__ void scan32(const unsigned* in, unsigned* out, unsigned long long* descriptors,
                       unsigned* counter, unsigned* done_flag, int tiles) {
    __shared__ unsigned scratch[kTile];
    __shared__ unsigned tile_id;
    if (threadIdx.x == 0) tile_id = atomicAdd(counter, 1u);
    __syncthreads();
    const unsigned tile = tile_id;
    const unsigned aggregate = block_sum(in[tile * kTile + threadIdx.x], scratch);
    if (threadIdx.x == 0) {
        unsigned prefix = 0;
        if (tile == 0) {
            publish32(&descriptors[0], kInclusive, aggregate);
        } else {
            publish32(&descriptors[tile], kPartial, aggregate);
            for (int j = static_cast<int>(tile) - 1; j >= 0; --j) {
                unsigned status, value;
                do { poll32(&descriptors[j], status, value); } while (status == kInvalid);
                prefix += value;
                if (status == kInclusive) break;
            }
            publish32(&descriptors[tile], kInclusive, prefix + aggregate);
        }
        out[tile] = prefix + aggregate;
        if (tile == static_cast<unsigned>(tiles) - 1)
            asm volatile("st.release.gpu.u32 [%0], %1;" ::"l"(done_flag), "r"(1u) : "memory");
    }
}

__global__ void scan64(const unsigned long long* in, unsigned long long* out,
                       ulonglong2* descriptors, unsigned* counter, int tiles) {
    __shared__ unsigned long long scratch[kTile];
    __shared__ unsigned tile_id;
    if (threadIdx.x == 0) tile_id = atomicAdd(counter, 1u);
    __syncthreads();
    const unsigned tile = tile_id;
    const unsigned long long aggregate = block_sum(in[tile * kTile + threadIdx.x], scratch);
    if (threadIdx.x == 0) {
        unsigned long long prefix = 0;
        if (tile == 0) {
            publish64(&descriptors[0], kInclusive, aggregate);
        } else {
            publish64(&descriptors[tile], kPartial, aggregate);
            for (int j = static_cast<int>(tile) - 1; j >= 0; --j) {
                unsigned long long status, value;
                do { poll64(&descriptors[j], status, value); } while (status == kInvalid);
                prefix += value;
                if (status == kInclusive) break;
            }
            publish64(&descriptors[tile], kInclusive, prefix + aggregate);
        }
        out[tile] = prefix + aggregate;
    }
    (void)tiles;
}

__global__ void read_flag(const unsigned* flag, unsigned* seen) {
    unsigned value;
    asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(value) : "l"(flag) : "memory");
    *seen = value;
}

#define CHECK(call)                                                                  \
    do {                                                                             \
        cudaError_t e = (call);                                                      \
        if (e != cudaSuccess) {                                                      \
            std::printf("FAIL: %s -> %s\n", #call, cudaGetErrorName(e));            \
            return 1;                                                                \
        }                                                                            \
    } while (0)

int main() {
    const int tiles = 2048, n = tiles * kTile, reps = 20;
    std::vector<unsigned> h32(n);
    std::vector<unsigned long long> h64(n);
    for (int i = 0; i < n; ++i) {
        h32[i] = static_cast<unsigned>(i % 13);
        h64[i] = 3000000000ull + static_cast<unsigned long long>(i % 17);
    }
    std::vector<unsigned> want32(tiles);
    std::vector<unsigned long long> want64(tiles);
    unsigned run32 = 0;
    unsigned long long run64 = 0;
    for (int t = 0; t < tiles; ++t) {
        for (int i = 0; i < kTile; ++i) {
            run32 += h32[t * kTile + i];
            run64 += h64[t * kTile + i];
        }
        want32[t] = run32;
        want64[t] = run64;
    }

    unsigned *in32, *out32, *counter, *flag, *seen;
    unsigned long long *in64, *out64, *desc32;
    ulonglong2* desc64;
    CHECK(cudaMalloc(&in32, n * sizeof(unsigned)));
    CHECK(cudaMalloc(&out32, tiles * sizeof(unsigned)));
    CHECK(cudaMalloc(&in64, n * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&out64, tiles * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&desc32, tiles * sizeof(unsigned long long)));
    CHECK(cudaMalloc(&desc64, tiles * sizeof(ulonglong2)));
    CHECK(cudaMalloc(&counter, sizeof(unsigned)));
    CHECK(cudaMalloc(&flag, sizeof(unsigned)));
    CHECK(cudaMalloc(&seen, sizeof(unsigned)));
    CHECK(cudaMemcpy(in32, h32.data(), n * sizeof(unsigned), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(in64, h64.data(), n * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    int wrong32 = 0, wrong64 = 0;
    for (int r = 0; r < reps; ++r) {
        CHECK(cudaMemset(desc32, 0, tiles * sizeof(unsigned long long)));
        CHECK(cudaMemset(desc64, 0, tiles * sizeof(ulonglong2)));
        CHECK(cudaMemset(counter, 0, sizeof(unsigned)));
        CHECK(cudaMemset(flag, 0, sizeof(unsigned)));
        scan32<<<tiles, kTile>>>(in32, out32, desc32, counter, flag, tiles);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaMemset(counter, 0, sizeof(unsigned)));
        scan64<<<tiles, kTile>>>(in64, out64, desc64, counter, tiles);
        CHECK(cudaGetLastError());
        CHECK(cudaDeviceSynchronize());
        std::vector<unsigned> got32(tiles);
        std::vector<unsigned long long> got64(tiles);
        CHECK(cudaMemcpy(got32.data(), out32, tiles * sizeof(unsigned), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(got64.data(), out64, tiles * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
        for (int t = 0; t < tiles; ++t) {
            if (got32[t] != want32[t]) { ++wrong32; break; }
        }
        for (int t = 0; t < tiles; ++t) {
            if (got64[t] != want64[t]) { ++wrong64; break; }
        }
    }
    read_flag<<<1, 1>>>(flag, seen);
    CHECK(cudaDeviceSynchronize());
    unsigned seen_host = 0;
    CHECK(cudaMemcpy(&seen_host, seen, sizeof(unsigned), cudaMemcpyDeviceToHost));
    if (wrong32 != 0 || wrong64 != 0 || seen_host != 1) {
        std::printf("FAIL: look-back scan wrong in %d/%d (v2.u32) and %d/%d (v2.u64) runs; "
                    "release flag %u\n", wrong32, reps, wrong64, reps, seen_host);
        return 1;
    }
    std::printf("PASS: decoupled look-back scan with scoped PTX descriptors\n");
    return 0;
}
