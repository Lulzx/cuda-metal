// #216: a cuFFT transform is bitwise deterministic for a fixed plan, size
// and device, as in CUDA. Each case runs the same forward transform many
// times, through both a reused plan and freshly created plans, and requires
// identical output bytes every time.

#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int kRuns = 12;
int failures = 0;
bool print_digests = false;

std::uint32_t lcg(std::uint32_t* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

float unit(std::uint32_t* state) {
    return static_cast<float>(lcg(state) >> 8) * (1.0f / 16777216.0f) - 0.5f;
}

struct Case {
    const char* name;
    cufftType type;
    int rank;
    int n[3];
    int batch;
};

size_t real_count(const Case& c) {
    size_t count = static_cast<size_t>(c.batch);
    for (int i = 0; i < c.rank; ++i) count *= static_cast<size_t>(c.n[i]);
    return count;
}

size_t complex_out_count(const Case& c) {
    size_t count = static_cast<size_t>(c.batch);
    for (int i = 0; i < c.rank - 1; ++i) count *= static_cast<size_t>(c.n[i]);
    return count * static_cast<size_t>(c.n[c.rank - 1] / 2 + 1);
}

bool make_plan(const Case& c, cufftHandle* plan) {
    int n[3] = {c.n[0], c.n[1], c.n[2]};
    return cufftPlanMany(plan, c.rank, n, nullptr, 1, 0, nullptr, 1, 0, c.type, c.batch) ==
           CUFFT_SUCCESS;
}

// Returns the output bytes of one forward transform.
std::vector<unsigned char> run_once(const Case& c, cufftHandle plan, const void* input_host,
                                    size_t in_bytes, size_t out_bytes) {
    void* in = nullptr;
    void* out = nullptr;
    cudaMalloc(&in, in_bytes);
    cudaMalloc(&out, out_bytes);
    cudaMemcpy(in, input_host, in_bytes, cudaMemcpyHostToDevice);
    cudaMemset(out, 0xab, out_bytes);
    cufftResult r = CUFFT_INVALID_TYPE;
    if (c.type == CUFFT_C2C) {
        r = cufftExecC2C(plan, static_cast<cufftComplex*>(in), static_cast<cufftComplex*>(out),
                         CUFFT_FORWARD);
    } else {
        r = cufftExecR2C(plan, static_cast<cufftReal*>(in), static_cast<cufftComplex*>(out));
    }
    std::vector<unsigned char> bytes(out_bytes);
    if (r != CUFFT_SUCCESS || cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(bytes.data(), out, out_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: %s: transform failed (cufft=%d)\n", c.name, r);
        ++failures;
        bytes.clear();
    }
    cudaFree(in);
    cudaFree(out);
    return bytes;
}

void check_case(const Case& c) {
    const size_t elements = real_count(c);
    const bool c2c = c.type == CUFFT_C2C;
    const size_t in_bytes = elements * (c2c ? sizeof(cufftComplex) : sizeof(cufftReal));
    const size_t out_bytes = (c2c ? elements : complex_out_count(c)) * sizeof(cufftComplex);
    std::vector<float> input(in_bytes / sizeof(float));
    std::uint32_t state = 0x2160216u;
    for (float& v : input) v = unit(&state);

    cufftHandle reused = 0;
    if (!make_plan(c, &reused)) {
        std::fprintf(stderr, "FAIL: %s: plan creation failed\n", c.name);
        ++failures;
        return;
    }
    const std::vector<unsigned char> reference =
        run_once(c, reused, input.data(), in_bytes, out_bytes);
    if (reference.empty()) return;
    int distinct = 0;
    for (int run = 1; run < kRuns; ++run) {
        // Alternate a reused plan with a fresh plan of the same shape.
        cufftHandle plan = reused;
        const bool fresh = (run % 2) == 0;
        if (fresh && !make_plan(c, &plan)) {
            ++failures;
            return;
        }
        const std::vector<unsigned char> bytes = run_once(c, plan, input.data(), in_bytes, out_bytes);
        if (fresh) cufftDestroy(plan);
        if (bytes.size() != reference.size() ||
            std::memcmp(bytes.data(), reference.data(), bytes.size()) != 0) {
            ++distinct;
        }
    }
    cufftDestroy(reused);
    if (print_digests) {
        // FNV-1a of the output, so separate processes can be compared.
        std::uint64_t h = 0xcbf29ce484222325ull;
        for (unsigned char b : reference) h = (h ^ b) * 0x100000001b3ull;
        std::printf("DIGEST %s %016llx\n", c.name, static_cast<unsigned long long>(h));
    }
    if (distinct != 0) {
        std::fprintf(stderr, "FAIL: %s: %d of %d runs differ bitwise from the first\n", c.name,
                     distinct, kRuns - 1);
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    // --digests prints one output digest per case so a driver script can
    // compare separate processes (cold and warm kernel caches).
    print_digests = argc > 1 && std::strcmp(argv[1], "--digests") == 0;
    if (cudaFree(nullptr) != cudaSuccess) return 1;
    const Case cases[] = {
        // Power-of-two Stockham on the GPU.
        {"C2C 1D 8192", CUFFT_C2C, 1, {8192, 1, 1}, 1},
        {"C2C 1D 4096 x4", CUFFT_C2C, 1, {4096, 1, 1}, 4},
        // Mixed radix and Bluestein (prime) lengths.
        {"C2C 1D 6000", CUFFT_C2C, 1, {6000, 1, 1}, 1},
        {"C2C 1D 4099 (prime)", CUFFT_C2C, 1, {4099, 1, 1}, 1},
        {"R2C 1D 8192", CUFFT_R2C, 1, {8192, 1, 1}, 1},
        {"C2C 2D 128x128", CUFFT_C2C, 2, {128, 128, 1}, 1},
        {"R2C 2D 128x128", CUFFT_R2C, 2, {128, 128, 1}, 1},
        // Rank-3 R2C: VkFFT on Metal.
        {"R2C 3D 32x32x32", CUFFT_R2C, 3, {32, 32, 32}, 1},
        // Below the dispatch threshold: CPU.
        {"C2C 1D 64 (CPU)", CUFFT_C2C, 1, {64, 1, 1}, 1},
    };
    for (const Case& c : cases) check_case(c);
    if (failures != 0) return 1;
    std::printf("PASS: cuFFT forward transforms are bitwise deterministic\n");
    return 0;
}
