// Everyday CUDA idioms whose PTX lowering the direct PTX->MSL emitter got
// silently wrong: it compiled every one of these, and each ran and returned a
// plausible-looking wrong answer.
//
//   div_by_const   clang/nvcc lower `x / 3` to mul.hi.s32; the emitter dropped
//                  `.hi` and kept the low product.
//   shfl_float     a float warp shuffle was value-converted to uint and back,
//                  truncating every lane.
//   atomic_offset  atomicAdd(&s[1], v) accumulated into s[0]: the address
//                  displacement was discarded.
//   bool_flag      a `bool*` became `device float*`, so storing true wrote a
//                  4-byte 1.0f whose first byte is zero.
//   guarded_tail   work after `if (i < n) {...}` must still run for i >= n.
//   signed_ops     signed shift/compare on negative data.
//
// One kernel per idiom so a refusal or a wrong answer in one cannot mask the
// others. Each prints PASS/FAIL on its own line.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

__global__ void div_by_const(const int* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] / 3 + in[i] % 7;
}

__global__ void shfl_float(const float* in, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float v = in[i];
    out[i] = v + __shfl_down_sync(0xffffffffu, v, 1);
}

__global__ void atomic_offset(const float* in, float* sums, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&sums[1], in[i]);
}

__global__ void bool_flag(float* data, bool* flag, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f;
    if (i == 0) flag[1] = true;
}

__global__ void guarded_tail(float* data, int* touched, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 2.0f;
    touched[i] = 1;
}

__global__ void signed_ops(const int* in, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] < 0 ? (in[i] >> 1) : -in[i];
}

namespace {

constexpr int kThreads = 64;
int g_failures = 0;

void report(const char* name, bool ok, int bad_index, double got, double want) {
    if (ok) {
        std::printf("PASS: %s\n", name);
    } else {
        ++g_failures;
        std::printf("FAIL: %s [%d] got %.9g want %.9g\n", name, bad_index, got, want);
    }
}

bool launched(const char* name) {
    const cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        ++g_failures;
        std::printf("FAIL: %s did not run: %s\n", name, cudaGetErrorString(err));
        return false;
    }
    return true;
}

template <typename T>
T* upload(const T* host, int count) {
    T* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(T) * count);
    cudaMemcpy(device, host, sizeof(T) * count, cudaMemcpyHostToDevice);
    return device;
}

template <typename T>
void download(T* host, const T* device, int count) {
    cudaMemcpy(host, device, sizeof(T) * count, cudaMemcpyDeviceToHost);
}

void check_div_by_const() {
    int in[kThreads], out[kThreads];
    for (int i = 0; i < kThreads; ++i) in[i] = (i - 32) * 977 + 5;
    int* d_in = upload(in, kThreads);
    int* d_out = upload(in, kThreads);
    div_by_const<<<1, kThreads>>>(d_in, d_out, kThreads);
    if (!launched("div_by_const")) return;
    download(out, d_out, kThreads);
    for (int i = 0; i < kThreads; ++i) {
        const int want = in[i] / 3 + in[i] % 7;
        if (out[i] != want) return report("div_by_const", false, i, out[i], want);
    }
    report("div_by_const", true, 0, 0, 0);
}

void check_shfl_float() {
    float in[kThreads], out[kThreads];
    for (int i = 0; i < kThreads; ++i) in[i] = 0.25f + 1.5f * i;
    float* d_in = upload(in, kThreads);
    float* d_out = upload(in, kThreads);
    shfl_float<<<1, kThreads>>>(d_in, d_out);
    if (!launched("shfl_float")) return;
    download(out, d_out, kThreads);
    for (int i = 0; i < kThreads; ++i) {
        // Lanes 0..30 read their neighbor; lane 31 of each warp reads itself.
        const int src = (i % 32 == 31) ? i : i + 1;
        const float want = in[i] + in[src];
        if (out[i] != want) return report("shfl_float", false, i, out[i], want);
    }
    report("shfl_float", true, 0, 0, 0);
}

void check_atomic_offset() {
    float in[kThreads];
    float want = 0.0f;
    for (int i = 0; i < kThreads; ++i) {
        in[i] = 0.5f * (i % 4);  // exact in binary; order cannot change the sum
        want += in[i];
    }
    const float zeros[3] = {0.0f, 0.0f, 0.0f};
    float sums[3];
    float* d_in = upload(in, kThreads);
    float* d_sums = upload(zeros, 3);
    atomic_offset<<<1, kThreads>>>(d_in, d_sums, kThreads);
    if (!launched("atomic_offset")) return;
    download(sums, d_sums, 3);
    if (sums[0] != 0.0f) return report("atomic_offset", false, 0, sums[0], 0.0);
    if (sums[2] != 0.0f) return report("atomic_offset", false, 2, sums[2], 0.0);
    report("atomic_offset", sums[1] == want, 1, sums[1], want);
}

void check_bool_flag() {
    float data[kThreads];
    for (int i = 0; i < kThreads; ++i) data[i] = static_cast<float>(i);
    const bool flags_init[4] = {false, false, false, false};
    bool flags[4];
    float* d_data = upload(data, kThreads);
    bool* d_flags = upload(flags_init, 4);
    bool_flag<<<1, kThreads>>>(d_data, d_flags, kThreads);
    if (!launched("bool_flag")) return;
    download(flags, d_flags, 4);
    for (int i = 0; i < 4; ++i) {
        const bool want = i == 1;
        if (flags[i] != want) return report("bool_flag", false, i, flags[i], want);
    }
    report("bool_flag", true, 0, 0, 0);
}

void check_guarded_tail() {
    constexpr int n = kThreads / 2;
    float data[kThreads];
    int touched[kThreads];
    for (int i = 0; i < kThreads; ++i) {
        data[i] = static_cast<float>(i);
        touched[i] = 0;
    }
    float* d_data = upload(data, kThreads);
    int* d_touched = upload(touched, kThreads);
    guarded_tail<<<1, kThreads>>>(d_data, d_touched, n);
    if (!launched("guarded_tail")) return;
    download(data, d_data, kThreads);
    download(touched, d_touched, kThreads);
    for (int i = 0; i < kThreads; ++i) {
        const float want = i < n ? 2.0f * i : static_cast<float>(i);
        if (data[i] != want) return report("guarded_tail", false, i, data[i], want);
        if (touched[i] != 1) return report("guarded_tail", false, i, touched[i], 1);
    }
    report("guarded_tail", true, 0, 0, 0);
}

void check_signed_ops() {
    int in[kThreads], out[kThreads];
    for (int i = 0; i < kThreads; ++i) in[i] = (i - 40) * 3;
    int* d_in = upload(in, kThreads);
    int* d_out = upload(in, kThreads);
    signed_ops<<<1, kThreads>>>(d_in, d_out, kThreads);
    if (!launched("signed_ops")) return;
    download(out, d_out, kThreads);
    for (int i = 0; i < kThreads; ++i) {
        const int want = in[i] < 0 ? (in[i] >> 1) : -in[i];
        if (out[i] != want) return report("signed_ops", false, i, out[i], want);
    }
    report("signed_ops", true, 0, 0, 0);
}

}  // namespace

int main() {
    check_div_by_const();
    check_shfl_float();
    check_atomic_offset();
    check_bool_flag();
    check_guarded_tail();
    check_signed_ops();
    if (g_failures != 0) {
        std::printf("FAIL: %d PTX idiom(s) computed wrong results\n", g_failures);
        return 1;
    }
    std::printf("PASS: all PTX idioms\n");
    return 0;
}
