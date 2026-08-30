#include "cuda_runtime.h"
#include "curand.h"

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <limits>
#include <thread>
#include <vector>

namespace {
struct HostGate {
    std::atomic<bool> entered{false};
    std::atomic<bool> release{false};
};

void wait_on_gate(void* data) {
    auto* gate = static_cast<HostGate*>(data);
    gate->entered.store(true, std::memory_order_release);
    while (!gate->release.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
}
}  // namespace

int main() {
    constexpr std::size_t kCount = 4096;

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    int curand_version = 0;
    if (curandGetVersion(&curand_version) != CURAND_STATUS_SUCCESS || curand_version <= 0) {
        std::fprintf(stderr, "FAIL: curandGetVersion failed\n");
        return 1;
    }
    if (curandGetVersion(nullptr) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null version ptr\n");
        return 1;
    }

    curandGenerator_t generator = nullptr;
    if (curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandCreateGenerator failed\n");
        return 1;
    }
    if (curandSetPseudoRandomGeneratorSeed(generator, 0xC0FFEEULL) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandSetPseudoRandomGeneratorSeed failed\n");
        return 1;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }
    if (curandSetStream(generator, stream) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandSetStream failed\n");
        return 1;
    }
    cudaStream_t queried_stream = nullptr;
    if (curandGetStream(generator, &queried_stream) != CURAND_STATUS_SUCCESS ||
        queried_stream != stream) {
        std::fprintf(stderr, "FAIL: curandGetStream mismatch\n");
        return 1;
    }

    float* ordered_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&ordered_output), sizeof(float)) != cudaSuccess ||
        cudaMemset(ordered_output, 0, sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cuRAND ordering buffer setup failed\n");
        return 1;
    }
    HostGate gate;
    if (cudaLaunchHostFunc(stream, wait_on_gate, &gate) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cuRAND ordering gate enqueue failed\n");
        return 1;
    }
    while (!gate.entered.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }
    const auto generate_start = std::chrono::steady_clock::now();
    if (curandGenerateUniform(generator, ordered_output, 1) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: queued cuRAND generation failed\n");
        return 1;
    }
    const auto generate_elapsed = std::chrono::steady_clock::now() - generate_start;
    if (generate_elapsed > std::chrono::milliseconds(100) || ordered_output[0] != 0.0f) {
        std::fprintf(stderr, "FAIL: cuRAND generation blocked or overtook its stream\n");
        return 1;
    }
    gate.release.store(true, std::memory_order_release);
    if (cudaStreamSynchronize(stream) != cudaSuccess || ordered_output[0] == 0.0f) {
        std::fprintf(stderr, "FAIL: queued cuRAND generation did not complete after stream sync\n");
        return 1;
    }
    if (cudaFree(ordered_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cuRAND ordering buffer cleanup failed\n");
        return 1;
    }
    if (curandSetGeneratorOffset(nullptr, 0) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null generator offset\n");
        return 1;
    }

    constexpr std::size_t kOffsetSkip = 257;
    constexpr std::size_t kOffsetSample = 128;
    std::uint32_t* device_offset_sample = nullptr;
    std::uint32_t* device_offset_skipped = nullptr;
    std::uint32_t* device_offset_reference = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_offset_sample), kOffsetSample * sizeof(std::uint32_t)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_offset_skipped), kOffsetSample * sizeof(std::uint32_t)) !=
            cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&device_offset_reference),
                   (kOffsetSkip + kOffsetSample) * sizeof(std::uint32_t)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for offset test buffers failed\n");
        return 1;
    }

    if (curandSetGeneratorOffset(generator, 0) != CURAND_STATUS_SUCCESS ||
        curandGenerate(generator,
                       reinterpret_cast<unsigned int*>(device_offset_sample),
                       kOffsetSample) != CURAND_STATUS_SUCCESS ||
        curandSetGeneratorOffset(generator, kOffsetSkip) != CURAND_STATUS_SUCCESS ||
        curandGenerate(generator,
                       reinterpret_cast<unsigned int*>(device_offset_skipped),
                       kOffsetSample) != CURAND_STATUS_SUCCESS ||
        curandSetGeneratorOffset(generator, 0) != CURAND_STATUS_SUCCESS ||
        curandGenerate(generator,
                       reinterpret_cast<unsigned int*>(device_offset_reference),
                       kOffsetSkip + kOffsetSample) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curand offset sequence generation failed\n");
        return 1;
    }

    std::vector<std::uint32_t> host_offset_skipped(kOffsetSample, 0);
    std::vector<std::uint32_t> host_offset_reference(kOffsetSkip + kOffsetSample, 0);
    if (cudaMemcpy(host_offset_skipped.data(),
                   device_offset_skipped,
                   kOffsetSample * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(host_offset_reference.data(),
                   device_offset_reference,
                   (kOffsetSkip + kOffsetSample) * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy for offset sequence validation failed\n");
        return 1;
    }
    for (std::size_t i = 0; i < kOffsetSample; ++i) {
        if (host_offset_skipped[i] != host_offset_reference[kOffsetSkip + i]) {
            std::fprintf(stderr, "FAIL: curand offset mismatch at %zu\n", i);
            return 1;
        }
    }

    if (cudaFree(device_offset_sample) != cudaSuccess ||
        cudaFree(device_offset_skipped) != cudaSuccess ||
        cudaFree(device_offset_reference) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for offset test buffers failed\n");
        return 1;
    }
    if (curandSetPseudoRandomGeneratorSeed(generator, 0xC0FFEEULL) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandSetPseudoRandomGeneratorSeed reset failed\n");
        return 1;
    }

    float* device_output = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output), kCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (curandGenerateUniform(generator, device_output, kCount) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateUniform failed\n");
        return 1;
    }

    std::vector<float> host(kCount, 0.0f);
    if (cudaMemcpy(host.data(), device_output, kCount * sizeof(float), cudaMemcpyDeviceToHost) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    float min_value = 1.0f;
    float max_value = 0.0f;
    bool has_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        const float value = host[i];
        if (value < 0.0f || value > 1.0f) {
            std::fprintf(stderr, "FAIL: generated value out of range [0,1]: %f\n", value);
            return 1;
        }
        if (i > 0 && host[i] != host[i - 1]) {
            has_variation = true;
        }
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }
    }

    if (!has_variation) {
        std::fprintf(stderr, "FAIL: generated sequence has no variation\n");
        return 1;
    }
    if (min_value == max_value) {
        std::fprintf(stderr, "FAIL: generated sequence collapsed to a constant\n");
        return 1;
    }

    double* device_output_double = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_double), kCount * sizeof(double)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for double output failed\n");
        return 1;
    }
    if (curandGenerateUniformDouble(generator, device_output_double, kCount) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateUniformDouble failed\n");
        return 1;
    }

    std::vector<double> host_double(kCount, 0.0);
    if (cudaMemcpy(host_double.data(),
                   device_output_double,
                   kCount * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for double output failed\n");
        return 1;
    }
    bool has_double_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        const double value = host_double[i];
        if (value < 0.0 || value > 1.0) {
            std::fprintf(stderr, "FAIL: generated double value out of range [0,1]: %f\n", value);
            return 1;
        }
        if (i > 0 && host_double[i] != host_double[i - 1]) {
            has_double_variation = true;
        }
    }
    if (!has_double_variation) {
        std::fprintf(stderr, "FAIL: generated double sequence has no variation\n");
        return 1;
    }

    float* device_output_normal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_normal), kCount * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for normal output failed\n");
        return 1;
    }
    constexpr float kNormalMean = 1.5f;
    constexpr float kNormalStddev = 0.7f;
    if (curandGenerateNormal(generator, device_output_normal, kCount, kNormalMean, kNormalStddev) !=
        CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateNormal failed\n");
        return 1;
    }

    std::vector<float> host_normal(kCount, 0.0f);
    if (cudaMemcpy(host_normal.data(),
                   device_output_normal,
                   kCount * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for normal output failed\n");
        return 1;
    }
    double sum = 0.0;
    bool has_normal_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        sum += static_cast<double>(host_normal[i]);
        if (i > 0 && host_normal[i] != host_normal[i - 1]) {
            has_normal_variation = true;
        }
    }
    const double sample_mean = sum / static_cast<double>(kCount);
    if (!has_normal_variation) {
        std::fprintf(stderr, "FAIL: generated normal sequence has no variation\n");
        return 1;
    }
    if (sample_mean < 1.2 || sample_mean > 1.8) {
        std::fprintf(stderr,
                     "FAIL: normal sample mean out of expected range (got=%f expected~%f)\n",
                     sample_mean,
                     static_cast<double>(kNormalMean));
        return 1;
    }
    if (curandGenerateNormal(generator, device_output_normal, kCount, kNormalMean, 0.0f) !=
        CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for zero stddev\n");
        return 1;
    }
    if (curandGenerateNormal(generator, device_output_normal, 1, kNormalMean, kNormalStddev) !=
        CURAND_STATUS_LENGTH_NOT_MULTIPLE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_LENGTH_NOT_MULTIPLE for odd normal count\n");
        return 1;
    }
    if (curandGenerateNormal(generator, device_output_normal, kCount,
                             std::numeric_limits<float>::quiet_NaN(), kNormalStddev) !=
        CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for NaN normal mean\n");
        return 1;
    }

    double* device_output_normal_double = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_normal_double), kCount * sizeof(double)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for normal double output failed\n");
        return 1;
    }
    constexpr double kNormalDoubleMean = -0.75;
    constexpr double kNormalDoubleStddev = 1.1;
    if (curandGenerateNormalDouble(generator,
                                   device_output_normal_double,
                                   kCount,
                                   kNormalDoubleMean,
                                   kNormalDoubleStddev) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateNormalDouble failed\n");
        return 1;
    }

    std::vector<double> host_normal_double(kCount, 0.0);
    if (cudaMemcpy(host_normal_double.data(),
                   device_output_normal_double,
                   kCount * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for normal double output failed\n");
        return 1;
    }
    double sum_double = 0.0;
    bool has_normal_double_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        sum_double += host_normal_double[i];
        if (i > 0 && host_normal_double[i] != host_normal_double[i - 1]) {
            has_normal_double_variation = true;
        }
    }
    const double sample_double_mean = sum_double / static_cast<double>(kCount);
    if (!has_normal_double_variation) {
        std::fprintf(stderr, "FAIL: generated normal double sequence has no variation\n");
        return 1;
    }
    if (sample_double_mean < -1.05 || sample_double_mean > -0.45) {
        std::fprintf(stderr,
                     "FAIL: normal double sample mean out of expected range (got=%f expected~%f)\n",
                     sample_double_mean,
                     kNormalDoubleMean);
        return 1;
    }
    if (curandGenerateNormalDouble(generator,
                                   device_output_normal_double,
                                   kCount,
                                   kNormalDoubleMean,
                                   0.0) != CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr,
                     "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for zero stddev (double)\n");
        return 1;
    }

    float* device_output_lognormal = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_lognormal), kCount * sizeof(float)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for lognormal output failed\n");
        return 1;
    }
    if (curandGenerateLogNormal(generator, device_output_lognormal, kCount, 0.1f, 0.6f) !=
        CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateLogNormal failed\n");
        return 1;
    }
    std::vector<float> host_lognormal(kCount, 0.0f);
    if (cudaMemcpy(host_lognormal.data(),
                   device_output_lognormal,
                   kCount * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for lognormal output failed\n");
        return 1;
    }
    bool has_lognormal_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        if (host_lognormal[i] <= 0.0f) {
            std::fprintf(stderr, "FAIL: lognormal output must be > 0 (got=%f)\n", host_lognormal[i]);
            return 1;
        }
        if (i > 0 && host_lognormal[i] != host_lognormal[i - 1]) {
            has_lognormal_variation = true;
        }
    }
    if (!has_lognormal_variation) {
        std::fprintf(stderr, "FAIL: generated lognormal sequence has no variation\n");
        return 1;
    }
    if (curandGenerateLogNormal(generator, device_output_lognormal, kCount, 0.1f, 0.0f) !=
        CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for zero lognormal stddev\n");
        return 1;
    }
    if (curandGenerateLogNormal(generator, device_output_lognormal, 1, 0.1f, 0.6f) !=
        CURAND_STATUS_LENGTH_NOT_MULTIPLE) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_LENGTH_NOT_MULTIPLE for odd lognormal count\n");
        return 1;
    }

    double* device_output_lognormal_double = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_lognormal_double), kCount * sizeof(double)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for lognormal double output failed\n");
        return 1;
    }
    if (curandGenerateLogNormalDouble(generator,
                                      device_output_lognormal_double,
                                      kCount,
                                      -0.2,
                                      0.9) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateLogNormalDouble failed\n");
        return 1;
    }
    std::vector<double> host_lognormal_double(kCount, 0.0);
    if (cudaMemcpy(host_lognormal_double.data(),
                   device_output_lognormal_double,
                   kCount * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for lognormal double output failed\n");
        return 1;
    }
    bool has_lognormal_double_variation = false;
    for (std::size_t i = 0; i < kCount; ++i) {
        if (host_lognormal_double[i] <= 0.0) {
            std::fprintf(stderr,
                         "FAIL: lognormal double output must be > 0 (got=%f)\n",
                         host_lognormal_double[i]);
            return 1;
        }
        if (i > 0 && host_lognormal_double[i] != host_lognormal_double[i - 1]) {
            has_lognormal_double_variation = true;
        }
    }
    if (!has_lognormal_double_variation) {
        std::fprintf(stderr, "FAIL: generated lognormal double sequence has no variation\n");
        return 1;
    }
    if (curandGenerateLogNormalDouble(generator,
                                      device_output_lognormal_double,
                                      kCount,
                                      -0.2,
                                      0.0) != CURAND_STATUS_OUT_OF_RANGE) {
        std::fprintf(stderr,
                     "FAIL: expected CURAND_STATUS_OUT_OF_RANGE for zero lognormal stddev (double)\n");
        return 1;
    }

    std::uint32_t* device_output_uint = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_uint), kCount * sizeof(std::uint32_t)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for uint output failed\n");
        return 1;
    }
    if (curandGenerate(generator, reinterpret_cast<unsigned int*>(device_output_uint), kCount) !=
        CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerate failed\n");
        return 1;
    }
    std::vector<std::uint32_t> host_uint(kCount, 0);
    if (cudaMemcpy(host_uint.data(),
                   device_output_uint,
                   kCount * sizeof(std::uint32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for uint output failed\n");
        return 1;
    }
    bool has_uint_variation = false;
    for (std::size_t i = 1; i < kCount; ++i) {
        if (host_uint[i] != host_uint[i - 1]) {
            has_uint_variation = true;
            break;
        }
    }
    if (!has_uint_variation) {
        std::fprintf(stderr, "FAIL: generated uint sequence has no variation\n");
        return 1;
    }

    std::uint64_t* device_output_ull = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_output_ull), kCount * sizeof(std::uint64_t)) !=
        cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc for uint64 output failed\n");
        return 1;
    }
    if (curandGenerateLongLong(generator,
                               reinterpret_cast<unsigned long long*>(device_output_ull),
                               kCount) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandGenerateLongLong failed\n");
        return 1;
    }
    std::vector<std::uint64_t> host_ull(kCount, 0);
    if (cudaMemcpy(host_ull.data(),
                   device_output_ull,
                   kCount * sizeof(std::uint64_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host for uint64 output failed\n");
        return 1;
    }
    bool has_ull_variation = false;
    for (std::size_t i = 1; i < kCount; ++i) {
        if (host_ull[i] != host_ull[i - 1]) {
            has_ull_variation = true;
            break;
        }
    }
    if (!has_ull_variation) {
        std::fprintf(stderr, "FAIL: generated uint64 sequence has no variation\n");
        return 1;
    }

    std::vector<unsigned int> host_uint_out(kCount, 0u);
    if (curandGenerate(generator, host_uint_out.data(), kCount) != CURAND_STATUS_TYPE_ERROR) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_TYPE_ERROR for host uint output pointer\n");
        return 1;
    }
    std::vector<unsigned long long> host_ull_out(kCount, 0ull);
    if (curandGenerateLongLong(generator, host_ull_out.data(), kCount) != CURAND_STATUS_TYPE_ERROR) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_TYPE_ERROR for host uint64 output pointer\n");
        return 1;
    }

    std::vector<float> host_output(kCount, 0.0f);
    if (curandGenerateUniform(generator, host_output.data(), kCount) != CURAND_STATUS_TYPE_ERROR) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_TYPE_ERROR for host output pointer\n");
        return 1;
    }

    if (curandCreateGenerator(nullptr, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null generator out ptr\n");
        return 1;
    }
    if (curandSetStream(nullptr, stream) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null generator in setStream\n");
        return 1;
    }
    if (curandGetStream(generator, nullptr) != CURAND_STATUS_NOT_INITIALIZED) {
        std::fprintf(stderr, "FAIL: expected CURAND_STATUS_NOT_INITIALIZED for null out stream ptr\n");
        return 1;
    }

    if (cudaFree(device_output) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }
    if (cudaFree(device_output_double) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for double output failed\n");
        return 1;
    }
    if (cudaFree(device_output_normal) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for normal output failed\n");
        return 1;
    }
    if (cudaFree(device_output_normal_double) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for normal double output failed\n");
        return 1;
    }
    if (cudaFree(device_output_lognormal) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for lognormal output failed\n");
        return 1;
    }
    if (cudaFree(device_output_lognormal_double) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for lognormal double output failed\n");
        return 1;
    }
    if (cudaFree(device_output_uint) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for uint output failed\n");
        return 1;
    }
    if (cudaFree(device_output_ull) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree for uint64 output failed\n");
        return 1;
    }
    if (curandDestroyGenerator(generator) != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: curandDestroyGenerator failed\n");
        return 1;
    }
    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    std::printf("PASS: cuRAND generation + stream binding shim works on device allocations\n");
    return 0;
}
