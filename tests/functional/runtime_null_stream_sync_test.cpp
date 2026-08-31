#include "cuda_runtime.h"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kElementCount = 1u << 18;
constexpr std::size_t kThreadsPerBlock = 256;
// Long enough for the marker on another queue to overtake it when a required
// legacy-stream wait is missing, but short enough to stay below Metal's GPU
// watchdog even after the external conformance workload has heated the device.
constexpr unsigned int kSpinIterations = 10000000u;

bool nearly_equal(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 64;
    }

    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

    if (cudaInit(0) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaInit failed\n");
        return 1;
    }

    std::vector<float> host_a(kElementCount);
    std::vector<float> host_b(kElementCount);
    std::vector<float> host_out(kElementCount, 0.0f);

    for (std::size_t i = 0; i < kElementCount; ++i) {
        host_a[i] = static_cast<float>((i * 13) % 29) * 0.75f;
        host_b[i] = static_cast<float>((i * 5) % 31) * 0.5f;
    }

    void* dev_a = nullptr;
    void* dev_b = nullptr;
    void* dev_tmp = nullptr;
    void* dev_out = nullptr;
    const std::size_t bytes = kElementCount * sizeof(float);
    if (cudaMalloc(&dev_a, bytes) != cudaSuccess || cudaMalloc(&dev_b, bytes) != cudaSuccess ||
        cudaMalloc(&dev_tmp, bytes) != cudaSuccess || cudaMalloc(&dev_out, bytes) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc failed\n");
        return 1;
    }

    if (cudaMemcpy(dev_a, host_a.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dev_b, host_b.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy host->device failed\n");
        return 1;
    }

    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess || stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreate failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BUFFER, 0},
    };
    const cumetalKernel_t kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "vector_add",
        .arg_count = 3,
        .arg_info = kArgInfo,
    };

    const dim3 block_dim(static_cast<unsigned int>(kThreadsPerBlock), 1, 1);
    const dim3 grid_dim(static_cast<unsigned int>((kElementCount + kThreadsPerBlock - 1) /
                                                   kThreadsPerBlock),
                        1, 1);

    void* stage1_a = dev_a;
    void* stage1_b = dev_b;
    void* stage1_tmp = dev_tmp;
    void* stage1_args[] = {&stage1_a, &stage1_b, &stage1_tmp};
    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, stage1_args, 0, stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stage1 launch on non-default stream failed\n");
        return 1;
    }

    // No explicit stream/event synchronization here. Legacy null-stream semantics require that
    // this launch wait for prior work submitted to non-default streams.
    void* stage2_tmp = dev_tmp;
    void* stage2_b = dev_b;
    void* stage2_out = dev_out;
    void* stage2_args[] = {&stage2_tmp, &stage2_b, &stage2_out};
    if (cudaLaunchKernel(&kernel, grid_dim, block_dim, stage2_args, 0, nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: stage2 launch on null stream failed\n");
        return 1;
    }

    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaDeviceSynchronize failed\n");
        return 1;
    }

    if (cudaMemcpy(host_out.data(), dev_out, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMemcpy device->host failed\n");
        return 1;
    }

    for (std::size_t i = 0; i < kElementCount; ++i) {
        const float expected = host_a[i] + host_b[i] + host_b[i];
        if (!nearly_equal(host_out[i], expected)) {
            std::fprintf(stderr,
                         "FAIL: mismatch at %zu (got=%f expected=%f)\n",
                         i,
                         static_cast<double>(host_out[i]),
                         static_cast<double>(expected));
            return 1;
        }
    }

    void* spin_output_a = nullptr;
    void* spin_output_b = nullptr;
    void* marker_output_a = nullptr;
    void* marker_output_b = nullptr;
    if (cudaMalloc(&spin_output_a, sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&spin_output_b, sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&marker_output_a, sizeof(unsigned int)) != cudaSuccess ||
        cudaMalloc(&marker_output_b, sizeof(unsigned int)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: ordering-probe cudaMalloc failed\n");
        return 1;
    }

    static const cumetalKernelArgInfo_t kProbeArgInfo[] = {
        {CUMETAL_ARG_BUFFER, 0},
        {CUMETAL_ARG_BYTES, sizeof(unsigned int)},
    };
    const cumetalKernel_t spin_kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "spin_store",
        .arg_count = 2,
        .arg_info = kProbeArgInfo,
    };
    const cumetalKernel_t marker_kernel{
        .metallib_path = metallib_path.c_str(),
        .kernel_name = "marker_store",
        .arg_count = 2,
        .arg_info = kProbeArgInfo,
    };
    auto launch_spin = [&](void* output, cudaStream_t target) {
        void* output_arg = output;
        unsigned int iterations = kSpinIterations;
        void* launch_args[] = {&output_arg, &iterations};
        return cudaLaunchKernel(
            &spin_kernel, dim3(1, 1, 1), dim3(1, 1, 1), launch_args, 0, target);
    };
    auto launch_marker = [&](void* output, unsigned int value, cudaStream_t target) {
        void* output_arg = output;
        void* launch_args[] = {&output_arg, &value};
        return cudaLaunchKernel(
            &marker_kernel, dim3(1, 1, 1), dim3(1, 1, 1), launch_args, 0, target);
    };

    // Blocking stream -> legacy stream, with no shared buffers between the commands.
    if (launch_spin(spin_output_a, stream) != cudaSuccess ||
        launch_marker(marker_output_a, 0xA1u, nullptr) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess ||
        cudaStreamQuery(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: legacy stream did not wait for prior blocking-stream work\n");
        return 1;
    }

    // Legacy stream -> blocking stream, again using disjoint buffers.
    if (launch_spin(spin_output_b, nullptr) != cudaSuccess ||
        launch_marker(marker_output_b, 0xB2u, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        cudaStreamQuery(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: blocking stream did not wait for prior legacy-stream work\n");
        return 1;
    }

    cudaStream_t nonblocking_stream = nullptr;
    if (cudaStreamCreateWithFlags(&nonblocking_stream, cudaStreamNonBlocking) != cudaSuccess ||
        nonblocking_stream == nullptr) {
        std::fprintf(stderr, "FAIL: cudaStreamCreateWithFlags(nonblocking) failed\n");
        return 1;
    }

    // A legacy command must not wait for unrelated non-blocking-stream work.
    if (launch_spin(spin_output_a, nonblocking_stream) != cudaSuccess ||
        launch_marker(marker_output_b, 0xC3u, nullptr) != cudaSuccess ||
        cudaStreamSynchronize(nullptr) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: nonblocking -> legacy probe launch failed\n");
        return 1;
    }
    if (cudaStreamQuery(nonblocking_stream) != cudaErrorNotReady) {
        std::fprintf(stderr, "FAIL: legacy stream incorrectly waited for nonblocking stream\n");
        return 1;
    }
    if (cudaStreamSynchronize(nonblocking_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: nonblocking stream cleanup sync failed\n");
        return 1;
    }

    // Hold the legacy command behind unrelated blocking-stream work. A
    // non-blocking marker submitted afterwards must bypass both commands.
    if (launch_spin(spin_output_b, stream) != cudaSuccess ||
        launch_marker(marker_output_b, 0xD4u, nullptr) != cudaSuccess ||
        launch_marker(marker_output_a, 0xE5u, nonblocking_stream) != cudaSuccess ||
        cudaStreamSynchronize(nonblocking_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: legacy -> nonblocking probe launch failed\n");
        return 1;
    }
    if (cudaStreamQuery(nullptr) != cudaErrorNotReady) {
        std::fprintf(stderr, "FAIL: nonblocking stream incorrectly waited for legacy stream\n");
        return 1;
    }
    if (cudaStreamSynchronize(nullptr) != cudaSuccess ||
        cudaStreamDestroy(nonblocking_stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: ordering-probe cleanup failed\n");
        return 1;
    }

    if (cudaStreamDestroy(stream) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaStreamDestroy failed\n");
        return 1;
    }

    if (cudaFree(dev_a) != cudaSuccess || cudaFree(dev_b) != cudaSuccess ||
        cudaFree(dev_tmp) != cudaSuccess || cudaFree(dev_out) != cudaSuccess ||
        cudaFree(spin_output_a) != cudaSuccess || cudaFree(spin_output_b) != cudaSuccess ||
        cudaFree(marker_output_a) != cudaSuccess || cudaFree(marker_output_b) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaFree failed\n");
        return 1;
    }

    std::printf("PASS: runtime legacy stream orders blocking streams and excludes nonblocking streams\n");
    return 0;
}
