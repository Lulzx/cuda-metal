// #222 (Cuda-On-Metal-Lab overlay): one FP64 kernel in a module must not send
// an FP32 kernel from the same module down the FP64 path. On the registration
// path each kernel is lowered from its own slice of the module, so the FP32
// kernel's launch provenance names plain generic lowering, while the FP64
// kernel's names its FP64 mode. Provenance is read from this process's own
// stderr under CUMETAL_TRACE_GPU.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <string>
#include <unistd.h>

__global__ void softmax_scale_f32(float* x, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 0.5f + 1.0f;
}

__global__ void unused_neighbor_f64(double* x, unsigned n) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] * 3.0 + 0.25;
}

static int fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

// Waits for the completion handler's provenance line for `kernel` and returns it.
static std::string provenance_line(FILE* log, const char* kernel) {
    for (int attempt = 0; attempt < 200; ++attempt) {
        std::fflush(stderr);
        std::rewind(log);
        char line[2048];
        while (std::fgets(line, sizeof line, log) != nullptr) {
            if (std::strstr(line, "event=kernel_launch") != nullptr &&
                std::strstr(line, kernel) != nullptr)
                return line;
        }
        usleep(25000);
    }
    return {};
}

int main() {
    setenv("CUMETAL_TRACE_GPU", "1", 1);
    FILE* log = std::tmpfile();
    if (log == nullptr) return fail("tmpfile");
    const int saved_stderr = dup(STDERR_FILENO);
    std::fflush(stderr);
    dup2(fileno(log), STDERR_FILENO);

    constexpr unsigned n = 256;
    float* x = nullptr;
    double* d = nullptr;
    cudaMalloc(&x, n * sizeof(float));
    cudaMalloc(&d, n * sizeof(double));
    cudaMemset(x, 0, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(double));
    softmax_scale_f32<<<1, n>>>(x, n);
    const cudaError_t f32_status = cudaDeviceSynchronize();
    const std::string f32_line = provenance_line(log, "softmax_scale_f32");
    unused_neighbor_f64<<<1, n>>>(d, n);
    const cudaError_t f64_status = cudaDeviceSynchronize();
    const std::string f64_line = provenance_line(log, "unused_neighbor_f64");

    std::fflush(stderr);
    dup2(saved_stderr, STDERR_FILENO);
    // Hand the captured diagnostics on, so a harness still sees them.
    std::rewind(log);
    char chunk[4096];
    for (size_t got; (got = std::fread(chunk, 1, sizeof chunk, log)) > 0;)
        std::fwrite(chunk, 1, got, stderr);
    if (f32_status != cudaSuccess || f64_status != cudaSuccess) return fail("kernel launch");
    if (f32_line.empty() || f64_line.empty()) return fail("no provenance line for a launch");
    if (f32_line.find("provenance=generic_ptx_lowering ") == std::string::npos) {
        std::printf("FP32 kernel provenance: %s", f32_line.c_str());
        return fail("the FP32 kernel took its FP64 neighbour's lowering");
    }
    if (f64_line.find("provenance=generic_ptx_lowering_fp64_") == std::string::npos) {
        std::printf("FP64 kernel provenance: %s", f64_line.c_str());
        return fail("the FP64 kernel did not report its FP64 lowering");
    }
    float host[n];
    cudaMemcpy(host, x, sizeof host, cudaMemcpyDeviceToHost);
    for (unsigned i = 0; i < n; ++i)
        if (host[i] != 1.0f) return fail("FP32 kernel result");
    std::printf("PASS: FP64 handling is decided per kernel\n");
    return 0;
}
