// #215: every library shim call that does work reports a CUMETAL_PROVENANCE
// line under CUMETAL_TRACE_GPU naming the library, routine and the backend
// that actually ran: MPS, a CuMetal library kernel, or the host fallback.
// Calls that do no work (refused, or tracing off) report nothing.

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusolverDn.h>
#include <cusparse.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// Runs `body` with stderr redirected to a temporary file and returns what it
// printed.
template <typename Body>
std::string capture_stderr(Body body) {
    std::fflush(stderr);
    char path[] = "/tmp/cumetal_library_provenance_XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return "<mkstemp failed>";
    const int saved = dup(STDERR_FILENO);
    dup2(fd, STDERR_FILENO);
    body();
    std::fflush(stderr);
    dup2(saved, STDERR_FILENO);
    close(saved);
    std::string text;
    lseek(fd, 0, SEEK_SET);
    char buffer[4096];
    ssize_t n = 0;
    while ((n = read(fd, buffer, sizeof(buffer))) > 0) text.append(buffer, static_cast<size_t>(n));
    close(fd);
    unlink(path);
    return text;
}

std::vector<std::string> library_lines(const std::string& text) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start < text.size()) {
        size_t end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        const std::string line = text.substr(start, end - start);
        if (line.find("CUMETAL_PROVENANCE event=library_call ") != std::string::npos) {
            lines.push_back(line);
        }
        start = end + 1;
    }
    return lines;
}

int failures = 0;

// Exactly one library line, for `routine`, with the expected backend field.
void expect_line(const char* what, const std::string& text, const char* library,
                 const char* routine, const char* backend) {
    const std::vector<std::string> lines = library_lines(text);
    const std::string want = std::string("library=") + library + " routine=" + routine +
                             " backend=" + backend + " ";
    if (lines.size() != 1 || lines[0].find(want) == std::string::npos) {
        std::fprintf(stderr, "FAIL: %s: want one line with '%s', got %zu:\n%s\n", what,
                     want.c_str(), lines.size(), text.c_str());
        ++failures;
        return;
    }
    const bool host = std::strstr(backend, "host_fallback") != nullptr;
    const std::string host_field = host ? "host_work=true" : "host_work=false";
    if (lines[0].find(host_field) == std::string::npos) {
        std::fprintf(stderr, "FAIL: %s: expected %s in '%s'\n", what, host_field.c_str(),
                     lines[0].c_str());
        ++failures;
    }
}

void expect_none(const char* what, const std::string& text) {
    if (!library_lines(text).empty()) {
        std::fprintf(stderr, "FAIL: %s: expected no library provenance, got:\n%s\n", what,
                     text.c_str());
        ++failures;
    }
}

float* device_floats(size_t count, float value) {
    float* p = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "FAIL: cudaMalloc\n");
        std::exit(1);
    }
    std::vector<float> host(count, value);
    cudaMemcpy(p, host.data(), count * sizeof(float), cudaMemcpyHostToDevice);
    return p;
}

}  // namespace

int main() {
    setenv("CUMETAL_TRACE_GPU", "1", 1);
    // A tiny SpMV stays on the CPU in auto mode; force the Metal kernel so the
    // cuSPARSE case exercises the kernel backend.
    setenv("CUMETAL_SPARSE_METAL", "1", 1);
    if (cudaFree(nullptr) != cudaSuccess) return 1;

    // cuBLAS: SGEMM runs on MPS.
    {
        cublasHandle_t handle = nullptr;
        cublasCreate(&handle);
        constexpr int n = 64;
        float* a = device_floats(n * n, 1.0f);
        float* b = device_floats(n * n, 2.0f);
        float* c = device_floats(n * n, 0.0f);
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
        const std::string text = capture_stderr([&] {
            st = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, n, b, n,
                             &beta, c, n);
            cudaDeviceSynchronize();
        });
        if (st != CUBLAS_STATUS_SUCCESS) ++failures;
        expect_line("cublasSgemm", text, "cuBLAS", "cublasSgemm", "mps");

        // Refused call: no work, no line.
        const std::string refused = capture_stderr([&] {
            cublasSgemm(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, a, n, b, n, &beta,
                        c, n);
        });
        expect_none("refused cublasSgemm", refused);
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cublasDestroy(handle);
    }

    // cuFFT: a large C2C runs on CuMetal's FFT kernels; a tiny one stays on
    // the CPU below the dispatch-cost threshold.
    {
        for (const int n : {8192, 16}) {
            cufftHandle plan = 0;
            cufftPlan1d(&plan, n, CUFFT_C2C, 1);
            cufftComplex* data = nullptr;
            cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(cufftComplex));
            cudaMemset(data, 0, n * sizeof(cufftComplex));
            cufftResult r = CUFFT_SUCCESS;
            const std::string text = capture_stderr([&] {
                r = cufftExecC2C(plan, data, data, CUFFT_FORWARD);
                cudaDeviceSynchronize();
            });
            if (r != CUFFT_SUCCESS) ++failures;
            expect_line(n == 8192 ? "cufftExecC2C large" : "cufftExecC2C small", text, "cuFFT",
                        "cufftExecC2C", n == 8192 ? "cumetal_kernel" : "host_fallback");
            cudaFree(data);
            cufftDestroy(plan);
        }
    }

    // cuRAND: the host API generators compute on the CPU.
    {
        curandGenerator_t gen = nullptr;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        float* out = device_floats(256, 0.0f);
        curandStatus_t st = CURAND_STATUS_SUCCESS;
        const std::string text = capture_stderr([&] {
            st = curandGenerateUniform(gen, out, 256);
            cudaDeviceSynchronize();
        });
        if (st != CURAND_STATUS_SUCCESS) ++failures;
        expect_line("curandGenerateUniform", text, "cuRAND", "curandGenerateUniform",
                    "host_fallback");
        cudaFree(out);
        curandDestroyGenerator(gen);
    }

    // cuSPARSE: CSR SpMV runs on a CuMetal library kernel (forced above).
    {
        cusparseHandle_t handle = nullptr;
        cusparseCreate(&handle);
        int* row = nullptr;
        int* col = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&row), 4 * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&col), 3 * sizeof(int));
        const int host_row[] = {0, 1, 2, 3};
        const int host_col[] = {0, 1, 2};
        cudaMemcpy(row, host_row, sizeof(host_row), cudaMemcpyHostToDevice);
        cudaMemcpy(col, host_col, sizeof(host_col), cudaMemcpyHostToDevice);
        float* values = device_floats(3, 2.0f);
        float* x = device_floats(3, 1.0f);
        float* y = device_floats(3, 0.0f);
        cusparseSpMatDescr_t mat = nullptr;
        cusparseDnVecDescr_t vx = nullptr, vy = nullptr;
        cusparseCreateCsr(&mat, 3, 3, 3, row, col, values, CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        cusparseCreateDnVec(&vx, 3, x, CUDA_R_32F);
        cusparseCreateDnVec(&vy, 3, y, CUDA_R_32F);
        const float alpha = 1.0f, beta = 0.0f;
        cusparseStatus_t st = CUSPARSE_STATUS_SUCCESS;
        const std::string text = capture_stderr([&] {
            st = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vx, &beta,
                              vy, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
            cudaDeviceSynchronize();
        });
        if (st != CUSPARSE_STATUS_SUCCESS) ++failures;
        expect_line("cusparseSpMV", text, "cuSPARSE", "cusparseSpMV", "cumetal_kernel");
        cusparseDestroySpMat(mat);
        cusparseDestroyDnVec(vx);
        cusparseDestroyDnVec(vy);
        cudaFree(row);
        cudaFree(col);
        cudaFree(values);
        cudaFree(x);
        cudaFree(y);
        cusparseDestroy(handle);
    }

    // cuSOLVER: LU factorization runs on the CPU (Accelerate LAPACK).
    {
        cusolverDnHandle_t handle = nullptr;
        cusolverDnCreate(&handle);
        float* a = device_floats(4, 0.0f);
        const float host_a[] = {2.0f, 0.0f, 0.0f, 3.0f};
        cudaMemcpy(a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
        float* work = device_floats(4, 0.0f);
        int* ipiv = nullptr;
        int* info = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&ipiv), 2 * sizeof(int));
        cudaMalloc(reinterpret_cast<void**>(&info), sizeof(int));
        cusolverStatus_t st = CUSOLVER_STATUS_SUCCESS;
        const std::string text = capture_stderr([&] {
            st = cusolverDnSgetrf(handle, 2, 2, a, 2, work, ipiv, info);
            cudaDeviceSynchronize();
        });
        if (st != CUSOLVER_STATUS_SUCCESS) ++failures;
        expect_line("cusolverDnSgetrf", text, "cuSOLVER", "cusolverDnSgetrf", "host_fallback");
        cudaFree(a);
        cudaFree(work);
        cudaFree(ipiv);
        cudaFree(info);
        cusolverDnDestroy(handle);
    }

    // cuDNN: activation runs on the CPU.
    {
        cudnnHandle_t handle = nullptr;
        cudnnCreate(&handle);
        cudnnTensorDescriptor_t desc = nullptr;
        cudnnCreateTensorDescriptor(&desc);
        cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, 1, 8);
        cudnnActivationDescriptor_t act = nullptr;
        cudnnCreateActivationDescriptor(&act);
        cudnnSetActivationDescriptor(act, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 0.0);
        float* x = device_floats(8, -1.0f);
        float* y = device_floats(8, 5.0f);
        const float alpha = 1.0f, beta = 0.0f;
        cudnnStatus_t st = CUDNN_STATUS_SUCCESS;
        const std::string text = capture_stderr([&] {
            st = cudnnActivationForward(handle, act, &alpha, desc, x, &beta, desc, y);
            cudaDeviceSynchronize();
        });
        if (st != CUDNN_STATUS_SUCCESS) ++failures;
        expect_line("cudnnActivationForward", text, "cuDNN", "cudnnActivationForward",
                    "host_fallback");
        cudaFree(x);
        cudaFree(y);
        cudnnDestroyActivationDescriptor(act);
        cudnnDestroyTensorDescriptor(desc);
        cudnnDestroy(handle);
    }

    // Tracing off: the same call that reported above prints nothing.
    {
        unsetenv("CUMETAL_TRACE_GPU");
        curandGenerator_t gen = nullptr;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        float* out = device_floats(64, 0.0f);
        const std::string text = capture_stderr([&] {
            curandGenerateUniform(gen, out, 64);
            cudaDeviceSynchronize();
        });
        expect_none("tracing disabled", text);
        cudaFree(out);
        curandDestroyGenerator(gen);
    }

    if (failures != 0) return 1;
    std::printf("PASS: library provenance\n");
    return 0;
}
