// End-to-end cover for the NVRTC shim: CUDA source in, a loadable module out.
//
// The point of the shim is that a caller written against NVRTC never learns it
// is talking to a Metal toolchain, so this test follows the sequence such a
// caller uses -- create, compile, read the "CUBIN", hand it to
// cuModuleLoadDataEx -- and checks the failure paths report themselves through
// the program log rather than silently producing something unloadable.
#include "cuda.h"
#include "nvPTXCompiler.h"
#include "nvrtc.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

const char* const kKernelSource = R"(
extern "C" __global__ void scale_kernel(float* out, const float* in, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = in[index] * 2.0f;
    }
}
)";

// The options Warp's wp_cuda_compile_program passes for a release build, minus
// the include path it points at its own headers.
const std::vector<const char*> kWarpLikeOptions = {
    "--gpu-architecture=sm_80",
    "--std=c++17",
    "--define-macro=NDEBUG",
    "--undefine-macro=WP_VERIFY_FP",
    "--fmad=false",
    "--device-as-default-execution-space",
    "--extra-device-vectorization",
    "--restrict",
    "--diag-suppress=177,550",
};

std::string program_log(nvrtcProgram program) {
    std::size_t size = 0;
    if (nvrtcGetProgramLogSize(program, &size) != NVRTC_SUCCESS || size == 0) return {};
    std::vector<char> log(size);
    if (nvrtcGetProgramLog(program, log.data()) != NVRTC_SUCCESS) return {};
    return std::string(log.data());
}

bool compile_ok(const char* source,
                const char* name,
                const std::vector<const char*>& options,
                std::vector<char>* cubin) {
    nvrtcProgram program = nullptr;
    if (nvrtcCreateProgram(&program, source, name, 0, nullptr, nullptr) != NVRTC_SUCCESS) {
        std::fprintf(stderr, "FAIL: nvrtcCreateProgram(%s)\n", name);
        return false;
    }
    const nvrtcResult compiled =
        nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
    if (compiled != NVRTC_SUCCESS) {
        std::fprintf(stderr, "FAIL: nvrtcCompileProgram(%s): %s\n%s\n", name,
                     nvrtcGetErrorString(compiled), program_log(program).c_str());
        nvrtcDestroyProgram(&program);
        return false;
    }

    std::size_t size = 0;
    const bool sized = nvrtcGetCUBINSize(program, &size) == NVRTC_SUCCESS && size > 4;
    if (sized) {
        cubin->resize(size);
        if (nvrtcGetCUBIN(program, cubin->data()) != NVRTC_SUCCESS) {
            std::fprintf(stderr, "FAIL: nvrtcGetCUBIN(%s)\n", name);
            nvrtcDestroyProgram(&program);
            return false;
        }
    } else {
        std::fprintf(stderr, "FAIL: nvrtcGetCUBINSize(%s) returned %zu\n", name, size);
    }
    nvrtcDestroyProgram(&program);
    return sized;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-cumetalc>\n", argv[0]);
        return 64;
    }
    if (!std::filesystem::exists(argv[1])) {
        std::fprintf(stderr, "SKIP: cumetalc not found at %s\n", argv[1]);
        return 77;
    }
    // Pin the shim to the compiler from this build tree rather than whatever an
    // installed prefix or PATH happens to offer.
    ::setenv("CUMETAL_NVRTC_COMPILER", argv[1], 1);

    int major = 0;
    int minor = 0;
    if (!expect(nvrtcVersion(&major, &minor) == NVRTC_SUCCESS && major == CUDA_VERSION / 1000 &&
                    minor == (CUDA_VERSION % 1000) / 10,
                "nvrtcVersion matches the toolkit version")) {
        return 1;
    }

    int arch_count = 0;
    int archs[4] = {0, 0, 0, 0};
    if (!expect(nvrtcGetNumSupportedArchs(&arch_count) == NVRTC_SUCCESS && arch_count == 1,
                "one supported architecture")) {
        return 1;
    }
    if (!expect(nvrtcGetSupportedArchs(archs) == NVRTC_SUCCESS && archs[0] == 80,
                "supported architecture is sm_80")) {
        return 1;
    }

    std::vector<char> cubin;
    if (!compile_ok(kKernelSource, "scale_module", kWarpLikeOptions, &cubin)) {
        return 1;
    }
    // The "CUBIN" is a CuMetal module image: an 8-byte magic, two 64-bit
    // lengths, then the metallib and the kernel ABI sidecar cumetalc wrote
    // beside it. The sidecar rides inside the image because nothing in the
    // NVRTC -> cache -> cuModuleLoadData round trip carries a second file.
    {
        constexpr std::size_t kHeader = 8 + 2 * sizeof(std::uint64_t);
        if (!expect(cubin.size() > kHeader && std::memcmp(cubin.data(), "CUMTLMD1", 8) == 0,
                    "compiled output is a CuMetal module image")) {
            return 1;
        }
        std::uint64_t sizes[2] = {0, 0};
        std::memcpy(sizes, cubin.data() + 8, sizeof(sizes));
        // Plus the trailing NUL that NVRTC's CUBIN size counts.
        if (!expect(sizes[0] > 4 && sizes[1] > 0 &&
                        kHeader + sizes[0] + sizes[1] + 1 == cubin.size(),
                    "module image lengths cover the buffer up to its NUL")) {
            return 1;
        }
        if (!expect(std::memcmp(cubin.data() + kHeader, "MTLB", 4) == 0,
                    "module image carries a Metal library")) {
            return 1;
        }
        const std::string sidecar(cubin.data() + kHeader + sizes[0], static_cast<std::size_t>(sizes[1]));
        if (!expect(sidecar.rfind("CUMETAL_ABI_V", 0) == 0 &&
                        sidecar.find("kernel scale_kernel") != std::string::npos,
                    "module image carries the kernel ABI sidecar")) {
            std::fprintf(stderr, "%s\n", sidecar.c_str());
            return 1;
        }
    }

    // An in-memory header must reach the compile the way a quoted include
    // expects to find it.
    {
        const char* const header = "__device__ float twice(float x) { return x + x; }\n";
        const char* const include_name = "helper.h";
        const char* const source = R"(
#include "helper.h"
extern "C" __global__ void header_kernel(float* out, const float* in, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        out[index] = twice(in[index]);
    }
}
)";
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, source, "header_module", 1, &header,
                                       &include_name) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram accepts an in-memory header")) {
            return 1;
        }
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(kWarpLikeOptions.size()), kWarpLikeOptions.data());
        if (!expect(compiled == NVRTC_SUCCESS, "in-memory header resolves at compile time")) {
            std::fprintf(stderr, "%s\n", program_log(program).c_str());
            nvrtcDestroyProgram(&program);
            return 1;
        }
        nvrtcDestroyProgram(&program);
    }

    // NVRTC treats every unannotated function as device code
    // (--device-as-default-execution-space). A helper with no __device__ on it
    // must compile under that option and be rejected without it, which is the
    // difference between Warp's tile headers building and not.
    {
        const char* const source = R"(
inline float twice_unannotated(float x) { return x + x; }
struct Scale { float factor; Scale(float f) : factor(f) {} float apply(float x) const { return x * factor; } };
extern "C" __global__ void unannotated_kernel(float* out, const float* in, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        Scale scale(3.0f);
        out[index] = scale.apply(twice_unannotated(in[index]));
    }
}
)";
        std::vector<char> image;
        if (!compile_ok(source, "unannotated_module", kWarpLikeOptions, &image)) {
            return 1;
        }
        std::vector<const char*> host_default;
        for (const char* option : kWarpLikeOptions) {
            if (std::strcmp(option, "--device-as-default-execution-space") != 0) {
                host_default.push_back(option);
            }
        }
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, source, "host_default_module", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the host-default case")) {
            return 1;
        }
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(host_default.size()), host_default.data());
        const std::string log = program_log(program);
        nvrtcDestroyProgram(&program);
        if (!expect(compiled == NVRTC_ERROR_COMPILATION &&
                        log.find("__host__") != std::string::npos,
                    "without --device-as-default-execution-space an unannotated helper is "
                    "host-only")) {
            std::fprintf(stderr, "%s\n", log.c_str());
            return 1;
        }
    }

    // A virtual architecture asks for PTX. It must be real NVPTX output for the
    // kernel, loadable through cuModuleLoadData, and computing the right answer.
    std::string ptx_text;
    {
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, kKernelSource, "ptx_module", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the PTX case")) {
            return 1;
        }
        const char* const options[] = {"--gpu-architecture=compute_75"};
        const nvrtcResult compiled = nvrtcCompileProgram(program, 1, options);
        if (!expect(compiled == NVRTC_SUCCESS, "compute_XX compiles to PTX")) {
            std::fprintf(stderr, "%s\n", program_log(program).c_str());
            nvrtcDestroyProgram(&program);
            return 1;
        }
        std::size_t size = 0;
        if (!expect(nvrtcGetPTXSize(program, &size) == NVRTC_SUCCESS && size > 1,
                    "nvrtcGetPTXSize reports the PTX")) {
            nvrtcDestroyProgram(&program);
            return 1;
        }
        std::vector<char> ptx(size);
        if (!expect(nvrtcGetPTX(program, ptx.data()) == NVRTC_SUCCESS && ptx.back() == '\0',
                    "nvrtcGetPTX returns NUL-terminated text")) {
            nvrtcDestroyProgram(&program);
            return 1;
        }
        nvrtcDestroyProgram(&program);
        ptx_text = ptx.data();
        if (!expect(ptx_text.find(".version") != std::string::npos &&
                        ptx_text.find(".entry scale_kernel") != std::string::npos,
                    "the PTX defines the kernel entry")) {
            std::fprintf(stderr, "%s\n", ptx_text.c_str());
            return 1;
        }
    }

    // A real-architecture compile produces a Metal library, so there is no PTX
    // to read; that is an error rather than empty output.
    {
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, kKernelSource, "ptx_query", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the PTX query")) {
            return 1;
        }
        std::size_t size = 0;
        const nvrtcResult before = nvrtcGetPTXSize(program, &size);
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(kWarpLikeOptions.size()), kWarpLikeOptions.data());
        const nvrtcResult after = nvrtcGetPTXSize(program, &size);
        nvrtcDestroyProgram(&program);
        if (!expect(before == NVRTC_ERROR_INVALID_PROGRAM && compiled == NVRTC_SUCCESS &&
                        after == NVRTC_ERROR_INVALID_PROGRAM,
                    "nvrtcGetPTXSize fails before compilation and for sm_XX")) {
            return 1;
        }
    }

    // Templated kernels are found through name expressions, which must map to
    // the symbols the device compiler actually emitted.
    const char* const kTemplateSource = R"(
namespace ns {
template <typename T, int Factor>
__global__ void scale_t(T* out, const T* in, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) out[index] = in[index] * T(Factor);
}
}
__device__ float device_bias = 0.5f;
extern "C" __global__ void plain_kernel(float* out) { out[0] = device_bias; }
)";
    const char* const kExpressions[] = {"ns::scale_t<float, 3>", "ns::scale_t<float, 5>",
                                        "plain_kernel", "&device_bias"};
    std::string lowered[4];
    std::vector<char> template_image;
    {
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, kTemplateSource, "template_module", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for name expressions")) {
            return 1;
        }
        for (const char* expression : kExpressions) {
            if (!expect(nvrtcAddNameExpression(program, expression) == NVRTC_SUCCESS,
                        "nvrtcAddNameExpression")) {
                return 1;
            }
        }
        const char* name = nullptr;
        if (!expect(nvrtcGetLoweredName(program, kExpressions[0], &name) ==
                        NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
                    "lowered names wait for compilation")) {
            return 1;
        }
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(kWarpLikeOptions.size()), kWarpLikeOptions.data());
        if (!expect(compiled == NVRTC_SUCCESS, "templated program compiles")) {
            std::fprintf(stderr, "%s\n", program_log(program).c_str());
            nvrtcDestroyProgram(&program);
            return 1;
        }
        for (int i = 0; i < 4; ++i) {
            if (!expect(nvrtcGetLoweredName(program, kExpressions[i], &name) == NVRTC_SUCCESS &&
                            name != nullptr,
                        "every registered expression has a lowered name")) {
                nvrtcDestroyProgram(&program);
                return 1;
            }
            lowered[i] = name;
        }
        if (!expect(nvrtcGetLoweredName(program, "ns::scale_t<float, 7>", &name) ==
                        NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
                    "an unregistered expression has no lowered name")) {
            return 1;
        }
        if (!expect(nvrtcAddNameExpression(program, "plain_kernel") ==
                        NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
                    "name expressions close at compilation")) {
            return 1;
        }
        std::size_t size = 0;
        if (nvrtcGetCUBINSize(program, &size) == NVRTC_SUCCESS && size > 0) {
            template_image.resize(size);
            nvrtcGetCUBIN(program, template_image.data());
        }
        nvrtcDestroyProgram(&program);
        if (!expect(lowered[0] == "_ZN2ns7scale_tIfLi3EEEvPT_PKS1_i" &&
                        lowered[1] == "_ZN2ns7scale_tIfLi5EEEvPT_PKS1_i" &&
                        lowered[2] == "plain_kernel" && lowered[3] == "device_bias",
                    "lowered names are the Itanium-mangled symbols")) {
            for (const std::string& n : lowered) std::fprintf(stderr, "  %s\n", n.c_str());
            return 1;
        }
        if (!expect(!template_image.empty(), "templated program produced a module")) return 1;
    }

    // A name expression that does not name anything is a compile error.
    {
        nvrtcProgram program = nullptr;
        nvrtcCreateProgram(&program, kKernelSource, "bad_name", 0, nullptr, nullptr);
        nvrtcAddNameExpression(program, "no_such_kernel<int>");
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(kWarpLikeOptions.size()), kWarpLikeOptions.data());
        nvrtcDestroyProgram(&program);
        if (!expect(compiled == NVRTC_ERROR_COMPILATION, "an invalid name expression fails")) {
            return 1;
        }
    }

    // Both outputs must run: the PTX through the driver's PTX path, and the
    // templated module through the mangled names NVRTC reported.
    {
        CUdevice device = 0;
        CUcontext context = nullptr;
        if (!expect(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&device, 0) == CUDA_SUCCESS &&
                        cuCtxCreate(&context, 0, device) == CUDA_SUCCESS,
                    "driver context")) {
            return 1;
        }
        constexpr int kCount = 64;
        CUdeviceptr in = 0, out = 0;
        if (!expect(cuMemAlloc(&in, kCount * sizeof(float)) == CUDA_SUCCESS &&
                        cuMemAlloc(&out, kCount * sizeof(float)) == CUDA_SUCCESS,
                    "cuMemAlloc")) {
            return 1;
        }
        std::vector<float> host(kCount);
        for (int i = 0; i < kCount; ++i) host[i] = static_cast<float>(i) - 10.0f;
        cuMemcpyHtoD(in, host.data(), kCount * sizeof(float));

        const auto run = [&](CUmodule module, const char* kernel_name, float factor) -> bool {
            CUfunction function = nullptr;
            if (!expect(cuModuleGetFunction(&function, module, kernel_name) == CUDA_SUCCESS,
                        "cuModuleGetFunction")) {
                std::fprintf(stderr, "  kernel %s\n", kernel_name);
                return false;
            }
            int count = kCount;
            void* args[] = {&out, &in, &count};
            cuMemsetD32(out, 0, kCount);
            if (!expect(cuLaunchKernel(function, 1, 1, 1, kCount, 1, 1, 0, nullptr, args,
                                       nullptr) == CUDA_SUCCESS &&
                            cuCtxSynchronize() == CUDA_SUCCESS,
                        "launch")) {
                return false;
            }
            std::vector<float> result(kCount);
            cuMemcpyDtoH(result.data(), out, kCount * sizeof(float));
            for (int i = 0; i < kCount; ++i) {
                if (result[i] != host[i] * factor) {
                    std::fprintf(stderr, "FAIL: %s[%d] = %f, want %f\n", kernel_name, i,
                                 result[i], host[i] * factor);
                    return false;
                }
            }
            return true;
        };

        CUmodule ptx_module = nullptr;
        if (!expect(cuModuleLoadData(&ptx_module, ptx_text.c_str()) == CUDA_SUCCESS,
                    "cuModuleLoadData accepts NVRTC's PTX") ||
            !run(ptx_module, "scale_kernel", 2.0f)) {
            return 1;
        }
        CUmodule template_module = nullptr;
        if (!expect(cuModuleLoadData(&template_module, template_image.data()) == CUDA_SUCCESS,
                    "cuModuleLoadData accepts the templated module") ||
            !run(template_module, lowered[0].c_str(), 3.0f) ||
            !run(template_module, lowered[1].c_str(), 5.0f)) {
            return 1;
        }
        // CuPy keeps nvrtcGetCUBINSize() - 1 bytes because NVRTC's size counts
        // a trailing NUL. Before the shim appended one, that cut the sidecar's
        // last byte and every CuPy launch failed with CUDA_ERROR_INVALID_VALUE.
        if (!expect(!cubin.empty() && cubin.back() == '\0', "the CUBIN ends in a NUL")) return 1;
        std::vector<char> stripped(cubin.begin(), cubin.end() - 1);
        CUmodule stripped_module = nullptr;
        if (!expect(cuModuleLoadData(&stripped_module, stripped.data()) == CUDA_SUCCESS,
                    "cuModuleLoadData accepts the CUBIN without its NUL") ||
            !run(stripped_module, "scale_kernel", 2.0f)) {
            return 1;
        }
        cuModuleUnload(stripped_module);
        CUdeviceptr bias = 0;
        std::size_t bias_size = 0;
        float bias_value = 0.0f;
        if (!expect(cuModuleGetGlobal(&bias, &bias_size, template_module, lowered[3].c_str()) ==
                            CUDA_SUCCESS &&
                        bias_size == sizeof(float) &&
                        cuMemcpyDtoH(&bias_value, bias, sizeof(float)) == CUDA_SUCCESS &&
                        bias_value == 0.5f,
                    "the lowered variable name resolves through cuModuleGetGlobal")) {
            return 1;
        }
        cuModuleUnload(ptx_module);
        cuModuleUnload(template_module);
        cuMemFree(in);
        cuMemFree(out);
        cuCtxDestroy(context);
    }

    // A compile error must surface as NVRTC_ERROR_COMPILATION with the
    // compiler's own diagnostics in the log.
    {
        nvrtcProgram program = nullptr;
        const char* const broken = "extern \"C\" __global__ void k() { this is not C++; }\n";
        if (!expect(nvrtcCreateProgram(&program, broken, "broken_module", 0, nullptr, nullptr) ==
                        NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the failure case")) {
            return 1;
        }
        const nvrtcResult compiled = nvrtcCompileProgram(
            program, static_cast<int>(kWarpLikeOptions.size()), kWarpLikeOptions.data());
        const std::string log = program_log(program);
        std::size_t cubin_size = 0;
        const nvrtcResult sized = nvrtcGetCUBINSize(program, &cubin_size);
        nvrtcDestroyProgram(&program);
        if (!expect(compiled == NVRTC_ERROR_COMPILATION, "invalid source fails to compile")) {
            return 1;
        }
        if (!expect(!log.empty(), "a failed compile leaves diagnostics in the log")) return 1;
        if (!expect(sized == NVRTC_ERROR_INVALID_PROGRAM,
                    "a failed compile has no CUBIN to read")) {
            return 1;
        }
    }

    // nvPTXCompiler passes PTX through: the driver is what compiles it.
    {
        const std::string ptx = ".version 7.0\n.target sm_80\n";
        nvPTXCompilerHandle compiler = nullptr;
        if (!expect(nvPTXCompilerCreate(&compiler, ptx.size(), ptx.data()) == NVPTXCOMPILE_SUCCESS,
                    "nvPTXCompilerCreate")) {
            return 1;
        }
        const char* const options[] = {"--gpu-name=sm_80"};
        if (!expect(nvPTXCompilerCompile(compiler, 1, options) == NVPTXCOMPILE_SUCCESS,
                    "nvPTXCompilerCompile")) {
            return 1;
        }
        std::size_t size = 0;
        if (!expect(nvPTXCompilerGetCompiledProgramSize(compiler, &size) == NVPTXCOMPILE_SUCCESS &&
                        size == ptx.size() + 1,
                    "compiled program is the PTX plus its terminator")) {
            return 1;
        }
        std::vector<char> image(size);
        if (!expect(nvPTXCompilerGetCompiledProgram(compiler, image.data()) ==
                            NVPTXCOMPILE_SUCCESS &&
                        std::string(image.data()) == ptx,
                    "compiled program round-trips the PTX")) {
            return 1;
        }
        if (!expect(nvPTXCompilerDestroy(&compiler) == NVPTXCOMPILE_SUCCESS && compiler == nullptr,
                    "nvPTXCompilerDestroy clears the handle")) {
            return 1;
        }
    }

    // The whole point: the driver takes what NVRTC produced.
    if (!expect(cuInit(0) == CUDA_SUCCESS, "cuInit")) return 1;
    CUdevice device = 0;
    if (!expect(cuDeviceGet(&device, 0) == CUDA_SUCCESS, "cuDeviceGet")) return 1;
    CUcontext context = nullptr;
    if (!expect(cuCtxCreate(&context, 0, device) == CUDA_SUCCESS, "cuCtxCreate")) return 1;

    CUmodule module = nullptr;
    if (!expect(cuModuleLoadDataEx(&module, cubin.data(), 0, nullptr, nullptr) == CUDA_SUCCESS,
                "cuModuleLoadDataEx accepts the NVRTC output")) {
        cuCtxDestroy(context);
        return 1;
    }
    CUfunction function = nullptr;
    if (!expect(cuModuleGetFunction(&function, module, "scale_kernel") == CUDA_SUCCESS &&
                    function != nullptr,
                "the compiled kernel is present in the module")) {
        cuModuleUnload(module);
        cuCtxDestroy(context);
        return 1;
    }

    cuModuleUnload(module);
    cuCtxDestroy(context);

    std::printf("PASS: nvrtc compile and module load\n");
    return 0;
}
