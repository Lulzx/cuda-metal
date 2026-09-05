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
    if (!expect(cubin.size() > 4 && std::memcmp(cubin.data(), "MTLB", 4) == 0,
                "compiled output is a Metal library")) {
        return 1;
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

    // Asking for a virtual architecture means asking for PTX, which CuMetal
    // cannot emit from CUDA source. Fail at compile time, with a log that says
    // why, rather than handing back bytes the caller will mis-handle.
    {
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, kKernelSource, "ptx_module", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the PTX case")) {
            return 1;
        }
        const char* const options[] = {"--gpu-architecture=compute_75"};
        const nvrtcResult compiled = nvrtcCompileProgram(program, 1, options);
        const std::string log = program_log(program);
        nvrtcDestroyProgram(&program);
        if (!expect(compiled == NVRTC_ERROR_INVALID_OPTION,
                    "compute_XX is rejected as an invalid option")) {
            return 1;
        }
        if (!expect(log.find("cannot emit PTX") != std::string::npos,
                    "the PTX rejection explains itself in the log")) {
            return 1;
        }
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

    // PTX retrieval is unavailable, and says so rather than returning success
    // with empty output.
    {
        nvrtcProgram program = nullptr;
        if (!expect(nvrtcCreateProgram(&program, kKernelSource, "ptx_query", 0, nullptr,
                                       nullptr) == NVRTC_SUCCESS,
                    "nvrtcCreateProgram for the PTX query")) {
            return 1;
        }
        std::size_t size = 0;
        const nvrtcResult result = nvrtcGetPTXSize(program, &size);
        nvrtcDestroyProgram(&program);
        if (!expect(result == NVRTC_ERROR_INVALID_PROGRAM, "nvrtcGetPTXSize reports failure")) {
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
