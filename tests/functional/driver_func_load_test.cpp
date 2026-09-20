// cuFuncLoad / cuFuncIsLoaded: explicit function preparation (issue #126).
//
// The point of these is a boundary: cuModuleGetFunction resolves lazily, so a
// successful lookup proves nothing about whether the kernel compiled. A caller
// wants to separate translation, Metal compilation and pipeline creation from
// execution, and report a compilation failure before interpreting any result.
#include "cuda.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace {

int fail(const char* what) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <path-to-metallib>\n", argv[0]);
        return 2;
    }
    const std::string metallib_path = argv[1];
    if (!std::filesystem::exists(metallib_path)) {
        std::fprintf(stderr, "SKIP: metallib not found at %s\n", metallib_path.c_str());
        return 77;
    }

    // Before cuInit, both entry points must say so rather than crash.
    CUfunctionLoadingState uninitialized_state = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&uninitialized_state, nullptr) != CUDA_ERROR_NOT_INITIALIZED)
        return fail("cuFuncIsLoaded before cuInit did not report NOT_INITIALIZED");
    if (cuFuncLoad(nullptr) != CUDA_ERROR_NOT_INITIALIZED)
        return fail("cuFuncLoad before cuInit did not report NOT_INITIALIZED");

    if (cuInit(0) != CUDA_SUCCESS) return fail("cuInit failed");
    CUdevice device = 0;
    if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) return fail("cuDeviceGet failed");
    CUcontext context = nullptr;
    if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS) return fail("cuCtxCreate failed");

    CUmodule module = nullptr;
    if (cuModuleLoad(&module, metallib_path.c_str()) != CUDA_SUCCESS)
        return fail("cuModuleLoad failed");
    CUfunction function = nullptr;
    if (cuModuleGetFunction(&function, module, "matrix_mul") != CUDA_SUCCESS)
        return fail("cuModuleGetFunction failed");

    // A resolved handle is not a prepared kernel: nothing has compiled yet.
    CUfunctionLoadingState state = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&state, function) != CUDA_SUCCESS)
        return fail("cuFuncIsLoaded on a resolved function failed");
    if (state != CU_FUNCTION_LOADING_STATE_UNLOADED)
        return fail("cuFuncIsLoaded reported LOADED before any preparation");

    // A query must not itself start compilation.
    CUfunctionLoadingState repeated = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&repeated, function) != CUDA_SUCCESS)
        return fail("repeated cuFuncIsLoaded failed");
    if (repeated != CU_FUNCTION_LOADING_STATE_UNLOADED)
        return fail("cuFuncIsLoaded prepared the function as a side effect");

    if (cuFuncLoad(function) != CUDA_SUCCESS) return fail("cuFuncLoad failed");
    if (cuFuncIsLoaded(&state, function) != CUDA_SUCCESS)
        return fail("cuFuncIsLoaded after cuFuncLoad failed");
    if (state != CU_FUNCTION_LOADING_STATE_LOADED)
        return fail("cuFuncIsLoaded did not report LOADED after cuFuncLoad");

    // Idempotent: loading an already-prepared function succeeds and does not
    // repeat the work.
    if (cuFuncLoad(function) != CUDA_SUCCESS) return fail("repeated cuFuncLoad failed");

    // Readiness must reflect preparation through any path, not only cuFuncLoad.
    // A second handle to the same artifact, prepared by an attribute query, is
    // reported loaded without anyone calling cuFuncLoad on it.
    CUmodule reloaded = nullptr;
    if (cuModuleLoad(&reloaded, metallib_path.c_str()) != CUDA_SUCCESS)
        return fail("second cuModuleLoad failed");
    CUfunction reloaded_function = nullptr;
    if (cuModuleGetFunction(&reloaded_function, reloaded, "matrix_mul") != CUDA_SUCCESS)
        return fail("second cuModuleGetFunction failed");
    int max_threads = 0;
    if (cuFuncGetAttribute(&max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
                           reloaded_function) != CUDA_SUCCESS)
        return fail("cuFuncGetAttribute failed");
    CUfunctionLoadingState reloaded_state = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&reloaded_state, reloaded_function) != CUDA_SUCCESS)
        return fail("cuFuncIsLoaded on the reloaded function failed");
    if (reloaded_state != CU_FUNCTION_LOADING_STATE_LOADED)
        return fail("preparation through cuFuncGetAttribute was not observed");

    // A handle whose module is gone must not report loaded because a
    // process-wide pipeline cache still holds the same path and name.
    if (cuModuleUnload(reloaded) != CUDA_SUCCESS) return fail("cuModuleUnload failed");
    CUfunctionLoadingState stale = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&stale, reloaded_function) != CUDA_ERROR_INVALID_HANDLE)
        return fail("cuFuncIsLoaded accepted a function from an unloaded module");
    if (cuFuncLoad(reloaded_function) != CUDA_ERROR_INVALID_HANDLE)
        return fail("cuFuncLoad accepted a function from an unloaded module");

    // Negative paths.
    if (cuFuncIsLoaded(nullptr, function) != CUDA_ERROR_INVALID_VALUE)
        return fail("cuFuncIsLoaded accepted a null state pointer");
    if (cuFuncLoad(nullptr) != CUDA_ERROR_INVALID_HANDLE)
        return fail("cuFuncLoad accepted a null function");
    CUfunctionLoadingState bogus_state = CU_FUNCTION_LOADING_STATE_MAX;
    if (cuFuncIsLoaded(&bogus_state, reinterpret_cast<CUfunction>(0x1234)) !=
        CUDA_ERROR_INVALID_HANDLE)
        return fail("cuFuncIsLoaded accepted an invalid function handle");

    cuModuleUnload(module);
    cuCtxDestroy(context);
    std::printf("PASS: cuFuncLoad prepares, cuFuncIsLoaded observes without preparing\n");
    return 0;
}
