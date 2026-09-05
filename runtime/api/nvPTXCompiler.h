#pragma once
// CuMetal: define CUDA's canonical include-guard macros so third-party code
// that feature-detects on them sees a real nvPTXCompiler surface.
#ifndef __nvPTXCompiler_h__
#define __nvPTXCompiler_h__ 1
#endif

// CuMetal's nvPTXCompiler surface is a pass-through. The real library turns PTX
// into a SASS cubin ahead of the driver; CuMetal's driver consumes PTX text
// directly (`cuModuleLoadDataEx` parses it), so "compiling" here means handing
// the same PTX back as the compiled program. Callers that fall back to this
// path because they believe the driver is too old still end up loading a module
// that works.

#include <stddef.h>

#ifndef NVPTXCOMPILER_MAX_VERSION_MAJOR
#define NVPTXCOMPILER_MAX_VERSION_MAJOR 12
#endif
#ifndef NVPTXCOMPILER_MAX_VERSION_MINOR
#define NVPTXCOMPILER_MAX_VERSION_MINOR 0
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nvPTXCompileResult {
    NVPTXCOMPILE_SUCCESS = 0,
    NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE = 1,
    NVPTXCOMPILE_ERROR_INVALID_INPUT = 2,
    NVPTXCOMPILE_ERROR_COMPILATION_FAILURE = 3,
    NVPTXCOMPILE_ERROR_INTERNAL = 4,
    NVPTXCOMPILE_ERROR_OUT_OF_MEMORY = 5,
    NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE = 6,
    NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION = 7,
} nvPTXCompileResult;

typedef struct nvPTXCompiler* nvPTXCompilerHandle;

nvPTXCompileResult nvPTXCompilerGetVersion(unsigned int* major, unsigned int* minor);

nvPTXCompileResult nvPTXCompilerCreate(nvPTXCompilerHandle* compiler,
                                       size_t ptxCodeLen,
                                       const char* ptxCode);
nvPTXCompileResult nvPTXCompilerDestroy(nvPTXCompilerHandle* compiler);

nvPTXCompileResult nvPTXCompilerCompile(nvPTXCompilerHandle compiler,
                                        int numCompileOptions,
                                        const char* const* compileOptions);

nvPTXCompileResult nvPTXCompilerGetCompiledProgramSize(nvPTXCompilerHandle compiler,
                                                       size_t* binaryImageSize);
nvPTXCompileResult nvPTXCompilerGetCompiledProgram(nvPTXCompilerHandle compiler,
                                                   void* binaryImage);

nvPTXCompileResult nvPTXCompilerGetErrorLogSize(nvPTXCompilerHandle compiler, size_t* errorLogSize);
nvPTXCompileResult nvPTXCompilerGetErrorLog(nvPTXCompilerHandle compiler, char* errorLog);

nvPTXCompileResult nvPTXCompilerGetInfoLogSize(nvPTXCompilerHandle compiler, size_t* infoLogSize);
nvPTXCompileResult nvPTXCompilerGetInfoLog(nvPTXCompilerHandle compiler, char* infoLog);

#ifdef __cplusplus
}
#endif
