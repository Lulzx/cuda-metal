#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// feature-detects on these to decide whether to declare its NVRTC-dependent
// helpers, so a header that only uses `#pragma once` silently compiles to
// nothing useful downstream.
#ifndef __NVRTC_H__
#define __NVRTC_H__ 1
#endif

// CuMetal's NVRTC surface compiles CUDA C++ at runtime by driving `cumetalc`,
// so the "CUBIN" a program yields is a Metal library. `cuModuleLoadDataEx`
// accepts those bytes directly, which is what makes the substitution invisible
// to callers that hand NVRTC output straight to the driver.
//
// PTX output has no analogue: `cumetalc` lowers CUDA source to AIR, never to
// PTX. `nvrtcGetPTX` therefore fails, and `nvrtcGetSupportedArchs` reports the
// one architecture CuMetal presents (sm_80), so architecture-driven callers
// pick the CUBIN path on their own.

#include <stddef.h>

#ifndef NVRTCAPI
#define NVRTCAPI
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
} nvrtcResult;

typedef struct _nvrtcProgram* nvrtcProgram;

const char* NVRTCAPI nvrtcGetErrorString(nvrtcResult result);

nvrtcResult NVRTCAPI nvrtcVersion(int* major, int* minor);

nvrtcResult NVRTCAPI nvrtcGetNumSupportedArchs(int* numArchs);
nvrtcResult NVRTCAPI nvrtcGetSupportedArchs(int* supportedArchs);

nvrtcResult NVRTCAPI nvrtcCreateProgram(nvrtcProgram* prog,
                                        const char* src,
                                        const char* name,
                                        int numHeaders,
                                        const char* const* headers,
                                        const char* const* includeNames);
nvrtcResult NVRTCAPI nvrtcDestroyProgram(nvrtcProgram* prog);

nvrtcResult NVRTCAPI nvrtcCompileProgram(nvrtcProgram prog,
                                         int numOptions,
                                         const char* const* options);

nvrtcResult NVRTCAPI nvrtcGetPTXSize(nvrtcProgram prog, size_t* ptxSizeRet);
nvrtcResult NVRTCAPI nvrtcGetPTX(nvrtcProgram prog, char* ptx);

nvrtcResult NVRTCAPI nvrtcGetCUBINSize(nvrtcProgram prog, size_t* cubinSizeRet);
nvrtcResult NVRTCAPI nvrtcGetCUBIN(nvrtcProgram prog, char* cubin);

nvrtcResult NVRTCAPI nvrtcGetLTOIRSize(nvrtcProgram prog, size_t* LTOIRSizeRet);
nvrtcResult NVRTCAPI nvrtcGetLTOIR(nvrtcProgram prog, char* LTOIR);

nvrtcResult NVRTCAPI nvrtcGetProgramLogSize(nvrtcProgram prog, size_t* logSizeRet);
nvrtcResult NVRTCAPI nvrtcGetProgramLog(nvrtcProgram prog, char* log);

nvrtcResult NVRTCAPI nvrtcAddNameExpression(nvrtcProgram prog, const char* nameExpression);
nvrtcResult NVRTCAPI nvrtcGetLoweredName(nvrtcProgram prog,
                                         const char* nameExpression,
                                         const char** loweredName);

#ifdef __cplusplus
}
#endif
