#pragma once
#include "library_types.h"
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUSOLVER_COMMON_H_
#define CUSOLVER_COMMON_H_ 1
#endif


#ifdef __cplusplus
extern "C" {
#endif

typedef enum cusolverStatus_t {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
} cusolverStatus_t;

typedef enum cusolverEigType_t {
    CUSOLVER_EIG_TYPE_1 = 1,
    CUSOLVER_EIG_TYPE_2 = 2,
    CUSOLVER_EIG_TYPE_3 = 3,
} cusolverEigType_t;

typedef enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
} cusolverEigMode_t;

#ifdef __cplusplus
}
#endif
