#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __LIBRARY_TYPES_H__
#define __LIBRARY_TYPES_H__ 1
#endif

// CuMetal: real CUDA declares cudaDataType_t and libraryPropertyType here, and
// every library header (cublas, cusparse, cufft, curand, cusolver) includes it.
// They used to be duplicated per-header, and cusparse.h's copy was `typedef int
// cudaDataType` -- which collided with cublas_v2.h's enum, so any translation
// unit including both failed with a typedef redefinition.
#ifdef __cplusplus
extern "C" {
#endif

#ifndef CUMETAL_CUDA_DATA_TYPE_DEFINED
#define CUMETAL_CUDA_DATA_TYPE_DEFINED
typedef enum cudaDataType_t {
    CUDA_R_16F  =  2,
    CUDA_C_16F  =  6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F  =  0,
    CUDA_C_32F  =  4,
    CUDA_R_64F  =  1,
    CUDA_C_64F  =  5,
    CUDA_R_8I   =  3,
    CUDA_R_8U   =  8,
    CUDA_R_32I  = 10,
} cudaDataType_t;
typedef cudaDataType_t cudaDataType;
#endif

#ifndef CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
#define CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
typedef enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
} libraryPropertyType;
#endif

#ifdef __cplusplus
}
#endif
