#pragma once

#include <stddef.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int cufftHandle;

typedef struct {
    float x;
    float y;
} cufftComplex;

typedef struct {
    double x;
    double y;
} cufftDoubleComplex;

typedef float cufftReal;
typedef double cufftDoubleReal;

typedef enum cufftResult {
    CUFFT_SUCCESS = 0x0,
    CUFFT_INVALID_PLAN = 0x1,
    CUFFT_ALLOC_FAILED = 0x2,
    CUFFT_INVALID_TYPE = 0x3,
    CUFFT_INVALID_VALUE = 0x4,
    CUFFT_INTERNAL_ERROR = 0x5,
    CUFFT_EXEC_FAILED = 0x6,
    CUFFT_SETUP_FAILED = 0x7,
    CUFFT_INVALID_SIZE = 0x8,
    CUFFT_UNALIGNED_DATA = 0x9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 0xA,
    CUFFT_INVALID_DEVICE = 0xB,
    CUFFT_PARSE_ERROR = 0xC,
    CUFFT_NO_WORKSPACE = 0xD,
    CUFFT_NOT_IMPLEMENTED = 0xE,
    CUFFT_LICENSE_ERROR = 0x0F,
    CUFFT_NOT_SUPPORTED = 0x10,
} cufftResult;

typedef enum cufftType {
    CUFFT_R2C = 0x2a,   // real → complex (single)
    CUFFT_C2R = 0x2c,   // complex → real (single)
    CUFFT_C2C = 0x29,   // complex ↔ complex (single)
    CUFFT_D2Z = 0x6a,   // real → complex (double)
    CUFFT_Z2D = 0x6c,   // complex → real (double)
    CUFFT_Z2Z = 0x69,   // complex ↔ complex (double)
} cufftType;

#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE  (1)

// libraryPropertyType — mirrors CUDA library_types.h; guarded for multi-header includes.
#ifndef CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
#define CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
typedef enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL   = 2,
} libraryPropertyType;
#endif

cufftResult cufftGetVersion(int* version);
// SetWorkArea — ignored on UMA (vDSP manages its own scratch buffers).
cufftResult cufftSetWorkArea(cufftHandle plan, void* workArea);
// Estimate* — return a conservative scratch-size estimate without building a plan.
cufftResult cufftEstimate1d(int nx, cufftType type, int batch, size_t* workSize);
cufftResult cufftEstimate2d(int nx, int ny, cufftType type, size_t* workSize);
cufftResult cufftEstimate3d(int nx, int ny, int nz, cufftType type, size_t* workSize);
cufftResult cufftEstimateMany(int rank, int* n,
                               int* inembed, int istride, int idist,
                               int* onembed, int ostride, int odist,
                               cufftType type, int batch, size_t* workSize);
cufftResult cufftCreate(cufftHandle* plan);
cufftResult cufftDestroy(cufftHandle plan);
cufftResult cufftSetStream(cufftHandle plan, cudaStream_t stream);
cufftResult cufftGetSize(cufftHandle plan, size_t* workSize);

cufftResult cufftPlan1d(cufftHandle* plan, int nx, cufftType type, int batch);
cufftResult cufftPlan2d(cufftHandle* plan, int nx, int ny, cufftType type);
cufftResult cufftPlan3d(cufftHandle* plan, int nx, int ny, int nz, cufftType type);
cufftResult cufftPlanMany(cufftHandle* plan,
                          int rank,
                          int* n,
                          int* inembed,
                          int istride,
                          int idist,
                          int* onembed,
                          int ostride,
                          int odist,
                          cufftType type,
                          int batch);

cufftResult cufftMakePlan1d(cufftHandle plan, int nx, cufftType type, int batch,
                             size_t* workSize);
cufftResult cufftMakePlan2d(cufftHandle plan, int nx, int ny, cufftType type,
                             size_t* workSize);
cufftResult cufftMakePlan3d(cufftHandle plan, int nx, int ny, int nz, cufftType type,
                             size_t* workSize);
cufftResult cufftMakePlanMany(cufftHandle plan,
                               int rank,
                               int* n,
                               int* inembed,
                               int istride,
                               int idist,
                               int* onembed,
                               int ostride,
                               int odist,
                               cufftType type,
                               int batch,
                               size_t* workSize);

cufftResult cufftExecC2C(cufftHandle plan,
                          cufftComplex* idata,
                          cufftComplex* odata,
                          int direction);
cufftResult cufftExecR2C(cufftHandle plan, cufftReal* idata, cufftComplex* odata);
cufftResult cufftExecC2R(cufftHandle plan, cufftComplex* idata, cufftReal* odata);
cufftResult cufftExecZ2Z(cufftHandle plan,
                          cufftDoubleComplex* idata,
                          cufftDoubleComplex* odata,
                          int direction);
cufftResult cufftExecD2Z(cufftHandle plan,
                          cufftDoubleReal* idata,
                          cufftDoubleComplex* odata);
cufftResult cufftExecZ2D(cufftHandle plan,
                          cufftDoubleComplex* idata,
                          cufftDoubleReal* odata);

// cufftGetProperty — query cuFFT library version fields.
cufftResult cufftGetProperty(libraryPropertyType type, int* value);

#ifdef __cplusplus
}
#endif
