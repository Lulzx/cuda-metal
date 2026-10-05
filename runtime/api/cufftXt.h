#pragma once
// Clean-room subset of CUDA's cuFFT extensible-plan interface.
#ifndef CUFFT_XT_H_
#define CUFFT_XT_H_ 1
#endif

#include "cufft.h"

#ifdef __cplusplus
extern "C" {
#endif

// Multi-GPU descriptor types (CUDA's cudalibxt.h). CuMetal exposes one device,
// so descriptor execution refuses; the types exist so callers compile.
#define CUDA_XT_DESCRIPTOR_VERSION 0x01000000
#define MAX_CUDA_DESCRIPTOR_GPUS 64

typedef enum cudaXtCopyType_t {
    LIB_XT_COPY_HOST_TO_DEVICE = 0,
    LIB_XT_COPY_DEVICE_TO_HOST = 1,
    LIB_XT_COPY_DEVICE_TO_DEVICE = 2
} cudaLibXtCopyType;

typedef enum libFormat_t {
    LIB_FORMAT_CUFFT = 0x0,
    LIB_FORMAT_UNDEFINED = 0x1
} libFormat;

typedef struct cudaXtDesc_t {
    int version;
    int nGPUs;
    int GPUs[MAX_CUDA_DESCRIPTOR_GPUS];
    void* data[MAX_CUDA_DESCRIPTOR_GPUS];
    size_t size[MAX_CUDA_DESCRIPTOR_GPUS];
    void* cudaXtState;
} cudaXtDesc;

typedef struct cudaLibXtDesc_t {
    int version;
    cudaXtDesc* descriptor;
    libFormat library;
    int subFormat;
    void* libDescriptor;
} cudaLibXtDesc;

typedef enum cufftXtSubFormat_t {
    CUFFT_XT_FORMAT_INPUT = 0x00,
    CUFFT_XT_FORMAT_OUTPUT = 0x01,
    CUFFT_XT_FORMAT_INPLACE = 0x02,
    CUFFT_XT_FORMAT_INPLACE_SHUFFLED = 0x03,
    CUFFT_XT_FORMAT_1D_INPUT_SHUFFLED = 0x04,
    CUFFT_XT_FORMAT_DISTRIBUTED_INPUT = 0x05,
    CUFFT_XT_FORMAT_DISTRIBUTED_OUTPUT = 0x06,
    CUFFT_FORMAT_UNDEFINED = 0x07
} cufftXtSubFormat;

typedef enum cufftXtCopyType_t {
    CUFFT_COPY_HOST_TO_DEVICE = 0x00,
    CUFFT_COPY_DEVICE_TO_HOST = 0x01,
    CUFFT_COPY_DEVICE_TO_DEVICE = 0x02,
    CUFFT_COPY_UNDEFINED = 0x03
} cufftXtCopyType;

typedef enum cufftXtCallbackType_t {
    CUFFT_CB_LD_COMPLEX = 0x0,
    CUFFT_CB_LD_COMPLEX_DOUBLE = 0x1,
    CUFFT_CB_LD_REAL = 0x2,
    CUFFT_CB_LD_REAL_DOUBLE = 0x3,
    CUFFT_CB_ST_COMPLEX = 0x4,
    CUFFT_CB_ST_COMPLEX_DOUBLE = 0x5,
    CUFFT_CB_ST_REAL = 0x6,
    CUFFT_CB_ST_REAL_DOUBLE = 0x7,
    CUFFT_CB_UNDEFINED = 0x8
} cufftXtCallbackType;

// Only the single CuMetal device (ordinal 0) is accepted.
cufftResult cufftXtSetGPUs(cufftHandle plan, int nGPUs, int* whichGPUs);
cufftResult cufftXtSetWorkArea(cufftHandle plan, void** workArea);
// Executes with the transform type the plan was made for.
cufftResult cufftXtExec(cufftHandle plan, void* input, void* output, int direction);
// Descriptor (multi-GPU) data movement and execution: CUFFT_NOT_SUPPORTED.
cufftResult cufftXtMemcpy(cufftHandle plan, void* dstPointer, void* srcPointer,
                          cufftXtCopyType type);
cufftResult cufftXtExecDescriptorC2C(cufftHandle plan, cudaLibXtDesc* input,
                                     cudaLibXtDesc* output, int direction);
cufftResult cufftXtExecDescriptorZ2Z(cufftHandle plan, cudaLibXtDesc* input,
                                     cudaLibXtDesc* output, int direction);
// Load/store callbacks are device code injected into the transform kernel,
// which CuMetal's host FFT path has no place to run: CUFFT_NOT_SUPPORTED.
cufftResult cufftXtSetCallback(cufftHandle plan, void** callbackRoutine,
                               cufftXtCallbackType type, void** callerInfo);

cufftResult cufftXtMakePlanMany(cufftHandle plan,
                                 int rank,
                                 long long int* n,
                                 long long int* inembed,
                                 long long int istride,
                                 long long int idist,
                                 cudaDataType inputtype,
                                 long long int* onembed,
                                 long long int ostride,
                                 long long int odist,
                                 cudaDataType outputtype,
                                 long long int batch,
                                 size_t* workSize,
                                 cudaDataType executiontype);

#ifdef __cplusplus
}
#endif
