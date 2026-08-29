#pragma once
// CuMetal device-runtime declarations used by Clang's CUDA launch lowering.
// The device implementation is supplied by CuMetal's PTX-to-Metal path; host
// code retains an explicit unsupported result rather than attempting a device
// launch through the ordinary runtime ABI.

#include "cuda_runtime.h"

// Provide the type so code that checks for it compiles
#ifdef __cplusplus
extern "C" {
#endif

// cudaLaunchDevice is the device-side kernel launch function Clang looks up
// when compiling `child<<<grid, block, shared, stream>>>(...)` under
// relocatable-device-code mode.
#if defined(__clang__) && defined(__CUDA__)
__device__ cudaError_t cudaLaunchDevice(void*, void**, dim3, dim3,
                                        unsigned int, cudaStream_t);
#else
static inline cudaError_t cudaLaunchDevice(void*, void**, dim3, dim3, unsigned int, cudaStream_t) {
    return cudaErrorUnknown;
}
#endif

#ifdef __cplusplus
}
#endif
