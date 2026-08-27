#pragma once
// Clean-room subset of CUDA's cuFFT extensible-plan interface.
#ifndef CUFFT_XT_H_
#define CUFFT_XT_H_ 1
#endif

#include "cufft.h"

#ifdef __cplusplus
extern "C" {
#endif

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
