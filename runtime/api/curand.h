#pragma once

#include <stddef.h>

#include "cuda_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct curandGenerator_st* curandGenerator_t;

typedef enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999,
} curandStatus_t;

typedef enum curandRngType {
    CURAND_RNG_PSEUDO_DEFAULT = 100,
} curandRngType_t;

// libraryPropertyType — mirrors CUDA library_types.h; guarded for multi-header includes.
#ifndef CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
#define CUMETAL_LIBRARY_PROPERTY_TYPE_DEFINED
typedef enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL   = 2,
} libraryPropertyType;
#endif

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
// Host generator — on Apple Silicon UMA, host and device share memory; identical to device generator.
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandGetVersion(int* version);
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream);
curandStatus_t curandGetStream(curandGenerator_t generator, cudaStream_t* stream);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                   unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* output_ptr, size_t num);
curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double* output_ptr,
                                           size_t num);
curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                    float* output_ptr,
                                    size_t num,
                                    float mean,
                                    float stddev);
curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator,
                                          double* output_ptr,
                                          size_t num,
                                          double mean,
                                          double stddev);
curandStatus_t curandGenerateLogNormal(curandGenerator_t generator,
                                       float* output_ptr,
                                       size_t num,
                                       float mean,
                                       float stddev);
curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator,
                                             double* output_ptr,
                                             size_t num,
                                             double mean,
                                             double stddev);
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* output_ptr, size_t num);
curandStatus_t curandGenerateLongLong(curandGenerator_t generator,
                                      unsigned long long* output_ptr,
                                      size_t num);
curandStatus_t curandGeneratePoisson(curandGenerator_t generator,
                                     unsigned int* output_ptr,
                                     size_t num,
                                     double lambda);
curandStatus_t curandGetProperty(libraryPropertyType type, int* value);

// Exponential distribution: X ~ Exp(1) = -ln(U), U ~ Uniform(0,1).
curandStatus_t curandGenerateExponential(curandGenerator_t generator,
                                          float* output_ptr,
                                          size_t num);
curandStatus_t curandGenerateExponentialDouble(curandGenerator_t generator,
                                               double* output_ptr,
                                               size_t num);

#ifdef __cplusplus
}
#endif
