#include "curand.h"

#include "cuda_runtime.h"

#include <mutex>
#include <new>
#include <random>

struct curandGenerator_st {
    std::mt19937_64 engine{0};
    std::uniform_real_distribution<float> uniform{0.0f, 1.0f};
    std::uniform_real_distribution<double> uniform_double{0.0, 1.0};
    unsigned long long seed = 0;
    unsigned long long offset = 0;
    cudaStream_t stream = nullptr;
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

namespace {
constexpr int kCurandCompatVersion = 12000;
}  // namespace

extern "C" {

// Host generator — on Apple Silicon UMA, host and device share memory; identical to device generator.
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) {
    return curandCreateGenerator(generator, rng_type);
}

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (rng_type != CURAND_RNG_PSEUDO_DEFAULT) {
        return CURAND_STATUS_TYPE_ERROR;
    }

    curandGenerator_t created = new (std::nothrow) curandGenerator_st();
    if (created == nullptr) {
        return CURAND_STATUS_ALLOCATION_FAILED;
    }

    *generator = created;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    delete generator;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetVersion(int* version) {
    if (version == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    *version = kCurandCompatVersion;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->stream = stream;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetStream(curandGenerator_t generator, cudaStream_t* stream) {
    if (generator == nullptr || stream == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    *stream = generator->stream;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator,
                                                   unsigned long long seed) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->seed = seed;
    generator->offset = 0;
    generator->engine.seed(seed);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->engine.seed(generator->seed);
    generator->engine.discard(offset);
    generator->offset = offset;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* output_ptr, size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = generator->uniform(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateUniformDouble(curandGenerator_t generator,
                                           double* output_ptr,
                                           size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = generator->uniform_double(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormal(curandGenerator_t generator,
                                    float* output_ptr,
                                    size_t num,
                                    float mean,
                                    float stddev) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (stddev <= 0.0f) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::normal_distribution<float> distribution(mean, stddev);
    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = distribution(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateNormalDouble(curandGenerator_t generator,
                                          double* output_ptr,
                                          size_t num,
                                          double mean,
                                          double stddev) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (stddev <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::normal_distribution<double> distribution(mean, stddev);
    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = distribution(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormal(curandGenerator_t generator,
                                       float* output_ptr,
                                       size_t num,
                                       float mean,
                                       float stddev) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (stddev <= 0.0f) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lognormal_distribution<float> distribution(mean, stddev);
    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = distribution(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLogNormalDouble(curandGenerator_t generator,
                                             double* output_ptr,
                                             size_t num,
                                             double mean,
                                             double stddev) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (stddev <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lognormal_distribution<double> distribution(mean, stddev);
    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = distribution(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* output_ptr, size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = static_cast<unsigned int>(generator->engine());
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateLongLong(curandGenerator_t generator,
                                      unsigned long long* output_ptr,
                                      size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = generator->engine();
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGeneratePoisson(curandGenerator_t generator,
                                     unsigned int* output_ptr,
                                     size_t num,
                                     double lambda) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (lambda <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::poisson_distribution<unsigned int> distribution(lambda);
    std::lock_guard<std::mutex> lock(generator->mutex);
    for (size_t i = 0; i < num; ++i) {
        output_ptr[i] = distribution(generator->engine);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateExponential(curandGenerator_t generator,
                                          float* output_ptr,
                                          size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (size_t i = 0; i < num; ++i) {
        // X = -ln(U), U in (0,1); clamp to avoid -ln(0) = inf
        float u_val = u(generator->engine);
        if (u_val <= 0.0f) u_val = 1e-38f;
        output_ptr[i] = -std::log(u_val);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGenerateExponentialDouble(curandGenerator_t generator,
                                               double* output_ptr,
                                               size_t num) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (output_ptr == nullptr && num > 0) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    if (cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (cudaStreamSynchronize(generator->stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
    }

    std::lock_guard<std::mutex> lock(generator->mutex);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    for (size_t i = 0; i < num; ++i) {
        double u_val = u(generator->engine);
        if (u_val <= 0.0) u_val = 1e-300;
        output_ptr[i] = -std::log(u_val);
    }
    generator->offset += static_cast<unsigned long long>(num);
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandGetProperty(libraryPropertyType type, int* value) {
    if (value == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    switch (type) {
        case MAJOR_VERSION: *value = 12; break;
        case MINOR_VERSION: *value = 0;  break;
        case PATCH_LEVEL:   *value = 0;  break;
        default:
            return CURAND_STATUS_TYPE_ERROR;
    }
    return CURAND_STATUS_SUCCESS;
}

}  // extern "C"
