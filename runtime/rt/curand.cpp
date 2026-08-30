#include "curand.h"

#include "cuda_runtime.h"
#include "runtime_internal.h"

#include <cmath>
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
    curandRngType_t rng_type = CURAND_RNG_PSEUDO_DEFAULT;
    curandOrdering_t ordering = CURAND_ORDERING_PSEUDO_DEFAULT;
    unsigned int quasi_dimensions = 1;
    bool host_generator = false;
    std::mutex mutex;
};

extern "C" int cumetalRuntimeIsDevicePointer(const void* ptr);

namespace {
constexpr int kCurandCompatVersion = 12000;

bool supports_generation(curandRngType_t rng_type) {
    // CURAND_RNG_PSEUDO_DEFAULT is CuMetal's documented compatibility stream.
    // Named NVIDIA algorithms must not silently return a different bitstream.
    return rng_type == CURAND_RNG_PSEUDO_DEFAULT ||
           rng_type == CURAND_RNG_PSEUDO_MTGP32;
}

curandStatus_t create_generator(curandGenerator_t* generator,
                                curandRngType_t rng_type,
                                bool host_generator) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    const bool pseudo = (rng_type == CURAND_RNG_PSEUDO_DEFAULT   ||
                         rng_type == CURAND_RNG_PSEUDO_XORWOW    ||
                         rng_type == CURAND_RNG_PSEUDO_MRG32K3A  ||
                         rng_type == CURAND_RNG_PSEUDO_MTGP32    ||
                         rng_type == CURAND_RNG_PSEUDO_MT19937   ||
                         rng_type == CURAND_RNG_PSEUDO_PHILOX4_32_10);
    const bool quasi  = (rng_type == CURAND_RNG_QUASI_DEFAULT    ||
                         rng_type == CURAND_RNG_QUASI_SOBOL32    ||
                         rng_type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 ||
                         rng_type == CURAND_RNG_QUASI_SOBOL64    ||
                         rng_type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
    if (!pseudo && !quasi) {
        return CURAND_STATUS_TYPE_ERROR;
    }

    curandGenerator_t created = new (std::nothrow) curandGenerator_st();
    if (created == nullptr) {
        return CURAND_STATUS_ALLOCATION_FAILED;
    }
    created->rng_type = rng_type;
    created->host_generator = host_generator;
    *generator = created;
    return CURAND_STATUS_SUCCESS;
}

template <typename Operation>
curandStatus_t enqueue_generation(curandGenerator_t generator, Operation operation) {
    cudaStream_t stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(generator->mutex);
        stream = generator->stream;
    }
    return cumetal::rt::enqueue_host_operation(stream, std::move(operation)) == cudaSuccess
               ? CURAND_STATUS_SUCCESS
               : CURAND_STATUS_PREEXISTING_FAILURE;
}

template <typename Output, typename Operation>
curandStatus_t dispatch_generation(curandGenerator_t generator,
                                   Output* output_ptr,
                                   std::size_t count,
                                   Operation operation) {
    if (!supports_generation(generator->rng_type)) return CURAND_STATUS_TYPE_ERROR;
    if (!generator->host_generator && cumetalRuntimeIsDevicePointer(output_ptr) == 0) {
        return CURAND_STATUS_TYPE_ERROR;
    }
    if (!generator->host_generator) {
        cumetal::rt::AllocationTable::ResolvedAllocation resolved;
        if (!cumetal::rt::resolve_allocation_for_pointer(output_ptr, &resolved)) {
            return CURAND_STATUS_TYPE_ERROR;
        }
        if (count > resolved.remaining_size / sizeof(Output)) {
            return CURAND_STATUS_OUT_OF_RANGE;
        }
    }
    if (generator->host_generator) {
        operation();
        return CURAND_STATUS_SUCCESS;
    }
    return enqueue_generation(generator, std::move(operation));
}
}  // namespace

extern "C" {

curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type) {
    return create_generator(generator, rng_type, true);
}

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type) {
    return create_generator(generator, rng_type, false);
}

curandStatus_t curandDestroyGenerator(curandGenerator_t generator) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }

    cudaStream_t stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(generator->mutex);
        stream = generator->stream;
    }
    if (!generator->host_generator && cudaStreamSynchronize(stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
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

    cudaStream_t old_stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(generator->mutex);
        old_stream = generator->stream;
    }
    if (!generator->host_generator && cudaStreamSynchronize(old_stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
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

    cudaStream_t stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(generator->mutex);
        stream = generator->stream;
    }
    if (!generator->host_generator && cudaStreamSynchronize(stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
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

    cudaStream_t stream = nullptr;
    {
        std::lock_guard<std::mutex> lock(generator->mutex);
        stream = generator->stream;
    }
    if (!generator->host_generator && cudaStreamSynchronize(stream) != cudaSuccess) {
        return CURAND_STATUS_PREEXISTING_FAILURE;
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = generator->uniform(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = generator->uniform_double(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0f) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if ((num & 1U) != 0) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::normal_distribution<float> distribution(mean, stddev);
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = distribution(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if ((num & 1U) != 0) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::normal_distribution<double> distribution(mean, stddev);
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = distribution(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0f) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if ((num & 1U) != 0) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lognormal_distribution<float> distribution(mean, stddev);
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = distribution(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    if (!std::isfinite(mean) || !std::isfinite(stddev) || stddev <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if ((num & 1U) != 0) return CURAND_STATUS_LENGTH_NOT_MULTIPLE;
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lognormal_distribution<double> distribution(mean, stddev);
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = distribution(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = static_cast<unsigned int>(generator->engine());
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = generator->engine();
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    if (!std::isfinite(lambda) || lambda <= 0.0) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    if (num == 0) {
        return CURAND_STATUS_SUCCESS;
    }
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::poisson_distribution<unsigned int> distribution(lambda);
        std::lock_guard<std::mutex> lock(generator->mutex);
        for (size_t i = 0; i < num; ++i) {
            output_ptr[i] = distribution(generator->engine);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        std::uniform_real_distribution<float> u(0.0f, 1.0f);
        for (size_t i = 0; i < num; ++i) {
            float u_val = u(generator->engine);
            if (u_val <= 0.0f) u_val = 1e-38f;
            output_ptr[i] = -std::log(u_val);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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
    return dispatch_generation(generator, output_ptr, num, [=]() {
        std::lock_guard<std::mutex> lock(generator->mutex);
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (size_t i = 0; i < num; ++i) {
            double u_val = u(generator->engine);
            if (u_val <= 0.0) u_val = 1e-300;
            output_ptr[i] = -std::log(u_val);
        }
        generator->offset += static_cast<unsigned long long>(num);
    });
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

// ── Generator introspection / ordering / quasi-dimensions (batch 5) ──────────

curandStatus_t curandGetGeneratorType(curandGenerator_t generator, curandRngType_t* rng_type) {
    if (generator == nullptr || rng_type == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    std::lock_guard<std::mutex> lock(generator->mutex);
    *rng_type = generator->rng_type;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    // Validate ordering is a known value.
    if (order != CURAND_ORDERING_PSEUDO_BEST    &&
        order != CURAND_ORDERING_PSEUDO_DEFAULT &&
        order != CURAND_ORDERING_PSEUDO_SEEDED  &&
        order != CURAND_ORDERING_PSEUDO_LEGACY  &&
        order != CURAND_ORDERING_QUASI_DEFAULT) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->ordering = order;
    return CURAND_STATUS_SUCCESS;
}

curandStatus_t curandSetQuasiRandomGeneratorDimensions(curandGenerator_t generator,
                                                        unsigned int num_dimensions) {
    if (generator == nullptr) {
        return CURAND_STATUS_NOT_INITIALIZED;
    }
    if (num_dimensions == 0 || num_dimensions > 20000) {
        return CURAND_STATUS_OUT_OF_RANGE;
    }
    std::lock_guard<std::mutex> lock(generator->mutex);
    generator->quasi_dimensions = num_dimensions;
    return CURAND_STATUS_SUCCESS;
}

}  // extern "C"
