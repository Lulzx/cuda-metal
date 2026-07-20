#pragma once

#ifndef CUMETAL_FRONTEND_TEST_VALUE
#error "cumetalc did not forward the requested CUDA frontend definition"
#endif

static __device__ __forceinline__ float cumetal_frontend_test_bias() {
    return static_cast<float>(CUMETAL_FRONTEND_TEST_VALUE);
}
