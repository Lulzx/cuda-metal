// CuMetal: cuda_runtime_api.h — forwarding header.
// Many CUDA programs include <cuda_runtime_api.h> for API-only declarations
// (no device code). Forward to our full cuda_runtime.h.
#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __CUDA_RUNTIME_API_H__
// Special device ids accepted by memory-advice and prefetch APIs.
#define cudaCpuDeviceId ((int)-1)
#define cudaInvalidDeviceId ((int)-2)

#define __CUDA_RUNTIME_API_H__ 1
#endif

#include "cuda_runtime.h"
