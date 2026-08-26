#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUBLAS_API_H_
#define CUBLAS_API_H_ 1
#endif

// CuMetal: forwarding header — legacy cuBLAS v1 API types are in cublas_v2.h.
#include "cublas_v2.h"
