#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __MATH_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__ 1
#endif

// CuMetal: forwarding header — device math intrinsics (__expf, __sinf, etc.)
// are defined in cuda_runtime.h.
#include "cuda_runtime.h"
