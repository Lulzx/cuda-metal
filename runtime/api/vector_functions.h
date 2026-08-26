#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __VECTOR_FUNCTIONS_H__
#define __VECTOR_FUNCTIONS_H__ 1
#endif


// CUDA-compatible vector constructor surface.
//
// The clean-room make_* implementations live beside CuMetal's vector ABI
// declarations in cuda_runtime.h. CUDA code commonly includes this header
// directly after vector_types.h.
#include "cuda_runtime.h"
