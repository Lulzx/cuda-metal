#pragma once
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef __VECTOR_TYPES_H__
#define __VECTOR_TYPES_H__ 1
#endif


// CUDA-compatible vector type surface.
//
// CuMetal keeps the clean-room vector ABI declarations in cuda_runtime.h.
// Some CUDA projects, including PhysX, include vector_types.h directly, so
// expose the same declarations through the conventional standalone header.
#include "cuda_runtime.h"
