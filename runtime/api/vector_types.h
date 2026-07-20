#pragma once

// CUDA-compatible vector type surface.
//
// CuMetal keeps the clean-room vector ABI declarations in cuda_runtime.h.
// Some CUDA projects, including PhysX, include vector_types.h directly, so
// expose the same declarations through the conventional standalone header.
#include "cuda_runtime.h"
