#pragma once

// CUDA-compatible vector constructor surface.
//
// The clean-room make_* implementations live beside CuMetal's vector ABI
// declarations in cuda_runtime.h. CUDA code commonly includes this header
// directly after vector_types.h.
#include "cuda_runtime.h"
