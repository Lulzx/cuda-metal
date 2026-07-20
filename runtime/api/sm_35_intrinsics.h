#pragma once

// CUDA 3.5-era device intrinsics are exposed by CuMetal's clean-room runtime
// header. Projects such as PhysX include this NVIDIA compatibility header
// directly even though they only use the ordinary shuffle/vote surface.
#include "cuda_runtime.h"
