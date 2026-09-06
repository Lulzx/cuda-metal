// Force-included ahead of the translation unit when the compile was asked for
// NVRTC's --device-as-default-execution-space: from here on, every function
// without an execution-space annotation is __host__ __device__ instead of
// __host__. NVRTC compiles device code only and treats unannotated functions
// as device code; a header-only library written for it (NVIDIA Warp's tile.h,
// for one) calls `constexpr` helpers, constructors and lambdas that carry no
// __device__ at all.
//
// The region is deliberately never closed: nothing follows the main file, and
// Clang accepts an unmatched `begin`. Headers the SDK provides are included
// before this point by cuda_runtime.h, so their declarations keep their host
// execution space and a later include inside the region is guard-skipped.
#pragma clang force_cuda_host_device begin
