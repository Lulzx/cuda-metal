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
//
// C++ standard library headers follow the same rule. Opened inside the region,
// libc++'s classes turn __host__ __device__ while their virtual bases stay
// host-only, and every `override` of what() or the like fails to compile.
// CCCL (CUB block algorithms through NVRTC, as CuPy uses them) includes these,
// so they are included here first and keep their host execution space. CCCL
// sees Clang, not NVRTC, and takes its host-compiler branches, which include
// <ostream> (and with it <ios>, <locale> and <bitset>).
#ifdef __cplusplus
#include <exception>
#include <functional>
#include <memory>
#include <new>
#include <optional>
#include <ostream>
#include <stdexcept>
#include <tuple>
#include <typeinfo>
#include <utility>
#endif
#pragma clang force_cuda_host_device begin
