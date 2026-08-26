#pragma once
// Real cusparse.h declares every entry point with CUSPARSEAPI (empty off Windows).
// helper_cuda.h keys its cuSPARSE error strings off `#ifdef CUSPARSEAPI`.
#ifndef CUSPARSEAPI
#define CUSPARSEAPI
#endif
// CuMetal: define CUDA's canonical include-guard macros. Third-party code
// (NVIDIA's own Common/helper_cuda.h, among others) feature-detects on these
// to decide whether to declare its CUDA-dependent helpers, so a header that
// only uses `#pragma once` silently compiles to nothing useful downstream.
#ifndef CUSPARSE_V2_H_
#define CUSPARSE_V2_H_ 1
#endif

// CuMetal: forwarding header — cusparse_v2.h routes to cusparse.h.
#include "cusparse.h"
