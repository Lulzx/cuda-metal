#pragma once

// CuMetal thrust shim: umbrella header.
// On Apple Silicon UMA, all thrust operations are CPU-backed.
// device_vector allocations go through cudaMalloc (UMA shared memory).

#include "device_ptr.h"
#include "device_vector.h"
#include "host_vector.h"
#include "sort.h"
#include "reduce.h"
#include "scan.h"
#include "transform.h"
#include "functional.h"
#include "unique.h"
#include "copy.h"
#include "generate.h"
#include "random.h"
#include "sequence.h"
#include "adjacent_difference.h"
#include "device_malloc.h"
#include "device_free.h"
#include "fill.h"
#include "find.h"
#include "for_each.h"
#include "pair.h"
#include "execution_policy.h"
#include "iterator/counting_iterator.h"
#include "iterator/zip_iterator.h"
